/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.bruteforce;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.sandbox.AbstractNativeEngineService;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import static org.opensearch.knn.sandbox.bruteforce.BruteForceLibrary.ENGINE_NAME;
import static org.opensearch.knn.sandbox.bruteforce.BruteForceLibrary.PARAM_SPACE_TYPE;

/**
 * The example engine's {@code NativeEngineService}, implemented entirely on-heap: "native" only in the sense
 * that it services the routed native-index lifecycle. Index handles are keys into a map rather than memory
 * addresses; core treats the {@code long} as opaque, which is exactly what the seam allows.
 *
 * <p>Two spots below bridge the seam's native-shaped contracts into Java, and both are candidate SPI
 * refinements surfaced by this example: {@link #readStagedVectors} (the off-heap staging contract) and
 * {@link #lucene} (the JNI-shaped read wrapper). Unsupported operations (template builds, radial search)
 * inherit descriptive throws from {@link AbstractNativeEngineService}; the library's capability flags keep
 * those paths unreachable, including filtering, which core runs itself for engines that do not claim it.
 */
public final class BruteForceNativeEngineService extends AbstractNativeEngineService {

    public static final BruteForceNativeEngineService INSTANCE = new BruteForceNativeEngineService();

    /** In-heap index: parallel doc-id and vector arrays plus the space fixed at creation. */
    private record ScanIndex(SpaceType spaceType, int dimension, int[] docIds, float[][] vectors) {
    }

    private final ConcurrentHashMap<Long, ScanIndex> indexes = new ConcurrentHashMap<>();
    private final AtomicLong nextHandle = new AtomicLong(1);

    private BruteForceNativeEngineService() {
        super(ENGINE_NAME);
    }

    @Override
    public long initIndex(long numDocs, int dim, Map<String, Object> parameters) {
        final Object space = parameters == null ? null : parameters.get(PARAM_SPACE_TYPE);
        final SpaceType spaceType = space == null ? SpaceType.L2 : SpaceType.getSpace(space.toString());
        return store(new ScanIndex(spaceType, dim, new int[0], new float[0][]));
    }

    @Override
    public void insertToIndex(int[] docs, long vectorsAddress, int dimension, Map<String, Object> parameters, long indexAddress) {
        final ScanIndex index = existing(indexAddress);
        final int[] docIds = Arrays.copyOf(index.docIds, index.docIds.length + docs.length);
        System.arraycopy(docs, 0, docIds, index.docIds.length, docs.length);
        final float[][] vectors = Arrays.copyOf(index.vectors, index.vectors.length + docs.length);
        readStagedVectors(vectorsAddress, docs.length, dimension, vectors, index.vectors.length);
        indexes.put(indexAddress, new ScanIndex(index.spaceType, index.dimension, docIds, vectors));
    }

    @Override
    public void writeIndex(IndexOutputWithBuffer output, long indexAddress, Map<String, Object> parameters, boolean skipFlat) {
        final ScanIndex index = existing(indexAddress);
        try {
            final ByteArrayOutputStream payload = new ByteArrayOutputStream();
            try (DataOutputStream data = new DataOutputStream(payload)) {
                data.writeUTF(index.spaceType.getValue());
                data.writeInt(index.dimension);
                data.writeInt(index.docIds.length);
                for (int docId : index.docIds) {
                    data.writeInt(docId);
                }
                for (float[] vector : index.vectors) {
                    for (float component : vector) {
                        data.writeFloat(component);
                    }
                }
            }
            // Length-framed so the loader stops before the Lucene codec footer that follows in the file.
            final ByteArrayOutputStream framed = new ByteArrayOutputStream();
            new DataOutputStream(framed).writeLong(payload.size());
            payload.writeTo(framed);
            output.writeFromStreamWithBuffer(new ByteArrayInputStream(framed.toByteArray()), 64 * 1024);
        } catch (IOException e) {
            throw new RuntimeException("Failed to write bruteforce index", e);
        }
    }

    @Override
    public long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters) {
        try {
            final org.apache.lucene.store.IndexInput input = lucene(readStream);
            final byte[] frame = new byte[Long.BYTES];
            input.readBytes(frame, 0, frame.length);
            final byte[] payload = new byte[Math.toIntExact(new DataInputStream(new ByteArrayInputStream(frame)).readLong())];
            input.readBytes(payload, 0, payload.length);
            final DataInputStream data = new DataInputStream(new ByteArrayInputStream(payload));
            final SpaceType spaceType = SpaceType.getSpace(data.readUTF());
            final int dimension = data.readInt();
            final int count = data.readInt();
            final int[] docIds = new int[count];
            for (int i = 0; i < count; i++) {
                docIds[i] = data.readInt();
            }
            final float[][] vectors = new float[count][dimension];
            for (int i = 0; i < count; i++) {
                for (int d = 0; d < dimension; d++) {
                    vectors[i][d] = data.readFloat();
                }
            }
            return store(new ScanIndex(spaceType, dimension, docIds, vectors));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load bruteforce index", e);
        }
    }

    @Override
    public KNNQueryResult[] queryIndex(
        long indexPointer,
        float[] queryVector,
        int k,
        Map<String, ?> methodParameters,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        final ScanIndex index = existing(indexPointer);
        // Max-heap by raw distance keeps the k closest; ties broken on doc id for determinism.
        final PriorityQueue<float[]> worstFirst = new PriorityQueue<>(
            Comparator.<float[]>comparingDouble(e -> e[0]).thenComparingDouble(e -> e[1]).reversed()
        );
        for (int i = 0; i < index.docIds.length; i++) {
            final float distance = squaredL2(queryVector, index.vectors[i]);
            if (worstFirst.size() < k) {
                worstFirst.add(new float[] { distance, index.docIds[i] });
            } else if (worstFirst.peek()[0] > distance) {
                worstFirst.poll();
                worstFirst.add(new float[] { distance, index.docIds[i] });
            }
        }
        final float[][] closest = worstFirst.toArray(new float[0][]);
        Arrays.sort(closest, Comparator.comparingDouble(e -> e[0]));
        final KNNQueryResult[] results = new KNNQueryResult[closest.length];
        for (int i = 0; i < closest.length; i++) {
            // Raw squared-L2 distance; core translates it to a score through the engine's library.
            results[i] = new KNNQueryResult((int) closest[i][1], closest[i][0]);
        }
        return results;
    }

    @Override
    public void free(long indexPointer, boolean isBinaryIndex) {
        indexes.remove(indexPointer);
    }

    private long store(ScanIndex index) {
        final long handle = nextHandle.getAndIncrement();
        indexes.put(handle, index);
        return handle;
    }

    private ScanIndex existing(long handle) {
        final ScanIndex index = indexes.get(handle);
        if (index == null) {
            throw new IllegalStateException("Unknown bruteforce index handle " + handle);
        }
        return index;
    }

    private static float squaredL2(float[] query, float[] vector) {
        float sum = 0f;
        for (int i = 0; i < query.length; i++) {
            final float diff = query[i] - vector[i];
            sum += diff * diff;
        }
        return sum;
    }

    // ------------------------------------------------------------------
    // The two native-shaped contracts this pure-JVM tenant has to bridge.
    // ------------------------------------------------------------------

    private static final sun.misc.Unsafe UNSAFE;

    static {
        try {
            final Field field = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
            field.setAccessible(true);
            UNSAFE = (sun.misc.Unsafe) field.get(null);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("The bruteforce example engine requires sun.misc.Unsafe to read staged vectors", e);
        }
    }

    /**
     * {@code vectorsAddress} is the address of a C++ {@code std::vector<float>} OBJECT, not of the float
     * data: native tenants {@code reinterpret_cast} it and call {@code ->data()} (see
     * {@code jni/src/commons.cpp}). The equivalent here: the first pointer-sized word of the libstdc++ layout
     * ({@code _M_start}) is the address of {@code count * dimension} contiguous native-order floats. The
     * layout being ABI-implicit is a candidate SPI documentation refinement.
     */
    private static void readStagedVectors(long vectorsAddress, int count, int dimension, float[][] into, int atRow) {
        final long dataAddress = UNSAFE.getLong(vectorsAddress); // _M_start
        final long available = (UNSAFE.getLong(vectorsAddress + Long.BYTES) - dataAddress) / Float.BYTES; // to _M_finish
        if (available < (long) count * dimension) {
            throw new IllegalStateException("Staged data holds " + available + " floats; expected " + ((long) count * dimension));
        }
        long cursor = dataAddress;
        for (int row = 0; row < count; row++) {
            final float[] vector = new float[dimension];
            for (int col = 0; col < dimension; col++) {
                vector[col] = UNSAFE.getFloat(cursor);
                cursor += Float.BYTES;
            }
            into[atRow + row] = vector;
        }
    }

    /**
     * {@link IndexInputWithBuffer}'s read methods are private by design (they are invoked from the JNI
     * layer), so a pure-JVM tenant reads the underlying Lucene {@code IndexInput} instead. A public read
     * path on the wrapper would remove this reflection; candidate SPI refinement.
     */
    private static org.apache.lucene.store.IndexInput lucene(IndexInputWithBuffer readStream) {
        try {
            final Field field = IndexInputWithBuffer.class.getDeclaredField("indexInput");
            field.setAccessible(true);
            return (org.apache.lucene.store.IndexInput) field.get(readStream);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to unwrap IndexInputWithBuffer", e);
        }
    }
}
