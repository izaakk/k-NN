/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.jni.NativeEngineService;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Pure-Java, in-memory {@link NativeEngineService} for the fixture engine. There is deliberately no vector
 * math and no off-heap access here: the seam under test is {@code JNIService}'s routing — that every native
 * operation invoked with the fixture engine arrives at THIS service with its arguments intact, and never at
 * the built-in Faiss/Nmslib services. Each call is recorded in an op log the tests assert on.
 *
 * <p>Operations a minimal tenant would not support (template builds, radial search) throw
 * {@link UnsupportedOperationException} after logging, mirroring how a real tenant declines them — the
 * tests assert the throw originates here, proving even unsupported-op dispatch is engine-owned.
 */
public final class FixtureNativeEngineService implements NativeEngineService {

    /**
     * The single instance handed to the core through {@link FixtureEngineProvider}. The engine table built
     * by {@code KNNEngine} holds this exact instance, so tests reach the op log through it.
     */
    public static final FixtureNativeEngineService INSTANCE = new FixtureNativeEngineService();

    private final List<String> opLog = Collections.synchronizedList(new ArrayList<>());
    private final AtomicLong nextHandle = new AtomicLong(1000);

    private FixtureNativeEngineService() {}

    /** Snapshot of the recorded operations, in call order. */
    public List<String> opLog() {
        return List.copyOf(opLog);
    }

    /** Clears the recorded operations between tests. */
    public void reset() {
        opLog.clear();
    }

    @Override
    public long initIndex(long numDocs, int dim, Map<String, Object> parameters) {
        final long handle = nextHandle.incrementAndGet();
        opLog.add(String.format(Locale.ROOT, "initIndex(numDocs=%d, dim=%d) -> %d", numDocs, dim, handle));
        return handle;
    }

    @Override
    public void insertToIndex(int[] docs, long vectorsAddress, int dimension, Map<String, Object> parameters, long indexAddress) {
        opLog.add(String.format(Locale.ROOT, "insertToIndex(docs=%d, dim=%d, handle=%d)", docs.length, dimension, indexAddress));
    }

    @Override
    public boolean prefersJavaVectors() {
        // The fixture is a pure-JVM engine: it consumes vector batches on-heap (float[][]) — the iterative
        // build path delivers them via the on-heap insertToIndex instead of an off-heap address.
        return true;
    }

    @Override
    public void insertToIndex(int[] docs, float[][] vectors, Map<String, Object> parameters, long indexAddress) {
        opLog.add(
            String.format(
                Locale.ROOT,
                "insertToIndex(docs=%d, vectors=%d, dim=%d, handle=%d)",
                docs.length,
                vectors.length,
                vectors.length == 0 ? 0 : vectors[0].length,
                indexAddress
            )
        );
    }

    /** Bytes the fixture writes as its "segment file": pins the pure-Java write/read path end-to-end. */
    static final byte[] FIXTURE_SEGMENT_BYTES = { 'f', 'i', 'x', 't', 'u', 'r', 'e' };

    @Override
    public void writeIndex(IndexOutputWithBuffer output, long indexAddress, Map<String, Object> parameters, boolean skipFlat) {
        // A pure-Java engine streams its serialized index through the wrapper's on-heap write path.
        output.writeBytes(FIXTURE_SEGMENT_BYTES, 0, FIXTURE_SEGMENT_BYTES.length);
        opLog.add(String.format(Locale.ROOT, "writeIndex(handle=%d, bytes=%d)", indexAddress, FIXTURE_SEGMENT_BYTES.length));
    }

    @Override
    public void createIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        int dim,
        IndexOutputWithBuffer output,
        byte[] templateIndex,
        Map<String, Object> parameters
    ) {
        opLog.add("createIndexFromTemplate");
        throw new UnsupportedOperationException("Template-based index builds are not supported by the fixture engine");
    }

    @Override
    public long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters) {
        // A pure-Java engine reads its serialized index back through the wrapper's on-heap read path;
        // the fixture consumes the whole stream so the dispatch test can assert the write/read round-trip.
        final byte[] content = new byte[(int) readStream.length()];
        int read = 0;
        try {
            while (read < content.length) {
                final int n = readStream.readBytes(content, read, content.length - read);
                if (n < 0) {
                    break;
                }
                read += n;
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        final long handle = nextHandle.incrementAndGet();
        opLog.add(String.format(Locale.ROOT, "loadIndex(bytes=%d) -> %d", read, handle));
        return handle;
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
        opLog.add(String.format(Locale.ROOT, "queryIndex(handle=%d, k=%d, methodParameters=%s)", indexPointer, k, methodParameters));
        final KNNQueryResult[] results = new KNNQueryResult[k];
        for (int i = 0; i < k; i++) {
            results[i] = new KNNQueryResult(i, 1.0f / (1 + i));
        }
        return results;
    }

    @Override
    public KNNQueryResult[] radiusQueryIndex(
        long indexPointer,
        float[] queryVector,
        float radius,
        Map<String, ?> methodParameters,
        int indexMaxResultWindow,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        opLog.add("radiusQueryIndex");
        throw new UnsupportedOperationException("Radial search is not supported by the fixture engine");
    }

    @Override
    public void free(long indexPointer, boolean isBinaryIndex) {
        opLog.add(String.format(Locale.ROOT, "free(handle=%d)", indexPointer));
    }
}
