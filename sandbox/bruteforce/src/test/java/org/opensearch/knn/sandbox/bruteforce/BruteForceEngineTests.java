/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.bruteforce;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.test.OpenSearchTestCase;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Map;

import static org.opensearch.knn.sandbox.bruteforce.BruteForceLibrary.ENGINE_NAME;
import static org.opensearch.knn.sandbox.bruteforce.BruteForceLibrary.EXTENSION;
import static org.opensearch.knn.sandbox.bruteforce.BruteForceLibrary.PARAM_SPACE_TYPE;

/**
 * Registration through the SPI, plus the full routed lifecycle exactly as core drives it:
 * init → insert (vectors staged off-heap, std::vector layout) → write (Lucene output) → load (Lucene input)
 * → query → free, with ground-truth nearest-neighbor assertions.
 */
public class BruteForceEngineTests extends OpenSearchTestCase {

    public void testEngineRegistersAsFirstClassKNNEngine() {
        final KNNEngine engine = KNNEngine.getEngine(ENGINE_NAME);
        assertNotNull(engine);
        assertEquals(ENGINE_NAME, engine.getName());
        assertTrue(Arrays.asList(KNNEngine.values()).contains(engine));
        assertSame(BruteForceNativeEngineService.INSTANCE, engine.getNativeService());
        assertEquals(EXTENSION, engine.getExtension());
        assertTrue(engine.supportsIterativeBuild());
        assertTrue(engine.createsCustomSegmentFiles());
        assertFalse(engine.supportsRadialSearch());
        assertFalse(engine.supportsNestedFields());
        assertFalse(engine.supportsFilters());
    }

    public void testFullLifecycleReturnsExactNeighbors() throws Exception {
        final BruteForceNativeEngineService service = BruteForceNativeEngineService.INSTANCE;
        final int dimension = 4;
        final int count = 64;

        // Deterministic vectors: doc i at (i, i, i, i).
        final float[][] vectors = new float[count][dimension];
        final int[] docIds = new int[count];
        for (int i = 0; i < count; i++) {
            docIds[i] = i;
            Arrays.fill(vectors[i], i);
        }

        final long address = stageAsStdVector(vectors);
        final long buildHandle;
        try {
            buildHandle = service.initIndex(count, dimension, Map.of(PARAM_SPACE_TYPE, "l2"));
            service.insertToIndex(docIds, address, dimension, Map.of(), buildHandle);
        } finally {
            freeStdVector(address);
        }

        try (Directory directory = new ByteBuffersDirectory()) {
            try (IndexOutput output = directory.createOutput("index" + EXTENSION, IOContext.DEFAULT)) {
                service.writeIndex(new IndexOutputWithBuffer(output), buildHandle, Map.of(), false);
            }
            service.free(buildHandle, false);

            try (IndexInput input = directory.openInput("index" + EXTENSION, IOContext.DEFAULT)) {
                final long searchHandle = service.loadIndex(new IndexInputWithBuffer(input), Map.of());

                // Query at (10.2, ...): nearest docs by L2 are 10, 11, 9, 12, 8.
                final float[] query = new float[dimension];
                Arrays.fill(query, 10.2f);
                final KNNQueryResult[] results = service.queryIndex(searchHandle, query, 5, null, null, 0, null);

                assertEquals(5, results.length);
                assertEquals(10, results[0].getId());
                assertEquals(11, results[1].getId());
                assertEquals(9, results[2].getId());
                assertEquals(12, results[3].getId());
                assertEquals(8, results[4].getId());
                // Raw scores are squared-L2 distances, ascending.
                assertEquals(dimension * 0.2f * 0.2f, results[0].getScore(), 1e-4);
                for (int i = 1; i < results.length; i++) {
                    assertTrue(results[i].getScore() >= results[i - 1].getScore());
                }

                service.free(searchHandle, false);
            }
        }
    }

    // --- staging that emulates JNICommons#storeVectorData: the address points at a std::vector<float>
    // object (libstdc++ layout: start/finish/capacity pointers); the floats live behind its first word. ---

    private static sun.misc.Unsafe unsafe() throws Exception {
        final Field field = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
        field.setAccessible(true);
        return (sun.misc.Unsafe) field.get(null);
    }

    private static long stageAsStdVector(float[][] vectors) throws Exception {
        final sun.misc.Unsafe unsafe = unsafe();
        final long dataBytes = (long) vectors.length * vectors[0].length * Float.BYTES;
        final long dataAddress = unsafe.allocateMemory(dataBytes);
        long cursor = dataAddress;
        for (final float[] vector : vectors) {
            for (final float component : vector) {
                unsafe.putFloat(cursor, component);
                cursor += Float.BYTES;
            }
        }
        final long vectorObject = unsafe.allocateMemory(3 * Long.BYTES);
        unsafe.putLong(vectorObject, dataAddress); // _M_start
        unsafe.putLong(vectorObject + Long.BYTES, dataAddress + dataBytes); // _M_finish
        unsafe.putLong(vectorObject + 2 * Long.BYTES, dataAddress + dataBytes); // _M_end_of_storage
        return vectorObject;
    }

    private static void freeStdVector(long vectorObject) throws Exception {
        final sun.misc.Unsafe unsafe = unsafe();
        unsafe.freeMemory(unsafe.getLong(vectorObject));
        unsafe.freeMemory(vectorObject);
    }
}
