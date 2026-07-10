/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;

import java.util.Map;

/**
 * Generic native-index lifecycle contract for an engine that is contributed at runtime (rather than being a
 * built-in such as Faiss or NMSLIB). {@link JNIService} routes every native operation for the registered
 * engine to an implementation of this interface, so it can drive its own isolated JNI library without
 * {@code JNIService} holding any compile-time reference to that engine.
 *
 * <p>The method set mirrors {@link JNIService}'s per-engine entry points. An implementation that does not
 * support a particular operation (for example template-based builds or radial search) should throw
 * {@link UnsupportedOperationException}; the corresponding capability checks in the core typically keep those
 * paths unreachable, so the throws are defensive backstops.
 */
@ExperimentalApi
public interface NativeEngineService {

    long initIndex(long numDocs, int dim, Map<String, Object> parameters);

    void insertToIndex(int[] docs, long vectorsAddress, int dimension, Map<String, Object> parameters, long indexAddress);

    void writeIndex(IndexOutputWithBuffer output, long indexAddress, Map<String, Object> parameters, boolean skipFlat);

    void createIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        int dim,
        IndexOutputWithBuffer output,
        byte[] templateIndex,
        Map<String, Object> parameters
    );

    long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters);

    KNNQueryResult[] queryIndex(
        long indexPointer,
        float[] queryVector,
        int k,
        Map<String, ?> methodParameters,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    );

    KNNQueryResult[] radiusQueryIndex(
        long indexPointer,
        float[] queryVector,
        float radius,
        Map<String, ?> methodParameters,
        int indexMaxResultWindow,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    );

    void free(long indexPointer, boolean isBinaryIndex);
}
