/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.bruteforce;

import org.apache.commons.lang3.ArrayUtils;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.sandbox.AbstractNativeEngineService;

import java.util.Map;

/**
 * Routes the native index operations to {@link BruteForceService}. Operations the engine does not
 * support inherit descriptive throws from {@link AbstractNativeEngineService}.
 */
public final class BruteForceNativeEngineService extends AbstractNativeEngineService {

    public static final BruteForceNativeEngineService INSTANCE = new BruteForceNativeEngineService();

    private BruteForceNativeEngineService() {
        super(BruteForceLibrary.ENGINE_NAME);
    }

    @Override
    public long initIndex(long numDocs, int dim, Map<String, Object> parameters) {
        return BruteForceService.initIndex(numDocs, dim, parameters);
    }

    @Override
    public void insertToIndex(int[] docs, long vectorsAddress, int dimension, Map<String, Object> parameters, long indexAddress) {
        BruteForceService.insertToIndex(docs, vectorsAddress, dimension, indexAddress);
    }

    @Override
    public void writeIndex(IndexOutputWithBuffer output, long indexAddress, Map<String, Object> parameters, boolean skipFlat) {
        BruteForceService.writeIndex(indexAddress, output);
    }

    @Override
    public long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters) {
        return BruteForceService.loadIndexWithStream(readStream);
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
        // The engine does not claim filters or nested fields, so core should never send these.
        // Fail loudly rather than return silently unfiltered results if that ever changes.
        if (ArrayUtils.isNotEmpty(filteredIds) || ArrayUtils.isNotEmpty(parentIds)) {
            throw new UnsupportedOperationException("The bruteforce engine does not support filters or nested fields");
        }
        return BruteForceService.queryIndex(indexPointer, queryVector, k, methodParameters);
    }

    @Override
    public void free(long indexPointer, boolean isBinaryIndex) {
        BruteForceService.free(indexPointer);
    }
}
