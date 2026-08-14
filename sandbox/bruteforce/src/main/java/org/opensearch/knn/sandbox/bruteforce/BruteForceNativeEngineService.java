/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.bruteforce;

import org.apache.commons.lang3.ArrayUtils;
import org.opensearch.knn.index.engine.NativeIndexBuildParams;
import org.opensearch.knn.index.engine.NativeSearchParams;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.sandbox.AbstractNativeEngineService;

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
    public long initIndex(NativeIndexBuildParams params) {
        return BruteForceService.initIndex(params.numDocs(), params.dim(), params.engineParameters().raw());
    }

    @Override
    public void insertToIndex(int[] docs, long vectorsAddress, long indexAddress, NativeIndexBuildParams params) {
        BruteForceService.insertToIndex(docs, vectorsAddress, params.dim(), indexAddress);
    }

    @Override
    public void writeIndex(IndexOutputWithBuffer output, long indexAddress, NativeIndexBuildParams params) {
        BruteForceService.writeIndex(indexAddress, output);
    }

    @Override
    public long loadIndex(IndexInputWithBuffer readStream, NativeIndexBuildParams params) {
        return BruteForceService.loadIndexWithStream(readStream);
    }

    @Override
    public KNNQueryResult[] queryIndex(long indexPointer, NativeSearchParams params) {
        // The engine does not claim filters or nested fields, so core should never send these.
        // Fail loudly rather than return silently unfiltered results if that ever changes.
        if (ArrayUtils.isNotEmpty(params.filteredIds()) || ArrayUtils.isNotEmpty(params.parentIds())) {
            throw new UnsupportedOperationException("The bruteforce engine does not support filters or nested fields");
        }
        // The typed read replaces the map fishing that used to happen inside the JNI wrapper, -1 means
        // scan everything.
        final int scanLimit = params.methodParameters().get(BruteForceLibrary.SCAN_LIMIT_KEY, -1);
        return BruteForceService.queryIndex(indexPointer, params.queryVector(), params.k(), scanLimit);
    }

    @Override
    public void free(long indexPointer) {
        BruteForceService.free(indexPointer);
    }
}
