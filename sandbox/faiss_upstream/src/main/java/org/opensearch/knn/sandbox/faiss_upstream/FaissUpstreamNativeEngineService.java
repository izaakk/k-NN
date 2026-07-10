/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.faiss_upstream;

import org.apache.commons.lang3.ArrayUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.jni.NativeEngineService;
import org.opensearch.knn.sandbox.AbstractNativeEngineService;

import java.util.Map;

/**
 * {@link NativeEngineService} for the {@code faiss_upstream} engine: routes every native index operation
 * to {@link FaissUpstreamService} (the isolated {@code libopensearchknn_faiss_upstream}), keeping the whole
 * native lifecycle behind the generic core SPI. Operations outside this engine's scope (template builds,
 * radial search) inherit descriptive rejections from {@link AbstractNativeEngineService}.
 */
public class FaissUpstreamNativeEngineService extends AbstractNativeEngineService {

    public FaissUpstreamNativeEngineService() {
        super(FaissUpstreamConstants.FAISS_UPSTREAM_ENGINE_NAME);
    }

    @Override
    public long initIndex(long numDocs, int dim, Map<String, Object> parameters) {
        return FaissUpstreamService.initIndex(numDocs, dim, parameters);
    }

    @Override
    public void insertToIndex(int[] docs, long vectorsAddress, int dimension, Map<String, Object> parameters, long indexAddress) {
        int threadCount = (int) parameters.getOrDefault(KNNConstants.INDEX_THREAD_QTY, 0);
        FaissUpstreamService.insertToIndex(docs, vectorsAddress, dimension, indexAddress, threadCount);
    }

    @Override
    public void writeIndex(IndexOutputWithBuffer output, long indexAddress, Map<String, Object> parameters, boolean skipFlat) {
        FaissUpstreamService.writeIndex(indexAddress, output);
    }

    @Override
    public long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters) {
        return FaissUpstreamService.loadIndexWithStream(readStream);
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
        if (ArrayUtils.isNotEmpty(parentIds)) {
            throw new UnsupportedOperationException("Nested fields are not supported by the faiss_upstream engine");
        }
        if (ArrayUtils.isNotEmpty(filteredIds)) {
            // supportsFilters() is false, so the core answers filtered queries with exact search instead.
            throw new UnsupportedOperationException("Filtered search is not supported by the faiss_upstream engine");
        }
        return FaissUpstreamService.queryIndex(indexPointer, queryVector, k, methodParameters);
    }

    @Override
    public void free(long indexPointer, boolean isBinaryIndex) {
        // isBinaryIndex is ignored: this engine has no binary indices.
        FaissUpstreamService.free(indexPointer);
    }
}
