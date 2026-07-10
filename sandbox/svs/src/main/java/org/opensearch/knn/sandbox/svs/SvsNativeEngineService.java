/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.apache.commons.lang3.ArrayUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.jni.NativeEngineService;
import org.opensearch.knn.sandbox.AbstractNativeEngineService;

import java.util.Map;

/**
 * {@link NativeEngineService} for the experimental Intel SVS engine. Routes every native index operation to
 * {@link SvsService} (the isolated {@code libopensearchknn_svs}), keeping all native lifecycle behind the
 * generic core SPI so the core {@code JNIService} holds no compile-time reference to SVS.
 *
 * <p>Operations outside the SVS scope (template builds, radial, nested queries) throw
 * {@link UnsupportedOperationException}; the corresponding capability checks in the core already exclude this
 * engine, so these are defensive backstops.
 */
public class SvsNativeEngineService extends AbstractNativeEngineService {

    public SvsNativeEngineService() {
        super(SVSConstants.SVS_ENGINE_NAME);
    }

    @Override
    public long initIndex(long numDocs, int dim, Map<String, Object> parameters) {
        return SvsService.initIndex(numDocs, dim, parameters);
    }

    @Override
    public void insertToIndex(int[] docs, long vectorsAddress, int dimension, Map<String, Object> parameters, long indexAddress) {
        int threadCount = (int) parameters.getOrDefault(KNNConstants.INDEX_THREAD_QTY, 0);
        SvsService.insertToIndex(docs, vectorsAddress, dimension, indexAddress, threadCount);
    }

    @Override
    public void writeIndex(IndexOutputWithBuffer output, long indexAddress, Map<String, Object> parameters, boolean skipFlat) {
        SvsService.writeIndex(indexAddress, output);
    }

    @Override
    public long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters) {
        return SvsService.loadIndexWithStream(readStream);
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
            // Reject rather than silently ignore parentIds, which would return multiple children per parent.
            throw new UnsupportedOperationException("Nested fields are not supported by the experimental SVS engine");
        }
        if (ArrayUtils.isNotEmpty(filteredIds)) {
            return SvsService.queryIndexWithFilter(indexPointer, queryVector, k, methodParameters, filteredIds, filterIdsType);
        }
        return SvsService.queryIndex(indexPointer, queryVector, k, methodParameters);
    }

    @Override
    public void free(long indexPointer, boolean isBinaryIndex) {
        // isBinaryIndex is ignored: the SVS engine has no binary indices.
        SvsService.free(indexPointer);
    }
}
