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
import org.opensearch.knn.index.engine.NativeEngineService;
import org.opensearch.knn.sandbox.AbstractNativeEngineService;

import java.util.Map;

/**
 * {@link NativeEngineService} for the experimental Intel SVS engine. Routes every native index operation to
 * {@link SvsService} (the isolated {@code libopensearchknn_svs}), keeping all native lifecycle behind the
 * generic core SPI so the core {@code JNIService} holds no compile-time reference to SVS.
 *
 * <p>Operations outside the SVS scope (template builds) throw {@link UnsupportedOperationException}; the
 * corresponding capability checks in the core already exclude this engine, so these are defensive backstops.
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
        // Non-empty parentIds makes this a nested query: the native layer passes an IDGrouper to the SVS
        // runtime so the graph search returns one best child per parent and k counts distinct parents
        // (empty arrays are normalized to null so the native layer has a single "no parents" signal).
        int[] parentIdsOrNull = ArrayUtils.isNotEmpty(parentIds) ? parentIds : null;
        if (ArrayUtils.isNotEmpty(filteredIds)) {
            return SvsService.queryIndexWithFilter(
                indexPointer,
                queryVector,
                k,
                methodParameters,
                filteredIds,
                filterIdsType,
                parentIdsOrNull
            );
        }
        return SvsService.queryIndex(indexPointer, queryVector, k, methodParameters, parentIdsOrNull);
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
        if (radius <= 0) {
            // The SVS index only accepts a strictly positive faiss-domain radius. Core converts the
            // user-facing threshold per space type (Faiss#distanceToRadialThreshold/#scoreToRadialThreshold),
            // so this rejects e.g. inner-product max_distance >= 0 or cosine min_score <= 0.5.
            throw new UnsupportedOperationException(
                "The SVS engine does not support radial thresholds that resolve to a non-positive radius "
                    + "(converted radius was "
                    + radius
                    + "); use a stricter max_distance/min_score for this space type"
            );
        }
        // Non-empty parentIds reduces the in-radius results to the best child per parent (nested radial).
        int[] parentIdsOrNull = ArrayUtils.isNotEmpty(parentIds) ? parentIds : null;
        if (ArrayUtils.isNotEmpty(filteredIds)) {
            return SvsService.radiusQueryIndexWithFilter(
                indexPointer,
                queryVector,
                radius,
                methodParameters,
                indexMaxResultWindow,
                filteredIds,
                filterIdsType,
                parentIdsOrNull
            );
        }
        return SvsService.radiusQueryIndex(indexPointer, queryVector, radius, methodParameters, indexMaxResultWindow, parentIdsOrNull);
    }

    @Override
    public void free(long indexPointer, boolean isBinaryIndex) {
        // isBinaryIndex is ignored: the SVS engine has no binary indices.
        SvsService.free(indexPointer);
    }
}
