/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;

/**
 * The search side parameters of a {@link NativeEngineService} query. k feeds queryIndex, radius and
 * indexMaxResultWindow feed radiusQueryIndex, everything else feeds both. filterIdsType discriminates
 * the filteredIds encoding.
 */
@ExperimentalApi
public record NativeSearchParams(float[] queryVector, int k, float radius, int indexMaxResultWindow, EngineParameters methodParameters,
    long[] filteredIds, int filterIdsType, int[] parentIds) {
}
