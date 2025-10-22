/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.model.QueryContext;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SEARCH_WINDOW_SIZE;

/**
 * Faiss SVS Vamana search context. Defines query-time parameters that can be tuned
 * to trade off between accuracy and performance.
 *
 * These parameters can be set per-query to override the index-time defaults:
 * - search_window_size: Controls the size of the search window during graph traversal
 * - search_buffer_capacity: Controls the buffer capacity during search
 */
public final class FaissSVSVamanaSearchContext implements KNNLibrarySearchContext {

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return Map.of(
            METHOD_PARAMETER_SEARCH_WINDOW_SIZE,
            new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_WINDOW_SIZE, null, value -> value > 0),
            METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY,
            new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY, null, value -> value > 0)
        );
    }
}
