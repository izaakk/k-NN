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
 * Faiss SVS Vamana search context.
 */
public final class FaissSVSVamanaSearchContext implements KNNLibrarySearchContext {

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return Map.of(
            METHOD_PARAMETER_SEARCH_WINDOW_SIZE,
            new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_WINDOW_SIZE, null, (v, context) -> v > 0),
            METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY,
            new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY, null, (v, context) -> v > 0)
        );
    }
}
