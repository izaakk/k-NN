/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.faiss_upstream;

import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.sandbox.ExperimentalAlgorithm;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;

/**
 * Search context for {@code hnsw_flat_panorama}: the query-time tunable is the standard HNSW
 * {@code ef_search} (beam width — recall/latency knob). Panorama's progressive-distance pruning is
 * automatic and has no query-time parameter.
 */
@ExperimentalAlgorithm(description = "faiss_upstream HNSWFlatPanorama search context", since = "3.9.0")
public final class HNSWFlatPanoramaSearchContext implements KNNLibrarySearchContext {

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return Map.of(METHOD_PARAMETER_EF_SEARCH, new Parameter.IntegerParameter(METHOD_PARAMETER_EF_SEARCH, null, (v, context) -> v > 0));
    }
}
