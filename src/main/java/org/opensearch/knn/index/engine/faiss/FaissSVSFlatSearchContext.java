/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.model.QueryContext;

import java.util.Collections;
import java.util.Map;

/**
 * Search context for SVS Flat method. SVS Flat is an exhaustive (brute-force) search method 
 * that does not support any search-time parameters, as it always computes exact distances 
 * to all vectors.
 */
public final class FaissSVSFlatSearchContext implements KNNLibrarySearchContext {

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        // SVS Flat is exhaustive search with no tunable search parameters
        return Collections.emptyMap();
    }
}
