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
 * Search context for SVS Flat method. SVS Flat is an exhaustive search method that does not
 * support any search-time parameters (unlike HNSW which has ef_search).
 *
 * Phase 2 may add support for search parameters if needed.
 */
public final class FaissSVSFlatSearchContext implements KNNLibrarySearchContext {

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        // SVS Flat is exhaustive search with no tunable search parameters
        return Collections.emptyMap();
    }
}
