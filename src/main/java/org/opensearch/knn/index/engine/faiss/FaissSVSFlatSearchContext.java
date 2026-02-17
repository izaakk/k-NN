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
 * Search context for SVS Flat method. No search-time tuning parameters.
 */
public final class FaissSVSFlatSearchContext implements KNNLibrarySearchContext {

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return Collections.emptyMap();
    }
}
