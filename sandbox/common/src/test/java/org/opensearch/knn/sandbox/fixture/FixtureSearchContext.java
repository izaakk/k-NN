/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.model.QueryContext;

import java.util.Map;

import static org.opensearch.knn.sandbox.fixture.FixtureConstants.METHOD_PARAMETER_FIXTURE_WINDOW;

/**
 * Search context for the fixture engine: declares the engine-specific query parameter
 * {@code fixture_window} (valid iff &gt; 0). This is the engine-side half of the query extension point —
 * the core defers parameters it does not know to this declaration, and the engine-aware validation in
 * {@code KNNQueryBuilder#doToQuery} judges them against it.
 */
public final class FixtureSearchContext implements KNNLibrarySearchContext {

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return Map.of(
            METHOD_PARAMETER_FIXTURE_WINDOW,
            new Parameter.IntegerParameter(METHOD_PARAMETER_FIXTURE_WINDOW, null, (v, context) -> v > 0)
        );
    }
}
