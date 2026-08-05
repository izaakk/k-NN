/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.EngineQueryParameter;
import org.opensearch.knn.index.engine.EngineQueryParameter.ValueType;
import org.opensearch.knn.index.engine.KNNEngineContext;
import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;

import java.util.Set;

/**
 * A well-formed definition that declares two query parameters sharing one name, a within-engine duplicate.
 * The registry must skip the engine at validation and never call its initialize.
 */
public final class MisdeclaredParamFixtureEngineProvider implements KNNEngineDefinition {

    static volatile boolean initialized = false;

    @Override
    public String engineName() {
        return "misdeclared-param";
    }

    @Override
    public KNNLibrary library() {
        return PlainFixtureLibrary.INSTANCE;
    }

    @Override
    public Set<EngineQueryParameter> engineSpecificQueryParameters() {
        return Set.of(new EngineQueryParameter("dup", ValueType.INTEGER), new EngineQueryParameter("dup", ValueType.STRING));
    }

    @Override
    public void initialize(KNNEngineContext context) {
        initialized = true;
    }
}
