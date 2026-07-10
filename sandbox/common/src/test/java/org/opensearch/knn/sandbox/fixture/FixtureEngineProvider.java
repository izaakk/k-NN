/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.jni.NativeEngineService;

import java.util.Set;

/**
 * {@link KNNEngineDefinition} for the test-only fixture engine, discovered by the core
 * {@code KNNEngineRegistry} via the {@code META-INF/services} entry in the sandbox TEST resources — so the
 * fixture registers only on the sandbox test classpath and can never appear in a shipped artifact.
 *
 * <p>This class is also the tutorial's reference for the registration step: a real tenant's provider looks
 * exactly like this, in the tenant's MAIN sources, with the service file under the tenant's main resources.
 */
public final class FixtureEngineProvider implements KNNEngineDefinition {

    @Override
    public String engineName() {
        return FixtureConstants.FIXTURE_ENGINE_NAME;
    }

    @Override
    public KNNLibrary library() {
        return FixtureLibrary.INSTANCE;
    }

    @Override
    public NativeEngineService nativeService() {
        return FixtureNativeEngineService.INSTANCE;
    }

    @Override
    public Set<String> engineSpecificQueryParameters() {
        // Parse-time allowlist for the engine's query method_parameters; the value rules live in
        // FixtureSearchContext, which KNNQueryBuilder#doToQuery validates against.
        return Set.of(FixtureConstants.METHOD_PARAMETER_FIXTURE_WINDOW);
    }
}
