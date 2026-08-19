/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.bruteforce;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.index.engine.NativeEngineService;
import org.opensearch.knn.index.mapper.EngineFieldStrategy;
import org.opensearch.knn.index.mapper.FaissFieldStrategy;

import java.util.Set;

import static org.opensearch.knn.sandbox.bruteforce.BruteForceLibrary.ENGINE_NAME;
import static org.opensearch.knn.sandbox.bruteforce.BruteForceLibrary.SCAN_LIMIT;

/**
 * {@link KNNEngineDefinition} of the example brute-force engine, discovered via
 * {@code META-INF/services}. Declares one engine-specific query parameter ({@code scan_limit}), so the
 * REST/gRPC parsers accept the name and defer its validation to the engine's search context.
 */
public final class BruteForceEngineProvider implements KNNEngineDefinition {

    @Override
    public String engineName() {
        return ENGINE_NAME;
    }

    @Override
    public KNNLibrary library() {
        return BruteForceLibrary.INSTANCE;
    }

    @Override
    public NativeEngineService nativeService() {
        return BruteForceNativeEngineService.INSTANCE;
    }

    @Override
    public Set<String> engineSpecificQueryParameters() {
        return Set.of(SCAN_LIMIT);
    }

    @Override
    public EngineFieldStrategy fieldStrategy() {
        // The strategy shared by the native built-ins; brute-force fields are mapped and indexed exactly
        // like other native-engine fields, only the search-time path differs.
        return FaissFieldStrategy.INSTANCE;
    }
}
