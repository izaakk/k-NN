/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.bruteforce;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.index.engine.NativeEngineService;

import static org.opensearch.knn.sandbox.bruteforce.BruteForceLibrary.ENGINE_NAME;

/**
 * {@link KNNEngineDefinition} of the example brute-force engine, discovered via
 * {@code META-INF/services}. Declares no engine-specific query parameters: the scan takes none.
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
}
