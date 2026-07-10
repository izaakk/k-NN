/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.faiss_upstream;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.jni.NativeEngineService;

/**
 * {@link KNNEngineDefinition} for the experimental {@code faiss_upstream} engine, discovered by the core
 * {@code KNNEngineRegistry} via {@code META-INF/services}. Registration is lazy-safe: constructing the
 * {@link FaissUpstreamNativeEngineService} does not class-init {@link FaissUpstreamService}, so the native
 * library loads only on first native use.
 */
public class FaissUpstreamEngineProvider implements KNNEngineDefinition {

    private final NativeEngineService nativeService = new FaissUpstreamNativeEngineService();

    @Override
    public String engineName() {
        return FaissUpstreamConstants.FAISS_UPSTREAM_ENGINE_NAME;
    }

    @Override
    public KNNLibrary library() {
        return FaissUpstreamLibrary.INSTANCE;
    }

    @Override
    public NativeEngineService nativeService() {
        return nativeService;
    }
}
