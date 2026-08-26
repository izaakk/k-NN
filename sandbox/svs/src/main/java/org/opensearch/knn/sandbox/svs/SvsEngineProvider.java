/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.index.engine.NativeEngineService;
import org.opensearch.knn.index.mapper.EngineFieldStrategy;
import org.opensearch.knn.index.mapper.FaissFieldStrategy;

import java.util.Set;

/**
 * {@link KNNEngineDefinition} for the experimental Intel SVS engine. Discovered by the core
 * {@code KNNEngineRegistry} via {@code META-INF/services}, it registers as a first-class {@code KNNEngine}
 * (resolved by name); it supplies its identity: the engine name {@code "svs"}, the {@link SvsLibrary} (methods/extension/scoring/
 * capabilities), and the native index lifecycle ({@link SvsNativeEngineService}, backed by the isolated
 * {@code libopensearchknn_svs}).
 *
 * <p>{@link #engineName()} and {@link #library()} are evaluated when {@code KNNEngine} initializes; they do
 * not touch {@link SvsService}, so discovery alone never loads the native library (which loads lazily on the
 * first native call).
 */
public class SvsEngineProvider implements KNNEngineDefinition {

    private final NativeEngineService nativeService = new SvsNativeEngineService();

    @Override
    public String engineName() {
        return SVSConstants.SVS_ENGINE_NAME;
    }

    @Override
    public KNNLibrary library() {
        return SvsLibrary.INSTANCE;
    }

    @Override
    public NativeEngineService nativeService() {
        return nativeService;
    }

    @Override
    public EngineFieldStrategy fieldStrategy() {
        // SVS indexes are faiss-format and score like faiss (SvsLibrary delegates scoring/threshold
        // translation to Faiss), so the faiss field strategy applies unchanged.
        return FaissFieldStrategy.INSTANCE;
    }

    @Override
    public Set<String> engineSpecificQueryParameters() {
        // Parse-time allowlist for the engine's query method_parameters; the value rules live in
        // FaissSVSVamanaSearchContext, which KNNQueryBuilder#doToQuery validates against.
        return Set.of(SVSConstants.METHOD_PARAMETER_SEARCH_WINDOW_SIZE, SVSConstants.METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY);
    }
}
