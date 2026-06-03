/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.faiss.SandboxFaissMethodProvider;
import org.opensearch.knn.sandbox.ExperimentalAlgorithm;

import java.util.Map;

import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_SVS_VAMANA;

/**
 * {@link SandboxFaissMethodProvider} implementation that registers the experimental Intel SVS
 * method ({@code svs_vamana}) with the main Faiss library at runtime.
 *
 * <p>Discovered via {@link java.util.ServiceLoader}; declared in
 * {@code META-INF/services/org.opensearch.knn.index.engine.faiss.SandboxFaissMethodProvider}.</p>
 */
@ExperimentalAlgorithm(description = "Intel SVS Vamana ServiceLoader provider", since = "3.7.0")
public class SVSMethodProvider implements SandboxFaissMethodProvider {

    @Override
    public Map<String, KNNMethod> methods() {
        return Map.of(METHOD_SVS_VAMANA, new FaissSVSVamanaMethod());
    }

    @Override
    public Map<String, MethodComponent> methodComponents() {
        return Map.of(METHOD_SVS_VAMANA, FaissSVSVamanaMethod.METHOD_COMPONENT);
    }

    @Override
    public Map<String, Map<String, Encoder>> encoderMaps() {
        return Map.of(METHOD_SVS_VAMANA, FaissSVSVamanaMethod.SUPPORTED_ENCODERS);
    }
}
