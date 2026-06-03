/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.MethodComponent;

import java.util.Map;

/**
 * SPI for contributing Faiss methods at runtime via {@link java.util.ServiceLoader}. Lets experimental engines
 * in the {@code :sandbox} module register with {@link Faiss} and {@link FaissMethodResolver} without the main
 * module holding any compile-time reference to the sandbox classes.
 */
public interface SandboxFaissMethodProvider {

    /** Method name to {@link KNNMethod}, merged into the Faiss METHODS map. */
    Map<String, KNNMethod> methods();

    /** Method name to its {@link MethodComponent}, used by the method resolver. */
    Map<String, MethodComponent> methodComponents();

    /** Method name to its supported encoder map, used by the method resolver. */
    Map<String, Map<String, Encoder>> encoderMaps();
}
