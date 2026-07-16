/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Incubation environment for experimental algorithms, optimizations, and architectural
 * components within the OpenSearch k-NN plugin.
 *
 * <p>All classes in this package and its sub-packages are <strong>experimental</strong>. Sandbox jars
 * are bundled only into snapshot builds explicitly flagged with {@code -Pknn.sandbox.enabled=true};
 * they are never included in release artifacts.</p>
 *
 * <p>Every public class in this package should be annotated with
 * {@link org.opensearch.knn.sandbox.ExperimentalAlgorithm @ExperimentalAlgorithm}.
 * See sandbox/README.md for the sandbox lifecycle and graduation criteria.</p>
 *
 * @see org.opensearch.knn.sandbox.ExperimentalAlgorithm
 */
package org.opensearch.knn.sandbox;
