/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marker annotation indicating that a class is an experimental algorithm or component
 * within the k-NN sandbox module. Annotated classes are not part of the stable,
 * production-ready API surface: they may change, be renamed, or be removed without
 * prior notice in any future release. Metadata: {@link #description()} and {@link #since()}.
 *
 * <p>Usage example:</p>
 * <pre>{@code
 * @ExperimentalAlgorithm(description = "Vamana graph-based ANN", since = "3.8.0")
 * public class VamanaAlgorithm { ... }
 * }</pre>
 *
 * <p>See sandbox/README.md for the sandbox lifecycle and graduation criteria.</p>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target({ ElementType.TYPE })
public @interface ExperimentalAlgorithm {

    /**
     * A brief description of the experimental algorithm or component.
     *
     * @return the description string, empty by default
     */
    String description() default "";

    /**
     * The version in which this experimental component was first introduced.
     *
     * @return the version string (e.g., "3.8.0"), empty by default
     */
    String since() default "";
}
