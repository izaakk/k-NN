/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox;

import org.opensearch.test.OpenSearchTestCase;

/**
 * Tests for {@link ExperimentalAlgorithm} marker annotation: it must be readable at runtime (RUNTIME
 * retention is what lets tooling and reviewers discover experimental classes reflectively) with its
 * attributes and defaults intact.
 */
public class ExperimentalAlgorithmTests extends OpenSearchTestCase {

    @ExperimentalAlgorithm(description = "Test algorithm for validation", since = "3.7.0")
    private static class SampleExperimentalAlgorithm {
        // Intentionally empty — used only to verify annotation behavior
    }

    @ExperimentalAlgorithm
    private static class MinimalExperimentalAlgorithm {
        // Intentionally empty — verifies default annotation values
    }

    public void testAnnotationIsReadableAtRuntimeWithAttributes() {
        assertTrue(SampleExperimentalAlgorithm.class.isAnnotationPresent(ExperimentalAlgorithm.class));
        ExperimentalAlgorithm annotation = SampleExperimentalAlgorithm.class.getAnnotation(ExperimentalAlgorithm.class);
        assertNotNull("RUNTIME retention must make the annotation reflectively visible", annotation);
        assertEquals("Test algorithm for validation", annotation.description());
        assertEquals("3.7.0", annotation.since());
    }

    public void testAnnotationDefaultValues() {
        ExperimentalAlgorithm annotation = MinimalExperimentalAlgorithm.class.getAnnotation(ExperimentalAlgorithm.class);
        assertNotNull("Annotation should not be null", annotation);
        assertEquals("Default description should be empty", "", annotation.description());
        assertEquals("Default since should be empty", "", annotation.since());
    }
}
