/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.bruteforce;

import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Arrays;

import org.opensearch.knn.index.VectorQueryType;
import org.opensearch.knn.index.engine.model.QueryContext;

import static org.opensearch.knn.sandbox.bruteforce.BruteForceLibrary.ENGINE_NAME;
import static org.opensearch.knn.sandbox.bruteforce.BruteForceLibrary.EXTENSION;
import static org.opensearch.knn.sandbox.bruteforce.BruteForceLibrary.SCAN_LIMIT;

/**
 * Registration through the SPI: present on the classpath via META-INF/services, the example engine becomes
 * a first-class {@code KNNEngine}. The native lifecycle itself is exercised against a real cluster (the
 * tenant smoke test); these tests deliberately never touch {@code BruteForceService}, whose class load
 * would pull in the native library.
 */
public class BruteForceEngineTests extends OpenSearchTestCase {

    public void testEngineRegistersAsFirstClassKNNEngine() {
        final KNNEngine engine = KNNEngine.getEngine(ENGINE_NAME);
        assertNotNull(engine);
        assertEquals(ENGINE_NAME, engine.getName());
        assertTrue(Arrays.asList(KNNEngine.values()).contains(engine));
        assertNotNull(engine.getNativeService());
        assertEquals(EXTENSION, engine.getExtension());
    }

    public void testScanLimitIsDeclared() {
        final KNNEngine engine = KNNEngine.getEngine(ENGINE_NAME);
        final var parameters = engine.getKNNLibrarySearchContext("scan").supportedMethodParameters(new QueryContext(VectorQueryType.K));
        assertTrue(parameters.containsKey(SCAN_LIMIT));
    }

    public void testCapabilityFlagsAreFolded() {
        final KNNEngine engine = KNNEngine.getEngine(ENGINE_NAME);
        assertTrue(engine.supportsIterativeBuild());
        assertTrue(engine.createsCustomSegmentFiles());
        assertFalse(engine.supportsRadialSearch());
        assertFalse(engine.supportsNestedFields());
        assertFalse(engine.supportsFilters());
    }
}
