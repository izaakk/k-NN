/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;

import static org.opensearch.knn.common.KNNConstants.METHOD_SVS_FLAT;

/**
 * Unit tests for FaissSVSFlatMethod
 */
public class FaissSVSFlatMethodTests extends KNNTestCase {

    private FaissSVSFlatMethod svsMethod;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        svsMethod = new FaissSVSFlatMethod();
    }

    /**
     * Test that the method supports L2 space type
     */
    public void testSupportedSpaceTypes_whenL2_thenSupported() {
        assertTrue("SVS Flat should support L2 space type", FaissSVSFlatMethod.SUPPORTED_SPACES.contains(SpaceType.L2));
    }

    /**
     * Test that the method supports INNER_PRODUCT space type
     */
    public void testSupportedSpaceTypes_whenInnerProduct_thenSupported() {
        assertTrue(
            "SVS Flat should support INNER_PRODUCT space type",
            FaissSVSFlatMethod.SUPPORTED_SPACES.contains(SpaceType.INNER_PRODUCT)
        );
    }

    /**
     * Test that the method does not support COSINESIMIL space type
     */
    public void testSupportedSpaceTypes_whenCosine_thenNotSupported() {
        assertFalse(
            "SVS Flat should not support COSINESIMIL space type",
            FaissSVSFlatMethod.SUPPORTED_SPACES.contains(SpaceType.COSINESIMIL)
        );
    }

    /**
     * Test that method component is correctly initialized
     */
    public void testMethodComponent_whenInitialized_thenNotNull() {
        assertNotNull("Method component should not be null", FaissSVSFlatMethod.METHOD_COMPONENT);
    }

    /**
     * Test that method name is correctly set
     */
    public void testMethodName_whenAccessed_thenReturnsSVSFlat() {
        assertEquals("Method component name should be svs_flat", METHOD_SVS_FLAT, FaissSVSFlatMethod.METHOD_COMPONENT.getName());
    }

    /**
     * Test that supported spaces list contains exactly L2 and INNER_PRODUCT
     */
    public void testSupportedSpaces_whenChecked_thenContainsExactlyTwoSpaces() {
        assertEquals("SVS Flat should support exactly 2 space types", 2, FaissSVSFlatMethod.SUPPORTED_SPACES.size());
    }
}
