/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Map;
import java.util.Collections;

import static org.opensearch.knn.common.KNNConstants.METHOD_SVS_FLAT;
import static org.opensearch.knn.common.KNNConstants.SVS_PARAMETER_COMPRESSION;

public class FaissSVSFlatMethodTests extends KNNTestCase {

    public void testMethodComponent_hasSupportedDataTypes() {
        MethodComponent component = FaissSVSFlatMethod.SVS_FLAT_COMPONENT;
        assertTrue("Should support FLOAT", component.getSupportedDataTypes().contains(VectorDataType.FLOAT));
        assertEquals("Should only support FLOAT", 1, component.getSupportedDataTypes().size());
    }

    public void testMethodComponent_hasSupportedSpaces() {
        FaissSVSFlatMethod method = new FaissSVSFlatMethod();
        assertTrue("Should support L2", method.getSupportedSpaces().contains(SpaceType.L2));
        assertTrue("Should support INNER_PRODUCT", method.getSupportedSpaces().contains(SpaceType.INNER_PRODUCT));
        assertTrue("Should support COSINESIMIL", method.getSupportedSpaces().contains(SpaceType.COSINESIMIL));
    }

    public void testMethodComponent_hasCompressionParameter() {
        MethodComponent component = FaissSVSFlatMethod.SVS_FLAT_COMPONENT;
        assertTrue("Should have compression parameter", component.getParameters().containsKey(SVS_PARAMETER_COMPRESSION));
        
        // Test parameter validation
        String defaultValue = (String) component.getParameters().get(SVS_PARAMETER_COMPRESSION).getDefaultValue();
        assertEquals("Default should be uncompressed", FaissSVSFlatMethod.COMPRESSION_UNCOMPRESSED, defaultValue);
    }

    public void testGetInstance() {
        FaissSVSFlatMethod instance1 = FaissSVSFlatMethod.getInstance();
        FaissSVSFlatMethod instance2 = FaissSVSFlatMethod.getInstance();
        
        assertNotNull("Instance should not be null", instance1);
        assertNotNull("Instance should not be null", instance2);
        // Note: getInstance creates new instances, so they won't be the same object
    }

    public void testCompressionConstants() {
        // Test that compression constants are properly defined
        assertEquals("uncompressed", FaissSVSFlatMethod.COMPRESSION_UNCOMPRESSED);
        assertEquals("LVQ4x0", FaissSVSFlatMethod.COMPRESSION_LVQ4x0);
        assertEquals("LVQ4x4", FaissSVSFlatMethod.COMPRESSION_LVQ4x4);
        assertEquals("LVQ4x8", FaissSVSFlatMethod.COMPRESSION_LVQ4x8);
        assertEquals("LeanVec4x4", FaissSVSFlatMethod.COMPRESSION_LEANVEC4x4);
        assertEquals("LeanVec4x8", FaissSVSFlatMethod.COMPRESSION_LEANVEC4x8);
        assertEquals("LeanVec8x8", FaissSVSFlatMethod.COMPRESSION_LEANVEC8x8);
    }
}
