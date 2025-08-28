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

import static org.opensearch.knn.common.KNNConstants.METHOD_SVS_VAMANA;
import static org.opensearch.knn.common.KNNConstants.SVS_PARAMETER_DEGREE;
import static org.opensearch.knn.common.KNNConstants.SVS_PARAMETER_COMPRESSION;

public class FaissSVSVamanaMethodTests extends KNNTestCase {

    public void testMethodComponent_hasSupportedDataTypes() {
        MethodComponent component = FaissSVSVamanaMethod.SVS_VAMANA_COMPONENT;
        assertTrue("Should support FLOAT", component.getSupportedDataTypes().contains(VectorDataType.FLOAT));
        assertEquals("Should only support FLOAT", 1, component.getSupportedDataTypes().size());
    }

    public void testMethodComponent_hasSupportedSpaces() {
        FaissSVSVamanaMethod method = new FaissSVSVamanaMethod();
        assertTrue("Should support L2", method.getSupportedSpaces().contains(SpaceType.L2));
        assertTrue("Should support INNER_PRODUCT", method.getSupportedSpaces().contains(SpaceType.INNER_PRODUCT));
        assertTrue("Should support COSINESIMIL", method.getSupportedSpaces().contains(SpaceType.COSINESIMIL));
    }

    public void testMethodComponent_hasDegreeParameter() {
        MethodComponent component = FaissSVSVamanaMethod.SVS_VAMANA_COMPONENT;
        assertTrue("Should have degree parameter", component.getParameters().containsKey(SVS_PARAMETER_DEGREE));
        
        // Test parameter validation
        Integer defaultValue = (Integer) component.getParameters().get(SVS_PARAMETER_DEGREE).getDefaultValue();
        assertEquals("Default should be 64", Integer.valueOf(64), defaultValue);
    }

    public void testMethodComponent_hasCompressionParameter() {
        MethodComponent component = FaissSVSVamanaMethod.SVS_VAMANA_COMPONENT;
        assertTrue("Should have compression parameter", component.getParameters().containsKey(SVS_PARAMETER_COMPRESSION));
        
        // Test parameter validation
        String defaultValue = (String) component.getParameters().get(SVS_PARAMETER_COMPRESSION).getDefaultValue();
        assertEquals("Default should be uncompressed", FaissSVSVamanaMethod.COMPRESSION_UNCOMPRESSED, defaultValue);
    }

    public void testGetInstance() {
        FaissSVSVamanaMethod instance1 = FaissSVSVamanaMethod.getInstance();
        FaissSVSVamanaMethod instance2 = FaissSVSVamanaMethod.getInstance();
        
        assertNotNull("Instance should not be null", instance1);
        assertNotNull("Instance should not be null", instance2);
        // Note: getInstance creates new instances, so they won't be the same object
    }

    public void testCompressionConstants() {
        // Test that compression constants are properly defined
        assertEquals("uncompressed", FaissSVSVamanaMethod.COMPRESSION_UNCOMPRESSED);
        assertEquals("LVQ4x0", FaissSVSVamanaMethod.COMPRESSION_LVQ4x0);
        assertEquals("LVQ4x4", FaissSVSVamanaMethod.COMPRESSION_LVQ4x4);
        assertEquals("LVQ4x8", FaissSVSVamanaMethod.COMPRESSION_LVQ4x8);
        assertEquals("LeanVec4x4", FaissSVSVamanaMethod.COMPRESSION_LEANVEC4x4);
        assertEquals("LeanVec4x8", FaissSVSVamanaMethod.COMPRESSION_LEANVEC4x8);
        assertEquals("LeanVec8x8", FaissSVSVamanaMethod.COMPRESSION_LEANVEC8x8);
    }

    public void testDegreeParameterValidation() {
        // Test that degree parameter accepts valid values
        MethodComponent component = FaissSVSVamanaMethod.SVS_VAMANA_COMPONENT;
        Parameter degreeParam = component.getParameters().get(SVS_PARAMETER_DEGREE);
        
        // Valid degrees should be between 32 and 128
        assertTrue("Should accept degree 32", degreeParam.validate(32, null));
        assertTrue("Should accept degree 64", degreeParam.validate(64, null));
        assertTrue("Should accept degree 96", degreeParam.validate(96, null));
        assertTrue("Should accept degree 128", degreeParam.validate(128, null));
        
        // Invalid degrees should be rejected
        assertFalse("Should reject degree 16", degreeParam.validate(16, null));
        assertFalse("Should reject degree 256", degreeParam.validate(256, null));
    }
}
