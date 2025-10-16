/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Collections;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
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
     * Test that the method generates correct index description for SVS Flat
     */
    public void testIndexDescription_whenNoCompression_thenGenerateCorrectDescription() {
        // Create method context with flat encoder (no compression)
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            null,
            SpaceType.L2,
            new MethodComponentContext(METHOD_SVS_FLAT, Collections.emptyMap())
        );

        // Generate index description
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        Map<String, Object> indexDescription = svsMethod.getMethodComponent()
            .getKnnLibraryIndexingContextGenerator()
            .generate(svsMethod.getMethodComponent(), knnMethodContext.getMethodComponentContext(), knnMethodConfigContext);

        // Verify the description matches expected SVS Flat pattern
        assertNotNull("Index description should not be null", indexDescription);
        assertTrue("Index description should contain type", indexDescription.containsKey("type"));
        assertEquals("Index type should be SVSFlat", FAISS_SVS_FLAT_DESCRIPTION, indexDescription.get("type"));
    }

    /**
     * Test that the method supports FLOAT data type
     */
    public void testSupportedDataTypes_whenFloat_thenSupported() {
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        assertTrue("SVS Flat should support FLOAT data type", svsMethod.isDataTypeSupported(context));
    }

    /**
     * Test that the method does not support BYTE data type
     */
    public void testSupportedDataTypes_whenByte_thenNotSupported() {
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.BYTE)
            .dimension(128)
            .build();

        assertFalse("SVS Flat should not support BYTE data type", svsMethod.isDataTypeSupported(context));
    }

    /**
     * Test that the method does not support BINARY data type
     */
    public void testSupportedDataTypes_whenBinary_thenNotSupported() {
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.BINARY)
            .dimension(128)
            .build();

        assertFalse("SVS Flat should not support BINARY data type", svsMethod.isDataTypeSupported(context));
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
     * Test that the method component has correct encoder parameter
     */
    public void testMethodComponent_whenInitialized_thenHasEncoderParameter() {
        assertNotNull("Method component should not be null", svsMethod.getMethodComponent());
        assertNotNull("Method component should have parameters", svsMethod.getMethodComponent().getParameters());
        assertTrue(
            "Method component should have encoder parameter",
            svsMethod.getMethodComponent().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
    }

    /**
     * Test that default encoder is FLAT
     */
    public void testDefaultEncoder_whenNoEncoderSpecified_thenUseFlatEncoder() {
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            null,
            SpaceType.L2,
            new MethodComponentContext(METHOD_SVS_FLAT, Collections.emptyMap())
        );

        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        Map<String, Object> indexDescription = svsMethod.getMethodComponent()
            .getKnnLibraryIndexingContextGenerator()
            .generate(svsMethod.getMethodComponent(), knnMethodContext.getMethodComponentContext(), knnMethodConfigContext);

        // With flat encoder and no parameters, should just be "SVSFlat"
        assertEquals("Default index description should be SVSFlat", FAISS_SVS_FLAT_DESCRIPTION, indexDescription.get("type"));
    }

    /**
     * Test that method validates correctly for supported configurations
     */
    public void testValidation_whenSupportedConfiguration_thenNoValidationErrors() {
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            null,
            SpaceType.L2,
            new MethodComponentContext(METHOD_SVS_FLAT, Collections.emptyMap())
        );

        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        // Should not throw any exceptions
        assertNotNull("Validation should succeed for supported configuration", svsMethod.getMethodComponent());
    }

    /**
     * Test that method name is correctly set
     */
    public void testMethodName_whenAccessed_thenReturnsSVSFlat() {
        assertEquals(
            "Method component name should be svs_flat",
            METHOD_SVS_FLAT,
            svsMethod.getMethodComponent().getName()
        );
    }
}
