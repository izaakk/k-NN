/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import lombok.extern.slf4j.Slf4j;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.DefaultHnswSearchContext;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_SVS_FLAT;
import static org.opensearch.knn.common.KNNConstants.SVS_PARAMETER_COMPRESSION;

/**
 * Intel SVS Flat method implementation for Faiss
 * Provides exhaustive search with optional compression
 */
@Slf4j
public class FaissSVSFlatMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(
        VectorDataType.FLOAT
    );

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.L2,
        SpaceType.INNER_PRODUCT,
        SpaceType.COSINESIMIL
    );

    // SVS compression techniques
    public static final String COMPRESSION_UNCOMPRESSED = "uncompressed";
    public static final String COMPRESSION_LVQ4x0 = "LVQ4x0";
    public static final String COMPRESSION_LVQ4x4 = "LVQ4x4";
    public static final String COMPRESSION_LVQ4x8 = "LVQ4x8";
    public static final String COMPRESSION_LEANVEC4x4 = "LeanVec4x4";
    public static final String COMPRESSION_LEANVEC4x8 = "LeanVec4x8";
    public static final String COMPRESSION_LEANVEC8x8 = "LeanVec8x8";

    // Package private so that the method resolving logic can access the methods
    final static Encoder FLAT_ENCODER = new FaissFlatEncoder();
    final static Map<String, Encoder> SUPPORTED_ENCODERS = Map.of(
        FLAT_ENCODER.getName(),
        FLAT_ENCODER
    );
    
    public final static MethodComponent SVS_FLAT_COMPONENT = initMethodComponent();

    /**
     * Constructor for FaissSVSFlatMethod
     *
     * @see AbstractKNNMethod
     */
    public FaissSVSFlatMethod() {
        super(SVS_FLAT_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new DefaultHnswSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_SVS_FLAT)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(
                SVS_PARAMETER_COMPRESSION,
                new Parameter.StringParameter(SVS_PARAMETER_COMPRESSION, COMPRESSION_UNCOMPRESSED, (v, context) -> 
                    COMPRESSION_UNCOMPRESSED.equals(v) ||
                    COMPRESSION_LVQ4x0.equals(v) ||
                    COMPRESSION_LVQ4x4.equals(v) ||
                    COMPRESSION_LVQ4x8.equals(v) ||
                    COMPRESSION_LEANVEC4x4.equals(v) ||
                    COMPRESSION_LEANVEC4x8.equals(v) ||
                    COMPRESSION_LEANVEC8x8.equals(v)
                )
            )
            .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                String compression = (String) methodComponentContext.getParameters()
                    .getOrDefault(SVS_PARAMETER_COMPRESSION, COMPRESSION_UNCOMPRESSED);
                
                String indexDescription;
                if (COMPRESSION_UNCOMPRESSED.equals(compression)) {
                    indexDescription = "SVS,Flat";
                } else {
                    indexDescription = "SVS,Flat," + compression;
                }
                
                MethodAsMapBuilder methodAsMapBuilder = MethodAsMapBuilder.builder(
                    indexDescription,
                    methodComponent,
                    methodComponentContext,
                    knnMethodConfigContext
                );
                return adjustIndexDescription(methodAsMapBuilder, methodComponentContext, knnMethodConfigContext);
            }))
            .build();
    }

    public static FaissSVSFlatMethod getInstance() {
        return new FaissSVSFlatMethod();
    }
}
