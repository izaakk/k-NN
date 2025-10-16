/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.Parameter;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_SVS_FLAT;

/**
 * SVS Flat method implementation. Provides exhaustive (brute force) search with optional compression.
 * 
 * Supported index descriptions:
 * - "SVSFlat" (FP32, no compression)
 * - "SVSFlat,FP16" (FP16 compression, 2x reduction)
 * 
 * Note: IndexSVSFlat does NOT support range_search(). Use SVS Vamana for range queries.
 */
public class FaissSVSFlatMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT);

    private final static MethodComponent.MethodComponentContext DEFAULT_ENCODER_CONTEXT = new MethodComponent.MethodComponentContext(
        ENCODER_FLAT,
        Collections.emptyMap()
    );

    final static MethodComponent METHOD_COMPONENT = initMethodComponent();

    /**
     * Constructor for FaissSVSFlatMethod
     */
    public FaissSVSFlatMethod() {
        super(METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), null);
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_SVS_FLAT)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(METHOD_ENCODER_PARAMETER, initEncoderParameter())
            .setKnnLibraryIndexingContextGenerator(
                ((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                    // Build index description: "SVSFlat" or "SVSFlat,FP16"
                    MethodAsMapBuilder methodAsMapBuilder = MethodAsMapBuilder.builder(
                        FAISS_SVS_FLAT_DESCRIPTION,
                        methodComponent,
                        methodComponentContext,
                        knnMethodConfigContext
                    ).addParameter(METHOD_ENCODER_PARAMETER, ",", "");

                    return methodAsMapBuilder.build();
                })
            )
            .build();
    }

    private static Parameter.MethodComponentContextParameter initEncoderParameter() {
        // For now, only support Flat encoder. FP16 encoder will be added in Phase 2
        return new Parameter.MethodComponentContextParameter(
            METHOD_ENCODER_PARAMETER,
            DEFAULT_ENCODER_CONTEXT,
            Collections.singletonMap(FaissHNSWMethod.FLAT_ENCODER.getName(), FaissHNSWMethod.FLAT_ENCODER.getMethodComponent())
        );
    }
}
