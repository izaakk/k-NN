/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_ENCODER_LEANVEC;
import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_ENCODER_LVQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_ENCODER_SQ8;
import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_VAMANA_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_DEGREE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SEARCH_WINDOW_SIZE;
import static org.opensearch.knn.common.KNNConstants.METHOD_SVS_VAMANA;

/**
 * SVS Vamana method implementation. Provides graph-based approximate search using
 * the Vamana graph construction algorithm.
 */
public class FaissSVSVamanaMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT);

    /**
     * Supported encoders for SVS Vamana.
     * FLAT (no compression), FP16, SQ8, LVQ, and LeanVec.
     */
    public final static Map<String, Encoder> SUPPORTED_ENCODERS = Map.of(
        ENCODER_FLAT,
        new FaissFlatEncoder(),
        FAISS_SVS_ENCODER_FP16,
        new FaissSVSFP16Encoder(),
        FAISS_SVS_ENCODER_SQ8,
        new FaissSVSSQ8Encoder(),
        FAISS_SVS_ENCODER_LVQ,
        new FaissSVSLVQEncoder(),
        FAISS_SVS_ENCODER_LEANVEC,
        new FaissSVSLeanVecEncoder()
    );

    private final static MethodComponentContext DEFAULT_ENCODER_CONTEXT = new MethodComponentContext(
        ENCODER_FLAT,
        Collections.emptyMap()
    );

    final static MethodComponent METHOD_COMPONENT = initMethodComponent();

    public FaissSVSVamanaMethod() {
        super(METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new FaissSVSVamanaSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_SVS_VAMANA)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(
                METHOD_PARAMETER_DEGREE,
                new Parameter.IntegerParameter(METHOD_PARAMETER_DEGREE, 64, (v, context) -> v > 0 && v <= 256)
            )
            .addParameter(
                METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_CONSTRUCTION_WINDOW_SIZE,
                    (v, context) -> v > 0
                )
            )
            .addParameter(
                METHOD_PARAMETER_SEARCH_WINDOW_SIZE,
                new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_WINDOW_SIZE, 10, (v, context) -> v > 0)
            )
            .addParameter(
                METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY,
                new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY, 10, (v, context) -> v > 0)
            )
            .addParameter(
                METHOD_ENCODER_PARAMETER,
                new Parameter.MethodComponentContextParameter(
                    METHOD_ENCODER_PARAMETER,
                    DEFAULT_ENCODER_CONTEXT,
                    SUPPORTED_ENCODERS.values().stream().collect(Collectors.toMap(Encoder::getName, Encoder::getMethodComponent))
                )
            )
            .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                // Build index description: "SVSVamana{degree}" or "SVSVamana{degree},{encoder}"
                MethodAsMapBuilder methodAsMapBuilder = MethodAsMapBuilder.builder(
                    FAISS_SVS_VAMANA_DESCRIPTION,
                    methodComponent,
                    methodComponentContext,
                    knnMethodConfigContext
                );

                // Add degree parameter to the index description (e.g., "64")
                methodAsMapBuilder.addParameter(METHOD_PARAMETER_DEGREE, "", "");

                // Encoder parameter must be processed for JSON serialization.
                methodAsMapBuilder.addParameter(METHOD_ENCODER_PARAMETER, ",", "");

                // Remove ",Flat" suffix (Flat is default, excluded from description).
                Map<String, Object> parameters = methodComponentContext.getParameters();
                Object encoderParam = parameters.get(METHOD_ENCODER_PARAMETER);
                if (encoderParam instanceof MethodComponentContext) {
                    MethodComponentContext encoderContext = (MethodComponentContext) encoderParam;
                    if (ENCODER_FLAT.equals(encoderContext.getName())) {
                        String desc = methodAsMapBuilder.indexDescription;
                        if (desc.endsWith(",Flat")) {
                            methodAsMapBuilder.indexDescription = desc.substring(0, desc.length() - ",Flat".length());
                        }
                    }
                }

                return methodAsMapBuilder.build();
            }))
            .build();
    }
}
