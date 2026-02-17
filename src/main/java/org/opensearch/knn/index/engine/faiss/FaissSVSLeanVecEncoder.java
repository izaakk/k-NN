/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_ENCODER_LEANVEC;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_LEANVEC_PRIMARY_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_LEANVEC_RESIDUAL_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_LEANVEC_DIMENSIONS;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_LEANVEC_INITIAL_TRAINING_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_DEFAULT_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_DEFAULT_INITIAL_THRESHOLD;

/**
 * LeanVec encoder for SVS indexes. Uses learned linear projections for adaptive compression.
 * Training is deferred to segment merges when cumulative vector count crosses the configured threshold.
 */
public class FaissSVSLeanVecEncoder implements Encoder {

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(FAISS_SVS_ENCODER_LEANVEC)
        .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
        .addParameter(
            METHOD_PARAMETER_LEANVEC_PRIMARY_BITS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LEANVEC_PRIMARY_BITS, 4, (v, context) -> v >= 1 && v <= 8)
        )
        .addParameter(
            METHOD_PARAMETER_LEANVEC_RESIDUAL_BITS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LEANVEC_RESIDUAL_BITS, 4, (v, context) -> v >= 0 && v <= 8)
        )
        .addParameter(
            METHOD_PARAMETER_LEANVEC_DIMENSIONS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LEANVEC_DIMENSIONS, 0, (v, context) -> v >= 0)
        )
        .addParameter(
            METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD,
            new Parameter.IntegerParameter(
                METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD,
                DEFERRED_TRAINING_DEFAULT_THRESHOLD,
                (v, context) -> v == 0 || v >= 1000
            )
        )
        .addParameter(
            METHOD_PARAMETER_LEANVEC_INITIAL_TRAINING_THRESHOLD,
            new Parameter.IntegerParameter(
                METHOD_PARAMETER_LEANVEC_INITIAL_TRAINING_THRESHOLD,
                DEFERRED_TRAINING_DEFAULT_INITIAL_THRESHOLD,
                (v, context) -> v == 0 || v >= 1000
            )
        )
        .setKnnLibraryIndexingContextGenerator(
            ((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                MethodAsMapBuilder builder = MethodAsMapBuilder.builder(
                    "LeanVec",
                    methodComponent,
                    methodComponentContext,
                    knnMethodConfigContext
                );
                // Builds index description in format: LeanVec{primary}x{residual}[_{dims}]
                builder.addParameter(METHOD_PARAMETER_LEANVEC_PRIMARY_BITS, "", "x");
                builder.addParameter(METHOD_PARAMETER_LEANVEC_RESIDUAL_BITS, "", "");

                Object dimensions = methodComponentContext.getParameters().get(METHOD_PARAMETER_LEANVEC_DIMENSIONS);
                if (dimensions instanceof Integer && (Integer) dimensions > 0) {
                    builder.addParameter(METHOD_PARAMETER_LEANVEC_DIMENSIONS, "_", "");
                }

                return builder.build();
            })
        )
        .build();

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext encoderContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        return CompressionLevel.NOT_CONFIGURED;
    }
}
