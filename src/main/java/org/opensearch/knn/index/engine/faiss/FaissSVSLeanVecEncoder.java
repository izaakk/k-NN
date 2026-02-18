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
 * LeanVec (Adaptive vector quantization) encoder for SVS indexes.
 * Provides adaptive compression with optional dimension specification.
 * 
 * Parameters:
 * - primary_bits: bits for primary quantization (1-8, default: 4)
 * - residual_bits: bits for residual quantization (0-8, default: 4)
 * - dimensions: dimension count (0 = use all, default: 0, nullable)
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
                (v, context) -> v == 0 || v >= 1000  // 0=use default, otherwise must be >= 1000
            )
        )
        .addParameter(
            METHOD_PARAMETER_LEANVEC_INITIAL_TRAINING_THRESHOLD,
            new Parameter.IntegerParameter(
                METHOD_PARAMETER_LEANVEC_INITIAL_TRAINING_THRESHOLD,
                DEFERRED_TRAINING_DEFAULT_INITIAL_THRESHOLD,
                (v, context) -> v == 0 || v >= 1000  // 0=use default, otherwise must be >= 1000
            )
        )
        // LeanVec uses automatic deferred per-shard training during segment merges.
        .setKnnLibraryIndexingContextGenerator(
            ((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                MethodAsMapBuilder builder = MethodAsMapBuilder.builder(
                    "LeanVec",  // Faiss expects "LeanVec", not "leanvec"
                    methodComponent,
                    methodComponentContext,
                    knnMethodConfigContext
                );
                // Builds "LeanVec4x4" format (primary_bits "4x" + residual_bits "4")
                builder.addParameter(METHOD_PARAMETER_LEANVEC_PRIMARY_BITS, "", "x");
                builder.addParameter(METHOD_PARAMETER_LEANVEC_RESIDUAL_BITS, "", "");
                
                // If dimensions is specified (non-zero), append it: "LeanVec4x4_128"
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
        // LeanVec has adaptive compression, so return NOT_CONFIGURED
        return CompressionLevel.NOT_CONFIGURED;
    }
}
