/*
 * Copyright OpenSearch Copublic class FaissSVSLVQEncoder implements Encoder {

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(FAISS_SVS_ENCODER_LVQ)
        .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
        .addParameter(
            METHOD_PARAMETER_LVQ_PRIMARY_BITS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LVQ_PRIMARY_BITS, 4, (v, context) -> v > 0 && v <= 8)
        )
        .addParameter(
            METHOD_PARAMETER_LVQ_RESIDUAL_BITS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LVQ_RESIDUAL_BITS, 4, (v, context) -> v > 0 && v <= 8)
        )
        .build(); SPDX-License-Identifier: Apache-2.0
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

import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_ENCODER_LVQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_LVQ_PRIMARY_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_LVQ_RESIDUAL_BITS;

/**
 * LVQ (Locally-adaptive Vector Quantization) encoder for SVS indexes.
 * Reduces memory by 8x with configurable bit allocation.
 * 
 * Parameters:
 * - primary_bits: bits for primary quantization (1-8, default: 4)
 * - residual_bits: bits for residual quantization (0-8, default: 4)
 */
public class FaissSVSLVQEncoder implements Encoder {

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(FAISS_SVS_ENCODER_LVQ)
        .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
        .addParameter(
            METHOD_PARAMETER_LVQ_PRIMARY_BITS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LVQ_PRIMARY_BITS, 4, (v, context) -> v >= 1 && v <= 8)
        )
        .addParameter(
            METHOD_PARAMETER_LVQ_RESIDUAL_BITS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LVQ_RESIDUAL_BITS, 4, (v, context) -> v >= 0 && v <= 8)
        )
        .setKnnLibraryIndexingContextGenerator(
            ((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                MethodAsMapBuilder builder = MethodAsMapBuilder.builder(
                    "LVQ",  // Faiss expects "LVQ", not "lvq"
                    methodComponent,
                    methodComponentContext,
                    knnMethodConfigContext
                );
                // Add primary_bits and residual_bits parameters: "LVQ4x4"
                // First adds "4x", second adds "4" → "LVQ4x4"
                builder.addParameter(METHOD_PARAMETER_LVQ_PRIMARY_BITS, "", "x");
                builder.addParameter(METHOD_PARAMETER_LVQ_RESIDUAL_BITS, "", "");
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
        return CompressionLevel.x8;
    }
}
