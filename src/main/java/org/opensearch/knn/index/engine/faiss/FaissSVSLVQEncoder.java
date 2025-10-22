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
    }

    /**
     * Generates the LVQ string for index description.
     * Format: "LVQ{primary_bits}x{residual_bits}"
     * Examples: "LVQ4x4", "LVQ4x8", "LVQ8x8"
     */
    public static String getLVQString(Map<String, Object> parameters) {
        int primaryBits = parameters.containsKey(PRIMARY_BITS) 
            ? (Integer) parameters.get(PRIMARY_BITS) 
            : DEFAULT_PRIMARY_BITS;
        int residualBits = parameters.containsKey(RESIDUAL_BITS) 
            ? (Integer) parameters.get(RESIDUAL_BITS) 
            : DEFAULT_RESIDUAL_BITS;
        return String.format("LVQ%dx%d", primaryBits, residualBits);
    }
}
