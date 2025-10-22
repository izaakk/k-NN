/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_ENCODER_LVQ;

/**
 * LVQ (Locally-adaptive Vector Quantization) encoder for SVS indexes.
 * Reduces memory by 8x with configurable bit allocation.
 * 
 * Parameters:
 * - primary_bits: 4 or 8 (default: 4)
 * - residual_bits: 0, 4, or 8 (default: 4)
 */
public class FaissSVSLVQEncoder implements Encoder {

    private static final String PRIMARY_BITS = "primary_bits";
    private static final String RESIDUAL_BITS = "residual_bits";
    private static final int DEFAULT_PRIMARY_BITS = 4;
    private static final int DEFAULT_RESIDUAL_BITS = 4;
    private static final Set<Integer> VALID_PRIMARY_BITS = Set.of(4, 8);
    private static final Set<Integer> VALID_RESIDUAL_BITS = Set.of(0, 4, 8);

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(FAISS_SVS_ENCODER_LVQ)
        .addParameter(
            PRIMARY_BITS,
            new Parameter.IntegerParameter(PRIMARY_BITS, DEFAULT_PRIMARY_BITS, (v, context) -> VALID_PRIMARY_BITS.contains(v))
        )
        .addParameter(
            RESIDUAL_BITS,
            new Parameter.IntegerParameter(RESIDUAL_BITS, DEFAULT_RESIDUAL_BITS, (v, context) -> VALID_RESIDUAL_BITS.contains(v))
        )
        .build();

    @Override
    public MethodComponent.MethodComponentContext getMethodComponent() {
        return METHOD_COMPONENT.getMethodComponentContext();
    }

    @Override
    public CompressionLevel getCompressionLevel() {
        return CompressionLevel.x8;
    }

    @Override
    public String getName() {
        return FAISS_SVS_ENCODER_LVQ;
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
