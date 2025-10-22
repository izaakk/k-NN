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

import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_ENCODER_LEANVEC;

/**
 * LeanVec (Adaptive vector quantization) encoder for SVS indexes.
 * Provides adaptive compression with optional dimension specification.
 * 
 * Parameters:
 * - primary_bits: 4 or 8 (default: 4)
 * - residual_bits: 4 or 8 (default: 4)
 * - dimensions: 0 (use all) or positive integer (default: 0)
 */
public class FaissSVSLeanVecEncoder implements Encoder {

    private static final String PRIMARY_BITS = "primary_bits";
    private static final String RESIDUAL_BITS = "residual_bits";
    private static final String DIMENSIONS = "dimensions";
    private static final int DEFAULT_PRIMARY_BITS = 4;
    private static final int DEFAULT_RESIDUAL_BITS = 4;
    private static final int DEFAULT_DIMENSIONS = 0;  // 0 means use all dimensions
    private static final Set<Integer> VALID_PRIMARY_BITS = Set.of(4, 8);
    private static final Set<Integer> VALID_RESIDUAL_BITS = Set.of(4, 8);

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(FAISS_SVS_ENCODER_LEANVEC)
        .addParameter(
            PRIMARY_BITS,
            new Parameter.IntegerParameter(PRIMARY_BITS, DEFAULT_PRIMARY_BITS, (v, context) -> VALID_PRIMARY_BITS.contains(v))
        )
        .addParameter(
            RESIDUAL_BITS,
            new Parameter.IntegerParameter(RESIDUAL_BITS, DEFAULT_RESIDUAL_BITS, (v, context) -> VALID_RESIDUAL_BITS.contains(v))
        )
        .addParameter(
            DIMENSIONS,
            new Parameter.IntegerParameter(DIMENSIONS, DEFAULT_DIMENSIONS, (v, context) -> v >= 0)
        )
        .build();

    @Override
    public MethodComponent.MethodComponentContext getMethodComponent() {
        return METHOD_COMPONENT.getMethodComponentContext();
    }

    @Override
    public CompressionLevel getCompressionLevel() {
        return CompressionLevel.x8;  // Adaptive, but typically 8x
    }

    @Override
    public String getName() {
        return FAISS_SVS_ENCODER_LEANVEC;
    }

    /**
     * Generates the LeanVec string for index description.
     * Format: "LeanVec{primary_bits}x{residual_bits}" or "LeanVec{primary_bits}x{residual_bits}_{dimensions}"
     * Examples: "LeanVec4x4", "LeanVec4x4_128", "LeanVec8x8_256"
     */
    public static String getLeanVecString(Map<String, Object> parameters) {
        int primaryBits = parameters.containsKey(PRIMARY_BITS) 
            ? (Integer) parameters.get(PRIMARY_BITS) 
            : DEFAULT_PRIMARY_BITS;
        int residualBits = parameters.containsKey(RESIDUAL_BITS) 
            ? (Integer) parameters.get(RESIDUAL_BITS) 
            : DEFAULT_RESIDUAL_BITS;
        int dimensions = parameters.containsKey(DIMENSIONS) 
            ? (Integer) parameters.get(DIMENSIONS) 
            : DEFAULT_DIMENSIONS;
        
        String base = String.format("LeanVec%dx%d", primaryBits, residualBits);
        if (dimensions > 0) {
            return base + "_" + dimensions;
        }
        return base;
    }
}
