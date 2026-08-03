/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Locale;
import java.util.Set;

import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_ENCODER_LEANVEC;
import static org.opensearch.knn.sandbox.svs.SVSConstants.LEANVEC_DEFAULT_ROUGH_TRAINING_THRESHOLD;
import static org.opensearch.knn.sandbox.svs.SVSConstants.LEANVEC_DEFAULT_TRAINING_THRESHOLD;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LEANVEC_DIMENSIONS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LEANVEC_ROUGH_TRAINING_THRESHOLD;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_PRIMARY_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_RESIDUAL_BITS;

/**
 * LeanVec encoder for SVS indexes: a learned dimensionality-reducing projection over LVQ storage, trained
 * <b>deferred</b> at segment-build time (no Train API, no model registry). The native layer applies a
 * per-segment quality ladder driven by two thresholds: segments under {@code rough_training_threshold} build
 * as the equivalent LVQ index (training-free fallback); between the thresholds the projection trains on the
 * whole segment; at or above {@code training_threshold} it trains on a {@code training_threshold}-sized
 * sample. Merges re-train, so quality upgrades monotonically with segment size.
 *
 * <p>Parameters: {@code primary_bits}x{@code residual_bits} select the LeanVec storage kind (supported:
 * 4x4, 4x8, 8x8); {@code dimensions} sets the reduced dimensionality (0, the default, lets the SVS runtime
 * pick {@code dim/2}); the two threshold parameters tune the ladder (0 = use the built-in default).
 */
public class FaissSVSLeanVecEncoder implements Encoder {

    static final int DEFAULT_PRIMARY_BITS = 4;
    static final int DEFAULT_RESIDUAL_BITS = 8;

    /** Supported (primary_bits, residual_bits) combinations, mirroring the SVS {@code SVSStorageKind} LeanVec kinds. */
    private static final Set<String> SUPPORTED_BIT_COMBINATIONS = Set.of("4x4", "4x8", "8x8");

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(FAISS_SVS_ENCODER_LEANVEC)
        .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
        .addParameter(
            METHOD_PARAMETER_LVQ_PRIMARY_BITS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LVQ_PRIMARY_BITS, DEFAULT_PRIMARY_BITS, (v, context) -> v >= 1 && v <= 8)
        )
        .addParameter(
            METHOD_PARAMETER_LVQ_RESIDUAL_BITS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LVQ_RESIDUAL_BITS, DEFAULT_RESIDUAL_BITS, (v, context) -> v >= 0 && v <= 8)
        )
        .addParameter(
            METHOD_PARAMETER_LEANVEC_DIMENSIONS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LEANVEC_DIMENSIONS, 0, (v, context) -> v >= 0)
        )
        .addParameter(
            METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD,
            new Parameter.IntegerParameter(
                METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD,
                LEANVEC_DEFAULT_TRAINING_THRESHOLD,
                (v, context) -> v == 0 || v >= 1000
            )
        )
        .addParameter(
            METHOD_PARAMETER_LEANVEC_ROUGH_TRAINING_THRESHOLD,
            new Parameter.IntegerParameter(
                METHOD_PARAMETER_LEANVEC_ROUGH_TRAINING_THRESHOLD,
                LEANVEC_DEFAULT_ROUGH_TRAINING_THRESHOLD,
                (v, context) -> v == 0 || v >= 1000
            )
        )
        .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
            int primaryBits = readIntParameter(methodComponentContext, METHOD_PARAMETER_LVQ_PRIMARY_BITS, DEFAULT_PRIMARY_BITS);
            int residualBits = readIntParameter(methodComponentContext, METHOD_PARAMETER_LVQ_RESIDUAL_BITS, DEFAULT_RESIDUAL_BITS);
            validateBitCombination(primaryBits, residualBits);
            FaissSVSLVQEncoder.validatePlatformSupportsLvq();
            validateThresholds(
                readIntParameter(methodComponentContext, METHOD_PARAMETER_LEANVEC_ROUGH_TRAINING_THRESHOLD, 0),
                readIntParameter(methodComponentContext, METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD, 0)
            );

            SvsMethodAsMapBuilder builder = SvsMethodAsMapBuilder.builder(
                "LeanVec",
                methodComponent,
                methodComponentContext,
                knnMethodConfigContext
            );
            // Builds the "LeanVec{primary}x{residual}" token, e.g. "LeanVec4x8".
            builder.addParameter(METHOD_PARAMETER_LVQ_PRIMARY_BITS, "", "x");
            builder.addParameter(METHOD_PARAMETER_LVQ_RESIDUAL_BITS, "", "");
            // Optional reduced dimensionality: "LeanVec4x8_192". 0 = let the runtime default to dim/2.
            Object dimensions = methodComponentContext.getParameters().get(METHOD_PARAMETER_LEANVEC_DIMENSIONS);
            if (dimensions instanceof Integer && (Integer) dimensions > 0) {
                builder.addParameter(METHOD_PARAMETER_LEANVEC_DIMENSIONS, "_", "");
            }
            // The threshold parameters are not description tokens; they stay in the encoder parameter map,
            // which the native InitIndex reads to drive the deferred-training ladder.
            return builder.build();
        }))
        .build();

    private static int readIntParameter(MethodComponentContext methodComponentContext, String name, int defaultValue) {
        if (methodComponentContext == null) {
            return defaultValue;
        }
        Object value = methodComponentContext.getParameters().get(name);
        return value instanceof Integer ? (Integer) value : defaultValue;
    }

    static void validateBitCombination(int primaryBits, int residualBits) {
        String combination = String.format(Locale.ROOT, "%dx%d", primaryBits, residualBits);
        if (SUPPORTED_BIT_COMBINATIONS.contains(combination) == false) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Unsupported LeanVec (primary_bits, residual_bits) combination [%s] for encoder [%s]. "
                        + "Supported combinations are: 4x4, 4x8, 8x8.",
                    combination,
                    FAISS_SVS_ENCODER_LEANVEC
                )
            );
        }
    }

    static void validateThresholds(int roughThreshold, int trainingThreshold) {
        int effectiveRough = roughThreshold == 0 ? LEANVEC_DEFAULT_ROUGH_TRAINING_THRESHOLD : roughThreshold;
        int effectiveFinal = trainingThreshold == 0 ? LEANVEC_DEFAULT_TRAINING_THRESHOLD : trainingThreshold;
        if (effectiveRough > effectiveFinal) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[%s] (%d) cannot exceed [%s] (%d) for encoder [%s].",
                    METHOD_PARAMETER_LEANVEC_ROUGH_TRAINING_THRESHOLD,
                    effectiveRough,
                    METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD,
                    effectiveFinal,
                    FAISS_SVS_ENCODER_LEANVEC
                )
            );
        }
    }

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext encoderContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        // Compression varies with the reduced dimensionality and bit widths; do not claim a fixed level.
        return CompressionLevel.NOT_CONFIGURED;
    }
}
