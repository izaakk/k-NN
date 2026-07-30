/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.bruteforce;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContextImpl;
import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.NativeLibrary;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;
import org.opensearch.knn.index.mapper.SpaceVectorValidator;
import org.opensearch.knn.index.mapper.VectorTransformerFactory;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.NAME;

/**
 * The example engine's {@link org.opensearch.knn.index.engine.KNNLibrary}: one method ({@code scan}),
 * L2 float vectors, no compression.
 */
public final class BruteForceLibrary extends NativeLibrary {

    public static final String ENGINE_NAME = "bruteforce";
    public static final String METHOD_SCAN = "scan";
    public static final String EXTENSION = ".bruteforce";
    /** The engine's one query-time parameter: scan only the first {@code scan_limit} vectors. */
    public static final String SCAN_LIMIT = "scan_limit";

    /** Typed key for the same parameter, used for the read in the engine service. */
    static final org.opensearch.knn.index.engine.ParameterKey<Integer> SCAN_LIMIT_KEY = org.opensearch.knn.index.engine.ParameterKey.intKey(
        SCAN_LIMIT
    );
    /** Key used to hand the space type to the engine service. */
    static final String PARAM_SPACE_TYPE = "space_type";

    public static final BruteForceLibrary INSTANCE = new BruteForceLibrary();

    private BruteForceLibrary() {
        super(Map.of(METHOD_SCAN, new ScanMethod()), Map.of(), "1", EXTENSION);
    }

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        if (knnMethodConfigContext != null) {
            final CompressionLevel compression = knnMethodConfigContext.getCompressionLevel();
            if ((CompressionLevel.isConfigured(compression) && compression != CompressionLevel.x1)
                || Mode.ON_DISK == knnMethodConfigContext.getMode()) {
                final ValidationException validationException = new ValidationException();
                validationException.addValidationError("The bruteforce engine does not support compression or on_disk mode");
                throw validationException;
            }
        }
        final KNNMethodContext resolved = knnMethodContext != null
            ? new KNNMethodContext(knnMethodContext)
            : new KNNMethodContext(KNNEngine.getEngine(ENGINE_NAME), spaceType, new MethodComponentContext(METHOD_SCAN, new HashMap<>()));
        return ResolvedMethodContext.builder().knnMethodContext(resolved).compressionLevel(CompressionLevel.x1).build();
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        return distance;
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return score;
    }

    @Override
    public boolean supportsIterativeBuild() {
        return true;
    }

    @Override
    public boolean createsCustomSegmentFiles() {
        return true;
    }

    /** The single method: an exact scan. L2 and float only, one query parameter, no training. */
    private static final class ScanMethod implements KNNMethod {

        @Override
        public boolean isSpaceTypeSupported(SpaceType space) {
            return space == SpaceType.L2;
        }

        @Override
        public ValidationException validate(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
            if (knnMethodConfigContext != null && knnMethodConfigContext.getVectorDataType() != VectorDataType.FLOAT) {
                final ValidationException validationException = new ValidationException();
                validationException.addValidationError("The bruteforce engine only supports float vectors");
                return validationException;
            }
            return null;
        }

        @Override
        public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
            return false;
        }

        @Override
        public int estimateOverheadInKB(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
            return 0;
        }

        @Override
        public KNNLibraryIndexingContext getKNNLibraryIndexingContext(
            KNNMethodContext knnMethodContext,
            KNNMethodConfigContext knnMethodConfigContext
        ) {
            final SpaceType spaceType = knnMethodContext == null ? SpaceType.L2 : knnMethodContext.getSpaceType();
            // The field mapper reads these validators from the context. These are the core defaults.
            return KNNLibraryIndexingContextImpl.builder()
                .parameters(Map.of(NAME, METHOD_SCAN, PARAM_SPACE_TYPE, spaceType.getValue()))
                .vectorValidator(new SpaceVectorValidator(spaceType))
                .perDimensionValidator(PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR)
                .perDimensionProcessor(PerDimensionProcessor.NOOP_PROCESSOR)
                .vectorTransformer(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER)
                .build();
        }

        @Override
        public KNNLibrarySearchContext getKNNLibrarySearchContext() {
            // The engine declares scan_limit and validates it here. It must be positive.
            return ctx -> Map.of(SCAN_LIMIT, new Parameter.IntegerParameter(SCAN_LIMIT, null, (v, context) -> v > 0));
        }
    }
}
