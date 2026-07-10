/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.faiss_upstream;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.sandbox.ExperimentalAlgorithm;

import java.util.Locale;
import java.util.Set;

import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.FAISS_UPSTREAM_ENGINE_NAME;
import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.METHOD_HNSW_FLAT_PANORAMA;

/**
 * {@link org.opensearch.knn.index.engine.MethodResolver} for the {@code faiss_upstream} engine. The sole
 * method is {@code hnsw_flat_panorama}; there are no encoders, so only {@code x1} compression is accepted,
 * and training contexts / {@code on_disk} mode are rejected.
 */
@ExperimentalAlgorithm(description = "faiss_upstream method resolver", since = "3.9.0")
public class FaissUpstreamMethodResolver extends AbstractMethodResolver {

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(CompressionLevel.x1);

    // Resolved by name from the engine registry (no core enum slot).
    private static KNNEngine faissUpstreamEngine() {
        return KNNEngine.getEngine(FAISS_UPSTREAM_ENGINE_NAME);
    }

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        validateConfig(knnMethodConfigContext, shouldRequireTraining);

        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            faissUpstreamEngine(),
            spaceType,
            METHOD_HNSW_FLAT_PANORAMA
        );

        String methodName = resolvedKNNMethodContext.getMethodComponentContext().getName();
        if (METHOD_HNSW_FLAT_PANORAMA.equals(methodName) == false) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "Invalid method name [%s] for engine [%s]. The only supported method is [%s].",
                    methodName,
                    faissUpstreamEngine().getName(),
                    METHOD_HNSW_FLAT_PANORAMA
                )
            );
            throw validationException;
        }

        resolveMethodParams(
            resolvedKNNMethodContext.getMethodComponentContext(),
            knnMethodConfigContext,
            HNSWFlatPanoramaMethod.METHOD_COMPONENT
        );
        knnMethodConfigContext.setCompressionLevel(CompressionLevel.x1);

        return ResolvedMethodContext.builder().knnMethodContext(resolvedKNNMethodContext).compressionLevel(CompressionLevel.x1).build();
    }

    private void validateConfig(KNNMethodConfigContext knnMethodConfigContext, boolean shouldRequireTraining) {
        ValidationException validationException = validateNotTrainingContext(shouldRequireTraining, faissUpstreamEngine(), null);
        validationException = validateCompressionSupported(
            knnMethodConfigContext.getCompressionLevel(),
            SUPPORTED_COMPRESSION_LEVELS,
            faissUpstreamEngine(),
            validationException
        );
        if (knnMethodConfigContext.getMode() == Mode.ON_DISK) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "mode=on_disk is not supported with %s; use mode=in_memory or a different method.",
                    METHOD_HNSW_FLAT_PANORAMA
                )
            );
        }
        if (validationException != null) {
            throw validationException;
        }
    }
}
