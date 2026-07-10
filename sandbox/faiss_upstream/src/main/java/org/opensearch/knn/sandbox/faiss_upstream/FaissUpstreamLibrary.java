/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.faiss_upstream;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodResolver;
import org.opensearch.knn.index.engine.NativeLibrary;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.engine.faiss.Faiss;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;
import org.opensearch.knn.sandbox.ExperimentalAlgorithm;

import java.util.Map;

import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.CURRENT_VERSION;
import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.FAISS_UPSTREAM_EXTENSION;
import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.METHOD_HNSW_FLAT_PANORAMA;

/**
 * The {@link org.opensearch.knn.index.engine.KNNLibrary} of the experimental {@code faiss_upstream}
 * engine. Its indices are plain faiss-format indices (written by the vendored upstream faiss), so scoring
 * and radial translations delegate to the core faiss library; only the file extension and the method set
 * differ.
 */
@ExperimentalAlgorithm(description = "faiss_upstream engine library (latest upstream faiss index types)", since = "3.9.0")
public class FaissUpstreamLibrary extends NativeLibrary {

    public static final FaissUpstreamLibrary INSTANCE = new FaissUpstreamLibrary();

    private final MethodResolver methodResolver = new FaissUpstreamMethodResolver();

    private FaissUpstreamLibrary() {
        // Score translation delegates to faiss below, so the base map stays empty.
        super(
            Map.<String, KNNMethod>of(METHOD_HNSW_FLAT_PANORAMA, new HNSWFlatPanoramaMethod()),
            Map.of(),
            CURRENT_VERSION,
            FAISS_UPSTREAM_EXTENSION
        );
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        return Faiss.INSTANCE.score(rawScore, spaceType);
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        return Faiss.INSTANCE.distanceToRadialThreshold(distance, spaceType);
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return Faiss.INSTANCE.scoreToRadialThreshold(score, spaceType);
    }

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        return methodResolver.resolveMethod(knnMethodContext, knnMethodConfigContext, shouldRequireTraining, spaceType);
    }

    @Override
    public VectorSearcherFactory getVectorSearcherFactory() {
        // No memory-optimized (direct-vector) search: queries run through the vendored faiss in
        // libopensearchknn_faiss_upstream — that is the whole point of the tenant.
        return null;
    }

    @Override
    public boolean supportsIterativeBuild() {
        // The native wrapper accumulates the streamed batches and trains/builds at write time, so the
        // engine participates in the iterative (memory-optimized) build path like faiss does.
        return true;
    }

    @Override
    public boolean createsCustomSegmentFiles() {
        return true;
    }

    @Override
    public boolean supportsFilters() {
        // Keep the first iteration minimal: no pre-filtered search. The core falls back to exact search
        // for filtered queries against this engine.
        return false;
    }
}
