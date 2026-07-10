/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.faiss_upstream;

/**
 * Names and defaults for the experimental {@code faiss_upstream} engine. The engine's premise: build
 * against the latest upstream faiss (vendored at its own pinned commit by
 * {@code jni/sandbox/faiss_upstream/tenant.cmake}) and expose index types that exist in faiss mainstream
 * but not in the k-NN faiss pin. The first such type is the Panorama family's
 * {@code IndexHNSWFlatPanorama} (faiss 1.13/1.14): HNSW-Flat with level-oriented storage and
 * progressive-distance pruning during graph traversal. It requires no training and builds
 * incrementally, exactly like the built-in HNSW methods.
 */
public final class FaissUpstreamConstants {

    /** Engine name users type in their mapping; also how {@code KNNEngine.getEngine(name)} resolves. */
    public static final String FAISS_UPSTREAM_ENGINE_NAME = "faiss_upstream";

    /** The engine's first method: faiss {@code IndexHNSWFlatPanorama}. */
    public static final String METHOD_HNSW_FLAT_PANORAMA = "hnsw_flat_panorama";

    /** Extension for this engine's native segment files. */
    public static final String FAISS_UPSTREAM_EXTENSION = ".faissup";

    /**
     * Compatibility version tag baked into the file name. The file payload is standard faiss
     * serialization from the vendored upstream faiss.
     */
    public static final String CURRENT_VERSION = "165";

    /**
     * Number of Panorama levels: each vector's dimensions are chunked into levels with cached residual
     * energies, so a candidate's distance evaluation can stop early once the partial-distance bound
     * proves it cannot enter the result set. More levels = finer-grained pruning. Clamped to the vector
     * dimension at mapping time (faiss requires n_levels <= d).
     */
    public static final String METHOD_PARAMETER_N_LEVELS = "n_levels";
    public static final int DEFAULT_N_LEVELS = 8;

    private FaissUpstreamConstants() {}
}
