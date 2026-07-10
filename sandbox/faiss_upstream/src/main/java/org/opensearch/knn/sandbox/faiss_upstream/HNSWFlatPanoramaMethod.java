/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.faiss_upstream;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContextImpl;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.faiss.AbstractFaissMethod;
import org.opensearch.knn.sandbox.ExperimentalAlgorithm;

import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.DEFAULT_N_LEVELS;
import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.METHOD_HNSW_FLAT_PANORAMA;
import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.METHOD_PARAMETER_N_LEVELS;

/**
 * The {@code hnsw_flat_panorama} method: faiss {@code IndexHNSWFlatPanorama} (Panorama-adapted HNSW-Flat,
 * arXiv:2510.00566) from upstream faiss — an index type the k-NN faiss pin does not have. Standard HNSW
 * semantics ({@code m}, {@code ef_construction}; query-time {@code ef_search}) plus {@code n_levels}
 * (Panorama levels per vector for progressive-distance pruning during graph traversal). Requires no
 * training and builds incrementally, exactly like the built-in HNSW methods.
 */
@ExperimentalAlgorithm(description = "faiss IndexHNSWFlatPanorama method (upstream faiss Panorama family)", since = "3.9.0")
public class HNSWFlatPanoramaMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = Set.of(VectorDataType.FLOAT);

    public static final List<SpaceType> SUPPORTED_SPACES = List.of(SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL);

    private static final int DEFAULT_M = 16;
    private static final int DEFAULT_EF_CONSTRUCTION = 100;

    public static final MethodComponent METHOD_COMPONENT = initMethodComponent();

    public HNSWFlatPanoramaMethod() {
        super(METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new HNSWFlatPanoramaSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_HNSW_FLAT_PANORAMA)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(METHOD_PARAMETER_M, new Parameter.IntegerParameter(METHOD_PARAMETER_M, DEFAULT_M, (v, context) -> v > 0))
            .addParameter(
                METHOD_PARAMETER_EF_CONSTRUCTION,
                new Parameter.IntegerParameter(METHOD_PARAMETER_EF_CONSTRUCTION, DEFAULT_EF_CONSTRUCTION, (v, context) -> v > 0)
            )
            .addParameter(
                METHOD_PARAMETER_N_LEVELS,
                new Parameter.IntegerParameter(METHOD_PARAMETER_N_LEVELS, DEFAULT_N_LEVELS, (v, context) -> v > 0 && v <= 64)
            )
            .setKnnLibraryIndexingContextGenerator((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                Map<String, Object> parameters = MethodComponent.getParameterMapWithDefaultsAdded(
                    methodComponentContext,
                    methodComponent,
                    knnMethodConfigContext
                );

                // Build the faiss factory description, e.g. "HNSW16,FlatPanorama8". faiss requires
                // n_levels <= d, so clamp here where the field's dimension is known — this keeps the
                // default valid for low-dimension fields.
                int m = (Integer) parameters.getOrDefault(METHOD_PARAMETER_M, DEFAULT_M);
                int nLevels = (Integer) parameters.getOrDefault(METHOD_PARAMETER_N_LEVELS, DEFAULT_N_LEVELS);
                int dimension = knnMethodConfigContext.getDimension();
                if (dimension > 0) {
                    nLevels = Math.min(nLevels, dimension);
                }

                Map<String, Object> methodAsMap = new HashMap<>();
                methodAsMap.put(NAME, methodComponent.getName());
                methodAsMap.put(PARAMETERS, parameters);
                methodAsMap.put(KNNConstants.INDEX_DESCRIPTION_PARAMETER, String.format(Locale.ROOT, "HNSW%d,FlatPanorama%d", m, nLevels));
                return KNNLibraryIndexingContextImpl.builder().parameters(methodAsMap).build();
            })
            .build();
    }
}
