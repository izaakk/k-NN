/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.Parameter;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_VAMANA_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_DEGREE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SEARCH_WINDOW_SIZE;
import static org.opensearch.knn.common.KNNConstants.METHOD_SVS_VAMANA;

/**
 * SVS Vamana method implementation. Provides graph-based approximate search using
 * the Vamana graph construction algorithm.
 *
 * Phase 3A (Current): Basic FP32 Vamana index without compression
 * - Supported index description: "SVSVamana{degree}" (e.g., "SVSVamana64")
 * - Parameters: degree (graph degree), search_window_size, search_buffer_capacity
 *
 * Phase 3B (Future): Compression support will be added
 * - Planned: "SVSVamana64,FP16", "SVSVamana64,LVQ4x4", etc.
 */
public class FaissSVSVamanaMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT);

    final static MethodComponent METHOD_COMPONENT = initMethodComponent();

    /**
     * Constructor for FaissSVSVamanaMethod
     */
    public FaissSVSVamanaMethod() {
        super(METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new FaissSVSVamanaSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_SVS_VAMANA)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(
                METHOD_PARAMETER_DEGREE,
                new Parameter.IntegerParameter(METHOD_PARAMETER_DEGREE, 64, value -> value > 0 && value <= 256)
            )
            .addParameter(
                METHOD_PARAMETER_SEARCH_WINDOW_SIZE,
                new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_WINDOW_SIZE, 10, value -> value > 0)
            )
            .addParameter(
                METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY,
                new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY, 10, value -> value > 0)
            )
            // Note: Encoder parameter will be added in Phase 3B (FP16, LVQ, LeanVec support)
            .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                // Build index description: "SVSVamana{degree}"
                MethodAsMapBuilder methodAsMapBuilder = MethodAsMapBuilder.builder(
                    FAISS_SVS_VAMANA_DESCRIPTION,
                    methodComponent,
                    methodComponentContext,
                    knnMethodConfigContext
                );

                // Add degree parameter to the index description
                methodAsMapBuilder.addParameter(METHOD_PARAMETER_DEGREE, "", "");

                return methodAsMapBuilder.build();
            }))
            .build();
    }
}
