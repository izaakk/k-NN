/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.MethodComponent;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_SVS_FLAT;

/**
 * SVS Flat method implementation. Provides exhaustive (brute force) search.
 */
public class FaissSVSFlatMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT);

    final static MethodComponent METHOD_COMPONENT = initMethodComponent();

    public FaissSVSFlatMethod() {
        super(METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new FaissSVSFlatSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_SVS_FLAT)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                MethodAsMapBuilder methodAsMapBuilder = MethodAsMapBuilder.builder(
                    FAISS_SVS_FLAT_DESCRIPTION,
                    methodComponent,
                    methodComponentContext,
                    knnMethodConfigContext
                );

                return methodAsMapBuilder.build();
            }))
            .build();
    }
}
