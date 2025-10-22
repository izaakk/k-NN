/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.mapper.CompressionLevel;

import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_ENCODER_SQ8;

/**
 * SQ8 (8-bit scalar quantization) encoder for SVS indexes. 
 * Reduces memory by 4x (32-bit float → 8-bit integer).
 * This encoder has no parameters.
 */
public class FaissSVSSQ8Encoder implements Encoder {

    private final static MethodComponent.MethodComponentContext METHOD_COMPONENT_CONTEXT = new MethodComponent.MethodComponentContext(
        FAISS_SVS_ENCODER_SQ8,
        java.util.Map.of()
    );

    @Override
    public MethodComponent.MethodComponentContext getMethodComponent() {
        return METHOD_COMPONENT_CONTEXT;
    }

    @Override
    public CompressionLevel getCompressionLevel() {
        return CompressionLevel.x4;
    }

    @Override
    public String getName() {
        return FAISS_SVS_ENCODER_SQ8;
    }
}
