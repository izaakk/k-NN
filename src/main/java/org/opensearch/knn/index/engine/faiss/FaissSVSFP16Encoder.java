/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.mapper.CompressionLevel;

import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_ENCODER_FP16;

/**
 * FP16 encoder for SVS indexes. Reduces memory by 2x (32-bit float → 16-bit float).
 * This is the simplest SVS encoder with no parameters.
 */
public class FaissSVSFP16Encoder implements Encoder {

    private final static MethodComponent.MethodComponentContext METHOD_COMPONENT_CONTEXT = new MethodComponent.MethodComponentContext(
        FAISS_SVS_ENCODER_FP16,
        java.util.Map.of()
    );

    @Override
    public MethodComponent.MethodComponentContext getMethodComponent() {
        return METHOD_COMPONENT_CONTEXT;
    }

    @Override
    public CompressionLevel getCompressionLevel() {
        return CompressionLevel.x2;
    }

    @Override
    public String getName() {
        return FAISS_SVS_ENCODER_FP16;
    }
}
