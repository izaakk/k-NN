/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.common.Nullable;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.util.IndexUtil;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.plugin.stats.KNNGraphValue;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.common.FieldInfoExtractor.extractKNNEngine;
import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.common.FieldInfoExtractor.isDeferredLeanVecEnabled;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.SHARD_MODEL_BLOB_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;
import static org.opensearch.knn.index.engine.faiss.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

/**
 * Writes KNN Index for a field in a segment. This is intended to be used for native engines.
 *
 * Supports deferred LeanVec training: when a shard model blob is provided, it routes to the
 * template (DefaultIndexBuildStrategy) path. When no model is available for a deferred-training
 * field, it swaps the index_description to use a LVQ fallback encoding.
 */
@Log4j2
public class NativeIndexWriter {
    private static final Long CRC32_CHECKSUM_SANITY = 0xFFFFFFFF00000000L;

    private final SegmentWriteState state;
    private final FieldInfo fieldInfo;
    private final NativeIndexBuildStrategyFactory indexBuilderFactory;
    @Nullable
    private final QuantizationState quantizationState;
    @Nullable
    private final byte[] shardModelBlob;

    /**
     * Full constructor with all parameters including shard model blob for deferred training.
     */
    NativeIndexWriter(
        SegmentWriteState state,
        FieldInfo fieldInfo,
        NativeIndexBuildStrategyFactory indexBuilderFactory,
        @Nullable QuantizationState quantizationState,
        @Nullable byte[] shardModelBlob
    ) {
        this.state = state;
        this.fieldInfo = fieldInfo;
        this.indexBuilderFactory = indexBuilderFactory;
        this.quantizationState = quantizationState;
        this.shardModelBlob = shardModelBlob;
    }

    /**
     * Gets the correct writer type from fieldInfo (no quantization, no shard model)
     */
    public static NativeIndexWriter getWriter(final FieldInfo fieldInfo, SegmentWriteState state) {
        return createWriter(fieldInfo, state, null, new NativeIndexBuildStrategyFactory(), null);
    }

    /**
     * Gets the correct writer type with quantization state (no shard model).
     */
    public static NativeIndexWriter getWriter(
        final FieldInfo fieldInfo,
        final SegmentWriteState state,
        final QuantizationState quantizationState,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        return createWriter(fieldInfo, state, quantizationState, nativeIndexBuildStrategyFactory, null);
    }

    /**
     * Gets the correct writer type with quantization state and shard model blob for deferred training.
     */
    public static NativeIndexWriter getWriter(
        final FieldInfo fieldInfo,
        final SegmentWriteState state,
        final QuantizationState quantizationState,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory,
        @Nullable final byte[] shardModelBlob
    ) {
        return createWriter(fieldInfo, state, quantizationState, nativeIndexBuildStrategyFactory, shardModelBlob);
    }

    /**
     * flushes the index
     */
    public void flushIndex(final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier, int totalLiveDocs) throws IOException {
        buildAndWriteIndex(knnVectorValuesSupplier, totalLiveDocs, true);
        recordRefreshStats();
    }

    /**
     * Merges kNN index
     */
    public void mergeIndex(final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier, int totalLiveDocs) throws IOException {
        KNNVectorValues<?> knnVectorValues = knnVectorValuesSupplier.get();
        initializeVectorValues(knnVectorValues);
        if (knnVectorValues.docId() == NO_MORE_DOCS) {
            log.debug("Skipping mergeIndex, vector values are already iterated for {}", fieldInfo.name);
            return;
        }

        long bytesPerVector = knnVectorValues.bytesPerVector();
        startMergeStats(totalLiveDocs, bytesPerVector);
        buildAndWriteIndex(knnVectorValuesSupplier, totalLiveDocs, false);
        endMergeStats(totalLiveDocs, bytesPerVector);
    }

    private void buildAndWriteIndex(final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier, int totalLiveDocs, boolean isFlush)
        throws IOException {
        if (totalLiveDocs == 0) {
            log.debug("No live docs for field {}", fieldInfo.name);
            return;
        }

        final KNNEngine knnEngine = extractKNNEngine(fieldInfo);
        final String engineFileName = buildEngineFileName(
            state.segmentInfo.name,
            knnEngine.getVersion(),
            fieldInfo.name,
            knnEngine.getExtension()
        );
        try (IndexOutput output = state.directory.createOutput(engineFileName, state.context)) {
            final IndexOutputWithBuffer indexOutputWithBuffer = new IndexOutputWithBuffer(output);
            final BuildIndexParams nativeIndexParams = indexParams(
                fieldInfo,
                indexOutputWithBuffer,
                knnEngine,
                knnVectorValuesSupplier,
                totalLiveDocs,
                isFlush
            );
            NativeIndexBuildStrategy indexBuilder = indexBuilderFactory.getBuildStrategy(
                fieldInfo,
                totalLiveDocs,
                knnVectorValuesSupplier.get(),
                nativeIndexParams
            );
            indexBuilder.buildAndWriteIndex(nativeIndexParams);
            CodecUtil.writeFooter(output);
        }
    }

    // The logic for building parameters need to be cleaned up. There are various cases handled here
    // Currently it falls under two categories - with model and without model. Without model is further divided based on vector data type
    // TODO: Refactor this so its scalable. Possibly move it out of this class
    private BuildIndexParams indexParams(
        FieldInfo fieldInfo,
        IndexOutputWithBuffer indexOutputWithBuffer,
        KNNEngine knnEngine,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        int totalLiveDocs,
        boolean isFlush
    ) throws IOException {
        final Map<String, Object> parameters;
        VectorDataType vectorDataType;
        if (quantizationState != null) {
            vectorDataType = QuantizationService.getInstance().getVectorDataTypeForTransfer(fieldInfo);
        } else {
            vectorDataType = extractVectorDataType(fieldInfo);
        }

        if (fieldInfo.attributes().containsKey(MODEL_ID)) {
            // Standard model-based path (explicit model_id in mapping)
            Model model = getModel(fieldInfo);
            parameters = getTemplateParameters(fieldInfo, model);
        } else if (shardModelBlob != null) {
            // Deferred training: shard model available — use template path
            parameters = getShardModelParameters(fieldInfo);
        } else if (isDeferredLeanVecEnabled(fieldInfo)) {
            // Deferred training: no model yet — use LVQ fallback
            parameters = getDeferredTrainingFallbackParameters(fieldInfo, vectorDataType, knnEngine);
        } else {
            // Standard non-model path
            parameters = getParameters(fieldInfo, vectorDataType, knnEngine);
        }

        return BuildIndexParams.builder()
            .fieldName(fieldInfo.name)
            .parameters(parameters)
            .vectorDataType(vectorDataType)
            .knnEngine(knnEngine)
            .indexOutputWithBuffer(indexOutputWithBuffer)
            .quantizationState(quantizationState)
            .knnVectorValuesSupplier(knnVectorValuesSupplier)
            .totalLiveDocs(totalLiveDocs)
            .segmentWriteState(state)
            .isFlush(isFlush)
            .build();
    }

    private Map<String, Object> getParameters(FieldInfo fieldInfo, VectorDataType vectorDataType, KNNEngine knnEngine) throws IOException {
        Map<String, Object> parameters = new HashMap<>();
        Map<String, String> fieldAttributes = fieldInfo.attributes();
        String parametersString = fieldAttributes.get(KNNConstants.PARAMETERS);

        // parametersString will be null when legacy mapper is used
        if (parametersString == null) {
            parameters.put(KNNConstants.SPACE_TYPE, fieldAttributes.getOrDefault(KNNConstants.SPACE_TYPE, SpaceType.DEFAULT.getValue()));

            String efConstruction = fieldAttributes.get(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION);
            Map<String, Object> algoParams = new HashMap<>();
            if (efConstruction != null) {
                algoParams.put(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, Integer.parseInt(efConstruction));
            }

            String m = fieldAttributes.get(KNNConstants.HNSW_ALGO_M);
            if (m != null) {
                algoParams.put(KNNConstants.METHOD_PARAMETER_M, Integer.parseInt(m));
            }
            parameters.put(PARAMETERS, algoParams);
        } else {
            // Use explicit JSON media type, not runtime-dependent default
            parameters.putAll(
                XContentHelper.createParser(
                    NamedXContentRegistry.EMPTY,
                    DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                    new BytesArray(parametersString),
                    MediaTypeRegistry.JSON
                ).map()
            );
        }

        parameters.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        // In OpenSearch 2.16, we added the prefix for binary indices in the index description in the codec logic.
        // After 2.16, we added the binary prefix in the faiss library code. However, to ensure backwards compatibility,
        // we need to ensure that if the description does not contain the prefix but the type is binary, we add the
        // description.
        maybeAddBinaryPrefixForFaissBWC(knnEngine, parameters, fieldAttributes);

        // Used to determine how many threads to use when indexing
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.getIndexThreadQty());

        return parameters;
    }

    private void maybeAddBinaryPrefixForFaissBWC(KNNEngine knnEngine, Map<String, Object> parameters, Map<String, String> fieldAttributes) {
        if (KNNEngine.FAISS != knnEngine) {
            return;
        }

        if (!VectorDataType.BINARY.getValue()
            .equals(fieldAttributes.getOrDefault(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue()))) {
            return;
        }

        if (parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER) == null) {
            return;
        }

        if (parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER).toString().startsWith(FAISS_BINARY_INDEX_DESCRIPTION_PREFIX)) {
            return;
        }

        parameters.put(
            KNNConstants.INDEX_DESCRIPTION_PARAMETER,
            FAISS_BINARY_INDEX_DESCRIPTION_PREFIX + parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER).toString()
        );
        IndexUtil.updateVectorDataTypeToParameters(parameters, VectorDataType.BINARY);
    }

    private Map<String, Object> getTemplateParameters(FieldInfo fieldInfo, Model model) throws IOException {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.getIndexThreadQty());
        parameters.put(KNNConstants.MODEL_ID, fieldInfo.attributes().get(MODEL_ID));
        parameters.put(KNNConstants.MODEL_BLOB_PARAMETER, model.getModelBlob());
        if (FieldInfoExtractor.extractQuantizationConfig(fieldInfo) != QuantizationConfig.EMPTY) {
            IndexUtil.updateVectorDataTypeToParameters(parameters, VectorDataType.BINARY);
        } else {
            IndexUtil.updateVectorDataTypeToParameters(parameters, model.getModelMetadata().getVectorDataType());
        }

        return parameters;
    }

    /**
     * Builds parameters using the shard model blob (deferred training, post-training).
     * Uses SHARD_MODEL_BLOB_PARAMETER (not MODEL_ID) to avoid polluting the model registry namespace.
     */
    private Map<String, Object> getShardModelParameters(FieldInfo fieldInfo) {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.getIndexThreadQty());
        parameters.put(SHARD_MODEL_BLOB_PARAMETER, shardModelBlob);
        // include space_type so FAISS can configure the correct distance metric
        parameters.put(KNNConstants.SPACE_TYPE, fieldInfo.attributes().getOrDefault(KNNConstants.SPACE_TYPE, SpaceType.DEFAULT.getValue()));
        IndexUtil.updateVectorDataTypeToParameters(parameters, VectorDataType.FLOAT);
        return parameters;
    }

    /**
     * Builds parameters for pre-training fallback (deferred training, before model exists).
     * Swaps the LeanVec index_description to a LVQ fallback so the iterative build path can succeed
     * without training.
     *
     * Parsing is structured: split on comma, find the LeanVec component, replace with LVQ equivalent.
     * Throws IllegalStateException if index_description is missing or has no LeanVec component.
     */
    private Map<String, Object> getDeferredTrainingFallbackParameters(
        FieldInfo fieldInfo,
        VectorDataType vectorDataType,
        KNNEngine knnEngine
    ) throws IOException {
        Map<String, Object> parameters = getParameters(fieldInfo, vectorDataType, knnEngine);

        Object indexDesc = parameters.get(INDEX_DESCRIPTION_PARAMETER);
        if (indexDesc == null) {
            throw new IllegalStateException(
                "Deferred LeanVec training enabled but no index_description found for field " + fieldInfo.name
            );
        }

        String desc = indexDesc.toString();
        // Structured parse: split on comma, find the LeanVec component, replace with LVQ
        // Example: "SVSVamana64,LeanVec4x4_192" -> components ["SVSVamana64", "LeanVec4x4_192"]
        String[] components = desc.split(",");
        boolean found = false;
        for (int i = 0; i < components.length; i++) {
            String comp = components[i].trim();
            if (comp.startsWith("LeanVec")) {
                // Extract quantization bits: "LeanVec4x4_192" -> "4x4"
                String remainder = comp.substring("LeanVec".length());
                int underscoreIdx = remainder.indexOf('_');
                String bits = underscoreIdx >= 0 ? remainder.substring(0, underscoreIdx) : remainder;
                components[i] = "LVQ" + bits;
                found = true;
                break;
            }
        }

        if (!found) {
            throw new IllegalStateException(
                "Deferred LeanVec training enabled but index_description has no LeanVec component: " + desc
            );
        }

        String fallbackDesc = String.join(",", components);
        parameters.put(INDEX_DESCRIPTION_PARAMETER, fallbackDesc);
        log.debug("Deferred training fallback: swapped index_description from '{}' to '{}'", desc, fallbackDesc);

        // Remove training_threshold from nested encoder parameters before passing to JNI.
        // training_threshold is a codec-layer parameter with no meaning to FAISS/SVS.
        removeLeanVecTrainingThreshold(parameters);

        return parameters;
    }

    /**
     * Removes training_threshold from the nested encoder parameters (C-R3-4 fix).
     * training_threshold is a codec-layer parameter that leaks through MethodComponent's
     * getParameterMapWithDefaultsAdded() into the FAISS parameters JSON. It has no meaning
     * to FAISS/SVS and may cause JNI errors if the native layer rejects unknown parameters.
     */
    @SuppressWarnings("unchecked")
    private static void removeLeanVecTrainingThreshold(Map<String, Object> parameters) {
        Object subParamsObj = parameters.get(PARAMETERS);
        if (subParamsObj instanceof Map) {
            Map<String, Object> subParams = (Map<String, Object>) subParamsObj;
            Object encoderObj = subParams.get("encoder");
            if (encoderObj instanceof Map) {
                Map<String, Object> encoder = (Map<String, Object>) encoderObj;
                Object encoderParamsObj = encoder.get("parameters");
                if (encoderParamsObj instanceof Map) {
                    ((Map<String, Object>) encoderParamsObj).remove(METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD);
                }
            }
        }
    }

    private Model getModel(FieldInfo fieldInfo) {
        String modelId = fieldInfo.attributes().get(MODEL_ID);
        Model model = ModelCache.getInstance().get(modelId);
        if (model.getModelBlob() == null) {
            throw new RuntimeException(String.format("There is no trained model with id \"%s\"", modelId));
        }
        return model;
    }

    private void startMergeStats(int numDocs, long bytesPerVector) {
        KNNGraphValue.MERGE_CURRENT_OPERATIONS.increment();
        KNNGraphValue.MERGE_CURRENT_DOCS.incrementBy(numDocs);
        KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.incrementBy(bytesPerVector);
        KNNGraphValue.MERGE_TOTAL_OPERATIONS.increment();
        KNNGraphValue.MERGE_TOTAL_DOCS.incrementBy(numDocs);
        KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.incrementBy(bytesPerVector);
    }

    private void endMergeStats(int numDocs, long arraySize) {
        KNNGraphValue.MERGE_CURRENT_OPERATIONS.decrement();
        KNNGraphValue.MERGE_CURRENT_DOCS.decrementBy(numDocs);
        KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.decrementBy(arraySize);
    }

    private void recordRefreshStats() {
        KNNGraphValue.REFRESH_TOTAL_OPERATIONS.increment();
    }

    /**
     * Helper method to create the appropriate NativeIndexWriter based on the field info and quantization state.
     */
    private static NativeIndexWriter createWriter(
        final FieldInfo fieldInfo,
        final SegmentWriteState state,
        @Nullable final QuantizationState quantizationState,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory,
        @Nullable final byte[] shardModelBlob
    ) {
        return new NativeIndexWriter(state, fieldInfo, nativeIndexBuildStrategyFactory, quantizationState, shardModelBlob);
    }
}
