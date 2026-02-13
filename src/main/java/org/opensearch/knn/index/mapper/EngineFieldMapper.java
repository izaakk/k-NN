/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.DerivedKnnByteVectorField;
import org.opensearch.knn.index.DerivedKnnFloatVectorField;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_DEFAULT_INITIAL_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_DEFAULT_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_ENABLED;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_INITIAL_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_LEANVEC_DIMS;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.FAISS_SVS_ENCODER_LEANVEC;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_LEANVEC_DIMENSIONS;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_LEANVEC_INITIAL_TRAINING_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.buildDocValuesFieldType;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForByteVector;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForFloatVector;

/**
 *  Field mapper for all supported engines.
 */
@lombok.extern.log4j.Log4j2
public class EngineFieldMapper extends KNNVectorFieldMapper {

    private final FieldType vectorFieldType;
    private final PerDimensionProcessor perDimensionProcessor;
    private final PerDimensionValidator perDimensionValidator;
    private final VectorValidator vectorValidator;
    private final VectorTransformer vectorTransformer;
    private final boolean isLuceneEngine;

    public static EngineFieldMapper createFieldMapper(
        String fullname,
        String simpleName,
        Map<String, String> metaValue,
        KNNMethodConfigContext knnMethodConfigContext,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        OriginalMappingParameters originalMappingParameters,
        Version indexCreatedVersion
    ) {
        KNNMethodContext methodContext = originalMappingParameters.getResolvedKnnMethodContext();
        KNNLibraryIndexingContext libraryContext = methodContext.getKnnEngine()
            .getKNNLibraryIndexingContext(methodContext, knnMethodConfigContext);
        boolean isLuceneEngine = KNNEngine.LUCENE.equals(methodContext.getKnnEngine());

        KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(
            fullname,
            metaValue,
            knnMethodConfigContext.getVectorDataType(),
            new KNNMappingConfig() {
                @Override
                public Optional<KNNMethodContext> getKnnMethodContext() {
                    return Optional.of(methodContext);
                }

                @Override
                public int getDimension() {
                    return knnMethodConfigContext.getDimension();
                }

                @Override
                public Mode getMode() {
                    return Mode.fromName(originalMappingParameters.getMode());
                }

                @Override
                public CompressionLevel getCompressionLevel() {
                    return knnMethodConfigContext.getCompressionLevel();
                }

                @Override
                public Version getIndexCreatedVersion() {
                    return knnMethodConfigContext.getVersionCreated();
                }

                @Override
                public QuantizationConfig getQuantizationConfig() {
                    return Optional.ofNullable(libraryContext)
                        .map(KNNLibraryIndexingContext::getQuantizationConfig)
                        .orElse(QuantizationConfig.EMPTY);
                }

                @Override
                public KNNLibraryIndexingContext getKnnLibraryIndexingContext() {
                    return libraryContext;
                }
            },
            indexCreatedVersion
        );

        return new EngineFieldMapper(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            knnMethodConfigContext,
            originalMappingParameters,
            isLuceneEngine
        );
    }

    private EngineFieldMapper(
        String name,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        KNNMethodConfigContext knnMethodConfigContext,
        OriginalMappingParameters originalMappingParameters,
        boolean isLuceneEngine
    ) {
        super(
            name,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            knnMethodConfigContext.getVersionCreated(),
            originalMappingParameters
        );
        this.isLuceneEngine = isLuceneEngine;
        updateEngineStats();
        KNNMappingConfig knnMappingConfig = mappedFieldType.getKnnMappingConfig();
        VectorDataType vectorDataType = mappedFieldType.getVectorDataType();
        KNNMethodContext resolvedKnnMethodContext = originalMappingParameters.getResolvedKnnMethodContext();

        final KNNVectorSimilarityFunction knnVectorSimilarityFunction = resolvedKnnMethodContext.getSpaceType()
            .getKnnVectorSimilarityFunction();
        KNNLibraryIndexingContext knnLibraryIndexingContext = resolvedKnnMethodContext.getKnnEngine()
            .getKNNLibraryIndexingContext(resolvedKnnMethodContext, knnMethodConfigContext);

        // LuceneFieldMapper attributes
        if (this.isLuceneEngine) {
            this.fieldType = vectorDataType.createKnnVectorFieldType(knnMappingConfig.getDimension(), knnVectorSimilarityFunction);

            if (this.hasDocValues) {
                this.vectorFieldType = buildDocValuesFieldType(resolvedKnnMethodContext.getKnnEngine());
            } else {
                this.vectorFieldType = null;
            }
            this.vectorTransformer = null;
        } else {
            // MethodFieldMapper attributes
            this.vectorFieldType = null;
            this.useLuceneBasedVectorField = KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(indexCreatedVersion);
            KNNEngine knnEngine = resolvedKnnMethodContext.getKnnEngine();
            QuantizationConfig quantizationConfig = knnLibraryIndexingContext.getQuantizationConfig();
            this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
            this.fieldType.putAttribute(DIMENSION, String.valueOf(knnMappingConfig.getDimension()));
            this.fieldType.putAttribute(SPACE_TYPE, resolvedKnnMethodContext.getSpaceType().getValue());
            // Conditionally add quantization config
            if (quantizationConfig != null && quantizationConfig != QuantizationConfig.EMPTY) {
                this.fieldType.putAttribute(QFRAMEWORK_CONFIG, QuantizationConfigParser.toCsv(quantizationConfig));
            }

            this.fieldType.putAttribute(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
            this.fieldType.putAttribute(KNN_ENGINE, knnEngine.getName());
            try {
                this.fieldType.putAttribute(
                    PARAMETERS,
                    XContentFactory.jsonBuilder().map(knnLibraryIndexingContext.getLibraryParameters()).toString()
                );
            } catch (IOException ioe) {
                throw new RuntimeException(String.format("Unable to create KNNVectorFieldMapper: %s", ioe), ioe);
            }

            // Add deferred training attributes for LeanVec without model_id
            addDeferredTrainingAttributes(resolvedKnnMethodContext, knnMappingConfig.getDimension());

            if (useLuceneBasedVectorField) {
                int adjustedDimension = mappedFieldType.vectorDataType == VectorDataType.BINARY
                    ? knnMappingConfig.getDimension() / 8
                    : knnMappingConfig.getDimension();
                final VectorEncoding encoding = mappedFieldType.vectorDataType == VectorDataType.FLOAT
                    ? VectorEncoding.FLOAT32
                    : VectorEncoding.BYTE;
                final VectorSimilarityFunction similarityFunction = findBestMatchingVectorSimilarityFunction(
                    resolvedKnnMethodContext.getSpaceType()
                );
                fieldType.setVectorAttributes(adjustedDimension, encoding, similarityFunction);
            } else {
                fieldType.setDocValuesType(DocValuesType.BINARY);
            }

            this.fieldType.freeze();
            this.vectorTransformer = knnLibraryIndexingContext.getVectorTransformer();
        }

        // Common Attributes
        this.perDimensionProcessor = knnLibraryIndexingContext.getPerDimensionProcessor();
        this.perDimensionValidator = knnLibraryIndexingContext.getPerDimensionValidator();
        this.vectorValidator = knnLibraryIndexingContext.getVectorValidator();
    }

    private VectorSimilarityFunction findBestMatchingVectorSimilarityFunction(final SpaceType spaceType) {
        if (indexCreatedVersion.onOrAfter(Version.V_3_0_0)) {
            // We need to find the best matching similarity function and not just save DEFAULT space type after 3.0.
            // This is required for memory optimized search where utilizing .vec file to retrieve vectors.
            // During the retrieval, it will locate similarity function from the meta info. Without this best effort, always the default
            // similarity function will be used even when other space type is configured in a mapping.
            // However, for keeping the backward compatibility, we only apply this to indices created after 3.0+.
            try {
                return spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction();
            } catch (Exception e) {
                // ignore
            }
        }

        return SpaceType.DEFAULT.getKnnVectorSimilarityFunction().getVectorSimilarityFunction();
    }

    @Override
    protected List<Field> getFieldsForFloatVector(final float[] array, boolean isDerivedSourceEnabled) {
        if (this.isLuceneEngine) {
            final List<Field> fields = new ArrayList<>();
            fields.add(new DerivedKnnFloatVectorField(name(), array, fieldType, isDerivedSourceEnabled));
            if (hasDocValues && vectorFieldType != null) {
                fields.add(new VectorField(name(), array, vectorFieldType));
            }
            if (stored) {
                fields.add(createStoredFieldForFloatVector(name(), array));
            }
            return fields;
        }
        return super.getFieldsForFloatVector(array, isDerivedSourceEnabled);
    }

    @Override
    protected List<Field> getFieldsForByteVector(final byte[] array, boolean isDerivedSourceEnabled) {
        if (this.isLuceneEngine) {
            final List<Field> fields = new ArrayList<>();
            fields.add(new DerivedKnnByteVectorField(name(), array, fieldType, isDerivedSourceEnabled));
            if (hasDocValues && vectorFieldType != null) {
                fields.add(new VectorField(name(), array, vectorFieldType));
            }
            if (stored) {
                fields.add(createStoredFieldForByteVector(name(), array));
            }
            return fields;
        }
        return super.getFieldsForByteVector(array, isDerivedSourceEnabled);
    }

    @Override
    protected VectorValidator getVectorValidator() {
        return vectorValidator;
    }

    @Override
    protected PerDimensionValidator getPerDimensionValidator() {
        return perDimensionValidator;
    }

    @Override
    protected PerDimensionProcessor getPerDimensionProcessor() {
        return perDimensionProcessor;
    }

    @Override
    protected VectorTransformer getVectorTransformer() {
        if (isLuceneEngine) {
            return super.getVectorTransformer();
        }
        return vectorTransformer;
    }

    @Override
    void updateEngineStats() {
        Optional.ofNullable(originalMappingParameters)
            .ifPresent(params -> params.getResolvedKnnMethodContext().getKnnEngine().setInitialized(true));
    }

    /**
     * Adds deferred training attributes to the field type when LeanVec encoder is used without a model_id.
     * These attributes signal to the codec layer that deferred per-shard training should be used.
     */
    private void addDeferredTrainingAttributes(KNNMethodContext methodContext, int dimension) {
        if (methodContext == null) {
            return;
        }
        MethodComponentContext encoderContext = getEncoderContext(methodContext);
        if (encoderContext == null || !FAISS_SVS_ENCODER_LEANVEC.equals(encoderContext.getName())) {
            return;
        }

        // LeanVec without model_id → enable deferred training
        this.fieldType.putAttribute(DEFERRED_TRAINING_ENABLED, "true");

        // Get training thresholds from encoder parameters
        Map<String, Object> encoderParams = encoderContext.getParameters();
        int finalThreshold = DEFERRED_TRAINING_DEFAULT_THRESHOLD;
        if (encoderParams != null && encoderParams.containsKey(METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD)) {
            finalThreshold = ((Number) encoderParams.get(METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD)).intValue();
        }
        int initialThreshold = DEFERRED_TRAINING_DEFAULT_INITIAL_THRESHOLD;
        if (encoderParams != null && encoderParams.containsKey(METHOD_PARAMETER_LEANVEC_INITIAL_TRAINING_THRESHOLD)) {
            initialThreshold = ((Number) encoderParams.get(METHOD_PARAMETER_LEANVEC_INITIAL_TRAINING_THRESHOLD)).intValue();
        }

        // Validate ordering: initial <= final (W-1 fix)
        if (initialThreshold > finalThreshold) {
            log.warn("initial_training_threshold ({}) > training_threshold ({}), swapping values",
                initialThreshold, finalThreshold);
            int temp = initialThreshold;
            initialThreshold = finalThreshold;
            finalThreshold = temp;
        }
        this.fieldType.putAttribute(DEFERRED_TRAINING_THRESHOLD, String.valueOf(finalThreshold));
        this.fieldType.putAttribute(DEFERRED_TRAINING_INITIAL_THRESHOLD, String.valueOf(initialThreshold));

        // Get LeanVec target dimensions (default: dim/2 if not specified or 0)
        int leanvecDims = 0;
        if (encoderParams != null && encoderParams.containsKey(METHOD_PARAMETER_LEANVEC_DIMENSIONS)) {
            leanvecDims = ((Number) encoderParams.get(METHOD_PARAMETER_LEANVEC_DIMENSIONS)).intValue();
        }
        if (leanvecDims <= 0) {
            leanvecDims = (dimension + 1) / 2; // SVS default
        }
        this.fieldType.putAttribute(DEFERRED_TRAINING_LEANVEC_DIMS, String.valueOf(leanvecDims));
    }

    /**
     * Extracts the encoder MethodComponentContext from a KNNMethodContext, if present.
     */
    private static MethodComponentContext getEncoderContext(KNNMethodContext methodContext) {
        MethodComponentContext methodComponent = methodContext.getMethodComponentContext();
        if (methodComponent == null || methodComponent.getParameters() == null) {
            return null;
        }
        Object encoder = methodComponent.getParameters().get(METHOD_ENCODER_PARAMETER);
        if (encoder instanceof MethodComponentContext) {
            return (MethodComponentContext) encoder;
        }
        return null;
    }
}
