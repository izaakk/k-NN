/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.index.engine.faiss.FaissHNSWMethod.HNSW_COMPONENT;
import static org.opensearch.knn.index.engine.faiss.FaissIVFMethod.IVF_COMPONENT;

public class FaissMethodResolver extends AbstractMethodResolver {

    private static final Logger logger = LogManager.getLogger(FaissMethodResolver.class);

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(
        CompressionLevel.x1,
        CompressionLevel.x2,
        CompressionLevel.x8,
        CompressionLevel.x16,
        CompressionLevel.x32
    );

    // svs_vamana additionally supports x4 (via lvq 4x4); the QFrame-based HNSW/IVF do not, so x4 stays out
    // of the global set above.
    private static final Set<CompressionLevel> SVS_SUPPORTED_COMPRESSION_LEVELS = Set.of(
        CompressionLevel.x1,
        CompressionLevel.x2,
        CompressionLevel.x4,
        CompressionLevel.x8
    );

    // SVS marker names matched as literals: the sandbox module owns the SVS classes, so main cannot
    // reference the sandbox SVSConstants directly.
    private static final String METHOD_SVS_VAMANA = "svs_vamana";
    private static final String SVS_ENCODER_LVQ = "lvq";
    private static final String SVS_LVQ_PRIMARY_BITS = "primary_bits";
    private static final String SVS_LVQ_RESIDUAL_BITS = "residual_bits";

    // Method registries contributed at runtime via SandboxFaissMethodProvider/ServiceLoader, so main needs no
    // compile-time reference to any FaissSVS* class.
    private static final Map<String, MethodComponent> SANDBOX_METHOD_COMPONENTS = new HashMap<>();
    private static final Map<String, Map<String, Encoder>> SANDBOX_ENCODER_MAPS = new HashMap<>();

    static {
        // Iterate defensively: a throwing/misconfigured sandbox provider must not take down engine init
        // (which would also disable the built-in HNSW/IVF methods).
        Iterator<SandboxFaissMethodProvider> providers = ServiceLoader.load(
            SandboxFaissMethodProvider.class,
            FaissMethodResolver.class.getClassLoader()
        ).iterator();
        while (true) {
            try {
                if (providers.hasNext() == false) {
                    break;
                }
                SandboxFaissMethodProvider provider = providers.next();
                SANDBOX_METHOD_COMPONENTS.putAll(provider.methodComponents());
                SANDBOX_ENCODER_MAPS.putAll(provider.encoderMaps());
            } catch (ServiceConfigurationError | RuntimeException e) {
                logger.warn("Skipping misconfigured SandboxFaissMethodProvider during FaissMethodResolver init", e);
            }
        }
    }

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        boolean isSvsVamana = knnMethodContext != null
            && knnMethodContext.getMethodComponentContext() != null
            && METHOD_SVS_VAMANA.equals(knnMethodContext.getMethodComponentContext().getName());

        // Initial validation to ensure that there are no contradictions in provided parameters
        validateConfig(knnMethodConfigContext, isSvsVamana);

        // svs_vamana is an in-memory-only execution path; reject mode=on_disk explicitly.
        if (isSvsVamana && knnMethodConfigContext.getMode() == Mode.ON_DISK) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                "mode=on_disk is not supported with svs_vamana; SVS is an in-memory execution path. "
                    + "Use mode=in_memory or a different method."
            );
            throw validationException;
        }

        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            KNNEngine.FAISS,
            spaceType,
            shouldRequireTraining ? METHOD_IVF : METHOD_HNSW
        );
        // Built-ins are handled directly; other methods come from the sandbox registry, else IVF default.
        String methodName = resolvedKNNMethodContext.getMethodComponentContext().getName();
        MethodComponent method;
        Map<String, Encoder> encoderMap;
        if (METHOD_HNSW.equals(methodName)) {
            method = HNSW_COMPONENT;
            encoderMap = FaissHNSWMethod.SUPPORTED_ENCODERS;
        } else if (SANDBOX_METHOD_COMPONENTS.containsKey(methodName)) {
            method = SANDBOX_METHOD_COMPONENTS.get(methodName);
            encoderMap = SANDBOX_ENCODER_MAPS.getOrDefault(methodName, Map.of());
        } else {
            method = IVF_COMPONENT;
            encoderMap = FaissIVFMethod.SUPPORTED_ENCODERS;
        }

        // Fill in parameters for the encoder and then the method.
        resolveEncoder(resolvedKNNMethodContext, knnMethodConfigContext, encoderMap);
        // From the resolved method context, get the compression level and validate it against the passed in
        // configuration
        CompressionLevel resolvedCompressionLevel = resolveCompressionLevelFromMethodContext(
            resolvedKNNMethodContext,
            knnMethodConfigContext,
            encoderMap
        );

        // Validate encoder parameters
        validateEncoderConfig(resolvedKNNMethodContext, knnMethodConfigContext, encoderMap);

        // Validate that resolved compression doesnt have any conflicts
        validateCompressionConflicts(knnMethodConfigContext.getCompressionLevel(), resolvedCompressionLevel);
        knnMethodConfigContext.setCompressionLevel(resolvedCompressionLevel);
        resolveMethodParams(resolvedKNNMethodContext.getMethodComponentContext(), knnMethodConfigContext, method);

        return ResolvedMethodContext.builder()
            .knnMethodContext(resolvedKNNMethodContext)
            .compressionLevel(resolvedCompressionLevel)
            .build();
    }

    private void resolveEncoder(
        KNNMethodContext resolvedKNNMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        Map<String, Encoder> encoderMap
    ) {
        if (shouldEncoderBeResolved(resolvedKNNMethodContext, knnMethodConfigContext) == false) {
            return;
        }

        CompressionLevel resolvedCompressionLevel = getDefaultCompressionLevel(knnMethodConfigContext);
        if (resolvedCompressionLevel == CompressionLevel.x1) {
            return;
        }

        // svs_vamana uses its own sq/lvq encoders, not QFrame; return before the QFrame chain, which would
        // otherwise NPE on the absent QFrame entry in the SVS encoder map.
        if (METHOD_SVS_VAMANA.equals(resolvedKNNMethodContext.getMethodComponentContext().getName())) {
            resolveSVSEncoder(resolvedKNNMethodContext, knnMethodConfigContext, encoderMap, resolvedCompressionLevel);
            return;
        }

        // TODO: This chain of if-blocks mapping compression levels to encoder configs is too complex.
        // Need to refactor it into a strategy or registry pattern where each CompressionLevel declares
        // its own encoder factory, e.g. CompressionLevel.x2.createEncoder(context, encoderMap). That
        // would make it easier to add new compression level resolutions.
        MethodComponentContext encoderComponentContext = new MethodComponentContext(ENCODER_FLAT, new HashMap<>());
        Encoder encoder = encoderMap.get(ENCODER_FLAT);
        if (CompressionLevel.x2 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(ENCODER_SQ, new HashMap<>());
            encoder = encoderMap.get(ENCODER_SQ);
            encoderComponentContext.getParameters().put(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16);
            // On 3.6.0+, also set bits for consistency with the new bits-based validation
            if (knnMethodConfigContext.getVersionCreated() != null
                && knnMethodConfigContext.getVersionCreated().onOrAfter(Version.V_3_6_0)) {
                encoderComponentContext.getParameters().put(SQ_BITS, FaissSQEncoder.Bits.SIXTEEN.getValue());
            }
        }

        if (CompressionLevel.x8 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, new HashMap<>());
            encoder = encoderMap.get(QFrameBitEncoder.NAME);
            encoderComponentContext.getParameters().put(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x8.numBitsForFloat32());
        }

        if (CompressionLevel.x16 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, new HashMap<>());
            encoder = encoderMap.get(QFrameBitEncoder.NAME);
            encoderComponentContext.getParameters().put(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x16.numBitsForFloat32());
        }

        if (CompressionLevel.x32 == resolvedCompressionLevel) {
            if (shouldUseSQOneBitForX32(knnMethodConfigContext, encoderMap)) {
                encoderComponentContext = new MethodComponentContext(ENCODER_SQ, new HashMap<>());
                encoder = encoderMap.get(ENCODER_SQ);
                encoderComponentContext.getParameters().put(SQ_BITS, FaissSQEncoder.Bits.ONE.getValue());
            } else {
                encoderComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, new HashMap<>());
                encoder = encoderMap.get(QFrameBitEncoder.NAME);
                encoderComponentContext.getParameters().put(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x32.numBitsForFloat32());
            }
        }

        Map<String, Object> resolvedParams = MethodComponent.getParameterMapWithDefaultsAdded(
            encoderComponentContext,
            encoder.getMethodComponent(),
            knnMethodConfigContext
        );
        encoderComponentContext.getParameters().putAll(resolvedParams);

        // When auto-resolved to bits=1, remove the type and clip defaults that were injected —
        // the 1-bit quantization path doesn't use them, and validateEncoderConfig would reject them.
        if (encoderComponentContext.getParameters().get(SQ_BITS) instanceof Integer bitsVal
            && bitsVal == FaissSQEncoder.Bits.ONE.getValue()) {
            encoderComponentContext.getParameters().remove(FAISS_SQ_TYPE);
            encoderComponentContext.getParameters().remove(FAISS_SQ_CLIP);
        }

        resolvedKNNMethodContext.getMethodComponentContext().getParameters().put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
    }

    /**
     * Resolves the SVS encoder for a user-supplied {@code compression_level} when no explicit encoder was
     * given: {@code 2x} -> {@code sq}(fp16), {@code 4x} -> {@code lvq}(4,4), {@code 8x} -> {@code lvq}(4,0).
     * Other levels require an explicit {@code encoder} block.
     */
    private void resolveSVSEncoder(
        KNNMethodContext resolvedKNNMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        Map<String, Encoder> encoderMap,
        CompressionLevel resolvedCompressionLevel
    ) {
        MethodComponentContext encoderComponentContext;
        Encoder encoder;
        if (CompressionLevel.x2 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(ENCODER_SQ, new HashMap<>());
            encoderComponentContext.getParameters().put(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16);
            encoder = encoderMap.get(ENCODER_SQ);
        } else if (CompressionLevel.x4 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(SVS_ENCODER_LVQ, new HashMap<>());
            encoderComponentContext.getParameters().put(SVS_LVQ_PRIMARY_BITS, 4);
            encoderComponentContext.getParameters().put(SVS_LVQ_RESIDUAL_BITS, 4);
            encoder = encoderMap.get(SVS_ENCODER_LVQ);
        } else if (CompressionLevel.x8 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(SVS_ENCODER_LVQ, new HashMap<>());
            encoderComponentContext.getParameters().put(SVS_LVQ_PRIMARY_BITS, 4);
            encoderComponentContext.getParameters().put(SVS_LVQ_RESIDUAL_BITS, 0);
            encoder = encoderMap.get(SVS_ENCODER_LVQ);
        } else {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "Compression level [%s] is not supported for svs_vamana via compression_level. "
                        + "Supported levels are 2x, 4x, 8x; for other levels specify an explicit encoder.",
                    resolvedCompressionLevel.getName()
                )
            );
            throw validationException;
        }

        if (encoder == null) {
            // SVS encoder set not registered (sandbox module absent); nothing to resolve.
            return;
        }

        Map<String, Object> resolvedParams = MethodComponent.getParameterMapWithDefaultsAdded(
            encoderComponentContext,
            encoder.getMethodComponent(),
            knnMethodConfigContext
        );
        encoderComponentContext.getParameters().putAll(resolvedParams);
        resolvedKNNMethodContext.getMethodComponentContext().getParameters().put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
    }

    // Method validates for explicit contradictions in the config
    private void validateConfig(KNNMethodConfigContext knnMethodConfigContext, boolean isSvsVamana) {
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();
        ValidationException validationException = validateCompressionSupported(
            compressionLevel,
            isSvsVamana ? SVS_SUPPORTED_COMPRESSION_LEVELS : SUPPORTED_COMPRESSION_LEVELS,
            KNNEngine.FAISS,
            null
        );
        if (validationException != null) {
            throw validationException;
        }
    }

    protected void validateEncoderConfig(
        KNNMethodContext resolvedKnnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        Map<String, Encoder> encoderMap
    ) {
        if (isEncoderSpecified(resolvedKnnMethodContext) == false) {
            return;
        }
        Encoder encoder = encoderMap.get(getEncoderName(resolvedKnnMethodContext));
        if (encoder == null) {
            return;
        }

        TrainingConfigValidationInput.TrainingConfigValidationInputBuilder inputBuilder = TrainingConfigValidationInput.builder();

        TrainingConfigValidationOutput validationOutput = encoder.validateEncoderConfig(
            inputBuilder.knnMethodContext(resolvedKnnMethodContext).knnMethodConfigContext(knnMethodConfigContext).build()
        );

        if (validationOutput.getValid() != null && !validationOutput.getValid()) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(validationOutput.getErrorMessage());
            throw validationException;
        }
    }

    private CompressionLevel getDefaultCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        if (CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())) {
            return knnMethodConfigContext.getCompressionLevel();
        }
        if (knnMethodConfigContext.getMode() == Mode.ON_DISK) {
            return CompressionLevel.x32;
        }
        return CompressionLevel.x1;
    }

    /**
     * Starting 3.6.0, x32 compression can use sq(bits=1) instead of the older QFrameBitEncoder (binary).
     * 1-bit quantization delegates to Lucene's flat format rather than the k-NN quantization
     * framework, which gives better recall. The encoderMap guard is needed because IVF doesn't
     * register the sq encoder — only HNSW does.
     *
     * Currently disabled — the SQ writer pipeline is not yet fully stable for auto-resolved
     * indices. Users can still explicitly specify sq(bits=1) to opt in. This will be enabled
     * as the default in Part 2.
     * TODO: Enable once the Faiss1040ScalarQuantizedKnnVectorsWriter pipeline is validated end-to-end.
     */
    private static boolean shouldUseSQOneBitForX32(KNNMethodConfigContext knnMethodConfigContext, Map<String, Encoder> encoderMap) {
        return knnMethodConfigContext.getVersionCreated() != null
            && knnMethodConfigContext.getVersionCreated().onOrAfter(Version.V_3_6_0)
            && encoderMap.containsKey(ENCODER_SQ);
    }
}
