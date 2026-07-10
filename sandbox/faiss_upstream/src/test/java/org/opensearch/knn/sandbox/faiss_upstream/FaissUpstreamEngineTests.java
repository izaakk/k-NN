/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.faiss_upstream;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorQueryType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.DEFAULT_N_LEVELS;
import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.FAISS_UPSTREAM_ENGINE_NAME;
import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.FAISS_UPSTREAM_EXTENSION;
import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.METHOD_HNSW_FLAT_PANORAMA;
import static org.opensearch.knn.sandbox.faiss_upstream.FaissUpstreamConstants.METHOD_PARAMETER_N_LEVELS;

/**
 * Unit tests for the faiss_upstream engine's Java surface: registration through the SPI, method
 * validation, resolver behavior, and the declared query parameters. The native path is covered by
 * FaissUpstreamPanoramaIT on a sandbox-enabled build.
 */
public class FaissUpstreamEngineTests extends OpenSearchTestCase {

    private KNNMethodConfigContext configContext() {
        return KNNMethodConfigContext.builder().versionCreated(Version.CURRENT).vectorDataType(VectorDataType.FLOAT).dimension(128).build();
    }

    public void testEngineIsRegistered() {
        KNNEngine engine = KNNEngine.getEngine(FAISS_UPSTREAM_ENGINE_NAME);
        assertEquals(FAISS_UPSTREAM_ENGINE_NAME, engine.getName());
        assertNotNull(engine.getNativeService());
        assertEquals(FAISS_UPSTREAM_EXTENSION, engine.getExtension());
        assertTrue(engine.supportsIterativeBuild());
        assertTrue(engine.createsCustomSegmentFiles());
        assertFalse(engine.supportsFilters());
        assertSame(engine, KNNEngine.getEngineNameFromPath("_0_165_vec" + FAISS_UPSTREAM_EXTENSION));
    }

    public void testSearchContextDeclaresEfSearch() {
        KNNEngine engine = KNNEngine.getEngine(FAISS_UPSTREAM_ENGINE_NAME);
        Map<String, ?> supported = engine.getKNNLibrarySearchContext(METHOD_HNSW_FLAT_PANORAMA)
            .supportedMethodParameters(new QueryContext(VectorQueryType.K));
        assertTrue(supported.containsKey(METHOD_PARAMETER_EF_SEARCH));
        assertEquals(1, supported.size());
    }

    public void testMethodParameterValidation() {
        MethodComponent component = HNSWFlatPanoramaMethod.METHOD_COMPONENT;
        assertNull(
            component.validate(
                new MethodComponentContext(
                    METHOD_HNSW_FLAT_PANORAMA,
                    Map.of(METHOD_PARAMETER_M, 16, METHOD_PARAMETER_EF_CONSTRUCTION, 100, METHOD_PARAMETER_N_LEVELS, 8)
                ),
                configContext()
            )
        );
        assertNotNull(
            component.validate(new MethodComponentContext(METHOD_HNSW_FLAT_PANORAMA, Map.of(METHOD_PARAMETER_M, 0)), configContext())
        );
        assertNotNull(
            component.validate(
                new MethodComponentContext(METHOD_HNSW_FLAT_PANORAMA, Map.of(METHOD_PARAMETER_N_LEVELS, 65)),
                configContext()
            )
        );
    }

    public void testIndexingContextCarriesDefaultsAndDescription() {
        KNNLibraryIndexingContext context = HNSWFlatPanoramaMethod.METHOD_COMPONENT.getKNNLibraryIndexingContext(
            new MethodComponentContext(METHOD_HNSW_FLAT_PANORAMA, Map.of()),
            configContext()
        );
        Map<String, Object> parameters = context.getLibraryParameters();
        assertEquals(METHOD_HNSW_FLAT_PANORAMA, parameters.get(NAME));
        // dim=128 in configContext(), so the default n_levels=8 is not clamped.
        assertEquals("HNSW16,FlatPanorama8", parameters.get(INDEX_DESCRIPTION_PARAMETER));
        @SuppressWarnings("unchecked")
        Map<String, Object> methodParameters = (Map<String, Object>) parameters.get(PARAMETERS);
        assertEquals(16, methodParameters.get(METHOD_PARAMETER_M));
        assertEquals(100, methodParameters.get(METHOD_PARAMETER_EF_CONSTRUCTION));
        assertEquals(DEFAULT_N_LEVELS, methodParameters.get(METHOD_PARAMETER_N_LEVELS));
    }

    public void testIndexingContextClampsLevelsToDimension() {
        // faiss requires n_levels <= d; the description generator clamps using the field's dimension.
        KNNMethodConfigContext lowDim = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(3)
            .build();
        KNNLibraryIndexingContext context = HNSWFlatPanoramaMethod.METHOD_COMPONENT.getKNNLibraryIndexingContext(
            new MethodComponentContext(METHOD_HNSW_FLAT_PANORAMA, Map.of()),
            lowDim
        );
        assertEquals("HNSW16,FlatPanorama3", context.getLibraryParameters().get(INDEX_DESCRIPTION_PARAMETER));
    }

    public void testResolverFillsDefaults() {
        ResolvedMethodContext resolved = FaissUpstreamLibrary.INSTANCE.resolveMethod(null, configContext(), false, SpaceType.L2);
        KNNMethodContext resolvedContext = resolved.getKnnMethodContext();
        assertEquals(METHOD_HNSW_FLAT_PANORAMA, resolvedContext.getMethodComponentContext().getName());
        assertEquals(CompressionLevel.x1, resolved.getCompressionLevel());
        assertTrue(resolvedContext.getMethodComponentContext().getParameters().containsKey(METHOD_PARAMETER_M));
        assertTrue(resolvedContext.getMethodComponentContext().getParameters().containsKey(METHOD_PARAMETER_N_LEVELS));
    }

    public void testResolverRejectsOnDiskMode() {
        KNNMethodConfigContext onDisk = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .mode(Mode.ON_DISK)
            .build();
        expectThrows(ValidationException.class, () -> FaissUpstreamLibrary.INSTANCE.resolveMethod(null, onDisk, false, SpaceType.L2));
    }

    public void testResolverRejectsCompression() {
        KNNMethodConfigContext compressed = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .compressionLevel(CompressionLevel.x8)
            .build();
        expectThrows(ValidationException.class, () -> FaissUpstreamLibrary.INSTANCE.resolveMethod(null, compressed, false, SpaceType.L2));
    }

    public void testResolverRejectsTrainingContext() {
        expectThrows(
            ValidationException.class,
            () -> FaissUpstreamLibrary.INSTANCE.resolveMethod(null, configContext(), true, SpaceType.L2)
        );
    }
}
