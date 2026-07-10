/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.common.ValidationException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.index.Index;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorQueryType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.query.parser.MethodParametersParser;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.engine.validation.ParameterValidator.validateParameters;
import static org.opensearch.knn.sandbox.fixture.FixtureConstants.FIXTURE_ENGINE_NAME;
import static org.opensearch.knn.sandbox.fixture.FixtureConstants.METHOD_FIXTURE;
import static org.opensearch.knn.sandbox.fixture.FixtureConstants.METHOD_PARAMETER_FIXTURE_WINDOW;

/**
 * Exercises the Query-layer extension point end-to-end with a parameter only the engine knows about
 * ({@code fixture_window}, declared through {@code KNNEngineDefinition#engineSpecificQueryParameters} and
 * absent from the core {@code MethodParameter} enum):
 * <ol>
 *   <li>the REST parse layer defers the declared name instead of rejecting it — while a name no registered
 *       engine declared is still rejected at parse exactly as upstream;</li>
 *   <li>the engine-aware validation (the same call {@code KNNQueryBuilder#doToQuery} makes against the
 *       engine's {@code KNNLibrarySearchContext}) is the authority that accepts valid values and rejects
 *       invalid or unknown ones;</li>
 *   <li>the node-to-node wire carries it in the feature-gated appendix after the unchanged upstream
 *       positional block — and when any node lacks the feature, serialization fails loudly instead of
 *       silently dropping the parameter (the B5 finding: a dropped engine parameter means a query that
 *       "succeeds" with the wrong search parameters).</li>
 * </ol>
 */
public class FixtureQueryParamDeferralTests extends OpenSearchTestCase {

    public void testRestLayerDefersEngineParameterInsteadOfRejecting() throws Exception {
        // Preliminary validation defers names a registered engine has declared...
        assertNull(MethodParametersParser.validateMethodParameters(Map.of(METHOD_PARAMETER_FIXTURE_WINDOW, 32)));
        // ...still validates the parameters it knows...
        ValidationException efSearchInvalid = MethodParametersParser.validateMethodParameters(Map.of("ef_search", 0));
        assertNotNull(efSearchInvalid);
        // ...and still rejects a name no registered engine declared, even with the fixture engine present.
        ValidationException undeclaredInvalid = MethodParametersParser.validateMethodParameters(Map.of("totally_undeclared_param", 1));
        assertNotNull(undeclaredInvalid);
        assertTrue(undeclaredInvalid.getMessage().contains("totally_undeclared_param is not a valid method parameter"));

        // XContent parsing passes the declared parameter through with its raw value preserved.
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(METHOD_PARAMETER_FIXTURE_WINDOW, 32)
            .field("ef_search", 10)
            .endObject();
        try (XContentParser parser = createParser(builder)) {
            final Map<String, ?> parsed = MethodParametersParser.fromXContent(parser);
            assertEquals(32, parsed.get(METHOD_PARAMETER_FIXTURE_WINDOW));
            assertEquals(10, parsed.get("ef_search"));
        }

        // An undeclared name is rejected at parse time exactly as upstream — registering an engine does not
        // relax parsing for anyone else's names.
        XContentBuilder undeclared = XContentFactory.jsonBuilder().startObject().field("totally_undeclared_param", 1).endObject();
        try (XContentParser parser = createParser(undeclared)) {
            expectThrows(ParsingException.class, () -> MethodParametersParser.fromXContent(parser));
        }
    }

    public void testEngineAwareValidationIsTheAuthority() {
        // The same validation KNNQueryBuilder#doToQuery runs, against the registered engine's declared
        // search context — resolved through the engine, exactly as production code resolves it.
        final Map<String, Parameter<?>> supported = KNNEngine.getEngine(FIXTURE_ENGINE_NAME)
            .getKNNLibrarySearchContext(METHOD_FIXTURE)
            .supportedMethodParameters(new QueryContext(VectorQueryType.K));

        assertNull(validateParameters(supported, Map.of(METHOD_PARAMETER_FIXTURE_WINDOW, 32), KNNMethodConfigContext.EMPTY));
        // Invalid value: the engine's validator (> 0) rejects what the REST layer deferred.
        assertNotNull(validateParameters(supported, Map.of(METHOD_PARAMETER_FIXTURE_WINDOW, 0), KNNMethodConfigContext.EMPTY));
        // Genuinely unknown name: rejected here, so deferral does not mean anything-goes.
        assertNotNull(validateParameters(supported, Map.of("totally_made_up_param", 1), KNNMethodConfigContext.EMPTY));
    }

    public void testAppendixWireCarriesEngineParameterAcrossNodes() throws Exception {
        final Map<String, Object> methodParameters = Map.of(METHOD_PARAMETER_FIXTURE_WINDOW, 42, "ef_search", 10);
        // Every node advertises the generic-appendix feature (a uniform-version cluster).
        final Function<String, Boolean> allFeatures = name -> true;

        try (BytesStreamOutput out = new BytesStreamOutput()) {
            MethodParametersParser.streamOutput(out, methodParameters, allFeatures);
            final Map<String, ?> read = MethodParametersParser.streamInput(out.bytes().streamInput(), allFeatures);
            // The core parameter rides the unchanged upstream positional block; the engine parameter rides
            // the appendix. Both arrive.
            assertEquals(42, read.get(METHOD_PARAMETER_FIXTURE_WINDOW));
            assertEquals(10, read.get("ef_search"));
        }
    }

    public void testMixedClusterFailsLoudlyInsteadOfDroppingEngineParameter_theB5Guarantee() throws Exception {
        final Map<String, Object> methodParameters = Map.of(METHOD_PARAMETER_FIXTURE_WINDOW, 42, "ef_search", 10);
        // A mixed-version cluster with a pre-appendix node: the engine parameter cannot be represented on
        // the wire for that node. Serialization must fail loudly — the silent alternative is the B5 failure
        // mode, a query that "succeeds" with the wrong search parameters.
        final Function<String, Boolean> legacyCluster = name -> !KNNConstants.GENERIC_METHOD_PARAMETERS_FEATURE.equals(name);

        try (BytesStreamOutput out = new BytesStreamOutput()) {
            IllegalArgumentException e = expectThrows(
                IllegalArgumentException.class,
                () -> MethodParametersParser.streamOutput(out, methodParameters, legacyCluster)
            );
            assertTrue(e.getMessage().contains(METHOD_PARAMETER_FIXTURE_WINDOW));
        }
    }

    public void testCoreOnlyParametersUnaffectedByMixedCluster() throws Exception {
        // Without engine parameters the wire is exactly the upstream legacy format on a mixed cluster —
        // core-known parameters are never held hostage by the appendix feature.
        final Map<String, Object> methodParameters = Map.of("ef_search", 10);
        final Function<String, Boolean> legacyCluster = name -> !KNNConstants.GENERIC_METHOD_PARAMETERS_FEATURE.equals(name);

        try (BytesStreamOutput out = new BytesStreamOutput()) {
            MethodParametersParser.streamOutput(out, methodParameters, legacyCluster);
            final Map<String, ?> read = MethodParametersParser.streamInput(out.bytes().streamInput(), legacyCluster);
            assertEquals(methodParameters, read);
        }
    }

    public void testNoMethodContextFallback_rejectsEngineParameterAimedAtModelBasedIndex() throws Exception {
        // fixture_window passes parse and preliminary validation (the fixture engine declared it), but a
        // model-based index belongs to a built-in engine (model training has no registered-engine path) and
        // this one carries no serialized method information, so the engine-aware validation in doToQuery
        // cannot run. The no-method fallback must reject the engine parameter rather than silently ignore
        // it — this is the only path on which a declared name can reach the fallback.
        final Index dummyIndex = new Index("dummy", "dummy");
        final QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        final KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        final String modelId = "fixture-model-no-method";
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(new KNNMappingConfig() {
            @Override
            public Optional<String> getModelId() {
                return Optional.of(modelId);
            }

            @Override
            public int getDimension() {
                return 4;
            }
        });
        final ModelMetadata modelMetadata = mock(ModelMetadata.class);
        when(modelMetadata.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(modelMetadata.getSpaceType()).thenReturn(SpaceType.COSINESIMIL);
        when(modelMetadata.getState()).thenReturn(ModelState.CREATED);
        // The model carries no serialized method component information.
        when(modelMetadata.getMethodComponentContext()).thenReturn(null);
        when(modelMetadata.getVectorDataType()).thenReturn(VectorDataType.DEFAULT);
        final ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(modelMetadata);
        KNNQueryBuilder.initialize(modelDao);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

        final KNNQueryBuilder engineParamBuilder = KNNQueryBuilder.builder()
            .fieldName("test-field")
            .vector(new float[] { 1f, 2f, 3f, 4f })
            .k(3)
            .methodParameters(Map.of(METHOD_PARAMETER_FIXTURE_WINDOW, 7))
            .build();
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> engineParamBuilder.toQuery(mockQueryShardContext));
        assertTrue(e.getMessage().contains(METHOD_PARAMETER_FIXTURE_WINDOW));

        // A core-known parameter still passes on the same no-method path.
        final KNNQueryBuilder knownParamBuilder = KNNQueryBuilder.builder()
            .fieldName("test-field")
            .vector(new float[] { 1f, 2f, 3f, 4f })
            .k(3)
            .methodParameters(Map.of("ef_search", 10))
            .build();
        assertNotNull(knownParamBuilder.toQuery(mockQueryShardContext));
    }
}
