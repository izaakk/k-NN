/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.query.parser;

import lombok.SneakyThrows;
import org.opensearch.common.ValidationException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;

import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.index.query.parser.MethodParametersParser.doXContent;
import static org.opensearch.knn.index.query.parser.MethodParametersParser.validateMethodParameters;

public class MethodParametersParserTests extends KNNTestCase {

    public void testValidateMethodParameters() {
        // No engine is registered on the core test classpath, so an unknown name is rejected exactly as it
        // always was. (Deferral of engine-declared names is exercised by the sandbox fixture engine's tests.)
        ValidationException validationException = validateMethodParameters(Map.of("dummy", 0));
        assertEquals("Validation Failed: 1: dummy is not a valid method parameter;", validationException.getMessage());

        ValidationException validationException2 = validateMethodParameters(Map.of("ef_search", 0));
        assertTrue(validationException2.getMessage().contains("Validation Failed: 1: ef_search should be greater than 0"));

        ValidationException validationException3 = validateMethodParameters(Map.of("ef_search", 10));
        assertNull(validationException3);

        ValidationException validationException4 = validateMethodParameters(Map.of("nprobes", 0));
        assertTrue(validationException4.getMessage().contains("Validation Failed: 1: nprobes should be greater than 0"));
    }

    @SneakyThrows
    public void testDoXContent() {
        Map<String, ?> params = Map.of("ef_search", 10);
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("method_parameters")
            .field("ef_search", 10)
            .endObject()
            .endObject();

        XContentBuilder builder2 = XContentFactory.jsonBuilder().startObject();
        doXContent(builder2, params);
        builder2.endObject();
        assertEquals(builder.toString(), builder2.toString());

        XContentBuilder b3 = XContentFactory.jsonBuilder();
        XContentBuilder b4 = XContentFactory.jsonBuilder();

        doXContent(b4, null);
        assertEquals(b3.toString(), b4.toString());
    }

    @SneakyThrows
    public void testFromXContent() {
        // efsearch string
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field("ef_search", "string").endObject();
        XContentParser parser1 = createParser(builder);
        expectThrows(ParsingException.class, () -> MethodParametersParser.fromXContent(parser1));

        // unknown method parameter: rejected at parse time, exactly as upstream — no engine is registered on
        // the core test classpath. (Engine-declared names are deferred; see the sandbox fixture tests.)
        builder = XContentFactory.jsonBuilder().startObject().field("unknown", "10").endObject();
        XContentParser parser2 = createParser(builder);
        expectThrows(ParsingException.class, () -> MethodParametersParser.fromXContent(parser2));

        // Valid
        builder = XContentFactory.jsonBuilder().startObject().field("ef_search", 10).endObject();
        XContentParser parser3 = createParser(builder);
        assertEquals(Map.of("ef_search", 10), MethodParametersParser.fromXContent(parser3));

        // empty map
        builder = XContentFactory.jsonBuilder().startObject().endObject();
        XContentParser parser4 = createParser(builder);
        expectThrows(ParsingException.class, () -> MethodParametersParser.fromXContent(parser4));

        // nprobes string
        builder = XContentFactory.jsonBuilder().startObject().field("nprobes", "string").endObject();
        XContentParser parser5 = createParser(builder);
        expectThrows(ParsingException.class, () -> MethodParametersParser.fromXContent(parser5));

        // nprobes Valid
        builder = XContentFactory.jsonBuilder().startObject().field("nprobes", 10).endObject();
        XContentParser parser6 = createParser(builder);
        assertEquals(Map.of("nprobes", 10), MethodParametersParser.fromXContent(parser6));
    }

    @SneakyThrows
    public void testStreamCoreParametersRideTheLegacyBlockUnchanged() {
        // Core-known parameters cross the wire in the upstream positional block regardless of the
        // generic-appendix feature; the appendix (present only when the whole cluster supports it) is empty.
        final Map<String, Object> methodParameters = Map.of("ef_search", 10);
        for (boolean appendixSupported : new boolean[] { false, true }) {
            final Function<String, Boolean> versionCheck = name -> appendixSupported
                || !KNNConstants.GENERIC_METHOD_PARAMETERS_FEATURE.equals(name);
            try (BytesStreamOutput out = new BytesStreamOutput()) {
                MethodParametersParser.streamOutput(out, methodParameters, versionCheck);
                final Map<String, ?> read = MethodParametersParser.streamInput(out.bytes().streamInput(), versionCheck);
                assertEquals(methodParameters, read);
            }
        }
    }

    @SneakyThrows
    public void testStreamEngineParameterRoundTripsThroughAppendix() {
        // A parameter unknown to the core enum (as contributed by a registered engine) rides the appendix
        // when every node supports the feature, alongside — not instead of — the positional block.
        final Map<String, Object> methodParameters = Map.of("ef_search", 10, "engine_only_param", 42);
        final Function<String, Boolean> allFeatures = name -> true;
        try (BytesStreamOutput out = new BytesStreamOutput()) {
            MethodParametersParser.streamOutput(out, methodParameters, allFeatures);
            final Map<String, ?> read = MethodParametersParser.streamInput(out.bytes().streamInput(), allFeatures);
            assertEquals(methodParameters, read);
        }
    }

    @SneakyThrows
    public void testStreamEngineParameterFailsLoudlyWhenClusterCannotCarryIt() {
        // If any node lacks the generic-appendix feature, an engine-contributed parameter cannot be
        // represented on the wire; serialization must fail loudly rather than silently drop the parameter
        // (a query that "succeeds" with the wrong search parameters).
        final Map<String, Object> methodParameters = Map.of("ef_search", 10, "engine_only_param", 42);
        final Function<String, Boolean> legacyCluster = name -> !KNNConstants.GENERIC_METHOD_PARAMETERS_FEATURE.equals(name);
        try (BytesStreamOutput out = new BytesStreamOutput()) {
            IllegalArgumentException e = expectThrows(
                IllegalArgumentException.class,
                () -> MethodParametersParser.streamOutput(out, methodParameters, legacyCluster)
            );
            assertTrue(e.getMessage().contains("engine_only_param"));
        }
    }
}
