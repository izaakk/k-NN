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

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.opensearch.common.ValidationException;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.request.MethodParameter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.index.query.KNNQueryBuilder.METHOD_PARAMS_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.NAME;

/**
 * Note: This parser is used by neural plugin as well, breaking changes will require changes in neural as well
 */
@EqualsAndHashCode
@Getter
@AllArgsConstructor
public class MethodParametersParser {

    // Validation on rest layer
    public static ValidationException validateMethodParameters(final Map<String, ?> methodParameters) {
        final List<String> errors = new ArrayList<>();
        for (final Map.Entry<String, ?> methodParameter : methodParameters.entrySet()) {
            final MethodParameter parameter = MethodParameter.enumOf(methodParameter.getKey());
            if (parameter != null) {
                final ValidationException validationException = parameter.validate(methodParameter.getValue());
                if (validationException != null) {
                    errors.add(validationException.getMessage());
                }
            } else if (!KNNEngine.isEngineContributedQueryParameter(methodParameter.getKey())) {
                // Should never happen if used in the right sequence
                errors.add(methodParameter.getKey() + " is not a valid method parameter");
            }
            // A name declared by a registered engine (KNNEngineDefinition#engineSpecificQueryParameters) is
            // deferred: the engine-aware validation in KNNQueryBuilder#doToQuery judges its value against the
            // engine's KNNLibrarySearchContext. In a default build no engine is registered, so this method
            // behaves exactly as upstream.
        }

        if (!errors.isEmpty()) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationErrors(errors);
            return validationException;
        }
        return null;
    }

    // deserialize for node to node communication
    public static Map<String, ?> streamInput(StreamInput in, Function<String, Boolean> minClusterVersionCheck) throws IOException {
        if (!in.readBoolean()) {
            return null;
        }

        final Map<String, Object> methodParameters = new HashMap<>();
        // Positional block — unchanged from upstream. Core-known parameters always cross the wire in the
        // battle-tested enum-keyed format, including the typed parse() normalization on read.
        for (final MethodParameter methodParameter : MethodParameter.values()) {
            if (minClusterVersionCheck.apply(methodParameter.getName())) {
                String name = in.readString();
                Object value = in.readGenericValue();
                if (value != null) {
                    methodParameters.put(name, methodParameter.parse(value));
                }
            }
        }
        // Appendix — strictly symmetric with streamOutput's write condition: parameters contributed by a
        // registered engine (names unknown to the core enum) ride an appended map once the whole cluster
        // understands it. Empty on any default build.
        if (minClusterVersionCheck.apply(KNNConstants.GENERIC_METHOD_PARAMETERS_FEATURE)) {
            methodParameters.putAll(in.readMap(StreamInput::readString, StreamInput::readGenericValue));
        }

        return !methodParameters.isEmpty() ? methodParameters : null;
    }

    // serialize for node to node communication
    public static void streamOutput(StreamOutput out, Map<String, ?> methodParameters, Function<String, Boolean> minClusterVersionCheck)
        throws IOException {
        if (methodParameters == null || methodParameters.isEmpty()) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            // Positional block — unchanged from upstream, always written. All values are written to
            // deserialize without ambiguity.
            for (final MethodParameter methodParameter : MethodParameter.values()) {
                if (minClusterVersionCheck.apply(methodParameter.getName())) {
                    out.writeString(methodParameter.getName());
                    out.writeGenericValue(methodParameters.get(methodParameter.getName()));
                }
            }
            // Appendix: parameters contributed by a registered engine (names unknown to the core enum),
            // written only when the whole cluster understands it. Empty in any default build — the parse
            // layers admit a non-core name only when a registered engine declared it.
            final Map<String, Object> engineParameters = engineContributedSubset(methodParameters);
            if (minClusterVersionCheck.apply(KNNConstants.GENERIC_METHOD_PARAMETERS_FEATURE)) {
                out.writeMap(engineParameters, StreamOutput::writeString, StreamOutput::writeGenericValue);
            } else if (engineParameters.isEmpty() == false) {
                // A node below the feature version cannot receive these. Failing loudly here is deliberate:
                // the alternative is the parameter silently vanishing on the hop to that node — a query that
                // "succeeds" with the wrong search parameters.
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "engine-specific method_parameters %s require every node in the cluster to support %s",
                        engineParameters.keySet(),
                        KNNConstants.GENERIC_METHOD_PARAMETERS_FEATURE
                    )
                );
            }
        }
    }

    // The subset of methodParameters whose names are unknown to the core MethodParameter enum — i.e. the
    // parameters contributed by a registered engine.
    private static Map<String, Object> engineContributedSubset(Map<String, ?> methodParameters) {
        final Map<String, Object> engineParameters = new HashMap<>();
        for (final Map.Entry<String, ?> entry : methodParameters.entrySet()) {
            if (MethodParameter.enumOf(entry.getKey()) == null) {
                engineParameters.put(entry.getKey(), entry.getValue());
            }
        }
        return engineParameters;
    }

    public static void doXContent(final XContentBuilder builder, final Map<String, ?> methodParameters) throws IOException {
        if (methodParameters == null || methodParameters.isEmpty()) {
            return;
        }
        builder.startObject(METHOD_PARAMS_FIELD.getPreferredName());
        for (final Map.Entry<String, ?> entry : methodParameters.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null) {
                builder.field(entry.getKey(), entry.getValue());
            }
        }
        builder.endObject();
    }

    public static Map<String, ?> fromXContent(final XContentParser parser) throws IOException {
        final Map<String, Object> methodParametersJson = parser.map();
        if (methodParametersJson.isEmpty()) {
            throw new ParsingException(parser.getTokenLocation(), METHOD_PARAMS_FIELD.getPreferredName() + " cannot be empty");
        }

        final Map<String, Object> methodParameters = new HashMap<>();
        for (Map.Entry<String, Object> requestParameter : methodParametersJson.entrySet()) {
            final String name = requestParameter.getKey();
            final Object value = requestParameter.getValue();
            final MethodParameter parameter = MethodParameter.enumOf(name);
            if (parameter == null) {
                if (KNNEngine.isEngineContributedQueryParameter(name)) {
                    // A registered engine declared this name (KNNEngineDefinition#engineSpecificQueryParameters);
                    // pass the raw value through for the engine-aware validation in KNNQueryBuilder#doToQuery,
                    // which judges it against the engine's KNNLibrarySearchContext. Unreachable in a default
                    // build, where the rejection below is identical to upstream.
                    methodParameters.put(name, value);
                    continue;
                }
                throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] unknown method parameter found [" + name + "]");
            }

            try {
                // This makes sure that we throw parsing exception on rest layer.
                methodParameters.put(name, parameter.parse(value));
            } catch (final Exception exception) {
                throw new ParsingException(parser.getTokenLocation(), exception.getMessage());
            }
        }
        return methodParameters.isEmpty() ? null : methodParameters;
    }
}
