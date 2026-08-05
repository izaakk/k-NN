/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;

/**
 * One query time parameter declared by a registered engine, a name and
 * a value type. Data only, captured at discovery, the parse layer type
 * checks values with core code and never runs engine code. Matching is
 * case sensitive. Value rules beyond the type stay with the engine's
 * search context at query time.
 */
@ExperimentalApi
public record EngineQueryParameter(String name, ValueType type) {

    public enum ValueType {
        INTEGER,
        FLOAT,
        BOOLEAN,
        STRING
    }

    /**
     * True when a parsed request value matches the type. Null, lists and maps never match. INTEGER is
     * strictly Integer, matching what the engine side Parameter.IntegerParameter accepts, request
     * parsing produces Integer for every value that fits an int and Long only beyond that, which an
     * int parameter should reject anyway.
     */
    public static boolean matches(ValueType type, Object value) {
        return switch (type) {
            case INTEGER -> value instanceof Integer;
            case FLOAT -> value instanceof Number;
            case BOOLEAN -> value instanceof Boolean;
            case STRING -> value instanceof String;
        };
    }
}
