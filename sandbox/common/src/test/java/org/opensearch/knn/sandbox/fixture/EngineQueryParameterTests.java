/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.EngineQueryParameter;
import org.opensearch.knn.index.engine.EngineQueryParameter.ValueType;
import org.opensearch.test.OpenSearchTestCase;

import java.util.List;
import java.util.Map;

/**
 * The core type match matrix for {@link EngineQueryParameter}. Lives in the sandbox suite so it runs on the
 * local toolchain without the JNI build.
 */
public class EngineQueryParameterTests extends OpenSearchTestCase {

    public void testIntegerMatchesIntegerOnly() {
        assertTrue(EngineQueryParameter.matches(ValueType.INTEGER, 5));
        assertFalse(EngineQueryParameter.matches(ValueType.INTEGER, 5L));
        assertFalse(EngineQueryParameter.matches(ValueType.INTEGER, 5.0d));
        assertFalse(EngineQueryParameter.matches(ValueType.INTEGER, "5"));
        assertFalse(EngineQueryParameter.matches(ValueType.INTEGER, true));
        assertFalse(EngineQueryParameter.matches(ValueType.INTEGER, null));
        assertFalse(EngineQueryParameter.matches(ValueType.INTEGER, List.of(1)));
        assertFalse(EngineQueryParameter.matches(ValueType.INTEGER, Map.of()));
    }

    public void testFloatMatchesAnyNumber() {
        assertTrue(EngineQueryParameter.matches(ValueType.FLOAT, 5));
        assertTrue(EngineQueryParameter.matches(ValueType.FLOAT, 5.0d));
        assertTrue(EngineQueryParameter.matches(ValueType.FLOAT, 5L));
        assertFalse(EngineQueryParameter.matches(ValueType.FLOAT, "5.0"));
        assertFalse(EngineQueryParameter.matches(ValueType.FLOAT, true));
        assertFalse(EngineQueryParameter.matches(ValueType.FLOAT, null));
        assertFalse(EngineQueryParameter.matches(ValueType.FLOAT, List.of(1)));
        assertFalse(EngineQueryParameter.matches(ValueType.FLOAT, Map.of()));
    }

    public void testBooleanMatchesBooleanOnly() {
        assertTrue(EngineQueryParameter.matches(ValueType.BOOLEAN, true));
        assertTrue(EngineQueryParameter.matches(ValueType.BOOLEAN, false));
        assertFalse(EngineQueryParameter.matches(ValueType.BOOLEAN, "true"));
        assertFalse(EngineQueryParameter.matches(ValueType.BOOLEAN, 1));
        assertFalse(EngineQueryParameter.matches(ValueType.BOOLEAN, null));
        assertFalse(EngineQueryParameter.matches(ValueType.BOOLEAN, List.of(true)));
        assertFalse(EngineQueryParameter.matches(ValueType.BOOLEAN, Map.of()));
    }

    public void testStringMatchesStringOnly() {
        assertTrue(EngineQueryParameter.matches(ValueType.STRING, "text"));
        assertFalse(EngineQueryParameter.matches(ValueType.STRING, 1));
        assertFalse(EngineQueryParameter.matches(ValueType.STRING, true));
        assertFalse(EngineQueryParameter.matches(ValueType.STRING, null));
        assertFalse(EngineQueryParameter.matches(ValueType.STRING, List.of("a")));
        assertFalse(EngineQueryParameter.matches(ValueType.STRING, Map.of()));
    }

    public void testEveryTypeRejectsNull() {
        for (ValueType type : ValueType.values()) {
            assertFalse(EngineQueryParameter.matches(type, null));
        }
    }
}
