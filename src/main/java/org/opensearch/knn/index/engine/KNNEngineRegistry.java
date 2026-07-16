/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.jni.NativeEngineService;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * Discovers engines contributed at runtime via {@link java.util.ServiceLoader} of {@link KNNEngineDefinition}.
 * Each discovered definition is fully materialized here into a {@link RegisteredEngine}; a definition that
 * throws is skipped with a warning rather than failing the plugin — one bad experimental jar must not take the
 * node down. {@link KNNEngine} consults {@link #all()} once at class load; the registry is not on the
 * query/index hot path. When nothing is registered (the default build) the registry is empty.
 */
@Log4j2
final class KNNEngineRegistry {

    // Built-in engine names, spelled out here (not read off KNNEngine) because this registry loads during
    // KNNEngine's class initialization. KNNEngine asserts its built-ins against this set when seeding.
    static final Set<String> BUILT_IN_ENGINE_NAMES = Set.of("faiss", "lucene", "nmslib", "undefined");

    /** A fully-materialized registered engine; every definition method has already been invoked successfully. */
    record RegisteredEngine(String engineName, KNNLibrary library, NativeEngineService nativeService, Set<String> queryParameterNames) {
    }

    private static final Map<String, RegisteredEngine> BY_NAME;
    private static final Set<String> QUERY_PARAMETER_NAMES;

    static {
        final Map<String, RegisteredEngine> byName = new LinkedHashMap<>();
        final Set<String> queryParameterNames = new HashSet<>();
        for (KNNEngineDefinition definition : ServiceLoader.load(KNNEngineDefinition.class, KNNEngineRegistry.class.getClassLoader())) {
            try {
                final String name = definition.engineName();
                final String key = name.toLowerCase(Locale.ROOT);
                if (BUILT_IN_ENGINE_NAMES.contains(key)) {
                    log.warn(
                        "KNNEngineDefinition [{}] collides with built-in engine name [{}]; ignoring",
                        definition.getClass().getName(),
                        key
                    );
                    continue;
                }
                if (byName.containsKey(key)) {
                    log.warn("Duplicate KNNEngineDefinition for name [{}]; ignoring [{}]", key, definition.getClass().getName());
                    continue;
                }
                final RegisteredEngine engine = new RegisteredEngine(
                    name,
                    definition.library(),
                    definition.nativeService(),
                    Set.copyOf(definition.engineSpecificQueryParameters())
                );
                byName.put(key, engine);
                queryParameterNames.addAll(engine.queryParameterNames());
            } catch (Exception | LinkageError e) {
                log.warn("Skipping misconfigured KNNEngineDefinition", e);
            }
        }
        BY_NAME = Collections.unmodifiableMap(byName);
        QUERY_PARAMETER_NAMES = Collections.unmodifiableSet(queryParameterNames);
    }

    private KNNEngineRegistry() {}

    /** All registered engines discovered on the classpath (empty in a default build). */
    static Collection<RegisteredEngine> all() {
        return BY_NAME.values();
    }

    /**
     * Query-time {@code method_parameters} names contributed by registered engines (see
     * {@link KNNEngineDefinition#engineSpecificQueryParameters()}); empty in a default build.
     */
    static Set<String> engineContributedQueryParameterNames() {
        return QUERY_PARAMETER_NAMES;
    }
}
