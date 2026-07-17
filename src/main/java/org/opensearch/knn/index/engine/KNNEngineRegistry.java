/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.extern.log4j.Log4j2;

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
 * A definition that throws or is misconfigured is skipped with a warning rather than failing the plugin.
 * Empty in a default build.
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
                if (name == null || name.isBlank()) {
                    log.warn("KNNEngineDefinition [{}] returned a null or blank engine name; ignoring", definition.getClass().getName());
                    continue;
                }
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
                if (engine.library() == null) {
                    log.warn("KNNEngineDefinition [{}] returned a null library; ignoring", definition.getClass().getName());
                    continue;
                }
                if (engine.library().createsCustomSegmentFiles() && engine.nativeService() == null) {
                    log.warn(
                        "KNNEngineDefinition [{}] creates custom segment files but supplies no NativeEngineService to serve them; ignoring",
                        definition.getClass().getName()
                    );
                    continue;
                }
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
