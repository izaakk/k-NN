/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.extern.log4j.Log4j2;

import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * Discovers engines contributed at runtime by modules on the classpath, via {@link java.util.ServiceLoader} of
 * {@link KNNEngineDefinition}. This is the generic extension point: the core names no specific engine and holds
 * no compile-time reference to one. Unlike a single reserved slot, this registry supports any number of
 * registered engines simultaneously — each becomes a first-class {@link KNNEngine} instance (see
 * {@link KNNEngine#getEngine(String)} / {@link KNNEngine#values()}).
 *
 * <p>{@link KNNEngine} consults {@link #all()} once at class load to build its engine table; each registered
 * engine carries its own {@code NativeEngineService}, so {@code JNIService} dispatches native operations by
 * asking the engine instance — the registry is not consulted on the query/index hot path. When nothing is
 * registered (the default build) the registry is empty and the plugin behaves exactly as upstream.
 */
@Log4j2
final class KNNEngineRegistry {

    // Built-in engine names, spelled out here (not read off KNNEngine) because this registry loads during
    // KNNEngine's class initialization. A definition colliding with one of these is skipped with a warning
    // rather than failing the plugin: one bad experimental jar must not take the node down.
    private static final Set<String> BUILT_IN_ENGINE_NAMES = Set.of("faiss", "lucene", "nmslib", "undefined");

    private static final Map<String, KNNEngineDefinition> BY_NAME;
    private static final Set<String> QUERY_PARAMETER_NAMES;

    static {
        final Map<String, KNNEngineDefinition> byName = new LinkedHashMap<>();
        final Set<String> queryParameterNames = new HashSet<>();
        for (KNNEngineDefinition definition : ServiceLoader.load(KNNEngineDefinition.class, KNNEngineRegistry.class.getClassLoader())) {
            try {
                final String key = definition.engineName().toLowerCase(Locale.ROOT);
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
                // Read the parameter names before registering, so a definition that throws here is skipped
                // entirely rather than half-registered.
                final Set<String> engineQueryParameters = definition.engineSpecificQueryParameters();
                byName.put(key, definition);
                queryParameterNames.addAll(engineQueryParameters);
            } catch (Exception | LinkageError e) {
                log.warn("Skipping misconfigured KNNEngineDefinition", e);
            }
        }
        BY_NAME = Collections.unmodifiableMap(byName);
        QUERY_PARAMETER_NAMES = Collections.unmodifiableSet(queryParameterNames);
    }

    private KNNEngineRegistry() {}

    /** All engine definitions discovered on the classpath (empty in a default build). */
    static Iterable<KNNEngineDefinition> all() {
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
