/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.knn.jni.NativeEngineService;

import java.util.Set;

/**
 * Service-provider interface that contributes a complete engine to the core k-NN module at runtime, discovered
 * via {@link java.util.ServiceLoader} (see {@link KNNEngineRegistry}). It is the generic extension point by
 * which a module on the classpath supplies an engine's identity and behavior; the core wires it in as a
 * first-class {@link KNNEngine} (resolved by name) without holding any compile-time reference to that module,
 * and any number of engines can be registered simultaneously.
 *
 * <p>An implementation supplies everything the registered engine needs to behave like a built-in engine:
 * <ul>
 *   <li>{@link #engineName()} — the engine name users type in their mapping; also how
 *       {@code KNNEngine.getEngine(name)} resolves to this engine.</li>
 *   <li>{@link #library()} — the {@link KNNLibrary} driving method resolution, the file extension (so the codec
 *       writes/reads the engine's files), validation, scoring and capability flags.</li>
 *   <li>{@link #nativeService()} — the native index lifecycle; {@code JNIService} routes every operation for the
 *       registered engine here, to the engine's own JNI library, fully separate from the built-in
 *       {@code FaissService}/{@code NmslibService}.</li>
 * </ul>
 *
 * <p>Routing is purely by engine: a registered-engine index is created/loaded under this engine and its files
 * carry the engine's extension, so create, load, query and free all dispatch here uniformly — no per-op routing
 * key is needed. When no definition is on the classpath (the default build) the registry is empty and the
 * plugin is byte-for-byte upstream. The core holds no compile-time reference to any specific engine.
 */
@ExperimentalApi
public interface KNNEngineDefinition {

    String engineName();

    KNNLibrary library();

    /**
     * The engine's native index lifecycle, or {@code null} for a pure-JVM engine whose library does not
     * create custom segment files (such an engine never reaches {@code JNIService}).
     *
     * <p>Implementations must not touch {@code KNNEngine} statics during construction or from this method:
     * definitions are consulted while {@code KNNEngine}'s own class initialization builds the engine table,
     * so doing so creates a class-initialization cycle.
     *
     * @return the engine's {@link NativeEngineService}, or {@code null} for a pure-JVM engine
     */
    default NativeEngineService nativeService() {
        return null;
    }

    /**
     * Query-time {@code method_parameters} names this engine contributes (for example {@code search_window_size}),
     * beyond the core-known names in {@code org.opensearch.knn.index.query.request.MethodParameter}. This is a
     * parse-time allowlist only: it tells the REST/gRPC layers not to reject the name, so the engine-aware
     * validation in {@code KNNQueryBuilder#doToQuery} (against the engine's {@link KNNLibrarySearchContext})
     * can judge the value. Names, not semantics — the search context remains the validation authority, so a
     * name declared here but absent from the search context is accepted at parse and then rejected there,
     * never silently honored.
     *
     * <p>The same class-initialization rule as {@link #nativeService()} applies: do not touch
     * {@code KNNEngine} statics from this method.
     *
     * @return the engine-specific query parameter names; empty (the default) if the engine's query-time
     *         parameters are all core-known
     */
    default Set<String> engineSpecificQueryParameters() {
        return Set.of();
    }
}
