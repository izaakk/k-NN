/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

/**
 * Names used by the test-only fixture engine. The fixture lives exclusively in the sandbox test sources —
 * it is never part of any shipped artifact — and exists to exercise the three generic engine extension
 * points (registration, JNIService dispatch, query-parameter deferral) in CI without native code.
 */
public final class FixtureConstants {

    /** Engine name users would type in a mapping; also what {@code KNNEngine.getEngine(name)} resolves. */
    public static final String FIXTURE_ENGINE_NAME = "sandbox_fixture";

    /** The fixture library's sole method. */
    public static final String METHOD_FIXTURE = "fixture_method";

    /** File extension for the fixture's (imaginary) custom segment files. */
    public static final String FIXTURE_EXTENSION = ".fixturebin";

    /**
     * The fixture's engine-specific query-time parameter. Deliberately NOT in the core
     * {@code MethodParameter} enum: it exists to prove that a parameter only the engine knows about is
     * deferred by the REST/gRPC layers and carried by the generic node-to-node wire.
     */
    public static final String METHOD_PARAMETER_FIXTURE_WINDOW = "fixture_window";

    private FixtureConstants() {}
}
