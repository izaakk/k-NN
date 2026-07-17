# The k-NN Sandbox

The `sandbox/` tree is an incubation environment for experimental k-NN engines, algorithms, and
optimizations ("tenants"). It gives high-risk/high-reward ideas a structured home inside the repository —
with real CI, real reviews, and a defined path to production — without touching what a default build ships.

It is modeled on the [OpenSearch core sandbox](https://github.com/opensearch-project/OpenSearch/tree/main/sandbox)
(gated out of the build graph by default, never in a release artifact, promoted into the project proper when
mature) and grew out of [#3283](https://github.com/opensearch-project/k-NN/issues/3283) and the module
skeleton in [#3296](https://github.com/opensearch-project/k-NN/pull/3296).

> **⚠️ Experimental**: everything under `sandbox/` and `jni/sandbox/` is experimental, is **never included
> in release artifacts**, and may change or be removed without notice.

## Lifecycle

```
1. PROPOSAL  →  2. INCUBATION  →  3. GRADUATION (or REMOVAL)
```

1. **Proposal** — open an issue describing the experimental engine/algorithm (link it to
   [#3283](https://github.com/opensearch-project/k-NN/issues/3283)).
2. **Incubation** — implement it as a sandbox tenant (see "Anatomy of a tenant") and gather feedback from
   early adopters running snapshot builds with the sandbox enabled. The experimental label is structural —
   the `sandbox/` location, the build flag, snapshot-only bundling, and this README — not an annotation.
3. **Graduation** — when it meets the criteria at the end of this document, promote it into the plugin
   proper (`src/main` + `jni/`); the sandbox is deliberately shaped so the code moves with minimal change.
   Removal is equally clean: delete two directories (see "Removing a tenant").

## How the gating works

| State | What happens |
|---|---|
| Default build (`./gradlew build`, `scripts/build.sh`, releases) | The `sandbox/` tree is **not in the Gradle build graph at all** (see `settings.gradle`): not compiled, not tested, not bundled. `jni/sandbox/` is not configured by CMake (`CONFIG_SANDBOX` is `OFF` and never auto-enabled). The produced plugin is identical to one built from a tree with no `sandbox/` directory. |
| `-Pknn.sandbox.enabled=true` | Sandbox subprojects join the build; each tenant jar is bundled into the plugin zip as a **runtime-only** artifact (the root project has no compile dependency on any tenant — discovery is via `ServiceLoader`); each tenant's isolated JNI library is built. |
| Release builds | There is **no release guard**: release artifacts exclude the sandbox simply because the release scripts (`scripts/build.sh`) never pass the flag, so the sandbox is not in the build graph there — exactly the default-build row. |

The gate is a Gradle project property (`-Pknn.sandbox.enabled=true`) rather than core's JVM system property
(`-Dsandbox.enabled=true`): this gate was agreed in the #3283/#3296 discussions, it is namespaced so the
OpenSearch distribution build can never trip it, and `-P` is this repo's house style for build knobs.
Everything else follows the core sandbox conventions.

## The three extension points

A tenant engine plugs into the core through one SPI and behaves like a built-in engine everywhere:

1. **KNNEngine layer** — implement
   [`KNNEngineDefinition`](../src/main/java/org/opensearch/knn/index/engine/KNNEngineDefinition.java) and
   register it via `META-INF/services`. `KNNEngineRegistry` discovers it at startup and the engine becomes
   a first-class `KNNEngine`: resolvable by name in mappings, present in `KNNEngine.values()`, folded into
   the core capability sets through the generic `KNNLibrary` flags (`supportsIterativeBuild()`,
   `createsCustomSegmentFiles()`, `supportsFilters()`, ...) — the core never names a tenant engine.
   A definition whose name collides with a built-in (`faiss`, `lucene`, `nmslib`) or an already-registered
   tenant is **skipped with a warning** at startup — deliberately, so one bad experimental jar cannot take
   the node down.
2. **JNIService layer** — implement
   [`NativeEngineService`](../src/main/java/org/opensearch/knn/index/engine/NativeEngineService.java) (the
   8-op native index lifecycle). `JNIService` routes the 8 lifecycle/search ops (init/insert/write/
   template/load/query/radiusQuery/free) to it with a single uniform check
   (`knnEngine.getNativeService() != null`) — binary indexes, training, and shared index state remain
   core-only today; adding an engine touches **zero** core dispatch code. Pure-JVM tenants skip this
   layer entirely (`nativeService()` defaults to `null`).
3. **Query layer** — declare engine-specific query parameter names in the `KNNEngineDefinition`
   (`engineSpecificQueryParameters()`) and their value rules in a
   [`KNNLibrarySearchContext`](../src/main/java/org/opensearch/knn/index/engine/KNNLibrarySearchContext.java).
   The REST and gRPC layers defer declared names to the engine-aware validation in
   `KNNQueryBuilder#doToQuery`; a name no registered engine declares is rejected at parse, exactly as
   before. On the node-to-node wire, core-known parameters ride the unchanged upstream format; declared
   engine parameters ride a version-gated appendix. If any node in the cluster is too old to carry the
   appendix, serialization **fails loudly** at the coordinator — an engine parameter is never silently
   dropped on a multi-node hop.

These contracts are pinned in CI at two levels. The **fixture engine** in
[`sandbox/common/src/test/java/org/opensearch/knn/sandbox/fixture/`](common/src/test/java/org/opensearch/knn/sandbox/fixture/)
— a complete, pure-Java tenant in a handful of small classes — exercises registration, capability folding,
JNIService dispatch, and query-parameter deferral in the sandbox test run. The node-to-node wire behavior
(appendix serialization, version gating, loud failure) is pinned by the core
`MethodParametersParserTests`, which run in default-build CI on every PR.

## Anatomy of a tenant

A tenant named `acme` (lowercase, no dashes — the name becomes a package name and a JNI library suffix)
consists of the following pieces. The fixture engine is the minimal in-tree reference; the first real
tenant PR (Intel SVS) serves as the full-size worked example, including the native pieces.

### Module: `sandbox/acme/`

`settings.gradle` discovers any `sandbox/<dir>` containing a `build.gradle` when the flag is on. The
build file is ~10 lines because plugins, repositories, the dependency set (`compileOnly project(':')`
etc.), and test conventions are inherited from [`sandbox/build.gradle`](build.gradle):

```groovy
/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
description = "Experimental ACME engine for k-NN"
```

Verify: `./gradlew projects -Pknn.sandbox.enabled=true | grep acme` lists `:sandbox:acme`; without the
flag, no sandbox project appears.

### Engine definition + SPI registration

One `KNNEngineDefinition` implementation in main sources
(reference: [`FixtureEngineProvider`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureEngineProvider.java)):

```java
public class AcmeEngineProvider implements KNNEngineDefinition {
    public String engineName() { return "acme"; }             // "engine": "acme" in mappings
    public KNNLibrary library() { return AcmeLibrary.INSTANCE; }
    public NativeEngineService nativeService() { return nativeService; }   // null for pure-JVM tenants
    public Set<String> engineSpecificQueryParameters() { return Set.of("acme_beam_width"); }
}
```

plus the service file
`sandbox/acme/src/main/resources/META-INF/services/org.opensearch.knn.index.engine.KNNEngineDefinition`
containing the implementation's fully qualified name. Any number of tenants can register simultaneously.

### Library

A `KNNLibrary` (typically extending
[`NativeLibrary`](../src/main/java/org/opensearch/knn/index/engine/NativeLibrary.java)) declaring the
engine's methods, segment-file extension, score translation, method resolver, and capability flags
(reference: [`FixtureLibrary`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureLibrary.java)).
The flags are how the core folds a tenant into its behavior without knowing its name:

| Flag | What it buys |
|---|---|
| `supportsIterativeBuild()` | The memory-efficient iterative build path (`initIndex` + `insertToIndex` batches + `writeIndex`). |
| `createsCustomSegmentFiles()` | The codec writes/reads segment files with the tenant's extension and routes them back to its engine. |
| `supportsFilters()` | Pre-filtered k-NN search is allowed on the engine. |
| `supportsRadialSearch()` | Radial queries are allowed; `NativeEngineService.radiusQueryIndex` becomes reachable. |
| `supportsNestedFields()` | k-NN on nested fields is allowed on the engine. |

Value rules for declared query parameters go in the library's `KNNLibrarySearchContext`
(reference: [`FixtureSearchContext`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureSearchContext.java)):

```java
public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
    return Map.of("acme_beam_width", new Parameter.IntegerParameter("acme_beam_width", null, (v, c) -> v > 0));
}
```

Engine parameters do **not** go in the core `MethodParameter` enum — that enum is only for parameters the
core itself owns.

### Native service (native tenants only)

A `NativeEngineService` implementation — typically extending `AbstractNativeEngineService` from
`:sandbox:common`, which supplies descriptive `UnsupportedOperationException`s for undeclared operations —
that adapts each of the 8 lifecycle ops onto a static JNI binding class. `JNIService` hands the tenant
the raw method-parameters map — the tenant handles its own tuning parameters (e.g. thread counts); the
core does not pre-extract `INDEX_THREAD_QTY` for tenants. The binding class loads its library through
the core loader:

```java
static {
    // Picks the best available SIMD variant for the host and falls back to the plain library.
    KNNLibraryLoader.loadLibraryByVariant("opensearchknn_acme");
    initLibrary();
    KNNEngine.getEngine(AcmeConstants.ACME_ENGINE_NAME).setInitialized(true);
}
```

The variant-suffix scheme (`_avx512_spr`/`_avx512`/`_avx2`/plain) mirrors faiss's opt-levels, but shipping
variants is optional — a single unsuffixed `.so` is fully supported via the loader's fallback.

Two conventions, checked in review:

* All `System.loadLibrary` calls go through `KNNLibraryLoader` — tenants must not load native libraries
  directly.
* The `NativeEngineService` must hold no static reference that class-initializes the JNI binding class:
  registration happens at node startup, but the native library must load **lazily on first use**. The
  first native tenant PR demonstrates this pattern end to end.

### Native build (native tenants only): `jni/sandbox/acme/`

Configured only when `-DCONFIG_SANDBOX=ON` (passed automatically by `-Pknn.sandbox.enabled=true`). A
`jni/sandbox/acme/tenant.cmake` — discovered automatically by `jni/sandbox/tenants.cmake` — vendors the
tenant's own copy of its underlying library (static, PIC, pinned to an exact commit) and hands the JNI
sources to the shared helper in [`jni/sandbox/cmake/SandboxTenant.cmake`](../jni/sandbox/cmake/SandboxTenant.cmake):

```cmake
knn_sandbox_add_jni_library(opensearchknn_acme
    SOURCES        ${KNN_SANDBOX_TENANT_DIR}/src/org_opensearch_knn_sandbox_acme_AcmeService.cpp
    INCLUDE_DIRS   ${KNN_SANDBOX_TENANT_DIR}/include
    LINK_LIBRARIES acme_vendor
    DEPENDS        acme_vendor_ep)
```

The helper compiles tenant code with hidden visibility (JNI entry points stay exported via `JNIEXPORT`)
and, on Linux, links with `-Wl,--exclude-libs,ALL` so symbols from statically linked archives become
local — a guarantee that is Linux-only today (macOS/Windows un-addressed) — plus an `$ORIGIN` rpath for
any runtime `.so` shipped alongside. This matters most when the tenant embeds a different version of a C++ library the
plugin already ships (e.g. faiss): two exported `faiss::*` symbol sets in one JVM interpose and route
calls into the wrong build; export-only-JNI makes the tenant a black box. Verify:

```bash
nm -D --defined-only jni/build/release/libopensearchknn_acme*.so | grep -cv ' Java_\| JNI_'   # → 0
```

Additional native-build rules:

* **Supply chain**: anything `tenant.cmake` downloads must be checksum-pinned (`URL_HASH SHA256=...` or an
  exact git commit); a user-overridable URL requires a paired `-D..._SHA256`.
* **JNI headers**: compiling the tenant's Java generates headers under
  `sandbox/acme/build/generated-jni-headers/`; copy them into `jni/sandbox/acme/include/` (headers are
  checked in so the native build never depends on a Java compile). Note that `_` in package/class names
  mangles to `_1` in JNI symbols — the generated header gets this right.
* **faiss-based tenants**: a shared `knn_sandbox_vendor_faiss` helper packaging the static+PIC faiss
  recipe (SIMD-variant selection, BLAS/LAPACK/OpenMP re-supply) ships with the first faiss-based tenant.
* **Tenant-specific cmake flags** pass through `-Psandbox.cmake.args="-DYOUR_FLAG=value;..."` — a tenant
  never edits the root `build.gradle`.

Gradle discovers the native target by convention: a tenant with `jni/sandbox/acme/tenant.cmake` builds
`libopensearchknn_acme*.so` under `./gradlew buildJniLib -Pknn.sandbox.enabled=true`.

### Tests

Unit tests live in `sandbox/acme/src/test/...` and run via
`./gradlew :sandbox:acme:test -Pknn.sandbox.enabled=true`; the fixture test classes
([`FixtureEngineRegistrationTests`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureEngineRegistrationTests.java),
[`FixtureJNIServiceDispatchTests`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureJNIServiceDispatchTests.java),
[`FixtureQueryParamDeferralTests`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureQueryParamDeferralTests.java))
are ready-made templates for registration, dispatch, and deferral/engine-aware validation.

REST integration tests are auto-wired: `*IT` classes extending `KNNRestTestCase` in the tenant's test
sources are detected by the shared sandbox config, which creates a `:sandbox:acme:integTest` task whose
cluster runs the sandbox-enabled plugin zip with the JNI build dir on the node's `java.library.path`.
Nothing to copy into the tenant's build.gradle — the wiring lives once in
[`sandbox/build.gradle`](build.gradle).

To run a live node: `./gradlew run -Pknn.sandbox.enabled=true`, then map a field with
`"engine": "acme"`. On a default build the same mapping is rejected with `Invalid engine type: acme` —
correct behavior: the tenant does not exist there.

## CI and contribution hygiene

* The [`sandbox-check`](../.github/workflows/sandbox-check.yml) workflow runs
  `./gradlew -p sandbox test -Pknn.sandbox.enabled=true` on every PR — tenant unit tests join CI with zero
  workflow changes. It runs on all PRs (not just sandbox ones) because a core refactor can break the SPI
  seam without touching `sandbox/**`.
* Only the tenant **jar** (plus `:sandbox:common`'s) is bundled into the plugin zip — an `implementation`
  dependency on anything OpenSearch does not already provide will be missing at runtime. Keep tenant
  runtime dependencies to OpenSearch-provided libraries, or vendor the code.
* Sandbox PRs follow the normal [CONTRIBUTING](../CONTRIBUTING.md) flow: CHANGELOG entry, DCO sign-off,
  `./gradlew spotlessApply -Pknn.sandbox.enabled=true`, no unrelated changes. Sandbox PRs get the same
  review bar as any other.

## Graduation

This section is the canonical statement of the graduation criteria.

**Graduation criteria** (all of them):

| Criterion | Threshold |
|---|---|
| **Stability** | No critical bugs after a minimum incubation period |
| **Performance** | Meets or exceeds benchmarks comparable to existing stable algorithms |
| **Test coverage** | Comprehensive unit and integration tests |
| **Community feedback** | Positive reception from early adopters running sandbox-enabled snapshots |
| **Documentation** | Complete API documentation and usage guides |

**Mechanics** — the sandbox is shaped so incubated code moves, not rewrites: Java sources move from
`sandbox/acme/src/main/...` into `src/main/...` (a graduated engine may keep its `KNNEngineDefinition` or
be added as a built-in singleton); native code moves from `jni/sandbox/acme/` into `jni/src` +
`jni/CMakeLists.txt` proper (or keeps its isolated-library shape if its vendored dependencies require it);
tests move to the corresponding core locations. The
graduation PR is where BWC policy, settings, and docs get the full production treatment.

**Removing a tenant** that didn't pan out:

```bash
git rm -r sandbox/acme jni/sandbox/acme
```

No other file changes — nothing else in the repository references a tenant by name. Be honest in the
removal PR: indices created with the removed engine become unopenable (their segment files carry the
tenant's extension); acceptable for an experimental, snapshot-only engine, but say it.

## Design notes

* **No reserved engine slot**: the core carries no compile-time knowledge of any tenant — no enum entry,
  no name string, no dispatch branch. Discovery is `ServiceLoader`-only, so a default build's engine set,
  parse behavior, and wire behavior are identical to a tree with no `sandbox/` directory.
* **Per-tenant subprojects** (vs one sandbox module with tenant packages): tenants stay independently
  buildable, testable, and removable, and two tenants can pin conflicting dependency versions. Core's
  `sandbox/{libs,modules,plugins}` mirror what a feature graduates *to*; k-NN tenants graduate into the
  plugin itself, so the mirror is "one subtree per incubating engine".
* **Test-scoped fixture**: the fixture engine lives in test sources so "never shipped" is structural — a
  test source set is excluded from every jar by construction; no packaging rule can regress it.
* **Gating layers**: the sandbox is excluded from the build graph (settings.gradle) and from CMake
  configuration (`CONFIG_SANDBOX=OFF`); release artifacts exclude tenants because the release scripts
  never pass the flag. Per-tenant selection (enabling a subset of tenants) is planned follow-up work —
  currently the flag enables all tenants.
* **Query-param deferral ships together with the wire appendix**: deferral without a wire that can carry
  the parameter appears to work on a single node and silently drops engine parameters on any multi-node
  hop. The version-gated appendix carries them when every node supports it; when one can't, serialization
  fails loudly at the coordinator — a silent drop is impossible by construction. Core
  `MethodParametersParserTests` pin this behavior in default-build CI.
