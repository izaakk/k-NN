# The k-NN Sandbox

The `sandbox/` tree is a **first-class incubation environment** for experimental k-NN engines, algorithms,
and optimizations ("tenants"). It gives bold, high-risk/high-reward ideas a structured home inside the
repository — with real CI, real reviews, and a defined path to production — without ever touching what a
default build ships.

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
2. **Incubation** — implement it as a sandbox tenant (this document is the how-to), annotate public classes
   with `@ExperimentalAlgorithm`, gather feedback from early adopters running snapshot builds with the
   sandbox enabled.
3. **Graduation** — when it meets the criteria at the end of this document, promote it into the plugin
   proper (`src/main` + `jni/`); the sandbox is deliberately shaped so the code moves with minimal change.
   Removal is equally clean: delete two directories (see "Removing a tenant").

## How the gating works

| State | What happens |
|---|---|
| Default build (`./gradlew build`, `scripts/build.sh`, releases) | The `sandbox/` tree is **not in the Gradle build graph at all** (see `settings.gradle`): not compiled, not tested, not bundled. `jni/sandbox/` is not configured by CMake (`CONFIG_SANDBOX` is `OFF` and never auto-enabled). The produced plugin is identical to one built from a tree with no `sandbox/` directory. |
| `-Pknn.sandbox.enabled=true` on a **snapshot** build | Sandbox subprojects join the build; each tenant jar is bundled into the plugin zip as a **runtime-only** artifact (the root project has no compile dependency on any tenant — discovery is via `ServiceLoader`); each tenant's isolated JNI library is built. |
| `-Pknn.sandbox.enabled=true` on a **release** build (`-Dbuild.snapshot=false`) | The build **fails** with an explanatory error. Sandbox tenants can never ship in a release artifact — this mirrors the OpenSearch core sandbox's "snapshot + flag" double condition, enforced loudly instead of silently. |

The gate is a Gradle project property (`-Pknn.sandbox.enabled=true`) rather than core's JVM system property
(`-Dsandbox.enabled=true`): this exact gate was agreed and approved in the #3283/#3296 discussions, it is
namespaced so the OpenSearch distribution build can never trip it, and `-P` is this repo's house style for
build knobs. Everything else follows the core sandbox conventions.

## The three extension points (what the core gives you)

A tenant engine plugs into the core through one SPI and behaves like a built-in engine everywhere:

1. **KNNEngine layer** — implement
   [`KNNEngineDefinition`](../src/main/java/org/opensearch/knn/index/engine/KNNEngineDefinition.java) and
   register it via `META-INF/services`. `KNNEngineRegistry` discovers it at startup and your engine becomes
   a first-class `KNNEngine`: resolvable by name in mappings, present in `KNNEngine.values()`, folded into
   the core capability sets through the generic `KNNLibrary` flags (`supportsIterativeBuild()`,
   `createsCustomSegmentFiles()`, `supportsFilters()`) — the core never names your engine.
2. **JNIService layer** — implement
   [`NativeEngineService`](../src/main/java/org/opensearch/knn/jni/NativeEngineService.java) (the 8-op
   native index lifecycle). `JNIService` routes every native operation for your engine to it with a single
   uniform check (`knnEngine.getNativeService() != null`); adding an engine touches **zero** core dispatch
   code. A pure-JVM engine implements the same service in plain Java — no JNI, no native build (see the
   pure-JVM notes in steps 3 and 5).
3. **Query layer** — declare your engine-specific query parameter names in your `KNNEngineDefinition`
   (`engineSpecificQueryParameters()`) and their value rules in a
   [`KNNLibrarySearchContext`](../src/main/java/org/opensearch/knn/index/engine/KNNLibrarySearchContext.java).
   The REST and gRPC layers defer the names you declared to the engine-aware validation in
   `KNNQueryBuilder#doToQuery` (undeclared names are rejected at parse, exactly as before), and the
   node-to-node wire carries your parameters across a multi-node cluster on a feature-gated appendix —
   you get correct distributed behavior for free, and a cluster that cannot carry your parameter fails
   loudly instead of dropping it.

Every claim above is exercised in CI by the **fixture engine** in
[`sandbox/common/src/test/java/org/opensearch/knn/sandbox/fixture/`](common/src/test/java/org/opensearch/knn/sandbox/fixture/)
— a complete, pure-Java tenant in ~5 small classes. It is the reference implementation for this tutorial:
every Java step below points at a fixture file you can open and copy.

---

# Tutorial: onboarding a tenant

The steps below onboard a new engine tenant called `acme`. Substitute your own name (lowercase, no dashes —
it becomes a package name and a JNI library suffix). Steps 1–5 give you a working engine; steps 6–8 cover
query parameters, tests, and running a live cluster; steps 9–10 cover CI and graduation.

### Step 0 — prerequisites

* A branch off `main`. Sandbox work follows the normal [CONTRIBUTING](../CONTRIBUTING.md) flow (DCO
  sign-off, CHANGELOG entry, Spotless).
* For the native path (step 5): the same toolchain as a normal k-NN native build (CMake ≥ 3.24, gcc,
  `JAVA_HOME` set).

### Step 1 — module skeleton (1 file)

```bash
mkdir -p sandbox/acme/src/main/java/org/opensearch/knn/sandbox/acme
mkdir -p sandbox/acme/src/main/resources/META-INF/services
mkdir -p sandbox/acme/src/test/java/org/opensearch/knn/sandbox/acme
```

`sandbox/acme/build.gradle` — this is the whole file; plugins, repositories, the dependency set
(`compileOnly project(':')` etc.) and test conventions are inherited from `sandbox/build.gradle`:

```groovy
/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

description = "Experimental ACME engine for k-NN"
```

That's it — `settings.gradle` discovers any `sandbox/<dir>` containing a `build.gradle` when the flag is
on. Verify:

```bash
./gradlew projects -Pknn.sandbox.enabled=true | grep acme     # → Project ':sandbox:acme'
./gradlew projects | grep sandbox                             # → nothing (default build unaffected)
```

### Step 2 — the library: what your engine can do

Extend [`NativeLibrary`](../src/main/java/org/opensearch/knn/index/engine/NativeLibrary.java) and declare
your method(s), file extension, and capability flags. The complete minimal set below **compiles and maps
as-is** — steps 3–8 build on these exact classes. (In-tree reference:
[`FixtureLibrary`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureLibrary.java); the Intel
SVS tenant from [#3370](https://github.com/opensearch-project/k-NN/pull/3370) is the full-size example
with encoders and compression-level mapping.)

`AcmeConstants.java`:

```java
public final class AcmeConstants {
    public static final String ACME_ENGINE_NAME = "acme";
    public static final String METHOD_ACME = "acme_method";
    public static final String ACME_EXTENSION = ".acme";
    private AcmeConstants() {}
}
```

`AcmeMethod.java` — one method, L2 only, no training, no encoders (grow it later):

```java
@ExperimentalAlgorithm(description = "ACME method", since = "3.9.0")
public class AcmeMethod extends AbstractFaissMethod {

    public static final MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(AcmeConstants.METHOD_ACME)
        .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
        .setKnnLibraryIndexingContextGenerator((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
            Map<String, Object> methodAsMap = new HashMap<>();
            methodAsMap.put(KNNConstants.NAME, methodComponent.getName());
            methodAsMap.put(
                KNNConstants.PARAMETERS,
                MethodComponent.getParameterMapWithDefaultsAdded(methodComponentContext, methodComponent, knnMethodConfigContext)
            );
            return KNNLibraryIndexingContextImpl.builder().parameters(methodAsMap).build();
        })
        .build();

    public AcmeMethod() {
        super(METHOD_COMPONENT, Set.of(SpaceType.L2), new AcmeSearchContext());
    }
}
```

`AcmeLibrary.java` — with a working minimal resolver, so a live mapping resolves defaults:

```java
@ExperimentalAlgorithm(description = "ACME engine library", since = "3.9.0")
public class AcmeLibrary extends NativeLibrary {

    public static final AcmeLibrary INSTANCE = new AcmeLibrary();

    private final MethodResolver methodResolver = new AcmeMethodResolver();

    private AcmeLibrary() {
        super(Map.of(AcmeConstants.METHOD_ACME, new AcmeMethod()), Map.of(), "1", AcmeConstants.ACME_EXTENSION);
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        // faiss-format distances? delegate: return Faiss.INSTANCE.score(rawScore, spaceType);
        return spaceType.scoreTranslation(rawScore);
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        return distance;
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return score;
    }

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        return methodResolver.resolveMethod(knnMethodContext, knnMethodConfigContext, shouldRequireTraining, spaceType);
    }

    @Override
    public boolean supportsIterativeBuild() {
        return true;
    }

    @Override
    public boolean createsCustomSegmentFiles() {
        return true;
    }
}
```

`AcmeMethodResolver.java` — fills defaults, rejects what the engine can't serve:

```java
@ExperimentalAlgorithm(description = "ACME method resolver", since = "3.9.0")
public class AcmeMethodResolver extends AbstractMethodResolver {

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        ValidationException validationException = validateNotTrainingContext(
            shouldRequireTraining,
            KNNEngine.getEngine(AcmeConstants.ACME_ENGINE_NAME),
            null
        );
        if (validationException != null) {
            throw validationException;
        }
        KNNMethodContext resolved = initResolvedKNNMethodContext(
            knnMethodContext,
            KNNEngine.getEngine(AcmeConstants.ACME_ENGINE_NAME),
            spaceType,
            AcmeConstants.METHOD_ACME
        );
        resolveMethodParams(resolved.getMethodComponentContext(), knnMethodConfigContext, AcmeMethod.METHOD_COMPONENT);
        knnMethodConfigContext.setCompressionLevel(CompressionLevel.x1);
        return ResolvedMethodContext.builder().knnMethodContext(resolved).compressionLevel(CompressionLevel.x1).build();
    }
}
```

The capability flags are how the core folds you into its behavior **without knowing your name**:

| Flag | What it buys you |
|---|---|
| `supportsIterativeBuild()` | The memory-efficient iterative build path (`initIndex` + `insertToIndex` batches + `writeIndex`) instead of the one-shot build. |
| `createsCustomSegmentFiles()` | The codec writes/reads segment files with your extension and routes them back to your engine (`KNNEngine.getEngineNameFromPath`). |
| `supportsFilters()` | Pre-filtered k-NN search is allowed on your engine. |
| `supportsRadialSearch()` | Distance/score-threshold (radial) queries are allowed, and `NativeEngineService.radiusQueryIndex` becomes reachable. |
| `supportsNestedFields()` | k-NN on nested fields is allowed on your engine. |

Your method's `KNNMethod` also returns your `KNNLibrarySearchContext` (step 6) and your
`KNNLibraryIndexingContext`. Annotate public classes with
[`@ExperimentalAlgorithm`](common/src/main/java/org/opensearch/knn/sandbox/ExperimentalAlgorithm.java):

```java
@ExperimentalAlgorithm(description = "ACME graph engine", since = "3.9.0")
```

### Step 3 — the native service: your engine's index lifecycle

Implement [`NativeEngineService`](../src/main/java/org/opensearch/knn/jni/NativeEngineService.java).
Reference: [`FixtureNativeEngineService`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureNativeEngineService.java)
shows the shape; a native tenant's implementation is a thin adapter onto its own static JNI class:

```java
public class AcmeNativeEngineService extends AbstractNativeEngineService {

    public AcmeNativeEngineService() {
        super(AcmeConstants.ACME_ENGINE_NAME);
    }

    @Override
    public long initIndex(long numDocs, int dim, Map<String, Object> parameters) {
        return AcmeService.initIndex(numDocs, dim, parameters);
    }
    // ... override insertToIndex/writeIndex/loadIndex/queryIndex/free the same way ...
    // Everything you DON'T override (template builds, radial search, ...) inherits a descriptive
    // UnsupportedOperationException from AbstractNativeEngineService — declining is your engine's
    // decision, not the core's. (Implementing NativeEngineService directly is equally valid; a
    // pure-JVM tenant can skip the native service entirely — KNNEngineDefinition.nativeService()
    // defaults to null.)
}
```

And the JNI binding class (only for native tenants — the validated SVS tenant's `SvsService` is the
reference; note **all** `System.loadLibrary` calls must go through `KNNLibraryLoader`, enforced by the
`validateLibraryUsage` build task):

```java
public class AcmeService {
    private static final String ACME_JNI_LIBRARY_NAME = "opensearchknn_acme";

    static {
        // Loads the best SIMD variant (_avx512_spr/_avx512/_avx2/plain) for the host, same policy as faiss.
        KNNLibraryLoader.loadLibraryByVariant(ACME_JNI_LIBRARY_NAME);
        initLibrary();
        KNNEngine.getEngine(AcmeConstants.ACME_ENGINE_NAME).setInitialized(true);
    }

    public static native long initIndex(long numDocs, int dim, Map<String, Object> parameters);
    // ... your native surface ...
    public static native void initLibrary();
}
```

Keep the `NativeEngineService` free of static references to the JNI class's initialization triggers:
registration happens at node startup, but the native library must load **lazily on first use** (the
`new AcmeNativeEngineService()` your provider creates must not class-init `AcmeService`). The fixture and
SVS both follow this pattern; it was validated explicitly (`SVS native lib not loaded until first use`).

**Pure-JVM engine?** Implement this same service in plain Java — no JNI binding class, no native library,
and skip step 5 entirely. Two extras make the data path JVM-clean:

* Override `prefersJavaVectors()` to return `true` and implement the on-heap
  `insertToIndex(int[] docs, float[][] vectors, …)` overload — the iterative build path then delivers each
  vector batch on-heap (same batching, bounded by `knn.vector_streaming_memory.limit`) instead of as the
  address of an off-heap C++ vector, which plain Java could not portably dereference. This also skips the
  off-heap copy that only exists for JNI's benefit.
* Everything else is already JVM-clean: `writeIndex`/`loadIndex` are stream-shaped, and the wrappers have
  a pure-Java surface — `IndexOutputWithBuffer.writeBytes(byte[], off, len)` to stream your serialized
  index out, `IndexInputWithBuffer.length()`/`readBytes(byte[], off, len)` to read it back (the JNI
  transfer-buffer methods are for native tenants). `queryIndex` receives `float[]`, and your "memory
  addresses" are opaque handles — use keys into your own on-heap registry, exactly like the fixture
  service (the running existence proof of a pure-Java `NativeEngineService`).

Declare your engine's Java library in `bundledRuntime` so it ships in the sandbox-enabled zip (step 9).

### Step 4 — register the engine (the SPI)

Reference: [`FixtureEngineProvider`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureEngineProvider.java) — a
real tenant's provider is identical, in **main** sources:

```java
public class AcmeEngineProvider implements KNNEngineDefinition {
    private final NativeEngineService nativeService = new AcmeNativeEngineService();

    @Override
    public String engineName() {
        return "acme";           // what users type in their mapping: "engine": "acme"
    }

    @Override
    public KNNLibrary library() {
        return AcmeLibrary.INSTANCE;
    }

    @Override
    public NativeEngineService nativeService() {
        return nativeService;
    }
}
```

`sandbox/acme/src/main/resources/META-INF/services/org.opensearch.knn.index.engine.KNNEngineDefinition`:

```
org.opensearch.knn.sandbox.acme.AcmeEngineProvider
```

Engine names must not collide with built-ins (`faiss`, `lucene`, `nmslib`) — the registry rejects the
collision at startup. Any number of tenants can be registered simultaneously.

### Step 5 — the native build: your own `.so`, your own library versions

**Pure-JVM tenant?** Skip this step — with no `jni/sandbox/acme/` directory the native build is untouched
(your engine's Java library ships via `bundledRuntime`, step 9).

Everything native lives under `jni/sandbox/acme/`, configured only when `-DCONFIG_SANDBOX=ON` (which
`-Pknn.sandbox.enabled=true` passes for you).

```bash
mkdir -p jni/sandbox/acme/{src,include}
```

`jni/sandbox/acme/tenant.cmake` (discovered automatically by `jni/sandbox/tenants.cmake`) vendors the
library your engine is built on with standard CMake and hands it to the shared helper in
[`jni/sandbox/cmake/SandboxTenant.cmake`](../jni/sandbox/cmake/SandboxTenant.cmake) — the extension point
is `knn_sandbox_add_jni_library`, and it is library-agnostic:

```cmake
include(ExternalProject)

# Vendor YOUR OWN copy of the library your engine is built on, at YOUR OWN pinned commit, built statically
# (PIC) inside this tenant's binary dir. Pin an exact commit — the supply-chain rule below applies to
# anything tenant.cmake downloads.
ExternalProject_Add(acme_vendor_ep
    GIT_REPOSITORY    https://github.com/example/libacme.git
    GIT_TAG           0123456789abcdef0123456789abcdef01234567
    PREFIX            ${CMAKE_BINARY_DIR}/acme_vendor
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_SHARED_LIBS=OFF
    INSTALL_COMMAND   ""
    BUILD_BYPRODUCTS  ${CMAKE_BINARY_DIR}/acme_vendor/src/acme_vendor_ep-build/libacme.a
)
add_library(acme_vendor STATIC IMPORTED)
set_target_properties(acme_vendor PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/acme_vendor/src/acme_vendor_ep-build/libacme.a")

# The isolated JNI library: hidden visibility + JNI-only exports + private static marshalling helpers +
# $ORIGIN rpath. See "Symbol isolation" below for what this buys you.
knn_sandbox_add_jni_library(opensearchknn_acme
    SOURCES
        ${KNN_SANDBOX_TENANT_DIR}/src/org_opensearch_knn_sandbox_acme_AcmeService.cpp
        ${KNN_SANDBOX_TENANT_DIR}/src/acme_wrapper.cpp
    INCLUDE_DIRS
        ${KNN_SANDBOX_TENANT_DIR}/include
        ${CMAKE_BINARY_DIR}/acme_vendor/src/acme_vendor_ep
    LINK_LIBRARIES
        acme_vendor
    DEPENDS
        acme_vendor_ep
)
```

The isolation recipe applies whatever the library is — hidden visibility and the JNI-only export list keep
*your* library's symbols local — and a prebuilt runtime `.so` also works: install it next to your JNI
library and the `$ORIGIN` rpath resolves it at load time.

**Building on faiss?** The one uniquely hazardous case is embedding a *different* faiss beside the one this
plugin already ships (two `faiss::*` symbol sets in one JVM — which is exactly what the isolation recipe
exists for), and doing it well needs host-side knowledge: build your faiss static + PIC with a
SIMD-variant choice mirroring the core's `init-faiss.cmake` detection, and re-supply BLAS/LAPACK/OpenMP on
your link line (a static faiss archive doesn't carry them). A shared `knn_sandbox_vendor_faiss` helper
packaging that exact recipe ships with the first faiss-based tenant, so faiss tenants get it as a
one-liner — follow that tenant's `tenant.cmake` as the worked example once it lands.

Gradle discovers your native target by convention: a tenant with `jni/sandbox/acme/tenant.cmake` builds
`opensearchknn_acme` automatically.

Build it:

```bash
./gradlew buildJniLib -Pknn.sandbox.enabled=true
ls jni/build/release/ | grep acme          # → libopensearchknn_acme*.so
```

**JNI headers are generated for you**: because `jni/sandbox/acme/` exists, compiling your Java
(`./gradlew :sandbox:acme:compileJava -Pknn.sandbox.enabled=true`) writes the header for `AcmeService`'s
native methods to `sandbox/acme/build/generated-jni-headers/`. Copy it into `jni/sandbox/acme/include/`
(headers are checked in so the native build never depends on a Java compile) and re-copy whenever the
native method surface changes. ⚠ **Underscores mangle**: a package or class name containing `_` becomes
`_1` in JNI symbol and file names (e.g. package `my_engine` → `Java_org_opensearch_knn_sandbox_my_1engine_...`);
the generated header gets this right — hand-written symbols often don't.

Implement the header in `src/org_opensearch_knn_sandbox_acme_AcmeService.cpp` using the marshalling helpers
from `jni/include/jni_util.h` and `jni/include/commons.h` (compiled privately into your library by the
harness). The native contract in brief:

| You receive | What it is |
|---|---|
| `parameters` (jobject) | A Java `Map` — read with `jniUtil->ConvertJavaMapToCppMap`; top level has `spaceType`, `indexThreadQty`; your method parameters are under the `parameters` sub-map. |
| `vectorsAddress` (jlong) | A pointer to an off-heap `std::vector<float>` owned by the JVM side (`reinterpret_cast<std::vector<float>*>`); read it, never free it. |
| `filterIdsType` / `filteredIds` | Pre-filter doc-id set: type 0 = a Lucene FixedBitSet-style jlong bitmap, 1 = a plain id batch. Only reachable if you declare `supportsFilters()`. |
| `parentIds` | Nested-field parent boundaries; only reachable if you declare `supportsNestedFields()`. |
| Return of `initIndex`/`loadIndex` | An opaque handle (usually a heap pointer you `reinterpret_cast`); the core passes it back to your other ops and to `free` — you own that memory. |

**Symbol isolation** — why the harness recipe matters: your library statically embeds *your* faiss while
`libopensearchknn_faiss` embeds the *k-NN-pinned* faiss, and both are dlopen'd into one JVM. The harness
compiles you with hidden visibility and links you with a version script
([`jni/sandbox/cmake/jni_exports.version`](../jni/sandbox/cmake/jni_exports.version)) that exports **only**
`JNI_OnLoad`/`JNI_OnUnload`/`Java_org_opensearch_knn_*`, so the two `faiss::*` symbol sets cannot interpose
(an ODR hazard that would route calls into the wrong faiss). This exact recipe was validated for the SVS
tenant: 0 foreign symbols in the core faiss library, JNI-only exports in the tenant library. Verify yours:

```bash
nm -D --defined-only jni/build/release/libopensearchknn_acme*.so | grep -cv ' Java_\| JNI_'   # → 0
```

**Supply-chain rule**: any artifact your `tenant.cmake` downloads (a runtime `.so`, a dataset) **must** be
checksum-pinned (`URL_HASH SHA256=...`); if you accept a user-overridable URL, require a paired
`-D..._SHA256` with, at most, an explicit dev-only bypass flag. If your tenant ships a runtime shared
library next to the JNI library, the harness's `$ORIGIN` rpath resolves it from the same directory.

**Tenant-specific cmake overrides**: users pass them through the generic
`-Psandbox.cmake.args="-DYOUR_FLAG=value;-DANOTHER=value"` pass-through (semicolon-separated) — your
tenant never edits the root `build.gradle`.

### Step 6 — engine-specific query parameters

Two declarations, one per half of the contract. **The name** goes in your `KNNEngineDefinition` (this is
the parse-time allowlist — it tells the REST/gRPC layers the name is yours instead of a typo; a name no
registered engine declares is rejected at parse exactly as it always was):

```java
public class AcmeEngineProvider implements KNNEngineDefinition {
    // ... engineName() / library() / nativeService() from step 4 ...

    @Override
    public Set<String> engineSpecificQueryParameters() {
        return Set.of("acme_beam_width");
    }
}
```

**The value rules** go in your search context — reference:
[`FixtureSearchContext`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureSearchContext.java):

```java
public class AcmeSearchContext implements KNNLibrarySearchContext {
    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return Map.of("acme_beam_width", new Parameter.IntegerParameter("acme_beam_width", null, (v, context) -> v > 0));
    }
}
```

Because of the query extension point:

* REST/gRPC accept `"method_parameters": {"acme_beam_width": 64}` and defer validation to your engine
  (a genuinely unknown parameter is still rejected — at parse if no engine declared it, or by the
  engine-aware check with your engine's name in the message if it targets the wrong engine);
* the node-to-node wire carries the parameter to every shard in a multi-node cluster (core-known
  parameters ride the unchanged upstream format; yours ride a feature-gated appendix) and your validator
  runs per-shard. If a node in the cluster is too old to carry your parameter, the query **fails loudly**
  at the coordinator — it is never silently dropped;
* your `NativeEngineService.queryIndex` receives it in `methodParameters`.

Do **not** add engine parameters to the core `MethodParameter` enum — that enum is only for parameters the
core itself owns.

### Step 7 — tests

**Unit tests** (`sandbox/acme/src/test/...`, run by `./gradlew :sandbox:acme:test -Pknn.sandbox.enabled=true`):
extend `OpenSearchTestCase`; the three fixture test classes are ready-made templates —
[`FixtureEngineRegistrationTests`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureEngineRegistrationTests.java)
(your engine resolves, capabilities fold),
[`FixtureJNIServiceDispatchTests`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureJNIServiceDispatchTests.java)
(dispatch reaches your service), and
[`FixtureQueryParamDeferralTests`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureQueryParamDeferralTests.java)
(deferral, engine-aware validation, wire round-trip).

**REST integration tests are auto-wired** (real cluster, real native library): put `*IT` classes
extending `KNNRestTestCase` in your test sources and you're done — the shared sandbox config detects them
and creates an `integTest` task whose cluster runs the sandbox-enabled plugin zip with the JNI build dir on
the node's `java.library.path` (security-enabled runs honor the usual `-Dhttps/-Duser/-Dpassword`). Nothing
to copy into your build.gradle — the wiring lives once in [`sandbox/build.gradle`](build.gradle) (cluster from the bundled plugin zip, node-side `java.library.path`, `tests.rest.cluster` endpoints).

```bash
./gradlew :sandbox:acme:integTest -Pknn.sandbox.enabled=true
```

### Step 8 — run it live

```bash
./gradlew run -Pknn.sandbox.enabled=true          # single node with the sandbox-enabled plugin
```

```bash
curl -X PUT "localhost:9200/acme-index" -H 'Content-Type: application/json' -d '{
  "settings": { "index.knn": true },
  "mappings": { "properties": { "vec": {
    "type": "knn_vector", "dimension": 4,
    "method": { "name": "acme_method", "engine": "acme", "space_type": "l2" }
  } } }
}'

curl -X POST "localhost:9200/acme-index/_search" -H 'Content-Type: application/json' -d '{
  "query": { "knn": { "vec": {
    "vector": [1.0, 2.0, 3.0, 4.0], "k": 3,
    "method_parameters": { "acme_beam_width": 64 }
  } } }
}'
```

On a default build (`./gradlew run`), `"engine": "acme"` is rejected with `Invalid engine type: acme` —
the correct behavior: your tenant does not exist there.

### Step 9 — CI and contribution hygiene

* The [`sandbox-check`](../.github/workflows/sandbox-check.yml) workflow runs
  `./gradlew -p sandbox test -Pknn.sandbox.enabled=true` on every PR — your unit tests join CI with zero
  workflow changes. It runs on all PRs (not just sandbox ones) because a core refactor can break the SPI
  seam without touching `sandbox/**`.
* ⚠ Your tenant **jar** (plus `:sandbox:common`'s) is bundled into the plugin zip automatically — but a
  plain `implementation` dependency on anything OpenSearch does not already provide will be **missing at
  runtime**. A third-party Java library your engine is built on (a pure-JVM engine's equivalent of
  vendoring) goes in the `bundledRuntime` configuration instead; every jar it resolves is bundled beside
  your tenant jar and it feeds your compile/test classpaths:

  ```groovy
  dependencies {
      bundledRuntime("com.example:acme-engine:1.2.3") {
          // The k-NN plugin already bundles slf4j-api — a duplicate jar fails the jarhell check.
          exclude group: "org.slf4j", module: "slf4j-api"
      }
  }
  ```

  Pin exact versions and keep the set minimal: exclude anything OpenSearch or this plugin already
  provides, and remember these jars are snapshot-only experiments — the full license/audit treatment
  happens at graduation.
* Add a CHANGELOG entry; sign off your commits (DCO); run
  `./gradlew spotlessApply -Pknn.sandbox.enabled=true` before pushing.
* Keep your tenant PR free of unrelated changes — sandbox PRs get the same review bar as any other.

### Step 10 — graduation (or removal)

**Graduation criteria** (all of them):

| Criterion | Threshold |
|---|---|
| **Stability** | No critical bugs after a minimum incubation period |
| **Performance** | Meets or exceeds benchmarks comparable to existing stable algorithms |
| **Test coverage** | Comprehensive unit and integration tests |
| **Community feedback** | Positive reception from early adopters running sandbox-enabled snapshots |
| **Documentation** | Complete API documentation and usage guides |

**What graduation looks like mechanically** — the sandbox is shaped so incubated code moves, not rewrites:
Java sources move from `sandbox/acme/src/main/...` into `src/main/...` (the SPI classes can stay as they
are or become direct wiring — a graduated engine may keep its `KNNEngineDefinition` or be added as a
built-in singleton); native code moves from `jni/sandbox/acme/` into `jni/src` + `jni/CMakeLists.txt`
proper (or keeps its isolated-library shape if its vendored dependencies require it); tests move to the
corresponding core locations; the `@ExperimentalAlgorithm` annotations come off. The graduation PR is where
BWC policy, settings, and docs get the full production treatment.

**Removing a tenant** that didn't pan out:

```bash
git rm -r sandbox/acme jni/sandbox/acme
```

No other file changes — nothing else in the repository references a tenant by name. Be honest in the
removal PR: indices created with the removed engine become unopenable (their segment files carry the
tenant's extension); acceptable for an experimental, snapshot-only engine, but say it.

---

## Design notes (for reviewers and the curious)

* **Why per-tenant subprojects** (vs one sandbox module with tenant packages): tenants stay independently
  buildable, testable, and removable, and two tenants can pin conflicting dependency versions. This is the
  core-sandbox shape adapted to a single-plugin repo: core's `sandbox/{libs,modules,plugins}` mirror what a
  feature graduates *to*; k-NN tenants graduate into the plugin itself, so the mirror is "one subtree per
  incubating engine".
* **Why the fixture engine lives in test sources**: "never shipped" is structural — a test source set is
  excluded from every jar by construction, so no packaging rule can regress it.
* **Why the version script + hidden visibility are non-negotiable** for native tenants: two different
  versions of the same C++ library in one process interpose through their exported symbols; the validated
  failure mode is calls silently routing into the wrong build. Export-only-JNI makes the tenant a black box.
* **Why query-param deferral ships together with the wire appendix**: deferral without a wire that can
  carry the parameter *appears* to work on a single node and silently drops engine parameters on any
  multi-node hop (validated with positive and negative controls in the #3370 work). The appendix carries
  them when every node supports it; when one can't, serialization fails loudly at the coordinator — a
  silent drop is impossible by construction. And because parse admits only names a registered engine
  declared, a default build's parse and wire behavior is identical to a tree with no sandbox at all. The
  sandbox tests pin all of it.
