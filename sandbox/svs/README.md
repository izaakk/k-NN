# SVS sandbox tenant

Intel SVS (Scalable Vector Search) as a sandbox tenant engine: `engine: "svs"`, method `svs_vamana`,
encoders `flat` / `sq` (fp16, sq8) / `lvq` (4x0, 4x4, 4x8). Built entirely on the sandbox extension
points — zero core code. See `sandbox/README.md` for the extension-point contract and gating; this file
documents tenant-specific behavior a user or reviewer needs to know.

## Search surface

- **Top-k kNN** with optional efficient pre-filtering, and query-time `method_parameters`:
  `search_window_size` (the SVS accuracy knob).
- **Radial search** (`max_distance` / `min_score`), also with optional filtering, with two caveats:
  - The SVS index only accepts a strictly positive faiss-domain radius. Thresholds that convert to a
    non-positive radius — inner-product `max_distance >= 0`, inner-product `min_score < 1`, cosine
    `max_distance >= 1`, cosine `min_score <= 0.5` — are rejected at query time with a descriptive
    error. L2 is fully supported.
  - Under compressed storage (LVQ/SQ), distances are computed in the compressed domain, so membership
    near the radial boundary is approximate; unlike core faiss's 1-bit-SQ path there is no rescore
    wrapper. Graph-based radial search is approximate for faiss HNSW too — compression only widens the
    boundary fuzz.
  - Filtered radial queries whose filter selects a small candidate set fall back to core's exact search,
    which today gates radial on `engine == FAISS` and rejects registered engines (a core follow-up; the
    check should be capability-driven). Workaround until then: set the index setting
    `index.knn.advanced.filtered_exact_search_threshold: 0` to keep filtered radial on the native ANN
    path.
- **Nested (multi-vector) fields are supported** through the SVS runtime's `IDGrouper`: the native layer
  passes a one-best-per-parent grouper to `VamanaIndex#search`/`#range_search`, so `k` counts distinct
  parents with the same guarantee the patched-faiss engines provide (exact grouping during graph search,
  not an oversampled post-grouping emulation). Requires a `libsvs-runtime` build that carries the
  `IDGrouper` API (see the upstream feature request); the grouped search does not use the runtime's
  filtered-search early-exit heuristics, so highly-restrictive nested+filter queries scan until k parents
  are found or the segment's graph is exhausted.
  - Nested **radial** search is wired the same way (best in-radius child per parent). Filtered nested
    radial is subject to the same core exact-search fallback gate as plain filtered radial (below).

## Platform notes

- **LVQ requires Intel AVX-512.** The check runs on the node that validates the mapping (via
  `SvsService#isLvqLeanvecEnabled`), not on every data node: in a heterogeneous cluster a mapping
  accepted by an AVX-512 coordinating node will still fail at index-build time on a data node without
  AVX-512. Keep clusters homogeneous when using LVQ.
- **SIMD variant selection reuses the faiss-named settings.** The tenant `.so` is built at the host's
  best variant (e.g. `libopensearchknn_svs_avx512_spr.so`) with a plain-library fallback at load time,
  and the `knn.faiss.avx512.disabled` / `knn.faiss.avx512_spr.disabled` settings gate variant selection
  for **every** library, including this tenant — disabling a faiss variant also disables the SVS variant.
- **OpenMP**: the tenant links `libgomp` dynamically per the sandbox README's threading rule; the
  vendored faiss and SVS runtime share the process-global OpenMP runtime with the built-in libraries.

## Build

Requires `-Pknn.sandbox.enabled=true` (Gradle) which passes `-DCONFIG_SANDBOX=ON` to CMake.
`jni/sandbox/svs/tenant.cmake` vendors an unpatched upstream faiss pinned to an SVS-capable commit
(built with `FAISS_ENABLE_SVS=ON`) and fetches the prebuilt, sha256-pinned `libsvs-runtime` conda
artifact; `libsvs_runtime.so.0` is installed beside the JNI libraries and resolved via `$ORIGIN`.
Offline builds: `-DSVS_RUNTIME_PREFIX=<dir>` or `-DSVS_RUNTIME_URL=<artifact>` +
`-DSVS_RUNTIME_SHA256=<hex>`, passed through `-Psandbox.cmake.args`.
