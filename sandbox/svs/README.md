# SVS sandbox tenant

Intel SVS (Scalable Vector Search) as a sandbox tenant engine: `engine: "svs"`, method `svs_vamana`
(static Vamana graph), encoders `flat` / `sq` (fp16, sq8) / `lvq` (4x0, 4x4, 4x8) / `leanvec` (4x4, 4x8,
8x8 with dimensionality reduction, trained at segment-build time). Built entirely on the sandbox
extension points, with zero core code. See `sandbox/README.md` for the extension-point contract and gating; this
file documents tenant-specific behavior a user or reviewer needs to know.

## Index build

Every segment is built as a **static** SVS Vamana index (`IndexSVSVamana::is_static`): OpenSearch
segments are write-once, so the JNI layer buffers the segment's vector stream and hands it to the
runtime in a single one-shot add at write time. The dynamic SVS index is never used. Note for
upgrades within the sandbox's experimental life: the vendored faiss pin that introduced static
support also extended the SVS serialization format, so `.svs` segments written by older sandbox
builds are not readable by this one (reindex required; the library version string was bumped to
mark the break).

Build memory: because the static index is built in one shot, a segment build holds the whole
segment's vectors in native memory as float32 (`vectors x dimension x 4` bytes) in addition to the
graph under construction. Size data-node headroom for the largest merge, not for the flush size.

### LeanVec encoder (deferred training)

`leanvec` stores LVQ-style compressed vectors under a learned projection to a reduced dimensionality,
searching the graph on the reduced vectors and re-ranking on the full ones. Parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `primary_bits` x `residual_bits` | 4 x 8 | storage kind; supported: 4x4, 4x8, 8x8 |
| `dimensions` | 0 | reduced dimensionality; 0 lets the runtime pick `dimension / 2`; must be `<= dimension` |
| `training_threshold` | 100000 | vectors sampled to train the projection (`0` = default, else `>= 1000`) |
| `rough_training_threshold` | 10000 | segments smaller than this skip LeanVec entirely (see ladder) |

The projection needs a training sample, and OpenSearch has no separate training step, so training
happens at segment-build time on a uniform sample of the segment's own vectors (never the first N:
the head of a merged stream is distributionally skewed). This yields a per-segment quality ladder:
segments below `rough_training_threshold` are built as the training-free LVQ equivalent (4x4 ->
LVQ4x4, 4x8 and 8x8 -> LVQ4x8) and upgrade to real LeanVec when a merge produces a segment at or
above the threshold; from `training_threshold` upward the sample size is capped there. Every
segment serves searches at each rung, so recall improves monotonically with merges. A mapping may
not combine `compression_level` with `leanvec` (its ratio depends on `dimensions`); the mapping is
rejected rather than the level being silently dropped.

## Search surface

- **Top-k kNN** with optional efficient pre-filtering, and query-time `method_parameters`:
  `search_window_size` (the SVS accuracy knob, analogous to `ef_search`) and `search_buffer_capacity`
  (the candidate-retention pool; for LeanVec it is the re-rank pool, so capacity above the window trades
  a little latency for recall). Both override the index-level defaults set in the mapping; capacity is
  clamped natively to `>= search_window_size`. Radial search honors both parameters the same way.
- **Radial search** (`max_distance` / `min_score`), also with optional filtering, with two caveats:
  - The SVS index only accepts a strictly positive faiss-domain radius. Thresholds that convert to a
    non-positive radius (inner-product `max_distance >= 0`, inner-product `min_score < 1`, cosine
    `max_distance >= 1`, cosine `min_score <= 0.5`) are rejected at query time with a descriptive
    error. L2 is fully supported.
  - Under compressed storage (LVQ/SQ), distances are computed in the compressed domain, so membership
    near the radial boundary is approximate; unlike core faiss's 1-bit-SQ path there is no rescore
    wrapper. Graph-based radial search is approximate for faiss HNSW too, so compression only widens the
    boundary fuzz.
  - Filtered radial queries whose filter selects a small candidate set fall back to core's exact search,
    which today gates radial on `engine == FAISS` and rejects registered engines (a core follow-up; the
    check should be capability-driven). Workaround until then: set the index setting
    `index.knn.advanced.filtered_exact_search_threshold: 0` to keep filtered radial on the native ANN
    path.
- **Nested fields are not supported** and are rejected at query time: per-parent grouping must happen
  inside the SVS runtime's graph traversal to guarantee k distinct parents, and the prebuilt
  `libsvs_runtime` search API does not expose a grouping construct yet (feature request filed upstream).

## Platform notes

- **LVQ requires Intel AVX-512.** The check runs on the node that validates the mapping (via
  `SvsService#isLvqLeanvecEnabled`), not on every data node: in a heterogeneous cluster a mapping
  accepted by an AVX-512 coordinating node will still fail at index-build time on a data node without
  AVX-512. Keep clusters homogeneous when using LVQ.
- **SIMD variant selection reuses the faiss-named settings.** The tenant `.so` is built at the host's
  best variant (e.g. `libopensearchknn_svs_avx512_spr.so`) with a plain-library fallback at load time,
  and the `knn.faiss.avx512.disabled` / `knn.faiss.avx512_spr.disabled` settings gate variant selection
  for **every** library, including this tenant, so disabling a faiss variant also disables the SVS variant.
- **OpenMP**: the tenant links `libgomp` dynamically per the sandbox README's threading rule; the
  vendored faiss and SVS runtime share the process-global OpenMP runtime with the built-in libraries.

## Build

Requires `-Pknn.sandbox.enabled=true` (Gradle) which passes `-DCONFIG_SANDBOX=ON` to CMake.
`jni/sandbox/svs/tenant.cmake` vendors an unpatched upstream faiss pinned to an SVS-capable commit
(built with `FAISS_ENABLE_SVS=ON`) and fetches the prebuilt, sha256-pinned `libsvs-runtime` conda
artifact; `libsvs_runtime.so.0` is installed beside the JNI libraries and resolved via `$ORIGIN`.
Offline builds: `-DSVS_RUNTIME_PREFIX=<dir>` or `-DSVS_RUNTIME_URL=<artifact>` +
`-DSVS_RUNTIME_SHA256=<hex>`, passed through `-Psandbox.cmake.args`.
