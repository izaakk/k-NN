/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

// Micro-benchmark for the SQ (4-bit query x 1-bit data) bulk similarity kernel, standalone
// (no JNI / Faiss dependency), in the style of the FP16 bulk benchmarks of #3172 / #3341.
//
// Kernels:
//   BM_Old      : avx512_4bitDotProductBatch as on main @5437d97 (nibble-LUT popcount, scalar byte tail for
//                 codeBytes % 64, scalar int4BitDotProduct for the vector remainder).
//   BM_New      : masked-load tail, same nibble-LUT popcount (what the avx512 build gets).
//   BM_NewBitalg: masked-load tail + vpopcntb (what the avx512_spr build gets); only when compiled with
//                 -mavx512bitalg.
//
// Compile (AVX-512 host):
//   g++ -O3 -std=c++17 -mavx512f -mavx512bw -mavx512vl -mavx512vpopcntdq -mavx512bitalg -mprefer-vector-width=512 \
//       sq_bulk_similarity_bench.cpp -o sq_bulk_similarity_bench -lbenchmark -lpthread
// Run:
//   ./sq_bulk_similarity_bench --benchmark_counters_tabular=true
//   (optionally: setarch $(uname -m) -R ./sq_bulk_similarity_bench --benchmark_perf_counters=cycles,instructions)
//
// Each iteration scores `vecs` random vectors of dimension `dim` against one query, exactly like one
// calculateSimilarityInBulk call: dot products in batches of 8 / 4 / 1, then the OSQ correction terms.
// items_per_second = vectors scored per second.

#include <benchmark/benchmark.h>
#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE inline __attribute__((always_inline))
#else
    #define FORCE_INLINE inline
#endif

namespace {

constexpr int32_t CORRECTION_BYTES = 3 * sizeof(float) + sizeof(int32_t);
constexpr float FOUR_BIT_SCALE = 1.0f / 15.0f;

// ---------------------------------------------------------------------------------------------
// Shared pieces (identical in old and new)
// ---------------------------------------------------------------------------------------------
FORCE_INLINE void readDataCorrections(const uint8_t* ptr, float& ax, float& lx, float& additional, float& x1) {
    float lower, upper;
    std::memcpy(&lower, ptr, sizeof(float));
    std::memcpy(&upper, ptr + 4, sizeof(float));
    std::memcpy(&additional, ptr + 8, sizeof(float));
    int32_t componentSum;
    std::memcpy(&componentSum, ptr + 12, sizeof(int32_t));
    ax = lower;
    lx = upper - lower;
    x1 = static_cast<float>(componentSum);
}

FORCE_INLINE __m512i lut_popcnt_epi8(const __m512i v) {
    alignas(64) static const __m512i popLut = _mm512_setr_epi64(
        0x0302020102010100LL, 0x0403030203020201LL, 0x0302020102010100LL, 0x0403030203020201LL,
        0x0302020102010100LL, 0x0403030203020201LL, 0x0302020102010100LL, 0x0403030203020201LL);
    const __m512i lowMask = _mm512_set1_epi8(0x0F);
    const __m512i lo = _mm512_and_si512(v, lowMask);
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), lowMask);
    return _mm512_add_epi8(_mm512_shuffle_epi8(popLut, lo), _mm512_shuffle_epi8(popLut, hi));
}

#if defined(__AVX512BITALG__)
FORCE_INLINE __m512i bitalg_popcnt_epi8(const __m512i v) {
    return _mm512_popcnt_epi8(v);
}
#endif

// ---------------------------------------------------------------------------------------------
// Old kernel (main @5437d97)
// ---------------------------------------------------------------------------------------------
FORCE_INLINE int64_t old_int4BitDotProduct(const uint8_t* q, const uint8_t* d, const int32_t binaryCodeBytes) {
    int64_t result = 0;
    for (int32_t bitPlane = 0; bitPlane < 4; ++bitPlane) {
        const int32_t words = binaryCodeBytes >> 3;
        int64_t subResult = 0;
        for (int32_t w = 0; w < words; ++w) {
            uint64_t qWord, dWord;
            std::memcpy(&qWord, q + bitPlane * binaryCodeBytes + w * 8, sizeof(uint64_t));
            std::memcpy(&dWord, d + w * 8, sizeof(uint64_t));
            subResult += __builtin_popcountll(qWord & dWord);
        }
        for (int32_t r = words * 8; r < binaryCodeBytes; ++r) {
            subResult += __builtin_popcount((q[bitPlane * binaryCodeBytes + r] & d[r]) & 0xFF);
        }
        result += subResult << bitPlane;
    }
    return result;
}

template <int BATCH_SIZE>
FORCE_INLINE void old_4bitDotProductBatch(const uint8_t* queryPtr, uint8_t** dataVecs, const int32_t binaryCodeBytes, float* results) {
    const uint8_t* plane0 = queryPtr;
    const uint8_t* plane1 = queryPtr + binaryCodeBytes;
    const uint8_t* plane2 = queryPtr + 2 * binaryCodeBytes;
    const uint8_t* plane3 = queryPtr + 3 * binaryCodeBytes;
    __m512i acc[BATCH_SIZE];
    for (int32_t b = 0; b < BATCH_SIZE; ++b) acc[b] = _mm512_setzero_si512();
    int32_t i = 0;
    for (; i + 64 <= binaryCodeBytes; i += 64) {
        const __m512i q0 = _mm512_loadu_si512(plane0 + i), q1 = _mm512_loadu_si512(plane1 + i);
        const __m512i q2 = _mm512_loadu_si512(plane2 + i), q3 = _mm512_loadu_si512(plane3 + i);
        if (i + 64 < binaryCodeBytes) {
            __builtin_prefetch(plane0 + i + 64);
            for (int32_t b = 0; b < BATCH_SIZE; ++b) __builtin_prefetch(dataVecs[b] + i + 64);
        }
        for (int32_t b = 0; b < BATCH_SIZE; ++b) {
            const __m512i d = _mm512_loadu_si512(dataVecs[b] + i);
            const __m512i p0 = lut_popcnt_epi8(_mm512_and_si512(q0, d)), p1 = lut_popcnt_epi8(_mm512_and_si512(q1, d));
            const __m512i p2 = lut_popcnt_epi8(_mm512_and_si512(q2, d)), p3 = lut_popcnt_epi8(_mm512_and_si512(q3, d));
            __m512i weighted = _mm512_add_epi8(p0, _mm512_slli_epi16(p1, 1));
            weighted = _mm512_add_epi8(weighted, _mm512_slli_epi16(p2, 2));
            weighted = _mm512_add_epi8(weighted, _mm512_slli_epi16(p3, 3));
            acc[b] = _mm512_add_epi64(acc[b], _mm512_sad_epu8(weighted, _mm512_setzero_si512()));
        }
    }
    for (int32_t b = 0; b < BATCH_SIZE; ++b) results[b] = static_cast<float>(_mm512_reduce_add_epi64(acc[b]));
    for (; i < binaryCodeBytes; ++i) {
        const uint8_t q0b = plane0[i], q1b = plane1[i], q2b = plane2[i], q3b = plane3[i];
        for (int32_t b = 0; b < BATCH_SIZE; ++b) {
            const uint8_t db = dataVecs[b][i];
            results[b] += static_cast<float>(
                __builtin_popcount((q0b & db) & 0xFF) * 1 + __builtin_popcount((q1b & db) & 0xFF) * 2
                + __builtin_popcount((q2b & db) & 0xFF) * 4 + __builtin_popcount((q3b & db) & 0xFF) * 8);
        }
    }
}

struct OldKernel {
    template <int BATCH_SIZE>
    static FORCE_INLINE void batch(const uint8_t* q, uint8_t** vecs, const int32_t codeBytes, float* out) {
        old_4bitDotProductBatch<BATCH_SIZE>(q, vecs, codeBytes, out);
    }
    static FORCE_INLINE float single(const uint8_t* q, const uint8_t* vec, const int32_t codeBytes) {
        return static_cast<float>(old_int4BitDotProduct(q, vec, codeBytes));
    }
};

// ---------------------------------------------------------------------------------------------
// New kernel (this change): masked tail, one routine for every path, optional vpopcntb
// ---------------------------------------------------------------------------------------------
template <int BATCH_SIZE, __m512i (*POPCNT)(__m512i)>
FORCE_INLINE void new_chunk(const __mmask64 mask, const uint8_t* p0, const uint8_t* p1, const uint8_t* p2, const uint8_t* p3,
                            uint8_t** dataVecs, __m512i* acc) {
    const __m512i q0 = _mm512_maskz_loadu_epi8(mask, p0), q1 = _mm512_maskz_loadu_epi8(mask, p1);
    const __m512i q2 = _mm512_maskz_loadu_epi8(mask, p2), q3 = _mm512_maskz_loadu_epi8(mask, p3);
    for (int32_t b = 0; b < BATCH_SIZE; ++b) {
        const __m512i d = _mm512_maskz_loadu_epi8(mask, dataVecs[b]);
        const __m512i c0 = POPCNT(_mm512_and_si512(q0, d)), c1 = POPCNT(_mm512_and_si512(q1, d));
        const __m512i c2 = POPCNT(_mm512_and_si512(q2, d)), c3 = POPCNT(_mm512_and_si512(q3, d));
        __m512i weighted = _mm512_add_epi8(c0, _mm512_slli_epi16(c1, 1));
        weighted = _mm512_add_epi8(weighted, _mm512_slli_epi16(c2, 2));
        weighted = _mm512_add_epi8(weighted, _mm512_slli_epi16(c3, 3));
        acc[b] = _mm512_add_epi64(acc[b], _mm512_sad_epu8(weighted, _mm512_setzero_si512()));
    }
}

template <int BATCH_SIZE, __m512i (*POPCNT)(__m512i)>
FORCE_INLINE void new_4bitDotProductBatch(const uint8_t* queryPtr, uint8_t** dataVecs, const int32_t binaryCodeBytes, float* results) {
    const uint8_t* plane0 = queryPtr;
    const uint8_t* plane1 = queryPtr + binaryCodeBytes;
    const uint8_t* plane2 = queryPtr + 2 * binaryCodeBytes;
    const uint8_t* plane3 = queryPtr + 3 * binaryCodeBytes;
    __m512i acc[BATCH_SIZE];
    for (int32_t b = 0; b < BATCH_SIZE; ++b) acc[b] = _mm512_setzero_si512();
    uint8_t* chunkVecs[BATCH_SIZE];
    int32_t i = 0;
    for (; i + 64 <= binaryCodeBytes; i += 64) {
        if (i + 64 < binaryCodeBytes) {
            __builtin_prefetch(plane0 + i + 64);
            for (int32_t b = 0; b < BATCH_SIZE; ++b) __builtin_prefetch(dataVecs[b] + i + 64);
        }
        for (int32_t b = 0; b < BATCH_SIZE; ++b) chunkVecs[b] = dataVecs[b] + i;
        new_chunk<BATCH_SIZE, POPCNT>(~static_cast<__mmask64>(0), plane0 + i, plane1 + i, plane2 + i, plane3 + i, chunkVecs, acc);
    }
    if (i < binaryCodeBytes) {
        const __mmask64 tailMask = static_cast<__mmask64>((1ULL << (binaryCodeBytes - i)) - 1ULL);
        for (int32_t b = 0; b < BATCH_SIZE; ++b) chunkVecs[b] = dataVecs[b] + i;
        new_chunk<BATCH_SIZE, POPCNT>(tailMask, plane0 + i, plane1 + i, plane2 + i, plane3 + i, chunkVecs, acc);
    }
    for (int32_t b = 0; b < BATCH_SIZE; ++b) results[b] = static_cast<float>(_mm512_reduce_add_epi64(acc[b]));
}

template <__m512i (*POPCNT)(__m512i)>
struct NewKernel {
    template <int BATCH_SIZE>
    static FORCE_INLINE void batch(const uint8_t* q, uint8_t** vecs, const int32_t codeBytes, float* out) {
        new_4bitDotProductBatch<BATCH_SIZE, POPCNT>(q, vecs, codeBytes, out);
    }
    static FORCE_INLINE float single(const uint8_t* q, const uint8_t* vec, const int32_t codeBytes) {
        uint8_t* vecs[1] = {const_cast<uint8_t*>(vec)};
        float r;
        new_4bitDotProductBatch<1, POPCNT>(q, vecs, codeBytes, &r);
        return r;
    }
};

// ---------------------------------------------------------------------------------------------
// calculateSimilarityInBulk (SQ_IP) shape, parameterised by the dot-product kernel
// ---------------------------------------------------------------------------------------------
struct Dataset {
    int32_t dim = 0, codeBytes = 0, oneVectorBytes = 0, numVectors = 0;
    std::vector<uint8_t> query;    // 4 planes
    std::vector<uint8_t> vectors;  // codes + corrections
    float qLower = -0.3f, qUpper = 0.4f, qAdditional = 0.1f, centroidDp = 0.05f;
    int32_t qComponentSum = 0;
    uint8_t* vectorPtr(const int32_t id) const {
        return const_cast<uint8_t*>(vectors.data()) + static_cast<size_t>(id) * oneVectorBytes;
    }
};

Dataset makeDataset(const int32_t dim, const int32_t numVectors, const uint32_t seed) {
    Dataset ds;
    ds.dim = dim;
    ds.codeBytes = (dim + 7) / 8;
    ds.oneVectorBytes = ds.codeBytes + CORRECTION_BYTES;
    ds.numVectors = numVectors;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> byteDist(0, 255);
    std::uniform_int_distribution<int32_t> sumDist(0, dim);
    ds.query.resize(4 * ds.codeBytes);
    for (auto& b : ds.query) b = static_cast<uint8_t>(byteDist(rng));
    ds.qComponentSum = 15 * sumDist(rng);
    ds.vectors.resize(static_cast<size_t>(numVectors) * ds.oneVectorBytes);
    for (int32_t v = 0; v < numVectors; ++v) {
        uint8_t* vec = ds.vectorPtr(v);
        for (int32_t i = 0; i < ds.codeBytes; ++i) vec[i] = static_cast<uint8_t>(byteDist(rng));
        const float lower = -0.5f, upper = 0.6f, additional = 0.2f;
        const int32_t sum = sumDist(rng);
        std::memcpy(vec + ds.codeBytes, &lower, 4);
        std::memcpy(vec + ds.codeBytes + 4, &upper, 4);
        std::memcpy(vec + ds.codeBytes + 8, &additional, 4);
        std::memcpy(vec + ds.codeBytes + 12, &sum, 4);
    }
    return ds;
}

template <class Kernel>
void sqBulkIp(const Dataset& ds, const int32_t* ids, float* scores, const int32_t numVectors) {
    const uint8_t* queryPtr = ds.query.data();
    const int32_t dim = ds.dim;
    const int32_t binaryCodeBytes = ds.codeBytes;
    const float ay = ds.qLower;
    const float ly = (ds.qUpper - ds.qLower) * FOUR_BIT_SCALE;
    const float queryAdditional = ds.qAdditional;
    const float y1 = static_cast<float>(ds.qComponentSum);
    const float centroidDp = ds.centroidDp;
    uint8_t* vectors[8];
    int32_t processed = 0;
    auto finish = [&](const int32_t idx, uint8_t* vec, const float qc) {
        float ax, lx, additional, x1;
        readDataCorrections(vec + binaryCodeBytes, ax, lx, additional, x1);
        float s = ax * ay * dim + ay * lx * x1 + ax * ly * y1 + lx * ly * qc;
        s += queryAdditional + additional - centroidDp;
        scores[idx] = s < 0 ? 1 / (1 - s) : (1 + s);
    };
    for (; processed + 8 <= numVectors; processed += 8) {
        for (int32_t i = 0; i < 8; ++i) vectors[i] = ds.vectorPtr(ids[processed + i]);
        Kernel::template batch<8>(queryPtr, vectors, binaryCodeBytes, &scores[processed]);
        for (int32_t i = 0; i < 8; ++i) finish(processed + i, vectors[i], scores[processed + i]);
    }
    for (; processed + 4 <= numVectors; processed += 4) {
        for (int32_t i = 0; i < 4; ++i) vectors[i] = ds.vectorPtr(ids[processed + i]);
        Kernel::template batch<4>(queryPtr, vectors, binaryCodeBytes, &scores[processed]);
        for (int32_t i = 0; i < 4; ++i) finish(processed + i, vectors[i], scores[processed + i]);
    }
    for (; processed < numVectors; ++processed) {
        uint8_t* vec = ds.vectorPtr(ids[processed]);
        finish(processed, vec, Kernel::single(queryPtr, vec, binaryCodeBytes));
    }
}

template <class Kernel>
void runBenchmark(benchmark::State& state) {
    const int32_t dim = static_cast<int32_t>(state.range(0));
    const int32_t vecs = static_cast<int32_t>(state.range(1));
    // 64k-vector pool (> L2 for d >= 256) so the memory behaviour resembles a real segment.
    const int32_t pool = 65536;
    const Dataset ds = makeDataset(dim, pool, 42u);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int32_t> idDist(0, pool - 1);
    std::vector<int32_t> ids(vecs);
    std::vector<float> scores(vecs);
    for (auto _ : state) {
        for (auto& id : ids) id = idDist(rng);
        sqBulkIp<Kernel>(ds, ids.data(), scores.data(), vecs);
        benchmark::DoNotOptimize(scores.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * vecs);
    state.counters["dim"] = dim;
    state.counters["vecs"] = vecs;
}

void BM_Old(benchmark::State& state) { runBenchmark<OldKernel>(state); }
void BM_New(benchmark::State& state) { runBenchmark<NewKernel<lut_popcnt_epi8>>(state); }
#if defined(__AVX512BITALG__)
void BM_NewBitalg(benchmark::State& state) { runBenchmark<NewKernel<bitalg_popcnt_epi8>>(state); }
#endif

void Args(benchmark::internal::Benchmark* b) {
    for (const int64_t dim : {128, 256, 384, 512, 768, 1024, 1536, 3072}) {
        for (const int64_t vecs : {64, 256}) {
            b->Args({dim, vecs});
        }
    }
    b->ArgNames({"dim", "vecs"});
}

}  // namespace

BENCHMARK(BM_Old)->Apply(Args);
BENCHMARK(BM_New)->Apply(Args);
#if defined(__AVX512BITALG__)
BENCHMARK(BM_NewBitalg)->Apply(Args);
#endif

BENCHMARK_MAIN();
