/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

// Tests for the SQ (4-bit query x 1-bit data) similarity functions (SQ_IP / SQ_L2 / SQ_COSINE) through
// SimilarityFunction::saveSearchContext, against a scalar reference, at the build's SIMD level.

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "simd/similarity_function/similarity_function.h"

using knn_jni::simd::similarity_function::NativeSimilarityFunctionType;
using knn_jni::simd::similarity_function::SimdVectorSearchContext;
using knn_jni::simd::similarity_function::SimilarityFunction;

namespace {

constexpr int32_t CORRECTION_BYTES = 3 * sizeof(float) + sizeof(int32_t);
constexpr float FOUR_BIT_SCALE = 1.0f / 15.0f;

// The exact comparisons below rely on the (0, 15) query interval giving ly == 1.0f exactly.
static_assert(15.0f * FOUR_BIT_SCALE == 1.0f, "ly = (15 - 0) * FOUR_BIT_SCALE must be exactly 1.0f");

// The SIMD level is fixed at build time; skip instead of SIGILL on a host that cannot execute it.
bool hostSupportsBuildIsa() {
#if defined(__x86_64__) || defined(__i386__)
    #if defined(KNN_HAVE_AVX512_SPR)
    return __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512bitalg");
    #elif defined(KNN_HAVE_AVX512)
    return __builtin_cpu_supports("avx512bw");
    #elif defined(KNN_HAVE_AVX2_F16C)
    return __builtin_cpu_supports("avx2");
    #else
    return true;
    #endif
#else
    return true;
#endif
}

#define SKIP_IF_HOST_LACKS_BUILD_ISA()                                                                 \
    if (!hostSupportsBuildIsa()) {                                                                     \
        GTEST_SKIP() << "host CPU lacks the SIMD level this library was built for";                    \
    }

struct QueryCorrections {
    float lower = 0;
    float upper = 15;
    float additional = 0;
    int32_t componentSum = 0;
    float centroidDp = 0;
};

struct SqDataset {
    int32_t dim = 0;
    int32_t codeBytes = 0;
    int32_t oneVectorBytes = 0;
    int32_t numVectors = 0;
    std::vector<uint8_t> query;    // 4 bit planes, codeBytes each
    std::vector<uint8_t> vectors;  // numVectors x (codeBytes + CORRECTION_BYTES)
    QueryCorrections q;
};

int32_t codeBytesFor(const int32_t dim) {
    return (dim + 7) / 8;
}

// Lucene packs MSB first, so the padding bits of a partial last byte are its low bits (zero).
uint8_t lastByteMaskFor(const int32_t dim) {
    const int32_t validBits = dim - 8 * (codeBytesFor(dim) - 1);  // 1..8
    return static_cast<uint8_t>(0xFFu << (8 - validBits));
}

void writeCorrections(uint8_t* dst, const float lower, const float upper, const float additional, const int32_t componentSum) {
    std::memcpy(dst, &lower, sizeof(float));
    std::memcpy(dst + 4, &upper, sizeof(float));
    std::memcpy(dst + 8, &additional, sizeof(float));
    std::memcpy(dst + 12, &componentSum, sizeof(int32_t));
}

void readCorrections(const uint8_t* src, float& lower, float& upper, float& additional, int32_t& componentSum) {
    std::memcpy(&lower, src, sizeof(float));
    std::memcpy(&upper, src + 4, sizeof(float));
    std::memcpy(&additional, src + 8, sizeof(float));
    std::memcpy(&componentSum, src + 12, sizeof(int32_t));
}

// Scalar reference: popcount(plane_k AND code) * 2^k summed over the four query planes.
int64_t referenceDotProduct(const uint8_t* query, const uint8_t* code, const int32_t codeBytes) {
    int64_t result = 0;
    for (int32_t plane = 0; plane < 4; ++plane) {
        for (int32_t i = 0; i < codeBytes; ++i) {
            result += static_cast<int64_t>(__builtin_popcount(query[plane * codeBytes + i] & code[i])) << plane;
        }
    }
    return result;
}

struct ReferenceScore {
    float score;
    float errorBound;  // sum of |terms| entering the score; the transforms all have |derivative| <= 1
};

// Scalar reference of the complete score. The kernel TU is compiled with FMA contraction and this one is
// not, so the comparison tolerance is 16 * FLT_EPSILON * errorBound rather than exact.
ReferenceScore referenceScore(const SqDataset& ds, const int32_t id, const NativeSimilarityFunctionType type) {
    const uint8_t* vec = ds.vectors.data() + static_cast<size_t>(id) * ds.oneVectorBytes;
    const float qc = static_cast<float>(referenceDotProduct(ds.query.data(), vec, ds.codeBytes));
    float lower, upper, additional;
    int32_t x1Raw;
    readCorrections(vec + ds.codeBytes, lower, upper, additional, x1Raw);
    const float ax = lower;
    const float lx = upper - lower;
    const float x1 = static_cast<float>(x1Raw);
    const float ay = ds.q.lower;
    const float ly = (ds.q.upper - ds.q.lower) * FOUR_BIT_SCALE;
    const float y1 = static_cast<float>(ds.q.componentSum);
    const float t0 = ax * ay * ds.dim, t1 = ay * lx * x1, t2 = ax * ly * y1, t3 = lx * ly * qc;
    float score = t0 + t1 + t2 + t3;
    const float dotTerms = std::abs(t0) + std::abs(t1) + std::abs(t2) + std::abs(t3);
    if (type == NativeSimilarityFunctionType::SQ_IP) {
        score += ds.q.additional + additional - ds.q.centroidDp;
        const float bound = dotTerms + std::abs(ds.q.additional) + std::abs(additional) + std::abs(ds.q.centroidDp);
        return {score < 0 ? 1 / (1 - score) : (1 + score), bound};
    }
    if (type == NativeSimilarityFunctionType::SQ_COSINE) {
        score += ds.q.additional + additional - ds.q.centroidDp;
        const float bound = dotTerms + std::abs(ds.q.additional) + std::abs(additional) + std::abs(ds.q.centroidDp);
        const float raw = (1.0f + score) * 0.5f;
        return {raw < 0.0f ? 0.0f : (raw > 1.0f ? 1.0f : raw), bound};
    }
    const float bound = 2 * dotTerms + std::abs(ds.q.additional) + std::abs(additional);
    score = std::max(0.0F, ds.q.additional + additional - 2 * score);
    return {1.0f / (1.0f + score), bound};
}

float toleranceFor(const ReferenceScore& ref) {
    return 16 * FLT_EPSILON * ref.errorBound + 1e-7f;
}

enum class Fill { RANDOM, ALL_ONES };

// neutral: corrections such that score(IP) == 1 + dotProduct exactly. Otherwise random corrections on
// the scale of a real OSQ segment (intervals ~1/dim), so the pre-transform score lands in [-1, 1] and
// SQ_COSINE exercises its arithmetic rather than its 0 / 1 clamp.
SqDataset makeDataset(const int32_t dim, const int32_t numVectors, const bool neutral, const uint32_t seed,
                      const Fill fill = Fill::RANDOM) {
    SqDataset ds;
    ds.dim = dim;
    ds.codeBytes = codeBytesFor(dim);
    ds.oneVectorBytes = ds.codeBytes + CORRECTION_BYTES;
    ds.numVectors = numVectors;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> byteDist(0, 255);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> docSumDist(0, dim);
    std::uniform_int_distribution<int32_t> querySumDist(0, 15 * dim);
    const float scale = 1.0f / static_cast<float>(std::max(dim, 16));
    const uint8_t lastByteMask = lastByteMaskFor(dim);
    auto fillByte = [&]() { return fill == Fill::ALL_ONES ? uint8_t {0xFF} : static_cast<uint8_t>(byteDist(rng)); };

    ds.query.resize(4 * ds.codeBytes);
    for (auto& b : ds.query) {
        b = fillByte();
    }
    for (int32_t plane = 0; plane < 4; ++plane) {
        ds.query[plane * ds.codeBytes + ds.codeBytes - 1] &= lastByteMask;
    }

    ds.vectors.resize(static_cast<size_t>(numVectors) * ds.oneVectorBytes);
    for (int32_t v = 0; v < numVectors; ++v) {
        uint8_t* vec = ds.vectors.data() + static_cast<size_t>(v) * ds.oneVectorBytes;
        for (int32_t i = 0; i < ds.codeBytes; ++i) {
            vec[i] = fillByte();
        }
        vec[ds.codeBytes - 1] &= lastByteMask;
        if (neutral) {
            writeCorrections(vec + ds.codeBytes, 0.0f, 1.0f, 0.0f, docSumDist(rng));
        } else {
            const float lower = -std::abs(unit(rng)) * scale;
            const float upper = lower + (0.5f + std::abs(unit(rng))) * scale;
            writeCorrections(vec + ds.codeBytes, lower, upper, 0.25f * unit(rng), docSumDist(rng));
        }
    }

    if (neutral) {
        ds.q = QueryCorrections {0.0f, 15.0f, 0.0f, querySumDist(rng), 0.0f};
    } else {
        const float lower = -std::abs(unit(rng)) * scale;
        ds.q = QueryCorrections {lower, lower + (0.5f + std::abs(unit(rng))) * scale, 0.25f * unit(rng), querySumDist(rng), 0.25f * unit(rng)};
    }
    return ds;
}

SimdVectorSearchContext* setupContext(SqDataset& ds, const NativeSimilarityFunctionType type, std::vector<int64_t>& addressAndSize) {
    SimdVectorSearchContext* ctx = SimilarityFunction::saveSearchContext(
        ds.query.data(),
        static_cast<int32_t>(ds.query.size()),
        ds.dim,
        addressAndSize.data(),
        static_cast<int32_t>(addressAndSize.size()),
        static_cast<int32_t>(type));
    // Same layout Java_org_opensearch_knn_jni_SimdVectorComputeService_saveSQSearchContext writes.
    ctx->tmpBuffer.resize(5 * sizeof(float));
    auto* correctionPtr = reinterpret_cast<float*>(ctx->tmpBuffer.data());
    correctionPtr[0] = ds.q.lower;
    correctionPtr[1] = ds.q.upper;
    correctionPtr[2] = ds.q.additional;
    std::memcpy(&correctionPtr[3], &ds.q.componentSum, sizeof(int32_t));
    correctionPtr[4] = ds.q.centroidDp;
    return ctx;
}

SimdVectorSearchContext* setupSingleRegionContext(SqDataset& ds, const NativeSimilarityFunctionType type) {
    std::vector<int64_t> addressAndSize {
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(ds.vectors.data())),
        static_cast<int64_t>(ds.vectors.size())};
    return setupContext(ds, type, addressAndSize);
}

std::vector<float> bulkScores(SimdVectorSearchContext* ctx, std::vector<int32_t> ids) {
    std::vector<float> scores(ids.size(), 0.0f);
    ctx->similarityFunction->calculateSimilarityInBulk(ctx, ids.data(), scores.data(), static_cast<int32_t>(ids.size()));
    return scores;
}

std::vector<int32_t> identityIds(const int32_t n) {
    std::vector<int32_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    return ids;
}

// Every code-byte length 1..130 (all tail lengths, exact multiples of 64) plus partial-last-byte and large dims.
std::vector<int32_t> allTestDims() {
    std::vector<int32_t> dims {1, 7, 9, 63, 65, 100, 127, 129, 500, 511, 513, 1000, 1536, 2048, 3072};
    for (int32_t dim = 8; dim <= 130 * 8; dim += 8) {
        dims.push_back(dim);
    }
    std::sort(dims.begin(), dims.end());
    return dims;
}

const std::vector<int32_t> TEST_DIMS = allTestDims();
const std::vector<int32_t> TEST_NUM_VECTORS {1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 33, 100};
const std::vector<NativeSimilarityFunctionType> SQ_TYPES {
    NativeSimilarityFunctionType::SQ_IP, NativeSimilarityFunctionType::SQ_L2, NativeSimilarityFunctionType::SQ_COSINE};

// Two anonymous pages, the second PROT_NONE.
struct GuardedPage {
    long pageSize = sysconf(_SC_PAGESIZE);
    uint8_t* base = nullptr;
    GuardedPage() {
        void* mapping = mmap(nullptr, 2 * pageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping != MAP_FAILED && mprotect(static_cast<uint8_t*>(mapping) + pageSize, pageSize, PROT_NONE) == 0) {
            base = static_cast<uint8_t*>(mapping);
        } else if (mapping != MAP_FAILED) {
            munmap(mapping, 2 * pageSize);
        }
    }
    ~GuardedPage() {
        if (base != nullptr) {
            munmap(base, 2 * pageSize);
        }
    }
    uint8_t* guardStart() const { return base + pageSize; }
};

}  // namespace

TEST(SimdSqSimilarityFunctionTest, BulkDotProductIsExactForEveryTailLengthAndBatchShape) {
    SKIP_IF_HOST_LACKS_BUILD_ISA();
    uint32_t seed = 1;
    for (const int32_t dim : TEST_DIMS) {
        for (const int32_t numVectors : TEST_NUM_VECTORS) {
            SqDataset ds = makeDataset(dim, numVectors, /*neutral=*/true, seed++);
            SimdVectorSearchContext* ctx = setupSingleRegionContext(ds, NativeSimilarityFunctionType::SQ_IP);
            const std::vector<float> scores = bulkScores(ctx, identityIds(numVectors));
            for (int32_t id = 0; id < numVectors; ++id) {
                const uint8_t* vec = ds.vectors.data() + static_cast<size_t>(id) * ds.oneVectorBytes;
                const float expected = 1.0f + static_cast<float>(referenceDotProduct(ds.query.data(), vec, ds.codeBytes));
                ASSERT_EQ(expected, scores[id]) << "dim=" << dim << " numVectors=" << numVectors << " id=" << id;
                ASSERT_EQ(expected, ctx->similarityFunction->calculateSimilarity(ctx, id))
                    << "single, dim=" << dim << " numVectors=" << numVectors << " id=" << id;
            }
        }
    }
}

// Every byte at the maximum weighted popcount (120), every u64 lane at 960.
TEST(SimdSqSimilarityFunctionTest, AllOnesCodesReachTheMaximumWeightedPopcount) {
    SKIP_IF_HOST_LACKS_BUILD_ISA();
    for (const int32_t dim : {8, 64, 100, 512, 1000, 3072}) {
        for (const int32_t numVectors : {4, 9}) {
            SqDataset ds = makeDataset(dim, numVectors, /*neutral=*/true, 5u, Fill::ALL_ONES);
            SimdVectorSearchContext* ctx = setupSingleRegionContext(ds, NativeSimilarityFunctionType::SQ_IP);
            const std::vector<float> scores = bulkScores(ctx, identityIds(numVectors));
            const float expected = 1.0f + 15.0f * static_cast<float>(dim);
            for (int32_t id = 0; id < numVectors; ++id) {
                ASSERT_EQ(expected, scores[id]) << "dim=" << dim << " numVectors=" << numVectors << " id=" << id;
                ASSERT_EQ(expected, ctx->similarityFunction->calculateSimilarity(ctx, id)) << "single dim=" << dim << " id=" << id;
            }
        }
    }
}

TEST(SimdSqSimilarityFunctionTest, BulkAndSingleMatchReferenceForAllMetricModes) {
    SKIP_IF_HOST_LACKS_BUILD_ISA();
    uint32_t seed = 1000;
    int32_t cosineTotal = 0, cosineInsideUnitInterval = 0;
    for (const NativeSimilarityFunctionType type : SQ_TYPES) {
        for (const int32_t dim : {1, 64, 128, 129, 384, 768, 1024, 3072}) {
            for (const int32_t numVectors : {1, 4, 9, 17, 100}) {
                SqDataset ds = makeDataset(dim, numVectors, /*neutral=*/false, seed++);
                SimdVectorSearchContext* ctx = setupSingleRegionContext(ds, type);
                std::vector<int32_t> ids = identityIds(numVectors);
                std::shuffle(ids.begin(), ids.end(), std::mt19937(seed));
                const std::vector<float> scores = bulkScores(ctx, ids);
                for (int32_t i = 0; i < numVectors; ++i) {
                    const ReferenceScore ref = referenceScore(ds, ids[i], type);
                    ASSERT_NEAR(ref.score, scores[i], toleranceFor(ref))
                        << "bulk type=" << static_cast<int32_t>(type) << " dim=" << dim << " id=" << ids[i];
                    ASSERT_FLOAT_EQ(scores[i], ctx->similarityFunction->calculateSimilarity(ctx, ids[i]))
                        << "single type=" << static_cast<int32_t>(type) << " dim=" << dim << " id=" << ids[i];
                    if (type == NativeSimilarityFunctionType::SQ_COSINE) {
                        ++cosineTotal;
                        cosineInsideUnitInterval += (ref.score > 0.0f && ref.score < 1.0f) ? 1 : 0;
                    }
                }
            }
        }
    }
    // Most cosine comparisons must exercise the arithmetic, not the 0 / 1 clamp.
    ASSERT_GE(cosineInsideUnitInterval, (cosineTotal * 9) / 10) << cosineInsideUnitInterval << " of " << cosineTotal;
}

// Three regions: vector 5 is split inside its code, vector 6 inside its correction bytes.
TEST(SimdSqSimilarityFunctionTest, VectorsStraddlingMemoryRegionsScoreIdentically) {
    SKIP_IF_HOST_LACKS_BUILD_ISA();
    for (const int32_t dim : {128, 768, 1000}) {
        const int32_t numVectors = 37;
        SqDataset ds = makeDataset(dim, numVectors, /*neutral=*/false, 7u + dim);
        SimdVectorSearchContext* ctx = setupSingleRegionContext(ds, NativeSimilarityFunctionType::SQ_IP);
        const std::vector<float> expected = bulkScores(ctx, identityIds(numVectors));

        const int64_t split1 = 5 * ds.oneVectorBytes + ds.codeBytes / 2 + 1;
        const int64_t split2 = 6 * ds.oneVectorBytes + ds.codeBytes + 6;
        const auto addr = [&](const int64_t offset) {
            return static_cast<int64_t>(reinterpret_cast<uintptr_t>(ds.vectors.data() + offset));
        };
        std::vector<int64_t> addressAndSize {
            addr(0), split1,
            addr(split1), split2 - split1,
            addr(split2), static_cast<int64_t>(ds.vectors.size()) - split2};
        ctx = setupContext(ds, NativeSimilarityFunctionType::SQ_IP, addressAndSize);
        const std::vector<float> actual = bulkScores(ctx, identityIds(numVectors));
        for (int32_t id = 0; id < numVectors; ++id) {
            ASSERT_EQ(expected[id], actual[id]) << "dim=" << dim << " id=" << id;
            ASSERT_EQ(expected[id], ctx->similarityFunction->calculateSimilarity(ctx, id)) << "single dim=" << dim << " id=" << id;
        }
    }
}

// The last byte of the last vector is the last byte before a PROT_NONE page; any over-read SIGSEGVs.
TEST(SimdSqSimilarityFunctionTest, MaskedTailDoesNotReadPastTheMappedRegion) {
    SKIP_IF_HOST_LACKS_BUILD_ISA();
    for (const int32_t dim : {8, 128, 384, 768, 1000}) {
        GuardedPage page;
        ASSERT_NE(page.base, nullptr);
        const int32_t oneVectorBytes = codeBytesFor(dim) + CORRECTION_BYTES;
        const int32_t numVectors = static_cast<int32_t>(page.pageSize / oneVectorBytes);
        SqDataset ds = makeDataset(dim, numVectors, /*neutral=*/false, 99u + dim);
        ASSERT_LE(static_cast<long>(ds.vectors.size()), page.pageSize);

        uint8_t* region = page.guardStart() - ds.vectors.size();
        std::memcpy(region, ds.vectors.data(), ds.vectors.size());

        std::vector<int64_t> addressAndSize {
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(region)), static_cast<int64_t>(ds.vectors.size())};
        SimdVectorSearchContext* ctx = setupContext(ds, NativeSimilarityFunctionType::SQ_L2, addressAndSize);
        const std::vector<float> scores = bulkScores(ctx, identityIds(numVectors));
        for (int32_t id = 0; id < numVectors; ++id) {
            const ReferenceScore ref = referenceScore(ds, id, NativeSimilarityFunctionType::SQ_L2);
            ASSERT_NEAR(ref.score, scores[id], toleranceFor(ref)) << "dim=" << dim << " id=" << id;
        }
        ASSERT_FLOAT_EQ(scores[numVectors - 1], ctx->similarityFunction->calculateSimilarity(ctx, numVectors - 1));
    }
}
