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

// Bit-exact tests for the SQ (4-bit query x 1-bit data) SIMD similarity functions
// (SQ_IP / SQ_L2 / SQ_COSINE) exposed through SimilarityFunction::saveSearchContext.
//
// The reference is a plain scalar byte loop. Every code-byte length (so every masked-tail
// length), every bulk batch shape (8 / 4 / remainder) and the single-vector path are compared
// against it, for the build's SIMD level (avx2 / avx512 / avx512_spr / neon / generic).

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
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

// Scalar reference of the complete score, mirroring AVX512SQSimilarityFunction / default SQ function.
float referenceScore(const SqDataset& ds, const int32_t id, const NativeSimilarityFunctionType type) {
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
    float score = ax * ay * ds.dim + ay * lx * x1 + ax * ly * y1 + lx * ly * qc;
    if (type == NativeSimilarityFunctionType::SQ_IP) {
        score += ds.q.additional + additional - ds.q.centroidDp;
        return score < 0 ? 1 / (1 - score) : (1 + score);
    }
    if (type == NativeSimilarityFunctionType::SQ_COSINE) {
        score += ds.q.additional + additional - ds.q.centroidDp;
        const float raw = (1.0f + score) * 0.5f;
        return raw < 0.0f ? 0.0f : (raw > 1.0f ? 1.0f : raw);
    }
    score = std::max(0.0F, ds.q.additional + additional - 2 * score);
    return 1.0f / (1.0f + score);
}

// neutral == true: corrections chosen so that score(IP) == 1 + dotProduct exactly
// (ay = 0, ly = 1, ax = 0, lx = 1, no additive terms), which makes the SIMD result comparable
// bit-for-bit with the integer reference. neutral == false: realistic random corrections.
SqDataset makeDataset(const int32_t dim, const int32_t numVectors, const bool neutral, const uint32_t seed) {
    SqDataset ds;
    ds.dim = dim;
    ds.codeBytes = codeBytesFor(dim);
    ds.oneVectorBytes = ds.codeBytes + CORRECTION_BYTES;
    ds.numVectors = numVectors;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> byteDist(0, 255);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> sumDist(0, 15 * dim);

    ds.query.resize(4 * ds.codeBytes);
    for (auto& b : ds.query) {
        b = static_cast<uint8_t>(byteDist(rng));
    }
    // Bits above `dim` in the last byte of each plane must be zero, as the Java quantizer writes them.
    const int32_t validBitsInLastByte = dim - 8 * (ds.codeBytes - 1);
    const uint8_t lastByteMask = static_cast<uint8_t>((1u << validBitsInLastByte) - 1u);
    for (int32_t plane = 0; plane < 4; ++plane) {
        ds.query[plane * ds.codeBytes + ds.codeBytes - 1] &= lastByteMask;
    }

    ds.vectors.resize(static_cast<size_t>(numVectors) * ds.oneVectorBytes);
    for (int32_t v = 0; v < numVectors; ++v) {
        uint8_t* vec = ds.vectors.data() + static_cast<size_t>(v) * ds.oneVectorBytes;
        for (int32_t i = 0; i < ds.codeBytes; ++i) {
            vec[i] = static_cast<uint8_t>(byteDist(rng));
        }
        vec[ds.codeBytes - 1] &= lastByteMask;
        if (neutral) {
            writeCorrections(vec + ds.codeBytes, 0.0f, 1.0f, 0.0f, sumDist(rng));
        } else {
            const float lower = -std::abs(unit(rng));
            const float upper = lower + 0.5f + std::abs(unit(rng));
            writeCorrections(vec + ds.codeBytes, lower, upper, unit(rng), sumDist(rng));
        }
    }

    if (neutral) {
        ds.q = QueryCorrections {0.0f, 15.0f, 0.0f, sumDist(rng), 0.0f};
    } else {
        const float lower = -std::abs(unit(rng));
        ds.q = QueryCorrections {lower, lower + 0.5f + std::abs(unit(rng)), unit(rng), sumDist(rng), unit(rng)};
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

const std::vector<int32_t> TEST_DIMS {
    1, 7, 8, 9, 16, 63, 64, 65, 100, 127, 128, 129, 200, 256, 384, 500, 511, 512, 513, 768, 1000, 1024, 1536, 2048, 3072};
const std::vector<int32_t> TEST_NUM_VECTORS {1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 33, 100};
const std::vector<NativeSimilarityFunctionType> SQ_TYPES {
    NativeSimilarityFunctionType::SQ_IP, NativeSimilarityFunctionType::SQ_L2, NativeSimilarityFunctionType::SQ_COSINE};

}  // namespace

// The integer dot product must be exact for every tail length (codeBytes % 64) and every batch shape.
TEST(SimdSqSimilarityFunctionTest, BulkDotProductIsExactForEveryTailLengthAndBatchShape) {
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

// Bulk and single-vector scoring must agree with each other and with the scalar reference for all
// three metric modes with realistic correction factors, including a shuffled id order.
TEST(SimdSqSimilarityFunctionTest, BulkAndSingleMatchReferenceForAllMetricModes) {
    uint32_t seed = 1000;
    for (const NativeSimilarityFunctionType type : SQ_TYPES) {
        for (const int32_t dim : {1, 64, 128, 129, 384, 768, 1024, 3072}) {
            for (const int32_t numVectors : {1, 4, 9, 17, 100}) {
                SqDataset ds = makeDataset(dim, numVectors, /*neutral=*/false, seed++);
                SimdVectorSearchContext* ctx = setupSingleRegionContext(ds, type);
                std::vector<int32_t> ids = identityIds(numVectors);
                std::shuffle(ids.begin(), ids.end(), std::mt19937(seed));
                const std::vector<float> scores = bulkScores(ctx, ids);
                for (int32_t i = 0; i < numVectors; ++i) {
                    const float expected = referenceScore(ds, ids[i], type);
                    const float tolerance = 1e-5f * std::abs(expected) + 1e-6f;
                    ASSERT_NEAR(expected, scores[i], tolerance)
                        << "bulk type=" << static_cast<int32_t>(type) << " dim=" << dim << " id=" << ids[i];
                    ASSERT_FLOAT_EQ(scores[i], ctx->similarityFunction->calculateSimilarity(ctx, ids[i]))
                        << "single type=" << static_cast<int32_t>(type) << " dim=" << dim << " id=" << ids[i];
                }
            }
        }
    }
}

// A vector that straddles two mmap regions is copied into tmpBuffer by getVectorPointer; the scores
// must be identical to the single-region case.
TEST(SimdSqSimilarityFunctionTest, VectorsStraddlingTwoMemoryRegionsScoreIdentically) {
    for (const int32_t dim : {128, 768, 1000}) {
        const int32_t numVectors = 37;
        SqDataset ds = makeDataset(dim, numVectors, /*neutral=*/false, 7u + dim);
        SimdVectorSearchContext* ctx = setupSingleRegionContext(ds, NativeSimilarityFunctionType::SQ_IP);
        const std::vector<float> expected = bulkScores(ctx, identityIds(numVectors));

        // Split in the middle of vector 5 so that it, and only it, straddles the boundary.
        const int64_t split = 5 * ds.oneVectorBytes + ds.codeBytes / 2 + 1;
        std::vector<int64_t> addressAndSize {
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(ds.vectors.data())), split,
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(ds.vectors.data() + split)), static_cast<int64_t>(ds.vectors.size()) - split};
        ctx = setupContext(ds, NativeSimilarityFunctionType::SQ_IP, addressAndSize);
        const std::vector<float> actual = bulkScores(ctx, identityIds(numVectors));
        for (int32_t id = 0; id < numVectors; ++id) {
            ASSERT_EQ(expected[id], actual[id]) << "dim=" << dim << " id=" << id;
            ASSERT_EQ(expected[id], ctx->similarityFunction->calculateSimilarity(ctx, id)) << "single dim=" << dim << " id=" << id;
        }
    }
}

// The masked tail must never touch memory past the last vector: place the vectors so that the
// region ends exactly at a PROT_NONE guard page. An over-read would SIGSEGV here.
TEST(SimdSqSimilarityFunctionTest, MaskedTailDoesNotReadPastTheMappedRegion) {
    const long pageSize = sysconf(_SC_PAGESIZE);
    for (const int32_t dim : {8, 128, 384, 768, 1000}) {
        SqDataset ds = makeDataset(dim, /*numVectors=*/1, /*neutral=*/false, 99u + dim);
        const int32_t numVectors = static_cast<int32_t>(pageSize / ds.oneVectorBytes);
        ds = makeDataset(dim, numVectors, /*neutral=*/false, 99u + dim);
        ASSERT_LE(static_cast<long>(ds.vectors.size()), pageSize);

        void* mapping = mmap(nullptr, 2 * pageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ASSERT_NE(mapping, MAP_FAILED);
        ASSERT_EQ(0, mprotect(static_cast<uint8_t*>(mapping) + pageSize, pageSize, PROT_NONE));
        uint8_t* region = static_cast<uint8_t*>(mapping) + pageSize - ds.vectors.size();
        std::memcpy(region, ds.vectors.data(), ds.vectors.size());

        std::vector<int64_t> addressAndSize {
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(region)), static_cast<int64_t>(ds.vectors.size())};
        SimdVectorSearchContext* ctx = setupContext(ds, NativeSimilarityFunctionType::SQ_L2, addressAndSize);
        const std::vector<float> scores = bulkScores(ctx, identityIds(numVectors));
        for (int32_t id = 0; id < numVectors; ++id) {
            const float expected = referenceScore(ds, id, NativeSimilarityFunctionType::SQ_L2);
            ASSERT_NEAR(expected, scores[id], 1e-5f * std::abs(expected) + 1e-6f) << "dim=" << dim << " id=" << id;
        }
        ASSERT_FLOAT_EQ(scores[numVectors - 1], ctx->similarityFunction->calculateSimilarity(ctx, numVectors - 1));
        munmap(mapping, 2 * pageSize);
    }
}
