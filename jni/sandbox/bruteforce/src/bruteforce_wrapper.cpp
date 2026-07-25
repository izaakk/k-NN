/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bruteforce_wrapper.h"

#include "commons.h"
#include "native_engines_stream_support.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int32_t SERIALIZATION_MAGIC = 0x42465631;  // "BFV1"
constexpr const char *SPACE_TYPE_PARAM = "space_type";
constexpr const char *SCAN_LIMIT_PARAM = "scan_limit";
constexpr const char *L2_SPACE = "l2";

// The whole engine state: doc ids and their vectors, row-major, plus the space.
struct BruteForceIndex {
    std::string spaceType = L2_SPACE;
    int32_t dim = 0;
    std::vector<int32_t> ids;
    std::vector<float> vectors;  // ids.size() * dim floats
};

BruteForceIndex *existing(jlong indexPointer) {
    auto *index = reinterpret_cast<BruteForceIndex *>(indexPointer);
    if (index == nullptr) {
        throw std::runtime_error("bruteforce: null index pointer");
    }
    return index;
}

float squaredL2Distance(const float *a, const float *b, int32_t dim) {
    float sum = 0.0f;
    for (int32_t d = 0; d < dim; d++) {
        const float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

// Builds the Java KNNQueryResult[] from the first resultSize (distance, doc id) pairs.
jobjectArray toJavaResults(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env,
                           const std::vector<std::pair<float, int32_t>> &closest, size_t resultSize) {
    jclass resultClass = jniUtil->FindClass(env, "org/opensearch/knn/index/query/KNNQueryResult");
    jmethodID allArgs = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");
    jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);
    for (size_t i = 0; i < resultSize; i++) {
        jobject result = jniUtil->NewObject(env, resultClass, allArgs, closest[i].second, closest[i].first);
        jniUtil->SetObjectArrayElement(env, results, i, result);
    }
    return results;
}

void writeString(knn_jni::stream::NativeEngineIndexOutputMediator &mediator, const std::string &value) {
    const int32_t length = static_cast<int32_t>(value.size());
    mediator.writeBytes(reinterpret_cast<const uint8_t *>(&length), sizeof(length));
    mediator.writeBytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

std::string readString(knn_jni::stream::NativeEngineIndexInputMediator &mediator) {
    int32_t length = 0;
    mediator.copyBytes(sizeof(length), reinterpret_cast<uint8_t *>(&length));
    if (length < 0 || length > 64) {
        throw std::runtime_error("bruteforce: corrupt string length in serialized index");
    }
    std::string value(length, '\0');
    mediator.copyBytes(length, reinterpret_cast<uint8_t *>(value.data()));
    return value;
}

}  // namespace

jlong knn_jni::bruteforce_wrapper::InitIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong numDocs, jint dim,
                                             jobject parametersJ) {
    auto index = std::make_unique<BruteForceIndex>();
    if (parametersJ != nullptr) {
        auto parameters = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);
        auto space = parameters.find(SPACE_TYPE_PARAM);
        if (space != parameters.end()) {
            index->spaceType = jniUtil->ConvertJavaObjectToCppString(env, space->second);
        }
    }
    if (index->spaceType != L2_SPACE) {
        throw std::runtime_error("bruteforce: only the l2 space is supported");
    }
    index->dim = dim;
    index->ids.reserve(numDocs);
    index->vectors.reserve(static_cast<size_t>(numDocs) * dim);
    return reinterpret_cast<jlong>(index.release());
}

void knn_jni::bruteforce_wrapper::InsertToIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jintArray idsJ,
                                                jlong vectorsAddressJ, jint dim, jlong indexAddress) {
    BruteForceIndex *index = existing(indexAddress);
    if (dim != index->dim) {
        throw std::runtime_error("bruteforce: dimension mismatch on insert");
    }
    // The staged vectors arrive as a std::vector<float>, the same way every native tenant gets them.
    auto *staged = reinterpret_cast<std::vector<float> *>(vectorsAddressJ);
    auto ids = jniUtil->ConvertJavaIntArrayToCppIntVector(env, idsJ);
    if (staged->size() < ids.size() * static_cast<size_t>(dim)) {
        throw std::runtime_error("bruteforce: staged vector data shorter than ids * dim");
    }
    for (const auto id : ids) {
        index->ids.push_back(static_cast<int32_t>(id));
    }
    index->vectors.insert(index->vectors.end(), staged->begin(), staged->begin() + ids.size() * dim);
}

void knn_jni::bruteforce_wrapper::WriteIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong indexAddress,
                                             jobject outputJ) {
    BruteForceIndex *index = existing(indexAddress);
    knn_jni::stream::NativeEngineIndexOutputMediator mediator{jniUtil, env, outputJ};
    const int32_t count = static_cast<int32_t>(index->ids.size());
    mediator.writeBytes(reinterpret_cast<const uint8_t *>(&SERIALIZATION_MAGIC), sizeof(SERIALIZATION_MAGIC));
    writeString(mediator, index->spaceType);
    const int32_t header[2] = { index->dim, count };
    mediator.writeBytes(reinterpret_cast<const uint8_t *>(header), sizeof(header));
    mediator.writeBytes(reinterpret_cast<const uint8_t *>(index->ids.data()), count * sizeof(int32_t));
    mediator.writeBytes(reinterpret_cast<const uint8_t *>(index->vectors.data()), index->vectors.size() * sizeof(float));
    mediator.flush();
}

jlong knn_jni::bruteforce_wrapper::LoadIndexWithStream(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jobject readStreamJ) {
    knn_jni::stream::NativeEngineIndexInputMediator mediator{jniUtil, env, readStreamJ};
    int32_t magic = 0;
    mediator.copyBytes(sizeof(magic), reinterpret_cast<uint8_t *>(&magic));
    if (magic != SERIALIZATION_MAGIC) {
        throw std::runtime_error("bruteforce: not a serialized bruteforce index (bad magic)");
    }
    auto index = std::make_unique<BruteForceIndex>();
    index->spaceType = readString(mediator);
    if (index->spaceType != L2_SPACE) {
        throw std::runtime_error("bruteforce: unsupported space in serialized index: " + index->spaceType);
    }
    int32_t header[2];
    mediator.copyBytes(sizeof(header), reinterpret_cast<uint8_t *>(header));
    index->dim = header[0];
    const int32_t count = header[1];
    index->ids.resize(count);
    index->vectors.resize(static_cast<size_t>(count) * index->dim);
    mediator.copyBytes(count * sizeof(int32_t), reinterpret_cast<uint8_t *>(index->ids.data()));
    mediator.copyBytes(index->vectors.size() * sizeof(float), reinterpret_cast<uint8_t *>(index->vectors.data()));
    return reinterpret_cast<jlong>(index.release());
}

jobjectArray knn_jni::bruteforce_wrapper::QueryIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong indexPointer,
                                                     jfloatArray queryVectorJ, jint k, jobject methodParamsJ) {
    BruteForceIndex *index = existing(indexPointer);
    if (jniUtil->GetJavaFloatArrayLength(env, queryVectorJ) != index->dim) {
        throw std::runtime_error("bruteforce: query dimension mismatch");
    }

    // scan_limit caps how many vectors are scanned. Absent means scan everything.
    std::unordered_map<std::string, jobject> methodParams;
    if (methodParamsJ != nullptr) {
        methodParams = jniUtil->ConvertJavaMapToCppMap(env, methodParamsJ);
    }
    const size_t scanRows = std::min(
        index->ids.size(),
        static_cast<size_t>(knn_jni::commons::getIntegerMethodParameter(
            env, jniUtil, methodParams, SCAN_LIMIT_PARAM, static_cast<int>(index->ids.size()))));

    jfloat *query = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);
    std::vector<std::pair<float, int32_t>> closest;
    closest.reserve(scanRows);
    for (size_t row = 0; row < scanRows; row++) {
        const float *rowVector = index->vectors.data() + row * index->dim;
        closest.emplace_back(squaredL2Distance(query, rowVector, index->dim), index->ids[row]);
    }
    jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, query, JNI_ABORT);

    // Keep the k smallest raw distances. Core turns them into scores through the engine's library.
    const size_t resultSize = std::min(static_cast<size_t>(k), closest.size());
    std::partial_sort(closest.begin(), closest.begin() + resultSize, closest.end());
    return toJavaResults(jniUtil, env, closest, resultSize);
}

void knn_jni::bruteforce_wrapper::Free(jlong indexPointer) {
    delete reinterpret_cast<BruteForceIndex *>(indexPointer);
}
