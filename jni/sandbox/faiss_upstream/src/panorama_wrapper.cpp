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

/*
 * Native implementation for the isolated faiss_upstream engine (libopensearchknn_faiss_upstream).
 * Deliberately minimal and self-contained: it only knows how to build, write, load, query (top-k) and free
 * an IndexHNSWFlatPanorama — using nothing beyond unmodified upstream faiss APIs (index_factory,
 * IndexIDMap, read_index/write_index, SearchParametersHNSW) from the tenant's own vendored faiss
 * (v1.14.3). Like the built-in HNSW, the index needs no training and builds incrementally, so the JNI
 * lifecycle is a straight passthrough. It shares no translation units with the main faiss JNI library
 * other than the generic JNI marshalling helpers (jni_util/commons), which are faiss-free.
 */

#include "panorama_wrapper.h"

#include "jni_util.h"
#include "commons.h"
#include "faiss_stream_support.h"

#include "faiss/Index.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"

#include <omp.h>

#include <algorithm>
#include <jni.h>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

faiss::MetricType translateSpaceToMetric(const std::string& spaceType) {
    if (spaceType == knn_jni::L2) {
        return faiss::METRIC_L2;
    }
    if (spaceType == knn_jni::INNER_PRODUCT) {
        return faiss::METRIC_INNER_PRODUCT;
    }
    // Vectors are normalized at the Java layer for cosine, so cosine is equivalent to inner product.
    if (spaceType == knn_jni::COSINESIMIL) {
        return faiss::METRIC_INNER_PRODUCT;
    }
    throw std::runtime_error("Invalid spaceType: " + spaceType);
}

// Extracts the IndexHNSWFlatPanorama out of the IndexIDMap wrapper, throwing if the index is anything
// else. This library only ever serves .faissup files containing HNSW-Flat-Panorama indices.
faiss::IndexHNSWFlatPanorama* extractPanorama(faiss::IndexIDMap* idMap) {
    if (idMap == nullptr) {
        throw std::runtime_error("Invalid pointer to index");
    }
    auto panorama = dynamic_cast<faiss::IndexHNSWFlatPanorama*>(idMap->index);
    if (panorama == nullptr) {
        throw std::runtime_error("Index is not an HNSWFlatPanorama index");
    }
    return panorama;
}

// Builds the KNNQueryResult[] from raw search output; results shorter than k are trimmed at the -1 sentinel.
jobjectArray buildQueryResults(knn_jni::JNIUtilInterface* jniUtil, JNIEnv* env,
                               const std::vector<faiss::idx_t>& ids, const std::vector<float>& dis, int k) {
    int resultSize = k;
    auto it = std::find(ids.begin(), ids.end(), -1);
    if (it != ids.end()) {
        resultSize = it - ids.begin();
    }

    jclass resultClass = jniUtil->FindClass(env, "org/opensearch/knn/index/query/KNNQueryResult");
    jmethodID allArgs = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");

    jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);
    for (int i = 0; i < resultSize; ++i) {
        jobject result = jniUtil->NewObject(env, resultClass, allArgs, ids[i], dis[i]);
        jniUtil->SetObjectArrayElement(env, results, i, result);
        env->DeleteLocalRef(result);
    }
    return results;
}

}  // namespace

jlong knn_jni::panorama_wrapper::InitIndex(knn_jni::JNIUtilInterface* jniUtil, JNIEnv* env, jlong numDocs, jint dimJ,
                                           jobject parametersJ) {
    if (dimJ <= 0) {
        throw std::runtime_error("Vectors dimensions cannot be less than or equal to 0");
    }
    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);

    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(jniUtil->ConvertJavaObjectToCppString(env, spaceTypeJ));
    faiss::MetricType metric = translateSpaceToMetric(spaceTypeCpp);
    jniUtil->DeleteLocalRef(env, spaceTypeJ);

    // The description (e.g. "HNSW16,FlatPanorama8") comes from the sandbox method mapping (n_levels is
    // already clamped to the dimension on the Java side) and is parsed by the unmodified upstream factory.
    jobject indexDescriptionJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::INDEX_DESCRIPTION);
    std::string indexDescriptionCpp(jniUtil->ConvertJavaObjectToCppString(env, indexDescriptionJ));
    jniUtil->DeleteLocalRef(env, indexDescriptionJ);

    int threadCount = 0;
    if (parametersCpp.find(knn_jni::INDEX_THREAD_QUANTITY) != parametersCpp.end()) {
        threadCount = jniUtil->ConvertJavaObjectToCppInteger(env, parametersCpp[knn_jni::INDEX_THREAD_QUANTITY]);
    }
    if (threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    std::unique_ptr<faiss::Index> index(faiss::index_factory(static_cast<int>(dimJ), indexDescriptionCpp.c_str(), metric));

    auto panorama = dynamic_cast<faiss::IndexHNSWFlatPanorama*>(index.get());
    if (panorama == nullptr) {
        throw std::runtime_error("Index description \"" + indexDescriptionCpp + "\" is not an HNSWFlatPanorama index");
    }

    // ef_construction is a plain HNSW field, not part of the factory description.
    if (parametersCpp.find(knn_jni::PARAMETERS) != parametersCpp.end()) {
        auto subParametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersCpp[knn_jni::PARAMETERS]);
        auto value = subParametersCpp.find(knn_jni::EF_CONSTRUCTION);
        if (value != subParametersCpp.end()) {
            panorama->hnsw.efConstruction = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }
    }

    // HNSW is is_trained from construction; no training step exists on this path.
    auto idMap = std::make_unique<faiss::IndexIDMap>(index.get());
    // The IDMap must free the inner index when it is itself freed.
    idMap->own_fields = true;
    index.release();

    return reinterpret_cast<jlong>(idMap.release());
}

void knn_jni::panorama_wrapper::InsertToIndex(knn_jni::JNIUtilInterface* jniUtil, JNIEnv* env, jintArray idsJ,
                                              jlong vectorsAddressJ, jint dimJ, jlong indexAddressJ, jint threadCount) {
    if (idsJ == nullptr) {
        throw std::runtime_error("IDs cannot be null");
    }
    if (vectorsAddressJ <= 0) {
        throw std::runtime_error("VectorsAddress cannot be less than 0");
    }
    if (dimJ <= 0) {
        throw std::runtime_error("Vectors dimensions cannot be less than or equal to 0");
    }

    auto* inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddressJ);
    int dim = static_cast<int>(dimJ);
    int numVectors = static_cast<int>(inputVectors->size() / static_cast<uint64_t>(dim));
    if (numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }
    auto ids = jniUtil->ConvertJavaIntArrayToCppIntVector(env, idsJ);

    if (threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    auto* idMap = reinterpret_cast<faiss::IndexIDMap*>(indexAddressJ);
    extractPanorama(idMap);  // validate the pointer really is our index type
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());
}

void knn_jni::panorama_wrapper::WriteIndex(knn_jni::JNIUtilInterface* jniUtil, JNIEnv* env, jobject output,
                                           jlong indexAddressJ) {
    if (output == nullptr) {
        throw std::runtime_error("Index output stream cannot be null");
    }

    knn_jni::stream::NativeEngineIndexOutputMediator mediator{jniUtil, env, output};
    knn_jni::stream::FaissOpenSearchIOWriter writer{&mediator};

    // The index is freed after writing (the build strategy creates it solely to write it).
    std::unique_ptr<faiss::IndexIDMap> idMap(reinterpret_cast<faiss::IndexIDMap*>(indexAddressJ));
    try {
        faiss::write_index(idMap.get(), &writer);
        writer.flush();
    } catch (std::exception& e) {
        throw std::runtime_error(std::string("Failed to write index to disk, error=") + e.what());
    }
}

jlong knn_jni::panorama_wrapper::LoadIndexWithStream(faiss::IOReader* ioReader) {
    if (ioReader == nullptr) {
        throw std::runtime_error("IOReader cannot be null");
    }

    std::unique_ptr<faiss::Index> indexReader(faiss::read_index(ioReader, faiss::IO_FLAG_READ_ONLY));

    // .faissup files only ever contain IDMap(IndexHNSWFlatPanorama); refuse anything else outright.
    auto idMap = dynamic_cast<faiss::IndexIDMap*>(indexReader.get());
    if (idMap == nullptr || dynamic_cast<faiss::IndexHNSWFlatPanorama*>(idMap->index) == nullptr) {
        throw std::runtime_error("Loaded index is not an HNSWFlatPanorama index");
    }

    return reinterpret_cast<jlong>(indexReader.release());
}

jobjectArray knn_jni::panorama_wrapper::QueryIndex(knn_jni::JNIUtilInterface* jniUtil, JNIEnv* env, jlong indexPointerJ,
                                                   jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ) {
    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }

    auto* idMap = reinterpret_cast<faiss::IndexIDMap*>(indexPointerJ);
    auto* panorama = extractPanorama(idMap);

    std::unordered_map<std::string, jobject> methodParams;
    if (methodParamsJ != nullptr) {
        methodParams = jniUtil->ConvertJavaMapToCppMap(env, methodParamsJ);
    }

    faiss::SearchParametersHNSW hnswParams;
    // Query-time ef_search supersedes the index-level default.
    hnswParams.efSearch = knn_jni::commons::getIntegerMethodParameter(
        env, jniUtil, methodParams, knn_jni::EF_SEARCH, panorama->hnsw.efSearch);

    int k = static_cast<int>(std::min<int64_t>(static_cast<int64_t>(kJ), panorama->ntotal));
    if (k <= 0) {
        std::vector<faiss::idx_t> emptyIds;
        std::vector<float> emptyDis;
        return buildQueryResults(jniUtil, env, emptyIds, emptyDis, 0);
    }
    std::vector<float> dis(k, std::numeric_limits<float>::infinity());
    std::vector<faiss::idx_t> ids(k, -1);
    float* rawQueryVector = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);

    // Set omp threads to 1 so no new OMP threads are created under the search threadpool.
    omp_set_num_threads(1);

    try {
        idMap->search(1, rawQueryVector, k, dis.data(), ids.data(), &hnswParams);
    } catch (...) {
        jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
        throw;
    }
    jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);

    return buildQueryResults(jniUtil, env, ids, dis, k);
}

void knn_jni::panorama_wrapper::Free(jlong indexPointerJ) {
    auto* index = reinterpret_cast<faiss::Index*>(indexPointerJ);
    delete index;
}

void knn_jni::panorama_wrapper::InitLibrary() {
    // No global initialization required today; kept as the single hook for any future setup.
}
