/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_JNI_SVS_CONSTANTS_H
#define OPENSEARCH_KNN_JNI_SVS_CONSTANTS_H

#include <cstdint>
#include <string>

namespace knn_jni {
    // Intel SVS Vamana parameter names. These live ONLY in the isolated SVS JNI library
    // (libopensearchknn_svs); the main faiss JNI library carries no SVS symbols. They were
    // intentionally removed from jni_util when SVS was pulled out of the shared layer.
    extern const std::string CONSTRUCTION_WINDOW_SIZE;
    extern const std::string ALPHA;
    extern const std::string SEARCH_WINDOW_SIZE;
    extern const std::string SEARCH_BUFFER_CAPACITY;

    // Deferred-training LeanVec: the encoder sub-map key and its threshold parameters, plus the
    // fork-proven defaults applied when the parameters are absent or 0.
    extern const std::string ENCODER;
    extern const std::string LEANVEC_TRAINING_THRESHOLD;
    extern const std::string LEANVEC_ROUGH_TRAINING_THRESHOLD;
    constexpr int64_t LEANVEC_DEFAULT_TRAINING_THRESHOLD = 100000;
    constexpr int64_t LEANVEC_DEFAULT_ROUGH_TRAINING_THRESHOLD = 10000;
}

#endif //OPENSEARCH_KNN_JNI_SVS_CONSTANTS_H
