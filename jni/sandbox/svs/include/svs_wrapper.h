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

#ifndef OPENSEARCH_KNN_SVS_WRAPPER_H
#define OPENSEARCH_KNN_SVS_WRAPPER_H

#include "jni_util.h"

#include "faiss/impl/io.h"
#include "faiss/impl/AuxIndexStructures.h"

#include <jni.h>
#include <iostream>

namespace knn_jni {
    namespace svs_wrapper {
        // Create an empty SVS Vamana index from the parameters map (space type, index description, thread
        // count, and the svs_vamana build parameters) and return the address of its build context (which
        // wraps the IndexIDMap). For LeanVec descriptions the build context drives deferred training:
        // below the rough threshold the description is rewritten to the LVQ equivalent; otherwise the
        // context buffers min(training_threshold, numDocs) vectors, trains the projection, then streams.
        jlong InitIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong numDocs, jint dimJ, jobject parametersJ);

        // Add the vectors at vectorsAddressJ (with the given ids) to the build context created by InitIndex
        // (buffering them first when deferred LeanVec training is still pending).
        void InsertToIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ, jlong vectorsAddressJ,
                           jint dimJ, jlong indexAddressJ, jint threadCount);

        // Serialize the built index to the provided IndexOutputWithBuffer and free the build context
        // (training first if the deferred buffer never filled).
        void WriteIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jobject output, jlong indexAddressJ);

        // Load an SVS index from a faiss IOReader; returns the address of the loaded index.
        jlong LoadIndexWithStream(faiss::IOReader* ioReader);

        // Top-k search; methodParamsJ may carry search_window_size / search_buffer_capacity overrides.
        jobjectArray QueryIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ);

        // Top-k search restricted to filterIdsJ (bitmap or batch encoding, per filterIdsTypeJ).
        jobjectArray QueryIndex_WithFilter(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                           jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ,
                                           jlongArray filterIdsJ, jint filterIdsTypeJ);

        // Radial (range) search: every neighbor within radiusJ (faiss-domain, must be > 0), capped at
        // maxResultWindowJ; methodParamsJ may carry a search_window_size override.
        jobjectArray RangeSearch(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                 jfloatArray queryVectorJ, jfloat radiusJ, jobject methodParamsJ,
                                 jint maxResultWindowJ);

        // Radial search restricted to filterIdsJ (bitmap or batch encoding, per filterIdsTypeJ).
        jobjectArray RangeSearch_WithFilter(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                            jfloatArray queryVectorJ, jfloat radiusJ, jobject methodParamsJ,
                                            jint maxResultWindowJ, jlongArray filterIdsJ, jint filterIdsTypeJ);

        // Free the index at the given address.
        void Free(jlong indexPointerJ);

        // One-time native library initialization.
        void InitLibrary();

        // Whether the linked SVS runtime supports LVQ/LeanVec storage (requires Intel AVX-512).
        bool IsLvqLeanvecEnabled();

        // Map a k-NN space type string to the faiss metric ("cosinesimil" maps to inner product: the Java
        // layer normalizes vectors for cosine, making the two equivalent).
        faiss::MetricType TranslateSpaceToMetric(const std::string& spaceType);

        // Interrupt callback that aborts native index building when the Lucene merge driving it is aborted.
        // Identical to the main faiss library's callback; duplicated so this library stays self-contained.
        struct OpenSearchMergeInterruptCallback : faiss::InterruptCallback {

            OpenSearchMergeInterruptCallback(JNIUtil *jniUtil) {
                jutil = jniUtil;
                JNIEnv* jenv = jutil->GetJNICurrentEnv();
                mergeHelperClass = jniUtil->FindClass(jenv, "org/apache/lucene/index/MergeAbortChecker");
                isAbortedMethod = jniUtil->FindMethod(jenv, "org/apache/lucene/index/MergeAbortChecker", "isMergeAborted");
            }

            bool want_interrupt() override {
                JNIEnv* jenv = jutil->GetJNICurrentEnv();
                if (jenv == nullptr) {
                    std::cerr << "JNIEnv not found\n";
                    return false;
                }
                if (mergeHelperClass == nullptr) {
                    std::cerr << "MergeAbortChecker class not found\n";
                    return false;
                }
                if (isAbortedMethod == nullptr) {
                    std::cerr << "isMergeAborted method not found\n";
                    return false;
                }
                return (bool) jenv->CallStaticBooleanMethod(mergeHelperClass, isAbortedMethod);
            }

            JNIUtil *jutil;
            jclass mergeHelperClass;
            jmethodID isAbortedMethod;
        };
    }
}

#endif //OPENSEARCH_KNN_SVS_WRAPPER_H
