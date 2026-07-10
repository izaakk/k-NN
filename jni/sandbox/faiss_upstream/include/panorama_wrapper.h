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

#ifndef OPENSEARCH_KNN_PANORAMA_WRAPPER_H
#define OPENSEARCH_KNN_PANORAMA_WRAPPER_H

#include <jni.h>

#include "jni_util.h"
#include "faiss/impl/io.h"

namespace knn_jni {
    namespace panorama_wrapper {

        // Creates an IndexHNSWFlatPanorama from the faiss factory description in the parameters (e.g.
        // "HNSW16,FlatPanorama8"), wrapped in an IndexIDMap. HNSW builds incrementally — no training.
        jlong InitIndex(knn_jni::JNIUtilInterface* jniUtil, JNIEnv* env, jlong numDocs, jint dimJ, jobject parametersJ);

        // Adds a batch of ids + off-heap vectors to the index (incremental HNSW graph insertion).
        void InsertToIndex(
            knn_jni::JNIUtilInterface* jniUtil,
            JNIEnv* env,
            jintArray idsJ,
            jlong vectorsAddressJ,
            jint dimJ,
            jlong indexAddressJ,
            jint threadCount);

        // Serializes the index to the OpenSearch output stream and frees it.
        void WriteIndex(knn_jni::JNIUtilInterface* jniUtil, JNIEnv* env, jobject output, jlong indexAddressJ);

        // Loads a serialized IDMap(IndexHNSWFlatPanorama) from the OpenSearch input stream.
        jlong LoadIndexWithStream(faiss::IOReader* ioReader);

        // Top-k query; honors the "ef_search" method parameter.
        jobjectArray QueryIndex(
            knn_jni::JNIUtilInterface* jniUtil,
            JNIEnv* env,
            jlong indexPointerJ,
            jfloatArray queryVectorJ,
            jint kJ,
            jobject methodParamsJ);

        void Free(jlong indexPointerJ);

        void InitLibrary();
    }
}

#endif  // OPENSEARCH_KNN_PANORAMA_WRAPPER_H
