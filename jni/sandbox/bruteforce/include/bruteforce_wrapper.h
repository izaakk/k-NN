/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_SANDBOX_BRUTEFORCE_WRAPPER_H
#define OPENSEARCH_KNN_SANDBOX_BRUTEFORCE_WRAPPER_H

#include <jni.h>

#include "jni_util.h"

namespace knn_jni {
namespace bruteforce_wrapper {

jlong InitIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong numDocs, jint dim, jobject parametersJ);

void InsertToIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jintArray idsJ, jlong vectorsAddressJ, jint dim, jlong indexAddress);

void WriteIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong indexAddress, jobject outputJ);

jlong LoadIndexWithStream(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jobject readStreamJ);

jobjectArray QueryIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong indexPointer, jfloatArray queryVectorJ, jint k, jobject methodParamsJ);

void Free(jlong indexPointer);

}
}

#endif //OPENSEARCH_KNN_SANDBOX_BRUTEFORCE_WRAPPER_H
