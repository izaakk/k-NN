/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "org_opensearch_knn_sandbox_bruteforce_BruteForceService.h"

#include <jni.h>

#include "bruteforce_wrapper.h"
#include "jni_util.h"

static knn_jni::JNIUtil jniUtil;
static const jint KNN_BRUTEFORCE_JNI_VERSION = JNI_VERSION_1_1;

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv *env;
    if (vm->GetEnv((void **)&env, KNN_BRUTEFORCE_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }
    jniUtil.Initialize(env, vm);
    return KNN_BRUTEFORCE_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv *env;
    vm->GetEnv((void **)&env, KNN_BRUTEFORCE_JNI_VERSION);
    jniUtil.Uninitialize(env);
}

extern "C" {

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_sandbox_bruteforce_BruteForceService_initIndex(JNIEnv *env, jclass cls,
                                                                                               jlong numDocs, jint dim,
                                                                                               jobject parametersJ) {
    try {
        return knn_jni::bruteforce_wrapper::InitIndex(&jniUtil, env, numDocs, dim, parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_bruteforce_BruteForceService_insertToIndex(JNIEnv *env, jclass cls,
                                                                                                  jintArray idsJ,
                                                                                                  jlong vectorsAddressJ,
                                                                                                  jint dim,
                                                                                                  jlong indexAddress) {
    try {
        knn_jni::bruteforce_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dim, indexAddress);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_bruteforce_BruteForceService_writeIndex(JNIEnv *env, jclass cls,
                                                                                               jlong indexAddress,
                                                                                               jobject outputJ) {
    try {
        knn_jni::bruteforce_wrapper::WriteIndex(&jniUtil, env, indexAddress, outputJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_sandbox_bruteforce_BruteForceService_loadIndexWithStream(JNIEnv *env, jclass cls,
                                                                                                         jobject readStreamJ) {
    try {
        return knn_jni::bruteforce_wrapper::LoadIndexWithStream(&jniUtil, env, readStreamJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_sandbox_bruteforce_BruteForceService_queryIndex(JNIEnv *env, jclass cls,
                                                                                                       jlong indexPointer,
                                                                                                       jfloatArray queryVectorJ,
                                                                                                       jint k,
                                                                                                       jobject methodParamsJ) {
    try {
        return knn_jni::bruteforce_wrapper::QueryIndex(&jniUtil, env, indexPointer, queryVectorJ, k, methodParamsJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_bruteforce_BruteForceService_free(JNIEnv *env, jclass cls,
                                                                                         jlong indexPointer) {
    try {
        knn_jni::bruteforce_wrapper::Free(indexPointer);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

}  // extern "C"
