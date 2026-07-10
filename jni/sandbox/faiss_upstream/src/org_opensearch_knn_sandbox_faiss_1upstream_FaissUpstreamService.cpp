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

#include "org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService.h"

#include <jni.h>

#include "jni_util.h"
#include "panorama_wrapper.h"
#include "faiss_stream_support.h"

static knn_jni::JNIUtil jniUtil;
static const jint KNN_FAISS_UPSTREAM_JNI_VERSION = JNI_VERSION_1_1;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, KNN_FAISS_UPSTREAM_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }
    jniUtil.Initialize(env, vm);
    return KNN_FAISS_UPSTREAM_JNI_VERSION;
}

void JNI_OnUnload(JavaVM* vm, void* reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_FAISS_UPSTREAM_JNI_VERSION);
    jniUtil.Uninitialize(env);
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_initIndex(
    JNIEnv* env, jclass cls, jlong numDocs, jint dimJ, jobject parametersJ) {
    try {
        return knn_jni::panorama_wrapper::InitIndex(&jniUtil, env, numDocs, dimJ, parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_insertToIndex(
    JNIEnv* env, jclass cls, jintArray idsJ, jlong vectorsAddressJ, jint dimJ, jlong indexAddress, jint threadCount) {
    try {
        knn_jni::panorama_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexAddress, threadCount);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_writeIndex(
    JNIEnv* env, jclass cls, jlong indexAddress, jobject output) {
    try {
        knn_jni::panorama_wrapper::WriteIndex(&jniUtil, env, output, indexAddress);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_loadIndexWithStream(
    JNIEnv* env, jclass cls, jobject readStream) {
    try {
        // `readStream` is an IndexInputWithBuffer; bridge it to a faiss IOReader.
        knn_jni::stream::NativeEngineIndexInputMediator mediator{&jniUtil, env, readStream};
        knn_jni::stream::FaissOpenSearchIOReader faissOpenSearchIOReader{&mediator};
        return knn_jni::panorama_wrapper::LoadIndexWithStream(&faissOpenSearchIOReader);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_queryIndex(
    JNIEnv* env, jclass cls, jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ) {
    try {
        return knn_jni::panorama_wrapper::QueryIndex(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_free(
    JNIEnv* env, jclass cls, jlong indexPointerJ) {
    try {
        knn_jni::panorama_wrapper::Free(indexPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_initLibrary(
    JNIEnv* env, jclass cls) {
    try {
        knn_jni::panorama_wrapper::InitLibrary();
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}
