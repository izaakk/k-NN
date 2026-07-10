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

/* Header for class org_opensearch_knn_sandbox_faiss_upstream_FaissUpstreamService */

#include <jni.h>

#ifndef _Included_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService
#define _Included_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_opensearch_knn_sandbox_faiss_upstream_FaissUpstreamService
 * Method:    initIndex
 * Signature: (JILjava/util/Map;)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_initIndex(
    JNIEnv*, jclass, jlong, jint, jobject);

/*
 * Class:     org_opensearch_knn_sandbox_faiss_upstream_FaissUpstreamService
 * Method:    insertToIndex
 * Signature: ([IJIJI)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_insertToIndex(
    JNIEnv*, jclass, jintArray, jlong, jint, jlong, jint);

/*
 * Class:     org_opensearch_knn_sandbox_faiss_upstream_FaissUpstreamService
 * Method:    writeIndex
 * Signature: (JLorg/opensearch/knn/index/store/IndexOutputWithBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_writeIndex(
    JNIEnv*, jclass, jlong, jobject);

/*
 * Class:     org_opensearch_knn_sandbox_faiss_upstream_FaissUpstreamService
 * Method:    loadIndexWithStream
 * Signature: (Lorg/opensearch/knn/index/store/IndexInputWithBuffer;)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_loadIndexWithStream(
    JNIEnv*, jclass, jobject);

/*
 * Class:     org_opensearch_knn_sandbox_faiss_upstream_FaissUpstreamService
 * Method:    queryIndex
 * Signature: (J[FILjava/util/Map;)[Lorg/opensearch/knn/index/query/KNNQueryResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_queryIndex(
    JNIEnv*, jclass, jlong, jfloatArray, jint, jobject);

/*
 * Class:     org_opensearch_knn_sandbox_faiss_upstream_FaissUpstreamService
 * Method:    free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_free(JNIEnv*, jclass, jlong);

/*
 * Class:     org_opensearch_knn_sandbox_faiss_upstream_FaissUpstreamService
 * Method:    initLibrary
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService_initLibrary(JNIEnv*, jclass);

#ifdef __cplusplus
}
#endif
#endif
