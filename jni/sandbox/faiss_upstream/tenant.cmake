#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Native build for the experimental faiss_upstream tenant (libopensearchknn_faiss_upstream).
# Included by jni/sandbox/tenants.cmake only when -DCONFIG_SANDBOX=ON.
#
# The tenant vendors its OWN upstream faiss — pinned to v1.14.3 (0ca9df47), well ahead of the k-NN
# submodule pin — because its purpose is to expose index types (the Panorama family) that the pinned faiss
# does not have. The vendored build is static and every one of its symbols stays private to this library
# (hidden visibility + JNI-only export list via knn_sandbox_add_jni_library), so it coexists in one JVM
# with the different-version faiss inside libopensearchknn_faiss.

# v1.14.3 == 0ca9df4792b173d573044ee14ca0704780176e82 (pin the SHA, not the tag: tags can move).
knn_sandbox_vendor_faiss(faiss_upstream_faiss
    GIT_TAG 0ca9df4792b173d573044ee14ca0704780176e82
)

find_package(OpenMP REQUIRED)

knn_sandbox_add_jni_library(opensearchknn_faiss_upstream
    SOURCES
        ${KNN_SANDBOX_TENANT_DIR}/src/org_opensearch_knn_sandbox_faiss_1upstream_FaissUpstreamService.cpp
        ${KNN_SANDBOX_TENANT_DIR}/src/panorama_wrapper.cpp
    INCLUDE_DIRS
        ${KNN_SANDBOX_TENANT_DIR}/include
        ${faiss_upstream_faiss_INCLUDE_DIR}
    LINK_LIBRARIES
        ${faiss_upstream_faiss_LINK_LIB}
        OpenMP::OpenMP_CXX
    DEPENDS
        ${faiss_upstream_faiss_EP_TARGET}
)
