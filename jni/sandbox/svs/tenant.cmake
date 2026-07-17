#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Native build for the experimental Intel SVS tenant (libopensearchknn_svs).
# Included by jni/sandbox/tenants.cmake only when -DCONFIG_SANDBOX=ON.
#
# The tenant vendors its OWN upstream faiss — pinned to an SVS-capable commit (carries IndexSVSVamana, the
# SVS factory descriptions and the find_package(svs_runtime) model), built with FAISS_ENABLE_SVS=ON and
# EXACTLY as upstream ships it (no k-NN patches; the wrapper uses only upstream faiss APIs). Every vendored
# symbol stays private to this library (hidden visibility + --exclude-libs), so it coexists in one JVM with
# the different-version faiss inside libopensearchknn_faiss.

# Provide the prebuilt Intel SVS runtime (conda-forge libsvs-runtime, sha256-pinned; override with
# -DSVS_RUNTIME_PREFIX / -DSVS_RUNTIME_URL via -Psandbox.cmake.args). Sets SVS_RUNTIME_PREFIX and resolves
# find_package(svs_runtime) for this build.
include(${KNN_SANDBOX_TENANT_DIR}/cmake/fetch-svs-runtime.cmake)

knn_sandbox_vendor_faiss(svs_faiss
    GIT_TAG 67f066f7a02f76d3178baccf4c31b4839ff0fee8
    CMAKE_ARGS
        -DFAISS_ENABLE_SVS=ON
        -DCMAKE_PREFIX_PATH=${SVS_RUNTIME_PREFIX}
)

find_package(OpenMP REQUIRED)

knn_sandbox_add_jni_library(opensearchknn_svs
    SOURCES
        ${KNN_SANDBOX_TENANT_DIR}/src/org_opensearch_knn_sandbox_svs_SvsService.cpp
        ${KNN_SANDBOX_TENANT_DIR}/src/svs_wrapper.cpp
        ${KNN_SANDBOX_TENANT_DIR}/src/svs_constants.cpp
    INCLUDE_DIRS
        ${KNN_SANDBOX_TENANT_DIR}/include
        ${svs_faiss_INCLUDE_DIR}
    LINK_LIBRARIES
        ${svs_faiss_LINK_LIB}
        svs::svs_runtime
        OpenMP::OpenMP_CXX
    DEPENDS
        ${svs_faiss_EP_TARGET}
)

target_compile_definitions(opensearchknn_svs PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=v0)

# Install the SVS runtime beside the JNI libraries so the $ORIGIN rpath resolves it in any deployment
# (the build tree's absolute link path must never be relied on).
add_custom_command(TARGET opensearchknn_svs POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SVS_RUNTIME_PREFIX}/lib/libsvs_runtime.so.0 ${CMAKE_BINARY_DIR}/release/
    COMMENT "Installing libsvs_runtime beside the tenant JNI library"
)
