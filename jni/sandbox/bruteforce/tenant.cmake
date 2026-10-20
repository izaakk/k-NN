#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Native build for the example brute-force tenant (libopensearchknn_bruteforce).
# Included by jni/sandbox/tenants.cmake only when -DCONFIG_SANDBOX=ON.

knn_sandbox_add_jni_library(opensearchknn_bruteforce
    SOURCES
        ${KNN_SANDBOX_TENANT_DIR}/src/org_opensearch_knn_sandbox_bruteforce_BruteForceService.cpp
        ${KNN_SANDBOX_TENANT_DIR}/src/bruteforce_wrapper.cpp
    INCLUDE_DIRS
        ${KNN_SANDBOX_TENANT_DIR}/include
)
