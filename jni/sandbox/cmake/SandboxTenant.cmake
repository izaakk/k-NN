#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Shared build harness for sandbox tenant JNI libraries. A tenant's jni/sandbox/<tenant>/tenant.cmake calls
# these helpers to produce a fully isolated libopensearchknn_<tenant> shared library that can coexist in one
# JVM with the built-in libopensearchknn_faiss — even when both embed different versions of the same vector
# library. The isolation recipe (hidden visibility + a JNI-only export list + a private static copy of the
# marshalling helpers + $ORIGIN rpath) is exactly the one validated end-to-end in the PR #3370 work. See sandbox/README.md ("The native path") for the tutorial that walks through using these.

# ---------------------------------------------------------------------------------------------------------
# knn_sandbox_add_jni_library(<target>
#     SOURCES <src>...
#     [INCLUDE_DIRS <dir>...]
#     [LINK_LIBRARIES <lib>...]
#     [DEPENDS <target>...]
# )
#
# Defines the tenant's SHARED JNI library with the full isolation recipe:
#   * a private STATIC copy of the generic JNI marshalling helpers (jni/src/jni_util.cpp + commons.cpp) is
#     compiled in, so the tenant library is runtime-independent of the SHARED opensearchknn_util that the
#     built-in libraries link;
#   * hidden symbol visibility plus a linker version script that exports ONLY the JNI entry points
#     (JNI_OnLoad / JNI_OnUnload / Java_org_opensearch_knn_*). Every other symbol — including the entire
#     symbol set of any statically embedded vector library — stays local to the tenant .so, so it cannot
#     interpose with a different version of the same library embedded in libopensearchknn_faiss (an ODR
#     hazard that could otherwise route calls into the wrong build);
#   * BUILD_RPATH/INSTALL_RPATH of $ORIGIN, so a tenant that ships a runtime .so alongside its JNI library
#     resolves it from its own directory;
#   * the repo-common target properties and registration in TARGET_LIBS, so `cmake --build ... --target
#     opensearchknn_<tenant>` and the install step behave like the built-in libraries.
# ---------------------------------------------------------------------------------------------------------
function(knn_sandbox_add_jni_library target)
    cmake_parse_arguments(TENANT "" "" "SOURCES;INCLUDE_DIRS;LINK_LIBRARIES;DEPENDS" ${ARGN})
    if(NOT TENANT_SOURCES)
        message(FATAL_ERROR "knn_sandbox_add_jni_library(${target}): SOURCES is required")
    endif()

    # One shared static helpers target for all tenants (created on first use).
    if(NOT TARGET opensearchknn_sandbox_jni_helpers)
        add_library(opensearchknn_sandbox_jni_helpers STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src/jni_util.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/commons.cpp)
        set_property(TARGET opensearchknn_sandbox_jni_helpers PROPERTY POSITION_INDEPENDENT_CODE ON)
        target_include_directories(opensearchknn_sandbox_jni_helpers PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            $ENV{JAVA_HOME}/include
            $ENV{JAVA_HOME}/include/${JVM_OS_TYPE})
    endif()

    add_library(${target} SHARED ${TENANT_SOURCES})
    if(TENANT_DEPENDS)
        add_dependencies(${target} ${TENANT_DEPENDS})
    endif()
    target_link_libraries(${target} opensearchknn_sandbox_jni_helpers ${TENANT_LINK_LIBRARIES})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        $ENV{JAVA_HOME}/include
        $ENV{JAVA_HOME}/include/${JVM_OS_TYPE}
        ${TENANT_INCLUDE_DIRS})
    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN")
    if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
        target_link_options(${target} PRIVATE
            "-Wl,--version-script=${KNN_SANDBOX_DIR}/cmake/jni_exports.version")
    endif()
    opensearch_set_common_properties(${target})
    list(APPEND TARGET_LIBS ${target})
    set(TARGET_LIBS "${TARGET_LIBS}" PARENT_SCOPE)
endfunction()
