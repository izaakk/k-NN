# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/_deps/svs_runtime_pkg-src"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/_deps/svs_runtime_pkg-build"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/_deps/svs_runtime_pkg-subbuild/svs_runtime_pkg-populate-prefix"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/_deps/svs_runtime_pkg-subbuild/svs_runtime_pkg-populate-prefix/tmp"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/_deps/svs_runtime_pkg-subbuild/svs_runtime_pkg-populate-prefix/src/svs_runtime_pkg-populate-stamp"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/_deps/svs_runtime_pkg-subbuild/svs_runtime_pkg-populate-prefix/src"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/_deps/svs_runtime_pkg-subbuild/svs_runtime_pkg-populate-prefix/src/svs_runtime_pkg-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/_deps/svs_runtime_pkg-subbuild/svs_runtime_pkg-populate-prefix/src/svs_runtime_pkg-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/_deps/svs_runtime_pkg-subbuild/svs_runtime_pkg-populate-prefix/src/svs_runtime_pkg-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
