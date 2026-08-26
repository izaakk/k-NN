# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-build"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/tmp"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src"
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp${cfgdir}") # cfgdir has leading slash
endif()
