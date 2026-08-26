# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp/svs_faiss_ep-gitclone-lastrun.txt" AND EXISTS "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp/svs_faiss_ep-gitinfo.txt" AND
  "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp/svs_faiss_ep-gitclone-lastrun.txt" IS_NEWER_THAN "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp/svs_faiss_ep-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp/svs_faiss_ep-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"
            clone --no-checkout --config "advice.detachedHead=false" "https://github.com/facebookresearch/faiss.git" "svs_faiss_ep"
    WORKING_DIRECTORY "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/facebookresearch/faiss.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"
          checkout "67f066f7a02f76d3178baccf4c31b4839ff0fee8" --
  WORKING_DIRECTORY "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '67f066f7a02f76d3178baccf4c31b4839ff0fee8'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp/svs_faiss_ep-gitinfo.txt" "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp/svs_faiss_ep-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/build-avx512/svs_faiss/src/svs_faiss_ep-stamp/svs_faiss_ep-gitclone-lastrun.txt'")
endif()
