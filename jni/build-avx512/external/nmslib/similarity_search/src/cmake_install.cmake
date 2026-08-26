# Install script for directory: /home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/cmd_options.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/distcomp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/distcomp_edist.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/eval_metrics.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/eval_results.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/experimentconf.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/experiments.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/factory" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/factory/init_methods.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/factory" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/factory/init_spaces.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/falconn_heap_mod.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/flexbuff.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/floatdiff.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/global.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/gold_standard.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/hnswquery.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/idtype.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/incremental_quick_select.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/index.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/init.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/inmem_inv_index.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/knnquery.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/knnqueue.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/logging.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/memory.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/meta_analysis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/method" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/method/dummy.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/method" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/method/hnsw.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/method" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/method/hnsw_distfunc_opt_impl_inline.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/method" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/method/permutation_prefix_index.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/method" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/method/pivot_neighb_common.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/method" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/method/pivot_neighb_invindx.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/method" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/method/seqsearch.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/method" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/method/simple_inverted_index.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/method" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/method/small_world_rand.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/method" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/method/vptree.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/method" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/method/vptree_utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/methodfactory.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/my_isnan_isinf.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/object.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/params.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/params_cmdline.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/params_def.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/permutation_type.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/permutation_utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/portable_align.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/portable_intrinsics.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/portable_popcount.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/portable_prefetch.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/portable_simd.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/ported_boost_progress.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/pow.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/query.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/query_creator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/rangequery.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/read_data.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/report.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/report_intr_dim.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/searchoracle.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/simddebug.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/sort_arr_bi.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_ab_diverg.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_bit_hamming.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_bit_jaccard.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_bit_vector.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_bregman.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_dummy.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_js.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_l2sqr_sift.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_leven.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_lp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_rank_correl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_renyi_diverg.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_scalar.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_sparse_bin_common.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_sparse_dense_fusion.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_sparse_jaccard.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_sparse_lp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_sparse_scalar.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_sparse_scalar_bin_fast.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_sparse_scalar_fast.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_sparse_vector.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_sparse_vector_inter.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_sqfd.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_string.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_vector.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_vector_gen.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/space" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/space/space_word_embed.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/spacefactory.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/thread_pool.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/tune.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/include/ztimer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/ubuntu/workspaces/knn-pr3-svs-tenant/repos/k-NN/jni/external/nmslib/similarity_search/release/libNonMetricSpaceLib.a")
endif()

