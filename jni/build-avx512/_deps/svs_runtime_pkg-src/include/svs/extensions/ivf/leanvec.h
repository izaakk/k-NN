/**
 *    Copyright (C) 2025 Intel Corporation
 *
 *    This software and the related documents are Intel copyrighted materials,
 *    and your use of them is governed by the express license under which they
 *    were provided to you ("License"). Unless the License provides otherwise,
 *    you may not use, modify, copy, publish, distribute, disclose or transmit
 *    this software or the related documents without Intel's prior written
 *    permission.
 *
 *    This software and the related documents are provided as is, with no
 *    express or implied warranties, other than those that are expressly stated
 *    in the License.
 */

#pragma once

#include "svs/extensions/ivf/lvq.h"
#include "svs/index/ivf/extensions.h"
#include "svs/leanvec/leanvec.h"

namespace svs::leanvec {

template <IsLeanDataset Data, typename Distance>
auto svs_invoke(
    svs::tag_t<svs::index::ivf::extensions::per_thread_batch_search_setup>,
    const Data& data,
    const Distance& distance
) {
    return std::make_tuple(
        threads::shallow_copy(distance),
        data.adapt(data.view_primary_dataset(), distance),
        data.adapt_secondary(data.view_secondary_dataset(), distance)
    );
}

// Extension for get_distance_ext with LeanVec datasets (used in dynamic IVF)
template <IsLeanDataset Data, typename Distance, typename Query>
double svs_invoke(
    svs::tag_t<svs::index::ivf::extensions::get_distance_ext>,
    const Data& data,
    const Distance& distance,
    size_t pos,
    const Query& query
) {
    auto secondary_distance = data.adapt_secondary(data.view_secondary_dataset(), distance);
    svs::distance::maybe_fix_argument(secondary_distance, query);
    auto secondary_data = data.get_secondary(pos);
    auto dist = svs::distance::compute(secondary_distance, query, secondary_data);
    return static_cast<double>(dist);
}

// Extension for single_search with LeanVec datasets (used in batch iterator)
template <
    IsLeanDataset Data,
    typename Cluster,
    typename BufferCentroids,
    typename BufferLeaves,
    typename Scratch,
    typename Query,
    typename SearchCentroids,
    typename SearchLeaves>
void svs_invoke(
    svs::tag_t<svs::index::ivf::extensions::single_search>,
    const Data& dataset,
    const Cluster& cluster,
    BufferCentroids& buffer_centroids,
    BufferLeaves& buffer_leaves,
    Scratch& scratch,
    const Query& query,
    const SearchCentroids& search_centroids,
    const SearchLeaves& search_leaves
) {
    size_t n_inner_threads = buffer_leaves.size();

    auto& [distance, distance_primary, distance_secondary] = scratch;

    // Preprocess the query (single query case)
    auto query_data = data::SimpleData<float>(1, query.size());
    std::copy(query.begin(), query.end(), query_data.data());
    auto processed_queries = dataset.preprocess_queries(distance, query_data.cview());
    const auto& processed_query = processed_queries.get_datum(0);

    // Search centroids to find nearest clusters
    search_centroids(query, buffer_centroids);

    // Search within selected clusters using primary distance
    search_leaves(processed_query, distance_primary, buffer_centroids, buffer_leaves);

    // Accumulate results from intra-query threads into buffer_leaves[0]
    for (size_t j = 1; j < n_inner_threads; ++j) {
        for (size_t k = 0; k < buffer_leaves[j].size(); ++k) {
            buffer_leaves[0].insert(buffer_leaves[j][k]);
        }
    }

    // Rerank using secondary distance and convert to global IDs
    distance::maybe_fix_argument(distance_secondary, query);
    for (size_t j = 0; j < buffer_leaves[0].size(); ++j) {
        auto& neighbor = buffer_leaves[0][j];
        auto cluster_id = neighbor.id();
        auto local_id = neighbor.get_local_id();
        auto global_id = cluster.get_global_id(cluster_id, local_id);
        neighbor.set_id(global_id);

        // Rerank the results using secondary data
        auto new_distance = distance::compute(
            distance_secondary, query, cluster.get_secondary(cluster_id, local_id)
        );
        neighbor.set_distance(new_distance);
    }

    // Sort buffer to get valid results in order
    buffer_leaves[0].sort();
}

template <
    IsLeanDataset Data,
    typename Cluster,
    typename BufferCentroids,
    typename BufferLeaves,
    typename Scratch,
    typename Queries,
    std::integral I,
    typename SearchCentroids,
    typename SearchLeaves>
void svs_invoke(
    svs::tag_t<svs::index::ivf::extensions::per_thread_batch_search>,
    const Data& dataset,
    const Cluster& cluster,
    BufferCentroids& buffer_centroids,
    BufferLeaves& buffer_leaves,
    Scratch& scratch,
    const Queries& queries,
    QueryResultView<I>& result,
    threads::UnitRange<size_t> thread_indices,
    const size_t tid,
    const SearchCentroids& search_centroids,
    const SearchLeaves& search_leaves
) {
    size_t n_inner_threads = buffer_leaves.size();
    size_t num_neighbors = result.n_neighbors();

    auto& [distance, distance_primary, distance_secondary] = scratch;

    size_t batch_start = thread_indices.start();
    data::SimpleData<float> processed_queries;
    using QueryType = typename Queries::element_type;

    if constexpr (std::is_same_v<std::remove_cv_t<QueryType>, float>) {
        auto query_batch = data::ConstSimpleDataView<float>(
            &queries.get_datum(thread_indices.front()).front(),
            thread_indices.size(),
            queries.dimensions()
        );
        processed_queries = dataset.preprocess_queries(distance, query_batch);
    } else {
        // TODO: To minimize conversion overhead for Float16 queries, add support for
        // Float16 directly within the `preprocess_queries` function.
        auto queries_f32 =
            svs::data::SimpleData<float>{queries.size(), queries.dimensions()};
        svs::data::copy(queries, queries_f32);
        auto query_batch = data::ConstSimpleDataView<float>(
            &queries_f32.get_datum(thread_indices.front()).front(),
            thread_indices.size(),
            queries_f32.dimensions()
        );
        processed_queries = dataset.preprocess_queries(distance, query_batch);
    }

    for (auto i : thread_indices) {
        const auto& query = queries.get_datum(i);
        search_centroids(query, buffer_centroids, i);

        const auto& processed_query = processed_queries.get_datum(i - batch_start);
        search_leaves(
            processed_query, distance_primary, buffer_centroids, buffer_leaves, tid
        );

        // Accumulate results from intra-query threads
        for (size_t j = 1; j < n_inner_threads; ++j) {
            for (size_t k = 0; k < buffer_leaves[j].size(); ++k) {
                buffer_leaves[0].insert(buffer_leaves[j][k]);
            }
        }

        distance::maybe_fix_argument(distance_secondary, query);
        for (size_t j = 0; j < buffer_leaves[0].size(); ++j) {
            auto& neighbor = buffer_leaves[0][j];
            auto cluster_id = neighbor.id();
            auto local_id = neighbor.get_local_id();
            auto global_id = cluster.get_global_id(cluster_id, local_id);
            neighbor.set_id(global_id);

            // Rerank the results
            auto new_distance = distance::compute(
                distance_secondary, query, cluster.get_secondary(cluster_id, local_id)
            );
            neighbor.set_distance(new_distance);
        }
        buffer_leaves[0].sort();

        // Store results
        for (size_t j = 0; j < num_neighbors; ++j) {
            result.set(buffer_leaves[0][j], i, j);
        }
    }
}

template <IsLeanDataset Data, typename Alloc>
auto create_leanvec_from(const Data& original, size_t new_size, const Alloc& allocator) {
    auto new_primary = index::ivf::extensions::create_dense_cluster(
        original.view_primary_dataset(), new_size, allocator
    );
    auto new_secondary = index::ivf::extensions::create_dense_cluster(
        original.view_secondary_dataset(), new_size, allocator
    );

    return LeanDataset<
        typename Data::primary_data_type,
        typename Data::secondary_data_type,
        Data::leanvec_extent,
        Data::extent,
        Alloc>(
        std::move(new_primary),
        std::move(new_secondary),
        original.get_leanvec_matrices(),
        original.get_means(),
        original.is_pca()
    );
}

// Default version: use lib::Allocator for non-blocked allocators
template <IsLeanDataset Data, typename Alloc>
auto svs_invoke(
    svs::tag_t<index::ivf::extensions::create_dense_cluster>,
    const Data& original,
    size_t new_size,
    const Alloc& SVS_UNUSED(allocator)
) {
    return create_leanvec_from(original, new_size, lib::Allocator<std::byte>());
}

// Specialization for Blocked allocators: pass the allocator through
template <IsLeanDataset Data, typename BlockedAlloc>
auto svs_invoke(
    svs::tag_t<index::ivf::extensions::create_dense_cluster>,
    const Data& original,
    size_t new_size,
    const data::Blocked<BlockedAlloc>& SVS_UNUSED(allocator)
) {
    auto blocked_alloc = data::Blocked<lib::Allocator<std::byte>>();
    return create_leanvec_from(original, new_size, blocked_alloc);
}

template <IsLeanDataset Src, IsLeanDataset Dst, typename Idx>
void svs_invoke(
    svs::tag_t<index::ivf::extensions::set_dense_cluster>,
    const Src& src,
    Dst& dst,
    const std::vector<Idx>& src_ids,
    std::vector<Idx>& dst_ids
) {
    size_t i = 0;
    for (auto id : src_ids) {
        dst.set_primary(i, src.get_datum(id));
        dst.set_secondary(i, src.get_secondary(id));
        dst_ids[i] = id;
        ++i;
    }
}

} // namespace svs::leanvec

/////
///// DataTypeTraits specialization for LeanVec datasets
/////

namespace svs::index::ivf {

/// @brief Get LeanVec kind name as string
template <typename Kind> constexpr std::string_view get_leanvec_kind_name() {
    if constexpr (std::is_same_v<Kind, float>) {
        return "float32";
    } else if constexpr (std::is_same_v<Kind, Float16>) {
        return "float16";
    } else if constexpr (std::is_same_v<Kind, leanvec::UsingLVQ<4>>) {
        return "lvq4";
    } else if constexpr (std::is_same_v<Kind, leanvec::UsingLVQ<8>>) {
        return "lvq8";
    } else {
        return "unknown";
    }
}

/// @brief Specialization of DataTypeTraits for LeanVec datasets
///
/// This enables automatic save/load of LeanVec data type information in IVF indices.
template <typename T1, typename T2, size_t LeanVecDims, size_t Extent, typename Alloc>
struct DataTypeTraits<leanvec::LeanDataset<T1, T2, LeanVecDims, Extent, Alloc>> {
    using Data = leanvec::LeanDataset<T1, T2, LeanVecDims, Extent, Alloc>;

    static DataTypeConfig get_config() {
        DataTypeConfig config;
        config.schema = std::string(leanvec::lean_dataset_schema);
        config.primary_kind = std::string(get_leanvec_kind_name<T1>());
        config.secondary_kind = std::string(get_leanvec_kind_name<T2>());
        // Use 0 to represent dynamic dimensions (SIZE_MAX) for serialization
        config.leanvec_dims = (LeanVecDims == Dynamic) ? 0 : LeanVecDims;
        return config;
    }
};

} // namespace svs::index::ivf
