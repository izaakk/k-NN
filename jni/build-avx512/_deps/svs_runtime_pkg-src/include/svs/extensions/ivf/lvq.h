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

#include "svs/index/ivf/data_traits.h"
#include "svs/index/ivf/extensions.h"
#include "svs/quantization/lvq/lvq.h"

namespace svs::quantization::lvq {

// Only extend search for two-level dataset.
// Use primary accesor for search part
template <IsTwoLevelDataset Data>
PrimaryAccessor svs_invoke(
    svs::tag_t<svs::index::ivf::extensions::accessor>, const Data& SVS_UNUSED(data)
) {
    return PrimaryAccessor{};
}

template <IsLVQDataset Data, typename Distance>
biased_distance_t<Distance> svs_invoke(
    svs::tag_t<svs::index::ivf::extensions::per_thread_batch_search_setup>,
    const Data& dataset,
    const Distance& distance
);

// Extension for single_search with two-level LVQ datasets (used in batch iterator)
// Two-level LVQ requires reranking using the full (primary + residual) vectors
template <
    IsTwoLevelDataset Data,
    typename Cluster,
    typename BufferCentroids,
    typename BufferLeaves,
    typename Distance,
    typename Query,
    typename SearchCentroids,
    typename SearchLeaves>
void svs_invoke(
    svs::tag_t<svs::index::ivf::extensions::single_search>,
    const Data& SVS_UNUSED(dataset),
    const Cluster& cluster,
    BufferCentroids& buffer_centroids,
    BufferLeaves& buffer_leaves,
    Distance& distance,
    const Query& query,
    const SearchCentroids& search_centroids,
    const SearchLeaves& search_leaves
) {
    size_t n_inner_threads = buffer_leaves.size();

    // Search centroids to find nearest clusters
    search_centroids(query, buffer_centroids);

    // Search within selected clusters using primary (biased) distance
    search_leaves(query, distance, buffer_centroids, buffer_leaves);

    // Accumulate results from intra-query threads into buffer_leaves[0]
    for (size_t j = 1; j < n_inner_threads; ++j) {
        for (size_t k = 0; k < buffer_leaves[j].size(); ++k) {
            buffer_leaves[0].insert(buffer_leaves[j][k]);
        }
    }

    // Convert to global IDs and rerank using full (primary + residual) vectors
    for (size_t j = 0; j < buffer_leaves[0].size(); ++j) {
        auto& neighbor = buffer_leaves[0][j];
        auto cluster_id = neighbor.id();
        auto local_id = neighbor.get_local_id();
        auto global_id = cluster.get_global_id(cluster_id, local_id);
        neighbor.set_id(global_id);

        // Rerank using full vector (primary + residual)
        auto new_distance =
            distance::compute(distance, query, cluster.get_datum(cluster_id, local_id));
        neighbor.set_distance(new_distance);
    }

    // Sort buffer to get valid results in order after reranking
    buffer_leaves[0].sort();
}

// Only extend search for two-level dataset.
// One level datasets can use the default implementation directly.
template <
    IsTwoLevelDataset Data,
    typename Cluster,
    typename BufferCentroids,
    typename BufferLeaves,
    typename Distance,
    typename Queries,
    std::integral I,
    typename SearchCentroids,
    typename SearchLeaves>
void svs_invoke(
    svs::tag_t<svs::index::ivf::extensions::per_thread_batch_search>,
    const Data& SVS_UNUSED(dataset),
    const Cluster& cluster,
    BufferCentroids& buffer_centroids,
    BufferLeaves& buffer_leaves,
    Distance& distance,
    const Queries& queries,
    QueryResultView<I>& result,
    threads::UnitRange<size_t> thread_indices,
    const size_t tid,
    const SearchCentroids& search_centroids,
    const SearchLeaves& search_leaves
) {
    size_t n_inner_threads = buffer_leaves.size();
    size_t num_neighbors = result.n_neighbors();

    for (auto i : thread_indices) {
        const auto& query = queries.get_datum(i);
        search_centroids(query, buffer_centroids, i);
        search_leaves(query, distance, buffer_centroids, buffer_leaves, tid);

        // Accumulate results from intra-query threads
        for (size_t j = 1; j < n_inner_threads; ++j) {
            for (size_t k = 0; k < buffer_leaves[j].size(); ++k) {
                buffer_leaves[0].insert(buffer_leaves[j][k]);
            }
        }

        for (size_t j = 0; j < buffer_leaves[0].size(); ++j) {
            auto& neighbor = buffer_leaves[0][j];
            auto cluster_id = neighbor.id();
            auto local_id = neighbor.get_local_id();
            auto global_id = cluster.get_global_id(cluster_id, local_id);
            neighbor.set_id(global_id);

            // Rerank the results
            auto new_distance =
                distance::compute(distance, query, cluster.get_datum(cluster_id, local_id));
            neighbor.set_distance(new_distance);
        }
        buffer_leaves[0].sort();

        // Store results
        for (size_t j = 0; j < num_neighbors; ++j) {
            result.set(buffer_leaves[0][j], i, j);
        }
    }
}

template <IsLVQDataset Data, typename Alloc>
auto create_onelevel_from(const Data& original, size_t new_size, const Alloc& allocator) {
    constexpr size_t Bits = Data::primary_bits;
    constexpr size_t Extent = Data::extent;
    using Strategy = typename Data::strategy;
    auto& primary = original.get_primary_dataset();

    // Allocate a new primary dataset of the requested size.
    auto new_primary = ScaledBiasedDataset<Bits, Extent, Strategy, Alloc>(
        new_size, primary.static_dims(), primary.get_alignment(), allocator
    );
    return LVQDataset<Bits, 0, Extent, Strategy, Alloc>(
        std::move(new_primary), *original.view_centroids()
    );
}

// For two level LVQ
template <IsTwoLevelDataset Data, typename Alloc>
auto create_twolevel_from(const Data& original, size_t new_size, const Alloc& allocator) {
    constexpr size_t PBits = Data::primary_bits;
    constexpr size_t SBits = Data::residual_bits;
    constexpr size_t Extent = Data::extent;
    using Strategy = typename Data::strategy;
    auto& primary = original.get_primary_dataset();

    // Allocate new primary/residual dataset of the requested size.
    auto new_primary = ScaledBiasedDataset<PBits, Extent, Strategy, Alloc>(
        new_size, primary.static_dims(), primary.get_alignment(), allocator
    );
    auto new_residual =
        CompressedDataset<quantization::lvq::Unsigned, SBits, Extent, Alloc>(
            new_size, primary.static_dims(), allocator
        );

    return LVQDataset<PBits, SBits, Extent, Strategy, Alloc>(
        std::move(new_primary), std::move(new_residual), *original.view_centroids()
    );
}

// Default: for non-blocked allocators, always use lib::Allocator<std::byte>
template <IsLVQDataset Data, typename Alloc>
auto svs_invoke(
    svs::tag_t<index::ivf::extensions::create_dense_cluster>,
    const Data& original,
    size_t new_size,
    const Alloc& SVS_UNUSED(allocator)
) {
    if constexpr (Data::residual_bits != 0) {
        return create_twolevel_from(original, new_size, lib::Allocator<std::byte>());
    } else {
        return create_onelevel_from(original, new_size, lib::Allocator<std::byte>());
    }
}

// Specialization for Blocked allocators - needed for Dynamic IVF
template <IsLVQDataset Data, typename BlockedAlloc>
auto svs_invoke(
    svs::tag_t<index::ivf::extensions::create_dense_cluster>,
    const Data& original,
    size_t new_size,
    const data::Blocked<BlockedAlloc>& SVS_UNUSED(allocator)
) {
    auto blocked_alloc = data::Blocked<lib::Allocator<std::byte>>();
    if constexpr (Data::residual_bits != 0) {
        return create_twolevel_from(original, new_size, blocked_alloc);
    } else {
        return create_onelevel_from(original, new_size, blocked_alloc);
    }
}

template <IsTwoLevelDataset Src, IsTwoLevelDataset Dst, typename Idx>
void svs_invoke(
    svs::tag_t<index::ivf::extensions::set_dense_cluster>,
    const Src& src,
    Dst& dst,
    const std::vector<Idx>& src_ids,
    std::vector<Idx>& dst_ids
) {
    size_t i = 0;
    for (auto id : src_ids) {
        dst.set_primary(i, src.get_primary(id));
        dst.set_residual(i, src.get_residual(id));
        dst_ids[i] = id;
        ++i;
    }
}

} // namespace svs::quantization::lvq

/////
///// DataTypeTraits specialization for LVQ datasets
/////

namespace svs::index::ivf {

/// @brief Get LVQ strategy name as string
template <typename Strategy> constexpr std::string_view get_lvq_strategy_name() {
    if constexpr (std::is_same_v<Strategy, quantization::lvq::Sequential>) {
        return "sequential";
    } else {
        return "turbo";
    }
}

/// @brief Specialization of DataTypeTraits for LVQ datasets
///
/// This enables automatic save/load of LVQ data type information in IVF indices.
template <size_t Primary, size_t Residual, size_t Extent, typename Strategy, typename Alloc>
struct DataTypeTraits<
    quantization::lvq::LVQDataset<Primary, Residual, Extent, Strategy, Alloc>> {
    using Data = quantization::lvq::LVQDataset<Primary, Residual, Extent, Strategy, Alloc>;

    static DataTypeConfig get_config() {
        DataTypeConfig config;
        if constexpr (Residual == 0) {
            config.schema = std::string(quantization::lvq::one_level_serialization_schema);
        } else {
            config.schema = std::string(quantization::lvq::two_level_serialization_schema);
        }
        config.primary_bits = Primary;
        config.residual_bits = Residual;
        config.strategy = std::string(get_lvq_strategy_name<Strategy>());
        return config;
    }
};

} // namespace svs::index::ivf
