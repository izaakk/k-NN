/**
 *    Copyright (C) 2023 Intel Corporation
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

#include "svs/index/vamana/extensions.h"
#include "svs/leanvec/leanvec.h"
// We need to include the `lvq` extensions because LeanVec is LVQ compatible and we may
// need to access specializations defined by LVQ.
#include "svs/extensions/vamana/lvq.h"

namespace svs::leanvec {

/////
///// Entry Point Computation
/////

// Delegate to the entry-point computation for the primary dataset.
template <IsLeanDataset Data, threads::ThreadPool Pool, typename Predicate>
size_t svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::compute_entry_point>,
    const Data& data,
    Pool& threadpool,
    Predicate&& predicate
) {
    return svs::index::vamana::extensions::compute_entry_point(
        data.view_primary_dataset(), threadpool, SVS_FWD(predicate)
    );
}

template <IsLeanDataset Data>
svs::index::vamana::GreedySearchPrefetchParameters svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::estimate_prefetch_parameters>,
    const Data& SVS_UNUSED(data)
) {
    // Conservative prefetching.
    return svs::index::vamana::GreedySearchPrefetchParameters{1, 1};
}

/////
///// Vamana Build
/////

template <typename Distance> struct VamanaBuildAdaptor {
  public:
    using search_distance_type = Distance;
    using general_distance_type = Distance;

    explicit VamanaBuildAdaptor(Distance distance)
        : distance_{std::move(distance)} {}

    // For graph construction, primary data is used for all purposes
    template <IsLeanDataset Data>
    auto access_query_for_graph_search(const Data& dataset, size_t i) const {
        return dataset.get_datum(i);
    }

    template <IsLeanDataset Data, typename Query>
    const Query& modify_post_search_query(
        const Data& SVS_UNUSED(data), size_t SVS_UNUSED(i), const Query& query
    ) const {
        return query;
    }

    // As such, there is no-need to call `maybe_fix_argument` following graph search.
    static constexpr bool refix_argument_after_search = false;

    // Search functor used for the graph search portion of index construction.
    search_distance_type& graph_search_distance() { return distance_; }

    // Only access the primary data
    data::GetDatumAccessor graph_search_accessor() const {
        return data::GetDatumAccessor{};
    }

    // Using only the primary data for graph construction, no need for reranking
    template <IsLeanDataset Data, typename Query, NeighborLike N>
    Neighbor<typename N::index_type> post_search_modify(
        const Data& SVS_UNUSED(dataset),
        general_distance_type& SVS_UNUSED(d),
        const Query& SVS_UNUSED(query),
        const N& n
    ) const {
        return n;
    }

    // General distance computations use the same underlying distance functor as graph
    // search distances.
    general_distance_type& general_distance() { return distance_; }

    // Use primary data for graph construction in all cases
    data::GetDatumAccessor general_accessor() const { return data::GetDatumAccessor{}; }

  public:
    Distance distance_;
};

template <IsLeanDataset Dataset, typename Distance>
auto svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::build_adaptor>,
    const Dataset& dataset,
    const Distance& distance
) {
    return VamanaBuildAdaptor{
        dataset.adapt_for_self(dataset.view_primary_dataset(), distance)};
}

/////
///// Vamana Search
/////

template <IsLeanDataset Data, typename Distance>
auto svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::single_search_setup>,
    const Data& data,
    const Distance& distance
) {
    return std::make_tuple(
        threads::shallow_copy(distance),
        data.adapt(data.view_primary_dataset(), distance),
        data.adapt_secondary(data.view_secondary_dataset(), distance)
    );
}

template <
    IsLeanDataset Data,
    typename SearchBuffer,
    typename Scratch,
    typename Query,
    typename Search,
    typename Index>
void svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::single_search>,
    const Data& dataset,
    SearchBuffer& search_buffer,
    Scratch& scratch,
    const Query& query,
    const Search& search,
    const Index& index,
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
) {
    auto& [distance, distance_primary, distance_secondary] = scratch;
    using QueryType = typename Query::element_type;

    data::SimpleData<float> processed_queries;

    if constexpr (std::is_same_v<std::remove_cv_t<QueryType>, float>) {
        auto query_data = data::ConstSimpleDataView<float>(query.data(), 1, query.size());
        processed_queries = dataset.preprocess_queries(distance, query_data);
    } else {
        // TODO: To minimize conversion overhead for Float16 queries, add support for
        // Float16 directly within the `preprocess_queries` function.
        auto query_data = data::SimpleData<float>(1, query.size());
        query_data.set_datum(0, query);
        processed_queries = dataset.preprocess_queries(distance, query_data.cview());
    }

    // Perform graph search.
    assert(processed_queries.size() == 1);
    const auto& processed_query = processed_queries.get_datum(0);
    {
        auto accessor = data::GetDatumAccessor();
        search(processed_query, accessor, distance_primary, search_buffer);
    }

    // Check if request to cancel the search
    if (cancel()) {
        return;
    }

    // For LeanVec, always rerank the result
    distance::maybe_fix_argument(distance_secondary, query);

    for (size_t j = 0, jmax = search_buffer.size(); j < jmax; ++j) {
        auto& neighbor = search_buffer[j];
        auto id = neighbor.id();
        auto new_distance =
            distance::compute(distance_secondary, query, dataset.get_secondary(id));
        neighbor.set_distance(new_distance);
    }
    if constexpr (Index::needs_id_translation) {
        svs::index::vamana::extensions::check_and_supplement_search_buffer(
            index, search_buffer, query
        );
    }
    search_buffer.sort();
}

// Returning a tuple consisting of:
//
// * The original abstract distance (to be used in pre-processing)
// * The distance modified for the primary dataset.
// * The distance modified for the secondary dataset.
//
template <IsLeanDataset Data, typename Distance>
auto svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::per_thread_batch_search_setup>,
    const Data& data,
    const Distance& distance
) {
    return std::make_tuple(
        threads::shallow_copy(distance),
        data.adapt(data.view_primary_dataset(), distance),
        data.adapt_secondary(data.view_secondary_dataset(), distance)
    );
}

template <
    IsLeanDataset Data,
    typename SearchBuffer,
    typename Scratch,
    typename Queries,
    std::integral I,
    typename Search,
    typename Index>
void svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::per_thread_batch_search>,
    const Data& dataset,
    SearchBuffer& search_buffer,
    Scratch& scratch,
    const Queries& queries,
    QueryResultView<I>& result,
    threads::UnitRange<size_t> thread_indices,
    const Search& search,
    const Index& index,
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
) {
    size_t num_neighbors = result.n_neighbors();
    size_t batch_start = thread_indices.start();

    auto& [distance, distance_primary, distance_secondary] = scratch;

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

    // Perform graph search.
    for (auto i : thread_indices) {
        const auto& query = queries.get_datum(i);
        const auto& processed_query = processed_queries.get_datum(i - batch_start);

        {
            auto accessor = data::GetDatumAccessor();
            search(processed_query, accessor, distance_primary, search_buffer);
        }

        // Check if request to cancel the search
        if (cancel()) {
            return;
        }

        // For LeanVec, always rerank the result
        distance::maybe_fix_argument(distance_secondary, query);
        for (size_t j = 0, jmax = search_buffer.size(); j < jmax; ++j) {
            auto& neighbor = search_buffer[j];
            auto id = neighbor.id();
            auto new_distance =
                distance::compute(distance_secondary, query, dataset.get_secondary(id));
            neighbor.set_distance(new_distance);
        }
        if constexpr (Index::needs_id_translation) {
            svs::index::vamana::extensions::check_and_supplement_search_buffer(
                index, search_buffer, query
            );
        }
        search_buffer.sort();

        // Copy back results.
        for (size_t j = 0; j < num_neighbors; ++j) {
            result.set(search_buffer[j], i, j);
        }
    }
}

/////
///// Calibration
/////

template <IsLeanDataset Dataset>
constexpr bool svs_invoke(svs::index::vamana::extensions::UsesReranking<Dataset>) {
    return true;
}

/////
///// Reconstruction
/////

namespace detail {
template <IsLeanDataset Data> using secondary_dataset_type = typename Data::secondary_type;

// An auxiliary accessor that accesses the secondary dataset using the nested accessor.
template <typename T> struct SecondaryReconstructor {
    // The return-type dance here basically says that we return whetever the result of
    // invoking the `secondary_accessor_` on the secondary dataset returns.
    template <IsLeanDataset Data>
    std::invoke_result_t<T, const secondary_dataset_type<Data>&, size_t>
    operator()(const Data& data, size_t i) {
        return secondary_accessor_(data.view_secondary_dataset(), i);
    }

    ///// Members
    // Auxiliary accessor for the secondary dataset.
    T secondary_accessor_;
};

// Get the type of the accessor returned by the secondary dataset for this customization
// point object.
template <IsLeanDataset Data>
using secondary_accessor_t = svs::svs_invoke_result_t<
    svs::tag_t<svs::index::vamana::extensions::reconstruct_accessor>,
    const detail::secondary_dataset_type<Data>&>;

} // namespace detail

// Compose the reconstruction accessor for the secondary dataset with an accessor that
// grabs the secondary dataset.
template <IsLeanDataset Dataset>
detail::SecondaryReconstructor<detail::secondary_accessor_t<Dataset>> svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::reconstruct_accessor> cpo,
    const Dataset& data
) {
    using T = detail::secondary_accessor_t<Dataset>;
    return detail::SecondaryReconstructor<T>{cpo(data.view_secondary_dataset())};
}

/////
///// Distance
/////

template <IsLeanDataset Data, typename Distance, typename Query>
double svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::get_distance_ext>,
    const Data& data,
    const Distance& distance,
    size_t internal_id,
    const Query& query
) {
    //  For Leanvec dataset, use secondary data to calculate the distance
    auto secondary_distance = data.adapt_secondary(data.view_secondary_dataset(), distance);
    svs::distance::maybe_fix_argument(secondary_distance, query);
    auto secondary_span = data.get_secondary(internal_id);

    auto dist = svs::distance::compute(secondary_distance, query, secondary_span);

    return static_cast<double>(dist);
}
} // namespace svs::leanvec
