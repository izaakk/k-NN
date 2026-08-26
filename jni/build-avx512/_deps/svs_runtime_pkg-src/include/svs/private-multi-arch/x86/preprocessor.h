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

#define DECOMPRESS_TEMPLATE_HELPER(SPEC, AVX, EXTENT)                                                                           \
    SPEC void decompress<                                                                                                       \
        AVX>(std::span<float, std::dynamic_extent>, const ScaledBiasedVector<4, EXTENT, Sequential>&, const float*);            \
    SPEC void decompress<                                                                                                       \
        AVX>(std::span<float, std::dynamic_extent>, const ScaledBiasedVector<4, EXTENT, Turbo<16, 8>>&, const float*);          \
    SPEC void decompress<                                                                                                       \
        AVX>(std::span<float, std::dynamic_extent>, const ScaledBiasedVector<8, EXTENT, Sequential>&, const float*);            \
    SPEC void decompress<                                                                                                       \
        AVX>(std::span<float, std::dynamic_extent>, const ScaledBiasedVector<8, EXTENT, Turbo<16, 4>>&, const float*);          \
    SPEC void decompress<                                                                                                       \
        AVX>(std::span<float, std::dynamic_extent>, const ScaledBiasedWithResidual<4, 4, EXTENT, Sequential>&, const float*);   \
    SPEC void decompress<                                                                                                       \
        AVX>(std::span<float, std::dynamic_extent>, const ScaledBiasedWithResidual<4, 4, EXTENT, Turbo<16, 8>>&, const float*); \
    SPEC void decompress<                                                                                                       \
        AVX>(std::span<float, std::dynamic_extent>, const ScaledBiasedWithResidual<4, 8, EXTENT, Sequential>&, const float*);   \
    SPEC void decompress<                                                                                                       \
        AVX>(std::span<float, std::dynamic_extent>, const ScaledBiasedWithResidual<4, 8, EXTENT, Turbo<16, 8>>&, const float*); \
    SPEC void decompress<                                                                                                       \
        AVX>(std::span<float, std::dynamic_extent>, const ScaledBiasedWithResidual<8, 8, EXTENT, Sequential>&, const float*);

#define DECOMPRESS_INSTANTIATE_TEMPLATE(EXTENT, AVX) \
    DECOMPRESS_TEMPLATE_HELPER(template, AVX, EXTENT);

#define DECOMPRESS_EXTERN_TEMPLATE(EXTENT, AVX) \
    DECOMPRESS_TEMPLATE_HELPER(extern template, AVX, EXTENT);

#define COMPUTE_QUANTIZED_TEMPLATE_HELPER(SPEC, AVX, DISTANCE, EXTENT)                                              \
    SPEC float compute_quantized<                                                                                   \
        AVX>(const DISTANCE&, std::span<const float>, const ScaledBiasedVector<4, EXTENT, Sequential>&);            \
    SPEC float compute_quantized<                                                                                   \
        AVX>(const DISTANCE&, std::span<const float>, const ScaledBiasedVector<4, EXTENT, Turbo<16, 8>>&);          \
    SPEC float compute_quantized<                                                                                   \
        AVX>(const DISTANCE&, std::span<const float>, const ScaledBiasedVector<8, EXTENT, Sequential>&);            \
    SPEC float compute_quantized<                                                                                   \
        AVX>(const DISTANCE&, std::span<const float>, const ScaledBiasedVector<8, EXTENT, Turbo<16, 4>>&);          \
    SPEC float compute_quantized<                                                                                   \
        AVX>(const DISTANCE&, std::span<const float>, const ScaledBiasedWithResidual<4, 4, EXTENT, Sequential>&);   \
    SPEC float compute_quantized<                                                                                   \
        AVX>(const DISTANCE&, std::span<const float>, const ScaledBiasedWithResidual<4, 4, EXTENT, Turbo<16, 8>>&); \
    SPEC float compute_quantized<                                                                                   \
        AVX>(const DISTANCE&, std::span<const float>, const ScaledBiasedWithResidual<4, 8, EXTENT, Sequential>&);   \
    SPEC float compute_quantized<                                                                                   \
        AVX>(const DISTANCE&, std::span<const float>, const ScaledBiasedWithResidual<4, 8, EXTENT, Turbo<16, 8>>&); \
    SPEC float compute_quantized<                                                                                   \
        AVX>(const DISTANCE&, std::span<const float>, const ScaledBiasedWithResidual<8, 8, EXTENT, Sequential>&);

#define COMPUTE_QUANTIZED_TEMPLATE_DISTANCE(SPEC, AVX, EXTENT)                       \
    COMPUTE_QUANTIZED_TEMPLATE_HELPER(SPEC, AVX, svs::distance::DistanceL2, EXTENT); \
    COMPUTE_QUANTIZED_TEMPLATE_HELPER(SPEC, AVX, svs::distance::DistanceIP, EXTENT); \
    COMPUTE_QUANTIZED_TEMPLATE_HELPER(                                               \
        SPEC, AVX, svs::distance::DistanceCosineSimilarity, EXTENT                   \
    );                                                                               \
    COMPUTE_QUANTIZED_TEMPLATE_HELPER(SPEC, AVX, DistanceFastIP, EXTENT);            \
    COMPUTE_QUANTIZED_TEMPLATE_HELPER(SPEC, AVX, CosineSimilarityBiased, EXTENT);

#define COMPUTE_QUANTIZED_INSTANTIATE_TEMPLATE(EXTENT, AVX) \
    COMPUTE_QUANTIZED_TEMPLATE_DISTANCE(template, AVX, EXTENT);

#define COMPUTE_QUANTIZED_EXTERN_TEMPLATE(EXTENT, AVX) \
    COMPUTE_QUANTIZED_TEMPLATE_DISTANCE(extern template, AVX, EXTENT);
