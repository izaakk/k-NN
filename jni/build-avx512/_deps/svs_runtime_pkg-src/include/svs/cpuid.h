/**
 *    Copyright (C) 2024 Intel Corporation
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

#include <cpuid.h>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "svs/lib/avx_detection.h"

namespace svs {
namespace detail {

inline bool intel_enabled() {
    uint32_t eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);
    std::string vendor_id = std::string((const char*)&ebx, 4) +
                            std::string((const char*)&edx, 4) +
                            std::string((const char*)&ecx, 4);
    return vendor_id == "GenuineIntel";
}

inline void check_cpuid() {
    if (!intel_enabled()) {
        fprintf(
            stderr,
            "LVQ and Leanvec functionality of SVS is not supported on "
            "non-Intel hardware.\n"
        );
        exit(1);
    }
}

// This API can be used in downstream integration code
// to determine whether we should fallback to non LVQ/LeanVec solution.
inline bool lvq_leanvec_enabled() {
    return intel_enabled(); /* && avx_runtime_flags.is_avx2_supported(); */
}

} // namespace detail
} // namespace svs
