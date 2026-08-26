//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Project Contributors
  SPDX-License-Identifier: BSL-1.0
*/
//==================================================================================================
#pragma once

#include <eve/arch.hpp>
#include <eve/forward.hpp>
#include <eve/detail/abi.hpp>
#include <eve/concept/value.hpp>

#include <eve/detail/function/simd/common/iota.hpp>

#if defined(EVE_INCLUDE_SVE_HEADER)
#  include <eve/detail/function/simd/arm/sve/iota.hpp>
#endif

namespace eve::detail
{
  template<typename T>
  EVE_FORCEINLINE constexpr auto linear_ramp(as<T> const& tgt) noexcept
  {
    return linear_ramp(EVE_RETARGET(eve::current_api_type), tgt);
  }
}
