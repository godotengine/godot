
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_PREPROCESSOR_INTERNAL_STRINGIFY_HPP_INCLUDED
#define CATCH_PREPROCESSOR_INTERNAL_STRINGIFY_HPP_INCLUDED

#include <catch2/catch_user_config.hpp>

#if !defined(CATCH_CONFIG_DISABLE_STRINGIFICATION)
  #define CATCH_INTERNAL_STRINGIFY(...) #__VA_ARGS__##_catch_sr
#else
  #define CATCH_INTERNAL_STRINGIFY(...) "Disabled by CATCH_CONFIG_DISABLE_STRINGIFICATION"_catch_sr
#endif

#endif // CATCH_PREPROCESSOR_INTERNAL_STRINGIFY_HPP_INCLUDED
