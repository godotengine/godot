
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/** \file
 * Wrapper for the STATIC_ANALYSIS_SUPPORT configuration option
 *
 * Some of Catch2's macros can be defined differently to work better with
 * static analysis tools, like clang-tidy or coverity.
 * Currently the main use case is to show that `SECTION`s are executed
 * exclusively, and not all in one run of a `TEST_CASE`.
 */

#ifndef CATCH_CONFIG_STATIC_ANALYSIS_SUPPORT_HPP_INCLUDED
#define CATCH_CONFIG_STATIC_ANALYSIS_SUPPORT_HPP_INCLUDED

#include <catch2/catch_user_config.hpp>

#if defined(__clang_analyzer__) || defined(__COVERITY__)
    #define CATCH_INTERNAL_CONFIG_STATIC_ANALYSIS_SUPPORT
#endif

#if defined( CATCH_INTERNAL_CONFIG_STATIC_ANALYSIS_SUPPORT ) && \
    !defined( CATCH_CONFIG_NO_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT ) && \
    !defined( CATCH_CONFIG_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT )
#    define CATCH_CONFIG_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT
#endif


#endif // CATCH_CONFIG_STATIC_ANALYSIS_SUPPORT_HPP_INCLUDED
