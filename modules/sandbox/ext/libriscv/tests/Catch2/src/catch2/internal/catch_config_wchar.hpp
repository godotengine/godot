
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/** \file
 * Wrapper for the WCHAR configuration option
 *
 * We want to support platforms that do not provide `wchar_t`, so we
 * sometimes have to disable providing wchar_t overloads through Catch2,
 * e.g. the StringMaker specialization for `std::wstring`.
 */

#ifndef CATCH_CONFIG_WCHAR_HPP_INCLUDED
#define CATCH_CONFIG_WCHAR_HPP_INCLUDED

#include <catch2/catch_user_config.hpp>

// We assume that WCHAR should be enabled by default, and only disabled
// for a shortlist (so far only DJGPP) of compilers.

#if defined(__DJGPP__)
#  define CATCH_INTERNAL_CONFIG_NO_WCHAR
#endif // __DJGPP__

#if !defined( CATCH_INTERNAL_CONFIG_NO_WCHAR ) && \
    !defined( CATCH_CONFIG_NO_WCHAR ) && \
    !defined( CATCH_CONFIG_WCHAR )
#    define CATCH_CONFIG_WCHAR
#endif

#endif // CATCH_CONFIG_WCHAR_HPP_INCLUDED
