
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/** \file
 * Wrapper for UNCAUGHT_EXCEPTIONS configuration option
 *
 * For some functionality, Catch2 requires to know whether there is
 * an active exception. Because `std::uncaught_exception` is deprecated
 * in C++17, we want to use `std::uncaught_exceptions` if possible.
 */

#ifndef CATCH_CONFIG_UNCAUGHT_EXCEPTIONS_HPP_INCLUDED
#define CATCH_CONFIG_UNCAUGHT_EXCEPTIONS_HPP_INCLUDED

#include <catch2/catch_user_config.hpp>

#if defined(_MSC_VER)
#  if _MSC_VER >= 1900 // Visual Studio 2015 or newer
#    define CATCH_INTERNAL_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS
#  endif
#endif


#include <exception>

#if defined(__cpp_lib_uncaught_exceptions) \
    && !defined(CATCH_INTERNAL_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS)

#  define CATCH_INTERNAL_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS
#endif // __cpp_lib_uncaught_exceptions


#if defined(CATCH_INTERNAL_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS) \
    && !defined(CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS) \
    && !defined(CATCH_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS)

#  define CATCH_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS
#endif


#endif // CATCH_CONFIG_UNCAUGHT_EXCEPTIONS_HPP_INCLUDED
