
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/** \file
 * Wrapper for ANDROID_LOGWRITE configuration option
 *
 * We want to default to enabling it when compiled for android, but
 * users of the library should also be able to disable it if they want
 * to.
 */

#ifndef CATCH_CONFIG_ANDROID_LOGWRITE_HPP_INCLUDED
#define CATCH_CONFIG_ANDROID_LOGWRITE_HPP_INCLUDED

#include <catch2/catch_user_config.hpp>

#if defined(__ANDROID__)
#    define CATCH_INTERNAL_CONFIG_ANDROID_LOGWRITE
#endif


#if defined( CATCH_INTERNAL_CONFIG_ANDROID_LOGWRITE ) && \
    !defined( CATCH_CONFIG_NO_ANDROID_LOGWRITE ) &&      \
    !defined( CATCH_CONFIG_ANDROID_LOGWRITE )
#    define CATCH_CONFIG_ANDROID_LOGWRITE
#endif

#endif // CATCH_CONFIG_ANDROID_LOGWRITE_HPP_INCLUDED
