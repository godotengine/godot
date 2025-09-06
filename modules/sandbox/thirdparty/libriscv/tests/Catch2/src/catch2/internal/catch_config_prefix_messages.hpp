
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/** \file
 * Wrapper for the CATCH_CONFIG_PREFIX_MESSAGES configuration option
 *
 * CATCH_CONFIG_PREFIX_ALL can be used to avoid clashes with other macros
 * by prepending CATCH_. This may not be desirable if the only clashes are with
 * logger macros such as INFO and WARN. In this cases
 * CATCH_CONFIG_PREFIX_MESSAGES can be used to only prefix a small subset
 * of relevant macros.
 *
 */

#ifndef CATCH_CONFIG_PREFIX_MESSAGES_HPP_INCLUDED
#define CATCH_CONFIG_PREFIX_MESSAGES_HPP_INCLUDED

#include <catch2/catch_user_config.hpp>

#if defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_PREFIX_MESSAGES)
    #define CATCH_CONFIG_PREFIX_MESSAGES
#endif

#endif // CATCH_CONFIG_PREFIX_MESSAGES_HPP_INCLUDED
