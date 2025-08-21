
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_DEPRECATION_MACRO_HPP_INCLUDED
#define CATCH_DEPRECATION_MACRO_HPP_INCLUDED

#include <catch2/catch_user_config.hpp>

#if !defined( CATCH_CONFIG_NO_DEPRECATION_ANNOTATIONS )
#    define DEPRECATED( msg ) [[deprecated( msg )]]
#else
#    define DEPRECATED( msg )
#endif

#endif // CATCH_DEPRECATION_MACRO_HPP_INCLUDED
