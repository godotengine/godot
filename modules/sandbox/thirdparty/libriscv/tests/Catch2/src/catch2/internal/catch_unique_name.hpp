
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_UNIQUE_NAME_HPP_INCLUDED
#define CATCH_UNIQUE_NAME_HPP_INCLUDED

#include <catch2/internal/catch_config_counter.hpp>
#define INTERNAL_CATCH_UNIQUE_NAME_LINE2( name, line ) name##line
#define INTERNAL_CATCH_UNIQUE_NAME_LINE( name, line ) INTERNAL_CATCH_UNIQUE_NAME_LINE2( name, line )
#ifdef CATCH_CONFIG_COUNTER
#  define INTERNAL_CATCH_UNIQUE_NAME( name ) INTERNAL_CATCH_UNIQUE_NAME_LINE( name, __COUNTER__ )
#else
#  define INTERNAL_CATCH_UNIQUE_NAME( name ) INTERNAL_CATCH_UNIQUE_NAME_LINE( name, __LINE__ )
#endif

#endif // CATCH_UNIQUE_NAME_HPP_INCLUDED
