
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_CONSOLE_WIDTH_HPP_INCLUDED
#define CATCH_CONSOLE_WIDTH_HPP_INCLUDED

// This include must be kept so that user's configured value for CONSOLE_WIDTH
// is used before we attempt to provide a default value
#include <catch2/catch_user_config.hpp>

#ifndef CATCH_CONFIG_CONSOLE_WIDTH
#define CATCH_CONFIG_CONSOLE_WIDTH 80
#endif

#endif // CATCH_CONSOLE_WIDTH_HPP_INCLUDED
