
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_stdstreams.hpp>

#include <catch2/catch_user_config.hpp>

#include <iostream>

namespace Catch {

// If you #define this you must implement these functions
#if !defined( CATCH_CONFIG_NOSTDOUT )
    std::ostream& cout() { return std::cout; }
    std::ostream& cerr() { return std::cerr; }
    std::ostream& clog() { return std::clog; }
#endif

} // namespace Catch
