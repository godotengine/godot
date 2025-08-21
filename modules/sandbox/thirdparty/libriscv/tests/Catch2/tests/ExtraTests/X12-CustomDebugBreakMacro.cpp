
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that user-defined `CATCH_BREAK_INTO_DEBUGGER` is respected and used.
 */

#include <iostream>

void custom_debug_break() {
    std::cerr << "Pretty please, break into debugger\n";
}

#define CATCH_BREAK_INTO_DEBUGGER() custom_debug_break()

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

TEST_CASE("Failing test that breaks into debugger", "[macros]") {
    REQUIRE(1 == 2);
}
