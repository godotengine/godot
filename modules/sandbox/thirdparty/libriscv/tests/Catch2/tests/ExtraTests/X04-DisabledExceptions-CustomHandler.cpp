
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>

#include <iostream>

namespace Catch {
    [[noreturn]]
    void throw_exception(std::exception const& e) {
        std::cerr << "====== CUSTOM HANDLER ====== run terminates because an exception was thrown.\n"
                  << "The message was: " << e.what() << '\n';
        // Avoid abort and other exceptional exits -- there is no way
        // to tell CMake that abort is the desired outcome of a test.
        exit(1);
    }
}

TEST_CASE("Tests that run") {
    // All of these should be run and be reported
    CHECK(1 == 2);
    CHECK(1 == 1);
    CHECK(1 != 3);
    CHECK(1 == 4);
}



TEST_CASE("Tests that abort") {
    REQUIRE(1 == 1);
    REQUIRE(1 != 2);
    REQUIRE(1 == 3);
    // We should not get here, because the test above aborts
    REQUIRE(1 != 4);
}
