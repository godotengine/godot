
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that the user can define custom fallbackStringifier 
 *
 * This is done by defining a custom fallback stringifier that prints
 * out a specific string, and then asserting (to cause stringification)
 * over a type without stringification support.
 */

#include <string>

// A catch-all stringifier
template <typename T>
std::string fallbackStringifier(T const&) {
    return "{ !!! }";
}

#include <catch2/catch_test_macros.hpp>

struct foo {
    explicit operator bool() const {
        return true;
    }
};

TEST_CASE("aa") {
    REQUIRE(foo{});
}
