
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

// 310-Gen-VariablesInGenerator.cpp
// Shows how to use variables when creating generators.

// Note that using variables inside generators is dangerous and should
// be done only if you know what you are doing, because the generators
// _WILL_ outlive the variables -- thus they should be either captured
// by value directly, or copied by the generators during construction.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

TEST_CASE("Generate random doubles across different ranges",
          "[generator][example][advanced]") {
    // Workaround for old libstdc++
    using record = std::tuple<double, double>;
    // Set up 3 ranges to generate numbers from
    auto r = GENERATE(table<double, double>({
        record{3, 4},
        record{-4, -3},
        record{10, 1000}
    }));

    // This will not compile (intentionally), because it accesses a variable
    // auto number = GENERATE(take(50, random(std::get<0>(r), std::get<1>(r))));

   // GENERATE_COPY copies all variables mentioned inside the expression
   // thus this will work.
    auto number = GENERATE_COPY(take(50, random(std::get<0>(r), std::get<1>(r))));

    REQUIRE(std::abs(number) > 0);
}

// Compiling and running this file will result in 150 successful assertions

