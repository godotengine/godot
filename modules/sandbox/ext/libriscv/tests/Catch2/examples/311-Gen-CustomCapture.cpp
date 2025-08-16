
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

// 311-Gen-CustomCapture.cpp
// Shows how to provide custom capture list to the generator expression

// Note that using variables inside generators is dangerous and should
// be done only if you know what you are doing, because the generators
// _WILL_ outlive the variables. Also, even if you know what you are
// doing, you should probably use GENERATE_COPY or GENERATE_REF macros
// instead. However, if your use case requires having a
// per-variable custom capture list, this example shows how to achieve
// that.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

TEST_CASE("Generate random doubles across different ranges",
          "[generator][example][advanced]") {
    // Workaround for old libstdc++
    using record = std::tuple<double, double>;
    // Set up 3 ranges to generate numbers from
    auto r1 = GENERATE(table<double, double>({
        record{3, 4},
        record{-4, -3},
        record{10, 1000}
    }));

    auto r2(r1);

    // This will take r1 by reference and r2 by value.
    // Note that there are no advantages for doing so in this example,
    // it is done only for expository purposes.
    auto number = Catch::Generators::generate( "custom capture generator", CATCH_INTERNAL_LINEINFO,
        [&r1, r2]{
            using namespace Catch::Generators;
            return makeGenerators(take(50, random(std::get<0>(r1), std::get<1>(r2))));
        }
    );

    REQUIRE(std::abs(number) > 0);
}

// Compiling and running this file will result in 150 successful assertions

