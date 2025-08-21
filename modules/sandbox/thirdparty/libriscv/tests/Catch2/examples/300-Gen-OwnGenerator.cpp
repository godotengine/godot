
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

// 300-Gen-OwnGenerator.cpp
// Shows how to define a custom generator.

// Specifically we will implement a random number generator for integers
// It will have infinite capacity and settable lower/upper bound

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include <random>

namespace {

// This class shows how to implement a simple generator for Catch tests
class RandomIntGenerator final : public Catch::Generators::IGenerator<int> {
    std::minstd_rand m_rand;
    std::uniform_int_distribution<> m_dist;
    int current_number;
public:

    RandomIntGenerator(int low, int high):
        m_rand(std::random_device{}()),
        m_dist(low, high)
    {
        static_cast<void>(next());
    }

    int const& get() const override;
    bool next() override {
        current_number = m_dist(m_rand);
        return true;
    }
};

// Avoids -Wweak-vtables
int const& RandomIntGenerator::get() const {
    return current_number;
}

// This helper function provides a nicer UX when instantiating the generator
// Notice that it returns an instance of GeneratorWrapper<int>, which
// is a value-wrapper around std::unique_ptr<IGenerator<int>>.
Catch::Generators::GeneratorWrapper<int> random(int low, int high) {
    return Catch::Generators::GeneratorWrapper<int>(
        new RandomIntGenerator(low, high)
        // Another possibility:
        // Catch::Detail::make_unique<RandomIntGenerator>(low, high)
    );
}

} // end anonymous namespaces

// The two sections in this test case are equivalent, but the first one
// is much more readable/nicer to use
TEST_CASE("Generating random ints", "[example][generator]") {
    SECTION("Nice UX") {
        auto i = GENERATE(take(100, random(-100, 100)));
        REQUIRE(i >= -100);
        REQUIRE(i <= 100);
    }
    SECTION("Creating the random generator directly") {
        auto i = GENERATE(take(100, GeneratorWrapper<int>(Catch::Detail::make_unique<RandomIntGenerator>(-100, 100))));
        REQUIRE(i >= -100);
        REQUIRE(i <= 100);
    }
}

// Compiling and running this file will result in 400 successful assertions
