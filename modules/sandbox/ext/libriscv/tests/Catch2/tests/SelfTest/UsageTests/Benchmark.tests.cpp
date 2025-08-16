
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
// Adapted from donated nonius code.

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_constructor.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <map>

namespace {
    std::uint64_t Fibonacci(std::uint64_t number) {
        return number < 2 ? number : Fibonacci(number - 1) + Fibonacci(number - 2);
    }
}

TEST_CASE("Benchmark Fibonacci", "[!benchmark]") {
    CHECK(Fibonacci(0) == 0);
    // some more asserts..
    CHECK(Fibonacci(5) == 5);
    // some more asserts..

    REQUIRE( Fibonacci( 20 ) == 6'765 );
    BENCHMARK( "Fibonacci 20" ) {
        return Fibonacci(20);
    };

    REQUIRE( Fibonacci( 25 ) == 75'025 );
    BENCHMARK( "Fibonacci 25" ) {
        return Fibonacci(25);
    };

    BENCHMARK("Fibonacci 30") {
        return Fibonacci(30);
    };

    BENCHMARK("Fibonacci 35") {
        return Fibonacci(35);
    };
}

TEST_CASE("Benchmark containers", "[!benchmark]") {
    static const int size = 100;

    std::vector<int> v;
    std::map<int, int> m;

    SECTION("without generator") {
        BENCHMARK("Load up a vector") {
            v = std::vector<int>();
            for (int i = 0; i < size; ++i)
                v.push_back(i);
        };
        REQUIRE(v.size() == size);

        // test optimizer control
        BENCHMARK("Add up a vector's content") {
            uint64_t add = 0;
            for (int i = 0; i < size; ++i)
                add += v[i];
            return add;
        };

        BENCHMARK("Load up a map") {
            m = std::map<int, int>();
            for (int i = 0; i < size; ++i)
                m.insert({ i, i + 1 });
        };
        REQUIRE(m.size() == size);

        BENCHMARK("Reserved vector") {
            v = std::vector<int>();
            v.reserve(size);
            for (int i = 0; i < size; ++i)
                v.push_back(i);
        };
        REQUIRE(v.size() == size);

        BENCHMARK("Resized vector") {
            v = std::vector<int>();
            v.resize(size);
            for (int i = 0; i < size; ++i)
                v[i] = i;
        };
        REQUIRE(v.size() == size);

        int array[size] {};
        BENCHMARK("A fixed size array that should require no allocations") {
            for (int i = 0; i < size; ++i)
                array[i] = i;
        };
        int sum = 0;
        for (int val : array)
            sum += val;
        REQUIRE(sum > size);

        SECTION("XYZ") {

            BENCHMARK_ADVANCED("Load up vector with chronometer")(Catch::Benchmark::Chronometer meter) {
                std::vector<int> k;
                meter.measure([&](int idx) {
                    k = std::vector<int>();
                    for (int i = 0; i < size; ++i)
                        k.push_back(idx);
                });
                REQUIRE(k.size() == size);
            };

            int runs = 0;
            BENCHMARK("Fill vector indexed", benchmarkIndex) {
                v = std::vector<int>();
                v.resize(size);
                for (int i = 0; i < size; ++i)
                    v[i] = benchmarkIndex;
                runs = benchmarkIndex;
            };

            for (int val : v) {
                REQUIRE(val == runs);
            }
        }
    }

    SECTION("with generator") {
        auto generated = GENERATE(range(0, 10));
        BENCHMARK("Fill vector generated") {
            v = std::vector<int>();
            v.resize(size);
            for (int i = 0; i < size; ++i)
                v[i] = generated;
        };
        for (int val : v) {
            REQUIRE(val == generated);
        }
    }

    SECTION("construct and destroy example") {
        BENCHMARK_ADVANCED("construct")(Catch::Benchmark::Chronometer meter) {
            std::vector<Catch::Benchmark::storage_for<std::string>> storage(meter.runs());
            meter.measure([&](int i) { storage[i].construct("thing"); });
        };

        BENCHMARK_ADVANCED("destroy")(Catch::Benchmark::Chronometer meter) {
            std::vector<Catch::Benchmark::destructable_object<std::string>> storage(meter.runs());
            for(auto&& o : storage)
                o.construct("thing");
            meter.measure([&](int i) { storage[i].destruct(); });
        };
    }
}

TEST_CASE("Skip benchmark macros", "[!benchmark]") {
    std::vector<int> v;
    BENCHMARK("fill vector") {
        v.emplace_back(1);
        v.emplace_back(2);
        v.emplace_back(3);
    };
    REQUIRE(v.size() == 0);

    std::size_t counter{0};
    BENCHMARK_ADVANCED("construct vector")(Catch::Benchmark::Chronometer meter) {
        std::vector<Catch::Benchmark::storage_for<std::string>> storage(meter.runs());
        meter.measure([&](int i) { storage[i].construct("thing"); counter++; });
    };
    REQUIRE(counter == 0);
}
