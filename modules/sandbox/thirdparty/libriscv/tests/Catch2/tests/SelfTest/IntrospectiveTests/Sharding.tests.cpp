
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <catch2/internal/catch_sharding.hpp>

#include <unordered_map>
#include <vector>

TEST_CASE("Sharding Function", "[approvals]") {
    std::vector<int> testContainer = { 0, 1, 2, 3, 4, 5, 6 };
    std::unordered_map<int, std::vector<std::size_t>> expectedShardSizes = {
        {1, {7}},
        {2, {4, 3}},
        {3, {3, 2, 2}},
        {4, {2, 2, 2, 1}},
        {5, {2, 2, 1, 1, 1}},
        {6, {2, 1, 1, 1, 1, 1}},
        {7, {1, 1, 1, 1, 1, 1, 1}},
    };

    auto shardCount = GENERATE(range(1, 7));
    auto shardIndex = GENERATE_COPY(filter([=](int i) { return i < shardCount; }, range(0, 6)));

    std::vector<int> result = Catch::createShard(testContainer, shardCount, shardIndex);

    auto& sizes = expectedShardSizes[shardCount];
    REQUIRE(result.size() == sizes[shardIndex]);

    std::size_t startIndex = 0;
    for(int i = 0; i < shardIndex; i++) {
        startIndex += sizes[i];
    }

    for(std::size_t i = 0; i < sizes[shardIndex]; i++) {
        CHECK(result[i] == testContainer[i + startIndex]);
    }
}
