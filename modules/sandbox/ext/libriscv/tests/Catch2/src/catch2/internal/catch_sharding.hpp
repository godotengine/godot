
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_SHARDING_HPP_INCLUDED
#define CATCH_SHARDING_HPP_INCLUDED

#include <cassert>
#include <algorithm>

namespace Catch {

    template<typename Container>
    Container createShard(Container const& container, std::size_t const shardCount, std::size_t const shardIndex) {
        assert(shardCount > shardIndex);

        if (shardCount == 1) {
            return container;
        }

        const std::size_t totalTestCount = container.size();

        const std::size_t shardSize = totalTestCount / shardCount;
        const std::size_t leftoverTests = totalTestCount % shardCount;

        const std::size_t startIndex = shardIndex * shardSize + (std::min)(shardIndex, leftoverTests);
        const std::size_t endIndex = (shardIndex + 1) * shardSize + (std::min)(shardIndex + 1, leftoverTests);

        auto startIterator = std::next(container.begin(), static_cast<std::ptrdiff_t>(startIndex));
        auto endIterator = std::next(container.begin(), static_cast<std::ptrdiff_t>(endIndex));

        return Container(startIterator, endIterator);
    }

}

#endif // CATCH_SHARDING_HPP_INCLUDED
