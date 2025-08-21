
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_RANDOM_SEED_GENERATION_HPP_INCLUDED
#define CATCH_RANDOM_SEED_GENERATION_HPP_INCLUDED

#include <cstdint>

namespace Catch {

    enum class GenerateFrom {
        Time,
        RandomDevice,
        //! Currently equivalent to RandomDevice, but can change at any point
        Default
    };

    std::uint32_t generateRandomSeed(GenerateFrom from);

} // end namespace Catch

#endif // CATCH_RANDOM_SEED_GENERATION_HPP_INCLUDED
