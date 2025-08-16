
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_GET_RANDOM_SEED_HPP_INCLUDED
#define CATCH_GET_RANDOM_SEED_HPP_INCLUDED

#include <cstdint>

namespace Catch {
    //! Returns Catch2's current RNG seed.
    std::uint32_t getSeed();
}

#endif // CATCH_GET_RANDOM_SEED_HPP_INCLUDED
