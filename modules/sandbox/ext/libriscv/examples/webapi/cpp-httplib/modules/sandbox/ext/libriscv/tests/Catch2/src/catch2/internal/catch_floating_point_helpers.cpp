
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_floating_point_helpers.hpp>

#include <cstring>

namespace Catch {
    namespace Detail {

        uint32_t convertToBits(float f) {
            static_assert(sizeof(float) == sizeof(uint32_t), "Important ULP matcher assumption violated");
            uint32_t i;
            std::memcpy(&i, &f, sizeof(f));
            return i;
        }

        uint64_t convertToBits(double d) {
            static_assert(sizeof(double) == sizeof(uint64_t), "Important ULP matcher assumption violated");
            uint64_t i;
            std::memcpy(&i, &d, sizeof(d));
            return i;
        }

#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        bool directCompare( float lhs, float rhs ) { return lhs == rhs; }
        bool directCompare( double lhs, double rhs ) { return lhs == rhs; }
#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic pop
#endif


    } // end namespace Detail
} // end namespace Catch

