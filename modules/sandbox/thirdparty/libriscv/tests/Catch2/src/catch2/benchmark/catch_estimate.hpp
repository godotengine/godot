
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
// Adapted from donated nonius code.

#ifndef CATCH_ESTIMATE_HPP_INCLUDED
#define CATCH_ESTIMATE_HPP_INCLUDED

namespace Catch {
    namespace Benchmark {
        template <typename Type>
        struct Estimate {
            Type point;
            Type lower_bound;
            Type upper_bound;
            double confidence_interval;
        };
    } // namespace Benchmark
} // namespace Catch

#endif // CATCH_ESTIMATE_HPP_INCLUDED
