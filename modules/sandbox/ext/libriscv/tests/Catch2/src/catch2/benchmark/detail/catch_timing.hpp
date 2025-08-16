
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
// Adapted from donated nonius code.

#ifndef CATCH_TIMING_HPP_INCLUDED
#define CATCH_TIMING_HPP_INCLUDED

#include <catch2/benchmark/catch_clock.hpp>
#include <catch2/benchmark/detail/catch_complete_invoke.hpp>

namespace Catch {
    namespace Benchmark {
        template <typename Result>
        struct Timing {
            IDuration elapsed;
            Result result;
            int iterations;
        };
        template <typename Func, typename... Args>
        using TimingOf = Timing<Detail::CompleteType_t<FunctionReturnType<Func, Args...>>>;
    } // namespace Benchmark
} // namespace Catch

#endif // CATCH_TIMING_HPP_INCLUDED
