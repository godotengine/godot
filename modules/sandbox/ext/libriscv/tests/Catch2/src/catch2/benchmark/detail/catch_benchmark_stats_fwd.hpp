
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_BENCHMARK_STATS_FWD_HPP_INCLUDED
#define CATCH_BENCHMARK_STATS_FWD_HPP_INCLUDED

#include <catch2/benchmark/catch_clock.hpp>

namespace Catch {

    // We cannot forward declare the type with default template argument
    // multiple times, so it is split out into a separate header so that
    // we can prevent multiple declarations in dependees
    template <typename Duration = Benchmark::FDuration>
    struct BenchmarkStats;

} // end namespace Catch

#endif // CATCH_BENCHMARK_STATS_FWD_HPP_INCLUDED
