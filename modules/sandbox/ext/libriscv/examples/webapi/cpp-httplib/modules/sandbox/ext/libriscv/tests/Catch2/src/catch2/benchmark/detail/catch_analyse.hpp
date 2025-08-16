
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
// Adapted from donated nonius code.

#ifndef CATCH_ANALYSE_HPP_INCLUDED
#define CATCH_ANALYSE_HPP_INCLUDED

#include <catch2/benchmark/catch_clock.hpp>
#include <catch2/benchmark/catch_sample_analysis.hpp>


namespace Catch {
    class IConfig;

    namespace Benchmark {
        namespace Detail {
            SampleAnalysis analyse(const IConfig &cfg, FDuration* first, FDuration* last);
        } // namespace Detail
    } // namespace Benchmark
} // namespace Catch

#endif // CATCH_ANALYSE_HPP_INCLUDED
