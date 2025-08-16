
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
// Adapted from donated nonius code.

#ifndef CATCH_SAMPLE_ANALYSIS_HPP_INCLUDED
#define CATCH_SAMPLE_ANALYSIS_HPP_INCLUDED

#include <catch2/benchmark/catch_estimate.hpp>
#include <catch2/benchmark/catch_outlier_classification.hpp>
#include <catch2/benchmark/catch_clock.hpp>

#include <vector>

namespace Catch {
    namespace Benchmark {
        struct SampleAnalysis {
            std::vector<FDuration> samples;
            Estimate<FDuration> mean;
            Estimate<FDuration> standard_deviation;
            OutlierClassification outliers;
            double outlier_variance;
        };
    } // namespace Benchmark
} // namespace Catch

#endif // CATCH_SAMPLE_ANALYSIS_HPP_INCLUDED
