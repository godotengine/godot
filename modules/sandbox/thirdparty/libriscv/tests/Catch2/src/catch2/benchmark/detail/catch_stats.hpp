
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
// Adapted from donated nonius code.

#ifndef CATCH_STATS_HPP_INCLUDED
#define CATCH_STATS_HPP_INCLUDED

#include <catch2/benchmark/catch_estimate.hpp>
#include <catch2/benchmark/catch_outlier_classification.hpp>

#include <vector>

namespace Catch {
    namespace Benchmark {
        namespace Detail {
            using sample = std::vector<double>;

            double weighted_average_quantile( int k,
                                              int q,
                                              double* first,
                                              double* last );

            OutlierClassification
            classify_outliers( double const* first, double const* last );

            double mean( double const* first, double const* last );

            double normal_cdf( double x );

            double erfc_inv(double x);

            double normal_quantile(double p);

            Estimate<double>
            bootstrap( double confidence_level,
                       double* first,
                       double* last,
                       sample const& resample,
                       double ( *estimator )( double const*, double const* ) );

            struct bootstrap_analysis {
                Estimate<double> mean;
                Estimate<double> standard_deviation;
                double outlier_variance;
            };

            bootstrap_analysis analyse_samples(double confidence_level,
                                               unsigned int n_resamples,
                                               double* first,
                                               double* last);
        } // namespace Detail
    } // namespace Benchmark
} // namespace Catch

#endif // CATCH_STATS_HPP_INCLUDED
