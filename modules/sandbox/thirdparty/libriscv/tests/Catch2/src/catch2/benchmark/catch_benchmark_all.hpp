
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
/** \file
 * This is a convenience header for Catch2's benchmarking. It includes
 * **all** of Catch2 headers related to benchmarking.
 *
 * Generally the Catch2 users should use specific includes they need,
 * but this header can be used instead for ease-of-experimentation, or
 * just plain convenience, at the cost of (significantly) increased
 * compilation times.
 *
 * When a new header is added to either the `benchmark` folder, or to
 * the corresponding internal (detail) subfolder, it should be added here.
 */

#ifndef CATCH_BENCHMARK_ALL_HPP_INCLUDED
#define CATCH_BENCHMARK_ALL_HPP_INCLUDED

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_chronometer.hpp>
#include <catch2/benchmark/catch_clock.hpp>
#include <catch2/benchmark/catch_constructor.hpp>
#include <catch2/benchmark/catch_environment.hpp>
#include <catch2/benchmark/catch_estimate.hpp>
#include <catch2/benchmark/catch_execution_plan.hpp>
#include <catch2/benchmark/catch_optimizer.hpp>
#include <catch2/benchmark/catch_outlier_classification.hpp>
#include <catch2/benchmark/catch_sample_analysis.hpp>
#include <catch2/benchmark/detail/catch_analyse.hpp>
#include <catch2/benchmark/detail/catch_benchmark_function.hpp>
#include <catch2/benchmark/detail/catch_benchmark_stats.hpp>
#include <catch2/benchmark/detail/catch_benchmark_stats_fwd.hpp>
#include <catch2/benchmark/detail/catch_complete_invoke.hpp>
#include <catch2/benchmark/detail/catch_estimate_clock.hpp>
#include <catch2/benchmark/detail/catch_measure.hpp>
#include <catch2/benchmark/detail/catch_repeat.hpp>
#include <catch2/benchmark/detail/catch_run_for_at_least.hpp>
#include <catch2/benchmark/detail/catch_stats.hpp>
#include <catch2/benchmark/detail/catch_timing.hpp>

#endif // CATCH_BENCHMARK_ALL_HPP_INCLUDED
