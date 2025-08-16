
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
// Adapted from donated nonius code.

#ifndef CATCH_BENCHMARK_HPP_INCLUDED
#define CATCH_BENCHMARK_HPP_INCLUDED

#include <catch2/catch_user_config.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>
#include <catch2/internal/catch_context.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_test_failure_exception.hpp>
#include <catch2/internal/catch_unique_name.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/interfaces/catch_interfaces_registry_hub.hpp>
#include <catch2/benchmark/detail/catch_benchmark_stats.hpp>
#include <catch2/benchmark/catch_clock.hpp>
#include <catch2/benchmark/catch_environment.hpp>
#include <catch2/benchmark/catch_execution_plan.hpp>
#include <catch2/benchmark/detail/catch_estimate_clock.hpp>
#include <catch2/benchmark/detail/catch_analyse.hpp>
#include <catch2/benchmark/detail/catch_benchmark_function.hpp>
#include <catch2/benchmark/detail/catch_run_for_at_least.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <string>
#include <cmath>

namespace Catch {
    namespace Benchmark {
        struct Benchmark {
            Benchmark(std::string&& benchmarkName)
                : name(CATCH_MOVE(benchmarkName)) {}

            template <class FUN>
            Benchmark(std::string&& benchmarkName , FUN &&func)
                : fun(CATCH_MOVE(func)), name(CATCH_MOVE(benchmarkName)) {}

            template <typename Clock>
            ExecutionPlan prepare(const IConfig &cfg, Environment env) {
                auto min_time = env.clock_resolution.mean * Detail::minimum_ticks;
                auto run_time = std::max(min_time, std::chrono::duration_cast<decltype(min_time)>(cfg.benchmarkWarmupTime()));
                auto&& test = Detail::run_for_at_least<Clock>(std::chrono::duration_cast<IDuration>(run_time), 1, fun);
                int new_iters = static_cast<int>(std::ceil(min_time * test.iterations / test.elapsed));
                return { new_iters, test.elapsed / test.iterations * new_iters * cfg.benchmarkSamples(), CATCH_MOVE(fun), std::chrono::duration_cast<FDuration>(cfg.benchmarkWarmupTime()), Detail::warmup_iterations };
            }

            template <typename Clock = default_clock>
            void run() {
                static_assert( Clock::is_steady,
                               "Benchmarking clock should be steady" );
                auto const* cfg = getCurrentContext().getConfig();

                auto env = Detail::measure_environment<Clock>();

                getResultCapture().benchmarkPreparing(name);
                CATCH_TRY{
                    auto plan = user_code([&] {
                        return prepare<Clock>(*cfg, env);
                    });

                    BenchmarkInfo info {
                        CATCH_MOVE(name),
                        plan.estimated_duration.count(),
                        plan.iterations_per_sample,
                        cfg->benchmarkSamples(),
                        cfg->benchmarkResamples(),
                        env.clock_resolution.mean.count(),
                        env.clock_cost.mean.count()
                    };

                    getResultCapture().benchmarkStarting(info);

                    auto samples = user_code([&] {
                        return plan.template run<Clock>(*cfg, env);
                    });

                    auto analysis = Detail::analyse(*cfg, samples.data(), samples.data() + samples.size());
                    BenchmarkStats<> stats{ CATCH_MOVE(info), CATCH_MOVE(analysis.samples), analysis.mean, analysis.standard_deviation, analysis.outliers, analysis.outlier_variance };
                    getResultCapture().benchmarkEnded(stats);
                } CATCH_CATCH_ALL {
                    getResultCapture().benchmarkFailed(translateActiveException());
                    // We let the exception go further up so that the
                    // test case is marked as failed.
                    std::rethrow_exception(std::current_exception());
                }
            }

            // sets lambda to be used in fun *and* executes benchmark!
            template <typename Fun, std::enable_if_t<!Detail::is_related_v<Fun, Benchmark>, int> = 0>
                Benchmark & operator=(Fun func) {
                auto const* cfg = getCurrentContext().getConfig();
                if (!cfg->skipBenchmarks()) {
                    fun = Detail::BenchmarkFunction(func);
                    run();
                }
                return *this;
            }

            explicit operator bool() {
                return true;
            }

        private:
            Detail::BenchmarkFunction fun;
            std::string name;
        };
    }
} // namespace Catch

#define INTERNAL_CATCH_GET_1_ARG(arg1, arg2, ...) arg1
#define INTERNAL_CATCH_GET_2_ARG(arg1, arg2, ...) arg2

#define INTERNAL_CATCH_BENCHMARK(BenchmarkName, name, benchmarkIndex)\
    if( Catch::Benchmark::Benchmark BenchmarkName{name} ) \
        BenchmarkName = [&](int benchmarkIndex)

#define INTERNAL_CATCH_BENCHMARK_ADVANCED(BenchmarkName, name)\
    if( Catch::Benchmark::Benchmark BenchmarkName{name} ) \
        BenchmarkName = [&]

#if defined(CATCH_CONFIG_PREFIX_ALL)

#define CATCH_BENCHMARK(...) \
    INTERNAL_CATCH_BENCHMARK(INTERNAL_CATCH_UNIQUE_NAME(CATCH2_INTERNAL_BENCHMARK_), INTERNAL_CATCH_GET_1_ARG(__VA_ARGS__,,), INTERNAL_CATCH_GET_2_ARG(__VA_ARGS__,,))
#define CATCH_BENCHMARK_ADVANCED(name) \
    INTERNAL_CATCH_BENCHMARK_ADVANCED(INTERNAL_CATCH_UNIQUE_NAME(CATCH2_INTERNAL_BENCHMARK_), name)

#else

#define BENCHMARK(...) \
    INTERNAL_CATCH_BENCHMARK(INTERNAL_CATCH_UNIQUE_NAME(CATCH2_INTERNAL_BENCHMARK_), INTERNAL_CATCH_GET_1_ARG(__VA_ARGS__,,), INTERNAL_CATCH_GET_2_ARG(__VA_ARGS__,,))
#define BENCHMARK_ADVANCED(name) \
    INTERNAL_CATCH_BENCHMARK_ADVANCED(INTERNAL_CATCH_UNIQUE_NAME(CATCH2_INTERNAL_BENCHMARK_), name)

#endif

#endif // CATCH_BENCHMARK_HPP_INCLUDED
