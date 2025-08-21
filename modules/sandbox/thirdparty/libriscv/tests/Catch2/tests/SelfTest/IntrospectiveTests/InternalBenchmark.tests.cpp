
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
// Adapted from donated nonius code.


#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif


#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_config.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_chronometer.hpp>
#include <catch2/benchmark/detail/catch_analyse.hpp>
#include <catch2/benchmark/detail/catch_benchmark_function.hpp>
#include <catch2/benchmark/detail/catch_estimate_clock.hpp>

#include <numeric>

namespace {
    struct manual_clock {
    public:
        using duration = std::chrono::nanoseconds;
        using time_point = std::chrono::time_point<manual_clock, duration>;
        using rep = duration::rep;
        using period = duration::period;
        enum { is_steady = true };

        static time_point now() {
            return time_point(duration(tick()));
        }

        static void advance(int ticks = 1) {
            tick() += ticks;
        }

    private:
        static rep& tick() {
            static rep the_tick = 0;
            return the_tick;
        }
    };

    struct counting_clock {
    public:
        using duration = std::chrono::nanoseconds;
        using time_point = std::chrono::time_point<counting_clock, duration>;
        using rep = duration::rep;
        using period = duration::period;
        enum { is_steady = true };

        static time_point now() {
            static rep ticks = 0;
            return time_point(duration(ticks += rate()));
        }

        static void set_rate(rep new_rate) { rate() = new_rate; }

    private:
        static rep& rate() {
            static rep the_rate = 1;
            return the_rate;
        }
    };

    struct TestChronometerModel : Catch::Benchmark::Detail::ChronometerConcept {
        int started = 0;
        int finished = 0;

        void start() override { ++started; }
        void finish() override { ++finished; }
    };
} // namespace

TEST_CASE("warmup", "[benchmark]") {
    auto rate = 1000;
    counting_clock::set_rate(rate);

    auto start = counting_clock::now();
    auto iterations = Catch::Benchmark::Detail::warmup<counting_clock>();
    auto end = counting_clock::now();

    REQUIRE((iterations * rate) > Catch::Benchmark::Detail::warmup_time.count());
    REQUIRE((end - start) > Catch::Benchmark::Detail::warmup_time);
}

TEST_CASE("resolution", "[benchmark]") {
    auto rate = 1000;
    counting_clock::set_rate(rate);

    size_t count = 10;
    auto res = Catch::Benchmark::Detail::resolution<counting_clock>(static_cast<int>(count));

    REQUIRE(res.size() == count);

    for (size_t i = 1; i < count; ++i) {
        REQUIRE(res[i] == rate);
    }
}

TEST_CASE("estimate_clock_resolution", "[benchmark]") {
    auto rate = 2'000;
    counting_clock::set_rate(rate);

    int iters = 160'000;
    auto res = Catch::Benchmark::Detail::estimate_clock_resolution<counting_clock>(iters);

    REQUIRE(res.mean.count() == rate);
    REQUIRE(res.outliers.total() == 0);
}

TEST_CASE("benchmark function call", "[benchmark]") {
    SECTION("without chronometer") {
        auto called = 0;
        auto model = TestChronometerModel{};
        auto meter = Catch::Benchmark::Chronometer{ model, 1 };
        auto fn = Catch::Benchmark::Detail::BenchmarkFunction{ [&] {
                CHECK(model.started == 1);
                CHECK(model.finished == 0);
                ++called;
            } };

        fn(meter);

        CHECK(model.started == 1);
        CHECK(model.finished == 1);
        CHECK(called == 1);
    }

    SECTION("with chronometer") {
        auto called = 0;
        auto model = TestChronometerModel{};
        auto meter = Catch::Benchmark::Chronometer{ model, 1 };
        auto fn = Catch::Benchmark::Detail::BenchmarkFunction{ [&](Catch::Benchmark::Chronometer) {
                CHECK(model.started == 0);
                CHECK(model.finished == 0);
                ++called;
            } };

        fn(meter);

        CHECK(model.started == 0);
        CHECK(model.finished == 0);
        CHECK(called == 1);
    }
}

TEST_CASE("uniform samples", "[benchmark]") {
    std::vector<double> samples(100);
    std::fill(samples.begin(), samples.end(), 23);

    auto e = Catch::Benchmark::Detail::bootstrap(
        0.95,
        samples.data(),
        samples.data() + samples.size(),
        samples,
        []( double const* a, double const* b ) {
        auto sum = std::accumulate(a, b, 0.);
        return sum / (b - a);
    });
    CHECK(e.point == 23);
    CHECK(e.upper_bound == 23);
    CHECK(e.lower_bound == 23);
    CHECK(e.confidence_interval == 0.95);
}


TEST_CASE("normal_cdf", "[benchmark][approvals]") {
    using Catch::Benchmark::Detail::normal_cdf;
    using Catch::Approx;
    CHECK(normal_cdf(0.000000) == Approx(0.50000000000000000));
    CHECK(normal_cdf(1.000000) == Approx(0.84134474606854293));
    CHECK(normal_cdf(-1.000000) == Approx(0.15865525393145705));
    CHECK(normal_cdf(2.809729) == Approx(0.99752083845315409));
    CHECK(normal_cdf(-1.352570) == Approx(0.08809652095066035));
}

TEST_CASE("erfc_inv", "[benchmark]") {
    using Catch::Benchmark::Detail::erfc_inv;
    using Catch::Approx;
    CHECK(erfc_inv(1.103560) == Approx(-0.09203687623843015));
    CHECK(erfc_inv(1.067400) == Approx(-0.05980291115763361));
    CHECK(erfc_inv(0.050000) == Approx(1.38590382434967796));
}

TEST_CASE("normal_quantile", "[benchmark]") {
    using Catch::Benchmark::Detail::normal_quantile;
    using Catch::Approx;
    CHECK(normal_quantile(0.551780) == Approx(0.13015979861484198));
    CHECK(normal_quantile(0.533700) == Approx(0.08457408802851875));
    CHECK(normal_quantile(0.025000) == Approx(-1.95996398454005449));
}


TEST_CASE("mean", "[benchmark]") {
    std::vector<double> x{ 10., 20., 14., 16., 30., 24. };

    auto m = Catch::Benchmark::Detail::mean(x.data(), x.data() + x.size());

    REQUIRE(m == 19.);
}

TEST_CASE("weighted_average_quantile", "[benchmark]") {
    std::vector<double> x{ 10., 20., 14., 16., 30., 24. };

    auto q1 = Catch::Benchmark::Detail::weighted_average_quantile(1, 4, x.data(), x.data() + x.size());
    auto med = Catch::Benchmark::Detail::weighted_average_quantile(1, 2, x.data(), x.data() + x.size());
    auto q3 = Catch::Benchmark::Detail::weighted_average_quantile(3, 4, x.data(), x.data() + x.size());

    REQUIRE(q1 == 14.5);
    REQUIRE(med == 18.);
    REQUIRE(q3 == 23.);
}

TEST_CASE("classify_outliers", "[benchmark]") {
    auto require_outliers = [](Catch::Benchmark::OutlierClassification o, int los, int lom, int him, int his) {
        REQUIRE(o.low_severe == los);
        REQUIRE(o.low_mild == lom);
        REQUIRE(o.high_mild == him);
        REQUIRE(o.high_severe == his);
        REQUIRE(o.total() == los + lom + him + his);
    };

    SECTION("none") {
        std::vector<double> x{ 10., 20., 14., 16., 30., 24. };

        auto o = Catch::Benchmark::Detail::classify_outliers(
            x.data(), x.data() + x.size() );

        REQUIRE(o.samples_seen == static_cast<int>(x.size()));
        require_outliers(o, 0, 0, 0, 0);
    }
    SECTION("low severe") {
        std::vector<double> x{ -12., 20., 14., 16., 30., 24. };

        auto o = Catch::Benchmark::Detail::classify_outliers(
            x.data(), x.data() + x.size() );

        REQUIRE(o.samples_seen == static_cast<int>(x.size()));
        require_outliers(o, 1, 0, 0, 0);
    }
    SECTION("low mild") {
        std::vector<double> x{ 1., 20., 14., 16., 30., 24. };

        auto o = Catch::Benchmark::Detail::classify_outliers(
            x.data(), x.data() + x.size() );

        REQUIRE(o.samples_seen == static_cast<int>(x.size()));
        require_outliers(o, 0, 1, 0, 0);
    }
    SECTION("high mild") {
        std::vector<double> x{ 10., 20., 14., 16., 36., 24. };

        auto o = Catch::Benchmark::Detail::classify_outliers(
            x.data(), x.data() + x.size() );

        REQUIRE(o.samples_seen == static_cast<int>(x.size()));
        require_outliers(o, 0, 0, 1, 0);
    }
    SECTION("high severe") {
        std::vector<double> x{ 10., 20., 14., 16., 49., 24. };

        auto o = Catch::Benchmark::Detail::classify_outliers(
            x.data(), x.data() + x.size() );

        REQUIRE(o.samples_seen == static_cast<int>(x.size()));
        require_outliers(o, 0, 0, 0, 1);
    }
    SECTION("mixed") {
        std::vector<double> x{ -20., 20., 14., 16., 39., 24. };

        auto o = Catch::Benchmark::Detail::classify_outliers(
            x.data(), x.data() + x.size() );

        REQUIRE(o.samples_seen == static_cast<int>(x.size()));
        require_outliers(o, 1, 0, 1, 0);
    }
}

TEST_CASE("analyse", "[approvals][benchmark]") {
    Catch::ConfigData data{};
    data.benchmarkConfidenceInterval = 0.95;
    data.benchmarkNoAnalysis = false;
    data.benchmarkResamples = 1000;
    data.benchmarkSamples = 99;
    Catch::Config config{data};

    using FDuration = Catch::Benchmark::FDuration;
    std::vector<FDuration> samples(99);
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = FDuration(23 + (i % 3 - 1));
    }

    auto analysis = Catch::Benchmark::Detail::analyse(config, samples.data(), samples.data() + samples.size());
    CHECK( analysis.mean.point.count() == 23 );
    CHECK( analysis.mean.lower_bound.count() < 23 );
    CHECK(analysis.mean.lower_bound.count() > 22);
    CHECK(analysis.mean.upper_bound.count() > 23);
    CHECK(analysis.mean.upper_bound.count() < 24);

    CHECK(analysis.standard_deviation.point.count() > 0.5);
    CHECK(analysis.standard_deviation.point.count() < 1);
    CHECK(analysis.standard_deviation.lower_bound.count() > 0.5);
    CHECK(analysis.standard_deviation.lower_bound.count() < 1);
    CHECK(analysis.standard_deviation.upper_bound.count() > 0.5);
    CHECK(analysis.standard_deviation.upper_bound.count() < 1);

    CHECK(analysis.outliers.total() == 0);
    CHECK(analysis.outliers.low_mild == 0);
    CHECK(analysis.outliers.low_severe == 0);
    CHECK(analysis.outliers.high_mild == 0);
    CHECK(analysis.outliers.high_severe == 0);
    CHECK(analysis.outliers.samples_seen == static_cast<int>(samples.size()));

    CHECK(analysis.outlier_variance < 0.5);
    CHECK(analysis.outlier_variance > 0);
}

TEST_CASE("analyse no analysis", "[benchmark]") {
    Catch::ConfigData data{};
    data.benchmarkConfidenceInterval = 0.95;
    data.benchmarkNoAnalysis = true;
    data.benchmarkResamples = 1000;
    data.benchmarkSamples = 99;
    Catch::Config config{ data };

    using FDuration = Catch::Benchmark::FDuration;
    std::vector<FDuration> samples(99);
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = FDuration(23 + (i % 3 - 1));
    }

    auto analysis = Catch::Benchmark::Detail::analyse(config, samples.data(), samples.data() + samples.size());
    CHECK(analysis.mean.point.count() == 23);
    CHECK(analysis.mean.lower_bound.count() == 23);
    CHECK(analysis.mean.upper_bound.count() == 23);

    CHECK(analysis.standard_deviation.point.count() == 0);
    CHECK(analysis.standard_deviation.lower_bound.count() == 0);
    CHECK(analysis.standard_deviation.upper_bound.count() == 0);

    CHECK(analysis.outliers.total() == 0);
    CHECK(analysis.outliers.low_mild == 0);
    CHECK(analysis.outliers.low_severe == 0);
    CHECK(analysis.outliers.high_mild == 0);
    CHECK(analysis.outliers.high_severe == 0);
    CHECK(analysis.outliers.samples_seen == 0);

    CHECK(analysis.outlier_variance == 0);
}

TEST_CASE("run_for_at_least, int", "[benchmark]") {
    manual_clock::duration time(100);

    int old_x = 1;
    auto Timing = Catch::Benchmark::Detail::run_for_at_least<manual_clock>(time, 1, [&old_x](int x) -> int {
        CHECK(x >= old_x);
        manual_clock::advance(x);
        old_x = x;
        return x + 17;
    });

    REQUIRE(Timing.elapsed >= time);
    REQUIRE(Timing.result == Timing.iterations + 17);
    REQUIRE(Timing.iterations >= time.count());
}

TEST_CASE("run_for_at_least, chronometer", "[benchmark]") {
    manual_clock::duration time(100);

    int old_runs = 1;
    auto Timing = Catch::Benchmark::Detail::run_for_at_least<manual_clock>(time, 1, [&old_runs](Catch::Benchmark::Chronometer meter) -> int {
        CHECK(meter.runs() >= old_runs);
        manual_clock::advance(100);
        meter.measure([] {
            manual_clock::advance(1);
        });
        old_runs = meter.runs();
        return meter.runs() + 17;
    });

    REQUIRE(Timing.elapsed >= time);
    REQUIRE(Timing.result == Timing.iterations + 17);
    REQUIRE(Timing.iterations >= time.count());
}


TEST_CASE("measure", "[benchmark]") {
    auto r = Catch::Benchmark::Detail::measure<manual_clock>([](int x) -> int {
        CHECK(x == 17);
        manual_clock::advance(42);
        return 23;
    }, 17);
    auto s = Catch::Benchmark::Detail::measure<manual_clock>([](int x) -> int {
        CHECK(x == 23);
        manual_clock::advance(69);
        return 17;
    }, 23);

    CHECK(r.elapsed.count() == 42);
    CHECK(r.result == 23);
    CHECK(r.iterations == 1);

    CHECK(s.elapsed.count() == 69);
    CHECK(s.result == 17);
    CHECK(s.iterations == 1);
}

TEST_CASE("run benchmark", "[benchmark][approvals]") {
    counting_clock::set_rate(1000);
    auto start = counting_clock::now();

    Catch::Benchmark::Benchmark bench{ "Test Benchmark", [](Catch::Benchmark::Chronometer meter) {
        counting_clock::set_rate(100000);
        meter.measure([] { return counting_clock::now(); });
    } };

    bench.run<counting_clock>();
    auto end = counting_clock::now();

    CHECK((end - start).count() == 2867251000);
}

TEST_CASE("Failing benchmarks", "[!benchmark][.approvals]") {
    SECTION("empty", "Benchmark that has been optimized away (because it is empty)") {
        BENCHMARK("Empty benchmark") {};
    }
    SECTION("throw", "Benchmark that throws an exception") {
        BENCHMARK("Throwing benchmark") {
            throw "just a plain literal, bleh";
        };
    }
    SECTION("assert", "Benchmark that asserts inside") {
        BENCHMARK("Asserting benchmark") {
            REQUIRE(1 == 2);
        };
    }
    SECTION("fail", "Benchmark that fails inside") {
        BENCHMARK("FAIL'd benchmark") {
            FAIL("This benchmark only fails, nothing else");
        };
    }
}

TEST_CASE( "Failing benchmark respects should-fail",
           "[!shouldfail][!benchmark][approvals]" ) {
    BENCHMARK( "Asserting benchmark" ) { REQUIRE( 1 == 2 ); };
}
