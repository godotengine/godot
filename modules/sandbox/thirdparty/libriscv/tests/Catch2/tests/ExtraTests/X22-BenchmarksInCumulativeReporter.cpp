
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that the cumulative reporter base stores both assertions and
 * benchmarks, and stores them in the right order.
 *
 * This is done through a custom reporter that writes out the assertions
 * and benchmarks and checking that the output is in right order.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/reporters/catch_reporter_cumulative_base.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <iostream>
#include <utility>

class CumulativeBenchmarkReporter final : public Catch::CumulativeReporterBase {

public:
    CumulativeBenchmarkReporter(Catch::ReporterConfig&& _config) :
        CumulativeReporterBase(std::move(_config)) {
        m_preferences.shouldReportAllAssertions = true;
    }

    static std::string getDescription() {
        return "Custom reporter for testing cumulative reporter base";
    }

    void testRunEndedCumulative() override;
};

CATCH_REGISTER_REPORTER("testReporter", CumulativeBenchmarkReporter)

#include <chrono>
#include <thread>

TEST_CASE("Some assertions and benchmarks") {
    using namespace std::chrono_literals;

    REQUIRE(1);
    BENCHMARK("2") {
        std::this_thread::sleep_for(1ms);
    };
    REQUIRE(3);
    BENCHMARK("4") {
        std::this_thread::sleep_for(1ms);
    };
    REQUIRE(5);
}

void CumulativeBenchmarkReporter::testRunEndedCumulative() {
    auto const& testCases = m_testRun->children;
    assert(testCases.size() == 1);

    auto const& testCase = *testCases.front();
    auto const& sections = testCase.children;
    assert(sections.size() == 1);

    auto const& section = *sections.front();
    assert(section.childSections.empty());
    for (auto const& aob : section.assertionsAndBenchmarks) {
        if (aob.isAssertion()) {
            auto const& assertion = aob.asAssertion();
            std::cout << assertion.assertionResult.getExpandedExpression() << '\n';
        }
        if (aob.isBenchmark()) {
            auto const& bench = aob.asBenchmark();
            std::cout << bench.info.name << '\n';
        }
    }
}
