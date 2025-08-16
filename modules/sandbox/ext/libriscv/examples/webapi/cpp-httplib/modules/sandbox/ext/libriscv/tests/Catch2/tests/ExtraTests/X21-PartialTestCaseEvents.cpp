
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Registers custom reporter that reports testCase* events
 *
 * The resulting executable can then be used by an external Python script
 * to verify that testCase{Starting,Ended} and testCasePartial{Starting,Ended}
 * events are properly nested.
 */


#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/generators/catch_generators.hpp>


#include <iostream>

using Catch::TestCaseInfo;
using Catch::TestCaseStats;

class PartialReporter : public Catch::StreamingReporterBase {
public:
    using StreamingReporterBase::StreamingReporterBase;

    ~PartialReporter() override; // = default

    static std::string getDescription() {
        return "Special reporter for testing TestCasePartialStarting/Ended events";
    }

    //! Called _once_ for each TEST_CASE, no matter how many times it is entered
    void testCaseStarting(TestCaseInfo const& testInfo) override {
        std::cout << "TestCaseStarting: " << testInfo.name << '\n';
    }
    //! Called _every time_ a TEST_CASE is entered, including repeats (due to sections)
    void testCasePartialStarting(TestCaseInfo const& testInfo, uint64_t partNumber) override {
        std::cout << "TestCaseStartingPartial: " << testInfo.name << '#' << partNumber << '\n';
    }


    //! Called _every time_ a TEST_CASE is entered, including repeats (due to sections)
    void testCasePartialEnded(TestCaseStats const& testCaseStats, uint64_t partNumber) override {
        std::cout << "TestCasePartialEnded: " << testCaseStats.testInfo->name << '#' << partNumber << '\n';
    }
    //! Called _once_ for each TEST_CASE, no matter how many times it is entered
    void testCaseEnded(TestCaseStats const& testCaseStats) override {
        std::cout << "TestCaseEnded: " << testCaseStats.testInfo->name << '\n';
    }
};
PartialReporter::~PartialReporter() = default;


CATCH_REGISTER_REPORTER("partial", PartialReporter)

TEST_CASE("section") {
    SECTION("A") {}
    SECTION("B") {}
    SECTION("C") {}
    SECTION("D") {}
}

TEST_CASE("generator") {
    auto _ = GENERATE(1, 2, 3, 4);
    (void)_;
}
