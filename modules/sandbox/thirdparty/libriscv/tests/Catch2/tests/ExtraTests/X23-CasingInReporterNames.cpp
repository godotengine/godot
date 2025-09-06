
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that reporter registration is case-preserving, selection is
 * case-insensitive.
 *
 * This is done by registering a custom reporter that prints out a marker
 * string upon construction and then invoking the binary with different
 * casings of the name.
 */

#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <iostream>
#include <utility>

class TestReporter : public Catch::StreamingReporterBase {
public:
    TestReporter(Catch::ReporterConfig&& _config):
        StreamingReporterBase(std::move(_config)) {
        std::cout << "TestReporter constructed\n";
    }

    static std::string getDescription() {
        return "Reporter for testing casing handling in reporter registration/selection";
    }

    ~TestReporter() override;
};

TestReporter::~TestReporter() = default;

CATCH_REGISTER_REPORTER("testReporterCASED", TestReporter)

