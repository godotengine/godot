
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_AUTOMAKE_HPP_INCLUDED
#define CATCH_REPORTER_AUTOMAKE_HPP_INCLUDED

#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

#include <string>

namespace Catch {

    class AutomakeReporter final : public StreamingReporterBase {
    public:
        // GCC5 compat: we cannot use inherited constructor, because it
        //              doesn't implement backport of P0136
        AutomakeReporter( ReporterConfig&& _config ):
            StreamingReporterBase( CATCH_MOVE( _config ) ) {
            m_preferences.shouldReportAllAssertionStarts = false;
        }

        ~AutomakeReporter() override;

        static std::string getDescription() {
            using namespace std::string_literals;
            return "Reports test results in the format of Automake .trs files"s;
        }

        void testCaseEnded(TestCaseStats const& _testCaseStats) override;
        void skipTest(TestCaseInfo const& testInfo) override;
    };

} // end namespace Catch

#endif // CATCH_REPORTER_AUTOMAKE_HPP_INCLUDED
