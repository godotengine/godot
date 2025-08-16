
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/reporters/catch_reporter_compact.hpp>

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_spec.hpp>
#include <catch2/reporters/catch_reporter_helpers.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/internal/catch_platform.hpp>
#include <catch2/internal/catch_console_colour.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/internal/catch_stringref.hpp>

#include <ostream>

namespace Catch {
namespace {

    // Colour::LightGrey
    static constexpr Colour::Code compactDimColour = Colour::FileName;

#ifdef CATCH_PLATFORM_MAC
    static constexpr Catch::StringRef compactFailedString = "FAILED"_sr;
    static constexpr Catch::StringRef compactPassedString = "PASSED"_sr;
#else
    static constexpr Catch::StringRef compactFailedString = "failed"_sr;
    static constexpr Catch::StringRef compactPassedString = "passed"_sr;
#endif

// Implementation of CompactReporter formatting
class AssertionPrinter {
public:
    AssertionPrinter& operator= (AssertionPrinter const&) = delete;
    AssertionPrinter(AssertionPrinter const&) = delete;
    AssertionPrinter(std::ostream& _stream, AssertionStats const& _stats, bool _printInfoMessages, ColourImpl* colourImpl_)
        : stream(_stream)
        , result(_stats.assertionResult)
        , messages(_stats.infoMessages)
        , itMessage(_stats.infoMessages.begin())
        , printInfoMessages(_printInfoMessages)
        , colourImpl(colourImpl_)
    {}

    void print() {
        printSourceInfo();

        itMessage = messages.begin();

        switch (result.getResultType()) {
        case ResultWas::Ok:
            printResultType(Colour::ResultSuccess, compactPassedString);
            printOriginalExpression();
            printReconstructedExpression();
            if (!result.hasExpression())
                printRemainingMessages(Colour::None);
            else
                printRemainingMessages();
            break;
        case ResultWas::ExpressionFailed:
            if (result.isOk())
                printResultType(Colour::ResultSuccess, compactFailedString + " - but was ok"_sr);
            else
                printResultType(Colour::Error, compactFailedString);
            printOriginalExpression();
            printReconstructedExpression();
            printRemainingMessages();
            break;
        case ResultWas::ThrewException:
            printResultType(Colour::Error, compactFailedString);
            printIssue("unexpected exception with message:");
            printMessage();
            printExpressionWas();
            printRemainingMessages();
            break;
        case ResultWas::FatalErrorCondition:
            printResultType(Colour::Error, compactFailedString);
            printIssue("fatal error condition with message:");
            printMessage();
            printExpressionWas();
            printRemainingMessages();
            break;
        case ResultWas::DidntThrowException:
            printResultType(Colour::Error, compactFailedString);
            printIssue("expected exception, got none");
            printExpressionWas();
            printRemainingMessages();
            break;
        case ResultWas::Info:
            printResultType(Colour::None, "info"_sr);
            printMessage();
            printRemainingMessages();
            break;
        case ResultWas::Warning:
            printResultType(Colour::None, "warning"_sr);
            printMessage();
            printRemainingMessages();
            break;
        case ResultWas::ExplicitFailure:
            printResultType(Colour::Error, compactFailedString);
            printIssue("explicitly");
            printRemainingMessages(Colour::None);
            break;
        case ResultWas::ExplicitSkip:
            printResultType(Colour::Skip, "skipped"_sr);
            printMessage();
            printRemainingMessages();
            break;
            // These cases are here to prevent compiler warnings
        case ResultWas::Unknown:
        case ResultWas::FailureBit:
        case ResultWas::Exception:
            printResultType(Colour::Error, "** internal error **");
            break;
        }
    }

private:
    void printSourceInfo() const {
        stream << colourImpl->guardColour( Colour::FileName )
               << result.getSourceInfo() << ':';
    }

    void printResultType(Colour::Code colour, StringRef passOrFail) const {
        if (!passOrFail.empty()) {
            stream << colourImpl->guardColour(colour) << ' ' << passOrFail;
            stream << ':';
        }
    }

    void printIssue(char const* issue) const {
        stream << ' ' << issue;
    }

    void printExpressionWas() {
        if (result.hasExpression()) {
            stream << ';';
            {
                stream << colourImpl->guardColour(compactDimColour) << " expression was:";
            }
            printOriginalExpression();
        }
    }

    void printOriginalExpression() const {
        if (result.hasExpression()) {
            stream << ' ' << result.getExpression();
        }
    }

    void printReconstructedExpression() const {
        if (result.hasExpandedExpression()) {
            stream << colourImpl->guardColour(compactDimColour) << " for: ";
            stream << result.getExpandedExpression();
        }
    }

    void printMessage() {
        if (itMessage != messages.end()) {
            stream << " '" << itMessage->message << '\'';
            ++itMessage;
        }
    }

    void printRemainingMessages(Colour::Code colour = compactDimColour) {
        if (itMessage == messages.end())
            return;

        const auto itEnd = messages.cend();
        const auto N = static_cast<std::size_t>(itEnd - itMessage);

        stream << colourImpl->guardColour( colour ) << " with "
               << pluralise( N, "message"_sr ) << ':';

        while (itMessage != itEnd) {
            // If this assertion is a warning ignore any INFO messages
            if (printInfoMessages || itMessage->type != ResultWas::Info) {
                printMessage();
                if (itMessage != itEnd) {
                    stream << colourImpl->guardColour(compactDimColour) << " and";
                }
                continue;
            }
            ++itMessage;
        }
    }

private:
    std::ostream& stream;
    AssertionResult const& result;
    std::vector<MessageInfo> const& messages;
    std::vector<MessageInfo>::const_iterator itMessage;
    bool printInfoMessages;
    ColourImpl* colourImpl;
};

} // anon namespace

        std::string CompactReporter::getDescription() {
            return "Reports test results on a single line, suitable for IDEs";
        }

        void CompactReporter::noMatchingTestCases( StringRef unmatchedSpec ) {
            m_stream << "No test cases matched '" << unmatchedSpec << "'\n";
        }

        void CompactReporter::testRunStarting( TestRunInfo const& ) {
            if ( m_config->testSpec().hasFilters() ) {
                m_stream << m_colour->guardColour( Colour::BrightYellow )
                         << "Filters: "
                         << m_config->testSpec()
                         << '\n';
            }
            m_stream << "RNG seed: " << getSeed() << '\n'
                     << std::flush;
        }

        void CompactReporter::assertionEnded( AssertionStats const& _assertionStats ) {
            AssertionResult const& result = _assertionStats.assertionResult;

            bool printInfoMessages = true;

            // Drop out if result was successful and we're not printing those
            if( !m_config->includeSuccessfulResults() && result.isOk() ) {
                if( result.getResultType() != ResultWas::Warning && result.getResultType() != ResultWas::ExplicitSkip )
                    return;
                printInfoMessages = false;
            }

            AssertionPrinter printer( m_stream, _assertionStats, printInfoMessages, m_colour.get() );
            printer.print();

            m_stream << '\n' << std::flush;
        }

        void CompactReporter::sectionEnded(SectionStats const& _sectionStats) {
            double dur = _sectionStats.durationInSeconds;
            if ( shouldShowDuration( *m_config, dur ) ) {
                m_stream << getFormattedDuration( dur ) << " s: " << _sectionStats.sectionInfo.name << '\n' << std::flush;
            }
        }

        void CompactReporter::testRunEnded( TestRunStats const& _testRunStats ) {
            printTestRunTotals( m_stream, *m_colour, _testRunStats.totals );
            m_stream << "\n\n" << std::flush;
            StreamingReporterBase::testRunEnded( _testRunStats );
        }

        CompactReporter::~CompactReporter() = default;

} // end namespace Catch
