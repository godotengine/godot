
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/reporters/catch_reporter_tap.hpp>
#include <catch2/internal/catch_console_colour.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/catch_test_spec.hpp>
#include <catch2/reporters/catch_reporter_helpers.hpp>

#include <algorithm>
#include <ostream>

namespace Catch {

    namespace {
        // Yes, this has to be outside the class and namespaced by naming.
        // Making older compiler happy is hard.
        static constexpr StringRef tapFailedString = "not ok"_sr;
        static constexpr StringRef tapPassedString = "ok"_sr;
        static constexpr Colour::Code tapDimColour = Colour::FileName;

        class TapAssertionPrinter {
        public:
            TapAssertionPrinter& operator= (TapAssertionPrinter const&) = delete;
            TapAssertionPrinter(TapAssertionPrinter const&) = delete;
            TapAssertionPrinter(std::ostream& _stream, AssertionStats const& _stats, std::size_t _counter, ColourImpl* colour_)
                : stream(_stream)
                , result(_stats.assertionResult)
                , messages(_stats.infoMessages)
                , itMessage(_stats.infoMessages.begin())
                , printInfoMessages(true)
                , counter(_counter)
                , colourImpl( colour_ ) {}

            void print() {
                itMessage = messages.begin();

                switch (result.getResultType()) {
                case ResultWas::Ok:
                    printResultType(tapPassedString);
                    printOriginalExpression();
                    printReconstructedExpression();
                    if (!result.hasExpression())
                        printRemainingMessages(Colour::None);
                    else
                        printRemainingMessages();
                    break;
                case ResultWas::ExpressionFailed:
                    if (result.isOk()) {
                        printResultType(tapPassedString);
                    } else {
                        printResultType(tapFailedString);
                    }
                    printOriginalExpression();
                    printReconstructedExpression();
                    if (result.isOk()) {
                        printIssue(" # TODO");
                    }
                    printRemainingMessages();
                    break;
                case ResultWas::ThrewException:
                    printResultType(tapFailedString);
                    printIssue("unexpected exception with message:"_sr);
                    printMessage();
                    printExpressionWas();
                    printRemainingMessages();
                    break;
                case ResultWas::FatalErrorCondition:
                    printResultType(tapFailedString);
                    printIssue("fatal error condition with message:"_sr);
                    printMessage();
                    printExpressionWas();
                    printRemainingMessages();
                    break;
                case ResultWas::DidntThrowException:
                    printResultType(tapFailedString);
                    printIssue("expected exception, got none"_sr);
                    printExpressionWas();
                    printRemainingMessages();
                    break;
                case ResultWas::Info:
                    printResultType("info"_sr);
                    printMessage();
                    printRemainingMessages();
                    break;
                case ResultWas::Warning:
                    printResultType("warning"_sr);
                    printMessage();
                    printRemainingMessages();
                    break;
                case ResultWas::ExplicitFailure:
                    printResultType(tapFailedString);
                    printIssue("explicitly"_sr);
                    printRemainingMessages(Colour::None);
                    break;
                case ResultWas::ExplicitSkip:
                    printResultType(tapPassedString);
                    printIssue(" # SKIP"_sr);
                    printMessage();
                    printRemainingMessages();
                    break;
                    // These cases are here to prevent compiler warnings
                case ResultWas::Unknown:
                case ResultWas::FailureBit:
                case ResultWas::Exception:
                    printResultType("** internal error **"_sr);
                    break;
                }
            }

        private:
            void printResultType(StringRef passOrFail) const {
                if (!passOrFail.empty()) {
                    stream << passOrFail << ' ' << counter << " -";
                }
            }

            void printIssue(StringRef issue) const {
                stream << ' ' << issue;
            }

            void printExpressionWas() {
                if (result.hasExpression()) {
                    stream << ';';
                    stream << colourImpl->guardColour( tapDimColour )
                           << " expression was:";
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
                    stream << colourImpl->guardColour( tapDimColour ) << " for: ";

                    std::string expr = result.getExpandedExpression();
                    std::replace(expr.begin(), expr.end(), '\n', ' ');
                    stream << expr;
                }
            }

            void printMessage() {
                if (itMessage != messages.end()) {
                    stream << " '" << itMessage->message << '\'';
                    ++itMessage;
                }
            }

            void printRemainingMessages(Colour::Code colour = tapDimColour) {
                if (itMessage == messages.end()) {
                    return;
                }

                // using messages.end() directly (or auto) yields compilation error:
                std::vector<MessageInfo>::const_iterator itEnd = messages.end();
                const std::size_t N = static_cast<std::size_t>(itEnd - itMessage);

                stream << colourImpl->guardColour( colour ) << " with "
                       << pluralise( N, "message"_sr ) << ':';

                for (; itMessage != itEnd; ) {
                    // If this assertion is a warning ignore any INFO messages
                    if (printInfoMessages || itMessage->type != ResultWas::Info) {
                        stream << " '" << itMessage->message << '\'';
                        if (++itMessage != itEnd) {
                            stream << colourImpl->guardColour(tapDimColour) << " and";
                        }
                    }
                }
            }

        private:
            std::ostream& stream;
            AssertionResult const& result;
            std::vector<MessageInfo> const& messages;
            std::vector<MessageInfo>::const_iterator itMessage;
            bool printInfoMessages;
            std::size_t counter;
            ColourImpl* colourImpl;
        };

    } // End anonymous namespace

    void TAPReporter::testRunStarting( TestRunInfo const& ) {
        if ( m_config->testSpec().hasFilters() ) {
            m_stream << "# filters: " << m_config->testSpec() << '\n';
        }
        m_stream << "# rng-seed: " << m_config->rngSeed() << '\n'
                 << std::flush;
    }

    void TAPReporter::noMatchingTestCases( StringRef unmatchedSpec ) {
        m_stream << "# No test cases matched '" << unmatchedSpec << "'\n";
    }

    void TAPReporter::assertionEnded(AssertionStats const& _assertionStats) {
        ++counter;

        m_stream << "# " << currentTestCaseInfo->name << '\n';
        TapAssertionPrinter printer(m_stream, _assertionStats, counter, m_colour.get());
        printer.print();

        m_stream << '\n' << std::flush;
    }

    void TAPReporter::testRunEnded(TestRunStats const& _testRunStats) {
        m_stream << "1.." << _testRunStats.totals.assertions.total();
        if (_testRunStats.totals.testCases.total() == 0) {
            m_stream << " # Skipped: No tests ran.";
        }
        m_stream << "\n\n" << std::flush;
        StreamingReporterBase::testRunEnded(_testRunStats);
    }




} // end namespace Catch
