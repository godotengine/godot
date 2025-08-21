
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/reporters/catch_reporter_console.hpp>

#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/catch_test_spec.hpp>
#include <catch2/internal/catch_console_colour.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/catch_version.hpp>
#include <catch2/internal/catch_textflow.hpp>
#include <catch2/internal/catch_reusable_string_stream.hpp>
#include <catch2/internal/catch_stringref.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/internal/catch_console_width.hpp>
#include <catch2/reporters/catch_reporter_helpers.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/catch_get_random_seed.hpp>

#include <cstdio>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4061) // Not all labels are EXPLICITLY handled in switch
 // Note that 4062 (not all labels are handled and default is missing) is enabled
#endif

#if defined(__clang__)
#  pragma clang diagnostic push
// For simplicity, benchmarking-only helpers are always enabled
#  pragma clang diagnostic ignored "-Wunused-function"
#endif



namespace Catch {

namespace {

// Formatter impl for ConsoleReporter
class ConsoleAssertionPrinter {
public:
    ConsoleAssertionPrinter& operator= (ConsoleAssertionPrinter const&) = delete;
    ConsoleAssertionPrinter(ConsoleAssertionPrinter const&) = delete;
    ConsoleAssertionPrinter(std::ostream& _stream, AssertionStats const& _stats, ColourImpl* colourImpl_, bool _printInfoMessages)
        : stream(_stream),
        stats(_stats),
        result(_stats.assertionResult),
        colour(Colour::None),
        messages(_stats.infoMessages),
        colourImpl(colourImpl_),
        printInfoMessages(_printInfoMessages) {
        switch (result.getResultType()) {
        case ResultWas::Ok:
            colour = Colour::Success;
            passOrFail = "PASSED"_sr;
            //if( result.hasMessage() )
            if (messages.size() == 1)
                messageLabel = "with message"_sr;
            if (messages.size() > 1)
                messageLabel = "with messages"_sr;
            break;
        case ResultWas::ExpressionFailed:
            if (result.isOk()) {
                colour = Colour::Success;
                passOrFail = "FAILED - but was ok"_sr;
            } else {
                colour = Colour::Error;
                passOrFail = "FAILED"_sr;
            }
            if (messages.size() == 1)
                messageLabel = "with message"_sr;
            if (messages.size() > 1)
                messageLabel = "with messages"_sr;
            break;
        case ResultWas::ThrewException:
            colour = Colour::Error;
            passOrFail = "FAILED"_sr;
            // todo switch
            switch (messages.size()) { case 0:
                messageLabel = "due to unexpected exception with "_sr;
                break;
            case 1:
                messageLabel = "due to unexpected exception with message"_sr;
                break;
            default:
                messageLabel = "due to unexpected exception with messages"_sr;
                break;
            }
            break;
        case ResultWas::FatalErrorCondition:
            colour = Colour::Error;
            passOrFail = "FAILED"_sr;
            messageLabel = "due to a fatal error condition"_sr;
            break;
        case ResultWas::DidntThrowException:
            colour = Colour::Error;
            passOrFail = "FAILED"_sr;
            messageLabel = "because no exception was thrown where one was expected"_sr;
            break;
        case ResultWas::Info:
            messageLabel = "info"_sr;
            break;
        case ResultWas::Warning:
            messageLabel = "warning"_sr;
            break;
        case ResultWas::ExplicitFailure:
            passOrFail = "FAILED"_sr;
            colour = Colour::Error;
            if (messages.size() == 1)
                messageLabel = "explicitly with message"_sr;
            if (messages.size() > 1)
                messageLabel = "explicitly with messages"_sr;
            break;
        case ResultWas::ExplicitSkip:
            colour = Colour::Skip;
            passOrFail = "SKIPPED"_sr;
            if (messages.size() == 1)
                messageLabel = "explicitly with message"_sr;
            if (messages.size() > 1)
                messageLabel = "explicitly with messages"_sr;
            break;
            // These cases are here to prevent compiler warnings
        case ResultWas::Unknown:
        case ResultWas::FailureBit:
        case ResultWas::Exception:
            passOrFail = "** internal error **"_sr;
            colour = Colour::Error;
            break;
        }
    }

    void print() const {
        printSourceInfo();
        if (stats.totals.assertions.total() > 0) {
            printResultType();
            printOriginalExpression();
            printReconstructedExpression();
        } else {
            stream << '\n';
        }
        printMessage();
    }

private:
    void printResultType() const {
        if (!passOrFail.empty()) {
            stream << colourImpl->guardColour(colour) << passOrFail << ":\n";
        }
    }
    void printOriginalExpression() const {
        if (result.hasExpression()) {
            stream << colourImpl->guardColour( Colour::OriginalExpression )
                   << "  " << result.getExpressionInMacro() << '\n';
        }
    }
    void printReconstructedExpression() const {
        if (result.hasExpandedExpression()) {
            stream << "with expansion:\n";
            stream << colourImpl->guardColour( Colour::ReconstructedExpression )
                   << TextFlow::Column( result.getExpandedExpression() )
                          .indent( 2 )
                   << '\n';
        }
    }
    void printMessage() const {
        if (!messageLabel.empty())
            stream << messageLabel << ':' << '\n';
        for (auto const& msg : messages) {
            // If this assertion is a warning ignore any INFO messages
            if (printInfoMessages || msg.type != ResultWas::Info)
                stream << TextFlow::Column(msg.message).indent(2) << '\n';
        }
    }
    void printSourceInfo() const {
        stream << colourImpl->guardColour( Colour::FileName )
               << result.getSourceInfo() << ": ";
    }

    std::ostream& stream;
    AssertionStats const& stats;
    AssertionResult const& result;
    Colour::Code colour;
    StringRef passOrFail;
    StringRef messageLabel;
    std::vector<MessageInfo> const& messages;
    ColourImpl* colourImpl;
    bool printInfoMessages;
};

std::size_t makeRatio( std::uint64_t number, std::uint64_t total ) {
    const auto ratio = total > 0 ? CATCH_CONFIG_CONSOLE_WIDTH * number / total : 0;
    return (ratio == 0 && number > 0) ? 1 : static_cast<std::size_t>(ratio);
}

std::size_t&
findMax( std::size_t& i, std::size_t& j, std::size_t& k, std::size_t& l ) {
    if (i > j && i > k && i > l)
        return i;
    else if (j > k && j > l)
        return j;
    else if (k > l)
        return k;
    else
        return l;
}

struct ColumnBreak {};
struct RowBreak {};
struct OutputFlush {};

class Duration {
    enum class Unit : uint8_t {
        Auto,
        Nanoseconds,
        Microseconds,
        Milliseconds,
        Seconds,
        Minutes
    };
    static const uint64_t s_nanosecondsInAMicrosecond = 1000;
    static const uint64_t s_nanosecondsInAMillisecond = 1000 * s_nanosecondsInAMicrosecond;
    static const uint64_t s_nanosecondsInASecond = 1000 * s_nanosecondsInAMillisecond;
    static const uint64_t s_nanosecondsInAMinute = 60 * s_nanosecondsInASecond;

    double m_inNanoseconds;
    Unit m_units;

public:
    explicit Duration(double inNanoseconds, Unit units = Unit::Auto)
        : m_inNanoseconds(inNanoseconds),
        m_units(units) {
        if (m_units == Unit::Auto) {
            if (m_inNanoseconds < s_nanosecondsInAMicrosecond)
                m_units = Unit::Nanoseconds;
            else if (m_inNanoseconds < s_nanosecondsInAMillisecond)
                m_units = Unit::Microseconds;
            else if (m_inNanoseconds < s_nanosecondsInASecond)
                m_units = Unit::Milliseconds;
            else if (m_inNanoseconds < s_nanosecondsInAMinute)
                m_units = Unit::Seconds;
            else
                m_units = Unit::Minutes;
        }

    }

    auto value() const -> double {
        switch (m_units) {
        case Unit::Microseconds:
            return m_inNanoseconds / static_cast<double>(s_nanosecondsInAMicrosecond);
        case Unit::Milliseconds:
            return m_inNanoseconds / static_cast<double>(s_nanosecondsInAMillisecond);
        case Unit::Seconds:
            return m_inNanoseconds / static_cast<double>(s_nanosecondsInASecond);
        case Unit::Minutes:
            return m_inNanoseconds / static_cast<double>(s_nanosecondsInAMinute);
        default:
            return m_inNanoseconds;
        }
    }
    StringRef unitsAsString() const {
        switch (m_units) {
        case Unit::Nanoseconds:
            return "ns"_sr;
        case Unit::Microseconds:
            return "us"_sr;
        case Unit::Milliseconds:
            return "ms"_sr;
        case Unit::Seconds:
            return "s"_sr;
        case Unit::Minutes:
            return "m"_sr;
        default:
            return "** internal error **"_sr;
        }

    }
    friend auto operator << (std::ostream& os, Duration const& duration) -> std::ostream& {
        return os << duration.value() << ' ' << duration.unitsAsString();
    }
};
} // end anon namespace

enum class Justification : uint8_t {
    Left,
    Right
};

struct ColumnInfo {
    std::string name;
    std::size_t width;
    Justification justification;
};

class TablePrinter {
    std::ostream& m_os;
    std::vector<ColumnInfo> m_columnInfos;
    ReusableStringStream m_oss;
    int m_currentColumn = -1;
    bool m_isOpen = false;

public:
    TablePrinter( std::ostream& os, std::vector<ColumnInfo> columnInfos )
    :   m_os( os ),
        m_columnInfos( CATCH_MOVE( columnInfos ) ) {}

    auto columnInfos() const -> std::vector<ColumnInfo> const& {
        return m_columnInfos;
    }

    void open() {
        if (!m_isOpen) {
            m_isOpen = true;
            *this << RowBreak();

			TextFlow::Columns headerCols;
			for (auto const& info : m_columnInfos) {
                assert(info.width > 2);
				headerCols += TextFlow::Column(info.name).width(info.width - 2);
                headerCols += TextFlow::Spacer( 2 );
			}
			m_os << headerCols << '\n';

            m_os << lineOfChars('-') << '\n';
        }
    }
    void close() {
        if (m_isOpen) {
            *this << RowBreak();
            m_os << '\n' << std::flush;
            m_isOpen = false;
        }
    }

    template<typename T>
    friend TablePrinter& operator<< (TablePrinter& tp, T const& value) {
        tp.m_oss << value;
        return tp;
    }

    friend TablePrinter& operator<< (TablePrinter& tp, ColumnBreak) {
        auto colStr = tp.m_oss.str();
        const auto strSize = colStr.size();
        tp.m_oss.str("");
        tp.open();
        if (tp.m_currentColumn == static_cast<int>(tp.m_columnInfos.size() - 1)) {
            tp.m_currentColumn = -1;
            tp.m_os << '\n';
        }
        tp.m_currentColumn++;

        auto colInfo = tp.m_columnInfos[tp.m_currentColumn];
        auto padding = (strSize + 1 < colInfo.width)
            ? std::string(colInfo.width - (strSize + 1), ' ')
            : std::string();
        if (colInfo.justification == Justification::Left)
            tp.m_os << colStr << padding << ' ';
        else
            tp.m_os << padding << colStr << ' ';
        return tp;
    }

    friend TablePrinter& operator<< (TablePrinter& tp, RowBreak) {
        if (tp.m_currentColumn > 0) {
            tp.m_os << '\n';
            tp.m_currentColumn = -1;
        }
        return tp;
    }

    friend TablePrinter& operator<<(TablePrinter& tp, OutputFlush) {
        tp.m_os << std::flush;
        return tp;
    }
};

ConsoleReporter::ConsoleReporter(ReporterConfig&& config):
    StreamingReporterBase( CATCH_MOVE( config ) ),
    m_tablePrinter(Detail::make_unique<TablePrinter>(m_stream,
        [&config]() -> std::vector<ColumnInfo> {
        if (config.fullConfig()->benchmarkNoAnalysis())
        {
            return{
                { "benchmark name", CATCH_CONFIG_CONSOLE_WIDTH - 43, Justification::Left },
                { "     samples", 14, Justification::Right },
                { "  iterations", 14, Justification::Right },
                { "        mean", 14, Justification::Right }
            };
        }
        else
        {
            return{
                { "benchmark name", CATCH_CONFIG_CONSOLE_WIDTH - 43, Justification::Left },
                { "samples      mean       std dev", 14, Justification::Right },
                { "iterations   low mean   low std dev", 14, Justification::Right },
                { "est run time high mean  high std dev", 14, Justification::Right }
            };
        }
    }())) {
    m_preferences.shouldReportAllAssertionStarts = false;
}

ConsoleReporter::~ConsoleReporter() = default;

std::string ConsoleReporter::getDescription() {
    return "Reports test results as plain lines of text";
}

void ConsoleReporter::noMatchingTestCases( StringRef unmatchedSpec ) {
    m_stream << "No test cases matched '" << unmatchedSpec << "'\n";
}

void ConsoleReporter::reportInvalidTestSpec( StringRef arg ) {
    m_stream << "Invalid Filter: " << arg << '\n';
}

void ConsoleReporter::assertionEnded(AssertionStats const& _assertionStats) {
    AssertionResult const& result = _assertionStats.assertionResult;

    bool includeResults = m_config->includeSuccessfulResults() || !result.isOk();

    // Drop out if result was successful but we're not printing them.
    // TODO: Make configurable whether skips should be printed
    if (!includeResults && result.getResultType() != ResultWas::Warning && result.getResultType() != ResultWas::ExplicitSkip)
        return;

    lazyPrint();

    ConsoleAssertionPrinter printer(m_stream, _assertionStats, m_colour.get(), includeResults);
    printer.print();
    m_stream << '\n' << std::flush;
}

void ConsoleReporter::sectionStarting(SectionInfo const& _sectionInfo) {
    m_tablePrinter->close();
    m_headerPrinted = false;
    StreamingReporterBase::sectionStarting(_sectionInfo);
}
void ConsoleReporter::sectionEnded(SectionStats const& _sectionStats) {
    m_tablePrinter->close();
    if (_sectionStats.missingAssertions) {
        lazyPrint();
        auto guard =
            m_colour->guardColour( Colour::ResultError ).engage( m_stream );
        if (m_sectionStack.size() > 1)
            m_stream << "\nNo assertions in section";
        else
            m_stream << "\nNo assertions in test case";
        m_stream << " '" << _sectionStats.sectionInfo.name << "'\n\n" << std::flush;
    }
    double dur = _sectionStats.durationInSeconds;
    if (shouldShowDuration(*m_config, dur)) {
        m_stream << getFormattedDuration(dur) << " s: " << _sectionStats.sectionInfo.name << '\n' << std::flush;
    }
    if (m_headerPrinted) {
        m_headerPrinted = false;
    }
    StreamingReporterBase::sectionEnded(_sectionStats);
}

void ConsoleReporter::benchmarkPreparing( StringRef name ) {
	lazyPrintWithoutClosingBenchmarkTable();

	auto nameCol = TextFlow::Column( static_cast<std::string>( name ) )
                       .width( m_tablePrinter->columnInfos()[0].width - 2 );

	bool firstLine = true;
	for (auto line : nameCol) {
		if (!firstLine)
			(*m_tablePrinter) << ColumnBreak() << ColumnBreak() << ColumnBreak();
		else
			firstLine = false;

		(*m_tablePrinter) << line << ColumnBreak();
	}
}

void ConsoleReporter::benchmarkStarting(BenchmarkInfo const& info) {
    (*m_tablePrinter) << info.samples << ColumnBreak()
        << info.iterations << ColumnBreak();
    if ( !m_config->benchmarkNoAnalysis() ) {
        ( *m_tablePrinter )
            << Duration( info.estimatedDuration ) << ColumnBreak();
    }
    ( *m_tablePrinter ) << OutputFlush{};
}
void ConsoleReporter::benchmarkEnded(BenchmarkStats<> const& stats) {
    if (m_config->benchmarkNoAnalysis())
    {
        (*m_tablePrinter) << Duration(stats.mean.point.count()) << ColumnBreak();
    }
    else
    {
        (*m_tablePrinter) << ColumnBreak()
            << Duration(stats.mean.point.count()) << ColumnBreak()
            << Duration(stats.mean.lower_bound.count()) << ColumnBreak()
            << Duration(stats.mean.upper_bound.count()) << ColumnBreak() << ColumnBreak()
            << Duration(stats.standardDeviation.point.count()) << ColumnBreak()
            << Duration(stats.standardDeviation.lower_bound.count()) << ColumnBreak()
            << Duration(stats.standardDeviation.upper_bound.count()) << ColumnBreak() << ColumnBreak() << ColumnBreak() << ColumnBreak() << ColumnBreak();
    }
}

void ConsoleReporter::benchmarkFailed( StringRef error ) {
    auto guard = m_colour->guardColour( Colour::Red ).engage( m_stream );
    (*m_tablePrinter)
        << "Benchmark failed (" << error << ')'
        << ColumnBreak() << RowBreak();
}

void ConsoleReporter::testCaseEnded(TestCaseStats const& _testCaseStats) {
    m_tablePrinter->close();
    StreamingReporterBase::testCaseEnded(_testCaseStats);
    m_headerPrinted = false;
}
void ConsoleReporter::testRunEnded(TestRunStats const& _testRunStats) {
    printTotalsDivider(_testRunStats.totals);
    printTestRunTotals( m_stream, *m_colour, _testRunStats.totals );
    m_stream << '\n' << std::flush;
    StreamingReporterBase::testRunEnded(_testRunStats);
}
void ConsoleReporter::testRunStarting(TestRunInfo const& _testRunInfo) {
    StreamingReporterBase::testRunStarting(_testRunInfo);
    if ( m_config->testSpec().hasFilters() ) {
        m_stream << m_colour->guardColour( Colour::BrightYellow ) << "Filters: "
                 << m_config->testSpec() << '\n';
    }
    m_stream << "Randomness seeded to: " << getSeed() << '\n'
             << std::flush;
}

void ConsoleReporter::lazyPrint() {

    m_tablePrinter->close();
    lazyPrintWithoutClosingBenchmarkTable();
}

void ConsoleReporter::lazyPrintWithoutClosingBenchmarkTable() {

    if ( !m_testRunInfoPrinted ) {
        lazyPrintRunInfo();
    }
    if (!m_headerPrinted) {
        printTestCaseAndSectionHeader();
        m_headerPrinted = true;
    }
}
void ConsoleReporter::lazyPrintRunInfo() {
    m_stream << '\n'
             << lineOfChars( '~' ) << '\n'
             << m_colour->guardColour( Colour::SecondaryText )
             << currentTestRunInfo.name << " is a Catch2 v" << libraryVersion()
             << " host application.\n"
             << "Run with -? for options\n\n";

    m_testRunInfoPrinted = true;
}
void ConsoleReporter::printTestCaseAndSectionHeader() {
    assert(!m_sectionStack.empty());
    printOpenHeader(currentTestCaseInfo->name);

    if (m_sectionStack.size() > 1) {
        auto guard = m_colour->guardColour( Colour::Headers ).engage( m_stream );

        auto
            it = m_sectionStack.begin() + 1, // Skip first section (test case)
            itEnd = m_sectionStack.end();
        for (; it != itEnd; ++it)
            printHeaderString(it->name, 2);
    }

    SourceLineInfo lineInfo = m_sectionStack.back().lineInfo;


    m_stream << lineOfChars( '-' ) << '\n'
             << m_colour->guardColour( Colour::FileName ) << lineInfo << '\n'
             << lineOfChars( '.' ) << "\n\n"
             << std::flush;
}

void ConsoleReporter::printClosedHeader(std::string const& _name) {
    printOpenHeader(_name);
    m_stream << lineOfChars('.') << '\n';
}
void ConsoleReporter::printOpenHeader(std::string const& _name) {
    m_stream << lineOfChars('-') << '\n';
    {
        auto guard = m_colour->guardColour( Colour::Headers ).engage( m_stream );
        printHeaderString(_name);
    }
}

void ConsoleReporter::printHeaderString(std::string const& _string, std::size_t indent) {
    // We want to get a bit fancy with line breaking here, so that subsequent
    // lines start after ":" if one is present, e.g.
    // ```
    // blablabla: Fancy
    //            linebreaking
    // ```
    // but we also want to avoid problems with overly long indentation causing
    // the text to take up too many lines, e.g.
    // ```
    // blablabla: F
    //            a
    //            n
    //            c
    //            y
    //            .
    //            .
    //            .
    // ```
    // So we limit the prefix indentation check to first quarter of the possible
    // width
    std::size_t idx = _string.find( ": " );
    if ( idx != std::string::npos && idx < CATCH_CONFIG_CONSOLE_WIDTH / 4 ) {
        idx += 2;
    } else {
        idx = 0;
    }
    m_stream << TextFlow::Column( _string )
                  .indent( indent + idx )
                  .initialIndent( indent )
           << '\n';
}

void ConsoleReporter::printTotalsDivider(Totals const& totals) {
    if (totals.testCases.total() > 0) {
        std::size_t failedRatio = makeRatio(totals.testCases.failed, totals.testCases.total());
        std::size_t failedButOkRatio = makeRatio(totals.testCases.failedButOk, totals.testCases.total());
        std::size_t passedRatio = makeRatio(totals.testCases.passed, totals.testCases.total());
        std::size_t skippedRatio = makeRatio(totals.testCases.skipped, totals.testCases.total());
        while (failedRatio + failedButOkRatio + passedRatio + skippedRatio < CATCH_CONFIG_CONSOLE_WIDTH - 1)
            findMax(failedRatio, failedButOkRatio, passedRatio, skippedRatio)++;
        while (failedRatio + failedButOkRatio + passedRatio > CATCH_CONFIG_CONSOLE_WIDTH - 1)
            findMax(failedRatio, failedButOkRatio, passedRatio, skippedRatio)--;

        m_stream << m_colour->guardColour( Colour::Error )
                 << std::string( failedRatio, '=' )
                 << m_colour->guardColour( Colour::ResultExpectedFailure )
                 << std::string( failedButOkRatio, '=' );
        if ( totals.testCases.allPassed() ) {
            m_stream << m_colour->guardColour( Colour::ResultSuccess )
                     << std::string( passedRatio, '=' );
        } else {
            m_stream << m_colour->guardColour( Colour::Success )
                     << std::string( passedRatio, '=' );
        }
        m_stream << m_colour->guardColour( Colour::Skip )
                 << std::string( skippedRatio, '=' );
    } else {
        m_stream << m_colour->guardColour( Colour::Warning )
                 << std::string( CATCH_CONFIG_CONSOLE_WIDTH - 1, '=' );
    }
    m_stream << '\n';
}

} // end namespace Catch

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(__clang__)
#  pragma clang diagnostic pop
#endif
