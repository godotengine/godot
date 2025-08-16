
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/reporters/catch_reporter_xml.hpp>

#include <catch2/reporters/catch_reporter_helpers.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/catch_test_spec.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/internal/catch_list.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/catch_version.hpp>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4061) // Not all labels are EXPLICITLY handled in switch
                              // Note that 4062 (not all labels are handled
                              // and default is missing) is enabled
#endif

namespace Catch {
    XmlReporter::XmlReporter( ReporterConfig&& _config )
    :   StreamingReporterBase( CATCH_MOVE(_config) ),
        m_xml(m_stream)
    {
        m_preferences.shouldRedirectStdOut = true;
        m_preferences.shouldReportAllAssertions = true;
        m_preferences.shouldReportAllAssertionStarts = false;
    }

    XmlReporter::~XmlReporter() = default;

    std::string XmlReporter::getDescription() {
        return "Reports test results as an XML document";
    }

    std::string XmlReporter::getStylesheetRef() const {
        return std::string();
    }

    void XmlReporter::writeSourceInfo( SourceLineInfo const& sourceInfo ) {
        m_xml
            .writeAttribute( "filename"_sr, sourceInfo.file )
            .writeAttribute( "line"_sr, sourceInfo.line );
    }

    void XmlReporter::testRunStarting( TestRunInfo const& testInfo ) {
        StreamingReporterBase::testRunStarting( testInfo );
        std::string stylesheetRef = getStylesheetRef();
        if( !stylesheetRef.empty() )
            m_xml.writeStylesheetRef( stylesheetRef );
        m_xml.startElement("Catch2TestRun")
             .writeAttribute("name"_sr, m_config->name())
             .writeAttribute("rng-seed"_sr, m_config->rngSeed())
             .writeAttribute("xml-format-version"_sr, 3)
             .writeAttribute("catch2-version"_sr, libraryVersion());
        if ( m_config->testSpec().hasFilters() ) {
            m_xml.writeAttribute( "filters"_sr, m_config->testSpec() );
        }
    }

    void XmlReporter::testCaseStarting( TestCaseInfo const& testInfo ) {
        StreamingReporterBase::testCaseStarting(testInfo);
        m_xml.startElement( "TestCase" )
            .writeAttribute( "name"_sr, trim( StringRef(testInfo.name) ) )
            .writeAttribute( "tags"_sr, testInfo.tagsAsString() );

        writeSourceInfo( testInfo.lineInfo );

        if ( m_config->showDurations() == ShowDurations::Always )
            m_testCaseTimer.start();
        m_xml.ensureTagClosed();
    }

    void XmlReporter::sectionStarting( SectionInfo const& sectionInfo ) {
        StreamingReporterBase::sectionStarting( sectionInfo );
        if( m_sectionDepth++ > 0 ) {
            m_xml.startElement( "Section" )
                .writeAttribute( "name"_sr, trim( StringRef(sectionInfo.name) ) );
            writeSourceInfo( sectionInfo.lineInfo );
            m_xml.ensureTagClosed();
        }
    }

    void XmlReporter::assertionEnded( AssertionStats const& assertionStats ) {

        AssertionResult const& result = assertionStats.assertionResult;

        bool includeResults = m_config->includeSuccessfulResults() || !result.isOk();

        if( includeResults || result.getResultType() == ResultWas::Warning ) {
            // Print any info messages in <Info> tags.
            for( auto const& msg : assertionStats.infoMessages ) {
                if( msg.type == ResultWas::Info && includeResults ) {
                    auto t = m_xml.scopedElement( "Info" );
                    writeSourceInfo( msg.lineInfo );
                    t.writeText( msg.message );
                } else if ( msg.type == ResultWas::Warning ) {
                    auto t = m_xml.scopedElement( "Warning" );
                    writeSourceInfo( msg.lineInfo );
                    t.writeText( msg.message );
                }
            }
        }

        // Drop out if result was successful but we're not printing them.
        if ( !includeResults && result.getResultType() != ResultWas::Warning &&
             result.getResultType() != ResultWas::ExplicitSkip ) {
            return;
        }

        // Print the expression if there is one.
        if( result.hasExpression() ) {
            m_xml.startElement( "Expression" )
                .writeAttribute( "success"_sr, result.succeeded() )
                .writeAttribute( "type"_sr, result.getTestMacroName() );

            writeSourceInfo( result.getSourceInfo() );

            m_xml.scopedElement( "Original" )
                .writeText( result.getExpression() );
            m_xml.scopedElement( "Expanded" )
                .writeText( result.getExpandedExpression() );
        }

        // And... Print a result applicable to each result type.
        switch( result.getResultType() ) {
            case ResultWas::ThrewException:
                m_xml.startElement( "Exception" );
                writeSourceInfo( result.getSourceInfo() );
                m_xml.writeText( result.getMessage() );
                m_xml.endElement();
                break;
            case ResultWas::FatalErrorCondition:
                m_xml.startElement( "FatalErrorCondition" );
                writeSourceInfo( result.getSourceInfo() );
                m_xml.writeText( result.getMessage() );
                m_xml.endElement();
                break;
            case ResultWas::Info:
                m_xml.scopedElement( "Info" )
                     .writeText( result.getMessage() );
                break;
            case ResultWas::Warning:
                // Warning will already have been written
                break;
            case ResultWas::ExplicitFailure:
                m_xml.startElement( "Failure" );
                writeSourceInfo( result.getSourceInfo() );
                m_xml.writeText( result.getMessage() );
                m_xml.endElement();
                break;
            case ResultWas::ExplicitSkip:
                m_xml.startElement( "Skip" );
                writeSourceInfo( result.getSourceInfo() );
                m_xml.writeText( result.getMessage() );
                m_xml.endElement();
                break;
            default:
                break;
        }

        if( result.hasExpression() )
            m_xml.endElement();
    }

    void XmlReporter::sectionEnded( SectionStats const& sectionStats ) {
        StreamingReporterBase::sectionEnded( sectionStats );
        if ( --m_sectionDepth > 0 ) {
            {
                XmlWriter::ScopedElement e = m_xml.scopedElement( "OverallResults" );
                e.writeAttribute( "successes"_sr, sectionStats.assertions.passed );
                e.writeAttribute( "failures"_sr, sectionStats.assertions.failed );
                e.writeAttribute( "expectedFailures"_sr, sectionStats.assertions.failedButOk );
                e.writeAttribute( "skipped"_sr, sectionStats.assertions.skipped > 0 );

                if ( m_config->showDurations() == ShowDurations::Always )
                    e.writeAttribute( "durationInSeconds"_sr, sectionStats.durationInSeconds );
            }
            // Ends assertion tag
            m_xml.endElement();
        }
    }

    void XmlReporter::testCaseEnded( TestCaseStats const& testCaseStats ) {
        StreamingReporterBase::testCaseEnded( testCaseStats );
        XmlWriter::ScopedElement e = m_xml.scopedElement( "OverallResult" );
        e.writeAttribute( "success"_sr, testCaseStats.totals.assertions.allOk() );
        e.writeAttribute( "skips"_sr, testCaseStats.totals.assertions.skipped );

        if ( m_config->showDurations() == ShowDurations::Always )
            e.writeAttribute( "durationInSeconds"_sr, m_testCaseTimer.getElapsedSeconds() );
        if( !testCaseStats.stdOut.empty() )
            m_xml.scopedElement( "StdOut" ).writeText( trim( StringRef(testCaseStats.stdOut) ), XmlFormatting::Newline );
        if( !testCaseStats.stdErr.empty() )
            m_xml.scopedElement( "StdErr" ).writeText( trim( StringRef(testCaseStats.stdErr) ), XmlFormatting::Newline );

        m_xml.endElement();
    }

    void XmlReporter::testRunEnded( TestRunStats const& testRunStats ) {
        StreamingReporterBase::testRunEnded( testRunStats );
        m_xml.scopedElement( "OverallResults" )
            .writeAttribute( "successes"_sr, testRunStats.totals.assertions.passed )
            .writeAttribute( "failures"_sr, testRunStats.totals.assertions.failed )
            .writeAttribute( "expectedFailures"_sr, testRunStats.totals.assertions.failedButOk )
            .writeAttribute( "skips"_sr, testRunStats.totals.assertions.skipped );
        m_xml.scopedElement( "OverallResultsCases")
            .writeAttribute( "successes"_sr, testRunStats.totals.testCases.passed )
            .writeAttribute( "failures"_sr, testRunStats.totals.testCases.failed )
            .writeAttribute( "expectedFailures"_sr, testRunStats.totals.testCases.failedButOk )
            .writeAttribute( "skips"_sr, testRunStats.totals.testCases.skipped );
        m_xml.endElement();
    }

    void XmlReporter::benchmarkPreparing( StringRef name ) {
        m_xml.startElement("BenchmarkResults")
             .writeAttribute("name"_sr, name);
    }

    void XmlReporter::benchmarkStarting(BenchmarkInfo const &info) {
        m_xml.writeAttribute("samples"_sr, info.samples)
            .writeAttribute("resamples"_sr, info.resamples)
            .writeAttribute("iterations"_sr, info.iterations)
            .writeAttribute("clockResolution"_sr, info.clockResolution)
            .writeAttribute("estimatedDuration"_sr, info.estimatedDuration)
            .writeComment("All values in nano seconds"_sr);
    }

    void XmlReporter::benchmarkEnded(BenchmarkStats<> const& benchmarkStats) {
        m_xml.scopedElement("mean")
            .writeAttribute("value"_sr, benchmarkStats.mean.point.count())
            .writeAttribute("lowerBound"_sr, benchmarkStats.mean.lower_bound.count())
            .writeAttribute("upperBound"_sr, benchmarkStats.mean.upper_bound.count())
            .writeAttribute("ci"_sr, benchmarkStats.mean.confidence_interval);
        m_xml.scopedElement("standardDeviation")
            .writeAttribute("value"_sr, benchmarkStats.standardDeviation.point.count())
            .writeAttribute("lowerBound"_sr, benchmarkStats.standardDeviation.lower_bound.count())
            .writeAttribute("upperBound"_sr, benchmarkStats.standardDeviation.upper_bound.count())
            .writeAttribute("ci"_sr, benchmarkStats.standardDeviation.confidence_interval);
        m_xml.scopedElement("outliers")
            .writeAttribute("variance"_sr, benchmarkStats.outlierVariance)
            .writeAttribute("lowMild"_sr, benchmarkStats.outliers.low_mild)
            .writeAttribute("lowSevere"_sr, benchmarkStats.outliers.low_severe)
            .writeAttribute("highMild"_sr, benchmarkStats.outliers.high_mild)
            .writeAttribute("highSevere"_sr, benchmarkStats.outliers.high_severe);
        m_xml.endElement();
    }

    void XmlReporter::benchmarkFailed(StringRef error) {
        m_xml.scopedElement("failed").
            writeAttribute("message"_sr, error);
        m_xml.endElement();
    }

    void XmlReporter::listReporters(std::vector<ReporterDescription> const& descriptions) {
        auto outerTag = m_xml.scopedElement("AvailableReporters");
        for (auto const& reporter : descriptions) {
            auto inner = m_xml.scopedElement("Reporter");
            m_xml.startElement("Name", XmlFormatting::Indent)
                 .writeText(reporter.name, XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
            m_xml.startElement("Description", XmlFormatting::Indent)
                 .writeText(reporter.description, XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
        }
    }

    void XmlReporter::listListeners(std::vector<ListenerDescription> const& descriptions) {
        auto outerTag = m_xml.scopedElement( "RegisteredListeners" );
        for ( auto const& listener : descriptions ) {
            auto inner = m_xml.scopedElement( "Listener" );
            m_xml.startElement( "Name", XmlFormatting::Indent )
                .writeText( listener.name, XmlFormatting::None )
                .endElement( XmlFormatting::Newline );
            m_xml.startElement( "Description", XmlFormatting::Indent )
                .writeText( listener.description, XmlFormatting::None )
                .endElement( XmlFormatting::Newline );
        }
    }

    void XmlReporter::listTests(std::vector<TestCaseHandle> const& tests) {
        auto outerTag = m_xml.scopedElement("MatchingTests");
        for (auto const& test : tests) {
            auto innerTag = m_xml.scopedElement("TestCase");
            auto const& testInfo = test.getTestCaseInfo();
            m_xml.startElement("Name", XmlFormatting::Indent)
                 .writeText(testInfo.name, XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
            m_xml.startElement("ClassName", XmlFormatting::Indent)
                 .writeText(testInfo.className, XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
            m_xml.startElement("Tags", XmlFormatting::Indent)
                 .writeText(testInfo.tagsAsString(), XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);

            auto sourceTag = m_xml.scopedElement("SourceInfo");
            m_xml.startElement("File", XmlFormatting::Indent)
                 .writeText(testInfo.lineInfo.file, XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
            m_xml.startElement("Line", XmlFormatting::Indent)
                 .writeText(std::to_string(testInfo.lineInfo.line), XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
        }
    }

    void XmlReporter::listTags(std::vector<TagInfo> const& tags) {
        auto outerTag = m_xml.scopedElement("TagsFromMatchingTests");
        for (auto const& tag : tags) {
            auto innerTag = m_xml.scopedElement("Tag");
            m_xml.startElement("Count", XmlFormatting::Indent)
                 .writeText(std::to_string(tag.count), XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
            auto aliasTag = m_xml.scopedElement("Aliases");
            for (auto const& alias : tag.spellings) {
                m_xml.startElement("Alias", XmlFormatting::Indent)
                     .writeText(alias, XmlFormatting::None)
                     .endElement(XmlFormatting::Newline);
            }
        }
    }

} // end namespace Catch

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
