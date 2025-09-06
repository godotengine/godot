
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_INTERFACES_CAPTURE_HPP_INCLUDED
#define CATCH_INTERFACES_CAPTURE_HPP_INCLUDED

#include <string>

#include <catch2/internal/catch_stringref.hpp>
#include <catch2/internal/catch_result_type.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/benchmark/detail/catch_benchmark_stats_fwd.hpp>

namespace Catch {

    class AssertionResult;
    struct AssertionInfo;
    struct SectionInfo;
    struct SectionEndInfo;
    struct MessageInfo;
    struct MessageBuilder;
    struct Counts;
    struct AssertionReaction;
    struct SourceLineInfo;

    class ITransientExpression;
    class IGeneratorTracker;

    struct BenchmarkInfo;

    namespace Generators {
        class GeneratorUntypedBase;
        using GeneratorBasePtr = Catch::Detail::unique_ptr<GeneratorUntypedBase>;
    }


    class IResultCapture {
    public:
        virtual ~IResultCapture();

        virtual void notifyAssertionStarted( AssertionInfo const& info ) = 0;
        virtual bool sectionStarted( StringRef sectionName,
                                     SourceLineInfo const& sectionLineInfo,
                                     Counts& assertions ) = 0;
        virtual void sectionEnded( SectionEndInfo&& endInfo ) = 0;
        virtual void sectionEndedEarly( SectionEndInfo&& endInfo ) = 0;

        virtual IGeneratorTracker*
        acquireGeneratorTracker( StringRef generatorName,
                                 SourceLineInfo const& lineInfo ) = 0;
        virtual IGeneratorTracker*
        createGeneratorTracker( StringRef generatorName,
                                SourceLineInfo lineInfo,
                                Generators::GeneratorBasePtr&& generator ) = 0;

        virtual void benchmarkPreparing( StringRef name ) = 0;
        virtual void benchmarkStarting( BenchmarkInfo const& info ) = 0;
        virtual void benchmarkEnded( BenchmarkStats<> const& stats ) = 0;
        virtual void benchmarkFailed( StringRef error ) = 0;

        virtual void pushScopedMessage( MessageInfo const& message ) = 0;
        virtual void popScopedMessage( MessageInfo const& message ) = 0;

        virtual void emplaceUnscopedMessage( MessageBuilder&& builder ) = 0;

        virtual void handleFatalErrorCondition( StringRef message ) = 0;

        virtual void handleExpr
                (   AssertionInfo const& info,
                    ITransientExpression const& expr,
                    AssertionReaction& reaction ) = 0;
        virtual void handleMessage
                (   AssertionInfo const& info,
                    ResultWas::OfType resultType,
                    std::string&& message,
                    AssertionReaction& reaction ) = 0;
        virtual void handleUnexpectedExceptionNotThrown
                (   AssertionInfo const& info,
                    AssertionReaction& reaction ) = 0;
        virtual void handleUnexpectedInflightException
                (   AssertionInfo const& info,
                    std::string&& message,
                    AssertionReaction& reaction ) = 0;
        virtual void handleIncomplete
                (   AssertionInfo const& info ) = 0;
        virtual void handleNonExpr
                (   AssertionInfo const &info,
                    ResultWas::OfType resultType,
                    AssertionReaction &reaction ) = 0;


        virtual bool lastAssertionPassed() = 0;

        // Deprecated, do not use:
        virtual std::string getCurrentTestName() const = 0;
        virtual const AssertionResult* getLastResult() const = 0;
        virtual void exceptionEarlyReported() = 0;
    };

    IResultCapture& getResultCapture();
}

#endif // CATCH_INTERFACES_CAPTURE_HPP_INCLUDED
