
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TEST_MACRO_IMPL_HPP_INCLUDED
#define CATCH_TEST_MACRO_IMPL_HPP_INCLUDED

#include <catch2/catch_user_config.hpp>
#include <catch2/internal/catch_assertion_handler.hpp>
#include <catch2/internal/catch_preprocessor_internal_stringify.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/internal/catch_stringref.hpp>
#include <catch2/internal/catch_source_line_info.hpp>

// We need this suppression to leak, because it took until GCC 10
// for the front end to handle local suppression via _Pragma properly
#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && __GNUC__ <= 9
  #pragma GCC diagnostic ignored "-Wparentheses"
#endif

#if !defined(CATCH_CONFIG_DISABLE)

#if defined(CATCH_CONFIG_FAST_COMPILE) || defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)

///////////////////////////////////////////////////////////////////////////////
// Another way to speed-up compilation is to omit local try-catch for REQUIRE*
// macros.
#define INTERNAL_CATCH_TRY
#define INTERNAL_CATCH_CATCH( capturer )

#else // CATCH_CONFIG_FAST_COMPILE

#define INTERNAL_CATCH_TRY try
#define INTERNAL_CATCH_CATCH( handler ) catch(...) { (handler).handleUnexpectedInflightException(); }

#endif

///////////////////////////////////////////////////////////////////////////////
#define INTERNAL_CATCH_TEST( macroName, resultDisposition, ... ) \
    do { /* NOLINT(bugprone-infinite-loop) */ \
        /* The expression should not be evaluated, but warnings should hopefully be checked */ \
        CATCH_INTERNAL_IGNORE_BUT_WARN(__VA_ARGS__); \
        Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(__VA_ARGS__), resultDisposition ); \
        INTERNAL_CATCH_TRY { \
            CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
            CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS \
            catchAssertionHandler.handleExpr( Catch::Decomposer() <= __VA_ARGS__ ); /* NOLINT(bugprone-chained-comparison) */ \
            CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
        } INTERNAL_CATCH_CATCH( catchAssertionHandler ) \
        catchAssertionHandler.complete(); \
    } while( (void)0, (false) && static_cast<const bool&>( !!(__VA_ARGS__) ) ) // the expression here is never evaluated at runtime but it forces the compiler to give it a look
    // The double negation silences MSVC's C4800 warning, the static_cast forces short-circuit evaluation if the type has overloaded &&.

///////////////////////////////////////////////////////////////////////////////
#define INTERNAL_CATCH_IF( macroName, resultDisposition, ... ) \
    INTERNAL_CATCH_TEST( macroName, resultDisposition, __VA_ARGS__ ); \
    if( Catch::getResultCapture().lastAssertionPassed() )

///////////////////////////////////////////////////////////////////////////////
#define INTERNAL_CATCH_ELSE( macroName, resultDisposition, ... ) \
    INTERNAL_CATCH_TEST( macroName, resultDisposition, __VA_ARGS__ ); \
    if( !Catch::getResultCapture().lastAssertionPassed() )

///////////////////////////////////////////////////////////////////////////////
#define INTERNAL_CATCH_NO_THROW( macroName, resultDisposition, ... ) \
    do { \
        Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(__VA_ARGS__), resultDisposition ); \
        try { \
            CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
            CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS \
            static_cast<void>(__VA_ARGS__); \
            CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
            catchAssertionHandler.handleExceptionNotThrownAsExpected(); \
        } \
        catch( ... ) { \
            catchAssertionHandler.handleUnexpectedInflightException(); \
        } \
        catchAssertionHandler.complete(); \
    } while( false )

///////////////////////////////////////////////////////////////////////////////
#define INTERNAL_CATCH_THROWS( macroName, resultDisposition, ... ) \
    do { \
        Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(__VA_ARGS__), resultDisposition); \
        if( catchAssertionHandler.allowThrows() ) \
            try { \
                CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
                CATCH_INTERNAL_SUPPRESS_UNUSED_RESULT \
                CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS \
                static_cast<void>(__VA_ARGS__); \
                CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
                catchAssertionHandler.handleUnexpectedExceptionNotThrown(); \
            } \
            catch( ... ) { \
                catchAssertionHandler.handleExceptionThrownAsExpected(); \
            } \
        else \
            catchAssertionHandler.handleThrowingCallSkipped(); \
        catchAssertionHandler.complete(); \
    } while( false )

///////////////////////////////////////////////////////////////////////////////
#define INTERNAL_CATCH_THROWS_AS( macroName, exceptionType, resultDisposition, expr ) \
    do { \
        Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(expr) ", " CATCH_INTERNAL_STRINGIFY(exceptionType), resultDisposition ); \
        if( catchAssertionHandler.allowThrows() ) \
            try { \
                CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
                CATCH_INTERNAL_SUPPRESS_UNUSED_RESULT \
                CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS \
                static_cast<void>(expr); \
                CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
                catchAssertionHandler.handleUnexpectedExceptionNotThrown(); \
            } \
            catch( exceptionType const& ) { \
                catchAssertionHandler.handleExceptionThrownAsExpected(); \
            } \
            catch( ... ) { \
                catchAssertionHandler.handleUnexpectedInflightException(); \
            } \
        else \
            catchAssertionHandler.handleThrowingCallSkipped(); \
        catchAssertionHandler.complete(); \
    } while( false )



///////////////////////////////////////////////////////////////////////////////
// Although this is matcher-based, it can be used with just a string
#define INTERNAL_CATCH_THROWS_STR_MATCHES( macroName, resultDisposition, matcher, ... ) \
    do { \
        Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(__VA_ARGS__) ", " CATCH_INTERNAL_STRINGIFY(matcher), resultDisposition ); \
        if( catchAssertionHandler.allowThrows() ) \
            try { \
                CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
                CATCH_INTERNAL_SUPPRESS_UNUSED_RESULT \
                CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS \
                static_cast<void>(__VA_ARGS__); \
                CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
                catchAssertionHandler.handleUnexpectedExceptionNotThrown(); \
            } \
            catch( ... ) { \
                Catch::handleExceptionMatchExpr( catchAssertionHandler, matcher ); \
            } \
        else \
            catchAssertionHandler.handleThrowingCallSkipped(); \
        catchAssertionHandler.complete(); \
    } while( false )

#endif // CATCH_CONFIG_DISABLE

#endif // CATCH_TEST_MACRO_IMPL_HPP_INCLUDED
