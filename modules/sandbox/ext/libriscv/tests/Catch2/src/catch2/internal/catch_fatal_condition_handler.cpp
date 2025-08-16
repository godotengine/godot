
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/** \file
 * This file provides platform specific implementations of FatalConditionHandler
 *
 * This means that there is a lot of conditional compilation, and platform
 * specific code. Currently, Catch2 supports a dummy handler (if no
 * handler is desired), and 2 platform specific handlers:
 *  * Windows' SEH
 *  * POSIX signals
 *
 * Consequently, various pieces of code below are compiled if either of
 * the platform specific handlers is enabled, or if none of them are
 * enabled. It is assumed that both cannot be enabled at the same time,
 * and doing so should cause a compilation error.
 *
 * If another platform specific handler is added, the compile guards
 * below will need to be updated taking these assumptions into account.
 */

#include <catch2/internal/catch_fatal_condition_handler.hpp>

#include <catch2/internal/catch_compiler_capabilities.hpp>
#include <catch2/internal/catch_context.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/internal/catch_windows_h_proxy.hpp>
#include <catch2/internal/catch_stdstreams.hpp>

#include <algorithm>

#if !defined( CATCH_CONFIG_WINDOWS_SEH ) && !defined( CATCH_CONFIG_POSIX_SIGNALS )

namespace Catch {

    // If neither SEH nor signal handling is required, the handler impls
    // do not have to do anything, and can be empty.
    void FatalConditionHandler::engage_platform() {}
    void FatalConditionHandler::disengage_platform() noexcept {}
    FatalConditionHandler::FatalConditionHandler() = default;
    FatalConditionHandler::~FatalConditionHandler() = default;

} // end namespace Catch

#endif // !CATCH_CONFIG_WINDOWS_SEH && !CATCH_CONFIG_POSIX_SIGNALS

#if defined( CATCH_CONFIG_WINDOWS_SEH ) && defined( CATCH_CONFIG_POSIX_SIGNALS )
#error "Inconsistent configuration: Windows' SEH handling and POSIX signals cannot be enabled at the same time"
#endif // CATCH_CONFIG_WINDOWS_SEH && CATCH_CONFIG_POSIX_SIGNALS

#if defined( CATCH_CONFIG_WINDOWS_SEH ) || defined( CATCH_CONFIG_POSIX_SIGNALS )

namespace {
    //! Signals fatal error message to the run context
    void reportFatal( char const * const message ) {
        Catch::getCurrentContext().getResultCapture()->handleFatalErrorCondition( message );
    }

    //! Minimal size Catch2 needs for its own fatal error handling.
    //! Picked empirically, so it might not be sufficient on all
    //! platforms, and for all configurations.
    constexpr std::size_t minStackSizeForErrors = 32 * 1024;
} // end unnamed namespace

#endif // CATCH_CONFIG_WINDOWS_SEH || CATCH_CONFIG_POSIX_SIGNALS

#if defined( CATCH_CONFIG_WINDOWS_SEH )

namespace Catch {

    struct SignalDefs { DWORD id; const char* name; };

    // There is no 1-1 mapping between signals and windows exceptions.
    // Windows can easily distinguish between SO and SigSegV,
    // but SigInt, SigTerm, etc are handled differently.
    static constexpr SignalDefs signalDefs[] = {
        { EXCEPTION_ILLEGAL_INSTRUCTION,  "SIGILL - Illegal instruction signal" },
        { EXCEPTION_STACK_OVERFLOW, "SIGSEGV - Stack overflow" },
        { EXCEPTION_ACCESS_VIOLATION, "SIGSEGV - Segmentation violation signal" },
        { EXCEPTION_INT_DIVIDE_BY_ZERO, "Divide by zero error" },
    };

    static LONG CALLBACK topLevelExceptionFilter(PEXCEPTION_POINTERS ExceptionInfo) {
        for (auto const& def : signalDefs) {
            if (ExceptionInfo->ExceptionRecord->ExceptionCode == def.id) {
                reportFatal(def.name);
            }
        }
        // If its not an exception we care about, pass it along.
        // This stops us from eating debugger breaks etc.
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // Since we do not support multiple instantiations, we put these
    // into global variables and rely on cleaning them up in outlined
    // constructors/destructors
    static LPTOP_LEVEL_EXCEPTION_FILTER previousTopLevelExceptionFilter = nullptr;


    // For MSVC, we reserve part of the stack memory for handling
    // memory overflow structured exception.
    FatalConditionHandler::FatalConditionHandler() {
        ULONG guaranteeSize = static_cast<ULONG>(minStackSizeForErrors);
        if (!SetThreadStackGuarantee(&guaranteeSize)) {
            // We do not want to fully error out, because needing
            // the stack reserve should be rare enough anyway.
            Catch::cerr()
                << "Failed to reserve piece of stack."
                << " Stack overflows will not be reported successfully.";
        }
    }

    // We do not attempt to unset the stack guarantee, because
    // Windows does not support lowering the stack size guarantee.
    FatalConditionHandler::~FatalConditionHandler() = default;


    void FatalConditionHandler::engage_platform() {
        // Register as a the top level exception filter.
        previousTopLevelExceptionFilter = SetUnhandledExceptionFilter(topLevelExceptionFilter);
    }

    void FatalConditionHandler::disengage_platform() noexcept {
        if (SetUnhandledExceptionFilter(previousTopLevelExceptionFilter) != topLevelExceptionFilter) {
            Catch::cerr()
                << "Unexpected SEH unhandled exception filter on disengage."
                << " The filter was restored, but might be rolled back unexpectedly.";
        }
        previousTopLevelExceptionFilter = nullptr;
    }

} // end namespace Catch

#endif // CATCH_CONFIG_WINDOWS_SEH

#if defined( CATCH_CONFIG_POSIX_SIGNALS )

#include <signal.h>

namespace Catch {

    struct SignalDefs {
        int id;
        const char* name;
    };

    static constexpr SignalDefs signalDefs[] = {
        { SIGINT,  "SIGINT - Terminal interrupt signal" },
        { SIGILL,  "SIGILL - Illegal instruction signal" },
        { SIGFPE,  "SIGFPE - Floating point error signal" },
        { SIGSEGV, "SIGSEGV - Segmentation violation signal" },
        { SIGTERM, "SIGTERM - Termination request signal" },
        { SIGABRT, "SIGABRT - Abort (abnormal termination) signal" }
    };

// Older GCCs trigger -Wmissing-field-initializers for T foo = {}
// which is zero initialization, but not explicit. We want to avoid
// that.
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

    static char* altStackMem = nullptr;
    static std::size_t altStackSize = 0;
    static stack_t oldSigStack{};
    static struct sigaction oldSigActions[sizeof(signalDefs) / sizeof(SignalDefs)]{};

    static void restorePreviousSignalHandlers() noexcept {
        // We set signal handlers back to the previous ones. Hopefully
        // nobody overwrote them in the meantime, and doesn't expect
        // their signal handlers to live past ours given that they
        // installed them after ours..
        for (std::size_t i = 0; i < sizeof(signalDefs) / sizeof(SignalDefs); ++i) {
            sigaction(signalDefs[i].id, &oldSigActions[i], nullptr);
        }
        // Return the old stack
        sigaltstack(&oldSigStack, nullptr);
    }

    static void handleSignal( int sig ) {
        char const * name = "<unknown signal>";
        for (auto const& def : signalDefs) {
            if (sig == def.id) {
                name = def.name;
                break;
            }
        }
        // We need to restore previous signal handlers and let them do
        // their thing, so that the users can have the debugger break
        // when a signal is raised, and so on.
        restorePreviousSignalHandlers();
        reportFatal( name );
        raise( sig );
    }

    FatalConditionHandler::FatalConditionHandler() {
        assert(!altStackMem && "Cannot initialize POSIX signal handler when one already exists");
        if (altStackSize == 0) {
            altStackSize = std::max(static_cast<size_t>(SIGSTKSZ), minStackSizeForErrors);
        }
        altStackMem = new char[altStackSize]();
    }

    FatalConditionHandler::~FatalConditionHandler() {
        delete[] altStackMem;
        // We signal that another instance can be constructed by zeroing
        // out the pointer.
        altStackMem = nullptr;
    }

    void FatalConditionHandler::engage_platform() {
        stack_t sigStack;
        sigStack.ss_sp = altStackMem;
        sigStack.ss_size = altStackSize;
        sigStack.ss_flags = 0;
        sigaltstack(&sigStack, &oldSigStack);
        struct sigaction sa = { };

        sa.sa_handler = handleSignal;
        sa.sa_flags = SA_ONSTACK;
        for (std::size_t i = 0; i < sizeof(signalDefs)/sizeof(SignalDefs); ++i) {
            sigaction(signalDefs[i].id, &sa, &oldSigActions[i]);
        }
    }

#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif


    void FatalConditionHandler::disengage_platform() noexcept {
        restorePreviousSignalHandlers();
    }

} // end namespace Catch

#endif // CATCH_CONFIG_POSIX_SIGNALS
