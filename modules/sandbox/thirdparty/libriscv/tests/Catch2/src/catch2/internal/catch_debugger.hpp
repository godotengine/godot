
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_DEBUGGER_HPP_INCLUDED
#define CATCH_DEBUGGER_HPP_INCLUDED

#include <catch2/internal/catch_platform.hpp>

namespace Catch {
    bool isDebuggerActive();
}

#if !defined( CATCH_TRAP ) && defined( __clang__ ) && defined( __has_builtin )
#    if __has_builtin( __builtin_debugtrap )
#        define CATCH_TRAP() __builtin_debugtrap()
#    endif
#endif

#if !defined( CATCH_TRAP ) && defined( _MSC_VER )
#    define CATCH_TRAP() __debugbreak()
#endif

#if !defined(CATCH_TRAP) // If we couldn't use compiler-specific impl from above, we get into platform-specific options
#ifdef CATCH_PLATFORM_MAC

    #if defined(__i386__) || defined(__x86_64__)
        #define CATCH_TRAP() __asm__("int $3\n" : : ) /* NOLINT */
    #elif defined(__aarch64__)
        #define CATCH_TRAP() __asm__(".inst 0xd43e0000")
    #elif defined(__POWERPC__)
        #define CATCH_TRAP() __asm__("li r0, 20\nsc\nnop\nli r0, 37\nli r4, 2\nsc\nnop\n" \
        : : : "memory","r0","r3","r4" ) /* NOLINT */
    #endif

#elif defined(CATCH_PLATFORM_IPHONE)

    // use inline assembler
    #if defined(__i386__) || defined(__x86_64__)
        #define CATCH_TRAP()  __asm__("int $3")
    #elif defined(__aarch64__)
        #define CATCH_TRAP()  __asm__(".inst 0xd4200000")
    #elif defined(__arm__) && !defined(__thumb__)
        #define CATCH_TRAP()  __asm__(".inst 0xe7f001f0")
    #elif defined(__arm__) &&  defined(__thumb__)
        #define CATCH_TRAP()  __asm__(".inst 0xde01")
    #endif

#elif defined(CATCH_PLATFORM_LINUX)
    // If we can use inline assembler, do it because this allows us to break
    // directly at the location of the failing check instead of breaking inside
    // raise() called from it, i.e. one stack frame below.
    #if defined(__GNUC__) && (defined(__i386) || defined(__x86_64))
        #define CATCH_TRAP() asm volatile ("int $3") /* NOLINT */
    #else // Fall back to the generic way.
        #include <signal.h>

        #define CATCH_TRAP() raise(SIGTRAP)
    #endif
#elif defined(__MINGW32__)
    extern "C" __declspec(dllimport) void __stdcall DebugBreak();
    #define CATCH_TRAP() DebugBreak()
#endif
#endif // ^^ CATCH_TRAP is not defined yet, so we define it


#if !defined(CATCH_BREAK_INTO_DEBUGGER)
    #if defined(CATCH_TRAP)
        #define CATCH_BREAK_INTO_DEBUGGER() []{ if( Catch::isDebuggerActive() ) { CATCH_TRAP(); } }()
    #else
        #define CATCH_BREAK_INTO_DEBUGGER() []{}()
    #endif
#endif

#endif // CATCH_DEBUGGER_HPP_INCLUDED
