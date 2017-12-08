// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_CORE_DEBUG_H
#define NV_CORE_DEBUG_H

#include "nvcore.h"

#include <stdarg.h> // va_list

#if NV_OS_IOS //ACS: maybe we want this for OSX too?
#   ifdef __APPLE__
#       include <TargetConditionals.h>
#       include <signal.h>
#   endif
#endif

// Make sure we are using our assert.
#undef assert

#define NV_ABORT_DEBUG      1
#define NV_ABORT_IGNORE     2
#define NV_ABORT_EXIT       3

#define nvNoAssert(exp) \
    NV_MULTI_LINE_MACRO_BEGIN \
    (void)sizeof(exp); \
    NV_MULTI_LINE_MACRO_END

#if NV_NO_ASSERT

#   define nvAssert(exp) nvNoAssert(exp)
#   define nvCheck(exp) nvNoAssert(exp)
#   define nvDebugAssert(exp) nvNoAssert(exp)
#   define nvDebugCheck(exp) nvNoAssert(exp)
#   define nvDebugBreak() nvNoAssert(0)

#else // NV_NO_ASSERT

#   if NV_CC_MSVC
        // @@ Does this work in msvc-6 and earlier?
#       define nvDebugBreak()       __debugbreak()
//#       define nvDebugBreak()        __asm { int 3 }
#   elif NV_OS_ORBIS
#       define nvDebugBreak()       __debugbreak()
#   elif NV_OS_IOS && TARGET_OS_IPHONE
#       define nvDebugBreak()       raise(SIGINT)
#   elif NV_CC_CLANG
#       define nvDebugBreak()       __builtin_debugtrap()
#   elif NV_CC_GNUC
//#       define nvDebugBreak()       __builtin_debugtrap()     // Does GCC have debugtrap?
#       define nvDebugBreak()		__builtin_trap()
/*
#   elif NV_CC_GNUC && NV_CPU_PPC && NV_OS_DARWIN
// @@ Use __builtin_trap() on GCC
#       define nvDebugBreak()       __asm__ volatile ("trap")
#   elif NV_CC_GNUC && NV_CPU_X86 && NV_OS_DARWIN
#       define nvDebugBreak()       __asm__ volatile ("int3")
#   elif NV_CC_GNUC && NV_CPU_X86 
#       define nvDebugBreak()       __asm__ ( "int %0" : :"I"(3) )
#   elif NV_OS_ORBIS
#       define nvDebugBreak()       __asm volatile ("int $0x41")
#   else
#       include <signal.h>
#       define nvDebugBreak()       raise(SIGTRAP); 
// define nvDebugBreak()        *((int *)(0)) = 0
*/
#   endif

#  if NV_CC_MSVC
#   define nvExpect(expr) (expr)
#else
#   define nvExpect(expr) __builtin_expect((expr) != 0, true)
#endif

#if NV_CC_CLANG 
#   if __has_feature(attribute_analyzer_noreturn)
#       define NV_ANALYZER_NORETURN __attribute__((analyzer_noreturn))
#   else
#       define NV_ANALYZER_NORETURN
#   endif
#else
#   define NV_ANALYZER_NORETURN
#endif

#define nvDebugBreakOnce() \
    NV_MULTI_LINE_MACRO_BEGIN \
    static bool firstTime = true; \
    if (firstTime) { firstTime = false; nvDebugBreak(); } \
    NV_MULTI_LINE_MACRO_END

#define nvAssertMacro(exp) \
    NV_MULTI_LINE_MACRO_BEGIN \
    if (!nvExpect(exp)) { \
        if (nvAbort(#exp, __FILE__, __LINE__, __FUNC__) == NV_ABORT_DEBUG) { \
            nvDebugBreak(); \
        } \
    } \
    NV_MULTI_LINE_MACRO_END

// GCC, LLVM need "##" before the __VA_ARGS__, MSVC doesn't care
#define nvAssertMacroWithIgnoreAll(exp,...) \
    NV_MULTI_LINE_MACRO_BEGIN \
        static bool ignoreAll = false; \
        if (!ignoreAll && !nvExpect(exp)) { \
            int _result = nvAbort(#exp, __FILE__, __LINE__, __FUNC__, ##__VA_ARGS__); \
            if (_result == NV_ABORT_DEBUG) { \
                nvDebugBreak(); \
            } else if (_result == NV_ABORT_IGNORE) { \
                ignoreAll = true; \
            } \
        } \
    NV_MULTI_LINE_MACRO_END

// Interesting assert macro from Insomniac:
// http://www.gdcvault.com/play/1015319/Developing-Imperfect-Software-How-to
// Used as follows:
// if (nvCheck(i < count)) {
//     normal path
// } else {
//     fixup code.
// }
// This style of macro could be combined with __builtin_expect to let the compiler know failure is unlikely.
#define nvCheckMacro(exp) \
    (\
        (exp) ? true : ( \
            (nvAbort(#exp, __FILE__, __LINE__, __FUNC__) == NV_ABORT_DEBUG) ? (nvDebugBreak(), true) : ( false ) \
        ) \
    )


#define nvAssert(exp)    nvAssertMacro(exp)
#define nvCheck(exp)     nvAssertMacro(exp)

#if defined(_DEBUG)
#   define nvDebugAssert(exp)   nvAssertMacro(exp)
#   define nvDebugCheck(exp)    nvAssertMacro(exp)
#else // _DEBUG
#   define nvDebugAssert(exp)   nvNoAssert(exp)
#   define nvDebugCheck(exp)    nvNoAssert(exp)
#endif // _DEBUG

#endif // NV_NO_ASSERT

// Use nvAssume for very simple expresions only: nvAssume(0), nvAssume(value == true), etc.
/*#if !defined(_DEBUG)
#   if NV_CC_MSVC
#       define nvAssume(exp)    __assume(exp)
#   else
#       define nvAssume(exp)    nvCheck(exp)
#   endif
#else
#   define nvAssume(exp)    nvCheck(exp)
#endif*/

#if defined(_DEBUG)
#  if NV_CC_MSVC
#   define nvUnreachable() nvAssert(0 && "unreachable"); __assume(0)
#  else
#   define nvUnreachable() nvAssert(0 && "unreachable"); __builtin_unreachable()
#  endif
#else
#  if NV_CC_MSVC
#   define nvUnreachable() __assume(0)
#  else
#   define nvUnreachable() __builtin_unreachable()
#  endif
#endif

#define nvError(x)      nvAbort(x, __FILE__, __LINE__, __FUNC__)
#define nvWarning(x)    nvDebugPrint("*** Warning %s/%d: %s\n", __FILE__, __LINE__, (x))

#ifndef NV_DEBUG_PRINT
#define NV_DEBUG_PRINT 1 //defined(_DEBUG)
#endif

#if NV_DEBUG_PRINT
#define nvDebug(...)    nvDebugPrint(__VA_ARGS__)
#else
#if NV_CC_MSVC
#define nvDebug(...)    __noop(__VA_ARGS__)
#else
#define nvDebug(...)    ((void)0) // Non-msvc platforms do not evaluate arguments?
#endif
#endif


NVCORE_API int nvAbort(const char *exp, const char *file, int line, const char * func = NULL, const char * msg = NULL, ...) __attribute__((format (printf, 5, 6))) NV_ANALYZER_NORETURN;
NVCORE_API void NV_CDECL nvDebugPrint( const char *msg, ... ) __attribute__((format (printf, 1, 2)));

namespace nv
{
    inline bool isValidPtr(const void * ptr) {
    #if NV_OS_DARWIN
        return true;    // IC: Not sure what ranges are OK on OSX.
    #endif
        
    #if NV_CPU_X86_64
        if (ptr == NULL) return true;
        if (reinterpret_cast<uint64>(ptr) < 0x10000ULL) return false;
        if (reinterpret_cast<uint64>(ptr) >= 0x000007FFFFFEFFFFULL) return false;
    #else
	    if (reinterpret_cast<uint32>(ptr) == 0xcccccccc) return false;
	    if (reinterpret_cast<uint32>(ptr) == 0xcdcdcdcd) return false;
	    if (reinterpret_cast<uint32>(ptr) == 0xdddddddd) return false;
	    if (reinterpret_cast<uint32>(ptr) == 0xffffffff) return false;
    #endif
        return true;
    }

    // Message handler interface.
    struct MessageHandler {
        virtual void log(const char * str, va_list arg) = 0;
        virtual ~MessageHandler() {}
    };

    // Assert handler interface.
    struct AssertHandler {
        virtual int assertion(const char *exp, const char *file, int line, const char *func, const char *msg, va_list arg) = 0;
        virtual ~AssertHandler() {}
    };


    namespace debug
    {
        NVCORE_API void dumpInfo();
        NVCORE_API void dumpCallstack( MessageHandler *messageHandler, int callstackLevelsToSkip = 0 );

        NVCORE_API void setMessageHandler( MessageHandler * messageHandler );
        NVCORE_API void resetMessageHandler();

        NVCORE_API void setAssertHandler( AssertHandler * assertHanlder );
        NVCORE_API void resetAssertHandler();

        NVCORE_API void enableSigHandler(bool interactive);
        NVCORE_API void disableSigHandler();

        NVCORE_API bool isDebuggerPresent();
        NVCORE_API bool attachToDebugger();

        NVCORE_API void terminate(int code);
    }

} // nv namespace

#endif // NV_CORE_DEBUG_H
