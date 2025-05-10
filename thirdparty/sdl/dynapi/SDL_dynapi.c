/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "SDL_build_config.h"
#include "SDL_dynapi.h"
#include "SDL_dynapi_unsupported.h"

#if SDL_DYNAMIC_API

#define SDL_DYNAMIC_API_ENVVAR "SDL3_DYNAMIC_API"
#define SDL_SLOW_MEMCPY
#define SDL_SLOW_MEMMOVE
#define SDL_SLOW_MEMSET

#ifdef HAVE_STDIO_H
#include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include <SDL3/SDL.h>
#define SDL_MAIN_NOIMPL // don't drag in header-only implementation of SDL_main
#include <SDL3/SDL_main.h>


// These headers have system specific definitions, so aren't included above
#include <SDL3/SDL_vulkan.h>

#if defined(WIN32) || defined(_WIN32) || defined(SDL_PLATFORM_CYGWIN)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#endif

/* This is the version of the dynamic API. This doesn't match the SDL version
   and should not change until there's been a major revamp in API/ABI.
   So 2.0.5 adds functions over 2.0.4? This number doesn't change;
   the sizeof(jump_table) changes instead. But 2.1.0 changes how a function
   works in an incompatible way or removes a function? This number changes,
   since sizeof(jump_table) isn't sufficient anymore. It's likely
   we'll forget to bump every time we add a function, so this is the
   failsafe switch for major API change decisions. Respect it and use it
   sparingly. */
#define SDL_DYNAPI_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

static void SDL_InitDynamicAPI(void);

/* BE CAREFUL CALLING ANY SDL CODE IN HERE, IT WILL BLOW UP.
   Even self-contained stuff might call SDL_SetError() and break everything. */

// behold, the macro salsa!

// Can't use the macro for varargs nonsense. This is atrocious.
#define SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, logname, prio)                                     \
    _static void SDLCALL SDL_Log##logname##name(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) \
    {                                                                                                        \
        va_list ap;                                                                                          \
        initcall;                                                                                            \
        va_start(ap, fmt);                                                                                   \
        jump_table.SDL_LogMessageV(category, SDL_LOG_PRIORITY_##prio, fmt, ap);                              \
        va_end(ap);                                                                                          \
    }

#define SDL_DYNAPI_VARARGS(_static, name, initcall)                                                                                       \
    _static bool SDLCALL SDL_SetError##name(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)                                            \
    {                                                                                                                                     \
        char buf[128], *str = buf;                                                                                                        \
        int result;                                                                                                                       \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        result = jump_table.SDL_vsnprintf(buf, sizeof(buf), fmt, ap);                                                                     \
        va_end(ap);                                                                                                                       \
        if (result >= 0 && (size_t)result >= sizeof(buf)) {                                                                               \
            str = NULL;                                                                                                                   \
            va_start(ap, fmt);                                                                                                            \
            result = jump_table.SDL_vasprintf(&str, fmt, ap);                                                                                        \
            va_end(ap);                                                                                                                   \
        }                                                                                                                                 \
        if (result >= 0) {                                                                                                                \
            jump_table.SDL_SetError("%s", str);                                                                                           \
        }                                                                                                                                 \
        if (str != buf) {                                                                                                                 \
            jump_table.SDL_free(str);                                                                                                     \
        }                                                                                                                                 \
        return false;                                                                                                                 \
    }                                                                                                                                     \
    _static int SDLCALL SDL_sscanf##name(const char *buf, SDL_SCANF_FORMAT_STRING const char *fmt, ...)                                   \
    {                                                                                                                                     \
        int result;                                                                                                                       \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        result = jump_table.SDL_vsscanf(buf, fmt, ap);                                                                                    \
        va_end(ap);                                                                                                                       \
        return result;                                                                                                                    \
    }                                                                                                                                     \
    _static int SDLCALL SDL_snprintf##name(SDL_OUT_Z_CAP(maxlen) char *buf, size_t maxlen, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) \
    {                                                                                                                                     \
        int result;                                                                                                                       \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        result = jump_table.SDL_vsnprintf(buf, maxlen, fmt, ap);                                                                          \
        va_end(ap);                                                                                                                       \
        return result;                                                                                                                    \
    }                                                                                                                                     \
    _static int SDLCALL SDL_swprintf##name(SDL_OUT_Z_CAP(maxlen) wchar_t *buf, size_t maxlen, SDL_PRINTF_FORMAT_STRING const wchar_t *fmt, ...) \
    {                                                                                                                                     \
        int result;                                                                                                                       \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        result = jump_table.SDL_vswprintf(buf, maxlen, fmt, ap);                                                                          \
        va_end(ap);                                                                                                                       \
        return result;                                                                                                                    \
    }                                                                                                                                     \
    _static int SDLCALL SDL_asprintf##name(char **strp, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)                                    \
    {                                                                                                                                     \
        int result;                                                                                                                       \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        result = jump_table.SDL_vasprintf(strp, fmt, ap);                                                                                 \
        va_end(ap);                                                                                                                       \
        return result;                                                                                                                    \
    }                                                                                                                                     \
    _static size_t SDLCALL SDL_IOprintf##name(SDL_IOStream *context, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)                          \
    {                                                                                                                                     \
        size_t result;                                                                                                                    \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        result = jump_table.SDL_IOvprintf(context, fmt, ap);                                                                              \
        va_end(ap);                                                                                                                       \
        return result;                                                                                                                    \
    }                                                                                                                                     \
    _static bool SDLCALL SDL_RenderDebugTextFormat##name(SDL_Renderer *renderer, float x, float y, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) \
    {                                                                                                                                     \
        char buf[128], *str = buf;                                                                                                        \
        int result;                                                                                                                       \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        result = jump_table.SDL_vsnprintf(buf, sizeof(buf), fmt, ap);                                                                     \
        va_end(ap);                                                                                                                       \
        if (result >= 0 && (size_t)result >= sizeof(buf)) {                                                                               \
            str = NULL;                                                                                                                   \
            va_start(ap, fmt);                                                                                                            \
            result = jump_table.SDL_vasprintf(&str, fmt, ap);                                                                             \
            va_end(ap);                                                                                                                   \
        }                                                                                                                                 \
        bool retval = false;                                                                                                              \
        if (result >= 0) {                                                                                                                \
            retval = jump_table.SDL_RenderDebugTextFormat(renderer, x, y, "%s", str);                                                     \
        }                                                                                                                                 \
        if (str != buf) {                                                                                                                 \
            jump_table.SDL_free(str);                                                                                                     \
        }                                                                                                                                 \
        return retval;                                                                                                                    \
    }                                                                                                                                     \
    _static void SDLCALL SDL_Log##name(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)                                                     \
    {                                                                                                                                     \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        jump_table.SDL_LogMessageV(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO, fmt, ap);                                         \
        va_end(ap);                                                                                                                       \
    }                                                                                                                                     \
    _static void SDLCALL SDL_LogMessage##name(int category, SDL_LogPriority priority, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)      \
    {                                                                                                                                     \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        jump_table.SDL_LogMessageV(category, priority, fmt, ap);                                                                          \
        va_end(ap);                                                                                                                       \
    }                                                                                                                                     \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Trace, TRACE)                                                                   \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Verbose, VERBOSE)                                                                   \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Debug, DEBUG)                                                                       \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Info, INFO)                                                                         \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Warn, WARN)                                                                         \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Error, ERROR)                                                                       \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Critical, CRITICAL)

// Typedefs for function pointers for jump table, and predeclare funcs
// The DEFAULT funcs will init jump table and then call real function.
// The REAL funcs are the actual functions, name-mangled to not clash.
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) \
    typedef rc (SDLCALL *SDL_DYNAPIFN_##fn) params;\
    static rc SDLCALL fn##_DEFAULT params;         \
    extern rc SDLCALL fn##_REAL params;
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC

// The jump table!
typedef struct
{
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) SDL_DYNAPIFN_##fn fn;
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
} SDL_DYNAPI_jump_table;

// The actual jump table.
static SDL_DYNAPI_jump_table jump_table = {
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) fn##_DEFAULT,
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
};

// Default functions init the function table then call right thing.
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) \
    static rc SDLCALL fn##_DEFAULT params          \
    {                                              \
        SDL_InitDynamicAPI();                      \
        ret jump_table.fn args;                    \
    }
#define SDL_DYNAPI_PROC_NO_VARARGS 1
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
#undef SDL_DYNAPI_PROC_NO_VARARGS
SDL_DYNAPI_VARARGS(static, _DEFAULT, SDL_InitDynamicAPI())

// Public API functions to jump into the jump table.
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) \
    rc SDLCALL fn params                           \
    {                                              \
        ret jump_table.fn args;                    \
    }
#define SDL_DYNAPI_PROC_NO_VARARGS 1
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
#undef SDL_DYNAPI_PROC_NO_VARARGS
SDL_DYNAPI_VARARGS(, , )

#define ENABLE_SDL_CALL_LOGGING 0
#if ENABLE_SDL_CALL_LOGGING
static bool SDLCALL SDL_SetError_LOGSDLCALLS(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    char buf[512]; // !!! FIXME: dynamic allocation
    va_list ap;
    SDL_Log_REAL("SDL3CALL SDL_SetError");
    va_start(ap, fmt);
    SDL_vsnprintf_REAL(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return SDL_SetError_REAL("%s", buf);
}
static int SDLCALL SDL_sscanf_LOGSDLCALLS(const char *buf, SDL_SCANF_FORMAT_STRING const char *fmt, ...)
{
    int result;
    va_list ap;
    SDL_Log_REAL("SDL3CALL SDL_sscanf");
    va_start(ap, fmt);
    result = SDL_vsscanf_REAL(buf, fmt, ap);
    va_end(ap);
    return result;
}
static int SDLCALL SDL_snprintf_LOGSDLCALLS(SDL_OUT_Z_CAP(maxlen) char *buf, size_t maxlen, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    int result;
    va_list ap;
    SDL_Log_REAL("SDL3CALL SDL_snprintf");
    va_start(ap, fmt);
    result = SDL_vsnprintf_REAL(buf, maxlen, fmt, ap);
    va_end(ap);
    return result;
}
static int SDLCALL SDL_asprintf_LOGSDLCALLS(char **strp, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    int result;
    va_list ap;
    SDL_Log_REAL("SDL3CALL SDL_asprintf");
    va_start(ap, fmt);
    result = SDL_vasprintf_REAL(strp, fmt, ap);
    va_end(ap);
    return result;
}
static int SDLCALL SDL_swprintf_LOGSDLCALLS(SDL_OUT_Z_CAP(maxlen) wchar_t *buf, size_t maxlen, SDL_PRINTF_FORMAT_STRING const wchar_t *fmt, ...)
{
    int result;
    va_list ap;
    SDL_Log_REAL("SDL3CALL SDL_swprintf");
    va_start(ap, fmt);
    result = SDL_vswprintf_REAL(buf, maxlen, fmt, ap);
    va_end(ap);
    return result;
}
static size_t SDLCALL SDL_IOprintf_LOGSDLCALLS(SDL_IOStream *context, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    size_t result;
    va_list ap;
    SDL_Log_REAL("SDL3CALL SDL_IOprintf");
    va_start(ap, fmt);
    result = SDL_IOvprintf_REAL(context, fmt, ap);
    va_end(ap);
    return result;
}
static bool SDLCALL SDL_RenderDebugTextFormat_LOGSDLCALLS(SDL_Renderer *renderer, float x, float y, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    char buf[128], *str = buf;
    int result;
    va_list ap;
    SDL_Log_REAL("SDL3CALL SDL_RenderDebugTextFormat");
    va_start(ap, fmt);
    result = jump_table.SDL_vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (result >= 0 && (size_t)result >= sizeof(buf)) {
        str = NULL;
        va_start(ap, fmt);
        result = SDL_vasprintf_REAL(&str, fmt, ap);
        va_end(ap);
    }
    bool retval = false;
    if (result >= 0) {
        retval = SDL_RenderDebugTextFormat_REAL(renderer, x, y, "%s", str);
    }
    if (str != buf) {
        jump_table.SDL_free(str);
    }
    return retval;
}
static void SDLCALL SDL_Log_LOGSDLCALLS(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;
    SDL_Log_REAL("SDL3CALL SDL_Log");
    va_start(ap, fmt);
    SDL_LogMessageV_REAL(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO, fmt, ap);
    va_end(ap);
}
static void SDLCALL SDL_LogMessage_LOGSDLCALLS(int category, SDL_LogPriority priority, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;
    SDL_Log_REAL("SDL3CALL SDL_LogMessage");
    va_start(ap, fmt);
    SDL_LogMessageV_REAL(category, priority, fmt, ap);
    va_end(ap);
}
#define SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(logname, prio)                                                         \
    static void SDLCALL SDL_Log##logname##_LOGSDLCALLS(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) \
    {                                                                                                               \
        va_list ap;                                                                                                 \
        va_start(ap, fmt);                                                                                          \
        SDL_Log_REAL("SDL3CALL SDL_Log%s", #logname);                                                               \
        SDL_LogMessageV_REAL(category, SDL_LOG_PRIORITY_##prio, fmt, ap);                                           \
        va_end(ap);                                                                                                 \
    }
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Trace, TRACE)
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Verbose, VERBOSE)
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Debug, DEBUG)
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Info, INFO)
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Warn, WARN)
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Error, ERROR)
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Critical, CRITICAL)
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) \
    rc SDLCALL fn##_LOGSDLCALLS params             \
    {                                              \
        SDL_Log_REAL("SDL3CALL %s", #fn);          \
        ret fn##_REAL args;                        \
    }
#define SDL_DYNAPI_PROC_NO_VARARGS 1
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
#undef SDL_DYNAPI_PROC_NO_VARARGS
#endif

/* we make this a static function so we can call the correct one without the
   system's dynamic linker resolving to the wrong version of this. */
static Sint32 initialize_jumptable(Uint32 apiver, void *table, Uint32 tablesize)
{
    SDL_DYNAPI_jump_table *output_jump_table = (SDL_DYNAPI_jump_table *)table;

    if (apiver != SDL_DYNAPI_VERSION) {
        // !!! FIXME: can maybe handle older versions?
        return -1; // not compatible.
    } else if (tablesize > sizeof(jump_table)) {
        return -1; // newer version of SDL with functions we can't provide.
    }

// Init our jump table first.
#if ENABLE_SDL_CALL_LOGGING
    {
        const char *env = SDL_getenv_unsafe_REAL("SDL_DYNAPI_LOG_CALLS");
        const bool log_calls = (env && SDL_atoi_REAL(env));
        if (log_calls) {
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) jump_table.fn = fn##_LOGSDLCALLS;
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
        } else {
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) jump_table.fn = fn##_REAL;
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
        }
    }
#else
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) jump_table.fn = fn##_REAL;
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
#endif

    // Then the external table...
    if (output_jump_table != &jump_table) {
        jump_table.SDL_memcpy(output_jump_table, &jump_table, tablesize);
    }

    // Safe to call SDL functions now; jump table is initialized!

    return 0; // success!
}

// Here's the exported entry point that fills in the jump table.
// Use specific types when an "int" might suffice to keep this sane.
typedef Sint32 (SDLCALL *SDL_DYNAPI_ENTRYFN)(Uint32 apiver, void *table, Uint32 tablesize);
extern SDL_DECLSPEC Sint32 SDLCALL SDL_DYNAPI_entry(Uint32, void *, Uint32);

Sint32 SDL_DYNAPI_entry(Uint32 apiver, void *table, Uint32 tablesize)
{
    return initialize_jumptable(apiver, table, tablesize);
}

#ifdef __cplusplus
}
#endif

// Obviously we can't use SDL_LoadObject() to load SDL.  :)
// Also obviously, we never close the loaded library.
#if defined(WIN32) || defined(_WIN32) || defined(SDL_PLATFORM_CYGWIN)
static SDL_INLINE void *get_sdlapi_entry(const char *fname, const char *sym)
{
    HMODULE lib = LoadLibraryA(fname);
    void *result = NULL;
    if (lib) {
        result = (void *) GetProcAddress(lib, sym);
        if (!result) {
            FreeLibrary(lib);
        }
    }
    return result;
}

#elif defined(SDL_PLATFORM_UNIX) || defined(SDL_PLATFORM_APPLE) || defined(SDL_PLATFORM_HAIKU)
#include <dlfcn.h>
static SDL_INLINE void *get_sdlapi_entry(const char *fname, const char *sym)
{
    void *lib = dlopen(fname, RTLD_NOW | RTLD_LOCAL);
    void *result = NULL;
    if (lib) {
        result = dlsym(lib, sym);
        if (!result) {
            dlclose(lib);
        }
    }
    return result;
}

#else
#error Please define your platform.
#endif

static void dynapi_warn(const char *msg)
{
    const char *caption = "SDL Dynamic API Failure!";
    (void)caption;
// SDL_ShowSimpleMessageBox() is a too heavy for here.
#if (defined(WIN32) || defined(_WIN32) || defined(SDL_PLATFORM_CYGWIN)) && !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    MessageBoxA(NULL, msg, caption, MB_OK | MB_ICONERROR);
#elif defined(HAVE_STDIO_H)
    fprintf(stderr, "\n\n%s\n%s\n\n", caption, msg);
    fflush(stderr);
#endif
}

/* This is not declared in any header, although it is shared between some
    parts of SDL, because we don't want anything calling it without an
    extremely good reason. */
#ifdef __cplusplus
extern "C" {
#endif
extern SDL_NORETURN void SDL_ExitProcess(int exitcode);
#ifdef __WATCOMC__
#pragma aux SDL_ExitProcess aborts;
#endif
#ifdef __cplusplus
}
#endif

static void SDL_InitDynamicAPILocked(void)
{
    // this can't use SDL_getenv_unsafe_REAL, because it might allocate memory before the app can set their allocator.
#if defined(WIN32) || defined(_WIN32) || defined(__CYGWIN__)
    // We've always used LoadLibraryA for this, so this has never worked with Unicode paths on Windows. Sorry.
    char envbuf[512];  // overflows will just report as environment variable being unset, but LoadLibraryA has a MAX_PATH of 260 anyhow, apparently.
    const DWORD rc = GetEnvironmentVariableA(SDL_DYNAMIC_API_ENVVAR, envbuf, (DWORD) sizeof (envbuf));
    char *libname = ((rc != 0) && (rc < sizeof (envbuf))) ? envbuf : NULL;
#else
    char *libname = getenv(SDL_DYNAMIC_API_ENVVAR);
#endif

    SDL_DYNAPI_ENTRYFN entry = NULL; // funcs from here by default.
    bool use_internal = true;

    if (libname) {
        while (*libname && !entry) {
            // This is evil, but we're not making any permanent changes...
            char *ptr = (char *)libname;
            while (true) {
                char ch = *ptr;
                if ((ch == ',') || (ch == '\0')) {
                    *ptr = '\0';
                    entry = (SDL_DYNAPI_ENTRYFN)get_sdlapi_entry(libname, "SDL_DYNAPI_entry");
                    *ptr = ch;
                    libname = (ch == '\0') ? ptr : (ptr + 1);
                    break;
                } else {
                    ptr++;
                }
            }
        }
        if (!entry) {
            dynapi_warn("Couldn't load an overriding SDL library. Please fix or remove the " SDL_DYNAMIC_API_ENVVAR " environment variable. Using the default SDL.");
            // Just fill in the function pointers from this library, later.
        }
    }

    if (entry) {
        if (entry(SDL_DYNAPI_VERSION, &jump_table, sizeof(jump_table)) < 0) {
            dynapi_warn("Couldn't override SDL library. Using a newer SDL build might help. Please fix or remove the " SDL_DYNAMIC_API_ENVVAR " environment variable. Using the default SDL.");
            // Just fill in the function pointers from this library, later.
        } else {
            use_internal = false; // We overrode SDL! Don't use the internal version!
        }
    }

    // Just fill in the function pointers from this library.
    if (use_internal) {
        if (initialize_jumptable(SDL_DYNAPI_VERSION, &jump_table, sizeof(jump_table)) < 0) {
            // Now we're screwed. Should definitely abort now.
            dynapi_warn("Failed to initialize internal SDL dynapi. As this would otherwise crash, we have to abort now.");
#ifndef NDEBUG
            SDL_TriggerBreakpoint();
#endif
            SDL_ExitProcess(86);
        }
    }

    // we intentionally never close the newly-loaded lib, of course.
}

static void SDL_InitDynamicAPI(void)
{
    /* So the theory is that every function in the jump table defaults to
     *  calling this function, and then replaces itself with a version that
     *  doesn't call this function anymore. But it's possible that, in an
     *  extreme corner case, you can have a second thread hit this function
     *  while the jump table is being initialized by the first.
     * In this case, a spinlock is really painful compared to what spinlocks
     *  _should_ be used for, but this would only happen once, and should be
     *  insanely rare, as you would have to spin a thread outside of SDL (as
     *  SDL_CreateThread() would also call this function before building the
     *  new thread).
     */
    static bool already_initialized = false;

    static SDL_SpinLock lock = 0;
    SDL_LockSpinlock_REAL(&lock);

    if (!already_initialized) {
        SDL_InitDynamicAPILocked();
        already_initialized = true;
    }

    SDL_UnlockSpinlock_REAL(&lock);
}

#else // SDL_DYNAMIC_API

#include <SDL3/SDL.h>

Sint32 SDL_DYNAPI_entry(Uint32 apiver, void *table, Uint32 tablesize);
Sint32 SDL_DYNAPI_entry(Uint32 apiver, void *table, Uint32 tablesize)
{
    (void)apiver;
    (void)table;
    (void)tablesize;
    return -1; // not compatible.
}

#endif // SDL_DYNAMIC_API
