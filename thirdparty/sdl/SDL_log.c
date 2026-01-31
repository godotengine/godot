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
#include "SDL_internal.h"

#if defined(SDL_PLATFORM_WINDOWS)
#include "core/windows/SDL_windows.h"
#endif

// Simple log messages in SDL

#include "SDL_log_c.h"

#ifdef HAVE_STDIO_H
#include <stdio.h>
#endif

#ifdef SDL_PLATFORM_ANDROID
#include <android/log.h>
#endif

#include "stdlib/SDL_vacopy.h"

// The size of the stack buffer to use for rendering log messages.
#define SDL_MAX_LOG_MESSAGE_STACK 256

#define DEFAULT_CATEGORY -1

typedef struct SDL_LogLevel
{
    int category;
    SDL_LogPriority priority;
    struct SDL_LogLevel *next;
} SDL_LogLevel;


// The default log output function
static void SDLCALL SDL_LogOutput(void *userdata, int category, SDL_LogPriority priority, const char *message);

static void CleanupLogPriorities(void);
static void CleanupLogPrefixes(void);

static SDL_InitState SDL_log_init;
static SDL_Mutex *SDL_log_lock;
static SDL_Mutex *SDL_log_function_lock;
static SDL_LogLevel *SDL_loglevels SDL_GUARDED_BY(SDL_log_lock);
static SDL_LogPriority SDL_log_priorities[SDL_LOG_CATEGORY_CUSTOM] SDL_GUARDED_BY(SDL_log_lock);
static SDL_LogPriority SDL_log_default_priority SDL_GUARDED_BY(SDL_log_lock);
static SDL_LogOutputFunction SDL_log_function SDL_GUARDED_BY(SDL_log_function_lock) = SDL_LogOutput;
static void *SDL_log_userdata SDL_GUARDED_BY(SDL_log_function_lock) = NULL;

#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

// If this list changes, update the documentation for SDL_HINT_LOGGING
static const char * const SDL_priority_names[] = {
    NULL,
    "TRACE",
    "VERBOSE",
    "DEBUG",
    "INFO",
    "WARN",
    "ERROR",
    "CRITICAL"
};
SDL_COMPILE_TIME_ASSERT(priority_names, SDL_arraysize(SDL_priority_names) == SDL_LOG_PRIORITY_COUNT);

// This is guarded by SDL_log_function_lock because it's the logging function that calls GetLogPriorityPrefix()
static char *SDL_priority_prefixes[SDL_LOG_PRIORITY_COUNT] SDL_GUARDED_BY(SDL_log_function_lock);

// If this list changes, update the documentation for SDL_HINT_LOGGING
static const char * const SDL_category_names[] = {
    "APP",
    "ERROR",
    "ASSERT",
    "SYSTEM",
    "AUDIO",
    "VIDEO",
    "RENDER",
    "INPUT",
    "TEST",
    "GPU"
};
SDL_COMPILE_TIME_ASSERT(category_names, SDL_arraysize(SDL_category_names) == SDL_LOG_CATEGORY_RESERVED2);

#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic pop
#endif

#ifdef SDL_PLATFORM_ANDROID
static int SDL_android_priority[] = {
    ANDROID_LOG_UNKNOWN,
    ANDROID_LOG_VERBOSE,
    ANDROID_LOG_VERBOSE,
    ANDROID_LOG_DEBUG,
    ANDROID_LOG_INFO,
    ANDROID_LOG_WARN,
    ANDROID_LOG_ERROR,
    ANDROID_LOG_FATAL
};
SDL_COMPILE_TIME_ASSERT(android_priority, SDL_arraysize(SDL_android_priority) == SDL_LOG_PRIORITY_COUNT);
#endif // SDL_PLATFORM_ANDROID

static void SDLCALL SDL_LoggingChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_ResetLogPriorities();
}

void SDL_InitLog(void)
{
    if (!SDL_ShouldInit(&SDL_log_init)) {
        return;
    }

    // If these fail we'll continue without them.
    SDL_log_lock = SDL_CreateMutex();
    SDL_log_function_lock = SDL_CreateMutex();

    SDL_AddHintCallback(SDL_HINT_LOGGING, SDL_LoggingChanged, NULL);

    SDL_SetInitialized(&SDL_log_init, true);
}

void SDL_QuitLog(void)
{
    if (!SDL_ShouldQuit(&SDL_log_init)) {
        return;
    }

    SDL_RemoveHintCallback(SDL_HINT_LOGGING, SDL_LoggingChanged, NULL);

    CleanupLogPriorities();
    CleanupLogPrefixes();

    if (SDL_log_lock) {
        SDL_DestroyMutex(SDL_log_lock);
        SDL_log_lock = NULL;
    }
    if (SDL_log_function_lock) {
        SDL_DestroyMutex(SDL_log_function_lock);
        SDL_log_function_lock = NULL;
    }

    SDL_SetInitialized(&SDL_log_init, false);
}

static void SDL_CheckInitLog(void)
{
    int status = SDL_GetAtomicInt(&SDL_log_init.status);
    if (status == SDL_INIT_STATUS_INITIALIZED ||
        (status == SDL_INIT_STATUS_INITIALIZING && SDL_log_init.thread == SDL_GetCurrentThreadID())) {
        return;
    }

    SDL_InitLog();
}

static void CleanupLogPriorities(void)
{
    while (SDL_loglevels) {
        SDL_LogLevel *entry = SDL_loglevels;
        SDL_loglevels = entry->next;
        SDL_free(entry);
    }
}

void SDL_SetLogPriorities(SDL_LogPriority priority)
{
    SDL_CheckInitLog();

    SDL_LockMutex(SDL_log_lock);
    {
        CleanupLogPriorities();

        SDL_log_default_priority = priority;
        for (int i = 0; i < SDL_arraysize(SDL_log_priorities); ++i) {
            SDL_log_priorities[i] = priority;
        }
    }
    SDL_UnlockMutex(SDL_log_lock);
}

void SDL_SetLogPriority(int category, SDL_LogPriority priority)
{
    SDL_LogLevel *entry;

    SDL_CheckInitLog();

    SDL_LockMutex(SDL_log_lock);
    {
        if (category >= 0 && category < SDL_arraysize(SDL_log_priorities)) {
            SDL_log_priorities[category] = priority;
        } else {
            for (entry = SDL_loglevels; entry; entry = entry->next) {
                if (entry->category == category) {
                    entry->priority = priority;
                    break;
                }
            }

            if (!entry) {
                entry = (SDL_LogLevel *)SDL_malloc(sizeof(*entry));
                if (entry) {
                    entry->category = category;
                    entry->priority = priority;
                    entry->next = SDL_loglevels;
                    SDL_loglevels = entry;
                }
            }
        }
    }
    SDL_UnlockMutex(SDL_log_lock);
}

SDL_LogPriority SDL_GetLogPriority(int category)
{
    SDL_LogLevel *entry;
    SDL_LogPriority priority = SDL_LOG_PRIORITY_INVALID;

    SDL_CheckInitLog();

    // Bypass the lock for known categories
    // Technically if the priority was set on a different CPU the value might not
    // be visible on this CPU for a while, but in practice it's fast enough that
    // this performance improvement is worthwhile.
    if (category >= 0 && category < SDL_arraysize(SDL_log_priorities)) {
        return SDL_log_priorities[category];
    }

    SDL_LockMutex(SDL_log_lock);
    {
        if (category >= 0 && category < SDL_arraysize(SDL_log_priorities)) {
            priority = SDL_log_priorities[category];
        } else {
            for (entry = SDL_loglevels; entry; entry = entry->next) {
                if (entry->category == category) {
                    priority = entry->priority;
                    break;
                }
            }
            if (priority == SDL_LOG_PRIORITY_INVALID) {
                priority = SDL_log_default_priority;
            }
        }
    }
    SDL_UnlockMutex(SDL_log_lock);

    return priority;
}

static bool ParseLogCategory(const char *string, size_t length, int *category)
{
    int i;

    if (SDL_isdigit(*string)) {
        *category = SDL_atoi(string);
        return true;
    }

    if (*string == '*') {
        *category = DEFAULT_CATEGORY;
        return true;
    }

    for (i = 0; i < SDL_arraysize(SDL_category_names); ++i) {
        if (SDL_strncasecmp(string, SDL_category_names[i], length) == 0) {
            *category = i;
            return true;
        }
    }
    return false;
}

static bool ParseLogPriority(const char *string, size_t length, SDL_LogPriority *priority)
{
    int i;

    if (SDL_isdigit(*string)) {
        i = SDL_atoi(string);
        if (i == 0) {
            // 0 has a special meaning of "disable this category"
            *priority = SDL_LOG_PRIORITY_COUNT;
            return true;
        }
        if (i > SDL_LOG_PRIORITY_INVALID && i < SDL_LOG_PRIORITY_COUNT) {
            *priority = (SDL_LogPriority)i;
            return true;
        }
        return false;
    }

    if (SDL_strncasecmp(string, "quiet", length) == 0) {
        *priority = SDL_LOG_PRIORITY_COUNT;
        return true;
    }

    for (i = SDL_LOG_PRIORITY_INVALID + 1; i < SDL_LOG_PRIORITY_COUNT; ++i) {
        if (SDL_strncasecmp(string, SDL_priority_names[i], length) == 0) {
            *priority = (SDL_LogPriority)i;
            return true;
        }
    }
    return false;
}

static void ParseLogPriorities(const char *hint)
{
    const char *name, *next;
    int category = DEFAULT_CATEGORY;
    SDL_LogPriority priority = SDL_LOG_PRIORITY_INVALID;

    if (SDL_strchr(hint, '=') == NULL) {
        if (ParseLogPriority(hint, SDL_strlen(hint), &priority)) {
            SDL_SetLogPriorities(priority);
        }
        return;
    }

    for (name = hint; name; name = next) {
        const char *sep = SDL_strchr(name, '=');
        if (!sep) {
            break;
        }
        next = SDL_strchr(sep, ',');
        if (next) {
            ++next;
        }

        if (ParseLogCategory(name, (sep - name), &category)) {
            const char *value = sep + 1;
            size_t len;
            if (next) {
                len = (next - value - 1);
            } else {
                len = SDL_strlen(value);
            }
            if (ParseLogPriority(value, len, &priority)) {
                if (category == DEFAULT_CATEGORY) {
                    for (int i = 0; i < SDL_arraysize(SDL_log_priorities); ++i) {
                        if (SDL_log_priorities[i] == SDL_LOG_PRIORITY_INVALID) {
                            SDL_log_priorities[i] = priority;
                        }
                    }
                    SDL_log_default_priority = priority;
                } else {
                    SDL_SetLogPriority(category, priority);
                }
            }
        }
    }
}

void SDL_ResetLogPriorities(void)
{
    SDL_CheckInitLog();

    SDL_LockMutex(SDL_log_lock);
    {
        CleanupLogPriorities();

        SDL_log_default_priority = SDL_LOG_PRIORITY_INVALID;
        for (int i = 0; i < SDL_arraysize(SDL_log_priorities); ++i) {
            SDL_log_priorities[i] = SDL_LOG_PRIORITY_INVALID;
        }

        const char *hint = SDL_GetHint(SDL_HINT_LOGGING);
        if (hint) {
            ParseLogPriorities(hint);
        }

        if (SDL_log_default_priority == SDL_LOG_PRIORITY_INVALID) {
            SDL_log_default_priority = SDL_LOG_PRIORITY_ERROR;
        }
        for (int i = 0; i < SDL_arraysize(SDL_log_priorities); ++i) {
            if (SDL_log_priorities[i] != SDL_LOG_PRIORITY_INVALID) {
                continue;
            }

            switch (i) {
            case SDL_LOG_CATEGORY_APPLICATION:
                SDL_log_priorities[i] = SDL_LOG_PRIORITY_INFO;
                break;
            case SDL_LOG_CATEGORY_ASSERT:
                SDL_log_priorities[i] = SDL_LOG_PRIORITY_WARN;
                break;
            case SDL_LOG_CATEGORY_TEST:
                SDL_log_priorities[i] = SDL_LOG_PRIORITY_VERBOSE;
                break;
            default:
                SDL_log_priorities[i] = SDL_LOG_PRIORITY_ERROR;
                break;
            }
        }
    }
    SDL_UnlockMutex(SDL_log_lock);
}

static void CleanupLogPrefixes(void)
{
    for (int i = 0; i < SDL_arraysize(SDL_priority_prefixes); ++i) {
        if (SDL_priority_prefixes[i]) {
            SDL_free(SDL_priority_prefixes[i]);
            SDL_priority_prefixes[i] = NULL;
        }
    }
}

static const char *GetLogPriorityPrefix(SDL_LogPriority priority)
{
    if (priority <= SDL_LOG_PRIORITY_INVALID || priority >= SDL_LOG_PRIORITY_COUNT) {
        return "";
    }

    if (SDL_priority_prefixes[priority]) {
        return SDL_priority_prefixes[priority];
    }

    switch (priority) {
    case SDL_LOG_PRIORITY_WARN:
        return "WARNING: ";
    case SDL_LOG_PRIORITY_ERROR:
        return "ERROR: ";
    case SDL_LOG_PRIORITY_CRITICAL:
        return "ERROR: ";
    default:
        return "";
    }
}

bool SDL_SetLogPriorityPrefix(SDL_LogPriority priority, const char *prefix)
{
    char *prefix_copy;

    if (priority <= SDL_LOG_PRIORITY_INVALID || priority >= SDL_LOG_PRIORITY_COUNT) {
        return SDL_InvalidParamError("priority");
    }

    if (!prefix || !*prefix) {
        prefix_copy = SDL_strdup("");
    } else {
        prefix_copy = SDL_strdup(prefix);
    }
    if (!prefix_copy) {
        return false;
    }

    SDL_LockMutex(SDL_log_function_lock);
    {
        if (SDL_priority_prefixes[priority]) {
            SDL_free(SDL_priority_prefixes[priority]);
        }
        SDL_priority_prefixes[priority] = prefix_copy;
    }
    SDL_UnlockMutex(SDL_log_function_lock);

    return true;
}

void SDL_Log(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    SDL_LogMessageV(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO, fmt, ap);
    va_end(ap);
}

void SDL_LogTrace(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    SDL_LogMessageV(category, SDL_LOG_PRIORITY_TRACE, fmt, ap);
    va_end(ap);
}

void SDL_LogVerbose(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    SDL_LogMessageV(category, SDL_LOG_PRIORITY_VERBOSE, fmt, ap);
    va_end(ap);
}

void SDL_LogDebug(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    SDL_LogMessageV(category, SDL_LOG_PRIORITY_DEBUG, fmt, ap);
    va_end(ap);
}

void SDL_LogInfo(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    SDL_LogMessageV(category, SDL_LOG_PRIORITY_INFO, fmt, ap);
    va_end(ap);
}

void SDL_LogWarn(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    SDL_LogMessageV(category, SDL_LOG_PRIORITY_WARN, fmt, ap);
    va_end(ap);
}

void SDL_LogError(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    SDL_LogMessageV(category, SDL_LOG_PRIORITY_ERROR, fmt, ap);
    va_end(ap);
}

void SDL_LogCritical(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    SDL_LogMessageV(category, SDL_LOG_PRIORITY_CRITICAL, fmt, ap);
    va_end(ap);
}

void SDL_LogMessage(int category, SDL_LogPriority priority, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    SDL_LogMessageV(category, priority, fmt, ap);
    va_end(ap);
}

#ifdef SDL_PLATFORM_ANDROID
static const char *GetCategoryPrefix(int category)
{
    if (category < SDL_LOG_CATEGORY_RESERVED2) {
        return SDL_category_names[category];
    }
    if (category < SDL_LOG_CATEGORY_CUSTOM) {
        return "RESERVED";
    }
    return "CUSTOM";
}
#endif // SDL_PLATFORM_ANDROID

void SDL_LogMessageV(int category, SDL_LogPriority priority, SDL_PRINTF_FORMAT_STRING const char *fmt, va_list ap)
{
    char *message = NULL;
    char stack_buf[SDL_MAX_LOG_MESSAGE_STACK];
    size_t len_plus_term;
    int len;
    va_list aq;

    // Nothing to do if we don't have an output function
    if (!SDL_log_function) {
        return;
    }

    // See if we want to do anything with this message
    if (priority < SDL_GetLogPriority(category)) {
        return;
    }

    // Render into stack buffer
    va_copy(aq, ap);
    len = SDL_vsnprintf(stack_buf, sizeof(stack_buf), fmt, aq);
    va_end(aq);

    if (len < 0) {
        return;
    }

    // If message truncated, allocate and re-render
    if (len >= sizeof(stack_buf) && SDL_size_add_check_overflow(len, 1, &len_plus_term)) {
        // Allocate exactly what we need, including the zero-terminator
        message = (char *)SDL_malloc(len_plus_term);
        if (!message) {
            return;
        }
        va_copy(aq, ap);
        len = SDL_vsnprintf(message, len_plus_term, fmt, aq);
        va_end(aq);
    } else {
        message = stack_buf;
    }

    // Chop off final endline.
    if ((len > 0) && (message[len - 1] == '\n')) {
        message[--len] = '\0';
        if ((len > 0) && (message[len - 1] == '\r')) { // catch "\r\n", too.
            message[--len] = '\0';
        }
    }

    SDL_LockMutex(SDL_log_function_lock);
    {
        SDL_log_function(SDL_log_userdata, category, priority, message);
    }
    SDL_UnlockMutex(SDL_log_function_lock);

    // Free only if dynamically allocated
    if (message != stack_buf) {
        SDL_free(message);
    }
}

#if defined(SDL_PLATFORM_WIN32) && !defined(SDL_PLATFORM_GDK)
enum {
    CONSOLE_UNATTACHED = 0,
    CONSOLE_ATTACHED_CONSOLE = 1,
    CONSOLE_ATTACHED_FILE = 2,
    CONSOLE_ATTACHED_ERROR = -1,
} consoleAttached = CONSOLE_UNATTACHED;

// Handle to stderr output of console.
static HANDLE stderrHandle = NULL;
#endif

static void SDLCALL SDL_LogOutput(void *userdata, int category, SDL_LogPriority priority,
                                  const char *message)
{
#if defined(SDL_PLATFORM_WINDOWS)
    // Way too many allocations here, urgh
    // Note: One can't call SDL_SetError here, since that function itself logs.
    {
        char *output;
        size_t length;
        LPTSTR tstr;
        bool isstack;

#if !defined(SDL_PLATFORM_GDK)
        BOOL attachResult;
        DWORD attachError;
        DWORD consoleMode;
        DWORD charsWritten;

        // Maybe attach console and get stderr handle
        if (consoleAttached == CONSOLE_UNATTACHED) {
            attachResult = AttachConsole(ATTACH_PARENT_PROCESS);
            if (!attachResult) {
                attachError = GetLastError();
                if (attachError == ERROR_INVALID_HANDLE) {
                    // This is expected when running from Visual Studio
                    // OutputDebugString(TEXT("Parent process has no console\r\n"));
                    consoleAttached = CONSOLE_ATTACHED_ERROR;
                } else if (attachError == ERROR_GEN_FAILURE) {
                    OutputDebugString(TEXT("Could not attach to console of parent process\r\n"));
                    consoleAttached = CONSOLE_ATTACHED_ERROR;
                } else if (attachError == ERROR_ACCESS_DENIED) {
                    // Already attached
                    consoleAttached = CONSOLE_ATTACHED_CONSOLE;
                } else {
                    OutputDebugString(TEXT("Error attaching console\r\n"));
                    consoleAttached = CONSOLE_ATTACHED_ERROR;
                }
            } else {
                // Newly attached
                consoleAttached = CONSOLE_ATTACHED_CONSOLE;
            }

            if (consoleAttached == CONSOLE_ATTACHED_CONSOLE) {
                stderrHandle = GetStdHandle(STD_ERROR_HANDLE);

                if (GetConsoleMode(stderrHandle, &consoleMode) == 0) {
                    // WriteConsole fails if the output is redirected to a file. Must use WriteFile instead.
                    consoleAttached = CONSOLE_ATTACHED_FILE;
                }
            }
        }
#endif // !defined(SDL_PLATFORM_GDK)
        length = SDL_strlen(GetLogPriorityPrefix(priority)) + SDL_strlen(message) + 1 + 1 + 1;
        output = SDL_small_alloc(char, length, &isstack);
        if (!output) {
            return;
        }
        (void)SDL_snprintf(output, length, "%s%s\r\n", GetLogPriorityPrefix(priority), message);
        tstr = WIN_UTF8ToString(output);

        // Output to debugger
        OutputDebugString(tstr);

#if !defined(SDL_PLATFORM_GDK)
        // Screen output to stderr, if console was attached.
        if (consoleAttached == CONSOLE_ATTACHED_CONSOLE) {
            if (!WriteConsole(stderrHandle, tstr, (DWORD)SDL_tcslen(tstr), &charsWritten, NULL)) {
                OutputDebugString(TEXT("Error calling WriteConsole\r\n"));
                if (GetLastError() == ERROR_NOT_ENOUGH_MEMORY) {
                    OutputDebugString(TEXT("Insufficient heap memory to write message\r\n"));
                }
            }

        } else if (consoleAttached == CONSOLE_ATTACHED_FILE) {
            if (!WriteFile(stderrHandle, output, (DWORD)SDL_strlen(output), &charsWritten, NULL)) {
                OutputDebugString(TEXT("Error calling WriteFile\r\n"));
            }
        }
#endif // !defined(SDL_PLATFORM_GDK)

        SDL_free(tstr);
        SDL_small_free(output, isstack);
    }
#elif defined(SDL_PLATFORM_ANDROID)
    {
        char tag[32];

        SDL_snprintf(tag, SDL_arraysize(tag), "SDL/%s", GetCategoryPrefix(category));
        __android_log_write(SDL_android_priority[priority], tag, message);
    }
#elif defined(SDL_PLATFORM_APPLE) && (defined(SDL_VIDEO_DRIVER_COCOA) || defined(SDL_VIDEO_DRIVER_UIKIT))
    /* Technically we don't need Cocoa/UIKit, but that's where this function is defined for now.
     */
    extern void SDL_NSLog(const char *prefix, const char *text);
    {
        SDL_NSLog(GetLogPriorityPrefix(priority), message);
        return;
    }
#elif defined(SDL_PLATFORM_PSP) || defined(SDL_PLATFORM_PS2)
    {
        FILE *pFile;
        pFile = fopen("SDL_Log.txt", "a");
        if (pFile) {
            (void)fprintf(pFile, "%s%s\n", GetLogPriorityPrefix(priority), message);
            (void)fclose(pFile);
        }
    }
#elif defined(SDL_PLATFORM_VITA)
    {
        FILE *pFile;
        pFile = fopen("ux0:/data/SDL_Log.txt", "a");
        if (pFile) {
            (void)fprintf(pFile, "%s%s\n", GetLogPriorityPrefix(priority), message);
            (void)fclose(pFile);
        }
    }
#elif defined(SDL_PLATFORM_3DS)
    {
        FILE *pFile;
        pFile = fopen("sdmc:/3ds/SDL_Log.txt", "a");
        if (pFile) {
            (void)fprintf(pFile, "%s%s\n", GetLogPriorityPrefix(priority), message);
            (void)fclose(pFile);
        }
    }
#endif
#if defined(HAVE_STDIO_H) && \
    !(defined(SDL_PLATFORM_APPLE) && (defined(SDL_VIDEO_DRIVER_COCOA) || defined(SDL_VIDEO_DRIVER_UIKIT))) && \
    !(defined(SDL_PLATFORM_WIN32))
    (void)fprintf(stderr, "%s%s\n", GetLogPriorityPrefix(priority), message);
#endif
}

SDL_LogOutputFunction SDL_GetDefaultLogOutputFunction(void)
{
    return SDL_LogOutput;
}

void SDL_GetLogOutputFunction(SDL_LogOutputFunction *callback, void **userdata)
{
    SDL_LockMutex(SDL_log_function_lock);
    {
        if (callback) {
            *callback = SDL_log_function;
        }
        if (userdata) {
            *userdata = SDL_log_userdata;
        }
    }
    SDL_UnlockMutex(SDL_log_function_lock);
}

void SDL_SetLogOutputFunction(SDL_LogOutputFunction callback, void *userdata)
{
    SDL_LockMutex(SDL_log_function_lock);
    {
        SDL_log_function = callback;
        SDL_log_userdata = userdata;
    }
    SDL_UnlockMutex(SDL_log_function_lock);
}
