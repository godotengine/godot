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

/**
 * # CategoryLog
 *
 * Simple log messages with priorities and categories. A message's
 * SDL_LogPriority signifies how important the message is. A message's
 * SDL_LogCategory signifies from what domain it belongs to. Every category
 * has a minimum priority specified: when a message belongs to that category,
 * it will only be sent out if it has that minimum priority or higher.
 *
 * SDL's own logs are sent below the default priority threshold, so they are
 * quiet by default.
 *
 * You can change the log verbosity programmatically using
 * SDL_SetLogPriority() or with SDL_SetHint(SDL_HINT_LOGGING, ...), or with
 * the "SDL_LOGGING" environment variable. This variable is a comma separated
 * set of category=level tokens that define the default logging levels for SDL
 * applications.
 *
 * The category can be a numeric category, one of "app", "error", "assert",
 * "system", "audio", "video", "render", "input", "test", or `*` for any
 * unspecified category.
 *
 * The level can be a numeric level, one of "trace", "verbose", "debug",
 * "info", "warn", "error", "critical", or "quiet" to disable that category.
 *
 * You can omit the category if you want to set the logging level for all
 * categories.
 *
 * If this hint isn't set, the default log levels are equivalent to:
 *
 * `app=info,assert=warn,test=verbose,*=error`
 *
 * Here's where the messages go on different platforms:
 *
 * - Windows: debug output stream
 * - Android: log output
 * - Others: standard error output (stderr)
 *
 * You don't need to have a newline (`\n`) on the end of messages, the
 * functions will do that for you. For consistent behavior cross-platform, you
 * shouldn't have any newlines in messages, such as to log multiple lines in
 * one call; unusual platform-specific behavior can be observed in such usage.
 * Do one log call per line instead, with no newlines in messages.
 *
 * Each log call is atomic, so you won't see log messages cut off one another
 * when logging from multiple threads.
 */

#ifndef SDL_log_h_
#define SDL_log_h_

#include <SDL3/SDL_stdinc.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * The predefined log categories
 *
 * By default the application and gpu categories are enabled at the INFO
 * level, the assert category is enabled at the WARN level, test is enabled at
 * the VERBOSE level and all other categories are enabled at the ERROR level.
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_LogCategory
{
    SDL_LOG_CATEGORY_APPLICATION,
    SDL_LOG_CATEGORY_ERROR,
    SDL_LOG_CATEGORY_ASSERT,
    SDL_LOG_CATEGORY_SYSTEM,
    SDL_LOG_CATEGORY_AUDIO,
    SDL_LOG_CATEGORY_VIDEO,
    SDL_LOG_CATEGORY_RENDER,
    SDL_LOG_CATEGORY_INPUT,
    SDL_LOG_CATEGORY_TEST,
    SDL_LOG_CATEGORY_GPU,

    /* Reserved for future SDL library use */
    SDL_LOG_CATEGORY_RESERVED2,
    SDL_LOG_CATEGORY_RESERVED3,
    SDL_LOG_CATEGORY_RESERVED4,
    SDL_LOG_CATEGORY_RESERVED5,
    SDL_LOG_CATEGORY_RESERVED6,
    SDL_LOG_CATEGORY_RESERVED7,
    SDL_LOG_CATEGORY_RESERVED8,
    SDL_LOG_CATEGORY_RESERVED9,
    SDL_LOG_CATEGORY_RESERVED10,

    /* Beyond this point is reserved for application use, e.g.
       enum {
           MYAPP_CATEGORY_AWESOME1 = SDL_LOG_CATEGORY_CUSTOM,
           MYAPP_CATEGORY_AWESOME2,
           MYAPP_CATEGORY_AWESOME3,
           ...
       };
     */
    SDL_LOG_CATEGORY_CUSTOM
} SDL_LogCategory;

/**
 * The predefined log priorities
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_LogPriority
{
    SDL_LOG_PRIORITY_INVALID,
    SDL_LOG_PRIORITY_TRACE,
    SDL_LOG_PRIORITY_VERBOSE,
    SDL_LOG_PRIORITY_DEBUG,
    SDL_LOG_PRIORITY_INFO,
    SDL_LOG_PRIORITY_WARN,
    SDL_LOG_PRIORITY_ERROR,
    SDL_LOG_PRIORITY_CRITICAL,
    SDL_LOG_PRIORITY_COUNT
} SDL_LogPriority;


/**
 * Set the priority of all log categories.
 *
 * \param priority the SDL_LogPriority to assign.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_ResetLogPriorities
 * \sa SDL_SetLogPriority
 */
extern SDL_DECLSPEC void SDLCALL SDL_SetLogPriorities(SDL_LogPriority priority);

/**
 * Set the priority of a particular log category.
 *
 * \param category the category to assign a priority to.
 * \param priority the SDL_LogPriority to assign.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetLogPriority
 * \sa SDL_ResetLogPriorities
 * \sa SDL_SetLogPriorities
 */
extern SDL_DECLSPEC void SDLCALL SDL_SetLogPriority(int category, SDL_LogPriority priority);

/**
 * Get the priority of a particular log category.
 *
 * \param category the category to query.
 * \returns the SDL_LogPriority for the requested category.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetLogPriority
 */
extern SDL_DECLSPEC SDL_LogPriority SDLCALL SDL_GetLogPriority(int category);

/**
 * Reset all priorities to default.
 *
 * This is called by SDL_Quit().
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetLogPriorities
 * \sa SDL_SetLogPriority
 */
extern SDL_DECLSPEC void SDLCALL SDL_ResetLogPriorities(void);

/**
 * Set the text prepended to log messages of a given priority.
 *
 * By default SDL_LOG_PRIORITY_INFO and below have no prefix, and
 * SDL_LOG_PRIORITY_WARN and higher have a prefix showing their priority, e.g.
 * "WARNING: ".
 *
 * \param priority the SDL_LogPriority to modify.
 * \param prefix the prefix to use for that log priority, or NULL to use no
 *               prefix.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetLogPriorities
 * \sa SDL_SetLogPriority
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetLogPriorityPrefix(SDL_LogPriority priority, const char *prefix);

/**
 * Log a message with SDL_LOG_CATEGORY_APPLICATION and SDL_LOG_PRIORITY_INFO.
 *
 * \param fmt a printf() style message format string.
 * \param ... additional parameters matching % tokens in the `fmt` string, if
 *            any.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_LogCritical
 * \sa SDL_LogDebug
 * \sa SDL_LogError
 * \sa SDL_LogInfo
 * \sa SDL_LogMessage
 * \sa SDL_LogMessageV
 * \sa SDL_LogTrace
 * \sa SDL_LogVerbose
 * \sa SDL_LogWarn
 */
extern SDL_DECLSPEC void SDLCALL SDL_Log(SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(1);

/**
 * Log a message with SDL_LOG_PRIORITY_TRACE.
 *
 * \param category the category of the message.
 * \param fmt a printf() style message format string.
 * \param ... additional parameters matching % tokens in the **fmt** string,
 *            if any.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_Log
 * \sa SDL_LogCritical
 * \sa SDL_LogDebug
 * \sa SDL_LogError
 * \sa SDL_LogInfo
 * \sa SDL_LogMessage
 * \sa SDL_LogMessageV
 * \sa SDL_LogTrace
 * \sa SDL_LogVerbose
 * \sa SDL_LogWarn
 */
extern SDL_DECLSPEC void SDLCALL SDL_LogTrace(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(2);

/**
 * Log a message with SDL_LOG_PRIORITY_VERBOSE.
 *
 * \param category the category of the message.
 * \param fmt a printf() style message format string.
 * \param ... additional parameters matching % tokens in the **fmt** string,
 *            if any.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_Log
 * \sa SDL_LogCritical
 * \sa SDL_LogDebug
 * \sa SDL_LogError
 * \sa SDL_LogInfo
 * \sa SDL_LogMessage
 * \sa SDL_LogMessageV
 * \sa SDL_LogWarn
 */
extern SDL_DECLSPEC void SDLCALL SDL_LogVerbose(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(2);

/**
 * Log a message with SDL_LOG_PRIORITY_DEBUG.
 *
 * \param category the category of the message.
 * \param fmt a printf() style message format string.
 * \param ... additional parameters matching % tokens in the **fmt** string,
 *            if any.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_Log
 * \sa SDL_LogCritical
 * \sa SDL_LogError
 * \sa SDL_LogInfo
 * \sa SDL_LogMessage
 * \sa SDL_LogMessageV
 * \sa SDL_LogTrace
 * \sa SDL_LogVerbose
 * \sa SDL_LogWarn
 */
extern SDL_DECLSPEC void SDLCALL SDL_LogDebug(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(2);

/**
 * Log a message with SDL_LOG_PRIORITY_INFO.
 *
 * \param category the category of the message.
 * \param fmt a printf() style message format string.
 * \param ... additional parameters matching % tokens in the **fmt** string,
 *            if any.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_Log
 * \sa SDL_LogCritical
 * \sa SDL_LogDebug
 * \sa SDL_LogError
 * \sa SDL_LogMessage
 * \sa SDL_LogMessageV
 * \sa SDL_LogTrace
 * \sa SDL_LogVerbose
 * \sa SDL_LogWarn
 */
extern SDL_DECLSPEC void SDLCALL SDL_LogInfo(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(2);

/**
 * Log a message with SDL_LOG_PRIORITY_WARN.
 *
 * \param category the category of the message.
 * \param fmt a printf() style message format string.
 * \param ... additional parameters matching % tokens in the **fmt** string,
 *            if any.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_Log
 * \sa SDL_LogCritical
 * \sa SDL_LogDebug
 * \sa SDL_LogError
 * \sa SDL_LogInfo
 * \sa SDL_LogMessage
 * \sa SDL_LogMessageV
 * \sa SDL_LogTrace
 * \sa SDL_LogVerbose
 */
extern SDL_DECLSPEC void SDLCALL SDL_LogWarn(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(2);

/**
 * Log a message with SDL_LOG_PRIORITY_ERROR.
 *
 * \param category the category of the message.
 * \param fmt a printf() style message format string.
 * \param ... additional parameters matching % tokens in the **fmt** string,
 *            if any.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_Log
 * \sa SDL_LogCritical
 * \sa SDL_LogDebug
 * \sa SDL_LogInfo
 * \sa SDL_LogMessage
 * \sa SDL_LogMessageV
 * \sa SDL_LogTrace
 * \sa SDL_LogVerbose
 * \sa SDL_LogWarn
 */
extern SDL_DECLSPEC void SDLCALL SDL_LogError(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(2);

/**
 * Log a message with SDL_LOG_PRIORITY_CRITICAL.
 *
 * \param category the category of the message.
 * \param fmt a printf() style message format string.
 * \param ... additional parameters matching % tokens in the **fmt** string,
 *            if any.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_Log
 * \sa SDL_LogDebug
 * \sa SDL_LogError
 * \sa SDL_LogInfo
 * \sa SDL_LogMessage
 * \sa SDL_LogMessageV
 * \sa SDL_LogTrace
 * \sa SDL_LogVerbose
 * \sa SDL_LogWarn
 */
extern SDL_DECLSPEC void SDLCALL SDL_LogCritical(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(2);

/**
 * Log a message with the specified category and priority.
 *
 * \param category the category of the message.
 * \param priority the priority of the message.
 * \param fmt a printf() style message format string.
 * \param ... additional parameters matching % tokens in the **fmt** string,
 *            if any.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_Log
 * \sa SDL_LogCritical
 * \sa SDL_LogDebug
 * \sa SDL_LogError
 * \sa SDL_LogInfo
 * \sa SDL_LogMessageV
 * \sa SDL_LogTrace
 * \sa SDL_LogVerbose
 * \sa SDL_LogWarn
 */
extern SDL_DECLSPEC void SDLCALL SDL_LogMessage(int category,
                                            SDL_LogPriority priority,
                                            SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(3);

/**
 * Log a message with the specified category and priority.
 *
 * \param category the category of the message.
 * \param priority the priority of the message.
 * \param fmt a printf() style message format string.
 * \param ap a variable argument list.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_Log
 * \sa SDL_LogCritical
 * \sa SDL_LogDebug
 * \sa SDL_LogError
 * \sa SDL_LogInfo
 * \sa SDL_LogMessage
 * \sa SDL_LogTrace
 * \sa SDL_LogVerbose
 * \sa SDL_LogWarn
 */
extern SDL_DECLSPEC void SDLCALL SDL_LogMessageV(int category,
                                             SDL_LogPriority priority,
                                             SDL_PRINTF_FORMAT_STRING const char *fmt, va_list ap) SDL_PRINTF_VARARG_FUNCV(3);

/**
 * The prototype for the log output callback function.
 *
 * This function is called by SDL when there is new text to be logged. A mutex
 * is held so that this function is never called by more than one thread at
 * once.
 *
 * \param userdata what was passed as `userdata` to
 *                 SDL_SetLogOutputFunction().
 * \param category the category of the message.
 * \param priority the priority of the message.
 * \param message the message being output.
 *
 * \since This datatype is available since SDL 3.2.0.
 */
typedef void (SDLCALL *SDL_LogOutputFunction)(void *userdata, int category, SDL_LogPriority priority, const char *message);

/**
 * Get the default log output function.
 *
 * \returns the default log output callback.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetLogOutputFunction
 * \sa SDL_GetLogOutputFunction
 */
extern SDL_DECLSPEC SDL_LogOutputFunction SDLCALL SDL_GetDefaultLogOutputFunction(void);

/**
 * Get the current log output function.
 *
 * \param callback an SDL_LogOutputFunction filled in with the current log
 *                 callback.
 * \param userdata a pointer filled in with the pointer that is passed to
 *                 `callback`.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetDefaultLogOutputFunction
 * \sa SDL_SetLogOutputFunction
 */
extern SDL_DECLSPEC void SDLCALL SDL_GetLogOutputFunction(SDL_LogOutputFunction *callback, void **userdata);

/**
 * Replace the default log output function with one of your own.
 *
 * \param callback an SDL_LogOutputFunction to call instead of the default.
 * \param userdata a pointer that is passed to `callback`.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetDefaultLogOutputFunction
 * \sa SDL_GetLogOutputFunction
 */
extern SDL_DECLSPEC void SDLCALL SDL_SetLogOutputFunction(SDL_LogOutputFunction callback, void *userdata);


/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_log_h_ */
