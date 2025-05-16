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

#include "SDL_assert_c.h"
#include "video/SDL_sysvideo.h"

#if defined(SDL_PLATFORM_WINDOWS)
#ifndef WS_OVERLAPPEDWINDOW
#define WS_OVERLAPPEDWINDOW 0
#endif
#endif

#ifdef SDL_PLATFORM_EMSCRIPTEN
#include <emscripten.h>
#endif

// The size of the stack buffer to use for rendering assert messages.
#define SDL_MAX_ASSERT_MESSAGE_STACK 256

static SDL_AssertState SDLCALL SDL_PromptAssertion(const SDL_AssertData *data, void *userdata);

/*
 * We keep all triggered assertions in a singly-linked list so we can
 *  generate a report later.
 */
static SDL_AssertData *triggered_assertions = NULL;

#ifndef SDL_THREADS_DISABLED
static SDL_Mutex *assertion_mutex = NULL;
#endif

static SDL_AssertionHandler assertion_handler = SDL_PromptAssertion;
static void *assertion_userdata = NULL;

#ifdef __GNUC__
static void debug_print(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
#endif

static void debug_print(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    SDL_LogMessageV(SDL_LOG_CATEGORY_ASSERT, SDL_LOG_PRIORITY_WARN, fmt, ap);
    va_end(ap);
}

static void SDL_AddAssertionToReport(SDL_AssertData *data)
{
    /* (data) is always a static struct defined with the assert macros, so
       we don't have to worry about copying or allocating them. */
    data->trigger_count++;
    if (data->trigger_count == 1) { // not yet added?
        data->next = triggered_assertions;
        triggered_assertions = data;
    }
}

#if defined(SDL_PLATFORM_WINDOWS)
#define ENDLINE "\r\n"
#else
#define ENDLINE "\n"
#endif

static int SDL_RenderAssertMessage(char *buf, size_t buf_len, const SDL_AssertData *data)
{
    return SDL_snprintf(buf, buf_len,
                        "Assertion failure at %s (%s:%d), triggered %u %s:" ENDLINE "  '%s'",
                        data->function, data->filename, data->linenum,
                        data->trigger_count, (data->trigger_count == 1) ? "time" : "times",
                        data->condition);
}

static void SDL_GenerateAssertionReport(void)
{
    const SDL_AssertData *item = triggered_assertions;

    // only do this if the app hasn't assigned an assertion handler.
    if ((item) && (assertion_handler != SDL_PromptAssertion)) {
        debug_print("\n\nSDL assertion report.\n");
        debug_print("All SDL assertions between last init/quit:\n\n");

        while (item) {
            debug_print(
                "'%s'\n"
                "    * %s (%s:%d)\n"
                "    * triggered %u time%s.\n"
                "    * always ignore: %s.\n",
                item->condition, item->function, item->filename,
                item->linenum, item->trigger_count,
                (item->trigger_count == 1) ? "" : "s",
                item->always_ignore ? "yes" : "no");
            item = item->next;
        }
        debug_print("\n");

        SDL_ResetAssertionReport();
    }
}

/* This is not declared in any header, although it is shared between some
    parts of SDL, because we don't want anything calling it without an
    extremely good reason. */
#ifdef __WATCOMC__
extern void SDL_ExitProcess(int exitcode);
#pragma aux SDL_ExitProcess aborts;
#endif
extern SDL_NORETURN void SDL_ExitProcess(int exitcode);

#ifdef __WATCOMC__
static void SDL_AbortAssertion(void);
#pragma aux SDL_AbortAssertion aborts;
#endif
static SDL_NORETURN void SDL_AbortAssertion(void)
{
    SDL_Quit();
    SDL_ExitProcess(42);
}

static SDL_AssertState SDLCALL SDL_PromptAssertion(const SDL_AssertData *data, void *userdata)
{
    SDL_AssertState state = SDL_ASSERTION_ABORT;
    SDL_Window *window;
    SDL_MessageBoxData messagebox;
    SDL_MessageBoxButtonData buttons[] = {
        { 0, SDL_ASSERTION_RETRY, "Retry" },
        { 0, SDL_ASSERTION_BREAK, "Break" },
        { 0, SDL_ASSERTION_ABORT, "Abort" },
        { SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT,
          SDL_ASSERTION_IGNORE, "Ignore" },
        { SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT,
          SDL_ASSERTION_ALWAYS_IGNORE, "Always Ignore" }
    };
    int selected;

    char stack_buf[SDL_MAX_ASSERT_MESSAGE_STACK];
    char *message = stack_buf;
    size_t buf_len = sizeof(stack_buf);
    int len;

    (void)userdata; // unused in default handler.

    // Assume the output will fit...
    len = SDL_RenderAssertMessage(message, buf_len, data);

    // .. and if it didn't, try to allocate as much room as we actually need.
    if (len >= (int)buf_len) {
        if (SDL_size_add_check_overflow(len, 1, &buf_len)) {
            message = (char *)SDL_malloc(buf_len);
            if (message) {
                len = SDL_RenderAssertMessage(message, buf_len, data);
            } else {
                message = stack_buf;
            }
        }
    }

    // Something went very wrong
    if (len < 0) {
        if (message != stack_buf) {
            SDL_free(message);
        }
        return SDL_ASSERTION_ABORT;
    }

    debug_print("\n\n%s\n\n", message);

    // let env. variable override, so unit tests won't block in a GUI.
    const char *hint = SDL_GetHint(SDL_HINT_ASSERT);
    if (hint) {
        if (message != stack_buf) {
            SDL_free(message);
        }

        if (SDL_strcmp(hint, "abort") == 0) {
            return SDL_ASSERTION_ABORT;
        } else if (SDL_strcmp(hint, "break") == 0) {
            return SDL_ASSERTION_BREAK;
        } else if (SDL_strcmp(hint, "retry") == 0) {
            return SDL_ASSERTION_RETRY;
        } else if (SDL_strcmp(hint, "ignore") == 0) {
            return SDL_ASSERTION_IGNORE;
        } else if (SDL_strcmp(hint, "always_ignore") == 0) {
            return SDL_ASSERTION_ALWAYS_IGNORE;
        } else {
            return SDL_ASSERTION_ABORT; // oh well.
        }
    }

    // Leave fullscreen mode, if possible (scary!)
    window = SDL_GetToplevelForKeyboardFocus();
    if (window) {
        if (window->fullscreen_exclusive) {
            SDL_MinimizeWindow(window);
        } else {
            // !!! FIXME: ungrab the input if we're not fullscreen?
            // No need to mess with the window
            window = NULL;
        }
    }

    // Show a messagebox if we can, otherwise fall back to stdio
    SDL_zero(messagebox);
    messagebox.flags = SDL_MESSAGEBOX_WARNING;
    messagebox.window = window;
    messagebox.title = "Assertion Failed";
    messagebox.message = message;
    messagebox.numbuttons = SDL_arraysize(buttons);
    messagebox.buttons = buttons;

    if (SDL_ShowMessageBox(&messagebox, &selected)) {
        if (selected == -1) {
            state = SDL_ASSERTION_IGNORE;
        } else {
            state = (SDL_AssertState)selected;
        }
    } else {
#ifdef SDL_PLATFORM_PRIVATE_ASSERT
        SDL_PRIVATE_PROMPTASSERTION();
#elif defined(SDL_PLATFORM_EMSCRIPTEN)
        // This is nasty, but we can't block on a custom UI.
        for (;;) {
            bool okay = true;
            /* *INDENT-OFF* */ // clang-format off
            int reply = MAIN_THREAD_EM_ASM_INT({
                var str =
                    UTF8ToString($0) + '\n\n' +
                    'Abort/Retry/Ignore/AlwaysIgnore? [ariA] :';
                var reply = window.prompt(str, "i");
                if (reply === null) {
                    reply = "i";
                }
                return reply.length === 1 ? reply.charCodeAt(0) : -1;
            }, message);
            /* *INDENT-ON* */ // clang-format on

            switch (reply) {
            case 'a':
                state = SDL_ASSERTION_ABORT;
                break;
#if 0 // (currently) no break functionality on Emscripten
            case 'b':
                state = SDL_ASSERTION_BREAK;
                break;
#endif
            case 'r':
                state = SDL_ASSERTION_RETRY;
                break;
            case 'i':
                state = SDL_ASSERTION_IGNORE;
                break;
            case 'A':
                state = SDL_ASSERTION_ALWAYS_IGNORE;
                break;
            default:
                okay = false;
                break;
            }

            if (okay) {
                break;
            }
        }
#elif defined(HAVE_STDIO_H) && !defined(SDL_PLATFORM_3DS)
        // this is a little hacky.
        for (;;) {
            char buf[32];
            (void)fprintf(stderr, "Abort/Break/Retry/Ignore/AlwaysIgnore? [abriA] : ");
            (void)fflush(stderr);
            if (fgets(buf, sizeof(buf), stdin) == NULL) {
                break;
            }

            if (SDL_strncmp(buf, "a", 1) == 0) {
                state = SDL_ASSERTION_ABORT;
                break;
            } else if (SDL_strncmp(buf, "b", 1) == 0) {
                state = SDL_ASSERTION_BREAK;
                break;
            } else if (SDL_strncmp(buf, "r", 1) == 0) {
                state = SDL_ASSERTION_RETRY;
                break;
            } else if (SDL_strncmp(buf, "i", 1) == 0) {
                state = SDL_ASSERTION_IGNORE;
                break;
            } else if (SDL_strncmp(buf, "A", 1) == 0) {
                state = SDL_ASSERTION_ALWAYS_IGNORE;
                break;
            }
        }
#else
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_WARNING, "Assertion Failed", message, window);
#endif // HAVE_STDIO_H
    }

    // Re-enter fullscreen mode
    if (window) {
        SDL_RestoreWindow(window);
    }

    if (message != stack_buf) {
        SDL_free(message);
    }

    return state;
}

SDL_AssertState SDL_ReportAssertion(SDL_AssertData *data, const char *func, const char *file, int line)
{
    SDL_AssertState state = SDL_ASSERTION_IGNORE;
    static int assertion_running = 0;

#ifndef SDL_THREADS_DISABLED
    static SDL_SpinLock spinlock = 0;
    SDL_LockSpinlock(&spinlock);
    if (!assertion_mutex) { // never called SDL_Init()?
        assertion_mutex = SDL_CreateMutex();
        if (!assertion_mutex) {
            SDL_UnlockSpinlock(&spinlock);
            return SDL_ASSERTION_IGNORE; // oh well, I guess.
        }
    }
    SDL_UnlockSpinlock(&spinlock);

    SDL_LockMutex(assertion_mutex);
#endif // !SDL_THREADS_DISABLED

    // doing this because Visual C is upset over assigning in the macro.
    if (data->trigger_count == 0) {
        data->function = func;
        data->filename = file;
        data->linenum = line;
    }

    SDL_AddAssertionToReport(data);

    assertion_running++;
    if (assertion_running > 1) { // assert during assert! Abort.
        if (assertion_running == 2) {
            SDL_AbortAssertion();
        } else if (assertion_running == 3) { // Abort asserted!
            SDL_ExitProcess(42);
        } else {
            while (1) { // do nothing but spin; what else can you do?!
            }
        }
    }

    if (!data->always_ignore) {
        state = assertion_handler(data, assertion_userdata);
    }

    switch (state) {
    case SDL_ASSERTION_ALWAYS_IGNORE:
        state = SDL_ASSERTION_IGNORE;
        data->always_ignore = true;
        break;

    case SDL_ASSERTION_IGNORE:
    case SDL_ASSERTION_RETRY:
    case SDL_ASSERTION_BREAK:
        break; // macro handles these.

    case SDL_ASSERTION_ABORT:
        SDL_AbortAssertion();
        // break;  ...shouldn't return, but oh well.
    }

    assertion_running--;

#ifndef SDL_THREADS_DISABLED
    SDL_UnlockMutex(assertion_mutex);
#endif

    return state;
}

void SDL_AssertionsQuit(void)
{
#if SDL_ASSERT_LEVEL > 0
    SDL_GenerateAssertionReport();
#ifndef SDL_THREADS_DISABLED
    if (assertion_mutex) {
        SDL_DestroyMutex(assertion_mutex);
        assertion_mutex = NULL;
    }
#endif
#endif // SDL_ASSERT_LEVEL > 0
}

void SDL_SetAssertionHandler(SDL_AssertionHandler handler, void *userdata)
{
    if (handler != NULL) {
        assertion_handler = handler;
        assertion_userdata = userdata;
    } else {
        assertion_handler = SDL_PromptAssertion;
        assertion_userdata = NULL;
    }
}

const SDL_AssertData *SDL_GetAssertionReport(void)
{
    return triggered_assertions;
}

void SDL_ResetAssertionReport(void)
{
    SDL_AssertData *next = NULL;
    SDL_AssertData *item;
    for (item = triggered_assertions; item; item = next) {
        next = (SDL_AssertData *)item->next;
        item->always_ignore = false;
        item->trigger_count = 0;
        item->next = NULL;
    }

    triggered_assertions = NULL;
}

SDL_AssertionHandler SDL_GetDefaultAssertionHandler(void)
{
    return SDL_PromptAssertion;
}

SDL_AssertionHandler SDL_GetAssertionHandler(void **userdata)
{
    if (userdata) {
        *userdata = assertion_userdata;
    }
    return assertion_handler;
}
