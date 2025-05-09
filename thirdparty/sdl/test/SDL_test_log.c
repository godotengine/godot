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

/*

 Used by the test framework and test cases.

*/

/* quiet windows compiler warnings */
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <SDL3/SDL_test.h>

#include <time.h> /* Needed for localtime() */

/* work around compiler warning on older GCCs. */
#if (defined(__GNUC__) && (__GNUC__ <= 2))
static size_t strftime_gcc2_workaround(char *s, size_t max, const char *fmt, const struct tm *tm)
{
    return strftime(s, max, fmt, tm);
}
#ifdef strftime
#undef strftime
#endif
#define strftime strftime_gcc2_workaround
#endif

/**
 * Converts unix timestamp to its ascii representation in localtime
 *
 * Note: Uses a static buffer internally, so the return value
 * isn't valid after the next call of this function. If you
 * want to retain the return value, make a copy of it.
 *
 * \param timestamp A Timestamp, i.e. time(0)
 *
 * \return Ascii representation of the timestamp in localtime in the format '08/23/01 14:55:02'
 */
static const char *SDLTest_TimestampToString(const time_t timestamp)
{
    time_t copy;
    static char buffer[64];
    struct tm *local;
    size_t result = 0;

    SDL_memset(buffer, 0, sizeof(buffer));
    copy = timestamp;
    local = localtime(&copy);
    result = strftime(buffer, sizeof(buffer), "%x %X", local);
    if (result == 0) {
        return "";
    }

    return buffer;
}

/*
 * Prints given message with a timestamp in the TEST category and INFO priority.
 */
void SDLTest_Log(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list list;
    char logMessage[SDLTEST_MAX_LOGMESSAGE_LENGTH];

    /* Print log message into a buffer */
    SDL_memset(logMessage, 0, SDLTEST_MAX_LOGMESSAGE_LENGTH);
    va_start(list, fmt);
    (void)SDL_vsnprintf(logMessage, SDLTEST_MAX_LOGMESSAGE_LENGTH - 1, fmt, list);
    va_end(list);

    /* Log with timestamp and newline */
    SDL_LogMessage(SDL_LOG_CATEGORY_TEST, SDL_LOG_PRIORITY_INFO, " %s: %s", SDLTest_TimestampToString(time(0)), logMessage);
}

/*
 * Prints given message with a timestamp in the TEST category and the ERROR priority.
 */
void SDLTest_LogError(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list list;
    char logMessage[SDLTEST_MAX_LOGMESSAGE_LENGTH];

    /* Print log message into a buffer */
    SDL_memset(logMessage, 0, SDLTEST_MAX_LOGMESSAGE_LENGTH);
    va_start(list, fmt);
    (void)SDL_vsnprintf(logMessage, SDLTEST_MAX_LOGMESSAGE_LENGTH - 1, fmt, list);
    va_end(list);

    /* Log with timestamp and newline */
    SDL_LogMessage(SDL_LOG_CATEGORY_TEST, SDL_LOG_PRIORITY_ERROR, "%s: %s", SDLTest_TimestampToString(time(0)), logMessage);
}

static char nibble_to_char(Uint8 nibble)
{
    if (nibble < 0xa) {
        return '0' + nibble;
    } else {
        return 'a' + nibble - 10;
    }
}

void SDLTest_LogEscapedString(const char *prefix, const void *buffer, size_t size)
{
    const Uint8 *data = buffer;
    char logMessage[SDLTEST_MAX_LOGMESSAGE_LENGTH];

    if (data) {
        size_t i;
        size_t pos = 0;
        #define NEED_X_CHARS(N) \
            if (pos + (N) > sizeof(logMessage) - 2) { \
                break;                                \
            }

        logMessage[pos++] = '"';
        for (i = 0; i < size; i++) {
            Uint8 c = data[i];
            size_t pos_start = pos;
            switch (c) {
            case '\0':
                NEED_X_CHARS(2);
                logMessage[pos++] = '\\';
                logMessage[pos++] = '0';
                break;
            case '"':
                NEED_X_CHARS(2);
                logMessage[pos++] = '\\';
                logMessage[pos++] = '"';
                break;
            case '\n':
                NEED_X_CHARS(2);
                logMessage[pos++] = '\\';
                logMessage[pos++] = 'n';
                break;
            case '\r':
                NEED_X_CHARS(2);
                logMessage[pos++] = '\\';
                logMessage[pos++] = 'r';
                break;
            case '\t':
                NEED_X_CHARS(2);
                logMessage[pos++] = '\\';
                logMessage[pos++] = 't';
                break;
            case '\f':
                NEED_X_CHARS(2);
                logMessage[pos++] = '\\';
                logMessage[pos++] = 'f';
                break;
            case '\b':
                NEED_X_CHARS(2);
                logMessage[pos++] = '\\';
                logMessage[pos++] = 'b';
                break;
            case '\\':
                NEED_X_CHARS(2);
                logMessage[pos++] = '\\';
                logMessage[pos++] = '\\';
                break;
            default:
                if (SDL_isprint(c)) {
                    NEED_X_CHARS(1);
                    logMessage[pos++] = c;
                } else {
                    NEED_X_CHARS(4);
                    logMessage[pos++] = '\\';
                    logMessage[pos++] = 'x';
                    logMessage[pos++] = nibble_to_char(c >> 4);
                    logMessage[pos++] = nibble_to_char(c & 0xf);
                }
                break;
            }
            if (pos == pos_start) {
                break;
            }
        }
        if (i < size) {
            logMessage[sizeof(logMessage) - 4] = '.';
            logMessage[sizeof(logMessage) - 3] = '.';
            logMessage[sizeof(logMessage) - 2] = '.';
            logMessage[sizeof(logMessage) - 1] = '\0';
        } else {
            logMessage[pos++] = '"';
            logMessage[pos] = '\0';
        }
    } else {
        SDL_strlcpy(logMessage, "(nil)", sizeof(logMessage));
    }

    SDLTest_Log("%s%s", prefix, logMessage);
}
