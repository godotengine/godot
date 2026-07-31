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

// Simple error handling in SDL

#include "SDL_error_c.h"

bool SDL_SetError(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;
    bool result;

    va_start(ap, fmt);
    result = SDL_SetErrorV(fmt, ap);
    va_end(ap);
    return result;
}

bool SDL_SetErrorV(SDL_PRINTF_FORMAT_STRING const char *fmt, va_list ap)
{
    // Ignore call if invalid format pointer was passed
    if (fmt) {
        int result;
        SDL_error *error = SDL_GetErrBuf(true);
        va_list ap2;

        error->error = SDL_ErrorCodeGeneric;

        va_copy(ap2, ap);
        result = SDL_vsnprintf(error->str, error->len, fmt, ap2);
        va_end(ap2);

        if (result >= 0 && (size_t)result >= error->len && error->realloc_func) {
            size_t len = (size_t)result + 1;
            char *str = (char *)error->realloc_func(error->str, len);
            if (str) {
                error->str = str;
                error->len = len;
                va_copy(ap2, ap);
                (void)SDL_vsnprintf(error->str, error->len, fmt, ap2);
                va_end(ap2);
            }
        }

// Enable this if you want to see all errors printed as they occur.
// Note that there are many recoverable errors that may happen internally and
// can be safely ignored if the public API doesn't return an error code.
#if 0
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "%s", error->str);
#endif
    }

    return false;
}

const char *SDL_GetError(void)
{
    const SDL_error *error = SDL_GetErrBuf(false);

    if (!error) {
        return "";
    }

    switch (error->error) {
    case SDL_ErrorCodeGeneric:
        return error->str;
    case SDL_ErrorCodeOutOfMemory:
        return "Out of memory";
    default:
        return "";
    }
}

bool SDL_ClearError(void)
{
    SDL_error *error = SDL_GetErrBuf(false);

    if (error) {
        error->error = SDL_ErrorCodeNone;
    }
    return true;
}

bool SDL_OutOfMemory(void)
{
    SDL_error *error = SDL_GetErrBuf(true);

    if (error) {
        error->error = SDL_ErrorCodeOutOfMemory;
    }
    return false;
}

