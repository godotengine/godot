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
 *  Logging related functions of SDL test framework.
 *
 *  This code is a part of the SDL test library, not the main SDL library.
 */

/*
 *
 *  Wrapper to log in the TEST category
 *
 */

#ifndef SDL_test_log_h_
#define SDL_test_log_h_

#include <SDL3/SDL_stdinc.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Prints given message with a timestamp in the TEST category and INFO priority.
 *
 * \param fmt Message to be logged
 */
void SDLCALL SDLTest_Log(SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(1);

/**
 * Prints given prefix and buffer.
 * Non-printible characters in the raw data are substituted by printible alternatives.
 *
 * \param prefix Prefix message.
 * \param buffer Raw data to be escaped.
 * \param size Number of bytes in buffer.
 */
void SDLCALL SDLTest_LogEscapedString(const char *prefix, const void *buffer, size_t size);

/**
 * Prints given message with a timestamp in the TEST category and the ERROR priority.
 *
 * \param fmt Message to be logged
 */
void SDLCALL SDLTest_LogError(SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(1);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_test_log_h_ */
