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

/* This file defines a structure that carries language-independent
   error messages
*/

#ifndef SDL_error_c_h_
#define SDL_error_c_h_

typedef enum
{
    SDL_ErrorCodeNone,
    SDL_ErrorCodeGeneric,
    SDL_ErrorCodeOutOfMemory,
} SDL_ErrorCode;

typedef struct SDL_error
{
    SDL_ErrorCode error;
    char *str;
    size_t len;
    SDL_realloc_func realloc_func;
    SDL_free_func free_func;
} SDL_error;

// Defined in SDL_thread.c
extern SDL_error *SDL_GetErrBuf(bool create);

// Macros to save and restore error values
#define SDL_PushError() \
    char *saved_error = SDL_strdup(SDL_GetError())

#define SDL_PopError()                          \
    do {                                        \
        if (saved_error) {                      \
            SDL_SetError("%s", saved_error);    \
            SDL_free(saved_error);              \
        }                                       \
    } while (0)

#endif // SDL_error_c_h_
