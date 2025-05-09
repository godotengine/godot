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
 *  Memory tracking related functions of SDL test framework.
 *
 *  This code is a part of the SDL test library, not the main SDL library.
 */

#ifndef SDL_test_memory_h_
#define SDL_test_memory_h_

#include <SDL3/SDL_stdinc.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Start tracking SDL memory allocations
 *
 * \note This should be called before any other SDL functions for complete tracking coverage
 */
void SDLCALL SDLTest_TrackAllocations(void);

/**
 * Fill allocations with random data
 *
 * \note This implicitly calls SDLTest_TrackAllocations()
 */
void SDLCALL SDLTest_RandFillAllocations(void);

/**
 * Print a log of any outstanding allocations
 *
 * \note This can be called after SDL_Quit()
 */
void SDLCALL SDLTest_LogAllocations(void);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_test_memory_h_ */
