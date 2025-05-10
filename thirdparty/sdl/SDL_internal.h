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
#ifndef SDL_internal_h_
#define SDL_internal_h_

// Many of SDL's features require _GNU_SOURCE on various platforms
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

// Need this so Linux systems define fseek64o, ftell64o and off64_t
#ifndef _LARGEFILE64_SOURCE
#define _LARGEFILE64_SOURCE 1
#endif

/* This is for a variable-length array at the end of a struct:
    struct x { int y; char z[SDL_VARIABLE_LENGTH_ARRAY]; };
   Use this because GCC 2 needs different magic than other compilers. */
#if (defined(__GNUC__) && (__GNUC__ <= 2)) || defined(__CC_ARM) || defined(__cplusplus)
#define SDL_VARIABLE_LENGTH_ARRAY 1
#else
#define SDL_VARIABLE_LENGTH_ARRAY
#endif

#if (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))) || defined(__clang__)
#define HAVE_GCC_DIAGNOSTIC_PRAGMA 1
#endif

#ifdef _MSC_VER // We use constant comparison for generated code
#pragma warning(disable : 6326)
#endif

#ifdef _MSC_VER // SDL_MAX_SMALL_ALLOC_STACKSIZE is smaller than _ALLOCA_S_THRESHOLD and should be generally safe
#pragma warning(disable : 6255)
#endif
#define SDL_MAX_SMALL_ALLOC_STACKSIZE          128
#define SDL_small_alloc(type, count, pisstack) ((*(pisstack) = ((sizeof(type) * (count)) < SDL_MAX_SMALL_ALLOC_STACKSIZE)), (*(pisstack) ? SDL_stack_alloc(type, count) : (type *)SDL_malloc(sizeof(type) * (count))))
#define SDL_small_free(ptr, isstack) \
    if ((isstack)) {                 \
        SDL_stack_free(ptr);         \
    } else {                         \
        SDL_free(ptr);               \
    }

#include "SDL_build_config.h"

#include "dynapi/SDL_dynapi.h"

#if SDL_DYNAMIC_API
#include "dynapi/SDL_dynapi_overrides.h"
/* force SDL_DECLSPEC off...it's all internal symbols now.
   These will have actual #defines during SDL_dynapi.c only */
#ifdef SDL_DECLSPEC
#undef SDL_DECLSPEC
#endif
#define SDL_DECLSPEC
#endif

#ifdef SDL_PLATFORM_APPLE
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1 // for memset_pattern4()
#endif
#include <Availability.h>

#ifndef __IPHONE_OS_VERSION_MAX_ALLOWED
#define __IPHONE_OS_VERSION_MAX_ALLOWED 0
#endif
#ifndef __APPLETV_OS_VERSION_MAX_ALLOWED
#define __APPLETV_OS_VERSION_MAX_ALLOWED 0
#endif
#ifndef __MAC_OS_X_VERSION_MAX_ALLOWED
#define __MAC_OS_X_VERSION_MAX_ALLOWED 0
#endif
#endif // SDL_PLATFORM_APPLE

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_STDIO_H
#include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#elif defined(HAVE_MALLOC_H)
#include <malloc.h>
#endif
#ifdef HAVE_STDDEF_H
#include <stddef.h>
#endif
#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif
#ifdef HAVE_STRING_H
#ifdef HAVE_MEMORY_H
#include <memory.h>
#endif
#include <string.h>
#endif
#ifdef HAVE_STRINGS_H
#include <strings.h>
#endif
#ifdef HAVE_WCHAR_H
#include <wchar.h>
#endif
#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#elif defined(HAVE_STDINT_H)
#include <stdint.h>
#endif
#ifdef HAVE_MATH_H
#include <math.h>
#endif
#ifdef HAVE_FLOAT_H
#include <float.h>
#endif

// If you run into a warning that O_CLOEXEC is redefined, update the SDL configuration header for your platform to add HAVE_O_CLOEXEC
#ifndef HAVE_O_CLOEXEC
#define O_CLOEXEC 0
#endif

/* A few #defines to reduce SDL footprint.
   Only effective when library is statically linked. */

/* Optimized functions from 'SDL_blit_0.c'
   - blit with source bits_per_pixel < 8, palette */
#if !defined(SDL_HAVE_BLIT_0) && !defined(SDL_LEAN_AND_MEAN)
#define SDL_HAVE_BLIT_0 1
#endif

/* Optimized functions from 'SDL_blit_1.c'
   - blit with source bytes_per_pixel == 1, palette */
#if !defined(SDL_HAVE_BLIT_1) && !defined(SDL_LEAN_AND_MEAN)
#define SDL_HAVE_BLIT_1 1
#endif

/* Optimized functions from 'SDL_blit_A.c'
   - blit with 'SDL_BLENDMODE_BLEND' blending mode */
#if !defined(SDL_HAVE_BLIT_A) && !defined(SDL_LEAN_AND_MEAN)
#define SDL_HAVE_BLIT_A 1
#endif

/* Optimized functions from 'SDL_blit_N.c'
   - blit with COLORKEY mode, or nothing */
#if !defined(SDL_HAVE_BLIT_N) && !defined(SDL_LEAN_AND_MEAN)
#define SDL_HAVE_BLIT_N 1
#endif

/* Optimized functions from 'SDL_blit_N.c'
   - RGB565 conversion with Lookup tables */
#if !defined(SDL_HAVE_BLIT_N_RGB565) && !defined(SDL_LEAN_AND_MEAN)
#define SDL_HAVE_BLIT_N_RGB565 1
#endif

/* Optimized functions from 'SDL_blit_AUTO.c'
   - blit with modulate color, modulate alpha, any blending mode
   - scaling or not */
#if !defined(SDL_HAVE_BLIT_AUTO) && !defined(SDL_LEAN_AND_MEAN)
#define SDL_HAVE_BLIT_AUTO 1
#endif

/* Run-Length-Encoding
   - SDL_SetSurfaceColorKey() called with SDL_RLEACCEL flag */
#if !defined(SDL_HAVE_RLE) && !defined(SDL_LEAN_AND_MEAN)
#define SDL_HAVE_RLE 1
#endif

/* Software SDL_Renderer
   - creation of software renderer
   - *not* general blitting functions
   - {blend,draw}{fillrect,line,point} internal functions */
#if !defined(SDL_VIDEO_RENDER_SW) && !defined(SDL_LEAN_AND_MEAN)
#define SDL_VIDEO_RENDER_SW 1
#endif

/* STB image conversion */
#if !defined(SDL_HAVE_STB) && !defined(SDL_LEAN_AND_MEAN)
#define SDL_HAVE_STB 1
#endif

/* YUV formats
   - handling of YUV surfaces
   - blitting and conversion functions */
#if !defined(SDL_HAVE_YUV) && !defined(SDL_LEAN_AND_MEAN)
#define SDL_HAVE_YUV 1
#endif

#ifdef SDL_CAMERA_DISABLED
#undef SDL_CAMERA_DRIVER_ANDROID
#undef SDL_CAMERA_DRIVER_COREMEDIA
#undef SDL_CAMERA_DRIVER_DUMMY
#undef SDL_CAMERA_DRIVER_EMSCRIPTEN
#undef SDL_CAMERA_DRIVER_MEDIAFOUNDATION
#undef SDL_CAMERA_DRIVER_PIPEWIRE
#undef SDL_CAMERA_DRIVER_V4L2
#undef SDL_CAMERA_DRIVER_VITA
#endif

#ifdef SDL_RENDER_DISABLED
#undef SDL_VIDEO_RENDER_SW
#undef SDL_VIDEO_RENDER_D3D
#undef SDL_VIDEO_RENDER_D3D11
#undef SDL_VIDEO_RENDER_D3D12
#undef SDL_VIDEO_RENDER_GPU
#undef SDL_VIDEO_RENDER_METAL
#undef SDL_VIDEO_RENDER_OGL
#undef SDL_VIDEO_RENDER_OGL_ES2
#undef SDL_VIDEO_RENDER_PS2
#undef SDL_VIDEO_RENDER_PSP
#undef SDL_VIDEO_RENDER_VITA_GXM
#undef SDL_VIDEO_RENDER_VULKAN
#endif // SDL_RENDER_DISABLED

#ifdef SDL_GPU_DISABLED
#undef SDL_GPU_D3D12
#undef SDL_GPU_METAL
#undef SDL_GPU_VULKAN
#undef SDL_VIDEO_RENDER_GPU
#endif // SDL_GPU_DISABLED

#if !defined(HAVE_LIBC)
// If not using _any_ C runtime, these have to be defined before SDL_thread.h
// gets included, so internal SDL_CreateThread calls will not try to reference
// the (unavailable and unneeded) _beginthreadex/_endthreadex functions.
#define SDL_BeginThreadFunction NULL
#define SDL_EndThreadFunction NULL
#endif

#ifdef SDL_NOLONGLONG
#error We cannot build a valid SDL3 library without long long support
#endif

/* Enable internal definitions in SDL API headers */
#define SDL_INTERNAL

#include <SDL3/SDL.h>
#include <SDL3/SDL_intrin.h>

#define SDL_MAIN_NOIMPL // don't drag in header-only implementation of SDL_main
#include <SDL3/SDL_main.h>

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

#include "SDL_utils_c.h"
#include "SDL_hashtable.h"

// Do any initialization that needs to happen before threads are started
extern void SDL_InitMainThread(void);

/* The internal implementations of these functions have up to nanosecond precision.
   We can expose these functions as part of the API if we want to later.
*/
extern bool SDLCALL SDL_WaitSemaphoreTimeoutNS(SDL_Semaphore *sem, Sint64 timeoutNS);
extern bool SDLCALL SDL_WaitConditionTimeoutNS(SDL_Condition *cond, SDL_Mutex *mutex, Sint64 timeoutNS);
extern bool SDLCALL SDL_WaitEventTimeoutNS(SDL_Event *event, Sint64 timeoutNS);

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif

#endif // SDL_internal_h_
