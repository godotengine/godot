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


#ifdef SDL_memmove
#undef SDL_memmove
#endif
#if SDL_DYNAMIC_API
#define SDL_memmove SDL_memmove_REAL
#endif
void *SDL_memmove(SDL_OUT_BYTECAP(len) void *dst, SDL_IN_BYTECAP(len) const void *src, size_t len)
{
#if defined(__GNUC__) && (defined(HAVE_LIBC) && HAVE_LIBC)
    // Presumably this is well tuned for speed.
    return __builtin_memmove(dst, src, len);
#elif defined(HAVE_MEMMOVE)
    return memmove(dst, src, len);
#else
    char *srcp = (char *)src;
    char *dstp = (char *)dst;

    if (src < dst) {
        srcp += len - 1;
        dstp += len - 1;
        while (len--) {
            *dstp-- = *srcp--;
        }
    } else {
        while (len--) {
            *dstp++ = *srcp++;
        }
    }
    return dst;
#endif // HAVE_MEMMOVE
}


#ifndef HAVE_LIBC
// NOLINTNEXTLINE(readability-redundant-declaration)
extern void *memmove(void *dst, const void *src, size_t len);
#if defined(_MSC_VER) && !defined(__INTEL_LLVM_COMPILER)
#pragma intrinsic(memmove)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#pragma function(memmove)
#endif
// NOLINTNEXTLINE(readability-inconsistent-declaration-parameter-name)
void *memmove(void *dst, const void *src, size_t len)
{
    return SDL_memmove(dst, src, len);
}
#endif // !HAVE_LIBC

