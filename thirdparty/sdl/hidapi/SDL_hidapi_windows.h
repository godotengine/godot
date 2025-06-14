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

/* Define standard library functions in terms of SDL */

/* #pragma push_macro/pop_macro works correctly only as of gcc >= 4.4.3
   clang-3.0 _seems_ to be OK. */
#pragma push_macro("calloc")
#pragma push_macro("free")
#pragma push_macro("malloc")
#pragma push_macro("memcmp")
#pragma push_macro("swprintf")
#pragma push_macro("towupper")
#pragma push_macro("wcscmp")
#pragma push_macro("_wcsdup")
#pragma push_macro("wcslen")
#pragma push_macro("wcsncpy")
#pragma push_macro("wcsstr")
#pragma push_macro("wcstol")

#undef calloc
#undef free
#undef malloc
#undef memcmp
#undef swprintf
#undef towupper
#undef wcscmp
#undef _wcsdup
#undef wcslen
#undef wcsncpy
#undef wcsstr
#undef wcstol

#define calloc      SDL_calloc
#define free        SDL_free
#define malloc      SDL_malloc
#define memcmp      SDL_memcmp
#define swprintf    SDL_swprintf
#define towupper    (wchar_t)SDL_toupper
#define wcscmp      SDL_wcscmp
#define _wcsdup     SDL_wcsdup
#define wcslen      SDL_wcslen
#define wcsncpy     SDL_wcslcpy
#define wcsstr      SDL_wcsstr
#define wcstol      SDL_wcstol

// These functions conflict when linking both SDL and hidapi statically
#define hid_winapi_descriptor_reconstruct_pp_data SDL_hid_winapi_descriptor_reconstruct_pp_data
#define hid_winapi_get_container_id SDL_hid_winapi_get_container_id

#undef HIDAPI_H__
#include "windows/hid.c"
#define HAVE_PLATFORM_BACKEND 1
#define udev_ctx              1

/* restore libc function macros */
#pragma pop_macro("calloc")
#pragma pop_macro("free")
#pragma pop_macro("malloc")
#pragma pop_macro("memcmp")
#pragma pop_macro("swprintf")
#pragma pop_macro("towupper")
#pragma pop_macro("wcscmp")
#pragma pop_macro("_wcsdup")
#pragma pop_macro("wcslen")
#pragma pop_macro("wcsncpy")
#pragma pop_macro("wcsstr")
#pragma pop_macro("wcstol")
