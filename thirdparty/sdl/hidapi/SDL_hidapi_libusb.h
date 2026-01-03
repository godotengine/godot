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
#pragma push_macro("malloc")
#pragma push_macro("realloc")
#pragma push_macro("free")
#pragma push_macro("iconv_t")
#pragma push_macro("iconv")
#pragma push_macro("iconv_open")
#pragma push_macro("iconv_close")
#pragma push_macro("setlocale")
#pragma push_macro("snprintf")
#pragma push_macro("strcmp")
#pragma push_macro("strdup")
#pragma push_macro("strncpy")
#pragma push_macro("tolower")
#pragma push_macro("wcscmp")
#pragma push_macro("wcsdup")
#pragma push_macro("wcsncpy")

#undef calloc
#undef malloc
#undef realloc
#undef free
#undef iconv_t
#undef iconv
#undef iconv_open
#undef iconv_close
#undef setlocale
#undef snprintf
#undef strcmp
#undef strdup
#undef strncpy
#undef tolower
#undef wcscmp
#undef wcsdup
#undef wcsncpy

#define calloc          SDL_calloc
#define malloc          SDL_malloc
#define realloc         SDL_realloc
#define free            SDL_free
#define iconv_t         SDL_iconv_t
#ifndef ICONV_CONST
#define ICONV_CONST
#define UNDEF_ICONV_CONST
#endif
#define iconv(a,b,c,d,e) SDL_iconv(a, (const char **)b, c, d, e)
#define iconv_open      SDL_iconv_open
#define iconv_close     SDL_iconv_close
#define setlocale(X, Y) NULL
#define snprintf        SDL_snprintf
#define strcmp          SDL_strcmp
#define strdup          SDL_strdup
#define strncpy         SDL_strlcpy
#define tolower         SDL_tolower
#define wcscmp          SDL_wcscmp
#define wcsdup          SDL_wcsdup
#define wcsncpy         SDL_wcslcpy


#ifndef SDL_PLATFORM_FREEBSD
/* this is awkwardly inlined, so we need to re-implement it here
 * so we can override the libusb_control_transfer call */
static int SDL_libusb_get_string_descriptor(libusb_device_handle *dev,
                                 uint8_t descriptor_index, uint16_t lang_id,
                                 unsigned char *data, int length)
{
    return libusb_control_transfer(dev, LIBUSB_ENDPOINT_IN | 0x0, LIBUSB_REQUEST_GET_DESCRIPTOR, (LIBUSB_DT_STRING << 8) | descriptor_index, lang_id,
                                   data, (uint16_t)length, 1000); /* Endpoint 0 IN */
}
#define libusb_get_string_descriptor SDL_libusb_get_string_descriptor
#endif /* SDL_PLATFORM_FREEBSD */

#define HIDAPI_THREAD_MODEL_INCLUDE "hidapi_thread_sdl.h"
#ifndef LIBUSB_API_VERSION
#ifdef LIBUSBX_API_VERSION
#define LIBUSB_API_VERSION LIBUSBX_API_VERSION
#else
#define LIBUSB_API_VERSION 0x0
#endif
#endif
/* we need libusb >= 1.0.16 because of libusb_get_port_numbers */
/* we don't need libusb_wrap_sys_device: */
#define HIDAPI_TARGET_LIBUSB_API_VERSION 0x01000102

#undef HIDAPI_H__
#include "libusb/hid.c"

/* restore libc function macros */
#ifdef UNDEF_ICONV_CONST
#undef ICONV_CONST
#undef UNDEF_ICONV_CONST
#endif
#pragma pop_macro("calloc")
#pragma pop_macro("malloc")
#pragma pop_macro("realloc")
#pragma pop_macro("free")
#pragma pop_macro("iconv_t")
#pragma pop_macro("iconv")
#pragma pop_macro("iconv_open")
#pragma pop_macro("iconv_close")
#pragma pop_macro("setlocale")
#pragma pop_macro("snprintf")
#pragma pop_macro("strcmp")
#pragma pop_macro("strdup")
#pragma pop_macro("strncpy")
#pragma pop_macro("tolower")
#pragma pop_macro("wcscmp")
#pragma pop_macro("wcsdup")
#pragma pop_macro("wcsncpy")
