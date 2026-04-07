/*
 * Copyright (c) 2020 - 2026 ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _TVG_COMMON_H_
#define _TVG_COMMON_H_

#ifdef _WIN32
    #if defined(WINAPI_ENTRY)
        #if (WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP)
            #include <windows.h>
        #endif
    #elif !defined(APIENTRY) && !defined(__CYGWIN__) && !defined(__SCITECH_SNAP__)
        #include <windows.h>
    #endif
#endif
#include <string>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "config.h"
#include "thorvg.h"

using namespace std;
using namespace tvg;

//for MSVC Compat
#ifdef _MSC_VER
    #define TVG_UNUSED
    #define strncasecmp _strnicmp
    #define strcasecmp _stricmp
    #define strtok_r strtok_s
#else
    #define TVG_UNUSED __attribute__ ((__unused__))
#endif

// Portable 'fallthrough' attribute
#if __has_cpp_attribute(fallthrough)
    #ifdef _MSC_VER
        #define TVG_FALLTHROUGH [[fallthrough]];
    #else
        #define TVG_FALLTHROUGH __attribute__ ((fallthrough));
    #endif
#else
    #define TVG_FALLTHROUGH
#endif

#if defined(_MSC_VER) && defined(__clang__)
    #define strncpy strncpy_s
#endif

namespace tvg {

    enum class FileType { Png = 0, Jpg, Webp, Svg, Lot, Ttf, Raw, Gif, Unknown };

    #ifdef THORVG_LOG_ENABLED
        constexpr auto ErrorColor = "\033[31m";  //red
        constexpr auto ErrorBgColor = "\033[41m";//bg red
        constexpr auto LogColor = "\033[32m";    //green
        constexpr auto LogBgColor = "\033[42m";  //bg green
        constexpr auto GreyColor = "\033[90m";   //grey
        constexpr auto ResetColors = "\033[0m";  //default
        #define TVGERR(tag, fmt, ...) fprintf(stderr, "%s[E]%s %s" tag "%s (%s %d): %s" fmt "\n", ErrorBgColor, ResetColors, ErrorColor, GreyColor, __FILE__, __LINE__, ResetColors, ##__VA_ARGS__)
        #define TVGLOG(tag, fmt, ...) fprintf(stdout, "%s[L]%s %s" tag "%s (%s %d): %s" fmt "\n", LogBgColor, ResetColors, LogColor, GreyColor, __FILE__, __LINE__, ResetColors, ##__VA_ARGS__)
    #else
        #define TVGERR(...) do {} while(0)
        #define TVGLOG(...) do {} while(0)
    #endif

    template<typename T>
    static inline T* to(const Paint* p)
    {
        return static_cast<T*>(const_cast<Paint*>(p));
    }

    uint16_t THORVG_VERSION_NUMBER();

    extern int engineInit;
}

#include "tvgAllocator.h"

#endif //_TVG_COMMON_H_
