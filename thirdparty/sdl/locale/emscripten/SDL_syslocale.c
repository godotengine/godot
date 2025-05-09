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

#include <emscripten.h>

#include "SDL_internal.h"
#include "../SDL_syslocale.h"

bool SDL_SYS_GetPreferredLocales(char *buf, size_t buflen)
{
    /* *INDENT-OFF* */ // clang-format off
    EM_ASM({
        var buf = $0;
        var buflen = $1;
        var list = undefined;

        if (navigator.languages && navigator.languages.length) {
            list = navigator.languages;
        } else {
            var oneOfThese = navigator.userLanguage || navigator.language || navigator.browserLanguage || navigator.systemLanguage;
            if (oneOfThese !== undefined) {
                list = [ oneOfThese ];
            }
        }

        if (list === undefined) {
            return;  // we've got nothing.
        }

        var str = "";  // Can't do list.join() because we need to fit in buflen.
        for (var i = 0; i < list.length; i++) {
            var item = list[i];
            if ((str.length + item.length + 1) > buflen) {
                break;   // don't add, we're out of space.
            }
            if (str.length > 0) {
                str += ",";
            }
            str += item;
        }

        str = str.replace(/-/g, "_");
        if (buflen > str.length) {
            buflen = str.length;  // clamp to size of string.
        }

        for (var i = 0; i < buflen; i++) {
            setValue(buf + i, str.charCodeAt(i), "i8");  // fill in C array.
        }
    }, buf, buflen);
    /* *INDENT-ON* */ // clang-format on
    return true;
}
