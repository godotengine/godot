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
#include "../SDL_syslocale.h"

static void normalize_locale_str(char *dst, char *str, size_t buflen)
{
    char *ptr;

    ptr = SDL_strchr(str, '.'); // chop off encoding if specified.
    if (ptr) {
        *ptr = '\0';
    }

    ptr = SDL_strchr(str, '@'); // chop off extra bits if specified.
    if (ptr) {
        *ptr = '\0';
    }

    // The "C" locale isn't useful for our needs, ignore it if you see it.
    if ((str[0] == 'C') && (str[1] == '\0')) {
        return;
    }

    if (*str) {
        if (*dst) {
            SDL_strlcat(dst, ",", buflen); // SDL has these split by commas
        }
        SDL_strlcat(dst, str, buflen);
    }
}

static void normalize_locales(char *dst, char *src, size_t buflen)
{
    char *ptr;

    // entries are separated by colons
    while ((ptr = SDL_strchr(src, ':')) != NULL) {
        *ptr = '\0';
        normalize_locale_str(dst, src, buflen);
        src = ptr + 1;
    }
    normalize_locale_str(dst, src, buflen);
}

bool SDL_SYS_GetPreferredLocales(char *buf, size_t buflen)
{
    // !!! FIXME: should we be using setlocale()? Or some D-Bus thing?
    bool isstack;
    const char *envr;
    char *tmp;

    SDL_assert(buflen > 0);
    tmp = SDL_small_alloc(char, buflen, &isstack);
    if (!tmp) {
        return false;
    }

    *tmp = '\0';

    // LANG is the primary locale (maybe)
    envr = SDL_getenv("LANG");
    if (envr) {
        SDL_strlcpy(tmp, envr, buflen);
    }

    // fallback languages
    envr = SDL_getenv("LANGUAGE");
    if (envr) {
        if (*tmp) {
            SDL_strlcat(tmp, ":", buflen);
        }
        SDL_strlcat(tmp, envr, buflen);
    }

    if (*tmp == '\0') {
        SDL_SetError("LANG environment variable isn't set");
    } else {
        normalize_locales(buf, tmp, buflen);
    }

    SDL_small_free(tmp, isstack);
    return true;
}
