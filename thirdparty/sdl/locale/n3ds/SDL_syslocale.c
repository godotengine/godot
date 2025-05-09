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

#include "../SDL_syslocale.h"
#include "SDL_internal.h"

#include <3ds.h>

// Used when the CFGU fails to work.
#define BAD_LOCALE 255

static u8 GetLocaleIndex(void);

bool SDL_SYS_GetPreferredLocales(char *buf, size_t buflen)
{
    // The 3DS only supports these 12 languages, only one can be active at a time
    static const char AVAILABLE_LOCALES[][6] = { "ja_JP", "en_US", "fr_FR", "de_DE",
                                                 "it_IT", "es_ES", "zh_CN", "ko_KR",
                                                 "nl_NL", "pt_PT", "ru_RU", "zh_TW" };
    u8 current_locale = GetLocaleIndex();
    if (current_locale != BAD_LOCALE) {
        SDL_strlcpy(buf, AVAILABLE_LOCALES[current_locale], buflen);
    }
    return true;
}

static u8 GetLocaleIndex(void)
{
    u8 current_locale;
    Result result;
    if (R_FAILED(cfguInit())) {
        return BAD_LOCALE;
    }
    result = CFGU_GetSystemLanguage(&current_locale);
    cfguExit();
    return R_SUCCEEDED(result) ? current_locale : BAD_LOCALE;
}
