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

#include <psputility.h>

bool SDL_SYS_GetPreferredLocales(char *buf, size_t buflen)
{
    int current_locale_int = PSP_SYSTEMPARAM_LANGUAGE_ENGLISH;

    SDL_assert(buflen > 0);

    sceUtilityGetSystemParamInt(PSP_SYSTEMPARAM_ID_INT_LANGUAGE, &current_locale_int);
    switch(current_locale_int) {
        case PSP_SYSTEMPARAM_LANGUAGE_JAPANESE:
            SDL_strlcpy(buf, "ja_JP", buflen);
            break;
        case PSP_SYSTEMPARAM_LANGUAGE_ENGLISH:
            SDL_strlcpy(buf, "en_US", buflen);
            break;
        case PSP_SYSTEMPARAM_LANGUAGE_FRENCH:
            SDL_strlcpy(buf, "fr_FR", buflen);
            break;
        case PSP_SYSTEMPARAM_LANGUAGE_SPANISH:
            SDL_strlcpy(buf, "es_ES", buflen);
            break;
        case PSP_SYSTEMPARAM_LANGUAGE_GERMAN:
            SDL_strlcpy(buf, "de_DE", buflen);
            break;
        case PSP_SYSTEMPARAM_LANGUAGE_ITALIAN:
            SDL_strlcpy(buf, "it_IT", buflen);
            break;
        case PSP_SYSTEMPARAM_LANGUAGE_DUTCH:
            SDL_strlcpy(buf, "nl_NL", buflen);
            break;
        case PSP_SYSTEMPARAM_LANGUAGE_PORTUGUESE:
            SDL_strlcpy(buf, "pt_PT", buflen);
            break;
        case PSP_SYSTEMPARAM_LANGUAGE_RUSSIAN:
            SDL_strlcpy(buf, "ru_RU", buflen);
            break;
        case PSP_SYSTEMPARAM_LANGUAGE_KOREAN:
            SDL_strlcpy(buf, "ko_KR", buflen);
            break;
        case PSP_SYSTEMPARAM_LANGUAGE_CHINESE_TRADITIONAL:
            SDL_strlcpy(buf, "zh_TW", buflen);
            break;
        case PSP_SYSTEMPARAM_LANGUAGE_CHINESE_SIMPLIFIED:
            SDL_strlcpy(buf, "zh_CN", buflen);
            break;
        default:
            SDL_strlcpy(buf, "en_US", buflen);
            break;
    }
    return true;
}

/* vi: set ts=4 sw=4 expandtab: */
