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

#include <psp2/apputil.h>
#include <psp2/system_param.h>

bool SDL_SYS_GetPreferredLocales(char *buf, size_t buflen)
{
    const char *vita_locales[] = {
        "ja_JP",
        "en_US",
        "fr_FR",
        "es_ES",
        "de_DE",
        "it_IT",
        "nl_NL",
        "pt_PT",
        "ru_RU",
        "ko_KR",
        "zh_TW",
        "zh_CN",
        "fi_FI",
        "sv_SE",
        "da_DK",
        "no_NO",
        "pl_PL",
        "pt_BR",
        "en_GB",
        "tr_TR",
    };

    Sint32 language = SCE_SYSTEM_PARAM_LANG_ENGLISH_US;
    SceAppUtilInitParam initParam;
    SceAppUtilBootParam bootParam;
    SDL_zero(initParam);
    SDL_zero(bootParam);
    sceAppUtilInit(&initParam, &bootParam);
    sceAppUtilSystemParamGetInt(SCE_SYSTEM_PARAM_ID_LANG, &language);

    if (language < 0 || language > SCE_SYSTEM_PARAM_LANG_TURKISH) {
        language = SCE_SYSTEM_PARAM_LANG_ENGLISH_US; // default to english
    }

    SDL_strlcpy(buf, vita_locales[language], buflen);

    sceAppUtilShutdown();
    return true;
}
