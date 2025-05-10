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

#include <AppKit.h>
#include <LocaleRoster.h>
#include <TypeConstants.h>

#include "SDL_internal.h"
#include "../SDL_syslocale.h"

bool SDL_SYS_GetPreferredLocales(char *buf, size_t buflen)
{
    BLocaleRoster *roster = BLocaleRoster::Default();
    roster->Refresh();

    BMessage msg;
    if (roster->GetPreferredLanguages(&msg) != B_OK) {
        return SDL_SetError("BLocaleRoster couldn't get preferred languages");
    }

    const char *key = "language";
    type_code typ = B_ANY_TYPE;
    int32 numlangs = 0;
    if ((msg.GetInfo(key, &typ, &numlangs) != B_OK) || (typ != B_STRING_TYPE)) {
        return SDL_SetError("BLocaleRoster message was wrong");
    }

    for (int32 i = 0; i < numlangs; i++) {
        const char *str = NULL;
        if (msg.FindString(key, i, &str) != B_OK) {
            continue;
        }

        const size_t len = SDL_strlen(str);
        if (buflen <= len) {
            break;  // can't fit it, we're done.
        }

        SDL_strlcpy(buf, str, buflen);
        buf += len;
        buflen -= len;

        if (i < (numlangs - 1)) {
            if (buflen <= 1) {
                break;  // out of room, stop looking.
            }
            buf[0] = ',';  // add a comma between entries.
            buf[1] = '\0';
            buf++;
            buflen--;
        }
    }
    return true;
}
