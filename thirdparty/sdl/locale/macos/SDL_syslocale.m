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

#import <Foundation/Foundation.h>

bool SDL_SYS_GetPreferredLocales(char *buf, size_t buflen)
{
    @autoreleasepool {
        NSArray *languages = NSLocale.preferredLanguages;
        size_t numlangs = 0;
        size_t i;

        numlangs = (size_t)[languages count];

        for (i = 0; i < numlangs; i++) {
            NSString *nsstr = [languages objectAtIndex:i];
            size_t len;
            char *ptr;

            if (nsstr == nil) {
                break;
            }

            [nsstr getCString:buf maxLength:buflen encoding:NSASCIIStringEncoding];
            len = SDL_strlen(buf);

            // convert '-' to '_'...
            //  These are always full lang-COUNTRY, so we search from the back,
            //  so things like zh-Hant-CN find the right '-' to convert.
            ptr = SDL_strrchr(buf, '-');
            if (ptr != NULL) {
                *ptr = '_';
            }

            if (buflen <= len) {
                *buf = '\0'; // drop this one and stop, we can't fit anymore.
                break;
            }

            buf += len;
            buflen -= len;

            if (i < (numlangs - 1)) {
                if (buflen <= 1) {
                    break; // out of room, stop looking.
                }
                buf[0] = ','; // add a comma between entries.
                buf[1] = '\0';
                buf++;
                buflen--;
            }
        }
    }
    return true;
}
