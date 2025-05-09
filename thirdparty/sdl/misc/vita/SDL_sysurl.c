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

#include "../SDL_sysurl.h"

#include <string.h>
#include <psp2/apputil.h>

bool SDL_SYS_OpenURL(const char *url)
{
    SceAppUtilInitParam init_param;
    SceAppUtilBootParam boot_param;
    SceAppUtilWebBrowserParam browser_param;
    SDL_zero(init_param);
    SDL_zero(boot_param);
    sceAppUtilInit(&init_param, &boot_param);
    SDL_zero(browser_param);
    browser_param.str = url;
    browser_param.strlen = SDL_strlen(url);
    sceAppUtilLaunchWebBrowser(&browser_param);
    return true;
}
