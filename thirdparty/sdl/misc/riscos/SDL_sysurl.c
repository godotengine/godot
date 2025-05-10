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

#include <kernel.h>
#include <swis.h>

#ifndef URI_Dispatch
#define URI_Dispatch 0x4e381
#endif

bool SDL_SYS_OpenURL(const char *url)
{
    _kernel_swi_regs regs;
    _kernel_oserror *error;

    regs.r[0] = 0;
    regs.r[1] = (int)url;
    regs.r[2] = 0;
    error = _kernel_swi(URI_Dispatch, &regs, &regs);
    if (error) {
        return SDL_SetError("Couldn't open given URL: %s", error->errmess);
    }

    if (regs.r[0] & 1) {
        return SDL_SetError("Couldn't open given URL.");
    }
    return true;
}
