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

#ifdef SDL_VIDEO_DRIVER_X11

#include "SDL_x11video.h"
#include "SDL_x11touch.h"
#include "SDL_x11xinput2.h"
#include "../../events/SDL_touch_c.h"

void X11_InitTouch(SDL_VideoDevice *_this)
{
    X11_InitXinput2Multitouch(_this);
}

void X11_QuitTouch(SDL_VideoDevice *_this)
{
    SDL_QuitTouch();
}

void X11_ResetTouch(SDL_VideoDevice *_this)
{
    X11_QuitTouch(_this);
    X11_InitTouch(_this);
}

#endif // SDL_VIDEO_DRIVER_X11
