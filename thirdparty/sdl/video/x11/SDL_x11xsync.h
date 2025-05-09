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

#ifndef SDL_x11xsync_h_
#define SDL_x11xsync_h_

#ifdef SDL_VIDEO_DRIVER_X11_XSYNC

extern void X11_InitXsync(SDL_VideoDevice *_this);
extern bool X11_XsyncIsInitialized(void);
extern bool X11_InitResizeSync(SDL_Window *window);
extern void X11_TermResizeSync(SDL_Window *window);
extern void X11_HandleSyncRequest(SDL_Window *window, XClientMessageEvent *event);
extern void X11_HandleConfigure(SDL_Window *window, XConfigureEvent *event);
extern void X11_HandlePresent(SDL_Window *window);

#endif // SDL_VIDEO_DRIVER_X11_XSYNC

#endif // SDL_x11xsync_h_
