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

#ifndef SDL_x11events_h_
#define SDL_x11events_h_

extern void X11_PumpEvents(SDL_VideoDevice *_this);
extern int X11_WaitEventTimeout(SDL_VideoDevice *_this, Sint64 timeoutNS);
extern void X11_SendWakeupEvent(SDL_VideoDevice *_this, SDL_Window *window);
extern bool X11_SuspendScreenSaver(SDL_VideoDevice *_this);
extern void X11_ReconcileKeyboardState(SDL_VideoDevice *_this);
extern void X11_GetBorderValues(SDL_WindowData *data);
extern Uint64 X11_GetEventTimestamp(unsigned long time);
extern void X11_HandleKeyEvent(SDL_VideoDevice *_this, SDL_WindowData *windowdata, SDL_KeyboardID keyboardID, XEvent *xevent);
extern void X11_HandleButtonPress(SDL_VideoDevice *_this, SDL_WindowData *windowdata, SDL_MouseID mouseID, int button, float x, float y, unsigned long time);
extern void X11_HandleButtonRelease(SDL_VideoDevice *_this, SDL_WindowData *windowdata, SDL_MouseID mouseID, int button, unsigned long time);
extern SDL_WindowData *X11_FindWindow(SDL_VideoDevice *_this, Window window);
extern bool X11_ProcessHitTest(SDL_VideoDevice *_this, SDL_WindowData *data, const float x, const float y, bool force_new_result);
extern bool X11_TriggerHitTestAction(SDL_VideoDevice *_this, SDL_WindowData *data, const float x, const float y);

#endif // SDL_x11events_h_
