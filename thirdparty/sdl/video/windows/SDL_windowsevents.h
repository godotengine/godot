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

#ifndef SDL_windowsevents_h_
#define SDL_windowsevents_h_

extern LPTSTR SDL_Appname;
extern Uint32 SDL_Appstyle;
extern HINSTANCE SDL_Instance;

extern LRESULT CALLBACK WIN_KeyboardHookProc(int nCode, WPARAM wParam, LPARAM lParam);
extern LRESULT CALLBACK WIN_WindowProc(HWND hwnd, UINT msg, WPARAM wParam,
                                       LPARAM lParam);
extern void WIN_PollRawInput(SDL_VideoDevice *_this, Uint64 poll_start);
extern void WIN_CheckKeyboardAndMouseHotplug(SDL_VideoDevice *_this, bool initial_check);
extern void WIN_PumpEvents(SDL_VideoDevice *_this);
extern void WIN_SendWakeupEvent(SDL_VideoDevice *_this, SDL_Window *window);
extern int WIN_WaitEventTimeout(SDL_VideoDevice *_this, Sint64 timeoutNS);

#endif // SDL_windowsevents_h_
