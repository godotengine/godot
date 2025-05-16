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

#ifndef SDL_windowevents_c_h_
#define SDL_windowevents_c_h_

typedef enum
{
    SDL_WINDOW_EVENT_WATCH_EARLY,
    SDL_WINDOW_EVENT_WATCH_NORMAL
} SDL_WindowEventWatchPriority;

extern void SDL_InitWindowEventWatch(void);
extern void SDL_QuitWindowEventWatch(void);
extern void SDL_AddWindowEventWatch(SDL_WindowEventWatchPriority priority, SDL_EventFilter filter, void *userdata);
extern void SDL_RemoveWindowEventWatch(SDL_WindowEventWatchPriority priority, SDL_EventFilter filter, void *userdata);

extern bool SDL_SendWindowEvent(SDL_Window *window, SDL_EventType windowevent, int data1, int data2);

#endif // SDL_windowevents_c_h_
