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

#ifndef SDL_events_c_h_
#define SDL_events_c_h_

#include "SDL_internal.h"

// Useful functions and variables from SDL_events.c
#include "../video/SDL_sysvideo.h"

#include "SDL_clipboardevents_c.h"
#include "SDL_displayevents_c.h"
#include "SDL_dropevents_c.h"
#include "SDL_keyboard_c.h"
#include "SDL_mouse_c.h"
#include "SDL_touch_c.h"
#include "SDL_pen_c.h"
#include "SDL_windowevents_c.h"

// Start and stop the event processing loop
extern bool SDL_StartEventLoop(void);
extern void SDL_StopEventLoop(void);
extern void SDL_QuitInterrupt(void);

extern void SDL_SendAppEvent(SDL_EventType eventType);
extern void SDL_SendKeymapChangedEvent(void);
extern void SDL_SendLocaleChangedEvent(void);
extern void SDL_SendSystemThemeChangedEvent(void);

extern void *SDL_AllocateTemporaryMemory(size_t size);
extern const char *SDL_CreateTemporaryString(const char *string);
extern void *SDL_ClaimTemporaryMemory(const void *mem);
extern void SDL_FreeTemporaryMemory(void);

extern void SDL_PumpEventMaintenance(void);

extern void SDL_SendQuit(void);

extern bool SDL_InitEvents(void);
extern void SDL_QuitEvents(void);

extern void SDL_SendPendingSignalEvents(void);

extern bool SDL_InitQuit(void);
extern void SDL_QuitQuit(void);

#endif // SDL_events_c_h_
