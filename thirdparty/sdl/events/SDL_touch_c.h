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

#ifndef SDL_touch_c_h_
#define SDL_touch_c_h_

typedef struct SDL_Touch
{
    SDL_TouchID id;
    SDL_TouchDeviceType type;
    int num_fingers;
    int max_fingers;
    SDL_Finger **fingers;
    char *name;
} SDL_Touch;

// Initialize the touch subsystem
extern bool SDL_InitTouch(void);

// Returns true if _any_ connected touch devices are known to SDL
extern bool SDL_TouchDevicesAvailable(void);

// Add a touch, returning the index of the touch, or -1 if there was an error.
extern int SDL_AddTouch(SDL_TouchID id, SDL_TouchDeviceType type, const char *name);

// Set or update the name of a touch.
extern void SDL_SetTouchName(SDL_TouchID id, const char *name);

// Get the touch with a given id
extern SDL_Touch *SDL_GetTouch(SDL_TouchID id);

// Send a touch down/up event for a touch
extern void SDL_SendTouch(Uint64 timestamp, SDL_TouchID id, SDL_FingerID fingerid, SDL_Window *window, SDL_EventType type, float x, float y, float pressure);

// Send a touch motion event for a touch
extern void SDL_SendTouchMotion(Uint64 timestamp, SDL_TouchID id, SDL_FingerID fingerid, SDL_Window *window, float x, float y, float pressure);

// Remove a touch
extern void SDL_DelTouch(SDL_TouchID id);

// Shutdown the touch subsystem
extern void SDL_QuitTouch(void);

#endif // SDL_touch_c_h_
