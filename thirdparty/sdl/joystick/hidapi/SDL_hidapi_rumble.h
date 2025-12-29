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

#ifdef SDL_JOYSTICK_HIDAPI

// Handle rumble on a separate thread so it doesn't block the application

// Advanced API
#ifdef SDL_THREAD_SAFETY_ANALYSIS
extern SDL_Mutex *SDL_HIDAPI_rumble_lock;
#endif
bool SDL_HIDAPI_LockRumble(void) SDL_TRY_ACQUIRE(true, SDL_HIDAPI_rumble_lock);
bool SDL_HIDAPI_GetPendingRumbleLocked(SDL_HIDAPI_Device *device, Uint8 **data, int **size, int *maximum_size);
int SDL_HIDAPI_SendRumbleAndUnlock(SDL_HIDAPI_Device *device, const Uint8 *data, int size) SDL_RELEASE(SDL_HIDAPI_rumble_lock);
typedef void (*SDL_HIDAPI_RumbleSentCallback)(void *userdata);
int SDL_HIDAPI_SendRumbleWithCallbackAndUnlock(SDL_HIDAPI_Device *device, const Uint8 *data, int size, SDL_HIDAPI_RumbleSentCallback callback, void *userdata) SDL_RELEASE(SDL_HIDAPI_rumble_lock);
void SDL_HIDAPI_UnlockRumble(void) SDL_RELEASE(SDL_HIDAPI_rumble_lock);

// Simple API, will replace any pending rumble with the new data
int SDL_HIDAPI_SendRumble(SDL_HIDAPI_Device *device, const Uint8 *data, int size);
void SDL_HIDAPI_QuitRumble(void);

#endif // SDL_JOYSTICK_HIDAPI
