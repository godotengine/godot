/*
  Simple DirectMedia Layer
  Copyright (C) 2025 Katharine Chui <katharine.chui@gmail.com>

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

#ifndef SDL_hidapihaptic_c_h_
#define SDL_hidapihaptic_c_h_

#include "SDL3/SDL_haptic.h"
#include "SDL3/SDL_joystick.h"
#include "../SDL_syshaptic.h"
#include "../../joystick/SDL_joystick_c.h" // accessing _SDL_JoystickDriver
#include "../../joystick/SDL_sysjoystick.h" // accessing _SDL_Joystick

#define SDL_HAPTIC_HIDAPI_LG4FF

typedef struct SDL_HIDAPI_HapticDriver SDL_HIDAPI_HapticDriver;
typedef struct SDL_HIDAPI_HapticDevice
{
    SDL_Haptic *haptic; /* related haptic ref */
    SDL_Joystick *joystick; /* related hidapi joystick */
    SDL_HIDAPI_HapticDriver *driver; /* driver to use */
    void *ctx; /* driver specific context */
} SDL_HIDAPI_HapticDevice;

struct SDL_HIDAPI_HapticDriver
{
    bool (*JoystickSupported)(SDL_Joystick *joystick); /* return true if haptic can be opened from the joystick */
    void *(*Open)(SDL_Joystick *joystick); /* returns a driver context allocated with SDL_malloc, or null if it cannot be allocated */
  
    /* functions below need to handle the possibility of a null joystick instance, indicating the absence of the joystick */
    void (*Close)(SDL_HIDAPI_HapticDevice *device); /* cleanup resources allocated during Open, do NOT free driver context created in Open */
  
    /* below mirror SDL_haptic.h effect interfaces */
    int (*NumEffects)(SDL_HIDAPI_HapticDevice *device); /* returns supported number of effects the device can store */
    int (*NumEffectsPlaying)(SDL_HIDAPI_HapticDevice *device); /* returns supported number of effects the device can play concurrently */
    Uint32 (*GetFeatures)(SDL_HIDAPI_HapticDevice *device); /* returns supported effects in a bitmask */
    int (*NumAxes)(SDL_HIDAPI_HapticDevice *device); /* returns the number of haptic axes */
    SDL_HapticEffectID (*CreateEffect)(SDL_HIDAPI_HapticDevice *device, const SDL_HapticEffect *data); /* returns effect id if created correctly, negative number on error */
    bool (*UpdateEffect)(SDL_HIDAPI_HapticDevice *device, SDL_HapticEffectID id, const SDL_HapticEffect *data); /* returns true on success, false on error */
    bool (*RunEffect)(SDL_HIDAPI_HapticDevice *device, SDL_HapticEffectID id, Uint32 iterations); /* returns true on success, false on error */
    bool (*StopEffect)(SDL_HIDAPI_HapticDevice *device, SDL_HapticEffectID id); /* returns true on success, false on error */
    void (*DestroyEffect)(SDL_HIDAPI_HapticDevice *device, SDL_HapticEffectID id);
    bool (*GetEffectStatus)(SDL_HIDAPI_HapticDevice *device, SDL_HapticEffectID id); /* returns true if playing, false if not playing or on error */
    bool (*SetGain)(SDL_HIDAPI_HapticDevice *device, int gain); /* gain 0 - 100, returns true on success, false on error */
    bool (*SetAutocenter)(SDL_HIDAPI_HapticDevice *device, int autocenter); /* autocenter 0 - 100, returns true on success, false on error */
    bool (*Pause)(SDL_HIDAPI_HapticDevice *device); /* returns true on success, false on error */
    bool (*Resume)(SDL_HIDAPI_HapticDevice *device); /* returns true on success, false on error */
    bool (*StopEffects)(SDL_HIDAPI_HapticDevice *device); /* returns true on success, false on error */
};

extern SDL_HIDAPI_HapticDriver SDL_HIDAPI_HapticDriverLg4ff;

#endif //SDL_joystick_c_h_
