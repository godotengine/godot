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

#include "SDL_windowshaptic_c.h"

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

extern bool SDL_DINPUT_HapticInit(void);
extern bool SDL_DINPUT_HapticMaybeAddDevice(const DIDEVICEINSTANCE *pdidInstance);
extern bool SDL_DINPUT_HapticMaybeRemoveDevice(const DIDEVICEINSTANCE *pdidInstance);
extern bool SDL_DINPUT_HapticOpen(SDL_Haptic *haptic, SDL_hapticlist_item *item);
extern bool SDL_DINPUT_JoystickSameHaptic(SDL_Haptic *haptic, SDL_Joystick *joystick);
extern bool SDL_DINPUT_HapticOpenFromJoystick(SDL_Haptic *haptic, SDL_Joystick *joystick);
extern void SDL_DINPUT_HapticClose(SDL_Haptic *haptic);
extern void SDL_DINPUT_HapticQuit(void);
extern bool SDL_DINPUT_HapticNewEffect(SDL_Haptic *haptic, struct haptic_effect *effect, const SDL_HapticEffect *base);
extern bool SDL_DINPUT_HapticUpdateEffect(SDL_Haptic *haptic, struct haptic_effect *effect, const SDL_HapticEffect *data);
extern bool SDL_DINPUT_HapticRunEffect(SDL_Haptic *haptic, struct haptic_effect *effect, Uint32 iterations);
extern bool SDL_DINPUT_HapticStopEffect(SDL_Haptic *haptic, struct haptic_effect *effect);
extern void SDL_DINPUT_HapticDestroyEffect(SDL_Haptic *haptic, struct haptic_effect *effect);
extern int SDL_DINPUT_HapticGetEffectStatus(SDL_Haptic *haptic, struct haptic_effect *effect);
extern bool SDL_DINPUT_HapticSetGain(SDL_Haptic *haptic, int gain);
extern bool SDL_DINPUT_HapticSetAutocenter(SDL_Haptic *haptic, int autocenter);
extern bool SDL_DINPUT_HapticPause(SDL_Haptic *haptic);
extern bool SDL_DINPUT_HapticResume(SDL_Haptic *haptic);
extern bool SDL_DINPUT_HapticStopAll(SDL_Haptic *haptic);

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif
