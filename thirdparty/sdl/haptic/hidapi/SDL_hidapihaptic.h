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

/*
  All hid command sent and effect rendering are ported from https://github.com/berarma/new-lg4ff
*/

#ifndef SDL_hidapihaptic_h_
#define SDL_hidapihaptic_h_

bool SDL_HIDAPI_HapticInit(void);
bool SDL_HIDAPI_HapticIsHidapi(SDL_Haptic *haptic);
bool SDL_HIDAPI_JoystickIsHaptic(SDL_Joystick *joystick);
bool SDL_HIDAPI_HapticOpenFromJoystick(SDL_Haptic *haptic, SDL_Joystick *joystick);
bool SDL_HIDAPI_JoystickSameHaptic(SDL_Haptic *haptic, SDL_Joystick *joystick);
void SDL_HIDAPI_HapticClose(SDL_Haptic *haptic);
void SDL_HIDAPI_HapticQuit(void);
SDL_HapticEffectID SDL_HIDAPI_HapticNewEffect(SDL_Haptic *haptic, const SDL_HapticEffect *base);
bool SDL_HIDAPI_HapticUpdateEffect(SDL_Haptic *haptic, SDL_HapticEffectID id, const SDL_HapticEffect *data);
bool SDL_HIDAPI_HapticRunEffect(SDL_Haptic *haptic, SDL_HapticEffectID id, Uint32 iterations);
bool SDL_HIDAPI_HapticStopEffect(SDL_Haptic *haptic, SDL_HapticEffectID id);
void SDL_HIDAPI_HapticDestroyEffect(SDL_Haptic *haptic, SDL_HapticEffectID id);
bool SDL_HIDAPI_HapticGetEffectStatus(SDL_Haptic *haptic, SDL_HapticEffectID id);
bool SDL_HIDAPI_HapticSetGain(SDL_Haptic *haptic, int gain);
bool SDL_HIDAPI_HapticSetAutocenter(SDL_Haptic *haptic, int autocenter);
bool SDL_HIDAPI_HapticPause(SDL_Haptic *haptic);
bool SDL_HIDAPI_HapticResume(SDL_Haptic *haptic);
bool SDL_HIDAPI_HapticStopAll(SDL_Haptic *haptic);

#endif //SDL_hidapihaptic_h_
