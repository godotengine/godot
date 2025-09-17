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

#include "../../core/windows/SDL_xinput.h"

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

extern bool SDL_XINPUT_Enabled(void);
extern bool SDL_XINPUT_JoystickInit(void);
extern void SDL_XINPUT_JoystickDetect(JoyStick_DeviceData **pContext);
extern bool SDL_XINPUT_JoystickPresent(Uint16 vendor, Uint16 product, Uint16 version);
extern bool SDL_XINPUT_JoystickOpen(SDL_Joystick *joystick, JoyStick_DeviceData *joystickdevice);
extern bool SDL_XINPUT_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble);
extern void SDL_XINPUT_JoystickUpdate(SDL_Joystick *joystick);
extern void SDL_XINPUT_JoystickClose(SDL_Joystick *joystick);
extern void SDL_XINPUT_JoystickQuit(void);
extern int SDL_XINPUT_GetSteamVirtualGamepadSlot(Uint8 userid);

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif
