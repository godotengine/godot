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

#ifndef SDL_POWER_DISABLED
#ifdef SDL_POWER_EMSCRIPTEN

#include <emscripten/html5.h>

bool SDL_GetPowerInfo_Emscripten(SDL_PowerState *state, int *seconds, int *percent)
{
    EmscriptenBatteryEvent batteryState;
    int haveBattery = 0;

    if (emscripten_get_battery_status(&batteryState) == EMSCRIPTEN_RESULT_NOT_SUPPORTED) {
        return false;
    }

    haveBattery = batteryState.level != 1.0 || !batteryState.charging || batteryState.chargingTime != 0.0;

    if (!haveBattery) {
        *state = SDL_POWERSTATE_NO_BATTERY;
        *seconds = -1;
        *percent = -1;
        return true;
    }

    if (batteryState.charging) {
        *state = batteryState.chargingTime == 0.0 ? SDL_POWERSTATE_CHARGED : SDL_POWERSTATE_CHARGING;
    } else {
        *state = SDL_POWERSTATE_ON_BATTERY;
    }

    *seconds = (int)batteryState.dischargingTime;
    *percent = (int)batteryState.level * 100;

    return true;
}

#endif // SDL_POWER_EMSCRIPTEN
#endif // SDL_POWER_DISABLED
