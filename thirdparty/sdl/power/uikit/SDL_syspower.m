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
#ifdef SDL_POWER_UIKIT

#import <UIKit/UIKit.h>

#include "SDL_syspower.h"

#ifndef SDL_PLATFORM_TVOS
// turn off the battery monitor if it's been more than X ms since last check.
static const int BATTERY_MONITORING_TIMEOUT = 3000;
static Uint64 SDL_UIKitLastPowerInfoQuery = 0;

void SDL_UIKit_UpdateBatteryMonitoring(void)
{
    if (SDL_UIKitLastPowerInfoQuery) {
        if (SDL_GetTicks() >= (SDL_UIKitLastPowerInfoQuery + BATTERY_MONITORING_TIMEOUT)) {
            UIDevice *uidev = [UIDevice currentDevice];
            SDL_assert([uidev isBatteryMonitoringEnabled] == YES);
            [uidev setBatteryMonitoringEnabled:NO];
            SDL_UIKitLastPowerInfoQuery = 0;
        }
    }
}
#else
void SDL_UIKit_UpdateBatteryMonitoring(void)
{
    // Do nothing.
}
#endif // !SDL_PLATFORM_TVOS

bool SDL_GetPowerInfo_UIKit(SDL_PowerState *state, int *seconds, int *percent)
{
#ifdef SDL_PLATFORM_TVOS
    *state = SDL_POWERSTATE_NO_BATTERY;
    *seconds = -1;
    *percent = -1;
#else  // SDL_PLATFORM_TVOS
    @autoreleasepool {
        UIDevice *uidev = [UIDevice currentDevice];

        if (!SDL_UIKitLastPowerInfoQuery) {
            SDL_assert(uidev.isBatteryMonitoringEnabled == NO);
            uidev.batteryMonitoringEnabled = YES;
        }

        /* UIKit_GL_SwapWindow() (etc) will check this and disable the battery
         *  monitoring if the app hasn't queried it in the last X seconds.
         *  Apparently monitoring the battery burns battery life.  :)
         *  Apple's docs say not to monitor the battery unless you need it.
         */
        SDL_UIKitLastPowerInfoQuery = SDL_GetTicks();

        *seconds = -1; // no API to estimate this in UIKit.

        switch (uidev.batteryState) {
        case UIDeviceBatteryStateCharging:
            *state = SDL_POWERSTATE_CHARGING;
            break;

        case UIDeviceBatteryStateFull:
            *state = SDL_POWERSTATE_CHARGED;
            break;

        case UIDeviceBatteryStateUnplugged:
            *state = SDL_POWERSTATE_ON_BATTERY;
            break;

        case UIDeviceBatteryStateUnknown:
        default:
            *state = SDL_POWERSTATE_UNKNOWN;
            break;
        }

        const float level = uidev.batteryLevel;
        *percent = ((level < 0.0f) ? -1 : ((int)((level * 100) + 0.5f)));
    }
#endif // SDL_PLATFORM_TVOS

    return true; // always the definitive answer on iOS.
}

#endif // SDL_POWER_UIKIT
#endif // SDL_POWER_DISABLED
