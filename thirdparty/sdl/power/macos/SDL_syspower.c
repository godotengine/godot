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
#ifdef SDL_POWER_MACOSX

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>

// CoreFoundation is so verbose...
#define STRMATCH(a, b) (CFStringCompare(a, b, 0) == kCFCompareEqualTo)
#define GETVAL(k, v) \
    CFDictionaryGetValueIfPresent(dict, CFSTR(k), (const void **)v)

// Note that AC power sources also include a laptop battery it is charging.
static void checkps(CFDictionaryRef dict, bool *have_ac, bool *have_battery,
                    bool *charging, int *seconds, int *percent)
{
    CFStringRef strval; // don't CFRelease() this.
    CFBooleanRef bval;
    CFNumberRef numval;
    bool charge = false;
    bool choose = false;
    bool is_ac = false;
    int secs = -1;
    int maxpct = -1;
    int pct = -1;

    if ((GETVAL(kIOPSIsPresentKey, &bval)) && (bval == kCFBooleanFalse)) {
        return; // nothing to see here.
    }

    if (!GETVAL(kIOPSPowerSourceStateKey, &strval)) {
        return;
    }

    if (STRMATCH(strval, CFSTR(kIOPSACPowerValue))) {
        is_ac = *have_ac = true;
    } else if (!STRMATCH(strval, CFSTR(kIOPSBatteryPowerValue))) {
        return; // not a battery?
    }

    if ((GETVAL(kIOPSIsChargingKey, &bval)) && (bval == kCFBooleanTrue)) {
        charge = true;
    }

    if (GETVAL(kIOPSMaxCapacityKey, &numval)) {
        SInt32 val = -1;
        CFNumberGetValue(numval, kCFNumberSInt32Type, &val);
        if (val > 0) {
            *have_battery = true;
            maxpct = (int)val;
        }
    }

    if (GETVAL(kIOPSMaxCapacityKey, &numval)) {
        SInt32 val = -1;
        CFNumberGetValue(numval, kCFNumberSInt32Type, &val);
        if (val > 0) {
            *have_battery = true;
            maxpct = (int)val;
        }
    }

    if (GETVAL(kIOPSTimeToEmptyKey, &numval)) {
        SInt32 val = -1;
        CFNumberGetValue(numval, kCFNumberSInt32Type, &val);

        // macOS reports 0 minutes until empty if you're plugged in. :(
        if ((val == 0) && (is_ac)) {
            val = -1; // !!! FIXME: calc from timeToFull and capacity?
        }

        secs = (int)val;
        if (secs > 0) {
            secs *= 60; // value is in minutes, so convert to seconds.
        }
    }

    if (GETVAL(kIOPSCurrentCapacityKey, &numval)) {
        SInt32 val = -1;
        CFNumberGetValue(numval, kCFNumberSInt32Type, &val);
        pct = (int)val;
    }

    if ((pct > 0) && (maxpct > 0)) {
        pct = (int)((((double)pct) / ((double)maxpct)) * 100.0);
    }

    if (pct > 100) {
        pct = 100;
    }

    /*
     * We pick the battery that claims to have the most minutes left.
     *  (failing a report of minutes, we'll take the highest percent.)
     */
    if ((secs < 0) && (*seconds < 0)) {
        if ((pct < 0) && (*percent < 0)) {
            choose = true; // at least we know there's a battery.
        }
        if (pct > *percent) {
            choose = true;
        }
    } else if (secs > *seconds) {
        choose = true;
    }

    if (choose) {
        *seconds = secs;
        *percent = pct;
        *charging = charge;
    }
}

#undef GETVAL
#undef STRMATCH

bool SDL_GetPowerInfo_MacOSX(SDL_PowerState *state, int *seconds, int *percent)
{
    CFTypeRef blob = IOPSCopyPowerSourcesInfo();

    *seconds = -1;
    *percent = -1;
    *state = SDL_POWERSTATE_UNKNOWN;

    if (blob != NULL) {
        CFArrayRef list = IOPSCopyPowerSourcesList(blob);
        if (list != NULL) {
            // don't CFRelease() the list items, or dictionaries!
            bool have_ac = false;
            bool have_battery = false;
            bool charging = false;
            const CFIndex total = CFArrayGetCount(list);
            CFIndex i;
            for (i = 0; i < total; i++) {
                CFTypeRef ps = (CFTypeRef)CFArrayGetValueAtIndex(list, i);
                CFDictionaryRef dict =
                    IOPSGetPowerSourceDescription(blob, ps);
                if (dict != NULL) {
                    checkps(dict, &have_ac, &have_battery, &charging,
                            seconds, percent);
                }
            }

            if (!have_battery) {
                *state = SDL_POWERSTATE_NO_BATTERY;
            } else if (charging) {
                *state = SDL_POWERSTATE_CHARGING;
            } else if (have_ac) {
                *state = SDL_POWERSTATE_CHARGED;
            } else {
                *state = SDL_POWERSTATE_ON_BATTERY;
            }

            CFRelease(list);
        }
        CFRelease(blob);
    }

    return true; // always the definitive answer on macOS.
}

#endif // SDL_POWER_MACOSX
#endif // SDL_POWER_DISABLED
