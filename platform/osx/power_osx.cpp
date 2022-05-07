/*************************************************************************/
/*  power_osx.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

/*
Adapted from corresponding SDL 2.0 code.
*/

/*
 * Simple DirectMedia Layer
 * Copyright (C) 1997-2017 Sam Lantinga <slouken@libsdl.org>
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include "power_osx.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/ps/IOPSKeys.h>
#include <IOKit/ps/IOPowerSources.h>

//  CODE CHUNK IMPORTED FROM SDL 2.0

/* CoreFoundation is so verbose... */
#define STRMATCH(a, b) (CFStringCompare(a, b, 0) == kCFCompareEqualTo)
#define GETVAL(k, v) \
	CFDictionaryGetValueIfPresent(dict, CFSTR(k), (const void **)v)

/* Note that AC power sources also include a laptop battery it is charging. */
void PowerOSX::checkps(CFDictionaryRef dict, bool *have_ac, bool *have_battery, bool *charging) {
	CFStringRef strval; /* don't CFRelease() this. */
	CFBooleanRef bval;
	CFNumberRef numval;
	bool charge = false;
	bool choose = false;
	bool is_ac = false;
	int secs = -1;
	int maxpct = -1;
	int pct = -1;

	if ((GETVAL(kIOPSIsPresentKey, &bval)) && (bval == kCFBooleanFalse)) {
		return; /* nothing to see here. */
	}

	if (!GETVAL(kIOPSPowerSourceStateKey, &strval)) {
		return;
	}

	if (STRMATCH(strval, CFSTR(kIOPSACPowerValue))) {
		is_ac = *have_ac = true;
	} else if (!STRMATCH(strval, CFSTR(kIOPSBatteryPowerValue))) {
		return; /* not a battery? */
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

		/* Mac OS X reports 0 minutes until empty if you're plugged in. :( */
		if ((val == 0) && (is_ac)) {
			val = -1; /* !!! FIXME: calc from timeToFull and capacity? */
		}

		secs = (int)val;
		if (secs > 0) {
			secs *= 60; /* value is in minutes, so convert to seconds. */
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
	if ((secs < 0) && (nsecs_left < 0)) {
		if ((pct < 0) && (percent_left < 0)) {
			choose = true; /* at least we know there's a battery. */
		}
		if (pct > percent_left) {
			choose = true;
		}
	} else if (secs > nsecs_left) {
		choose = true;
	}

	if (choose) {
		nsecs_left = secs;
		percent_left = pct;
		*charging = charge;
	}
}

#undef GETVAL
#undef STRMATCH

//  CODE CHUNK IMPORTED FROM SDL 2.0
bool PowerOSX::GetPowerInfo_MacOSX() {
	CFTypeRef blob = IOPSCopyPowerSourcesInfo();

	nsecs_left = -1;
	percent_left = -1;
	power_state = OS::POWERSTATE_UNKNOWN;

	if (blob != NULL) {
		CFArrayRef list = IOPSCopyPowerSourcesList(blob);
		if (list != NULL) {
			/* don't CFRelease() the list items, or dictionaries! */
			bool have_ac = false;
			bool have_battery = false;
			bool charging = false;
			const CFIndex total = CFArrayGetCount(list);
			CFIndex i;
			for (i = 0; i < total; i++) {
				CFTypeRef ps = (CFTypeRef)CFArrayGetValueAtIndex(list, i);
				CFDictionaryRef dict = IOPSGetPowerSourceDescription(blob, ps);
				if (dict != NULL) {
					checkps(dict, &have_ac, &have_battery, &charging);
				}
			}

			if (!have_battery) {
				power_state = OS::POWERSTATE_NO_BATTERY;
			} else if (charging) {
				power_state = OS::POWERSTATE_CHARGING;
			} else if (have_ac) {
				power_state = OS::POWERSTATE_CHARGED;
			} else {
				power_state = OS::POWERSTATE_ON_BATTERY;
			}

			CFRelease(list);
		}
		CFRelease(blob);
	}

	return true; /* always the definitive answer on Mac OS X. */
}

bool PowerOSX::UpdatePowerInfo() {
	if (GetPowerInfo_MacOSX()) {
		return true;
	}
	return false;
}

OS::PowerState PowerOSX::get_power_state() {
	if (UpdatePowerInfo()) {
		return power_state;
	} else {
		return OS::POWERSTATE_UNKNOWN;
	}
}

int PowerOSX::get_power_seconds_left() {
	if (UpdatePowerInfo()) {
		return nsecs_left;
	} else {
		return -1;
	}
}

int PowerOSX::get_power_percent_left() {
	if (UpdatePowerInfo()) {
		return percent_left;
	} else {
		return -1;
	}
}

PowerOSX::PowerOSX() :
		nsecs_left(-1),
		percent_left(-1),
		power_state(OS::POWERSTATE_UNKNOWN) {
}

PowerOSX::~PowerOSX() {
}
