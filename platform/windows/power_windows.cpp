/*************************************************************************/
/*  power_windows.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
  Simple DirectMedia Layer
  Copyright (C) 1997-2017 Sam Lantinga <slouken@libsdl.org>

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

#include "power_windows.h"

// CODE CHUNK IMPORTED FROM SDL 2.0

bool PowerWindows::GetPowerInfo_Windows() {
	SYSTEM_POWER_STATUS status;
	bool need_details = FALSE;

	/* This API should exist back to Win95. */
	if (!GetSystemPowerStatus(&status)) {
		/* !!! FIXME: push GetLastError() into GetError() */
		power_state = OS::POWERSTATE_UNKNOWN;
	} else if (status.BatteryFlag == 0xFF) { /* unknown state */
		power_state = OS::POWERSTATE_UNKNOWN;
	} else if (status.BatteryFlag & (1 << 7)) { /* no battery */
		power_state = OS::POWERSTATE_NO_BATTERY;
	} else if (status.BatteryFlag & (1 << 3)) { /* charging */
		power_state = OS::POWERSTATE_CHARGING;
		need_details = TRUE;
	} else if (status.ACLineStatus == 1) {
		power_state = OS::POWERSTATE_CHARGED; /* on AC, not charging. */
		need_details = TRUE;
	} else {
		power_state = OS::POWERSTATE_ON_BATTERY; /* not on AC. */
		need_details = TRUE;
	}

	percent_left = -1;
	nsecs_left = -1;
	if (need_details) {
		const int pct = (int)status.BatteryLifePercent;
		const int secs = (int)status.BatteryLifeTime;

		if (pct != 255) { /* 255 == unknown */
			percent_left = (pct > 100) ? 100 : pct; /* clamp between 0%, 100% */
		}
		if (secs != 0xFFFFFFFF) { /* ((DWORD)-1) == unknown */
			nsecs_left = secs;
		}
	}

	return TRUE; /* always the definitive answer on Windows. */
}

OS::PowerState PowerWindows::get_power_state() {
	if (GetPowerInfo_Windows()) {
		return power_state;
	} else {
		return OS::POWERSTATE_UNKNOWN;
	}
}

int PowerWindows::get_power_seconds_left() {
	if (GetPowerInfo_Windows()) {
		return nsecs_left;
	} else {
		return -1;
	}
}

int PowerWindows::get_power_percent_left() {
	if (GetPowerInfo_Windows()) {
		return percent_left;
	} else {
		return -1;
	}
}

PowerWindows::PowerWindows() :
		nsecs_left(-1),
		percent_left(-1),
		power_state(OS::POWERSTATE_UNKNOWN) {
}

PowerWindows::~PowerWindows() {
}
