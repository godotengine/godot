/*************************************************************************/
/*  power_uwp.cpp                                                        */
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

#include "power_uwp.h"

PowerUWP::PowerUWP()
	: nsecs_left(-1), percent_left(-1), power_state(OS::POWERSTATE_UNKNOWN) {
}

PowerUWP::~PowerUWP() {
}

bool PowerUWP::UpdatePowerInfo() {
	// TODO, WinRT: Battery info is available on at least one WinRT platform (Windows Phone 8).  Implement UpdatePowerInfo as appropriate. */
	/* Notes from SDL:
	         - the Win32 function, GetSystemPowerStatus, is not available for use on WinRT
	         - Windows Phone 8 has a 'Battery' class, which is documented as available for C++
	             - More info on WP8's Battery class can be found at http://msdn.microsoft.com/library/windowsphone/develop/jj207231
	    */
	return false;
}

OS::PowerState PowerUWP::get_power_state() {
	if (UpdatePowerInfo()) {
		return power_state;
	} else {
		WARN_PRINT("Power management is not implemented on this platform, defaulting to POWERSTATE_UNKNOWN");
		return OS::POWERSTATE_UNKNOWN;
	}
}

int PowerUWP::get_power_seconds_left() {
	if (UpdatePowerInfo()) {
		return nsecs_left;
	} else {
		WARN_PRINT("Power management is not implemented on this platform, defaulting to -1");
		return -1;
	}
}

int PowerUWP::get_power_percent_left() {
	if (UpdatePowerInfo()) {
		return percent_left;
	} else {
		WARN_PRINT("Power management is not implemented on this platform, defaulting to -1");
		return -1;
	}
}
