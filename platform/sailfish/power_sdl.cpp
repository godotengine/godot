/*************************************************************************/
/*  power_sdl.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "power_sdl.h"
#include <SDL.h>

OS::PowerState PowerSDL::get_power_state() {
	switch (SDL_GetPowerInfo(NULL, NULL)) {
		case SDL_POWERSTATE_UNKNOWN:
			return OS::POWERSTATE_UNKNOWN;
		case SDL_POWERSTATE_ON_BATTERY:
			return OS::POWERSTATE_ON_BATTERY;
		case SDL_POWERSTATE_NO_BATTERY:
			return OS::POWERSTATE_NO_BATTERY;
		case SDL_POWERSTATE_CHARGING:
			return OS::POWERSTATE_CHARGING;
		case SDL_POWERSTATE_CHARGED:
			return OS::POWERSTATE_CHARGED;
		default:
			return OS::POWERSTATE_UNKNOWN;
	}
}

int PowerSDL::get_power_seconds_left() {
	int seconds_left = -1;

	SDL_GetPowerInfo(&seconds_left, NULL);
	return seconds_left;
}

int PowerSDL::get_power_percent_left() {
	int power_percentage = -1;

	SDL_GetPowerInfo(NULL, &power_percentage);
	return power_percentage;
}
