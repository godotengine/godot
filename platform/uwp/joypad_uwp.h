/*************************************************************************/
/*  joypad_uwp.h                                                         */
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

#ifndef JOYPAD_UWP_H
#define JOYPAD_UWP_H

#include "core/input/input.h"

ref class JoypadUWP sealed {
	/** clang-format breaks this, it does not understand this token. */
	/* clang-format off */
internal:
	void register_events();
	void process_controllers();
	/* clang-format on */

	JoypadUWP();
	JoypadUWP(InputDefault *p_input);

private:
	enum {
		MAX_CONTROLLERS = 4,
	};

	enum ControllerType {
		GAMEPAD_CONTROLLER,
		ARCADE_STICK_CONTROLLER,
		RACING_WHEEL_CONTROLLER,
	};

	struct ControllerDevice {
		Windows::Gaming::Input::IGameController ^ controller_reference;

		int id = -1;
		bool connected = false;
		ControllerType type = ControllerType::GAMEPAD_CONTROLLER;
		float ff_timestamp = 0;
		float ff_end_timestamp = 0;
		bool vibrating = false;
	};

	ControllerDevice controllers[MAX_CONTROLLERS];

	InputDefault *input;

	void OnGamepadAdded(Platform::Object ^ sender, Windows::Gaming::Input::Gamepad ^ value);
	void OnGamepadRemoved(Platform::Object ^ sender, Windows::Gaming::Input::Gamepad ^ value);

	InputDefault::JoyAxisValue axis_correct(double p_val, bool p_negate = false, bool p_trigger = false) const;
	void joypad_vibration_start(int p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration, uint64_t p_timestamp);
	void joypad_vibration_stop(int p_device, uint64_t p_timestamp);
};

#endif
