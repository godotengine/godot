/*************************************************************************/
/*  joypad_uwp.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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

#include "joypad_uwp.h"

using namespace Windows::Gaming::Input;
using namespace Windows::Foundation;

void JoypadUWP::register_events() {

	Gamepad::GamepadAdded +=
			ref new EventHandler<Gamepad ^>(this, &JoypadUWP::OnGamepadAdded);
	Gamepad::GamepadRemoved +=
			ref new EventHandler<Gamepad ^>(this, &JoypadUWP::OnGamepadRemoved);
}

void JoypadUWP::process_controllers() {

	for (int i = 0; i < MAX_CONTROLLERS; i++) {

		if (!controllers[i].connected) break;

		switch (controllers[i].type) {

			case ControllerType::GAMEPAD_CONTROLLER: {

				GamepadReading reading = ((Gamepad ^)controllers[i].controller_reference)->GetCurrentReading();

				int button_mask = (int)GamepadButtons::Menu;
				for (int j = 0; j < 14; j++) {

					input->joy_button(controllers[i].id, j, (int)reading.Buttons & button_mask);
					button_mask *= 2;
				}

				input->joy_axis(controllers[i].id, JOY_AXIS_0, axis_correct(reading.LeftThumbstickX));
				input->joy_axis(controllers[i].id, JOY_AXIS_1, axis_correct(reading.LeftThumbstickY, true));
				input->joy_axis(controllers[i].id, JOY_AXIS_2, axis_correct(reading.RightThumbstickX));
				input->joy_axis(controllers[i].id, JOY_AXIS_3, axis_correct(reading.RightThumbstickY, true));
				input->joy_axis(controllers[i].id, JOY_AXIS_4, axis_correct(reading.LeftTrigger, false, true));
				input->joy_axis(controllers[i].id, JOY_AXIS_5, axis_correct(reading.RightTrigger, false, true));

				break;
			}
		}
	}
}

JoypadUWP::JoypadUWP() {

	for (int i = 0; i < MAX_CONTROLLERS; i++)
		controllers[i].id = i;
}

JoypadUWP::JoypadUWP(InputDefault *p_input) {

	input = p_input;

	JoypadUWP();
}

void JoypadUWP::OnGamepadAdded(Platform::Object ^ sender, Windows::Gaming::Input::Gamepad ^ value) {

	short idx = -1;

	for (int i = 0; i < MAX_CONTROLLERS; i++) {

		if (!controllers[i].connected) {
			idx = i;
			break;
		}
	}

	ERR_FAIL_COND(idx == -1);

	controllers[idx].connected = true;
	controllers[idx].controller_reference = value;
	controllers[idx].id = idx;
	controllers[idx].type = ControllerType::GAMEPAD_CONTROLLER;

	input->joy_connection_changed(controllers[idx].id, true, "Xbox Controller", "__UWP_GAMEPAD__");
}

void JoypadUWP::OnGamepadRemoved(Platform::Object ^ sender, Windows::Gaming::Input::Gamepad ^ value) {

	short idx = -1;

	for (int i = 0; i < MAX_CONTROLLERS; i++) {

		if (controllers[i].controller_reference == value) {
			idx = i;
			break;
		}
	}

	ERR_FAIL_COND(idx == -1);

	for (int i = idx + 1; i < MAX_CONTROLLERS - 1; i++) {

		if (!controllers[i].connected) {
			break;
		}

		controllers[i - 1] = controllers[i];
	}
	controllers[MAX_CONTROLLERS - 1] = ControllerDevice();

	input->joy_connection_changed(idx, false, "Xbox Controller");
}

InputDefault::JoyAxis JoypadUWP::axis_correct(double p_val, bool p_negate, bool p_trigger) const {

	InputDefault::JoyAxis jx;

	jx.min = p_trigger ? 0 : -1;
	jx.value = (float)(p_negate ? -p_val : p_val);

	return jx;
}
