/*************************************************************************/
/*  joypad_uwp.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "core/os/os.h"

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
		ControllerDevice &joy = controllers[i];

		if (!joy.connected)
			break;

		switch (joy.type) {
			case ControllerType::GAMEPAD_CONTROLLER: {
				GamepadReading reading = ((Gamepad ^) joy.controller_reference)->GetCurrentReading();

				int button_mask = (int)GamepadButtons::Menu;
				for (int j = 0; j < 14; j++) {
					input->joy_button(joy.id, j, (int)reading.Buttons & button_mask);
					button_mask *= 2;
				}

				input->joy_axis(joy.id, JoyAxis::LEFT_X, axis_correct(reading.LeftThumbstickX));
				input->joy_axis(joy.id, JoyAxis::LEFT_Y, axis_correct(reading.LeftThumbstickY, true));
				input->joy_axis(joy.id, JoyAxis::RIGHT_X, axis_correct(reading.RightThumbstickX));
				input->joy_axis(joy.id, JoyAxis::RIGHT_Y, axis_correct(reading.RightThumbstickY, true));
				input->joy_axis(joy.id, JoyAxis::TRIGGER_LEFT, axis_correct(reading.LeftTrigger, false, true));
				input->joy_axis(joy.id, JoyAxis::TRIGGER_RIGHT, axis_correct(reading.RightTrigger, false, true));

				uint64_t timestamp = input->get_joy_vibration_timestamp(joy.id);
				if (timestamp > joy.ff_timestamp) {
					Vector2 strength = input->get_joy_vibration_strength(joy.id);
					float duration = input->get_joy_vibration_duration(joy.id);
					if (strength.x == 0 && strength.y == 0) {
						joypad_vibration_stop(i, timestamp);
					} else {
						joypad_vibration_start(i, strength.x, strength.y, duration, timestamp);
					}
				} else if (joy.vibrating && joy.ff_end_timestamp != 0) {
					uint64_t current_time = OS::get_singleton()->get_ticks_usec();
					if (current_time >= joy.ff_end_timestamp)
						joypad_vibration_stop(i, current_time);
				}

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

	controllers[idx] = ControllerDevice();

	input->joy_connection_changed(idx, false, "Xbox Controller");
}

InputDefault::JoyAxisValue JoypadUWP::axis_correct(double p_val, bool p_negate, bool p_trigger) const {
	InputDefault::JoyAxisValue jx;

	jx.min = p_trigger ? 0 : -1;
	jx.value = (float)(p_negate ? -p_val : p_val);

	return jx;
}

void JoypadUWP::joypad_vibration_start(int p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration, uint64_t p_timestamp) {
	ControllerDevice &joy = controllers[p_device];
	if (joy.connected) {
		GamepadVibration vibration;
		vibration.LeftMotor = p_strong_magnitude;
		vibration.RightMotor = p_weak_magnitude;
		((Gamepad ^) joy.controller_reference)->Vibration = vibration;

		joy.ff_timestamp = p_timestamp;
		joy.ff_end_timestamp = p_duration == 0 ? 0 : p_timestamp + (uint64_t)(p_duration * 1000000.0);
		joy.vibrating = true;
	}
}

void JoypadUWP::joypad_vibration_stop(int p_device, uint64_t p_timestamp) {
	ControllerDevice &joy = controllers[p_device];
	if (joy.connected) {
		GamepadVibration vibration;
		vibration.LeftMotor = 0.0;
		vibration.RightMotor = 0.0;
		((Gamepad ^) joy.controller_reference)->Vibration = vibration;

		joy.ff_timestamp = p_timestamp;
		joy.vibrating = false;
	}
}
