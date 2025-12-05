/**************************************************************************/
/*  joypad_sdl.cpp                                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "joypad_sdl.h"

#ifdef SDL_ENABLED

#include "core/input/default_controller_mappings.h"
#include "core/os/time.h"
#include "core/variant/dictionary.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_iostream.h>
#include <SDL3/SDL_joystick.h>

JoypadSDL *JoypadSDL::singleton = nullptr;

// Macro to skip the SDL joystick event handling if the device is an SDL gamepad, because
// there are separate events for SDL gamepads
#define SKIP_EVENT_FOR_GAMEPAD                    \
	if (SDL_IsGamepad(sdl_event.jdevice.which)) { \
		continue;                                 \
	}

JoypadSDL::JoypadSDL() {
	singleton = this;
}

JoypadSDL::~JoypadSDL() {
	// Process any remaining input events
	process_events();
	for (int i = 0; i < Input::JOYPADS_MAX; i++) {
		if (joypads[i].attached) {
			close_joypad(i);
		}
	}
	SDL_Quit();
	singleton = nullptr;
}

JoypadSDL *JoypadSDL::get_singleton() {
	return singleton;
}

Error JoypadSDL::initialize() {
	SDL_SetHint(SDL_HINT_JOYSTICK_THREAD, "1");
	SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
	ERR_FAIL_COND_V_MSG(!SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_GAMEPAD), FAILED, SDL_GetError());

	// Add Godot's mapping database from memory
	int i = 0;
	while (DefaultControllerMappings::mappings[i]) {
		String mapping_string = DefaultControllerMappings::mappings[i++];
		CharString data = mapping_string.utf8();
		SDL_IOStream *rw = SDL_IOFromMem((void *)data.ptr(), data.size());
		SDL_AddGamepadMappingsFromIO(rw, 1);
	}

	// Make sure that we handle already connected joypads when the driver is initialized.
	process_events();

	print_verbose("SDL: Init OK!");
	return OK;
}

void JoypadSDL::process_events() {
	// Update rumble first for it to be applied when we handle SDL events
	for (int i = 0; i < Input::JOYPADS_MAX; i++) {
		Joypad &joy = joypads[i];
		if (joy.attached && joy.supports_force_feedback) {
			uint64_t timestamp = Input::get_singleton()->get_joy_vibration_timestamp(i);

			// Update the joypad rumble only if there was a new vibration request
			if (timestamp > joy.ff_effect_timestamp) {
				joy.ff_effect_timestamp = timestamp;

				SDL_Joystick *sdl_joy = SDL_GetJoystickFromID(joypads[i].sdl_instance_idx);
				Vector2 strength = Input::get_singleton()->get_joy_vibration_strength(i);

				/*
					If the vibration was requested to start, SDL_RumbleJoystick will start it.
					If the vibration was requested to stop, strength and duration will be 0, so SDL will stop the rumble.

					Here strength.y goes first and then strength.x, because Input.get_joy_vibration_strength().x
					is vibration's weak magnitude (high frequency rumble), and .y is strong magnitude (low frequency rumble),
					SDL_RumbleJoystick takes low frequency rumble first and then high frequency rumble.
				*/
				SDL_RumbleJoystick(
						sdl_joy,
						// Rumble strength goes from 0 to 0xFFFF
						strength.y * UINT16_MAX,
						strength.x * UINT16_MAX,
						Input::get_singleton()->get_joy_vibration_duration(i) * 1000);
			}
		}
	}

	SDL_Event sdl_event;
	while (SDL_PollEvent(&sdl_event)) {
		// A new joypad was attached
		if (sdl_event.type == SDL_EVENT_JOYSTICK_ADDED) {
			int joy_id = Input::get_singleton()->get_unused_joy_id();
			if (joy_id == -1) {
				// There is no space for more joypads...
				print_error("A new joypad was attached but couldn't allocate a new id for it because joypad limit was reached.");
			} else {
				SDL_Joystick *joy = nullptr;
				SDL_Gamepad *gamepad = nullptr;
				String device_name;

				// Gamepads must be opened with SDL_OpenGamepad to get their special remapped events
				if (SDL_IsGamepad(sdl_event.jdevice.which)) {
					gamepad = SDL_OpenGamepad(sdl_event.jdevice.which);

					ERR_CONTINUE_MSG(!gamepad,
							vformat("Error opening gamepad at index %d: %s", sdl_event.jdevice.which, SDL_GetError()));

					device_name = SDL_GetGamepadName(gamepad);
					joy = SDL_GetGamepadJoystick(gamepad);

					print_verbose(vformat("SDL: Gamepad %s connected", SDL_GetGamepadName(gamepad)));
				} else {
					joy = SDL_OpenJoystick(sdl_event.jdevice.which);
					ERR_CONTINUE_MSG(!joy,
							vformat("Error opening joystick at index %d: %s", sdl_event.jdevice.which, SDL_GetError()));

					device_name = SDL_GetJoystickName(joy);

					print_verbose(vformat("SDL: Joystick %s connected", SDL_GetJoystickName(joy)));
				}

				const int MAX_GUID_SIZE = 64;
				char guid[MAX_GUID_SIZE] = {};

				SDL_GUIDToString(SDL_GetJoystickGUID(joy), guid, MAX_GUID_SIZE);
				SDL_PropertiesID propertiesID = SDL_GetJoystickProperties(joy);

				joypads[joy_id].attached = true;
				joypads[joy_id].sdl_instance_idx = sdl_event.jdevice.which;
				joypads[joy_id].supports_force_feedback = SDL_GetBooleanProperty(propertiesID, SDL_PROP_JOYSTICK_CAP_RUMBLE_BOOLEAN, false);
				joypads[joy_id].guid = StringName(String(guid));
				joypads[joy_id].supports_sensors = SDL_GamepadHasSensor(gamepad, SDL_SENSOR_ACCEL) && SDL_GamepadHasSensor(gamepad, SDL_SENSOR_GYRO);

				sdl_instance_id_to_joypad_id.insert(sdl_event.jdevice.which, joy_id);

				Dictionary joypad_info;
				// Skip Godot's mapping system if SDL already handles the joypad's mapping.
				joypad_info["mapping_handled"] = SDL_IsGamepad(sdl_event.jdevice.which);
				joypad_info["raw_name"] = String(SDL_GetJoystickName(joy));
				joypad_info["vendor_id"] = itos(SDL_GetJoystickVendor(joy));
				joypad_info["product_id"] = itos(SDL_GetJoystickProduct(joy));

				const uint64_t steam_handle = SDL_GetGamepadSteamHandle(gamepad);
				if (steam_handle != 0) {
					joypad_info["steam_input_index"] = itos(steam_handle);
				}

				const int player_index = SDL_GetJoystickPlayerIndex(joy);
				if (player_index >= 0) {
					// For XInput controllers SDL_GetJoystickPlayerIndex returns the XInput user index.
					joypad_info["xinput_index"] = itos(player_index);
				}

				Input::get_singleton()->joy_connection_changed(
						joy_id,
						true,
						device_name,
						joypads[joy_id].guid,
						joypad_info);

				Input::get_singleton()->set_joy_features(joy_id, &joypads[joy_id]);
			}
			// An event for an attached joypad
		} else if (sdl_event.type >= SDL_EVENT_JOYSTICK_AXIS_MOTION && sdl_event.type < SDL_EVENT_FINGER_DOWN && sdl_instance_id_to_joypad_id.has(sdl_event.jdevice.which)) {
			int joy_id = sdl_instance_id_to_joypad_id.get(sdl_event.jdevice.which);

			switch (sdl_event.type) {
				case SDL_EVENT_JOYSTICK_REMOVED:
					Input::get_singleton()->joy_connection_changed(joy_id, false, "");
					close_joypad(joy_id);
					break;

				case SDL_EVENT_JOYSTICK_AXIS_MOTION:
					SKIP_EVENT_FOR_GAMEPAD;

					Input::get_singleton()->joy_axis(
							joy_id,
							static_cast<JoyAxis>(sdl_event.jaxis.axis), // Godot joy axis constants are already intentionally the same as SDL's
							((sdl_event.jaxis.value - SDL_JOYSTICK_AXIS_MIN) / (float)(SDL_JOYSTICK_AXIS_MAX - SDL_JOYSTICK_AXIS_MIN) - 0.5f) * 2.0f);
					break;

				case SDL_EVENT_JOYSTICK_BUTTON_UP:
				case SDL_EVENT_JOYSTICK_BUTTON_DOWN:
					SKIP_EVENT_FOR_GAMEPAD;

					// Some devices report pressing buttons with indices like 232+, 241+, etc. that are not valid,
					// so we ignore them here.
					if (sdl_event.jbutton.button >= (int)JoyButton::MAX) {
						continue;
					}

					Input::get_singleton()->joy_button(
							joy_id,
							static_cast<JoyButton>(sdl_event.jbutton.button), // Godot button constants are intentionally the same as SDL's, so we can just straight up use them
							sdl_event.jbutton.down);
					break;

				case SDL_EVENT_JOYSTICK_HAT_MOTION:
					SKIP_EVENT_FOR_GAMEPAD;

					Input::get_singleton()->joy_hat(
							joy_id,
							(HatMask)sdl_event.jhat.value // Godot hat masks are identical to SDL hat masks, so we can just use them as-is.
					);
					break;

				case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
					float axis_value;

					if (sdl_event.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || sdl_event.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
						// Gamepad triggers go from 0 to SDL_JOYSTICK_AXIS_MAX
						axis_value = sdl_event.gaxis.value / (float)SDL_JOYSTICK_AXIS_MAX;
					} else {
						// Other axis go from SDL_JOYSTICK_AXIS_MIN to SDL_JOYSTICK_AXIS_MAX
						axis_value =
								((sdl_event.gaxis.value - SDL_JOYSTICK_AXIS_MIN) / (float)(SDL_JOYSTICK_AXIS_MAX - SDL_JOYSTICK_AXIS_MIN) - 0.5f) * 2.0f;
					}

					Input::get_singleton()->joy_axis(
							joy_id,
							static_cast<JoyAxis>(sdl_event.gaxis.axis), // Godot joy axis constants are already intentionally the same as SDL's
							axis_value);
				} break;

				// Do note SDL gamepads do not have separate events for the dpad
				case SDL_EVENT_GAMEPAD_BUTTON_UP:
				case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
					Input::get_singleton()->joy_button(
							joy_id,
							static_cast<JoyButton>(sdl_event.gbutton.button), // Godot button constants are intentionally the same as SDL's, so we can just straight up use them
							sdl_event.gbutton.down);
					break;
			}
		}
	}

	for (int i = 0; i < Input::JOYPADS_MAX; i++) {
		Joypad &joy = joypads[i];
		if (!joy.attached || !joy.supports_sensors) {
			continue;
		}
		SDL_Gamepad *gamepad = SDL_GetGamepadFromID(joy.sdl_instance_idx);
		// gamepad should not be NULL since joy.supports_sensors is true here.

		float accel_data[3];
		float gyro_data[3];
		SDL_GetGamepadSensorData(gamepad, SDL_SENSOR_ACCEL, accel_data, 3);
		SDL_GetGamepadSensorData(gamepad, SDL_SENSOR_GYRO, gyro_data, 3);

		Input::get_singleton()->process_joy_sensors(
				i,
				Vector3(-accel_data[0], -accel_data[1], -accel_data[2]),
				Vector3(gyro_data[0], gyro_data[1], gyro_data[2]));

		float data_rate = SDL_GetGamepadSensorDataRate(
				SDL_GetGamepadFromID(sdl_event.gsensor.which),
				SDL_SENSOR_ACCEL); // Data rate for all sensors should be the same.

		Input::get_singleton()->set_joy_sensor_rate(i, data_rate);
	}
}

void JoypadSDL::close_joypad(int p_pad_idx) {
	int sdl_instance_idx = joypads[p_pad_idx].sdl_instance_idx;

	joypads[p_pad_idx].attached = false;
	sdl_instance_id_to_joypad_id.erase(sdl_instance_idx);

	if (SDL_IsGamepad(sdl_instance_idx)) {
		SDL_Gamepad *gamepad = SDL_GetGamepadFromID(sdl_instance_idx);
		SDL_CloseGamepad(gamepad);
	} else {
		SDL_Joystick *joy = SDL_GetJoystickFromID(sdl_instance_idx);
		SDL_CloseJoystick(joy);
	}
}

bool JoypadSDL::Joypad::has_joy_light() const {
	SDL_PropertiesID properties_id = SDL_GetJoystickProperties(get_sdl_joystick());
	if (properties_id == 0) {
		return false;
	}
	return SDL_GetBooleanProperty(properties_id, SDL_PROP_JOYSTICK_CAP_RGB_LED_BOOLEAN, false) || SDL_GetBooleanProperty(properties_id, SDL_PROP_JOYSTICK_CAP_MONO_LED_BOOLEAN, false);
}

bool JoypadSDL::Joypad::set_joy_light(const Color &p_color) {
	Color linear = p_color.srgb_to_linear();
	return SDL_SetJoystickLED(get_sdl_joystick(), linear.get_r8(), linear.get_g8(), linear.get_b8());
}

bool JoypadSDL::Joypad::has_joy_sensors() const {
	return supports_sensors;
}

bool JoypadSDL::Joypad::set_joy_sensors_enabled(bool p_enable) {
	SDL_Gamepad *gamepad = get_sdl_gamepad();
	return SDL_SetGamepadSensorEnabled(gamepad, SDL_SENSOR_ACCEL, p_enable) && SDL_SetGamepadSensorEnabled(gamepad, SDL_SENSOR_GYRO, p_enable);
}

SDL_Joystick *JoypadSDL::Joypad::get_sdl_joystick() const {
	return SDL_GetJoystickFromID(sdl_instance_idx);
}

SDL_Gamepad *JoypadSDL::Joypad::get_sdl_gamepad() const {
	return SDL_GetGamepadFromID(sdl_instance_idx);
}

#endif // SDL_ENABLED
