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

#include "core/dictionary.h"
#include "core/os/time.h"
#include "main/default_controller_mappings.h"

#include <iterator>

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

JoypadSDL::JoypadSDL(InputDefault *in) {
	singleton = this;
	input = in;
}

JoypadSDL::~JoypadSDL() {
	// Process any remaining input events
	process_events();
	for (int i = 0; i < InputDefault::JOYPADS_MAX; i++) {
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
	for (int i = 0; i < InputDefault::JOYPADS_MAX; i++) {
		Joypad &joy = joypads[i];
		if (joy.attached && joy.supports_force_feedback) {
			uint64_t timestamp = input->get_joy_vibration_timestamp(i);

			// Update the joypad rumble only if there was a new vibration request
			if (timestamp > joy.ff_effect_timestamp) {
				joy.ff_effect_timestamp = timestamp;

				SDL_Joystick *sdl_joy = SDL_GetJoystickFromID(joypads[i].sdl_instance_idx);
				Vector2 strength = input->get_joy_vibration_strength(i);

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
						input->get_joy_vibration_duration(i) * 1000);
			}
		}
	}

	SDL_Event sdl_event;
	while (SDL_PollEvent(&sdl_event)) {
		// A new joypad was attached
		if (sdl_event.type == SDL_EVENT_JOYSTICK_ADDED) {
			int joy_id = input->get_unused_joy_id();
			if (joy_id == -1) {
				// There is no space for more joypads...
				print_error("A new joypad was attached but couldn't allocate a new id for it because joypad limit was reached.");
			} else {
				SDL_Joystick *joy = nullptr;
				String device_name;

				// Gamepads must be opened with SDL_OpenGamepad to get their special remapped events
				if (SDL_IsGamepad(sdl_event.jdevice.which)) {
					SDL_Gamepad *gamepad = SDL_OpenGamepad(sdl_event.jdevice.which);

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

				sdl_instance_id_to_joypad_id.set(sdl_event.jdevice.which, joy_id);

				input->joy_connection_changed(
						joy_id,
						true,
						device_name,
						joypads[joy_id].guid);
			}
			// An event for an attached joypad
		} else if (sdl_event.type >= SDL_EVENT_JOYSTICK_AXIS_MOTION && sdl_event.type < SDL_EVENT_FINGER_DOWN && sdl_instance_id_to_joypad_id.has(sdl_event.jdevice.which)) {
			int joy_id = sdl_instance_id_to_joypad_id.get(sdl_event.jdevice.which);

			switch (sdl_event.type) {
				case SDL_EVENT_JOYSTICK_REMOVED:
					input->joy_connection_changed(joy_id, false, "", "");
					close_joypad(joy_id);
					break;

				case SDL_EVENT_JOYSTICK_AXIS_MOTION:
					SKIP_EVENT_FOR_GAMEPAD;

					input->joy_axis(
							joy_id,
							map_sdl_axis_to_joystick_list(sdl_event.jaxis.axis),
							((sdl_event.jaxis.value - SDL_JOYSTICK_AXIS_MIN) / (float)(SDL_JOYSTICK_AXIS_MAX - SDL_JOYSTICK_AXIS_MIN) - 0.5f) * 2.0f);
					break;

				case SDL_EVENT_JOYSTICK_BUTTON_UP:
				case SDL_EVENT_JOYSTICK_BUTTON_DOWN:
					SKIP_EVENT_FOR_GAMEPAD;

					// Some devices report pressing buttons with indices like 232+, 241+, etc. that are not valid,
					// so we ignore them here.
					if (sdl_event.jbutton.button >= (int)SDLJoyButton::MAX) {
						continue;
					}

					input->joy_button(
							joy_id,
							map_sdl_button_to_joystick_list(sdl_event.jbutton.button),
							sdl_event.jbutton.down);
					break;

				case SDL_EVENT_JOYSTICK_HAT_MOTION:
					SKIP_EVENT_FOR_GAMEPAD;

					input->joy_hat(
							joy_id,
							(InputDefault::HatMask)sdl_event.jhat.value // Godot hat masks are identical to SDL hat masks, so we can just use them as-is.
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

					input->joy_axis(
							joy_id,
							map_sdl_axis_to_joystick_list(sdl_event.gaxis.axis),
							axis_value);
				} break;

				// Do note SDL gamepads do not have separate events for the dpad
				case SDL_EVENT_GAMEPAD_BUTTON_UP:
				case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
					input->joy_button(
							joy_id,
							map_sdl_button_to_joystick_list(sdl_event.gbutton.button),
							sdl_event.gbutton.down);
					break;
			}
		}
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

JoystickList JoypadSDL::map_sdl_button_to_joystick_list(int p_button) {
	SDLJoyButton sdl_joy_button = static_cast<SDLJoyButton>(p_button);
	switch (sdl_joy_button) {
		case SDLJoyButton::INVALID:
			return JOY_INVALID_OPTION;
		case SDLJoyButton::A:
			return JOY_XBOX_A;
		case SDLJoyButton::B:
			return JOY_XBOX_B;
		case SDLJoyButton::X:
			return JOY_XBOX_X;
		case SDLJoyButton::Y:
			return JOY_XBOX_Y;
		case SDLJoyButton::BACK:
			return JOY_SELECT;
		case SDLJoyButton::GUIDE:
			return JOY_GUIDE;
		case SDLJoyButton::START:
			return JOY_START;
		case SDLJoyButton::LEFT_STICK:
			return JOY_L3;
		case SDLJoyButton::RIGHT_STICK:
			return JOY_R3;
		case SDLJoyButton::LEFT_SHOULDER:
			return JOY_L;
		case SDLJoyButton::RIGHT_SHOULDER:
			return JOY_R;
		case SDLJoyButton::DPAD_UP:
			return JOY_DPAD_UP;
		case SDLJoyButton::DPAD_DOWN:
			return JOY_DPAD_DOWN;
		case SDLJoyButton::DPAD_LEFT:
			return JOY_DPAD_LEFT;
		case SDLJoyButton::DPAD_RIGHT:
			return JOY_DPAD_RIGHT;
		case SDLJoyButton::MISC1:
			return JOY_MISC1;
		case SDLJoyButton::PADDLE1:
			return JOY_PADDLE1;
		case SDLJoyButton::PADDLE2:
			return JOY_PADDLE2;
		case SDLJoyButton::PADDLE3:
			return JOY_PADDLE3;
		case SDLJoyButton::PADDLE4:
			return JOY_PADDLE4;
		case SDLJoyButton::TOUCHPAD:
			return JOY_TOUCHPAD;
		case SDLJoyButton::SDL_MAX:
			return JOY_BUTTON_MAX;
		case SDLJoyButton::MAX:
			return JOY_BUTTON_MAX;
		default:
			return static_cast<JoystickList>(p_button);
	}
}

JoystickList JoypadSDL::map_sdl_axis_to_joystick_list(int p_axis) {
	SDLJoyAxis sdl_joy_axis = static_cast<SDLJoyAxis>(p_axis);
	switch (sdl_joy_axis) {
		case SDLJoyAxis::INVALID:
			return JOY_INVALID_OPTION;
		case SDLJoyAxis::LEFT_X:
			return JOY_ANALOG_LX;
		case SDLJoyAxis::LEFT_Y:
			return JOY_ANALOG_LY;
		case SDLJoyAxis::RIGHT_X:
			return JOY_ANALOG_RX;
		case SDLJoyAxis::RIGHT_Y:
			return JOY_ANALOG_RY;
		case SDLJoyAxis::TRIGGER_LEFT:
			return JOY_ANALOG_L2;
		case SDLJoyAxis::TRIGGER_RIGHT:
			return JOY_ANALOG_R2;
		case SDLJoyAxis::SDL_MAX:
			return JOY_AXIS_MAX;
		case SDLJoyAxis::MAX:
			return JOY_AXIS_MAX;
		default:
			return static_cast<JoystickList>(p_axis);
	}
}

#endif // SDL_ENABLED
