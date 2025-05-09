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

#include <iterator>

#include "SDL3/SDL.h"
#include "SDL3/SDL_error.h"
#include "SDL3/SDL_events.h"
#include "SDL3/SDL_gamepad.h"
#include "SDL3/SDL_iostream.h"
#include "SDL3/SDL_joystick.h"

#include "core/input/default_controller_mappings.h"
#include "core/os/time.h"

JoypadSDL *JoypadSDL::singleton = nullptr;

#define HANDLE_SDL_ERROR(m_call) \
	error = m_call;              \
	ERR_FAIL_COND_V_MSG(error != 0, FAILED, SDL_GetError())

// Macro to skip the SDL joystick event handling if the device is an SDL gamepad, because
// there are separate events for SDL gamepads
#define SKIP_EVENT_FOR_GAMEPAD                            \
	if (is_sdl_device_gamepad(sdl_event.jdevice.which)) { \
		continue;                                         \
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

Error JoypadSDL::initialize() {
	SDL_SetHint(SDL_HINT_JOYSTICK_THREAD, "1");
	SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
	int error;
	HANDLE_SDL_ERROR(!SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_GAMEPAD));

	// Add godot's mapping database from memory
	int i = 0;
	while (DefaultControllerMappings::mappings[i]) {
		String mapping_string = DefaultControllerMappings::mappings[i++];
		CharString data = mapping_string.utf8();
		SDL_IOStream *rw = SDL_IOFromMem((void *)data.ptr(), data.size());
		SDL_AddGamepadMappingsFromIO(rw, 1);
	}

	print_verbose("SDL: Init OK!");
	return OK;
}

bool JoypadSDL::is_sdl_device_gamepad(int p_sdl_device_id) const {
	if (!sdl_instance_id_to_joypad_id.has(p_sdl_device_id)) {
		return false;
	}

	int joy_id = sdl_instance_id_to_joypad_id.get(p_sdl_device_id);
	return is_device_gamepad(joy_id);
}

void JoypadSDL::close_joypad(int p_pad_idx) const {
	int sdl_instance_idx = joypads[p_pad_idx].sdl_instance_idx;

	if (is_device_gamepad(p_pad_idx)) {
		SDL_Gamepad *gamepad = SDL_GetGamepadFromID(sdl_instance_idx);
		SDL_CloseGamepad(gamepad);
	} else {
		SDL_Joystick *joy = SDL_GetJoystickFromID(sdl_instance_idx);
		SDL_CloseJoystick(joy);
	}
}

JoypadSDL *JoypadSDL::get_singleton() {
	return singleton;
}

void JoypadSDL::process_events() {
	// Update rumble first for it to be applied when we handle SDL events
	for (int i = 0; i < Input::JOYPADS_MAX; i++) {
		Joypad &joy = joypads[i];
		if (joy.attached && joy.supports_force_feedback) {
			uint64_t timestamp = Input::get_singleton()->get_joy_vibration_timestamp(i);

			// Don't update the joypad rumble if there were no vibration requests
			if (timestamp > joy.ff_effect_timestamp) {
				joy.ff_effect_timestamp = timestamp;

				SDL_Joystick *sdl_joy = SDL_GetJoystickFromID(joypads[i].sdl_instance_idx);
				Vector2 strength = Input::get_singleton()->get_joy_vibration_strength(i);
				uint32_t duration_ms = 0;
				uint16_t weak = 0;
				uint16_t strong = 0;

				// If joypad rumble was requested to start
				if (strength.x != 0 || strength.y != 0) {
					duration_ms = Input::get_singleton()->get_joy_vibration_duration(i) * 1000;
					weak = strength.x * UINT16_MAX;
					strong = strength.y * UINT16_MAX;
				}
				// If joypad rumble was requested to stop, "weak" and "strong" variables are still 0 at this point
				// so when they're passed to SDL_RumbleJoystick the rumble stops

				bool result = SDL_RumbleJoystick(sdl_joy, weak, strong, duration_ms);

				// SDL_RumbleJoystick returns false if rumble is not supported
				if (!result) {
					print_error(vformat("Rumble is not supported on joypad %d.", i));
				}
			}
		}
	}

	SDL_Event sdl_event;
	while (SDL_PollEvent(&sdl_event)) {
		// A new joypad was attached
		if (sdl_event.type == SDL_EVENT_JOYSTICK_ADDED) {
			int joy_id = Input::get_singleton()->get_unused_joy_id();
			if (joy_id == -1) {
				// There ain't no space for more joypads...
				print_error("A new joypad was attached but couldn't allocate a new id for it because joypad limit was reached!");
			} else {
				SDL_Joystick *joy = nullptr;
				String device_name;
				JoypadType device_type;

				// Gamepads must be opened with SDL_OpenGamepad to get their special remapped events
				if (SDL_IsGamepad(sdl_event.jdevice.which)) {
					SDL_Gamepad *gamepad = SDL_OpenGamepad(sdl_event.jdevice.which);

					ERR_CONTINUE_MSG(!gamepad,
							vformat("Error opening gamepad at index %d: %s", sdl_event.jdevice.which, SDL_GetError()));

					device_type = JoypadType::GAMEPAD;
					device_name = SDL_GetGamepadName(gamepad);
					joy = SDL_GetGamepadJoystick(gamepad);

					print_verbose(vformat("SDL: Gamepad %s connected", SDL_GetGamepadName(gamepad)));

				} else {
					device_type = JoypadType::JOYSTICK;
					joy = SDL_OpenJoystick(sdl_event.jdevice.which);
					ERR_CONTINUE_MSG(!joy,
							vformat("Error opening joy device at index %d: %s", sdl_event.jdevice.which, SDL_GetError()));

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
				joypads[joy_id].type = device_type;
				joypads[joy_id].guid = StringName(String(guid));

				sdl_instance_id_to_joypad_id.insert(sdl_event.jdevice.which, joy_id);

				Input::get_singleton()->joy_connection_changed(
						joy_id,
						true,
						device_name,
						// Don't give joysticks of type gamepad a GUID to prevent godot from messing us up with its own remapping logic
						device_type == JoypadType::GAMEPAD ? "" : joypads[joy_id].guid);
			}
			// An event for an attached joypad
		} else if (sdl_event.type >= SDL_EVENT_JOYSTICK_AXIS_MOTION && sdl_event.type < SDL_EVENT_FINGER_DOWN && sdl_instance_id_to_joypad_id.has(sdl_event.jdevice.which)) {
			int joy_id = sdl_instance_id_to_joypad_id.get(sdl_event.jdevice.which);

			switch (sdl_event.type) {
				case SDL_EVENT_JOYSTICK_REMOVED: {
					MutexLock lock(joypad_mutexes[joy_id]);
					joypads[joy_id].attached = false;
					Input::get_singleton()->joy_connection_changed(joy_id, false, "");
					close_joypad(sdl_instance_id_to_joypad_id.get(sdl_event.jdevice.which));
					sdl_instance_id_to_joypad_id.erase(sdl_event.jdevice.which);
				} break;

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
}

bool JoypadSDL::is_device_gamepad(int p_joy_device_idx) const {
	return joypads[p_joy_device_idx].type == JoypadType::GAMEPAD;
}

StringName JoypadSDL::get_device_guid(int p_joy_device_idx) const {
	ERR_FAIL_INDEX_V(p_joy_device_idx, (int)std::size(joypads), "");
	ERR_FAIL_COND_V(!joypads[p_joy_device_idx].attached, "");

	return joypads[p_joy_device_idx].guid;
}

#ifdef WINDOWS_ENABLED
extern "C" {
HWND SDL_HelperWindow;
}

// Required for DInput joypads to work
void JoypadSDL::setup_sdl_helper_window(HWND p_hwnd) {
	SDL_HelperWindow = p_hwnd;
}
#endif

#endif // SDL_ENABLED
