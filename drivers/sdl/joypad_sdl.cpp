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

#include "core/error/error_macros.h"
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

void JoypadSDL::process_inputs_thread_func(void *p_userdata) {
	JoypadSDL *joy = static_cast<JoypadSDL *>(p_userdata);
	joy->process_inputs_run();
}

#define HANDLE_SDL_ERROR(call) \
	error = call;              \
	ERR_FAIL_COND_V_MSG(error != 0, FAILED, SDL_GetError())

// Macro to skip the SDL joystick event handling if the device is an SDL gamepad, because
// there are separate events for SDL gamepads
#define SKIP_EVENT_FOR_GAMEPAD                                      \
	if (SDL_GetGamepadFromID(sdl_event.jbutton.which) != nullptr) { \
		continue;                                                   \
	}

#define SETUP_JOYPAD_EVENT(event_type)                                \
	joypad_event.timestamp = Time::get_singleton()->get_ticks_usec(); \
	joypad_event.type = JoypadEventType::event_type;                  \
	joypad_event.sdl_joystick_instance_id = sdl_event.jbutton.which;

void JoypadSDL::process_inputs_run() {
	while (!process_inputs_exit.is_set()) {
		for (int i = 0; i < Input::JOYPADS_MAX; i++) {
			float ff_weak = 0.0f;
			float ff_strong = 0.0f;
			SDL_Joystick *joy = nullptr;
			uint32_t ff_duration_ms = 0;

			joypads_lock[i].lock();
			if (joypads[i].attached && joypads[i].supports_force_feedback && joypads[i].needs_ff_update) {
				joy = SDL_GetJoystickFromID(joypads[i].sdl_instance_idx);
				ff_weak = joypads[i].ff_weak;
				ff_strong = joypads[i].ff_strong;
				ff_duration_ms = joypads[i].ff_duration_ms;
				joypads[i].needs_ff_update = false;
			}
			joypads_lock[i].unlock();

			// Skip if we don't need to update a joypad's rumble
			// Or it may be that we've closed the joystick but the main thread isn't aware of this fact yet
			// because the event queue hasn't been processed
			if (joy == nullptr) {
				continue;
			}
			uint16_t weak = ff_weak * UINT16_MAX;
			uint16_t strong = ff_strong * UINT16_MAX;
			SDL_RumbleJoystick(joy, weak, strong, ff_duration_ms);
		}

		SDL_Event sdl_event;
		JoypadEvent joypad_event;
		int has_event = SDL_WaitEventTimeout(&sdl_event, 16);
		bool push_event = has_event;
		if (has_event != 0) {
			switch (sdl_event.type) {
				case SDL_EVENT_JOYSTICK_ADDED: {
					joypad_event.type = JoypadEventType::DEVICE_ADDED;
					SDL_Joystick *joy = nullptr;

					// Gamepads must be opened with SDL_OpenGamepad to get their special remapped events
					if (SDL_IsGamepad(sdl_event.jdevice.which)) {
						joypad_event.device_type = JoypadType::GAMEPAD;
						SDL_Gamepad *gamepad = SDL_OpenGamepad(sdl_event.jdevice.which);

						ERR_CONTINUE_MSG(!gamepad,
								vformat("Error opening gamepad at index %d: %s", sdl_event.jdevice.which, SDL_GetError()));

						joypad_event.device_name = SDL_GetGamepadName(gamepad);
						joy = SDL_GetGamepadJoystick(gamepad);

						print_verbose(vformat("SDL: Gamepad %s connected", SDL_GetGamepadName(gamepad)));

					} else {
						joypad_event.device_type = JoypadType::JOYSTICK;
						joy = SDL_OpenJoystick(sdl_event.jdevice.which);
						ERR_CONTINUE_MSG(!joy,
								vformat("Error opening joy device %d: %s", sdl_event.jdevice.which, SDL_GetError()));

						joypad_event.device_name = String(SDL_GetJoystickName(joy));

						print_verbose(vformat("SDL: Joystick %s connected", SDL_GetJoystickName(joy)));
					}

					joypad_event.sdl_joystick_instance_id = SDL_GetJoystickID(joy);

					const int MAX_GUID_SIZE = 64;
					char guid[MAX_GUID_SIZE] = {};

					SDL_GUIDToString(SDL_GetJoystickGUID(joy), guid, MAX_GUID_SIZE);
					joypad_event.device_guid = StringName(String(guid));
					SDL_PropertiesID propertiesID = SDL_GetJoystickProperties(joy);
					joypad_event.device_supports_force_feedback = SDL_GetBooleanProperty(propertiesID, SDL_PROP_JOYSTICK_CAP_RUMBLE_BOOLEAN, false);
				} break;

				case SDL_EVENT_JOYSTICK_REMOVED: {
					joypad_event.type = JoypadEventType::DEVICE_REMOVED;
					joypad_event.sdl_joystick_instance_id = sdl_event.jdevice.which;

					SDL_Gamepad *gamepad = SDL_GetGamepadFromID(sdl_event.jdevice.which);
					if (gamepad != nullptr) {
						SDL_CloseGamepad(gamepad);
					} else {
						SDL_CloseJoystick(SDL_GetJoystickFromID(sdl_event.jdevice.which));
					}
				} break;
				case SDL_EVENT_JOYSTICK_AXIS_MOTION:
					SKIP_EVENT_FOR_GAMEPAD;

					SETUP_JOYPAD_EVENT(AXIS);
					// Godot joy axis constants are already intentionally the same as SDL's
					joypad_event.axis = static_cast<JoyAxis>(sdl_event.jaxis.axis);

					joypad_event.value =
							((sdl_event.jaxis.value - SDL_JOYSTICK_AXIS_MIN) / (float)(SDL_JOYSTICK_AXIS_MAX - SDL_JOYSTICK_AXIS_MIN) - 0.5f) * 2.0f;
					break;

				case SDL_EVENT_JOYSTICK_BUTTON_UP:
				case SDL_EVENT_JOYSTICK_BUTTON_DOWN:
					SKIP_EVENT_FOR_GAMEPAD;

					SETUP_JOYPAD_EVENT(BUTTON);
					joypad_event.pressed = sdl_event.jbutton.down;

					// Godot button constants are intentionally the same as SDL's, so we can just straight up use them
					joypad_event.button = static_cast<JoyButton>(sdl_event.jbutton.button);
					break;

				case SDL_EVENT_JOYSTICK_HAT_MOTION:
					SKIP_EVENT_FOR_GAMEPAD;

					SETUP_JOYPAD_EVENT(HAT);
					// Godot hat masks are identical to SDL hat masks, so we can just use them as-is.
					joypad_event.hat_mask = (HatMask)sdl_event.jhat.value;
					break;

				case SDL_EVENT_GAMEPAD_AXIS_MOTION:
					SETUP_JOYPAD_EVENT(AXIS);
					// Godot joy axis constants are already intentionally the same as SDL's
					joypad_event.axis = static_cast<JoyAxis>(sdl_event.gaxis.axis);

					if (sdl_event.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || sdl_event.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
						// Gamepad triggers go from 0 to SDL_JOYSTICK_AXIS_MAX
						joypad_event.value = sdl_event.gaxis.value / (float)SDL_JOYSTICK_AXIS_MAX;
					} else {
						// Other axis go from SDL_JOYSTICK_AXIS_MIN to SDL_JOYSTICK_AXIS_MAX
						joypad_event.value =
								((sdl_event.gaxis.value - SDL_JOYSTICK_AXIS_MIN) / (float)(SDL_JOYSTICK_AXIS_MAX - SDL_JOYSTICK_AXIS_MIN) - 0.5f) * 2.0f;
					}
					break;

				// Do note SDL gamepads do not have separate events for the dpad
				case SDL_EVENT_GAMEPAD_BUTTON_UP:
				case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
					SETUP_JOYPAD_EVENT(BUTTON);
					joypad_event.pressed = sdl_event.gbutton.down;

					// Godot button constants are intentionally the same as SDL's, so we can just straight up use them
					joypad_event.button = static_cast<JoyButton>(sdl_event.gbutton.button);
					break;

				// No handled joypad events happened so nothing to push
				default:
					push_event = false;
					break;
			}
		}
		if (push_event) {
			MutexLock lock(joypad_event_queue_lock);
			joypad_event_queue.push_back(joypad_event);
		}
	}
}

void JoypadSDL::joypad_vibration_start(int p_pad_idx, float p_weak, float p_strong, float p_duration, uint64_t p_timestamp) {
	Joypad &pad = joypads[p_pad_idx];

	uint32_t duration_msec = p_duration * 1000;

	MutexLock lock(joypads_lock[p_pad_idx]);
	pad.needs_ff_update = true;
	pad.ff_duration_ms = duration_msec;
	pad.ff_weak = p_weak;
	pad.ff_strong = p_strong;
	pad.ff_effect_timestamp = p_timestamp;
}

void JoypadSDL::joypad_vibration_stop(int p_pad_idx, uint64_t p_timestamp) {
	Joypad &pad = joypads[p_pad_idx];

	MutexLock lock(joypads_lock[p_pad_idx]);
	pad.needs_ff_update = true;
	pad.ff_duration_ms = 0;
	pad.ff_weak = 0;
	pad.ff_strong = 0;
	pad.ff_effect_timestamp = p_timestamp;
}

JoypadSDL *JoypadSDL::get_singleton() {
	return singleton;
}

JoypadSDL::JoypadSDL(Input *p_input) {
	input = p_input;
	singleton = this;
}

JoypadSDL::~JoypadSDL() {
	if (process_inputs_thread.is_started()) {
		process_inputs_exit.set();
		process_inputs_thread.wait_to_finish();
		// Process any remaining input events
		process_events();
		for (int i = 0; i < Input::JOYPADS_MAX; i++) {
			if (joypads[i].attached) {
				if (joypads[i].type == JoypadType::GAMEPAD) {
					SDL_Gamepad *controller = SDL_GetGamepadFromID(joypads[i].sdl_instance_idx);
					SDL_CloseGamepad(controller);
				} else {
					SDL_Joystick *joy = SDL_GetJoystickFromID(joypads[i].sdl_instance_idx);
					SDL_CloseJoystick(joy);
				}
			}
		}
		SDL_Quit();
	}
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

	process_inputs_thread.start(&JoypadSDL::process_inputs_thread_func, this);
	return OK;
}

void JoypadSDL::process_events() {
	Vector<JoypadEvent> events;
	joypad_event_queue_lock.lock();
	events = joypad_event_queue;
	joypad_event_queue.clear();
	joypad_event_queue_lock.unlock();

	for (int i = 0; i < events.size(); i++) {
		JoypadEvent event = events[i];

		if (event.type == DEVICE_ADDED) {
			int joy_id = Input::get_singleton()->get_unused_joy_id();
			if (joy_id == -1) {
				// There ain't no space for more joypads...
				print_error("Joypad limit reached!");
			} else {
				joypads[joy_id].attached = true;
				joypads[joy_id].sdl_instance_idx = event.sdl_joystick_instance_id;
				joypads[joy_id].supports_force_feedback = event.device_supports_force_feedback;
				joypads[joy_id].type = event.device_type;
				joypads[joy_id].guid = event.device_guid;

				sdl_instance_id_to_joypad_id.insert(event.sdl_joystick_instance_id, joy_id);
				// Don't give joysticks of type gamepad a GUID to prevent godot from messing us up with its own remapping logic
				if (event.device_type == JoypadType::GAMEPAD) {
					input->joy_connection_changed(joy_id, true, event.device_name, "");
				} else {
					input->joy_connection_changed(joy_id, true, event.device_name, event.device_guid);
				}
			}
		} else if (sdl_instance_id_to_joypad_id.has(event.sdl_joystick_instance_id)) {
			int joy_id = sdl_instance_id_to_joypad_id.get(event.sdl_joystick_instance_id);

			switch (event.type) {
				case DEVICE_REMOVED: {
					MutexLock lock(joypads_lock[joy_id]);
					joypads[joy_id].attached = false;
					sdl_instance_id_to_joypad_id.erase(event.sdl_joystick_instance_id);
					input->joy_connection_changed(joy_id, false, "");

					joypads[joy_id].needs_ff_update = false;
				} break;
				case AXIS:
					input->joy_axis(joy_id, event.axis, event.value);
					break;
				case BUTTON:
					input->joy_button(joy_id, event.button, event.pressed);
					break;
				case HAT:
					input->joy_hat(joy_id, event.hat_mask);
					break;
				// To stop GCC from complaining
				default:
					break;
			}
		}
	}
	for (int i = 0; i < Input::JOYPADS_MAX; i++) {
		Joypad &joy = joypads[i];
		if (joy.attached && joy.supports_force_feedback) {
			uint64_t timestamp = input->get_joy_vibration_timestamp(i);
			if (timestamp > joy.ff_effect_timestamp) {
				Vector2 strength = input->get_joy_vibration_strength(i);
				float duration = input->get_joy_vibration_duration(i);
				if (strength.x == 0 && strength.y == 0) {
					joypad_vibration_stop(i, timestamp);
				} else {
					joypad_vibration_start(i, strength.x, strength.y, duration, timestamp);
				}
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

#endif // SDL_ENABLED
