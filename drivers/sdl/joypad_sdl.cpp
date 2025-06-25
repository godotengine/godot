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

#include <iterator>

#include <SDL3/SDL.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_iostream.h>
#include <SDL3/SDL_joystick.h>
#include <SDL3/SDL_log.h>

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

static void SDLCALL sdl_log(void *userdata, int category, SDL_LogPriority priority, const char *message) {
	print_verbose(vformat("SDL Debug (priority: %d): %s", priority, message));
}

Error JoypadSDL::initialize() {
	SDL_SetLogOutputFunction(sdl_log, nullptr);
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

				// If the vibration was requested to start, SDL_RumbleJoystick will start it.
				// If the vibration was requested to stop, strength and duration will be 0, so SDL will stop the rumble.
				SDL_RumbleJoystick(
						sdl_joy,
						// Rumble strength goes from 0 to 0xFFFF
						strength.x * UINT16_MAX,
						strength.y * UINT16_MAX,
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

				sdl_instance_id_to_joypad_id.insert(sdl_event.jdevice.which, joy_id);

				// Skip Godot's mapping system because SDL already handles the joypad's mapping
				Dictionary joypad_info;
				joypad_info["mapping_handled"] = true;

				Input::get_singleton()->joy_connection_changed(
						joy_id,
						true,
						device_name,
						joypads[joy_id].guid,
						joypad_info);
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

				case SDL_EVENT_JOYSTICK_BATTERY_UPDATED:
					// Gamepads also can have battery, so no SKIP_EVENT_FOR_GAMEPAD here

					Input::get_singleton()->set_joy_power_info(
							joy_id,
							static_cast<JoyPowerState>(sdl_event.jbattery.state),
							sdl_event.jbattery.percent);
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

				case SDL_EVENT_GAMEPAD_SENSOR_UPDATE: {
					// Godot currently doesn't support anything other than the main accelerometer and the main gyroscope sensors
					if (sdl_event.gsensor.sensor != SDL_SENSOR_ACCEL && sdl_event.gsensor.sensor != SDL_SENSOR_GYRO) {
						continue;
					}

					Vector3 value = Vector3(
							sdl_event.gsensor.data[0],
							sdl_event.gsensor.data[1],
							sdl_event.gsensor.data[2]);

					if (sdl_event.gsensor.sensor == SDL_SENSOR_ACCEL) {
						Input::get_singleton()->set_joy_accelerometer(joy_id, value);
					} else if (sdl_event.gsensor.sensor == SDL_SENSOR_GYRO) {
						// By default the rotation is positive in the counter-clockwise direction.
						// We revert it here to be positive in the clockwise direction.
						Input::get_singleton()->set_joy_gyroscope(joy_id, -value);
					}

					float data_rate = SDL_GetGamepadSensorDataRate(
							SDL_GetGamepadFromID(sdl_event.gsensor.which),
							(SDL_SensorType)sdl_event.gsensor.sensor);

					// Data rate for all sensors should be the same
					Input::get_singleton()->set_joy_sensor_rate(joy_id, data_rate);
				} break;

				case SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN:
				case SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION:
				case SDL_EVENT_GAMEPAD_TOUCHPAD_UP:
					Input::get_singleton()->set_joy_touchpad_finger(
							joy_id,
							sdl_event.gtouchpad.touchpad,
							sdl_event.gtouchpad.finger,
							sdl_event.gtouchpad.pressure,
							Vector2(sdl_event.gtouchpad.x, sdl_event.gtouchpad.y));
					break;
			}
		}
	}
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

bool JoypadSDL::enable_accelerometer(int p_pad_idx, bool p_enable) {
	bool result = SDL_SetGamepadSensorEnabled(get_sdl_gamepad(p_pad_idx), SDL_SENSOR_ACCEL, p_enable);
	if (!result) {
		print_verbose(vformat("Error while trying to enable joypad accelerometer: %s", SDL_GetError()));
	}
	return result;
}

bool JoypadSDL::enable_gyroscope(int p_pad_idx, bool p_enable) {
	bool result = SDL_SetGamepadSensorEnabled(get_sdl_gamepad(p_pad_idx), SDL_SENSOR_GYRO, p_enable);
	if (!result) {
		print_verbose(vformat("Error while trying to enable joypad gyroscope: %s", SDL_GetError()));
	}
	return result;
}

bool JoypadSDL::set_light(int p_pad_idx, Color p_color) {
	Color linear = p_color.srgb_to_linear();
	return SDL_SetJoystickLED(get_sdl_joystick(p_pad_idx), linear.get_r8(), linear.get_g8(), linear.get_b8());
}

bool JoypadSDL::has_joy_axis(int p_pad_idx, JoyAxis p_axis) const {
	SDL_Gamepad *gamepad = get_sdl_gamepad(p_pad_idx);
	if (gamepad != nullptr) {
		return SDL_GamepadHasAxis(gamepad, static_cast<SDL_GamepadAxis>(p_axis));
	}

	SDL_Joystick *joystick = get_sdl_joystick(p_pad_idx);
	if (joystick != nullptr) {
		return (int)p_axis >= 0 && (int)p_axis < SDL_GetNumJoystickAxes(joystick);
	}

	return false;
}

bool JoypadSDL::has_joy_button(int p_pad_idx, JoyButton p_button) const {
	SDL_Gamepad *gamepad = get_sdl_gamepad(p_pad_idx);
	if (gamepad != nullptr) {
		return SDL_GamepadHasButton(gamepad, static_cast<SDL_GamepadButton>(p_button));
	}

	SDL_Joystick *joystick = get_sdl_joystick(p_pad_idx);
	if (joystick != nullptr) {
		return (int)p_button >= 0 && (int)p_button < SDL_GetNumJoystickButtons(joystick);
	}

	return false;
}

String JoypadSDL::get_model_axis_string(JoyModel p_model, JoyAxis p_axis) {
	if (p_model == JoyModel::INVALID || p_model == JoyModel::UNKNOWN) {
		return "";
	}

	SDL_GamepadType gamepad_type = static_cast<SDL_GamepadType>(p_model);

	switch (p_axis) {
		case JoyAxis::LEFT_X:
			return "Left Stick X";
		case JoyAxis::LEFT_Y:
			return "Left Stick Y";
		case JoyAxis::RIGHT_X:
			return "Right Stick X";
		case JoyAxis::RIGHT_Y:
			return "Right Stick Y";

		case JoyAxis::TRIGGER_LEFT:
		case JoyAxis::TRIGGER_RIGHT:
			switch (gamepad_type) {
				case SDL_GAMEPAD_TYPE_XBOX360:
				case SDL_GAMEPAD_TYPE_XBOXONE:
					return p_axis == JoyAxis::TRIGGER_LEFT ? "LT" : "RT";

				case SDL_GAMEPAD_TYPE_PS3:
				case SDL_GAMEPAD_TYPE_PS4:
				case SDL_GAMEPAD_TYPE_PS5:
					return p_axis == JoyAxis::TRIGGER_LEFT ? "L2" : "R2";

				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO:
				//case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT: // Horizontal joycons don't have "trigger" buttons
				//case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT:
				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR:
					return p_axis == JoyAxis::TRIGGER_LEFT ? "ZL" : "ZR";

				default:
					return p_axis == JoyAxis::TRIGGER_LEFT ? "Left Trigger" : "Right Trigger";
			}

		default:
			break;
	}

	return "";
}

static const char *face_buttons_strings[] = {
	// Xbox/Nintendo names
	"A",
	"B",
	"X",
	"Y",
	// PlayStation names
	"Cross",
	"Circle",
	"Square",
	"Triangle",
};

static const char *horiz_joycon_face_buttons_strings[] = {
	// Horizontal joycons names
	"Face South",
	"Face East",
	"Face West",
	"Face North",
};

static const char *xb360_buttons[] = {
	"Back",
	"Guide",
	"Start",
};

static const char *xbone_buttons[] = {
	"View",
	"Xbox",
	"Menu",
};

static const char *ps3_buttons[] = {
	"Select",
	"PS",
	"Start",
};

static const char *ps45_buttons[] = {
	"Share",
	"PS",
	"Options",
};

static const char *switch_pro_buttons[] = {
	"Minus",
	"Home",
	"Plus",
};

static const char *default_paddles[] = {
	"Paddle 1",
	"Paddle 2",
	"Paddle 3",
	"Paddle 4",
};

static const char *dualsense_edge_paddles[] = {
	"Left Function",
	"Right Function",
	"Left Paddle",
	"Right Paddle",
};

static const char *joycon_paddles[] = {
	"SR (R)",
	"SL (L)",
	"SL (R)",
	"SR (L)",
};

static const char *joycon_horizontal_paddles[] = {
	"R (R)",
	"L (L)",
	"ZR (R)",
	"ZL (L)",
};

String JoypadSDL::get_model_button_string(JoyModel p_model, JoyButton p_button) {
	if (p_model == JoyModel::INVALID || p_model == JoyModel::UNKNOWN) {
		return "";
	}

	SDL_GamepadType gamepad_type = static_cast<SDL_GamepadType>(p_model);

	switch (p_button) {
		case JoyButton::A:
		case JoyButton::B:
		case JoyButton::X:
		case JoyButton::Y: {
			if (gamepad_type == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT || gamepad_type == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT) {
				return horiz_joycon_face_buttons_strings[(int)p_button - (int)JoyButton::A];
			}

			SDL_GamepadButtonLabel button_label = SDL_GetGamepadButtonLabelForType(gamepad_type, static_cast<SDL_GamepadButton>(p_button));

			// button_label == SDL_GAMEPAD_BUTTON_LABEL_UNKNOWN
			// Will SDL add new values that are less than 0? Not sure, so we make a check here just in case.
			if (button_label < SDL_GAMEPAD_BUTTON_LABEL_A || button_label > SDL_GAMEPAD_BUTTON_LABEL_TRIANGLE) {
				return "";
			}
			return face_buttons_strings[button_label - SDL_GAMEPAD_BUTTON_LABEL_A];
		}

		case JoyButton::BACK:
		case JoyButton::GUIDE:
		case JoyButton::START:
			switch (gamepad_type) {
				default:
				case SDL_GAMEPAD_TYPE_STANDARD:
				case SDL_GAMEPAD_TYPE_XBOX360:
					return xb360_buttons[(int)p_button - (int)JoyButton::BACK];

				case SDL_GAMEPAD_TYPE_XBOXONE:
					return xbone_buttons[(int)p_button - (int)JoyButton::BACK];

				case SDL_GAMEPAD_TYPE_PS3:
					return ps3_buttons[(int)p_button - (int)JoyButton::BACK];

				case SDL_GAMEPAD_TYPE_PS4:
				case SDL_GAMEPAD_TYPE_PS5:
					return ps45_buttons[(int)p_button - (int)JoyButton::BACK];

				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO:
				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR:
					return switch_pro_buttons[(int)p_button - (int)JoyButton::BACK];

				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT:
					// Plus button doesn't exist on the left joycon
					return p_button == JoyButton::GUIDE ? "Capture" : "Minus";
				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT:
					// Minus button doesn't exit on the right joycon
					return p_button == JoyButton::GUIDE ? "Home" : "Plus";
			}

		case JoyButton::LEFT_STICK:
			return "Left Stick";
		case JoyButton::RIGHT_STICK:
			return "Right Stick";

		case JoyButton::LEFT_SHOULDER:
		case JoyButton::RIGHT_SHOULDER:
			switch (gamepad_type) {
				case SDL_GAMEPAD_TYPE_XBOX360:
				case SDL_GAMEPAD_TYPE_XBOXONE:
					return p_button == JoyButton::LEFT_SHOULDER ? "LB" : "RB";

				case SDL_GAMEPAD_TYPE_PS3:
				case SDL_GAMEPAD_TYPE_PS4:
				case SDL_GAMEPAD_TYPE_PS5:
					return p_button == JoyButton::LEFT_SHOULDER ? "L1" : "R1";

				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO:
				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR:
					return p_button == JoyButton::LEFT_SHOULDER ? "L" : "R";
				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT:
				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT:
					return p_button == JoyButton::LEFT_SHOULDER ? "SL" : "SR";

				default:
					return p_button == JoyButton::LEFT_SHOULDER ? "Left Shoulder" : "Right Shoulder";
			}

		case JoyButton::DPAD_UP:
			return "D-pad Up";
		case JoyButton::DPAD_DOWN:
			return "D-pad Down";
		case JoyButton::DPAD_LEFT:
			return "D-pad Left";
		case JoyButton::DPAD_RIGHT:
			return "D-pad Right";

		case JoyButton::MISC1:
			switch (gamepad_type) {
				case SDL_GAMEPAD_TYPE_PS5:
					return "Mute";

				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO:
				//case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT: // Horizontal joycons don't have the Misc1 button
				//case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT: // (see Back, Guide, Start handling for those above)
				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR:
					return "Capture";

				default:
					return "Misc1";
			}

		case JoyButton::PADDLE1:
		case JoyButton::PADDLE2:
		case JoyButton::PADDLE3:
		case JoyButton::PADDLE4:
			switch (gamepad_type) {
				case SDL_GAMEPAD_TYPE_PS5:
					return dualsense_edge_paddles[(int)p_button - (int)JoyButton::PADDLE1];

				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR:
					return joycon_paddles[(int)p_button - (int)JoyButton::PADDLE1];

				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT:
				case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT:
					return joycon_horizontal_paddles[(int)p_button - (int)JoyButton::PADDLE1];

				default:
					return default_paddles[(int)p_button - (int)JoyButton::PADDLE1];
			}

		case JoyButton::TOUCHPAD:
			return "Touchpad";

		default:
			break;
	}
	return "";
}

// Override joypad model for controllers whose schemes can't detected by SDL
JoyModel JoypadSDL::get_scheme_override_model(int p_pad_idx) {
	SDL_Gamepad *gamepad = get_sdl_gamepad(p_pad_idx);
	if (gamepad == nullptr) {
		return JoyModel::UNKNOWN;
	}
	SDL_GamepadType gamepad_type = SDL_GetGamepadType(gamepad);
	// Can a gamepad have SDL_GAMEPAD_TYPE_UNKNOWN type? I'm not sure
	if (gamepad_type != SDL_GAMEPAD_TYPE_STANDARD) {
		return JoyModel::UNKNOWN;
	}

	String joy_name = String(SDL_GetGamepadName(gamepad)).to_lower();

	if (joy_name.contains("xbox")) {
		return JoyModel::XBOX360;
	} else if (joy_name.contains("playstation")) {
		return JoyModel::PS3;
	}

	return JoyModel::UNKNOWN;
}

void JoypadSDL::get_joypad_features(int p_pad_idx, Input::Joypad &p_js) {
	SDL_Joystick *joy = get_sdl_joystick(p_pad_idx);
	// Shouldn't happen, but I'll leave it here just in case
	ERR_FAIL_COND_MSG(joy == nullptr, "JoypadSDL::get_joypad_features: joy == nullptr");

	p_js.device_type = static_cast<JoyDeviceType>(SDL_GetJoystickType(joy));

	int battery_percent;
	SDL_PowerState power_state = SDL_GetJoystickPowerInfo(joy, &battery_percent);
	p_js.battery_percent = battery_percent;
	p_js.power_state = static_cast<JoyPowerState>(power_state);
	p_js.connection_state = static_cast<JoyConnectionState>(SDL_GetJoystickConnectionState(joy));

	SDL_PropertiesID propertiesID = SDL_GetJoystickProperties(joy);
	p_js.has_light = SDL_GetBooleanProperty(propertiesID, SDL_PROP_JOYSTICK_CAP_RGB_LED_BOOLEAN, false) || SDL_GetBooleanProperty(propertiesID, SDL_PROP_JOYSTICK_CAP_MONO_LED_BOOLEAN, false);

	p_js.num_buttons = SDL_GetNumJoystickButtons(joy);
	p_js.num_axes = SDL_GetNumJoystickAxes(joy);

	SDL_Gamepad *gamepad = get_sdl_gamepad(p_pad_idx);
	if (gamepad != nullptr) {
		if (SDL_GamepadHasSensor(gamepad, SDL_SENSOR_ACCEL)) {
			Input::get_singleton()->joy_motion[p_pad_idx].has_accelerometer = true;
		}
		if (SDL_GamepadHasSensor(gamepad, SDL_SENSOR_GYRO)) {
			Input::get_singleton()->joy_motion[p_pad_idx].has_gyroscope = true;
		}

		p_js.model = static_cast<JoyModel>(SDL_GetGamepadType(gamepad));

		// Since SDL_GetNumGamepadButtons, etc. don't exist, here's a more reliable way
		// to get the number of gamepad buttons, etc. since some of them may be remapped
		// to others (like a cheap controller's axes remapped to directional pad).

		int num_buttons = 0;
		int num_axes = 0;
		for (int i = SDL_GAMEPAD_BUTTON_SOUTH; i < SDL_GAMEPAD_BUTTON_COUNT; i++) {
			if (SDL_GamepadHasButton(gamepad, (SDL_GamepadButton)i)) {
				num_buttons++;
			}
		}
		for (int i = SDL_GAMEPAD_AXIS_LEFTX; i < SDL_GAMEPAD_AXIS_COUNT; i++) {
			if (SDL_GamepadHasAxis(gamepad, (SDL_GamepadAxis)i)) {
				num_axes++;
			}
		}
		p_js.num_buttons = num_buttons;
		p_js.num_axes = num_axes;

		if (SDL_GetNumGamepadTouchpads(gamepad) > 0) {
			Input::get_singleton()->joy_touch[p_pad_idx].num_touchpads = SDL_GetNumGamepadTouchpads(gamepad);
		}
	}
}

bool JoypadSDL::send_effect(int p_pad_idx, const void *p_data, int p_size) {
	return SDL_SendJoystickEffect(get_sdl_joystick(p_pad_idx), p_data, p_size);
}

void JoypadSDL::start_triggers_vibration(int p_pad_idx, float p_left_rumble, float p_right_rumble, float p_duration) {
	SDL_RumbleJoystickTriggers(get_sdl_joystick(p_pad_idx), p_left_rumble * 0xFFFF, p_right_rumble * 0xFFFF, p_duration * 1000);
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

SDL_Joystick *JoypadSDL::get_sdl_joystick(int p_pad_idx) const {
	if (p_pad_idx < 0 || p_pad_idx >= Input::JOYPADS_MAX || !joypads[p_pad_idx].attached) {
		return nullptr;
	}

	SDL_JoystickID sdl_instance_idx = joypads[p_pad_idx].sdl_instance_idx;
	return SDL_GetJoystickFromID(sdl_instance_idx);
}

SDL_Gamepad *JoypadSDL::get_sdl_gamepad(int p_pad_idx) const {
	if (p_pad_idx < 0 || p_pad_idx >= Input::JOYPADS_MAX || !joypads[p_pad_idx].attached) {
		return nullptr;
	}

	int sdl_instance_idx = joypads[p_pad_idx].sdl_instance_idx;

	if (!SDL_IsGamepad(sdl_instance_idx)) {
		return nullptr;
	}

	return SDL_GetGamepadFromID(sdl_instance_idx);
}

#endif // SDL_ENABLED
