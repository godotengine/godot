/**************************************************************************/
/*  input_enums.h                                                         */
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

#pragma once

#include "core/error/error_macros.h"

enum class InputEventType {
	INVALID = -1,
	KEY,
	MOUSE_BUTTON,
	MOUSE_MOTION,
	JOY_MOTION,
	JOY_BUTTON,
	SCREEN_TOUCH,
	SCREEN_DRAG,
	MAGNIFY_GESTURE,
	PAN_GESTURE,
	MIDI,
	SHORTCUT,
	ACTION,
	MAX,
};

enum class HatDir {
	UP = 0,
	RIGHT = 1,
	DOWN = 2,
	LEFT = 3,
	MAX = 4,
};

enum class HatMask {
	CENTER = 0,
	UP = 1,
	RIGHT = 2,
	DOWN = 4,
	LEFT = 8,
};

enum class JoyAxis {
	INVALID = -1,
	LEFT_X = 0,
	LEFT_Y = 1,
	RIGHT_X = 2,
	RIGHT_Y = 3,
	TRIGGER_LEFT = 4,
	TRIGGER_RIGHT = 5,
	SDL_MAX = 6,
	MAX = 10, // OpenVR supports up to 5 Joysticks making a total of 10 axes.
};

enum class JoyButton {
	INVALID = -1,
	A = 0,
	B = 1,
	X = 2,
	Y = 3,
	BACK = 4,
	GUIDE = 5,
	START = 6,
	LEFT_STICK = 7,
	RIGHT_STICK = 8,
	LEFT_SHOULDER = 9,
	RIGHT_SHOULDER = 10,
	DPAD_UP = 11,
	DPAD_DOWN = 12,
	DPAD_LEFT = 13,
	DPAD_RIGHT = 14,
	MISC1 = 15,
	PADDLE1 = 16,
	PADDLE2 = 17,
	PADDLE3 = 18,
	PADDLE4 = 19,
	TOUCHPAD = 20,
	SDL_MAX = 21,
	MAX = 128, // Android supports up to 36 buttons. DirectInput supports up to 128 buttons.
};

// See also SDL_GamepadType. JoyModel enum is not the same as SDL_GamepadType,
// but if new entries in SDL_GamepadType appear in the future, they should
// be added here as well.
enum class JoyModel {
	UNKNOWN = 0,
	XBOX_GENERIC = 1,
	PLAYSTATION_GENERIC = 2,
	NINTENDO_GENERIC = 3,
	XBOX360 = 4,
	XBOXONE = 5,
	STEAM_DECK = 6,
	PS3 = 7,
	PS4 = 8,
	PS5 = 9,
	SWITCH_PRO = 10,
	JOYCON_LEFT = 11,
	JOYCON_RIGHT = 12,
	JOYCON_PAIR = 13,
};

enum class JoyScheme {
	UNKNOWN = 0,
	XBOX = 1,
	PLAYSTATION = 2,
	NINTENDO = 3,
	JOYCON_HORIZONTAL = 4,
};

// See SDL_JoystickType (which is not the same as SDL_GamepadType).
enum class JoyDeviceType {
	UNKNOWN = 0,
	GAMEPAD = 1,
	WHEEL = 2,
	ARCADE_STICK = 3,
	FLIGHT_STICK = 4,
	DANCE_PAD = 5,
	GUITAR = 6,
	DRUM_KIT = 7,
	ARCADE_PAD = 8,
	THROTTLE = 9,
};

// See SDL_PowerState. SDL_POWERSTATE_ERROR equivalent was omitted for simplicity.
enum class JoyPowerState {
	UNKNOWN = 0,
	ON_BATTERY = 1,
	NO_BATTERY = 2,
	CHARGING = 3,
	FULL_BATTERY = 4,
};

// See SDL_JoystickConnectionState. SDL_JOYSTICK_CONNECTION_INVALID equivalent was omitted for simplicity.
enum class JoyConnectionState {
	UNKNOWN = 0,
	WIRED = 1,
	WIRELESS = 2,
};

enum class MIDIMessage {
	NONE = 0,
	NOTE_OFF = 0x8,
	NOTE_ON = 0x9,
	AFTERTOUCH = 0xA,
	CONTROL_CHANGE = 0xB,
	PROGRAM_CHANGE = 0xC,
	CHANNEL_PRESSURE = 0xD,
	PITCH_BEND = 0xE,
	SYSTEM_EXCLUSIVE = 0xF0,
	QUARTER_FRAME = 0xF1,
	SONG_POSITION_POINTER = 0xF2,
	SONG_SELECT = 0xF3,
	TUNE_REQUEST = 0xF6,
	TIMING_CLOCK = 0xF8,
	START = 0xFA,
	CONTINUE = 0xFB,
	STOP = 0xFC,
	ACTIVE_SENSING = 0xFE,
	SYSTEM_RESET = 0xFF,
};

enum class MouseButton {
	NONE = 0,
	LEFT = 1,
	RIGHT = 2,
	MIDDLE = 3,
	WHEEL_UP = 4,
	WHEEL_DOWN = 5,
	WHEEL_LEFT = 6,
	WHEEL_RIGHT = 7,
	MB_XBUTTON1 = 8, // "XBUTTON1" is a reserved word on Windows.
	MB_XBUTTON2 = 9, // "XBUTTON2" is a reserved word on Windows.
};

enum class MouseButtonMask {
	NONE = 0,
	LEFT = (1 << (int(MouseButton::LEFT) - 1)),
	RIGHT = (1 << (int(MouseButton::RIGHT) - 1)),
	MIDDLE = (1 << (int(MouseButton::MIDDLE) - 1)),
	MB_XBUTTON1 = (1 << (int(MouseButton::MB_XBUTTON1) - 1)),
	MB_XBUTTON2 = (1 << (int(MouseButton::MB_XBUTTON2) - 1)),
};

inline MouseButtonMask mouse_button_to_mask(MouseButton button) {
	ERR_FAIL_COND_V(button == MouseButton::NONE, MouseButtonMask::NONE);

	return MouseButtonMask(1 << ((int)button - 1));
}

constexpr MouseButtonMask operator|(MouseButtonMask p_a, MouseButtonMask p_b) {
	return static_cast<MouseButtonMask>(static_cast<int>(p_a) | static_cast<int>(p_b));
}

constexpr MouseButtonMask &operator|=(MouseButtonMask &p_a, MouseButtonMask p_b) {
	return p_a = p_a | p_b;
}
