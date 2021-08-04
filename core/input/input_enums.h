/*************************************************************************/
/*  input_enums.h                                                        */
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

#ifndef INPUT_ENUMS_H
#define INPUT_ENUMS_H

enum HatDir {
	HAT_UP = 0,
	HAT_RIGHT = 1,
	HAT_DOWN = 2,
	HAT_LEFT = 3,
	HAT_MAX = 4,
};

enum HatMask {
	HAT_MASK_CENTER = 0,
	HAT_MASK_UP = 1,
	HAT_MASK_RIGHT = 2,
	HAT_MASK_DOWN = 4,
	HAT_MASK_LEFT = 8,
};

enum JoyAxis {
	JOY_AXIS_INVALID = -1,
	JOY_AXIS_LEFT_X = 0,
	JOY_AXIS_LEFT_Y = 1,
	JOY_AXIS_RIGHT_X = 2,
	JOY_AXIS_RIGHT_Y = 3,
	JOY_AXIS_TRIGGER_LEFT = 4,
	JOY_AXIS_TRIGGER_RIGHT = 5,
	JOY_AXIS_SDL_MAX = 6,
	JOY_AXIS_MAX = 10, // OpenVR supports up to 5 Joysticks making a total of 10 axes.
};

enum JoyButton {
	JOY_BUTTON_INVALID = -1,
	JOY_BUTTON_A = 0,
	JOY_BUTTON_B = 1,
	JOY_BUTTON_X = 2,
	JOY_BUTTON_Y = 3,
	JOY_BUTTON_BACK = 4,
	JOY_BUTTON_GUIDE = 5,
	JOY_BUTTON_START = 6,
	JOY_BUTTON_LEFT_STICK = 7,
	JOY_BUTTON_RIGHT_STICK = 8,
	JOY_BUTTON_LEFT_SHOULDER = 9,
	JOY_BUTTON_RIGHT_SHOULDER = 10,
	JOY_BUTTON_DPAD_UP = 11,
	JOY_BUTTON_DPAD_DOWN = 12,
	JOY_BUTTON_DPAD_LEFT = 13,
	JOY_BUTTON_DPAD_RIGHT = 14,
	JOY_BUTTON_MISC1 = 15,
	JOY_BUTTON_PADDLE1 = 16,
	JOY_BUTTON_PADDLE2 = 17,
	JOY_BUTTON_PADDLE3 = 18,
	JOY_BUTTON_PADDLE4 = 19,
	JOY_BUTTON_TOUCHPAD = 20,
	JOY_BUTTON_SDL_MAX = 21,
	JOY_BUTTON_MAX = 36, // Android supports up to 36 buttons.
};

enum MIDIMessage {
	MIDI_MESSAGE_NONE = 0,
	MIDI_MESSAGE_NOTE_OFF = 0x8,
	MIDI_MESSAGE_NOTE_ON = 0x9,
	MIDI_MESSAGE_AFTERTOUCH = 0xA,
	MIDI_MESSAGE_CONTROL_CHANGE = 0xB,
	MIDI_MESSAGE_PROGRAM_CHANGE = 0xC,
	MIDI_MESSAGE_CHANNEL_PRESSURE = 0xD,
	MIDI_MESSAGE_PITCH_BEND = 0xE,
};

enum MouseButton {
	MOUSE_BUTTON_NONE = 0,
	MOUSE_BUTTON_LEFT = 1,
	MOUSE_BUTTON_RIGHT = 2,
	MOUSE_BUTTON_MIDDLE = 3,
	MOUSE_BUTTON_WHEEL_UP = 4,
	MOUSE_BUTTON_WHEEL_DOWN = 5,
	MOUSE_BUTTON_WHEEL_LEFT = 6,
	MOUSE_BUTTON_WHEEL_RIGHT = 7,
	MOUSE_BUTTON_XBUTTON1 = 8,
	MOUSE_BUTTON_XBUTTON2 = 9,
	MOUSE_BUTTON_MASK_LEFT = (1 << (MOUSE_BUTTON_LEFT - 1)),
	MOUSE_BUTTON_MASK_RIGHT = (1 << (MOUSE_BUTTON_RIGHT - 1)),
	MOUSE_BUTTON_MASK_MIDDLE = (1 << (MOUSE_BUTTON_MIDDLE - 1)),
	MOUSE_BUTTON_MASK_XBUTTON1 = (1 << (MOUSE_BUTTON_XBUTTON1 - 1)),
	MOUSE_BUTTON_MASK_XBUTTON2 = (1 << (MOUSE_BUTTON_XBUTTON2 - 1)),
};

inline MouseButton &operator|=(MouseButton &a, MouseButton b) {
	return (MouseButton &)((int &)a |= (int)b);
}

inline MouseButton &operator&=(MouseButton &a, MouseButton b) {
	return (MouseButton &)((int &)a &= (int)b);
}

#endif // INPUT_ENUMS_H
