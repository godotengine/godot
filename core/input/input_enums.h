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
	MAX = 36, // Android supports up to 36 buttons.
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
	MASK_LEFT = (1 << (LEFT - 1)),
	MASK_RIGHT = (1 << (RIGHT - 1)),
	MASK_MIDDLE = (1 << (MIDDLE - 1)),
	MASK_XBUTTON1 = (1 << (MB_XBUTTON1 - 1)),
	MASK_XBUTTON2 = (1 << (MB_XBUTTON2 - 1)),
};

inline MouseButton mouse_button_to_mask(MouseButton button) {
	return MouseButton(1 << ((int)button - 1));
}

inline MouseButton operator&(MouseButton a, MouseButton b) {
	return (MouseButton)((int)a & (int)b);
}

inline MouseButton operator|(MouseButton a, MouseButton b) {
	return (MouseButton)((int)a | (int)b);
}

inline MouseButton operator^(MouseButton a, MouseButton b) {
	return (MouseButton)((int)a ^ (int)b);
}

inline MouseButton &operator|=(MouseButton &a, MouseButton b) {
	return (MouseButton &)((int &)a |= (int)b);
}

inline MouseButton &operator&=(MouseButton &a, MouseButton b) {
	return (MouseButton &)((int &)a &= (int)b);
}

inline MouseButton operator~(MouseButton a) {
	return (MouseButton)(~(int)a);
}

inline HatMask operator|(HatMask a, HatMask b) {
	return (HatMask)((int)a | (int)b);
}

inline HatMask operator&(HatMask a, HatMask b) {
	return (HatMask)((int)a & (int)b);
}

inline HatMask &operator&=(HatMask &a, HatMask b) {
	return (HatMask &)((int &)a &= (int)b);
}

inline HatMask &operator|=(HatMask &a, HatMask b) {
	return (HatMask &)((int &)a |= (int)b);
}

inline HatMask operator~(HatMask a) {
	return (HatMask)(~(int)a);
}

#endif // INPUT_ENUMS_H
