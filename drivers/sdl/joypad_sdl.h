/**************************************************************************/
/*  joypad_sdl.h                                                          */
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

#include "main/input_default.h"

enum class SDLJoyAxis {
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

enum class SDLJoyButton {
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

typedef uint32_t SDL_JoystickID;
typedef struct HWND__ *HWND;

class JoypadSDL {
public:
	JoypadSDL(InputDefault *in);
#ifdef WINDOWS_ENABLED
	JoypadSDL(InputDefault *in, HWND p_helper_window);
#endif
	~JoypadSDL();

	static JoypadSDL *get_singleton();

	Error initialize();
	void process_events();

private:
	struct Joypad {
		bool attached = false;
		StringName guid;

		SDL_JoystickID sdl_instance_idx;

		bool supports_force_feedback = false;
		uint64_t ff_effect_timestamp = 0;
	};

	InputDefault *input;
	static JoypadSDL *singleton;

	Joypad joypads[InputDefault::JOYPADS_MAX];
	HashMap<SDL_JoystickID, int> sdl_instance_id_to_joypad_id;

	void close_joypad(int p_pad_idx);

	JoystickList map_sdl_button_to_joystick_list(int p_button);
	JoystickList map_sdl_axis_to_joystick_list(int p_axis);
};
