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

#include "core/input/input.h"
#include "core/os/thread.h"

typedef uint32_t SDL_JoystickID;
typedef struct HWND__ *HWND;
typedef struct SDL_Joystick SDL_Joystick;
typedef struct SDL_Gamepad SDL_Gamepad;

class JoypadSDL {
public:
	JoypadSDL();
	~JoypadSDL();

	static JoypadSDL *get_singleton();

	Error initialize();
	void process_events();
#ifdef WINDOWS_ENABLED
	void setup_sdl_helper_window(HWND p_hwnd);
#endif

	bool enable_accelerometer(int p_pad_idx, bool p_enable);
	bool enable_gyroscope(int p_pad_idx, bool p_enable);

	bool set_light(int p_pad_idx, Color p_color);

	bool has_joy_axis(int p_pad_idx, JoyAxis p_axis) const;
	bool has_joy_button(int p_pad_idx, JoyButton p_button) const;
	static String get_model_axis_string(JoyModel p_model, JoyAxis p_axis);
	static String get_model_button_string(JoyModel p_model, JoyButton p_button);
	JoyModel get_scheme_override_model(int p_pad_idx);

	void get_joypad_features(int p_pad_idx, Input::Joypad &p_js);

	bool send_effect(int p_pad_idx, const void *p_data, int p_size);
	void start_triggers_vibration(int p_pad_idx, float p_left_rumble, float p_right_rumble, float p_duration);

private:
	struct Joypad {
		bool attached = false;
		StringName guid;

		SDL_JoystickID sdl_instance_idx;

		bool supports_force_feedback = false;
		uint64_t ff_effect_timestamp = 0;
	};

	static JoypadSDL *singleton;

	Joypad joypads[Input::JOYPADS_MAX];
	HashMap<SDL_JoystickID, int> sdl_instance_id_to_joypad_id;

	void close_joypad(int p_pad_idx);
	SDL_Joystick *get_sdl_joystick(int p_pad_idx) const;
	SDL_Gamepad *get_sdl_gamepad(int p_pad_idx) const;
};
