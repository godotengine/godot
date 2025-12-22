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
typedef struct SDL_Joystick SDL_Joystick;
typedef struct SDL_Gamepad SDL_Gamepad;

class JoypadSDL {
public:
	JoypadSDL();
	~JoypadSDL();

	static JoypadSDL *get_singleton();

	Error initialize();
	void process_events();

private:
	class Joypad : public Input::JoypadFeatures {
	public:
		bool attached = false;
		StringName guid;

		SDL_JoystickID sdl_instance_idx;

		bool supports_force_feedback = false;
		uint64_t ff_effect_timestamp = 0;

		virtual bool has_joy_light() const override;
		virtual bool set_joy_light(const Color &p_color) override;

		SDL_Joystick *get_sdl_joystick() const;
		SDL_Gamepad *get_sdl_gamepad() const;
	};

	static JoypadSDL *singleton;

	Joypad joypads[Input::JOYPADS_MAX];
	HashMap<SDL_JoystickID, int> sdl_instance_id_to_joypad_id;

	void close_joypad(int p_pad_idx);
};
