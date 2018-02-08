/*************************************************************************/
/*  key_mapping_sdl.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "key_mapping_sdl.h"

/***** SCAN CODE CONVERSION ******/

struct KeyCodePair {

	SDL_Keycode sdl_keycode;
	unsigned int keycode;
};

static KeyCodePair sdl_keycode_to_keycode[] = {
	// misc keys

	{ SDLK_ESCAPE, KEY_ESCAPE },
	{ SDLK_TAB, KEY_TAB },
	{ SDLK_BACKSPACE, KEY_BACKSPACE },
	{ SDLK_RETURN, KEY_ENTER },
	{ SDLK_RETURN2, KEY_ENTER },
	{ SDLK_INSERT, KEY_INSERT },
	{ SDLK_DELETE, KEY_DELETE },
	{ SDLK_CLEAR, KEY_DELETE },
	{ SDLK_CLEARAGAIN, KEY_DELETE },
	{ SDLK_PAUSE, KEY_PAUSE },
	{ SDLK_PRINTSCREEN, KEY_PRINT },
	{ SDLK_HOME, KEY_HOME },
	{ SDLK_END, KEY_END },
	{ SDLK_LEFT, KEY_LEFT },
	{ SDLK_UP, KEY_UP },
	{ SDLK_RIGHT, KEY_RIGHT },
	{ SDLK_DOWN, KEY_DOWN },
	{ SDLK_PAGEUP, KEY_PAGEUP },
	{ SDLK_PAGEDOWN, KEY_PAGEDOWN },
	{ SDLK_LSHIFT, KEY_SHIFT },
	{ SDLK_RSHIFT, KEY_SHIFT },
	{ SDLK_LCTRL, KEY_CONTROL },
	{ SDLK_RCTRL, KEY_CONTROL },
	{ SDLK_LALT, KEY_ALT },
	{ SDLK_RALT, KEY_ALT },
	{ SDLK_LGUI, KEY_SUPER_L },
	{ SDLK_RGUI, KEY_SUPER_R },
	{ SDLK_CAPSLOCK, KEY_CAPSLOCK },
	{ SDLK_NUMLOCKCLEAR, KEY_NUMLOCK },
	{ SDLK_SCROLLLOCK, KEY_SCROLLLOCK },
	{ SDLK_MENU, KEY_MENU },
	{ SDLK_HELP, KEY_HELP },
	{ SDLK_KP_SPACE, KEY_SPACE },
	{ SDLK_TAB, KEY_TAB },
	{ SDLK_KP_ENTER, KEY_KP_ENTER },
	//{ XK_KP_Equal,                KEY_EQUAL   },
	//{ XK_KP_Separator,            KEY_COMMA   },
	{ SDLK_KP_PERIOD, KEY_KP_PERIOD },
	{ SDLK_KP_CLEAR, KEY_KP_PERIOD },
	{ SDLK_KP_MULTIPLY, KEY_KP_MULTIPLY },
	{ SDLK_KP_DIVIDE, KEY_KP_DIVIDE },
	{ SDLK_KP_MINUS, KEY_KP_SUBTRACT },
	{ SDLK_KP_PLUS, KEY_KP_ADD },
	{ SDLK_KP_0, KEY_KP_0 },
	{ SDLK_KP_1, KEY_KP_1 },
	{ SDLK_KP_2, KEY_KP_2 },
	{ SDLK_KP_3, KEY_KP_3 },
	{ SDLK_KP_4, KEY_KP_4 },
	{ SDLK_KP_5, KEY_KP_5 },
	{ SDLK_KP_6, KEY_KP_6 },
	{ SDLK_KP_7, KEY_KP_7 },
	{ SDLK_KP_8, KEY_KP_8 },
	{ SDLK_KP_9, KEY_KP_9 },
	// same but with numlock
	{ SDLK_F1, KEY_F1 },
	{ SDLK_F2, KEY_F2 },
	{ SDLK_F3, KEY_F3 },
	{ SDLK_F4, KEY_F4 },
	{ SDLK_F5, KEY_F5 },
	{ SDLK_F6, KEY_F6 },
	{ SDLK_F7, KEY_F7 },
	{ SDLK_F8, KEY_F8 },
	{ SDLK_F9, KEY_F9 },
	{ SDLK_F10, KEY_F10 },
	{ SDLK_F11, KEY_F11 },
	{ SDLK_F12, KEY_F12 },
	{ SDLK_F13, KEY_F13 },
	{ SDLK_F14, KEY_F14 },
	{ SDLK_F15, KEY_F15 },
	{ SDLK_F16, KEY_F16 },

	// media keys
	{ SDLK_AC_BACK, KEY_BACK },
	{ SDLK_AC_FORWARD, KEY_FORWARD },
	{ SDLK_AC_STOP, KEY_STOP },
	{ SDLK_AC_REFRESH, KEY_REFRESH },
	{ SDLK_AC_BOOKMARKS, KEY_FAVORITES },
	{ SDLK_WWW, KEY_OPENURL },
	{ SDLK_AC_HOME, KEY_HOMEPAGE },
	{ SDLK_AC_SEARCH, KEY_SEARCH },
	{ SDLK_VOLUMEDOWN, KEY_VOLUMEDOWN },
	{ SDLK_VOLUMEUP, KEY_VOLUMEUP },
	{ SDLK_AUDIOMUTE, KEY_VOLUMEMUTE },
	{ SDLK_AUDIOPLAY, KEY_MEDIAPLAY },
	{ SDLK_AUDIOSTOP, KEY_MEDIASTOP },
	{ SDLK_AUDIOPREV, KEY_MEDIAPREVIOUS },
	{ SDLK_AUDIONEXT, KEY_MEDIANEXT },

	// launch keys
	{ SDLK_MAIL, KEY_LAUNCHMAIL },
	{ SDLK_COMPUTER, KEY_LAUNCH0 },
	{ SDLK_CALCULATOR, KEY_LAUNCH1 },
	{ SDLK_SLEEP, KEY_STANDBY },
	{ 0, 0 }
};

unsigned int KeyMappingSDL::get_non_printable_keycode(SDL_Keycode p_keycode) {
	// kinda bruteforce.. could optimize.

	// look for special key
	for (int idx = 0; sdl_keycode_to_keycode[idx].sdl_keycode != 0; idx++) {

		if (sdl_keycode_to_keycode[idx].sdl_keycode == p_keycode)
			return sdl_keycode_to_keycode[idx].keycode;
	}

	return 0;
}
