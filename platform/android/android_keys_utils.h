/*************************************************************************/
/*  android_keys_utils.h                                                 */
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

#ifndef ANDROID_KEYS_UTILS_H
#define ANDROID_KEYS_UTILS_H

#include <android/input.h>
#include <core/os/keyboard.h>

struct _WinTranslatePair {
	Key keysym = Key::NONE;
	unsigned int keycode = 0;
};

static _WinTranslatePair _ak_to_keycode[] = {
	{ Key::TAB, AKEYCODE_TAB },
	{ Key::ENTER, AKEYCODE_ENTER },
	{ Key::SHIFT, AKEYCODE_SHIFT_LEFT },
	{ Key::SHIFT, AKEYCODE_SHIFT_RIGHT },
	{ Key::ALT, AKEYCODE_ALT_LEFT },
	{ Key::ALT, AKEYCODE_ALT_RIGHT },
	{ Key::MENU, AKEYCODE_MENU },
	{ Key::PAUSE, AKEYCODE_MEDIA_PLAY_PAUSE },
	{ Key::ESCAPE, AKEYCODE_BACK },
	{ Key::SPACE, AKEYCODE_SPACE },
	{ Key::PAGEUP, AKEYCODE_PAGE_UP },
	{ Key::PAGEDOWN, AKEYCODE_PAGE_DOWN },
	{ Key::HOME, AKEYCODE_HOME }, //(0x24)
	{ Key::LEFT, AKEYCODE_DPAD_LEFT },
	{ Key::UP, AKEYCODE_DPAD_UP },
	{ Key::RIGHT, AKEYCODE_DPAD_RIGHT },
	{ Key::DOWN, AKEYCODE_DPAD_DOWN },
	{ Key::PERIODCENTERED, AKEYCODE_DPAD_CENTER },
	{ Key::BACKSPACE, AKEYCODE_DEL },
	{ Key::KEY_0, AKEYCODE_0 },
	{ Key::KEY_1, AKEYCODE_1 },
	{ Key::KEY_2, AKEYCODE_2 },
	{ Key::KEY_3, AKEYCODE_3 },
	{ Key::KEY_4, AKEYCODE_4 },
	{ Key::KEY_5, AKEYCODE_5 },
	{ Key::KEY_6, AKEYCODE_6 },
	{ Key::KEY_7, AKEYCODE_7 },
	{ Key::KEY_8, AKEYCODE_8 },
	{ Key::KEY_9, AKEYCODE_9 },
	{ Key::A, AKEYCODE_A },
	{ Key::B, AKEYCODE_B },
	{ Key::C, AKEYCODE_C },
	{ Key::D, AKEYCODE_D },
	{ Key::E, AKEYCODE_E },
	{ Key::F, AKEYCODE_F },
	{ Key::G, AKEYCODE_G },
	{ Key::H, AKEYCODE_H },
	{ Key::I, AKEYCODE_I },
	{ Key::J, AKEYCODE_J },
	{ Key::K, AKEYCODE_K },
	{ Key::L, AKEYCODE_L },
	{ Key::M, AKEYCODE_M },
	{ Key::N, AKEYCODE_N },
	{ Key::O, AKEYCODE_O },
	{ Key::P, AKEYCODE_P },
	{ Key::Q, AKEYCODE_Q },
	{ Key::R, AKEYCODE_R },
	{ Key::S, AKEYCODE_S },
	{ Key::T, AKEYCODE_T },
	{ Key::U, AKEYCODE_U },
	{ Key::V, AKEYCODE_V },
	{ Key::W, AKEYCODE_W },
	{ Key::X, AKEYCODE_X },
	{ Key::Y, AKEYCODE_Y },
	{ Key::Z, AKEYCODE_Z },
	{ Key::HOMEPAGE, AKEYCODE_EXPLORER },
	{ Key::LAUNCH0, AKEYCODE_BUTTON_A },
	{ Key::LAUNCH1, AKEYCODE_BUTTON_B },
	{ Key::LAUNCH2, AKEYCODE_BUTTON_C },
	{ Key::LAUNCH3, AKEYCODE_BUTTON_X },
	{ Key::LAUNCH4, AKEYCODE_BUTTON_Y },
	{ Key::LAUNCH5, AKEYCODE_BUTTON_Z },
	{ Key::LAUNCH6, AKEYCODE_BUTTON_L1 },
	{ Key::LAUNCH7, AKEYCODE_BUTTON_R1 },
	{ Key::LAUNCH8, AKEYCODE_BUTTON_L2 },
	{ Key::LAUNCH9, AKEYCODE_BUTTON_R2 },
	{ Key::LAUNCHA, AKEYCODE_BUTTON_THUMBL },
	{ Key::LAUNCHB, AKEYCODE_BUTTON_THUMBR },
	{ Key::LAUNCHC, AKEYCODE_BUTTON_START },
	{ Key::LAUNCHD, AKEYCODE_BUTTON_SELECT },
	{ Key::LAUNCHE, AKEYCODE_BUTTON_MODE },
	{ Key::VOLUMEMUTE, AKEYCODE_MUTE },
	{ Key::VOLUMEDOWN, AKEYCODE_VOLUME_DOWN },
	{ Key::VOLUMEUP, AKEYCODE_VOLUME_UP },
	{ Key::BACK, AKEYCODE_MEDIA_REWIND },
	{ Key::FORWARD, AKEYCODE_MEDIA_FAST_FORWARD },
	{ Key::MEDIANEXT, AKEYCODE_MEDIA_NEXT },
	{ Key::MEDIAPREVIOUS, AKEYCODE_MEDIA_PREVIOUS },
	{ Key::MEDIASTOP, AKEYCODE_MEDIA_STOP },
	{ Key::PLUS, AKEYCODE_PLUS },
	{ Key::EQUAL, AKEYCODE_EQUALS }, // the '+' key
	{ Key::COMMA, AKEYCODE_COMMA }, // the ',' key
	{ Key::MINUS, AKEYCODE_MINUS }, // the '-' key
	{ Key::SLASH, AKEYCODE_SLASH }, // the '/?' key
	{ Key::BACKSLASH, AKEYCODE_BACKSLASH },
	{ Key::BRACKETLEFT, AKEYCODE_LEFT_BRACKET },
	{ Key::BRACKETRIGHT, AKEYCODE_RIGHT_BRACKET },
	{ Key::CTRL, AKEYCODE_CTRL_LEFT },
	{ Key::CTRL, AKEYCODE_CTRL_RIGHT },
	{ Key::UNKNOWN, 0 }
};
/*
TODO: map these android key:
	AKEYCODE_SOFT_LEFT       = 1,
	AKEYCODE_SOFT_RIGHT      = 2,
	AKEYCODE_CALL            = 5,
	AKEYCODE_ENDCALL         = 6,
	AKEYCODE_STAR            = 17,
	AKEYCODE_POUND           = 18,
	AKEYCODE_POWER           = 26,
	AKEYCODE_CAMERA          = 27,
	AKEYCODE_CLEAR           = 28,
	AKEYCODE_SYM             = 63,
	AKEYCODE_ENVELOPE        = 65,
	AKEYCODE_GRAVE           = 68,
	AKEYCODE_SEMICOLON       = 74,
	AKEYCODE_APOSTROPHE      = 75,
	AKEYCODE_AT              = 77,
	AKEYCODE_NUM             = 78,
	AKEYCODE_HEADSETHOOK     = 79,
	AKEYCODE_FOCUS           = 80,   // *Camera* focus
	AKEYCODE_NOTIFICATION    = 83,
	AKEYCODE_SEARCH          = 84,
	AKEYCODE_PICTSYMBOLS     = 94,
	AKEYCODE_SWITCH_CHARSET  = 95,
*/

Key android_get_keysym(unsigned int p_code);

#endif // ANDROID_KEYS_UTILS_H
