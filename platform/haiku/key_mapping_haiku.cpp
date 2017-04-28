/*************************************************************************/
/*  key_mapping_haiku.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include <InterfaceDefs.h>

#include "key_mapping_haiku.h"
#include "os/keyboard.h"

struct _HaikuTranslatePair {
	unsigned int keysym;
	int32 keycode;
};

static _HaikuTranslatePair _mod_to_keycode[] = {
	{ KEY_SHIFT, B_SHIFT_KEY },
	{ KEY_ALT, B_COMMAND_KEY },
	{ KEY_CONTROL, B_CONTROL_KEY },
	{ KEY_CAPSLOCK, B_CAPS_LOCK },
	{ KEY_SCROLLLOCK, B_SCROLL_LOCK },
	{ KEY_NUMLOCK, B_NUM_LOCK },
	{ KEY_SUPER_L, B_OPTION_KEY },
	{ KEY_MENU, B_MENU_KEY },
	{ KEY_SHIFT, B_LEFT_SHIFT_KEY },
	{ KEY_SHIFT, B_RIGHT_SHIFT_KEY },
	{ KEY_ALT, B_LEFT_COMMAND_KEY },
	{ KEY_ALT, B_RIGHT_COMMAND_KEY },
	{ KEY_CONTROL, B_LEFT_CONTROL_KEY },
	{ KEY_CONTROL, B_RIGHT_CONTROL_KEY },
	{ KEY_SUPER_L, B_LEFT_OPTION_KEY },
	{ KEY_SUPER_R, B_RIGHT_OPTION_KEY },
	{ KEY_UNKNOWN, 0 }
};

static _HaikuTranslatePair _fn_to_keycode[] = {
	{ KEY_F1, B_F1_KEY },
	{ KEY_F2, B_F2_KEY },
	{ KEY_F3, B_F3_KEY },
	{ KEY_F4, B_F4_KEY },
	{ KEY_F5, B_F5_KEY },
	{ KEY_F6, B_F6_KEY },
	{ KEY_F7, B_F7_KEY },
	{ KEY_F8, B_F8_KEY },
	{ KEY_F9, B_F9_KEY },
	{ KEY_F10, B_F10_KEY },
	{ KEY_F11, B_F11_KEY },
	{ KEY_F12, B_F12_KEY },
	//{ KEY_F13, ? },
	//{ KEY_F14, ? },
	//{ KEY_F15, ? },
	//{ KEY_F16, ? },
	{ KEY_PRINT, B_PRINT_KEY },
	{ KEY_SCROLLLOCK, B_SCROLL_KEY },
	{ KEY_PAUSE, B_PAUSE_KEY },
	{ KEY_UNKNOWN, 0 }
};

static _HaikuTranslatePair _hb_to_keycode[] = {
	{ KEY_BACKSPACE, B_BACKSPACE },
	{ KEY_TAB, B_TAB },
	{ KEY_RETURN, B_RETURN },
	{ KEY_CAPSLOCK, B_CAPS_LOCK },
	{ KEY_ESCAPE, B_ESCAPE },
	{ KEY_SPACE, B_SPACE },
	{ KEY_PAGEUP, B_PAGE_UP },
	{ KEY_PAGEDOWN, B_PAGE_DOWN },
	{ KEY_END, B_END },
	{ KEY_HOME, B_HOME },
	{ KEY_LEFT, B_LEFT_ARROW },
	{ KEY_UP, B_UP_ARROW },
	{ KEY_RIGHT, B_RIGHT_ARROW },
	{ KEY_DOWN, B_DOWN_ARROW },
	{ KEY_PRINT, B_PRINT_KEY },
	{ KEY_INSERT, B_INSERT },
	{ KEY_DELETE, B_DELETE },
	// { KEY_HELP, ??? },

	{ KEY_0, (0x30) },
	{ KEY_1, (0x31) },
	{ KEY_2, (0x32) },
	{ KEY_3, (0x33) },
	{ KEY_4, (0x34) },
	{ KEY_5, (0x35) },
	{ KEY_6, (0x36) },
	{ KEY_7, (0x37) },
	{ KEY_8, (0x38) },
	{ KEY_9, (0x39) },
	{ KEY_A, (0x61) },
	{ KEY_B, (0x62) },
	{ KEY_C, (0x63) },
	{ KEY_D, (0x64) },
	{ KEY_E, (0x65) },
	{ KEY_F, (0x66) },
	{ KEY_G, (0x67) },
	{ KEY_H, (0x68) },
	{ KEY_I, (0x69) },
	{ KEY_J, (0x6A) },
	{ KEY_K, (0x6B) },
	{ KEY_L, (0x6C) },
	{ KEY_M, (0x6D) },
	{ KEY_N, (0x6E) },
	{ KEY_O, (0x6F) },
	{ KEY_P, (0x70) },
	{ KEY_Q, (0x71) },
	{ KEY_R, (0x72) },
	{ KEY_S, (0x73) },
	{ KEY_T, (0x74) },
	{ KEY_U, (0x75) },
	{ KEY_V, (0x76) },
	{ KEY_W, (0x77) },
	{ KEY_X, (0x78) },
	{ KEY_Y, (0x79) },
	{ KEY_Z, (0x7A) },

	/*
{ KEY_PLAY, VK_PLAY},// (0xFA)
{ KEY_STANDBY,VK_SLEEP },//(0x5F)
{ KEY_BACK,VK_BROWSER_BACK},// (0xA6)
{ KEY_FORWARD,VK_BROWSER_FORWARD},// (0xA7)
{ KEY_REFRESH,VK_BROWSER_REFRESH},// (0xA8)
{ KEY_STOP,VK_BROWSER_STOP},// (0xA9)
{ KEY_SEARCH,VK_BROWSER_SEARCH},// (0xAA)
{ KEY_FAVORITES, VK_BROWSER_FAVORITES},// (0xAB)
{ KEY_HOMEPAGE,VK_BROWSER_HOME},// (0xAC)
{ KEY_VOLUMEMUTE,VK_VOLUME_MUTE},// (0xAD)
{ KEY_VOLUMEDOWN,VK_VOLUME_DOWN},// (0xAE)
{ KEY_VOLUMEUP,VK_VOLUME_UP},// (0xAF)
{ KEY_MEDIANEXT,VK_MEDIA_NEXT_TRACK},// (0xB0)
{ KEY_MEDIAPREVIOUS,VK_MEDIA_PREV_TRACK},// (0xB1)
{ KEY_MEDIASTOP,VK_MEDIA_STOP},// (0xB2)
{ KEY_LAUNCHMAIL, VK_LAUNCH_MAIL},// (0xB4)
{ KEY_LAUNCHMEDIA,VK_LAUNCH_MEDIA_SELECT},// (0xB5)
{ KEY_LAUNCH0,VK_LAUNCH_APP1},// (0xB6)
{ KEY_LAUNCH1,VK_LAUNCH_APP2},// (0xB7)
*/

	{ KEY_SEMICOLON, 0x3B },
	{ KEY_EQUAL, 0x3D },
	{ KEY_COLON, 0x2C },
	{ KEY_MINUS, 0x2D },
	{ KEY_PERIOD, 0x2E },
	{ KEY_SLASH, 0x2F },
	{ KEY_KP_MULTIPLY, 0x2A },
	{ KEY_KP_ADD, 0x2B },

	{ KEY_QUOTELEFT, 0x60 },
	{ KEY_BRACKETLEFT, 0x5B },
	{ KEY_BACKSLASH, 0x5C },
	{ KEY_BRACKETRIGHT, 0x5D },
	{ KEY_APOSTROPHE, 0x27 },

	{ KEY_UNKNOWN, 0 }
};

unsigned int KeyMappingHaiku::get_keysym(int32 raw_char, int32 key) {
	if (raw_char == B_INSERT && key == 0x64) {
		return KEY_KP_0;
	}
	if (raw_char == B_END && key == 0x58) {
		return KEY_KP_1;
	}
	if (raw_char == B_DOWN_ARROW && key == 0x59) {
		return KEY_KP_2;
	}
	if (raw_char == B_PAGE_DOWN && key == 0x5A) {
		return KEY_KP_3;
	}
	if (raw_char == B_LEFT_ARROW && key == 0x48) {
		return KEY_KP_4;
	}
	if (raw_char == 0x35 && key == 0x49) {
		return KEY_KP_5;
	}
	if (raw_char == B_RIGHT_ARROW && key == 0x4A) {
		return KEY_KP_6;
	}
	if (raw_char == B_HOME && key == 0x37) {
		return KEY_KP_7;
	}
	if (raw_char == B_UP_ARROW && key == 0x38) {
		return KEY_KP_8;
	}
	if (raw_char == B_PAGE_UP && key == 0x39) {
		return KEY_KP_9;
	}
	if (raw_char == 0x2F && key == 0x23) {
		return KEY_KP_DIVIDE;
	}
	if (raw_char == 0x2D && key == 0x25) {
		return KEY_KP_SUBTRACT;
	}
	if (raw_char == B_DELETE && key == 0x65) {
		return KEY_KP_PERIOD;
	}

	if (raw_char == 0x10) {
		for (int i = 0; _fn_to_keycode[i].keysym != KEY_UNKNOWN; i++) {
			if (_fn_to_keycode[i].keycode == key) {
				return _fn_to_keycode[i].keysym;
			}
		}

		return KEY_UNKNOWN;
	}

	for (int i = 0; _hb_to_keycode[i].keysym != KEY_UNKNOWN; i++) {
		if (_hb_to_keycode[i].keycode == raw_char) {
			return _hb_to_keycode[i].keysym;
		}
	}

	return KEY_UNKNOWN;
}

unsigned int KeyMappingHaiku::get_modifier_keysym(int32 key) {
	for (int i = 0; _mod_to_keycode[i].keysym != KEY_UNKNOWN; i++) {
		if ((_mod_to_keycode[i].keycode & key) != 0) {
			return _mod_to_keycode[i].keysym;
		}
	}

	return KEY_UNKNOWN;
}
