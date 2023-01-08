/**************************************************************************/
/*  key_mapping_xkb.cpp                                                   */
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

#include "key_mapping_xkb.h"

/***** SCAN CODE CONVERSION ******/

struct _XKBTranslatePair {
	xkb_keysym_t keysym;
	Key keycode;
};

static _XKBTranslatePair _xkb_keysym_to_keycode[] = {
	// misc keys

	{ XKB_KEY_Escape, Key::ESCAPE },
	{ XKB_KEY_Tab, Key::TAB },
	{ XKB_KEY_ISO_Left_Tab, Key::BACKTAB },
	{ XKB_KEY_BackSpace, Key::BACKSPACE },
	{ XKB_KEY_Return, Key::ENTER },
	{ XKB_KEY_Insert, Key::INSERT },
	{ XKB_KEY_Delete, Key::KEY_DELETE },
	{ XKB_KEY_Clear, Key::KEY_DELETE },
	{ XKB_KEY_Pause, Key::PAUSE },
	{ XKB_KEY_Print, Key::PRINT },
	{ XKB_KEY_Home, Key::HOME },
	{ XKB_KEY_End, Key::END },
	{ XKB_KEY_Left, Key::LEFT },
	{ XKB_KEY_Up, Key::UP },
	{ XKB_KEY_Right, Key::RIGHT },
	{ XKB_KEY_Down, Key::DOWN },
	{ XKB_KEY_Prior, Key::PAGEUP },
	{ XKB_KEY_Next, Key::PAGEDOWN },
	{ XKB_KEY_Shift_L, Key::SHIFT },
	{ XKB_KEY_Shift_R, Key::SHIFT },
	{ XKB_KEY_Shift_Lock, Key::SHIFT },
	{ XKB_KEY_Control_L, Key::CTRL },
	{ XKB_KEY_Control_R, Key::CTRL },
	{ XKB_KEY_Meta_L, Key::META },
	{ XKB_KEY_Meta_R, Key::META },
	{ XKB_KEY_Alt_L, Key::ALT },
	{ XKB_KEY_Alt_R, Key::ALT },
	{ XKB_KEY_Caps_Lock, Key::CAPSLOCK },
	{ XKB_KEY_Num_Lock, Key::NUMLOCK },
	{ XKB_KEY_Scroll_Lock, Key::SCROLLLOCK },
	{ XKB_KEY_Super_L, Key::SUPER_L },
	{ XKB_KEY_Super_R, Key::SUPER_R },
	{ XKB_KEY_Menu, Key::MENU },
	{ XKB_KEY_Hyper_L, Key::HYPER_L },
	{ XKB_KEY_Hyper_R, Key::HYPER_R },
	{ XKB_KEY_Help, Key::HELP },
	{ XKB_KEY_KP_Space, Key::SPACE },
	{ XKB_KEY_KP_Tab, Key::TAB },
	{ XKB_KEY_KP_Enter, Key::KP_ENTER },
	{ XKB_KEY_Home, Key::HOME },
	{ XKB_KEY_Left, Key::LEFT },
	{ XKB_KEY_Up, Key::UP },
	{ XKB_KEY_Right, Key::RIGHT },
	{ XKB_KEY_Down, Key::DOWN },
	{ XKB_KEY_Prior, Key::PAGEUP },
	{ XKB_KEY_Next, Key::PAGEDOWN },
	{ XKB_KEY_End, Key::END },
	{ XKB_KEY_Begin, Key::CLEAR },
	{ XKB_KEY_Insert, Key::INSERT },
	{ XKB_KEY_Delete, Key::KEY_DELETE },
	//{ XKB_KEY_KP_Equal,                Key::EQUAL   },
	//{ XKB_KEY_KP_Separator,            Key::COMMA   },
	{ XKB_KEY_KP_Decimal, Key::KP_PERIOD },
	{ XKB_KEY_KP_Delete, Key::KP_PERIOD },
	{ XKB_KEY_KP_Multiply, Key::KP_MULTIPLY },
	{ XKB_KEY_KP_Divide, Key::KP_DIVIDE },
	{ XKB_KEY_KP_Subtract, Key::KP_SUBTRACT },
	{ XKB_KEY_KP_Add, Key::KP_ADD },
	{ XKB_KEY_KP_0, Key::KP_0 },
	{ XKB_KEY_KP_1, Key::KP_1 },
	{ XKB_KEY_KP_2, Key::KP_2 },
	{ XKB_KEY_KP_3, Key::KP_3 },
	{ XKB_KEY_KP_4, Key::KP_4 },
	{ XKB_KEY_KP_5, Key::KP_5 },
	{ XKB_KEY_KP_6, Key::KP_6 },
	{ XKB_KEY_KP_7, Key::KP_7 },
	{ XKB_KEY_KP_8, Key::KP_8 },
	{ XKB_KEY_KP_9, Key::KP_9 },

	// same but with numlock
	{ XKB_KEY_KP_Insert, Key::KP_0 },
	{ XKB_KEY_KP_End, Key::KP_1 },
	{ XKB_KEY_KP_Down, Key::KP_2 },
	{ XKB_KEY_KP_Page_Down, Key::KP_3 },
	{ XKB_KEY_KP_Left, Key::KP_4 },
	{ XKB_KEY_KP_Begin, Key::KP_5 },
	{ XKB_KEY_KP_Right, Key::KP_6 },
	{ XKB_KEY_KP_Home, Key::KP_7 },
	{ XKB_KEY_KP_Up, Key::KP_8 },
	{ XKB_KEY_KP_Page_Up, Key::KP_9 },
	{ XKB_KEY_F1, Key::F1 },
	{ XKB_KEY_F2, Key::F2 },
	{ XKB_KEY_F3, Key::F3 },
	{ XKB_KEY_F4, Key::F4 },
	{ XKB_KEY_F5, Key::F5 },
	{ XKB_KEY_F6, Key::F6 },
	{ XKB_KEY_F7, Key::F7 },
	{ XKB_KEY_F8, Key::F8 },
	{ XKB_KEY_F9, Key::F9 },
	{ XKB_KEY_F10, Key::F10 },
	{ XKB_KEY_F11, Key::F11 },
	{ XKB_KEY_F12, Key::F12 },
	{ XKB_KEY_F13, Key::F13 },
	{ XKB_KEY_F14, Key::F14 },
	{ XKB_KEY_F15, Key::F15 },
	{ XKB_KEY_F16, Key::F16 },

	// media keys
	{ XKB_KEY_XF86Back, Key::BACK },
	{ XKB_KEY_XF86Forward, Key::FORWARD },
	{ XKB_KEY_XF86Stop, Key::STOP },
	{ XKB_KEY_XF86Refresh, Key::REFRESH },
	{ XKB_KEY_XF86Favorites, Key::FAVORITES },
	{ XKB_KEY_XF86AudioMedia, Key::LAUNCHMEDIA },
	{ XKB_KEY_XF86OpenURL, Key::OPENURL },
	{ XKB_KEY_XF86HomePage, Key::HOMEPAGE },
	{ XKB_KEY_XF86Search, Key::SEARCH },
	{ XKB_KEY_XF86AudioLowerVolume, Key::VOLUMEDOWN },
	{ XKB_KEY_XF86AudioMute, Key::VOLUMEMUTE },
	{ XKB_KEY_XF86AudioRaiseVolume, Key::VOLUMEUP },
	{ XKB_KEY_XF86AudioPlay, Key::MEDIAPLAY },
	{ XKB_KEY_XF86AudioStop, Key::MEDIASTOP },
	{ XKB_KEY_XF86AudioPrev, Key::MEDIAPREVIOUS },
	{ XKB_KEY_XF86AudioNext, Key::MEDIANEXT },
	{ XKB_KEY_XF86AudioRecord, Key::MEDIARECORD },

	// launch keys
	{ XKB_KEY_XF86Mail, Key::LAUNCHMAIL },
	{ XKB_KEY_XF86MyComputer, Key::LAUNCH0 },
	{ XKB_KEY_XF86Calculator, Key::LAUNCH1 },
	{ XKB_KEY_XF86Standby, Key::STANDBY },

	{ XKB_KEY_XF86Launch0, Key::LAUNCH2 },
	{ XKB_KEY_XF86Launch1, Key::LAUNCH3 },
	{ XKB_KEY_XF86Launch2, Key::LAUNCH4 },
	{ XKB_KEY_XF86Launch3, Key::LAUNCH5 },
	{ XKB_KEY_XF86Launch4, Key::LAUNCH6 },
	{ XKB_KEY_XF86Launch5, Key::LAUNCH7 },
	{ XKB_KEY_XF86Launch6, Key::LAUNCH8 },
	{ XKB_KEY_XF86Launch7, Key::LAUNCH9 },
	{ XKB_KEY_XF86Launch8, Key::LAUNCHA },
	{ XKB_KEY_XF86Launch9, Key::LAUNCHB },
	{ XKB_KEY_XF86LaunchA, Key::LAUNCHC },
	{ XKB_KEY_XF86LaunchB, Key::LAUNCHD },
	{ XKB_KEY_XF86LaunchC, Key::LAUNCHE },
	{ XKB_KEY_XF86LaunchD, Key::LAUNCHF },

	{ 0, Key::NONE }
};

struct _TranslatePair {
	Key keysym;
	unsigned int keycode;
};

static _TranslatePair _scancode_to_keycode[] = {
	{ Key::ESCAPE, 0x09 },
	{ Key::KEY_1, 0x0A },
	{ Key::KEY_2, 0x0B },
	{ Key::KEY_3, 0x0C },
	{ Key::KEY_4, 0x0D },
	{ Key::KEY_5, 0x0E },
	{ Key::KEY_6, 0x0F },
	{ Key::KEY_7, 0x10 },
	{ Key::KEY_8, 0x11 },
	{ Key::KEY_9, 0x12 },
	{ Key::KEY_0, 0x13 },
	{ Key::MINUS, 0x14 },
	{ Key::EQUAL, 0x15 },
	{ Key::BACKSPACE, 0x16 },
	{ Key::TAB, 0x17 },
	{ Key::Q, 0x18 },
	{ Key::W, 0x19 },
	{ Key::E, 0x1A },
	{ Key::R, 0x1B },
	{ Key::T, 0x1C },
	{ Key::Y, 0x1D },
	{ Key::U, 0x1E },
	{ Key::I, 0x1F },
	{ Key::O, 0x20 },
	{ Key::P, 0x21 },
	{ Key::BRACELEFT, 0x22 },
	{ Key::BRACERIGHT, 0x23 },
	{ Key::ENTER, 0x24 },
	{ Key::CTRL, 0x25 },
	{ Key::A, 0x26 },
	{ Key::S, 0x27 },
	{ Key::D, 0x28 },
	{ Key::F, 0x29 },
	{ Key::G, 0x2A },
	{ Key::H, 0x2B },
	{ Key::J, 0x2C },
	{ Key::K, 0x2D },
	{ Key::L, 0x2E },
	{ Key::SEMICOLON, 0x2F },
	{ Key::APOSTROPHE, 0x30 },
	{ Key::QUOTELEFT, 0x31 },
	{ Key::SHIFT, 0x32 },
	{ Key::BACKSLASH, 0x33 },
	{ Key::Z, 0x34 },
	{ Key::X, 0x35 },
	{ Key::C, 0x36 },
	{ Key::V, 0x37 },
	{ Key::B, 0x38 },
	{ Key::N, 0x39 },
	{ Key::M, 0x3A },
	{ Key::COMMA, 0x3B },
	{ Key::PERIOD, 0x3C },
	{ Key::SLASH, 0x3D },
	{ Key::SHIFT, 0x3E },
	{ Key::KP_MULTIPLY, 0x3F },
	{ Key::ALT, 0x40 },
	{ Key::SPACE, 0x41 },
	{ Key::CAPSLOCK, 0x42 },
	{ Key::F1, 0x43 },
	{ Key::F2, 0x44 },
	{ Key::F3, 0x45 },
	{ Key::F4, 0x46 },
	{ Key::F5, 0x47 },
	{ Key::F6, 0x48 },
	{ Key::F7, 0x49 },
	{ Key::F8, 0x4A },
	{ Key::F9, 0x4B },
	{ Key::F10, 0x4C },
	{ Key::NUMLOCK, 0x4D },
	{ Key::SCROLLLOCK, 0x4E },
	{ Key::KP_7, 0x4F },
	{ Key::KP_8, 0x50 },
	{ Key::KP_9, 0x51 },
	{ Key::KP_SUBTRACT, 0x52 },
	{ Key::KP_4, 0x53 },
	{ Key::KP_5, 0x54 },
	{ Key::KP_6, 0x55 },
	{ Key::KP_ADD, 0x56 },
	{ Key::KP_1, 0x57 },
	{ Key::KP_2, 0x58 },
	{ Key::KP_3, 0x59 },
	{ Key::KP_0, 0x5A },
	{ Key::KP_PERIOD, 0x5B },
	//{ Key::???, 0x5E }, //NON US BACKSLASH
	{ Key::F11, 0x5F },
	{ Key::F12, 0x60 },
	{ Key::KP_ENTER, 0x68 },
	{ Key::CTRL, 0x69 },
	{ Key::KP_DIVIDE, 0x6A },
	{ Key::PRINT, 0x6B },
	{ Key::ALT, 0x6C },
	{ Key::ENTER, 0x6D },
	{ Key::HOME, 0x6E },
	{ Key::UP, 0x6F },
	{ Key::PAGEUP, 0x70 },
	{ Key::LEFT, 0x71 },
	{ Key::RIGHT, 0x72 },
	{ Key::END, 0x73 },
	{ Key::DOWN, 0x74 },
	{ Key::PAGEDOWN, 0x75 },
	{ Key::INSERT, 0x76 },
	{ Key::KEY_DELETE, 0x77 },
	{ Key::VOLUMEMUTE, 0x79 },
	{ Key::VOLUMEDOWN, 0x7A },
	{ Key::VOLUMEUP, 0x7B },
	{ Key::PAUSE, 0x7F },
	{ Key::SUPER_L, 0x85 },
	{ Key::SUPER_R, 0x86 },
	{ Key::MENU, 0x87 },
	{ Key::UNKNOWN, 0 }
};

Key KeyMappingXKB::get_scancode(unsigned int p_code) {
	Key keycode = Key::UNKNOWN;
	for (int i = 0; _scancode_to_keycode[i].keysym != Key::UNKNOWN; i++) {
		if (_scancode_to_keycode[i].keycode == p_code) {
			keycode = _scancode_to_keycode[i].keysym;
			break;
		}
	}

	return keycode;
}

xkb_keycode_t KeyMappingXKB::get_xkb_keycode(Key p_keysym) {
	unsigned int code = 0;
	for (int i = 0; _scancode_to_keycode[i].keysym != Key::UNKNOWN; i++) {
		if (_scancode_to_keycode[i].keysym == p_keysym) {
			code = _scancode_to_keycode[i].keycode;
			break;
		}
	}

	return code;
}

Key KeyMappingXKB::get_keycode(xkb_keysym_t p_keysym) {
	// kinda bruteforce.. could optimize.

	if (p_keysym < 0x100) // Latin 1, maps 1-1
		return (Key)p_keysym;

	// look for special key
	for (int idx = 0; _xkb_keysym_to_keycode[idx].keysym != 0; idx++) {
		if (_xkb_keysym_to_keycode[idx].keysym == p_keysym)
			return _xkb_keysym_to_keycode[idx].keycode;
	}

	return Key::UNKNOWN;
}
