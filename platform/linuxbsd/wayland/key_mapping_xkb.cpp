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

void KeyMappingXKB::initialize() {
	// XKB keycode to Godot Key map.

	xkb_keycode_map[XKB_KEY_Escape] = Key::ESCAPE;
	xkb_keycode_map[XKB_KEY_Tab] = Key::TAB;
	xkb_keycode_map[XKB_KEY_ISO_Left_Tab] = Key::BACKTAB;
	xkb_keycode_map[XKB_KEY_BackSpace] = Key::BACKSPACE;
	xkb_keycode_map[XKB_KEY_Return] = Key::ENTER;
	xkb_keycode_map[XKB_KEY_Insert] = Key::INSERT;
	xkb_keycode_map[XKB_KEY_Delete] = Key::KEY_DELETE;
	xkb_keycode_map[XKB_KEY_Clear] = Key::KEY_DELETE;
	xkb_keycode_map[XKB_KEY_Pause] = Key::PAUSE;
	xkb_keycode_map[XKB_KEY_Print] = Key::PRINT;
	xkb_keycode_map[XKB_KEY_Home] = Key::HOME;
	xkb_keycode_map[XKB_KEY_End] = Key::END;
	xkb_keycode_map[XKB_KEY_Left] = Key::LEFT;
	xkb_keycode_map[XKB_KEY_Up] = Key::UP;
	xkb_keycode_map[XKB_KEY_Right] = Key::RIGHT;
	xkb_keycode_map[XKB_KEY_Down] = Key::DOWN;
	xkb_keycode_map[XKB_KEY_Prior] = Key::PAGEUP;
	xkb_keycode_map[XKB_KEY_Next] = Key::PAGEDOWN;
	xkb_keycode_map[XKB_KEY_Shift_L] = Key::SHIFT;
	xkb_keycode_map[XKB_KEY_Shift_R] = Key::SHIFT;
	xkb_keycode_map[XKB_KEY_Shift_Lock] = Key::SHIFT;
	xkb_keycode_map[XKB_KEY_Control_L] = Key::CTRL;
	xkb_keycode_map[XKB_KEY_Control_R] = Key::CTRL;
	xkb_keycode_map[XKB_KEY_Meta_L] = Key::META;
	xkb_keycode_map[XKB_KEY_Meta_R] = Key::META;
	xkb_keycode_map[XKB_KEY_Alt_L] = Key::ALT;
	xkb_keycode_map[XKB_KEY_Alt_R] = Key::ALT;
	xkb_keycode_map[XKB_KEY_Caps_Lock] = Key::CAPSLOCK;
	xkb_keycode_map[XKB_KEY_Num_Lock] = Key::NUMLOCK;
	xkb_keycode_map[XKB_KEY_Scroll_Lock] = Key::SCROLLLOCK;
	xkb_keycode_map[XKB_KEY_less] = Key::QUOTELEFT;
	xkb_keycode_map[XKB_KEY_grave] = Key::SECTION;
	xkb_keycode_map[XKB_KEY_Super_L] = Key::META;
	xkb_keycode_map[XKB_KEY_Super_R] = Key::META;
	xkb_keycode_map[XKB_KEY_Menu] = Key::MENU;
	xkb_keycode_map[XKB_KEY_Hyper_L] = Key::HYPER;
	xkb_keycode_map[XKB_KEY_Hyper_R] = Key::HYPER;
	xkb_keycode_map[XKB_KEY_Help] = Key::HELP;
	xkb_keycode_map[XKB_KEY_KP_Space] = Key::SPACE;
	xkb_keycode_map[XKB_KEY_KP_Tab] = Key::TAB;
	xkb_keycode_map[XKB_KEY_KP_Enter] = Key::KP_ENTER;
	xkb_keycode_map[XKB_KEY_Home] = Key::HOME;
	xkb_keycode_map[XKB_KEY_Left] = Key::LEFT;
	xkb_keycode_map[XKB_KEY_Up] = Key::UP;
	xkb_keycode_map[XKB_KEY_Right] = Key::RIGHT;
	xkb_keycode_map[XKB_KEY_Down] = Key::DOWN;
	xkb_keycode_map[XKB_KEY_Prior] = Key::PAGEUP;
	xkb_keycode_map[XKB_KEY_Next] = Key::PAGEDOWN;
	xkb_keycode_map[XKB_KEY_End] = Key::END;
	xkb_keycode_map[XKB_KEY_Begin] = Key::CLEAR;
	xkb_keycode_map[XKB_KEY_Insert] = Key::INSERT;
	xkb_keycode_map[XKB_KEY_Delete] = Key::KEY_DELETE;
	xkb_keycode_map[XKB_KEY_KP_Equal] = Key::EQUAL;
	xkb_keycode_map[XKB_KEY_KP_Separator] = Key::COMMA;
	xkb_keycode_map[XKB_KEY_KP_Decimal] = Key::KP_PERIOD;
	xkb_keycode_map[XKB_KEY_KP_Multiply] = Key::KP_MULTIPLY;
	xkb_keycode_map[XKB_KEY_KP_Divide] = Key::KP_DIVIDE;
	xkb_keycode_map[XKB_KEY_KP_Subtract] = Key::KP_SUBTRACT;
	xkb_keycode_map[XKB_KEY_KP_Add] = Key::KP_ADD;
	xkb_keycode_map[XKB_KEY_KP_0] = Key::KP_0;
	xkb_keycode_map[XKB_KEY_KP_1] = Key::KP_1;
	xkb_keycode_map[XKB_KEY_KP_2] = Key::KP_2;
	xkb_keycode_map[XKB_KEY_KP_3] = Key::KP_3;
	xkb_keycode_map[XKB_KEY_KP_4] = Key::KP_4;
	xkb_keycode_map[XKB_KEY_KP_5] = Key::KP_5;
	xkb_keycode_map[XKB_KEY_KP_6] = Key::KP_6;
	xkb_keycode_map[XKB_KEY_KP_7] = Key::KP_7;
	xkb_keycode_map[XKB_KEY_KP_8] = Key::KP_8;
	xkb_keycode_map[XKB_KEY_KP_9] = Key::KP_9;
	// Same keys but with numlock off.
	xkb_keycode_map[XKB_KEY_KP_Insert] = Key::INSERT;
	xkb_keycode_map[XKB_KEY_KP_Delete] = Key::KEY_DELETE;
	xkb_keycode_map[XKB_KEY_KP_End] = Key::END;
	xkb_keycode_map[XKB_KEY_KP_Down] = Key::DOWN;
	xkb_keycode_map[XKB_KEY_KP_Page_Down] = Key::PAGEDOWN;
	xkb_keycode_map[XKB_KEY_KP_Left] = Key::LEFT;
	// X11 documents this (numpad 5) as "begin of line" but no toolkit seems to interpret it this way.
	// On Windows this is emitting Key::Clear so for consistency it will be mapped to Key::Clear
	xkb_keycode_map[XKB_KEY_KP_Begin] = Key::CLEAR;
	xkb_keycode_map[XKB_KEY_KP_Right] = Key::RIGHT;
	xkb_keycode_map[XKB_KEY_KP_Home] = Key::HOME;
	xkb_keycode_map[XKB_KEY_KP_Up] = Key::UP;
	xkb_keycode_map[XKB_KEY_KP_Page_Up] = Key::PAGEUP;
	xkb_keycode_map[XKB_KEY_F1] = Key::F1;
	xkb_keycode_map[XKB_KEY_F2] = Key::F2;
	xkb_keycode_map[XKB_KEY_F3] = Key::F3;
	xkb_keycode_map[XKB_KEY_F4] = Key::F4;
	xkb_keycode_map[XKB_KEY_F5] = Key::F5;
	xkb_keycode_map[XKB_KEY_F6] = Key::F6;
	xkb_keycode_map[XKB_KEY_F7] = Key::F7;
	xkb_keycode_map[XKB_KEY_F8] = Key::F8;
	xkb_keycode_map[XKB_KEY_F9] = Key::F9;
	xkb_keycode_map[XKB_KEY_F10] = Key::F10;
	xkb_keycode_map[XKB_KEY_F11] = Key::F11;
	xkb_keycode_map[XKB_KEY_F12] = Key::F12;
	xkb_keycode_map[XKB_KEY_F13] = Key::F13;
	xkb_keycode_map[XKB_KEY_F14] = Key::F14;
	xkb_keycode_map[XKB_KEY_F15] = Key::F15;
	xkb_keycode_map[XKB_KEY_F16] = Key::F16;
	xkb_keycode_map[XKB_KEY_F17] = Key::F17;
	xkb_keycode_map[XKB_KEY_F18] = Key::F18;
	xkb_keycode_map[XKB_KEY_F19] = Key::F19;
	xkb_keycode_map[XKB_KEY_F20] = Key::F20;
	xkb_keycode_map[XKB_KEY_F21] = Key::F21;
	xkb_keycode_map[XKB_KEY_F22] = Key::F22;
	xkb_keycode_map[XKB_KEY_F23] = Key::F23;
	xkb_keycode_map[XKB_KEY_F24] = Key::F24;
	xkb_keycode_map[XKB_KEY_F25] = Key::F25;
	xkb_keycode_map[XKB_KEY_F26] = Key::F26;
	xkb_keycode_map[XKB_KEY_F27] = Key::F27;
	xkb_keycode_map[XKB_KEY_F28] = Key::F28;
	xkb_keycode_map[XKB_KEY_F29] = Key::F29;
	xkb_keycode_map[XKB_KEY_F30] = Key::F30;
	xkb_keycode_map[XKB_KEY_F31] = Key::F31;
	xkb_keycode_map[XKB_KEY_F32] = Key::F32;
	xkb_keycode_map[XKB_KEY_F33] = Key::F33;
	xkb_keycode_map[XKB_KEY_F34] = Key::F34;
	xkb_keycode_map[XKB_KEY_F35] = Key::F35;
	xkb_keycode_map[XKB_KEY_yen] = Key::YEN;
	xkb_keycode_map[XKB_KEY_section] = Key::SECTION;
	// Media keys.
	xkb_keycode_map[XKB_KEY_XF86Back] = Key::BACK;
	xkb_keycode_map[XKB_KEY_XF86Forward] = Key::FORWARD;
	xkb_keycode_map[XKB_KEY_XF86Stop] = Key::STOP;
	xkb_keycode_map[XKB_KEY_XF86Refresh] = Key::REFRESH;
	xkb_keycode_map[XKB_KEY_XF86Favorites] = Key::FAVORITES;
	xkb_keycode_map[XKB_KEY_XF86OpenURL] = Key::OPENURL;
	xkb_keycode_map[XKB_KEY_XF86HomePage] = Key::HOMEPAGE;
	xkb_keycode_map[XKB_KEY_XF86Search] = Key::SEARCH;
	xkb_keycode_map[XKB_KEY_XF86AudioLowerVolume] = Key::VOLUMEDOWN;
	xkb_keycode_map[XKB_KEY_XF86AudioMute] = Key::VOLUMEMUTE;
	xkb_keycode_map[XKB_KEY_XF86AudioRaiseVolume] = Key::VOLUMEUP;
	xkb_keycode_map[XKB_KEY_XF86AudioPlay] = Key::MEDIAPLAY;
	xkb_keycode_map[XKB_KEY_XF86AudioStop] = Key::MEDIASTOP;
	xkb_keycode_map[XKB_KEY_XF86AudioPrev] = Key::MEDIAPREVIOUS;
	xkb_keycode_map[XKB_KEY_XF86AudioNext] = Key::MEDIANEXT;
	xkb_keycode_map[XKB_KEY_XF86AudioRecord] = Key::MEDIARECORD;
	xkb_keycode_map[XKB_KEY_XF86Standby] = Key::STANDBY;
	// Launch keys.
	xkb_keycode_map[XKB_KEY_XF86Mail] = Key::LAUNCHMAIL;
	xkb_keycode_map[XKB_KEY_XF86AudioMedia] = Key::LAUNCHMEDIA;
	xkb_keycode_map[XKB_KEY_XF86MyComputer] = Key::LAUNCH0;
	xkb_keycode_map[XKB_KEY_XF86Calculator] = Key::LAUNCH1;
	xkb_keycode_map[XKB_KEY_XF86Launch0] = Key::LAUNCH2;
	xkb_keycode_map[XKB_KEY_XF86Launch1] = Key::LAUNCH3;
	xkb_keycode_map[XKB_KEY_XF86Launch2] = Key::LAUNCH4;
	xkb_keycode_map[XKB_KEY_XF86Launch3] = Key::LAUNCH5;
	xkb_keycode_map[XKB_KEY_XF86Launch4] = Key::LAUNCH6;
	xkb_keycode_map[XKB_KEY_XF86Launch5] = Key::LAUNCH7;
	xkb_keycode_map[XKB_KEY_XF86Launch6] = Key::LAUNCH8;
	xkb_keycode_map[XKB_KEY_XF86Launch7] = Key::LAUNCH9;
	xkb_keycode_map[XKB_KEY_XF86Launch8] = Key::LAUNCHA;
	xkb_keycode_map[XKB_KEY_XF86Launch9] = Key::LAUNCHB;
	xkb_keycode_map[XKB_KEY_XF86LaunchA] = Key::LAUNCHC;
	xkb_keycode_map[XKB_KEY_XF86LaunchB] = Key::LAUNCHD;
	xkb_keycode_map[XKB_KEY_XF86LaunchC] = Key::LAUNCHE;
	xkb_keycode_map[XKB_KEY_XF86LaunchD] = Key::LAUNCHF;

	// Scancode to Godot Key map.
	scancode_map[0x09] = Key::ESCAPE;
	scancode_map[0x0A] = Key::KEY_1;
	scancode_map[0x0B] = Key::KEY_2;
	scancode_map[0x0C] = Key::KEY_3;
	scancode_map[0x0D] = Key::KEY_4;
	scancode_map[0x0E] = Key::KEY_5;
	scancode_map[0x0F] = Key::KEY_6;
	scancode_map[0x10] = Key::KEY_7;
	scancode_map[0x11] = Key::KEY_8;
	scancode_map[0x12] = Key::KEY_9;
	scancode_map[0x13] = Key::KEY_0;
	scancode_map[0x14] = Key::MINUS;
	scancode_map[0x15] = Key::EQUAL;
	scancode_map[0x16] = Key::BACKSPACE;
	scancode_map[0x17] = Key::TAB;
	scancode_map[0x18] = Key::Q;
	scancode_map[0x19] = Key::W;
	scancode_map[0x1A] = Key::E;
	scancode_map[0x1B] = Key::R;
	scancode_map[0x1C] = Key::T;
	scancode_map[0x1D] = Key::Y;
	scancode_map[0x1E] = Key::U;
	scancode_map[0x1F] = Key::I;
	scancode_map[0x20] = Key::O;
	scancode_map[0x21] = Key::P;
	scancode_map[0x22] = Key::BRACELEFT;
	scancode_map[0x23] = Key::BRACERIGHT;
	scancode_map[0x24] = Key::ENTER;
	scancode_map[0x25] = Key::CTRL; // Left
	scancode_map[0x26] = Key::A;
	scancode_map[0x27] = Key::S;
	scancode_map[0x28] = Key::D;
	scancode_map[0x29] = Key::F;
	scancode_map[0x2A] = Key::G;
	scancode_map[0x2B] = Key::H;
	scancode_map[0x2C] = Key::J;
	scancode_map[0x2D] = Key::K;
	scancode_map[0x2E] = Key::L;
	scancode_map[0x2F] = Key::SEMICOLON;
	scancode_map[0x30] = Key::APOSTROPHE;
	scancode_map[0x31] = Key::SECTION;
	scancode_map[0x32] = Key::SHIFT; // Left
	scancode_map[0x33] = Key::BACKSLASH;
	scancode_map[0x34] = Key::Z;
	scancode_map[0x35] = Key::X;
	scancode_map[0x36] = Key::C;
	scancode_map[0x37] = Key::V;
	scancode_map[0x38] = Key::B;
	scancode_map[0x39] = Key::N;
	scancode_map[0x3A] = Key::M;
	scancode_map[0x3B] = Key::COMMA;
	scancode_map[0x3C] = Key::PERIOD;
	scancode_map[0x3D] = Key::SLASH;
	scancode_map[0x3E] = Key::SHIFT; // Right
	scancode_map[0x3F] = Key::KP_MULTIPLY;
	scancode_map[0x40] = Key::ALT; // Left
	scancode_map[0x41] = Key::SPACE;
	scancode_map[0x42] = Key::CAPSLOCK;
	scancode_map[0x43] = Key::F1;
	scancode_map[0x44] = Key::F2;
	scancode_map[0x45] = Key::F3;
	scancode_map[0x46] = Key::F4;
	scancode_map[0x47] = Key::F5;
	scancode_map[0x48] = Key::F6;
	scancode_map[0x49] = Key::F7;
	scancode_map[0x4A] = Key::F8;
	scancode_map[0x4B] = Key::F9;
	scancode_map[0x4C] = Key::F10;
	scancode_map[0x4D] = Key::NUMLOCK;
	scancode_map[0x4E] = Key::SCROLLLOCK;
	scancode_map[0x4F] = Key::KP_7;
	scancode_map[0x50] = Key::KP_8;
	scancode_map[0x51] = Key::KP_9;
	scancode_map[0x52] = Key::KP_SUBTRACT;
	scancode_map[0x53] = Key::KP_4;
	scancode_map[0x54] = Key::KP_5;
	scancode_map[0x55] = Key::KP_6;
	scancode_map[0x56] = Key::KP_ADD;
	scancode_map[0x57] = Key::KP_1;
	scancode_map[0x58] = Key::KP_2;
	scancode_map[0x59] = Key::KP_3;
	scancode_map[0x5A] = Key::KP_0;
	scancode_map[0x5B] = Key::KP_PERIOD;
	//scancode_map[0x5C]
	//scancode_map[0x5D] // Zenkaku Hankaku
	scancode_map[0x5E] = Key::QUOTELEFT;
	scancode_map[0x5F] = Key::F11;
	scancode_map[0x60] = Key::F12;
	//scancode_map[0x61] // Romaji
	//scancode_map[0x62] // Katakana
	//scancode_map[0x63] // Hiragana
	//scancode_map[0x64] // Henkan
	//scancode_map[0x65] // Hiragana Katakana
	//scancode_map[0x66] // Muhenkan
	scancode_map[0x67] = Key::COMMA; // KP_Separator
	scancode_map[0x68] = Key::KP_ENTER;
	scancode_map[0x69] = Key::CTRL; // Right
	scancode_map[0x6A] = Key::KP_DIVIDE;
	scancode_map[0x6B] = Key::PRINT;
	scancode_map[0x6C] = Key::ALT; // Right
	scancode_map[0x6D] = Key::ENTER;
	scancode_map[0x6E] = Key::HOME;
	scancode_map[0x6F] = Key::UP;
	scancode_map[0x70] = Key::PAGEUP;
	scancode_map[0x71] = Key::LEFT;
	scancode_map[0x72] = Key::RIGHT;
	scancode_map[0x73] = Key::END;
	scancode_map[0x74] = Key::DOWN;
	scancode_map[0x75] = Key::PAGEDOWN;
	scancode_map[0x76] = Key::INSERT;
	scancode_map[0x77] = Key::KEY_DELETE;
	//scancode_map[0x78] // Macro
	scancode_map[0x79] = Key::VOLUMEMUTE;
	scancode_map[0x7A] = Key::VOLUMEDOWN;
	scancode_map[0x7B] = Key::VOLUMEUP;
	//scancode_map[0x7C] // Power
	scancode_map[0x7D] = Key::EQUAL; // KP_Equal
	//scancode_map[0x7E] // KP_PlusMinus
	scancode_map[0x7F] = Key::PAUSE;
	scancode_map[0x80] = Key::LAUNCH0;
	scancode_map[0x81] = Key::COMMA; // KP_Comma
	//scancode_map[0x82] // Hangul
	//scancode_map[0x83] // Hangul_Hanja
	scancode_map[0x84] = Key::YEN;
	scancode_map[0x85] = Key::META; // Left
	scancode_map[0x86] = Key::META; // Right
	scancode_map[0x87] = Key::MENU;

	scancode_map[0xA6] = Key::BACK; // On Chromebooks
	scancode_map[0xA7] = Key::FORWARD; // On Chromebooks

	scancode_map[0xB5] = Key::REFRESH; // On Chromebooks

	scancode_map[0xBF] = Key::F13;
	scancode_map[0xC0] = Key::F14;
	scancode_map[0xC1] = Key::F15;
	scancode_map[0xC2] = Key::F16;
	scancode_map[0xC3] = Key::F17;
	scancode_map[0xC4] = Key::F18;
	scancode_map[0xC5] = Key::F19;
	scancode_map[0xC6] = Key::F20;
	scancode_map[0xC7] = Key::F21;
	scancode_map[0xC8] = Key::F22;
	scancode_map[0xC9] = Key::F23;
	scancode_map[0xCA] = Key::F24;
	scancode_map[0xCB] = Key::F25;
	scancode_map[0xCC] = Key::F26;
	scancode_map[0xCD] = Key::F27;
	scancode_map[0xCE] = Key::F28;
	scancode_map[0xCF] = Key::F29;
	scancode_map[0xD0] = Key::F30;
	scancode_map[0xD1] = Key::F31;
	scancode_map[0xD2] = Key::F32;
	scancode_map[0xD3] = Key::F33;
	scancode_map[0xD4] = Key::F34;
	scancode_map[0xD5] = Key::F35;

	// Godot to scancode map.
	for (const KeyValue<unsigned int, Key> &E : scancode_map) {
		scancode_map_inv[E.value] = E.key;
	}

	// Scancode to physical location map.
	// Ctrl.
	location_map[0x25] = KeyLocation::LEFT;
	location_map[0x69] = KeyLocation::RIGHT;
	// Shift.
	location_map[0x32] = KeyLocation::LEFT;
	location_map[0x3E] = KeyLocation::RIGHT;
	// Alt.
	location_map[0x40] = KeyLocation::LEFT;
	location_map[0x6C] = KeyLocation::RIGHT;
	// Meta.
	location_map[0x85] = KeyLocation::LEFT;
	location_map[0x86] = KeyLocation::RIGHT;
}

bool KeyMappingXKB::is_sym_numpad(xkb_keysym_t p_keysym) {
	switch (p_keysym) {
		case XKB_KEY_KP_Equal:
		case XKB_KEY_KP_Add:
		case XKB_KEY_KP_Subtract:
		case XKB_KEY_KP_Multiply:
		case XKB_KEY_KP_Divide:
		case XKB_KEY_KP_Separator:
		case XKB_KEY_KP_Decimal:
		case XKB_KEY_KP_0:
		case XKB_KEY_KP_1:
		case XKB_KEY_KP_2:
		case XKB_KEY_KP_3:
		case XKB_KEY_KP_4:
		case XKB_KEY_KP_5:
		case XKB_KEY_KP_6:
		case XKB_KEY_KP_7:
		case XKB_KEY_KP_8:
		case XKB_KEY_KP_9: {
			return true;
		} break;
	}

	return false;
}

Key KeyMappingXKB::get_keycode(xkb_keycode_t p_keysym) {
	if (p_keysym >= 0x20 && p_keysym < 0x7E) { // ASCII, maps 1-1
		if (p_keysym > 0x60 && p_keysym < 0x7B) { // Lowercase ASCII.
			return (Key)(p_keysym - 32);
		} else {
			return (Key)p_keysym;
		}
	}

	const Key *key = xkb_keycode_map.getptr(p_keysym);
	if (key) {
		return *key;
	}
	return Key::NONE;
}

Key KeyMappingXKB::get_scancode(unsigned int p_code) {
	const Key *key = scancode_map.getptr(p_code);
	if (key) {
		return *key;
	}

	return Key::NONE;
}

xkb_keycode_t KeyMappingXKB::get_xkb_keycode(Key p_key) {
	const unsigned int *key = scancode_map_inv.getptr(p_key);
	if (key) {
		return *key;
	}
	return 0x00;
}

KeyLocation KeyMappingXKB::get_location(unsigned int p_code) {
	const KeyLocation *location = location_map.getptr(p_code);
	if (location) {
		return *location;
	}
	return KeyLocation::UNSPECIFIED;
}
