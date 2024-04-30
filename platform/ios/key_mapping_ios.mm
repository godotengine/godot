/**************************************************************************/
/*  key_mapping_ios.mm                                                    */
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

#import "key_mapping_ios.h"

#include "core/templates/hash_map.h"

struct HashMapHasherKeys {
	static _FORCE_INLINE_ uint32_t hash(const Key p_key) { return hash_fmix32(static_cast<uint32_t>(p_key)); }
	static _FORCE_INLINE_ uint32_t hash(const CFIndex p_key) { return hash_fmix32(p_key); }
};

HashMap<CFIndex, Key, HashMapHasherKeys> keyusage_map;
HashMap<CFIndex, KeyLocation, HashMapHasherKeys> location_map;

void KeyMappingIOS::initialize() {
	if (@available(iOS 13.4, *)) {
		keyusage_map[UIKeyboardHIDUsageKeyboardA] = Key::A;
		keyusage_map[UIKeyboardHIDUsageKeyboardB] = Key::B;
		keyusage_map[UIKeyboardHIDUsageKeyboardC] = Key::C;
		keyusage_map[UIKeyboardHIDUsageKeyboardD] = Key::D;
		keyusage_map[UIKeyboardHIDUsageKeyboardE] = Key::E;
		keyusage_map[UIKeyboardHIDUsageKeyboardF] = Key::F;
		keyusage_map[UIKeyboardHIDUsageKeyboardG] = Key::G;
		keyusage_map[UIKeyboardHIDUsageKeyboardH] = Key::H;
		keyusage_map[UIKeyboardHIDUsageKeyboardI] = Key::I;
		keyusage_map[UIKeyboardHIDUsageKeyboardJ] = Key::J;
		keyusage_map[UIKeyboardHIDUsageKeyboardK] = Key::K;
		keyusage_map[UIKeyboardHIDUsageKeyboardL] = Key::L;
		keyusage_map[UIKeyboardHIDUsageKeyboardM] = Key::M;
		keyusage_map[UIKeyboardHIDUsageKeyboardN] = Key::N;
		keyusage_map[UIKeyboardHIDUsageKeyboardO] = Key::O;
		keyusage_map[UIKeyboardHIDUsageKeyboardP] = Key::P;
		keyusage_map[UIKeyboardHIDUsageKeyboardQ] = Key::Q;
		keyusage_map[UIKeyboardHIDUsageKeyboardR] = Key::R;
		keyusage_map[UIKeyboardHIDUsageKeyboardS] = Key::S;
		keyusage_map[UIKeyboardHIDUsageKeyboardT] = Key::T;
		keyusage_map[UIKeyboardHIDUsageKeyboardU] = Key::U;
		keyusage_map[UIKeyboardHIDUsageKeyboardV] = Key::V;
		keyusage_map[UIKeyboardHIDUsageKeyboardW] = Key::W;
		keyusage_map[UIKeyboardHIDUsageKeyboardX] = Key::X;
		keyusage_map[UIKeyboardHIDUsageKeyboardY] = Key::Y;
		keyusage_map[UIKeyboardHIDUsageKeyboardZ] = Key::Z;
		keyusage_map[UIKeyboardHIDUsageKeyboard0] = Key::KEY_0;
		keyusage_map[UIKeyboardHIDUsageKeyboard1] = Key::KEY_1;
		keyusage_map[UIKeyboardHIDUsageKeyboard2] = Key::KEY_2;
		keyusage_map[UIKeyboardHIDUsageKeyboard3] = Key::KEY_3;
		keyusage_map[UIKeyboardHIDUsageKeyboard4] = Key::KEY_4;
		keyusage_map[UIKeyboardHIDUsageKeyboard5] = Key::KEY_5;
		keyusage_map[UIKeyboardHIDUsageKeyboard6] = Key::KEY_6;
		keyusage_map[UIKeyboardHIDUsageKeyboard7] = Key::KEY_7;
		keyusage_map[UIKeyboardHIDUsageKeyboard8] = Key::KEY_8;
		keyusage_map[UIKeyboardHIDUsageKeyboard9] = Key::KEY_9;
		keyusage_map[UIKeyboardHIDUsageKeyboardBackslash] = Key::BACKSLASH;
		keyusage_map[UIKeyboardHIDUsageKeyboardCloseBracket] = Key::BRACKETRIGHT;
		keyusage_map[UIKeyboardHIDUsageKeyboardComma] = Key::COMMA;
		keyusage_map[UIKeyboardHIDUsageKeyboardEqualSign] = Key::EQUAL;
		keyusage_map[UIKeyboardHIDUsageKeyboardHyphen] = Key::MINUS;
		keyusage_map[UIKeyboardHIDUsageKeyboardNonUSBackslash] = Key::SECTION;
		keyusage_map[UIKeyboardHIDUsageKeyboardNonUSPound] = Key::ASCIITILDE;
		keyusage_map[UIKeyboardHIDUsageKeyboardOpenBracket] = Key::BRACKETLEFT;
		keyusage_map[UIKeyboardHIDUsageKeyboardPeriod] = Key::PERIOD;
		keyusage_map[UIKeyboardHIDUsageKeyboardQuote] = Key::QUOTEDBL;
		keyusage_map[UIKeyboardHIDUsageKeyboardSemicolon] = Key::SEMICOLON;
		keyusage_map[UIKeyboardHIDUsageKeyboardSeparator] = Key::SECTION;
		keyusage_map[UIKeyboardHIDUsageKeyboardSlash] = Key::SLASH;
		keyusage_map[UIKeyboardHIDUsageKeyboardSpacebar] = Key::SPACE;
		keyusage_map[UIKeyboardHIDUsageKeyboardCapsLock] = Key::CAPSLOCK;
		keyusage_map[UIKeyboardHIDUsageKeyboardLeftAlt] = Key::ALT;
		keyusage_map[UIKeyboardHIDUsageKeyboardLeftControl] = Key::CTRL;
		keyusage_map[UIKeyboardHIDUsageKeyboardLeftShift] = Key::SHIFT;
		keyusage_map[UIKeyboardHIDUsageKeyboardRightAlt] = Key::ALT;
		keyusage_map[UIKeyboardHIDUsageKeyboardRightControl] = Key::CTRL;
		keyusage_map[UIKeyboardHIDUsageKeyboardRightShift] = Key::SHIFT;
		keyusage_map[UIKeyboardHIDUsageKeyboardScrollLock] = Key::SCROLLLOCK;
		keyusage_map[UIKeyboardHIDUsageKeyboardLeftArrow] = Key::LEFT;
		keyusage_map[UIKeyboardHIDUsageKeyboardRightArrow] = Key::RIGHT;
		keyusage_map[UIKeyboardHIDUsageKeyboardUpArrow] = Key::UP;
		keyusage_map[UIKeyboardHIDUsageKeyboardDownArrow] = Key::DOWN;
		keyusage_map[UIKeyboardHIDUsageKeyboardPageUp] = Key::PAGEUP;
		keyusage_map[UIKeyboardHIDUsageKeyboardPageDown] = Key::PAGEDOWN;
		keyusage_map[UIKeyboardHIDUsageKeyboardHome] = Key::HOME;
		keyusage_map[UIKeyboardHIDUsageKeyboardEnd] = Key::END;
		keyusage_map[UIKeyboardHIDUsageKeyboardDeleteForward] = Key::KEY_DELETE;
		keyusage_map[UIKeyboardHIDUsageKeyboardDeleteOrBackspace] = Key::BACKSPACE;
		keyusage_map[UIKeyboardHIDUsageKeyboardEscape] = Key::ESCAPE;
		keyusage_map[UIKeyboardHIDUsageKeyboardInsert] = Key::INSERT;
		keyusage_map[UIKeyboardHIDUsageKeyboardReturn] = Key::ENTER;
		keyusage_map[UIKeyboardHIDUsageKeyboardTab] = Key::TAB;
		keyusage_map[UIKeyboardHIDUsageKeyboardF1] = Key::F1;
		keyusage_map[UIKeyboardHIDUsageKeyboardF2] = Key::F2;
		keyusage_map[UIKeyboardHIDUsageKeyboardF3] = Key::F3;
		keyusage_map[UIKeyboardHIDUsageKeyboardF4] = Key::F4;
		keyusage_map[UIKeyboardHIDUsageKeyboardF5] = Key::F5;
		keyusage_map[UIKeyboardHIDUsageKeyboardF6] = Key::F6;
		keyusage_map[UIKeyboardHIDUsageKeyboardF7] = Key::F7;
		keyusage_map[UIKeyboardHIDUsageKeyboardF8] = Key::F8;
		keyusage_map[UIKeyboardHIDUsageKeyboardF9] = Key::F9;
		keyusage_map[UIKeyboardHIDUsageKeyboardF10] = Key::F10;
		keyusage_map[UIKeyboardHIDUsageKeyboardF11] = Key::F11;
		keyusage_map[UIKeyboardHIDUsageKeyboardF12] = Key::F12;
		keyusage_map[UIKeyboardHIDUsageKeyboardF13] = Key::F13;
		keyusage_map[UIKeyboardHIDUsageKeyboardF14] = Key::F14;
		keyusage_map[UIKeyboardHIDUsageKeyboardF15] = Key::F15;
		keyusage_map[UIKeyboardHIDUsageKeyboardF16] = Key::F16;
		keyusage_map[UIKeyboardHIDUsageKeyboardF17] = Key::F17;
		keyusage_map[UIKeyboardHIDUsageKeyboardF18] = Key::F18;
		keyusage_map[UIKeyboardHIDUsageKeyboardF19] = Key::F19;
		keyusage_map[UIKeyboardHIDUsageKeyboardF20] = Key::F20;
		keyusage_map[UIKeyboardHIDUsageKeyboardF21] = Key::F21;
		keyusage_map[UIKeyboardHIDUsageKeyboardF22] = Key::F22;
		keyusage_map[UIKeyboardHIDUsageKeyboardF23] = Key::F23;
		keyusage_map[UIKeyboardHIDUsageKeyboardF24] = Key::F24;
		keyusage_map[UIKeyboardHIDUsageKeypad0] = Key::KP_0;
		keyusage_map[UIKeyboardHIDUsageKeypad1] = Key::KP_1;
		keyusage_map[UIKeyboardHIDUsageKeypad2] = Key::KP_2;
		keyusage_map[UIKeyboardHIDUsageKeypad3] = Key::KP_3;
		keyusage_map[UIKeyboardHIDUsageKeypad4] = Key::KP_4;
		keyusage_map[UIKeyboardHIDUsageKeypad5] = Key::KP_5;
		keyusage_map[UIKeyboardHIDUsageKeypad6] = Key::KP_6;
		keyusage_map[UIKeyboardHIDUsageKeypad7] = Key::KP_7;
		keyusage_map[UIKeyboardHIDUsageKeypad8] = Key::KP_8;
		keyusage_map[UIKeyboardHIDUsageKeypad9] = Key::KP_9;
		keyusage_map[UIKeyboardHIDUsageKeypadAsterisk] = Key::KP_MULTIPLY;
		keyusage_map[UIKeyboardHIDUsageKeyboardGraveAccentAndTilde] = Key::BAR;
		keyusage_map[UIKeyboardHIDUsageKeypadEnter] = Key::KP_ENTER;
		keyusage_map[UIKeyboardHIDUsageKeypadHyphen] = Key::KP_SUBTRACT;
		keyusage_map[UIKeyboardHIDUsageKeypadNumLock] = Key::NUMLOCK;
		keyusage_map[UIKeyboardHIDUsageKeypadPeriod] = Key::KP_PERIOD;
		keyusage_map[UIKeyboardHIDUsageKeypadPlus] = Key::KP_ADD;
		keyusage_map[UIKeyboardHIDUsageKeypadSlash] = Key::KP_DIVIDE;
		keyusage_map[UIKeyboardHIDUsageKeyboardPause] = Key::PAUSE;
		keyusage_map[UIKeyboardHIDUsageKeyboardStop] = Key::STOP;
		keyusage_map[UIKeyboardHIDUsageKeyboardMute] = Key::VOLUMEMUTE;
		keyusage_map[UIKeyboardHIDUsageKeyboardVolumeUp] = Key::VOLUMEUP;
		keyusage_map[UIKeyboardHIDUsageKeyboardVolumeDown] = Key::VOLUMEDOWN;
		keyusage_map[UIKeyboardHIDUsageKeyboardFind] = Key::SEARCH;
		keyusage_map[UIKeyboardHIDUsageKeyboardHelp] = Key::HELP;
		keyusage_map[UIKeyboardHIDUsageKeyboardLeftGUI] = Key::META;
		keyusage_map[UIKeyboardHIDUsageKeyboardRightGUI] = Key::META;
		keyusage_map[UIKeyboardHIDUsageKeyboardMenu] = Key::MENU;
		keyusage_map[UIKeyboardHIDUsageKeyboardPrintScreen] = Key::PRINT;
		keyusage_map[UIKeyboardHIDUsageKeyboardReturnOrEnter] = Key::ENTER;
		keyusage_map[UIKeyboardHIDUsageKeyboardSysReqOrAttention] = Key::SYSREQ;
		keyusage_map[0x01AE] = Key::KEYBOARD; // On-screen keyboard key on smart connector keyboard.
		keyusage_map[0x029D] = Key::GLOBE; // "Globe" key on smart connector / Mac keyboard.
		keyusage_map[UIKeyboardHIDUsageKeyboardLANG1] = Key::JIS_EISU;
		keyusage_map[UIKeyboardHIDUsageKeyboardLANG2] = Key::JIS_KANA;

		location_map[UIKeyboardHIDUsageKeyboardLeftAlt] = KeyLocation::LEFT;
		location_map[UIKeyboardHIDUsageKeyboardRightAlt] = KeyLocation::RIGHT;
		location_map[UIKeyboardHIDUsageKeyboardLeftControl] = KeyLocation::LEFT;
		location_map[UIKeyboardHIDUsageKeyboardRightControl] = KeyLocation::RIGHT;
		location_map[UIKeyboardHIDUsageKeyboardLeftShift] = KeyLocation::LEFT;
		location_map[UIKeyboardHIDUsageKeyboardRightShift] = KeyLocation::RIGHT;
		location_map[UIKeyboardHIDUsageKeyboardLeftGUI] = KeyLocation::LEFT;
		location_map[UIKeyboardHIDUsageKeyboardRightGUI] = KeyLocation::RIGHT;
	}
}

Key KeyMappingIOS::remap_key(CFIndex p_keycode) {
	if (@available(iOS 13.4, *)) {
		const Key *key = keyusage_map.getptr(p_keycode);
		if (key) {
			return *key;
		}
	}
	return Key::NONE;
}

KeyLocation KeyMappingIOS::key_location(CFIndex p_keycode) {
	if (@available(iOS 13.4, *)) {
		const KeyLocation *location = location_map.getptr(p_keycode);
		if (location) {
			return *location;
		}
	}
	return KeyLocation::UNSPECIFIED;
}
