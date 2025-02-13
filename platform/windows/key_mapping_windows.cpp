/**************************************************************************/
/*  key_mapping_windows.cpp                                               */
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

#include "key_mapping_windows.h"

#include "core/templates/hash_map.h"

// This provides translation from Windows virtual key codes to Godot and back.
// See WinUser.h and the below for documentation:
// https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes

struct HashMapHasherKeys {
	static _FORCE_INLINE_ uint32_t hash(const Key p_key) { return hash_fmix32(static_cast<uint32_t>(p_key)); }
	static _FORCE_INLINE_ uint32_t hash(const char32_t p_uchar) { return hash_fmix32(p_uchar); }
	static _FORCE_INLINE_ uint32_t hash(const unsigned p_key) { return hash_fmix32(p_key); }
};

HashMap<unsigned int, Key, HashMapHasherKeys> vk_map;
HashMap<unsigned int, Key, HashMapHasherKeys> scansym_map;
HashMap<Key, unsigned int, HashMapHasherKeys> scansym_map_inv;
HashMap<unsigned int, Key, HashMapHasherKeys> scansym_map_ext;
HashMap<unsigned int, KeyLocation, HashMapHasherKeys> location_map;

void KeyMappingWindows::initialize() {
	// VK_LBUTTON (0x01)
	// VK_RBUTTON (0x02)
	// VK_CANCEL (0x03)
	// VK_MBUTTON (0x04)
	// VK_XBUTTON1 (0x05)
	// VK_XBUTTON2 (0x06), We have no mappings for the above;as we only map keyboard buttons here.
	// 0x07 is undefined.
	vk_map[VK_BACK] = Key::BACKSPACE; // (0x08)
	vk_map[VK_TAB] = Key::TAB; // (0x09)
	// 0x0A-0B are reserved.
	vk_map[VK_CLEAR] = Key::CLEAR; // (0x0C)
	vk_map[VK_RETURN] = Key::ENTER; // (0x0D)
	// 0x0E-0F are undefined.
	vk_map[VK_SHIFT] = Key::SHIFT; // (0x10)
	vk_map[VK_CONTROL] = Key::CTRL; // (0x11)
	vk_map[VK_MENU] = Key::ALT; // (0x12)
	vk_map[VK_PAUSE] = Key::PAUSE; // (0x13)
	vk_map[VK_CAPITAL] = Key::CAPSLOCK; // (0x14)
	// 0x15-1A are IME keys.
	vk_map[VK_ESCAPE] = Key::ESCAPE; // (0x1B)
	// 0x1C-1F are IME keys.
	vk_map[VK_SPACE] = Key::SPACE; // (0x20)
	vk_map[VK_PRIOR] = Key::PAGEUP; // (0x21)
	vk_map[VK_NEXT] = Key::PAGEDOWN; // (0x22)
	vk_map[VK_END] = Key::END; // (0x23)
	vk_map[VK_HOME] = Key::HOME; // (0x24)
	vk_map[VK_LEFT] = Key::LEFT; // (0x25)
	vk_map[VK_UP] = Key::UP; // (0x26)
	vk_map[VK_RIGHT] = Key::RIGHT; // (0x27)
	vk_map[VK_DOWN] = Key::DOWN; // (0x28)
	// VK_SELECT (0x29), Old select key; e.g. on Digital Equipment Corporation keyboards.
	vk_map[VK_PRINT] = Key::PRINT; // (0x2A), Old IBM key; modern keyboards use VK_SNAPSHOT.
	// VK_EXECUTE (0x2B), Old and uncommon.
	vk_map[VK_SNAPSHOT] = Key::PRINT; // (0x2C)
	vk_map[VK_INSERT] = Key::INSERT; // (0x2D)
	vk_map[VK_DELETE] = Key::KEY_DELETE; // (0x2E)
	vk_map[VK_HELP] = Key::HELP; // (0x2F)
	vk_map[0x30] = Key::KEY_0; // 0 key.
	vk_map[0x31] = Key::KEY_1; // 1 key.
	vk_map[0x32] = Key::KEY_2; // 2 key.
	vk_map[0x33] = Key::KEY_3; // 3 key.
	vk_map[0x34] = Key::KEY_4; // 4 key.
	vk_map[0x35] = Key::KEY_5; // 5 key.
	vk_map[0x36] = Key::KEY_6; // 6 key.
	vk_map[0x37] = Key::KEY_7; // 7 key.
	vk_map[0x38] = Key::KEY_8; // 8 key.
	vk_map[0x39] = Key::KEY_9; // 9 key.
	// 0x3A-40 are undefined.
	vk_map[0x41] = Key::A; // A key.
	vk_map[0x42] = Key::B; // B key.
	vk_map[0x43] = Key::C; // C key.
	vk_map[0x44] = Key::D; // D key.
	vk_map[0x45] = Key::E; // E key.
	vk_map[0x46] = Key::F; // F key.
	vk_map[0x47] = Key::G; // G key.
	vk_map[0x48] = Key::H; // H key.
	vk_map[0x49] = Key::I; // I key
	vk_map[0x4A] = Key::J; // J key.
	vk_map[0x4B] = Key::K; // K key.
	vk_map[0x4C] = Key::L; // L key.
	vk_map[0x4D] = Key::M; // M key.
	vk_map[0x4E] = Key::N; // N key.
	vk_map[0x4F] = Key::O; // O key.
	vk_map[0x50] = Key::P; // P key.
	vk_map[0x51] = Key::Q; // Q key.
	vk_map[0x52] = Key::R; // R key.
	vk_map[0x53] = Key::S; // S key.
	vk_map[0x54] = Key::T; // T key.
	vk_map[0x55] = Key::U; // U key.
	vk_map[0x56] = Key::V; // V key.
	vk_map[0x57] = Key::W; // W key.
	vk_map[0x58] = Key::X; // X key.
	vk_map[0x59] = Key::Y; // Y key.
	vk_map[0x5A] = Key::Z; // Z key.
	vk_map[VK_LWIN] = (Key)Key::META; // (0x5B)
	vk_map[VK_RWIN] = (Key)Key::META; // (0x5C)
	vk_map[VK_APPS] = Key::MENU; // (0x5D)
	// 0x5E is reserved.
	vk_map[VK_SLEEP] = Key::STANDBY; // (0x5F)
	vk_map[VK_NUMPAD0] = Key::KP_0; // (0x60)
	vk_map[VK_NUMPAD1] = Key::KP_1; // (0x61)
	vk_map[VK_NUMPAD2] = Key::KP_2; // (0x62)
	vk_map[VK_NUMPAD3] = Key::KP_3; // (0x63)
	vk_map[VK_NUMPAD4] = Key::KP_4; // (0x64)
	vk_map[VK_NUMPAD5] = Key::KP_5; // (0x65)
	vk_map[VK_NUMPAD6] = Key::KP_6; // (0x66)
	vk_map[VK_NUMPAD7] = Key::KP_7; // (0x67)
	vk_map[VK_NUMPAD8] = Key::KP_8; // (0x68)
	vk_map[VK_NUMPAD9] = Key::KP_9; // (0x69)
	vk_map[VK_MULTIPLY] = Key::KP_MULTIPLY; // (0x6A)
	vk_map[VK_ADD] = Key::KP_ADD; // (0x6B)
	vk_map[VK_SEPARATOR] = Key::KP_PERIOD; // (0x6C)
	vk_map[VK_SUBTRACT] = Key::KP_SUBTRACT; // (0x6D)
	vk_map[VK_DECIMAL] = Key::KP_PERIOD; // (0x6E)
	vk_map[VK_DIVIDE] = Key::KP_DIVIDE; // (0x6F)
	vk_map[VK_F1] = Key::F1; // (0x70)
	vk_map[VK_F2] = Key::F2; // (0x71)
	vk_map[VK_F3] = Key::F3; // (0x72)
	vk_map[VK_F4] = Key::F4; // (0x73)
	vk_map[VK_F5] = Key::F5; // (0x74)
	vk_map[VK_F6] = Key::F6; // (0x75)
	vk_map[VK_F7] = Key::F7; // (0x76)
	vk_map[VK_F8] = Key::F8; // (0x77)
	vk_map[VK_F9] = Key::F9; // (0x78)
	vk_map[VK_F10] = Key::F10; // (0x79)
	vk_map[VK_F11] = Key::F11; // (0x7A)
	vk_map[VK_F12] = Key::F12; // (0x7B)
	vk_map[VK_F13] = Key::F13; // (0x7C)
	vk_map[VK_F14] = Key::F14; // (0x7D)
	vk_map[VK_F15] = Key::F15; // (0x7E)
	vk_map[VK_F16] = Key::F16; // (0x7F)
	vk_map[VK_F17] = Key::F17; // (0x80)
	vk_map[VK_F18] = Key::F18; // (0x81)
	vk_map[VK_F19] = Key::F19; // (0x82)
	vk_map[VK_F20] = Key::F20; // (0x83)
	vk_map[VK_F21] = Key::F21; // (0x84)
	vk_map[VK_F22] = Key::F22; // (0x85)
	vk_map[VK_F23] = Key::F23; // (0x86)
	vk_map[VK_F24] = Key::F24; // (0x87)
	// 0x88-8F are reserved for UI navigation.
	vk_map[VK_NUMLOCK] = Key::NUMLOCK; // (0x90)
	vk_map[VK_SCROLL] = Key::SCROLLLOCK; // (0x91)
	vk_map[VK_OEM_NEC_EQUAL] = Key::EQUAL; // (0x92), OEM NEC PC-9800 numpad '=' key.
	// 0x93-96 are OEM specific (e.g. used by Fujitsu/OASYS);
	// 0x97-9F are unassigned.
	vk_map[VK_LSHIFT] = Key::SHIFT; // (0xA0)
	vk_map[VK_RSHIFT] = Key::SHIFT; // (0xA1)
	vk_map[VK_LCONTROL] = Key::CTRL; // (0xA2)
	vk_map[VK_RCONTROL] = Key::CTRL; // (0xA3)
	vk_map[VK_LMENU] = Key::MENU; // (0xA4)
	vk_map[VK_RMENU] = Key::MENU; // (0xA5)
	vk_map[VK_BROWSER_BACK] = Key::BACK; // (0xA6)
	vk_map[VK_BROWSER_FORWARD] = Key::FORWARD; // (0xA7)
	vk_map[VK_BROWSER_REFRESH] = Key::REFRESH; // (0xA8)
	vk_map[VK_BROWSER_STOP] = Key::STOP; // (0xA9)
	vk_map[VK_BROWSER_SEARCH] = Key::SEARCH; // (0xAA)
	vk_map[VK_BROWSER_FAVORITES] = Key::FAVORITES; // (0xAB)
	vk_map[VK_BROWSER_HOME] = Key::HOMEPAGE; // (0xAC)
	vk_map[VK_VOLUME_MUTE] = Key::VOLUMEMUTE; // (0xAD)
	vk_map[VK_VOLUME_DOWN] = Key::VOLUMEDOWN; // (0xAE)
	vk_map[VK_VOLUME_UP] = Key::VOLUMEUP; // (0xAF)
	vk_map[VK_MEDIA_NEXT_TRACK] = Key::MEDIANEXT; // (0xB0)
	vk_map[VK_MEDIA_PREV_TRACK] = Key::MEDIAPREVIOUS; // (0xB1)
	vk_map[VK_MEDIA_STOP] = Key::MEDIASTOP; // (0xB2)
	vk_map[VK_MEDIA_PLAY_PAUSE] = Key::MEDIAPLAY; // (0xB3), Media button play/pause toggle.
	vk_map[VK_LAUNCH_MAIL] = Key::LAUNCHMAIL; // (0xB4)
	vk_map[VK_LAUNCH_MEDIA_SELECT] = Key::LAUNCHMEDIA; // (0xB5)
	vk_map[VK_LAUNCH_APP1] = Key::LAUNCH0; // (0xB6)
	vk_map[VK_LAUNCH_APP2] = Key::LAUNCH1; // (0xB7)
	// 0xB8-B9 are reserved.
	vk_map[VK_OEM_1] = Key::SEMICOLON; // (0xBA), Misc. character;can vary by keyboard/region. For US standard keyboards;the ';:' key.
	vk_map[VK_OEM_PLUS] = Key::EQUAL; // (0xBB)
	vk_map[VK_OEM_COMMA] = Key::COMMA; // (0xBC)
	vk_map[VK_OEM_MINUS] = Key::MINUS; // (0xBD)
	vk_map[VK_OEM_PERIOD] = Key::PERIOD; // (0xBE)
	vk_map[VK_OEM_2] = Key::SLASH; // (0xBF), For US standard keyboards;the '/?' key.
	vk_map[VK_OEM_3] = Key::QUOTELEFT; // (0xC0), For US standard keyboards;the '`~' key.
	// 0xC1-D7 are reserved. 0xD8-DA are unassigned.
	// 0xC3-DA may be used for old gamepads? Maybe we want to support this? See WinUser.h.
	vk_map[VK_OEM_4] = Key::BRACKETLEFT; // (0xDB),  For US standard keyboards;the '[{' key.
	vk_map[VK_OEM_5] = Key::BACKSLASH; // (0xDC), For US standard keyboards;the '\|' key.
	vk_map[VK_OEM_6] = Key::BRACKETRIGHT; // (0xDD), For US standard keyboards;the ']}' key.
	vk_map[VK_OEM_7] = Key::APOSTROPHE; // (0xDE), For US standard keyboards;single quote/double quote.
	// VK_OEM_8 (0xDF)
	// 0xE0 is reserved. 0xE1 is OEM specific.
	vk_map[VK_OEM_102] = Key::BAR; // (0xE2), Either angle bracket or backslash key on the RT 102-key keyboard.
	vk_map[VK_ICO_HELP] = Key::HELP; // (0xE3)
	// 0xE4 is OEM (e.g. ICO) specific.
	// VK_PROCESSKEY (0xE5), For IME.
	vk_map[VK_ICO_CLEAR] = Key::CLEAR; // (0xE6)
	// VK_PACKET (0xE7), Used to pass Unicode characters as if they were keystrokes.
	// 0xE8 is unassigned.
	// 0xE9-F5 are OEM (Nokia/Ericsson) specific.
	vk_map[VK_ATTN] = Key::ESCAPE; // (0xF6), Old IBM 'ATTN' key used on midrange computers ;e.g. AS/400.
	vk_map[VK_CRSEL] = Key::TAB; // (0xF7), Old IBM 3270 'CrSel' (cursor select) key; used to select data fields.
	// VK_EXSEL (0xF7), Old IBM 3270 extended selection key.
	// VK_EREOF (0xF8), Old IBM 3270 erase to end of field key.
	vk_map[VK_PLAY] = Key::MEDIAPLAY; // (0xFA), Old IBM 3270 'Play' key.
	// VK_ZOOM (0xFB), Old IBM 3290 'Zoom' key.
	// VK_NONAME (0xFC), Reserved.
	// VK_PA1 (0xFD), Old IBM 3270 PA1 key.
	vk_map[VK_OEM_CLEAR] = Key::CLEAR; // (0xFE), OEM specific clear key. Unclear how it differs from normal clear.

	scansym_map[0x00] = Key::PAUSE;
	scansym_map[0x01] = Key::ESCAPE;
	scansym_map[0x02] = Key::KEY_1;
	scansym_map[0x03] = Key::KEY_2;
	scansym_map[0x04] = Key::KEY_3;
	scansym_map[0x05] = Key::KEY_4;
	scansym_map[0x06] = Key::KEY_5;
	scansym_map[0x07] = Key::KEY_6;
	scansym_map[0x08] = Key::KEY_7;
	scansym_map[0x09] = Key::KEY_8;
	scansym_map[0x0A] = Key::KEY_9;
	scansym_map[0x0B] = Key::KEY_0;
	scansym_map[0x0C] = Key::MINUS;
	scansym_map[0x0D] = Key::EQUAL;
	scansym_map[0x0E] = Key::BACKSPACE;
	scansym_map[0x0F] = Key::TAB;
	scansym_map[0x10] = Key::Q;
	scansym_map[0x11] = Key::W;
	scansym_map[0x12] = Key::E;
	scansym_map[0x13] = Key::R;
	scansym_map[0x14] = Key::T;
	scansym_map[0x15] = Key::Y;
	scansym_map[0x16] = Key::U;
	scansym_map[0x17] = Key::I;
	scansym_map[0x18] = Key::O;
	scansym_map[0x19] = Key::P;
	scansym_map[0x1A] = Key::BRACKETLEFT;
	scansym_map[0x1B] = Key::BRACKETRIGHT;
	scansym_map[0x1C] = Key::ENTER;
	scansym_map[0x1D] = Key::CTRL;
	scansym_map[0x1E] = Key::A;
	scansym_map[0x1F] = Key::S;
	scansym_map[0x20] = Key::D;
	scansym_map[0x21] = Key::F;
	scansym_map[0x22] = Key::G;
	scansym_map[0x23] = Key::H;
	scansym_map[0x24] = Key::J;
	scansym_map[0x25] = Key::K;
	scansym_map[0x26] = Key::L;
	scansym_map[0x27] = Key::SEMICOLON;
	scansym_map[0x28] = Key::APOSTROPHE;
	scansym_map[0x29] = Key::QUOTELEFT;
	scansym_map[0x2A] = Key::SHIFT;
	scansym_map[0x2B] = Key::BACKSLASH;
	scansym_map[0x2C] = Key::Z;
	scansym_map[0x2D] = Key::X;
	scansym_map[0x2E] = Key::C;
	scansym_map[0x2F] = Key::V;
	scansym_map[0x30] = Key::B;
	scansym_map[0x31] = Key::N;
	scansym_map[0x32] = Key::M;
	scansym_map[0x33] = Key::COMMA;
	scansym_map[0x34] = Key::PERIOD;
	scansym_map[0x35] = Key::SLASH;
	scansym_map[0x36] = Key::SHIFT;
	scansym_map[0x37] = Key::KP_MULTIPLY;
	scansym_map[0x38] = Key::ALT;
	scansym_map[0x39] = Key::SPACE;
	scansym_map[0x3A] = Key::CAPSLOCK;
	scansym_map[0x3B] = Key::F1;
	scansym_map[0x3C] = Key::F2;
	scansym_map[0x3D] = Key::F3;
	scansym_map[0x3E] = Key::F4;
	scansym_map[0x3F] = Key::F5;
	scansym_map[0x40] = Key::F6;
	scansym_map[0x41] = Key::F7;
	scansym_map[0x42] = Key::F8;
	scansym_map[0x43] = Key::F9;
	scansym_map[0x44] = Key::F10;
	scansym_map[0x45] = Key::NUMLOCK;
	scansym_map[0x46] = Key::SCROLLLOCK;
	scansym_map[0x47] = Key::KP_7;
	scansym_map[0x48] = Key::KP_8;
	scansym_map[0x49] = Key::KP_9;
	scansym_map[0x4A] = Key::KP_SUBTRACT;
	scansym_map[0x4B] = Key::KP_4;
	scansym_map[0x4C] = Key::KP_5;
	scansym_map[0x4D] = Key::KP_6;
	scansym_map[0x4E] = Key::KP_ADD;
	scansym_map[0x4F] = Key::KP_1;
	scansym_map[0x50] = Key::KP_2;
	scansym_map[0x51] = Key::KP_3;
	scansym_map[0x52] = Key::KP_0;
	scansym_map[0x53] = Key::KP_PERIOD;
	scansym_map[0x56] = Key::SECTION;
	scansym_map[0x57] = Key::F11;
	scansym_map[0x58] = Key::F12;
	scansym_map[0x5B] = Key::META;
	scansym_map[0x5C] = Key::META;
	scansym_map[0x5D] = Key::MENU;
	scansym_map[0x64] = Key::F13;
	scansym_map[0x65] = Key::F14;
	scansym_map[0x66] = Key::F15;
	scansym_map[0x67] = Key::F16;
	scansym_map[0x68] = Key::F17;
	scansym_map[0x69] = Key::F18;
	scansym_map[0x6A] = Key::F19;
	scansym_map[0x6B] = Key::F20;
	scansym_map[0x6C] = Key::F21;
	scansym_map[0x6D] = Key::F22;
	scansym_map[0x6E] = Key::F23;
	//	scansym_map[0x71] = Key::JIS_KANA;
	//	scansym_map[0x72] = Key::JIS_EISU;
	scansym_map[0x76] = Key::F24;

	for (const KeyValue<unsigned int, Key> &E : scansym_map) {
		scansym_map_inv[E.value] = E.key;
	}

	scansym_map_ext[0x09] = Key::MENU;
	scansym_map_ext[0x10] = Key::MEDIAPREVIOUS;
	scansym_map_ext[0x19] = Key::MEDIANEXT;
	scansym_map_ext[0x1C] = Key::KP_ENTER;
	scansym_map_ext[0x20] = Key::VOLUMEMUTE;
	scansym_map_ext[0x21] = Key::LAUNCH1;
	scansym_map_ext[0x22] = Key::MEDIAPLAY;
	scansym_map_ext[0x24] = Key::MEDIASTOP;
	scansym_map_ext[0x2E] = Key::VOLUMEDOWN;
	scansym_map_ext[0x30] = Key::VOLUMEUP;
	scansym_map_ext[0x32] = Key::HOMEPAGE;
	scansym_map_ext[0x35] = Key::KP_DIVIDE;
	scansym_map_ext[0x37] = Key::PRINT;
	scansym_map_ext[0x3A] = Key::KP_ADD;
	scansym_map_ext[0x45] = Key::NUMLOCK;
	scansym_map_ext[0x47] = Key::HOME;
	scansym_map_ext[0x48] = Key::UP;
	scansym_map_ext[0x49] = Key::PAGEUP;
	scansym_map_ext[0x4A] = Key::KP_SUBTRACT;
	scansym_map_ext[0x4B] = Key::LEFT;
	scansym_map_ext[0x4C] = Key::KP_5;
	scansym_map_ext[0x4D] = Key::RIGHT;
	scansym_map_ext[0x4E] = Key::KP_ADD;
	scansym_map_ext[0x4F] = Key::END;
	scansym_map_ext[0x50] = Key::DOWN;
	scansym_map_ext[0x51] = Key::PAGEDOWN;
	scansym_map_ext[0x52] = Key::INSERT;
	scansym_map_ext[0x53] = Key::KEY_DELETE;
	scansym_map_ext[0x5D] = Key::MENU;
	scansym_map_ext[0x5F] = Key::STANDBY;
	scansym_map_ext[0x65] = Key::SEARCH;
	scansym_map_ext[0x66] = Key::FAVORITES;
	scansym_map_ext[0x67] = Key::REFRESH;
	scansym_map_ext[0x68] = Key::STOP;
	scansym_map_ext[0x69] = Key::FORWARD;
	scansym_map_ext[0x6A] = Key::BACK;
	scansym_map_ext[0x6B] = Key::LAUNCH0;
	scansym_map_ext[0x6C] = Key::LAUNCHMAIL;
	scansym_map_ext[0x6D] = Key::LAUNCHMEDIA;
	scansym_map_ext[0x78] = Key::MEDIARECORD;

	// Scancode to physical location map.
	// Shift.
	location_map[0x2A] = KeyLocation::LEFT;
	location_map[0x36] = KeyLocation::RIGHT;
	// Meta.
	location_map[0x5B] = KeyLocation::LEFT;
	location_map[0x5C] = KeyLocation::RIGHT;
	// Ctrl and Alt must be handled differently.
}

Key KeyMappingWindows::get_keysym(unsigned int p_code) {
	const Key *key = vk_map.getptr(p_code);
	if (key) {
		return *key;
	}
	return Key::UNKNOWN;
}

unsigned int KeyMappingWindows::get_scancode(Key p_keycode) {
	const unsigned int *key = scansym_map_inv.getptr(p_keycode);
	if (key) {
		return *key;
	}
	return 0;
}

Key KeyMappingWindows::get_scansym(unsigned int p_code, bool p_extended) {
	if (p_extended) {
		const Key *key = scansym_map_ext.getptr(p_code);
		if (key) {
			return *key;
		}
	}
	const Key *key = scansym_map.getptr(p_code);
	if (key) {
		return *key;
	}
	return Key::NONE;
}

bool KeyMappingWindows::is_extended_key(unsigned int p_code) {
	return p_code == VK_INSERT ||
			p_code == VK_DELETE ||
			p_code == VK_HOME ||
			p_code == VK_END ||
			p_code == VK_PRIOR ||
			p_code == VK_NEXT ||
			p_code == VK_LEFT ||
			p_code == VK_UP ||
			p_code == VK_RIGHT ||
			p_code == VK_DOWN;
}

KeyLocation KeyMappingWindows::get_location(unsigned int p_code, bool p_extended) {
	// Right- ctrl and alt have the same scancode as left, but are in the extended keys.
	const Key *key = scansym_map.getptr(p_code);
	if (key && (*key == Key::CTRL || *key == Key::ALT)) {
		return p_extended ? KeyLocation::RIGHT : KeyLocation::LEFT;
	}
	const KeyLocation *location = location_map.getptr(p_code);
	if (location) {
		return *location;
	}
	return KeyLocation::UNSPECIFIED;
}
