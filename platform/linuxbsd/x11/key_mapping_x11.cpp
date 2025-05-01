/**************************************************************************/
/*  key_mapping_x11.cpp                                                   */
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

#include "key_mapping_x11.h"

void KeyMappingX11::initialize() {
	// X11 Keysym to Godot Key map.

	xkeysym_map[XK_Escape] = Key::ESCAPE;
	xkeysym_map[XK_Tab] = Key::TAB;
	xkeysym_map[XK_ISO_Left_Tab] = Key::BACKTAB;
	xkeysym_map[XK_BackSpace] = Key::BACKSPACE;
	xkeysym_map[XK_Return] = Key::ENTER;
	xkeysym_map[XK_Insert] = Key::INSERT;
	xkeysym_map[XK_Delete] = Key::KEY_DELETE;
	xkeysym_map[XK_Clear] = Key::KEY_DELETE;
	xkeysym_map[XK_Pause] = Key::PAUSE;
	xkeysym_map[XK_Print] = Key::PRINT;
	xkeysym_map[XK_Home] = Key::HOME;
	xkeysym_map[XK_End] = Key::END;
	xkeysym_map[XK_Left] = Key::LEFT;
	xkeysym_map[XK_Up] = Key::UP;
	xkeysym_map[XK_Right] = Key::RIGHT;
	xkeysym_map[XK_Down] = Key::DOWN;
	xkeysym_map[XK_Prior] = Key::PAGEUP;
	xkeysym_map[XK_Next] = Key::PAGEDOWN;
	xkeysym_map[XK_Shift_L] = Key::SHIFT;
	xkeysym_map[XK_Shift_R] = Key::SHIFT;
	xkeysym_map[XK_Shift_Lock] = Key::SHIFT;
	xkeysym_map[XK_Control_L] = Key::CTRL;
	xkeysym_map[XK_Control_R] = Key::CTRL;
	xkeysym_map[XK_Meta_L] = Key::META;
	xkeysym_map[XK_Meta_R] = Key::META;
	xkeysym_map[XK_Alt_L] = Key::ALT;
	xkeysym_map[XK_Alt_R] = Key::ALT;
	xkeysym_map[XK_Caps_Lock] = Key::CAPSLOCK;
	xkeysym_map[XK_Num_Lock] = Key::NUMLOCK;
	xkeysym_map[XK_Scroll_Lock] = Key::SCROLLLOCK;
	xkeysym_map[XK_less] = Key::QUOTELEFT;
	xkeysym_map[XK_grave] = Key::SECTION;
	xkeysym_map[XK_Super_L] = Key::META;
	xkeysym_map[XK_Super_R] = Key::META;
	xkeysym_map[XK_Menu] = Key::MENU;
	xkeysym_map[XK_Hyper_L] = Key::HYPER;
	xkeysym_map[XK_Hyper_R] = Key::HYPER;
	xkeysym_map[XK_Help] = Key::HELP;
	xkeysym_map[XK_KP_Space] = Key::SPACE;
	xkeysym_map[XK_KP_Tab] = Key::TAB;
	xkeysym_map[XK_KP_Enter] = Key::KP_ENTER;
	xkeysym_map[XK_Home] = Key::HOME;
	xkeysym_map[XK_Left] = Key::LEFT;
	xkeysym_map[XK_Up] = Key::UP;
	xkeysym_map[XK_Right] = Key::RIGHT;
	xkeysym_map[XK_Down] = Key::DOWN;
	xkeysym_map[XK_Prior] = Key::PAGEUP;
	xkeysym_map[XK_Next] = Key::PAGEDOWN;
	xkeysym_map[XK_End] = Key::END;
	xkeysym_map[XK_Begin] = Key::CLEAR;
	xkeysym_map[XK_Insert] = Key::INSERT;
	xkeysym_map[XK_Delete] = Key::KEY_DELETE;
	xkeysym_map[XK_KP_Equal] = Key::EQUAL;
	xkeysym_map[XK_KP_Separator] = Key::COMMA;
	xkeysym_map[XK_KP_Decimal] = Key::KP_PERIOD;
	xkeysym_map[XK_KP_Multiply] = Key::KP_MULTIPLY;
	xkeysym_map[XK_KP_Divide] = Key::KP_DIVIDE;
	xkeysym_map[XK_KP_Subtract] = Key::KP_SUBTRACT;
	xkeysym_map[XK_KP_Add] = Key::KP_ADD;
	xkeysym_map[XK_KP_0] = Key::KP_0;
	xkeysym_map[XK_KP_1] = Key::KP_1;
	xkeysym_map[XK_KP_2] = Key::KP_2;
	xkeysym_map[XK_KP_3] = Key::KP_3;
	xkeysym_map[XK_KP_4] = Key::KP_4;
	xkeysym_map[XK_KP_5] = Key::KP_5;
	xkeysym_map[XK_KP_6] = Key::KP_6;
	xkeysym_map[XK_KP_7] = Key::KP_7;
	xkeysym_map[XK_KP_8] = Key::KP_8;
	xkeysym_map[XK_KP_9] = Key::KP_9;
	// Same keys but with numlock off.
	xkeysym_map[XK_KP_Insert] = Key::INSERT;
	xkeysym_map[XK_KP_Delete] = Key::KEY_DELETE;
	xkeysym_map[XK_KP_End] = Key::END;
	xkeysym_map[XK_KP_Down] = Key::DOWN;
	xkeysym_map[XK_KP_Page_Down] = Key::PAGEDOWN;
	xkeysym_map[XK_KP_Left] = Key::LEFT;
	// X11 documents this (numpad 5) as "begin of line" but no toolkit seems to interpret it this way.
	// On Windows this is emitting Key::Clear so for consistency it will be mapped to Key::Clear
	xkeysym_map[XK_KP_Begin] = Key::CLEAR;
	xkeysym_map[XK_KP_Right] = Key::RIGHT;
	xkeysym_map[XK_KP_Home] = Key::HOME;
	xkeysym_map[XK_KP_Up] = Key::UP;
	xkeysym_map[XK_KP_Page_Up] = Key::PAGEUP;
	xkeysym_map[XK_F1] = Key::F1;
	xkeysym_map[XK_F2] = Key::F2;
	xkeysym_map[XK_F3] = Key::F3;
	xkeysym_map[XK_F4] = Key::F4;
	xkeysym_map[XK_F5] = Key::F5;
	xkeysym_map[XK_F6] = Key::F6;
	xkeysym_map[XK_F7] = Key::F7;
	xkeysym_map[XK_F8] = Key::F8;
	xkeysym_map[XK_F9] = Key::F9;
	xkeysym_map[XK_F10] = Key::F10;
	xkeysym_map[XK_F11] = Key::F11;
	xkeysym_map[XK_F12] = Key::F12;
	xkeysym_map[XK_F13] = Key::F13;
	xkeysym_map[XK_F14] = Key::F14;
	xkeysym_map[XK_F15] = Key::F15;
	xkeysym_map[XK_F16] = Key::F16;
	xkeysym_map[XK_F17] = Key::F17;
	xkeysym_map[XK_F18] = Key::F18;
	xkeysym_map[XK_F19] = Key::F19;
	xkeysym_map[XK_F20] = Key::F20;
	xkeysym_map[XK_F21] = Key::F21;
	xkeysym_map[XK_F22] = Key::F22;
	xkeysym_map[XK_F23] = Key::F23;
	xkeysym_map[XK_F24] = Key::F24;
	xkeysym_map[XK_F25] = Key::F25;
	xkeysym_map[XK_F26] = Key::F26;
	xkeysym_map[XK_F27] = Key::F27;
	xkeysym_map[XK_F28] = Key::F28;
	xkeysym_map[XK_F29] = Key::F29;
	xkeysym_map[XK_F30] = Key::F30;
	xkeysym_map[XK_F31] = Key::F31;
	xkeysym_map[XK_F32] = Key::F32;
	xkeysym_map[XK_F33] = Key::F33;
	xkeysym_map[XK_F34] = Key::F34;
	xkeysym_map[XK_F35] = Key::F35;
	xkeysym_map[XK_yen] = Key::YEN;
	xkeysym_map[XK_section] = Key::SECTION;
	// Media keys.
	xkeysym_map[XF86XK_Back] = Key::BACK;
	xkeysym_map[XF86XK_Forward] = Key::FORWARD;
	xkeysym_map[XF86XK_Stop] = Key::STOP;
	xkeysym_map[XF86XK_Refresh] = Key::REFRESH;
	xkeysym_map[XF86XK_Favorites] = Key::FAVORITES;
	xkeysym_map[XF86XK_OpenURL] = Key::OPENURL;
	xkeysym_map[XF86XK_HomePage] = Key::HOMEPAGE;
	xkeysym_map[XF86XK_Search] = Key::SEARCH;
	xkeysym_map[XF86XK_AudioLowerVolume] = Key::VOLUMEDOWN;
	xkeysym_map[XF86XK_AudioMute] = Key::VOLUMEMUTE;
	xkeysym_map[XF86XK_AudioRaiseVolume] = Key::VOLUMEUP;
	xkeysym_map[XF86XK_AudioPlay] = Key::MEDIAPLAY;
	xkeysym_map[XF86XK_AudioStop] = Key::MEDIASTOP;
	xkeysym_map[XF86XK_AudioPrev] = Key::MEDIAPREVIOUS;
	xkeysym_map[XF86XK_AudioNext] = Key::MEDIANEXT;
	xkeysym_map[XF86XK_AudioRecord] = Key::MEDIARECORD;
	xkeysym_map[XF86XK_Standby] = Key::STANDBY;
	// Launch keys.
	xkeysym_map[XF86XK_Mail] = Key::LAUNCHMAIL;
	xkeysym_map[XF86XK_AudioMedia] = Key::LAUNCHMEDIA;
	xkeysym_map[XF86XK_MyComputer] = Key::LAUNCH0;
	xkeysym_map[XF86XK_Calculator] = Key::LAUNCH1;
	xkeysym_map[XF86XK_Launch0] = Key::LAUNCH2;
	xkeysym_map[XF86XK_Launch1] = Key::LAUNCH3;
	xkeysym_map[XF86XK_Launch2] = Key::LAUNCH4;
	xkeysym_map[XF86XK_Launch3] = Key::LAUNCH5;
	xkeysym_map[XF86XK_Launch4] = Key::LAUNCH6;
	xkeysym_map[XF86XK_Launch5] = Key::LAUNCH7;
	xkeysym_map[XF86XK_Launch6] = Key::LAUNCH8;
	xkeysym_map[XF86XK_Launch7] = Key::LAUNCH9;
	xkeysym_map[XF86XK_Launch8] = Key::LAUNCHA;
	xkeysym_map[XF86XK_Launch9] = Key::LAUNCHB;
	xkeysym_map[XF86XK_LaunchA] = Key::LAUNCHC;
	xkeysym_map[XF86XK_LaunchB] = Key::LAUNCHD;
	xkeysym_map[XF86XK_LaunchC] = Key::LAUNCHE;
	xkeysym_map[XF86XK_LaunchD] = Key::LAUNCHF;

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
	scancode_map[0x22] = Key::BRACKETLEFT;
	scancode_map[0x23] = Key::BRACKETRIGHT;
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
	scancode_map[0x31] = Key::QUOTELEFT;
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
	scancode_map[0x5E] = Key::SECTION;
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

	// Keysym to Unicode map, tables taken from FOX toolkit.
	xkeysym_unicode_map[0x01A1] = 0x0104;
	xkeysym_unicode_map[0x01A2] = 0x02D8;
	xkeysym_unicode_map[0x01A3] = 0x0141;
	xkeysym_unicode_map[0x01A5] = 0x013D;
	xkeysym_unicode_map[0x01A6] = 0x015A;
	xkeysym_unicode_map[0x01A9] = 0x0160;
	xkeysym_unicode_map[0x01AA] = 0x015E;
	xkeysym_unicode_map[0x01AB] = 0x0164;
	xkeysym_unicode_map[0x01AC] = 0x0179;
	xkeysym_unicode_map[0x01AE] = 0x017D;
	xkeysym_unicode_map[0x01AF] = 0x017B;
	xkeysym_unicode_map[0x01B1] = 0x0105;
	xkeysym_unicode_map[0x01B2] = 0x02DB;
	xkeysym_unicode_map[0x01B3] = 0x0142;
	xkeysym_unicode_map[0x01B5] = 0x013E;
	xkeysym_unicode_map[0x01B6] = 0x015B;
	xkeysym_unicode_map[0x01B7] = 0x02C7;
	xkeysym_unicode_map[0x01B9] = 0x0161;
	xkeysym_unicode_map[0x01BA] = 0x015F;
	xkeysym_unicode_map[0x01BB] = 0x0165;
	xkeysym_unicode_map[0x01BC] = 0x017A;
	xkeysym_unicode_map[0x01BD] = 0x02DD;
	xkeysym_unicode_map[0x01BE] = 0x017E;
	xkeysym_unicode_map[0x01BF] = 0x017C;
	xkeysym_unicode_map[0x01C0] = 0x0154;
	xkeysym_unicode_map[0x01C3] = 0x0102;
	xkeysym_unicode_map[0x01C5] = 0x0139;
	xkeysym_unicode_map[0x01C6] = 0x0106;
	xkeysym_unicode_map[0x01C8] = 0x010C;
	xkeysym_unicode_map[0x01CA] = 0x0118;
	xkeysym_unicode_map[0x01CC] = 0x011A;
	xkeysym_unicode_map[0x01CF] = 0x010E;
	xkeysym_unicode_map[0x01D0] = 0x0110;
	xkeysym_unicode_map[0x01D1] = 0x0143;
	xkeysym_unicode_map[0x01D2] = 0x0147;
	xkeysym_unicode_map[0x01D5] = 0x0150;
	xkeysym_unicode_map[0x01D8] = 0x0158;
	xkeysym_unicode_map[0x01D9] = 0x016E;
	xkeysym_unicode_map[0x01DB] = 0x0170;
	xkeysym_unicode_map[0x01DE] = 0x0162;
	xkeysym_unicode_map[0x01E0] = 0x0155;
	xkeysym_unicode_map[0x01E3] = 0x0103;
	xkeysym_unicode_map[0x01E5] = 0x013A;
	xkeysym_unicode_map[0x01E6] = 0x0107;
	xkeysym_unicode_map[0x01E8] = 0x010D;
	xkeysym_unicode_map[0x01EA] = 0x0119;
	xkeysym_unicode_map[0x01EC] = 0x011B;
	xkeysym_unicode_map[0x01EF] = 0x010F;
	xkeysym_unicode_map[0x01F0] = 0x0111;
	xkeysym_unicode_map[0x01F1] = 0x0144;
	xkeysym_unicode_map[0x01F2] = 0x0148;
	xkeysym_unicode_map[0x01F5] = 0x0151;
	xkeysym_unicode_map[0x01F8] = 0x0159;
	xkeysym_unicode_map[0x01F9] = 0x016F;
	xkeysym_unicode_map[0x01FB] = 0x0171;
	xkeysym_unicode_map[0x01FE] = 0x0163;
	xkeysym_unicode_map[0x01FF] = 0x02D9;
	xkeysym_unicode_map[0x02A1] = 0x0126;
	xkeysym_unicode_map[0x02A6] = 0x0124;
	xkeysym_unicode_map[0x02A9] = 0x0130;
	xkeysym_unicode_map[0x02AB] = 0x011E;
	xkeysym_unicode_map[0x02AC] = 0x0134;
	xkeysym_unicode_map[0x02B1] = 0x0127;
	xkeysym_unicode_map[0x02B6] = 0x0125;
	xkeysym_unicode_map[0x02B9] = 0x0131;
	xkeysym_unicode_map[0x02BB] = 0x011F;
	xkeysym_unicode_map[0x02BC] = 0x0135;
	xkeysym_unicode_map[0x02C5] = 0x010A;
	xkeysym_unicode_map[0x02C6] = 0x0108;
	xkeysym_unicode_map[0x02D5] = 0x0120;
	xkeysym_unicode_map[0x02D8] = 0x011C;
	xkeysym_unicode_map[0x02DD] = 0x016C;
	xkeysym_unicode_map[0x02DE] = 0x015C;
	xkeysym_unicode_map[0x02E5] = 0x010B;
	xkeysym_unicode_map[0x02E6] = 0x0109;
	xkeysym_unicode_map[0x02F5] = 0x0121;
	xkeysym_unicode_map[0x02F8] = 0x011D;
	xkeysym_unicode_map[0x02FD] = 0x016D;
	xkeysym_unicode_map[0x02FE] = 0x015D;
	xkeysym_unicode_map[0x03A2] = 0x0138;
	xkeysym_unicode_map[0x03A3] = 0x0156;
	xkeysym_unicode_map[0x03A5] = 0x0128;
	xkeysym_unicode_map[0x03A6] = 0x013B;
	xkeysym_unicode_map[0x03AA] = 0x0112;
	xkeysym_unicode_map[0x03AB] = 0x0122;
	xkeysym_unicode_map[0x03AC] = 0x0166;
	xkeysym_unicode_map[0x03B3] = 0x0157;
	xkeysym_unicode_map[0x03B5] = 0x0129;
	xkeysym_unicode_map[0x03B6] = 0x013C;
	xkeysym_unicode_map[0x03BA] = 0x0113;
	xkeysym_unicode_map[0x03BB] = 0x0123;
	xkeysym_unicode_map[0x03BC] = 0x0167;
	xkeysym_unicode_map[0x03BD] = 0x014A;
	xkeysym_unicode_map[0x03BF] = 0x014B;
	xkeysym_unicode_map[0x03C0] = 0x0100;
	xkeysym_unicode_map[0x03C7] = 0x012E;
	xkeysym_unicode_map[0x03CC] = 0x0116;
	xkeysym_unicode_map[0x03CF] = 0x012A;
	xkeysym_unicode_map[0x03D1] = 0x0145;
	xkeysym_unicode_map[0x03D2] = 0x014C;
	xkeysym_unicode_map[0x03D3] = 0x0136;
	xkeysym_unicode_map[0x03D9] = 0x0172;
	xkeysym_unicode_map[0x03DD] = 0x0168;
	xkeysym_unicode_map[0x03DE] = 0x016A;
	xkeysym_unicode_map[0x03E0] = 0x0101;
	xkeysym_unicode_map[0x03E7] = 0x012F;
	xkeysym_unicode_map[0x03EC] = 0x0117;
	xkeysym_unicode_map[0x03EF] = 0x012B;
	xkeysym_unicode_map[0x03F1] = 0x0146;
	xkeysym_unicode_map[0x03F2] = 0x014D;
	xkeysym_unicode_map[0x03F3] = 0x0137;
	xkeysym_unicode_map[0x03F9] = 0x0173;
	xkeysym_unicode_map[0x03FD] = 0x0169;
	xkeysym_unicode_map[0x03FE] = 0x016B;
	xkeysym_unicode_map[0x047E] = 0x203E;
	xkeysym_unicode_map[0x04A1] = 0x3002;
	xkeysym_unicode_map[0x04A2] = 0x300C;
	xkeysym_unicode_map[0x04A3] = 0x300D;
	xkeysym_unicode_map[0x04A4] = 0x3001;
	xkeysym_unicode_map[0x04A5] = 0x30FB;
	xkeysym_unicode_map[0x04A6] = 0x30F2;
	xkeysym_unicode_map[0x04A7] = 0x30A1;
	xkeysym_unicode_map[0x04A8] = 0x30A3;
	xkeysym_unicode_map[0x04A9] = 0x30A5;
	xkeysym_unicode_map[0x04AA] = 0x30A7;
	xkeysym_unicode_map[0x04AB] = 0x30A9;
	xkeysym_unicode_map[0x04AC] = 0x30E3;
	xkeysym_unicode_map[0x04AD] = 0x30E5;
	xkeysym_unicode_map[0x04AE] = 0x30E7;
	xkeysym_unicode_map[0x04AF] = 0x30C3;
	xkeysym_unicode_map[0x04B0] = 0x30FC;
	xkeysym_unicode_map[0x04B1] = 0x30A2;
	xkeysym_unicode_map[0x04B2] = 0x30A4;
	xkeysym_unicode_map[0x04B3] = 0x30A6;
	xkeysym_unicode_map[0x04B4] = 0x30A8;
	xkeysym_unicode_map[0x04B5] = 0x30AA;
	xkeysym_unicode_map[0x04B6] = 0x30AB;
	xkeysym_unicode_map[0x04B7] = 0x30AD;
	xkeysym_unicode_map[0x04B8] = 0x30AF;
	xkeysym_unicode_map[0x04B9] = 0x30B1;
	xkeysym_unicode_map[0x04BA] = 0x30B3;
	xkeysym_unicode_map[0x04BB] = 0x30B5;
	xkeysym_unicode_map[0x04BC] = 0x30B7;
	xkeysym_unicode_map[0x04BD] = 0x30B9;
	xkeysym_unicode_map[0x04BE] = 0x30BB;
	xkeysym_unicode_map[0x04BF] = 0x30BD;
	xkeysym_unicode_map[0x04C0] = 0x30BF;
	xkeysym_unicode_map[0x04C1] = 0x30C1;
	xkeysym_unicode_map[0x04C2] = 0x30C4;
	xkeysym_unicode_map[0x04C3] = 0x30C6;
	xkeysym_unicode_map[0x04C4] = 0x30C8;
	xkeysym_unicode_map[0x04C5] = 0x30CA;
	xkeysym_unicode_map[0x04C6] = 0x30CB;
	xkeysym_unicode_map[0x04C7] = 0x30CC;
	xkeysym_unicode_map[0x04C8] = 0x30CD;
	xkeysym_unicode_map[0x04C9] = 0x30CE;
	xkeysym_unicode_map[0x04CA] = 0x30CF;
	xkeysym_unicode_map[0x04CB] = 0x30D2;
	xkeysym_unicode_map[0x04CC] = 0x30D5;
	xkeysym_unicode_map[0x04CD] = 0x30D8;
	xkeysym_unicode_map[0x04CE] = 0x30DB;
	xkeysym_unicode_map[0x04CF] = 0x30DE;
	xkeysym_unicode_map[0x04D0] = 0x30DF;
	xkeysym_unicode_map[0x04D1] = 0x30E0;
	xkeysym_unicode_map[0x04D2] = 0x30E1;
	xkeysym_unicode_map[0x04D3] = 0x30E2;
	xkeysym_unicode_map[0x04D4] = 0x30E4;
	xkeysym_unicode_map[0x04D5] = 0x30E6;
	xkeysym_unicode_map[0x04D6] = 0x30E8;
	xkeysym_unicode_map[0x04D7] = 0x30E9;
	xkeysym_unicode_map[0x04D8] = 0x30EA;
	xkeysym_unicode_map[0x04D9] = 0x30EB;
	xkeysym_unicode_map[0x04DA] = 0x30EC;
	xkeysym_unicode_map[0x04DB] = 0x30ED;
	xkeysym_unicode_map[0x04DC] = 0x30EF;
	xkeysym_unicode_map[0x04DD] = 0x30F3;
	xkeysym_unicode_map[0x04DE] = 0x309B;
	xkeysym_unicode_map[0x04DF] = 0x309C;
	xkeysym_unicode_map[0x05AC] = 0x060C;
	xkeysym_unicode_map[0x05BB] = 0x061B;
	xkeysym_unicode_map[0x05BF] = 0x061F;
	xkeysym_unicode_map[0x05C1] = 0x0621;
	xkeysym_unicode_map[0x05C2] = 0x0622;
	xkeysym_unicode_map[0x05C3] = 0x0623;
	xkeysym_unicode_map[0x05C4] = 0x0624;
	xkeysym_unicode_map[0x05C5] = 0x0625;
	xkeysym_unicode_map[0x05C6] = 0x0626;
	xkeysym_unicode_map[0x05C7] = 0x0627;
	xkeysym_unicode_map[0x05C8] = 0x0628;
	xkeysym_unicode_map[0x05C9] = 0x0629;
	xkeysym_unicode_map[0x05CA] = 0x062A;
	xkeysym_unicode_map[0x05CB] = 0x062B;
	xkeysym_unicode_map[0x05CC] = 0x062C;
	xkeysym_unicode_map[0x05CD] = 0x062D;
	xkeysym_unicode_map[0x05CE] = 0x062E;
	xkeysym_unicode_map[0x05CF] = 0x062F;
	xkeysym_unicode_map[0x05D0] = 0x0630;
	xkeysym_unicode_map[0x05D1] = 0x0631;
	xkeysym_unicode_map[0x05D2] = 0x0632;
	xkeysym_unicode_map[0x05D3] = 0x0633;
	xkeysym_unicode_map[0x05D4] = 0x0634;
	xkeysym_unicode_map[0x05D5] = 0x0635;
	xkeysym_unicode_map[0x05D6] = 0x0636;
	xkeysym_unicode_map[0x05D7] = 0x0637;
	xkeysym_unicode_map[0x05D8] = 0x0638;
	xkeysym_unicode_map[0x05D9] = 0x0639;
	xkeysym_unicode_map[0x05DA] = 0x063A;
	xkeysym_unicode_map[0x05E0] = 0x0640;
	xkeysym_unicode_map[0x05E1] = 0x0641;
	xkeysym_unicode_map[0x05E2] = 0x0642;
	xkeysym_unicode_map[0x05E3] = 0x0643;
	xkeysym_unicode_map[0x05E4] = 0x0644;
	xkeysym_unicode_map[0x05E5] = 0x0645;
	xkeysym_unicode_map[0x05E6] = 0x0646;
	xkeysym_unicode_map[0x05E7] = 0x0647;
	xkeysym_unicode_map[0x05E8] = 0x0648;
	xkeysym_unicode_map[0x05E9] = 0x0649;
	xkeysym_unicode_map[0x05EA] = 0x064A;
	xkeysym_unicode_map[0x05EB] = 0x064B;
	xkeysym_unicode_map[0x05EC] = 0x064C;
	xkeysym_unicode_map[0x05ED] = 0x064D;
	xkeysym_unicode_map[0x05EE] = 0x064E;
	xkeysym_unicode_map[0x05EF] = 0x064F;
	xkeysym_unicode_map[0x05F0] = 0x0650;
	xkeysym_unicode_map[0x05F1] = 0x0651;
	xkeysym_unicode_map[0x05F2] = 0x0652;
	xkeysym_unicode_map[0x06A1] = 0x0452;
	xkeysym_unicode_map[0x06A2] = 0x0453;
	xkeysym_unicode_map[0x06A3] = 0x0451;
	xkeysym_unicode_map[0x06A4] = 0x0454;
	xkeysym_unicode_map[0x06A5] = 0x0455;
	xkeysym_unicode_map[0x06A6] = 0x0456;
	xkeysym_unicode_map[0x06A7] = 0x0457;
	xkeysym_unicode_map[0x06A8] = 0x0458;
	xkeysym_unicode_map[0x06A9] = 0x0459;
	xkeysym_unicode_map[0x06AA] = 0x045A;
	xkeysym_unicode_map[0x06AB] = 0x045B;
	xkeysym_unicode_map[0x06AC] = 0x045C;
	xkeysym_unicode_map[0x06AE] = 0x045E;
	xkeysym_unicode_map[0x06AF] = 0x045F;
	xkeysym_unicode_map[0x06B0] = 0x2116;
	xkeysym_unicode_map[0x06B1] = 0x0402;
	xkeysym_unicode_map[0x06B2] = 0x0403;
	xkeysym_unicode_map[0x06B3] = 0x0401;
	xkeysym_unicode_map[0x06B4] = 0x0404;
	xkeysym_unicode_map[0x06B5] = 0x0405;
	xkeysym_unicode_map[0x06B6] = 0x0406;
	xkeysym_unicode_map[0x06B7] = 0x0407;
	xkeysym_unicode_map[0x06B8] = 0x0408;
	xkeysym_unicode_map[0x06B9] = 0x0409;
	xkeysym_unicode_map[0x06BA] = 0x040A;
	xkeysym_unicode_map[0x06BB] = 0x040B;
	xkeysym_unicode_map[0x06BC] = 0x040C;
	xkeysym_unicode_map[0x06BE] = 0x040E;
	xkeysym_unicode_map[0x06BF] = 0x040F;
	xkeysym_unicode_map[0x06C0] = 0x044E;
	xkeysym_unicode_map[0x06C1] = 0x0430;
	xkeysym_unicode_map[0x06C2] = 0x0431;
	xkeysym_unicode_map[0x06C3] = 0x0446;
	xkeysym_unicode_map[0x06C4] = 0x0434;
	xkeysym_unicode_map[0x06C5] = 0x0435;
	xkeysym_unicode_map[0x06C6] = 0x0444;
	xkeysym_unicode_map[0x06C7] = 0x0433;
	xkeysym_unicode_map[0x06C8] = 0x0445;
	xkeysym_unicode_map[0x06C9] = 0x0438;
	xkeysym_unicode_map[0x06CA] = 0x0439;
	xkeysym_unicode_map[0x06CB] = 0x043A;
	xkeysym_unicode_map[0x06CC] = 0x043B;
	xkeysym_unicode_map[0x06CD] = 0x043C;
	xkeysym_unicode_map[0x06CE] = 0x043D;
	xkeysym_unicode_map[0x06CF] = 0x043E;
	xkeysym_unicode_map[0x06D0] = 0x043F;
	xkeysym_unicode_map[0x06D1] = 0x044F;
	xkeysym_unicode_map[0x06D2] = 0x0440;
	xkeysym_unicode_map[0x06D3] = 0x0441;
	xkeysym_unicode_map[0x06D4] = 0x0442;
	xkeysym_unicode_map[0x06D5] = 0x0443;
	xkeysym_unicode_map[0x06D6] = 0x0436;
	xkeysym_unicode_map[0x06D7] = 0x0432;
	xkeysym_unicode_map[0x06D8] = 0x044C;
	xkeysym_unicode_map[0x06D9] = 0x044B;
	xkeysym_unicode_map[0x06DA] = 0x0437;
	xkeysym_unicode_map[0x06DB] = 0x0448;
	xkeysym_unicode_map[0x06DC] = 0x044D;
	xkeysym_unicode_map[0x06DD] = 0x0449;
	xkeysym_unicode_map[0x06DE] = 0x0447;
	xkeysym_unicode_map[0x06DF] = 0x044A;
	xkeysym_unicode_map[0x06E0] = 0x042E;
	xkeysym_unicode_map[0x06E1] = 0x0410;
	xkeysym_unicode_map[0x06E2] = 0x0411;
	xkeysym_unicode_map[0x06E3] = 0x0426;
	xkeysym_unicode_map[0x06E4] = 0x0414;
	xkeysym_unicode_map[0x06E5] = 0x0415;
	xkeysym_unicode_map[0x06E6] = 0x0424;
	xkeysym_unicode_map[0x06E7] = 0x0413;
	xkeysym_unicode_map[0x06E8] = 0x0425;
	xkeysym_unicode_map[0x06E9] = 0x0418;
	xkeysym_unicode_map[0x06EA] = 0x0419;
	xkeysym_unicode_map[0x06EB] = 0x041A;
	xkeysym_unicode_map[0x06EC] = 0x041B;
	xkeysym_unicode_map[0x06ED] = 0x041C;
	xkeysym_unicode_map[0x06EE] = 0x041D;
	xkeysym_unicode_map[0x06EF] = 0x041E;
	xkeysym_unicode_map[0x06F0] = 0x041F;
	xkeysym_unicode_map[0x06F1] = 0x042F;
	xkeysym_unicode_map[0x06F2] = 0x0420;
	xkeysym_unicode_map[0x06F3] = 0x0421;
	xkeysym_unicode_map[0x06F4] = 0x0422;
	xkeysym_unicode_map[0x06F5] = 0x0423;
	xkeysym_unicode_map[0x06F6] = 0x0416;
	xkeysym_unicode_map[0x06F7] = 0x0412;
	xkeysym_unicode_map[0x06F8] = 0x042C;
	xkeysym_unicode_map[0x06F9] = 0x042B;
	xkeysym_unicode_map[0x06FA] = 0x0417;
	xkeysym_unicode_map[0x06FB] = 0x0428;
	xkeysym_unicode_map[0x06FC] = 0x042D;
	xkeysym_unicode_map[0x06FD] = 0x0429;
	xkeysym_unicode_map[0x06FE] = 0x0427;
	xkeysym_unicode_map[0x06FF] = 0x042A;
	xkeysym_unicode_map[0x07A1] = 0x0386;
	xkeysym_unicode_map[0x07A2] = 0x0388;
	xkeysym_unicode_map[0x07A3] = 0x0389;
	xkeysym_unicode_map[0x07A4] = 0x038A;
	xkeysym_unicode_map[0x07A5] = 0x03AA;
	xkeysym_unicode_map[0x07A7] = 0x038C;
	xkeysym_unicode_map[0x07A8] = 0x038E;
	xkeysym_unicode_map[0x07A9] = 0x03AB;
	xkeysym_unicode_map[0x07AB] = 0x038F;
	xkeysym_unicode_map[0x07AE] = 0x0385;
	xkeysym_unicode_map[0x07AF] = 0x2015;
	xkeysym_unicode_map[0x07B1] = 0x03AC;
	xkeysym_unicode_map[0x07B2] = 0x03AD;
	xkeysym_unicode_map[0x07B3] = 0x03AE;
	xkeysym_unicode_map[0x07B4] = 0x03AF;
	xkeysym_unicode_map[0x07B5] = 0x03CA;
	xkeysym_unicode_map[0x07B6] = 0x0390;
	xkeysym_unicode_map[0x07B7] = 0x03CC;
	xkeysym_unicode_map[0x07B8] = 0x03CD;
	xkeysym_unicode_map[0x07B9] = 0x03CB;
	xkeysym_unicode_map[0x07BA] = 0x03B0;
	xkeysym_unicode_map[0x07BB] = 0x03CE;
	xkeysym_unicode_map[0x07C1] = 0x0391;
	xkeysym_unicode_map[0x07C2] = 0x0392;
	xkeysym_unicode_map[0x07C3] = 0x0393;
	xkeysym_unicode_map[0x07C4] = 0x0394;
	xkeysym_unicode_map[0x07C5] = 0x0395;
	xkeysym_unicode_map[0x07C6] = 0x0396;
	xkeysym_unicode_map[0x07C7] = 0x0397;
	xkeysym_unicode_map[0x07C8] = 0x0398;
	xkeysym_unicode_map[0x07C9] = 0x0399;
	xkeysym_unicode_map[0x07CA] = 0x039A;
	xkeysym_unicode_map[0x07CB] = 0x039B;
	xkeysym_unicode_map[0x07CC] = 0x039C;
	xkeysym_unicode_map[0x07CD] = 0x039D;
	xkeysym_unicode_map[0x07CE] = 0x039E;
	xkeysym_unicode_map[0x07CF] = 0x039F;
	xkeysym_unicode_map[0x07D0] = 0x03A0;
	xkeysym_unicode_map[0x07D1] = 0x03A1;
	xkeysym_unicode_map[0x07D2] = 0x03A3;
	xkeysym_unicode_map[0x07D4] = 0x03A4;
	xkeysym_unicode_map[0x07D5] = 0x03A5;
	xkeysym_unicode_map[0x07D6] = 0x03A6;
	xkeysym_unicode_map[0x07D7] = 0x03A7;
	xkeysym_unicode_map[0x07D8] = 0x03A8;
	xkeysym_unicode_map[0x07D9] = 0x03A9;
	xkeysym_unicode_map[0x07E1] = 0x03B1;
	xkeysym_unicode_map[0x07E2] = 0x03B2;
	xkeysym_unicode_map[0x07E3] = 0x03B3;
	xkeysym_unicode_map[0x07E4] = 0x03B4;
	xkeysym_unicode_map[0x07E5] = 0x03B5;
	xkeysym_unicode_map[0x07E6] = 0x03B6;
	xkeysym_unicode_map[0x07E7] = 0x03B7;
	xkeysym_unicode_map[0x07E8] = 0x03B8;
	xkeysym_unicode_map[0x07E9] = 0x03B9;
	xkeysym_unicode_map[0x07EA] = 0x03BA;
	xkeysym_unicode_map[0x07EB] = 0x03BB;
	xkeysym_unicode_map[0x07EC] = 0x03BC;
	xkeysym_unicode_map[0x07ED] = 0x03BD;
	xkeysym_unicode_map[0x07EE] = 0x03BE;
	xkeysym_unicode_map[0x07EF] = 0x03BF;
	xkeysym_unicode_map[0x07F0] = 0x03C0;
	xkeysym_unicode_map[0x07F1] = 0x03C1;
	xkeysym_unicode_map[0x07F2] = 0x03C3;
	xkeysym_unicode_map[0x07F3] = 0x03C2;
	xkeysym_unicode_map[0x07F4] = 0x03C4;
	xkeysym_unicode_map[0x07F5] = 0x03C5;
	xkeysym_unicode_map[0x07F6] = 0x03C6;
	xkeysym_unicode_map[0x07F7] = 0x03C7;
	xkeysym_unicode_map[0x07F8] = 0x03C8;
	xkeysym_unicode_map[0x07F9] = 0x03C9;
	xkeysym_unicode_map[0x08A1] = 0x23B7;
	xkeysym_unicode_map[0x08A2] = 0x250C;
	xkeysym_unicode_map[0x08A3] = 0x2500;
	xkeysym_unicode_map[0x08A4] = 0x2320;
	xkeysym_unicode_map[0x08A5] = 0x2321;
	xkeysym_unicode_map[0x08A6] = 0x2502;
	xkeysym_unicode_map[0x08A7] = 0x23A1;
	xkeysym_unicode_map[0x08A8] = 0x23A3;
	xkeysym_unicode_map[0x08A9] = 0x23A4;
	xkeysym_unicode_map[0x08AA] = 0x23A6;
	xkeysym_unicode_map[0x08AB] = 0x239B;
	xkeysym_unicode_map[0x08AC] = 0x239D;
	xkeysym_unicode_map[0x08AD] = 0x239E;
	xkeysym_unicode_map[0x08AE] = 0x23A0;
	xkeysym_unicode_map[0x08AF] = 0x23A8;
	xkeysym_unicode_map[0x08B0] = 0x23AC;
	xkeysym_unicode_map[0x08BC] = 0x2264;
	xkeysym_unicode_map[0x08BD] = 0x2260;
	xkeysym_unicode_map[0x08BE] = 0x2265;
	xkeysym_unicode_map[0x08BF] = 0x222B;
	xkeysym_unicode_map[0x08C0] = 0x2234;
	xkeysym_unicode_map[0x08C1] = 0x221D;
	xkeysym_unicode_map[0x08C2] = 0x221E;
	xkeysym_unicode_map[0x08C5] = 0x2207;
	xkeysym_unicode_map[0x08C8] = 0x223C;
	xkeysym_unicode_map[0x08C9] = 0x2243;
	xkeysym_unicode_map[0x08CD] = 0x21D4;
	xkeysym_unicode_map[0x08CE] = 0x21D2;
	xkeysym_unicode_map[0x08CF] = 0x2261;
	xkeysym_unicode_map[0x08D6] = 0x221A;
	xkeysym_unicode_map[0x08DA] = 0x2282;
	xkeysym_unicode_map[0x08DB] = 0x2283;
	xkeysym_unicode_map[0x08DC] = 0x2229;
	xkeysym_unicode_map[0x08DD] = 0x222A;
	xkeysym_unicode_map[0x08DE] = 0x2227;
	xkeysym_unicode_map[0x08DF] = 0x2228;
	xkeysym_unicode_map[0x08EF] = 0x2202;
	xkeysym_unicode_map[0x08F6] = 0x0192;
	xkeysym_unicode_map[0x08FB] = 0x2190;
	xkeysym_unicode_map[0x08FC] = 0x2191;
	xkeysym_unicode_map[0x08FD] = 0x2192;
	xkeysym_unicode_map[0x08FE] = 0x2193;
	xkeysym_unicode_map[0x09E0] = 0x25C6;
	xkeysym_unicode_map[0x09E1] = 0x2592;
	xkeysym_unicode_map[0x09E2] = 0x2409;
	xkeysym_unicode_map[0x09E3] = 0x240C;
	xkeysym_unicode_map[0x09E4] = 0x240D;
	xkeysym_unicode_map[0x09E5] = 0x240A;
	xkeysym_unicode_map[0x09E8] = 0x2424;
	xkeysym_unicode_map[0x09E9] = 0x240B;
	xkeysym_unicode_map[0x09EA] = 0x2518;
	xkeysym_unicode_map[0x09EB] = 0x2510;
	xkeysym_unicode_map[0x09EC] = 0x250C;
	xkeysym_unicode_map[0x09ED] = 0x2514;
	xkeysym_unicode_map[0x09EE] = 0x253C;
	xkeysym_unicode_map[0x09EF] = 0x23BA;
	xkeysym_unicode_map[0x09F0] = 0x23BB;
	xkeysym_unicode_map[0x09F1] = 0x2500;
	xkeysym_unicode_map[0x09F2] = 0x23BC;
	xkeysym_unicode_map[0x09F3] = 0x23BD;
	xkeysym_unicode_map[0x09F4] = 0x251C;
	xkeysym_unicode_map[0x09F5] = 0x2524;
	xkeysym_unicode_map[0x09F6] = 0x2534;
	xkeysym_unicode_map[0x09F7] = 0x252C;
	xkeysym_unicode_map[0x09F8] = 0x2502;
	xkeysym_unicode_map[0x0AA1] = 0x2003;
	xkeysym_unicode_map[0x0AA2] = 0x2002;
	xkeysym_unicode_map[0x0AA3] = 0x2004;
	xkeysym_unicode_map[0x0AA4] = 0x2005;
	xkeysym_unicode_map[0x0AA5] = 0x2007;
	xkeysym_unicode_map[0x0AA6] = 0x2008;
	xkeysym_unicode_map[0x0AA7] = 0x2009;
	xkeysym_unicode_map[0x0AA8] = 0x200A;
	xkeysym_unicode_map[0x0AA9] = 0x2014;
	xkeysym_unicode_map[0x0AAA] = 0x2013;
	xkeysym_unicode_map[0x0AAE] = 0x2026;
	xkeysym_unicode_map[0x0AAF] = 0x2025;
	xkeysym_unicode_map[0x0AB0] = 0x2153;
	xkeysym_unicode_map[0x0AB1] = 0x2154;
	xkeysym_unicode_map[0x0AB2] = 0x2155;
	xkeysym_unicode_map[0x0AB3] = 0x2156;
	xkeysym_unicode_map[0x0AB4] = 0x2157;
	xkeysym_unicode_map[0x0AB5] = 0x2158;
	xkeysym_unicode_map[0x0AB6] = 0x2159;
	xkeysym_unicode_map[0x0AB7] = 0x215A;
	xkeysym_unicode_map[0x0AB8] = 0x2105;
	xkeysym_unicode_map[0x0ABB] = 0x2012;
	xkeysym_unicode_map[0x0ABC] = 0x2329;
	xkeysym_unicode_map[0x0ABE] = 0x232A;
	xkeysym_unicode_map[0x0AC3] = 0x215B;
	xkeysym_unicode_map[0x0AC4] = 0x215C;
	xkeysym_unicode_map[0x0AC5] = 0x215D;
	xkeysym_unicode_map[0x0AC6] = 0x215E;
	xkeysym_unicode_map[0x0AC9] = 0x2122;
	xkeysym_unicode_map[0x0ACA] = 0x2613;
	xkeysym_unicode_map[0x0ACC] = 0x25C1;
	xkeysym_unicode_map[0x0ACD] = 0x25B7;
	xkeysym_unicode_map[0x0ACE] = 0x25CB;
	xkeysym_unicode_map[0x0ACF] = 0x25AF;
	xkeysym_unicode_map[0x0AD0] = 0x2018;
	xkeysym_unicode_map[0x0AD1] = 0x2019;
	xkeysym_unicode_map[0x0AD2] = 0x201C;
	xkeysym_unicode_map[0x0AD3] = 0x201D;
	xkeysym_unicode_map[0x0AD4] = 0x211E;
	xkeysym_unicode_map[0x0AD6] = 0x2032;
	xkeysym_unicode_map[0x0AD7] = 0x2033;
	xkeysym_unicode_map[0x0AD9] = 0x271D;
	xkeysym_unicode_map[0x0ADB] = 0x25AC;
	xkeysym_unicode_map[0x0ADC] = 0x25C0;
	xkeysym_unicode_map[0x0ADD] = 0x25B6;
	xkeysym_unicode_map[0x0ADE] = 0x25CF;
	xkeysym_unicode_map[0x0ADF] = 0x25AE;
	xkeysym_unicode_map[0x0AE0] = 0x25E6;
	xkeysym_unicode_map[0x0AE1] = 0x25AB;
	xkeysym_unicode_map[0x0AE2] = 0x25AD;
	xkeysym_unicode_map[0x0AE3] = 0x25B3;
	xkeysym_unicode_map[0x0AE4] = 0x25BD;
	xkeysym_unicode_map[0x0AE5] = 0x2606;
	xkeysym_unicode_map[0x0AE6] = 0x2022;
	xkeysym_unicode_map[0x0AE7] = 0x25AA;
	xkeysym_unicode_map[0x0AE8] = 0x25B2;
	xkeysym_unicode_map[0x0AE9] = 0x25BC;
	xkeysym_unicode_map[0x0AEA] = 0x261C;
	xkeysym_unicode_map[0x0AEB] = 0x261E;
	xkeysym_unicode_map[0x0AEC] = 0x2663;
	xkeysym_unicode_map[0x0AED] = 0x2666;
	xkeysym_unicode_map[0x0AEE] = 0x2665;
	xkeysym_unicode_map[0x0AF0] = 0x2720;
	xkeysym_unicode_map[0x0AF1] = 0x2020;
	xkeysym_unicode_map[0x0AF2] = 0x2021;
	xkeysym_unicode_map[0x0AF3] = 0x2713;
	xkeysym_unicode_map[0x0AF4] = 0x2717;
	xkeysym_unicode_map[0x0AF5] = 0x266F;
	xkeysym_unicode_map[0x0AF6] = 0x266D;
	xkeysym_unicode_map[0x0AF7] = 0x2642;
	xkeysym_unicode_map[0x0AF8] = 0x2640;
	xkeysym_unicode_map[0x0AF9] = 0x260E;
	xkeysym_unicode_map[0x0AFA] = 0x2315;
	xkeysym_unicode_map[0x0AFB] = 0x2117;
	xkeysym_unicode_map[0x0AFC] = 0x2038;
	xkeysym_unicode_map[0x0AFD] = 0x201A;
	xkeysym_unicode_map[0x0AFE] = 0x201E;
	xkeysym_unicode_map[0x0BA3] = 0x003C;
	xkeysym_unicode_map[0x0BA6] = 0x003E;
	xkeysym_unicode_map[0x0BA8] = 0x2228;
	xkeysym_unicode_map[0x0BA9] = 0x2227;
	xkeysym_unicode_map[0x0BC0] = 0x00AF;
	xkeysym_unicode_map[0x0BC2] = 0x22A5;
	xkeysym_unicode_map[0x0BC3] = 0x2229;
	xkeysym_unicode_map[0x0BC4] = 0x230A;
	xkeysym_unicode_map[0x0BC6] = 0x005F;
	xkeysym_unicode_map[0x0BCA] = 0x2218;
	xkeysym_unicode_map[0x0BCC] = 0x2395;
	xkeysym_unicode_map[0x0BCE] = 0x22A4;
	xkeysym_unicode_map[0x0BCF] = 0x25CB;
	xkeysym_unicode_map[0x0BD3] = 0x2308;
	xkeysym_unicode_map[0x0BD6] = 0x222A;
	xkeysym_unicode_map[0x0BD8] = 0x2283;
	xkeysym_unicode_map[0x0BDA] = 0x2282;
	xkeysym_unicode_map[0x0BDC] = 0x22A2;
	xkeysym_unicode_map[0x0BFC] = 0x22A3;
	xkeysym_unicode_map[0x0CDF] = 0x2017;
	xkeysym_unicode_map[0x0CE0] = 0x05D0;
	xkeysym_unicode_map[0x0CE1] = 0x05D1;
	xkeysym_unicode_map[0x0CE2] = 0x05D2;
	xkeysym_unicode_map[0x0CE3] = 0x05D3;
	xkeysym_unicode_map[0x0CE4] = 0x05D4;
	xkeysym_unicode_map[0x0CE5] = 0x05D5;
	xkeysym_unicode_map[0x0CE6] = 0x05D6;
	xkeysym_unicode_map[0x0CE7] = 0x05D7;
	xkeysym_unicode_map[0x0CE8] = 0x05D8;
	xkeysym_unicode_map[0x0CE9] = 0x05D9;
	xkeysym_unicode_map[0x0CEA] = 0x05DA;
	xkeysym_unicode_map[0x0CEB] = 0x05DB;
	xkeysym_unicode_map[0x0CEC] = 0x05DC;
	xkeysym_unicode_map[0x0CED] = 0x05DD;
	xkeysym_unicode_map[0x0CEE] = 0x05DE;
	xkeysym_unicode_map[0x0CEF] = 0x05DF;
	xkeysym_unicode_map[0x0CF0] = 0x05E0;
	xkeysym_unicode_map[0x0CF1] = 0x05E1;
	xkeysym_unicode_map[0x0CF2] = 0x05E2;
	xkeysym_unicode_map[0x0CF3] = 0x05E3;
	xkeysym_unicode_map[0x0CF4] = 0x05E4;
	xkeysym_unicode_map[0x0CF5] = 0x05E5;
	xkeysym_unicode_map[0x0CF6] = 0x05E6;
	xkeysym_unicode_map[0x0CF7] = 0x05E7;
	xkeysym_unicode_map[0x0CF8] = 0x05E8;
	xkeysym_unicode_map[0x0CF9] = 0x05E9;
	xkeysym_unicode_map[0x0CFA] = 0x05EA;
	xkeysym_unicode_map[0x0DA1] = 0x0E01;
	xkeysym_unicode_map[0x0DA2] = 0x0E02;
	xkeysym_unicode_map[0x0DA3] = 0x0E03;
	xkeysym_unicode_map[0x0DA4] = 0x0E04;
	xkeysym_unicode_map[0x0DA5] = 0x0E05;
	xkeysym_unicode_map[0x0DA6] = 0x0E06;
	xkeysym_unicode_map[0x0DA7] = 0x0E07;
	xkeysym_unicode_map[0x0DA8] = 0x0E08;
	xkeysym_unicode_map[0x0DA9] = 0x0E09;
	xkeysym_unicode_map[0x0DAA] = 0x0E0A;
	xkeysym_unicode_map[0x0DAB] = 0x0E0B;
	xkeysym_unicode_map[0x0DAC] = 0x0E0C;
	xkeysym_unicode_map[0x0DAD] = 0x0E0D;
	xkeysym_unicode_map[0x0DAE] = 0x0E0E;
	xkeysym_unicode_map[0x0DAF] = 0x0E0F;
	xkeysym_unicode_map[0x0DB0] = 0x0E10;
	xkeysym_unicode_map[0x0DB1] = 0x0E11;
	xkeysym_unicode_map[0x0DB2] = 0x0E12;
	xkeysym_unicode_map[0x0DB3] = 0x0E13;
	xkeysym_unicode_map[0x0DB4] = 0x0E14;
	xkeysym_unicode_map[0x0DB5] = 0x0E15;
	xkeysym_unicode_map[0x0DB6] = 0x0E16;
	xkeysym_unicode_map[0x0DB7] = 0x0E17;
	xkeysym_unicode_map[0x0DB8] = 0x0E18;
	xkeysym_unicode_map[0x0DB9] = 0x0E19;
	xkeysym_unicode_map[0x0DBA] = 0x0E1A;
	xkeysym_unicode_map[0x0DBB] = 0x0E1B;
	xkeysym_unicode_map[0x0DBC] = 0x0E1C;
	xkeysym_unicode_map[0x0DBD] = 0x0E1D;
	xkeysym_unicode_map[0x0DBE] = 0x0E1E;
	xkeysym_unicode_map[0x0DBF] = 0x0E1F;
	xkeysym_unicode_map[0x0DC0] = 0x0E20;
	xkeysym_unicode_map[0x0DC1] = 0x0E21;
	xkeysym_unicode_map[0x0DC2] = 0x0E22;
	xkeysym_unicode_map[0x0DC3] = 0x0E23;
	xkeysym_unicode_map[0x0DC4] = 0x0E24;
	xkeysym_unicode_map[0x0DC5] = 0x0E25;
	xkeysym_unicode_map[0x0DC6] = 0x0E26;
	xkeysym_unicode_map[0x0DC7] = 0x0E27;
	xkeysym_unicode_map[0x0DC8] = 0x0E28;
	xkeysym_unicode_map[0x0DC9] = 0x0E29;
	xkeysym_unicode_map[0x0DCA] = 0x0E2A;
	xkeysym_unicode_map[0x0DCB] = 0x0E2B;
	xkeysym_unicode_map[0x0DCC] = 0x0E2C;
	xkeysym_unicode_map[0x0DCD] = 0x0E2D;
	xkeysym_unicode_map[0x0DCE] = 0x0E2E;
	xkeysym_unicode_map[0x0DCF] = 0x0E2F;
	xkeysym_unicode_map[0x0DD0] = 0x0E30;
	xkeysym_unicode_map[0x0DD1] = 0x0E31;
	xkeysym_unicode_map[0x0DD2] = 0x0E32;
	xkeysym_unicode_map[0x0DD3] = 0x0E33;
	xkeysym_unicode_map[0x0DD4] = 0x0E34;
	xkeysym_unicode_map[0x0DD5] = 0x0E35;
	xkeysym_unicode_map[0x0DD6] = 0x0E36;
	xkeysym_unicode_map[0x0DD7] = 0x0E37;
	xkeysym_unicode_map[0x0DD8] = 0x0E38;
	xkeysym_unicode_map[0x0DD9] = 0x0E39;
	xkeysym_unicode_map[0x0DDA] = 0x0E3A;
	xkeysym_unicode_map[0x0DDF] = 0x0E3F;
	xkeysym_unicode_map[0x0DE0] = 0x0E40;
	xkeysym_unicode_map[0x0DE1] = 0x0E41;
	xkeysym_unicode_map[0x0DE2] = 0x0E42;
	xkeysym_unicode_map[0x0DE3] = 0x0E43;
	xkeysym_unicode_map[0x0DE4] = 0x0E44;
	xkeysym_unicode_map[0x0DE5] = 0x0E45;
	xkeysym_unicode_map[0x0DE6] = 0x0E46;
	xkeysym_unicode_map[0x0DE7] = 0x0E47;
	xkeysym_unicode_map[0x0DE8] = 0x0E48;
	xkeysym_unicode_map[0x0DE9] = 0x0E49;
	xkeysym_unicode_map[0x0DEA] = 0x0E4A;
	xkeysym_unicode_map[0x0DEB] = 0x0E4B;
	xkeysym_unicode_map[0x0DEC] = 0x0E4C;
	xkeysym_unicode_map[0x0DED] = 0x0E4D;
	xkeysym_unicode_map[0x0DF0] = 0x0E50;
	xkeysym_unicode_map[0x0DF1] = 0x0E51;
	xkeysym_unicode_map[0x0DF2] = 0x0E52;
	xkeysym_unicode_map[0x0DF3] = 0x0E53;
	xkeysym_unicode_map[0x0DF4] = 0x0E54;
	xkeysym_unicode_map[0x0DF5] = 0x0E55;
	xkeysym_unicode_map[0x0DF6] = 0x0E56;
	xkeysym_unicode_map[0x0DF7] = 0x0E57;
	xkeysym_unicode_map[0x0DF8] = 0x0E58;
	xkeysym_unicode_map[0x0DF9] = 0x0E59;
	xkeysym_unicode_map[0x0EA1] = 0x3131;
	xkeysym_unicode_map[0x0EA2] = 0x3132;
	xkeysym_unicode_map[0x0EA3] = 0x3133;
	xkeysym_unicode_map[0x0EA4] = 0x3134;
	xkeysym_unicode_map[0x0EA5] = 0x3135;
	xkeysym_unicode_map[0x0EA6] = 0x3136;
	xkeysym_unicode_map[0x0EA7] = 0x3137;
	xkeysym_unicode_map[0x0EA8] = 0x3138;
	xkeysym_unicode_map[0x0EA9] = 0x3139;
	xkeysym_unicode_map[0x0EAA] = 0x313A;
	xkeysym_unicode_map[0x0EAB] = 0x313B;
	xkeysym_unicode_map[0x0EAC] = 0x313C;
	xkeysym_unicode_map[0x0EAD] = 0x313D;
	xkeysym_unicode_map[0x0EAE] = 0x313E;
	xkeysym_unicode_map[0x0EAF] = 0x313F;
	xkeysym_unicode_map[0x0EB0] = 0x3140;
	xkeysym_unicode_map[0x0EB1] = 0x3141;
	xkeysym_unicode_map[0x0EB2] = 0x3142;
	xkeysym_unicode_map[0x0EB3] = 0x3143;
	xkeysym_unicode_map[0x0EB4] = 0x3144;
	xkeysym_unicode_map[0x0EB5] = 0x3145;
	xkeysym_unicode_map[0x0EB6] = 0x3146;
	xkeysym_unicode_map[0x0EB7] = 0x3147;
	xkeysym_unicode_map[0x0EB8] = 0x3148;
	xkeysym_unicode_map[0x0EB9] = 0x3149;
	xkeysym_unicode_map[0x0EBA] = 0x314A;
	xkeysym_unicode_map[0x0EBB] = 0x314B;
	xkeysym_unicode_map[0x0EBC] = 0x314C;
	xkeysym_unicode_map[0x0EBD] = 0x314D;
	xkeysym_unicode_map[0x0EBE] = 0x314E;
	xkeysym_unicode_map[0x0EBF] = 0x314F;
	xkeysym_unicode_map[0x0EC0] = 0x3150;
	xkeysym_unicode_map[0x0EC1] = 0x3151;
	xkeysym_unicode_map[0x0EC2] = 0x3152;
	xkeysym_unicode_map[0x0EC3] = 0x3153;
	xkeysym_unicode_map[0x0EC4] = 0x3154;
	xkeysym_unicode_map[0x0EC5] = 0x3155;
	xkeysym_unicode_map[0x0EC6] = 0x3156;
	xkeysym_unicode_map[0x0EC7] = 0x3157;
	xkeysym_unicode_map[0x0EC8] = 0x3158;
	xkeysym_unicode_map[0x0EC9] = 0x3159;
	xkeysym_unicode_map[0x0ECA] = 0x315A;
	xkeysym_unicode_map[0x0ECB] = 0x315B;
	xkeysym_unicode_map[0x0ECC] = 0x315C;
	xkeysym_unicode_map[0x0ECD] = 0x315D;
	xkeysym_unicode_map[0x0ECE] = 0x315E;
	xkeysym_unicode_map[0x0ECF] = 0x315F;
	xkeysym_unicode_map[0x0ED0] = 0x3160;
	xkeysym_unicode_map[0x0ED1] = 0x3161;
	xkeysym_unicode_map[0x0ED2] = 0x3162;
	xkeysym_unicode_map[0x0ED3] = 0x3163;
	xkeysym_unicode_map[0x0ED4] = 0x11A8;
	xkeysym_unicode_map[0x0ED5] = 0x11A9;
	xkeysym_unicode_map[0x0ED6] = 0x11AA;
	xkeysym_unicode_map[0x0ED7] = 0x11AB;
	xkeysym_unicode_map[0x0ED8] = 0x11AC;
	xkeysym_unicode_map[0x0ED9] = 0x11AD;
	xkeysym_unicode_map[0x0EDA] = 0x11AE;
	xkeysym_unicode_map[0x0EDB] = 0x11AF;
	xkeysym_unicode_map[0x0EDC] = 0x11B0;
	xkeysym_unicode_map[0x0EDD] = 0x11B1;
	xkeysym_unicode_map[0x0EDE] = 0x11B2;
	xkeysym_unicode_map[0x0EDF] = 0x11B3;
	xkeysym_unicode_map[0x0EE0] = 0x11B4;
	xkeysym_unicode_map[0x0EE1] = 0x11B5;
	xkeysym_unicode_map[0x0EE2] = 0x11B6;
	xkeysym_unicode_map[0x0EE3] = 0x11B7;
	xkeysym_unicode_map[0x0EE4] = 0x11B8;
	xkeysym_unicode_map[0x0EE5] = 0x11B9;
	xkeysym_unicode_map[0x0EE6] = 0x11BA;
	xkeysym_unicode_map[0x0EE7] = 0x11BB;
	xkeysym_unicode_map[0x0EE8] = 0x11BC;
	xkeysym_unicode_map[0x0EE9] = 0x11BD;
	xkeysym_unicode_map[0x0EEA] = 0x11BE;
	xkeysym_unicode_map[0x0EEB] = 0x11BF;
	xkeysym_unicode_map[0x0EEC] = 0x11C0;
	xkeysym_unicode_map[0x0EED] = 0x11C1;
	xkeysym_unicode_map[0x0EEE] = 0x11C2;
	xkeysym_unicode_map[0x0EEF] = 0x316D;
	xkeysym_unicode_map[0x0EF0] = 0x3171;
	xkeysym_unicode_map[0x0EF1] = 0x3178;
	xkeysym_unicode_map[0x0EF2] = 0x317F;
	xkeysym_unicode_map[0x0EF3] = 0x3181;
	xkeysym_unicode_map[0x0EF4] = 0x3184;
	xkeysym_unicode_map[0x0EF5] = 0x3186;
	xkeysym_unicode_map[0x0EF6] = 0x318D;
	xkeysym_unicode_map[0x0EF7] = 0x318E;
	xkeysym_unicode_map[0x0EF8] = 0x11EB;
	xkeysym_unicode_map[0x0EF9] = 0x11F0;
	xkeysym_unicode_map[0x0EFA] = 0x11F9;
	xkeysym_unicode_map[0x0EFF] = 0x20A9;
	xkeysym_unicode_map[0x13A4] = 0x20AC;
	xkeysym_unicode_map[0x13BC] = 0x0152;
	xkeysym_unicode_map[0x13BD] = 0x0153;
	xkeysym_unicode_map[0x13BE] = 0x0178;
	xkeysym_unicode_map[0x20AC] = 0x20AC;

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

bool KeyMappingX11::is_sym_numpad(KeySym p_keysym) {
	switch (p_keysym) {
		case XK_KP_Equal:
		case XK_KP_Add:
		case XK_KP_Subtract:
		case XK_KP_Multiply:
		case XK_KP_Divide:
		case XK_KP_Separator:
		case XK_KP_Decimal:
		case XK_KP_Delete:
		case XK_KP_0:
		case XK_KP_1:
		case XK_KP_2:
		case XK_KP_3:
		case XK_KP_4:
		case XK_KP_5:
		case XK_KP_6:
		case XK_KP_7:
		case XK_KP_8:
		case XK_KP_9: {
			return true;
		} break;
	}

	return false;
}

Key KeyMappingX11::get_keycode(KeySym p_keysym) {
	if (p_keysym >= 0x20 && p_keysym < 0x7E) { // ASCII, maps 1-1
		if (p_keysym > 0x60 && p_keysym < 0x7B) { // Lowercase ASCII.
			return (Key)(p_keysym - 32);
		} else {
			return (Key)p_keysym;
		}
	}

	const Key *key = xkeysym_map.getptr(p_keysym);
	if (key) {
		return *key;
	}
	return Key::NONE;
}

Key KeyMappingX11::get_scancode(unsigned int p_code) {
	const Key *key = scancode_map.getptr(p_code);
	if (key) {
		return *key;
	}

	return Key::NONE;
}

unsigned int KeyMappingX11::get_xlibcode(Key p_keysym) {
	const unsigned int *key = scancode_map_inv.getptr(p_keysym);
	if (key) {
		return *key;
	}
	return 0x00;
}

char32_t KeyMappingX11::get_unicode_from_keysym(KeySym p_keysym) {
	// Latin-1
	if (p_keysym >= 0x20 && p_keysym <= 0x7E) {
		return p_keysym;
	}
	if (p_keysym >= 0xA0 && p_keysym <= 0xFF) {
		return p_keysym;
	}

	// Keypad to Latin-1.
	if (p_keysym >= 0xFFAA && p_keysym <= 0xFFB9) {
		return p_keysym - 0xFF80;
	}

	// Unicode (may be present).
	if ((p_keysym & 0xFF000000) == 0x01000000) {
		return p_keysym & 0x00FFFFFF;
	}

	const char32_t *c = xkeysym_unicode_map.getptr(p_keysym);
	if (c) {
		return *c;
	}
	return 0;
}

KeyLocation KeyMappingX11::get_location(unsigned int p_code) {
	const KeyLocation *location = location_map.getptr(p_code);
	if (location) {
		return *location;
	}
	return KeyLocation::UNSPECIFIED;
}
