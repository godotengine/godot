/*************************************************************************/
/*  key_mapping_windows.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "key_mapping_windows.h"

#include <stdio.h>

struct _WinTranslatePair {
	Key keysym;
	unsigned int keycode;
};

static _WinTranslatePair _vk_to_keycode[] = {
	{ Key::BACKSPACE, VK_BACK }, // (0x08) // backspace
	{ Key::TAB, VK_TAB }, //(0x09)

	//VK_CLEAR (0x0C)

	{ Key::ENTER, VK_RETURN }, //(0x0D)

	{ Key::SHIFT, VK_SHIFT }, //(0x10)

	{ Key::CTRL, VK_CONTROL }, //(0x11)

	{ Key::ALT, VK_MENU }, //(0x12)

	{ Key::PAUSE, VK_PAUSE }, //(0x13)

	{ Key::CAPSLOCK, VK_CAPITAL }, //(0x14)

	{ Key::ESCAPE, VK_ESCAPE }, //(0x1B)

	{ Key::SPACE, VK_SPACE }, //(0x20)

	{ Key::PAGEUP, VK_PRIOR }, //(0x21)

	{ Key::PAGEDOWN, VK_NEXT }, //(0x22)

	{ Key::END, VK_END }, //(0x23)

	{ Key::HOME, VK_HOME }, //(0x24)

	{ Key::LEFT, VK_LEFT }, //(0x25)

	{ Key::UP, VK_UP }, //(0x26)

	{ Key::RIGHT, VK_RIGHT }, //(0x27)

	{ Key::DOWN, VK_DOWN }, // (0x28)

	//VK_SELECT (0x29)

	{ Key::PRINT, VK_PRINT }, // (0x2A)

	//VK_EXECUTE (0x2B)

	{ Key::PRINT, VK_SNAPSHOT }, // (0x2C)

	{ Key::INSERT, VK_INSERT }, // (0x2D)

	{ Key::KEY_DELETE, VK_DELETE }, // (0x2E)

	{ Key::HELP, VK_HELP }, // (0x2F)

	{ Key::KEY_0, (0x30) }, ////0 key
	{ Key::KEY_1, (0x31) }, ////1 key
	{ Key::KEY_2, (0x32) }, ////2 key
	{ Key::KEY_3, (0x33) }, ////3 key
	{ Key::KEY_4, (0x34) }, ////4 key
	{ Key::KEY_5, (0x35) }, ////5 key
	{ Key::KEY_6, (0x36) }, ////6 key
	{ Key::KEY_7, (0x37) }, ////7 key
	{ Key::KEY_8, (0x38) }, ////8 key
	{ Key::KEY_9, (0x39) }, ////9 key
	{ Key::A, (0x41) }, ////A key
	{ Key::B, (0x42) }, ////B key
	{ Key::C, (0x43) }, ////C key
	{ Key::D, (0x44) }, ////D key
	{ Key::E, (0x45) }, ////E key
	{ Key::F, (0x46) }, ////F key
	{ Key::G, (0x47) }, ////G key
	{ Key::H, (0x48) }, ////H key
	{ Key::I, (0x49) }, ////I key
	{ Key::J, (0x4A) }, ////J key
	{ Key::K, (0x4B) }, ////K key
	{ Key::L, (0x4C) }, ////L key
	{ Key::M, (0x4D) }, ////M key
	{ Key::N, (0x4E) }, ////N key
	{ Key::O, (0x4F) }, ////O key
	{ Key::P, (0x50) }, ////P key
	{ Key::Q, (0x51) }, ////Q key
	{ Key::R, (0x52) }, ////R key
	{ Key::S, (0x53) }, ////S key
	{ Key::T, (0x54) }, ////T key
	{ Key::U, (0x55) }, ////U key
	{ Key::V, (0x56) }, ////V key
	{ Key::W, (0x57) }, ////W key
	{ Key::X, (0x58) }, ////X key
	{ Key::Y, (0x59) }, ////Y key
	{ Key::Z, (0x5A) }, ////Z key

	{ (Key)KeyModifierMask::META, VK_LWIN }, //(0x5B)
	{ (Key)KeyModifierMask::META, VK_RWIN }, //(0x5C)
	{ Key::MENU, VK_APPS }, //(0x5D)
	{ Key::STANDBY, VK_SLEEP }, //(0x5F)
	{ Key::KP_0, VK_NUMPAD0 }, //(0x60)
	{ Key::KP_1, VK_NUMPAD1 }, //(0x61)
	{ Key::KP_2, VK_NUMPAD2 }, //(0x62)
	{ Key::KP_3, VK_NUMPAD3 }, //(0x63)
	{ Key::KP_4, VK_NUMPAD4 }, //(0x64)
	{ Key::KP_5, VK_NUMPAD5 }, //(0x65)
	{ Key::KP_6, VK_NUMPAD6 }, //(0x66)
	{ Key::KP_7, VK_NUMPAD7 }, //(0x67)
	{ Key::KP_8, VK_NUMPAD8 }, //(0x68)
	{ Key::KP_9, VK_NUMPAD9 }, //(0x69)
	{ Key::KP_MULTIPLY, VK_MULTIPLY }, // (0x6A)
	{ Key::KP_ADD, VK_ADD }, // (0x6B)
	//VK_SEPARATOR (0x6C)
	{ Key::KP_SUBTRACT, VK_SUBTRACT }, // (0x6D)
	{ Key::KP_PERIOD, VK_DECIMAL }, // (0x6E)
	{ Key::KP_DIVIDE, VK_DIVIDE }, // (0x6F)
	{ Key::F1, VK_F1 }, // (0x70)
	{ Key::F2, VK_F2 }, // (0x71)
	{ Key::F3, VK_F3 }, // (0x72)
	{ Key::F4, VK_F4 }, // (0x73)
	{ Key::F5, VK_F5 }, // (0x74)
	{ Key::F6, VK_F6 }, // (0x75)
	{ Key::F7, VK_F7 }, // (0x76)
	{ Key::F8, VK_F8 }, // (0x77)
	{ Key::F9, VK_F9 }, // (0x78)
	{ Key::F10, VK_F10 }, // (0x79)
	{ Key::F11, VK_F11 }, // (0x7A)
	{ Key::F12, VK_F12 }, // (0x7B)
	{ Key::F13, VK_F13 }, // (0x7C)
	{ Key::F14, VK_F14 }, // (0x7D)
	{ Key::F15, VK_F15 }, // (0x7E)
	{ Key::F16, VK_F16 }, // (0x7F)
	{ Key::NUMLOCK, VK_NUMLOCK }, // (0x90)
	{ Key::SCROLLLOCK, VK_SCROLL }, // (0x91)
	{ Key::SHIFT, VK_LSHIFT }, // (0xA0)
	{ Key::SHIFT, VK_RSHIFT }, // (0xA1)
	{ Key::CTRL, VK_LCONTROL }, // (0xA2)
	{ Key::CTRL, VK_RCONTROL }, // (0xA3)
	{ Key::MENU, VK_LMENU }, // (0xA4)
	{ Key::MENU, VK_RMENU }, // (0xA5)

	{ Key::BACK, VK_BROWSER_BACK }, // (0xA6)

	{ Key::FORWARD, VK_BROWSER_FORWARD }, // (0xA7)

	{ Key::REFRESH, VK_BROWSER_REFRESH }, // (0xA8)

	{ Key::STOP, VK_BROWSER_STOP }, // (0xA9)

	{ Key::SEARCH, VK_BROWSER_SEARCH }, // (0xAA)

	{ Key::FAVORITES, VK_BROWSER_FAVORITES }, // (0xAB)

	{ Key::HOMEPAGE, VK_BROWSER_HOME }, // (0xAC)

	{ Key::VOLUMEMUTE, VK_VOLUME_MUTE }, // (0xAD)

	{ Key::VOLUMEDOWN, VK_VOLUME_DOWN }, // (0xAE)

	{ Key::VOLUMEUP, VK_VOLUME_UP }, // (0xAF)

	{ Key::MEDIANEXT, VK_MEDIA_NEXT_TRACK }, // (0xB0)

	{ Key::MEDIAPREVIOUS, VK_MEDIA_PREV_TRACK }, // (0xB1)

	{ Key::MEDIASTOP, VK_MEDIA_STOP }, // (0xB2)

	//VK_MEDIA_PLAY_PAUSE (0xB3)

	{ Key::LAUNCHMAIL, VK_LAUNCH_MAIL }, // (0xB4)

	{ Key::LAUNCHMEDIA, VK_LAUNCH_MEDIA_SELECT }, // (0xB5)

	{ Key::LAUNCH0, VK_LAUNCH_APP1 }, // (0xB6)

	{ Key::LAUNCH1, VK_LAUNCH_APP2 }, // (0xB7)

	{ Key::SEMICOLON, VK_OEM_1 }, // (0xBA)

	{ Key::EQUAL, VK_OEM_PLUS }, // (0xBB) // Windows 2000/XP: For any country/region, the '+' key
	{ Key::COMMA, VK_OEM_COMMA }, // (0xBC) // Windows 2000/XP: For any country/region, the ',' key
	{ Key::MINUS, VK_OEM_MINUS }, // (0xBD) // Windows 2000/XP: For any country/region, the '-' key
	{ Key::PERIOD, VK_OEM_PERIOD }, // (0xBE) // Windows 2000/XP: For any country/region, the '.' key
	{ Key::SLASH, VK_OEM_2 }, // (0xBF) //Windows 2000/XP: For the US standard keyboard, the '/?' key

	{ Key::QUOTELEFT, VK_OEM_3 }, // (0xC0)
	{ Key::BRACELEFT, VK_OEM_4 }, // (0xDB)
	{ Key::BACKSLASH, VK_OEM_5 }, // (0xDC)
	{ Key::BRACERIGHT, VK_OEM_6 }, // (0xDD)
	{ Key::APOSTROPHE, VK_OEM_7 }, // (0xDE)
	/*
{VK_OEM_8 (0xDF)
{VK_OEM_102 (0xE2) // Windows 2000/XP: Either the angle bracket key or the backslash key on the RT 102-key keyboard
*/
	//{ Key::PLAY, VK_PLAY},// (0xFA)

	{ Key::UNKNOWN, 0 }
};

/*
VK_ZOOM (0xFB)
VK_NONAME (0xFC)
VK_PA1 (0xFD)
VK_OEM_CLEAR (0xFE)
*/

static _WinTranslatePair _scancode_to_keycode[] = {
	{ Key::ESCAPE, 0x01 },
	{ Key::KEY_1, 0x02 },
	{ Key::KEY_2, 0x03 },
	{ Key::KEY_3, 0x04 },
	{ Key::KEY_4, 0x05 },
	{ Key::KEY_5, 0x06 },
	{ Key::KEY_6, 0x07 },
	{ Key::KEY_7, 0x08 },
	{ Key::KEY_8, 0x09 },
	{ Key::KEY_9, 0x0A },
	{ Key::KEY_0, 0x0B },
	{ Key::MINUS, 0x0C },
	{ Key::EQUAL, 0x0D },
	{ Key::BACKSPACE, 0x0E },
	{ Key::TAB, 0x0F },
	{ Key::Q, 0x10 },
	{ Key::W, 0x11 },
	{ Key::E, 0x12 },
	{ Key::R, 0x13 },
	{ Key::T, 0x14 },
	{ Key::Y, 0x15 },
	{ Key::U, 0x16 },
	{ Key::I, 0x17 },
	{ Key::O, 0x18 },
	{ Key::P, 0x19 },
	{ Key::BRACELEFT, 0x1A },
	{ Key::BRACERIGHT, 0x1B },
	{ Key::ENTER, 0x1C },
	{ Key::CTRL, 0x1D },
	{ Key::A, 0x1E },
	{ Key::S, 0x1F },
	{ Key::D, 0x20 },
	{ Key::F, 0x21 },
	{ Key::G, 0x22 },
	{ Key::H, 0x23 },
	{ Key::J, 0x24 },
	{ Key::K, 0x25 },
	{ Key::L, 0x26 },
	{ Key::SEMICOLON, 0x27 },
	{ Key::APOSTROPHE, 0x28 },
	{ Key::QUOTELEFT, 0x29 },
	{ Key::SHIFT, 0x2A },
	{ Key::BACKSLASH, 0x2B },
	{ Key::Z, 0x2C },
	{ Key::X, 0x2D },
	{ Key::C, 0x2E },
	{ Key::V, 0x2F },
	{ Key::B, 0x30 },
	{ Key::N, 0x31 },
	{ Key::M, 0x32 },
	{ Key::COMMA, 0x33 },
	{ Key::PERIOD, 0x34 },
	{ Key::SLASH, 0x35 },
	{ Key::SHIFT, 0x36 },
	{ Key::PRINT, 0x37 },
	{ Key::ALT, 0x38 },
	{ Key::SPACE, 0x39 },
	{ Key::CAPSLOCK, 0x3A },
	{ Key::F1, 0x3B },
	{ Key::F2, 0x3C },
	{ Key::F3, 0x3D },
	{ Key::F4, 0x3E },
	{ Key::F5, 0x3F },
	{ Key::F6, 0x40 },
	{ Key::F7, 0x41 },
	{ Key::F8, 0x42 },
	{ Key::F9, 0x43 },
	{ Key::F10, 0x44 },
	{ Key::NUMLOCK, 0x45 },
	{ Key::SCROLLLOCK, 0x46 },
	{ Key::HOME, 0x47 },
	{ Key::UP, 0x48 },
	{ Key::PAGEUP, 0x49 },
	{ Key::KP_SUBTRACT, 0x4A },
	{ Key::LEFT, 0x4B },
	{ Key::KP_5, 0x4C },
	{ Key::RIGHT, 0x4D },
	{ Key::KP_ADD, 0x4E },
	{ Key::END, 0x4F },
	{ Key::DOWN, 0x50 },
	{ Key::PAGEDOWN, 0x51 },
	{ Key::INSERT, 0x52 },
	{ Key::KEY_DELETE, 0x53 },
	//{ Key::???, 0x56 }, //NON US BACKSLASH
	{ Key::F11, 0x57 },
	{ Key::F12, 0x58 },
	{ Key::META, 0x5B },
	{ Key::META, 0x5C },
	{ Key::MENU, 0x5D },
	{ Key::F13, 0x64 },
	{ Key::F14, 0x65 },
	{ Key::F15, 0x66 },
	{ Key::F16, 0x67 },
	{ Key::UNKNOWN, 0 }
};

Key KeyMappingWindows::get_keysym(unsigned int p_code) {
	for (int i = 0; _vk_to_keycode[i].keysym != Key::UNKNOWN; i++) {
		if (_vk_to_keycode[i].keycode == p_code) {
			//printf("outcode: %x\n",_vk_to_keycode[i].keysym);

			return _vk_to_keycode[i].keysym;
		}
	}

	return Key::UNKNOWN;
}

unsigned int KeyMappingWindows::get_scancode(Key p_keycode) {
	for (int i = 0; _scancode_to_keycode[i].keysym != Key::UNKNOWN; i++) {
		if (_scancode_to_keycode[i].keysym == p_keycode) {
			return _scancode_to_keycode[i].keycode;
		}
	}

	return 0;
}

Key KeyMappingWindows::get_scansym(unsigned int p_code, bool p_extended) {
	Key keycode = Key::UNKNOWN;
	for (int i = 0; _scancode_to_keycode[i].keysym != Key::UNKNOWN; i++) {
		if (_scancode_to_keycode[i].keycode == p_code) {
			keycode = _scancode_to_keycode[i].keysym;
			break;
		}
	}

	if (p_extended) {
		switch (keycode) {
			case Key::ENTER: {
				keycode = Key::KP_ENTER;
			} break;
			case Key::SLASH: {
				keycode = Key::KP_DIVIDE;
			} break;
			case Key::CAPSLOCK: {
				keycode = Key::KP_ADD;
			} break;
			default:
				break;
		}
	} else {
		switch (keycode) {
			case Key::NUMLOCK: {
				keycode = Key::PAUSE;
			} break;
			case Key::HOME: {
				keycode = Key::KP_7;
			} break;
			case Key::UP: {
				keycode = Key::KP_8;
			} break;
			case Key::PAGEUP: {
				keycode = Key::KP_9;
			} break;
			case Key::LEFT: {
				keycode = Key::KP_4;
			} break;
			case Key::RIGHT: {
				keycode = Key::KP_6;
			} break;
			case Key::END: {
				keycode = Key::KP_1;
			} break;
			case Key::DOWN: {
				keycode = Key::KP_2;
			} break;
			case Key::PAGEDOWN: {
				keycode = Key::KP_3;
			} break;
			case Key::INSERT: {
				keycode = Key::KP_0;
			} break;
			case Key::KEY_DELETE: {
				keycode = Key::KP_PERIOD;
			} break;
			case Key::PRINT: {
				keycode = Key::KP_MULTIPLY;
			} break;
			default:
				break;
		}
	}

	return keycode;
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
