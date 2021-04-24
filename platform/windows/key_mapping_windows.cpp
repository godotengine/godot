/*************************************************************************/
/*  key_mapping_windows.cpp                                              */
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

#include "key_mapping_windows.h"

#include <stdio.h>

struct _WinTranslatePair {
	unsigned int keysym;
	unsigned int keycode;
};

static _WinTranslatePair _vk_to_keycode[] = {
	{ KEY_BACKSPACE, VK_BACK }, // (0x08) // backspace
	{ KEY_TAB, VK_TAB }, //(0x09)

	//VK_CLEAR (0x0C)

	{ KEY_ENTER, VK_RETURN }, //(0x0D)

	{ KEY_SHIFT, VK_SHIFT }, //(0x10)

	{ KEY_CTRL, VK_CONTROL }, //(0x11)

	{ KEY_ALT, VK_MENU }, //(0x12)

	{ KEY_PAUSE, VK_PAUSE }, //(0x13)

	{ KEY_CAPSLOCK, VK_CAPITAL }, //(0x14)

	{ KEY_ESCAPE, VK_ESCAPE }, //(0x1B)

	{ KEY_SPACE, VK_SPACE }, //(0x20)

	{ KEY_PAGEUP, VK_PRIOR }, //(0x21)

	{ KEY_PAGEDOWN, VK_NEXT }, //(0x22)

	{ KEY_END, VK_END }, //(0x23)

	{ KEY_HOME, VK_HOME }, //(0x24)

	{ KEY_LEFT, VK_LEFT }, //(0x25)

	{ KEY_UP, VK_UP }, //(0x26)

	{ KEY_RIGHT, VK_RIGHT }, //(0x27)

	{ KEY_DOWN, VK_DOWN }, // (0x28)

	//VK_SELECT (0x29)

	{ KEY_PRINT, VK_PRINT }, // (0x2A)

	//VK_EXECUTE (0x2B)

	{ KEY_PRINT, VK_SNAPSHOT }, // (0x2C)

	{ KEY_INSERT, VK_INSERT }, // (0x2D)

	{ KEY_DELETE, VK_DELETE }, // (0x2E)

	{ KEY_HELP, VK_HELP }, // (0x2F)

	{ KEY_0, (0x30) }, ////0 key
	{ KEY_1, (0x31) }, ////1 key
	{ KEY_2, (0x32) }, ////2 key
	{ KEY_3, (0x33) }, ////3 key
	{ KEY_4, (0x34) }, ////4 key
	{ KEY_5, (0x35) }, ////5 key
	{ KEY_6, (0x36) }, ////6 key
	{ KEY_7, (0x37) }, ////7 key
	{ KEY_8, (0x38) }, ////8 key
	{ KEY_9, (0x39) }, ////9 key
	{ KEY_A, (0x41) }, ////A key
	{ KEY_B, (0x42) }, ////B key
	{ KEY_C, (0x43) }, ////C key
	{ KEY_D, (0x44) }, ////D key
	{ KEY_E, (0x45) }, ////E key
	{ KEY_F, (0x46) }, ////F key
	{ KEY_G, (0x47) }, ////G key
	{ KEY_H, (0x48) }, ////H key
	{ KEY_I, (0x49) }, ////I key
	{ KEY_J, (0x4A) }, ////J key
	{ KEY_K, (0x4B) }, ////K key
	{ KEY_L, (0x4C) }, ////L key
	{ KEY_M, (0x4D) }, ////M key
	{ KEY_N, (0x4E) }, ////N key
	{ KEY_O, (0x4F) }, ////O key
	{ KEY_P, (0x50) }, ////P key
	{ KEY_Q, (0x51) }, ////Q key
	{ KEY_R, (0x52) }, ////R key
	{ KEY_S, (0x53) }, ////S key
	{ KEY_T, (0x54) }, ////T key
	{ KEY_U, (0x55) }, ////U key
	{ KEY_V, (0x56) }, ////V key
	{ KEY_W, (0x57) }, ////W key
	{ KEY_X, (0x58) }, ////X key
	{ KEY_Y, (0x59) }, ////Y key
	{ KEY_Z, (0x5A) }, ////Z key

	{ KEY_MASK_META, VK_LWIN }, //(0x5B)
	{ KEY_MASK_META, VK_RWIN }, //(0x5C)
	{ KEY_MENU, VK_APPS }, //(0x5D)
	{ KEY_STANDBY, VK_SLEEP }, //(0x5F)
	{ KEY_KP_0, VK_NUMPAD0 }, //(0x60)
	{ KEY_KP_1, VK_NUMPAD1 }, //(0x61)
	{ KEY_KP_2, VK_NUMPAD2 }, //(0x62)
	{ KEY_KP_3, VK_NUMPAD3 }, //(0x63)
	{ KEY_KP_4, VK_NUMPAD4 }, //(0x64)
	{ KEY_KP_5, VK_NUMPAD5 }, //(0x65)
	{ KEY_KP_6, VK_NUMPAD6 }, //(0x66)
	{ KEY_KP_7, VK_NUMPAD7 }, //(0x67)
	{ KEY_KP_8, VK_NUMPAD8 }, //(0x68)
	{ KEY_KP_9, VK_NUMPAD9 }, //(0x69)
	{ KEY_KP_MULTIPLY, VK_MULTIPLY }, // (0x6A)
	{ KEY_KP_ADD, VK_ADD }, // (0x6B)
	//VK_SEPARATOR (0x6C)
	{ KEY_KP_SUBTRACT, VK_SUBTRACT }, // (0x6D)
	{ KEY_KP_PERIOD, VK_DECIMAL }, // (0x6E)
	{ KEY_KP_DIVIDE, VK_DIVIDE }, // (0x6F)
	{ KEY_F1, VK_F1 }, // (0x70)
	{ KEY_F2, VK_F2 }, // (0x71)
	{ KEY_F3, VK_F3 }, // (0x72)
	{ KEY_F4, VK_F4 }, // (0x73)
	{ KEY_F5, VK_F5 }, // (0x74)
	{ KEY_F6, VK_F6 }, // (0x75)
	{ KEY_F7, VK_F7 }, // (0x76)
	{ KEY_F8, VK_F8 }, // (0x77)
	{ KEY_F9, VK_F9 }, // (0x78)
	{ KEY_F10, VK_F10 }, // (0x79)
	{ KEY_F11, VK_F11 }, // (0x7A)
	{ KEY_F12, VK_F12 }, // (0x7B)
	{ KEY_F13, VK_F13 }, // (0x7C)
	{ KEY_F14, VK_F14 }, // (0x7D)
	{ KEY_F15, VK_F15 }, // (0x7E)
	{ KEY_F16, VK_F16 }, // (0x7F)
	{ KEY_NUMLOCK, VK_NUMLOCK }, // (0x90)
	{ KEY_SCROLLLOCK, VK_SCROLL }, // (0x91)
	{ KEY_SHIFT, VK_LSHIFT }, // (0xA0)
	{ KEY_SHIFT, VK_RSHIFT }, // (0xA1)
	{ KEY_CTRL, VK_LCONTROL }, // (0xA2)
	{ KEY_CTRL, VK_RCONTROL }, // (0xA3)
	{ KEY_MENU, VK_LMENU }, // (0xA4)
	{ KEY_MENU, VK_RMENU }, // (0xA5)

	{ KEY_BACK, VK_BROWSER_BACK }, // (0xA6)

	{ KEY_FORWARD, VK_BROWSER_FORWARD }, // (0xA7)

	{ KEY_REFRESH, VK_BROWSER_REFRESH }, // (0xA8)

	{ KEY_STOP, VK_BROWSER_STOP }, // (0xA9)

	{ KEY_SEARCH, VK_BROWSER_SEARCH }, // (0xAA)

	{ KEY_FAVORITES, VK_BROWSER_FAVORITES }, // (0xAB)

	{ KEY_HOMEPAGE, VK_BROWSER_HOME }, // (0xAC)

	{ KEY_VOLUMEMUTE, VK_VOLUME_MUTE }, // (0xAD)

	{ KEY_VOLUMEDOWN, VK_VOLUME_DOWN }, // (0xAE)

	{ KEY_VOLUMEUP, VK_VOLUME_UP }, // (0xAF)

	{ KEY_MEDIANEXT, VK_MEDIA_NEXT_TRACK }, // (0xB0)

	{ KEY_MEDIAPREVIOUS, VK_MEDIA_PREV_TRACK }, // (0xB1)

	{ KEY_MEDIASTOP, VK_MEDIA_STOP }, // (0xB2)

	//VK_MEDIA_PLAY_PAUSE (0xB3)

	{ KEY_LAUNCHMAIL, VK_LAUNCH_MAIL }, // (0xB4)

	{ KEY_LAUNCHMEDIA, VK_LAUNCH_MEDIA_SELECT }, // (0xB5)

	{ KEY_LAUNCH0, VK_LAUNCH_APP1 }, // (0xB6)

	{ KEY_LAUNCH1, VK_LAUNCH_APP2 }, // (0xB7)

	{ KEY_SEMICOLON, VK_OEM_1 }, // (0xBA)

	{ KEY_EQUAL, VK_OEM_PLUS }, // (0xBB) // Windows 2000/XP: For any country/region, the '+' key
	{ KEY_COMMA, VK_OEM_COMMA }, // (0xBC) // Windows 2000/XP: For any country/region, the ',' key
	{ KEY_MINUS, VK_OEM_MINUS }, // (0xBD) // Windows 2000/XP: For any country/region, the '-' key
	{ KEY_PERIOD, VK_OEM_PERIOD }, // (0xBE) // Windows 2000/XP: For any country/region, the '.' key
	{ KEY_SLASH, VK_OEM_2 }, // (0xBF) //Windows 2000/XP: For the US standard keyboard, the '/?' key

	{ KEY_QUOTELEFT, VK_OEM_3 }, // (0xC0)
	{ KEY_BRACELEFT, VK_OEM_4 }, // (0xDB)
	{ KEY_BACKSLASH, VK_OEM_5 }, // (0xDC)
	{ KEY_BRACERIGHT, VK_OEM_6 }, // (0xDD)
	{ KEY_APOSTROPHE, VK_OEM_7 }, // (0xDE)
	/*
{VK_OEM_8 (0xDF)
{VK_OEM_102 (0xE2) // Windows 2000/XP: Either the angle bracket key or the backslash key on the RT 102-key keyboard
*/
	//{ KEY_PLAY, VK_PLAY},// (0xFA)

	{ KEY_UNKNOWN, 0 }
};

/*
VK_ZOOM (0xFB)
VK_NONAME (0xFC)
VK_PA1 (0xFD)
VK_OEM_CLEAR (0xFE)
*/

static _WinTranslatePair _scancode_to_keycode[] = {
	{ KEY_ESCAPE, 0x01 },
	{ KEY_1, 0x02 },
	{ KEY_2, 0x03 },
	{ KEY_3, 0x04 },
	{ KEY_4, 0x05 },
	{ KEY_5, 0x06 },
	{ KEY_6, 0x07 },
	{ KEY_7, 0x08 },
	{ KEY_8, 0x09 },
	{ KEY_9, 0x0A },
	{ KEY_0, 0x0B },
	{ KEY_MINUS, 0x0C },
	{ KEY_EQUAL, 0x0D },
	{ KEY_BACKSPACE, 0x0E },
	{ KEY_TAB, 0x0F },
	{ KEY_Q, 0x10 },
	{ KEY_W, 0x11 },
	{ KEY_E, 0x12 },
	{ KEY_R, 0x13 },
	{ KEY_T, 0x14 },
	{ KEY_Y, 0x15 },
	{ KEY_U, 0x16 },
	{ KEY_I, 0x17 },
	{ KEY_O, 0x18 },
	{ KEY_P, 0x19 },
	{ KEY_BRACELEFT, 0x1A },
	{ KEY_BRACERIGHT, 0x1B },
	{ KEY_ENTER, 0x1C },
	{ KEY_CTRL, 0x1D },
	{ KEY_A, 0x1E },
	{ KEY_S, 0x1F },
	{ KEY_D, 0x20 },
	{ KEY_F, 0x21 },
	{ KEY_G, 0x22 },
	{ KEY_H, 0x23 },
	{ KEY_J, 0x24 },
	{ KEY_K, 0x25 },
	{ KEY_L, 0x26 },
	{ KEY_SEMICOLON, 0x27 },
	{ KEY_APOSTROPHE, 0x28 },
	{ KEY_QUOTELEFT, 0x29 },
	{ KEY_SHIFT, 0x2A },
	{ KEY_BACKSLASH, 0x2B },
	{ KEY_Z, 0x2C },
	{ KEY_X, 0x2D },
	{ KEY_C, 0x2E },
	{ KEY_V, 0x2F },
	{ KEY_B, 0x30 },
	{ KEY_N, 0x31 },
	{ KEY_M, 0x32 },
	{ KEY_COMMA, 0x33 },
	{ KEY_PERIOD, 0x34 },
	{ KEY_SLASH, 0x35 },
	{ KEY_SHIFT, 0x36 },
	{ KEY_PRINT, 0x37 },
	{ KEY_ALT, 0x38 },
	{ KEY_SPACE, 0x39 },
	{ KEY_CAPSLOCK, 0x3A },
	{ KEY_F1, 0x3B },
	{ KEY_F2, 0x3C },
	{ KEY_F3, 0x3D },
	{ KEY_F4, 0x3E },
	{ KEY_F5, 0x3F },
	{ KEY_F6, 0x40 },
	{ KEY_F7, 0x41 },
	{ KEY_F8, 0x42 },
	{ KEY_F9, 0x43 },
	{ KEY_F10, 0x44 },
	{ KEY_NUMLOCK, 0x45 },
	{ KEY_SCROLLLOCK, 0x46 },
	{ KEY_HOME, 0x47 },
	{ KEY_UP, 0x48 },
	{ KEY_PAGEUP, 0x49 },
	{ KEY_KP_SUBTRACT, 0x4A },
	{ KEY_LEFT, 0x4B },
	{ KEY_KP_5, 0x4C },
	{ KEY_RIGHT, 0x4D },
	{ KEY_KP_ADD, 0x4E },
	{ KEY_END, 0x4F },
	{ KEY_DOWN, 0x50 },
	{ KEY_PAGEDOWN, 0x51 },
	{ KEY_INSERT, 0x52 },
	{ KEY_DELETE, 0x53 },
	//{ KEY_???, 0x56 }, //NON US BACKSLASH
	{ KEY_F11, 0x57 },
	{ KEY_F12, 0x58 },
	{ KEY_META, 0x5B },
	{ KEY_META, 0x5C },
	{ KEY_MENU, 0x5D },
	{ KEY_F13, 0x64 },
	{ KEY_F14, 0x65 },
	{ KEY_F15, 0x66 },
	{ KEY_F16, 0x67 },
	{ KEY_UNKNOWN, 0 }
};

unsigned int KeyMappingWindows::get_keysym(unsigned int p_code) {
	for (int i = 0; _vk_to_keycode[i].keysym != KEY_UNKNOWN; i++) {
		if (_vk_to_keycode[i].keycode == p_code) {
			//printf("outcode: %x\n",_vk_to_keycode[i].keysym);

			return _vk_to_keycode[i].keysym;
		}
	}

	return KEY_UNKNOWN;
}

unsigned int KeyMappingWindows::get_scansym(unsigned int p_code, bool p_extended) {
	unsigned int keycode = KEY_UNKNOWN;
	for (int i = 0; _scancode_to_keycode[i].keysym != KEY_UNKNOWN; i++) {
		if (_scancode_to_keycode[i].keycode == p_code) {
			keycode = _scancode_to_keycode[i].keysym;
			break;
		}
	}

	if (p_extended) {
		switch (keycode) {
			case KEY_ENTER: {
				keycode = KEY_KP_ENTER;
			} break;
			case KEY_SLASH: {
				keycode = KEY_KP_DIVIDE;
			} break;
			case KEY_CAPSLOCK: {
				keycode = KEY_KP_ADD;
			} break;
		}
	} else {
		switch (keycode) {
			case KEY_NUMLOCK: {
				keycode = KEY_PAUSE;
			} break;
			case KEY_HOME: {
				keycode = KEY_KP_7;
			} break;
			case KEY_UP: {
				keycode = KEY_KP_8;
			} break;
			case KEY_PAGEUP: {
				keycode = KEY_KP_9;
			} break;
			case KEY_LEFT: {
				keycode = KEY_KP_4;
			} break;
			case KEY_RIGHT: {
				keycode = KEY_KP_6;
			} break;
			case KEY_END: {
				keycode = KEY_KP_1;
			} break;
			case KEY_DOWN: {
				keycode = KEY_KP_2;
			} break;
			case KEY_PAGEDOWN: {
				keycode = KEY_KP_3;
			} break;
			case KEY_INSERT: {
				keycode = KEY_KP_0;
			} break;
			case KEY_DELETE: {
				keycode = KEY_KP_PERIOD;
			} break;
			case KEY_PRINT: {
				keycode = KEY_KP_MULTIPLY;
			} break;
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
