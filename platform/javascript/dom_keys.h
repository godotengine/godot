/*************************************************************************/
/*  dom_keys.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef DOM_KEYS_H
#define DOM_KEYS_H

#include "os/keyboard.h"

// https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/keyCode#Constants_for_keyCode_value
#define DOM_VK_CANCEL 0x03
#define DOM_VK_HELP 0x06
#define DOM_VK_BACK_SPACE 0x08
#define DOM_VK_TAB 0x09
#define DOM_VK_CLEAR 0x0C
#define DOM_VK_RETURN 0x0D
#define DOM_VK_ENTER 0x0E // "Reserved, but not used."
#define DOM_VK_SHIFT 0x10
#define DOM_VK_CONTROL 0x11
#define DOM_VK_ALT 0x12
#define DOM_VK_PAUSE 0x13
#define DOM_VK_CAPS_LOCK 0x14
#define DOM_VK_KANA 0x15
#define DOM_VK_HANGUL 0x15
#define DOM_VK_EISU 0x16
#define DOM_VK_JUNJA 0x17
#define DOM_VK_FINAL 0x18
#define DOM_VK_HANJA 0x19
#define DOM_VK_KANJI 0x19
#define DOM_VK_ESCAPE 0x1B
#define DOM_VK_CONVERT 0x1C
#define DOM_VK_NONCONVERT 0x1D
#define DOM_VK_ACCEPT 0x1E
#define DOM_VK_MODECHANGE 0x1F
#define DOM_VK_SPACE 0x20
#define DOM_VK_PAGE_UP 0x21
#define DOM_VK_PAGE_DOWN 0x22
#define DOM_VK_END 0x23
#define DOM_VK_HOME 0x24
#define DOM_VK_LEFT 0x25
#define DOM_VK_UP 0x26
#define DOM_VK_RIGHT 0x27
#define DOM_VK_DOWN 0x28
#define DOM_VK_SELECT 0x29
#define DOM_VK_PRINT 0x2A
#define DOM_VK_EXECUTE 0x2B
#define DOM_VK_PRINTSCREEN 0x2C
#define DOM_VK_INSERT 0x2D
#define DOM_VK_DELETE 0x2E
#define DOM_VK_0 0x30
#define DOM_VK_1 0x31
#define DOM_VK_2 0x32
#define DOM_VK_3 0x33
#define DOM_VK_4 0x34
#define DOM_VK_5 0x35
#define DOM_VK_6 0x36
#define DOM_VK_7 0x37
#define DOM_VK_8 0x38
#define DOM_VK_9 0x39
#define DOM_VK_COLON 0x3A
#define DOM_VK_SEMICOLON 0x3B
#define DOM_VK_LESS_THAN 0x3C
#define DOM_VK_EQUALS 0x3D
#define DOM_VK_GREATER_THAN 0x3E
#define DOM_VK_QUESTION_MARK 0x3F
#define DOM_VK_AT 0x40
#define DOM_VK_A 0x41
#define DOM_VK_B 0x42
#define DOM_VK_C 0x43
#define DOM_VK_D 0x44
#define DOM_VK_E 0x45
#define DOM_VK_F 0x46
#define DOM_VK_G 0x47
#define DOM_VK_H 0x48
#define DOM_VK_I 0x49
#define DOM_VK_J 0x4A
#define DOM_VK_K 0x4B
#define DOM_VK_L 0x4C
#define DOM_VK_M 0x4D
#define DOM_VK_N 0x4E
#define DOM_VK_O 0x4F
#define DOM_VK_P 0x50
#define DOM_VK_Q 0x51
#define DOM_VK_R 0x52
#define DOM_VK_S 0x53
#define DOM_VK_T 0x54
#define DOM_VK_U 0x55
#define DOM_VK_V 0x56
#define DOM_VK_W 0x57
#define DOM_VK_X 0x58
#define DOM_VK_Y 0x59
#define DOM_VK_Z 0x5A
#define DOM_VK_WIN 0x5B
#define DOM_VK_CONTEXT_MENU 0x5D
#define DOM_VK_SLEEP 0x5F
#define DOM_VK_NUMPAD0 0x60
#define DOM_VK_NUMPAD1 0x61
#define DOM_VK_NUMPAD2 0x62
#define DOM_VK_NUMPAD3 0x63
#define DOM_VK_NUMPAD4 0x64
#define DOM_VK_NUMPAD5 0x65
#define DOM_VK_NUMPAD6 0x66
#define DOM_VK_NUMPAD7 0x67
#define DOM_VK_NUMPAD8 0x68
#define DOM_VK_NUMPAD9 0x69
#define DOM_VK_MULTIPLY 0x6A
#define DOM_VK_ADD 0x6B
#define DOM_VK_SEPARATOR 0x6C
#define DOM_VK_SUBTRACT 0x6D
#define DOM_VK_DECIMAL 0x6E
#define DOM_VK_DIVIDE 0x6F
#define DOM_VK_F1 0x70
#define DOM_VK_F2 0x71
#define DOM_VK_F3 0x72
#define DOM_VK_F4 0x73
#define DOM_VK_F5 0x74
#define DOM_VK_F6 0x75
#define DOM_VK_F7 0x76
#define DOM_VK_F8 0x77
#define DOM_VK_F9 0x78
#define DOM_VK_F10 0x79
#define DOM_VK_F11 0x7A
#define DOM_VK_F12 0x7B
#define DOM_VK_F13 0x7C
#define DOM_VK_F14 0x7D
#define DOM_VK_F15 0x7E
#define DOM_VK_F16 0x7F
#define DOM_VK_F17 0x80
#define DOM_VK_F18 0x81
#define DOM_VK_F19 0x82
#define DOM_VK_F20 0x83
#define DOM_VK_F21 0x84
#define DOM_VK_F22 0x85
#define DOM_VK_F23 0x86
#define DOM_VK_F24 0x87
#define DOM_VK_NUM_LOCK 0x90
#define DOM_VK_SCROLL_LOCK 0x91
#define DOM_VK_WIN_OEM_FJ_JISHO 0x92
#define DOM_VK_WIN_OEM_FJ_MASSHOU 0x93
#define DOM_VK_WIN_OEM_FJ_TOUROKU 0x94
#define DOM_VK_WIN_OEM_FJ_LOYA 0x95
#define DOM_VK_WIN_OEM_FJ_ROYA 0x96
#define DOM_VK_CIRCUMFLEX 0xA0
#define DOM_VK_EXCLAMATION 0xA1
#define DOM_VK_DOUBLE_QUOTE 0xA2
#define DOM_VK_HASH 0xA3
#define DOM_VK_DOLLAR 0xA4
#define DOM_VK_PERCENT 0xA5
#define DOM_VK_AMPERSAND 0xA6
#define DOM_VK_UNDERSCORE 0xA7
#define DOM_VK_OPEN_PAREN 0xA8
#define DOM_VK_CLOSE_PAREN 0xA9
#define DOM_VK_ASTERISK 0xAA
#define DOM_VK_PLUS 0xAB
#define DOM_VK_PIPE 0xAC
#define DOM_VK_HYPHEN_MINUS 0xAD
#define DOM_VK_OPEN_CURLY_BRACKET 0xAE
#define DOM_VK_CLOSE_CURLY_BRACKET 0xAF
#define DOM_VK_TILDE 0xB0
#define DOM_VK_VOLUME_MUTE 0xB5
#define DOM_VK_VOLUME_DOWN 0xB6
#define DOM_VK_VOLUME_UP 0xB7
#define DOM_VK_COMMA 0xBC
#define DOM_VK_PERIOD 0xBE
#define DOM_VK_SLASH 0xBF
#define DOM_VK_BACK_QUOTE 0xC0
#define DOM_VK_OPEN_BRACKET 0xDB
#define DOM_VK_BACK_SLASH 0xDC
#define DOM_VK_CLOSE_BRACKET 0xDD
#define DOM_VK_QUOTE 0xDE
#define DOM_VK_META 0xE0
#define DOM_VK_ALTGR 0xE1
#define DOM_VK_WIN_ICO_HELP 0xE3
#define DOM_VK_WIN_ICO_00 0xE4
#define DOM_VK_WIN_ICO_CLEAR 0xE6
#define DOM_VK_WIN_OEM_RESET 0xE9
#define DOM_VK_WIN_OEM_JUMP 0xEA
#define DOM_VK_WIN_OEM_PA1 0xEB
#define DOM_VK_WIN_OEM_PA2 0xEC
#define DOM_VK_WIN_OEM_PA3 0xED
#define DOM_VK_WIN_OEM_WSCTRL 0xEE
#define DOM_VK_WIN_OEM_CUSEL 0xEF
#define DOM_VK_WIN_OEM_ATTN 0xF0
#define DOM_VK_WIN_OEM_FINISH 0xF1
#define DOM_VK_WIN_OEM_COPY 0xF2
#define DOM_VK_WIN_OEM_AUTO 0xF3
#define DOM_VK_WIN_OEM_ENLW 0xF4
#define DOM_VK_WIN_OEM_BACKTAB 0xF5
#define DOM_VK_ATTN 0xF6
#define DOM_VK_CRSEL 0xF7
#define DOM_VK_EXSEL 0xF8
#define DOM_VK_EREOF 0xF9
#define DOM_VK_PLAY 0xFA
#define DOM_VK_ZOOM 0xFB
#define DOM_VK_PA1 0xFD
#define DOM_VK_WIN_OEM_CLEAR 0xFE

int dom2godot_scancode(int dom_keycode) {

	if (DOM_VK_0 <= dom_keycode && dom_keycode <= DOM_VK_Z) {
		// ASCII intersection
		return dom_keycode;
	}

	if (DOM_VK_NUMPAD0 <= dom_keycode && dom_keycode <= DOM_VK_NUMPAD9) {
		// Numpad numbers
		return KEY_KP_0 + (dom_keycode - DOM_VK_NUMPAD0);
	}

	if (DOM_VK_F1 <= dom_keycode && dom_keycode <= DOM_VK_F16) {
		// F1-F16
		return KEY_F1 + (dom_keycode - DOM_VK_F1);
	}

	switch (dom_keycode) {
		//case DOM_VK_CANCEL: return KEY_UNKNOWN;
		case DOM_VK_HELP: return KEY_HELP;
		case DOM_VK_BACK_SPACE: return KEY_BACKSPACE;
		case DOM_VK_TAB: return KEY_TAB;

		case DOM_VK_CLEAR:
		case DOM_VK_WIN_OEM_CLEAR: // OEM duplicate
			return KEY_CLEAR;

		case DOM_VK_RETURN:
		case DOM_VK_ENTER: // unused according to MDN
			return KEY_ENTER;

		case DOM_VK_SHIFT: return KEY_SHIFT;
		case DOM_VK_CONTROL: return KEY_CONTROL;

		case DOM_VK_ALT:
		case DOM_VK_ALTGR:
			return KEY_ALT;

		case DOM_VK_PAUSE: return KEY_PAUSE;
		case DOM_VK_CAPS_LOCK:
			return KEY_CAPSLOCK;

			/*
		case DOM_VK_KANA: return KEY_UNKNOWN;
		case DOM_VK_HANGUL: return KEY_UNKNOWN;
		case DOM_VK_EISU: return KEY_UNKNOWN;
		case DOM_VK_JUNJA: return KEY_UNKNOWN;
		case DOM_VK_FINAL: return KEY_UNKNOWN;
		case DOM_VK_HANJA: return KEY_UNKNOWN;
		case DOM_VK_KANJI: return KEY_UNKNOWN;
		*/

		case DOM_VK_ESCAPE:
			return KEY_ESCAPE;
			/*
		case DOM_VK_CONVERT: return KEY_UNKNOWN;
		case DOM_VK_NONCONVERT: return KEY_UNKNOWN;
		case DOM_VK_ACCEPT: return KEY_UNKNOWN;
		case DOM_VK_MODECHANGE: return KEY_UNKNOWN;
		*/

		case DOM_VK_SPACE: return KEY_SPACE;
		case DOM_VK_PAGE_UP: return KEY_PAGEUP;
		case DOM_VK_PAGE_DOWN: return KEY_PAGEDOWN;
		case DOM_VK_END: return KEY_END;
		case DOM_VK_HOME: return KEY_HOME;
		case DOM_VK_LEFT: return KEY_LEFT;
		case DOM_VK_UP: return KEY_UP;
		case DOM_VK_RIGHT: return KEY_RIGHT;
		case DOM_VK_DOWN:
			return KEY_DOWN;

			//case DOM_VK_SELECT: return KEY_UNKNOWN;

		case DOM_VK_PRINTSCREEN: // this is the usual printScreen key
		case DOM_VK_PRINT: // maybe for alt+printScreen or physical printers?
			return KEY_PRINT;

		//case DOM_VK_EXECUTE: return KEY_UNKNOWN;
		case DOM_VK_INSERT: return KEY_INSERT;
		case DOM_VK_DELETE: return KEY_DELETE;

		case DOM_VK_META:
		case DOM_VK_WIN:
			return KEY_META;

		case DOM_VK_CONTEXT_MENU: return KEY_MENU;
		case DOM_VK_SLEEP:
			return KEY_STANDBY;

		// these are numpad keys according to MDN
		case DOM_VK_MULTIPLY: return KEY_KP_MULTIPLY;
		case DOM_VK_ADD: return KEY_KP_ADD;
		case DOM_VK_SEPARATOR:
			return KEY_KP_PERIOD; // good enough?
		case DOM_VK_SUBTRACT: return KEY_KP_SUBTRACT;
		case DOM_VK_DECIMAL: return KEY_KP_PERIOD;
		case DOM_VK_DIVIDE:
			return KEY_KP_DIVIDE;

			/*
		case DOM_VK_F17: return KEY_UNKNOWN;
		case DOM_VK_F18: return KEY_UNKNOWN;
		case DOM_VK_F19: return KEY_UNKNOWN;
		case DOM_VK_F20: return KEY_UNKNOWN;
		case DOM_VK_F21: return KEY_UNKNOWN;
		case DOM_VK_F22: return KEY_UNKNOWN;
		case DOM_VK_F23: return KEY_UNKNOWN;
		case DOM_VK_F24: return KEY_UNKNOWN;
		*/

		case DOM_VK_NUM_LOCK: return KEY_NUMLOCK;
		case DOM_VK_SCROLL_LOCK:
			return KEY_SCROLLLOCK;

			/*
		case DOM_VK_WIN_OEM_FJ_JISHO: return KEY_UNKNOWN;
		case DOM_VK_WIN_OEM_FJ_MASSHOU: return KEY_UNKNOWN;
		case DOM_VK_WIN_OEM_FJ_TOUROKU: return KEY_UNKNOWN;
		case DOM_VK_WIN_OEM_FJ_LOYA: return KEY_UNKNOWN;
		case DOM_VK_WIN_OEM_FJ_ROYA: return KEY_UNKNOWN;
		*/

		case DOM_VK_CIRCUMFLEX: return KEY_ASCIICIRCUM;
		case DOM_VK_EXCLAMATION: return KEY_EXCLAM;
		case DOM_VK_DOUBLE_QUOTE: return KEY_QUOTEDBL;
		case DOM_VK_HASH: return KEY_NUMBERSIGN;
		case DOM_VK_DOLLAR: return KEY_DOLLAR;
		case DOM_VK_PERCENT: return KEY_PERCENT;
		case DOM_VK_AMPERSAND: return KEY_AMPERSAND;
		case DOM_VK_UNDERSCORE: return KEY_UNDERSCORE;
		case DOM_VK_OPEN_PAREN: return KEY_PARENLEFT;
		case DOM_VK_CLOSE_PAREN: return KEY_PARENRIGHT;
		case DOM_VK_ASTERISK: return KEY_ASTERISK;
		case DOM_VK_PLUS: return KEY_PLUS;
		case DOM_VK_PIPE: return KEY_BAR;
		case DOM_VK_HYPHEN_MINUS: return KEY_MINUS;
		case DOM_VK_OPEN_CURLY_BRACKET: return KEY_BRACELEFT;
		case DOM_VK_CLOSE_CURLY_BRACKET: return KEY_BRACERIGHT;
		case DOM_VK_TILDE: return KEY_ASCIITILDE;

		case DOM_VK_VOLUME_MUTE: return KEY_VOLUMEMUTE;
		case DOM_VK_VOLUME_DOWN: return KEY_VOLUMEDOWN;
		case DOM_VK_VOLUME_UP: return KEY_VOLUMEUP;

		case DOM_VK_COMMA: return KEY_COMMA;
		case DOM_VK_PERIOD: return KEY_PERIOD;
		case DOM_VK_SLASH: return KEY_SLASH;
		case DOM_VK_BACK_QUOTE: return KEY_QUOTELEFT;
		case DOM_VK_OPEN_BRACKET: return KEY_BRACKETLEFT;
		case DOM_VK_BACK_SLASH: return KEY_BACKSLASH;
		case DOM_VK_CLOSE_BRACKET: return KEY_BRACKETRIGHT;
		case DOM_VK_QUOTE:
			return KEY_APOSTROPHE;

			// rest is OEM/unusual

		default: return KEY_UNKNOWN;
	};
}

#endif
