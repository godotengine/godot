/*************************************************************************/
/*  nacl_keycodes.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
 * Copyright (C) 2006 Michael Emmel mike.emmel@gmail.com. All rights reserved.
 * Copyright (C) 2008, 2009 Google Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY APPLE COMPUTER, INC. ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL APPLE COMPUTER, INC. OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES, LOSS OF USE, DATA, OR
 * PROFITS, OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef BASE_KEYBOARD_CODES_POSIX_H_
#define BASE_KEYBOARD_CODES_POSIX_H_
#pragma once

#include "core/os/keyboard.h"

enum {
  VKEY_BACK = 0x08,
  VKEY_TAB = 0x09,
  VKEY_CLEAR = 0x0C,
  VKEY_RETURN = 0x0D,
  VKEY_SHIFT = 0x10,
  VKEY_CONTROL = 0x11,
  VKEY_MENU = 0x12,
  VKEY_PAUSE = 0x13,
  VKEY_CAPITAL = 0x14,
  VKEY_KANA = 0x15,
  VKEY_HANGUL = 0x15,
  VKEY_JUNJA = 0x17,
  VKEY_FINAL = 0x18,
  VKEY_HANJA = 0x19,
  VKEY_KANJI = 0x19,
  VKEY_ESCAPE = 0x1B,
  VKEY_CONVERT = 0x1C,
  VKEY_NONCONVERT = 0x1D,
  VKEY_ACCEPT = 0x1E,
  VKEY_MODECHANGE = 0x1F,
  VKEY_SPACE = 0x20,
  VKEY_PRIOR = 0x21,
  VKEY_NEXT = 0x22,
  VKEY_END = 0x23,
  VKEY_HOME = 0x24,
  VKEY_LEFT = 0x25,
  VKEY_UP = 0x26,
  VKEY_RIGHT = 0x27,
  VKEY_DOWN = 0x28,
  VKEY_SELECT = 0x29,
  VKEY_PRINT = 0x2A,
  VKEY_EXECUTE = 0x2B,
  VKEY_SNAPSHOT = 0x2C,
  VKEY_INSERT = 0x2D,
  VKEY_DELETE = 0x2E,
  VKEY_HELP = 0x2F,
  VKEY_0 = 0x30,
  VKEY_1 = 0x31,
  VKEY_2 = 0x32,
  VKEY_3 = 0x33,
  VKEY_4 = 0x34,
  VKEY_5 = 0x35,
  VKEY_6 = 0x36,
  VKEY_7 = 0x37,
  VKEY_8 = 0x38,
  VKEY_9 = 0x39,
  VKEY_A = 0x41,
  VKEY_B = 0x42,
  VKEY_C = 0x43,
  VKEY_D = 0x44,
  VKEY_E = 0x45,
  VKEY_F = 0x46,
  VKEY_G = 0x47,
  VKEY_H = 0x48,
  VKEY_I = 0x49,
  VKEY_J = 0x4A,
  VKEY_K = 0x4B,
  VKEY_L = 0x4C,
  VKEY_M = 0x4D,
  VKEY_N = 0x4E,
  VKEY_O = 0x4F,
  VKEY_P = 0x50,
  VKEY_Q = 0x51,
  VKEY_R = 0x52,
  VKEY_S = 0x53,
  VKEY_T = 0x54,
  VKEY_U = 0x55,
  VKEY_V = 0x56,
  VKEY_W = 0x57,
  VKEY_X = 0x58,
  VKEY_Y = 0x59,
  VKEY_Z = 0x5A,
  VKEY_LWIN = 0x5B,
  VKEY_COMMAND = VKEY_LWIN,  // Provide the Mac name for convenience.
  VKEY_RWIN = 0x5C,
  VKEY_APPS = 0x5D,
  VKEY_SLEEP = 0x5F,
  VKEY_NUMPAD0 = 0x60,
  VKEY_NUMPAD1 = 0x61,
  VKEY_NUMPAD2 = 0x62,
  VKEY_NUMPAD3 = 0x63,
  VKEY_NUMPAD4 = 0x64,
  VKEY_NUMPAD5 = 0x65,
  VKEY_NUMPAD6 = 0x66,
  VKEY_NUMPAD7 = 0x67,
  VKEY_NUMPAD8 = 0x68,
  VKEY_NUMPAD9 = 0x69,
  VKEY_MULTIPLY = 0x6A,
  VKEY_ADD = 0x6B,
  VKEY_SEPARATOR = 0x6C,
  VKEY_SUBTRACT = 0x6D,
  VKEY_DECIMAL = 0x6E,
  VKEY_DIVIDE = 0x6F,
  VKEY_F1 = 0x70,
  VKEY_F2 = 0x71,
  VKEY_F3 = 0x72,
  VKEY_F4 = 0x73,
  VKEY_F5 = 0x74,
  VKEY_F6 = 0x75,
  VKEY_F7 = 0x76,
  VKEY_F8 = 0x77,
  VKEY_F9 = 0x78,
  VKEY_F10 = 0x79,
  VKEY_F11 = 0x7A,
  VKEY_F12 = 0x7B,
  VKEY_F13 = 0x7C,
  VKEY_F14 = 0x7D,
  VKEY_F15 = 0x7E,
  VKEY_F16 = 0x7F,
  VKEY_F17 = 0x80,
  VKEY_F18 = 0x81,
  VKEY_F19 = 0x82,
  VKEY_F20 = 0x83,
  VKEY_F21 = 0x84,
  VKEY_F22 = 0x85,
  VKEY_F23 = 0x86,
  VKEY_F24 = 0x87,
  VKEY_NUMLOCK = 0x90,
  VKEY_SCROLL = 0x91,
  VKEY_LSHIFT = 0xA0,
  VKEY_RSHIFT = 0xA1,
  VKEY_LCONTROL = 0xA2,
  VKEY_RCONTROL = 0xA3,
  VKEY_LMENU = 0xA4,
  VKEY_RMENU = 0xA5,
  VKEY_BROWSER_BACK = 0xA6,
  VKEY_BROWSER_FORWARD = 0xA7,
  VKEY_BROWSER_REFRESH = 0xA8,
  VKEY_BROWSER_STOP = 0xA9,
  VKEY_BROWSER_SEARCH = 0xAA,
  VKEY_BROWSER_FAVORITES = 0xAB,
  VKEY_BROWSER_HOME = 0xAC,
  VKEY_VOLUME_MUTE = 0xAD,
  VKEY_VOLUME_DOWN = 0xAE,
  VKEY_VOLUME_UP = 0xAF,
  VKEY_MEDIA_NEXT_TRACK = 0xB0,
  VKEY_MEDIA_PREV_TRACK = 0xB1,
  VKEY_MEDIA_STOP = 0xB2,
  VKEY_MEDIA_PLAY_PAUSE = 0xB3,
  VKEY_MEDIA_LAUNCH_MAIL = 0xB4,
  VKEY_MEDIA_LAUNCH_MEDIA_SELECT = 0xB5,
  VKEY_MEDIA_LAUNCH_APP1 = 0xB6,
  VKEY_MEDIA_LAUNCH_APP2 = 0xB7,
  VKEY_OEM_1 = 0xBA,
  VKEY_OEM_PLUS = 0xBB,
  VKEY_OEM_COMMA = 0xBC,
  VKEY_OEM_MINUS = 0xBD,
  VKEY_OEM_PERIOD = 0xBE,
  VKEY_OEM_2 = 0xBF,
  VKEY_OEM_3 = 0xC0,
  VKEY_OEM_4 = 0xDB,
  VKEY_OEM_5 = 0xDC,
  VKEY_OEM_6 = 0xDD,
  VKEY_OEM_7 = 0xDE,
  VKEY_OEM_8 = 0xDF,
  VKEY_OEM_102 = 0xE2,
  VKEY_PROCESSKEY = 0xE5,
  VKEY_PACKET = 0xE7,
  VKEY_ATTN = 0xF6,
  VKEY_CRSEL = 0xF7,
  VKEY_EXSEL = 0xF8,
  VKEY_EREOF = 0xF9,
  VKEY_PLAY = 0xFA,
  VKEY_ZOOM = 0xFB,
  VKEY_NONAME = 0xFC,
  VKEY_PA1 = 0xFD,
  VKEY_OEM_CLEAR = 0xFE,
  VKEY_UNKNOWN = 0
};

static uint32_t godot_key(uint32_t p_key, bool& is_char) {

	is_char = false;

	switch (p_key) {

	case VKEY_BACK: return KEY_BACKSPACE;
	case VKEY_TAB: return KEY_TAB;
	case VKEY_CLEAR: return KEY_CLEAR;
	case VKEY_RETURN: return KEY_RETURN;
	case VKEY_SHIFT: return KEY_SHIFT;
	case VKEY_CONTROL: return KEY_CONTROL;
	case VKEY_MENU: return KEY_MENU;
	case VKEY_PAUSE: return KEY_PAUSE;
//	case VKEY_CAPITAL: return KEY_CAPITAL;
//	case VKEY_KANA: return KEY_KANA;
//	case VKEY_HANGUL: return KEY_HANGUL;
//	case VKEY_JUNJA: return KEY_JUNJA;
//	case VKEY_FINAL: return KEY_FINAL;
//	case VKEY_HANJA: return KEY_HANJA;
//	case VKEY_KANJI: return KEY_KANJI;
	case VKEY_ESCAPE: return KEY_ESCAPE;
//	case VKEY_CONVERT: return KEY_CONVERT;
//	case VKEY_NONCONVERT: return KEY_NONCONVERT;
//	case VKEY_ACCEPT: return KEY_ACCEPT;
//	case VKEY_MODECHANGE: return KEY_MODECHANGE;
//	case VKEY_PRIOR: return KEY_PRIOR;
//	case VKEY_NEXT: return KEY_NEXT;
	case VKEY_END: return KEY_END;
	case VKEY_HOME: return KEY_HOME;
	case VKEY_LEFT: return KEY_LEFT;
	case VKEY_UP: return KEY_UP;
	case VKEY_RIGHT: return KEY_RIGHT;
	case VKEY_DOWN: return KEY_DOWN;
//	case VKEY_SELECT: return KEY_SELECT;
	case VKEY_PRINT: return KEY_PRINT;
//	case VKEY_EXECUTE: return KEY_EXECUTE;
//	case VKEY_SNAPSHOT: return KEY_SNAPSHOT;
	case VKEY_INSERT: return KEY_INSERT;
	case VKEY_DELETE: return KEY_DELETE;
	case VKEY_HELP: return KEY_HELP;
//	case VKEY_LWIN: return KEY_LWIN;
//	case VKEY_RWIN: return KEY_RWIN;
//	case VKEY_APPS: return KEY_APPS;
//	case VKEY_SLEEP: return KEY_SLEEP;
	case VKEY_NUMPAD0: return KEY_KP_0;
	case VKEY_NUMPAD1: return KEY_KP_1;
	case VKEY_NUMPAD2: return KEY_KP_2;
	case VKEY_NUMPAD3: return KEY_KP_3;
	case VKEY_NUMPAD4: return KEY_KP_4;
	case VKEY_NUMPAD5: return KEY_KP_5;
	case VKEY_NUMPAD6: return KEY_KP_6;
	case VKEY_NUMPAD7: return KEY_KP_7;
	case VKEY_NUMPAD8: return KEY_KP_8;
	case VKEY_NUMPAD9: return KEY_KP_9;
	case VKEY_MULTIPLY: return KEY_KP_MULTIPLY;
	case VKEY_ADD: return KEY_KP_ADD;
//	case VKEY_SEPARATOR: return KEY_SEPARATOR;
	case VKEY_SUBTRACT: return KEY_KP_SUBTRACT;
	case VKEY_DECIMAL: return KEY_KP_PERIOD;
	case VKEY_DIVIDE: return KEY_KP_DIVIDE;
	case VKEY_F1: return KEY_F1;
	case VKEY_F2: return KEY_F2;
	case VKEY_F3: return KEY_F3;
	case VKEY_F4: return KEY_F4;
	case VKEY_F5: return KEY_F5;
	case VKEY_F6: return KEY_F6;
	case VKEY_F7: return KEY_F7;
	case VKEY_F8: return KEY_F8;
	case VKEY_F9: return KEY_F9;
	case VKEY_F10: return KEY_F10;
	case VKEY_F11: return KEY_F11;
	case VKEY_F12: return KEY_F12;
	case VKEY_F13: return KEY_F13;
	case VKEY_F14: return KEY_F14;
	case VKEY_F15: return KEY_F15;
	case VKEY_F16: return KEY_F16;
	/*
	case VKEY_F17: return KEY_F17;
	case VKEY_F18: return KEY_F18;
	case VKEY_F19: return KEY_F19;
	case VKEY_F20: return KEY_F20;
	case VKEY_F21: return KEY_F21;
	case VKEY_F22: return KEY_F22;
	case VKEY_F23: return KEY_F23;
	case VKEY_F24: return KEY_F24;
	*/
	case VKEY_NUMLOCK: return KEY_NUMLOCK;
	case VKEY_SCROLL: return KEY_SCROLLLOCK;
	case VKEY_LSHIFT: return KEY_SHIFT;
	case VKEY_RSHIFT: return KEY_SHIFT;
	case VKEY_LCONTROL: return KEY_CONTROL;
	case VKEY_RCONTROL: return KEY_CONTROL;
	case VKEY_LMENU: return KEY_MENU;
	case VKEY_RMENU: return KEY_MENU;
	case VKEY_BROWSER_BACK: return KEY_BACK;
	case VKEY_BROWSER_FORWARD: return KEY_FORWARD;
	case VKEY_BROWSER_REFRESH: return KEY_REFRESH;
	case VKEY_BROWSER_STOP: return KEY_STOP;
	case VKEY_BROWSER_SEARCH: return KEY_SEARCH;
	case VKEY_BROWSER_FAVORITES: return KEY_FAVORITES;
	case VKEY_BROWSER_HOME: return KEY_HOMEPAGE;
	case VKEY_VOLUME_MUTE: return KEY_VOLUMEMUTE;
	case VKEY_VOLUME_DOWN: return KEY_VOLUMEDOWN;
	case VKEY_VOLUME_UP: return KEY_VOLUMEUP;
	case VKEY_MEDIA_NEXT_TRACK: return KEY_MEDIANEXT;
	case VKEY_MEDIA_PREV_TRACK: return KEY_MEDIAPREVIOUS;
	case VKEY_MEDIA_STOP: return KEY_MEDIASTOP;
	case VKEY_MEDIA_PLAY_PAUSE: return KEY_MEDIAPLAY;
	case VKEY_MEDIA_LAUNCH_MAIL: return KEY_LAUNCHMAIL;
	case VKEY_MEDIA_LAUNCH_MEDIA_SELECT: return KEY_LAUNCHMEDIA; // FUCKING USELESS KEYS, HOW DO THEY WORK?
	case VKEY_MEDIA_LAUNCH_APP1: return KEY_LAUNCH0;
	case VKEY_MEDIA_LAUNCH_APP2: return KEY_LAUNCH0;
//	case VKEY_OEM_102: return KEY_OEM_102;
//	case VKEY_PROCESSKEY: return KEY_PROCESSKEY;
//	case VKEY_PACKET: return KEY_PACKET;
//	case VKEY_ATTN: return KEY_ATTN;
//	case VKEY_CRSEL: return KEY_CRSEL;
//	case VKEY_EXSEL: return KEY_EXSEL;
//	case VKEY_EREOF: return KEY_EREOF;
//	case VKEY_PLAY: return KEY_PLAY;
//	case VKEY_ZOOM: return KEY_ZOOM;
//	case VKEY_NONAME: return KEY_NONAME;
//	case VKEY_PA1: return KEY_PA1;
//	case VKEY_OEM_CLEAR: return KEY_OEM_CLEAR;

	default: break;
	};

	is_char = true;

	switch (p_key) {

	case VKEY_SPACE: return KEY_SPACE;
	case VKEY_0: return KEY_0;
	case VKEY_1: return KEY_1;
	case VKEY_2: return KEY_2;
	case VKEY_3: return KEY_3;
	case VKEY_4: return KEY_4;
	case VKEY_5: return KEY_5;
	case VKEY_6: return KEY_6;
	case VKEY_7: return KEY_7;
	case VKEY_8: return KEY_8;
	case VKEY_9: return KEY_9;
	case VKEY_A: return KEY_A;
	case VKEY_B: return KEY_B;
	case VKEY_C: return KEY_C;
	case VKEY_D: return KEY_D;
	case VKEY_E: return KEY_E;
	case VKEY_F: return KEY_F;
	case VKEY_G: return KEY_G;
	case VKEY_H: return KEY_H;
	case VKEY_I: return KEY_I;
	case VKEY_J: return KEY_J;
	case VKEY_K: return KEY_K;
	case VKEY_L: return KEY_L;
	case VKEY_M: return KEY_M;
	case VKEY_N: return KEY_N;
	case VKEY_O: return KEY_O;
	case VKEY_P: return KEY_P;
	case VKEY_Q: return KEY_Q;
	case VKEY_R: return KEY_R;
	case VKEY_S: return KEY_S;
	case VKEY_T: return KEY_T;
	case VKEY_U: return KEY_U;
	case VKEY_V: return KEY_V;
	case VKEY_W: return KEY_W;
	case VKEY_X: return KEY_X;
	case VKEY_Y: return KEY_Y;
	case VKEY_Z: return KEY_Z;
	/*
	case VKEY_OEM_PLUS: return KEY_PLUS;
	case VKEY_OEM_COMMA: return KEY_COMMA;
	case VKEY_OEM_MINUS: return KEY_MINUS;
	case VKEY_OEM_PERIOD: return KEY_PERIOD;
	case VKEY_OEM_1: return KEY_OEM_1;
	case VKEY_OEM_2: return KEY_OEM_2;
	case VKEY_OEM_3: return KEY_OEM_3;
	case VKEY_OEM_4: return KEY_OEM_4;
	case VKEY_OEM_5: return KEY_OEM_5;
	case VKEY_OEM_6: return KEY_OEM_6;
	case VKEY_OEM_7: return KEY_OEM_7;
	case VKEY_OEM_8: return KEY_OEM_8;
	*/
	default: break;

	};

	return 0;
};

#endif  // BASE_KEYBOARD_CODES_POSIX_H_
