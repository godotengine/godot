/**************************************************************************/
/*  android_keys_utils.h                                                  */
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

#ifndef ANDROID_KEYS_UTILS_H
#define ANDROID_KEYS_UTILS_H

#include <android/input.h>
#include <core/os/keyboard.h>

#define AKEYCODE_MAX 0xFFFF

struct AndroidGodotCodePair {
	unsigned int android_code = 0;
	unsigned int godot_code = 0;
};

static AndroidGodotCodePair android_godot_code_pairs[] = {
	{ AKEYCODE_UNKNOWN, KEY_UNKNOWN }, // (0) Unknown key code.
	{ AKEYCODE_BACK, KEY_BACK }, // (4) Back key.
	{ AKEYCODE_0, KEY_0 }, // (7) '0' key.
	{ AKEYCODE_1, KEY_1 }, // (8) '1' key.
	{ AKEYCODE_2, KEY_2 }, // (9) '2' key.
	{ AKEYCODE_3, KEY_3 }, // (10) '3' key.
	{ AKEYCODE_4, KEY_4 }, // (11) '4' key.
	{ AKEYCODE_5, KEY_5 }, // (12) '5' key.
	{ AKEYCODE_6, KEY_6 }, // (13) '6' key.
	{ AKEYCODE_7, KEY_7 }, // (14) '7' key.
	{ AKEYCODE_8, KEY_8 }, // (15) '8' key.
	{ AKEYCODE_9, KEY_9 }, // (16) '9' key.
	{ AKEYCODE_STAR, KEY_ASTERISK }, // (17) '*' key.
	{ AKEYCODE_POUND, KEY_NUMBERSIGN }, // (18) '#' key.
	{ AKEYCODE_DPAD_UP, KEY_UP }, // (19) Directional Pad Up key.
	{ AKEYCODE_DPAD_DOWN, KEY_DOWN }, // (20) Directional Pad Down key.
	{ AKEYCODE_DPAD_LEFT, KEY_LEFT }, // (21) Directional Pad Left key.
	{ AKEYCODE_DPAD_RIGHT, KEY_RIGHT }, // (22) Directional Pad Right key.
	{ AKEYCODE_DPAD_CENTER, KEY_ENTER }, // (23) Directional Pad Center key.
	{ AKEYCODE_VOLUME_UP, KEY_VOLUMEUP }, // (24) Volume Up key.
	{ AKEYCODE_VOLUME_DOWN, KEY_VOLUMEDOWN }, // (25) Volume Down key.
	{ AKEYCODE_POWER, KEY_STANDBY }, // (26) Power key.
	{ AKEYCODE_CLEAR, KEY_CLEAR }, // (28) Clear key.
	{ AKEYCODE_A, KEY_A }, // (29) 'A' key.
	{ AKEYCODE_B, KEY_B }, // (30) 'B' key.
	{ AKEYCODE_C, KEY_C }, // (31) 'C' key.
	{ AKEYCODE_D, KEY_D }, // (32) 'D' key.
	{ AKEYCODE_E, KEY_E }, // (33) 'E' key.
	{ AKEYCODE_F, KEY_F }, // (34) 'F' key.
	{ AKEYCODE_G, KEY_G }, // (35) 'G' key.
	{ AKEYCODE_H, KEY_H }, // (36) 'H' key.
	{ AKEYCODE_I, KEY_I }, // (37) 'I' key.
	{ AKEYCODE_J, KEY_J }, // (38) 'J' key.
	{ AKEYCODE_K, KEY_K }, // (39) 'K' key.
	{ AKEYCODE_L, KEY_L }, // (40) 'L' key.
	{ AKEYCODE_M, KEY_M }, // (41) 'M' key.
	{ AKEYCODE_N, KEY_N }, // (42) 'N' key.
	{ AKEYCODE_O, KEY_O }, // (43) 'O' key.
	{ AKEYCODE_P, KEY_P }, // (44) 'P' key.
	{ AKEYCODE_Q, KEY_Q }, // (45) 'Q' key.
	{ AKEYCODE_R, KEY_R }, // (46) 'R' key.
	{ AKEYCODE_S, KEY_S }, // (47) 'S' key.
	{ AKEYCODE_T, KEY_T }, // (48) 'T' key.
	{ AKEYCODE_U, KEY_U }, // (49) 'U' key.
	{ AKEYCODE_V, KEY_V }, // (50) 'V' key.
	{ AKEYCODE_W, KEY_W }, // (51) 'W' key.
	{ AKEYCODE_X, KEY_X }, // (52) 'X' key.
	{ AKEYCODE_Y, KEY_Y }, // (53) 'Y' key.
	{ AKEYCODE_Z, KEY_Z }, // (54) 'Z' key.
	{ AKEYCODE_COMMA, KEY_COMMA }, // (55) ',’ key.
	{ AKEYCODE_PERIOD, KEY_PERIOD }, // (56) '.' key.
	{ AKEYCODE_ALT_LEFT, KEY_ALT }, // (57) Left Alt modifier key.
	{ AKEYCODE_ALT_RIGHT, KEY_ALT }, // (58) Right Alt modifier key.
	{ AKEYCODE_SHIFT_LEFT, KEY_SHIFT }, // (59) Left Shift modifier key.
	{ AKEYCODE_SHIFT_RIGHT, KEY_SHIFT }, // (60) Right Shift modifier key.
	{ AKEYCODE_TAB, KEY_TAB }, // (61) Tab key.
	{ AKEYCODE_SPACE, KEY_SPACE }, // (62) Space key.
	{ AKEYCODE_ENVELOPE, KEY_LAUNCHMAIL }, // (65) Envelope special function key.
	{ AKEYCODE_ENTER, KEY_ENTER }, // (66) Enter key.
	{ AKEYCODE_DEL, KEY_BACKSPACE }, // (67) Backspace key.
	{ AKEYCODE_GRAVE, KEY_QUOTELEFT }, // (68) '`' (backtick) key.
	{ AKEYCODE_MINUS, KEY_MINUS }, // (69) '-'.
	{ AKEYCODE_EQUALS, KEY_EQUAL }, // (70) '=' key.
	{ AKEYCODE_LEFT_BRACKET, KEY_BRACKETLEFT }, // (71) '[' key.
	{ AKEYCODE_RIGHT_BRACKET, KEY_BRACKETRIGHT }, // (72) ']' key.
	{ AKEYCODE_BACKSLASH, KEY_BACKSLASH }, // (73) '\' key.
	{ AKEYCODE_SEMICOLON, KEY_SEMICOLON }, // (74) ';' key.
	{ AKEYCODE_APOSTROPHE, KEY_APOSTROPHE }, // (75) ''' (apostrophe) key.
	{ AKEYCODE_SLASH, KEY_SLASH }, // (76) '/' key.
	{ AKEYCODE_AT, KEY_AT }, // (77) '@' key.
	{ AKEYCODE_PLUS, KEY_PLUS }, // (81) '+' key.
	{ AKEYCODE_MENU, KEY_MENU }, // (82) Menu key.
	{ AKEYCODE_SEARCH, KEY_SEARCH }, // (84) Search key.
	{ AKEYCODE_MEDIA_STOP, KEY_MEDIASTOP }, // (86) Stop media key.
	{ AKEYCODE_MEDIA_NEXT, KEY_MEDIANEXT }, // (87) Play Next media key.
	{ AKEYCODE_MEDIA_PREVIOUS, KEY_MEDIAPREVIOUS }, // (88) Play Previous media key.
	{ AKEYCODE_PAGE_UP, KEY_PAGEUP }, // (92) Page Up key.
	{ AKEYCODE_PAGE_DOWN, KEY_PAGEDOWN }, // (93) Page Down key.
	{ AKEYCODE_ESCAPE, KEY_ESCAPE }, // (111) Escape key.
	{ AKEYCODE_FORWARD_DEL, KEY_DELETE }, // (112) Forward Delete key.
	{ AKEYCODE_CTRL_LEFT, KEY_CONTROL }, // (113) Left Control modifier key.
	{ AKEYCODE_CTRL_RIGHT, KEY_CONTROL }, // (114) Right Control modifier key.
	{ AKEYCODE_CAPS_LOCK, KEY_CAPSLOCK }, // (115) Caps Lock key.
	{ AKEYCODE_SCROLL_LOCK, KEY_SCROLLLOCK }, // (116) Scroll Lock key.
	{ AKEYCODE_META_LEFT, KEY_META }, // (117) Left Meta modifier key.
	{ AKEYCODE_META_RIGHT, KEY_META }, // (118) Right Meta modifier key.
	{ AKEYCODE_SYSRQ, KEY_PRINT }, // (120) System Request / Print Screen key.
	{ AKEYCODE_BREAK, KEY_PAUSE }, // (121) Break / Pause key.
	{ AKEYCODE_MOVE_HOME, KEY_HOME }, // (122) Home Movement key.
	{ AKEYCODE_MOVE_END, KEY_END }, // (123) End Movement key.
	{ AKEYCODE_INSERT, KEY_INSERT }, // (124) Insert key.
	{ AKEYCODE_FORWARD, KEY_FORWARD }, // (125) Forward key.
	{ AKEYCODE_MEDIA_PLAY, KEY_MEDIAPLAY }, // (126) Play media key.
	{ AKEYCODE_MEDIA_RECORD, KEY_MEDIARECORD }, // (130) Record media key.
	{ AKEYCODE_F1, KEY_F1 }, // (131) F1 key.
	{ AKEYCODE_F2, KEY_F2 }, // (132) F2 key.
	{ AKEYCODE_F3, KEY_F3 }, // (133) F3 key.
	{ AKEYCODE_F4, KEY_F4 }, // (134) F4 key.
	{ AKEYCODE_F5, KEY_F5 }, // (135) F5 key.
	{ AKEYCODE_F6, KEY_F6 }, // (136) F6 key.
	{ AKEYCODE_F7, KEY_F7 }, // (137) F7 key.
	{ AKEYCODE_F8, KEY_F8 }, // (138) F8 key.
	{ AKEYCODE_F9, KEY_F9 }, // (139) F9 key.
	{ AKEYCODE_F10, KEY_F10 }, // (140) F10 key.
	{ AKEYCODE_F11, KEY_F11 }, // (141) F11 key.
	{ AKEYCODE_F12, KEY_F12 }, // (142) F12 key.
	{ AKEYCODE_NUM_LOCK, KEY_NUMLOCK }, // (143) Num Lock key.
	{ AKEYCODE_NUMPAD_0, KEY_KP_0 }, // (144) Numeric keypad '0' key.
	{ AKEYCODE_NUMPAD_1, KEY_KP_1 }, // (145) Numeric keypad '1' key.
	{ AKEYCODE_NUMPAD_2, KEY_KP_2 }, // (146) Numeric keypad '2' key.
	{ AKEYCODE_NUMPAD_3, KEY_KP_3 }, // (147) Numeric keypad '3' key.
	{ AKEYCODE_NUMPAD_4, KEY_KP_4 }, // (148) Numeric keypad '4' key.
	{ AKEYCODE_NUMPAD_5, KEY_KP_5 }, // (149) Numeric keypad '5' key.
	{ AKEYCODE_NUMPAD_6, KEY_KP_6 }, // (150) Numeric keypad '6' key.
	{ AKEYCODE_NUMPAD_7, KEY_KP_7 }, // (151) Numeric keypad '7' key.
	{ AKEYCODE_NUMPAD_8, KEY_KP_8 }, // (152) Numeric keypad '8' key.
	{ AKEYCODE_NUMPAD_9, KEY_KP_9 }, // (153) Numeric keypad '9' key.
	{ AKEYCODE_NUMPAD_DIVIDE, KEY_KP_DIVIDE }, // (154) Numeric keypad '/' key (for division).
	{ AKEYCODE_NUMPAD_MULTIPLY, KEY_KP_MULTIPLY }, // (155) Numeric keypad '*' key (for multiplication).
	{ AKEYCODE_NUMPAD_SUBTRACT, KEY_KP_SUBTRACT }, // (156) Numeric keypad '-' key (for subtraction).
	{ AKEYCODE_NUMPAD_ADD, KEY_KP_ADD }, // (157) Numeric keypad '+' key (for addition).
	{ AKEYCODE_NUMPAD_DOT, KEY_KP_PERIOD }, // (158) Numeric keypad '.' key (for decimals or digit grouping).
	{ AKEYCODE_NUMPAD_ENTER, KEY_KP_ENTER }, // (160) Numeric keypad Enter key.
	{ AKEYCODE_VOLUME_MUTE, KEY_VOLUMEMUTE }, // (164) Volume Mute key.
	{ AKEYCODE_YEN, KEY_YEN }, // (216) Japanese Yen key.
	{ AKEYCODE_HELP, KEY_HELP }, // (259) Help key.
	{ AKEYCODE_REFRESH, KEY_REFRESH }, // (285) Refresh key.
	{ AKEYCODE_MAX, KEY_UNKNOWN }
};

unsigned int godot_code_from_android_code(unsigned int p_code);
unsigned int godot_code_from_unicode(unsigned int p_code);

#endif // ANDROID_KEYS_UTILS_H
