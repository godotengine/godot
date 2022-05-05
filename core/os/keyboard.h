/*************************************************************************/
/*  keyboard.h                                                           */
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

#ifndef KEYBOARD_H
#define KEYBOARD_H

#include "core/string/ustring.h"

enum class Key {
	NONE = 0,
	// Special key: The strategy here is similar to the one used by toolkits,
	// which consists in leaving the 24 bits unicode range for printable
	// characters, and use the upper 8 bits for special keys and modifiers.
	// This way everything (char/keycode) can fit nicely in one 32-bit
	// integer (the enum's underlying type is `int` by default).
	SPECIAL = (1 << 24),
	/* CURSOR/FUNCTION/BROWSER/MULTIMEDIA/MISC KEYS */
	ESCAPE = SPECIAL | 0x01,
	TAB = SPECIAL | 0x02,
	BACKTAB = SPECIAL | 0x03,
	BACKSPACE = SPECIAL | 0x04,
	ENTER = SPECIAL | 0x05,
	KP_ENTER = SPECIAL | 0x06,
	INSERT = SPECIAL | 0x07,
	KEY_DELETE = SPECIAL | 0x08, // "DELETE" is a reserved word on Windows.
	PAUSE = SPECIAL | 0x09,
	PRINT = SPECIAL | 0x0a,
	SYSREQ = SPECIAL | 0x0b,
	CLEAR = SPECIAL | 0x0c,
	HOME = SPECIAL | 0x0d,
	END = SPECIAL | 0x0e,
	LEFT = SPECIAL | 0x0f,
	UP = SPECIAL | 0x10,
	RIGHT = SPECIAL | 0x11,
	DOWN = SPECIAL | 0x12,
	PAGEUP = SPECIAL | 0x13,
	PAGEDOWN = SPECIAL | 0x14,
	SHIFT = SPECIAL | 0x15,
	CTRL = SPECIAL | 0x16,
	META = SPECIAL | 0x17,
	ALT = SPECIAL | 0x18,
	CAPSLOCK = SPECIAL | 0x19,
	NUMLOCK = SPECIAL | 0x1a,
	SCROLLLOCK = SPECIAL | 0x1b,
	F1 = SPECIAL | 0x1c,
	F2 = SPECIAL | 0x1d,
	F3 = SPECIAL | 0x1e,
	F4 = SPECIAL | 0x1f,
	F5 = SPECIAL | 0x20,
	F6 = SPECIAL | 0x21,
	F7 = SPECIAL | 0x22,
	F8 = SPECIAL | 0x23,
	F9 = SPECIAL | 0x24,
	F10 = SPECIAL | 0x25,
	F11 = SPECIAL | 0x26,
	F12 = SPECIAL | 0x27,
	F13 = SPECIAL | 0x28,
	F14 = SPECIAL | 0x29,
	F15 = SPECIAL | 0x2a,
	F16 = SPECIAL | 0x2b,
	KP_MULTIPLY = SPECIAL | 0x81,
	KP_DIVIDE = SPECIAL | 0x82,
	KP_SUBTRACT = SPECIAL | 0x83,
	KP_PERIOD = SPECIAL | 0x84,
	KP_ADD = SPECIAL | 0x85,
	KP_0 = SPECIAL | 0x86,
	KP_1 = SPECIAL | 0x87,
	KP_2 = SPECIAL | 0x88,
	KP_3 = SPECIAL | 0x89,
	KP_4 = SPECIAL | 0x8a,
	KP_5 = SPECIAL | 0x8b,
	KP_6 = SPECIAL | 0x8c,
	KP_7 = SPECIAL | 0x8d,
	KP_8 = SPECIAL | 0x8e,
	KP_9 = SPECIAL | 0x8f,
	SUPER_L = SPECIAL | 0x2c,
	SUPER_R = SPECIAL | 0x2d,
	MENU = SPECIAL | 0x2e,
	HYPER_L = SPECIAL | 0x2f,
	HYPER_R = SPECIAL | 0x30,
	HELP = SPECIAL | 0x31,
	DIRECTION_L = SPECIAL | 0x32,
	DIRECTION_R = SPECIAL | 0x33,
	BACK = SPECIAL | 0x40,
	FORWARD = SPECIAL | 0x41,
	STOP = SPECIAL | 0x42,
	REFRESH = SPECIAL | 0x43,
	VOLUMEDOWN = SPECIAL | 0x44,
	VOLUMEMUTE = SPECIAL | 0x45,
	VOLUMEUP = SPECIAL | 0x46,
	BASSBOOST = SPECIAL | 0x47,
	BASSUP = SPECIAL | 0x48,
	BASSDOWN = SPECIAL | 0x49,
	TREBLEUP = SPECIAL | 0x4a,
	TREBLEDOWN = SPECIAL | 0x4b,
	MEDIAPLAY = SPECIAL | 0x4c,
	MEDIASTOP = SPECIAL | 0x4d,
	MEDIAPREVIOUS = SPECIAL | 0x4e,
	MEDIANEXT = SPECIAL | 0x4f,
	MEDIARECORD = SPECIAL | 0x50,
	HOMEPAGE = SPECIAL | 0x51,
	FAVORITES = SPECIAL | 0x52,
	SEARCH = SPECIAL | 0x53,
	STANDBY = SPECIAL | 0x54,
	OPENURL = SPECIAL | 0x55,
	LAUNCHMAIL = SPECIAL | 0x56,
	LAUNCHMEDIA = SPECIAL | 0x57,
	LAUNCH0 = SPECIAL | 0x58,
	LAUNCH1 = SPECIAL | 0x59,
	LAUNCH2 = SPECIAL | 0x5a,
	LAUNCH3 = SPECIAL | 0x5b,
	LAUNCH4 = SPECIAL | 0x5c,
	LAUNCH5 = SPECIAL | 0x5d,
	LAUNCH6 = SPECIAL | 0x5e,
	LAUNCH7 = SPECIAL | 0x5f,
	LAUNCH8 = SPECIAL | 0x60,
	LAUNCH9 = SPECIAL | 0x61,
	LAUNCHA = SPECIAL | 0x62,
	LAUNCHB = SPECIAL | 0x63,
	LAUNCHC = SPECIAL | 0x64,
	LAUNCHD = SPECIAL | 0x65,
	LAUNCHE = SPECIAL | 0x66,
	LAUNCHF = SPECIAL | 0x67,

	UNKNOWN = SPECIAL | 0xffffff,

	/* PRINTABLE LATIN 1 CODES */

	SPACE = 0x0020,
	EXCLAM = 0x0021,
	QUOTEDBL = 0x0022,
	NUMBERSIGN = 0x0023,
	DOLLAR = 0x0024,
	PERCENT = 0x0025,
	AMPERSAND = 0x0026,
	APOSTROPHE = 0x0027,
	PARENLEFT = 0x0028,
	PARENRIGHT = 0x0029,
	ASTERISK = 0x002a,
	PLUS = 0x002b,
	COMMA = 0x002c,
	MINUS = 0x002d,
	PERIOD = 0x002e,
	SLASH = 0x002f,
	KEY_0 = 0x0030,
	KEY_1 = 0x0031,
	KEY_2 = 0x0032,
	KEY_3 = 0x0033,
	KEY_4 = 0x0034,
	KEY_5 = 0x0035,
	KEY_6 = 0x0036,
	KEY_7 = 0x0037,
	KEY_8 = 0x0038,
	KEY_9 = 0x0039,
	COLON = 0x003a,
	SEMICOLON = 0x003b,
	LESS = 0x003c,
	EQUAL = 0x003d,
	GREATER = 0x003e,
	QUESTION = 0x003f,
	AT = 0x0040,
	A = 0x0041,
	B = 0x0042,
	C = 0x0043,
	D = 0x0044,
	E = 0x0045,
	F = 0x0046,
	G = 0x0047,
	H = 0x0048,
	I = 0x0049,
	J = 0x004a,
	K = 0x004b,
	L = 0x004c,
	M = 0x004d,
	N = 0x004e,
	O = 0x004f,
	P = 0x0050,
	Q = 0x0051,
	R = 0x0052,
	S = 0x0053,
	T = 0x0054,
	U = 0x0055,
	V = 0x0056,
	W = 0x0057,
	X = 0x0058,
	Y = 0x0059,
	Z = 0x005a,
	BRACKETLEFT = 0x005b,
	BACKSLASH = 0x005c,
	BRACKETRIGHT = 0x005d,
	ASCIICIRCUM = 0x005e,
	UNDERSCORE = 0x005f,
	QUOTELEFT = 0x0060,
	BRACELEFT = 0x007b,
	BAR = 0x007c,
	BRACERIGHT = 0x007d,
	ASCIITILDE = 0x007e,
	NOBREAKSPACE = 0x00a0,
	EXCLAMDOWN = 0x00a1,
	CENT = 0x00a2,
	STERLING = 0x00a3,
	CURRENCY = 0x00a4,
	YEN = 0x00a5,
	BROKENBAR = 0x00a6,
	SECTION = 0x00a7,
	DIAERESIS = 0x00a8,
	COPYRIGHT = 0x00a9,
	ORDFEMININE = 0x00aa,
	GUILLEMOTLEFT = 0x00ab,
	NOTSIGN = 0x00ac,
	HYPHEN = 0x00ad,
	KEY_REGISTERED = 0x00ae, // "REGISTERED" is a reserved word on Windows.
	MACRON = 0x00af,
	DEGREE = 0x00b0,
	PLUSMINUS = 0x00b1,
	TWOSUPERIOR = 0x00b2,
	THREESUPERIOR = 0x00b3,
	ACUTE = 0x00b4,
	MU = 0x00b5,
	PARAGRAPH = 0x00b6,
	PERIODCENTERED = 0x00b7,
	CEDILLA = 0x00b8,
	ONESUPERIOR = 0x00b9,
	MASCULINE = 0x00ba,
	GUILLEMOTRIGHT = 0x00bb,
	ONEQUARTER = 0x00bc,
	ONEHALF = 0x00bd,
	THREEQUARTERS = 0x00be,
	QUESTIONDOWN = 0x00bf,
	AGRAVE = 0x00c0,
	AACUTE = 0x00c1,
	ACIRCUMFLEX = 0x00c2,
	ATILDE = 0x00c3,
	ADIAERESIS = 0x00c4,
	ARING = 0x00c5,
	AE = 0x00c6,
	CCEDILLA = 0x00c7,
	EGRAVE = 0x00c8,
	EACUTE = 0x00c9,
	ECIRCUMFLEX = 0x00ca,
	EDIAERESIS = 0x00cb,
	IGRAVE = 0x00cc,
	IACUTE = 0x00cd,
	ICIRCUMFLEX = 0x00ce,
	IDIAERESIS = 0x00cf,
	ETH = 0x00d0,
	NTILDE = 0x00d1,
	OGRAVE = 0x00d2,
	OACUTE = 0x00d3,
	OCIRCUMFLEX = 0x00d4,
	OTILDE = 0x00d5,
	ODIAERESIS = 0x00d6,
	MULTIPLY = 0x00d7,
	OOBLIQUE = 0x00d8,
	UGRAVE = 0x00d9,
	UACUTE = 0x00da,
	UCIRCUMFLEX = 0x00db,
	UDIAERESIS = 0x00dc,
	YACUTE = 0x00dd,
	THORN = 0x00de,
	SSHARP = 0x00df,

	DIVISION = 0x00f7,
	YDIAERESIS = 0x00ff,
	END_LATIN1 = 0x0100,
};

enum class KeyModifierMask {
	CODE_MASK = ((1 << 25) - 1), ///< Apply this mask to any keycode to remove modifiers.
	MODIFIER_MASK = (0x7f << 24), ///< Apply this mask to isolate modifiers.
	SHIFT = (1 << 25),
	ALT = (1 << 26),
	META = (1 << 27),
	CTRL = (1 << 28),
#ifdef APPLE_STYLE_KEYS
	CMD = META,
#else
	CMD = CTRL,
#endif
	KPAD = (1 << 29),
	GROUP_SWITCH = (1 << 30)
};

// To avoid having unnecessary operators, only define the ones that are needed.

inline Key operator-(uint32_t a, Key b) {
	return (Key)(a - (uint32_t)b);
}

inline Key &operator-=(Key &a, int b) {
	return (Key &)((int &)a -= b);
}

inline Key operator+(Key a, int b) {
	return (Key)((int)a + (int)b);
}

inline Key operator+(Key a, Key b) {
	return (Key)((int)a + (int)b);
}

inline Key operator-(Key a, Key b) {
	return (Key)((int)a - (int)b);
}

inline Key operator&(Key a, Key b) {
	return (Key)((int)a & (int)b);
}

inline Key operator|(Key a, Key b) {
	return (Key)((int)a | (int)b);
}

inline Key &operator|=(Key &a, Key b) {
	return (Key &)((int &)a |= (int)b);
}

inline Key &operator|=(Key &a, KeyModifierMask b) {
	return (Key &)((int &)a |= (int)b);
}

inline Key &operator&=(Key &a, KeyModifierMask b) {
	return (Key &)((int &)a &= (int)b);
}

inline Key operator|(Key a, KeyModifierMask b) {
	return (Key)((int)a | (int)b);
}

inline Key operator&(Key a, KeyModifierMask b) {
	return (Key)((int)a & (int)b);
}

inline Key operator+(KeyModifierMask a, Key b) {
	return (Key)((int)a + (int)b);
}

inline Key operator|(KeyModifierMask a, Key b) {
	return (Key)((int)a | (int)b);
}

inline KeyModifierMask operator+(KeyModifierMask a, KeyModifierMask b) {
	return (KeyModifierMask)((int)a + (int)b);
}

inline KeyModifierMask operator|(KeyModifierMask a, KeyModifierMask b) {
	return (KeyModifierMask)((int)a | (int)b);
}

String keycode_get_string(Key p_code);
bool keycode_has_unicode(Key p_keycode);
Key find_keycode(const String &p_code);
const char *find_keycode_name(Key p_keycode);
int keycode_get_count();
int keycode_get_value_by_index(int p_index);
const char *keycode_get_name_by_index(int p_index);

#endif // KEYBOARD_H
