/*************************************************************************/
/*  android_keys_utils.cpp                                               */
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

#include "android_keys_utils.h"

Key godot_code_from_android_code(unsigned int p_code) {
	for (int i = 0; android_godot_code_pairs[i].android_code != AKEYCODE_MAX; i++) {
		if (android_godot_code_pairs[i].android_code == p_code) {
			return android_godot_code_pairs[i].godot_code;
		}
	}
	return Key::UNKNOWN;
}

Key godot_code_from_unicode(unsigned int p_code) {
	unsigned int code = p_code;
	if (code > 0xFF) {
		return Key::UNKNOWN;
	}
	// Known control codes.
	if (code == '\b') { // 0x08
		return Key::BACKSPACE;
	}
	if (code == '\t') { // 0x09
		return Key::TAB;
	}
	if (code == '\n') { // 0x0A
		return Key::ENTER;
	}
	if (code == 0x1B) {
		return Key::ESCAPE;
	}
	if (code == 0x1F) {
		return Key::KEY_DELETE;
	}
	// Unknown control codes.
	if (code <= 0x1F || (code >= 0x80 && code <= 0x9F)) {
		return Key::UNKNOWN;
	}
	// Convert to uppercase.
	if (code >= 'a' && code <= 'z') { // 0x61 - 0x7A
		code -= ('a' - 'A');
	}
	if (code >= u'à' && code <= u'ö') { // 0xE0 - 0xF6
		code -= (u'à' - u'À'); // 0xE0 - 0xC0
	}
	if (code >= u'ø' && code <= u'þ') { // 0xF8 - 0xFF
		code -= (u'ø' - u'Ø'); // 0xF8 - 0xD8
	}
	return Key(code);
}
