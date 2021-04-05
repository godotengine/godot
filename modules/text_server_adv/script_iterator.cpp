/*************************************************************************/
/*  script_iterator.cpp                                                  */
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

#include "script_iterator.h"

bool ScriptIterator::same_script(int32_t p_script_one, int32_t p_script_two) {
	return p_script_one <= USCRIPT_INHERITED || p_script_two <= USCRIPT_INHERITED || p_script_one == p_script_two;
}

ScriptIterator::ScriptIterator(const String &p_string, int p_start, int p_length) {
	struct ParenStackEntry {
		int pair_index;
		UScriptCode script_code;
	};

	if (p_start >= p_length) {
		p_start = p_length - 1;
	}

	if (p_start < 0) {
		p_start = 0;
	}

	ParenStackEntry paren_stack[128];

	int script_start;
	int script_end = p_start;
	UScriptCode script_code;
	int paren_sp = -1;
	int start_sp = paren_sp;
	UErrorCode err = U_ZERO_ERROR;
	const char32_t *str = p_string.ptr();

	do {
		script_code = USCRIPT_COMMON;
		for (script_start = script_end; script_end < p_length; script_end++) {
			UChar32 ch = str[script_end];
			UScriptCode sc = uscript_getScript(ch, &err);
			if (U_FAILURE(err)) {
				ERR_FAIL_MSG(u_errorName(err));
			}
			if (u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) != U_BPT_NONE) {
				if (u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) == U_BPT_OPEN) {
					paren_stack[++paren_sp].pair_index = ch;
					paren_stack[paren_sp].script_code = script_code;
				} else if (paren_sp >= 0) {
					UChar32 paired_ch = u_getBidiPairedBracket(ch);
					while (paren_sp >= 0 && paren_stack[paren_sp].pair_index != paired_ch) {
						paren_sp -= 1;
					}
					if (paren_sp < start_sp) {
						start_sp = paren_sp;
					}
					if (paren_sp >= 0) {
						sc = paren_stack[paren_sp].script_code;
					}
				}
			}

			if (same_script(script_code, sc)) {
				if (script_code <= USCRIPT_INHERITED && sc > USCRIPT_INHERITED) {
					script_code = sc;
					while (start_sp < paren_sp) {
						paren_stack[++start_sp].script_code = script_code;
					}
				}
				if ((u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) == U_BPT_CLOSE) && paren_sp >= 0) {
					paren_sp -= 1;
					if (start_sp >= 0) {
						start_sp -= 1;
					}
				}
			} else {
				break;
			}
		}

		ScriptRange rng;
		rng.script = hb_icu_script_to_script(script_code);
		rng.start = script_start;
		rng.end = script_end;

		script_ranges.push_back(rng);
	} while (script_end < p_length);
}
