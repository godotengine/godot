/**************************************************************************/
/*  script_iterator.cpp                                                   */
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

#include "script_iterator.h"

// This implementation is derived from ICU: icu4c/source/extra/scrptrun/scrptrun.cpp

inline constexpr UChar32 ZERO_WIDTH_JOINER = 0x200d;
inline constexpr UChar32 VARIATION_SELECTOR_15 = 0xfe0e;
inline constexpr UChar32 VARIATION_SELECTOR_16 = 0xfe0f;

inline bool ScriptIterator::same_script(int32_t p_script_one, int32_t p_script_two) {
	return p_script_one <= USCRIPT_INHERITED || p_script_two <= USCRIPT_INHERITED || p_script_one == p_script_two;
}

inline bool ScriptIterator::is_emoji(UChar32 p_c, UChar32 p_next) {
	if (p_next == VARIATION_SELECTOR_15 && (u_hasBinaryProperty(p_c, UCHAR_EMOJI) || u_hasBinaryProperty(p_c, UCHAR_EXTENDED_PICTOGRAPHIC))) {
		return false;
	} else if (p_next == VARIATION_SELECTOR_16 && (u_hasBinaryProperty(p_c, UCHAR_EMOJI) || u_hasBinaryProperty(p_c, UCHAR_EXTENDED_PICTOGRAPHIC))) {
		return true;
	} else {
		return u_hasBinaryProperty(p_c, UCHAR_EMOJI_PRESENTATION) || u_hasBinaryProperty(p_c, UCHAR_EMOJI_MODIFIER) || u_hasBinaryProperty(p_c, UCHAR_REGIONAL_INDICATOR);
	}
}

ScriptIterator::ScriptIterator(const String &p_string, int p_start, int p_length) {
	struct ParenStackEntry {
		int pair_index;
		UScriptCode script_code;
	};

	struct EmojiSubrunEntry {
		int start;
		int end;
	};

	if (p_start >= p_length) {
		p_start = p_length - 1;
	}

	if (p_start < 0) {
		p_start = 0;
	}

	int paren_size = PAREN_STACK_DEPTH;
	ParenStackEntry starter_paren_stack[PAREN_STACK_DEPTH];
	ParenStackEntry *paren_stack = starter_paren_stack;

	int emoji_size = EMOJI_STACK_DEPTH;
	EmojiSubrunEntry starter_emoji_stack[EMOJI_STACK_DEPTH];
	EmojiSubrunEntry *emoji_stack = starter_emoji_stack;

	int script_start;
	int script_end = p_start;
	UScriptCode script_code;
	int paren_sp = -1;
	int start_sp = paren_sp;
	UErrorCode err = U_ZERO_ERROR;
	const char32_t *str = p_string.ptr();

	do {
		script_code = USCRIPT_COMMON;
		int emoji_sp = -1;
		bool emoji_run = false;
		for (script_start = script_end; script_end < p_length; script_end++) {
			UChar32 ch = str[script_end];
			UChar32 n = (script_end + 1 < p_length) ? str[script_end + 1] : 0;
			if (is_emoji(ch, n)) {
				if (!emoji_run) {
					emoji_run = true;
					emoji_sp++;
					if (unlikely(emoji_sp >= emoji_size)) {
						emoji_size += EMOJI_STACK_DEPTH;
						if (emoji_stack == starter_emoji_stack) {
							emoji_stack = static_cast<EmojiSubrunEntry *>(memalloc(emoji_size * sizeof(EmojiSubrunEntry)));
						} else {
							emoji_stack = static_cast<EmojiSubrunEntry *>(memrealloc(emoji_stack, emoji_size * sizeof(EmojiSubrunEntry)));
						}
					}
					emoji_stack[emoji_sp].start = script_end;
					emoji_stack[emoji_sp].end = script_end;
				}
			} else if (emoji_run && ch != ZERO_WIDTH_JOINER && ch != VARIATION_SELECTOR_16 && !(u_hasBinaryProperty(ch, UCHAR_EXTENDED_PICTOGRAPHIC) && n != VARIATION_SELECTOR_15)) {
				emoji_run = false;
				emoji_stack[emoji_sp].end = script_end;
			}

			UScriptCode sc = uscript_getScript(ch, &err);
			if (U_FAILURE(err)) {
				if (paren_stack != starter_paren_stack) {
					memfree(paren_stack);
				}
				ERR_FAIL_MSG(u_errorName(err));
			}

			if (u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) != U_BPT_NONE) {
				if (u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) == U_BPT_OPEN) {
					// If it's an open character, push it onto the stack.
					paren_sp++;
					if (unlikely(paren_sp >= paren_size)) {
						// If the stack is full, allocate more space to handle deeply nested parentheses. This is unlikely to happen with any real text.
						paren_size += PAREN_STACK_DEPTH;
						if (paren_stack == starter_paren_stack) {
							paren_stack = static_cast<ParenStackEntry *>(memalloc(paren_size * sizeof(ParenStackEntry)));
						} else {
							paren_stack = static_cast<ParenStackEntry *>(memrealloc(paren_stack, paren_size * sizeof(ParenStackEntry)));
						}
					}
					paren_stack[paren_sp].pair_index = ch;
					paren_stack[paren_sp].script_code = script_code;
				} else if (paren_sp >= 0) {
					// If it's a close character, find the matching open on the stack, and use that script code. Any non-matching open characters above it on the stack will be popped.
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
					// Now that we have a final script code, fix any open characters we pushed before we knew the script code.
					while (start_sp < paren_sp) {
						paren_stack[++start_sp].script_code = script_code;
					}
				}
				if ((u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) == U_BPT_CLOSE) && paren_sp >= 0) {
					// If this character is a close paired character pop the matching open character from the stack.
					paren_sp -= 1;
					if (start_sp >= 0) {
						start_sp -= 1;
					}
				}
			} else {
				break;
			}
		}
		if (emoji_run) {
			emoji_stack[emoji_sp].end = script_end;
		}

		for (int sub = 0; sub <= emoji_sp; sub++) {
			if (emoji_stack[sub].start > script_start) {
				ScriptRange rng;
				rng.script = hb_icu_script_to_script(script_code);
				rng.start = script_start;
				rng.end = emoji_stack[sub].start;
				script_ranges.push_back(rng);
			}
			ScriptRange rng;
			rng.script = (hb_script_t)HB_TAG('Z', 's', 'y', 'e');
			rng.start = emoji_stack[sub].start;
			rng.end = emoji_stack[sub].end;
			script_ranges.push_back(rng);

			script_start = emoji_stack[sub].end;
		}
		if (script_start != script_end) {
			ScriptRange rng;
			rng.script = hb_icu_script_to_script(script_code);
			rng.start = script_start;
			rng.end = script_end;
			script_ranges.push_back(rng);
		}

		if (emoji_stack != starter_emoji_stack) {
			memfree(emoji_stack);
		}
	} while (script_end < p_length);

	if (paren_stack != starter_paren_stack) {
		memfree(paren_stack);
	}
}
