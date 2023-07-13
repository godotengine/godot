/*
 * Copyright (c) 2023 Alexander Rothman <gnomesort@megate.ch>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "gs_cjk.h"

namespace gnomesort {

#define GNOMESORT_CJK_UC(value) case value:;

	bool is_cjk_cannot_end_line(const int p_char) {
		switch (p_char)
		{
		GNOMESORT_CJK_NO_END_LINE_CHARS
			return true;
		default:
			return false;
		}
	}

	bool is_cjk_cannot_begin_line(const int p_char) {
		switch (p_char)
		{
		GNOMESORT_CJK_NO_BEGIN_LINE_CHARS
			return true;
		default:
			return false;
		}
	}

	bool is_cjk_cannot_separate(const int p_char) {
		switch (p_char)
		{
		GNOMESORT_CJK_NO_SEPARATE_CHARS
			return true;
		default:
			return false;
		}
	}

	bool is_cjk_char(const int p_char) {
		return (p_char >= 0x2E08 && p_char <= 0x9FFF) || // CJK scripts and symbols.
					 (p_char >= 0xFF00 && p_char <= 0xFFEF) || // Fullwidth Latin characters.
					 (p_char >= 0xAC00 && p_char <= 0xD7FF) || // Hangul Syllables and Hangul Jamo Extended-B.
					 (p_char >= 0xF900 && p_char <= 0xFAFF) || // CJK Compatibility Ideographs.
					 (p_char >= 0xFE30 && p_char <= 0xFE4F) || // CJK Compatibility Forms.
					 (p_char >= 0xFF65 && p_char <= 0xFF9F) || // Halfwidth forms of katakana
					 (p_char >= 0xFFA0 && p_char <= 0xFFDC) || // Halfwidth forms of compatibility jamo characters for Hangul
					 (p_char >= 0x20000 && p_char <= 0x2FA1F) || // CJK Unified Ideographs Extension B ~ F and CJK Compatibility Ideographs Supplement.
					 (p_char >= 0x30000 && p_char <= 0x3134F); // CJK Unified Ideographs Extension G.
	}

	bool is_cjk_separatable_char(const int p_previous_char, const int p_char, const int p_next_char) {
		return (is_cjk_char(p_char) &&
					 !(is_cjk_cannot_separate(p_char) || is_cjk_cannot_separate(p_next_char)) &&
					 !(is_cjk_cannot_begin_line(p_char) || is_cjk_cannot_begin_line(p_next_char)) &&
					 !(is_cjk_cannot_end_line(p_next_char))) ||
					 (is_cjk_cannot_begin_line(p_previous_char)
						&& !is_cjk_cannot_begin_line(p_char)
						&& is_cjk_cannot_begin_line(p_next_char));
	}

#undef GNOMESORT_CJK_UC

}
