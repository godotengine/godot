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
#ifndef GNOMESORT_CJK_H
#define GNOMESORT_CJK_H

/**
 * @def GNOMESORT_CJK_NO_END_LINE_CHARS
 * @brief A list of characters that cannot appear at the end of a line in a CJK language.
 */
#define GNOMESORT_CJK_NO_END_LINE_CHARS \
	GNOMESORT_CJK_UC(0x0022) \
	GNOMESORT_CJK_UC(0x0027) \
	GNOMESORT_CJK_UC(0x0028) \
	GNOMESORT_CJK_UC(0x005b) \
	GNOMESORT_CJK_UC(0x00ab) \
	GNOMESORT_CJK_UC(0x3008) \
	GNOMESORT_CJK_UC(0x300a) \
	GNOMESORT_CJK_UC(0x300c) \
	GNOMESORT_CJK_UC(0x300e) \
	GNOMESORT_CJK_UC(0x3010) \
	GNOMESORT_CJK_UC(0x3014) \
	GNOMESORT_CJK_UC(0x3016) \
	GNOMESORT_CJK_UC(0x3018) \
	GNOMESORT_CJK_UC(0x301d) \
	GNOMESORT_CJK_UC(0xff5b) \
	GNOMESORT_CJK_UC(0xff5f) \
	GNOMESORT_CJK_UC(0xff08)

/**
 * @def GNOMESORT_CJK_NO_BEGIN_LINE_CHARS
 * @brief A list of characters that cannot appear at the beginning of a line in CJK language.
 */
#define GNOMESORT_CJK_NO_BEGIN_LINE_CHARS \
	GNOMESORT_CJK_UC(0x0022) \
	GNOMESORT_CJK_UC(0x0027) \
	GNOMESORT_CJK_UC(0x0029) \
	GNOMESORT_CJK_UC(0x005d) \
	GNOMESORT_CJK_UC(0x00bb) \
	GNOMESORT_CJK_UC(0x3009) \
	GNOMESORT_CJK_UC(0x300b) \
	GNOMESORT_CJK_UC(0x300d) \
	GNOMESORT_CJK_UC(0x300f) \
	GNOMESORT_CJK_UC(0x3011) \
	GNOMESORT_CJK_UC(0x3015) \
	GNOMESORT_CJK_UC(0x3017) \
	GNOMESORT_CJK_UC(0x3019) \
	GNOMESORT_CJK_UC(0x301f) \
	GNOMESORT_CJK_UC(0xff5d) \
	GNOMESORT_CJK_UC(0xff60) \
	GNOMESORT_CJK_UC(0x2010) \
	GNOMESORT_CJK_UC(0x2013) \
	GNOMESORT_CJK_UC(0x301c) \
	GNOMESORT_CJK_UC(0x30a0) \
	GNOMESORT_CJK_UC(0x0021) \
	GNOMESORT_CJK_UC(0x203c) \
	GNOMESORT_CJK_UC(0x2047) \
	GNOMESORT_CJK_UC(0x2048) \
	GNOMESORT_CJK_UC(0x2049) \
	GNOMESORT_CJK_UC(0xff1f) \
	GNOMESORT_CJK_UC(0x002c) \
	GNOMESORT_CJK_UC(0x003a) \
	GNOMESORT_CJK_UC(0x003b) \
	GNOMESORT_CJK_UC(0x3001) \
	GNOMESORT_CJK_UC(0x30fb) \
	GNOMESORT_CJK_UC(0x002e) \
	GNOMESORT_CJK_UC(0x3002) \
	GNOMESORT_CJK_UC(0x3005) \
	GNOMESORT_CJK_UC(0x303b) \
	GNOMESORT_CJK_UC(0x3041) \
	GNOMESORT_CJK_UC(0x3043) \
	GNOMESORT_CJK_UC(0x3045) \
	GNOMESORT_CJK_UC(0x3047) \
	GNOMESORT_CJK_UC(0x3049) \
	GNOMESORT_CJK_UC(0x3063) \
	GNOMESORT_CJK_UC(0x3083) \
	GNOMESORT_CJK_UC(0x3085) \
	GNOMESORT_CJK_UC(0x3087) \
	GNOMESORT_CJK_UC(0x308e) \
	GNOMESORT_CJK_UC(0x3095) \
	GNOMESORT_CJK_UC(0x3096) \
	GNOMESORT_CJK_UC(0x30a1) \
	GNOMESORT_CJK_UC(0x30a3) \
	GNOMESORT_CJK_UC(0x30a5) \
	GNOMESORT_CJK_UC(0x30a7) \
	GNOMESORT_CJK_UC(0x30a9) \
	GNOMESORT_CJK_UC(0x30c3) \
	GNOMESORT_CJK_UC(0x30e3) \
	GNOMESORT_CJK_UC(0x30e5) \
	GNOMESORT_CJK_UC(0x30e7) \
	GNOMESORT_CJK_UC(0x30ee) \
	GNOMESORT_CJK_UC(0x30f5) \
	GNOMESORT_CJK_UC(0x30f6) \
	GNOMESORT_CJK_UC(0x30fc) \
	GNOMESORT_CJK_UC(0x30fd) \
	GNOMESORT_CJK_UC(0x30fe) \
	GNOMESORT_CJK_UC(0x31f0) \
	GNOMESORT_CJK_UC(0x31f1) \
	GNOMESORT_CJK_UC(0x31f2) \
	GNOMESORT_CJK_UC(0x31f3) \
	GNOMESORT_CJK_UC(0x31f4) \
	GNOMESORT_CJK_UC(0x31f5) \
	GNOMESORT_CJK_UC(0x31f6) \
	GNOMESORT_CJK_UC(0x31f7) \
	GNOMESORT_CJK_UC(0x31f8) \
	GNOMESORT_CJK_UC(0x31f9) \
	GNOMESORT_CJK_UC(0x31fa) \
	GNOMESORT_CJK_UC(0x31fb) \
	GNOMESORT_CJK_UC(0x31fc) \
	GNOMESORT_CJK_UC(0x31fd) \
	GNOMESORT_CJK_UC(0x31fe) \
	GNOMESORT_CJK_UC(0x31ff) \
	GNOMESORT_CJK_UC(0xff0c) \
	GNOMESORT_CJK_UC(0xff09) \

/**
 * @def GNOMESORT_CJK_NO_SEPARATE_CHARS
 * @brief A list of characters that may not be separated in a CJK language.
 */
#define GNOMESORT_CJK_NO_SEPARATE_CHARS \
	GNOMESORT_CJK_UC(0x002e) \
	GNOMESORT_CJK_UC(0x2014) \
	GNOMESORT_CJK_UC(0x2025) \
	GNOMESORT_CJK_UC(0x3033) \
	GNOMESORT_CJK_UC(0x3034) \
	GNOMESORT_CJK_UC(0x3035)

namespace gnomesort {

	/**
	 * @brief Determine whether or not a unicode character can appear at the end of a line of CJK text.
	 * @param p_char The character to test.
	 * @return True if the input character is not allowed at the end of a line. False in all other cases.
	 */
	bool is_cjk_cannot_end_line(const int p_char);

	/**
	 * @brief Determine whether or not a unicode character can appear at the beginning of a line of CJK text.
	 * @param p_char The character to test.
	 * @return True if the input character is not allowed at the beginning of a line. False in all other cases.
	 */
	bool is_cjk_cannot_begin_line(const int p_char);

	/**
	 * @brief Determine whether or not a unicode character can be separated across lines of CJK text.
	 * @param p_char The character to test.
	 * @return True if the input character is not allowed to be separated. False in all other cases.
	 */
	bool is_cjk_cannot_separate(const int p_char);

	/**
	 * @brief Determine whether or not the input character is a CJK character.
	 * @param p_char The character to test.
	 * @return True if the input character is a CJK character. False in all other cases.
	 */
	bool is_cjk_char(const int p_char);

	/**
	 * @brief Determine whether or not it is safe to break a line of CJK text around p_char.
	 * @param p_previous_char The previous character in sequence with p_char.
	 * @param p_char The character to test.
	 * @param p_next_char The next character in sequence with p_char.
	 * @return True if a line break can be inserted around p_char. False in all other cases.
	 */
	bool is_cjk_separatable_char(const int p_previous_char, const int p_char, const int p_next_char);

}

#endif
