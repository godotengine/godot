/*************************************************************************/
/*  FBXParseTools.h                                                      */
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

#ifndef FBX_PARSE_TOOLS_H
#define FBX_PARSE_TOOLS_H

#include "core/error/error_macros.h"
#include "core/string/ustring.h"

#include <stdint.h>
#include <algorithm>
#include <locale>

template <class char_t>
inline bool IsNewLine(char_t c) {
	return c == '\n' || c == '\r';
}
template <class char_t>
inline bool IsSpace(char_t c) {
	return (c == (char_t)' ' || c == (char_t)'\t');
}

template <class char_t>
inline bool IsSpaceOrNewLine(char_t c) {
	return IsNewLine(c) || IsSpace(c);
}

template <class char_t>
inline bool IsLineEnd(char_t c) {
	return (c == (char_t)'\r' || c == (char_t)'\n' || c == (char_t)'\0' || c == (char_t)'\f');
}

// ------------------------------------------------------------------------------------
// Special version of the function, providing higher accuracy and safety
// It is mainly used by fast_atof to prevent ugly and unwanted integer overflows.
// ------------------------------------------------------------------------------------
inline uint64_t strtoul10_64(const char *in, bool &errored, const char **out = nullptr, unsigned int *max_inout = nullptr) {
	unsigned int cur = 0;
	uint64_t value = 0;

	errored = *in < '0' || *in > '9';
	ERR_FAIL_COND_V_MSG(errored, 0, "The string cannot be converted parser error");

	for (;;) {
		if (*in < '0' || *in > '9') {
			break;
		}

		const uint64_t new_value = (value * (uint64_t)10) + ((uint64_t)(*in - '0'));

		// numeric overflow, we rely on you
		if (new_value < value) {
			//WARN_PRINT( "Converting the string \" " + in + " \" into a value resulted in overflow." );
			return 0;
		}

		value = new_value;

		++in;
		++cur;

		if (max_inout && *max_inout == cur) {
			if (out) { /* skip to end */
				while (*in >= '0' && *in <= '9') {
					++in;
				}
				*out = in;
			}

			return value;
		}
	}
	if (out) {
		*out = in;
	}

	if (max_inout) {
		*max_inout = cur;
	}

	return value;
}

#endif // FBX_PARSE_TOOLS_H
