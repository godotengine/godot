/**************************************************************************/
/*  semver.cpp                                                            */
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

#include "semver.h"

bool godotsharp::SemVer::parse_digit_only_field(const String &p_field, uint64_t &r_result) {
	if (p_field.is_empty()) {
		return false;
	}

	int64_t integer = 0;

	for (int i = 0; i < p_field.length(); i++) {
		char32_t c = p_field[i];
		if (is_digit(c)) {
			bool overflow = ((uint64_t)integer > UINT64_MAX / 10) || ((uint64_t)integer == UINT64_MAX / 10 && c > '5');
			ERR_FAIL_COND_V_MSG(overflow, false, "Cannot represent '" + p_field + "' as a 64-bit unsigned integer, since the value is too large.");
			integer *= 10;
			integer += c - '0';
		} else {
			return false;
		}
	}

	r_result = (uint64_t)integer;
	return true;
}

int godotsharp::SemVer::cmp(const godotsharp::SemVer &p_a, const godotsharp::SemVer &p_b) {
	if (p_a.major != p_b.major) {
		return p_a.major > p_b.major ? 1 : -1;
	}

	if (p_a.minor != p_b.minor) {
		return p_a.minor > p_b.minor ? 1 : -1;
	}

	if (p_a.patch != p_b.patch) {
		return p_a.patch > p_b.patch ? 1 : -1;
	}

	if (p_a.prerelease.is_empty() && p_b.prerelease.is_empty()) {
		return 0;
	}

	if (p_a.prerelease.is_empty() || p_b.prerelease.is_empty()) {
		return p_a.prerelease.is_empty() ? 1 : -1;
	}

	if (p_a.prerelease != p_b.prerelease) {
		// This could be optimized, but I'm too lazy

		Vector<String> a_field_set = p_a.prerelease.split(".");
		Vector<String> b_field_set = p_b.prerelease.split(".");

		int a_field_count = a_field_set.size();
		int b_field_count = b_field_set.size();

		int min_field_count = MIN(a_field_count, b_field_count);

		for (int i = 0; i < min_field_count; i++) {
			const String &a_field = a_field_set[i];
			const String &b_field = b_field_set[i];

			if (a_field == b_field) {
				continue;
			}

			uint64_t a_num;
			bool a_is_digit_only = parse_digit_only_field(a_field, a_num);

			uint64_t b_num;
			bool b_is_digit_only = parse_digit_only_field(b_field, b_num);

			if (a_is_digit_only && b_is_digit_only) {
				// Identifiers consisting of only digits are compared numerically.

				if (a_num == b_num) {
					continue;
				}

				return a_num > b_num ? 1 : -1;
			}

			if (a_is_digit_only || b_is_digit_only) {
				// Numeric identifiers always have lower precedence than non-numeric identifiers.
				return b_is_digit_only ? 1 : -1;
			}

			// Identifiers with letters or hyphens are compared lexically in ASCII sort order.
			return a_field > b_field ? 1 : -1;
		}

		if (a_field_count != b_field_count) {
			// A larger set of pre-release fields has a higher precedence than a smaller set, if all of the preceding identifiers are equal.
			return a_field_count > b_field_count ? 1 : -1;
		}
	}

	return 0;
}

bool godotsharp::SemVerParser::parse(const String &p_ver_text, godotsharp::SemVer &r_semver) {
	if (!regex.is_valid() && regex.get_pattern().is_empty()) {
		regex.compile("^(?P<major>0|[1-9]\\d*)\\.(?P<minor>0|[1-9]\\d*)\\.(?P<patch>0|[1-9]\\d*)(?:-(?P<prerelease>(?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\\.(?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\\.[0-9a-zA-Z-]+)*))?$");
		ERR_FAIL_COND_V(!regex.is_valid(), false);
	}

	Ref<RegExMatch> match = regex.search(p_ver_text);

	if (match.is_valid()) {
		r_semver = SemVer(
				match->get_string("major").to_int(),
				match->get_string("minor").to_int(),
				match->get_string("patch").to_int(),
				match->get_string("prerelease"),
				match->get_string("buildmetadata"));
		return true;
	}

	return false;
}
