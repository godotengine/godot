/**************************************************************************/
/*  regex.cpp                                                             */
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

#include "regex.h"

#include "core/os/memory.h"

#ifndef PCRE2_STATIC
#define PCRE2_STATIC
#endif

#ifndef PCRE2_CODE_UNIT_WIDTH
#define PCRE2_CODE_UNIT_WIDTH 32
#endif

#include <thirdparty/pcre2/src/pcre2.h>

static void *_regex_malloc(PCRE2_SIZE p_size, void *p_user) {
	return memalloc(p_size);
}

static void _regex_free(void *p_ptr, void *p_user) {
	if (p_ptr) {
		memfree(p_ptr);
	}
}

int RegExMatch::_find(const String &p_name) const {
	const HashMap<String, int>::ConstIterator found = _names.find(p_name);
	if (found) {
		return found->value;
	}
	return -1;
}

int RegExMatch::_find(int p_index) const {
	if (p_index >= _data.size()) {
		return -1;
	}
	return p_index;
}

Vector<String> RegExMatch::get_strings() const {
	Vector<String> result;

	int size = _data.size();

	for (int i = 0; i < size; i++) {
		int start = _data[i].start;

		if (start == -1) {
			result.append(String());
			continue;
		}

		int length = _data[i].end - start;
		result.append(_subject.substr(start, length));
	}

	return result;
}

String RegExMatch::_get_string(int p_found) const {
	if (p_found < 0) {
		return String();
	}

	int start = _data[p_found].start;

	if (start == -1) {
		return String();
	}

	int length = _data[p_found].end - start;
	return _subject.substr(start, length);
}

int RegExMatch::_get_start(int p_found) const {
	return p_found < 0 ? -1 : _data[p_found].start;
}

int RegExMatch::_get_end(int p_found) const {
	return p_found < 0 ? -1 : _data[p_found].end;
}

void RegEx::_pattern_info(uint32_t p_what, void *r_where) const {
	pcre2_pattern_info_32((pcre2_code_32 *)_code, p_what, r_where);
}

void RegEx::clear() {
	if (_code) {
		pcre2_code_free_32((pcre2_code_32 *)_code);
		_code = nullptr;
	}
}

Error RegEx::compile(const String &p_pattern, bool p_show_error) {
	_pattern = p_pattern;
	clear();

	int err;
	PCRE2_SIZE offset;
	uint32_t flags = PCRE2_DUPNAMES;

	pcre2_general_context_32 *gctx = (pcre2_general_context_32 *)_general_context;
	pcre2_compile_context_32 *cctx = pcre2_compile_context_create_32(gctx);
	PCRE2_SPTR32 p = (PCRE2_SPTR32)_pattern.get_data();

	_code = pcre2_compile_32(p, _pattern.length(), flags, &err, &offset, cctx);

	pcre2_compile_context_free_32(cctx);

	if (!_code) {
		if (p_show_error) {
			PCRE2_UCHAR32 buf[256];
			pcre2_get_error_message_32(err, buf, 256);
			String message = String::num_int64(offset) + ": " + String((const char32_t *)buf);
			ERR_PRINT(message);
		}
		return FAILED;
	}
	return OK;
}

RegExMatch RegEx::search(const String &p_subject, int p_offset, int p_end) const {
	ERR_FAIL_COND_V(!is_valid(), RegExMatch());
	ERR_FAIL_COND_V_MSG(p_offset < 0, RegExMatch(), "RegEx search offset must be >= 0");

	RegExMatch result;

	int length = p_subject.length();
	if (p_end >= 0 && p_end < length) {
		length = p_end;
	}

	pcre2_code_32 *c = (pcre2_code_32 *)_code;
	pcre2_general_context_32 *gctx = (pcre2_general_context_32 *)_general_context;
	pcre2_match_context_32 *mctx = pcre2_match_context_create_32(gctx);
	PCRE2_SPTR32 s = (PCRE2_SPTR32)p_subject.get_data();

	pcre2_match_data_32 *match = pcre2_match_data_create_from_pattern_32(c, gctx);

	int res = pcre2_match_32(c, s, length, p_offset, 0, match, mctx);

	if (res < 0) {
		pcre2_match_data_free_32(match);
		pcre2_match_context_free_32(mctx);

		return RegExMatch();
	}

	uint32_t size = pcre2_get_ovector_count_32(match);
	PCRE2_SIZE *ovector = pcre2_get_ovector_pointer_32(match);

	result._data.resize(size);

	for (uint32_t i = 0; i < size; i++) {
		result._data.write[i].start = ovector[i * 2];
		result._data.write[i].end = ovector[i * 2 + 1];
	}

	pcre2_match_data_free_32(match);
	pcre2_match_context_free_32(mctx);

	result._subject = p_subject;

	uint32_t count;
	const char32_t *table;
	uint32_t entry_size;

	_pattern_info(PCRE2_INFO_NAMECOUNT, &count);
	_pattern_info(PCRE2_INFO_NAMETABLE, &table);
	_pattern_info(PCRE2_INFO_NAMEENTRYSIZE, &entry_size);

	for (uint32_t i = 0; i < count; i++) {
		char32_t id = table[i * entry_size];
		if (result._data[id].start == -1) {
			continue;
		}
		String name = &table[i * entry_size + 1];
		if (result._names.has(name)) {
			continue;
		}

		result._names.insert(name, id);
	}

	return result;
}

Vector<RegExMatch> RegEx::search_all(const String &p_subject, int p_offset, int p_end) const {
	ERR_FAIL_COND_V_MSG(p_offset < 0, {}, "RegEx search offset must be >= 0");

	int last_end = 0;
	Vector<RegExMatch> result;
	RegExMatch match = search(p_subject, p_offset, p_end);

	while (match.is_valid()) {
		last_end = match.get_end(0);
		if (match.get_start(0) == last_end) {
			last_end++;
		}

		result.push_back(match);
		match = search(p_subject, last_end, p_end);
	}
	return result;
}

int RegEx::_sub(const String &p_subject, const String &p_replacement, int p_offset, int p_end, uint32_t p_flags, String &r_output) const {
	// `safety_zone` is the number of chars we allocate in addition to the number of chars expected in order to
	// guard against the PCRE API writing one additional `\0` at the end. PCRE's API docs are unclear on whether
	// PCRE understands outlength in `pcre2_substitute(`) as counting an implicit additional terminating char or
	// not. Always allocating one char more than telling PCRE has us on the safe side.
	const int safety_zone = 1;

	PCRE2_SIZE olength = p_subject.length() + 1; // Space for output string and one terminating `\0` character.
	Vector<char32_t> output;
	output.resize(olength + safety_zone);

	PCRE2_SIZE length = p_subject.length();
	if (p_end >= 0 && (uint32_t)p_end < length) {
		length = p_end;
	}

	pcre2_code_32 *c = (pcre2_code_32 *)_code;
	pcre2_general_context_32 *gctx = (pcre2_general_context_32 *)_general_context;
	pcre2_match_context_32 *mctx = pcre2_match_context_create_32(gctx);
	PCRE2_SPTR32 s = (PCRE2_SPTR32)p_subject.get_data();
	PCRE2_SPTR32 r = (PCRE2_SPTR32)p_replacement.get_data();
	PCRE2_UCHAR32 *o = (PCRE2_UCHAR32 *)output.ptrw();

	pcre2_match_data_32 *match = pcre2_match_data_create_from_pattern_32(c, gctx);

	int res = pcre2_substitute_32(c, s, length, p_offset, p_flags, match, mctx, r, p_replacement.length(), o, &olength);

	if (res == PCRE2_ERROR_NOMEMORY) {
		output.resize(olength + safety_zone);
		o = (PCRE2_UCHAR32 *)output.ptrw();
		res = pcre2_substitute_32(c, s, length, p_offset, p_flags, match, mctx, r, p_replacement.length(), o, &olength);
	}

	pcre2_match_data_free_32(match);
	pcre2_match_context_free_32(mctx);

	if (res >= 0) {
		r_output = String::utf32(Span(output.ptr(), olength)) + p_subject.substr(length);
	}

	return res;
}

String RegEx::sub(const String &p_subject, const String &p_replacement, bool p_all, int p_offset, int p_end) const {
	ERR_FAIL_COND_V(!is_valid(), String());
	ERR_FAIL_COND_V_MSG(p_offset < 0, String(), "RegEx sub offset must be >= 0");

	uint32_t flags = PCRE2_SUBSTITUTE_OVERFLOW_LENGTH | PCRE2_SUBSTITUTE_UNSET_EMPTY;
	if (p_all) {
		flags |= PCRE2_SUBSTITUTE_GLOBAL;
	}

	String output;
	const int res = _sub(p_subject, p_replacement, p_offset, p_end, flags, output);

	if (res < 0) {
		PCRE2_UCHAR32 buf[256];
		pcre2_get_error_message_32(res, buf, 256);
		String message = "PCRE2 Error: " + String((const char32_t *)buf);
		ERR_PRINT(message);

		if (res == PCRE2_ERROR_NOSUBSTRING) {
			flags |= PCRE2_SUBSTITUTE_UNKNOWN_UNSET;
			_sub(p_subject, p_replacement, p_offset, p_end, flags, output);
		}
	}

	return output;
}

int RegEx::get_group_count() const {
	ERR_FAIL_COND_V(!is_valid(), 0);

	uint32_t count;

	_pattern_info(PCRE2_INFO_CAPTURECOUNT, &count);

	return count;
}

Vector<String> RegEx::get_names() const {
	Vector<String> result;

	ERR_FAIL_COND_V(!is_valid(), result);

	uint32_t count;
	const char32_t *table;
	uint32_t entry_size;

	_pattern_info(PCRE2_INFO_NAMECOUNT, &count);
	_pattern_info(PCRE2_INFO_NAMETABLE, &table);
	_pattern_info(PCRE2_INFO_NAMEENTRYSIZE, &entry_size);

	for (uint32_t i = 0; i < count; i++) {
		String name = &table[i * entry_size + 1];
		if (!result.has(name)) {
			result.append(name);
		}
	}

	return result;
}

RegEx::RegEx() {
	_general_context = pcre2_general_context_create_32(&_regex_malloc, &_regex_free, nullptr);
}

RegEx::RegEx(const String &p_pattern, bool p_show_error) : RegEx() {
	compile(p_pattern, p_show_error);
}

RegEx::~RegEx() {
	if (_code) {
		pcre2_code_free_32((pcre2_code_32 *)_code);
	}
	pcre2_general_context_free_32((pcre2_general_context_32 *)_general_context);
}
