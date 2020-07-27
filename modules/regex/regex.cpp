/*************************************************************************/
/*  regex.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "regex.h"
#include "core/os/memory.h"

extern "C" {
#include <pcre2.h>
}

static void *_regex_malloc(PCRE2_SIZE size, void *user) {
	return memalloc(size);
}

static void _regex_free(void *ptr, void *user) {
	memfree(ptr);
}

int RegExMatch::_find(const Variant &p_name) const {
	if (p_name.is_num()) {
		int i = (int)p_name;
		if (i >= data.size()) {
			return -1;
		}
		return i;

	} else if (p_name.get_type() == Variant::STRING) {
		const Map<String, int>::Element *found = names.find((String)p_name);
		if (found) {
			return found->value();
		}
	}

	return -1;
}

String RegExMatch::get_subject() const {
	return subject;
}

int RegExMatch::get_group_count() const {
	if (data.size() == 0) {
		return 0;
	}
	return data.size() - 1;
}

Dictionary RegExMatch::get_names() const {
	Dictionary result;

	for (const Map<String, int>::Element *i = names.front(); i != nullptr; i = i->next()) {
		result[i->key()] = i->value();
	}

	return result;
}

Array RegExMatch::get_strings() const {
	Array result;

	int size = data.size();

	for (int i = 0; i < size; i++) {
		int start = data[i].start;

		if (start == -1) {
			result.append(String());
			continue;
		}

		int length = data[i].end - start;

		result.append(subject.substr(start, length));
	}

	return result;
}

String RegExMatch::get_string(const Variant &p_name) const {
	int id = _find(p_name);

	if (id < 0) {
		return String();
	}

	int start = data[id].start;

	if (start == -1) {
		return String();
	}

	int length = data[id].end - start;

	return subject.substr(start, length);
}

int RegExMatch::get_start(const Variant &p_name) const {
	int id = _find(p_name);

	if (id < 0) {
		return -1;
	}

	return data[id].start;
}

int RegExMatch::get_end(const Variant &p_name) const {
	int id = _find(p_name);

	if (id < 0) {
		return -1;
	}

	return data[id].end;
}

void RegExMatch::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_subject"), &RegExMatch::get_subject);
	ClassDB::bind_method(D_METHOD("get_group_count"), &RegExMatch::get_group_count);
	ClassDB::bind_method(D_METHOD("get_names"), &RegExMatch::get_names);
	ClassDB::bind_method(D_METHOD("get_strings"), &RegExMatch::get_strings);
	ClassDB::bind_method(D_METHOD("get_string", "name"), &RegExMatch::get_string, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_start", "name"), &RegExMatch::get_start, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_end", "name"), &RegExMatch::get_end, DEFVAL(0));

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "subject"), "", "get_subject");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "names"), "", "get_names");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "strings"), "", "get_strings");
}

void RegEx::_pattern_info(uint32_t what, void *where) const {
	pcre2_pattern_info_32((pcre2_code_32 *)code, what, where);
}

void RegEx::clear() {
	if (code) {
		pcre2_code_free_32((pcre2_code_32 *)code);
		code = nullptr;
	}
}

Error RegEx::compile(const String &p_pattern) {
	pattern = p_pattern;
	clear();

	int err;
	PCRE2_SIZE offset;
	uint32_t flags = PCRE2_DUPNAMES;

	pcre2_general_context_32 *gctx = (pcre2_general_context_32 *)general_ctx;
	pcre2_compile_context_32 *cctx = pcre2_compile_context_create_32(gctx);
	PCRE2_SPTR32 p = (PCRE2_SPTR32)pattern.get_data();

	code = pcre2_compile_32(p, pattern.length(), flags, &err, &offset, cctx);

	pcre2_compile_context_free_32(cctx);

	if (!code) {
		PCRE2_UCHAR32 buf[256];
		pcre2_get_error_message_32(err, buf, 256);
		String message = String::num(offset) + ": " + String((const char32_t *)buf);
		ERR_PRINT(message.utf8());
		return FAILED;
	}
	return OK;
}

Ref<RegExMatch> RegEx::search(const String &p_subject, int p_offset, int p_end) const {
	ERR_FAIL_COND_V(!is_valid(), nullptr);

	Ref<RegExMatch> result = memnew(RegExMatch);

	int length = p_subject.length();
	if (p_end >= 0 && p_end < length) {
		length = p_end;
	}

	pcre2_code_32 *c = (pcre2_code_32 *)code;
	pcre2_general_context_32 *gctx = (pcre2_general_context_32 *)general_ctx;
	pcre2_match_context_32 *mctx = pcre2_match_context_create_32(gctx);
	PCRE2_SPTR32 s = (PCRE2_SPTR32)p_subject.get_data();

	pcre2_match_data_32 *match = pcre2_match_data_create_from_pattern_32(c, gctx);

	int res = pcre2_match_32(c, s, length, p_offset, 0, match, mctx);

	if (res < 0) {
		pcre2_match_data_free_32(match);
		pcre2_match_context_free_32(mctx);

		return nullptr;
	}

	uint32_t size = pcre2_get_ovector_count_32(match);
	PCRE2_SIZE *ovector = pcre2_get_ovector_pointer_32(match);

	result->data.resize(size);

	for (uint32_t i = 0; i < size; i++) {
		result->data.write[i].start = ovector[i * 2];
		result->data.write[i].end = ovector[i * 2 + 1];
	}

	pcre2_match_data_free_32(match);
	pcre2_match_context_free_32(mctx);

	result->subject = p_subject;

	uint32_t count;
	const char32_t *table;
	uint32_t entry_size;

	_pattern_info(PCRE2_INFO_NAMECOUNT, &count);
	_pattern_info(PCRE2_INFO_NAMETABLE, &table);
	_pattern_info(PCRE2_INFO_NAMEENTRYSIZE, &entry_size);

	for (uint32_t i = 0; i < count; i++) {
		char32_t id = table[i * entry_size];
		if (result->data[id].start == -1) {
			continue;
		}
		String name = &table[i * entry_size + 1];
		if (result->names.has(name)) {
			continue;
		}

		result->names.insert(name, id);
	}

	return result;
}

Array RegEx::search_all(const String &p_subject, int p_offset, int p_end) const {
	int last_end = -1;
	Array result;
	Ref<RegExMatch> match = search(p_subject, p_offset, p_end);
	while (match.is_valid()) {
		if (last_end == match->get_end(0)) {
			break;
		}
		result.push_back(match);
		last_end = match->get_end(0);
		match = search(p_subject, match->get_end(0), p_end);
	}
	return result;
}

String RegEx::sub(const String &p_subject, const String &p_replacement, bool p_all, int p_offset, int p_end) const {
	ERR_FAIL_COND_V(!is_valid(), String());

	// safety_zone is the number of chars we allocate in addition to the number of chars expected in order to
	// guard against the PCRE API writing one additional \0 at the end. PCRE's API docs are unclear on whether
	// PCRE understands outlength in pcre2_substitute() as counting an implicit additional terminating char or
	// not. always allocating one char more than telling PCRE has us on the safe side.
	const int safety_zone = 1;

	PCRE2_SIZE olength = p_subject.length() + 1; // space for output string and one terminating \0 character
	Vector<char32_t> output;
	output.resize(olength + safety_zone);

	uint32_t flags = PCRE2_SUBSTITUTE_OVERFLOW_LENGTH;
	if (p_all) {
		flags |= PCRE2_SUBSTITUTE_GLOBAL;
	}

	PCRE2_SIZE length = p_subject.length();
	if (p_end >= 0 && (uint32_t)p_end < length) {
		length = p_end;
	}

	pcre2_code_32 *c = (pcre2_code_32 *)code;
	pcre2_general_context_32 *gctx = (pcre2_general_context_32 *)general_ctx;
	pcre2_match_context_32 *mctx = pcre2_match_context_create_32(gctx);
	PCRE2_SPTR32 s = (PCRE2_SPTR32)p_subject.get_data();
	PCRE2_SPTR32 r = (PCRE2_SPTR32)p_replacement.get_data();
	PCRE2_UCHAR32 *o = (PCRE2_UCHAR32 *)output.ptrw();

	pcre2_match_data_32 *match = pcre2_match_data_create_from_pattern_32(c, gctx);

	int res = pcre2_substitute_32(c, s, length, p_offset, flags, match, mctx, r, p_replacement.length(), o, &olength);

	if (res == PCRE2_ERROR_NOMEMORY) {
		output.resize(olength + safety_zone);
		o = (PCRE2_UCHAR32 *)output.ptrw();
		res = pcre2_substitute_32(c, s, length, p_offset, flags, match, mctx, r, p_replacement.length(), o, &olength);
	}

	pcre2_match_data_free_32(match);
	pcre2_match_context_free_32(mctx);

	if (res < 0) {
		return String();
	}

	return String(output.ptr(), olength);
}

bool RegEx::is_valid() const {
	return (code != nullptr);
}

String RegEx::get_pattern() const {
	return pattern;
}

int RegEx::get_group_count() const {
	ERR_FAIL_COND_V(!is_valid(), 0);

	uint32_t count;

	_pattern_info(PCRE2_INFO_CAPTURECOUNT, &count);

	return count;
}

Array RegEx::get_names() const {
	Array result;

	ERR_FAIL_COND_V(!is_valid(), result);

	uint32_t count;
	const char32_t *table;
	uint32_t entry_size;

	_pattern_info(PCRE2_INFO_NAMECOUNT, &count);
	_pattern_info(PCRE2_INFO_NAMETABLE, &table);
	_pattern_info(PCRE2_INFO_NAMEENTRYSIZE, &entry_size);

	for (uint32_t i = 0; i < count; i++) {
		String name = &table[i * entry_size + 1];
		if (result.find(name) < 0) {
			result.append(name);
		}
	}

	return result;
}

RegEx::RegEx() {
	general_ctx = pcre2_general_context_create_32(&_regex_malloc, &_regex_free, nullptr);
	code = nullptr;
}

RegEx::RegEx(const String &p_pattern) {
	general_ctx = pcre2_general_context_create_32(&_regex_malloc, &_regex_free, nullptr);
	code = nullptr;
	compile(p_pattern);
}

RegEx::~RegEx() {
	if (code) {
		pcre2_code_free_32((pcre2_code_32 *)code);
	}
	pcre2_general_context_free_32((pcre2_general_context_32 *)general_ctx);
}

void RegEx::_bind_methods() {
	ClassDB::bind_method(D_METHOD("clear"), &RegEx::clear);
	ClassDB::bind_method(D_METHOD("compile", "pattern"), &RegEx::compile);
	ClassDB::bind_method(D_METHOD("search", "subject", "offset", "end"), &RegEx::search, DEFVAL(0), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("search_all", "subject", "offset", "end"), &RegEx::search_all, DEFVAL(0), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("sub", "subject", "replacement", "all", "offset", "end"), &RegEx::sub, DEFVAL(false), DEFVAL(0), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("is_valid"), &RegEx::is_valid);
	ClassDB::bind_method(D_METHOD("get_pattern"), &RegEx::get_pattern);
	ClassDB::bind_method(D_METHOD("get_group_count"), &RegEx::get_group_count);
	ClassDB::bind_method(D_METHOD("get_names"), &RegEx::get_names);
}
