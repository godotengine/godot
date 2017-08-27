/*************************************************************************/
/*  regex.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "nrex.hpp"

void RegEx::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("compile", "pattern", "capture"), &RegEx::compile, DEFVAL(9));
	ObjectTypeDB::bind_method(_MD("find", "text", "start", "end"), &RegEx::find, DEFVAL(0), DEFVAL(-1));
	ObjectTypeDB::bind_method(_MD("clear"), &RegEx::clear);
	ObjectTypeDB::bind_method(_MD("is_valid"), &RegEx::is_valid);
	ObjectTypeDB::bind_method(_MD("get_capture_count"), &RegEx::get_capture_count);
	ObjectTypeDB::bind_method(_MD("get_capture", "capture"), &RegEx::get_capture);
	ObjectTypeDB::bind_method(_MD("get_capture_start", "capture"), &RegEx::get_capture_start);
	ObjectTypeDB::bind_method(_MD("get_captures"), &RegEx::_bind_get_captures);
};

StringArray RegEx::_bind_get_captures() const {

	StringArray ret;
	int count = get_capture_count();
	for (int i = 0; i < count; i++) {

		String c = get_capture(i);
		ret.push_back(c);
	};

	return ret;
};

void RegEx::clear() {

	text.clear();
	captures.clear();
	exp.reset();
};

bool RegEx::is_valid() const {

	return exp.valid();
};

int RegEx::get_capture_count() const {

	ERR_FAIL_COND_V(!exp.valid(), 0);

	return exp.capture_size();
}

String RegEx::get_capture(int capture) const {

	ERR_FAIL_COND_V(get_capture_count() <= capture, String());

	return text.substr(captures[capture].start, captures[capture].length);
}

int RegEx::get_capture_start(int capture) const {

	ERR_FAIL_COND_V(get_capture_count() <= capture, -1);

	return captures[capture].start;
}

Error RegEx::compile(const String &p_pattern, int capture) {

	clear();

	exp.compile(p_pattern.c_str(), capture);

	ERR_FAIL_COND_V(!exp.valid(), FAILED);

	captures.resize(exp.capture_size());

	return OK;
};

int RegEx::find(const String &p_text, int p_start, int p_end) const {

	ERR_FAIL_COND_V(!exp.valid(), -1);
	ERR_FAIL_COND_V(p_text.length() < p_start, -1);
	ERR_FAIL_COND_V(p_text.length() < p_end, -1);

	bool res = exp.match(p_text.c_str(), &captures[0], p_start, p_end);

	if (res) {
		text = p_text;
		return captures[0].start;
	}
	text.clear();
	return -1;
};

RegEx::RegEx(const String &p_pattern) {

	compile(p_pattern);
};

RegEx::RegEx(){

};

RegEx::~RegEx() {

	clear();
};
