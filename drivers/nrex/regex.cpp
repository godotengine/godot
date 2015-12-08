/*************************************************/
/*  regex.cpp                                    */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2010 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#include "regex.h"
#include "nrex.hpp"
#include "core/os/memory.h"

void RegEx::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("compile","pattern", "capture"),&RegEx::compile, DEFVAL(9));
	ObjectTypeDB::bind_method(_MD("find","text","start","end"),&RegEx::find, DEFVAL(0), DEFVAL(-1));
	ObjectTypeDB::bind_method(_MD("clear"),&RegEx::clear);
	ObjectTypeDB::bind_method(_MD("is_valid"),&RegEx::is_valid);
	ObjectTypeDB::bind_method(_MD("get_capture_count"),&RegEx::get_capture_count);
	ObjectTypeDB::bind_method(_MD("get_capture","capture"),&RegEx::get_capture);
	ObjectTypeDB::bind_method(_MD("get_captures"),&RegEx::_bind_get_captures);

};

StringArray RegEx::_bind_get_captures() const {

	StringArray ret;
	int count = get_capture_count();
	for (int i=0; i<count; i++) {

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

	ERR_FAIL_COND_V( !exp.valid(), 0 );

	return exp.capture_size();
}

String RegEx::get_capture(int capture) const {

	ERR_FAIL_COND_V( get_capture_count() <= capture, String() );

	return text.substr(captures[capture].start, captures[capture].length);

}

Error RegEx::compile(const String& p_pattern, int capture) {

	clear();

	exp.compile(p_pattern.c_str(), capture);

	ERR_FAIL_COND_V( !exp.valid(), FAILED );

	captures.resize(exp.capture_size());

	return OK;

};

int RegEx::find(const String& p_text, int p_start, int p_end) const {

	ERR_FAIL_COND_V( !exp.valid(), -1 );
	ERR_FAIL_COND_V( p_text.length() < p_start, -1 );
	ERR_FAIL_COND_V( p_text.length() < p_end, -1 );

	bool res = exp.match(p_text.c_str(), &captures[0], p_start, p_end);

	if (res) {
		text = p_text;
		return captures[0].start;
	}
	text.clear();
	return -1;

};

RegEx::RegEx(const String& p_pattern) {

	compile(p_pattern);

};

RegEx::RegEx() {

};

RegEx::~RegEx() {

	clear();

};
