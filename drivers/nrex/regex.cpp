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

	ObjectTypeDB::bind_method(_MD("compile","pattern"),&RegEx::compile);
	ObjectTypeDB::bind_method(_MD("match","text","start","end"),&RegEx::match, DEFVAL(0), DEFVAL(-1));
	ObjectTypeDB::bind_method(_MD("get_capture","capture"),&RegEx::get_capture);
	ObjectTypeDB::bind_method(_MD("get_capture_list"),&RegEx::_bind_get_capture_list);

};

StringArray RegEx::_bind_get_capture_list() const {

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
	
	return exp.capture_size();
}

String RegEx::get_capture(int capture) const {

	ERR_FAIL_COND_V( get_capture_count() <= capture, String() );

	return text.substr(captures[capture].start, captures[capture].length);

}

Error RegEx::compile(const String& p_pattern) {
	
	clear();

	exp.compile(p_pattern.c_str());

	ERR_FAIL_COND_V( !exp.valid(), FAILED );
	
	captures.resize(exp.capture_size());

	return OK;

};

bool RegEx::match(const String& p_text, int p_start, int p_end) const {

	
	ERR_FAIL_COND_V( !exp.valid(), false );
	ERR_FAIL_COND_V( p_text.length() < p_start, false );
	ERR_FAIL_COND_V( p_text.length() < p_end, false );

	bool res = exp.match(p_text.c_str(), &captures[0], p_start, p_end);

	if (res) {
		text = p_text;
		return true;
	}
	text.clear();
	return false;

};

RegEx::RegEx(const String& p_pattern) {

	compile(p_pattern);

};

RegEx::RegEx() {

};

RegEx::~RegEx() {

	clear();

};
