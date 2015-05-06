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

extern "C" {

#define _UNICODE
#include "trex.h"

};

void RegEx::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("compile","pattern"),&RegEx::compile);
	ObjectTypeDB::bind_method(_MD("find","text", "start","end"),&RegEx::_bind_find, DEFVAL(0), DEFVAL(-1));
	ObjectTypeDB::bind_method(_MD("get_captures"),&RegEx::_bind_get_captures);
};

Error RegEx::compile(const String& p_pattern) {

	clear();
	const TRexChar* error;
	exp = trex_compile(p_pattern.c_str(), &error);
	ERR_FAIL_COND_V(!exp, FAILED);
	return OK;
};


int RegEx::_bind_find(const String& p_text, int p_start, int p_end) const {

	int start, end;
	bool ret = find(p_text, start, end, NULL, p_start, p_end);

	return ret?start:-1;
};

bool RegEx::find(const String& p_text, int& p_rstart, int &p_rend, List<String>* p_captures, int p_start, int p_end) const {

	ERR_FAIL_COND_V( !exp, false );
	text=p_text;

	const CharType* str = p_text.c_str();
	const CharType* start = str + p_start;
	const CharType* end = str + (p_end == -1?p_text.size():p_end);

	const CharType* out_begin;
	const CharType* out_end;

	bool ret = trex_searchrange(exp, start, end, &out_begin, &out_end);
	if (ret) {

		p_rstart = out_begin - str;
		p_rend = out_end - str;

		if (p_captures) {

			int count = get_capture_count();
			for (int i=0; i<count; i++) {

				int start, len;
				get_capture_limits(i, start, len);
				p_captures->push_back(p_text.substr(start, len));
			};
		};
	} else {

		p_rstart = -1;
	};

	return ret;
};


bool RegEx::match(const String& p_text, List<String>* p_captures, int p_start, int p_end) const {

	ERR_FAIL_COND_V( !exp, false );

	int start, end;
	return find(p_text, start, end, p_captures, p_start, p_end);
};

int RegEx::get_capture_count() const {

	ERR_FAIL_COND_V( exp == NULL, -1 );

	return trex_getsubexpcount(exp);
};

Error RegEx::get_capture_limits(int p_capture, int& p_start, int& p_len) const {

	ERR_FAIL_COND_V( exp == NULL, ERR_UNCONFIGURED );

	TRexMatch match;
	TRexBool res = trex_getsubexp(exp, p_capture, &match);
	ERR_FAIL_COND_V( !res, FAILED );
	p_start = (int)(match.begin - text.c_str());
	p_len = match.len;

	return OK;
};

String RegEx::get_capture(int p_idx) const {

	ERR_FAIL_COND_V( exp == NULL, "" );
	int start, len;
	Error ret = get_capture_limits(p_idx, start, len);
	ERR_FAIL_COND_V(ret != OK, "");
	if (len == 0)
		return "";
	return text.substr(start, len);
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

bool RegEx::is_valid() const {

	return exp != NULL;
};

void RegEx::clear() {

	if (exp) {

		trex_free(exp);
		exp = NULL;
	};
};

RegEx::RegEx(const String& p_pattern) {

	exp = NULL;
	compile(p_pattern);
};

RegEx::RegEx() {

	exp = NULL;
};

RegEx::~RegEx() {

	clear();
};
