/*************************************************/
/*  regex.h                                      */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2010 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#ifndef REGEX_H
#define REGEX_H

#include "ustring.h"
#include "list.h"
#include "core/reference.h"
struct TRex;

class RegEx : public Reference {

	OBJ_TYPE(RegEx, Reference);

	mutable String text;
	TRex *exp;

protected:

	static void _bind_methods();

	int _bind_find(const String& p_text, int p_start = 0, int p_end = -1) const;
	StringArray _bind_get_captures() const;
public:

	void clear();

	Error compile(const String& p_pattern);
	bool is_valid() const;
	bool match(const String& p_text, List<String>* p_captures = NULL, int p_start = 0, int p_end = -1) const;
	bool find(const String& p_text, int& p_rstart, int &p_rend, List<String>* p_captures = NULL, int p_start = 0, int p_end = -1) const;
	int get_capture_count() const;
	Error get_capture_limits(int p_capture, int& p_start, int& p_len) const;
	String get_capture(int p_idx) const;

	RegEx();
	RegEx(const String& p_pattern);
	~RegEx();
};

#endif // REGEX_H
