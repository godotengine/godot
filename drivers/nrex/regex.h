/*************************************************/
/*  regex.h                                      */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2016 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#ifndef REGEX_H
#define REGEX_H

#include "ustring.h"
#include "vector.h"
#include "core/reference.h"
#include "nrex.hpp"

class RegEx : public Reference {

	OBJ_TYPE(RegEx, Reference);

	mutable String text;
	mutable Vector<nrex_result> captures;
	nrex exp;

protected:

	static void _bind_methods();
	StringArray _bind_get_captures() const;

public:

	void clear();
	bool is_valid() const;
	int get_capture_count() const;
	String get_capture(int capture) const;
	Error compile(const String& p_pattern, int capture = 9);
	int find(const String& p_text, int p_start = 0, int p_end = -1) const;

	RegEx();
	RegEx(const String& p_pattern);
	~RegEx();
};

#endif // REGEX_H
