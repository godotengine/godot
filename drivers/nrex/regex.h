/*************************************************************************/
/*  regex.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef REGEX_H
#define REGEX_H

#include "core/reference.h"
#include "nrex.hpp"
#include "ustring.h"
#include "vector.h"

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
	int get_capture_start(int capture) const;
	String get_capture(int capture) const;
	Error compile(const String &p_pattern, int capture = 9);
	int find(const String &p_text, int p_start = 0, int p_end = -1) const;

	RegEx();
	RegEx(const String &p_pattern);
	~RegEx();
};

#endif // REGEX_H
