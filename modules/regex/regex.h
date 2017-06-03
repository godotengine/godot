/*************************************************************************/
/*  regex.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/dictionary.h"
#include "core/reference.h"
#include "core/resource.h"
#include "core/ustring.h"
#include "core/vector.h"

class RegExNode;

class RegExMatch : public Reference {

	GDCLASS(RegExMatch, Reference);

	struct Group {
		Variant name;
		int start;
		int length;
	};

	Vector<Group> captures;
	String string;

	friend class RegEx;
	friend class RegExSearch;
	friend class RegExNodeCapturing;
	friend class RegExNodeBackReference;

protected:
	static void _bind_methods();

public:
	String expand(const String &p_template) const;

	int get_group_count() const;
	Array get_group_array() const;

	Array get_names() const;
	Dictionary get_name_dict() const;

	String get_string(const Variant &p_name) const;
	int get_start(const Variant &p_name) const;
	int get_end(const Variant &p_name) const;

	RegExMatch();
};

class RegEx : public Resource {

	GDCLASS(RegEx, Resource);

	RegExNode *root;
	Vector<Variant> group_names;
	String pattern;
	int lookahead_depth;

protected:
	static void _bind_methods();

public:
	void clear();
	Error compile(const String &p_pattern);

	Ref<RegExMatch> search(const String &p_text, int p_start = 0, int p_end = -1) const;
	String sub(const String &p_text, const String &p_replacement, bool p_all = false, int p_start = 0, int p_end = -1) const;

	bool is_valid() const;
	String get_pattern() const;
	int get_group_count() const;
	Array get_names() const;

	RegEx();
	RegEx(const String &p_pattern);
	~RegEx();
};

#endif // REGEX_H
