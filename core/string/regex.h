/**************************************************************************/
/*  regex.h                                                               */
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

#pragma once

#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"

class RegEx;

class RegExMatch {
	friend class ::RegEx;

	struct Range {
		int start = 0;
		int end = 0;
	};

	String _subject;
	Vector<Range> _data;
	HashMap<String, int> _names;

	int _find(const String &p_name) const;
	int _find(int p_index) const;

	String _get_string(int p_found) const;
	int _get_start(int p_found) const;
	int _get_end(int p_found) const;

public:
	_FORCE_INLINE_ bool is_valid() const { return !_data.is_empty(); }
	_FORCE_INLINE_ const String &get_subject() const _LIFETIME_BOUND_ { return _subject; }
	_FORCE_INLINE_ int get_group_count() const { return is_valid() ? _data.size() - 1 : 0; }
	_FORCE_INLINE_ const HashMap<String, int> &get_names() const _LIFETIME_BOUND_ { return _names; }

	Vector<String> get_strings() const;
	_FORCE_INLINE_ String get_string(const String &p_name) const { return _get_string(_find(p_name)); }
	_FORCE_INLINE_ String get_string(int p_index) const { return _get_string(_find(p_index)); }
	_FORCE_INLINE_ int get_start(const String &p_name) const { return _get_start(_find(p_name)); }
	_FORCE_INLINE_ int get_start(int p_index) const { return _get_start(_find(p_index)); }
	_FORCE_INLINE_ int get_end(const String &p_name) const { return _get_end(_find(p_name)); }
	_FORCE_INLINE_ int get_end(int p_index) const { return _get_end(_find(p_index)); }
};

class RegEx {
	void *_general_context = nullptr;
	void *_code = nullptr;
	String _pattern;

	void _pattern_info(uint32_t p_what, void *r_where) const;
	int _sub(const String &p_subject, const String &p_replacement, int p_offset, int p_end, uint32_t p_flags, String &r_output) const;

public:
	void clear();
	Error compile(const String &p_pattern, bool p_show_error = true);

	RegExMatch search(const String &p_subject, int p_offset = 0, int p_end = -1) const;
	Vector<RegExMatch> search_all(const String &p_subject, int p_offset = 0, int p_end = -1) const;
	String sub(const String &p_subject, const String &p_replacement, bool p_all = false, int p_offset = 0, int p_end = -1) const;

	_FORCE_INLINE_ bool is_valid() const { return _code != nullptr; }
	_FORCE_INLINE_ String get_pattern() const { return _pattern; }
	int get_group_count() const;
	Vector<String> get_names() const;

	RegEx();
	RegEx(const String &p_pattern, bool p_show_error = true);
	~RegEx();
};
