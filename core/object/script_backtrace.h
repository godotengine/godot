/**************************************************************************/
/*  script_backtrace.h                                                    */
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

#include "core/object/ref_counted.h"

class ScriptLanguage;

class ScriptBacktrace : public RefCounted {
	GDCLASS(ScriptBacktrace, RefCounted);

	struct StackVariable {
		String name;
		Variant value;
	};

	struct StackFrame {
		LocalVector<StackVariable> local_variables;
		LocalVector<StackVariable> member_variables;
		String function;
		String file;
		int line = 0;
	};

	LocalVector<StackFrame> stack_frames;
	LocalVector<StackVariable> global_variables;
	String language_name;

	static void _store_variables(const LocalVector<Pair<String, Variant>> &p_variables, LocalVector<StackVariable> &r_variables);

protected:
	static void _bind_methods();

public:
	ScriptBacktrace() = default;
	ScriptBacktrace(ScriptLanguage *p_language, bool p_include_variables = false);

	String get_language_name() const { return language_name; }

	bool is_empty() const { return stack_frames.is_empty(); }
	int get_frame_count() const { return stack_frames.size(); }
	String get_frame_function(int p_index) const;
	String get_frame_file(int p_index) const;
	int get_frame_line(int p_index) const;

	int get_global_variable_count() const { return global_variables.size(); }
	String get_global_variable_name(int p_variable_index) const;
	Variant get_global_variable_value(int p_variable_index) const;

	int get_local_variable_count(int p_frame_index) const;
	String get_local_variable_name(int p_frame_index, int p_variable_index) const;
	Variant get_local_variable_value(int p_frame_index, int p_variable_index) const;

	int get_member_variable_count(int p_frame_index) const;
	String get_member_variable_name(int p_frame_index, int p_variable_index) const;
	Variant get_member_variable_value(int p_frame_index, int p_variable_index) const;

	String format(int p_indent_all = 0, int p_indent_frames = 4) const;
	virtual String _to_string() override { return format(); }
};
