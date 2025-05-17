/**************************************************************************/
/*  script_backtrace.cpp                                                  */
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

#include "script_backtrace.h"

#include "core/object/script_language.h"

void ScriptBacktrace::_store_variables(const List<String> &p_names, const List<Variant> &p_values, LocalVector<StackVariable> &r_variables) {
	r_variables.reserve(p_names.size());

	const List<String>::Element *name = p_names.front();
	const List<Variant>::Element *value = p_values.front();

	for (int i = 0; i < p_names.size(); i++) {
		StackVariable variable;
		variable.name = name->get();
		variable.value = value->get();
		r_variables.push_back(std::move(variable));

		name = name->next();
		value = value->next();
	}
}

void ScriptBacktrace::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_language_name"), &ScriptBacktrace::get_language_name);

	ClassDB::bind_method(D_METHOD("is_empty"), &ScriptBacktrace::is_empty);
	ClassDB::bind_method(D_METHOD("get_frame_count"), &ScriptBacktrace::get_frame_count);
	ClassDB::bind_method(D_METHOD("get_frame_function", "index"), &ScriptBacktrace::get_frame_function);
	ClassDB::bind_method(D_METHOD("get_frame_file", "index"), &ScriptBacktrace::get_frame_file);
	ClassDB::bind_method(D_METHOD("get_frame_line", "index"), &ScriptBacktrace::get_frame_line);

	ClassDB::bind_method(D_METHOD("get_global_variable_count"), &ScriptBacktrace::get_global_variable_count);
	ClassDB::bind_method(D_METHOD("get_global_variable_name", "variable_index"), &ScriptBacktrace::get_global_variable_name);
	ClassDB::bind_method(D_METHOD("get_global_variable_value", "variable_index"), &ScriptBacktrace::get_global_variable_value);

	ClassDB::bind_method(D_METHOD("get_local_variable_count", "frame_index"), &ScriptBacktrace::get_local_variable_count);
	ClassDB::bind_method(D_METHOD("get_local_variable_name", "frame_index", "variable_index"), &ScriptBacktrace::get_local_variable_name);
	ClassDB::bind_method(D_METHOD("get_local_variable_value", "frame_index", "variable_index"), &ScriptBacktrace::get_local_variable_value);

	ClassDB::bind_method(D_METHOD("get_member_variable_count", "frame_index"), &ScriptBacktrace::get_member_variable_count);
	ClassDB::bind_method(D_METHOD("get_member_variable_name", "frame_index", "variable_index"), &ScriptBacktrace::get_member_variable_name);
	ClassDB::bind_method(D_METHOD("get_member_variable_value", "frame_index", "variable_index"), &ScriptBacktrace::get_member_variable_value);

	ClassDB::bind_method(D_METHOD("format", "indent_all", "indent_frames"), &ScriptBacktrace::format, DEFVAL(0), DEFVAL(4));
}

ScriptBacktrace::ScriptBacktrace(ScriptLanguage *p_language, bool p_include_variables) {
	language_name = p_language->get_name();

	Vector<ScriptLanguage::StackInfo> stack_infos = p_language->debug_get_current_stack_info();
	stack_frames.reserve(stack_infos.size());

	if (p_include_variables) {
		List<String> globals_names;
		List<Variant> globals_values;
		p_language->debug_get_globals(&globals_names, &globals_values);
		_store_variables(globals_names, globals_values, global_variables);
	}

	for (int i = 0; i < stack_infos.size(); i++) {
		const ScriptLanguage::StackInfo &stack_info = stack_infos[i];

		StackFrame stack_frame;
		stack_frame.function = stack_info.func;
		stack_frame.file = stack_info.file;
		stack_frame.line = stack_info.line;

		if (p_include_variables) {
			List<String> locals_names;
			List<Variant> locals_values;
			p_language->debug_get_stack_level_locals(i, &locals_names, &locals_values);
			_store_variables(locals_names, locals_values, stack_frame.local_variables);

			List<String> members_names;
			List<Variant> members_values;
			p_language->debug_get_stack_level_members(i, &members_names, &members_values);
			_store_variables(members_names, members_values, stack_frame.member_variables);
		}

		stack_frames.push_back(std::move(stack_frame));
	}
}

String ScriptBacktrace::get_frame_function(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)stack_frames.size(), String());
	return stack_frames[p_index].function;
}

String ScriptBacktrace::get_frame_file(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)stack_frames.size(), String());
	return stack_frames[p_index].file;
}

int ScriptBacktrace::get_frame_line(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)stack_frames.size(), -1);
	return stack_frames[p_index].line;
}

String ScriptBacktrace::get_global_variable_name(int p_variable_index) const {
	ERR_FAIL_INDEX_V(p_variable_index, (int)global_variables.size(), String());
	return global_variables[p_variable_index].name;
}

Variant ScriptBacktrace::get_global_variable_value(int p_variable_index) const {
	ERR_FAIL_INDEX_V(p_variable_index, (int)global_variables.size(), String());
	return global_variables[p_variable_index].value;
}

int ScriptBacktrace::get_local_variable_count(int p_frame_index) const {
	ERR_FAIL_INDEX_V(p_frame_index, (int)stack_frames.size(), 0);
	return (int)stack_frames[p_frame_index].local_variables.size();
}

String ScriptBacktrace::get_local_variable_name(int p_frame_index, int p_variable_index) const {
	ERR_FAIL_INDEX_V(p_frame_index, (int)stack_frames.size(), String());
	const LocalVector<StackVariable> &local_variables = stack_frames[p_frame_index].local_variables;
	ERR_FAIL_INDEX_V(p_variable_index, (int)local_variables.size(), String());
	return local_variables[p_variable_index].name;
}

Variant ScriptBacktrace::get_local_variable_value(int p_frame_index, int p_variable_index) const {
	ERR_FAIL_INDEX_V(p_frame_index, (int)stack_frames.size(), String());
	const LocalVector<StackVariable> &variables = stack_frames[p_frame_index].local_variables;
	ERR_FAIL_INDEX_V(p_variable_index, (int)variables.size(), String());
	return variables[p_variable_index].value;
}

int ScriptBacktrace::get_member_variable_count(int p_frame_index) const {
	ERR_FAIL_INDEX_V(p_frame_index, (int)stack_frames.size(), 0);
	return (int)stack_frames[p_frame_index].member_variables.size();
}

String ScriptBacktrace::get_member_variable_name(int p_frame_index, int p_variable_index) const {
	ERR_FAIL_INDEX_V(p_frame_index, (int)stack_frames.size(), String());
	const LocalVector<StackVariable> &variables = stack_frames[p_frame_index].member_variables;
	ERR_FAIL_INDEX_V(p_variable_index, (int)variables.size(), String());
	return variables[p_variable_index].name;
}

Variant ScriptBacktrace::get_member_variable_value(int p_frame_index, int p_variable_index) const {
	ERR_FAIL_INDEX_V(p_frame_index, (int)stack_frames.size(), String());
	const LocalVector<StackVariable> &variables = stack_frames[p_frame_index].member_variables;
	ERR_FAIL_INDEX_V(p_variable_index, (int)variables.size(), String());
	return variables[p_variable_index].value;
}

String ScriptBacktrace::format(int p_indent_all, int p_indent_frames) const {
	if (is_empty()) {
		return String();
	}

	static const String space = String::chr(U' ');
	String indent_all = space.repeat(p_indent_all);
	String indent_frames = space.repeat(p_indent_frames);
	String indent_total = indent_all + indent_frames;

	String result = indent_all + language_name + " backtrace (most recent call first):";
	for (int i = 0; i < (int)stack_frames.size(); i++) {
		const StackFrame &stack_frame = stack_frames[i];
		result += "\n" + indent_total + "[" + itos(i) + "] " + stack_frame.function;

		if (!stack_frame.file.is_empty()) {
			result += " (" + stack_frame.file + ":" + itos(stack_frame.line) + ")";
		}
	}

	return result;
}
