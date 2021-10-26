/*************************************************************************/
/*  script_debugger.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SCRIPT_DEBUGGER_H
#define SCRIPT_DEBUGGER_H

#include "core/object/script_language.h"
#include "core/string/string_name.h"
#include "core/templates/map.h"
#include "core/templates/set.h"
#include "core/templates/vector.h"

class ScriptDebugger {
	typedef ScriptLanguage::StackInfo StackInfo;

	int lines_left = -1;
	int depth = -1;
	bool skip_breakpoints = false;

	Map<int, Set<StringName>> breakpoints;

	ScriptLanguage *break_lang = nullptr;
	Vector<StackInfo> error_stack_info;

public:
	void set_lines_left(int p_left);
	int get_lines_left() const;

	void set_depth(int p_depth);
	int get_depth() const;

	String breakpoint_find_source(const String &p_source) const;
	void set_break_language(ScriptLanguage *p_lang) { break_lang = p_lang; }
	ScriptLanguage *get_break_language() { return break_lang; }
	void set_skip_breakpoints(bool p_skip_breakpoints);
	bool is_skipping_breakpoints();
	void insert_breakpoint(int p_line, const StringName &p_source);
	void remove_breakpoint(int p_line, const StringName &p_source);
	bool is_breakpoint(int p_line, const StringName &p_source) const;
	bool is_breakpoint_line(int p_line) const;
	void clear_breakpoints();
	const Map<int, Set<StringName>> &get_breakpoints() const { return breakpoints; }

	void debug(ScriptLanguage *p_lang, bool p_can_continue = true, bool p_is_error_breakpoint = false);
	ScriptLanguage *get_break_language() const;

	void send_error(const String &p_func, const String &p_file, int p_line, const String &p_err, const String &p_descr, bool p_editor_notify, ErrorHandlerType p_type, const Vector<StackInfo> &p_stack_info);
	Vector<StackInfo> get_error_stack_info() const;
	ScriptDebugger() {}
};

#endif // SCRIPT_DEBUGGER_H
