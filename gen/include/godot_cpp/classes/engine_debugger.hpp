/**************************************************************************/
/*  engine_debugger.hpp                                                   */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class EngineProfiler;
class ScriptLanguage;
class String;
class StringName;

class EngineDebugger : public Object {
	GDEXTENSION_CLASS(EngineDebugger, Object)

	static EngineDebugger *singleton;

public:
	static EngineDebugger *get_singleton();

	bool is_active();
	void register_profiler(const StringName &p_name, const Ref<EngineProfiler> &p_profiler);
	void unregister_profiler(const StringName &p_name);
	bool is_profiling(const StringName &p_name);
	bool has_profiler(const StringName &p_name);
	void profiler_add_frame_data(const StringName &p_name, const Array &p_data);
	void profiler_enable(const StringName &p_name, bool p_enable, const Array &p_arguments = Array());
	void register_message_capture(const StringName &p_name, const Callable &p_callable);
	void unregister_message_capture(const StringName &p_name);
	bool has_capture(const StringName &p_name);
	void line_poll();
	void send_message(const String &p_message, const Array &p_data);
	void debug(bool p_can_continue = true, bool p_is_error_breakpoint = false);
	void script_debug(ScriptLanguage *p_language, bool p_can_continue = true, bool p_is_error_breakpoint = false);
	void set_lines_left(int32_t p_lines);
	int32_t get_lines_left() const;
	void set_depth(int32_t p_depth);
	int32_t get_depth() const;
	bool is_breakpoint(int32_t p_line, const StringName &p_source) const;
	bool is_skipping_breakpoints() const;
	void insert_breakpoint(int32_t p_line, const StringName &p_source);
	void remove_breakpoint(int32_t p_line, const StringName &p_source);
	void clear_breakpoints();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~EngineDebugger();

public:
};

} // namespace godot

