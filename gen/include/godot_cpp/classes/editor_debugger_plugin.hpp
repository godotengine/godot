/**************************************************************************/
/*  editor_debugger_plugin.hpp                                            */
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
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class EditorDebuggerSession;
class Script;
class String;

class EditorDebuggerPlugin : public RefCounted {
	GDEXTENSION_CLASS(EditorDebuggerPlugin, RefCounted)

public:
	Ref<EditorDebuggerSession> get_session(int32_t p_id);
	Array get_sessions();
	virtual void _setup_session(int32_t p_session_id);
	virtual bool _has_capture(const String &p_capture) const;
	virtual bool _capture(const String &p_message, const Array &p_data, int32_t p_session_id);
	virtual void _goto_script_line(const Ref<Script> &p_script, int32_t p_line);
	virtual void _breakpoints_cleared_in_tree();
	virtual void _breakpoint_set_in_tree(const Ref<Script> &p_script, int32_t p_line, bool p_enabled);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_setup_session), decltype(&T::_setup_session)>) {
			BIND_VIRTUAL_METHOD(T, _setup_session, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_has_capture), decltype(&T::_has_capture)>) {
			BIND_VIRTUAL_METHOD(T, _has_capture, 3927539163);
		}
		if constexpr (!std::is_same_v<decltype(&B::_capture), decltype(&T::_capture)>) {
			BIND_VIRTUAL_METHOD(T, _capture, 2607901833);
		}
		if constexpr (!std::is_same_v<decltype(&B::_goto_script_line), decltype(&T::_goto_script_line)>) {
			BIND_VIRTUAL_METHOD(T, _goto_script_line, 1208513123);
		}
		if constexpr (!std::is_same_v<decltype(&B::_breakpoints_cleared_in_tree), decltype(&T::_breakpoints_cleared_in_tree)>) {
			BIND_VIRTUAL_METHOD(T, _breakpoints_cleared_in_tree, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_breakpoint_set_in_tree), decltype(&T::_breakpoint_set_in_tree)>) {
			BIND_VIRTUAL_METHOD(T, _breakpoint_set_in_tree, 2338735218);
		}
	}

public:
};

} // namespace godot

