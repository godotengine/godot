/**************************************************************************/
/*  editor_debugger_plugin.cpp                                            */
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

#include <godot_cpp/classes/editor_debugger_plugin.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/editor_debugger_session.hpp>
#include <godot_cpp/classes/script.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

Ref<EditorDebuggerSession> EditorDebuggerPlugin::get_session(int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDebuggerPlugin::get_class_static()._native_ptr(), StringName("get_session")._native_ptr(), 3061968499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<EditorDebuggerSession>()));
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	return Ref<EditorDebuggerSession>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<EditorDebuggerSession>(_gde_method_bind, _owner, &p_id_encoded));
}

Array EditorDebuggerPlugin::get_sessions() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDebuggerPlugin::get_class_static()._native_ptr(), StringName("get_sessions")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

void EditorDebuggerPlugin::_setup_session(int32_t p_session_id) {}

bool EditorDebuggerPlugin::_has_capture(const String &p_capture) const {
	return false;
}

bool EditorDebuggerPlugin::_capture(const String &p_message, const Array &p_data, int32_t p_session_id) {
	return false;
}

void EditorDebuggerPlugin::_goto_script_line(const Ref<Script> &p_script, int32_t p_line) {}

void EditorDebuggerPlugin::_breakpoints_cleared_in_tree() {}

void EditorDebuggerPlugin::_breakpoint_set_in_tree(const Ref<Script> &p_script, int32_t p_line, bool p_enabled) {}

} // namespace godot
