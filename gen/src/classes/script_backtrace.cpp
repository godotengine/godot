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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/script_backtrace.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

String ScriptBacktrace::get_language_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_language_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool ScriptBacktrace::is_empty() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("is_empty")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t ScriptBacktrace::get_frame_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_frame_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String ScriptBacktrace::get_frame_function(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_frame_function")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

String ScriptBacktrace::get_frame_file(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_frame_file")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

int32_t ScriptBacktrace::get_frame_line(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_frame_line")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

int32_t ScriptBacktrace::get_global_variable_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_global_variable_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String ScriptBacktrace::get_global_variable_name(int32_t p_variable_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_global_variable_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_variable_index_encoded;
	PtrToArg<int64_t>::encode(p_variable_index, &p_variable_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_variable_index_encoded);
}

Variant ScriptBacktrace::get_global_variable_value(int32_t p_variable_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_global_variable_value")._native_ptr(), 4227898402);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_variable_index_encoded;
	PtrToArg<int64_t>::encode(p_variable_index, &p_variable_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_variable_index_encoded);
}

int32_t ScriptBacktrace::get_local_variable_count(int32_t p_frame_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_local_variable_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_frame_index_encoded;
	PtrToArg<int64_t>::encode(p_frame_index, &p_frame_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_frame_index_encoded);
}

String ScriptBacktrace::get_local_variable_name(int32_t p_frame_index, int32_t p_variable_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_local_variable_name")._native_ptr(), 1391810591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_frame_index_encoded;
	PtrToArg<int64_t>::encode(p_frame_index, &p_frame_index_encoded);
	int64_t p_variable_index_encoded;
	PtrToArg<int64_t>::encode(p_variable_index, &p_variable_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_frame_index_encoded, &p_variable_index_encoded);
}

Variant ScriptBacktrace::get_local_variable_value(int32_t p_frame_index, int32_t p_variable_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_local_variable_value")._native_ptr(), 678354945);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_frame_index_encoded;
	PtrToArg<int64_t>::encode(p_frame_index, &p_frame_index_encoded);
	int64_t p_variable_index_encoded;
	PtrToArg<int64_t>::encode(p_variable_index, &p_variable_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_frame_index_encoded, &p_variable_index_encoded);
}

int32_t ScriptBacktrace::get_member_variable_count(int32_t p_frame_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_member_variable_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_frame_index_encoded;
	PtrToArg<int64_t>::encode(p_frame_index, &p_frame_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_frame_index_encoded);
}

String ScriptBacktrace::get_member_variable_name(int32_t p_frame_index, int32_t p_variable_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_member_variable_name")._native_ptr(), 1391810591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_frame_index_encoded;
	PtrToArg<int64_t>::encode(p_frame_index, &p_frame_index_encoded);
	int64_t p_variable_index_encoded;
	PtrToArg<int64_t>::encode(p_variable_index, &p_variable_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_frame_index_encoded, &p_variable_index_encoded);
}

Variant ScriptBacktrace::get_member_variable_value(int32_t p_frame_index, int32_t p_variable_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("get_member_variable_value")._native_ptr(), 678354945);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_frame_index_encoded;
	PtrToArg<int64_t>::encode(p_frame_index, &p_frame_index_encoded);
	int64_t p_variable_index_encoded;
	PtrToArg<int64_t>::encode(p_variable_index, &p_variable_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_frame_index_encoded, &p_variable_index_encoded);
}

String ScriptBacktrace::format(int32_t p_indent_all, int32_t p_indent_frames) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptBacktrace::get_class_static()._native_ptr(), StringName("format")._native_ptr(), 3464456933);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_indent_all_encoded;
	PtrToArg<int64_t>::encode(p_indent_all, &p_indent_all_encoded);
	int64_t p_indent_frames_encoded;
	PtrToArg<int64_t>::encode(p_indent_frames, &p_indent_frames_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_indent_all_encoded, &p_indent_frames_encoded);
}

} // namespace godot
