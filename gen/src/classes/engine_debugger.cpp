/**************************************************************************/
/*  engine_debugger.cpp                                                   */
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

#include <godot_cpp/classes/engine_debugger.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/engine_profiler.hpp>
#include <godot_cpp/classes/script_language.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

EngineDebugger *EngineDebugger::singleton = nullptr;

EngineDebugger *EngineDebugger::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(EngineDebugger::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<EngineDebugger *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &EngineDebugger::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(EngineDebugger::get_class_static(), singleton);
		}
	}
	return singleton;
}

EngineDebugger::~EngineDebugger() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(EngineDebugger::get_class_static());
		singleton = nullptr;
	}
}

bool EngineDebugger::is_active() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("is_active")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EngineDebugger::register_profiler(const StringName &p_name, const Ref<EngineProfiler> &p_profiler) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("register_profiler")._native_ptr(), 3651669560);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_profiler != nullptr ? &p_profiler->_owner : nullptr));
}

void EngineDebugger::unregister_profiler(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("unregister_profiler")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

bool EngineDebugger::is_profiling(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("is_profiling")._native_ptr(), 2041966384);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool EngineDebugger::has_profiler(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("has_profiler")._native_ptr(), 2041966384);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

void EngineDebugger::profiler_add_frame_data(const StringName &p_name, const Array &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("profiler_add_frame_data")._native_ptr(), 1895267858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_data);
}

void EngineDebugger::profiler_enable(const StringName &p_name, bool p_enable, const Array &p_arguments) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("profiler_enable")._native_ptr(), 3192561009);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_enable_encoded, &p_arguments);
}

void EngineDebugger::register_message_capture(const StringName &p_name, const Callable &p_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("register_message_capture")._native_ptr(), 1874754934);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_callable);
}

void EngineDebugger::unregister_message_capture(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("unregister_message_capture")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

bool EngineDebugger::has_capture(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("has_capture")._native_ptr(), 2041966384);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

void EngineDebugger::line_poll() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("line_poll")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void EngineDebugger::send_message(const String &p_message, const Array &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("send_message")._native_ptr(), 1209351045);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_message, &p_data);
}

void EngineDebugger::debug(bool p_can_continue, bool p_is_error_breakpoint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("debug")._native_ptr(), 2751962654);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_can_continue_encoded;
	PtrToArg<bool>::encode(p_can_continue, &p_can_continue_encoded);
	int8_t p_is_error_breakpoint_encoded;
	PtrToArg<bool>::encode(p_is_error_breakpoint, &p_is_error_breakpoint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_can_continue_encoded, &p_is_error_breakpoint_encoded);
}

void EngineDebugger::script_debug(ScriptLanguage *p_language, bool p_can_continue, bool p_is_error_breakpoint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("script_debug")._native_ptr(), 2442343672);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_can_continue_encoded;
	PtrToArg<bool>::encode(p_can_continue, &p_can_continue_encoded);
	int8_t p_is_error_breakpoint_encoded;
	PtrToArg<bool>::encode(p_is_error_breakpoint, &p_is_error_breakpoint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_language != nullptr ? &p_language->_owner : nullptr), &p_can_continue_encoded, &p_is_error_breakpoint_encoded);
}

void EngineDebugger::set_lines_left(int32_t p_lines) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("set_lines_left")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_lines_encoded;
	PtrToArg<int64_t>::encode(p_lines, &p_lines_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_lines_encoded);
}

int32_t EngineDebugger::get_lines_left() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("get_lines_left")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void EngineDebugger::set_depth(int32_t p_depth) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("set_depth")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_depth_encoded;
	PtrToArg<int64_t>::encode(p_depth, &p_depth_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_depth_encoded);
}

int32_t EngineDebugger::get_depth() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("get_depth")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool EngineDebugger::is_breakpoint(int32_t p_line, const StringName &p_source) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("is_breakpoint")._native_ptr(), 921227809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_line_encoded, &p_source);
}

bool EngineDebugger::is_skipping_breakpoints() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("is_skipping_breakpoints")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EngineDebugger::insert_breakpoint(int32_t p_line, const StringName &p_source) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("insert_breakpoint")._native_ptr(), 3780747571);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_source);
}

void EngineDebugger::remove_breakpoint(int32_t p_line, const StringName &p_source) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("remove_breakpoint")._native_ptr(), 3780747571);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_source);
}

void EngineDebugger::clear_breakpoints() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EngineDebugger::get_class_static()._native_ptr(), StringName("clear_breakpoints")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
