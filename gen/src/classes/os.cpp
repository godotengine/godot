/**************************************************************************/
/*  os.cpp                                                                */
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

#include <godot_cpp/classes/os.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/logger.hpp>

namespace godot {

OS *OS::singleton = nullptr;

OS *OS::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(OS::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<OS *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &OS::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(OS::get_class_static(), singleton);
		}
	}
	return singleton;
}

OS::~OS() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(OS::get_class_static());
		singleton = nullptr;
	}
}

PackedByteArray OS::get_entropy(int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_entropy")._native_ptr(), 47165747);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, &p_size_encoded);
}

String OS::get_system_ca_certificates() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_system_ca_certificates")._native_ptr(), 2841200299);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

PackedStringArray OS::get_connected_midi_inputs() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_connected_midi_inputs")._native_ptr(), 2981934095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void OS::open_midi_inputs() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("open_midi_inputs")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void OS::close_midi_inputs() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("close_midi_inputs")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void OS::alert(const String &p_text, const String &p_title) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("alert")._native_ptr(), 1783970740);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text, &p_title);
}

void OS::crash(const String &p_message) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("crash")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_message);
}

void OS::set_low_processor_usage_mode(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("set_low_processor_usage_mode")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool OS::is_in_low_processor_usage_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("is_in_low_processor_usage_mode")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void OS::set_low_processor_usage_mode_sleep_usec(int32_t p_usec) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("set_low_processor_usage_mode_sleep_usec")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_usec_encoded;
	PtrToArg<int64_t>::encode(p_usec, &p_usec_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_usec_encoded);
}

int32_t OS::get_low_processor_usage_mode_sleep_usec() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_low_processor_usage_mode_sleep_usec")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OS::set_delta_smoothing(bool p_delta_smoothing_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("set_delta_smoothing")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_delta_smoothing_enabled_encoded;
	PtrToArg<bool>::encode(p_delta_smoothing_enabled, &p_delta_smoothing_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_delta_smoothing_enabled_encoded);
}

bool OS::is_delta_smoothing_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("is_delta_smoothing_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t OS::get_processor_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_processor_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String OS::get_processor_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_processor_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

PackedStringArray OS::get_system_fonts() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_system_fonts")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

String OS::get_system_font_path(const String &p_font_name, int32_t p_weight, int32_t p_stretch, bool p_italic) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_system_font_path")._native_ptr(), 626580860);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_weight_encoded;
	PtrToArg<int64_t>::encode(p_weight, &p_weight_encoded);
	int64_t p_stretch_encoded;
	PtrToArg<int64_t>::encode(p_stretch, &p_stretch_encoded);
	int8_t p_italic_encoded;
	PtrToArg<bool>::encode(p_italic, &p_italic_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_font_name, &p_weight_encoded, &p_stretch_encoded, &p_italic_encoded);
}

PackedStringArray OS::get_system_font_path_for_text(const String &p_font_name, const String &p_text, const String &p_locale, const String &p_script, int32_t p_weight, int32_t p_stretch, bool p_italic) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_system_font_path_for_text")._native_ptr(), 197317981);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	int64_t p_weight_encoded;
	PtrToArg<int64_t>::encode(p_weight, &p_weight_encoded);
	int64_t p_stretch_encoded;
	PtrToArg<int64_t>::encode(p_stretch, &p_stretch_encoded);
	int8_t p_italic_encoded;
	PtrToArg<bool>::encode(p_italic, &p_italic_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_font_name, &p_text, &p_locale, &p_script, &p_weight_encoded, &p_stretch_encoded, &p_italic_encoded);
}

String OS::get_executable_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_executable_path")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::read_string_from_stdin(int64_t p_buffer_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("read_string_from_stdin")._native_ptr(), 723587915);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_buffer_size_encoded;
	PtrToArg<int64_t>::encode(p_buffer_size, &p_buffer_size_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_buffer_size_encoded);
}

PackedByteArray OS::read_buffer_from_stdin(int64_t p_buffer_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("read_buffer_from_stdin")._native_ptr(), 3249455752);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	int64_t p_buffer_size_encoded;
	PtrToArg<int64_t>::encode(p_buffer_size, &p_buffer_size_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, &p_buffer_size_encoded);
}

OS::StdHandleType OS::get_stdin_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_stdin_type")._native_ptr(), 1704816237);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OS::StdHandleType(0)));
	return (OS::StdHandleType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

OS::StdHandleType OS::get_stdout_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_stdout_type")._native_ptr(), 1704816237);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OS::StdHandleType(0)));
	return (OS::StdHandleType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

OS::StdHandleType OS::get_stderr_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_stderr_type")._native_ptr(), 1704816237);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OS::StdHandleType(0)));
	return (OS::StdHandleType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t OS::execute(const String &p_path, const PackedStringArray &p_arguments, const Array &p_output, bool p_read_stderr, bool p_open_console) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("execute")._native_ptr(), 1488299882);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int8_t p_read_stderr_encoded;
	PtrToArg<bool>::encode(p_read_stderr, &p_read_stderr_encoded);
	int8_t p_open_console_encoded;
	PtrToArg<bool>::encode(p_open_console, &p_open_console_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_arguments, &p_output, &p_read_stderr_encoded, &p_open_console_encoded);
}

Dictionary OS::execute_with_pipe(const String &p_path, const PackedStringArray &p_arguments, bool p_blocking) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("execute_with_pipe")._native_ptr(), 2851312030);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int8_t p_blocking_encoded;
	PtrToArg<bool>::encode(p_blocking, &p_blocking_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_path, &p_arguments, &p_blocking_encoded);
}

int32_t OS::create_process(const String &p_path, const PackedStringArray &p_arguments, bool p_open_console) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("create_process")._native_ptr(), 2903767230);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int8_t p_open_console_encoded;
	PtrToArg<bool>::encode(p_open_console, &p_open_console_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_arguments, &p_open_console_encoded);
}

int32_t OS::create_instance(const PackedStringArray &p_arguments) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("create_instance")._native_ptr(), 1080601263);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_arguments);
}

Error OS::open_with_program(const String &p_program_path, const PackedStringArray &p_paths) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("open_with_program")._native_ptr(), 2848259907);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_program_path, &p_paths);
}

Error OS::kill(int32_t p_pid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("kill")._native_ptr(), 844576869);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_pid_encoded;
	PtrToArg<int64_t>::encode(p_pid, &p_pid_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_pid_encoded);
}

Error OS::shell_open(const String &p_uri) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("shell_open")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_uri);
}

Error OS::shell_show_in_file_manager(const String &p_file_or_dir_path, bool p_open_folder) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("shell_show_in_file_manager")._native_ptr(), 3565188097);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_open_folder_encoded;
	PtrToArg<bool>::encode(p_open_folder, &p_open_folder_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_file_or_dir_path, &p_open_folder_encoded);
}

bool OS::is_process_running(int32_t p_pid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("is_process_running")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_pid_encoded;
	PtrToArg<int64_t>::encode(p_pid, &p_pid_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_pid_encoded);
}

int32_t OS::get_process_exit_code(int32_t p_pid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_process_exit_code")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_pid_encoded;
	PtrToArg<int64_t>::encode(p_pid, &p_pid_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_pid_encoded);
}

int32_t OS::get_process_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_process_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool OS::has_environment(const String &p_variable) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("has_environment")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_variable);
}

String OS::get_environment(const String &p_variable) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_environment")._native_ptr(), 3135753539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_variable);
}

void OS::set_environment(const String &p_variable, const String &p_value) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("set_environment")._native_ptr(), 3605043004);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_variable, &p_value);
}

void OS::unset_environment(const String &p_variable) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("unset_environment")._native_ptr(), 3089850668);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_variable);
}

String OS::get_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::get_distribution_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_distribution_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::get_version() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_version")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::get_version_alias() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_version_alias")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

PackedStringArray OS::get_cmdline_args() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_cmdline_args")._native_ptr(), 2981934095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

PackedStringArray OS::get_cmdline_user_args() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_cmdline_user_args")._native_ptr(), 2981934095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

PackedStringArray OS::get_video_adapter_driver_info() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_video_adapter_driver_info")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void OS::set_restart_on_exit(bool p_restart, const PackedStringArray &p_arguments) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("set_restart_on_exit")._native_ptr(), 3331453935);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_restart_encoded;
	PtrToArg<bool>::encode(p_restart, &p_restart_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_restart_encoded, &p_arguments);
}

bool OS::is_restart_on_exit_set() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("is_restart_on_exit_set")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

PackedStringArray OS::get_restart_on_exit_arguments() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_restart_on_exit_arguments")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void OS::delay_usec(int32_t p_usec) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("delay_usec")._native_ptr(), 998575451);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_usec_encoded;
	PtrToArg<int64_t>::encode(p_usec, &p_usec_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_usec_encoded);
}

void OS::delay_msec(int32_t p_msec) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("delay_msec")._native_ptr(), 998575451);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msec_encoded;
	PtrToArg<int64_t>::encode(p_msec, &p_msec_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msec_encoded);
}

String OS::get_locale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_locale")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::get_locale_language() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_locale_language")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::get_model_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_model_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool OS::is_userfs_persistent() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("is_userfs_persistent")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool OS::is_stdout_verbose() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("is_stdout_verbose")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool OS::is_debug_build() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("is_debug_build")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

uint64_t OS::get_static_memory_usage() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_static_memory_usage")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t OS::get_static_memory_peak_usage() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_static_memory_peak_usage")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

Dictionary OS::get_memory_info() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_memory_info")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

Error OS::move_to_trash(const String &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("move_to_trash")._native_ptr(), 2113323047);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

String OS::get_user_data_dir() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_user_data_dir")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::get_system_dir(OS::SystemDir p_dir, bool p_shared_storage) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_system_dir")._native_ptr(), 3073895123);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_dir_encoded;
	PtrToArg<int64_t>::encode(p_dir, &p_dir_encoded);
	int8_t p_shared_storage_encoded;
	PtrToArg<bool>::encode(p_shared_storage, &p_shared_storage_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_dir_encoded, &p_shared_storage_encoded);
}

String OS::get_config_dir() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_config_dir")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::get_data_dir() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_data_dir")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::get_cache_dir() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_cache_dir")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::get_temp_dir() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_temp_dir")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::get_unique_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_unique_id")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String OS::get_keycode_string(Key p_code) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_keycode_string")._native_ptr(), 2261993717);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_code_encoded;
	PtrToArg<int64_t>::encode(p_code, &p_code_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_code_encoded);
}

bool OS::is_keycode_unicode(char32_t p_code) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("is_keycode_unicode")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_code_encoded;
	PtrToArg<int64_t>::encode(p_code, &p_code_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_code_encoded);
}

Key OS::find_keycode_from_string(const String &p_string) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("find_keycode_from_string")._native_ptr(), 1084858572);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_string);
}

void OS::set_use_file_access_save_and_swap(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("set_use_file_access_save_and_swap")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

Error OS::set_thread_name(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("set_thread_name")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_name);
}

uint64_t OS::get_thread_caller_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_thread_caller_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t OS::get_main_thread_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_main_thread_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

bool OS::has_feature(const String &p_tag_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("has_feature")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_tag_name);
}

bool OS::is_sandboxed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("is_sandboxed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool OS::request_permission(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("request_permission")._native_ptr(), 2323990056);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool OS::request_permissions() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("request_permissions")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

PackedStringArray OS::get_granted_permissions() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("get_granted_permissions")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void OS::revoke_granted_permissions() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("revoke_granted_permissions")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void OS::add_logger(const Ref<Logger> &p_logger) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("add_logger")._native_ptr(), 4261188958);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_logger != nullptr ? &p_logger->_owner : nullptr));
}

void OS::remove_logger(const Ref<Logger> &p_logger) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OS::get_class_static()._native_ptr(), StringName("remove_logger")._native_ptr(), 4261188958);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_logger != nullptr ? &p_logger->_owner : nullptr));
}

} // namespace godot
