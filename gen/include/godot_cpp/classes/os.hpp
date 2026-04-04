/**************************************************************************/
/*  os.hpp                                                                */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Logger;

class OS : public Object {
	GDEXTENSION_CLASS(OS, Object)

	static OS *singleton;

public:
	enum RenderingDriver {
		RENDERING_DRIVER_VULKAN = 0,
		RENDERING_DRIVER_OPENGL3 = 1,
		RENDERING_DRIVER_D3D12 = 2,
		RENDERING_DRIVER_METAL = 3,
	};

	enum SystemDir {
		SYSTEM_DIR_DESKTOP = 0,
		SYSTEM_DIR_DCIM = 1,
		SYSTEM_DIR_DOCUMENTS = 2,
		SYSTEM_DIR_DOWNLOADS = 3,
		SYSTEM_DIR_MOVIES = 4,
		SYSTEM_DIR_MUSIC = 5,
		SYSTEM_DIR_PICTURES = 6,
		SYSTEM_DIR_RINGTONES = 7,
	};

	enum StdHandleType {
		STD_HANDLE_INVALID = 0,
		STD_HANDLE_CONSOLE = 1,
		STD_HANDLE_FILE = 2,
		STD_HANDLE_PIPE = 3,
		STD_HANDLE_UNKNOWN = 4,
	};

	static OS *get_singleton();

	PackedByteArray get_entropy(int32_t p_size);
	String get_system_ca_certificates();
	PackedStringArray get_connected_midi_inputs();
	void open_midi_inputs();
	void close_midi_inputs();
	void alert(const String &p_text, const String &p_title = "Alert!");
	void crash(const String &p_message);
	void set_low_processor_usage_mode(bool p_enable);
	bool is_in_low_processor_usage_mode() const;
	void set_low_processor_usage_mode_sleep_usec(int32_t p_usec);
	int32_t get_low_processor_usage_mode_sleep_usec() const;
	void set_delta_smoothing(bool p_delta_smoothing_enabled);
	bool is_delta_smoothing_enabled() const;
	int32_t get_processor_count() const;
	String get_processor_name() const;
	PackedStringArray get_system_fonts() const;
	String get_system_font_path(const String &p_font_name, int32_t p_weight = 400, int32_t p_stretch = 100, bool p_italic = false) const;
	PackedStringArray get_system_font_path_for_text(const String &p_font_name, const String &p_text, const String &p_locale = String(), const String &p_script = String(), int32_t p_weight = 400, int32_t p_stretch = 100, bool p_italic = false) const;
	String get_executable_path() const;
	String read_string_from_stdin(int64_t p_buffer_size = 1024);
	PackedByteArray read_buffer_from_stdin(int64_t p_buffer_size = 1024);
	OS::StdHandleType get_stdin_type() const;
	OS::StdHandleType get_stdout_type() const;
	OS::StdHandleType get_stderr_type() const;
	int32_t execute(const String &p_path, const PackedStringArray &p_arguments, const Array &p_output = Array(), bool p_read_stderr = false, bool p_open_console = false);
	Dictionary execute_with_pipe(const String &p_path, const PackedStringArray &p_arguments, bool p_blocking = true);
	int32_t create_process(const String &p_path, const PackedStringArray &p_arguments, bool p_open_console = false);
	int32_t create_instance(const PackedStringArray &p_arguments);
	Error open_with_program(const String &p_program_path, const PackedStringArray &p_paths);
	Error kill(int32_t p_pid);
	Error shell_open(const String &p_uri);
	Error shell_show_in_file_manager(const String &p_file_or_dir_path, bool p_open_folder = true);
	bool is_process_running(int32_t p_pid) const;
	int32_t get_process_exit_code(int32_t p_pid) const;
	int32_t get_process_id() const;
	bool has_environment(const String &p_variable) const;
	String get_environment(const String &p_variable) const;
	void set_environment(const String &p_variable, const String &p_value) const;
	void unset_environment(const String &p_variable) const;
	String get_name() const;
	String get_distribution_name() const;
	String get_version() const;
	String get_version_alias() const;
	PackedStringArray get_cmdline_args();
	PackedStringArray get_cmdline_user_args();
	PackedStringArray get_video_adapter_driver_info() const;
	void set_restart_on_exit(bool p_restart, const PackedStringArray &p_arguments = PackedStringArray());
	bool is_restart_on_exit_set() const;
	PackedStringArray get_restart_on_exit_arguments() const;
	void delay_usec(int32_t p_usec) const;
	void delay_msec(int32_t p_msec) const;
	String get_locale() const;
	String get_locale_language() const;
	String get_model_name() const;
	bool is_userfs_persistent() const;
	bool is_stdout_verbose() const;
	bool is_debug_build() const;
	uint64_t get_static_memory_usage() const;
	uint64_t get_static_memory_peak_usage() const;
	Dictionary get_memory_info() const;
	Error move_to_trash(const String &p_path) const;
	String get_user_data_dir() const;
	String get_system_dir(OS::SystemDir p_dir, bool p_shared_storage = true) const;
	String get_config_dir() const;
	String get_data_dir() const;
	String get_cache_dir() const;
	String get_temp_dir() const;
	String get_unique_id() const;
	String get_keycode_string(Key p_code) const;
	bool is_keycode_unicode(char32_t p_code) const;
	Key find_keycode_from_string(const String &p_string) const;
	void set_use_file_access_save_and_swap(bool p_enabled);
	Error set_thread_name(const String &p_name);
	uint64_t get_thread_caller_id() const;
	uint64_t get_main_thread_id() const;
	bool has_feature(const String &p_tag_name) const;
	bool is_sandboxed() const;
	bool request_permission(const String &p_name);
	bool request_permissions();
	PackedStringArray get_granted_permissions() const;
	void revoke_granted_permissions();
	void add_logger(const Ref<Logger> &p_logger);
	void remove_logger(const Ref<Logger> &p_logger);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~OS();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(OS::RenderingDriver);
VARIANT_ENUM_CAST(OS::SystemDir);
VARIANT_ENUM_CAST(OS::StdHandleType);

