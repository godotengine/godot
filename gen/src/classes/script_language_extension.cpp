/**************************************************************************/
/*  script_language_extension.cpp                                         */
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

#include <godot_cpp/classes/script_language_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/script.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace godot {

String ScriptLanguageExtension::_get_name() const {
	return String();
}

void ScriptLanguageExtension::_init() {}

String ScriptLanguageExtension::_get_type() const {
	return String();
}

String ScriptLanguageExtension::_get_extension() const {
	return String();
}

void ScriptLanguageExtension::_finish() {}

PackedStringArray ScriptLanguageExtension::_get_reserved_words() const {
	return PackedStringArray();
}

bool ScriptLanguageExtension::_is_control_flow_keyword(const String &p_keyword) const {
	return false;
}

PackedStringArray ScriptLanguageExtension::_get_comment_delimiters() const {
	return PackedStringArray();
}

PackedStringArray ScriptLanguageExtension::_get_doc_comment_delimiters() const {
	return PackedStringArray();
}

PackedStringArray ScriptLanguageExtension::_get_string_delimiters() const {
	return PackedStringArray();
}

Ref<Script> ScriptLanguageExtension::_make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const {
	return Ref<Script>();
}

TypedArray<Dictionary> ScriptLanguageExtension::_get_built_in_templates(const StringName &p_object) const {
	return TypedArray<Dictionary>();
}

bool ScriptLanguageExtension::_is_using_templates() {
	return false;
}

Dictionary ScriptLanguageExtension::_validate(const String &p_script, const String &p_path, bool p_validate_functions, bool p_validate_errors, bool p_validate_warnings, bool p_validate_safe_lines) const {
	return Dictionary();
}

String ScriptLanguageExtension::_validate_path(const String &p_path) const {
	return String();
}

Object *ScriptLanguageExtension::_create_script() const {
	return nullptr;
}

bool ScriptLanguageExtension::_has_named_classes() const {
	return false;
}

bool ScriptLanguageExtension::_supports_builtin_mode() const {
	return false;
}

bool ScriptLanguageExtension::_supports_documentation() const {
	return false;
}

bool ScriptLanguageExtension::_can_inherit_from_file() const {
	return false;
}

int32_t ScriptLanguageExtension::_find_function(const String &p_function, const String &p_code) const {
	return 0;
}

String ScriptLanguageExtension::_make_function(const String &p_class_name, const String &p_function_name, const PackedStringArray &p_function_args) const {
	return String();
}

bool ScriptLanguageExtension::_can_make_function() const {
	return false;
}

Error ScriptLanguageExtension::_open_in_external_editor(const Ref<Script> &p_script, int32_t p_line, int32_t p_column) {
	return Error(0);
}

bool ScriptLanguageExtension::_overrides_external_editor() {
	return false;
}

ScriptLanguage::ScriptNameCasing ScriptLanguageExtension::_preferred_file_name_casing() const {
	return ScriptLanguage::ScriptNameCasing(0);
}

Dictionary ScriptLanguageExtension::_complete_code(const String &p_code, const String &p_path, Object *p_owner) const {
	return Dictionary();
}

Dictionary ScriptLanguageExtension::_lookup_code(const String &p_code, const String &p_symbol, const String &p_path, Object *p_owner) const {
	return Dictionary();
}

String ScriptLanguageExtension::_auto_indent_code(const String &p_code, int32_t p_from_line, int32_t p_to_line) const {
	return String();
}

void ScriptLanguageExtension::_add_global_constant(const StringName &p_name, const Variant &p_value) {}

void ScriptLanguageExtension::_add_named_global_constant(const StringName &p_name, const Variant &p_value) {}

void ScriptLanguageExtension::_remove_named_global_constant(const StringName &p_name) {}

void ScriptLanguageExtension::_thread_enter() {}

void ScriptLanguageExtension::_thread_exit() {}

String ScriptLanguageExtension::_debug_get_error() const {
	return String();
}

int32_t ScriptLanguageExtension::_debug_get_stack_level_count() const {
	return 0;
}

int32_t ScriptLanguageExtension::_debug_get_stack_level_line(int32_t p_level) const {
	return 0;
}

String ScriptLanguageExtension::_debug_get_stack_level_function(int32_t p_level) const {
	return String();
}

String ScriptLanguageExtension::_debug_get_stack_level_source(int32_t p_level) const {
	return String();
}

Dictionary ScriptLanguageExtension::_debug_get_stack_level_locals(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}

Dictionary ScriptLanguageExtension::_debug_get_stack_level_members(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}

void *ScriptLanguageExtension::_debug_get_stack_level_instance(int32_t p_level) {
	return nullptr;
}

Dictionary ScriptLanguageExtension::_debug_get_globals(int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}

String ScriptLanguageExtension::_debug_parse_stack_level_expression(int32_t p_level, const String &p_expression, int32_t p_max_subitems, int32_t p_max_depth) {
	return String();
}

TypedArray<Dictionary> ScriptLanguageExtension::_debug_get_current_stack_info() {
	return TypedArray<Dictionary>();
}

void ScriptLanguageExtension::_reload_all_scripts() {}

void ScriptLanguageExtension::_reload_scripts(const Array &p_scripts, bool p_soft_reload) {}

void ScriptLanguageExtension::_reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {}

PackedStringArray ScriptLanguageExtension::_get_recognized_extensions() const {
	return PackedStringArray();
}

TypedArray<Dictionary> ScriptLanguageExtension::_get_public_functions() const {
	return TypedArray<Dictionary>();
}

Dictionary ScriptLanguageExtension::_get_public_constants() const {
	return Dictionary();
}

TypedArray<Dictionary> ScriptLanguageExtension::_get_public_annotations() const {
	return TypedArray<Dictionary>();
}

void ScriptLanguageExtension::_profiling_start() {}

void ScriptLanguageExtension::_profiling_stop() {}

void ScriptLanguageExtension::_profiling_set_save_native_calls(bool p_enable) {}

int32_t ScriptLanguageExtension::_profiling_get_accumulated_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max) {
	return 0;
}

int32_t ScriptLanguageExtension::_profiling_get_frame_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max) {
	return 0;
}

void ScriptLanguageExtension::_frame() {}

bool ScriptLanguageExtension::_handles_global_class_type(const String &p_type) const {
	return false;
}

Dictionary ScriptLanguageExtension::_get_global_class_name(const String &p_path) const {
	return Dictionary();
}

} // namespace godot
