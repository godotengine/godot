/**************************************************************************/
/*  gdscript_language_wrapper.cpp                                         */
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

#include "gdscript_language_wrapper.h"

#include "gdscript_wrapper.h"
#include "modules/gdscript/gdscript.h"

void GDScriptLanguageWrapper::_bind_methods() {
	// No methods to bind - this is a pure wrapper
}

GDScriptLanguageWrapper::GDScriptLanguageWrapper() {
	original_language = nullptr;
}

GDScriptLanguageWrapper::~GDScriptLanguageWrapper() {
	// Don't delete original_language - it's managed elsewhere
	original_language = nullptr;
}

void GDScriptLanguageWrapper::set_original_language(GDScriptLanguage *p_original) {
	original_language = p_original;
}

// ScriptLanguage interface - all methods delegate to original
// Phase 0: 100% pass-through (strangler vine pattern)

String GDScriptLanguageWrapper::get_name() const {
	ERR_FAIL_NULL_V(original_language, "GDScript");
	return original_language->get_name();
}

void GDScriptLanguageWrapper::init() {
	ERR_FAIL_NULL(original_language);
	original_language->init();
}

String GDScriptLanguageWrapper::get_type() const {
	ERR_FAIL_NULL_V(original_language, "");
	return original_language->get_type();
}

String GDScriptLanguageWrapper::get_extension() const {
	ERR_FAIL_NULL_V(original_language, "");
	return original_language->get_extension();
}

void GDScriptLanguageWrapper::finish() {
	ERR_FAIL_NULL(original_language);
	original_language->finish();
}

Vector<String> GDScriptLanguageWrapper::get_reserved_words() const {
	ERR_FAIL_NULL_V(original_language, Vector<String>());
	return original_language->get_reserved_words();
}

bool GDScriptLanguageWrapper::is_control_flow_keyword(const String &p_string) const {
	ERR_FAIL_NULL_V(original_language, false);
	return original_language->is_control_flow_keyword(p_string);
}

Vector<String> GDScriptLanguageWrapper::get_comment_delimiters() const {
	ERR_FAIL_NULL_V(original_language, Vector<String>());
	return original_language->get_comment_delimiters();
}

Vector<String> GDScriptLanguageWrapper::get_doc_comment_delimiters() const {
	ERR_FAIL_NULL_V(original_language, Vector<String>());
	return original_language->get_doc_comment_delimiters();
}

Vector<String> GDScriptLanguageWrapper::get_string_delimiters() const {
	ERR_FAIL_NULL_V(original_language, Vector<String>());
	return original_language->get_string_delimiters();
}

Ref<Script> GDScriptLanguageWrapper::make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const {
	ERR_FAIL_NULL_V(original_language, Ref<Script>());
	return original_language->make_template(p_template, p_class_name, p_base_class_name);
}

Vector<ScriptLanguage::ScriptTemplate> GDScriptLanguageWrapper::get_built_in_templates(const StringName &p_object) {
	ERR_FAIL_NULL_V(original_language, Vector<ScriptTemplate>());
	return original_language->get_built_in_templates(p_object);
}

bool GDScriptLanguageWrapper::is_using_templates() {
	ERR_FAIL_NULL_V(original_language, false);
	return original_language->is_using_templates();
}

bool GDScriptLanguageWrapper::validate(const String &p_script, const String &p_path, List<String> *r_functions, List<ScriptError> *r_errors, List<Warning> *r_warnings, HashSet<int> *r_safe_lines) const {
	ERR_FAIL_NULL_V(original_language, false);
	return original_language->validate(p_script, p_path, r_functions, r_errors, r_warnings, r_safe_lines);
}

String GDScriptLanguageWrapper::validate_path(const String &p_path) const {
	ERR_FAIL_NULL_V(original_language, "");
	return original_language->validate_path(p_path);
}

Script *GDScriptLanguageWrapper::create_script() const {
	ERR_FAIL_NULL_V(original_language, nullptr);
	// Create original GDScript
	Script *original = original_language->create_script();
	if (!original) {
		return nullptr;
	}

	// Wrap it in GDScriptWrapper
	GDScript *gdscript = Object::cast_to<GDScript>(original);
	if (!gdscript) {
		// Not a GDScript, return as-is (shouldn't happen, but be safe)
		return original;
	}

	GDScriptWrapper *wrapper = memnew(GDScriptWrapper);
	wrapper->set_original_script(Ref<GDScript>(gdscript));

	// Copy path and other properties from original
	wrapper->set_path(original->get_path());

	return wrapper;
}

bool GDScriptLanguageWrapper::supports_builtin_mode() const {
	ERR_FAIL_NULL_V(original_language, false);
	return original_language->supports_builtin_mode();
}

bool GDScriptLanguageWrapper::supports_documentation() const {
	ERR_FAIL_NULL_V(original_language, false);
	return original_language->supports_documentation();
}

bool GDScriptLanguageWrapper::can_inherit_from_file() const {
	ERR_FAIL_NULL_V(original_language, false);
	return original_language->can_inherit_from_file();
}

int GDScriptLanguageWrapper::find_function(const String &p_function, const String &p_code) const {
	ERR_FAIL_NULL_V(original_language, -1);
	return original_language->find_function(p_function, p_code);
}

String GDScriptLanguageWrapper::make_function(const String &p_class, const String &p_name, const PackedStringArray &p_args) const {
	ERR_FAIL_NULL_V(original_language, "");
	return original_language->make_function(p_class, p_name, p_args);
}

bool GDScriptLanguageWrapper::can_make_function() const {
	ERR_FAIL_NULL_V(original_language, false);
	return original_language->can_make_function();
}

Error GDScriptLanguageWrapper::open_in_external_editor(const Ref<Script> &p_script, int p_line, int p_col) {
	ERR_FAIL_NULL_V(original_language, ERR_UNAVAILABLE);
	return original_language->open_in_external_editor(p_script, p_line, p_col);
}

bool GDScriptLanguageWrapper::overrides_external_editor() {
	ERR_FAIL_NULL_V(original_language, false);
	return original_language->overrides_external_editor();
}

ScriptLanguage::ScriptNameCasing GDScriptLanguageWrapper::preferred_file_name_casing() const {
	ERR_FAIL_NULL_V(original_language, SCRIPT_NAME_CASING_SNAKE_CASE);
	return original_language->preferred_file_name_casing();
}

Error GDScriptLanguageWrapper::complete_code(const String &p_code, const String &p_path, Object *p_owner, List<CodeCompletionOption> *r_options, bool &r_force, String &r_call_hint) {
	ERR_FAIL_NULL_V(original_language, ERR_UNAVAILABLE);
	return original_language->complete_code(p_code, p_path, p_owner, r_options, r_force, r_call_hint);
}

#ifdef TOOLS_ENABLED
Error GDScriptLanguageWrapper::lookup_code(const String &p_code, const String &p_symbol, const String &p_path, Object *p_owner, LookupResult &r_result) {
	ERR_FAIL_NULL_V(original_language, ERR_UNAVAILABLE);
	return original_language->lookup_code(p_code, p_symbol, p_path, p_owner, r_result);
}
#endif

void GDScriptLanguageWrapper::auto_indent_code(String &p_code, int p_from_line, int p_to_line) const {
	ERR_FAIL_NULL(original_language);
	original_language->auto_indent_code(p_code, p_from_line, p_to_line);
}

void GDScriptLanguageWrapper::add_global_constant(const StringName &p_variable, const Variant &p_value) {
	ERR_FAIL_NULL(original_language);
	original_language->add_global_constant(p_variable, p_value);
}

void GDScriptLanguageWrapper::add_named_global_constant(const StringName &p_name, const Variant &p_value) {
	ERR_FAIL_NULL(original_language);
	original_language->add_named_global_constant(p_name, p_value);
}

void GDScriptLanguageWrapper::remove_named_global_constant(const StringName &p_name) {
	ERR_FAIL_NULL(original_language);
	original_language->remove_named_global_constant(p_name);
}

void GDScriptLanguageWrapper::thread_enter() {
	ERR_FAIL_NULL(original_language);
	original_language->thread_enter();
}

void GDScriptLanguageWrapper::thread_exit() {
	ERR_FAIL_NULL(original_language);
	original_language->thread_exit();
}

String GDScriptLanguageWrapper::debug_get_error() const {
	ERR_FAIL_NULL_V(original_language, "");
	return original_language->debug_get_error();
}

int GDScriptLanguageWrapper::debug_get_stack_level_count() const {
	ERR_FAIL_NULL_V(original_language, 0);
	return original_language->debug_get_stack_level_count();
}

int GDScriptLanguageWrapper::debug_get_stack_level_line(int p_level) const {
	ERR_FAIL_NULL_V(original_language, -1);
	return original_language->debug_get_stack_level_line(p_level);
}

String GDScriptLanguageWrapper::debug_get_stack_level_function(int p_level) const {
	ERR_FAIL_NULL_V(original_language, "");
	return original_language->debug_get_stack_level_function(p_level);
}

String GDScriptLanguageWrapper::debug_get_stack_level_source(int p_level) const {
	ERR_FAIL_NULL_V(original_language, "");
	return original_language->debug_get_stack_level_source(p_level);
}

void GDScriptLanguageWrapper::debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	ERR_FAIL_NULL(original_language);
	original_language->debug_get_stack_level_locals(p_level, p_locals, p_values, p_max_subitems, p_max_depth);
}

void GDScriptLanguageWrapper::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	ERR_FAIL_NULL(original_language);
	original_language->debug_get_stack_level_members(p_level, p_members, p_values, p_max_subitems, p_max_depth);
}

ScriptInstance *GDScriptLanguageWrapper::debug_get_stack_level_instance(int p_level) {
	ERR_FAIL_NULL_V(original_language, nullptr);
	return original_language->debug_get_stack_level_instance(p_level);
}

void GDScriptLanguageWrapper::debug_get_globals(List<String> *p_globals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	ERR_FAIL_NULL(original_language);
	original_language->debug_get_globals(p_globals, p_values, p_max_subitems, p_max_depth);
}

void GDScriptLanguageWrapper::get_recognized_extensions(List<String> *p_extensions) const {
	ERR_FAIL_NULL(original_language);
	original_language->get_recognized_extensions(p_extensions);
}

void GDScriptLanguageWrapper::get_public_functions(List<MethodInfo> *p_functions) const {
	ERR_FAIL_NULL(original_language);
	original_language->get_public_functions(p_functions);
}

void GDScriptLanguageWrapper::get_public_constants(List<Pair<String, Variant>> *p_constants) const {
	ERR_FAIL_NULL(original_language);
	original_language->get_public_constants(p_constants);
}

void GDScriptLanguageWrapper::get_public_annotations(List<MethodInfo> *p_annotations) const {
	ERR_FAIL_NULL(original_language);
	original_language->get_public_annotations(p_annotations);
}

void GDScriptLanguageWrapper::profiling_set_save_native_calls(bool p_enable) {
	ERR_FAIL_NULL(original_language);
	original_language->profiling_set_save_native_calls(p_enable);
}

void GDScriptLanguageWrapper::frame() {
	ERR_FAIL_NULL(original_language);
	original_language->frame();
}

void GDScriptLanguageWrapper::reload_all_scripts() {
	ERR_FAIL_NULL(original_language);
	original_language->reload_all_scripts();
}

void GDScriptLanguageWrapper::reload_scripts(const Array &p_scripts, bool p_soft_reload) {
	ERR_FAIL_NULL(original_language);
	original_language->reload_scripts(p_scripts, p_soft_reload);
}

void GDScriptLanguageWrapper::reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {
	ERR_FAIL_NULL(original_language);
	original_language->reload_tool_script(p_script, p_soft_reload);
}

bool GDScriptLanguageWrapper::handles_global_class_type(const String &p_type) const {
	ERR_FAIL_NULL_V(original_language, false);
	return original_language->handles_global_class_type(p_type);
}

String GDScriptLanguageWrapper::get_global_class_name(const String &p_path, String *r_base_type, String *r_icon_path, bool *r_is_abstract, bool *r_is_tool) const {
	ERR_FAIL_NULL_V(original_language, "");
	return original_language->get_global_class_name(p_path, r_base_type, r_icon_path, r_is_abstract, r_is_tool);
}

Vector<ScriptLanguage::StackInfo> GDScriptLanguageWrapper::debug_get_current_stack_info() {
	ERR_FAIL_NULL_V(original_language, Vector<StackInfo>());
	return original_language->debug_get_current_stack_info();
}

String GDScriptLanguageWrapper::debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) {
	ERR_FAIL_NULL_V(original_language, "");
	return original_language->debug_parse_stack_level_expression(p_level, p_expression, p_max_subitems, p_max_depth);
}

void GDScriptLanguageWrapper::profiling_start() {
	ERR_FAIL_NULL(original_language);
	original_language->profiling_start();
}

void GDScriptLanguageWrapper::profiling_stop() {
	ERR_FAIL_NULL(original_language);
	original_language->profiling_stop();
}

int GDScriptLanguageWrapper::profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) {
	ERR_FAIL_NULL_V(original_language, 0);
	return original_language->profiling_get_accumulated_data(p_info_arr, p_info_max);
}

int GDScriptLanguageWrapper::profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) {
	ERR_FAIL_NULL_V(original_language, 0);
	return original_language->profiling_get_frame_data(p_info_arr, p_info_max);
}
