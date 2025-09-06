/**************************************************************************/
/*  script_language_elf.cpp                                               */
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

#include "script_language_elf.h"
#include "../script_language_common.h"
#include "core/config/engine.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#ifdef TOOLS_ENABLED
#include "editor/editor_interface.h"
#endif
#include "scene/gui/control.h"
#include "scene/resources/texture.h"
#include "scene/resources/theme.h"
#include "script_elf.h"

String ELFScriptLanguage::get_name() const {
	return "ELF";
}

void ELFScriptLanguage::init() {}

String ELFScriptLanguage::get_type() const {
	return "ELFScript";
}

String ELFScriptLanguage::get_extension() const {
	return "elf";
}

void ELFScriptLanguage::finish() {}

Vector<String> ELFScriptLanguage::get_reserved_words() const {
	Vector<String> reserved_words;
	return reserved_words;
}

bool ELFScriptLanguage::is_control_flow_keyword(const String &p_keyword) const {
	return false;
}

Vector<String> ELFScriptLanguage::get_comment_delimiters() const {
	Vector<String> comment_delimiters;
	comment_delimiters.push_back("/* */");
	comment_delimiters.push_back("//");
	return comment_delimiters;
}

Vector<String> ELFScriptLanguage::get_doc_comment_delimiters() const {
	Vector<String> doc_comment_delimiters;
	doc_comment_delimiters.push_back("///");
	doc_comment_delimiters.push_back("/** */");
	return doc_comment_delimiters;
}

Vector<String> ELFScriptLanguage::get_string_delimiters() const {
	Vector<String> string_delimiters;
	string_delimiters.push_back("' '");
	string_delimiters.push_back("\" \"");
	return string_delimiters;
}

Ref<Script> ELFScriptLanguage::make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const {
	ELFScript *elf_script = memnew(ELFScript);
	return Ref<Script>(elf_script);
}

Vector<ScriptLanguage::ScriptTemplate> ELFScriptLanguage::get_built_in_templates(const StringName &p_object) {
	Vector<ScriptLanguage::ScriptTemplate> templates;
	return templates;
}

bool ELFScriptLanguage::is_using_templates() {
	return false;
}

bool ELFScriptLanguage::validate(const String &p_script, const String &p_path, List<String> *r_functions, List<ScriptLanguage::ScriptError> *r_errors, List<ScriptLanguage::Warning> *r_warnings, HashSet<int> *r_safe_lines) const {
	return true; // For now, assume all ELF scripts are valid
}

Script *ELFScriptLanguage::create_script() const {
	ELFScript *elf_script = memnew(ELFScript);
	return elf_script;
}

bool ELFScriptLanguage::has_named_classes() const {
	return true;
}

bool ELFScriptLanguage::supports_builtin_mode() const {
	return true;
}

bool ELFScriptLanguage::supports_documentation() const {
	return false;
}

// These methods are implemented as inline in header with return true/false

int ELFScriptLanguage::find_function(const String &p_function, const String &p_code) const {
	return -1;
}

String ELFScriptLanguage::make_function(const String &p_class, const String &p_name, const PackedStringArray &p_args) const {
	return String();
}

// External editor methods not implemented for internal module

Error ELFScriptLanguage::complete_code(const String &p_code, const String &p_path, Object *p_owner, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_forced, String &r_call_hint) {
	return OK; // No code completion for ELF scripts
}

void ELFScriptLanguage::auto_indent_code(String &p_code, int p_from_line, int p_to_line) const {
	// No auto-indent for ELF scripts
}

// Code analysis methods not implemented for internal module

void ELFScriptLanguage::add_global_constant(const StringName &p_variable, const Variant &p_value) {}
void ELFScriptLanguage::add_named_global_constant(const StringName &p_name, const Variant &p_value) {}
void ELFScriptLanguage::remove_named_global_constant(const StringName &p_name) {}
// Thread methods not needed for internal module

String ELFScriptLanguage::debug_get_error() const {
	return String();
}

int ELFScriptLanguage::debug_get_stack_level_count() const {
	return 0;
}

int ELFScriptLanguage::debug_get_stack_level_line(int p_level) const {
	return 0;
}

String ELFScriptLanguage::debug_get_stack_level_function(int p_level) const {
	return String();
}

void ELFScriptLanguage::debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	// No debug locals available
}

void ELFScriptLanguage::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	// No debug members available
}

ScriptInstance *ELFScriptLanguage::debug_get_stack_level_instance(int p_level) {
	return nullptr;
}

void ELFScriptLanguage::debug_get_globals(List<String> *p_globals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	// No debug globals available
}

String ELFScriptLanguage::debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) {
	return String();
}

Vector<ScriptLanguage::StackInfo> ELFScriptLanguage::debug_get_current_stack_info() {
	Vector<ScriptLanguage::StackInfo> stack_info;
	return stack_info;
}

void ELFScriptLanguage::reload_all_scripts() {}
void ELFScriptLanguage::reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {}
void ELFScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("elf");
}

void ELFScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {
	// No public functions to add
}

void ELFScriptLanguage::get_public_constants(List<Pair<String, Variant>> *p_constants) const {
	// No public constants to add
}

void ELFScriptLanguage::get_public_annotations(List<MethodInfo> *p_annotations) const {
	// No annotations to add
}

void ELFScriptLanguage::profiling_set_save_native_calls(bool p_enable) {
	// Not implemented
}

void ELFScriptLanguage::profiling_start() {}

void ELFScriptLanguage::profiling_stop() {}

int ELFScriptLanguage::profiling_get_accumulated_data(ScriptLanguage::ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

int ELFScriptLanguage::profiling_get_frame_data(ScriptLanguage::ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

void ELFScriptLanguage::reload_scripts(const Array &p_scripts, bool p_soft_reload) {
	// No special reload handling needed
}

String ELFScriptLanguage::debug_get_stack_level_source(int p_level) const {
	return String();
}

void ELFScriptLanguage::frame() {
#if TOOLS_ENABLED
	static bool icon_registered = register_language_icons;
	if (!icon_registered && Engine::get_singleton()->is_editor_hint()) {
		icon_registered = true;
		// Manually register ELFScript icon
		load_icon();
		// Register theme callback
		EditorInterface::get_singleton()->get_base_control()->connect("theme_changed", callable_mp(this, &ELFScriptLanguage::load_icon));
	}
#endif
}

void ELFScriptLanguage::load_icon() {
#if TOOLS_ENABLED
	static bool reenter = false;
	if (reenter) {
		return;
	}
	reenter = true;
	static const String icon_path = "res://addons/godot_sandbox/Sandbox.svg";
	Ref<FileAccess> fa = FileAccess::open(icon_path, FileAccess::READ);
	if (Engine::get_singleton()->is_editor_hint() && fa.is_valid()) {
		Ref<Theme> editor_theme = EditorInterface::get_singleton()->get_editor_theme();
		if (editor_theme.is_valid() && !editor_theme->has_icon("ELFScript", "EditorIcons")) {
			Ref<Texture2D> tex = ResourceLoader::load(icon_path);
			editor_theme->set_icon("ELFScript", "EditorIcons", tex);
		}
	}
	reenter = false;
#endif
}

bool ELFScriptLanguage::handles_global_class_type(const String &p_type) const {
	return p_type == "ELFScript" || p_type == "Sandbox";
}

String ELFScriptLanguage::get_global_class_name(const String &p_path, String *r_base_type, String *r_icon_path, bool *r_is_abstract, bool *r_is_tool) const {
	Ref<Resource> resource = ResourceLoader::load(p_path);
	Ref<ELFScript> elf_model = Object::cast_to<ELFScript>(resource.ptr());
	if (elf_model.is_valid()) {
		if (r_base_type)
			*r_base_type = "Sandbox";
		if (r_icon_path)
			*r_icon_path = "res://addons/godot_sandbox/Sandbox.svg";
		if (r_is_abstract)
			*r_is_abstract = false;
		if (r_is_tool)
			*r_is_tool = true;
		return elf_model->get_global_name();
	}
	return String();
}
