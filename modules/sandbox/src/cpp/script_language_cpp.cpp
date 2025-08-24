/**************************************************************************/
/*  script_language_cpp.cpp                                               */
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

#include "script_language_cpp.h"
#include "core/config/engine.h"
#include "core/io/file_access.h"
#include "core/os/memory.h"
#include "script_cpp.h"

static CPPScriptLanguage *cpp_language = nullptr;

void CPPScriptLanguage::init_language() {
	cpp_language = new CPPScriptLanguage();
}

void CPPScriptLanguage::deinit() {
	if (cpp_language) {
		delete cpp_language;
		cpp_language = nullptr;
	}
}

CPPScriptLanguage *CPPScriptLanguage::get_singleton() {
	return cpp_language;
}

// ScriptLanguage interface implementation
String CPPScriptLanguage::get_name() const {
	return "C++";
}

void CPPScriptLanguage::init() {}

String CPPScriptLanguage::get_type() const {
	return "CPPScript";
}

String CPPScriptLanguage::get_extension() const {
	return "cpp";
}

void CPPScriptLanguage::finish() {}

Vector<String> CPPScriptLanguage::get_reserved_words() const {
	Vector<String> reserved_words;
	// C++ keywords
	static const char *_reserved_words[] = {
		"alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit", "atomic_noexcept",
		"auto", "bitand", "bitor", "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
		"class", "compl", "concept", "const", "consteval", "constexpr", "constinit", "const_cast", "continue",
		"co_await", "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast",
		"else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto", "if",
		"inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr",
		"operator", "or", "or_eq", "private", "protected", "public", "reflexpr", "register", "reinterpret_cast",
		"requires", "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
		"switch", "synchronized", "template", "this", "thread_local", "throw", "true", "try", "typedef",
		"typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t",
		"while", "xor", "xor_eq",
		nullptr
	};

	const char **w = _reserved_words;
	while (*w) {
		reserved_words.push_back(*w);
		w++;
	}
	return reserved_words;
}

bool CPPScriptLanguage::is_control_flow_keyword(const String &p_keyword) const {
	return p_keyword == "if" || p_keyword == "else" || p_keyword == "for" || p_keyword == "while" ||
			p_keyword == "do" || p_keyword == "switch" || p_keyword == "case" || p_keyword == "default" ||
			p_keyword == "break" || p_keyword == "continue" || p_keyword == "return" || p_keyword == "goto" ||
			p_keyword == "try" || p_keyword == "catch" || p_keyword == "throw";
}

Vector<String> CPPScriptLanguage::get_comment_delimiters() const {
	Vector<String> comment_delimiters;
	comment_delimiters.push_back("/* */");
	comment_delimiters.push_back("//");
	return comment_delimiters;
}

Vector<String> CPPScriptLanguage::get_doc_comment_delimiters() const {
	Vector<String> doc_comment_delimiters;
	doc_comment_delimiters.push_back("///");
	doc_comment_delimiters.push_back("/** */");
	return doc_comment_delimiters;
}

Vector<String> CPPScriptLanguage::get_string_delimiters() const {
	Vector<String> string_delimiters;
	string_delimiters.push_back("' '");
	string_delimiters.push_back("\" \"");
	return string_delimiters;
}

Ref<Script> CPPScriptLanguage::make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const {
	CPPScript *cpp_script = memnew(CPPScript);
	return Ref<Script>(cpp_script);
}

Vector<ScriptLanguage::ScriptTemplate> CPPScriptLanguage::get_built_in_templates(const StringName &p_object) {
	Vector<ScriptLanguage::ScriptTemplate> templates;
	return templates;
}

bool CPPScriptLanguage::is_using_templates() {
	return false;
}

bool CPPScriptLanguage::validate(const String &p_script, const String &p_path, List<String> *r_functions, List<ScriptLanguage::ScriptError> *r_errors, List<ScriptLanguage::Warning> *r_warnings, HashSet<int> *r_safe_lines) const {
	return true; // For now, assume all C++ scripts are valid
}

Script *CPPScriptLanguage::create_script() const {
	CPPScript *cpp_script = memnew(CPPScript);
	return cpp_script;
}

bool CPPScriptLanguage::has_named_classes() const {
	return true;
}

bool CPPScriptLanguage::supports_builtin_mode() const {
	return true;
}

bool CPPScriptLanguage::supports_documentation() const {
	return false;
}

bool CPPScriptLanguage::can_inherit_from_file() const {
	return true;
}

int CPPScriptLanguage::find_function(const String &p_function, const String &p_code) const {
	return -1;
}

String CPPScriptLanguage::make_function(const String &p_class, const String &p_name, const PackedStringArray &p_args) const {
	return String();
}

Error CPPScriptLanguage::complete_code(const String &p_code, const String &p_path, Object *p_owner, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_forced, String &r_call_hint) {
	return ERR_UNAVAILABLE; // No code completion for C++ proxy scripts
}

void CPPScriptLanguage::auto_indent_code(String &p_code, int p_from_line, int p_to_line) const {
	// No auto-indent for C++ proxy scripts
}

void CPPScriptLanguage::add_global_constant(const StringName &p_variable, const Variant &p_value) {}
void CPPScriptLanguage::add_named_global_constant(const StringName &p_name, const Variant &p_value) {}
void CPPScriptLanguage::remove_named_global_constant(const StringName &p_name) {}

void CPPScriptLanguage::frame() {
	// Frame update logic if needed
}

void CPPScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("cpp");
	p_extensions->push_back("cxx");
	p_extensions->push_back("cc");
}

void CPPScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {
	// No public functions to add
}

void CPPScriptLanguage::get_public_constants(List<Pair<String, Variant>> *p_constants) const {
	// No public constants to add
}

void CPPScriptLanguage::get_public_annotations(List<MethodInfo> *p_annotations) const {
	// No annotations to add
}

bool CPPScriptLanguage::handles_global_class_type(const String &p_type) const {
	return p_type == "CPPScript";
}

String CPPScriptLanguage::get_global_class_name(const String &p_path, String *r_base_type, String *r_icon_path, bool *r_is_abstract, bool *r_is_tool) const {
	return String();
}

// Debug functions
String CPPScriptLanguage::debug_get_error() const {
	return String();
}

int CPPScriptLanguage::debug_get_stack_level_count() const {
	return 0;
}

int CPPScriptLanguage::debug_get_stack_level_line(int p_level) const {
	return 0;
}

String CPPScriptLanguage::debug_get_stack_level_function(int p_level) const {
	return String();
}

String CPPScriptLanguage::debug_get_stack_level_source(int p_level) const {
	return String();
}

void CPPScriptLanguage::debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	// No debug locals available
}

void CPPScriptLanguage::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	// No debug members available
}

ScriptInstance *CPPScriptLanguage::debug_get_stack_level_instance(int p_level) {
	return nullptr;
}

void CPPScriptLanguage::debug_get_globals(List<String> *p_globals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	// No debug globals available
}

String CPPScriptLanguage::debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) {
	return String();
}

Vector<ScriptLanguage::StackInfo> CPPScriptLanguage::debug_get_current_stack_info() {
	Vector<ScriptLanguage::StackInfo> stack_info;
	return stack_info;
}

void CPPScriptLanguage::reload_all_scripts() {}
void CPPScriptLanguage::reload_scripts(const Array &p_scripts, bool p_soft_reload) {}
void CPPScriptLanguage::reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {}

// Profiling functions
void CPPScriptLanguage::profiling_start() {}

void CPPScriptLanguage::profiling_stop() {}

void CPPScriptLanguage::profiling_set_save_native_calls(bool p_enable) {
	// Not implemented
}

int CPPScriptLanguage::profiling_get_accumulated_data(ScriptLanguage::ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

int CPPScriptLanguage::profiling_get_frame_data(ScriptLanguage::ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

void CPPScriptLanguage::load_icon() {
	// Icon loading functionality (simplified for now)
}
