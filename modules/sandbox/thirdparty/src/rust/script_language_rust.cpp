#include "script_language_rust.h"
#include "../script_language_common.h"
#include "script_rust.h"
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/theme.hpp>
#include <string>
#include <unordered_set>
static constexpr const char *icon_path = "res://addons/godot_sandbox/RustScript.svg";

static RustScriptLanguage *rust_language;

void RustScriptLanguage::init() {
	rust_language = memnew(RustScriptLanguage);
	Engine::get_singleton()->register_script_language(rust_language);
}
void RustScriptLanguage::deinit() {
	if (rust_language) {
		Engine::get_singleton()->unregister_script_language(rust_language);
		memdelete(rust_language);
		rust_language = nullptr;
	}
}

RustScriptLanguage *RustScriptLanguage::get_singleton() {
	return rust_language;
}

String RustScriptLanguage::_get_name() const {
	return "RustScript";
}
void RustScriptLanguage::_init() {}
String RustScriptLanguage::_get_type() const {
	return "RustScript";
}
String RustScriptLanguage::_get_extension() const {
	return "rs";
}
void RustScriptLanguage::_finish() {}
PackedStringArray RustScriptLanguage::_get_reserved_words() const {
	static const PackedStringArray reserved_words{
		"as",
		"use",
		"extern crate",
		"break",
		"const",
		"continue",
		"crate",
		"else",
		"if",
		"if let",
		"enum",
		"extern",
		"false",
		"fn",
		"for",
		"if",
		"impl",
		"in",
		"for",
		"let",
		"loop",
		"match",
		"mod",
		"move",
		"mut",
		"pub",
		"impl",
		"ref",
		"return",
		"Self",
		"self",
		"static",
		"struct",
		"super",
		"trait",
		"true",
		"type",
		"unsafe",
		"use",
		"where",
		"while",
		"abstract",
		"alignof",
		"become",
		"box",
		"do",
		"final",
		"macro",
		"offsetof",
		"override",
		"priv",
		"proc",
		"pure",
		"sizeof",
		"typeof",
		"unsized",
		"virtual",
		"yield"
		// Integers, floats, and strings are not reserved words in Rust, but we want the highlighting to be consistent
		"i8", "i16", "i32", "i64", "i128", "isize",
		"u8", "u16", "u32", "u64", "u128", "usize",
		"f32", "f64",
		"GodotString", "Variant"
	};
	return reserved_words;
}
bool RustScriptLanguage::_is_control_flow_keyword(const String &p_keyword) const {
	static const std::unordered_set<std::string> control_flow_keywords{
		"if", "else", "switch", "case", "default", "while", "loop", "for", "break", "continue", "return", "goto"
	};
	return control_flow_keywords.find(p_keyword.utf8().get_data()) != control_flow_keywords.end();
}
PackedStringArray RustScriptLanguage::_get_comment_delimiters() const {
	PackedStringArray comment_delimiters;
	comment_delimiters.push_back("/* */");
	comment_delimiters.push_back("//");
	return comment_delimiters;
}
PackedStringArray RustScriptLanguage::_get_doc_comment_delimiters() const {
	PackedStringArray doc_comment_delimiters;
	doc_comment_delimiters.push_back("///");
	doc_comment_delimiters.push_back("/** */");
	return doc_comment_delimiters;
}
PackedStringArray RustScriptLanguage::_get_string_delimiters() const {
	PackedStringArray string_delimiters;
	string_delimiters.push_back("' '");
	string_delimiters.push_back("\" \"");
	return string_delimiters;
}
Ref<Script> RustScriptLanguage::_make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const {
	RustScript *script = memnew(RustScript);
	return Ref<Script>(script);
}
TypedArray<Dictionary> RustScriptLanguage::_get_built_in_templates(const StringName &p_object) const {
	return TypedArray<Dictionary>();
}
bool RustScriptLanguage::_is_using_templates() {
	return false;
}
Dictionary RustScriptLanguage::_validate(const String &p_script, const String &p_path, bool p_validate_functions, bool p_validate_errors, bool p_validate_warnings, bool p_validate_safe_lines) const {
	return Dictionary();
}
String RustScriptLanguage::_validate_path(const String &p_path) const {
	return String();
}
Object *RustScriptLanguage::_create_script() const {
	RustScript *script = memnew(RustScript);
	return script;
}
bool RustScriptLanguage::_has_named_classes() const {
	return false;
}
bool RustScriptLanguage::_supports_builtin_mode() const {
	return false;
}
bool RustScriptLanguage::_supports_documentation() const {
	return false;
}
bool RustScriptLanguage::_can_inherit_from_file() const {
	return false;
}
int32_t RustScriptLanguage::_find_function(const String &p_function, const String &p_code) const {
	return -1;
}
String RustScriptLanguage::_make_function(const String &p_class_name, const String &p_function_name, const PackedStringArray &p_function_args) const {
	return String();
}
Error RustScriptLanguage::_open_in_external_editor(const Ref<Script> &p_script, int32_t p_line, int32_t p_column) {
	return Error::OK;
}
bool RustScriptLanguage::_overrides_external_editor() {
	return false;
}
Dictionary RustScriptLanguage::_complete_code(const String &p_code, const String &p_path, Object *p_owner) const {
	return Dictionary();
}
Dictionary RustScriptLanguage::_lookup_code(const String &p_code, const String &p_symbol, const String &p_path, Object *p_owner) const {
	return Dictionary();
}
String RustScriptLanguage::_auto_indent_code(const String &p_code, int32_t p_from_line, int32_t p_to_line) const {
	return String();
}
void RustScriptLanguage::_add_global_constant(const StringName &p_name, const Variant &p_value) {}
void RustScriptLanguage::_add_named_global_constant(const StringName &p_name, const Variant &p_value) {}
void RustScriptLanguage::_remove_named_global_constant(const StringName &p_name) {}
void RustScriptLanguage::_thread_enter() {}
void RustScriptLanguage::_thread_exit() {}
String RustScriptLanguage::_debug_get_error() const {
	return String();
}
int32_t RustScriptLanguage::_debug_get_stack_level_count() const {
	return 0;
}
int32_t RustScriptLanguage::_debug_get_stack_level_line(int32_t p_level) const {
	return 0;
}
String RustScriptLanguage::_debug_get_stack_level_function(int32_t p_level) const {
	return String();
}
Dictionary RustScriptLanguage::_debug_get_stack_level_locals(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}
Dictionary RustScriptLanguage::_debug_get_stack_level_members(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}
void *RustScriptLanguage::_debug_get_stack_level_instance(int32_t p_level) {
	return nullptr;
}
Dictionary RustScriptLanguage::_debug_get_globals(int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}
String RustScriptLanguage::_debug_parse_stack_level_expression(int32_t p_level, const String &p_expression, int32_t p_max_subitems, int32_t p_max_depth) {
	return String();
}
TypedArray<Dictionary> RustScriptLanguage::_debug_get_current_stack_info() {
	return TypedArray<Dictionary>();
}
void RustScriptLanguage::_reload_all_scripts() {}
void RustScriptLanguage::_reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {}
PackedStringArray RustScriptLanguage::_get_recognized_extensions() const {
	PackedStringArray array;
	array.push_back("rs");
	return array;
}
TypedArray<Dictionary> RustScriptLanguage::_get_public_functions() const {
	return TypedArray<Dictionary>();
}
Dictionary RustScriptLanguage::_get_public_constants() const {
	return Dictionary();
}
TypedArray<Dictionary> RustScriptLanguage::_get_public_annotations() const {
	return TypedArray<Dictionary>();
}
void RustScriptLanguage::_profiling_start() {}
void RustScriptLanguage::_profiling_stop() {}
int32_t RustScriptLanguage::_profiling_get_accumulated_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max) {
	return 0;
}
int32_t RustScriptLanguage::_profiling_get_frame_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max) {
	return 0;
}
void RustScriptLanguage::_frame() {
	static bool icon_registered = register_language_icons;
	if (!icon_registered && Engine::get_singleton()->is_editor_hint()) {
		icon_registered = true;
		// Manually register language icon
		load_icon();
		// Register theme callback
		EditorInterface::get_singleton()->get_base_control()->connect("theme_changed", callable_mp(this, &RustScriptLanguage::load_icon));
	}
}
void RustScriptLanguage::load_icon() {
	static bool reenter = false;
	if (reenter)
		return;
	reenter = true;
	if (Engine::get_singleton()->is_editor_hint() && FileAccess::file_exists(icon_path)) {
		Ref<Theme> editor_theme = EditorInterface::get_singleton()->get_editor_theme();
		if (editor_theme.is_valid() && !editor_theme->has_icon("RustScript", "EditorIcons")) {
			ResourceLoader *resource_loader = ResourceLoader::get_singleton();
			Ref<Texture2D> tex = resource_loader->load(icon_path);
			editor_theme->set_icon("RustScript", "EditorIcons", tex);
		}
	}
	reenter = false;
}
bool RustScriptLanguage::_handles_global_class_type(const String &p_type) const {
	return p_type == "RustScript";
}
Dictionary RustScriptLanguage::_get_global_class_name(const String &p_path) const {
	return Dictionary();
}
