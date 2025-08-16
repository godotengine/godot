#include "script_language_elf.h"
#include "../script_language_common.h"
#include "script_elf.h"
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/theme.hpp>
static constexpr const char *icon_path = "res://addons/godot_sandbox/Sandbox.svg";

String ELFScriptLanguage::_get_name() const {
	return "ELF";
}
void ELFScriptLanguage::_init() {}
String ELFScriptLanguage::_get_type() const {
	return "ELFScript";
}
String ELFScriptLanguage::_get_extension() const {
	return "elf";
}
void ELFScriptLanguage::_finish() {}
PackedStringArray ELFScriptLanguage::_get_reserved_words() const {
	return PackedStringArray();
}
bool ELFScriptLanguage::_is_control_flow_keyword(const String &p_keyword) const {
	return false;
}
PackedStringArray ELFScriptLanguage::_get_comment_delimiters() const {
	PackedStringArray comment_delimiters;
	comment_delimiters.push_back("/* */");
	comment_delimiters.push_back("//");
	return comment_delimiters;
}
PackedStringArray ELFScriptLanguage::_get_doc_comment_delimiters() const {
	PackedStringArray doc_comment_delimiters;
	doc_comment_delimiters.push_back("///");
	doc_comment_delimiters.push_back("/** */");
	return doc_comment_delimiters;
}
PackedStringArray ELFScriptLanguage::_get_string_delimiters() const {
	PackedStringArray string_delimiters;
	string_delimiters.push_back("' '");
	string_delimiters.push_back("\" \"");
	return string_delimiters;
}
Ref<Script> ELFScriptLanguage::_make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const {
	ELFScript *elf_script = memnew(ELFScript);
	return Ref<Script>(elf_script);
}
TypedArray<Dictionary> ELFScriptLanguage::_get_built_in_templates(const StringName &p_object) const {
	return TypedArray<Dictionary>();
}
bool ELFScriptLanguage::_is_using_templates() {
	return false;
}
Dictionary ELFScriptLanguage::_validate(const String &p_script, const String &p_path, bool p_validate_functions, bool p_validate_errors, bool p_validate_warnings, bool p_validate_safe_lines) const {
	return Dictionary();
}
String ELFScriptLanguage::_validate_path(const String &p_path) const {
	return String();
}
Object *ELFScriptLanguage::_create_script() const {
	ELFScript *script = memnew(ELFScript);
	return script;
}
bool ELFScriptLanguage::_has_named_classes() const {
	return true;
}
bool ELFScriptLanguage::_supports_builtin_mode() const {
	return true;
}
bool ELFScriptLanguage::_supports_documentation() const {
	return false;
}
bool ELFScriptLanguage::_can_inherit_from_file() const {
	return false;
}
bool ELFScriptLanguage::_can_make_function() const {
	return false;
}
int32_t ELFScriptLanguage::_find_function(const String &p_function, const String &p_code) const {
	return -1;
}
String ELFScriptLanguage::_make_function(const String &p_class_name, const String &p_function_name, const PackedStringArray &p_function_args) const {
	return String();
}
Error ELFScriptLanguage::_open_in_external_editor(const Ref<Script> &p_script, int32_t p_line, int32_t p_column) {
	return Error::OK;
}
bool ELFScriptLanguage::_overrides_external_editor() {
	return false;
}
Dictionary ELFScriptLanguage::_complete_code(const String &p_code, const String &p_path, Object *p_owner) const {
	return Dictionary();
}
Dictionary ELFScriptLanguage::_lookup_code(const String &p_code, const String &p_symbol, const String &p_path, Object *p_owner) const {
	return Dictionary();
}
String ELFScriptLanguage::_auto_indent_code(const String &p_code, int32_t p_from_line, int32_t p_to_line) const {
	return String();
}
void ELFScriptLanguage::_add_global_constant(const StringName &p_name, const Variant &p_value) {}
void ELFScriptLanguage::_add_named_global_constant(const StringName &p_name, const Variant &p_value) {}
void ELFScriptLanguage::_remove_named_global_constant(const StringName &p_name) {}
void ELFScriptLanguage::_thread_enter() {}
void ELFScriptLanguage::_thread_exit() {}
String ELFScriptLanguage::_debug_get_error() const {
	return String();
}
int32_t ELFScriptLanguage::_debug_get_stack_level_count() const {
	return 0;
}
int32_t ELFScriptLanguage::_debug_get_stack_level_line(int32_t p_level) const {
	return 0;
}
String ELFScriptLanguage::_debug_get_stack_level_function(int32_t p_level) const {
	return String();
}
Dictionary ELFScriptLanguage::_debug_get_stack_level_locals(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}
Dictionary ELFScriptLanguage::_debug_get_stack_level_members(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}
void *ELFScriptLanguage::_debug_get_stack_level_instance(int32_t p_level) {
	return nullptr;
}
Dictionary ELFScriptLanguage::_debug_get_globals(int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}
String ELFScriptLanguage::_debug_parse_stack_level_expression(int32_t p_level, const String &p_expression, int32_t p_max_subitems, int32_t p_max_depth) {
	return String();
}
TypedArray<Dictionary> ELFScriptLanguage::_debug_get_current_stack_info() {
	return TypedArray<Dictionary>();
}
void ELFScriptLanguage::_reload_all_scripts() {}
void ELFScriptLanguage::_reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {}
PackedStringArray ELFScriptLanguage::_get_recognized_extensions() const {
	PackedStringArray array;
	array.push_back("elf");
	return array;
}
TypedArray<Dictionary> ELFScriptLanguage::_get_public_functions() const {
	return TypedArray<Dictionary>();
}
Dictionary ELFScriptLanguage::_get_public_constants() const {
	return Dictionary();
}
TypedArray<Dictionary> ELFScriptLanguage::_get_public_annotations() const {
	return TypedArray<Dictionary>();
}
void ELFScriptLanguage::_profiling_start() {}
void ELFScriptLanguage::_profiling_stop() {}
int32_t ELFScriptLanguage::_profiling_get_accumulated_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max) {
	return 0;
}
int32_t ELFScriptLanguage::_profiling_get_frame_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max) {
	return 0;
}
void ELFScriptLanguage::_frame() {
	static bool icon_registered = register_language_icons;
	if (!icon_registered && Engine::get_singleton()->is_editor_hint()) {
		icon_registered = true;
		// Manually register ELFScript icon
		load_icon();
		// Register theme callback
		EditorInterface::get_singleton()->get_base_control()->connect("theme_changed", callable_mp(this, &ELFScriptLanguage::load_icon));
	}
}
void ELFScriptLanguage::load_icon() {
	static bool reenter = false;
	if (reenter)
		return;
	reenter = true;
	if (Engine::get_singleton()->is_editor_hint() && FileAccess::file_exists(icon_path)) {
		Ref<Theme> editor_theme = EditorInterface::get_singleton()->get_editor_theme();
		if (editor_theme.is_valid() && !editor_theme->has_icon("ELFScript", "EditorIcons")) {
			ResourceLoader *resource_loader = ResourceLoader::get_singleton();
			Ref<Texture2D> tex = resource_loader->load(icon_path);
			editor_theme->set_icon("ELFScript", "EditorIcons", tex);
		}
	}
	reenter = false;
}
bool ELFScriptLanguage::_handles_global_class_type(const String &p_type) const {
	return p_type == "ELFScript" || p_type == "Sandbox";
}
Dictionary ELFScriptLanguage::_get_global_class_name(const String &p_path) const {
	Ref<Resource> resource = ResourceLoader::get_singleton()->load(p_path);
	Ref<ELFScript> elf_model = Object::cast_to<ELFScript>(resource.ptr());
	Dictionary dict;
	if (elf_model.is_valid()) {
		dict["name"] = elf_model->_get_global_name();
		dict["base_type"] = "Sandbox";
		dict["icon_path"] = String("res://addons/godot_sandbox/Sandbox.svg");
	}
	return dict;
}
