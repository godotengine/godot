#include "script_language_zig.h"
#include "../script_language_common.h"
#include "script_zig.h"
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/theme.hpp>
#include <string>
#include <unordered_set>
static constexpr const char *icon_path = "res://addons/godot_sandbox/ZigScript.svg";

static ZigScriptLanguage *zig_language;

void ZigScriptLanguage::init() {
	zig_language = memnew(ZigScriptLanguage);
	Engine::get_singleton()->register_script_language(zig_language);
}
void ZigScriptLanguage::deinit() {
	if (zig_language) {
		Engine::get_singleton()->unregister_script_language(zig_language);
		memdelete(zig_language);
		zig_language = nullptr;
	}
}

ZigScriptLanguage *ZigScriptLanguage::get_singleton() {
	return zig_language;
}

String ZigScriptLanguage::_get_name() const {
	return "ZigScript";
}
void ZigScriptLanguage::_init() {}
String ZigScriptLanguage::_get_type() const {
	return "ZigScript";
}
String ZigScriptLanguage::_get_extension() const {
	return "zig";
}
void ZigScriptLanguage::_finish() {}
PackedStringArray ZigScriptLanguage::_get_reserved_words() const {
	static const PackedStringArray reserved_words{
		"addrspace", "align", "and", "asm", "async", "await", "break", "catch", "comptime", "const", "continue", "defer", "else", "enum", "errdefer", "error", "export", "extern", "for", "if", "inline", "noalias", "noinline", "nosuspend", "opaque", "or", "orelse", "packed", "anyframe", "pub", "resume", "return", "linksection", "callconv", "struct", "suspend", "switch", "test", "threadlocal", "try", "union", "unreachable", "usingnamespace", "var", "volatile", "allowzero", "while", "anytype", "fn"
	};
	return reserved_words;
}
bool ZigScriptLanguage::_is_control_flow_keyword(const String &p_keyword) const {
	static const std::unordered_set<std::string> control_flow_keywords{
		"if", "else", "switch", "case", "default", "while", "loop", "for", "break", "continue", "return", "goto", "resume", "suspend", "defer", "errdefer", "try", "catch", "async", "await"
	};
	return control_flow_keywords.find(p_keyword.utf8().get_data()) != control_flow_keywords.end();
}
PackedStringArray ZigScriptLanguage::_get_comment_delimiters() const {
	PackedStringArray comment_delimiters;
	comment_delimiters.push_back("/* */");
	comment_delimiters.push_back("//");
	return comment_delimiters;
}
PackedStringArray ZigScriptLanguage::_get_doc_comment_delimiters() const {
	PackedStringArray doc_comment_delimiters;
	doc_comment_delimiters.push_back("///");
	doc_comment_delimiters.push_back("/** */");
	return doc_comment_delimiters;
}
PackedStringArray ZigScriptLanguage::_get_string_delimiters() const {
	PackedStringArray string_delimiters;
	string_delimiters.push_back("' '");
	string_delimiters.push_back("\" \"");
	return string_delimiters;
}
Ref<Script> ZigScriptLanguage::_make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const {
	ZigScript *script = memnew(ZigScript);
	return Ref<Script>(script);
}
TypedArray<Dictionary> ZigScriptLanguage::_get_built_in_templates(const StringName &p_object) const {
	return TypedArray<Dictionary>();
}
bool ZigScriptLanguage::_is_using_templates() {
	return false;
}
Dictionary ZigScriptLanguage::_validate(const String &p_script, const String &p_path, bool p_validate_functions, bool p_validate_errors, bool p_validate_warnings, bool p_validate_safe_lines) const {
	return Dictionary();
}
String ZigScriptLanguage::_validate_path(const String &p_path) const {
	return String();
}
Object *ZigScriptLanguage::_create_script() const {
	ZigScript *script = memnew(ZigScript);
	return script;
}
bool ZigScriptLanguage::_has_named_classes() const {
	return false;
}
bool ZigScriptLanguage::_supports_builtin_mode() const {
	return false;
}
bool ZigScriptLanguage::_supports_documentation() const {
	return false;
}
bool ZigScriptLanguage::_can_inherit_from_file() const {
	return false;
}
int32_t ZigScriptLanguage::_find_function(const String &p_function, const String &p_code) const {
	return -1;
}
String ZigScriptLanguage::_make_function(const String &p_class_name, const String &p_function_name, const PackedStringArray &p_function_args) const {
	return String();
}
Error ZigScriptLanguage::_open_in_external_editor(const Ref<Script> &p_script, int32_t p_line, int32_t p_column) {
	return Error::OK;
}
bool ZigScriptLanguage::_overrides_external_editor() {
	return false;
}
Dictionary ZigScriptLanguage::_complete_code(const String &p_code, const String &p_path, Object *p_owner) const {
	return Dictionary();
}
Dictionary ZigScriptLanguage::_lookup_code(const String &p_code, const String &p_symbol, const String &p_path, Object *p_owner) const {
	return Dictionary();
}
String ZigScriptLanguage::_auto_indent_code(const String &p_code, int32_t p_from_line, int32_t p_to_line) const {
	return String();
}
void ZigScriptLanguage::_add_global_constant(const StringName &p_name, const Variant &p_value) {}
void ZigScriptLanguage::_add_named_global_constant(const StringName &p_name, const Variant &p_value) {}
void ZigScriptLanguage::_remove_named_global_constant(const StringName &p_name) {}
void ZigScriptLanguage::_thread_enter() {}
void ZigScriptLanguage::_thread_exit() {}
String ZigScriptLanguage::_debug_get_error() const {
	return String();
}
int32_t ZigScriptLanguage::_debug_get_stack_level_count() const {
	return 0;
}
int32_t ZigScriptLanguage::_debug_get_stack_level_line(int32_t p_level) const {
	return 0;
}
String ZigScriptLanguage::_debug_get_stack_level_function(int32_t p_level) const {
	return String();
}
Dictionary ZigScriptLanguage::_debug_get_stack_level_locals(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}
Dictionary ZigScriptLanguage::_debug_get_stack_level_members(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}
void *ZigScriptLanguage::_debug_get_stack_level_instance(int32_t p_level) {
	return nullptr;
}
Dictionary ZigScriptLanguage::_debug_get_globals(int32_t p_max_subitems, int32_t p_max_depth) {
	return Dictionary();
}
String ZigScriptLanguage::_debug_parse_stack_level_expression(int32_t p_level, const String &p_expression, int32_t p_max_subitems, int32_t p_max_depth) {
	return String();
}
TypedArray<Dictionary> ZigScriptLanguage::_debug_get_current_stack_info() {
	return TypedArray<Dictionary>();
}
void ZigScriptLanguage::_reload_all_scripts() {}
void ZigScriptLanguage::_reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {}
PackedStringArray ZigScriptLanguage::_get_recognized_extensions() const {
	PackedStringArray array;
	array.push_back("zig");
	return array;
}
TypedArray<Dictionary> ZigScriptLanguage::_get_public_functions() const {
	return TypedArray<Dictionary>();
}
Dictionary ZigScriptLanguage::_get_public_constants() const {
	return Dictionary();
}
TypedArray<Dictionary> ZigScriptLanguage::_get_public_annotations() const {
	return TypedArray<Dictionary>();
}
void ZigScriptLanguage::_profiling_start() {}
void ZigScriptLanguage::_profiling_stop() {}
int32_t ZigScriptLanguage::_profiling_get_accumulated_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max) {
	return 0;
}
int32_t ZigScriptLanguage::_profiling_get_frame_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max) {
	return 0;
}
void ZigScriptLanguage::_frame() {
	static bool icon_registered = register_language_icons;
	if (!icon_registered && Engine::get_singleton()->is_editor_hint()) {
		icon_registered = true;
		// Manually register language icon
		load_icon();
		// Register theme callback
		EditorInterface::get_singleton()->get_base_control()->connect("theme_changed", callable_mp(this, &ZigScriptLanguage::load_icon));
	}
}
void ZigScriptLanguage::load_icon() {
	static bool reenter = false;
	if (reenter)
		return;
	reenter = true;
	if (Engine::get_singleton()->is_editor_hint() && FileAccess::file_exists(icon_path)) {
		Ref<Theme> editor_theme = EditorInterface::get_singleton()->get_editor_theme();
		if (editor_theme.is_valid() && !editor_theme->has_icon("ZigScript", "EditorIcons")) {
			ResourceLoader *resource_loader = ResourceLoader::get_singleton();
			Ref<Texture2D> tex = resource_loader->load(icon_path);
			editor_theme->set_icon("ZigScript", "EditorIcons", tex);
		}
	}
	reenter = false;
}
bool ZigScriptLanguage::_handles_global_class_type(const String &p_type) const {
	return p_type == "ZigScript";
}
Dictionary ZigScriptLanguage::_get_global_class_name(const String &p_path) const {
	return Dictionary();
}
