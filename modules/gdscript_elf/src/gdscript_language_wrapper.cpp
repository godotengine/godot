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

// Macro-based forwarding for pass-through methods
#define FORWARD_V(ret, name, ...) \
	ret GDScriptLanguageWrapper::name(__VA_ARGS__) const { \
		ERR_FAIL_NULL_V(original_language, ret{}); \
		return original_language->name(__VA_ARGS__); \
	}

#define FORWARD_VOID(name, ...) \
	void GDScriptLanguageWrapper::name(__VA_ARGS__) { \
		ERR_FAIL_NULL(original_language); \
		original_language->name(__VA_ARGS__); \
	}

#define FORWARD_NV(ret, name, ...) \
	ret GDScriptLanguageWrapper::name(__VA_ARGS__) { \
		ERR_FAIL_NULL_V(original_language, ret{}); \
		return original_language->name(__VA_ARGS__); \
	}

void GDScriptLanguageWrapper::_bind_methods() {
	// No methods to bind - this is a pure wrapper
}

GDScriptLanguageWrapper::GDScriptLanguageWrapper() {
	original_language = nullptr;
}

GDScriptLanguageWrapper::~GDScriptLanguageWrapper() {
	original_language = nullptr;
}

void GDScriptLanguageWrapper::set_original_language(GDScriptLanguage *p_original) {
	original_language = p_original;
}

// ScriptLanguage interface - all methods delegate to original
FORWARD_V(String, get_name)
FORWARD_VOID(init)
FORWARD_V(String, get_type)
FORWARD_V(String, get_extension)
FORWARD_VOID(finish)
FORWARD_V(Vector<String>, get_reserved_words)
FORWARD_V(bool, is_control_flow_keyword, const String &p_string)
FORWARD_V(Vector<String>, get_comment_delimiters)
FORWARD_V(Vector<String>, get_doc_comment_delimiters)
FORWARD_V(Vector<String>, get_string_delimiters)
FORWARD_V(Ref<Script>, make_template, const String &p_template, const String &p_class_name, const String &p_base_class_name)
FORWARD_NV(Vector<ScriptLanguage::ScriptTemplate>, get_built_in_templates, const StringName &p_object)
FORWARD_NV(bool, is_using_templates)
FORWARD_V(bool, validate, const String &p_script, const String &p_path, List<String> *r_functions, List<ScriptError> *r_errors, List<Warning> *r_warnings, HashSet<int> *r_safe_lines)
FORWARD_V(String, validate_path, const String &p_path)

Script *GDScriptLanguageWrapper::create_script() const {
	ERR_FAIL_NULL_V(original_language, nullptr);
	Script *original = original_language->create_script();
	if (!original) {
		return nullptr;
	}
	GDScript *gdscript = Object::cast_to<GDScript>(original);
	if (!gdscript) {
		return original;
	}
	GDScriptWrapper *wrapper = memnew(GDScriptWrapper);
	wrapper->set_original_script(Ref<GDScript>(gdscript));
	wrapper->set_path(original->get_path());
	return wrapper;
}

FORWARD_V(bool, supports_builtin_mode)
FORWARD_V(bool, supports_documentation)
FORWARD_V(bool, can_inherit_from_file)
FORWARD_V(int, find_function, const String &p_function, const String &p_code)
FORWARD_V(String, make_function, const String &p_class, const String &p_name, const PackedStringArray &p_args)
FORWARD_V(bool, can_make_function)
FORWARD_NV(Error, open_in_external_editor, const Ref<Script> &p_script, int p_line, int p_col)
FORWARD_NV(bool, overrides_external_editor)
FORWARD_V(ScriptLanguage::ScriptNameCasing, preferred_file_name_casing)
FORWARD_NV(Error, complete_code, const String &p_code, const String &p_path, Object *p_owner, List<CodeCompletionOption> *r_options, bool &r_force, String &r_call_hint)

#ifdef TOOLS_ENABLED
FORWARD_NV(Error, lookup_code, const String &p_code, const String &p_symbol, const String &p_path, Object *p_owner, LookupResult &r_result)
#endif

FORWARD_VOID(auto_indent_code, String &p_code, int p_from_line, int p_to_line)
FORWARD_VOID(add_global_constant, const StringName &p_variable, const Variant &p_value)
FORWARD_VOID(add_named_global_constant, const StringName &p_name, const Variant &p_value)
FORWARD_VOID(remove_named_global_constant, const StringName &p_name)
FORWARD_VOID(thread_enter)
FORWARD_VOID(thread_exit)
FORWARD_V(String, debug_get_error)
FORWARD_V(int, debug_get_stack_level_count)
FORWARD_V(int, debug_get_stack_level_line, int p_level)
FORWARD_V(String, debug_get_stack_level_function, int p_level)
FORWARD_V(String, debug_get_stack_level_source, int p_level)
FORWARD_VOID(debug_get_stack_level_locals, int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth)
FORWARD_VOID(debug_get_stack_level_members, int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth)
FORWARD_NV(ScriptInstance *, debug_get_stack_level_instance, int p_level)
FORWARD_VOID(debug_get_globals, List<String> *p_globals, List<Variant> *p_values, int p_max_subitems, int p_max_depth)
FORWARD_VOID(get_recognized_extensions, List<String> *p_extensions)
FORWARD_VOID(get_public_functions, List<MethodInfo> *p_functions)
FORWARD_VOID(get_public_constants, List<Pair<String, Variant>> *p_constants)
FORWARD_VOID(get_public_annotations, List<MethodInfo> *p_annotations)
FORWARD_VOID(profiling_set_save_native_calls, bool p_enable)
FORWARD_VOID(frame)
FORWARD_VOID(reload_all_scripts)
FORWARD_VOID(reload_scripts, const Array &p_scripts, bool p_soft_reload)
FORWARD_VOID(reload_tool_script, const Ref<Script> &p_script, bool p_soft_reload)
FORWARD_V(bool, handles_global_class_type, const String &p_type)
FORWARD_V(String, get_global_class_name, const String &p_path, String *r_base_type, String *r_icon_path, bool *r_is_abstract, bool *r_is_tool)
FORWARD_NV(Vector<ScriptLanguage::StackInfo>, debug_get_current_stack_info)
FORWARD_NV(String, debug_parse_stack_level_expression, int p_level, const String &p_expression, int p_max_subitems, int p_max_depth)
FORWARD_VOID(profiling_start)
FORWARD_VOID(profiling_stop)
FORWARD_NV(int, profiling_get_accumulated_data, ProfilingInfo *p_info_arr, int p_info_max)
FORWARD_NV(int, profiling_get_frame_data, ProfilingInfo *p_info_arr, int p_info_max)

#undef FORWARD_V
#undef FORWARD_VOID
#undef FORWARD_NV
