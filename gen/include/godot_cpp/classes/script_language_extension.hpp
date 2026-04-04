/**************************************************************************/
/*  script_language_extension.hpp                                         */
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
#include <godot_cpp/classes/script_language.hpp>
#include <godot_cpp/classes/script_language_extension_profiling_info.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Array;
class Object;
class Script;
class StringName;
class Variant;

class ScriptLanguageExtension : public ScriptLanguage {
	GDEXTENSION_CLASS(ScriptLanguageExtension, ScriptLanguage)

public:
	enum LookupResultType {
		LOOKUP_RESULT_SCRIPT_LOCATION = 0,
		LOOKUP_RESULT_CLASS = 1,
		LOOKUP_RESULT_CLASS_CONSTANT = 2,
		LOOKUP_RESULT_CLASS_PROPERTY = 3,
		LOOKUP_RESULT_CLASS_METHOD = 4,
		LOOKUP_RESULT_CLASS_SIGNAL = 5,
		LOOKUP_RESULT_CLASS_ENUM = 6,
		LOOKUP_RESULT_CLASS_TBD_GLOBALSCOPE = 7,
		LOOKUP_RESULT_CLASS_ANNOTATION = 8,
		LOOKUP_RESULT_LOCAL_CONSTANT = 9,
		LOOKUP_RESULT_LOCAL_VARIABLE = 10,
		LOOKUP_RESULT_MAX = 11,
	};

	enum CodeCompletionLocation {
		LOCATION_LOCAL = 0,
		LOCATION_PARENT_MASK = 256,
		LOCATION_OTHER_USER_CODE = 512,
		LOCATION_OTHER = 1024,
	};

	enum CodeCompletionKind {
		CODE_COMPLETION_KIND_CLASS = 0,
		CODE_COMPLETION_KIND_FUNCTION = 1,
		CODE_COMPLETION_KIND_SIGNAL = 2,
		CODE_COMPLETION_KIND_VARIABLE = 3,
		CODE_COMPLETION_KIND_MEMBER = 4,
		CODE_COMPLETION_KIND_ENUM = 5,
		CODE_COMPLETION_KIND_CONSTANT = 6,
		CODE_COMPLETION_KIND_NODE_PATH = 7,
		CODE_COMPLETION_KIND_FILE_PATH = 8,
		CODE_COMPLETION_KIND_PLAIN_TEXT = 9,
		CODE_COMPLETION_KIND_MAX = 10,
	};

	virtual String _get_name() const;
	virtual void _init();
	virtual String _get_type() const;
	virtual String _get_extension() const;
	virtual void _finish();
	virtual PackedStringArray _get_reserved_words() const;
	virtual bool _is_control_flow_keyword(const String &p_keyword) const;
	virtual PackedStringArray _get_comment_delimiters() const;
	virtual PackedStringArray _get_doc_comment_delimiters() const;
	virtual PackedStringArray _get_string_delimiters() const;
	virtual Ref<Script> _make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const;
	virtual TypedArray<Dictionary> _get_built_in_templates(const StringName &p_object) const;
	virtual bool _is_using_templates();
	virtual Dictionary _validate(const String &p_script, const String &p_path, bool p_validate_functions, bool p_validate_errors, bool p_validate_warnings, bool p_validate_safe_lines) const;
	virtual String _validate_path(const String &p_path) const;
	virtual Object *_create_script() const;
	virtual bool _has_named_classes() const;
	virtual bool _supports_builtin_mode() const;
	virtual bool _supports_documentation() const;
	virtual bool _can_inherit_from_file() const;
	virtual int32_t _find_function(const String &p_function, const String &p_code) const;
	virtual String _make_function(const String &p_class_name, const String &p_function_name, const PackedStringArray &p_function_args) const;
	virtual bool _can_make_function() const;
	virtual Error _open_in_external_editor(const Ref<Script> &p_script, int32_t p_line, int32_t p_column);
	virtual bool _overrides_external_editor();
	virtual ScriptLanguage::ScriptNameCasing _preferred_file_name_casing() const;
	virtual Dictionary _complete_code(const String &p_code, const String &p_path, Object *p_owner) const;
	virtual Dictionary _lookup_code(const String &p_code, const String &p_symbol, const String &p_path, Object *p_owner) const;
	virtual String _auto_indent_code(const String &p_code, int32_t p_from_line, int32_t p_to_line) const;
	virtual void _add_global_constant(const StringName &p_name, const Variant &p_value);
	virtual void _add_named_global_constant(const StringName &p_name, const Variant &p_value);
	virtual void _remove_named_global_constant(const StringName &p_name);
	virtual void _thread_enter();
	virtual void _thread_exit();
	virtual String _debug_get_error() const;
	virtual int32_t _debug_get_stack_level_count() const;
	virtual int32_t _debug_get_stack_level_line(int32_t p_level) const;
	virtual String _debug_get_stack_level_function(int32_t p_level) const;
	virtual String _debug_get_stack_level_source(int32_t p_level) const;
	virtual Dictionary _debug_get_stack_level_locals(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth);
	virtual Dictionary _debug_get_stack_level_members(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth);
	virtual void *_debug_get_stack_level_instance(int32_t p_level);
	virtual Dictionary _debug_get_globals(int32_t p_max_subitems, int32_t p_max_depth);
	virtual String _debug_parse_stack_level_expression(int32_t p_level, const String &p_expression, int32_t p_max_subitems, int32_t p_max_depth);
	virtual TypedArray<Dictionary> _debug_get_current_stack_info();
	virtual void _reload_all_scripts();
	virtual void _reload_scripts(const Array &p_scripts, bool p_soft_reload);
	virtual void _reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload);
	virtual PackedStringArray _get_recognized_extensions() const;
	virtual TypedArray<Dictionary> _get_public_functions() const;
	virtual Dictionary _get_public_constants() const;
	virtual TypedArray<Dictionary> _get_public_annotations() const;
	virtual void _profiling_start();
	virtual void _profiling_stop();
	virtual void _profiling_set_save_native_calls(bool p_enable);
	virtual int32_t _profiling_get_accumulated_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max);
	virtual int32_t _profiling_get_frame_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max);
	virtual void _frame();
	virtual bool _handles_global_class_type(const String &p_type) const;
	virtual Dictionary _get_global_class_name(const String &p_path) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		ScriptLanguage::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_name), decltype(&T::_get_name)>) {
			BIND_VIRTUAL_METHOD(T, _get_name, 201670096);
		}
		if constexpr (!std::is_same_v<decltype(&B::_init), decltype(&T::_init)>) {
			BIND_VIRTUAL_METHOD(T, _init, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_type), decltype(&T::_get_type)>) {
			BIND_VIRTUAL_METHOD(T, _get_type, 201670096);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_extension), decltype(&T::_get_extension)>) {
			BIND_VIRTUAL_METHOD(T, _get_extension, 201670096);
		}
		if constexpr (!std::is_same_v<decltype(&B::_finish), decltype(&T::_finish)>) {
			BIND_VIRTUAL_METHOD(T, _finish, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_reserved_words), decltype(&T::_get_reserved_words)>) {
			BIND_VIRTUAL_METHOD(T, _get_reserved_words, 1139954409);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_control_flow_keyword), decltype(&T::_is_control_flow_keyword)>) {
			BIND_VIRTUAL_METHOD(T, _is_control_flow_keyword, 3927539163);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_comment_delimiters), decltype(&T::_get_comment_delimiters)>) {
			BIND_VIRTUAL_METHOD(T, _get_comment_delimiters, 1139954409);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_doc_comment_delimiters), decltype(&T::_get_doc_comment_delimiters)>) {
			BIND_VIRTUAL_METHOD(T, _get_doc_comment_delimiters, 1139954409);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_string_delimiters), decltype(&T::_get_string_delimiters)>) {
			BIND_VIRTUAL_METHOD(T, _get_string_delimiters, 1139954409);
		}
		if constexpr (!std::is_same_v<decltype(&B::_make_template), decltype(&T::_make_template)>) {
			BIND_VIRTUAL_METHOD(T, _make_template, 3583744548);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_built_in_templates), decltype(&T::_get_built_in_templates)>) {
			BIND_VIRTUAL_METHOD(T, _get_built_in_templates, 3147814860);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_using_templates), decltype(&T::_is_using_templates)>) {
			BIND_VIRTUAL_METHOD(T, _is_using_templates, 2240911060);
		}
		if constexpr (!std::is_same_v<decltype(&B::_validate), decltype(&T::_validate)>) {
			BIND_VIRTUAL_METHOD(T, _validate, 1697887509);
		}
		if constexpr (!std::is_same_v<decltype(&B::_validate_path), decltype(&T::_validate_path)>) {
			BIND_VIRTUAL_METHOD(T, _validate_path, 3135753539);
		}
		if constexpr (!std::is_same_v<decltype(&B::_create_script), decltype(&T::_create_script)>) {
			BIND_VIRTUAL_METHOD(T, _create_script, 1981248198);
		}
		if constexpr (!std::is_same_v<decltype(&B::_has_named_classes), decltype(&T::_has_named_classes)>) {
			BIND_VIRTUAL_METHOD(T, _has_named_classes, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_supports_builtin_mode), decltype(&T::_supports_builtin_mode)>) {
			BIND_VIRTUAL_METHOD(T, _supports_builtin_mode, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_supports_documentation), decltype(&T::_supports_documentation)>) {
			BIND_VIRTUAL_METHOD(T, _supports_documentation, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_can_inherit_from_file), decltype(&T::_can_inherit_from_file)>) {
			BIND_VIRTUAL_METHOD(T, _can_inherit_from_file, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_find_function), decltype(&T::_find_function)>) {
			BIND_VIRTUAL_METHOD(T, _find_function, 2878152881);
		}
		if constexpr (!std::is_same_v<decltype(&B::_make_function), decltype(&T::_make_function)>) {
			BIND_VIRTUAL_METHOD(T, _make_function, 1243061914);
		}
		if constexpr (!std::is_same_v<decltype(&B::_can_make_function), decltype(&T::_can_make_function)>) {
			BIND_VIRTUAL_METHOD(T, _can_make_function, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_open_in_external_editor), decltype(&T::_open_in_external_editor)>) {
			BIND_VIRTUAL_METHOD(T, _open_in_external_editor, 552845695);
		}
		if constexpr (!std::is_same_v<decltype(&B::_overrides_external_editor), decltype(&T::_overrides_external_editor)>) {
			BIND_VIRTUAL_METHOD(T, _overrides_external_editor, 2240911060);
		}
		if constexpr (!std::is_same_v<decltype(&B::_preferred_file_name_casing), decltype(&T::_preferred_file_name_casing)>) {
			BIND_VIRTUAL_METHOD(T, _preferred_file_name_casing, 2969522789);
		}
		if constexpr (!std::is_same_v<decltype(&B::_complete_code), decltype(&T::_complete_code)>) {
			BIND_VIRTUAL_METHOD(T, _complete_code, 950756616);
		}
		if constexpr (!std::is_same_v<decltype(&B::_lookup_code), decltype(&T::_lookup_code)>) {
			BIND_VIRTUAL_METHOD(T, _lookup_code, 3143837309);
		}
		if constexpr (!std::is_same_v<decltype(&B::_auto_indent_code), decltype(&T::_auto_indent_code)>) {
			BIND_VIRTUAL_METHOD(T, _auto_indent_code, 2531480354);
		}
		if constexpr (!std::is_same_v<decltype(&B::_add_global_constant), decltype(&T::_add_global_constant)>) {
			BIND_VIRTUAL_METHOD(T, _add_global_constant, 3776071444);
		}
		if constexpr (!std::is_same_v<decltype(&B::_add_named_global_constant), decltype(&T::_add_named_global_constant)>) {
			BIND_VIRTUAL_METHOD(T, _add_named_global_constant, 3776071444);
		}
		if constexpr (!std::is_same_v<decltype(&B::_remove_named_global_constant), decltype(&T::_remove_named_global_constant)>) {
			BIND_VIRTUAL_METHOD(T, _remove_named_global_constant, 3304788590);
		}
		if constexpr (!std::is_same_v<decltype(&B::_thread_enter), decltype(&T::_thread_enter)>) {
			BIND_VIRTUAL_METHOD(T, _thread_enter, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_thread_exit), decltype(&T::_thread_exit)>) {
			BIND_VIRTUAL_METHOD(T, _thread_exit, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_debug_get_error), decltype(&T::_debug_get_error)>) {
			BIND_VIRTUAL_METHOD(T, _debug_get_error, 201670096);
		}
		if constexpr (!std::is_same_v<decltype(&B::_debug_get_stack_level_count), decltype(&T::_debug_get_stack_level_count)>) {
			BIND_VIRTUAL_METHOD(T, _debug_get_stack_level_count, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_debug_get_stack_level_line), decltype(&T::_debug_get_stack_level_line)>) {
			BIND_VIRTUAL_METHOD(T, _debug_get_stack_level_line, 923996154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_debug_get_stack_level_function), decltype(&T::_debug_get_stack_level_function)>) {
			BIND_VIRTUAL_METHOD(T, _debug_get_stack_level_function, 844755477);
		}
		if constexpr (!std::is_same_v<decltype(&B::_debug_get_stack_level_source), decltype(&T::_debug_get_stack_level_source)>) {
			BIND_VIRTUAL_METHOD(T, _debug_get_stack_level_source, 844755477);
		}
		if constexpr (!std::is_same_v<decltype(&B::_debug_get_stack_level_locals), decltype(&T::_debug_get_stack_level_locals)>) {
			BIND_VIRTUAL_METHOD(T, _debug_get_stack_level_locals, 335235777);
		}
		if constexpr (!std::is_same_v<decltype(&B::_debug_get_stack_level_members), decltype(&T::_debug_get_stack_level_members)>) {
			BIND_VIRTUAL_METHOD(T, _debug_get_stack_level_members, 335235777);
		}
		if constexpr (!std::is_same_v<decltype(&B::_debug_get_stack_level_instance), decltype(&T::_debug_get_stack_level_instance)>) {
			BIND_VIRTUAL_METHOD(T, _debug_get_stack_level_instance, 3744713108);
		}
		if constexpr (!std::is_same_v<decltype(&B::_debug_get_globals), decltype(&T::_debug_get_globals)>) {
			BIND_VIRTUAL_METHOD(T, _debug_get_globals, 4123630098);
		}
		if constexpr (!std::is_same_v<decltype(&B::_debug_parse_stack_level_expression), decltype(&T::_debug_parse_stack_level_expression)>) {
			BIND_VIRTUAL_METHOD(T, _debug_parse_stack_level_expression, 1135811067);
		}
		if constexpr (!std::is_same_v<decltype(&B::_debug_get_current_stack_info), decltype(&T::_debug_get_current_stack_info)>) {
			BIND_VIRTUAL_METHOD(T, _debug_get_current_stack_info, 2915620761);
		}
		if constexpr (!std::is_same_v<decltype(&B::_reload_all_scripts), decltype(&T::_reload_all_scripts)>) {
			BIND_VIRTUAL_METHOD(T, _reload_all_scripts, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_reload_scripts), decltype(&T::_reload_scripts)>) {
			BIND_VIRTUAL_METHOD(T, _reload_scripts, 3156113851);
		}
		if constexpr (!std::is_same_v<decltype(&B::_reload_tool_script), decltype(&T::_reload_tool_script)>) {
			BIND_VIRTUAL_METHOD(T, _reload_tool_script, 1957307671);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_recognized_extensions), decltype(&T::_get_recognized_extensions)>) {
			BIND_VIRTUAL_METHOD(T, _get_recognized_extensions, 1139954409);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_public_functions), decltype(&T::_get_public_functions)>) {
			BIND_VIRTUAL_METHOD(T, _get_public_functions, 3995934104);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_public_constants), decltype(&T::_get_public_constants)>) {
			BIND_VIRTUAL_METHOD(T, _get_public_constants, 3102165223);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_public_annotations), decltype(&T::_get_public_annotations)>) {
			BIND_VIRTUAL_METHOD(T, _get_public_annotations, 3995934104);
		}
		if constexpr (!std::is_same_v<decltype(&B::_profiling_start), decltype(&T::_profiling_start)>) {
			BIND_VIRTUAL_METHOD(T, _profiling_start, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_profiling_stop), decltype(&T::_profiling_stop)>) {
			BIND_VIRTUAL_METHOD(T, _profiling_stop, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_profiling_set_save_native_calls), decltype(&T::_profiling_set_save_native_calls)>) {
			BIND_VIRTUAL_METHOD(T, _profiling_set_save_native_calls, 2586408642);
		}
		if constexpr (!std::is_same_v<decltype(&B::_profiling_get_accumulated_data), decltype(&T::_profiling_get_accumulated_data)>) {
			BIND_VIRTUAL_METHOD(T, _profiling_get_accumulated_data, 50157827);
		}
		if constexpr (!std::is_same_v<decltype(&B::_profiling_get_frame_data), decltype(&T::_profiling_get_frame_data)>) {
			BIND_VIRTUAL_METHOD(T, _profiling_get_frame_data, 50157827);
		}
		if constexpr (!std::is_same_v<decltype(&B::_frame), decltype(&T::_frame)>) {
			BIND_VIRTUAL_METHOD(T, _frame, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_handles_global_class_type), decltype(&T::_handles_global_class_type)>) {
			BIND_VIRTUAL_METHOD(T, _handles_global_class_type, 3927539163);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_global_class_name), decltype(&T::_get_global_class_name)>) {
			BIND_VIRTUAL_METHOD(T, _get_global_class_name, 2248993622);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(ScriptLanguageExtension::LookupResultType);
VARIANT_ENUM_CAST(ScriptLanguageExtension::CodeCompletionLocation);
VARIANT_ENUM_CAST(ScriptLanguageExtension::CodeCompletionKind);

