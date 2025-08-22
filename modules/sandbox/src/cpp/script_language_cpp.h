/**************************************************************************/
/*  script_language_cpp.h                                                 */
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

#pragma once

#include <godot_cpp/classes/script.hpp>
#include "core/object/script_language.h"


class CPPScriptLanguage : public ScriptLanguageExtension {
	GDCLASS(CPPScriptLanguage, ScriptLanguageExtension);

protected:
	static void _bind_methods() {}
	String file;

public:
	static void init();
	static void deinit();
	static CPPScriptLanguage *get_singleton();

	virtual String _get_name() const override;
	virtual void _init() override;
	virtual String _get_type() const override;
	virtual String _get_extension() const override;
	virtual void _finish() override;
	virtual PackedStringArray _get_reserved_words() const override;
	virtual bool _is_control_flow_keyword(const String &p_keyword) const override;
	virtual PackedStringArray _get_comment_delimiters() const override;
	virtual PackedStringArray _get_doc_comment_delimiters() const override;
	virtual PackedStringArray _get_string_delimiters() const override;
	virtual Ref<Script> _make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const override;
	virtual TypedArray<Dictionary> _get_built_in_templates(const StringName &p_object) const override;
	virtual bool _is_using_templates() override;
	virtual Dictionary _validate(const String &p_script, const String &p_path, bool p_validate_functions, bool p_validate_errors, bool p_validate_warnings, bool p_validate_safe_lines) const override;
	virtual String _validate_path(const String &p_path) const override;
	virtual Object *_create_script() const override;
	virtual bool _has_named_classes() const override;
	virtual bool _supports_builtin_mode() const override;
	virtual bool _supports_documentation() const override;
	virtual bool _can_inherit_from_file() const override;
	virtual bool _can_make_function() const override;
	virtual int32_t _find_function(const String &p_function, const String &p_code) const override;
	virtual String _make_function(const String &p_class_name, const String &p_function_name, const PackedStringArray &p_function_args) const override;
	virtual Error _open_in_external_editor(const Ref<Script> &p_script, int32_t p_line, int32_t p_column) override;
	virtual bool _overrides_external_editor() override;
	virtual Dictionary _complete_code(const String &p_code, const String &p_path, Object *p_owner) const override;
	virtual Dictionary _lookup_code(const String &p_code, const String &p_symbol, const String &p_path, Object *p_owner) const override;
	virtual String _auto_indent_code(const String &p_code, int32_t p_from_line, int32_t p_to_line) const override;
	virtual void _add_global_constant(const StringName &p_name, const Variant &p_value) override;
	virtual void _add_named_global_constant(const StringName &p_name, const Variant &p_value) override;
	virtual void _remove_named_global_constant(const StringName &p_name) override;
	virtual void _thread_enter() override;
	virtual void _thread_exit() override;
	virtual String _debug_get_error() const override;
	virtual int32_t _debug_get_stack_level_count() const override;
	virtual int32_t _debug_get_stack_level_line(int32_t p_level) const override;
	virtual String _debug_get_stack_level_function(int32_t p_level) const override;
	virtual Dictionary _debug_get_stack_level_locals(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth) override;
	virtual Dictionary _debug_get_stack_level_members(int32_t p_level, int32_t p_max_subitems, int32_t p_max_depth) override;
	virtual void *_debug_get_stack_level_instance(int32_t p_level) override;
	virtual Dictionary _debug_get_globals(int32_t p_max_subitems, int32_t p_max_depth) override;
	virtual String _debug_parse_stack_level_expression(int32_t p_level, const String &p_expression, int32_t p_max_subitems, int32_t p_max_depth) override;
	virtual TypedArray<Dictionary> _debug_get_current_stack_info() override;
	virtual void _reload_all_scripts() override;
	virtual void _reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) override;
	virtual PackedStringArray _get_recognized_extensions() const override;
	virtual TypedArray<Dictionary> _get_public_functions() const override;
	virtual Dictionary _get_public_constants() const override;
	virtual TypedArray<Dictionary> _get_public_annotations() const override;
	virtual void _profiling_start() override;
	virtual void _profiling_stop() override;
	virtual int32_t _profiling_get_accumulated_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max) override;
	virtual int32_t _profiling_get_frame_data(ScriptLanguageExtensionProfilingInfo *p_info_array, int32_t p_info_max) override;
	virtual void _frame() override;
	virtual bool _handles_global_class_type(const String &p_type) const override;
	virtual Dictionary _get_global_class_name(const String &p_path) const override;
	void load_icon();

	CPPScriptLanguage() {}
	~CPPScriptLanguage() {}
};
