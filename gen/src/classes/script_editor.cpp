/**************************************************************************/
/*  script_editor.cpp                                                     */
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

#include <godot_cpp/classes/script_editor.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/editor_syntax_highlighter.hpp>
#include <godot_cpp/classes/script.hpp>
#include <godot_cpp/classes/script_editor_base.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

ScriptEditorBase *ScriptEditor::get_current_editor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("get_current_editor")._native_ptr(), 1906266726);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<ScriptEditorBase>(_gde_method_bind, _owner);
}

TypedArray<ScriptEditorBase> ScriptEditor::get_open_script_editors() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("get_open_script_editors")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<ScriptEditorBase>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<ScriptEditorBase>>(_gde_method_bind, _owner);
}

PackedStringArray ScriptEditor::get_breakpoints() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("get_breakpoints")._native_ptr(), 2981934095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void ScriptEditor::register_syntax_highlighter(const Ref<EditorSyntaxHighlighter> &p_syntax_highlighter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("register_syntax_highlighter")._native_ptr(), 1092774468);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_syntax_highlighter != nullptr ? &p_syntax_highlighter->_owner : nullptr));
}

void ScriptEditor::unregister_syntax_highlighter(const Ref<EditorSyntaxHighlighter> &p_syntax_highlighter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("unregister_syntax_highlighter")._native_ptr(), 1092774468);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_syntax_highlighter != nullptr ? &p_syntax_highlighter->_owner : nullptr));
}

void ScriptEditor::goto_line(int32_t p_line_number) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("goto_line")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_number_encoded;
	PtrToArg<int64_t>::encode(p_line_number, &p_line_number_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_number_encoded);
}

Ref<Script> ScriptEditor::get_current_script() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("get_current_script")._native_ptr(), 2146468882);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Script>()));
	return Ref<Script>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Script>(_gde_method_bind, _owner));
}

TypedArray<Ref<Script>> ScriptEditor::get_open_scripts() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("get_open_scripts")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Script>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Script>>>(_gde_method_bind, _owner);
}

void ScriptEditor::open_script_create_dialog(const String &p_base_name, const String &p_base_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("open_script_create_dialog")._native_ptr(), 3186203200);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_base_name, &p_base_path);
}

void ScriptEditor::goto_help(const String &p_topic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("goto_help")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_topic);
}

void ScriptEditor::update_docs_from_script(const Ref<Script> &p_script) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("update_docs_from_script")._native_ptr(), 3657522847);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_script != nullptr ? &p_script->_owner : nullptr));
}

void ScriptEditor::clear_docs_from_script(const Ref<Script> &p_script) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScriptEditor::get_class_static()._native_ptr(), StringName("clear_docs_from_script")._native_ptr(), 3657522847);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_script != nullptr ? &p_script->_owner : nullptr));
}

} // namespace godot
