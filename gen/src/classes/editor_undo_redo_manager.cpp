/**************************************************************************/
/*  editor_undo_redo_manager.cpp                                          */
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

#include <godot_cpp/classes/editor_undo_redo_manager.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace godot {

void EditorUndoRedoManager::create_action(const String &p_name, UndoRedo::MergeMode p_merge_mode, Object *p_custom_context, bool p_backward_undo_ops, bool p_mark_unsaved) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("create_action")._native_ptr(), 796197507);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_merge_mode_encoded;
	PtrToArg<int64_t>::encode(p_merge_mode, &p_merge_mode_encoded);
	int8_t p_backward_undo_ops_encoded;
	PtrToArg<bool>::encode(p_backward_undo_ops, &p_backward_undo_ops_encoded);
	int8_t p_mark_unsaved_encoded;
	PtrToArg<bool>::encode(p_mark_unsaved, &p_mark_unsaved_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_merge_mode_encoded, (p_custom_context != nullptr ? &p_custom_context->_owner : nullptr), &p_backward_undo_ops_encoded, &p_mark_unsaved_encoded);
}

void EditorUndoRedoManager::commit_action(bool p_execute) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("commit_action")._native_ptr(), 3216645846);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_execute_encoded;
	PtrToArg<bool>::encode(p_execute, &p_execute_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_execute_encoded);
}

bool EditorUndoRedoManager::is_committing_action() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("is_committing_action")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorUndoRedoManager::force_fixed_history() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("force_fixed_history")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void EditorUndoRedoManager::add_do_method_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("add_do_method")._native_ptr(), 1517810467);
	CHECK_METHOD_BIND(_gde_method_bind);
	GDExtensionCallError error;
	Variant ret;
	::godot::gdextension_interface::object_method_bind_call(_gde_method_bind, _owner, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count, &ret, &error);
}

void EditorUndoRedoManager::add_undo_method_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("add_undo_method")._native_ptr(), 1517810467);
	CHECK_METHOD_BIND(_gde_method_bind);
	GDExtensionCallError error;
	Variant ret;
	::godot::gdextension_interface::object_method_bind_call(_gde_method_bind, _owner, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count, &ret, &error);
}

void EditorUndoRedoManager::add_do_property(Object *p_object, const StringName &p_property, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("add_do_property")._native_ptr(), 1017172818);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr), &p_property, &p_value);
}

void EditorUndoRedoManager::add_undo_property(Object *p_object, const StringName &p_property, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("add_undo_property")._native_ptr(), 1017172818);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr), &p_property, &p_value);
}

void EditorUndoRedoManager::add_do_reference(Object *p_object) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("add_do_reference")._native_ptr(), 3975164845);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr));
}

void EditorUndoRedoManager::add_undo_reference(Object *p_object) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("add_undo_reference")._native_ptr(), 3975164845);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr));
}

int32_t EditorUndoRedoManager::get_object_history_id(Object *p_object) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("get_object_history_id")._native_ptr(), 1107568780);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr));
}

UndoRedo *EditorUndoRedoManager::get_history_undo_redo(int32_t p_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("get_history_undo_redo")._native_ptr(), 2417974513);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<UndoRedo>(_gde_method_bind, _owner, &p_id_encoded);
}

void EditorUndoRedoManager::clear_history(int32_t p_id, bool p_increase_version) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorUndoRedoManager::get_class_static()._native_ptr(), StringName("clear_history")._native_ptr(), 2020603371);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int8_t p_increase_version_encoded;
	PtrToArg<bool>::encode(p_increase_version, &p_increase_version_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id_encoded, &p_increase_version_encoded);
}

} // namespace godot
