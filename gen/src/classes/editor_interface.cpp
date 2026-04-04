/**************************************************************************/
/*  editor_interface.cpp                                                  */
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

#include <godot_cpp/classes/editor_interface.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/editor_command_palette.hpp>
#include <godot_cpp/classes/editor_file_system.hpp>
#include <godot_cpp/classes/editor_inspector.hpp>
#include <godot_cpp/classes/editor_paths.hpp>
#include <godot_cpp/classes/editor_resource_preview.hpp>
#include <godot_cpp/classes/editor_selection.hpp>
#include <godot_cpp/classes/editor_settings.hpp>
#include <godot_cpp/classes/editor_toaster.hpp>
#include <godot_cpp/classes/editor_undo_redo_manager.hpp>
#include <godot_cpp/classes/file_system_dock.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/classes/script.hpp>
#include <godot_cpp/classes/script_editor.hpp>
#include <godot_cpp/classes/sub_viewport.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/theme.hpp>
#include <godot_cpp/classes/v_box_container.hpp>
#include <godot_cpp/classes/window.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {

EditorInterface *EditorInterface::singleton = nullptr;

EditorInterface *EditorInterface::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(EditorInterface::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<EditorInterface *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &EditorInterface::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(EditorInterface::get_class_static(), singleton);
		}
	}
	return singleton;
}

EditorInterface::~EditorInterface() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(EditorInterface::get_class_static());
		singleton = nullptr;
	}
}

void EditorInterface::restart_editor(bool p_save) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("restart_editor")._native_ptr(), 3216645846);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_save_encoded;
	PtrToArg<bool>::encode(p_save, &p_save_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_save_encoded);
}

EditorCommandPalette *EditorInterface::get_command_palette() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_command_palette")._native_ptr(), 2471163807);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<EditorCommandPalette>(_gde_method_bind, _owner);
}

EditorFileSystem *EditorInterface::get_resource_filesystem() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_resource_filesystem")._native_ptr(), 780151678);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<EditorFileSystem>(_gde_method_bind, _owner);
}

EditorPaths *EditorInterface::get_editor_paths() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_editor_paths")._native_ptr(), 1595760068);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<EditorPaths>(_gde_method_bind, _owner);
}

EditorResourcePreview *EditorInterface::get_resource_previewer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_resource_previewer")._native_ptr(), 943486957);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<EditorResourcePreview>(_gde_method_bind, _owner);
}

EditorSelection *EditorInterface::get_selection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_selection")._native_ptr(), 2690272531);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<EditorSelection>(_gde_method_bind, _owner);
}

Ref<EditorSettings> EditorInterface::get_editor_settings() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_editor_settings")._native_ptr(), 4086932459);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<EditorSettings>()));
	return Ref<EditorSettings>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<EditorSettings>(_gde_method_bind, _owner));
}

EditorToaster *EditorInterface::get_editor_toaster() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_editor_toaster")._native_ptr(), 3612675797);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<EditorToaster>(_gde_method_bind, _owner);
}

EditorUndoRedoManager *EditorInterface::get_editor_undo_redo() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_editor_undo_redo")._native_ptr(), 3819628421);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<EditorUndoRedoManager>(_gde_method_bind, _owner);
}

TypedArray<Ref<Texture2D>> EditorInterface::make_mesh_previews(const TypedArray<Ref<Mesh>> &p_meshes, int32_t p_preview_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("make_mesh_previews")._native_ptr(), 878078554);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Texture2D>>()));
	int64_t p_preview_size_encoded;
	PtrToArg<int64_t>::encode(p_preview_size, &p_preview_size_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Texture2D>>>(_gde_method_bind, _owner, &p_meshes, &p_preview_size_encoded);
}

void EditorInterface::set_plugin_enabled(const String &p_plugin, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("set_plugin_enabled")._native_ptr(), 2678287736);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_plugin, &p_enabled_encoded);
}

bool EditorInterface::is_plugin_enabled(const String &p_plugin) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("is_plugin_enabled")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_plugin);
}

Ref<Theme> EditorInterface::get_editor_theme() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_editor_theme")._native_ptr(), 3846893731);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Theme>()));
	return Ref<Theme>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Theme>(_gde_method_bind, _owner));
}

Control *EditorInterface::get_base_control() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_base_control")._native_ptr(), 2783021301);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Control>(_gde_method_bind, _owner);
}

VBoxContainer *EditorInterface::get_editor_main_screen() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_editor_main_screen")._native_ptr(), 1706218421);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<VBoxContainer>(_gde_method_bind, _owner);
}

ScriptEditor *EditorInterface::get_script_editor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_script_editor")._native_ptr(), 90868003);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<ScriptEditor>(_gde_method_bind, _owner);
}

SubViewport *EditorInterface::get_editor_viewport_2d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_editor_viewport_2d")._native_ptr(), 3750751911);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<SubViewport>(_gde_method_bind, _owner);
}

SubViewport *EditorInterface::get_editor_viewport_3d(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_editor_viewport_3d")._native_ptr(), 1970834490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<SubViewport>(_gde_method_bind, _owner, &p_idx_encoded);
}

void EditorInterface::set_main_screen_editor(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("set_main_screen_editor")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void EditorInterface::set_distraction_free_mode(bool p_enter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("set_distraction_free_mode")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enter_encoded;
	PtrToArg<bool>::encode(p_enter, &p_enter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enter_encoded);
}

bool EditorInterface::is_distraction_free_mode_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("is_distraction_free_mode_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool EditorInterface::is_multi_window_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("is_multi_window_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

float EditorInterface::get_editor_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_editor_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

String EditorInterface::get_editor_language() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_editor_language")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool EditorInterface::is_node_3d_snap_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("is_node_3d_snap_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

float EditorInterface::get_node_3d_translate_snap() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_node_3d_translate_snap")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float EditorInterface::get_node_3d_rotate_snap() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_node_3d_rotate_snap")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float EditorInterface::get_node_3d_scale_snap() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_node_3d_scale_snap")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void EditorInterface::popup_dialog(Window *p_dialog, const Rect2i &p_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("popup_dialog")._native_ptr(), 2015770942);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_dialog != nullptr ? &p_dialog->_owner : nullptr), &p_rect);
}

void EditorInterface::popup_dialog_centered(Window *p_dialog, const Vector2i &p_minsize) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("popup_dialog_centered")._native_ptr(), 346557367);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_dialog != nullptr ? &p_dialog->_owner : nullptr), &p_minsize);
}

void EditorInterface::popup_dialog_centered_ratio(Window *p_dialog, float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("popup_dialog_centered_ratio")._native_ptr(), 2093669136);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_dialog != nullptr ? &p_dialog->_owner : nullptr), &p_ratio_encoded);
}

void EditorInterface::popup_dialog_centered_clamped(Window *p_dialog, const Vector2i &p_minsize, float p_fallback_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("popup_dialog_centered_clamped")._native_ptr(), 3763385571);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fallback_ratio_encoded;
	PtrToArg<double>::encode(p_fallback_ratio, &p_fallback_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_dialog != nullptr ? &p_dialog->_owner : nullptr), &p_minsize, &p_fallback_ratio_encoded);
}

String EditorInterface::get_current_feature_profile() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_current_feature_profile")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void EditorInterface::set_current_feature_profile(const String &p_profile_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("set_current_feature_profile")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_profile_name);
}

void EditorInterface::popup_node_selector(const Callable &p_callback, const TypedArray<StringName> &p_valid_types, Node *p_current_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("popup_node_selector")._native_ptr(), 2444591477);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callback, &p_valid_types, (p_current_value != nullptr ? &p_current_value->_owner : nullptr));
}

void EditorInterface::popup_property_selector(Object *p_object, const Callable &p_callback, const PackedInt32Array &p_type_filter, const String &p_current_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("popup_property_selector")._native_ptr(), 2955609011);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr), &p_callback, &p_type_filter, &p_current_value);
}

void EditorInterface::popup_method_selector(Object *p_object, const Callable &p_callback, const String &p_current_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("popup_method_selector")._native_ptr(), 3585505226);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr), &p_callback, &p_current_value);
}

void EditorInterface::popup_quick_open(const Callable &p_callback, const TypedArray<StringName> &p_base_types) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("popup_quick_open")._native_ptr(), 2271411043);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callback, &p_base_types);
}

void EditorInterface::popup_create_dialog(const Callable &p_callback, const StringName &p_base_type, const String &p_current_type, const String &p_dialog_title, const TypedArray<StringName> &p_type_blocklist) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("popup_create_dialog")._native_ptr(), 495277124);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callback, &p_base_type, &p_current_type, &p_dialog_title, &p_type_blocklist);
}

FileSystemDock *EditorInterface::get_file_system_dock() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_file_system_dock")._native_ptr(), 3751012327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<FileSystemDock>(_gde_method_bind, _owner);
}

void EditorInterface::select_file(const String &p_file) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("select_file")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_file);
}

PackedStringArray EditorInterface::get_selected_paths() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_selected_paths")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

String EditorInterface::get_current_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_current_path")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String EditorInterface::get_current_directory() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_current_directory")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

EditorInspector *EditorInterface::get_inspector() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_inspector")._native_ptr(), 3517113938);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<EditorInspector>(_gde_method_bind, _owner);
}

void EditorInterface::inspect_object(Object *p_object, const String &p_for_property, bool p_inspector_only) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("inspect_object")._native_ptr(), 127962172);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_inspector_only_encoded;
	PtrToArg<bool>::encode(p_inspector_only, &p_inspector_only_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr), &p_for_property, &p_inspector_only_encoded);
}

void EditorInterface::edit_resource(const Ref<Resource> &p_resource) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("edit_resource")._native_ptr(), 968641751);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_resource != nullptr ? &p_resource->_owner : nullptr));
}

void EditorInterface::edit_node(Node *p_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("edit_node")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr));
}

void EditorInterface::edit_script(const Ref<Script> &p_script, int32_t p_line, int32_t p_column, bool p_grab_focus) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("edit_script")._native_ptr(), 219829402);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_grab_focus_encoded;
	PtrToArg<bool>::encode(p_grab_focus, &p_grab_focus_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_script != nullptr ? &p_script->_owner : nullptr), &p_line_encoded, &p_column_encoded, &p_grab_focus_encoded);
}

void EditorInterface::open_scene_from_path(const String &p_scene_filepath, bool p_set_inherited) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("open_scene_from_path")._native_ptr(), 1168363258);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_set_inherited_encoded;
	PtrToArg<bool>::encode(p_set_inherited, &p_set_inherited_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scene_filepath, &p_set_inherited_encoded);
}

void EditorInterface::reload_scene_from_path(const String &p_scene_filepath) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("reload_scene_from_path")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scene_filepath);
}

void EditorInterface::set_object_edited(Object *p_object, bool p_edited) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("set_object_edited")._native_ptr(), 1462101905);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_edited_encoded;
	PtrToArg<bool>::encode(p_edited, &p_edited_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr), &p_edited_encoded);
}

bool EditorInterface::is_object_edited(Object *p_object) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("is_object_edited")._native_ptr(), 397768994);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr));
}

PackedStringArray EditorInterface::get_open_scenes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_open_scenes")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

TypedArray<Node> EditorInterface::get_open_scene_roots() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_open_scene_roots")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Node>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Node>>(_gde_method_bind, _owner);
}

Node *EditorInterface::get_edited_scene_root() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_edited_scene_root")._native_ptr(), 3160264692);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner);
}

void EditorInterface::add_root_node(Node *p_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("add_root_node")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr));
}

Error EditorInterface::save_scene() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("save_scene")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void EditorInterface::save_scene_as(const String &p_path, bool p_with_preview) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("save_scene_as")._native_ptr(), 3647332257);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_with_preview_encoded;
	PtrToArg<bool>::encode(p_with_preview, &p_with_preview_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_with_preview_encoded);
}

void EditorInterface::save_all_scenes() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("save_all_scenes")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Error EditorInterface::close_scene() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("close_scene")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void EditorInterface::mark_scene_as_unsaved() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("mark_scene_as_unsaved")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void EditorInterface::play_main_scene() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("play_main_scene")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void EditorInterface::play_current_scene() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("play_current_scene")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void EditorInterface::play_custom_scene(const String &p_scene_filepath) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("play_custom_scene")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scene_filepath);
}

void EditorInterface::stop_playing_scene() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("stop_playing_scene")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool EditorInterface::is_playing_scene() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("is_playing_scene")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

String EditorInterface::get_playing_scene() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("get_playing_scene")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void EditorInterface::set_movie_maker_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("set_movie_maker_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool EditorInterface::is_movie_maker_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInterface::get_class_static()._native_ptr(), StringName("is_movie_maker_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
