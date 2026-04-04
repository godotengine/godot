/**************************************************************************/
/*  scene_tree.cpp                                                        */
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

#include <godot_cpp/classes/scene_tree.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/multiplayer_api.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/packed_scene.hpp>
#include <godot_cpp/classes/scene_tree_timer.hpp>
#include <godot_cpp/classes/tween.hpp>
#include <godot_cpp/classes/window.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace godot {

Window *SceneTree::get_root() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("get_root")._native_ptr(), 1757182445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Window>(_gde_method_bind, _owner);
}

bool SceneTree::has_group(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("has_group")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool SceneTree::is_accessibility_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("is_accessibility_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool SceneTree::is_accessibility_supported() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("is_accessibility_supported")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool SceneTree::is_auto_accept_quit() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("is_auto_accept_quit")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SceneTree::set_auto_accept_quit(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_auto_accept_quit")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool SceneTree::is_quit_on_go_back() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("is_quit_on_go_back")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SceneTree::set_quit_on_go_back(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_quit_on_go_back")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

void SceneTree::set_debug_collisions_hint(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_debug_collisions_hint")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool SceneTree::is_debugging_collisions_hint() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("is_debugging_collisions_hint")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SceneTree::set_debug_paths_hint(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_debug_paths_hint")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool SceneTree::is_debugging_paths_hint() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("is_debugging_paths_hint")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SceneTree::set_debug_navigation_hint(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_debug_navigation_hint")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool SceneTree::is_debugging_navigation_hint() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("is_debugging_navigation_hint")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SceneTree::set_edited_scene_root(Node *p_scene) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_edited_scene_root")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_scene != nullptr ? &p_scene->_owner : nullptr));
}

Node *SceneTree::get_edited_scene_root() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("get_edited_scene_root")._native_ptr(), 3160264692);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner);
}

void SceneTree::set_pause(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_pause")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool SceneTree::is_paused() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("is_paused")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<SceneTreeTimer> SceneTree::create_timer(double p_time_sec, bool p_process_always, bool p_process_in_physics, bool p_ignore_time_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("create_timer")._native_ptr(), 2709170273);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SceneTreeTimer>()));
	double p_time_sec_encoded;
	PtrToArg<double>::encode(p_time_sec, &p_time_sec_encoded);
	int8_t p_process_always_encoded;
	PtrToArg<bool>::encode(p_process_always, &p_process_always_encoded);
	int8_t p_process_in_physics_encoded;
	PtrToArg<bool>::encode(p_process_in_physics, &p_process_in_physics_encoded);
	int8_t p_ignore_time_scale_encoded;
	PtrToArg<bool>::encode(p_ignore_time_scale, &p_ignore_time_scale_encoded);
	return Ref<SceneTreeTimer>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SceneTreeTimer>(_gde_method_bind, _owner, &p_time_sec_encoded, &p_process_always_encoded, &p_process_in_physics_encoded, &p_ignore_time_scale_encoded));
}

Ref<Tween> SceneTree::create_tween() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("create_tween")._native_ptr(), 3426978995);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner));
}

TypedArray<Ref<Tween>> SceneTree::get_processed_tweens() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("get_processed_tweens")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Tween>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Tween>>>(_gde_method_bind, _owner);
}

int32_t SceneTree::get_node_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("get_node_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int64_t SceneTree::get_frame() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("get_frame")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SceneTree::quit(int32_t p_exit_code) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("quit")._native_ptr(), 1995695955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_exit_code_encoded;
	PtrToArg<int64_t>::encode(p_exit_code, &p_exit_code_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_exit_code_encoded);
}

void SceneTree::set_physics_interpolation_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_physics_interpolation_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool SceneTree::is_physics_interpolation_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("is_physics_interpolation_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SceneTree::queue_delete(Object *p_obj) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("queue_delete")._native_ptr(), 3975164845);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_obj != nullptr ? &p_obj->_owner : nullptr));
}

void SceneTree::call_group_flags_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("call_group_flags")._native_ptr(), 1527739229);
	CHECK_METHOD_BIND(_gde_method_bind);
	GDExtensionCallError error;
	Variant ret;
	::godot::gdextension_interface::object_method_bind_call(_gde_method_bind, _owner, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count, &ret, &error);
}

void SceneTree::notify_group_flags(uint32_t p_call_flags, const StringName &p_group, int32_t p_notification) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("notify_group_flags")._native_ptr(), 1245489420);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_call_flags_encoded;
	PtrToArg<int64_t>::encode(p_call_flags, &p_call_flags_encoded);
	int64_t p_notification_encoded;
	PtrToArg<int64_t>::encode(p_notification, &p_notification_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_call_flags_encoded, &p_group, &p_notification_encoded);
}

void SceneTree::set_group_flags(uint32_t p_call_flags, const StringName &p_group, const String &p_property, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_group_flags")._native_ptr(), 3497599527);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_call_flags_encoded;
	PtrToArg<int64_t>::encode(p_call_flags, &p_call_flags_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_call_flags_encoded, &p_group, &p_property, &p_value);
}

void SceneTree::call_group_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("call_group")._native_ptr(), 1257962832);
	CHECK_METHOD_BIND(_gde_method_bind);
	GDExtensionCallError error;
	Variant ret;
	::godot::gdextension_interface::object_method_bind_call(_gde_method_bind, _owner, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count, &ret, &error);
}

void SceneTree::notify_group(const StringName &p_group, int32_t p_notification) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("notify_group")._native_ptr(), 2415702435);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_notification_encoded;
	PtrToArg<int64_t>::encode(p_notification, &p_notification_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_group, &p_notification_encoded);
}

void SceneTree::set_group(const StringName &p_group, const String &p_property, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_group")._native_ptr(), 1279312029);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_group, &p_property, &p_value);
}

TypedArray<Node> SceneTree::get_nodes_in_group(const StringName &p_group) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("get_nodes_in_group")._native_ptr(), 689397652);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Node>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Node>>(_gde_method_bind, _owner, &p_group);
}

Node *SceneTree::get_first_node_in_group(const StringName &p_group) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("get_first_node_in_group")._native_ptr(), 4071044623);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner, &p_group);
}

int32_t SceneTree::get_node_count_in_group(const StringName &p_group) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("get_node_count_in_group")._native_ptr(), 2458036349);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_group);
}

void SceneTree::set_current_scene(Node *p_child_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_current_scene")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_child_node != nullptr ? &p_child_node->_owner : nullptr));
}

Node *SceneTree::get_current_scene() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("get_current_scene")._native_ptr(), 3160264692);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner);
}

Error SceneTree::change_scene_to_file(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("change_scene_to_file")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

Error SceneTree::change_scene_to_packed(const Ref<PackedScene> &p_packed_scene) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("change_scene_to_packed")._native_ptr(), 107349098);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_packed_scene != nullptr ? &p_packed_scene->_owner : nullptr));
}

Error SceneTree::change_scene_to_node(Node *p_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("change_scene_to_node")._native_ptr(), 2584678054);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr));
}

Error SceneTree::reload_current_scene() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("reload_current_scene")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SceneTree::unload_current_scene() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("unload_current_scene")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void SceneTree::set_multiplayer(const Ref<MultiplayerAPI> &p_multiplayer, const NodePath &p_root_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_multiplayer")._native_ptr(), 2385607013);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_multiplayer != nullptr ? &p_multiplayer->_owner : nullptr), &p_root_path);
}

Ref<MultiplayerAPI> SceneTree::get_multiplayer(const NodePath &p_for_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("get_multiplayer")._native_ptr(), 3453401404);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<MultiplayerAPI>()));
	return Ref<MultiplayerAPI>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<MultiplayerAPI>(_gde_method_bind, _owner, &p_for_path));
}

void SceneTree::set_multiplayer_poll_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("set_multiplayer_poll_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool SceneTree::is_multiplayer_poll_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneTree::get_class_static()._native_ptr(), StringName("is_multiplayer_poll_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
