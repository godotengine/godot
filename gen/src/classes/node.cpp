/**************************************************************************/
/*  node.cpp                                                              */
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

#include <godot_cpp/classes/node.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/input_event.hpp>
#include <godot_cpp/classes/multiplayer_api.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/tween.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/classes/window.hpp>

namespace godot {

void Node::print_orphan_nodes() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("print_orphan_nodes")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, nullptr);
}

TypedArray<int> Node::get_orphan_node_ids() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_orphan_node_ids")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<int>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<int>>(_gde_method_bind, nullptr);
}

void Node::add_sibling(Node *p_sibling, bool p_force_readable_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("add_sibling")._native_ptr(), 2570952461);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_force_readable_name_encoded;
	PtrToArg<bool>::encode(p_force_readable_name, &p_force_readable_name_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_sibling != nullptr ? &p_sibling->_owner : nullptr), &p_force_readable_name_encoded);
}

void Node::set_name(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_name")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

StringName Node::get_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_name")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

void Node::add_child(Node *p_node, bool p_force_readable_name, Node::InternalMode p_internal) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("add_child")._native_ptr(), 3863233950);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_force_readable_name_encoded;
	PtrToArg<bool>::encode(p_force_readable_name, &p_force_readable_name_encoded);
	int64_t p_internal_encoded;
	PtrToArg<int64_t>::encode(p_internal, &p_internal_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr), &p_force_readable_name_encoded, &p_internal_encoded);
}

void Node::remove_child(Node *p_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("remove_child")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr));
}

void Node::reparent(Node *p_new_parent, bool p_keep_global_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("reparent")._native_ptr(), 3685795103);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_global_transform_encoded;
	PtrToArg<bool>::encode(p_keep_global_transform, &p_keep_global_transform_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_new_parent != nullptr ? &p_new_parent->_owner : nullptr), &p_keep_global_transform_encoded);
}

int32_t Node::get_child_count(bool p_include_internal) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_child_count")._native_ptr(), 894402480);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int8_t p_include_internal_encoded;
	PtrToArg<bool>::encode(p_include_internal, &p_include_internal_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_include_internal_encoded);
}

TypedArray<Node> Node::get_children(bool p_include_internal) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_children")._native_ptr(), 873284517);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Node>()));
	int8_t p_include_internal_encoded;
	PtrToArg<bool>::encode(p_include_internal, &p_include_internal_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Node>>(_gde_method_bind, _owner, &p_include_internal_encoded);
}

Node *Node::get_child(int32_t p_idx, bool p_include_internal) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_child")._native_ptr(), 541253412);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_include_internal_encoded;
	PtrToArg<bool>::encode(p_include_internal, &p_include_internal_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner, &p_idx_encoded, &p_include_internal_encoded);
}

bool Node::has_node(const NodePath &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("has_node")._native_ptr(), 861721659);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

Node *Node::get_node_internal(const NodePath &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_node")._native_ptr(), 2734337346);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner, &p_path);
}

Node *Node::get_node_or_null(const NodePath &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_node_or_null")._native_ptr(), 2734337346);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner, &p_path);
}

Node *Node::get_parent() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_parent")._native_ptr(), 3160264692);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner);
}

Node *Node::find_child(const String &p_pattern, bool p_recursive, bool p_owned) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("find_child")._native_ptr(), 2008217037);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int8_t p_recursive_encoded;
	PtrToArg<bool>::encode(p_recursive, &p_recursive_encoded);
	int8_t p_owned_encoded;
	PtrToArg<bool>::encode(p_owned, &p_owned_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner, &p_pattern, &p_recursive_encoded, &p_owned_encoded);
}

TypedArray<Node> Node::find_children(const String &p_pattern, const String &p_type, bool p_recursive, bool p_owned) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("find_children")._native_ptr(), 2560337219);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Node>()));
	int8_t p_recursive_encoded;
	PtrToArg<bool>::encode(p_recursive, &p_recursive_encoded);
	int8_t p_owned_encoded;
	PtrToArg<bool>::encode(p_owned, &p_owned_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Node>>(_gde_method_bind, _owner, &p_pattern, &p_type, &p_recursive_encoded, &p_owned_encoded);
}

Node *Node::find_parent(const String &p_pattern) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("find_parent")._native_ptr(), 1140089439);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner, &p_pattern);
}

bool Node::has_node_and_resource(const NodePath &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("has_node_and_resource")._native_ptr(), 861721659);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

Array Node::get_node_and_resource(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_node_and_resource")._native_ptr(), 502563882);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_path);
}

bool Node::is_inside_tree() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_inside_tree")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool Node::is_part_of_edited_scene() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_part_of_edited_scene")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool Node::is_ancestor_of(Node *p_node) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_ancestor_of")._native_ptr(), 3093956946);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr));
}

bool Node::is_greater_than(Node *p_node) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_greater_than")._native_ptr(), 3093956946);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr));
}

NodePath Node::get_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_path")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

NodePath Node::get_path_to(Node *p_node, bool p_use_unique_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_path_to")._native_ptr(), 498846349);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int8_t p_use_unique_path_encoded;
	PtrToArg<bool>::encode(p_use_unique_path, &p_use_unique_path_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr), &p_use_unique_path_encoded);
}

void Node::add_to_group(const StringName &p_group, bool p_persistent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("add_to_group")._native_ptr(), 3683006648);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_persistent_encoded;
	PtrToArg<bool>::encode(p_persistent, &p_persistent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_group, &p_persistent_encoded);
}

void Node::remove_from_group(const StringName &p_group) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("remove_from_group")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_group);
}

bool Node::is_in_group(const StringName &p_group) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_in_group")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_group);
}

void Node::move_child(Node *p_child_node, int32_t p_to_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("move_child")._native_ptr(), 3315886247);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_to_index_encoded;
	PtrToArg<int64_t>::encode(p_to_index, &p_to_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_child_node != nullptr ? &p_child_node->_owner : nullptr), &p_to_index_encoded);
}

TypedArray<StringName> Node::get_groups() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_groups")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<StringName>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<StringName>>(_gde_method_bind, _owner);
}

void Node::set_owner(Node *p_owner) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_owner")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_owner != nullptr ? &p_owner->_owner : nullptr));
}

Node *Node::get_owner() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_owner")._native_ptr(), 3160264692);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner);
}

int32_t Node::get_index(bool p_include_internal) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_index")._native_ptr(), 894402480);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int8_t p_include_internal_encoded;
	PtrToArg<bool>::encode(p_include_internal, &p_include_internal_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_include_internal_encoded);
}

void Node::print_tree() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("print_tree")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Node::print_tree_pretty() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("print_tree_pretty")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

String Node::get_tree_string() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_tree_string")._native_ptr(), 2841200299);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String Node::get_tree_string_pretty() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_tree_string_pretty")._native_ptr(), 2841200299);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Node::set_scene_file_path(const String &p_scene_file_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_scene_file_path")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scene_file_path);
}

String Node::get_scene_file_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_scene_file_path")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Node::propagate_notification(int32_t p_what) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("propagate_notification")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_what_encoded;
	PtrToArg<int64_t>::encode(p_what, &p_what_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_what_encoded);
}

void Node::propagate_call(const StringName &p_method, const Array &p_args, bool p_parent_first) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("propagate_call")._native_ptr(), 1871007965);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_parent_first_encoded;
	PtrToArg<bool>::encode(p_parent_first, &p_parent_first_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_method, &p_args, &p_parent_first_encoded);
}

void Node::set_physics_process(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_physics_process")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

double Node::get_physics_process_delta_time() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_physics_process_delta_time")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool Node::is_physics_processing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_physics_processing")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

double Node::get_process_delta_time() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_process_delta_time")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Node::set_process(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_process")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void Node::set_process_priority(int32_t p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_process_priority")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_priority_encoded;
	PtrToArg<int64_t>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_priority_encoded);
}

int32_t Node::get_process_priority() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_process_priority")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Node::set_physics_process_priority(int32_t p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_physics_process_priority")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_priority_encoded;
	PtrToArg<int64_t>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_priority_encoded);
}

int32_t Node::get_physics_process_priority() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_physics_process_priority")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Node::is_processing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_processing")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_process_input(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_process_input")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Node::is_processing_input() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_processing_input")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_process_shortcut_input(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_process_shortcut_input")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Node::is_processing_shortcut_input() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_processing_shortcut_input")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_process_unhandled_input(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_process_unhandled_input")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Node::is_processing_unhandled_input() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_processing_unhandled_input")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_process_unhandled_key_input(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_process_unhandled_key_input")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Node::is_processing_unhandled_key_input() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_processing_unhandled_key_input")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_process_mode(Node::ProcessMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_process_mode")._native_ptr(), 1841290486);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Node::ProcessMode Node::get_process_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_process_mode")._native_ptr(), 739966102);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Node::ProcessMode(0)));
	return (Node::ProcessMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Node::can_process() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("can_process")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_process_thread_group(Node::ProcessThreadGroup p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_process_thread_group")._native_ptr(), 2275442745);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Node::ProcessThreadGroup Node::get_process_thread_group() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_process_thread_group")._native_ptr(), 1866404740);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Node::ProcessThreadGroup(0)));
	return (Node::ProcessThreadGroup)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Node::set_process_thread_messages(BitField<Node::ProcessThreadMessages> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_process_thread_messages")._native_ptr(), 1357280998);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flags);
}

BitField<Node::ProcessThreadMessages> Node::get_process_thread_messages() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_process_thread_messages")._native_ptr(), 4228993612);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<Node::ProcessThreadMessages>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Node::set_process_thread_group_order(int32_t p_order) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_process_thread_group_order")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_order_encoded;
	PtrToArg<int64_t>::encode(p_order, &p_order_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_order_encoded);
}

int32_t Node::get_process_thread_group_order() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_process_thread_group_order")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Node::queue_accessibility_update() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("queue_accessibility_update")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

RID Node::get_accessibility_element() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_accessibility_element")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void Node::set_display_folded(bool p_fold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_display_folded")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_fold_encoded;
	PtrToArg<bool>::encode(p_fold, &p_fold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fold_encoded);
}

bool Node::is_displayed_folded() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_displayed_folded")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_process_internal(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_process_internal")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Node::is_processing_internal() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_processing_internal")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_physics_process_internal(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_physics_process_internal")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Node::is_physics_processing_internal() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_physics_processing_internal")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_physics_interpolation_mode(Node::PhysicsInterpolationMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_physics_interpolation_mode")._native_ptr(), 3202404928);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Node::PhysicsInterpolationMode Node::get_physics_interpolation_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_physics_interpolation_mode")._native_ptr(), 2920385216);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Node::PhysicsInterpolationMode(0)));
	return (Node::PhysicsInterpolationMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Node::is_physics_interpolated() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_physics_interpolated")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool Node::is_physics_interpolated_and_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_physics_interpolated_and_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::reset_physics_interpolation() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("reset_physics_interpolation")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Node::set_auto_translate_mode(Node::AutoTranslateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_auto_translate_mode")._native_ptr(), 776149714);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Node::AutoTranslateMode Node::get_auto_translate_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_auto_translate_mode")._native_ptr(), 2498906432);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Node::AutoTranslateMode(0)));
	return (Node::AutoTranslateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Node::can_auto_translate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("can_auto_translate")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_translation_domain_inherited() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_translation_domain_inherited")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Window *Node::get_window() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_window")._native_ptr(), 1757182445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Window>(_gde_method_bind, _owner);
}

Window *Node::get_last_exclusive_window() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_last_exclusive_window")._native_ptr(), 1757182445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Window>(_gde_method_bind, _owner);
}

SceneTree *Node::get_tree() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_tree")._native_ptr(), 2958820483);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<SceneTree>(_gde_method_bind, _owner);
}

Ref<Tween> Node::create_tween() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("create_tween")._native_ptr(), 3426978995);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner));
}

Node *Node::duplicate(int32_t p_flags) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("duplicate")._native_ptr(), 3511555459);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner, &p_flags_encoded);
}

void Node::replace_by(Node *p_node, bool p_keep_groups) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("replace_by")._native_ptr(), 2570952461);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_groups_encoded;
	PtrToArg<bool>::encode(p_keep_groups, &p_keep_groups_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr), &p_keep_groups_encoded);
}

void Node::set_scene_instance_load_placeholder(bool p_load_placeholder) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_scene_instance_load_placeholder")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_load_placeholder_encoded;
	PtrToArg<bool>::encode(p_load_placeholder, &p_load_placeholder_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_load_placeholder_encoded);
}

bool Node::get_scene_instance_load_placeholder() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_scene_instance_load_placeholder")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_editable_instance(Node *p_node, bool p_is_editable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_editable_instance")._native_ptr(), 2731852923);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_is_editable_encoded;
	PtrToArg<bool>::encode(p_is_editable, &p_is_editable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr), &p_is_editable_encoded);
}

bool Node::is_editable_instance(Node *p_node) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_editable_instance")._native_ptr(), 3093956946);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr));
}

Viewport *Node::get_viewport() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_viewport")._native_ptr(), 3596683776);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Viewport>(_gde_method_bind, _owner);
}

void Node::queue_free() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("queue_free")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Node::request_ready() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("request_ready")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool Node::is_node_ready() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_node_ready")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node::set_multiplayer_authority(int32_t p_id, bool p_recursive) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_multiplayer_authority")._native_ptr(), 972357352);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int8_t p_recursive_encoded;
	PtrToArg<bool>::encode(p_recursive, &p_recursive_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id_encoded, &p_recursive_encoded);
}

int32_t Node::get_multiplayer_authority() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_multiplayer_authority")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Node::is_multiplayer_authority() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_multiplayer_authority")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<MultiplayerAPI> Node::get_multiplayer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_multiplayer")._native_ptr(), 406750475);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<MultiplayerAPI>()));
	return Ref<MultiplayerAPI>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<MultiplayerAPI>(_gde_method_bind, _owner));
}

void Node::rpc_config(const StringName &p_method, const Variant &p_config) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("rpc_config")._native_ptr(), 3776071444);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_method, &p_config);
}

Variant Node::get_node_rpc_config() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_node_rpc_config")._native_ptr(), 1214101251);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner);
}

void Node::set_editor_description(const String &p_editor_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_editor_description")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_editor_description);
}

String Node::get_editor_description() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("get_editor_description")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Node::set_unique_name_in_owner(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_unique_name_in_owner")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Node::is_unique_name_in_owner() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("is_unique_name_in_owner")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

String Node::atr(const String &p_message, const StringName &p_context) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("atr")._native_ptr(), 3344478075);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_message, &p_context);
}

String Node::atr_n(const String &p_message, const StringName &p_plural_message, int32_t p_n, const StringName &p_context) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("atr_n")._native_ptr(), 259354841);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_n_encoded;
	PtrToArg<int64_t>::encode(p_n, &p_n_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_message, &p_plural_message, &p_n_encoded, &p_context);
}

Error Node::rpc_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("rpc")._native_ptr(), 4047867050);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	GDExtensionCallError error;
	Variant ret;
	::godot::gdextension_interface::object_method_bind_call(_gde_method_bind, _owner, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count, &ret, &error);
	return VariantCaster<Error>::cast(ret);
}

Error Node::rpc_id_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("rpc_id")._native_ptr(), 361499283);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	GDExtensionCallError error;
	Variant ret;
	::godot::gdextension_interface::object_method_bind_call(_gde_method_bind, _owner, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count, &ret, &error);
	return VariantCaster<Error>::cast(ret);
}

void Node::update_configuration_warnings() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("update_configuration_warnings")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Variant Node::call_deferred_thread_group_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("call_deferred_thread_group")._native_ptr(), 3400424181);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	GDExtensionCallError error;
	Variant ret;
	::godot::gdextension_interface::object_method_bind_call(_gde_method_bind, _owner, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count, &ret, &error);
	return ret;
}

void Node::set_deferred_thread_group(const StringName &p_property, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_deferred_thread_group")._native_ptr(), 3776071444);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_property, &p_value);
}

void Node::notify_deferred_thread_group(int32_t p_what) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("notify_deferred_thread_group")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_what_encoded;
	PtrToArg<int64_t>::encode(p_what, &p_what_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_what_encoded);
}

Variant Node::call_thread_safe_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("call_thread_safe")._native_ptr(), 3400424181);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	GDExtensionCallError error;
	Variant ret;
	::godot::gdextension_interface::object_method_bind_call(_gde_method_bind, _owner, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count, &ret, &error);
	return ret;
}

void Node::set_thread_safe(const StringName &p_property, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("set_thread_safe")._native_ptr(), 3776071444);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_property, &p_value);
}

void Node::notify_thread_safe(int32_t p_what) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node::get_class_static()._native_ptr(), StringName("notify_thread_safe")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_what_encoded;
	PtrToArg<int64_t>::encode(p_what, &p_what_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_what_encoded);
}

void Node::_process(double p_delta) {}

void Node::_physics_process(double p_delta) {}

void Node::_enter_tree() {}

void Node::_exit_tree() {}

void Node::_ready() {}

PackedStringArray Node::_get_configuration_warnings() const {
	return PackedStringArray();
}

PackedStringArray Node::_get_accessibility_configuration_warnings() const {
	return PackedStringArray();
}

void Node::_input(const Ref<InputEvent> &p_event) {}

void Node::_shortcut_input(const Ref<InputEvent> &p_event) {}

void Node::_unhandled_input(const Ref<InputEvent> &p_event) {}

void Node::_unhandled_key_input(const Ref<InputEvent> &p_event) {}

RID Node::_get_focused_accessibility_element() const {
	return RID();
}

} // namespace godot
