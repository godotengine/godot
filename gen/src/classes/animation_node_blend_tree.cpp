/**************************************************************************/
/*  animation_node_blend_tree.cpp                                         */
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

#include <godot_cpp/classes/animation_node_blend_tree.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/animation_node.hpp>

namespace godot {

void AnimationNodeBlendTree::add_node(const StringName &p_name, const Ref<AnimationNode> &p_node, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("add_node")._native_ptr(), 1980270704);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_node != nullptr ? &p_node->_owner : nullptr), &p_position);
}

Ref<AnimationNode> AnimationNodeBlendTree::get_node(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("get_node")._native_ptr(), 625644256);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AnimationNode>()));
	return Ref<AnimationNode>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AnimationNode>(_gde_method_bind, _owner, &p_name));
}

void AnimationNodeBlendTree::remove_node(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("remove_node")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void AnimationNodeBlendTree::rename_node(const StringName &p_name, const StringName &p_new_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("rename_node")._native_ptr(), 3740211285);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_new_name);
}

bool AnimationNodeBlendTree::has_node(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("has_node")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

void AnimationNodeBlendTree::connect_node(const StringName &p_input_node, int32_t p_input_index, const StringName &p_output_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("connect_node")._native_ptr(), 2168001410);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_input_index_encoded;
	PtrToArg<int64_t>::encode(p_input_index, &p_input_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_input_node, &p_input_index_encoded, &p_output_node);
}

void AnimationNodeBlendTree::disconnect_node(const StringName &p_input_node, int32_t p_input_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("disconnect_node")._native_ptr(), 2415702435);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_input_index_encoded;
	PtrToArg<int64_t>::encode(p_input_index, &p_input_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_input_node, &p_input_index_encoded);
}

TypedArray<StringName> AnimationNodeBlendTree::get_node_list() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("get_node_list")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<StringName>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<StringName>>(_gde_method_bind, _owner);
}

void AnimationNodeBlendTree::set_node_position(const StringName &p_name, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("set_node_position")._native_ptr(), 1999414630);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_position);
}

Vector2 AnimationNodeBlendTree::get_node_position(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("get_node_position")._native_ptr(), 3100822709);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_name);
}

void AnimationNodeBlendTree::set_graph_offset(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("set_graph_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector2 AnimationNodeBlendTree::get_graph_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendTree::get_class_static()._native_ptr(), StringName("get_graph_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

} // namespace godot
