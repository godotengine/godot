/**************************************************************************/
/*  animation_node_state_machine.cpp                                      */
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

#include <godot_cpp/classes/animation_node_state_machine.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/animation_node.hpp>
#include <godot_cpp/classes/animation_node_state_machine_transition.hpp>

namespace godot {

void AnimationNodeStateMachine::add_node(const StringName &p_name, const Ref<AnimationNode> &p_node, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("add_node")._native_ptr(), 1980270704);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_node != nullptr ? &p_node->_owner : nullptr), &p_position);
}

void AnimationNodeStateMachine::replace_node(const StringName &p_name, const Ref<AnimationNode> &p_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("replace_node")._native_ptr(), 2559412862);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_node != nullptr ? &p_node->_owner : nullptr));
}

Ref<AnimationNode> AnimationNodeStateMachine::get_node(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("get_node")._native_ptr(), 625644256);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AnimationNode>()));
	return Ref<AnimationNode>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AnimationNode>(_gde_method_bind, _owner, &p_name));
}

void AnimationNodeStateMachine::remove_node(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("remove_node")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void AnimationNodeStateMachine::rename_node(const StringName &p_name, const StringName &p_new_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("rename_node")._native_ptr(), 3740211285);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_new_name);
}

bool AnimationNodeStateMachine::has_node(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("has_node")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

StringName AnimationNodeStateMachine::get_node_name(const Ref<AnimationNode> &p_node) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("get_node_name")._native_ptr(), 739213945);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr));
}

TypedArray<StringName> AnimationNodeStateMachine::get_node_list() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("get_node_list")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<StringName>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<StringName>>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachine::set_node_position(const StringName &p_name, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("set_node_position")._native_ptr(), 1999414630);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_position);
}

Vector2 AnimationNodeStateMachine::get_node_position(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("get_node_position")._native_ptr(), 3100822709);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_name);
}

bool AnimationNodeStateMachine::has_transition(const StringName &p_from, const StringName &p_to) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("has_transition")._native_ptr(), 471820014);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_from, &p_to);
}

void AnimationNodeStateMachine::add_transition(const StringName &p_from, const StringName &p_to, const Ref<AnimationNodeStateMachineTransition> &p_transition) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("add_transition")._native_ptr(), 795486887);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from, &p_to, (p_transition != nullptr ? &p_transition->_owner : nullptr));
}

Ref<AnimationNodeStateMachineTransition> AnimationNodeStateMachine::get_transition(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("get_transition")._native_ptr(), 4192381260);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AnimationNodeStateMachineTransition>()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return Ref<AnimationNodeStateMachineTransition>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AnimationNodeStateMachineTransition>(_gde_method_bind, _owner, &p_idx_encoded));
}

StringName AnimationNodeStateMachine::get_transition_from(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("get_transition_from")._native_ptr(), 659327637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_idx_encoded);
}

StringName AnimationNodeStateMachine::get_transition_to(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("get_transition_to")._native_ptr(), 659327637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_idx_encoded);
}

int32_t AnimationNodeStateMachine::get_transition_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("get_transition_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachine::remove_transition_by_index(int32_t p_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("remove_transition_by_index")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded);
}

void AnimationNodeStateMachine::remove_transition(const StringName &p_from, const StringName &p_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("remove_transition")._native_ptr(), 3740211285);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from, &p_to);
}

void AnimationNodeStateMachine::set_graph_offset(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("set_graph_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector2 AnimationNodeStateMachine::get_graph_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("get_graph_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachine::set_state_machine_type(AnimationNodeStateMachine::StateMachineType p_state_machine_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("set_state_machine_type")._native_ptr(), 2584759088);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_state_machine_type_encoded;
	PtrToArg<int64_t>::encode(p_state_machine_type, &p_state_machine_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_state_machine_type_encoded);
}

AnimationNodeStateMachine::StateMachineType AnimationNodeStateMachine::get_state_machine_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("get_state_machine_type")._native_ptr(), 1140726469);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AnimationNodeStateMachine::StateMachineType(0)));
	return (AnimationNodeStateMachine::StateMachineType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachine::set_allow_transition_to_self(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("set_allow_transition_to_self")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool AnimationNodeStateMachine::is_allow_transition_to_self() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("is_allow_transition_to_self")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachine::set_reset_ends(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("set_reset_ends")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool AnimationNodeStateMachine::are_ends_reset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachine::get_class_static()._native_ptr(), StringName("are_ends_reset")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
