/**************************************************************************/
/*  animation_node_state_machine_transition.cpp                           */
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

#include <godot_cpp/classes/animation_node_state_machine_transition.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/curve.hpp>

namespace godot {

void AnimationNodeStateMachineTransition::set_switch_mode(AnimationNodeStateMachineTransition::SwitchMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("set_switch_mode")._native_ptr(), 2074906633);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

AnimationNodeStateMachineTransition::SwitchMode AnimationNodeStateMachineTransition::get_switch_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("get_switch_mode")._native_ptr(), 2138562085);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AnimationNodeStateMachineTransition::SwitchMode(0)));
	return (AnimationNodeStateMachineTransition::SwitchMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachineTransition::set_advance_mode(AnimationNodeStateMachineTransition::AdvanceMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("set_advance_mode")._native_ptr(), 1210869868);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

AnimationNodeStateMachineTransition::AdvanceMode AnimationNodeStateMachineTransition::get_advance_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("get_advance_mode")._native_ptr(), 61101689);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AnimationNodeStateMachineTransition::AdvanceMode(0)));
	return (AnimationNodeStateMachineTransition::AdvanceMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachineTransition::set_advance_condition(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("set_advance_condition")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

StringName AnimationNodeStateMachineTransition::get_advance_condition() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("get_advance_condition")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachineTransition::set_xfade_time(float p_secs) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("set_xfade_time")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_secs_encoded;
	PtrToArg<double>::encode(p_secs, &p_secs_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_secs_encoded);
}

float AnimationNodeStateMachineTransition::get_xfade_time() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("get_xfade_time")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachineTransition::set_xfade_curve(const Ref<Curve> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("set_xfade_curve")._native_ptr(), 270443179);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Curve> AnimationNodeStateMachineTransition::get_xfade_curve() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("get_xfade_curve")._native_ptr(), 2460114913);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner));
}

void AnimationNodeStateMachineTransition::set_break_loop_at_end(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("set_break_loop_at_end")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool AnimationNodeStateMachineTransition::is_loop_broken_at_end() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("is_loop_broken_at_end")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachineTransition::set_reset(bool p_reset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("set_reset")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_reset_encoded;
	PtrToArg<bool>::encode(p_reset, &p_reset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_reset_encoded);
}

bool AnimationNodeStateMachineTransition::is_reset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("is_reset")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachineTransition::set_priority(int32_t p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("set_priority")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_priority_encoded;
	PtrToArg<int64_t>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_priority_encoded);
}

int32_t AnimationNodeStateMachineTransition::get_priority() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("get_priority")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationNodeStateMachineTransition::set_advance_expression(const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("set_advance_expression")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text);
}

String AnimationNodeStateMachineTransition::get_advance_expression() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeStateMachineTransition::get_class_static()._native_ptr(), StringName("get_advance_expression")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

} // namespace godot
