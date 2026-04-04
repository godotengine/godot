/**************************************************************************/
/*  tween.cpp                                                             */
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

#include <godot_cpp/classes/tween.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/callback_tweener.hpp>
#include <godot_cpp/classes/interval_tweener.hpp>
#include <godot_cpp/classes/method_tweener.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/property_tweener.hpp>
#include <godot_cpp/classes/subtween_tweener.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/node_path.hpp>

namespace godot {

Ref<PropertyTweener> Tween::tween_property(Object *p_object, const NodePath &p_property, const Variant &p_final_val, double p_duration) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("tween_property")._native_ptr(), 4049770449);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<PropertyTweener>()));
	double p_duration_encoded;
	PtrToArg<double>::encode(p_duration, &p_duration_encoded);
	return Ref<PropertyTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<PropertyTweener>(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr), &p_property, &p_final_val, &p_duration_encoded));
}

Ref<IntervalTweener> Tween::tween_interval(double p_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("tween_interval")._native_ptr(), 413360199);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<IntervalTweener>()));
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	return Ref<IntervalTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<IntervalTweener>(_gde_method_bind, _owner, &p_time_encoded));
}

Ref<CallbackTweener> Tween::tween_callback(const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("tween_callback")._native_ptr(), 1540176488);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<CallbackTweener>()));
	return Ref<CallbackTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<CallbackTweener>(_gde_method_bind, _owner, &p_callback));
}

Ref<MethodTweener> Tween::tween_method(const Callable &p_method, const Variant &p_from, const Variant &p_to, double p_duration) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("tween_method")._native_ptr(), 2337877153);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<MethodTweener>()));
	double p_duration_encoded;
	PtrToArg<double>::encode(p_duration, &p_duration_encoded);
	return Ref<MethodTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<MethodTweener>(_gde_method_bind, _owner, &p_method, &p_from, &p_to, &p_duration_encoded));
}

Ref<SubtweenTweener> Tween::tween_subtween(const Ref<Tween> &p_subtween) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("tween_subtween")._native_ptr(), 1567358477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SubtweenTweener>()));
	return Ref<SubtweenTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SubtweenTweener>(_gde_method_bind, _owner, (p_subtween != nullptr ? &p_subtween->_owner : nullptr)));
}

bool Tween::custom_step(double p_delta) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("custom_step")._native_ptr(), 330693286);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	double p_delta_encoded;
	PtrToArg<double>::encode(p_delta, &p_delta_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_delta_encoded);
}

void Tween::stop() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("stop")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Tween::pause() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("pause")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Tween::play() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("play")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Tween::kill() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("kill")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

double Tween::get_total_elapsed_time() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("get_total_elapsed_time")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool Tween::is_running() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("is_running")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool Tween::is_valid() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("is_valid")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<Tween> Tween::bind_node(Node *p_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("bind_node")._native_ptr(), 2946786331);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr)));
}

Ref<Tween> Tween::set_process_mode(Tween::TweenProcessMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("set_process_mode")._native_ptr(), 855258840);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner, &p_mode_encoded));
}

Ref<Tween> Tween::set_pause_mode(Tween::TweenPauseMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("set_pause_mode")._native_ptr(), 3363368837);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner, &p_mode_encoded));
}

Ref<Tween> Tween::set_ignore_time_scale(bool p_ignore) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("set_ignore_time_scale")._native_ptr(), 1942052223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	int8_t p_ignore_encoded;
	PtrToArg<bool>::encode(p_ignore, &p_ignore_encoded);
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner, &p_ignore_encoded));
}

Ref<Tween> Tween::set_parallel(bool p_parallel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("set_parallel")._native_ptr(), 1942052223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	int8_t p_parallel_encoded;
	PtrToArg<bool>::encode(p_parallel, &p_parallel_encoded);
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner, &p_parallel_encoded));
}

Ref<Tween> Tween::set_loops(int32_t p_loops) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("set_loops")._native_ptr(), 2670836414);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	int64_t p_loops_encoded;
	PtrToArg<int64_t>::encode(p_loops, &p_loops_encoded);
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner, &p_loops_encoded));
}

int32_t Tween::get_loops_left() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("get_loops_left")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<Tween> Tween::set_speed_scale(float p_speed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("set_speed_scale")._native_ptr(), 3961971106);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	double p_speed_encoded;
	PtrToArg<double>::encode(p_speed, &p_speed_encoded);
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner, &p_speed_encoded));
}

Ref<Tween> Tween::set_trans(Tween::TransitionType p_trans) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("set_trans")._native_ptr(), 3965963875);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	int64_t p_trans_encoded;
	PtrToArg<int64_t>::encode(p_trans, &p_trans_encoded);
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner, &p_trans_encoded));
}

Ref<Tween> Tween::set_ease(Tween::EaseType p_ease) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("set_ease")._native_ptr(), 1208117252);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	int64_t p_ease_encoded;
	PtrToArg<int64_t>::encode(p_ease, &p_ease_encoded);
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner, &p_ease_encoded));
}

Ref<Tween> Tween::parallel() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("parallel")._native_ptr(), 3426978995);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner));
}

Ref<Tween> Tween::chain() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("chain")._native_ptr(), 3426978995);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Tween>()));
	return Ref<Tween>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Tween>(_gde_method_bind, _owner));
}

Variant Tween::interpolate_value(const Variant &p_initial_value, const Variant &p_delta_value, double p_elapsed_time, double p_duration, Tween::TransitionType p_trans_type, Tween::EaseType p_ease_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tween::get_class_static()._native_ptr(), StringName("interpolate_value")._native_ptr(), 3452526450);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	double p_elapsed_time_encoded;
	PtrToArg<double>::encode(p_elapsed_time, &p_elapsed_time_encoded);
	double p_duration_encoded;
	PtrToArg<double>::encode(p_duration, &p_duration_encoded);
	int64_t p_trans_type_encoded;
	PtrToArg<int64_t>::encode(p_trans_type, &p_trans_type_encoded);
	int64_t p_ease_type_encoded;
	PtrToArg<int64_t>::encode(p_ease_type, &p_ease_type_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, nullptr, &p_initial_value, &p_delta_value, &p_elapsed_time_encoded, &p_duration_encoded, &p_trans_type_encoded, &p_ease_type_encoded);
}

} // namespace godot
