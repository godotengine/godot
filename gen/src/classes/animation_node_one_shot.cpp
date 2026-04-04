/**************************************************************************/
/*  animation_node_one_shot.cpp                                           */
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

#include <godot_cpp/classes/animation_node_one_shot.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/curve.hpp>

namespace godot {

void AnimationNodeOneShot::set_fadein_time(double p_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("set_fadein_time")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_time_encoded);
}

double AnimationNodeOneShot::get_fadein_time() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("get_fadein_time")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AnimationNodeOneShot::set_fadein_curve(const Ref<Curve> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("set_fadein_curve")._native_ptr(), 270443179);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Curve> AnimationNodeOneShot::get_fadein_curve() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("get_fadein_curve")._native_ptr(), 2460114913);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner));
}

void AnimationNodeOneShot::set_fadeout_time(double p_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("set_fadeout_time")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_time_encoded);
}

double AnimationNodeOneShot::get_fadeout_time() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("get_fadeout_time")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AnimationNodeOneShot::set_fadeout_curve(const Ref<Curve> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("set_fadeout_curve")._native_ptr(), 270443179);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Curve> AnimationNodeOneShot::get_fadeout_curve() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("get_fadeout_curve")._native_ptr(), 2460114913);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner));
}

void AnimationNodeOneShot::set_break_loop_at_end(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("set_break_loop_at_end")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool AnimationNodeOneShot::is_loop_broken_at_end() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("is_loop_broken_at_end")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimationNodeOneShot::set_abort_on_reset(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("set_abort_on_reset")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool AnimationNodeOneShot::is_aborted_on_reset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("is_aborted_on_reset")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimationNodeOneShot::set_autorestart(bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("set_autorestart")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_active_encoded);
}

bool AnimationNodeOneShot::has_autorestart() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("has_autorestart")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimationNodeOneShot::set_autorestart_delay(double p_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("set_autorestart_delay")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_time_encoded);
}

double AnimationNodeOneShot::get_autorestart_delay() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("get_autorestart_delay")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AnimationNodeOneShot::set_autorestart_random_delay(double p_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("set_autorestart_random_delay")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_time_encoded);
}

double AnimationNodeOneShot::get_autorestart_random_delay() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("get_autorestart_random_delay")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AnimationNodeOneShot::set_mix_mode(AnimationNodeOneShot::MixMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("set_mix_mode")._native_ptr(), 1018899799);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

AnimationNodeOneShot::MixMode AnimationNodeOneShot::get_mix_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeOneShot::get_class_static()._native_ptr(), StringName("get_mix_mode")._native_ptr(), 3076550526);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AnimationNodeOneShot::MixMode(0)));
	return (AnimationNodeOneShot::MixMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
