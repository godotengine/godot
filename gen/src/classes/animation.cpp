/**************************************************************************/
/*  animation.cpp                                                         */
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

#include <godot_cpp/classes/animation.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

int32_t Animation::add_track(Animation::TrackType p_type, int32_t p_at_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("add_track")._native_ptr(), 3843682357);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_at_position_encoded;
	PtrToArg<int64_t>::encode(p_at_position, &p_at_position_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_type_encoded, &p_at_position_encoded);
}

void Animation::remove_track(int32_t p_track_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("remove_track")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded);
}

int32_t Animation::get_track_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("get_track_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Animation::TrackType Animation::track_get_type(int32_t p_track_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_get_type")._native_ptr(), 3445944217);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Animation::TrackType(0)));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	return (Animation::TrackType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded);
}

NodePath Animation::track_get_path(int32_t p_track_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_get_path")._native_ptr(), 408788394);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_track_idx_encoded);
}

void Animation::track_set_path(int32_t p_track_idx, const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_set_path")._native_ptr(), 2761262315);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_path);
}

int32_t Animation::find_track(const NodePath &p_path, Animation::TrackType p_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("find_track")._native_ptr(), 245376003);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_type_encoded);
}

void Animation::track_move_up(int32_t p_track_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_move_up")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded);
}

void Animation::track_move_down(int32_t p_track_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_move_down")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded);
}

void Animation::track_move_to(int32_t p_track_idx, int32_t p_to_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_move_to")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_to_idx_encoded;
	PtrToArg<int64_t>::encode(p_to_idx, &p_to_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_to_idx_encoded);
}

void Animation::track_swap(int32_t p_track_idx, int32_t p_with_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_swap")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_with_idx_encoded;
	PtrToArg<int64_t>::encode(p_with_idx, &p_with_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_with_idx_encoded);
}

void Animation::track_set_imported(int32_t p_track_idx, bool p_imported) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_set_imported")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int8_t p_imported_encoded;
	PtrToArg<bool>::encode(p_imported, &p_imported_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_imported_encoded);
}

bool Animation::track_is_imported(int32_t p_track_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_is_imported")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_track_idx_encoded);
}

void Animation::track_set_enabled(int32_t p_track_idx, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_set_enabled")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_enabled_encoded);
}

bool Animation::track_is_enabled(int32_t p_track_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_is_enabled")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_track_idx_encoded);
}

int32_t Animation::position_track_insert_key(int32_t p_track_idx, double p_time, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("position_track_insert_key")._native_ptr(), 2540608232);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_encoded, &p_position);
}

int32_t Animation::rotation_track_insert_key(int32_t p_track_idx, double p_time, const Quaternion &p_rotation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("rotation_track_insert_key")._native_ptr(), 4165004800);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_encoded, &p_rotation);
}

int32_t Animation::scale_track_insert_key(int32_t p_track_idx, double p_time, const Vector3 &p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("scale_track_insert_key")._native_ptr(), 2540608232);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_encoded, &p_scale);
}

int32_t Animation::blend_shape_track_insert_key(int32_t p_track_idx, double p_time, float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("blend_shape_track_insert_key")._native_ptr(), 1534913637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_encoded, &p_amount_encoded);
}

Vector3 Animation::position_track_interpolate(int32_t p_track_idx, double p_time_sec, bool p_backward) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("position_track_interpolate")._native_ptr(), 3530011197);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_sec_encoded;
	PtrToArg<double>::encode(p_time_sec, &p_time_sec_encoded);
	int8_t p_backward_encoded;
	PtrToArg<bool>::encode(p_backward, &p_backward_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_sec_encoded, &p_backward_encoded);
}

Quaternion Animation::rotation_track_interpolate(int32_t p_track_idx, double p_time_sec, bool p_backward) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("rotation_track_interpolate")._native_ptr(), 2915876792);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Quaternion()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_sec_encoded;
	PtrToArg<double>::encode(p_time_sec, &p_time_sec_encoded);
	int8_t p_backward_encoded;
	PtrToArg<bool>::encode(p_backward, &p_backward_encoded);
	return ::godot::internal::_call_native_mb_ret<Quaternion>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_sec_encoded, &p_backward_encoded);
}

Vector3 Animation::scale_track_interpolate(int32_t p_track_idx, double p_time_sec, bool p_backward) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("scale_track_interpolate")._native_ptr(), 3530011197);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_sec_encoded;
	PtrToArg<double>::encode(p_time_sec, &p_time_sec_encoded);
	int8_t p_backward_encoded;
	PtrToArg<bool>::encode(p_backward, &p_backward_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_sec_encoded, &p_backward_encoded);
}

float Animation::blend_shape_track_interpolate(int32_t p_track_idx, double p_time_sec, bool p_backward) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("blend_shape_track_interpolate")._native_ptr(), 2482365182);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_sec_encoded;
	PtrToArg<double>::encode(p_time_sec, &p_time_sec_encoded);
	int8_t p_backward_encoded;
	PtrToArg<bool>::encode(p_backward, &p_backward_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_sec_encoded, &p_backward_encoded);
}

int32_t Animation::track_insert_key(int32_t p_track_idx, double p_time, const Variant &p_key, float p_transition) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_insert_key")._native_ptr(), 808952278);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	double p_transition_encoded;
	PtrToArg<double>::encode(p_transition, &p_transition_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_encoded, &p_key, &p_transition_encoded);
}

void Animation::track_remove_key(int32_t p_track_idx, int32_t p_key_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_remove_key")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

void Animation::track_remove_key_at_time(int32_t p_track_idx, double p_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_remove_key_at_time")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_encoded);
}

void Animation::track_set_key_value(int32_t p_track_idx, int32_t p_key, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_set_key_value")._native_ptr(), 2060538656);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_encoded;
	PtrToArg<int64_t>::encode(p_key, &p_key_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_encoded, &p_value);
}

void Animation::track_set_key_transition(int32_t p_track_idx, int32_t p_key_idx, float p_transition) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_set_key_transition")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	double p_transition_encoded;
	PtrToArg<double>::encode(p_transition, &p_transition_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded, &p_transition_encoded);
}

void Animation::track_set_key_time(int32_t p_track_idx, int32_t p_key_idx, double p_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_set_key_time")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded, &p_time_encoded);
}

float Animation::track_get_key_transition(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_get_key_transition")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

int32_t Animation::track_get_key_count(int32_t p_track_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_get_key_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded);
}

Variant Animation::track_get_key_value(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_get_key_value")._native_ptr(), 678354945);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

double Animation::track_get_key_time(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_get_key_time")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

int32_t Animation::track_find_key(int32_t p_track_idx, double p_time, Animation::FindMode p_find_mode, bool p_limit, bool p_backward) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_find_key")._native_ptr(), 4230953007);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	int64_t p_find_mode_encoded;
	PtrToArg<int64_t>::encode(p_find_mode, &p_find_mode_encoded);
	int8_t p_limit_encoded;
	PtrToArg<bool>::encode(p_limit, &p_limit_encoded);
	int8_t p_backward_encoded;
	PtrToArg<bool>::encode(p_backward, &p_backward_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_encoded, &p_find_mode_encoded, &p_limit_encoded, &p_backward_encoded);
}

void Animation::track_set_interpolation_type(int32_t p_track_idx, Animation::InterpolationType p_interpolation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_set_interpolation_type")._native_ptr(), 4112932513);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_interpolation_encoded;
	PtrToArg<int64_t>::encode(p_interpolation, &p_interpolation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_interpolation_encoded);
}

Animation::InterpolationType Animation::track_get_interpolation_type(int32_t p_track_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_get_interpolation_type")._native_ptr(), 1530756894);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Animation::InterpolationType(0)));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	return (Animation::InterpolationType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded);
}

void Animation::track_set_interpolation_loop_wrap(int32_t p_track_idx, bool p_interpolation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_set_interpolation_loop_wrap")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int8_t p_interpolation_encoded;
	PtrToArg<bool>::encode(p_interpolation, &p_interpolation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_interpolation_encoded);
}

bool Animation::track_get_interpolation_loop_wrap(int32_t p_track_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_get_interpolation_loop_wrap")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_track_idx_encoded);
}

bool Animation::track_is_compressed(int32_t p_track_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("track_is_compressed")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_track_idx_encoded);
}

void Animation::value_track_set_update_mode(int32_t p_track_idx, Animation::UpdateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("value_track_set_update_mode")._native_ptr(), 2854058312);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_mode_encoded);
}

Animation::UpdateMode Animation::value_track_get_update_mode(int32_t p_track_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("value_track_get_update_mode")._native_ptr(), 1440326473);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Animation::UpdateMode(0)));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	return (Animation::UpdateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded);
}

Variant Animation::value_track_interpolate(int32_t p_track_idx, double p_time_sec, bool p_backward) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("value_track_interpolate")._native_ptr(), 747269075);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_sec_encoded;
	PtrToArg<double>::encode(p_time_sec, &p_time_sec_encoded);
	int8_t p_backward_encoded;
	PtrToArg<bool>::encode(p_backward, &p_backward_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_sec_encoded, &p_backward_encoded);
}

StringName Animation::method_track_get_name(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("method_track_get_name")._native_ptr(), 351665558);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

Array Animation::method_track_get_params(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("method_track_get_params")._native_ptr(), 2345056839);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

int32_t Animation::bezier_track_insert_key(int32_t p_track_idx, double p_time, float p_value, const Vector2 &p_in_handle, const Vector2 &p_out_handle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("bezier_track_insert_key")._native_ptr(), 3656773645);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_encoded, &p_value_encoded, &p_in_handle, &p_out_handle);
}

void Animation::bezier_track_set_key_value(int32_t p_track_idx, int32_t p_key_idx, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("bezier_track_set_key_value")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded, &p_value_encoded);
}

void Animation::bezier_track_set_key_in_handle(int32_t p_track_idx, int32_t p_key_idx, const Vector2 &p_in_handle, float p_balanced_value_time_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("bezier_track_set_key_in_handle")._native_ptr(), 1719223284);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	double p_balanced_value_time_ratio_encoded;
	PtrToArg<double>::encode(p_balanced_value_time_ratio, &p_balanced_value_time_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded, &p_in_handle, &p_balanced_value_time_ratio_encoded);
}

void Animation::bezier_track_set_key_out_handle(int32_t p_track_idx, int32_t p_key_idx, const Vector2 &p_out_handle, float p_balanced_value_time_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("bezier_track_set_key_out_handle")._native_ptr(), 1719223284);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	double p_balanced_value_time_ratio_encoded;
	PtrToArg<double>::encode(p_balanced_value_time_ratio, &p_balanced_value_time_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded, &p_out_handle, &p_balanced_value_time_ratio_encoded);
}

float Animation::bezier_track_get_key_value(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("bezier_track_get_key_value")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

Vector2 Animation::bezier_track_get_key_in_handle(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("bezier_track_get_key_in_handle")._native_ptr(), 3016396712);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

Vector2 Animation::bezier_track_get_key_out_handle(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("bezier_track_get_key_out_handle")._native_ptr(), 3016396712);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

float Animation::bezier_track_interpolate(int32_t p_track_idx, double p_time) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("bezier_track_interpolate")._native_ptr(), 1900462983);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_encoded);
}

int32_t Animation::audio_track_insert_key(int32_t p_track_idx, double p_time, const Ref<Resource> &p_stream, float p_start_offset, float p_end_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("audio_track_insert_key")._native_ptr(), 4021027286);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	double p_start_offset_encoded;
	PtrToArg<double>::encode(p_start_offset, &p_start_offset_encoded);
	double p_end_offset_encoded;
	PtrToArg<double>::encode(p_end_offset, &p_end_offset_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_encoded, (p_stream != nullptr ? &p_stream->_owner : nullptr), &p_start_offset_encoded, &p_end_offset_encoded);
}

void Animation::audio_track_set_key_stream(int32_t p_track_idx, int32_t p_key_idx, const Ref<Resource> &p_stream) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("audio_track_set_key_stream")._native_ptr(), 3886397084);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded, (p_stream != nullptr ? &p_stream->_owner : nullptr));
}

void Animation::audio_track_set_key_start_offset(int32_t p_track_idx, int32_t p_key_idx, float p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("audio_track_set_key_start_offset")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded, &p_offset_encoded);
}

void Animation::audio_track_set_key_end_offset(int32_t p_track_idx, int32_t p_key_idx, float p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("audio_track_set_key_end_offset")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded, &p_offset_encoded);
}

Ref<Resource> Animation::audio_track_get_key_stream(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("audio_track_get_key_stream")._native_ptr(), 635277205);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Resource>()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return Ref<Resource>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Resource>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded));
}

float Animation::audio_track_get_key_start_offset(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("audio_track_get_key_start_offset")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

float Animation::audio_track_get_key_end_offset(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("audio_track_get_key_end_offset")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

void Animation::audio_track_set_use_blend(int32_t p_track_idx, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("audio_track_set_use_blend")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_enable_encoded);
}

bool Animation::audio_track_is_use_blend(int32_t p_track_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("audio_track_is_use_blend")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_track_idx_encoded);
}

int32_t Animation::animation_track_insert_key(int32_t p_track_idx, double p_time, const StringName &p_animation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("animation_track_insert_key")._native_ptr(), 158676774);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_time_encoded, &p_animation);
}

void Animation::animation_track_set_key_animation(int32_t p_track_idx, int32_t p_key_idx, const StringName &p_animation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("animation_track_set_key_animation")._native_ptr(), 117615382);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded, &p_animation);
}

StringName Animation::animation_track_get_key_animation(int32_t p_track_idx, int32_t p_key_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("animation_track_get_key_animation")._native_ptr(), 351665558);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	int64_t p_key_idx_encoded;
	PtrToArg<int64_t>::encode(p_key_idx, &p_key_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_track_idx_encoded, &p_key_idx_encoded);
}

void Animation::add_marker(const StringName &p_name, double p_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("add_marker")._native_ptr(), 4135858297);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_time_encoded);
}

void Animation::remove_marker(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("remove_marker")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

bool Animation::has_marker(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("has_marker")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

StringName Animation::get_marker_at_time(double p_time) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("get_marker_at_time")._native_ptr(), 4079494655);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_time_encoded);
}

StringName Animation::get_next_marker(double p_time) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("get_next_marker")._native_ptr(), 4079494655);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_time_encoded);
}

StringName Animation::get_prev_marker(double p_time) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("get_prev_marker")._native_ptr(), 4079494655);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_time_encoded);
}

double Animation::get_marker_time(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("get_marker_time")._native_ptr(), 2349060816);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_name);
}

PackedStringArray Animation::get_marker_names() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("get_marker_names")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

Color Animation::get_marker_color(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("get_marker_color")._native_ptr(), 3742943038);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_name);
}

void Animation::set_marker_color(const StringName &p_name, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("set_marker_color")._native_ptr(), 4260178595);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_color);
}

void Animation::set_length(float p_time_sec) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("set_length")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_sec_encoded;
	PtrToArg<double>::encode(p_time_sec, &p_time_sec_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_time_sec_encoded);
}

float Animation::get_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("get_length")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Animation::set_loop_mode(Animation::LoopMode p_loop_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("set_loop_mode")._native_ptr(), 3155355575);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_loop_mode_encoded;
	PtrToArg<int64_t>::encode(p_loop_mode, &p_loop_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_loop_mode_encoded);
}

Animation::LoopMode Animation::get_loop_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("get_loop_mode")._native_ptr(), 1988889481);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Animation::LoopMode(0)));
	return (Animation::LoopMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Animation::set_step(float p_size_sec) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("set_step")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_sec_encoded;
	PtrToArg<double>::encode(p_size_sec, &p_size_sec_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_sec_encoded);
}

float Animation::get_step() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("get_step")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Animation::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Animation::copy_track(int32_t p_track_idx, const Ref<Animation> &p_to_animation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("copy_track")._native_ptr(), 148001024);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_idx_encoded;
	PtrToArg<int64_t>::encode(p_track_idx, &p_track_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_idx_encoded, (p_to_animation != nullptr ? &p_to_animation->_owner : nullptr));
}

void Animation::optimize(float p_allowed_velocity_err, float p_allowed_angular_err, int32_t p_precision) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("optimize")._native_ptr(), 3303583852);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_allowed_velocity_err_encoded;
	PtrToArg<double>::encode(p_allowed_velocity_err, &p_allowed_velocity_err_encoded);
	double p_allowed_angular_err_encoded;
	PtrToArg<double>::encode(p_allowed_angular_err, &p_allowed_angular_err_encoded);
	int64_t p_precision_encoded;
	PtrToArg<int64_t>::encode(p_precision, &p_precision_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allowed_velocity_err_encoded, &p_allowed_angular_err_encoded, &p_precision_encoded);
}

void Animation::compress(uint32_t p_page_size, uint32_t p_fps, float p_split_tolerance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("compress")._native_ptr(), 3608408117);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_page_size_encoded;
	PtrToArg<int64_t>::encode(p_page_size, &p_page_size_encoded);
	int64_t p_fps_encoded;
	PtrToArg<int64_t>::encode(p_fps, &p_fps_encoded);
	double p_split_tolerance_encoded;
	PtrToArg<double>::encode(p_split_tolerance, &p_split_tolerance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_page_size_encoded, &p_fps_encoded, &p_split_tolerance_encoded);
}

bool Animation::is_capture_included() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Animation::get_class_static()._native_ptr(), StringName("is_capture_included")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
