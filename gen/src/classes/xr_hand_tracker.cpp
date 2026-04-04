/**************************************************************************/
/*  xr_hand_tracker.cpp                                                   */
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

#include <godot_cpp/classes/xr_hand_tracker.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void XRHandTracker::set_has_tracking_data(bool p_has_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("set_has_tracking_data")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_has_data_encoded;
	PtrToArg<bool>::encode(p_has_data, &p_has_data_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_has_data_encoded);
}

bool XRHandTracker::get_has_tracking_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("get_has_tracking_data")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void XRHandTracker::set_hand_tracking_source(XRHandTracker::HandTrackingSource p_source) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("set_hand_tracking_source")._native_ptr(), 2958308861);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_encoded;
	PtrToArg<int64_t>::encode(p_source, &p_source_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_encoded);
}

XRHandTracker::HandTrackingSource XRHandTracker::get_hand_tracking_source() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("get_hand_tracking_source")._native_ptr(), 2475045250);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (XRHandTracker::HandTrackingSource(0)));
	return (XRHandTracker::HandTrackingSource)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void XRHandTracker::set_hand_joint_flags(XRHandTracker::HandJoint p_joint, BitField<XRHandTracker::HandJointFlags> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("set_hand_joint_flags")._native_ptr(), 3028437365);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_encoded, &p_flags);
}

BitField<XRHandTracker::HandJointFlags> XRHandTracker::get_hand_joint_flags(XRHandTracker::HandJoint p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("get_hand_joint_flags")._native_ptr(), 1730972401);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<XRHandTracker::HandJointFlags>(0)));
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_joint_encoded);
}

void XRHandTracker::set_hand_joint_transform(XRHandTracker::HandJoint p_joint, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("set_hand_joint_transform")._native_ptr(), 2529959613);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_encoded, &p_transform);
}

Transform3D XRHandTracker::get_hand_joint_transform(XRHandTracker::HandJoint p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("get_hand_joint_transform")._native_ptr(), 1090840196);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_joint_encoded);
}

void XRHandTracker::set_hand_joint_radius(XRHandTracker::HandJoint p_joint, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("set_hand_joint_radius")._native_ptr(), 2723659615);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_encoded, &p_radius_encoded);
}

float XRHandTracker::get_hand_joint_radius(XRHandTracker::HandJoint p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("get_hand_joint_radius")._native_ptr(), 3400025734);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_joint_encoded);
}

void XRHandTracker::set_hand_joint_linear_velocity(XRHandTracker::HandJoint p_joint, const Vector3 &p_linear_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("set_hand_joint_linear_velocity")._native_ptr(), 1978646737);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_encoded, &p_linear_velocity);
}

Vector3 XRHandTracker::get_hand_joint_linear_velocity(XRHandTracker::HandJoint p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("get_hand_joint_linear_velocity")._native_ptr(), 547240792);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_joint_encoded);
}

void XRHandTracker::set_hand_joint_angular_velocity(XRHandTracker::HandJoint p_joint, const Vector3 &p_angular_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("set_hand_joint_angular_velocity")._native_ptr(), 1978646737);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_encoded, &p_angular_velocity);
}

Vector3 XRHandTracker::get_hand_joint_angular_velocity(XRHandTracker::HandJoint p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRHandTracker::get_class_static()._native_ptr(), StringName("get_hand_joint_angular_velocity")._native_ptr(), 547240792);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_joint_encoded);
}

} // namespace godot
