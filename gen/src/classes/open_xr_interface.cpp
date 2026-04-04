/**************************************************************************/
/*  open_xr_interface.cpp                                                 */
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

#include <godot_cpp/classes/open_xr_interface.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/string.hpp>

namespace godot {

OpenXRInterface::SessionState OpenXRInterface::get_session_state() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_session_state")._native_ptr(), 896364779);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRInterface::SessionState(0)));
	return (OpenXRInterface::SessionState)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

float OpenXRInterface::get_display_refresh_rate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_display_refresh_rate")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRInterface::set_display_refresh_rate(float p_refresh_rate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("set_display_refresh_rate")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_refresh_rate_encoded;
	PtrToArg<double>::encode(p_refresh_rate, &p_refresh_rate_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_refresh_rate_encoded);
}

double OpenXRInterface::get_render_target_size_multiplier() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_render_target_size_multiplier")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRInterface::set_render_target_size_multiplier(double p_multiplier) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("set_render_target_size_multiplier")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_multiplier_encoded;
	PtrToArg<double>::encode(p_multiplier, &p_multiplier_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multiplier_encoded);
}

bool OpenXRInterface::is_foveation_supported() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("is_foveation_supported")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t OpenXRInterface::get_foveation_level() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_foveation_level")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRInterface::set_foveation_level(int32_t p_foveation_level) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("set_foveation_level")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_foveation_level_encoded;
	PtrToArg<int64_t>::encode(p_foveation_level, &p_foveation_level_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_foveation_level_encoded);
}

bool OpenXRInterface::get_foveation_dynamic() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_foveation_dynamic")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void OpenXRInterface::set_foveation_dynamic(bool p_foveation_dynamic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("set_foveation_dynamic")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_foveation_dynamic_encoded;
	PtrToArg<bool>::encode(p_foveation_dynamic, &p_foveation_dynamic_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_foveation_dynamic_encoded);
}

bool OpenXRInterface::is_action_set_active(const String &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("is_action_set_active")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

void OpenXRInterface::set_action_set_active(const String &p_name, bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("set_action_set_active")._native_ptr(), 2678287736);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_active_encoded);
}

Array OpenXRInterface::get_action_sets() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_action_sets")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

Array OpenXRInterface::get_available_display_refresh_rates() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_available_display_refresh_rates")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

void OpenXRInterface::set_motion_range(OpenXRInterface::Hand p_hand, OpenXRInterface::HandMotionRange p_motion_range) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("set_motion_range")._native_ptr(), 855158159);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_hand_encoded;
	PtrToArg<int64_t>::encode(p_hand, &p_hand_encoded);
	int64_t p_motion_range_encoded;
	PtrToArg<int64_t>::encode(p_motion_range, &p_motion_range_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hand_encoded, &p_motion_range_encoded);
}

OpenXRInterface::HandMotionRange OpenXRInterface::get_motion_range(OpenXRInterface::Hand p_hand) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_motion_range")._native_ptr(), 3955838114);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRInterface::HandMotionRange(0)));
	int64_t p_hand_encoded;
	PtrToArg<int64_t>::encode(p_hand, &p_hand_encoded);
	return (OpenXRInterface::HandMotionRange)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_hand_encoded);
}

OpenXRInterface::HandTrackedSource OpenXRInterface::get_hand_tracking_source(OpenXRInterface::Hand p_hand) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_hand_tracking_source")._native_ptr(), 4092421202);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRInterface::HandTrackedSource(0)));
	int64_t p_hand_encoded;
	PtrToArg<int64_t>::encode(p_hand, &p_hand_encoded);
	return (OpenXRInterface::HandTrackedSource)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_hand_encoded);
}

BitField<OpenXRInterface::HandJointFlags> OpenXRInterface::get_hand_joint_flags(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_hand_joint_flags")._native_ptr(), 720567706);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<OpenXRInterface::HandJointFlags>(0)));
	int64_t p_hand_encoded;
	PtrToArg<int64_t>::encode(p_hand, &p_hand_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_hand_encoded, &p_joint_encoded);
}

Quaternion OpenXRInterface::get_hand_joint_rotation(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_hand_joint_rotation")._native_ptr(), 1974618321);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Quaternion()));
	int64_t p_hand_encoded;
	PtrToArg<int64_t>::encode(p_hand, &p_hand_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Quaternion>(_gde_method_bind, _owner, &p_hand_encoded, &p_joint_encoded);
}

Vector3 OpenXRInterface::get_hand_joint_position(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_hand_joint_position")._native_ptr(), 3529194242);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_hand_encoded;
	PtrToArg<int64_t>::encode(p_hand, &p_hand_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_hand_encoded, &p_joint_encoded);
}

float OpenXRInterface::get_hand_joint_radius(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_hand_joint_radius")._native_ptr(), 901522724);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_hand_encoded;
	PtrToArg<int64_t>::encode(p_hand, &p_hand_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_hand_encoded, &p_joint_encoded);
}

Vector3 OpenXRInterface::get_hand_joint_linear_velocity(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_hand_joint_linear_velocity")._native_ptr(), 3529194242);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_hand_encoded;
	PtrToArg<int64_t>::encode(p_hand, &p_hand_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_hand_encoded, &p_joint_encoded);
}

Vector3 OpenXRInterface::get_hand_joint_angular_velocity(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_hand_joint_angular_velocity")._native_ptr(), 3529194242);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_hand_encoded;
	PtrToArg<int64_t>::encode(p_hand, &p_hand_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_hand_encoded, &p_joint_encoded);
}

bool OpenXRInterface::is_hand_tracking_supported() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("is_hand_tracking_supported")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool OpenXRInterface::is_hand_interaction_supported() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("is_hand_interaction_supported")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool OpenXRInterface::is_eye_gaze_interaction_supported() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("is_eye_gaze_interaction_supported")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

float OpenXRInterface::get_vrs_min_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_vrs_min_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRInterface::set_vrs_min_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("set_vrs_min_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float OpenXRInterface::get_vrs_strength() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("get_vrs_strength")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRInterface::set_vrs_strength(float p_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("set_vrs_strength")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_strength_encoded);
}

void OpenXRInterface::set_cpu_level(OpenXRInterface::PerfSettingsLevel p_level) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("set_cpu_level")._native_ptr(), 2940842095);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_level_encoded;
	PtrToArg<int64_t>::encode(p_level, &p_level_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_level_encoded);
}

void OpenXRInterface::set_gpu_level(OpenXRInterface::PerfSettingsLevel p_level) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInterface::get_class_static()._native_ptr(), StringName("set_gpu_level")._native_ptr(), 2940842095);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_level_encoded;
	PtrToArg<int64_t>::encode(p_level, &p_level_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_level_encoded);
}

} // namespace godot
