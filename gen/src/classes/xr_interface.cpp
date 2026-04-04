/**************************************************************************/
/*  xr_interface.cpp                                                      */
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

#include <godot_cpp/classes/xr_interface.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/string.hpp>

namespace godot {

StringName XRInterface::get_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_name")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

uint32_t XRInterface::get_capabilities() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_capabilities")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool XRInterface::is_primary() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("is_primary")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void XRInterface::set_primary(bool p_primary) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("set_primary")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_primary_encoded;
	PtrToArg<bool>::encode(p_primary, &p_primary_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_primary_encoded);
}

bool XRInterface::is_initialized() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("is_initialized")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool XRInterface::initialize() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("initialize")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void XRInterface::uninitialize() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("uninitialize")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Dictionary XRInterface::get_system_info() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_system_info")._native_ptr(), 2382534195);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

XRInterface::TrackingStatus XRInterface::get_tracking_status() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_tracking_status")._native_ptr(), 167423259);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (XRInterface::TrackingStatus(0)));
	return (XRInterface::TrackingStatus)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Vector2 XRInterface::get_render_target_size() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_render_target_size")._native_ptr(), 1497962370);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

uint32_t XRInterface::get_view_count() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_view_count")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void XRInterface::trigger_haptic_pulse(const String &p_action_name, const StringName &p_tracker_name, double p_frequency, double p_amplitude, double p_duration_sec, double p_delay_sec) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("trigger_haptic_pulse")._native_ptr(), 3752640163);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_frequency_encoded;
	PtrToArg<double>::encode(p_frequency, &p_frequency_encoded);
	double p_amplitude_encoded;
	PtrToArg<double>::encode(p_amplitude, &p_amplitude_encoded);
	double p_duration_sec_encoded;
	PtrToArg<double>::encode(p_duration_sec, &p_duration_sec_encoded);
	double p_delay_sec_encoded;
	PtrToArg<double>::encode(p_delay_sec, &p_delay_sec_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action_name, &p_tracker_name, &p_frequency_encoded, &p_amplitude_encoded, &p_duration_sec_encoded, &p_delay_sec_encoded);
}

bool XRInterface::supports_play_area_mode(XRInterface::PlayAreaMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("supports_play_area_mode")._native_ptr(), 3429955281);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_mode_encoded);
}

XRInterface::PlayAreaMode XRInterface::get_play_area_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_play_area_mode")._native_ptr(), 1615132885);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (XRInterface::PlayAreaMode(0)));
	return (XRInterface::PlayAreaMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool XRInterface::set_play_area_mode(XRInterface::PlayAreaMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("set_play_area_mode")._native_ptr(), 3429955281);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_mode_encoded);
}

PackedVector3Array XRInterface::get_play_area() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_play_area")._native_ptr(), 497664490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner);
}

bool XRInterface::get_anchor_detection_is_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_anchor_detection_is_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void XRInterface::set_anchor_detection_is_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("set_anchor_detection_is_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

int32_t XRInterface::get_camera_feed_id() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_camera_feed_id")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool XRInterface::is_passthrough_supported() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("is_passthrough_supported")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool XRInterface::is_passthrough_enabled() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("is_passthrough_enabled")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool XRInterface::start_passthrough() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("start_passthrough")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void XRInterface::stop_passthrough() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("stop_passthrough")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Transform3D XRInterface::get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_transform_for_view")._native_ptr(), 518934792);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	int64_t p_view_encoded;
	PtrToArg<int64_t>::encode(p_view, &p_view_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_view_encoded, &p_cam_transform);
}

Projection XRInterface::get_projection_for_view(uint32_t p_view, double p_aspect, double p_near, double p_far) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_projection_for_view")._native_ptr(), 3766090294);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Projection()));
	int64_t p_view_encoded;
	PtrToArg<int64_t>::encode(p_view, &p_view_encoded);
	double p_aspect_encoded;
	PtrToArg<double>::encode(p_aspect, &p_aspect_encoded);
	double p_near_encoded;
	PtrToArg<double>::encode(p_near, &p_near_encoded);
	double p_far_encoded;
	PtrToArg<double>::encode(p_far, &p_far_encoded);
	return ::godot::internal::_call_native_mb_ret<Projection>(_gde_method_bind, _owner, &p_view_encoded, &p_aspect_encoded, &p_near_encoded, &p_far_encoded);
}

Array XRInterface::get_supported_environment_blend_modes() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_supported_environment_blend_modes")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

bool XRInterface::set_environment_blend_mode(XRInterface::EnvironmentBlendMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("set_environment_blend_mode")._native_ptr(), 551152418);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_mode_encoded);
}

XRInterface::EnvironmentBlendMode XRInterface::get_environment_blend_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterface::get_class_static()._native_ptr(), StringName("get_environment_blend_mode")._native_ptr(), 1984334071);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (XRInterface::EnvironmentBlendMode(0)));
	return (XRInterface::EnvironmentBlendMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
