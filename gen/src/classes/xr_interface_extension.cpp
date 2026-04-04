/**************************************************************************/
/*  xr_interface_extension.cpp                                            */
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

#include <godot_cpp/classes/xr_interface_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/rect2i.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

RID XRInterfaceExtension::get_color_texture() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterfaceExtension::get_class_static()._native_ptr(), StringName("get_color_texture")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID XRInterfaceExtension::get_depth_texture() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterfaceExtension::get_class_static()._native_ptr(), StringName("get_depth_texture")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID XRInterfaceExtension::get_velocity_texture() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterfaceExtension::get_class_static()._native_ptr(), StringName("get_velocity_texture")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void XRInterfaceExtension::add_blit(const RID &p_render_target, const Rect2 &p_src_rect, const Rect2i &p_dst_rect, bool p_use_layer, uint32_t p_layer, bool p_apply_lens_distortion, const Vector2 &p_eye_center, double p_k1, double p_k2, double p_upscale, double p_aspect_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterfaceExtension::get_class_static()._native_ptr(), StringName("add_blit")._native_ptr(), 258596971);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_layer_encoded;
	PtrToArg<bool>::encode(p_use_layer, &p_use_layer_encoded);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_apply_lens_distortion_encoded;
	PtrToArg<bool>::encode(p_apply_lens_distortion, &p_apply_lens_distortion_encoded);
	double p_k1_encoded;
	PtrToArg<double>::encode(p_k1, &p_k1_encoded);
	double p_k2_encoded;
	PtrToArg<double>::encode(p_k2, &p_k2_encoded);
	double p_upscale_encoded;
	PtrToArg<double>::encode(p_upscale, &p_upscale_encoded);
	double p_aspect_ratio_encoded;
	PtrToArg<double>::encode(p_aspect_ratio, &p_aspect_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_render_target, &p_src_rect, &p_dst_rect, &p_use_layer_encoded, &p_layer_encoded, &p_apply_lens_distortion_encoded, &p_eye_center, &p_k1_encoded, &p_k2_encoded, &p_upscale_encoded, &p_aspect_ratio_encoded);
}

RID XRInterfaceExtension::get_render_target_texture(const RID &p_render_target) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRInterfaceExtension::get_class_static()._native_ptr(), StringName("get_render_target_texture")._native_ptr(), 41030802);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_render_target);
}

StringName XRInterfaceExtension::_get_name() const {
	return StringName();
}

uint32_t XRInterfaceExtension::_get_capabilities() const {
	return 0;
}

bool XRInterfaceExtension::_is_initialized() const {
	return false;
}

bool XRInterfaceExtension::_initialize() {
	return false;
}

void XRInterfaceExtension::_uninitialize() {}

Dictionary XRInterfaceExtension::_get_system_info() const {
	return Dictionary();
}

bool XRInterfaceExtension::_supports_play_area_mode(XRInterface::PlayAreaMode p_mode) const {
	return false;
}

XRInterface::PlayAreaMode XRInterfaceExtension::_get_play_area_mode() const {
	return XRInterface::PlayAreaMode(0);
}

bool XRInterfaceExtension::_set_play_area_mode(XRInterface::PlayAreaMode p_mode) const {
	return false;
}

PackedVector3Array XRInterfaceExtension::_get_play_area() const {
	return PackedVector3Array();
}

Vector2 XRInterfaceExtension::_get_render_target_size() {
	return Vector2();
}

uint32_t XRInterfaceExtension::_get_view_count() {
	return 0;
}

Transform3D XRInterfaceExtension::_get_camera_transform() {
	return Transform3D();
}

Transform3D XRInterfaceExtension::_get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform) {
	return Transform3D();
}

PackedFloat64Array XRInterfaceExtension::_get_projection_for_view(uint32_t p_view, double p_aspect, double p_z_near, double p_z_far) {
	return PackedFloat64Array();
}

RID XRInterfaceExtension::_get_vrs_texture() {
	return RID();
}

XRInterface::VRSTextureFormat XRInterfaceExtension::_get_vrs_texture_format() {
	return XRInterface::VRSTextureFormat(0);
}

void XRInterfaceExtension::_process() {}

void XRInterfaceExtension::_pre_render() {}

bool XRInterfaceExtension::_pre_draw_viewport(const RID &p_render_target) {
	return false;
}

void XRInterfaceExtension::_post_draw_viewport(const RID &p_render_target, const Rect2 &p_screen_rect) {}

void XRInterfaceExtension::_end_frame() {}

PackedStringArray XRInterfaceExtension::_get_suggested_tracker_names() const {
	return PackedStringArray();
}

PackedStringArray XRInterfaceExtension::_get_suggested_pose_names(const StringName &p_tracker_name) const {
	return PackedStringArray();
}

XRInterface::TrackingStatus XRInterfaceExtension::_get_tracking_status() const {
	return XRInterface::TrackingStatus(0);
}

void XRInterfaceExtension::_trigger_haptic_pulse(const String &p_action_name, const StringName &p_tracker_name, double p_frequency, double p_amplitude, double p_duration_sec, double p_delay_sec) {}

bool XRInterfaceExtension::_get_anchor_detection_is_enabled() const {
	return false;
}

void XRInterfaceExtension::_set_anchor_detection_is_enabled(bool p_enabled) {}

int32_t XRInterfaceExtension::_get_camera_feed_id() const {
	return 0;
}

RID XRInterfaceExtension::_get_color_texture() {
	return RID();
}

RID XRInterfaceExtension::_get_depth_texture() {
	return RID();
}

RID XRInterfaceExtension::_get_velocity_texture() {
	return RID();
}

} // namespace godot
