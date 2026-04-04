/**************************************************************************/
/*  rd_pipeline_rasterization_state.cpp                                   */
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

#include <godot_cpp/classes/rd_pipeline_rasterization_state.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void RDPipelineRasterizationState::set_enable_depth_clamp(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("set_enable_depth_clamp")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineRasterizationState::get_enable_depth_clamp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("get_enable_depth_clamp")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineRasterizationState::set_discard_primitives(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("set_discard_primitives")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineRasterizationState::get_discard_primitives() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("get_discard_primitives")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineRasterizationState::set_wireframe(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("set_wireframe")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineRasterizationState::get_wireframe() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("get_wireframe")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineRasterizationState::set_cull_mode(RenderingDevice::PolygonCullMode p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("set_cull_mode")._native_ptr(), 2662586502);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::PolygonCullMode RDPipelineRasterizationState::get_cull_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("get_cull_mode")._native_ptr(), 2192484313);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::PolygonCullMode(0)));
	return (RenderingDevice::PolygonCullMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineRasterizationState::set_front_face(RenderingDevice::PolygonFrontFace p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("set_front_face")._native_ptr(), 2637251213);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::PolygonFrontFace RDPipelineRasterizationState::get_front_face() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("get_front_face")._native_ptr(), 708793786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::PolygonFrontFace(0)));
	return (RenderingDevice::PolygonFrontFace)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineRasterizationState::set_depth_bias_enabled(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("set_depth_bias_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineRasterizationState::get_depth_bias_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("get_depth_bias_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineRasterizationState::set_depth_bias_constant_factor(float p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("set_depth_bias_constant_factor")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_member_encoded;
	PtrToArg<double>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

float RDPipelineRasterizationState::get_depth_bias_constant_factor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("get_depth_bias_constant_factor")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RDPipelineRasterizationState::set_depth_bias_clamp(float p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("set_depth_bias_clamp")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_member_encoded;
	PtrToArg<double>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

float RDPipelineRasterizationState::get_depth_bias_clamp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("get_depth_bias_clamp")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RDPipelineRasterizationState::set_depth_bias_slope_factor(float p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("set_depth_bias_slope_factor")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_member_encoded;
	PtrToArg<double>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

float RDPipelineRasterizationState::get_depth_bias_slope_factor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("get_depth_bias_slope_factor")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RDPipelineRasterizationState::set_line_width(float p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("set_line_width")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_member_encoded;
	PtrToArg<double>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

float RDPipelineRasterizationState::get_line_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("get_line_width")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RDPipelineRasterizationState::set_patch_control_points(uint32_t p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("set_patch_control_points")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

uint32_t RDPipelineRasterizationState::get_patch_control_points() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineRasterizationState::get_class_static()._native_ptr(), StringName("get_patch_control_points")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
