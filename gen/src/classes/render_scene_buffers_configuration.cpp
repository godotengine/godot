/**************************************************************************/
/*  render_scene_buffers_configuration.cpp                                */
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

#include <godot_cpp/classes/render_scene_buffers_configuration.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

RID RenderSceneBuffersConfiguration::get_render_target() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("get_render_target")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderSceneBuffersConfiguration::set_render_target(const RID &p_render_target) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("set_render_target")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_render_target);
}

Vector2i RenderSceneBuffersConfiguration::get_internal_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("get_internal_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void RenderSceneBuffersConfiguration::set_internal_size(const Vector2i &p_internal_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("set_internal_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_internal_size);
}

Vector2i RenderSceneBuffersConfiguration::get_target_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("get_target_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void RenderSceneBuffersConfiguration::set_target_size(const Vector2i &p_target_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("set_target_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_target_size);
}

uint32_t RenderSceneBuffersConfiguration::get_view_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("get_view_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RenderSceneBuffersConfiguration::set_view_count(uint32_t p_view_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("set_view_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_view_count_encoded;
	PtrToArg<int64_t>::encode(p_view_count, &p_view_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_view_count_encoded);
}

RenderingServer::ViewportScaling3DMode RenderSceneBuffersConfiguration::get_scaling_3d_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("get_scaling_3d_mode")._native_ptr(), 976778074);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingServer::ViewportScaling3DMode(0)));
	return (RenderingServer::ViewportScaling3DMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RenderSceneBuffersConfiguration::set_scaling_3d_mode(RenderingServer::ViewportScaling3DMode p_scaling_3d_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("set_scaling_3d_mode")._native_ptr(), 447477857);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_scaling_3d_mode_encoded;
	PtrToArg<int64_t>::encode(p_scaling_3d_mode, &p_scaling_3d_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scaling_3d_mode_encoded);
}

RenderingServer::ViewportMSAA RenderSceneBuffersConfiguration::get_msaa_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("get_msaa_3d")._native_ptr(), 3109158617);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingServer::ViewportMSAA(0)));
	return (RenderingServer::ViewportMSAA)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RenderSceneBuffersConfiguration::set_msaa_3d(RenderingServer::ViewportMSAA p_msaa_3d) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("set_msaa_3d")._native_ptr(), 3952630748);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msaa_3d_encoded;
	PtrToArg<int64_t>::encode(p_msaa_3d, &p_msaa_3d_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msaa_3d_encoded);
}

RenderingServer::ViewportScreenSpaceAA RenderSceneBuffersConfiguration::get_screen_space_aa() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("get_screen_space_aa")._native_ptr(), 641513172);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingServer::ViewportScreenSpaceAA(0)));
	return (RenderingServer::ViewportScreenSpaceAA)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RenderSceneBuffersConfiguration::set_screen_space_aa(RenderingServer::ViewportScreenSpaceAA p_screen_space_aa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("set_screen_space_aa")._native_ptr(), 139543108);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_screen_space_aa_encoded;
	PtrToArg<int64_t>::encode(p_screen_space_aa, &p_screen_space_aa_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_screen_space_aa_encoded);
}

float RenderSceneBuffersConfiguration::get_fsr_sharpness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("get_fsr_sharpness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RenderSceneBuffersConfiguration::set_fsr_sharpness(float p_fsr_sharpness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("set_fsr_sharpness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fsr_sharpness_encoded;
	PtrToArg<double>::encode(p_fsr_sharpness, &p_fsr_sharpness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fsr_sharpness_encoded);
}

float RenderSceneBuffersConfiguration::get_texture_mipmap_bias() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("get_texture_mipmap_bias")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RenderSceneBuffersConfiguration::set_texture_mipmap_bias(float p_texture_mipmap_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("set_texture_mipmap_bias")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_texture_mipmap_bias_encoded;
	PtrToArg<double>::encode(p_texture_mipmap_bias, &p_texture_mipmap_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture_mipmap_bias_encoded);
}

RenderingServer::ViewportAnisotropicFiltering RenderSceneBuffersConfiguration::get_anisotropic_filtering_level() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("get_anisotropic_filtering_level")._native_ptr(), 1617414954);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingServer::ViewportAnisotropicFiltering(0)));
	return (RenderingServer::ViewportAnisotropicFiltering)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RenderSceneBuffersConfiguration::set_anisotropic_filtering_level(RenderingServer::ViewportAnisotropicFiltering p_anisotropic_filtering_level) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersConfiguration::get_class_static()._native_ptr(), StringName("set_anisotropic_filtering_level")._native_ptr(), 2559658741);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_anisotropic_filtering_level_encoded;
	PtrToArg<int64_t>::encode(p_anisotropic_filtering_level, &p_anisotropic_filtering_level_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anisotropic_filtering_level_encoded);
}

} // namespace godot
