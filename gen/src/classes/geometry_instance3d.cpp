/**************************************************************************/
/*  geometry_instance3d.cpp                                               */
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

#include <godot_cpp/classes/geometry_instance3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

void GeometryInstance3D::set_material_override(const Ref<Material> &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_material_override")._native_ptr(), 2757459619);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_material != nullptr ? &p_material->_owner : nullptr));
}

Ref<Material> GeometryInstance3D::get_material_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_material_override")._native_ptr(), 5934680);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Material>()));
	return Ref<Material>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Material>(_gde_method_bind, _owner));
}

void GeometryInstance3D::set_material_overlay(const Ref<Material> &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_material_overlay")._native_ptr(), 2757459619);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_material != nullptr ? &p_material->_owner : nullptr));
}

Ref<Material> GeometryInstance3D::get_material_overlay() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_material_overlay")._native_ptr(), 5934680);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Material>()));
	return Ref<Material>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Material>(_gde_method_bind, _owner));
}

void GeometryInstance3D::set_cast_shadows_setting(GeometryInstance3D::ShadowCastingSetting p_shadow_casting_setting) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_cast_shadows_setting")._native_ptr(), 856677339);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shadow_casting_setting_encoded;
	PtrToArg<int64_t>::encode(p_shadow_casting_setting, &p_shadow_casting_setting_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shadow_casting_setting_encoded);
}

GeometryInstance3D::ShadowCastingSetting GeometryInstance3D::get_cast_shadows_setting() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_cast_shadows_setting")._native_ptr(), 3383019359);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GeometryInstance3D::ShadowCastingSetting(0)));
	return (GeometryInstance3D::ShadowCastingSetting)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_lod_bias(float p_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_lod_bias")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bias_encoded;
	PtrToArg<double>::encode(p_bias, &p_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bias_encoded);
}

float GeometryInstance3D::get_lod_bias() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_lod_bias")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_transparency(float p_transparency) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_transparency")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_transparency_encoded;
	PtrToArg<double>::encode(p_transparency, &p_transparency_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_transparency_encoded);
}

float GeometryInstance3D::get_transparency() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_transparency")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_visibility_range_end_margin(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_visibility_range_end_margin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float GeometryInstance3D::get_visibility_range_end_margin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_visibility_range_end_margin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_visibility_range_end(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_visibility_range_end")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float GeometryInstance3D::get_visibility_range_end() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_visibility_range_end")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_visibility_range_begin_margin(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_visibility_range_begin_margin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float GeometryInstance3D::get_visibility_range_begin_margin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_visibility_range_begin_margin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_visibility_range_begin(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_visibility_range_begin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float GeometryInstance3D::get_visibility_range_begin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_visibility_range_begin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_visibility_range_fade_mode(GeometryInstance3D::VisibilityRangeFadeMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_visibility_range_fade_mode")._native_ptr(), 1440117808);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

GeometryInstance3D::VisibilityRangeFadeMode GeometryInstance3D::get_visibility_range_fade_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_visibility_range_fade_mode")._native_ptr(), 2067221882);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GeometryInstance3D::VisibilityRangeFadeMode(0)));
	return (GeometryInstance3D::VisibilityRangeFadeMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_instance_shader_parameter(const StringName &p_name, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_instance_shader_parameter")._native_ptr(), 3776071444);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_value);
}

Variant GeometryInstance3D::get_instance_shader_parameter(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_instance_shader_parameter")._native_ptr(), 2760726917);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_name);
}

void GeometryInstance3D::set_extra_cull_margin(float p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_extra_cull_margin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_margin_encoded;
	PtrToArg<double>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded);
}

float GeometryInstance3D::get_extra_cull_margin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_extra_cull_margin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_lightmap_texel_scale(float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_lightmap_texel_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

float GeometryInstance3D::get_lightmap_texel_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_lightmap_texel_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_lightmap_scale(GeometryInstance3D::LightmapScale p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_lightmap_scale")._native_ptr(), 2462696582);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_scale_encoded;
	PtrToArg<int64_t>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

GeometryInstance3D::LightmapScale GeometryInstance3D::get_lightmap_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_lightmap_scale")._native_ptr(), 798767852);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GeometryInstance3D::LightmapScale(0)));
	return (GeometryInstance3D::LightmapScale)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_gi_mode(GeometryInstance3D::GIMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_gi_mode")._native_ptr(), 2548557163);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

GeometryInstance3D::GIMode GeometryInstance3D::get_gi_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_gi_mode")._native_ptr(), 2188566509);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GeometryInstance3D::GIMode(0)));
	return (GeometryInstance3D::GIMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_ignore_occlusion_culling(bool p_ignore_culling) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_ignore_occlusion_culling")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_ignore_culling_encoded;
	PtrToArg<bool>::encode(p_ignore_culling, &p_ignore_culling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ignore_culling_encoded);
}

bool GeometryInstance3D::is_ignoring_occlusion_culling() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("is_ignoring_occlusion_culling")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GeometryInstance3D::set_custom_aabb(const AABB &p_aabb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("set_custom_aabb")._native_ptr(), 259215842);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_aabb);
}

AABB GeometryInstance3D::get_custom_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GeometryInstance3D::get_class_static()._native_ptr(), StringName("get_custom_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

} // namespace godot
