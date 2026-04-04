/**************************************************************************/
/*  base_material3d.cpp                                                   */
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

#include <godot_cpp/classes/base_material3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

void BaseMaterial3D::set_albedo(const Color &p_albedo) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_albedo")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_albedo);
}

Color BaseMaterial3D::get_albedo() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_albedo")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_transparency(BaseMaterial3D::Transparency p_transparency) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_transparency")._native_ptr(), 3435651667);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_transparency_encoded;
	PtrToArg<int64_t>::encode(p_transparency, &p_transparency_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_transparency_encoded);
}

BaseMaterial3D::Transparency BaseMaterial3D::get_transparency() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_transparency")._native_ptr(), 990903061);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::Transparency(0)));
	return (BaseMaterial3D::Transparency)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_alpha_antialiasing(BaseMaterial3D::AlphaAntiAliasing p_alpha_aa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_alpha_antialiasing")._native_ptr(), 3212649852);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alpha_aa_encoded;
	PtrToArg<int64_t>::encode(p_alpha_aa, &p_alpha_aa_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alpha_aa_encoded);
}

BaseMaterial3D::AlphaAntiAliasing BaseMaterial3D::get_alpha_antialiasing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_alpha_antialiasing")._native_ptr(), 2889939400);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::AlphaAntiAliasing(0)));
	return (BaseMaterial3D::AlphaAntiAliasing)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_alpha_antialiasing_edge(float p_edge) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_alpha_antialiasing_edge")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_edge_encoded;
	PtrToArg<double>::encode(p_edge, &p_edge_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_edge_encoded);
}

float BaseMaterial3D::get_alpha_antialiasing_edge() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_alpha_antialiasing_edge")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_shading_mode(BaseMaterial3D::ShadingMode p_shading_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_shading_mode")._native_ptr(), 3368750322);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shading_mode_encoded;
	PtrToArg<int64_t>::encode(p_shading_mode, &p_shading_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shading_mode_encoded);
}

BaseMaterial3D::ShadingMode BaseMaterial3D::get_shading_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_shading_mode")._native_ptr(), 2132070559);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::ShadingMode(0)));
	return (BaseMaterial3D::ShadingMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_specular(float p_specular) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_specular")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_specular_encoded;
	PtrToArg<double>::encode(p_specular, &p_specular_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_specular_encoded);
}

float BaseMaterial3D::get_specular() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_specular")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_metallic(float p_metallic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_metallic")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_metallic_encoded;
	PtrToArg<double>::encode(p_metallic, &p_metallic_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_metallic_encoded);
}

float BaseMaterial3D::get_metallic() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_metallic")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_roughness(float p_roughness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_roughness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_roughness_encoded;
	PtrToArg<double>::encode(p_roughness, &p_roughness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_roughness_encoded);
}

float BaseMaterial3D::get_roughness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_roughness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_emission(const Color &p_emission) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_emission")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_emission);
}

Color BaseMaterial3D::get_emission() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_emission")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_emission_energy_multiplier(float p_emission_energy_multiplier) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_emission_energy_multiplier")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_emission_energy_multiplier_encoded;
	PtrToArg<double>::encode(p_emission_energy_multiplier, &p_emission_energy_multiplier_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_emission_energy_multiplier_encoded);
}

float BaseMaterial3D::get_emission_energy_multiplier() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_emission_energy_multiplier")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_emission_intensity(float p_emission_energy_multiplier) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_emission_intensity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_emission_energy_multiplier_encoded;
	PtrToArg<double>::encode(p_emission_energy_multiplier, &p_emission_energy_multiplier_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_emission_energy_multiplier_encoded);
}

float BaseMaterial3D::get_emission_intensity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_emission_intensity")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_normal_scale(float p_normal_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_normal_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_normal_scale_encoded;
	PtrToArg<double>::encode(p_normal_scale, &p_normal_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_normal_scale_encoded);
}

float BaseMaterial3D::get_normal_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_normal_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_rim(float p_rim) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_rim")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_rim_encoded;
	PtrToArg<double>::encode(p_rim, &p_rim_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rim_encoded);
}

float BaseMaterial3D::get_rim() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_rim")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_rim_tint(float p_rim_tint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_rim_tint")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_rim_tint_encoded;
	PtrToArg<double>::encode(p_rim_tint, &p_rim_tint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rim_tint_encoded);
}

float BaseMaterial3D::get_rim_tint() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_rim_tint")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_clearcoat(float p_clearcoat) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_clearcoat")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_clearcoat_encoded;
	PtrToArg<double>::encode(p_clearcoat, &p_clearcoat_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_clearcoat_encoded);
}

float BaseMaterial3D::get_clearcoat() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_clearcoat")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_clearcoat_roughness(float p_clearcoat_roughness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_clearcoat_roughness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_clearcoat_roughness_encoded;
	PtrToArg<double>::encode(p_clearcoat_roughness, &p_clearcoat_roughness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_clearcoat_roughness_encoded);
}

float BaseMaterial3D::get_clearcoat_roughness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_clearcoat_roughness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_anisotropy(float p_anisotropy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_anisotropy")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_anisotropy_encoded;
	PtrToArg<double>::encode(p_anisotropy, &p_anisotropy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anisotropy_encoded);
}

float BaseMaterial3D::get_anisotropy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_anisotropy")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_heightmap_scale(float p_heightmap_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_heightmap_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_heightmap_scale_encoded;
	PtrToArg<double>::encode(p_heightmap_scale, &p_heightmap_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_heightmap_scale_encoded);
}

float BaseMaterial3D::get_heightmap_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_heightmap_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_subsurface_scattering_strength(float p_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_subsurface_scattering_strength")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_strength_encoded);
}

float BaseMaterial3D::get_subsurface_scattering_strength() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_subsurface_scattering_strength")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_transmittance_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_transmittance_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color BaseMaterial3D::get_transmittance_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_transmittance_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_transmittance_depth(float p_depth) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_transmittance_depth")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_depth_encoded;
	PtrToArg<double>::encode(p_depth, &p_depth_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_depth_encoded);
}

float BaseMaterial3D::get_transmittance_depth() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_transmittance_depth")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_transmittance_boost(float p_boost) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_transmittance_boost")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_boost_encoded;
	PtrToArg<double>::encode(p_boost, &p_boost_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_boost_encoded);
}

float BaseMaterial3D::get_transmittance_boost() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_transmittance_boost")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_backlight(const Color &p_backlight) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_backlight")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_backlight);
}

Color BaseMaterial3D::get_backlight() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_backlight")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_refraction(float p_refraction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_refraction")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_refraction_encoded;
	PtrToArg<double>::encode(p_refraction, &p_refraction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_refraction_encoded);
}

float BaseMaterial3D::get_refraction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_refraction")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_point_size(float p_point_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_point_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_point_size_encoded;
	PtrToArg<double>::encode(p_point_size, &p_point_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_point_size_encoded);
}

float BaseMaterial3D::get_point_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_point_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_detail_uv(BaseMaterial3D::DetailUV p_detail_uv) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_detail_uv")._native_ptr(), 456801921);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_detail_uv_encoded;
	PtrToArg<int64_t>::encode(p_detail_uv, &p_detail_uv_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_detail_uv_encoded);
}

BaseMaterial3D::DetailUV BaseMaterial3D::get_detail_uv() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_detail_uv")._native_ptr(), 2306920512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::DetailUV(0)));
	return (BaseMaterial3D::DetailUV)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_blend_mode(BaseMaterial3D::BlendMode p_blend_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_blend_mode")._native_ptr(), 2830186259);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_blend_mode_encoded;
	PtrToArg<int64_t>::encode(p_blend_mode, &p_blend_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_blend_mode_encoded);
}

BaseMaterial3D::BlendMode BaseMaterial3D::get_blend_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_blend_mode")._native_ptr(), 4022690962);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::BlendMode(0)));
	return (BaseMaterial3D::BlendMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_depth_draw_mode(BaseMaterial3D::DepthDrawMode p_depth_draw_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_depth_draw_mode")._native_ptr(), 1456584748);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_depth_draw_mode_encoded;
	PtrToArg<int64_t>::encode(p_depth_draw_mode, &p_depth_draw_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_depth_draw_mode_encoded);
}

BaseMaterial3D::DepthDrawMode BaseMaterial3D::get_depth_draw_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_depth_draw_mode")._native_ptr(), 2578197639);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::DepthDrawMode(0)));
	return (BaseMaterial3D::DepthDrawMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_depth_test(BaseMaterial3D::DepthTest p_depth_test) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_depth_test")._native_ptr(), 3918692338);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_depth_test_encoded;
	PtrToArg<int64_t>::encode(p_depth_test, &p_depth_test_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_depth_test_encoded);
}

BaseMaterial3D::DepthTest BaseMaterial3D::get_depth_test() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_depth_test")._native_ptr(), 3434785811);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::DepthTest(0)));
	return (BaseMaterial3D::DepthTest)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_cull_mode(BaseMaterial3D::CullMode p_cull_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_cull_mode")._native_ptr(), 2338909218);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cull_mode_encoded;
	PtrToArg<int64_t>::encode(p_cull_mode, &p_cull_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cull_mode_encoded);
}

BaseMaterial3D::CullMode BaseMaterial3D::get_cull_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_cull_mode")._native_ptr(), 1941499586);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::CullMode(0)));
	return (BaseMaterial3D::CullMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_diffuse_mode(BaseMaterial3D::DiffuseMode p_diffuse_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_diffuse_mode")._native_ptr(), 1045299638);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_diffuse_mode_encoded;
	PtrToArg<int64_t>::encode(p_diffuse_mode, &p_diffuse_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_diffuse_mode_encoded);
}

BaseMaterial3D::DiffuseMode BaseMaterial3D::get_diffuse_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_diffuse_mode")._native_ptr(), 3973617136);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::DiffuseMode(0)));
	return (BaseMaterial3D::DiffuseMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_specular_mode(BaseMaterial3D::SpecularMode p_specular_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_specular_mode")._native_ptr(), 584737147);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_specular_mode_encoded;
	PtrToArg<int64_t>::encode(p_specular_mode, &p_specular_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_specular_mode_encoded);
}

BaseMaterial3D::SpecularMode BaseMaterial3D::get_specular_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_specular_mode")._native_ptr(), 2569953298);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::SpecularMode(0)));
	return (BaseMaterial3D::SpecularMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_flag(BaseMaterial3D::Flags p_flag, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_flag")._native_ptr(), 3070159527);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flag_encoded, &p_enable_encoded);
}

bool BaseMaterial3D::get_flag(BaseMaterial3D::Flags p_flag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_flag")._native_ptr(), 1286410065);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_flag_encoded);
}

void BaseMaterial3D::set_texture_filter(BaseMaterial3D::TextureFilter p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_texture_filter")._native_ptr(), 22904437);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

BaseMaterial3D::TextureFilter BaseMaterial3D::get_texture_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_texture_filter")._native_ptr(), 3289213076);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::TextureFilter(0)));
	return (BaseMaterial3D::TextureFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_feature(BaseMaterial3D::Feature p_feature, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_feature")._native_ptr(), 2819288693);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_feature_encoded;
	PtrToArg<int64_t>::encode(p_feature, &p_feature_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_feature_encoded, &p_enable_encoded);
}

bool BaseMaterial3D::get_feature(BaseMaterial3D::Feature p_feature) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_feature")._native_ptr(), 1965241794);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_feature_encoded;
	PtrToArg<int64_t>::encode(p_feature, &p_feature_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_feature_encoded);
}

void BaseMaterial3D::set_texture(BaseMaterial3D::TextureParam p_param, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_texture")._native_ptr(), 464208135);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_param_encoded, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> BaseMaterial3D::get_texture(BaseMaterial3D::TextureParam p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_texture")._native_ptr(), 329605813);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_param_encoded));
}

void BaseMaterial3D::set_detail_blend_mode(BaseMaterial3D::BlendMode p_detail_blend_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_detail_blend_mode")._native_ptr(), 2830186259);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_detail_blend_mode_encoded;
	PtrToArg<int64_t>::encode(p_detail_blend_mode, &p_detail_blend_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_detail_blend_mode_encoded);
}

BaseMaterial3D::BlendMode BaseMaterial3D::get_detail_blend_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_detail_blend_mode")._native_ptr(), 4022690962);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::BlendMode(0)));
	return (BaseMaterial3D::BlendMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_uv1_scale(const Vector3 &p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_uv1_scale")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale);
}

Vector3 BaseMaterial3D::get_uv1_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_uv1_scale")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_uv1_offset(const Vector3 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_uv1_offset")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector3 BaseMaterial3D::get_uv1_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_uv1_offset")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_uv1_triplanar_blend_sharpness(float p_sharpness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_uv1_triplanar_blend_sharpness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_sharpness_encoded;
	PtrToArg<double>::encode(p_sharpness, &p_sharpness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sharpness_encoded);
}

float BaseMaterial3D::get_uv1_triplanar_blend_sharpness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_uv1_triplanar_blend_sharpness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_uv2_scale(const Vector3 &p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_uv2_scale")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale);
}

Vector3 BaseMaterial3D::get_uv2_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_uv2_scale")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_uv2_offset(const Vector3 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_uv2_offset")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector3 BaseMaterial3D::get_uv2_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_uv2_offset")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_uv2_triplanar_blend_sharpness(float p_sharpness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_uv2_triplanar_blend_sharpness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_sharpness_encoded;
	PtrToArg<double>::encode(p_sharpness, &p_sharpness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sharpness_encoded);
}

float BaseMaterial3D::get_uv2_triplanar_blend_sharpness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_uv2_triplanar_blend_sharpness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_billboard_mode(BaseMaterial3D::BillboardMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_billboard_mode")._native_ptr(), 4202036497);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

BaseMaterial3D::BillboardMode BaseMaterial3D::get_billboard_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_billboard_mode")._native_ptr(), 1283840139);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::BillboardMode(0)));
	return (BaseMaterial3D::BillboardMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_particles_anim_h_frames(int32_t p_frames) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_particles_anim_h_frames")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_frames_encoded;
	PtrToArg<int64_t>::encode(p_frames, &p_frames_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_frames_encoded);
}

int32_t BaseMaterial3D::get_particles_anim_h_frames() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_particles_anim_h_frames")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_particles_anim_v_frames(int32_t p_frames) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_particles_anim_v_frames")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_frames_encoded;
	PtrToArg<int64_t>::encode(p_frames, &p_frames_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_frames_encoded);
}

int32_t BaseMaterial3D::get_particles_anim_v_frames() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_particles_anim_v_frames")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_particles_anim_loop(bool p_loop) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_particles_anim_loop")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_loop_encoded;
	PtrToArg<bool>::encode(p_loop, &p_loop_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_loop_encoded);
}

bool BaseMaterial3D::get_particles_anim_loop() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_particles_anim_loop")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_heightmap_deep_parallax(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_heightmap_deep_parallax")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool BaseMaterial3D::is_heightmap_deep_parallax_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("is_heightmap_deep_parallax_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_heightmap_deep_parallax_min_layers(int32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_heightmap_deep_parallax_min_layers")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded);
}

int32_t BaseMaterial3D::get_heightmap_deep_parallax_min_layers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_heightmap_deep_parallax_min_layers")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_heightmap_deep_parallax_max_layers(int32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_heightmap_deep_parallax_max_layers")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded);
}

int32_t BaseMaterial3D::get_heightmap_deep_parallax_max_layers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_heightmap_deep_parallax_max_layers")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_heightmap_deep_parallax_flip_tangent(bool p_flip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_heightmap_deep_parallax_flip_tangent")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_flip_encoded;
	PtrToArg<bool>::encode(p_flip, &p_flip_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flip_encoded);
}

bool BaseMaterial3D::get_heightmap_deep_parallax_flip_tangent() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_heightmap_deep_parallax_flip_tangent")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_heightmap_deep_parallax_flip_binormal(bool p_flip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_heightmap_deep_parallax_flip_binormal")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_flip_encoded;
	PtrToArg<bool>::encode(p_flip, &p_flip_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flip_encoded);
}

bool BaseMaterial3D::get_heightmap_deep_parallax_flip_binormal() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_heightmap_deep_parallax_flip_binormal")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_grow(float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_grow")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

float BaseMaterial3D::get_grow() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_grow")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_emission_operator(BaseMaterial3D::EmissionOperator p_operator) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_emission_operator")._native_ptr(), 3825128922);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_operator_encoded;
	PtrToArg<int64_t>::encode(p_operator, &p_operator_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_operator_encoded);
}

BaseMaterial3D::EmissionOperator BaseMaterial3D::get_emission_operator() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_emission_operator")._native_ptr(), 974205018);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::EmissionOperator(0)));
	return (BaseMaterial3D::EmissionOperator)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_ao_light_affect(float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_ao_light_affect")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

float BaseMaterial3D::get_ao_light_affect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_ao_light_affect")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_alpha_scissor_threshold(float p_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_alpha_scissor_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_threshold_encoded;
	PtrToArg<double>::encode(p_threshold, &p_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_encoded);
}

float BaseMaterial3D::get_alpha_scissor_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_alpha_scissor_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_alpha_hash_scale(float p_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_alpha_hash_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_threshold_encoded;
	PtrToArg<double>::encode(p_threshold, &p_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_encoded);
}

float BaseMaterial3D::get_alpha_hash_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_alpha_hash_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_grow_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_grow_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool BaseMaterial3D::is_grow_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("is_grow_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_metallic_texture_channel(BaseMaterial3D::TextureChannel p_channel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_metallic_texture_channel")._native_ptr(), 744167988);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_channel_encoded;
	PtrToArg<int64_t>::encode(p_channel, &p_channel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_channel_encoded);
}

BaseMaterial3D::TextureChannel BaseMaterial3D::get_metallic_texture_channel() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_metallic_texture_channel")._native_ptr(), 568133867);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::TextureChannel(0)));
	return (BaseMaterial3D::TextureChannel)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_roughness_texture_channel(BaseMaterial3D::TextureChannel p_channel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_roughness_texture_channel")._native_ptr(), 744167988);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_channel_encoded;
	PtrToArg<int64_t>::encode(p_channel, &p_channel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_channel_encoded);
}

BaseMaterial3D::TextureChannel BaseMaterial3D::get_roughness_texture_channel() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_roughness_texture_channel")._native_ptr(), 568133867);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::TextureChannel(0)));
	return (BaseMaterial3D::TextureChannel)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_ao_texture_channel(BaseMaterial3D::TextureChannel p_channel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_ao_texture_channel")._native_ptr(), 744167988);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_channel_encoded;
	PtrToArg<int64_t>::encode(p_channel, &p_channel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_channel_encoded);
}

BaseMaterial3D::TextureChannel BaseMaterial3D::get_ao_texture_channel() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_ao_texture_channel")._native_ptr(), 568133867);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::TextureChannel(0)));
	return (BaseMaterial3D::TextureChannel)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_refraction_texture_channel(BaseMaterial3D::TextureChannel p_channel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_refraction_texture_channel")._native_ptr(), 744167988);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_channel_encoded;
	PtrToArg<int64_t>::encode(p_channel, &p_channel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_channel_encoded);
}

BaseMaterial3D::TextureChannel BaseMaterial3D::get_refraction_texture_channel() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_refraction_texture_channel")._native_ptr(), 568133867);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::TextureChannel(0)));
	return (BaseMaterial3D::TextureChannel)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_proximity_fade_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_proximity_fade_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool BaseMaterial3D::is_proximity_fade_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("is_proximity_fade_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_proximity_fade_distance(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_proximity_fade_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float BaseMaterial3D::get_proximity_fade_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_proximity_fade_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_msdf_pixel_range(float p_range) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_msdf_pixel_range")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_range_encoded;
	PtrToArg<double>::encode(p_range, &p_range_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_range_encoded);
}

float BaseMaterial3D::get_msdf_pixel_range() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_msdf_pixel_range")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_msdf_outline_size(float p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_msdf_outline_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

float BaseMaterial3D::get_msdf_outline_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_msdf_outline_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_distance_fade(BaseMaterial3D::DistanceFadeMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_distance_fade")._native_ptr(), 1379478617);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

BaseMaterial3D::DistanceFadeMode BaseMaterial3D::get_distance_fade() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_distance_fade")._native_ptr(), 2694575734);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::DistanceFadeMode(0)));
	return (BaseMaterial3D::DistanceFadeMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_distance_fade_max_distance(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_distance_fade_max_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float BaseMaterial3D::get_distance_fade_max_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_distance_fade_max_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_distance_fade_min_distance(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_distance_fade_min_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float BaseMaterial3D::get_distance_fade_min_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_distance_fade_min_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_z_clip_scale(float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_z_clip_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

float BaseMaterial3D::get_z_clip_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_z_clip_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_fov_override(float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_fov_override")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

float BaseMaterial3D::get_fov_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_fov_override")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_stencil_mode(BaseMaterial3D::StencilMode p_stencil_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_stencil_mode")._native_ptr(), 2272367200);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stencil_mode_encoded;
	PtrToArg<int64_t>::encode(p_stencil_mode, &p_stencil_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stencil_mode_encoded);
}

BaseMaterial3D::StencilMode BaseMaterial3D::get_stencil_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_stencil_mode")._native_ptr(), 2908443456);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::StencilMode(0)));
	return (BaseMaterial3D::StencilMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_stencil_flags(int32_t p_stencil_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_stencil_flags")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stencil_flags_encoded;
	PtrToArg<int64_t>::encode(p_stencil_flags, &p_stencil_flags_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stencil_flags_encoded);
}

int32_t BaseMaterial3D::get_stencil_flags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_stencil_flags")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_stencil_compare(BaseMaterial3D::StencilCompare p_stencil_compare) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_stencil_compare")._native_ptr(), 3741726481);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stencil_compare_encoded;
	PtrToArg<int64_t>::encode(p_stencil_compare, &p_stencil_compare_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stencil_compare_encoded);
}

BaseMaterial3D::StencilCompare BaseMaterial3D::get_stencil_compare() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_stencil_compare")._native_ptr(), 2824600492);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::StencilCompare(0)));
	return (BaseMaterial3D::StencilCompare)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_stencil_reference(int32_t p_stencil_reference) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_stencil_reference")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stencil_reference_encoded;
	PtrToArg<int64_t>::encode(p_stencil_reference, &p_stencil_reference_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stencil_reference_encoded);
}

int32_t BaseMaterial3D::get_stencil_reference() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_stencil_reference")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_stencil_effect_color(const Color &p_stencil_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_stencil_effect_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stencil_color);
}

Color BaseMaterial3D::get_stencil_effect_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_stencil_effect_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void BaseMaterial3D::set_stencil_effect_outline_thickness(float p_stencil_outline_thickness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("set_stencil_effect_outline_thickness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_stencil_outline_thickness_encoded;
	PtrToArg<double>::encode(p_stencil_outline_thickness, &p_stencil_outline_thickness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stencil_outline_thickness_encoded);
}

float BaseMaterial3D::get_stencil_effect_outline_thickness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BaseMaterial3D::get_class_static()._native_ptr(), StringName("get_stencil_effect_outline_thickness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
