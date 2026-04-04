/**************************************************************************/
/*  lightmap_gi.cpp                                                       */
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

#include <godot_cpp/classes/lightmap_gi.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/camera_attributes.hpp>
#include <godot_cpp/classes/sky.hpp>

namespace godot {

void LightmapGI::set_light_data(const Ref<LightmapGIData> &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_light_data")._native_ptr(), 1790597277);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_data != nullptr ? &p_data->_owner : nullptr));
}

Ref<LightmapGIData> LightmapGI::get_light_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_light_data")._native_ptr(), 290354153);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<LightmapGIData>()));
	return Ref<LightmapGIData>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<LightmapGIData>(_gde_method_bind, _owner));
}

void LightmapGI::set_bake_quality(LightmapGI::BakeQuality p_bake_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_bake_quality")._native_ptr(), 1192215803);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bake_quality_encoded;
	PtrToArg<int64_t>::encode(p_bake_quality, &p_bake_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bake_quality_encoded);
}

LightmapGI::BakeQuality LightmapGI::get_bake_quality() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_bake_quality")._native_ptr(), 688832735);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (LightmapGI::BakeQuality(0)));
	return (LightmapGI::BakeQuality)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_bounces(int32_t p_bounces) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_bounces")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bounces_encoded;
	PtrToArg<int64_t>::encode(p_bounces, &p_bounces_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bounces_encoded);
}

int32_t LightmapGI::get_bounces() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_bounces")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_bounce_indirect_energy(float p_bounce_indirect_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_bounce_indirect_energy")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bounce_indirect_energy_encoded;
	PtrToArg<double>::encode(p_bounce_indirect_energy, &p_bounce_indirect_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bounce_indirect_energy_encoded);
}

float LightmapGI::get_bounce_indirect_energy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_bounce_indirect_energy")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LightmapGI::set_generate_probes(LightmapGI::GenerateProbes p_subdivision) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_generate_probes")._native_ptr(), 549981046);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_subdivision_encoded;
	PtrToArg<int64_t>::encode(p_subdivision, &p_subdivision_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_subdivision_encoded);
}

LightmapGI::GenerateProbes LightmapGI::get_generate_probes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_generate_probes")._native_ptr(), 3930596226);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (LightmapGI::GenerateProbes(0)));
	return (LightmapGI::GenerateProbes)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_bias(float p_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_bias")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bias_encoded;
	PtrToArg<double>::encode(p_bias, &p_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bias_encoded);
}

float LightmapGI::get_bias() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_bias")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LightmapGI::set_environment_mode(LightmapGI::EnvironmentMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_environment_mode")._native_ptr(), 2282650285);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

LightmapGI::EnvironmentMode LightmapGI::get_environment_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_environment_mode")._native_ptr(), 4128646479);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (LightmapGI::EnvironmentMode(0)));
	return (LightmapGI::EnvironmentMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_environment_custom_sky(const Ref<Sky> &p_sky) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_environment_custom_sky")._native_ptr(), 3336722921);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_sky != nullptr ? &p_sky->_owner : nullptr));
}

Ref<Sky> LightmapGI::get_environment_custom_sky() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_environment_custom_sky")._native_ptr(), 1177136966);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Sky>()));
	return Ref<Sky>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Sky>(_gde_method_bind, _owner));
}

void LightmapGI::set_environment_custom_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_environment_custom_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color LightmapGI::get_environment_custom_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_environment_custom_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void LightmapGI::set_environment_custom_energy(float p_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_environment_custom_energy")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_energy_encoded);
}

float LightmapGI::get_environment_custom_energy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_environment_custom_energy")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LightmapGI::set_texel_scale(float p_texel_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_texel_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_texel_scale_encoded;
	PtrToArg<double>::encode(p_texel_scale, &p_texel_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texel_scale_encoded);
}

float LightmapGI::get_texel_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_texel_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LightmapGI::set_max_texture_size(int32_t p_max_texture_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_max_texture_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_max_texture_size_encoded;
	PtrToArg<int64_t>::encode(p_max_texture_size, &p_max_texture_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_texture_size_encoded);
}

int32_t LightmapGI::get_max_texture_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_max_texture_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_supersampling_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_supersampling_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool LightmapGI::is_supersampling_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("is_supersampling_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_supersampling_factor(float p_factor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_supersampling_factor")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_factor_encoded;
	PtrToArg<double>::encode(p_factor, &p_factor_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_factor_encoded);
}

float LightmapGI::get_supersampling_factor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_supersampling_factor")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LightmapGI::set_use_denoiser(bool p_use_denoiser) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_use_denoiser")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_denoiser_encoded;
	PtrToArg<bool>::encode(p_use_denoiser, &p_use_denoiser_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_use_denoiser_encoded);
}

bool LightmapGI::is_using_denoiser() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("is_using_denoiser")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_denoiser_strength(float p_denoiser_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_denoiser_strength")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_denoiser_strength_encoded;
	PtrToArg<double>::encode(p_denoiser_strength, &p_denoiser_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_denoiser_strength_encoded);
}

float LightmapGI::get_denoiser_strength() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_denoiser_strength")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LightmapGI::set_denoiser_range(int32_t p_denoiser_range) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_denoiser_range")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_denoiser_range_encoded;
	PtrToArg<int64_t>::encode(p_denoiser_range, &p_denoiser_range_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_denoiser_range_encoded);
}

int32_t LightmapGI::get_denoiser_range() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_denoiser_range")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_interior(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_interior")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool LightmapGI::is_interior() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("is_interior")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_directional(bool p_directional) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_directional")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_directional_encoded;
	PtrToArg<bool>::encode(p_directional, &p_directional_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_directional_encoded);
}

bool LightmapGI::is_directional() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("is_directional")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_shadowmask_mode(LightmapGIData::ShadowmaskMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_shadowmask_mode")._native_ptr(), 3451066572);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

LightmapGIData::ShadowmaskMode LightmapGI::get_shadowmask_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_shadowmask_mode")._native_ptr(), 785478560);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (LightmapGIData::ShadowmaskMode(0)));
	return (LightmapGIData::ShadowmaskMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_use_texture_for_bounces(bool p_use_texture_for_bounces) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_use_texture_for_bounces")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_texture_for_bounces_encoded;
	PtrToArg<bool>::encode(p_use_texture_for_bounces, &p_use_texture_for_bounces_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_use_texture_for_bounces_encoded);
}

bool LightmapGI::is_using_texture_for_bounces() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("is_using_texture_for_bounces")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void LightmapGI::set_camera_attributes(const Ref<CameraAttributes> &p_camera_attributes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("set_camera_attributes")._native_ptr(), 2817810567);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_camera_attributes != nullptr ? &p_camera_attributes->_owner : nullptr));
}

Ref<CameraAttributes> LightmapGI::get_camera_attributes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGI::get_class_static()._native_ptr(), StringName("get_camera_attributes")._native_ptr(), 3921283215);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<CameraAttributes>()));
	return Ref<CameraAttributes>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<CameraAttributes>(_gde_method_bind, _owner));
}

} // namespace godot
