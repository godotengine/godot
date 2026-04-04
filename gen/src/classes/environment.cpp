/**************************************************************************/
/*  environment.cpp                                                       */
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

#include <godot_cpp/classes/environment.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/sky.hpp>
#include <godot_cpp/classes/texture.hpp>

namespace godot {

void Environment::set_background(Environment::BGMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_background")._native_ptr(), 4071623990);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Environment::BGMode Environment::get_background() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_background")._native_ptr(), 1843210413);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Environment::BGMode(0)));
	return (Environment::BGMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Environment::set_sky(const Ref<Sky> &p_sky) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sky")._native_ptr(), 3336722921);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_sky != nullptr ? &p_sky->_owner : nullptr));
}

Ref<Sky> Environment::get_sky() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sky")._native_ptr(), 1177136966);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Sky>()));
	return Ref<Sky>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Sky>(_gde_method_bind, _owner));
}

void Environment::set_sky_custom_fov(float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sky_custom_fov")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

float Environment::get_sky_custom_fov() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sky_custom_fov")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_sky_rotation(const Vector3 &p_euler_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sky_rotation")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_euler_radians);
}

Vector3 Environment::get_sky_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sky_rotation")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void Environment::set_bg_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_bg_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color Environment::get_bg_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_bg_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void Environment::set_bg_energy_multiplier(float p_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_bg_energy_multiplier")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_energy_encoded);
}

float Environment::get_bg_energy_multiplier() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_bg_energy_multiplier")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_bg_intensity(float p_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_bg_intensity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_energy_encoded);
}

float Environment::get_bg_intensity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_bg_intensity")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_canvas_max_layer(int32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_canvas_max_layer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded);
}

int32_t Environment::get_canvas_max_layer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_canvas_max_layer")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Environment::set_camera_feed_id(int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_camera_feed_id")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id_encoded);
}

int32_t Environment::get_camera_feed_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_camera_feed_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Environment::set_ambient_light_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ambient_light_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color Environment::get_ambient_light_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ambient_light_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void Environment::set_ambient_source(Environment::AmbientSource p_source) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ambient_source")._native_ptr(), 2607780160);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_encoded;
	PtrToArg<int64_t>::encode(p_source, &p_source_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_encoded);
}

Environment::AmbientSource Environment::get_ambient_source() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ambient_source")._native_ptr(), 67453933);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Environment::AmbientSource(0)));
	return (Environment::AmbientSource)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Environment::set_ambient_light_energy(float p_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ambient_light_energy")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_energy_encoded);
}

float Environment::get_ambient_light_energy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ambient_light_energy")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ambient_light_sky_contribution(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ambient_light_sky_contribution")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

float Environment::get_ambient_light_sky_contribution() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ambient_light_sky_contribution")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_reflection_source(Environment::ReflectionSource p_source) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_reflection_source")._native_ptr(), 299673197);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_encoded;
	PtrToArg<int64_t>::encode(p_source, &p_source_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_encoded);
}

Environment::ReflectionSource Environment::get_reflection_source() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_reflection_source")._native_ptr(), 777700713);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Environment::ReflectionSource(0)));
	return (Environment::ReflectionSource)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Environment::set_tonemapper(Environment::ToneMapper p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_tonemapper")._native_ptr(), 1509116664);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Environment::ToneMapper Environment::get_tonemapper() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_tonemapper")._native_ptr(), 2908408137);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Environment::ToneMapper(0)));
	return (Environment::ToneMapper)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Environment::set_tonemap_exposure(float p_exposure) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_tonemap_exposure")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_exposure_encoded;
	PtrToArg<double>::encode(p_exposure, &p_exposure_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_exposure_encoded);
}

float Environment::get_tonemap_exposure() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_tonemap_exposure")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_tonemap_white(float p_white) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_tonemap_white")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_white_encoded;
	PtrToArg<double>::encode(p_white, &p_white_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_white_encoded);
}

float Environment::get_tonemap_white() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_tonemap_white")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_tonemap_agx_white(float p_white) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_tonemap_agx_white")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_white_encoded;
	PtrToArg<double>::encode(p_white, &p_white_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_white_encoded);
}

float Environment::get_tonemap_agx_white() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_tonemap_agx_white")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_tonemap_agx_contrast(float p_contrast) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_tonemap_agx_contrast")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_contrast_encoded;
	PtrToArg<double>::encode(p_contrast, &p_contrast_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_contrast_encoded);
}

float Environment::get_tonemap_agx_contrast() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_tonemap_agx_contrast")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssr_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssr_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Environment::is_ssr_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_ssr_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_ssr_max_steps(int32_t p_max_steps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssr_max_steps")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_max_steps_encoded;
	PtrToArg<int64_t>::encode(p_max_steps, &p_max_steps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_steps_encoded);
}

int32_t Environment::get_ssr_max_steps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssr_max_steps")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Environment::set_ssr_fade_in(float p_fade_in) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssr_fade_in")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fade_in_encoded;
	PtrToArg<double>::encode(p_fade_in, &p_fade_in_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fade_in_encoded);
}

float Environment::get_ssr_fade_in() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssr_fade_in")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssr_fade_out(float p_fade_out) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssr_fade_out")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fade_out_encoded;
	PtrToArg<double>::encode(p_fade_out, &p_fade_out_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fade_out_encoded);
}

float Environment::get_ssr_fade_out() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssr_fade_out")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssr_depth_tolerance(float p_depth_tolerance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssr_depth_tolerance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_depth_tolerance_encoded;
	PtrToArg<double>::encode(p_depth_tolerance, &p_depth_tolerance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_depth_tolerance_encoded);
}

float Environment::get_ssr_depth_tolerance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssr_depth_tolerance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssao_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssao_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Environment::is_ssao_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_ssao_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_ssao_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssao_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float Environment::get_ssao_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssao_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssao_intensity(float p_intensity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssao_intensity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_intensity_encoded;
	PtrToArg<double>::encode(p_intensity, &p_intensity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_intensity_encoded);
}

float Environment::get_ssao_intensity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssao_intensity")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssao_power(float p_power) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssao_power")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_power_encoded;
	PtrToArg<double>::encode(p_power, &p_power_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_power_encoded);
}

float Environment::get_ssao_power() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssao_power")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssao_detail(float p_detail) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssao_detail")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_detail_encoded;
	PtrToArg<double>::encode(p_detail, &p_detail_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_detail_encoded);
}

float Environment::get_ssao_detail() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssao_detail")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssao_horizon(float p_horizon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssao_horizon")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_horizon_encoded;
	PtrToArg<double>::encode(p_horizon, &p_horizon_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_horizon_encoded);
}

float Environment::get_ssao_horizon() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssao_horizon")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssao_sharpness(float p_sharpness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssao_sharpness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_sharpness_encoded;
	PtrToArg<double>::encode(p_sharpness, &p_sharpness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sharpness_encoded);
}

float Environment::get_ssao_sharpness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssao_sharpness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssao_direct_light_affect(float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssao_direct_light_affect")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

float Environment::get_ssao_direct_light_affect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssao_direct_light_affect")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssao_ao_channel_affect(float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssao_ao_channel_affect")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

float Environment::get_ssao_ao_channel_affect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssao_ao_channel_affect")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssil_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssil_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Environment::is_ssil_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_ssil_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_ssil_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssil_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float Environment::get_ssil_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssil_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssil_intensity(float p_intensity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssil_intensity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_intensity_encoded;
	PtrToArg<double>::encode(p_intensity, &p_intensity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_intensity_encoded);
}

float Environment::get_ssil_intensity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssil_intensity")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssil_sharpness(float p_sharpness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssil_sharpness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_sharpness_encoded;
	PtrToArg<double>::encode(p_sharpness, &p_sharpness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sharpness_encoded);
}

float Environment::get_ssil_sharpness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssil_sharpness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_ssil_normal_rejection(float p_normal_rejection) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_ssil_normal_rejection")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_normal_rejection_encoded;
	PtrToArg<double>::encode(p_normal_rejection, &p_normal_rejection_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_normal_rejection_encoded);
}

float Environment::get_ssil_normal_rejection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_ssil_normal_rejection")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Environment::is_sdfgi_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_sdfgi_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_cascades(int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_cascades")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

int32_t Environment::get_sdfgi_cascades() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sdfgi_cascades")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_min_cell_size(float p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_min_cell_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

float Environment::get_sdfgi_min_cell_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sdfgi_min_cell_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_max_distance(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_max_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float Environment::get_sdfgi_max_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sdfgi_max_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_cascade0_distance(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_cascade0_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float Environment::get_sdfgi_cascade0_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sdfgi_cascade0_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_y_scale(Environment::SDFGIYScale p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_y_scale")._native_ptr(), 3608608372);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_scale_encoded;
	PtrToArg<int64_t>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

Environment::SDFGIYScale Environment::get_sdfgi_y_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sdfgi_y_scale")._native_ptr(), 2568002245);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Environment::SDFGIYScale(0)));
	return (Environment::SDFGIYScale)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_use_occlusion(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_use_occlusion")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Environment::is_sdfgi_using_occlusion() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_sdfgi_using_occlusion")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_bounce_feedback(float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_bounce_feedback")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

float Environment::get_sdfgi_bounce_feedback() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sdfgi_bounce_feedback")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_read_sky_light(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_read_sky_light")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Environment::is_sdfgi_reading_sky_light() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_sdfgi_reading_sky_light")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_energy(float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_energy")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

float Environment::get_sdfgi_energy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sdfgi_energy")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_normal_bias(float p_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_normal_bias")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bias_encoded;
	PtrToArg<double>::encode(p_bias, &p_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bias_encoded);
}

float Environment::get_sdfgi_normal_bias() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sdfgi_normal_bias")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_sdfgi_probe_bias(float p_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_sdfgi_probe_bias")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bias_encoded;
	PtrToArg<double>::encode(p_bias, &p_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bias_encoded);
}

float Environment::get_sdfgi_probe_bias() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_sdfgi_probe_bias")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_glow_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Environment::is_glow_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_glow_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_glow_level(int32_t p_idx, float p_intensity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_level")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	double p_intensity_encoded;
	PtrToArg<double>::encode(p_intensity, &p_intensity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_intensity_encoded);
}

float Environment::get_glow_level(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_glow_level")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_idx_encoded);
}

void Environment::set_glow_normalized(bool p_normalize) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_normalized")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_normalize_encoded;
	PtrToArg<bool>::encode(p_normalize, &p_normalize_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_normalize_encoded);
}

bool Environment::is_glow_normalized() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_glow_normalized")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_glow_intensity(float p_intensity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_intensity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_intensity_encoded;
	PtrToArg<double>::encode(p_intensity, &p_intensity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_intensity_encoded);
}

float Environment::get_glow_intensity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_glow_intensity")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_glow_strength(float p_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_strength")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_strength_encoded);
}

float Environment::get_glow_strength() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_glow_strength")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_glow_mix(float p_mix) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_mix")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_mix_encoded;
	PtrToArg<double>::encode(p_mix, &p_mix_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mix_encoded);
}

float Environment::get_glow_mix() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_glow_mix")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_glow_bloom(float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_bloom")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

float Environment::get_glow_bloom() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_glow_bloom")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_glow_blend_mode(Environment::GlowBlendMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_blend_mode")._native_ptr(), 2561587761);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Environment::GlowBlendMode Environment::get_glow_blend_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_glow_blend_mode")._native_ptr(), 1529667332);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Environment::GlowBlendMode(0)));
	return (Environment::GlowBlendMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Environment::set_glow_hdr_bleed_threshold(float p_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_hdr_bleed_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_threshold_encoded;
	PtrToArg<double>::encode(p_threshold, &p_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_encoded);
}

float Environment::get_glow_hdr_bleed_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_glow_hdr_bleed_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_glow_hdr_bleed_scale(float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_hdr_bleed_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

float Environment::get_glow_hdr_bleed_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_glow_hdr_bleed_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_glow_hdr_luminance_cap(float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_hdr_luminance_cap")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

float Environment::get_glow_hdr_luminance_cap() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_glow_hdr_luminance_cap")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_glow_map_strength(float p_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_map_strength")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_strength_encoded);
}

float Environment::get_glow_map_strength() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_glow_map_strength")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_glow_map(const Ref<Texture> &p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_glow_map")._native_ptr(), 1790811099);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_mode != nullptr ? &p_mode->_owner : nullptr));
}

Ref<Texture> Environment::get_glow_map() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_glow_map")._native_ptr(), 4037048985);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture>()));
	return Ref<Texture>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture>(_gde_method_bind, _owner));
}

void Environment::set_fog_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Environment::is_fog_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_fog_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_fog_mode(Environment::FogMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_mode")._native_ptr(), 3059806579);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Environment::FogMode Environment::get_fog_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_mode")._native_ptr(), 2456062483);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Environment::FogMode(0)));
	return (Environment::FogMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Environment::set_fog_light_color(const Color &p_light_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_light_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light_color);
}

Color Environment::get_fog_light_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_light_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void Environment::set_fog_light_energy(float p_light_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_light_energy")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_light_energy_encoded;
	PtrToArg<double>::encode(p_light_energy, &p_light_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light_energy_encoded);
}

float Environment::get_fog_light_energy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_light_energy")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_fog_sun_scatter(float p_sun_scatter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_sun_scatter")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_sun_scatter_encoded;
	PtrToArg<double>::encode(p_sun_scatter, &p_sun_scatter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sun_scatter_encoded);
}

float Environment::get_fog_sun_scatter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_sun_scatter")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_fog_density(float p_density) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_density")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_density_encoded;
	PtrToArg<double>::encode(p_density, &p_density_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_density_encoded);
}

float Environment::get_fog_density() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_density")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_fog_height(float p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_height")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_height_encoded);
}

float Environment::get_fog_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_height")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_fog_height_density(float p_height_density) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_height_density")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_height_density_encoded;
	PtrToArg<double>::encode(p_height_density, &p_height_density_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_height_density_encoded);
}

float Environment::get_fog_height_density() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_height_density")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_fog_aerial_perspective(float p_aerial_perspective) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_aerial_perspective")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_aerial_perspective_encoded;
	PtrToArg<double>::encode(p_aerial_perspective, &p_aerial_perspective_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_aerial_perspective_encoded);
}

float Environment::get_fog_aerial_perspective() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_aerial_perspective")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_fog_sky_affect(float p_sky_affect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_sky_affect")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_sky_affect_encoded;
	PtrToArg<double>::encode(p_sky_affect, &p_sky_affect_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sky_affect_encoded);
}

float Environment::get_fog_sky_affect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_sky_affect")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_fog_depth_curve(float p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_depth_curve")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_curve_encoded;
	PtrToArg<double>::encode(p_curve, &p_curve_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_curve_encoded);
}

float Environment::get_fog_depth_curve() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_depth_curve")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_fog_depth_begin(float p_begin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_depth_begin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_begin_encoded;
	PtrToArg<double>::encode(p_begin, &p_begin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_begin_encoded);
}

float Environment::get_fog_depth_begin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_depth_begin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_fog_depth_end(float p_end) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_fog_depth_end")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_end_encoded;
	PtrToArg<double>::encode(p_end, &p_end_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_end_encoded);
}

float Environment::get_fog_depth_end() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_fog_depth_end")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Environment::is_volumetric_fog_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_volumetric_fog_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_emission(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_emission")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color Environment::get_volumetric_fog_emission() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_volumetric_fog_emission")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_albedo(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_albedo")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color Environment::get_volumetric_fog_albedo() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_volumetric_fog_albedo")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_density(float p_density) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_density")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_density_encoded;
	PtrToArg<double>::encode(p_density, &p_density_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_density_encoded);
}

float Environment::get_volumetric_fog_density() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_volumetric_fog_density")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_emission_energy(float p_begin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_emission_energy")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_begin_encoded;
	PtrToArg<double>::encode(p_begin, &p_begin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_begin_encoded);
}

float Environment::get_volumetric_fog_emission_energy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_volumetric_fog_emission_energy")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_anisotropy(float p_anisotropy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_anisotropy")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_anisotropy_encoded;
	PtrToArg<double>::encode(p_anisotropy, &p_anisotropy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anisotropy_encoded);
}

float Environment::get_volumetric_fog_anisotropy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_volumetric_fog_anisotropy")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_length(float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_length")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_length_encoded);
}

float Environment::get_volumetric_fog_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_volumetric_fog_length")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_detail_spread(float p_detail_spread) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_detail_spread")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_detail_spread_encoded;
	PtrToArg<double>::encode(p_detail_spread, &p_detail_spread_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_detail_spread_encoded);
}

float Environment::get_volumetric_fog_detail_spread() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_volumetric_fog_detail_spread")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_gi_inject(float p_gi_inject) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_gi_inject")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_gi_inject_encoded;
	PtrToArg<double>::encode(p_gi_inject, &p_gi_inject_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gi_inject_encoded);
}

float Environment::get_volumetric_fog_gi_inject() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_volumetric_fog_gi_inject")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_ambient_inject(float p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_ambient_inject")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_enabled_encoded;
	PtrToArg<double>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

float Environment::get_volumetric_fog_ambient_inject() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_volumetric_fog_ambient_inject")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_sky_affect(float p_sky_affect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_sky_affect")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_sky_affect_encoded;
	PtrToArg<double>::encode(p_sky_affect, &p_sky_affect_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sky_affect_encoded);
}

float Environment::get_volumetric_fog_sky_affect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_volumetric_fog_sky_affect")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_temporal_reprojection_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_temporal_reprojection_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Environment::is_volumetric_fog_temporal_reprojection_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_volumetric_fog_temporal_reprojection_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_volumetric_fog_temporal_reprojection_amount(float p_temporal_reprojection_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_volumetric_fog_temporal_reprojection_amount")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_temporal_reprojection_amount_encoded;
	PtrToArg<double>::encode(p_temporal_reprojection_amount, &p_temporal_reprojection_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_temporal_reprojection_amount_encoded);
}

float Environment::get_volumetric_fog_temporal_reprojection_amount() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_volumetric_fog_temporal_reprojection_amount")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_adjustment_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_adjustment_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Environment::is_adjustment_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("is_adjustment_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Environment::set_adjustment_brightness(float p_brightness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_adjustment_brightness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_brightness_encoded;
	PtrToArg<double>::encode(p_brightness, &p_brightness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_brightness_encoded);
}

float Environment::get_adjustment_brightness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_adjustment_brightness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_adjustment_contrast(float p_contrast) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_adjustment_contrast")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_contrast_encoded;
	PtrToArg<double>::encode(p_contrast, &p_contrast_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_contrast_encoded);
}

float Environment::get_adjustment_contrast() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_adjustment_contrast")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_adjustment_saturation(float p_saturation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_adjustment_saturation")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_saturation_encoded;
	PtrToArg<double>::encode(p_saturation, &p_saturation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_saturation_encoded);
}

float Environment::get_adjustment_saturation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_adjustment_saturation")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Environment::set_adjustment_color_correction(const Ref<Texture> &p_color_correction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("set_adjustment_color_correction")._native_ptr(), 1790811099);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_color_correction != nullptr ? &p_color_correction->_owner : nullptr));
}

Ref<Texture> Environment::get_adjustment_color_correction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Environment::get_class_static()._native_ptr(), StringName("get_adjustment_color_correction")._native_ptr(), 4037048985);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture>()));
	return Ref<Texture>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture>(_gde_method_bind, _owner));
}

} // namespace godot
