/**************************************************************************/
/*  procedural_sky_material.cpp                                           */
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

#include <godot_cpp/classes/procedural_sky_material.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

void ProceduralSkyMaterial::set_sky_top_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_sky_top_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color ProceduralSkyMaterial::get_sky_top_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_sky_top_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_sky_horizon_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_sky_horizon_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color ProceduralSkyMaterial::get_sky_horizon_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_sky_horizon_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_sky_curve(float p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_sky_curve")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_curve_encoded;
	PtrToArg<double>::encode(p_curve, &p_curve_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_curve_encoded);
}

float ProceduralSkyMaterial::get_sky_curve() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_sky_curve")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_sky_energy_multiplier(float p_multiplier) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_sky_energy_multiplier")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_multiplier_encoded;
	PtrToArg<double>::encode(p_multiplier, &p_multiplier_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multiplier_encoded);
}

float ProceduralSkyMaterial::get_sky_energy_multiplier() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_sky_energy_multiplier")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_sky_cover(const Ref<Texture2D> &p_sky_cover) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_sky_cover")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_sky_cover != nullptr ? &p_sky_cover->_owner : nullptr));
}

Ref<Texture2D> ProceduralSkyMaterial::get_sky_cover() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_sky_cover")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void ProceduralSkyMaterial::set_sky_cover_modulate(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_sky_cover_modulate")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color ProceduralSkyMaterial::get_sky_cover_modulate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_sky_cover_modulate")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_ground_bottom_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_ground_bottom_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color ProceduralSkyMaterial::get_ground_bottom_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_ground_bottom_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_ground_horizon_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_ground_horizon_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color ProceduralSkyMaterial::get_ground_horizon_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_ground_horizon_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_ground_curve(float p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_ground_curve")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_curve_encoded;
	PtrToArg<double>::encode(p_curve, &p_curve_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_curve_encoded);
}

float ProceduralSkyMaterial::get_ground_curve() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_ground_curve")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_ground_energy_multiplier(float p_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_ground_energy_multiplier")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_energy_encoded);
}

float ProceduralSkyMaterial::get_ground_energy_multiplier() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_ground_energy_multiplier")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_sun_angle_max(float p_degrees) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_sun_angle_max")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_degrees_encoded;
	PtrToArg<double>::encode(p_degrees, &p_degrees_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_degrees_encoded);
}

float ProceduralSkyMaterial::get_sun_angle_max() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_sun_angle_max")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_sun_curve(float p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_sun_curve")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_curve_encoded;
	PtrToArg<double>::encode(p_curve, &p_curve_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_curve_encoded);
}

float ProceduralSkyMaterial::get_sun_curve() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_sun_curve")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_use_debanding(bool p_use_debanding) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_use_debanding")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_debanding_encoded;
	PtrToArg<bool>::encode(p_use_debanding, &p_use_debanding_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_use_debanding_encoded);
}

bool ProceduralSkyMaterial::get_use_debanding() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_use_debanding")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ProceduralSkyMaterial::set_energy_multiplier(float p_multiplier) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("set_energy_multiplier")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_multiplier_encoded;
	PtrToArg<double>::encode(p_multiplier, &p_multiplier_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multiplier_encoded);
}

float ProceduralSkyMaterial::get_energy_multiplier() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProceduralSkyMaterial::get_class_static()._native_ptr(), StringName("get_energy_multiplier")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
