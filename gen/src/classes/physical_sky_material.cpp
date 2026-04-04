/**************************************************************************/
/*  physical_sky_material.cpp                                             */
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

#include <godot_cpp/classes/physical_sky_material.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

void PhysicalSkyMaterial::set_rayleigh_coefficient(float p_rayleigh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("set_rayleigh_coefficient")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_rayleigh_encoded;
	PtrToArg<double>::encode(p_rayleigh, &p_rayleigh_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rayleigh_encoded);
}

float PhysicalSkyMaterial::get_rayleigh_coefficient() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("get_rayleigh_coefficient")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalSkyMaterial::set_rayleigh_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("set_rayleigh_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color PhysicalSkyMaterial::get_rayleigh_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("get_rayleigh_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void PhysicalSkyMaterial::set_mie_coefficient(float p_mie) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("set_mie_coefficient")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_mie_encoded;
	PtrToArg<double>::encode(p_mie, &p_mie_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mie_encoded);
}

float PhysicalSkyMaterial::get_mie_coefficient() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("get_mie_coefficient")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalSkyMaterial::set_mie_eccentricity(float p_eccentricity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("set_mie_eccentricity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_eccentricity_encoded;
	PtrToArg<double>::encode(p_eccentricity, &p_eccentricity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_eccentricity_encoded);
}

float PhysicalSkyMaterial::get_mie_eccentricity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("get_mie_eccentricity")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalSkyMaterial::set_mie_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("set_mie_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color PhysicalSkyMaterial::get_mie_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("get_mie_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void PhysicalSkyMaterial::set_turbidity(float p_turbidity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("set_turbidity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_turbidity_encoded;
	PtrToArg<double>::encode(p_turbidity, &p_turbidity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_turbidity_encoded);
}

float PhysicalSkyMaterial::get_turbidity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("get_turbidity")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalSkyMaterial::set_sun_disk_scale(float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("set_sun_disk_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

float PhysicalSkyMaterial::get_sun_disk_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("get_sun_disk_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalSkyMaterial::set_ground_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("set_ground_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color PhysicalSkyMaterial::get_ground_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("get_ground_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void PhysicalSkyMaterial::set_energy_multiplier(float p_multiplier) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("set_energy_multiplier")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_multiplier_encoded;
	PtrToArg<double>::encode(p_multiplier, &p_multiplier_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multiplier_encoded);
}

float PhysicalSkyMaterial::get_energy_multiplier() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("get_energy_multiplier")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalSkyMaterial::set_use_debanding(bool p_use_debanding) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("set_use_debanding")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_debanding_encoded;
	PtrToArg<bool>::encode(p_use_debanding, &p_use_debanding_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_use_debanding_encoded);
}

bool PhysicalSkyMaterial::get_use_debanding() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("get_use_debanding")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PhysicalSkyMaterial::set_night_sky(const Ref<Texture2D> &p_night_sky) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("set_night_sky")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_night_sky != nullptr ? &p_night_sky->_owner : nullptr));
}

Ref<Texture2D> PhysicalSkyMaterial::get_night_sky() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalSkyMaterial::get_class_static()._native_ptr(), StringName("get_night_sky")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

} // namespace godot
