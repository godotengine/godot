/**************************************************************************/
/*  particle_process_material.cpp                                         */
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

#include <godot_cpp/classes/particle_process_material.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

void ParticleProcessMaterial::set_direction(const Vector3 &p_degrees) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_direction")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_degrees);
}

Vector3 ParticleProcessMaterial::get_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_direction")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_inherit_velocity_ratio(double p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_inherit_velocity_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

double ParticleProcessMaterial::get_inherit_velocity_ratio() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_inherit_velocity_ratio")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_spread(float p_degrees) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_spread")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_degrees_encoded;
	PtrToArg<double>::encode(p_degrees, &p_degrees_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_degrees_encoded);
}

float ParticleProcessMaterial::get_spread() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_spread")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_flatness(float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_flatness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

float ParticleProcessMaterial::get_flatness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_flatness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_param(ParticleProcessMaterial::Parameter p_param, const Vector2 &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_param")._native_ptr(), 676779352);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_param_encoded, &p_value);
}

Vector2 ParticleProcessMaterial::get_param(ParticleProcessMaterial::Parameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_param")._native_ptr(), 2623708480);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_param_encoded);
}

void ParticleProcessMaterial::set_param_min(ParticleProcessMaterial::Parameter p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_param_min")._native_ptr(), 2295964248);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_param_encoded, &p_value_encoded);
}

float ParticleProcessMaterial::get_param_min(ParticleProcessMaterial::Parameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_param_min")._native_ptr(), 3903786503);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_param_encoded);
}

void ParticleProcessMaterial::set_param_max(ParticleProcessMaterial::Parameter p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_param_max")._native_ptr(), 2295964248);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_param_encoded, &p_value_encoded);
}

float ParticleProcessMaterial::get_param_max(ParticleProcessMaterial::Parameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_param_max")._native_ptr(), 3903786503);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_param_encoded);
}

void ParticleProcessMaterial::set_param_texture(ParticleProcessMaterial::Parameter p_param, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_param_texture")._native_ptr(), 526976089);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_param_encoded, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> ParticleProcessMaterial::get_param_texture(ParticleProcessMaterial::Parameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_param_texture")._native_ptr(), 3489372978);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_param_encoded));
}

void ParticleProcessMaterial::set_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color ParticleProcessMaterial::get_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_color_ramp(const Ref<Texture2D> &p_ramp) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_color_ramp")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_ramp != nullptr ? &p_ramp->_owner : nullptr));
}

Ref<Texture2D> ParticleProcessMaterial::get_color_ramp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_color_ramp")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void ParticleProcessMaterial::set_alpha_curve(const Ref<Texture2D> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_alpha_curve")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Texture2D> ParticleProcessMaterial::get_alpha_curve() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_alpha_curve")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void ParticleProcessMaterial::set_emission_curve(const Ref<Texture2D> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_curve")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Texture2D> ParticleProcessMaterial::get_emission_curve() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_curve")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void ParticleProcessMaterial::set_color_initial_ramp(const Ref<Texture2D> &p_ramp) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_color_initial_ramp")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_ramp != nullptr ? &p_ramp->_owner : nullptr));
}

Ref<Texture2D> ParticleProcessMaterial::get_color_initial_ramp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_color_initial_ramp")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void ParticleProcessMaterial::set_velocity_limit_curve(const Ref<Texture2D> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_velocity_limit_curve")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Texture2D> ParticleProcessMaterial::get_velocity_limit_curve() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_velocity_limit_curve")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void ParticleProcessMaterial::set_particle_flag(ParticleProcessMaterial::ParticleFlags p_particle_flag, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_particle_flag")._native_ptr(), 1711815571);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_particle_flag_encoded;
	PtrToArg<int64_t>::encode(p_particle_flag, &p_particle_flag_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particle_flag_encoded, &p_enable_encoded);
}

bool ParticleProcessMaterial::get_particle_flag(ParticleProcessMaterial::ParticleFlags p_particle_flag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_particle_flag")._native_ptr(), 3895316907);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_particle_flag_encoded;
	PtrToArg<int64_t>::encode(p_particle_flag, &p_particle_flag_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_particle_flag_encoded);
}

void ParticleProcessMaterial::set_velocity_pivot(const Vector3 &p_pivot) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_velocity_pivot")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pivot);
}

Vector3 ParticleProcessMaterial::get_velocity_pivot() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_velocity_pivot")._native_ptr(), 3783033775);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_emission_shape(ParticleProcessMaterial::EmissionShape p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_shape")._native_ptr(), 461501442);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_encoded);
}

ParticleProcessMaterial::EmissionShape ParticleProcessMaterial::get_emission_shape() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_shape")._native_ptr(), 3719733018);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ParticleProcessMaterial::EmissionShape(0)));
	return (ParticleProcessMaterial::EmissionShape)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_emission_sphere_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_sphere_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float ParticleProcessMaterial::get_emission_sphere_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_sphere_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_emission_box_extents(const Vector3 &p_extents) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_box_extents")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_extents);
}

Vector3 ParticleProcessMaterial::get_emission_box_extents() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_box_extents")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_emission_point_texture(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_point_texture")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> ParticleProcessMaterial::get_emission_point_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_point_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void ParticleProcessMaterial::set_emission_normal_texture(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_normal_texture")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> ParticleProcessMaterial::get_emission_normal_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_normal_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void ParticleProcessMaterial::set_emission_color_texture(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_color_texture")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> ParticleProcessMaterial::get_emission_color_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_color_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void ParticleProcessMaterial::set_emission_point_count(int32_t p_point_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_point_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_point_count_encoded;
	PtrToArg<int64_t>::encode(p_point_count, &p_point_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_point_count_encoded);
}

int32_t ParticleProcessMaterial::get_emission_point_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_point_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_emission_ring_axis(const Vector3 &p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_ring_axis")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_axis);
}

Vector3 ParticleProcessMaterial::get_emission_ring_axis() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_ring_axis")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_emission_ring_height(float p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_ring_height")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_height_encoded);
}

float ParticleProcessMaterial::get_emission_ring_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_ring_height")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_emission_ring_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_ring_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float ParticleProcessMaterial::get_emission_ring_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_ring_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_emission_ring_inner_radius(float p_inner_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_ring_inner_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_inner_radius_encoded;
	PtrToArg<double>::encode(p_inner_radius, &p_inner_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_inner_radius_encoded);
}

float ParticleProcessMaterial::get_emission_ring_inner_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_ring_inner_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_emission_ring_cone_angle(float p_cone_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_ring_cone_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_cone_angle_encoded;
	PtrToArg<double>::encode(p_cone_angle, &p_cone_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cone_angle_encoded);
}

float ParticleProcessMaterial::get_emission_ring_cone_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_ring_cone_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_emission_shape_offset(const Vector3 &p_emission_shape_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_shape_offset")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_emission_shape_offset);
}

Vector3 ParticleProcessMaterial::get_emission_shape_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_shape_offset")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_emission_shape_scale(const Vector3 &p_emission_shape_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_emission_shape_scale")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_emission_shape_scale);
}

Vector3 ParticleProcessMaterial::get_emission_shape_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_emission_shape_scale")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

bool ParticleProcessMaterial::get_turbulence_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_turbulence_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_turbulence_enabled(bool p_turbulence_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_turbulence_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_turbulence_enabled_encoded;
	PtrToArg<bool>::encode(p_turbulence_enabled, &p_turbulence_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_turbulence_enabled_encoded);
}

float ParticleProcessMaterial::get_turbulence_noise_strength() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_turbulence_noise_strength")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_turbulence_noise_strength(float p_turbulence_noise_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_turbulence_noise_strength")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_turbulence_noise_strength_encoded;
	PtrToArg<double>::encode(p_turbulence_noise_strength, &p_turbulence_noise_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_turbulence_noise_strength_encoded);
}

float ParticleProcessMaterial::get_turbulence_noise_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_turbulence_noise_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_turbulence_noise_scale(float p_turbulence_noise_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_turbulence_noise_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_turbulence_noise_scale_encoded;
	PtrToArg<double>::encode(p_turbulence_noise_scale, &p_turbulence_noise_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_turbulence_noise_scale_encoded);
}

float ParticleProcessMaterial::get_turbulence_noise_speed_random() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_turbulence_noise_speed_random")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_turbulence_noise_speed_random(float p_turbulence_noise_speed_random) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_turbulence_noise_speed_random")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_turbulence_noise_speed_random_encoded;
	PtrToArg<double>::encode(p_turbulence_noise_speed_random, &p_turbulence_noise_speed_random_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_turbulence_noise_speed_random_encoded);
}

Vector3 ParticleProcessMaterial::get_turbulence_noise_speed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_turbulence_noise_speed")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_turbulence_noise_speed(const Vector3 &p_turbulence_noise_speed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_turbulence_noise_speed")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_turbulence_noise_speed);
}

Vector3 ParticleProcessMaterial::get_gravity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_gravity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_gravity(const Vector3 &p_accel_vec) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_gravity")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_accel_vec);
}

void ParticleProcessMaterial::set_lifetime_randomness(double p_randomness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_lifetime_randomness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_randomness_encoded;
	PtrToArg<double>::encode(p_randomness, &p_randomness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_randomness_encoded);
}

double ParticleProcessMaterial::get_lifetime_randomness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_lifetime_randomness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

ParticleProcessMaterial::SubEmitterMode ParticleProcessMaterial::get_sub_emitter_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_sub_emitter_mode")._native_ptr(), 2399052877);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ParticleProcessMaterial::SubEmitterMode(0)));
	return (ParticleProcessMaterial::SubEmitterMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_sub_emitter_mode(ParticleProcessMaterial::SubEmitterMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_sub_emitter_mode")._native_ptr(), 2161806672);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

double ParticleProcessMaterial::get_sub_emitter_frequency() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_sub_emitter_frequency")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_sub_emitter_frequency(double p_hz) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_sub_emitter_frequency")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_hz_encoded;
	PtrToArg<double>::encode(p_hz, &p_hz_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hz_encoded);
}

int32_t ParticleProcessMaterial::get_sub_emitter_amount_at_end() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_sub_emitter_amount_at_end")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_sub_emitter_amount_at_end(int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_sub_emitter_amount_at_end")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

int32_t ParticleProcessMaterial::get_sub_emitter_amount_at_collision() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_sub_emitter_amount_at_collision")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_sub_emitter_amount_at_collision(int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_sub_emitter_amount_at_collision")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

int32_t ParticleProcessMaterial::get_sub_emitter_amount_at_start() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_sub_emitter_amount_at_start")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_sub_emitter_amount_at_start(int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_sub_emitter_amount_at_start")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

bool ParticleProcessMaterial::get_sub_emitter_keep_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_sub_emitter_keep_velocity")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_sub_emitter_keep_velocity(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_sub_emitter_keep_velocity")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void ParticleProcessMaterial::set_attractor_interaction_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_attractor_interaction_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool ParticleProcessMaterial::is_attractor_interaction_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("is_attractor_interaction_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_collision_mode(ParticleProcessMaterial::CollisionMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_collision_mode")._native_ptr(), 653804659);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

ParticleProcessMaterial::CollisionMode ParticleProcessMaterial::get_collision_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_collision_mode")._native_ptr(), 139371864);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ParticleProcessMaterial::CollisionMode(0)));
	return (ParticleProcessMaterial::CollisionMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_collision_use_scale(bool p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_collision_use_scale")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_radius_encoded;
	PtrToArg<bool>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

bool ParticleProcessMaterial::is_collision_using_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("is_collision_using_scale")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_collision_friction(float p_friction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_collision_friction")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_friction_encoded;
	PtrToArg<double>::encode(p_friction, &p_friction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_friction_encoded);
}

float ParticleProcessMaterial::get_collision_friction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_collision_friction")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ParticleProcessMaterial::set_collision_bounce(float p_bounce) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("set_collision_bounce")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bounce_encoded;
	PtrToArg<double>::encode(p_bounce, &p_bounce_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bounce_encoded);
}

float ParticleProcessMaterial::get_collision_bounce() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ParticleProcessMaterial::get_class_static()._native_ptr(), StringName("get_collision_bounce")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
