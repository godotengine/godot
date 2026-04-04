/**************************************************************************/
/*  cpu_particles3d.cpp                                                   */
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

#include <godot_cpp/classes/cpu_particles3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/curve.hpp>
#include <godot_cpp/classes/gradient.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/node.hpp>

namespace godot {

void CPUParticles3D::set_emitting(bool p_emitting) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emitting")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_emitting_encoded;
	PtrToArg<bool>::encode(p_emitting, &p_emitting_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_emitting_encoded);
}

void CPUParticles3D::set_amount(int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_amount")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

void CPUParticles3D::set_lifetime(double p_secs) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_lifetime")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_secs_encoded;
	PtrToArg<double>::encode(p_secs, &p_secs_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_secs_encoded);
}

void CPUParticles3D::set_one_shot(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_one_shot")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void CPUParticles3D::set_pre_process_time(double p_secs) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_pre_process_time")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_secs_encoded;
	PtrToArg<double>::encode(p_secs, &p_secs_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_secs_encoded);
}

void CPUParticles3D::set_explosiveness_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_explosiveness_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

void CPUParticles3D::set_randomness_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_randomness_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

void CPUParticles3D::set_visibility_aabb(const AABB &p_aabb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_visibility_aabb")._native_ptr(), 259215842);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_aabb);
}

void CPUParticles3D::set_lifetime_randomness(double p_random) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_lifetime_randomness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_random_encoded;
	PtrToArg<double>::encode(p_random, &p_random_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_random_encoded);
}

void CPUParticles3D::set_use_local_coordinates(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_use_local_coordinates")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void CPUParticles3D::set_fixed_fps(int32_t p_fps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_fixed_fps")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_fps_encoded;
	PtrToArg<int64_t>::encode(p_fps, &p_fps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fps_encoded);
}

void CPUParticles3D::set_fractional_delta(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_fractional_delta")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void CPUParticles3D::set_speed_scale(double p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_speed_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

bool CPUParticles3D::is_emitting() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("is_emitting")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t CPUParticles3D::get_amount() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_amount")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

double CPUParticles3D::get_lifetime() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_lifetime")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool CPUParticles3D::get_one_shot() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_one_shot")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

double CPUParticles3D::get_pre_process_time() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_pre_process_time")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float CPUParticles3D::get_explosiveness_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_explosiveness_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float CPUParticles3D::get_randomness_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_randomness_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

AABB CPUParticles3D::get_visibility_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_visibility_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

double CPUParticles3D::get_lifetime_randomness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_lifetime_randomness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool CPUParticles3D::get_use_local_coordinates() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_use_local_coordinates")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t CPUParticles3D::get_fixed_fps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_fixed_fps")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool CPUParticles3D::get_fractional_delta() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_fractional_delta")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

double CPUParticles3D::get_speed_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_speed_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_draw_order(CPUParticles3D::DrawOrder p_order) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_draw_order")._native_ptr(), 1427401774);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_order_encoded;
	PtrToArg<int64_t>::encode(p_order, &p_order_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_order_encoded);
}

CPUParticles3D::DrawOrder CPUParticles3D::get_draw_order() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_draw_order")._native_ptr(), 1321900776);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CPUParticles3D::DrawOrder(0)));
	return (CPUParticles3D::DrawOrder)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_mesh(const Ref<Mesh> &p_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_mesh")._native_ptr(), 194775623);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_mesh != nullptr ? &p_mesh->_owner : nullptr));
}

Ref<Mesh> CPUParticles3D::get_mesh() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_mesh")._native_ptr(), 1808005922);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Mesh>()));
	return Ref<Mesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Mesh>(_gde_method_bind, _owner));
}

void CPUParticles3D::set_use_fixed_seed(bool p_use_fixed_seed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_use_fixed_seed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_fixed_seed_encoded;
	PtrToArg<bool>::encode(p_use_fixed_seed, &p_use_fixed_seed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_use_fixed_seed_encoded);
}

bool CPUParticles3D::get_use_fixed_seed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_use_fixed_seed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_seed(uint32_t p_seed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_seed")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_seed_encoded;
	PtrToArg<int64_t>::encode(p_seed, &p_seed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_seed_encoded);
}

uint32_t CPUParticles3D::get_seed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_seed")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CPUParticles3D::restart(bool p_keep_seed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("restart")._native_ptr(), 107499316);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_seed_encoded;
	PtrToArg<bool>::encode(p_keep_seed, &p_keep_seed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_keep_seed_encoded);
}

void CPUParticles3D::request_particles_process(float p_process_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("request_particles_process")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_process_time_encoded;
	PtrToArg<double>::encode(p_process_time, &p_process_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_process_time_encoded);
}

AABB CPUParticles3D::capture_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("capture_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_direction(const Vector3 &p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_direction")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_direction);
}

Vector3 CPUParticles3D::get_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_direction")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_spread(float p_degrees) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_spread")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_degrees_encoded;
	PtrToArg<double>::encode(p_degrees, &p_degrees_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_degrees_encoded);
}

float CPUParticles3D::get_spread() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_spread")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_flatness(float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_flatness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

float CPUParticles3D::get_flatness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_flatness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_param_min(CPUParticles3D::Parameter p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_param_min")._native_ptr(), 557936109);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_param_encoded, &p_value_encoded);
}

float CPUParticles3D::get_param_min(CPUParticles3D::Parameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_param_min")._native_ptr(), 597646162);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_param_encoded);
}

void CPUParticles3D::set_param_max(CPUParticles3D::Parameter p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_param_max")._native_ptr(), 557936109);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_param_encoded, &p_value_encoded);
}

float CPUParticles3D::get_param_max(CPUParticles3D::Parameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_param_max")._native_ptr(), 597646162);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_param_encoded);
}

void CPUParticles3D::set_param_curve(CPUParticles3D::Parameter p_param, const Ref<Curve> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_param_curve")._native_ptr(), 4044142537);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_param_encoded, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Curve> CPUParticles3D::get_param_curve(CPUParticles3D::Parameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_param_curve")._native_ptr(), 4132790277);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner, &p_param_encoded));
}

void CPUParticles3D::set_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color CPUParticles3D::get_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_color_ramp(const Ref<Gradient> &p_ramp) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_color_ramp")._native_ptr(), 2756054477);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_ramp != nullptr ? &p_ramp->_owner : nullptr));
}

Ref<Gradient> CPUParticles3D::get_color_ramp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_color_ramp")._native_ptr(), 132272999);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Gradient>()));
	return Ref<Gradient>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Gradient>(_gde_method_bind, _owner));
}

void CPUParticles3D::set_color_initial_ramp(const Ref<Gradient> &p_ramp) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_color_initial_ramp")._native_ptr(), 2756054477);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_ramp != nullptr ? &p_ramp->_owner : nullptr));
}

Ref<Gradient> CPUParticles3D::get_color_initial_ramp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_color_initial_ramp")._native_ptr(), 132272999);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Gradient>()));
	return Ref<Gradient>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Gradient>(_gde_method_bind, _owner));
}

void CPUParticles3D::set_particle_flag(CPUParticles3D::ParticleFlags p_particle_flag, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_particle_flag")._native_ptr(), 3515406498);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_particle_flag_encoded;
	PtrToArg<int64_t>::encode(p_particle_flag, &p_particle_flag_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particle_flag_encoded, &p_enable_encoded);
}

bool CPUParticles3D::get_particle_flag(CPUParticles3D::ParticleFlags p_particle_flag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_particle_flag")._native_ptr(), 2845201987);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_particle_flag_encoded;
	PtrToArg<int64_t>::encode(p_particle_flag, &p_particle_flag_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_particle_flag_encoded);
}

void CPUParticles3D::set_emission_shape(CPUParticles3D::EmissionShape p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emission_shape")._native_ptr(), 491823814);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_encoded);
}

CPUParticles3D::EmissionShape CPUParticles3D::get_emission_shape() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_emission_shape")._native_ptr(), 2961454842);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CPUParticles3D::EmissionShape(0)));
	return (CPUParticles3D::EmissionShape)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_emission_sphere_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emission_sphere_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float CPUParticles3D::get_emission_sphere_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_emission_sphere_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_emission_box_extents(const Vector3 &p_extents) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emission_box_extents")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_extents);
}

Vector3 CPUParticles3D::get_emission_box_extents() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_emission_box_extents")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_emission_points(const PackedVector3Array &p_array) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emission_points")._native_ptr(), 334873810);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_array);
}

PackedVector3Array CPUParticles3D::get_emission_points() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_emission_points")._native_ptr(), 497664490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_emission_normals(const PackedVector3Array &p_array) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emission_normals")._native_ptr(), 334873810);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_array);
}

PackedVector3Array CPUParticles3D::get_emission_normals() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_emission_normals")._native_ptr(), 497664490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_emission_colors(const PackedColorArray &p_array) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emission_colors")._native_ptr(), 3546319833);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_array);
}

PackedColorArray CPUParticles3D::get_emission_colors() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_emission_colors")._native_ptr(), 1392750486);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedColorArray()));
	return ::godot::internal::_call_native_mb_ret<PackedColorArray>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_emission_ring_axis(const Vector3 &p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emission_ring_axis")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_axis);
}

Vector3 CPUParticles3D::get_emission_ring_axis() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_emission_ring_axis")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_emission_ring_height(float p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emission_ring_height")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_height_encoded);
}

float CPUParticles3D::get_emission_ring_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_emission_ring_height")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_emission_ring_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emission_ring_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float CPUParticles3D::get_emission_ring_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_emission_ring_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_emission_ring_inner_radius(float p_inner_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emission_ring_inner_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_inner_radius_encoded;
	PtrToArg<double>::encode(p_inner_radius, &p_inner_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_inner_radius_encoded);
}

float CPUParticles3D::get_emission_ring_inner_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_emission_ring_inner_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_emission_ring_cone_angle(float p_cone_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_emission_ring_cone_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_cone_angle_encoded;
	PtrToArg<double>::encode(p_cone_angle, &p_cone_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cone_angle_encoded);
}

float CPUParticles3D::get_emission_ring_cone_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_emission_ring_cone_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

Vector3 CPUParticles3D::get_gravity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_gravity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_gravity(const Vector3 &p_accel_vec) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_gravity")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_accel_vec);
}

bool CPUParticles3D::get_split_scale() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_split_scale")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CPUParticles3D::set_split_scale(bool p_split_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_split_scale")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_split_scale_encoded;
	PtrToArg<bool>::encode(p_split_scale, &p_split_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_split_scale_encoded);
}

Ref<Curve> CPUParticles3D::get_scale_curve_x() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_scale_curve_x")._native_ptr(), 2460114913);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner));
}

void CPUParticles3D::set_scale_curve_x(const Ref<Curve> &p_scale_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_scale_curve_x")._native_ptr(), 270443179);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_scale_curve != nullptr ? &p_scale_curve->_owner : nullptr));
}

Ref<Curve> CPUParticles3D::get_scale_curve_y() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_scale_curve_y")._native_ptr(), 2460114913);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner));
}

void CPUParticles3D::set_scale_curve_y(const Ref<Curve> &p_scale_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_scale_curve_y")._native_ptr(), 270443179);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_scale_curve != nullptr ? &p_scale_curve->_owner : nullptr));
}

Ref<Curve> CPUParticles3D::get_scale_curve_z() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("get_scale_curve_z")._native_ptr(), 2460114913);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner));
}

void CPUParticles3D::set_scale_curve_z(const Ref<Curve> &p_scale_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("set_scale_curve_z")._native_ptr(), 270443179);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_scale_curve != nullptr ? &p_scale_curve->_owner : nullptr));
}

void CPUParticles3D::convert_from_particles(Node *p_particles) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CPUParticles3D::get_class_static()._native_ptr(), StringName("convert_from_particles")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_particles != nullptr ? &p_particles->_owner : nullptr));
}

} // namespace godot
