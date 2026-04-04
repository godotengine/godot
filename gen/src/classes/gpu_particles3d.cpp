/**************************************************************************/
/*  gpu_particles3d.cpp                                                   */
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

#include <godot_cpp/classes/gpu_particles3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/skin.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector3.hpp>

namespace godot {

void GPUParticles3D::set_emitting(bool p_emitting) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_emitting")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_emitting_encoded;
	PtrToArg<bool>::encode(p_emitting, &p_emitting_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_emitting_encoded);
}

void GPUParticles3D::set_amount(int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_amount")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

void GPUParticles3D::set_lifetime(double p_secs) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_lifetime")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_secs_encoded;
	PtrToArg<double>::encode(p_secs, &p_secs_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_secs_encoded);
}

void GPUParticles3D::set_one_shot(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_one_shot")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void GPUParticles3D::set_pre_process_time(double p_secs) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_pre_process_time")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_secs_encoded;
	PtrToArg<double>::encode(p_secs, &p_secs_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_secs_encoded);
}

void GPUParticles3D::set_explosiveness_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_explosiveness_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

void GPUParticles3D::set_randomness_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_randomness_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

void GPUParticles3D::set_visibility_aabb(const AABB &p_aabb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_visibility_aabb")._native_ptr(), 259215842);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_aabb);
}

void GPUParticles3D::set_use_local_coordinates(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_use_local_coordinates")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void GPUParticles3D::set_fixed_fps(int32_t p_fps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_fixed_fps")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_fps_encoded;
	PtrToArg<int64_t>::encode(p_fps, &p_fps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fps_encoded);
}

void GPUParticles3D::set_fractional_delta(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_fractional_delta")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void GPUParticles3D::set_interpolate(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_interpolate")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void GPUParticles3D::set_process_material(const Ref<Material> &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_process_material")._native_ptr(), 2757459619);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_material != nullptr ? &p_material->_owner : nullptr));
}

void GPUParticles3D::set_speed_scale(double p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_speed_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

void GPUParticles3D::set_collision_base_size(float p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_collision_base_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

void GPUParticles3D::set_interp_to_end(float p_interp) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_interp_to_end")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_interp_encoded;
	PtrToArg<double>::encode(p_interp, &p_interp_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_interp_encoded);
}

bool GPUParticles3D::is_emitting() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("is_emitting")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t GPUParticles3D::get_amount() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_amount")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

double GPUParticles3D::get_lifetime() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_lifetime")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool GPUParticles3D::get_one_shot() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_one_shot")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

double GPUParticles3D::get_pre_process_time() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_pre_process_time")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float GPUParticles3D::get_explosiveness_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_explosiveness_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float GPUParticles3D::get_randomness_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_randomness_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

AABB GPUParticles3D::get_visibility_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_visibility_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

bool GPUParticles3D::get_use_local_coordinates() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_use_local_coordinates")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t GPUParticles3D::get_fixed_fps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_fixed_fps")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool GPUParticles3D::get_fractional_delta() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_fractional_delta")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool GPUParticles3D::get_interpolate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_interpolate")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<Material> GPUParticles3D::get_process_material() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_process_material")._native_ptr(), 5934680);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Material>()));
	return Ref<Material>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Material>(_gde_method_bind, _owner));
}

double GPUParticles3D::get_speed_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_speed_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float GPUParticles3D::get_collision_base_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_collision_base_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float GPUParticles3D::get_interp_to_end() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_interp_to_end")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GPUParticles3D::set_use_fixed_seed(bool p_use_fixed_seed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_use_fixed_seed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_fixed_seed_encoded;
	PtrToArg<bool>::encode(p_use_fixed_seed, &p_use_fixed_seed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_use_fixed_seed_encoded);
}

bool GPUParticles3D::get_use_fixed_seed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_use_fixed_seed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GPUParticles3D::set_seed(uint32_t p_seed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_seed")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_seed_encoded;
	PtrToArg<int64_t>::encode(p_seed, &p_seed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_seed_encoded);
}

uint32_t GPUParticles3D::get_seed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_seed")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GPUParticles3D::set_draw_order(GPUParticles3D::DrawOrder p_order) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_draw_order")._native_ptr(), 1208074815);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_order_encoded;
	PtrToArg<int64_t>::encode(p_order, &p_order_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_order_encoded);
}

GPUParticles3D::DrawOrder GPUParticles3D::get_draw_order() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_draw_order")._native_ptr(), 3770381780);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GPUParticles3D::DrawOrder(0)));
	return (GPUParticles3D::DrawOrder)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GPUParticles3D::set_draw_passes(int32_t p_passes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_draw_passes")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_passes_encoded;
	PtrToArg<int64_t>::encode(p_passes, &p_passes_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_passes_encoded);
}

void GPUParticles3D::set_draw_pass_mesh(int32_t p_pass, const Ref<Mesh> &p_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_draw_pass_mesh")._native_ptr(), 969122797);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_pass_encoded;
	PtrToArg<int64_t>::encode(p_pass, &p_pass_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pass_encoded, (p_mesh != nullptr ? &p_mesh->_owner : nullptr));
}

int32_t GPUParticles3D::get_draw_passes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_draw_passes")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<Mesh> GPUParticles3D::get_draw_pass_mesh(int32_t p_pass) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_draw_pass_mesh")._native_ptr(), 1576363275);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Mesh>()));
	int64_t p_pass_encoded;
	PtrToArg<int64_t>::encode(p_pass, &p_pass_encoded);
	return Ref<Mesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Mesh>(_gde_method_bind, _owner, &p_pass_encoded));
}

void GPUParticles3D::set_skin(const Ref<Skin> &p_skin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_skin")._native_ptr(), 3971435618);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_skin != nullptr ? &p_skin->_owner : nullptr));
}

Ref<Skin> GPUParticles3D::get_skin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_skin")._native_ptr(), 2074563878);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Skin>()));
	return Ref<Skin>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Skin>(_gde_method_bind, _owner));
}

void GPUParticles3D::restart(bool p_keep_seed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("restart")._native_ptr(), 107499316);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_seed_encoded;
	PtrToArg<bool>::encode(p_keep_seed, &p_keep_seed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_keep_seed_encoded);
}

AABB GPUParticles3D::capture_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("capture_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

void GPUParticles3D::set_sub_emitter(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_sub_emitter")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

NodePath GPUParticles3D::get_sub_emitter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_sub_emitter")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void GPUParticles3D::emit_particle(const Transform3D &p_xform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("emit_particle")._native_ptr(), 992173727);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_xform, &p_velocity, &p_color, &p_custom, &p_flags_encoded);
}

void GPUParticles3D::set_trail_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_trail_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

void GPUParticles3D::set_trail_lifetime(double p_secs) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_trail_lifetime")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_secs_encoded;
	PtrToArg<double>::encode(p_secs, &p_secs_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_secs_encoded);
}

bool GPUParticles3D::is_trail_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("is_trail_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

double GPUParticles3D::get_trail_lifetime() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_trail_lifetime")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GPUParticles3D::set_transform_align(GPUParticles3D::TransformAlign p_align) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_transform_align")._native_ptr(), 3892425954);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_align_encoded;
	PtrToArg<int64_t>::encode(p_align, &p_align_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_align_encoded);
}

GPUParticles3D::TransformAlign GPUParticles3D::get_transform_align() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_transform_align")._native_ptr(), 2100992166);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GPUParticles3D::TransformAlign(0)));
	return (GPUParticles3D::TransformAlign)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GPUParticles3D::convert_from_particles(Node *p_particles) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("convert_from_particles")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_particles != nullptr ? &p_particles->_owner : nullptr));
}

void GPUParticles3D::set_amount_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("set_amount_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

float GPUParticles3D::get_amount_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("get_amount_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GPUParticles3D::request_particles_process(float p_process_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticles3D::get_class_static()._native_ptr(), StringName("request_particles_process")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_process_time_encoded;
	PtrToArg<double>::encode(p_process_time, &p_process_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_process_time_encoded);
}

} // namespace godot
