/**************************************************************************/
/*  voxel_gi_data.cpp                                                     */
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

#include <godot_cpp/classes/voxel_gi_data.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void VoxelGIData::allocate(const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3 &p_octree_size, const PackedByteArray &p_octree_cells, const PackedByteArray &p_data_cells, const PackedByteArray &p_distance_field, const PackedInt32Array &p_level_counts) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("allocate")._native_ptr(), 4041601946);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_to_cell_xform, &p_aabb, &p_octree_size, &p_octree_cells, &p_data_cells, &p_distance_field, &p_level_counts);
}

AABB VoxelGIData::get_bounds() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("get_bounds")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

Vector3 VoxelGIData::get_octree_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("get_octree_size")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Transform3D VoxelGIData::get_to_cell_xform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("get_to_cell_xform")._native_ptr(), 3229777777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

PackedByteArray VoxelGIData::get_octree_cells() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("get_octree_cells")._native_ptr(), 2362200018);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner);
}

PackedByteArray VoxelGIData::get_data_cells() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("get_data_cells")._native_ptr(), 2362200018);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner);
}

PackedInt32Array VoxelGIData::get_level_counts() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("get_level_counts")._native_ptr(), 1930428628);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void VoxelGIData::set_dynamic_range(float p_dynamic_range) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("set_dynamic_range")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_dynamic_range_encoded;
	PtrToArg<double>::encode(p_dynamic_range, &p_dynamic_range_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_dynamic_range_encoded);
}

float VoxelGIData::get_dynamic_range() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("get_dynamic_range")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VoxelGIData::set_energy(float p_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("set_energy")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_energy_encoded);
}

float VoxelGIData::get_energy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("get_energy")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VoxelGIData::set_bias(float p_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("set_bias")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bias_encoded;
	PtrToArg<double>::encode(p_bias, &p_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bias_encoded);
}

float VoxelGIData::get_bias() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("get_bias")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VoxelGIData::set_normal_bias(float p_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("set_normal_bias")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bias_encoded;
	PtrToArg<double>::encode(p_bias, &p_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bias_encoded);
}

float VoxelGIData::get_normal_bias() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("get_normal_bias")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VoxelGIData::set_propagation(float p_propagation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("set_propagation")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_propagation_encoded;
	PtrToArg<double>::encode(p_propagation, &p_propagation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_propagation_encoded);
}

float VoxelGIData::get_propagation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("get_propagation")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VoxelGIData::set_interior(bool p_interior) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("set_interior")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_interior_encoded;
	PtrToArg<bool>::encode(p_interior, &p_interior_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_interior_encoded);
}

bool VoxelGIData::is_interior() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("is_interior")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void VoxelGIData::set_use_two_bounces(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("set_use_two_bounces")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool VoxelGIData::is_using_two_bounces() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGIData::get_class_static()._native_ptr(), StringName("is_using_two_bounces")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
