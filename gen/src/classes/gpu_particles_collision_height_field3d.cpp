/**************************************************************************/
/*  gpu_particles_collision_height_field3d.cpp                            */
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

#include <godot_cpp/classes/gpu_particles_collision_height_field3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void GPUParticlesCollisionHeightField3D::set_size(const Vector3 &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("set_size")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector3 GPUParticlesCollisionHeightField3D::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void GPUParticlesCollisionHeightField3D::set_resolution(GPUParticlesCollisionHeightField3D::Resolution p_resolution) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("set_resolution")._native_ptr(), 1009996517);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_resolution_encoded;
	PtrToArg<int64_t>::encode(p_resolution, &p_resolution_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_resolution_encoded);
}

GPUParticlesCollisionHeightField3D::Resolution GPUParticlesCollisionHeightField3D::get_resolution() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("get_resolution")._native_ptr(), 1156065644);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GPUParticlesCollisionHeightField3D::Resolution(0)));
	return (GPUParticlesCollisionHeightField3D::Resolution)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GPUParticlesCollisionHeightField3D::set_update_mode(GPUParticlesCollisionHeightField3D::UpdateMode p_update_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("set_update_mode")._native_ptr(), 673680859);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_update_mode_encoded;
	PtrToArg<int64_t>::encode(p_update_mode, &p_update_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_update_mode_encoded);
}

GPUParticlesCollisionHeightField3D::UpdateMode GPUParticlesCollisionHeightField3D::get_update_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("get_update_mode")._native_ptr(), 1998141380);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GPUParticlesCollisionHeightField3D::UpdateMode(0)));
	return (GPUParticlesCollisionHeightField3D::UpdateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GPUParticlesCollisionHeightField3D::set_heightfield_mask(uint32_t p_heightfield_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("set_heightfield_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_heightfield_mask_encoded;
	PtrToArg<int64_t>::encode(p_heightfield_mask, &p_heightfield_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_heightfield_mask_encoded);
}

uint32_t GPUParticlesCollisionHeightField3D::get_heightfield_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("get_heightfield_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GPUParticlesCollisionHeightField3D::set_heightfield_mask_value(int32_t p_layer_number, bool p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("set_heightfield_mask_value")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	int8_t p_value_encoded;
	PtrToArg<bool>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_number_encoded, &p_value_encoded);
}

bool GPUParticlesCollisionHeightField3D::get_heightfield_mask_value(int32_t p_layer_number) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("get_heightfield_mask_value")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_number_encoded);
}

void GPUParticlesCollisionHeightField3D::set_follow_camera_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("set_follow_camera_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool GPUParticlesCollisionHeightField3D::is_follow_camera_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GPUParticlesCollisionHeightField3D::get_class_static()._native_ptr(), StringName("is_follow_camera_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
