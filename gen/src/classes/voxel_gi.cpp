/**************************************************************************/
/*  voxel_gi.cpp                                                          */
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

#include <godot_cpp/classes/voxel_gi.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/camera_attributes.hpp>
#include <godot_cpp/classes/voxel_gi_data.hpp>

namespace godot {

void VoxelGI::set_probe_data(const Ref<VoxelGIData> &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGI::get_class_static()._native_ptr(), StringName("set_probe_data")._native_ptr(), 1637849675);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_data != nullptr ? &p_data->_owner : nullptr));
}

Ref<VoxelGIData> VoxelGI::get_probe_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGI::get_class_static()._native_ptr(), StringName("get_probe_data")._native_ptr(), 1730645405);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<VoxelGIData>()));
	return Ref<VoxelGIData>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<VoxelGIData>(_gde_method_bind, _owner));
}

void VoxelGI::set_subdiv(VoxelGI::Subdiv p_subdiv) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGI::get_class_static()._native_ptr(), StringName("set_subdiv")._native_ptr(), 2240898472);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_subdiv_encoded;
	PtrToArg<int64_t>::encode(p_subdiv, &p_subdiv_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_subdiv_encoded);
}

VoxelGI::Subdiv VoxelGI::get_subdiv() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGI::get_class_static()._native_ptr(), StringName("get_subdiv")._native_ptr(), 4261647950);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VoxelGI::Subdiv(0)));
	return (VoxelGI::Subdiv)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void VoxelGI::set_size(const Vector3 &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGI::get_class_static()._native_ptr(), StringName("set_size")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector3 VoxelGI::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGI::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void VoxelGI::set_camera_attributes(const Ref<CameraAttributes> &p_camera_attributes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGI::get_class_static()._native_ptr(), StringName("set_camera_attributes")._native_ptr(), 2817810567);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_camera_attributes != nullptr ? &p_camera_attributes->_owner : nullptr));
}

Ref<CameraAttributes> VoxelGI::get_camera_attributes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGI::get_class_static()._native_ptr(), StringName("get_camera_attributes")._native_ptr(), 3921283215);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<CameraAttributes>()));
	return Ref<CameraAttributes>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<CameraAttributes>(_gde_method_bind, _owner));
}

void VoxelGI::bake(Node *p_from_node, bool p_create_visual_debug) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGI::get_class_static()._native_ptr(), StringName("bake")._native_ptr(), 2781551026);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_create_visual_debug_encoded;
	PtrToArg<bool>::encode(p_create_visual_debug, &p_create_visual_debug_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_from_node != nullptr ? &p_from_node->_owner : nullptr), &p_create_visual_debug_encoded);
}

void VoxelGI::debug_bake() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VoxelGI::get_class_static()._native_ptr(), StringName("debug_bake")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
