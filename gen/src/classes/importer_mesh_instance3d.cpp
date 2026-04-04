/**************************************************************************/
/*  importer_mesh_instance3d.cpp                                          */
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

#include <godot_cpp/classes/importer_mesh_instance3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/importer_mesh.hpp>
#include <godot_cpp/classes/skin.hpp>

namespace godot {

void ImporterMeshInstance3D::set_mesh(const Ref<ImporterMesh> &p_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("set_mesh")._native_ptr(), 2255166972);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_mesh != nullptr ? &p_mesh->_owner : nullptr));
}

Ref<ImporterMesh> ImporterMeshInstance3D::get_mesh() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("get_mesh")._native_ptr(), 3161779525);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ImporterMesh>()));
	return Ref<ImporterMesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ImporterMesh>(_gde_method_bind, _owner));
}

void ImporterMeshInstance3D::set_skin(const Ref<Skin> &p_skin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("set_skin")._native_ptr(), 3971435618);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_skin != nullptr ? &p_skin->_owner : nullptr));
}

Ref<Skin> ImporterMeshInstance3D::get_skin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("get_skin")._native_ptr(), 2074563878);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Skin>()));
	return Ref<Skin>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Skin>(_gde_method_bind, _owner));
}

void ImporterMeshInstance3D::set_skeleton_path(const NodePath &p_skeleton_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("set_skeleton_path")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skeleton_path);
}

NodePath ImporterMeshInstance3D::get_skeleton_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("get_skeleton_path")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void ImporterMeshInstance3D::set_layer_mask(uint32_t p_layer_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("set_layer_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_mask_encoded;
	PtrToArg<int64_t>::encode(p_layer_mask, &p_layer_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_mask_encoded);
}

uint32_t ImporterMeshInstance3D::get_layer_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("get_layer_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ImporterMeshInstance3D::set_cast_shadows_setting(GeometryInstance3D::ShadowCastingSetting p_shadow_casting_setting) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("set_cast_shadows_setting")._native_ptr(), 856677339);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shadow_casting_setting_encoded;
	PtrToArg<int64_t>::encode(p_shadow_casting_setting, &p_shadow_casting_setting_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shadow_casting_setting_encoded);
}

GeometryInstance3D::ShadowCastingSetting ImporterMeshInstance3D::get_cast_shadows_setting() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("get_cast_shadows_setting")._native_ptr(), 3383019359);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GeometryInstance3D::ShadowCastingSetting(0)));
	return (GeometryInstance3D::ShadowCastingSetting)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ImporterMeshInstance3D::set_visibility_range_end_margin(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("set_visibility_range_end_margin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float ImporterMeshInstance3D::get_visibility_range_end_margin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("get_visibility_range_end_margin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ImporterMeshInstance3D::set_visibility_range_end(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("set_visibility_range_end")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float ImporterMeshInstance3D::get_visibility_range_end() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("get_visibility_range_end")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ImporterMeshInstance3D::set_visibility_range_begin_margin(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("set_visibility_range_begin_margin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float ImporterMeshInstance3D::get_visibility_range_begin_margin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("get_visibility_range_begin_margin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ImporterMeshInstance3D::set_visibility_range_begin(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("set_visibility_range_begin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float ImporterMeshInstance3D::get_visibility_range_begin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("get_visibility_range_begin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ImporterMeshInstance3D::set_visibility_range_fade_mode(GeometryInstance3D::VisibilityRangeFadeMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("set_visibility_range_fade_mode")._native_ptr(), 1440117808);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

GeometryInstance3D::VisibilityRangeFadeMode ImporterMeshInstance3D::get_visibility_range_fade_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImporterMeshInstance3D::get_class_static()._native_ptr(), StringName("get_visibility_range_fade_mode")._native_ptr(), 2067221882);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GeometryInstance3D::VisibilityRangeFadeMode(0)));
	return (GeometryInstance3D::VisibilityRangeFadeMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
