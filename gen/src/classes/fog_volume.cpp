/**************************************************************************/
/*  fog_volume.cpp                                                        */
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

#include <godot_cpp/classes/fog_volume.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/material.hpp>

namespace godot {

void FogVolume::set_size(const Vector3 &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FogVolume::get_class_static()._native_ptr(), StringName("set_size")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector3 FogVolume::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FogVolume::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void FogVolume::set_shape(RenderingServer::FogVolumeShape p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FogVolume::get_class_static()._native_ptr(), StringName("set_shape")._native_ptr(), 1416323362);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_encoded);
}

RenderingServer::FogVolumeShape FogVolume::get_shape() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FogVolume::get_class_static()._native_ptr(), StringName("get_shape")._native_ptr(), 3920334604);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingServer::FogVolumeShape(0)));
	return (RenderingServer::FogVolumeShape)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FogVolume::set_material(const Ref<Material> &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FogVolume::get_class_static()._native_ptr(), StringName("set_material")._native_ptr(), 2757459619);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_material != nullptr ? &p_material->_owner : nullptr));
}

Ref<Material> FogVolume::get_material() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FogVolume::get_class_static()._native_ptr(), StringName("get_material")._native_ptr(), 5934680);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Material>()));
	return Ref<Material>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Material>(_gde_method_bind, _owner));
}

} // namespace godot
