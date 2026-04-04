/**************************************************************************/
/*  height_map_shape3d.cpp                                                */
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

#include <godot_cpp/classes/height_map_shape3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/image.hpp>

namespace godot {

void HeightMapShape3D::set_map_width(int32_t p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HeightMapShape3D::get_class_static()._native_ptr(), StringName("set_map_width")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_width_encoded);
}

int32_t HeightMapShape3D::get_map_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HeightMapShape3D::get_class_static()._native_ptr(), StringName("get_map_width")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void HeightMapShape3D::set_map_depth(int32_t p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HeightMapShape3D::get_class_static()._native_ptr(), StringName("set_map_depth")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_height_encoded);
}

int32_t HeightMapShape3D::get_map_depth() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HeightMapShape3D::get_class_static()._native_ptr(), StringName("get_map_depth")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void HeightMapShape3D::set_map_data(const PackedFloat32Array &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HeightMapShape3D::get_class_static()._native_ptr(), StringName("set_map_data")._native_ptr(), 2899603908);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_data);
}

PackedFloat32Array HeightMapShape3D::get_map_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HeightMapShape3D::get_class_static()._native_ptr(), StringName("get_map_data")._native_ptr(), 675695659);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedFloat32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedFloat32Array>(_gde_method_bind, _owner);
}

float HeightMapShape3D::get_min_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HeightMapShape3D::get_class_static()._native_ptr(), StringName("get_min_height")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float HeightMapShape3D::get_max_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HeightMapShape3D::get_class_static()._native_ptr(), StringName("get_max_height")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void HeightMapShape3D::update_map_data_from_image(const Ref<Image> &p_image, float p_height_min, float p_height_max) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HeightMapShape3D::get_class_static()._native_ptr(), StringName("update_map_data_from_image")._native_ptr(), 2636652979);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_height_min_encoded;
	PtrToArg<double>::encode(p_height_min, &p_height_min_encoded);
	double p_height_max_encoded;
	PtrToArg<double>::encode(p_height_max, &p_height_max_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_image != nullptr ? &p_image->_owner : nullptr), &p_height_min_encoded, &p_height_max_encoded);
}

} // namespace godot
