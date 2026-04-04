/**************************************************************************/
/*  visual_instance3d.cpp                                                 */
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

#include <godot_cpp/classes/visual_instance3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void VisualInstance3D::set_base(const RID &p_base) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("set_base")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_base);
}

RID VisualInstance3D::get_base() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("get_base")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID VisualInstance3D::get_instance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("get_instance")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void VisualInstance3D::set_layer_mask(uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("set_layer_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mask_encoded);
}

uint32_t VisualInstance3D::get_layer_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("get_layer_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void VisualInstance3D::set_layer_mask_value(int32_t p_layer_number, bool p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("set_layer_mask_value")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	int8_t p_value_encoded;
	PtrToArg<bool>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_number_encoded, &p_value_encoded);
}

bool VisualInstance3D::get_layer_mask_value(int32_t p_layer_number) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("get_layer_mask_value")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_number_encoded);
}

void VisualInstance3D::set_sorting_offset(float p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("set_sorting_offset")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset_encoded);
}

float VisualInstance3D::get_sorting_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("get_sorting_offset")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VisualInstance3D::set_sorting_use_aabb_center(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("set_sorting_use_aabb_center")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool VisualInstance3D::is_sorting_use_aabb_center() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("is_sorting_use_aabb_center")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

AABB VisualInstance3D::get_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualInstance3D::get_class_static()._native_ptr(), StringName("get_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

AABB VisualInstance3D::_get_aabb() const {
	return AABB();
}

} // namespace godot
