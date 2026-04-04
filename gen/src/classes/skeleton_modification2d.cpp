/**************************************************************************/
/*  skeleton_modification2d.cpp                                           */
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

#include <godot_cpp/classes/skeleton_modification2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/skeleton_modification_stack2d.hpp>

namespace godot {

void SkeletonModification2D::set_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2D::get_class_static()._native_ptr(), StringName("set_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool SkeletonModification2D::get_enabled() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2D::get_class_static()._native_ptr(), StringName("get_enabled")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<SkeletonModificationStack2D> SkeletonModification2D::get_modification_stack() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2D::get_class_static()._native_ptr(), StringName("get_modification_stack")._native_ptr(), 2137761694);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SkeletonModificationStack2D>()));
	return Ref<SkeletonModificationStack2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SkeletonModificationStack2D>(_gde_method_bind, _owner));
}

void SkeletonModification2D::set_is_setup(bool p_is_setup) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2D::get_class_static()._native_ptr(), StringName("set_is_setup")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_is_setup_encoded;
	PtrToArg<bool>::encode(p_is_setup, &p_is_setup_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_is_setup_encoded);
}

bool SkeletonModification2D::get_is_setup() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2D::get_class_static()._native_ptr(), StringName("get_is_setup")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SkeletonModification2D::set_execution_mode(int32_t p_execution_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2D::get_class_static()._native_ptr(), StringName("set_execution_mode")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_execution_mode_encoded;
	PtrToArg<int64_t>::encode(p_execution_mode, &p_execution_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_execution_mode_encoded);
}

int32_t SkeletonModification2D::get_execution_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2D::get_class_static()._native_ptr(), StringName("get_execution_mode")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

float SkeletonModification2D::clamp_angle(float p_angle, float p_min, float p_max, bool p_invert) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2D::get_class_static()._native_ptr(), StringName("clamp_angle")._native_ptr(), 1229502682);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	double p_min_encoded;
	PtrToArg<double>::encode(p_min, &p_min_encoded);
	double p_max_encoded;
	PtrToArg<double>::encode(p_max, &p_max_encoded);
	int8_t p_invert_encoded;
	PtrToArg<bool>::encode(p_invert, &p_invert_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_angle_encoded, &p_min_encoded, &p_max_encoded, &p_invert_encoded);
}

void SkeletonModification2D::set_editor_draw_gizmo(bool p_draw_gizmo) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2D::get_class_static()._native_ptr(), StringName("set_editor_draw_gizmo")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_draw_gizmo_encoded;
	PtrToArg<bool>::encode(p_draw_gizmo, &p_draw_gizmo_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_gizmo_encoded);
}

bool SkeletonModification2D::get_editor_draw_gizmo() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2D::get_class_static()._native_ptr(), StringName("get_editor_draw_gizmo")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SkeletonModification2D::_execute(double p_delta) {}

void SkeletonModification2D::_setup_modification(const Ref<SkeletonModificationStack2D> &p_modification_stack) {}

void SkeletonModification2D::_draw_editor_gizmo() {}

} // namespace godot
