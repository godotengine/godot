/**************************************************************************/
/*  skeleton_modification_stack2d.cpp                                     */
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

#include <godot_cpp/classes/skeleton_modification_stack2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/skeleton2d.hpp>
#include <godot_cpp/classes/skeleton_modification2d.hpp>

namespace godot {

void SkeletonModificationStack2D::setup() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("setup")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void SkeletonModificationStack2D::execute(float p_delta, int32_t p_execution_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("execute")._native_ptr(), 1005356550);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_delta_encoded;
	PtrToArg<double>::encode(p_delta, &p_delta_encoded);
	int64_t p_execution_mode_encoded;
	PtrToArg<int64_t>::encode(p_execution_mode, &p_execution_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_delta_encoded, &p_execution_mode_encoded);
}

void SkeletonModificationStack2D::enable_all_modifications(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("enable_all_modifications")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

Ref<SkeletonModification2D> SkeletonModificationStack2D::get_modification(int32_t p_mod_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("get_modification")._native_ptr(), 2570274329);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SkeletonModification2D>()));
	int64_t p_mod_idx_encoded;
	PtrToArg<int64_t>::encode(p_mod_idx, &p_mod_idx_encoded);
	return Ref<SkeletonModification2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SkeletonModification2D>(_gde_method_bind, _owner, &p_mod_idx_encoded));
}

void SkeletonModificationStack2D::add_modification(const Ref<SkeletonModification2D> &p_modification) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("add_modification")._native_ptr(), 354162120);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_modification != nullptr ? &p_modification->_owner : nullptr));
}

void SkeletonModificationStack2D::delete_modification(int32_t p_mod_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("delete_modification")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mod_idx_encoded;
	PtrToArg<int64_t>::encode(p_mod_idx, &p_mod_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mod_idx_encoded);
}

void SkeletonModificationStack2D::set_modification(int32_t p_mod_idx, const Ref<SkeletonModification2D> &p_modification) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("set_modification")._native_ptr(), 1098262544);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mod_idx_encoded;
	PtrToArg<int64_t>::encode(p_mod_idx, &p_mod_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mod_idx_encoded, (p_modification != nullptr ? &p_modification->_owner : nullptr));
}

void SkeletonModificationStack2D::set_modification_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("set_modification_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t SkeletonModificationStack2D::get_modification_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("get_modification_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool SkeletonModificationStack2D::get_is_setup() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("get_is_setup")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SkeletonModificationStack2D::set_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("set_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool SkeletonModificationStack2D::get_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("get_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SkeletonModificationStack2D::set_strength(float p_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("set_strength")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_strength_encoded);
}

float SkeletonModificationStack2D::get_strength() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("get_strength")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

Skeleton2D *SkeletonModificationStack2D::get_skeleton() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModificationStack2D::get_class_static()._native_ptr(), StringName("get_skeleton")._native_ptr(), 1697361217);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Skeleton2D>(_gde_method_bind, _owner);
}

} // namespace godot
