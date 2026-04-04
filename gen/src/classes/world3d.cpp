/**************************************************************************/
/*  world3d.cpp                                                           */
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

#include <godot_cpp/classes/world3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/camera_attributes.hpp>
#include <godot_cpp/classes/environment.hpp>
#include <godot_cpp/classes/physics_direct_space_state3d.hpp>

namespace godot {

RID World3D::get_space() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(World3D::get_class_static()._native_ptr(), StringName("get_space")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID World3D::get_navigation_map() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(World3D::get_class_static()._native_ptr(), StringName("get_navigation_map")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID World3D::get_scenario() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(World3D::get_class_static()._native_ptr(), StringName("get_scenario")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void World3D::set_environment(const Ref<Environment> &p_env) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(World3D::get_class_static()._native_ptr(), StringName("set_environment")._native_ptr(), 4143518816);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_env != nullptr ? &p_env->_owner : nullptr));
}

Ref<Environment> World3D::get_environment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(World3D::get_class_static()._native_ptr(), StringName("get_environment")._native_ptr(), 3082064660);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Environment>()));
	return Ref<Environment>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Environment>(_gde_method_bind, _owner));
}

void World3D::set_fallback_environment(const Ref<Environment> &p_env) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(World3D::get_class_static()._native_ptr(), StringName("set_fallback_environment")._native_ptr(), 4143518816);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_env != nullptr ? &p_env->_owner : nullptr));
}

Ref<Environment> World3D::get_fallback_environment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(World3D::get_class_static()._native_ptr(), StringName("get_fallback_environment")._native_ptr(), 3082064660);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Environment>()));
	return Ref<Environment>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Environment>(_gde_method_bind, _owner));
}

void World3D::set_camera_attributes(const Ref<CameraAttributes> &p_attributes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(World3D::get_class_static()._native_ptr(), StringName("set_camera_attributes")._native_ptr(), 2817810567);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_attributes != nullptr ? &p_attributes->_owner : nullptr));
}

Ref<CameraAttributes> World3D::get_camera_attributes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(World3D::get_class_static()._native_ptr(), StringName("get_camera_attributes")._native_ptr(), 3921283215);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<CameraAttributes>()));
	return Ref<CameraAttributes>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<CameraAttributes>(_gde_method_bind, _owner));
}

PhysicsDirectSpaceState3D *World3D::get_direct_space_state() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(World3D::get_class_static()._native_ptr(), StringName("get_direct_space_state")._native_ptr(), 2069328350);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<PhysicsDirectSpaceState3D>(_gde_method_bind, _owner);
}

} // namespace godot
