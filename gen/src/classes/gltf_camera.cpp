/**************************************************************************/
/*  gltf_camera.cpp                                                       */
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

#include <godot_cpp/classes/gltf_camera.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/camera3d.hpp>

namespace godot {

Ref<GLTFCamera> GLTFCamera::from_node(Camera3D *p_camera_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("from_node")._native_ptr(), 237784);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFCamera>()));
	return Ref<GLTFCamera>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFCamera>(_gde_method_bind, nullptr, (p_camera_node != nullptr ? &p_camera_node->_owner : nullptr)));
}

Camera3D *GLTFCamera::to_node() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("to_node")._native_ptr(), 2285090890);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Camera3D>(_gde_method_bind, _owner);
}

Ref<GLTFCamera> GLTFCamera::from_dictionary(const Dictionary &p_dictionary) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("from_dictionary")._native_ptr(), 2495512509);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFCamera>()));
	return Ref<GLTFCamera>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFCamera>(_gde_method_bind, nullptr, &p_dictionary));
}

Dictionary GLTFCamera::to_dictionary() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("to_dictionary")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

bool GLTFCamera::get_perspective() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("get_perspective")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GLTFCamera::set_perspective(bool p_perspective) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("set_perspective")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_perspective_encoded;
	PtrToArg<bool>::encode(p_perspective, &p_perspective_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_perspective_encoded);
}

float GLTFCamera::get_fov() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("get_fov")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFCamera::set_fov(float p_fov) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("set_fov")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fov_encoded;
	PtrToArg<double>::encode(p_fov, &p_fov_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fov_encoded);
}

float GLTFCamera::get_size_mag() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("get_size_mag")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFCamera::set_size_mag(float p_size_mag) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("set_size_mag")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_mag_encoded;
	PtrToArg<double>::encode(p_size_mag, &p_size_mag_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_mag_encoded);
}

float GLTFCamera::get_depth_far() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("get_depth_far")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFCamera::set_depth_far(float p_zdepth_far) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("set_depth_far")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_zdepth_far_encoded;
	PtrToArg<double>::encode(p_zdepth_far, &p_zdepth_far_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_zdepth_far_encoded);
}

float GLTFCamera::get_depth_near() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("get_depth_near")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFCamera::set_depth_near(float p_zdepth_near) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFCamera::get_class_static()._native_ptr(), StringName("set_depth_near")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_zdepth_near_encoded;
	PtrToArg<double>::encode(p_zdepth_near, &p_zdepth_near_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_zdepth_near_encoded);
}

} // namespace godot
