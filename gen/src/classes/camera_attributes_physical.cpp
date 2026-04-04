/**************************************************************************/
/*  camera_attributes_physical.cpp                                        */
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

#include <godot_cpp/classes/camera_attributes_physical.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void CameraAttributesPhysical::set_aperture(float p_aperture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("set_aperture")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_aperture_encoded;
	PtrToArg<double>::encode(p_aperture, &p_aperture_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_aperture_encoded);
}

float CameraAttributesPhysical::get_aperture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("get_aperture")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CameraAttributesPhysical::set_shutter_speed(float p_shutter_speed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("set_shutter_speed")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_shutter_speed_encoded;
	PtrToArg<double>::encode(p_shutter_speed, &p_shutter_speed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shutter_speed_encoded);
}

float CameraAttributesPhysical::get_shutter_speed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("get_shutter_speed")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CameraAttributesPhysical::set_focal_length(float p_focal_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("set_focal_length")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_focal_length_encoded;
	PtrToArg<double>::encode(p_focal_length, &p_focal_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_focal_length_encoded);
}

float CameraAttributesPhysical::get_focal_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("get_focal_length")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CameraAttributesPhysical::set_focus_distance(float p_focus_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("set_focus_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_focus_distance_encoded;
	PtrToArg<double>::encode(p_focus_distance, &p_focus_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_focus_distance_encoded);
}

float CameraAttributesPhysical::get_focus_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("get_focus_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CameraAttributesPhysical::set_near(float p_near) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("set_near")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_near_encoded;
	PtrToArg<double>::encode(p_near, &p_near_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_near_encoded);
}

float CameraAttributesPhysical::get_near() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("get_near")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CameraAttributesPhysical::set_far(float p_far) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("set_far")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_far_encoded;
	PtrToArg<double>::encode(p_far, &p_far_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_far_encoded);
}

float CameraAttributesPhysical::get_far() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("get_far")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float CameraAttributesPhysical::get_fov() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("get_fov")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CameraAttributesPhysical::set_auto_exposure_max_exposure_value(float p_exposure_value_max) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("set_auto_exposure_max_exposure_value")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_exposure_value_max_encoded;
	PtrToArg<double>::encode(p_exposure_value_max, &p_exposure_value_max_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_exposure_value_max_encoded);
}

float CameraAttributesPhysical::get_auto_exposure_max_exposure_value() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("get_auto_exposure_max_exposure_value")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CameraAttributesPhysical::set_auto_exposure_min_exposure_value(float p_exposure_value_min) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("set_auto_exposure_min_exposure_value")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_exposure_value_min_encoded;
	PtrToArg<double>::encode(p_exposure_value_min, &p_exposure_value_min_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_exposure_value_min_encoded);
}

float CameraAttributesPhysical::get_auto_exposure_min_exposure_value() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraAttributesPhysical::get_class_static()._native_ptr(), StringName("get_auto_exposure_min_exposure_value")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
