/**************************************************************************/
/*  open_xr_composition_layer_equirect.cpp                                */
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

#include <godot_cpp/classes/open_xr_composition_layer_equirect.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void OpenXRCompositionLayerEquirect::set_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayerEquirect::get_class_static()._native_ptr(), StringName("set_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float OpenXRCompositionLayerEquirect::get_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayerEquirect::get_class_static()._native_ptr(), StringName("get_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayerEquirect::set_central_horizontal_angle(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayerEquirect::get_class_static()._native_ptr(), StringName("set_central_horizontal_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

float OpenXRCompositionLayerEquirect::get_central_horizontal_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayerEquirect::get_class_static()._native_ptr(), StringName("get_central_horizontal_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayerEquirect::set_upper_vertical_angle(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayerEquirect::get_class_static()._native_ptr(), StringName("set_upper_vertical_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

float OpenXRCompositionLayerEquirect::get_upper_vertical_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayerEquirect::get_class_static()._native_ptr(), StringName("get_upper_vertical_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayerEquirect::set_lower_vertical_angle(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayerEquirect::get_class_static()._native_ptr(), StringName("set_lower_vertical_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

float OpenXRCompositionLayerEquirect::get_lower_vertical_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayerEquirect::get_class_static()._native_ptr(), StringName("get_lower_vertical_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayerEquirect::set_fallback_segments(uint32_t p_segments) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayerEquirect::get_class_static()._native_ptr(), StringName("set_fallback_segments")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_segments_encoded;
	PtrToArg<int64_t>::encode(p_segments, &p_segments_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_segments_encoded);
}

uint32_t OpenXRCompositionLayerEquirect::get_fallback_segments() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayerEquirect::get_class_static()._native_ptr(), StringName("get_fallback_segments")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
