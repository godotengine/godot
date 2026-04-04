/**************************************************************************/
/*  mobile_vr_interface.cpp                                               */
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

#include <godot_cpp/classes/mobile_vr_interface.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void MobileVRInterface::set_eye_height(double p_eye_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("set_eye_height")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_eye_height_encoded;
	PtrToArg<double>::encode(p_eye_height, &p_eye_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_eye_height_encoded);
}

double MobileVRInterface::get_eye_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("get_eye_height")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void MobileVRInterface::set_iod(double p_iod) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("set_iod")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_iod_encoded;
	PtrToArg<double>::encode(p_iod, &p_iod_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_iod_encoded);
}

double MobileVRInterface::get_iod() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("get_iod")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void MobileVRInterface::set_display_width(double p_display_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("set_display_width")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_display_width_encoded;
	PtrToArg<double>::encode(p_display_width, &p_display_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_display_width_encoded);
}

double MobileVRInterface::get_display_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("get_display_width")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void MobileVRInterface::set_display_to_lens(double p_display_to_lens) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("set_display_to_lens")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_display_to_lens_encoded;
	PtrToArg<double>::encode(p_display_to_lens, &p_display_to_lens_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_display_to_lens_encoded);
}

double MobileVRInterface::get_display_to_lens() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("get_display_to_lens")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void MobileVRInterface::set_offset_rect(const Rect2 &p_offset_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("set_offset_rect")._native_ptr(), 2046264180);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset_rect);
}

Rect2 MobileVRInterface::get_offset_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("get_offset_rect")._native_ptr(), 1639390495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner);
}

void MobileVRInterface::set_oversample(double p_oversample) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("set_oversample")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_oversample_encoded;
	PtrToArg<double>::encode(p_oversample, &p_oversample_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_oversample_encoded);
}

double MobileVRInterface::get_oversample() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("get_oversample")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void MobileVRInterface::set_k1(double p_k) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("set_k1")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_k_encoded;
	PtrToArg<double>::encode(p_k, &p_k_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_k_encoded);
}

double MobileVRInterface::get_k1() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("get_k1")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void MobileVRInterface::set_k2(double p_k) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("set_k2")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_k_encoded;
	PtrToArg<double>::encode(p_k, &p_k_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_k_encoded);
}

double MobileVRInterface::get_k2() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("get_k2")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float MobileVRInterface::get_vrs_min_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("get_vrs_min_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void MobileVRInterface::set_vrs_min_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("set_vrs_min_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float MobileVRInterface::get_vrs_strength() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("get_vrs_strength")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void MobileVRInterface::set_vrs_strength(float p_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MobileVRInterface::get_class_static()._native_ptr(), StringName("set_vrs_strength")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_strength_encoded);
}

} // namespace godot
