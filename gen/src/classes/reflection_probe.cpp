/**************************************************************************/
/*  reflection_probe.cpp                                                  */
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

#include <godot_cpp/classes/reflection_probe.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void ReflectionProbe::set_intensity(float p_intensity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_intensity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_intensity_encoded;
	PtrToArg<double>::encode(p_intensity, &p_intensity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_intensity_encoded);
}

float ReflectionProbe::get_intensity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_intensity")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_blend_distance(float p_blend_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_blend_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_blend_distance_encoded;
	PtrToArg<double>::encode(p_blend_distance, &p_blend_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_blend_distance_encoded);
}

float ReflectionProbe::get_blend_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_blend_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_ambient_mode(ReflectionProbe::AmbientMode p_ambient) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_ambient_mode")._native_ptr(), 1748981278);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_ambient_encoded;
	PtrToArg<int64_t>::encode(p_ambient, &p_ambient_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ambient_encoded);
}

ReflectionProbe::AmbientMode ReflectionProbe::get_ambient_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_ambient_mode")._native_ptr(), 1014607621);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ReflectionProbe::AmbientMode(0)));
	return (ReflectionProbe::AmbientMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_ambient_color(const Color &p_ambient) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_ambient_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ambient);
}

Color ReflectionProbe::get_ambient_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_ambient_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_ambient_color_energy(float p_ambient_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_ambient_color_energy")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ambient_energy_encoded;
	PtrToArg<double>::encode(p_ambient_energy, &p_ambient_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ambient_energy_encoded);
}

float ReflectionProbe::get_ambient_color_energy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_ambient_color_energy")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_max_distance(float p_max_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_max_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_max_distance_encoded;
	PtrToArg<double>::encode(p_max_distance, &p_max_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_distance_encoded);
}

float ReflectionProbe::get_max_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_max_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_mesh_lod_threshold(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_mesh_lod_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

float ReflectionProbe::get_mesh_lod_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_mesh_lod_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_size(const Vector3 &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_size")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector3 ReflectionProbe::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_origin_offset(const Vector3 &p_origin_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_origin_offset")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_origin_offset);
}

Vector3 ReflectionProbe::get_origin_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_origin_offset")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_as_interior(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_as_interior")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool ReflectionProbe::is_set_as_interior() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("is_set_as_interior")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_enable_box_projection(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_enable_box_projection")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool ReflectionProbe::is_box_projection_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("is_box_projection_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_enable_shadows(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_enable_shadows")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool ReflectionProbe::are_shadows_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("are_shadows_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_cull_mask(uint32_t p_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_cull_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layers_encoded);
}

uint32_t ReflectionProbe::get_cull_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_cull_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_reflection_mask(uint32_t p_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_reflection_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layers_encoded);
}

uint32_t ReflectionProbe::get_reflection_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_reflection_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ReflectionProbe::set_update_mode(ReflectionProbe::UpdateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("set_update_mode")._native_ptr(), 4090221187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

ReflectionProbe::UpdateMode ReflectionProbe::get_update_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ReflectionProbe::get_class_static()._native_ptr(), StringName("get_update_mode")._native_ptr(), 2367550552);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ReflectionProbe::UpdateMode(0)));
	return (ReflectionProbe::UpdateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
