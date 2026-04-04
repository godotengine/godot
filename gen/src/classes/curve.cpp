/**************************************************************************/
/*  curve.cpp                                                             */
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

#include <godot_cpp/classes/curve.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

int32_t Curve::get_point_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_point_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Curve::set_point_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_point_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t Curve::add_point(const Vector2 &p_position, float p_left_tangent, float p_right_tangent, Curve::TangentMode p_left_mode, Curve::TangentMode p_right_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("add_point")._native_ptr(), 434072736);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	double p_left_tangent_encoded;
	PtrToArg<double>::encode(p_left_tangent, &p_left_tangent_encoded);
	double p_right_tangent_encoded;
	PtrToArg<double>::encode(p_right_tangent, &p_right_tangent_encoded);
	int64_t p_left_mode_encoded;
	PtrToArg<int64_t>::encode(p_left_mode, &p_left_mode_encoded);
	int64_t p_right_mode_encoded;
	PtrToArg<int64_t>::encode(p_right_mode, &p_right_mode_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_position, &p_left_tangent_encoded, &p_right_tangent_encoded, &p_left_mode_encoded, &p_right_mode_encoded);
}

void Curve::remove_point(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("remove_point")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

void Curve::clear_points() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("clear_points")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Vector2 Curve::get_point_position(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_point_position")._native_ptr(), 2299179447);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_index_encoded);
}

void Curve::set_point_value(int32_t p_index, float p_y) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_point_value")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_y_encoded;
	PtrToArg<double>::encode(p_y, &p_y_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_y_encoded);
}

int32_t Curve::set_point_offset(int32_t p_index, float p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_point_offset")._native_ptr(), 3780573764);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded, &p_offset_encoded);
}

float Curve::sample(float p_offset) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("sample")._native_ptr(), 3919130443);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_offset_encoded);
}

float Curve::sample_baked(float p_offset) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("sample_baked")._native_ptr(), 3919130443);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_offset_encoded);
}

float Curve::get_point_left_tangent(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_point_left_tangent")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded);
}

float Curve::get_point_right_tangent(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_point_right_tangent")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded);
}

Curve::TangentMode Curve::get_point_left_mode(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_point_left_mode")._native_ptr(), 426950354);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Curve::TangentMode(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (Curve::TangentMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

Curve::TangentMode Curve::get_point_right_mode(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_point_right_mode")._native_ptr(), 426950354);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Curve::TangentMode(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (Curve::TangentMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void Curve::set_point_left_tangent(int32_t p_index, float p_tangent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_point_left_tangent")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_tangent_encoded;
	PtrToArg<double>::encode(p_tangent, &p_tangent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_tangent_encoded);
}

void Curve::set_point_right_tangent(int32_t p_index, float p_tangent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_point_right_tangent")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_tangent_encoded;
	PtrToArg<double>::encode(p_tangent, &p_tangent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_tangent_encoded);
}

void Curve::set_point_left_mode(int32_t p_index, Curve::TangentMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_point_left_mode")._native_ptr(), 1217242874);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_mode_encoded);
}

void Curve::set_point_right_mode(int32_t p_index, Curve::TangentMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_point_right_mode")._native_ptr(), 1217242874);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_mode_encoded);
}

float Curve::get_min_value() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_min_value")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Curve::set_min_value(float p_min) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_min_value")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_min_encoded;
	PtrToArg<double>::encode(p_min, &p_min_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_min_encoded);
}

float Curve::get_max_value() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_max_value")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Curve::set_max_value(float p_max) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_max_value")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_max_encoded;
	PtrToArg<double>::encode(p_max, &p_max_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_encoded);
}

float Curve::get_value_range() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_value_range")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float Curve::get_min_domain() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_min_domain")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Curve::set_min_domain(float p_min) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_min_domain")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_min_encoded;
	PtrToArg<double>::encode(p_min, &p_min_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_min_encoded);
}

float Curve::get_max_domain() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_max_domain")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Curve::set_max_domain(float p_max) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_max_domain")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_max_encoded;
	PtrToArg<double>::encode(p_max, &p_max_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_encoded);
}

float Curve::get_domain_range() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_domain_range")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Curve::clean_dupes() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("clean_dupes")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Curve::bake() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("bake")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

int32_t Curve::get_bake_resolution() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("get_bake_resolution")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Curve::set_bake_resolution(int32_t p_resolution) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve::get_class_static()._native_ptr(), StringName("set_bake_resolution")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_resolution_encoded;
	PtrToArg<int64_t>::encode(p_resolution, &p_resolution_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_resolution_encoded);
}

} // namespace godot
