/**************************************************************************/
/*  curve3d.cpp                                                           */
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

#include <godot_cpp/classes/curve3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

int32_t Curve3D::get_point_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_point_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Curve3D::set_point_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("set_point_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

void Curve3D::add_point(const Vector3 &p_position, const Vector3 &p_in, const Vector3 &p_out, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("add_point")._native_ptr(), 2931053748);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position, &p_in, &p_out, &p_index_encoded);
}

void Curve3D::set_point_position(int32_t p_idx, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("set_point_position")._native_ptr(), 1530502735);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_position);
}

Vector3 Curve3D::get_point_position(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_point_position")._native_ptr(), 711720468);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_idx_encoded);
}

void Curve3D::set_point_tilt(int32_t p_idx, float p_tilt) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("set_point_tilt")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	double p_tilt_encoded;
	PtrToArg<double>::encode(p_tilt, &p_tilt_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_tilt_encoded);
}

float Curve3D::get_point_tilt(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_point_tilt")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_idx_encoded);
}

void Curve3D::set_point_in(int32_t p_idx, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("set_point_in")._native_ptr(), 1530502735);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_position);
}

Vector3 Curve3D::get_point_in(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_point_in")._native_ptr(), 711720468);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_idx_encoded);
}

void Curve3D::set_point_out(int32_t p_idx, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("set_point_out")._native_ptr(), 1530502735);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_position);
}

Vector3 Curve3D::get_point_out(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_point_out")._native_ptr(), 711720468);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_idx_encoded);
}

void Curve3D::remove_point(int32_t p_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("remove_point")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded);
}

void Curve3D::clear_points() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("clear_points")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Vector3 Curve3D::sample(int32_t p_idx, float p_t) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("sample")._native_ptr(), 3285246857);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	double p_t_encoded;
	PtrToArg<double>::encode(p_t, &p_t_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_idx_encoded, &p_t_encoded);
}

Vector3 Curve3D::samplef(float p_fofs) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("samplef")._native_ptr(), 2553580215);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	double p_fofs_encoded;
	PtrToArg<double>::encode(p_fofs, &p_fofs_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_fofs_encoded);
}

void Curve3D::set_closed(bool p_closed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("set_closed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_closed_encoded;
	PtrToArg<bool>::encode(p_closed, &p_closed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_closed_encoded);
}

bool Curve3D::is_closed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("is_closed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Curve3D::set_bake_interval(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("set_bake_interval")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float Curve3D::get_bake_interval() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_bake_interval")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Curve3D::set_up_vector_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("set_up_vector_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Curve3D::is_up_vector_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("is_up_vector_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

float Curve3D::get_baked_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_baked_length")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

Vector3 Curve3D::sample_baked(float p_offset, bool p_cubic) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("sample_baked")._native_ptr(), 1350085894);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	int8_t p_cubic_encoded;
	PtrToArg<bool>::encode(p_cubic, &p_cubic_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_offset_encoded, &p_cubic_encoded);
}

Transform3D Curve3D::sample_baked_with_rotation(float p_offset, bool p_cubic, bool p_apply_tilt) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("sample_baked_with_rotation")._native_ptr(), 1939359131);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	int8_t p_cubic_encoded;
	PtrToArg<bool>::encode(p_cubic, &p_cubic_encoded);
	int8_t p_apply_tilt_encoded;
	PtrToArg<bool>::encode(p_apply_tilt, &p_apply_tilt_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_offset_encoded, &p_cubic_encoded, &p_apply_tilt_encoded);
}

Vector3 Curve3D::sample_baked_up_vector(float p_offset, bool p_apply_tilt) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("sample_baked_up_vector")._native_ptr(), 1362627031);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	int8_t p_apply_tilt_encoded;
	PtrToArg<bool>::encode(p_apply_tilt, &p_apply_tilt_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_offset_encoded, &p_apply_tilt_encoded);
}

PackedVector3Array Curve3D::get_baked_points() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_baked_points")._native_ptr(), 497664490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner);
}

PackedFloat32Array Curve3D::get_baked_tilts() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_baked_tilts")._native_ptr(), 675695659);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedFloat32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedFloat32Array>(_gde_method_bind, _owner);
}

PackedVector3Array Curve3D::get_baked_up_vectors() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_baked_up_vectors")._native_ptr(), 497664490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner);
}

Vector3 Curve3D::get_closest_point(const Vector3 &p_to_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_closest_point")._native_ptr(), 192990374);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_to_point);
}

float Curve3D::get_closest_offset(const Vector3 &p_to_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("get_closest_offset")._native_ptr(), 1109078154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_to_point);
}

PackedVector3Array Curve3D::tessellate(int32_t p_max_stages, float p_tolerance_degrees) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("tessellate")._native_ptr(), 1519759391);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	int64_t p_max_stages_encoded;
	PtrToArg<int64_t>::encode(p_max_stages, &p_max_stages_encoded);
	double p_tolerance_degrees_encoded;
	PtrToArg<double>::encode(p_tolerance_degrees, &p_tolerance_degrees_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_max_stages_encoded, &p_tolerance_degrees_encoded);
}

PackedVector3Array Curve3D::tessellate_even_length(int32_t p_max_stages, float p_tolerance_length) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Curve3D::get_class_static()._native_ptr(), StringName("tessellate_even_length")._native_ptr(), 133237049);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	int64_t p_max_stages_encoded;
	PtrToArg<int64_t>::encode(p_max_stages, &p_max_stages_encoded);
	double p_tolerance_length_encoded;
	PtrToArg<double>::encode(p_tolerance_length, &p_tolerance_length_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_max_stages_encoded, &p_tolerance_length_encoded);
}

} // namespace godot
