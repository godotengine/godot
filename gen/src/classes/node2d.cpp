/**************************************************************************/
/*  node2d.cpp                                                            */
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

#include <godot_cpp/classes/node2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/node.hpp>

namespace godot {

void Node2D::set_position(const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_position")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

void Node2D::set_rotation(float p_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_rotation")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radians_encoded;
	PtrToArg<double>::encode(p_radians, &p_radians_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radians_encoded);
}

void Node2D::set_rotation_degrees(float p_degrees) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_rotation_degrees")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_degrees_encoded;
	PtrToArg<double>::encode(p_degrees, &p_degrees_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_degrees_encoded);
}

void Node2D::set_skew(float p_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_skew")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radians_encoded;
	PtrToArg<double>::encode(p_radians, &p_radians_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radians_encoded);
}

void Node2D::set_scale(const Vector2 &p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_scale")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale);
}

Vector2 Node2D::get_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

float Node2D::get_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_rotation")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float Node2D::get_rotation_degrees() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_rotation_degrees")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float Node2D::get_skew() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_skew")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

Vector2 Node2D::get_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_scale")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Node2D::rotate(float p_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("rotate")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radians_encoded;
	PtrToArg<double>::encode(p_radians, &p_radians_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radians_encoded);
}

void Node2D::move_local_x(float p_delta, bool p_scaled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("move_local_x")._native_ptr(), 2087892650);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_delta_encoded;
	PtrToArg<double>::encode(p_delta, &p_delta_encoded);
	int8_t p_scaled_encoded;
	PtrToArg<bool>::encode(p_scaled, &p_scaled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_delta_encoded, &p_scaled_encoded);
}

void Node2D::move_local_y(float p_delta, bool p_scaled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("move_local_y")._native_ptr(), 2087892650);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_delta_encoded;
	PtrToArg<double>::encode(p_delta, &p_delta_encoded);
	int8_t p_scaled_encoded;
	PtrToArg<bool>::encode(p_scaled, &p_scaled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_delta_encoded, &p_scaled_encoded);
}

void Node2D::translate(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("translate")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

void Node2D::global_translate(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("global_translate")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

void Node2D::apply_scale(const Vector2 &p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("apply_scale")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio);
}

void Node2D::set_global_position(const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_global_position")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

Vector2 Node2D::get_global_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_global_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Node2D::set_global_rotation(float p_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_global_rotation")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radians_encoded;
	PtrToArg<double>::encode(p_radians, &p_radians_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radians_encoded);
}

void Node2D::set_global_rotation_degrees(float p_degrees) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_global_rotation_degrees")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_degrees_encoded;
	PtrToArg<double>::encode(p_degrees, &p_degrees_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_degrees_encoded);
}

float Node2D::get_global_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_global_rotation")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float Node2D::get_global_rotation_degrees() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_global_rotation_degrees")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Node2D::set_global_skew(float p_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_global_skew")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radians_encoded;
	PtrToArg<double>::encode(p_radians, &p_radians_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radians_encoded);
}

float Node2D::get_global_skew() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_global_skew")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Node2D::set_global_scale(const Vector2 &p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_global_scale")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale);
}

Vector2 Node2D::get_global_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_global_scale")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Node2D::set_transform(const Transform2D &p_xform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_transform")._native_ptr(), 2761652528);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_xform);
}

void Node2D::set_global_transform(const Transform2D &p_xform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("set_global_transform")._native_ptr(), 2761652528);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_xform);
}

void Node2D::look_at(const Vector2 &p_point) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("look_at")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_point);
}

float Node2D::get_angle_to(const Vector2 &p_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_angle_to")._native_ptr(), 2276447920);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_point);
}

Vector2 Node2D::to_local(const Vector2 &p_global_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("to_local")._native_ptr(), 2656412154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_global_point);
}

Vector2 Node2D::to_global(const Vector2 &p_local_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("to_global")._native_ptr(), 2656412154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_local_point);
}

Transform2D Node2D::get_relative_transform_to_parent(Node *p_parent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node2D::get_class_static()._native_ptr(), StringName("get_relative_transform_to_parent")._native_ptr(), 904556875);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner, (p_parent != nullptr ? &p_parent->_owner : nullptr));
}

} // namespace godot
