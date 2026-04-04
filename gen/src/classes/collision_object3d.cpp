/**************************************************************************/
/*  collision_object3d.cpp                                                */
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

#include <godot_cpp/classes/collision_object3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/input_event.hpp>
#include <godot_cpp/classes/shape3d.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/vector3.hpp>

namespace godot {

void CollisionObject3D::set_collision_layer(uint32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("set_collision_layer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded);
}

uint32_t CollisionObject3D::get_collision_layer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("get_collision_layer")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CollisionObject3D::set_collision_mask(uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("set_collision_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mask_encoded);
}

uint32_t CollisionObject3D::get_collision_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("get_collision_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CollisionObject3D::set_collision_layer_value(int32_t p_layer_number, bool p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("set_collision_layer_value")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	int8_t p_value_encoded;
	PtrToArg<bool>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_number_encoded, &p_value_encoded);
}

bool CollisionObject3D::get_collision_layer_value(int32_t p_layer_number) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("get_collision_layer_value")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_number_encoded);
}

void CollisionObject3D::set_collision_mask_value(int32_t p_layer_number, bool p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("set_collision_mask_value")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	int8_t p_value_encoded;
	PtrToArg<bool>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_number_encoded, &p_value_encoded);
}

bool CollisionObject3D::get_collision_mask_value(int32_t p_layer_number) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("get_collision_mask_value")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_number_encoded);
}

void CollisionObject3D::set_collision_priority(float p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("set_collision_priority")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_priority_encoded;
	PtrToArg<double>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_priority_encoded);
}

float CollisionObject3D::get_collision_priority() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("get_collision_priority")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CollisionObject3D::set_disable_mode(CollisionObject3D::DisableMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("set_disable_mode")._native_ptr(), 1623620376);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

CollisionObject3D::DisableMode CollisionObject3D::get_disable_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("get_disable_mode")._native_ptr(), 410164780);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CollisionObject3D::DisableMode(0)));
	return (CollisionObject3D::DisableMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CollisionObject3D::set_ray_pickable(bool p_ray_pickable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("set_ray_pickable")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_ray_pickable_encoded;
	PtrToArg<bool>::encode(p_ray_pickable, &p_ray_pickable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ray_pickable_encoded);
}

bool CollisionObject3D::is_ray_pickable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("is_ray_pickable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CollisionObject3D::set_capture_input_on_drag(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("set_capture_input_on_drag")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool CollisionObject3D::get_capture_input_on_drag() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("get_capture_input_on_drag")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

RID CollisionObject3D::get_rid() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("get_rid")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

uint32_t CollisionObject3D::create_shape_owner(Object *p_owner) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("create_shape_owner")._native_ptr(), 3429307534);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_owner != nullptr ? &p_owner->_owner : nullptr));
}

void CollisionObject3D::remove_shape_owner(uint32_t p_owner_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("remove_shape_owner")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_owner_id_encoded);
}

PackedInt32Array CollisionObject3D::get_shape_owners() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("get_shape_owners")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void CollisionObject3D::shape_owner_set_transform(uint32_t p_owner_id, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("shape_owner_set_transform")._native_ptr(), 3616898986);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_owner_id_encoded, &p_transform);
}

Transform3D CollisionObject3D::shape_owner_get_transform(uint32_t p_owner_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("shape_owner_get_transform")._native_ptr(), 1965739696);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_owner_id_encoded);
}

Object *CollisionObject3D::shape_owner_get_owner(uint32_t p_owner_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("shape_owner_get_owner")._native_ptr(), 3332903315);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Object>(_gde_method_bind, _owner, &p_owner_id_encoded);
}

void CollisionObject3D::shape_owner_set_disabled(uint32_t p_owner_id, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("shape_owner_set_disabled")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_owner_id_encoded, &p_disabled_encoded);
}

bool CollisionObject3D::is_shape_owner_disabled(uint32_t p_owner_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("is_shape_owner_disabled")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_owner_id_encoded);
}

void CollisionObject3D::shape_owner_add_shape(uint32_t p_owner_id, const Ref<Shape3D> &p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("shape_owner_add_shape")._native_ptr(), 2566676345);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_owner_id_encoded, (p_shape != nullptr ? &p_shape->_owner : nullptr));
}

int32_t CollisionObject3D::shape_owner_get_shape_count(uint32_t p_owner_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("shape_owner_get_shape_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_owner_id_encoded);
}

Ref<Shape3D> CollisionObject3D::shape_owner_get_shape(uint32_t p_owner_id, int32_t p_shape_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("shape_owner_get_shape")._native_ptr(), 4015519174);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Shape3D>()));
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	int64_t p_shape_id_encoded;
	PtrToArg<int64_t>::encode(p_shape_id, &p_shape_id_encoded);
	return Ref<Shape3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Shape3D>(_gde_method_bind, _owner, &p_owner_id_encoded, &p_shape_id_encoded));
}

int32_t CollisionObject3D::shape_owner_get_shape_index(uint32_t p_owner_id, int32_t p_shape_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("shape_owner_get_shape_index")._native_ptr(), 3175239445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	int64_t p_shape_id_encoded;
	PtrToArg<int64_t>::encode(p_shape_id, &p_shape_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_owner_id_encoded, &p_shape_id_encoded);
}

void CollisionObject3D::shape_owner_remove_shape(uint32_t p_owner_id, int32_t p_shape_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("shape_owner_remove_shape")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	int64_t p_shape_id_encoded;
	PtrToArg<int64_t>::encode(p_shape_id, &p_shape_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_owner_id_encoded, &p_shape_id_encoded);
}

void CollisionObject3D::shape_owner_clear_shapes(uint32_t p_owner_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("shape_owner_clear_shapes")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_owner_id_encoded);
}

uint32_t CollisionObject3D::shape_find_owner(int32_t p_shape_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CollisionObject3D::get_class_static()._native_ptr(), StringName("shape_find_owner")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_shape_index_encoded;
	PtrToArg<int64_t>::encode(p_shape_index, &p_shape_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shape_index_encoded);
}

void CollisionObject3D::_input_event(Camera3D *p_camera, const Ref<InputEvent> &p_event, const Vector3 &p_event_position, const Vector3 &p_normal, int32_t p_shape_idx) {}

void CollisionObject3D::_mouse_enter() {}

void CollisionObject3D::_mouse_exit() {}

} // namespace godot
