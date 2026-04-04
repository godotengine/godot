/**************************************************************************/
/*  animation_node_blend_space2d.cpp                                      */
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

#include <godot_cpp/classes/animation_node_blend_space2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void AnimationNodeBlendSpace2D::add_blend_point(const Ref<AnimationRootNode> &p_node, const Vector2 &p_pos, int32_t p_at_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("add_blend_point")._native_ptr(), 402261981);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_at_index_encoded;
	PtrToArg<int64_t>::encode(p_at_index, &p_at_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr), &p_pos, &p_at_index_encoded);
}

void AnimationNodeBlendSpace2D::set_blend_point_position(int32_t p_point, const Vector2 &p_pos) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("set_blend_point_position")._native_ptr(), 163021252);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_point_encoded;
	PtrToArg<int64_t>::encode(p_point, &p_point_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_point_encoded, &p_pos);
}

Vector2 AnimationNodeBlendSpace2D::get_blend_point_position(int32_t p_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_blend_point_position")._native_ptr(), 2299179447);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_point_encoded;
	PtrToArg<int64_t>::encode(p_point, &p_point_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_point_encoded);
}

void AnimationNodeBlendSpace2D::set_blend_point_node(int32_t p_point, const Ref<AnimationRootNode> &p_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("set_blend_point_node")._native_ptr(), 4240341528);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_point_encoded;
	PtrToArg<int64_t>::encode(p_point, &p_point_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_point_encoded, (p_node != nullptr ? &p_node->_owner : nullptr));
}

Ref<AnimationRootNode> AnimationNodeBlendSpace2D::get_blend_point_node(int32_t p_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_blend_point_node")._native_ptr(), 665599029);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AnimationRootNode>()));
	int64_t p_point_encoded;
	PtrToArg<int64_t>::encode(p_point, &p_point_encoded);
	return Ref<AnimationRootNode>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AnimationRootNode>(_gde_method_bind, _owner, &p_point_encoded));
}

void AnimationNodeBlendSpace2D::remove_blend_point(int32_t p_point) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("remove_blend_point")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_point_encoded;
	PtrToArg<int64_t>::encode(p_point, &p_point_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_point_encoded);
}

int32_t AnimationNodeBlendSpace2D::get_blend_point_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_blend_point_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationNodeBlendSpace2D::add_triangle(int32_t p_x, int32_t p_y, int32_t p_z, int32_t p_at_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("add_triangle")._native_ptr(), 753017335);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_x_encoded;
	PtrToArg<int64_t>::encode(p_x, &p_x_encoded);
	int64_t p_y_encoded;
	PtrToArg<int64_t>::encode(p_y, &p_y_encoded);
	int64_t p_z_encoded;
	PtrToArg<int64_t>::encode(p_z, &p_z_encoded);
	int64_t p_at_index_encoded;
	PtrToArg<int64_t>::encode(p_at_index, &p_at_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_x_encoded, &p_y_encoded, &p_z_encoded, &p_at_index_encoded);
}

int32_t AnimationNodeBlendSpace2D::get_triangle_point(int32_t p_triangle, int32_t p_point) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_triangle_point")._native_ptr(), 50157827);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_triangle_encoded;
	PtrToArg<int64_t>::encode(p_triangle, &p_triangle_encoded);
	int64_t p_point_encoded;
	PtrToArg<int64_t>::encode(p_point, &p_point_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_triangle_encoded, &p_point_encoded);
}

void AnimationNodeBlendSpace2D::remove_triangle(int32_t p_triangle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("remove_triangle")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_triangle_encoded;
	PtrToArg<int64_t>::encode(p_triangle, &p_triangle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_triangle_encoded);
}

int32_t AnimationNodeBlendSpace2D::get_triangle_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_triangle_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationNodeBlendSpace2D::set_min_space(const Vector2 &p_min_space) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("set_min_space")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_min_space);
}

Vector2 AnimationNodeBlendSpace2D::get_min_space() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_min_space")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void AnimationNodeBlendSpace2D::set_max_space(const Vector2 &p_max_space) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("set_max_space")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_space);
}

Vector2 AnimationNodeBlendSpace2D::get_max_space() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_max_space")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void AnimationNodeBlendSpace2D::set_snap(const Vector2 &p_snap) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("set_snap")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_snap);
}

Vector2 AnimationNodeBlendSpace2D::get_snap() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_snap")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void AnimationNodeBlendSpace2D::set_x_label(const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("set_x_label")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text);
}

String AnimationNodeBlendSpace2D::get_x_label() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_x_label")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void AnimationNodeBlendSpace2D::set_y_label(const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("set_y_label")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text);
}

String AnimationNodeBlendSpace2D::get_y_label() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_y_label")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void AnimationNodeBlendSpace2D::set_auto_triangles(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("set_auto_triangles")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool AnimationNodeBlendSpace2D::get_auto_triangles() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_auto_triangles")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimationNodeBlendSpace2D::set_blend_mode(AnimationNodeBlendSpace2D::BlendMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("set_blend_mode")._native_ptr(), 81193520);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

AnimationNodeBlendSpace2D::BlendMode AnimationNodeBlendSpace2D::get_blend_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("get_blend_mode")._native_ptr(), 1398433632);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AnimationNodeBlendSpace2D::BlendMode(0)));
	return (AnimationNodeBlendSpace2D::BlendMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationNodeBlendSpace2D::set_use_sync(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("set_use_sync")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool AnimationNodeBlendSpace2D::is_using_sync() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNodeBlendSpace2D::get_class_static()._native_ptr(), StringName("is_using_sync")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
