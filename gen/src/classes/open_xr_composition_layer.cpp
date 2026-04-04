/**************************************************************************/
/*  open_xr_composition_layer.cpp                                         */
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

#include <godot_cpp/classes/open_xr_composition_layer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/java_object.hpp>
#include <godot_cpp/classes/sub_viewport.hpp>
#include <godot_cpp/variant/vector3.hpp>

namespace godot {

void OpenXRCompositionLayer::set_layer_viewport(SubViewport *p_viewport) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_layer_viewport")._native_ptr(), 3888077664);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_viewport != nullptr ? &p_viewport->_owner : nullptr));
}

SubViewport *OpenXRCompositionLayer::get_layer_viewport() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_layer_viewport")._native_ptr(), 3750751911);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<SubViewport>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_use_android_surface(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_use_android_surface")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool OpenXRCompositionLayer::get_use_android_surface() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_use_android_surface")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_android_surface_size(const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_android_surface_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector2i OpenXRCompositionLayer::get_android_surface_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_android_surface_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_enable_hole_punch(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_enable_hole_punch")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool OpenXRCompositionLayer::get_enable_hole_punch() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_enable_hole_punch")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_sort_order(int32_t p_order) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_sort_order")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_order_encoded;
	PtrToArg<int64_t>::encode(p_order, &p_order_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_order_encoded);
}

int32_t OpenXRCompositionLayer::get_sort_order() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_sort_order")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_alpha_blend(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_alpha_blend")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool OpenXRCompositionLayer::get_alpha_blend() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_alpha_blend")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<JavaObject> OpenXRCompositionLayer::get_android_surface() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_android_surface")._native_ptr(), 3277089691);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<JavaObject>()));
	return Ref<JavaObject>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<JavaObject>(_gde_method_bind, _owner));
}

bool OpenXRCompositionLayer::is_natively_supported() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("is_natively_supported")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool OpenXRCompositionLayer::is_protected_content() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("is_protected_content")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_protected_content(bool p_protected_content) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_protected_content")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_protected_content_encoded;
	PtrToArg<bool>::encode(p_protected_content, &p_protected_content_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_protected_content_encoded);
}

void OpenXRCompositionLayer::set_min_filter(OpenXRCompositionLayer::Filter p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_min_filter")._native_ptr(), 3653437593);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

OpenXRCompositionLayer::Filter OpenXRCompositionLayer::get_min_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_min_filter")._native_ptr(), 845677307);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRCompositionLayer::Filter(0)));
	return (OpenXRCompositionLayer::Filter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_mag_filter(OpenXRCompositionLayer::Filter p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_mag_filter")._native_ptr(), 3653437593);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

OpenXRCompositionLayer::Filter OpenXRCompositionLayer::get_mag_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_mag_filter")._native_ptr(), 845677307);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRCompositionLayer::Filter(0)));
	return (OpenXRCompositionLayer::Filter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_mipmap_mode(OpenXRCompositionLayer::MipmapMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_mipmap_mode")._native_ptr(), 3271133183);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

OpenXRCompositionLayer::MipmapMode OpenXRCompositionLayer::get_mipmap_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_mipmap_mode")._native_ptr(), 3962697095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRCompositionLayer::MipmapMode(0)));
	return (OpenXRCompositionLayer::MipmapMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_horizontal_wrap(OpenXRCompositionLayer::Wrap p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_horizontal_wrap")._native_ptr(), 15634990);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

OpenXRCompositionLayer::Wrap OpenXRCompositionLayer::get_horizontal_wrap() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_horizontal_wrap")._native_ptr(), 2798816834);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRCompositionLayer::Wrap(0)));
	return (OpenXRCompositionLayer::Wrap)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_vertical_wrap(OpenXRCompositionLayer::Wrap p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_vertical_wrap")._native_ptr(), 15634990);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

OpenXRCompositionLayer::Wrap OpenXRCompositionLayer::get_vertical_wrap() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_vertical_wrap")._native_ptr(), 2798816834);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRCompositionLayer::Wrap(0)));
	return (OpenXRCompositionLayer::Wrap)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_red_swizzle(OpenXRCompositionLayer::Swizzle p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_red_swizzle")._native_ptr(), 741598951);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

OpenXRCompositionLayer::Swizzle OpenXRCompositionLayer::get_red_swizzle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_red_swizzle")._native_ptr(), 2334776767);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRCompositionLayer::Swizzle(0)));
	return (OpenXRCompositionLayer::Swizzle)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_green_swizzle(OpenXRCompositionLayer::Swizzle p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_green_swizzle")._native_ptr(), 741598951);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

OpenXRCompositionLayer::Swizzle OpenXRCompositionLayer::get_green_swizzle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_green_swizzle")._native_ptr(), 2334776767);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRCompositionLayer::Swizzle(0)));
	return (OpenXRCompositionLayer::Swizzle)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_blue_swizzle(OpenXRCompositionLayer::Swizzle p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_blue_swizzle")._native_ptr(), 741598951);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

OpenXRCompositionLayer::Swizzle OpenXRCompositionLayer::get_blue_swizzle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_blue_swizzle")._native_ptr(), 2334776767);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRCompositionLayer::Swizzle(0)));
	return (OpenXRCompositionLayer::Swizzle)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_alpha_swizzle(OpenXRCompositionLayer::Swizzle p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_alpha_swizzle")._native_ptr(), 741598951);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

OpenXRCompositionLayer::Swizzle OpenXRCompositionLayer::get_alpha_swizzle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_alpha_swizzle")._native_ptr(), 2334776767);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRCompositionLayer::Swizzle(0)));
	return (OpenXRCompositionLayer::Swizzle)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_max_anisotropy(float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_max_anisotropy")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

float OpenXRCompositionLayer::get_max_anisotropy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_max_anisotropy")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRCompositionLayer::set_border_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("set_border_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color OpenXRCompositionLayer::get_border_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("get_border_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

Vector2 OpenXRCompositionLayer::intersects_ray(const Vector3 &p_origin, const Vector3 &p_direction) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRCompositionLayer::get_class_static()._native_ptr(), StringName("intersects_ray")._native_ptr(), 1091262597);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_origin, &p_direction);
}

} // namespace godot
