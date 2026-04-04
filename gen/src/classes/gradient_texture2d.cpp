/**************************************************************************/
/*  gradient_texture2d.cpp                                                */
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

#include <godot_cpp/classes/gradient_texture2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/gradient.hpp>

namespace godot {

void GradientTexture2D::set_gradient(const Ref<Gradient> &p_gradient) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("set_gradient")._native_ptr(), 2756054477);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_gradient != nullptr ? &p_gradient->_owner : nullptr));
}

Ref<Gradient> GradientTexture2D::get_gradient() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("get_gradient")._native_ptr(), 132272999);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Gradient>()));
	return Ref<Gradient>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Gradient>(_gde_method_bind, _owner));
}

void GradientTexture2D::set_width(int32_t p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("set_width")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_width_encoded);
}

void GradientTexture2D::set_height(int32_t p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("set_height")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_height_encoded);
}

void GradientTexture2D::set_use_hdr(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("set_use_hdr")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool GradientTexture2D::is_using_hdr() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("is_using_hdr")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GradientTexture2D::set_fill(GradientTexture2D::Fill p_fill) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("set_fill")._native_ptr(), 3623927636);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_fill_encoded;
	PtrToArg<int64_t>::encode(p_fill, &p_fill_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fill_encoded);
}

GradientTexture2D::Fill GradientTexture2D::get_fill() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("get_fill")._native_ptr(), 1876227217);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GradientTexture2D::Fill(0)));
	return (GradientTexture2D::Fill)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GradientTexture2D::set_fill_from(const Vector2 &p_fill_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("set_fill_from")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fill_from);
}

Vector2 GradientTexture2D::get_fill_from() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("get_fill_from")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void GradientTexture2D::set_fill_to(const Vector2 &p_fill_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("set_fill_to")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fill_to);
}

Vector2 GradientTexture2D::get_fill_to() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("get_fill_to")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void GradientTexture2D::set_repeat(GradientTexture2D::Repeat p_repeat) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("set_repeat")._native_ptr(), 1357597002);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_repeat_encoded;
	PtrToArg<int64_t>::encode(p_repeat, &p_repeat_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_repeat_encoded);
}

GradientTexture2D::Repeat GradientTexture2D::get_repeat() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GradientTexture2D::get_class_static()._native_ptr(), StringName("get_repeat")._native_ptr(), 3351758665);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GradientTexture2D::Repeat(0)));
	return (GradientTexture2D::Repeat)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
