/**************************************************************************/
/*  xrvrs.cpp                                                             */
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

#include <godot_cpp/classes/xrvrs.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

namespace godot {

float XRVRS::get_vrs_min_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRVRS::get_class_static()._native_ptr(), StringName("get_vrs_min_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void XRVRS::set_vrs_min_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRVRS::get_class_static()._native_ptr(), StringName("set_vrs_min_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float XRVRS::get_vrs_strength() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRVRS::get_class_static()._native_ptr(), StringName("get_vrs_strength")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void XRVRS::set_vrs_strength(float p_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRVRS::get_class_static()._native_ptr(), StringName("set_vrs_strength")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_strength_encoded);
}

Rect2i XRVRS::get_vrs_render_region() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRVRS::get_class_static()._native_ptr(), StringName("get_vrs_render_region")._native_ptr(), 410525958);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner);
}

void XRVRS::set_vrs_render_region(const Rect2i &p_render_region) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRVRS::get_class_static()._native_ptr(), StringName("set_vrs_render_region")._native_ptr(), 1763793166);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_render_region);
}

RID XRVRS::make_vrs_texture(const Vector2 &p_target_size, const PackedVector2Array &p_eye_foci) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRVRS::get_class_static()._native_ptr(), StringName("make_vrs_texture")._native_ptr(), 3647044786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_target_size, &p_eye_foci);
}

} // namespace godot
