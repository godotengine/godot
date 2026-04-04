/**************************************************************************/
/*  nine_patch_rect.cpp                                                   */
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

#include <godot_cpp/classes/nine_patch_rect.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

void NinePatchRect::set_texture(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("set_texture")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> NinePatchRect::get_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("get_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void NinePatchRect::set_patch_margin(Side p_margin, int32_t p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("set_patch_margin")._native_ptr(), 437707142);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded, &p_value_encoded);
}

int32_t NinePatchRect::get_patch_margin(Side p_margin) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("get_patch_margin")._native_ptr(), 1983885014);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_margin_encoded);
}

void NinePatchRect::set_region_rect(const Rect2 &p_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("set_region_rect")._native_ptr(), 2046264180);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rect);
}

Rect2 NinePatchRect::get_region_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("get_region_rect")._native_ptr(), 1639390495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner);
}

void NinePatchRect::set_draw_center(bool p_draw_center) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("set_draw_center")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_draw_center_encoded;
	PtrToArg<bool>::encode(p_draw_center, &p_draw_center_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_center_encoded);
}

bool NinePatchRect::is_draw_center_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("is_draw_center_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NinePatchRect::set_h_axis_stretch_mode(NinePatchRect::AxisStretchMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("set_h_axis_stretch_mode")._native_ptr(), 3219608417);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

NinePatchRect::AxisStretchMode NinePatchRect::get_h_axis_stretch_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("get_h_axis_stretch_mode")._native_ptr(), 3317113799);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NinePatchRect::AxisStretchMode(0)));
	return (NinePatchRect::AxisStretchMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void NinePatchRect::set_v_axis_stretch_mode(NinePatchRect::AxisStretchMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("set_v_axis_stretch_mode")._native_ptr(), 3219608417);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

NinePatchRect::AxisStretchMode NinePatchRect::get_v_axis_stretch_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NinePatchRect::get_class_static()._native_ptr(), StringName("get_v_axis_stretch_mode")._native_ptr(), 3317113799);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NinePatchRect::AxisStretchMode(0)));
	return (NinePatchRect::AxisStretchMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
