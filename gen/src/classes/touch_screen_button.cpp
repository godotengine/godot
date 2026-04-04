/**************************************************************************/
/*  touch_screen_button.cpp                                               */
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

#include <godot_cpp/classes/touch_screen_button.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/bit_map.hpp>
#include <godot_cpp/classes/shape2d.hpp>
#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

void TouchScreenButton::set_texture_normal(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("set_texture_normal")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> TouchScreenButton::get_texture_normal() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("get_texture_normal")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void TouchScreenButton::set_texture_pressed(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("set_texture_pressed")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> TouchScreenButton::get_texture_pressed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("get_texture_pressed")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void TouchScreenButton::set_bitmask(const Ref<BitMap> &p_bitmask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("set_bitmask")._native_ptr(), 698588216);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_bitmask != nullptr ? &p_bitmask->_owner : nullptr));
}

Ref<BitMap> TouchScreenButton::get_bitmask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("get_bitmask")._native_ptr(), 2459671998);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<BitMap>()));
	return Ref<BitMap>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<BitMap>(_gde_method_bind, _owner));
}

void TouchScreenButton::set_shape(const Ref<Shape2D> &p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("set_shape")._native_ptr(), 771364740);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_shape != nullptr ? &p_shape->_owner : nullptr));
}

Ref<Shape2D> TouchScreenButton::get_shape() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("get_shape")._native_ptr(), 522005891);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Shape2D>()));
	return Ref<Shape2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Shape2D>(_gde_method_bind, _owner));
}

void TouchScreenButton::set_shape_centered(bool p_bool) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("set_shape_centered")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_bool_encoded;
	PtrToArg<bool>::encode(p_bool, &p_bool_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bool_encoded);
}

bool TouchScreenButton::is_shape_centered() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("is_shape_centered")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TouchScreenButton::set_shape_visible(bool p_bool) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("set_shape_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_bool_encoded;
	PtrToArg<bool>::encode(p_bool, &p_bool_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bool_encoded);
}

bool TouchScreenButton::is_shape_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("is_shape_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TouchScreenButton::set_action(const String &p_action) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("set_action")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action);
}

String TouchScreenButton::get_action() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("get_action")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void TouchScreenButton::set_visibility_mode(TouchScreenButton::VisibilityMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("set_visibility_mode")._native_ptr(), 3031128463);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

TouchScreenButton::VisibilityMode TouchScreenButton::get_visibility_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("get_visibility_mode")._native_ptr(), 2558996468);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TouchScreenButton::VisibilityMode(0)));
	return (TouchScreenButton::VisibilityMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TouchScreenButton::set_passby_press(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("set_passby_press")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TouchScreenButton::is_passby_press_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("is_passby_press_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool TouchScreenButton::is_pressed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TouchScreenButton::get_class_static()._native_ptr(), StringName("is_pressed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
