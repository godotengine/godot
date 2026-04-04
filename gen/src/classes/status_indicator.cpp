/**************************************************************************/
/*  status_indicator.cpp                                                  */
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

#include <godot_cpp/classes/status_indicator.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

void StatusIndicator::set_tooltip(const String &p_tooltip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StatusIndicator::get_class_static()._native_ptr(), StringName("set_tooltip")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tooltip);
}

String StatusIndicator::get_tooltip() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StatusIndicator::get_class_static()._native_ptr(), StringName("get_tooltip")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void StatusIndicator::set_icon(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StatusIndicator::get_class_static()._native_ptr(), StringName("set_icon")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> StatusIndicator::get_icon() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StatusIndicator::get_class_static()._native_ptr(), StringName("get_icon")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void StatusIndicator::set_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StatusIndicator::get_class_static()._native_ptr(), StringName("set_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool StatusIndicator::is_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StatusIndicator::get_class_static()._native_ptr(), StringName("is_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void StatusIndicator::set_menu(const NodePath &p_menu) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StatusIndicator::get_class_static()._native_ptr(), StringName("set_menu")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu);
}

NodePath StatusIndicator::get_menu() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StatusIndicator::get_class_static()._native_ptr(), StringName("get_menu")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

Rect2 StatusIndicator::get_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StatusIndicator::get_class_static()._native_ptr(), StringName("get_rect")._native_ptr(), 1639390495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner);
}

} // namespace godot
