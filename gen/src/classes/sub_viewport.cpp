/**************************************************************************/
/*  sub_viewport.cpp                                                      */
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

#include <godot_cpp/classes/sub_viewport.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void SubViewport::set_size(const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SubViewport::get_class_static()._native_ptr(), StringName("set_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector2i SubViewport::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SubViewport::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void SubViewport::set_size_2d_override(const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SubViewport::get_class_static()._native_ptr(), StringName("set_size_2d_override")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector2i SubViewport::get_size_2d_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SubViewport::get_class_static()._native_ptr(), StringName("get_size_2d_override")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void SubViewport::set_size_2d_override_stretch(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SubViewport::get_class_static()._native_ptr(), StringName("set_size_2d_override_stretch")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool SubViewport::is_size_2d_override_stretch_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SubViewport::get_class_static()._native_ptr(), StringName("is_size_2d_override_stretch_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SubViewport::set_update_mode(SubViewport::UpdateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SubViewport::get_class_static()._native_ptr(), StringName("set_update_mode")._native_ptr(), 1295690030);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

SubViewport::UpdateMode SubViewport::get_update_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SubViewport::get_class_static()._native_ptr(), StringName("get_update_mode")._native_ptr(), 2980171553);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SubViewport::UpdateMode(0)));
	return (SubViewport::UpdateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SubViewport::set_clear_mode(SubViewport::ClearMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SubViewport::get_class_static()._native_ptr(), StringName("set_clear_mode")._native_ptr(), 2834454712);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

SubViewport::ClearMode SubViewport::get_clear_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SubViewport::get_class_static()._native_ptr(), StringName("get_clear_mode")._native_ptr(), 331324495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SubViewport::ClearMode(0)));
	return (SubViewport::ClearMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
