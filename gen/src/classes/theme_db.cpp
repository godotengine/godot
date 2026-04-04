/**************************************************************************/
/*  theme_db.cpp                                                          */
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

#include <godot_cpp/classes/theme_db.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/style_box.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/theme.hpp>

namespace godot {

ThemeDB *ThemeDB::singleton = nullptr;

ThemeDB *ThemeDB::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(ThemeDB::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<ThemeDB *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &ThemeDB::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(ThemeDB::get_class_static(), singleton);
		}
	}
	return singleton;
}

ThemeDB::~ThemeDB() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(ThemeDB::get_class_static());
		singleton = nullptr;
	}
}

Ref<Theme> ThemeDB::get_default_theme() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("get_default_theme")._native_ptr(), 754276358);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Theme>()));
	return Ref<Theme>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Theme>(_gde_method_bind, _owner));
}

Ref<Theme> ThemeDB::get_project_theme() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("get_project_theme")._native_ptr(), 754276358);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Theme>()));
	return Ref<Theme>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Theme>(_gde_method_bind, _owner));
}

void ThemeDB::set_fallback_base_scale(float p_base_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("set_fallback_base_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_base_scale_encoded;
	PtrToArg<double>::encode(p_base_scale, &p_base_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_base_scale_encoded);
}

float ThemeDB::get_fallback_base_scale() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("get_fallback_base_scale")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ThemeDB::set_fallback_font(const Ref<Font> &p_font) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("set_fallback_font")._native_ptr(), 1262170328);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_font != nullptr ? &p_font->_owner : nullptr));
}

Ref<Font> ThemeDB::get_fallback_font() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("get_fallback_font")._native_ptr(), 3656929885);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Font>()));
	return Ref<Font>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Font>(_gde_method_bind, _owner));
}

void ThemeDB::set_fallback_font_size(int32_t p_font_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("set_fallback_font_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_size_encoded);
}

int32_t ThemeDB::get_fallback_font_size() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("get_fallback_font_size")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ThemeDB::set_fallback_icon(const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("set_fallback_icon")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

Ref<Texture2D> ThemeDB::get_fallback_icon() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("get_fallback_icon")._native_ptr(), 255860311);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void ThemeDB::set_fallback_stylebox(const Ref<StyleBox> &p_stylebox) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("set_fallback_stylebox")._native_ptr(), 2797200388);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_stylebox != nullptr ? &p_stylebox->_owner : nullptr));
}

Ref<StyleBox> ThemeDB::get_fallback_stylebox() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ThemeDB::get_class_static()._native_ptr(), StringName("get_fallback_stylebox")._native_ptr(), 496040854);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<StyleBox>()));
	return Ref<StyleBox>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<StyleBox>(_gde_method_bind, _owner));
}

} // namespace godot
