/**************************************************************************/
/*  canvas_texture.cpp                                                    */
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

#include <godot_cpp/classes/canvas_texture.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void CanvasTexture::set_diffuse_texture(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("set_diffuse_texture")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> CanvasTexture::get_diffuse_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("get_diffuse_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void CanvasTexture::set_normal_texture(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("set_normal_texture")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> CanvasTexture::get_normal_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("get_normal_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void CanvasTexture::set_specular_texture(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("set_specular_texture")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> CanvasTexture::get_specular_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("get_specular_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void CanvasTexture::set_specular_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("set_specular_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color CanvasTexture::get_specular_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("get_specular_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void CanvasTexture::set_specular_shininess(float p_shininess) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("set_specular_shininess")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_shininess_encoded;
	PtrToArg<double>::encode(p_shininess, &p_shininess_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shininess_encoded);
}

float CanvasTexture::get_specular_shininess() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("get_specular_shininess")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CanvasTexture::set_texture_filter(CanvasItem::TextureFilter p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("set_texture_filter")._native_ptr(), 1037999706);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_filter_encoded;
	PtrToArg<int64_t>::encode(p_filter, &p_filter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter_encoded);
}

CanvasItem::TextureFilter CanvasTexture::get_texture_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("get_texture_filter")._native_ptr(), 121960042);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CanvasItem::TextureFilter(0)));
	return (CanvasItem::TextureFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CanvasTexture::set_texture_repeat(CanvasItem::TextureRepeat p_repeat) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("set_texture_repeat")._native_ptr(), 1716472974);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_repeat_encoded;
	PtrToArg<int64_t>::encode(p_repeat, &p_repeat_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_repeat_encoded);
}

CanvasItem::TextureRepeat CanvasTexture::get_texture_repeat() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasTexture::get_class_static()._native_ptr(), StringName("get_texture_repeat")._native_ptr(), 2667158319);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CanvasItem::TextureRepeat(0)));
	return (CanvasItem::TextureRepeat)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
