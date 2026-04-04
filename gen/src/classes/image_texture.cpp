/**************************************************************************/
/*  image_texture.cpp                                                     */
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

#include <godot_cpp/classes/image_texture.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/vector2i.hpp>

namespace godot {

Ref<ImageTexture> ImageTexture::create_from_image(const Ref<Image> &p_image) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImageTexture::get_class_static()._native_ptr(), StringName("create_from_image")._native_ptr(), 2775144163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ImageTexture>()));
	return Ref<ImageTexture>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ImageTexture>(_gde_method_bind, nullptr, (p_image != nullptr ? &p_image->_owner : nullptr)));
}

Image::Format ImageTexture::get_format() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImageTexture::get_class_static()._native_ptr(), StringName("get_format")._native_ptr(), 3847873762);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Image::Format(0)));
	return (Image::Format)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ImageTexture::set_image(const Ref<Image> &p_image) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImageTexture::get_class_static()._native_ptr(), StringName("set_image")._native_ptr(), 532598488);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_image != nullptr ? &p_image->_owner : nullptr));
}

void ImageTexture::update(const Ref<Image> &p_image) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImageTexture::get_class_static()._native_ptr(), StringName("update")._native_ptr(), 532598488);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_image != nullptr ? &p_image->_owner : nullptr));
}

void ImageTexture::set_size_override(const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ImageTexture::get_class_static()._native_ptr(), StringName("set_size_override")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

} // namespace godot
