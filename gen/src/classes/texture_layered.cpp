/**************************************************************************/
/*  texture_layered.cpp                                                   */
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

#include <godot_cpp/classes/texture_layered.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Image::Format TextureLayered::get_format() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextureLayered::get_class_static()._native_ptr(), StringName("get_format")._native_ptr(), 3847873762);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Image::Format(0)));
	return (Image::Format)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

TextureLayered::LayeredType TextureLayered::get_layered_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextureLayered::get_class_static()._native_ptr(), StringName("get_layered_type")._native_ptr(), 518123893);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextureLayered::LayeredType(0)));
	return (TextureLayered::LayeredType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TextureLayered::get_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextureLayered::get_class_static()._native_ptr(), StringName("get_width")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TextureLayered::get_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextureLayered::get_class_static()._native_ptr(), StringName("get_height")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TextureLayered::get_layers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextureLayered::get_class_static()._native_ptr(), StringName("get_layers")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool TextureLayered::has_mipmaps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextureLayered::get_class_static()._native_ptr(), StringName("has_mipmaps")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<Image> TextureLayered::get_layer_data(int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextureLayered::get_class_static()._native_ptr(), StringName("get_layer_data")._native_ptr(), 3655284255);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_layer_encoded));
}

Image::Format TextureLayered::_get_format() const {
	return Image::Format(0);
}

uint32_t TextureLayered::_get_layered_type() const {
	return 0;
}

int32_t TextureLayered::_get_width() const {
	return 0;
}

int32_t TextureLayered::_get_height() const {
	return 0;
}

int32_t TextureLayered::_get_layers() const {
	return 0;
}

bool TextureLayered::_has_mipmaps() const {
	return false;
}

Ref<Image> TextureLayered::_get_layer_data(int32_t p_layer_index) const {
	return Ref<Image>();
}

} // namespace godot
