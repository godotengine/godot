/**************************************************************************/
/*  portable_compressed_texture2d.cpp                                     */
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

#include <godot_cpp/classes/portable_compressed_texture2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void PortableCompressedTexture2D::create_from_image(const Ref<Image> &p_image, PortableCompressedTexture2D::CompressionMode p_compression_mode, bool p_normal_map, float p_lossy_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PortableCompressedTexture2D::get_class_static()._native_ptr(), StringName("create_from_image")._native_ptr(), 3679243433);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_compression_mode_encoded;
	PtrToArg<int64_t>::encode(p_compression_mode, &p_compression_mode_encoded);
	int8_t p_normal_map_encoded;
	PtrToArg<bool>::encode(p_normal_map, &p_normal_map_encoded);
	double p_lossy_quality_encoded;
	PtrToArg<double>::encode(p_lossy_quality, &p_lossy_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_image != nullptr ? &p_image->_owner : nullptr), &p_compression_mode_encoded, &p_normal_map_encoded, &p_lossy_quality_encoded);
}

Image::Format PortableCompressedTexture2D::get_format() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PortableCompressedTexture2D::get_class_static()._native_ptr(), StringName("get_format")._native_ptr(), 3847873762);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Image::Format(0)));
	return (Image::Format)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

PortableCompressedTexture2D::CompressionMode PortableCompressedTexture2D::get_compression_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PortableCompressedTexture2D::get_class_static()._native_ptr(), StringName("get_compression_mode")._native_ptr(), 3265612739);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PortableCompressedTexture2D::CompressionMode(0)));
	return (PortableCompressedTexture2D::CompressionMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void PortableCompressedTexture2D::set_size_override(const Vector2 &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PortableCompressedTexture2D::get_class_static()._native_ptr(), StringName("set_size_override")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector2 PortableCompressedTexture2D::get_size_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PortableCompressedTexture2D::get_class_static()._native_ptr(), StringName("get_size_override")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void PortableCompressedTexture2D::set_keep_compressed_buffer(bool p_keep) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PortableCompressedTexture2D::get_class_static()._native_ptr(), StringName("set_keep_compressed_buffer")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_encoded;
	PtrToArg<bool>::encode(p_keep, &p_keep_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_keep_encoded);
}

bool PortableCompressedTexture2D::is_keeping_compressed_buffer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PortableCompressedTexture2D::get_class_static()._native_ptr(), StringName("is_keeping_compressed_buffer")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PortableCompressedTexture2D::set_basisu_compressor_params(int32_t p_uastc_level, float p_rdo_quality_loss) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PortableCompressedTexture2D::get_class_static()._native_ptr(), StringName("set_basisu_compressor_params")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_uastc_level_encoded;
	PtrToArg<int64_t>::encode(p_uastc_level, &p_uastc_level_encoded);
	double p_rdo_quality_loss_encoded;
	PtrToArg<double>::encode(p_rdo_quality_loss, &p_rdo_quality_loss_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_uastc_level_encoded, &p_rdo_quality_loss_encoded);
}

void PortableCompressedTexture2D::set_keep_all_compressed_buffers(bool p_keep) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PortableCompressedTexture2D::get_class_static()._native_ptr(), StringName("set_keep_all_compressed_buffers")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_encoded;
	PtrToArg<bool>::encode(p_keep, &p_keep_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, nullptr, &p_keep_encoded);
}

bool PortableCompressedTexture2D::is_keeping_all_compressed_buffers() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PortableCompressedTexture2D::get_class_static()._native_ptr(), StringName("is_keeping_all_compressed_buffers")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, nullptr);
}

} // namespace godot
