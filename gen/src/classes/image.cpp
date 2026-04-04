/**************************************************************************/
/*  image.cpp                                                             */
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

#include <godot_cpp/classes/image.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/string.hpp>

namespace godot {

int32_t Image::get_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_width")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t Image::get_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_height")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Vector2i Image::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

bool Image::has_mipmaps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("has_mipmaps")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Image::Format Image::get_format() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_format")._native_ptr(), 3847873762);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Image::Format(0)));
	return (Image::Format)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

PackedByteArray Image::get_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_data")._native_ptr(), 2362200018);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner);
}

int64_t Image::get_data_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_data_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Image::convert(Image::Format p_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("convert")._native_ptr(), 2120693146);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_format_encoded);
}

int32_t Image::get_mipmap_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_mipmap_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int64_t Image::get_mipmap_offset(int32_t p_mipmap) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_mipmap_offset")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_mipmap_encoded;
	PtrToArg<int64_t>::encode(p_mipmap, &p_mipmap_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_mipmap_encoded);
}

void Image::resize_to_po2(bool p_square, Image::Interpolation p_interpolation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("resize_to_po2")._native_ptr(), 4189212329);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_square_encoded;
	PtrToArg<bool>::encode(p_square, &p_square_encoded);
	int64_t p_interpolation_encoded;
	PtrToArg<int64_t>::encode(p_interpolation, &p_interpolation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_square_encoded, &p_interpolation_encoded);
}

void Image::resize(int32_t p_width, int32_t p_height, Image::Interpolation p_interpolation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("resize")._native_ptr(), 994498151);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int64_t p_interpolation_encoded;
	PtrToArg<int64_t>::encode(p_interpolation, &p_interpolation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_width_encoded, &p_height_encoded, &p_interpolation_encoded);
}

void Image::shrink_x2() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("shrink_x2")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Image::crop(int32_t p_width, int32_t p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("crop")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_width_encoded, &p_height_encoded);
}

void Image::flip_x() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("flip_x")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Image::flip_y() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("flip_y")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Error Image::generate_mipmaps(bool p_renormalize) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("generate_mipmaps")._native_ptr(), 1633102583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_renormalize_encoded;
	PtrToArg<bool>::encode(p_renormalize, &p_renormalize_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_renormalize_encoded);
}

void Image::clear_mipmaps() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("clear_mipmaps")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Ref<Image> Image::create(int32_t p_width, int32_t p_height, bool p_use_mipmaps, Image::Format p_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("create")._native_ptr(), 986942177);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int8_t p_use_mipmaps_encoded;
	PtrToArg<bool>::encode(p_use_mipmaps, &p_use_mipmaps_encoded);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, nullptr, &p_width_encoded, &p_height_encoded, &p_use_mipmaps_encoded, &p_format_encoded));
}

Ref<Image> Image::create_empty(int32_t p_width, int32_t p_height, bool p_use_mipmaps, Image::Format p_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("create_empty")._native_ptr(), 986942177);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int8_t p_use_mipmaps_encoded;
	PtrToArg<bool>::encode(p_use_mipmaps, &p_use_mipmaps_encoded);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, nullptr, &p_width_encoded, &p_height_encoded, &p_use_mipmaps_encoded, &p_format_encoded));
}

Ref<Image> Image::create_from_data(int32_t p_width, int32_t p_height, bool p_use_mipmaps, Image::Format p_format, const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("create_from_data")._native_ptr(), 299398494);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int8_t p_use_mipmaps_encoded;
	PtrToArg<bool>::encode(p_use_mipmaps, &p_use_mipmaps_encoded);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, nullptr, &p_width_encoded, &p_height_encoded, &p_use_mipmaps_encoded, &p_format_encoded, &p_data));
}

void Image::set_data(int32_t p_width, int32_t p_height, bool p_use_mipmaps, Image::Format p_format, const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("set_data")._native_ptr(), 2740482212);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int8_t p_use_mipmaps_encoded;
	PtrToArg<bool>::encode(p_use_mipmaps, &p_use_mipmaps_encoded);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_width_encoded, &p_height_encoded, &p_use_mipmaps_encoded, &p_format_encoded, &p_data);
}

bool Image::is_empty() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("is_empty")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Error Image::load(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

Ref<Image> Image::load_from_file(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load_from_file")._native_ptr(), 736337515);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, nullptr, &p_path));
}

Error Image::save_png(const String &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("save_png")._native_ptr(), 2113323047);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

PackedByteArray Image::save_png_to_buffer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("save_png_to_buffer")._native_ptr(), 2362200018);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner);
}

Error Image::save_jpg(const String &p_path, float p_quality) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("save_jpg")._native_ptr(), 2800019068);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	double p_quality_encoded;
	PtrToArg<double>::encode(p_quality, &p_quality_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_quality_encoded);
}

PackedByteArray Image::save_jpg_to_buffer(float p_quality) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("save_jpg_to_buffer")._native_ptr(), 592235273);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	double p_quality_encoded;
	PtrToArg<double>::encode(p_quality, &p_quality_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, &p_quality_encoded);
}

Error Image::save_exr(const String &p_path, bool p_grayscale) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("save_exr")._native_ptr(), 3108122999);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_grayscale_encoded;
	PtrToArg<bool>::encode(p_grayscale, &p_grayscale_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_grayscale_encoded);
}

PackedByteArray Image::save_exr_to_buffer(bool p_grayscale) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("save_exr_to_buffer")._native_ptr(), 3178917920);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	int8_t p_grayscale_encoded;
	PtrToArg<bool>::encode(p_grayscale, &p_grayscale_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, &p_grayscale_encoded);
}

Error Image::save_dds(const String &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("save_dds")._native_ptr(), 2113323047);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

PackedByteArray Image::save_dds_to_buffer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("save_dds_to_buffer")._native_ptr(), 2362200018);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner);
}

Error Image::save_webp(const String &p_path, bool p_lossy, float p_quality) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("save_webp")._native_ptr(), 2781156876);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_lossy_encoded;
	PtrToArg<bool>::encode(p_lossy, &p_lossy_encoded);
	double p_quality_encoded;
	PtrToArg<double>::encode(p_quality, &p_quality_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_lossy_encoded, &p_quality_encoded);
}

PackedByteArray Image::save_webp_to_buffer(bool p_lossy, float p_quality) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("save_webp_to_buffer")._native_ptr(), 1214628238);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	int8_t p_lossy_encoded;
	PtrToArg<bool>::encode(p_lossy, &p_lossy_encoded);
	double p_quality_encoded;
	PtrToArg<double>::encode(p_quality, &p_quality_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, &p_lossy_encoded, &p_quality_encoded);
}

Image::AlphaMode Image::detect_alpha() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("detect_alpha")._native_ptr(), 2030116505);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Image::AlphaMode(0)));
	return (Image::AlphaMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Image::is_invisible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("is_invisible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Image::UsedChannels Image::detect_used_channels(Image::CompressSource p_source) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("detect_used_channels")._native_ptr(), 2703139984);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Image::UsedChannels(0)));
	int64_t p_source_encoded;
	PtrToArg<int64_t>::encode(p_source, &p_source_encoded);
	return (Image::UsedChannels)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_source_encoded);
}

Error Image::compress(Image::CompressMode p_mode, Image::CompressSource p_source, Image::ASTCFormat p_astc_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("compress")._native_ptr(), 2975424957);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	int64_t p_source_encoded;
	PtrToArg<int64_t>::encode(p_source, &p_source_encoded);
	int64_t p_astc_format_encoded;
	PtrToArg<int64_t>::encode(p_astc_format, &p_astc_format_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_mode_encoded, &p_source_encoded, &p_astc_format_encoded);
}

Error Image::compress_from_channels(Image::CompressMode p_mode, Image::UsedChannels p_channels, Image::ASTCFormat p_astc_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("compress_from_channels")._native_ptr(), 4212890953);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	int64_t p_channels_encoded;
	PtrToArg<int64_t>::encode(p_channels, &p_channels_encoded);
	int64_t p_astc_format_encoded;
	PtrToArg<int64_t>::encode(p_astc_format, &p_astc_format_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_mode_encoded, &p_channels_encoded, &p_astc_format_encoded);
}

Error Image::decompress() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("decompress")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Image::is_compressed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("is_compressed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Image::rotate_90(ClockDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("rotate_90")._native_ptr(), 1901204267);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_direction_encoded);
}

void Image::rotate_180() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("rotate_180")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Image::fix_alpha_edges() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("fix_alpha_edges")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Image::premultiply_alpha() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("premultiply_alpha")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Image::srgb_to_linear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("srgb_to_linear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Image::linear_to_srgb() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("linear_to_srgb")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Image::normal_map_to_xy() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("normal_map_to_xy")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Ref<Image> Image::rgbe_to_srgb() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("rgbe_to_srgb")._native_ptr(), 564927088);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner));
}

void Image::bump_map_to_normal_map(float p_bump_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("bump_map_to_normal_map")._native_ptr(), 3423495036);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bump_scale_encoded;
	PtrToArg<double>::encode(p_bump_scale, &p_bump_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bump_scale_encoded);
}

Dictionary Image::compute_image_metrics(const Ref<Image> &p_compared_image, bool p_use_luma) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("compute_image_metrics")._native_ptr(), 3080961247);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int8_t p_use_luma_encoded;
	PtrToArg<bool>::encode(p_use_luma, &p_use_luma_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, (p_compared_image != nullptr ? &p_compared_image->_owner : nullptr), &p_use_luma_encoded);
}

void Image::blit_rect(const Ref<Image> &p_src, const Rect2i &p_src_rect, const Vector2i &p_dst) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("blit_rect")._native_ptr(), 2903928755);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_src != nullptr ? &p_src->_owner : nullptr), &p_src_rect, &p_dst);
}

void Image::blit_rect_mask(const Ref<Image> &p_src, const Ref<Image> &p_mask, const Rect2i &p_src_rect, const Vector2i &p_dst) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("blit_rect_mask")._native_ptr(), 3383581145);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_src != nullptr ? &p_src->_owner : nullptr), (p_mask != nullptr ? &p_mask->_owner : nullptr), &p_src_rect, &p_dst);
}

void Image::blend_rect(const Ref<Image> &p_src, const Rect2i &p_src_rect, const Vector2i &p_dst) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("blend_rect")._native_ptr(), 2903928755);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_src != nullptr ? &p_src->_owner : nullptr), &p_src_rect, &p_dst);
}

void Image::blend_rect_mask(const Ref<Image> &p_src, const Ref<Image> &p_mask, const Rect2i &p_src_rect, const Vector2i &p_dst) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("blend_rect_mask")._native_ptr(), 3383581145);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_src != nullptr ? &p_src->_owner : nullptr), (p_mask != nullptr ? &p_mask->_owner : nullptr), &p_src_rect, &p_dst);
}

void Image::fill(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("fill")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

void Image::fill_rect(const Rect2i &p_rect, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("fill_rect")._native_ptr(), 514693913);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rect, &p_color);
}

Rect2i Image::get_used_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_used_rect")._native_ptr(), 410525958);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner);
}

Ref<Image> Image::get_region(const Rect2i &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_region")._native_ptr(), 2601441065);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_region));
}

void Image::copy_from(const Ref<Image> &p_src) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("copy_from")._native_ptr(), 532598488);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_src != nullptr ? &p_src->_owner : nullptr));
}

Color Image::get_pixelv(const Vector2i &p_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_pixelv")._native_ptr(), 1532707496);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_point);
}

Color Image::get_pixel(int32_t p_x, int32_t p_y) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("get_pixel")._native_ptr(), 2165839948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_x_encoded;
	PtrToArg<int64_t>::encode(p_x, &p_x_encoded);
	int64_t p_y_encoded;
	PtrToArg<int64_t>::encode(p_y, &p_y_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_x_encoded, &p_y_encoded);
}

void Image::set_pixelv(const Vector2i &p_point, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("set_pixelv")._native_ptr(), 287851464);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_point, &p_color);
}

void Image::set_pixel(int32_t p_x, int32_t p_y, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("set_pixel")._native_ptr(), 3733378741);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_x_encoded;
	PtrToArg<int64_t>::encode(p_x, &p_x_encoded);
	int64_t p_y_encoded;
	PtrToArg<int64_t>::encode(p_y, &p_y_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_x_encoded, &p_y_encoded, &p_color);
}

void Image::adjust_bcs(float p_brightness, float p_contrast, float p_saturation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("adjust_bcs")._native_ptr(), 2385087082);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_brightness_encoded;
	PtrToArg<double>::encode(p_brightness, &p_brightness_encoded);
	double p_contrast_encoded;
	PtrToArg<double>::encode(p_contrast, &p_contrast_encoded);
	double p_saturation_encoded;
	PtrToArg<double>::encode(p_saturation, &p_saturation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_brightness_encoded, &p_contrast_encoded, &p_saturation_encoded);
}

Error Image::load_png_from_buffer(const PackedByteArray &p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load_png_from_buffer")._native_ptr(), 680677267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer);
}

Error Image::load_jpg_from_buffer(const PackedByteArray &p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load_jpg_from_buffer")._native_ptr(), 680677267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer);
}

Error Image::load_webp_from_buffer(const PackedByteArray &p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load_webp_from_buffer")._native_ptr(), 680677267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer);
}

Error Image::load_tga_from_buffer(const PackedByteArray &p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load_tga_from_buffer")._native_ptr(), 680677267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer);
}

Error Image::load_bmp_from_buffer(const PackedByteArray &p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load_bmp_from_buffer")._native_ptr(), 680677267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer);
}

Error Image::load_ktx_from_buffer(const PackedByteArray &p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load_ktx_from_buffer")._native_ptr(), 680677267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer);
}

Error Image::load_dds_from_buffer(const PackedByteArray &p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load_dds_from_buffer")._native_ptr(), 680677267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer);
}

Error Image::load_exr_from_buffer(const PackedByteArray &p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load_exr_from_buffer")._native_ptr(), 680677267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer);
}

Error Image::load_svg_from_buffer(const PackedByteArray &p_buffer, float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load_svg_from_buffer")._native_ptr(), 311853421);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer, &p_scale_encoded);
}

Error Image::load_svg_from_string(const String &p_svg_str, float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Image::get_class_static()._native_ptr(), StringName("load_svg_from_string")._native_ptr(), 3254053600);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_svg_str, &p_scale_encoded);
}

} // namespace godot
