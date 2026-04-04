/**************************************************************************/
/*  font_file.cpp                                                         */
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

#include <godot_cpp/classes/font_file.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

Error FontFile::load_bitmap_font(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("load_bitmap_font")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

Error FontFile::load_dynamic_font(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("load_dynamic_font")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

void FontFile::set_data(const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_data")._native_ptr(), 2971499966);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_data);
}

PackedByteArray FontFile::get_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_data")._native_ptr(), 2362200018);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner);
}

void FontFile::set_font_name(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_font_name")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void FontFile::set_font_style_name(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_font_style_name")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void FontFile::set_font_style(BitField<TextServer::FontStyle> p_style) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_font_style")._native_ptr(), 918070724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_style);
}

void FontFile::set_font_weight(int32_t p_weight) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_font_weight")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_weight_encoded;
	PtrToArg<int64_t>::encode(p_weight, &p_weight_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_weight_encoded);
}

void FontFile::set_font_stretch(int32_t p_stretch) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_font_stretch")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stretch_encoded;
	PtrToArg<int64_t>::encode(p_stretch, &p_stretch_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stretch_encoded);
}

void FontFile::set_antialiasing(TextServer::FontAntialiasing p_antialiasing) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_antialiasing")._native_ptr(), 1669900);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_antialiasing_encoded;
	PtrToArg<int64_t>::encode(p_antialiasing, &p_antialiasing_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_antialiasing_encoded);
}

TextServer::FontAntialiasing FontFile::get_antialiasing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_antialiasing")._native_ptr(), 4262718649);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::FontAntialiasing(0)));
	return (TextServer::FontAntialiasing)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FontFile::set_disable_embedded_bitmaps(bool p_disable_embedded_bitmaps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_disable_embedded_bitmaps")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_embedded_bitmaps_encoded;
	PtrToArg<bool>::encode(p_disable_embedded_bitmaps, &p_disable_embedded_bitmaps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_disable_embedded_bitmaps_encoded);
}

bool FontFile::get_disable_embedded_bitmaps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_disable_embedded_bitmaps")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void FontFile::set_generate_mipmaps(bool p_generate_mipmaps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_generate_mipmaps")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_generate_mipmaps_encoded;
	PtrToArg<bool>::encode(p_generate_mipmaps, &p_generate_mipmaps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_generate_mipmaps_encoded);
}

bool FontFile::get_generate_mipmaps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_generate_mipmaps")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void FontFile::set_multichannel_signed_distance_field(bool p_msdf) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_multichannel_signed_distance_field")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_msdf_encoded;
	PtrToArg<bool>::encode(p_msdf, &p_msdf_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msdf_encoded);
}

bool FontFile::is_multichannel_signed_distance_field() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("is_multichannel_signed_distance_field")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void FontFile::set_msdf_pixel_range(int32_t p_msdf_pixel_range) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_msdf_pixel_range")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msdf_pixel_range_encoded;
	PtrToArg<int64_t>::encode(p_msdf_pixel_range, &p_msdf_pixel_range_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msdf_pixel_range_encoded);
}

int32_t FontFile::get_msdf_pixel_range() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_msdf_pixel_range")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FontFile::set_msdf_size(int32_t p_msdf_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_msdf_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msdf_size_encoded;
	PtrToArg<int64_t>::encode(p_msdf_size, &p_msdf_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msdf_size_encoded);
}

int32_t FontFile::get_msdf_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_msdf_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FontFile::set_fixed_size(int32_t p_fixed_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_fixed_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_fixed_size_encoded;
	PtrToArg<int64_t>::encode(p_fixed_size, &p_fixed_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fixed_size_encoded);
}

int32_t FontFile::get_fixed_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_fixed_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FontFile::set_fixed_size_scale_mode(TextServer::FixedSizeScaleMode p_fixed_size_scale_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_fixed_size_scale_mode")._native_ptr(), 1660989956);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_fixed_size_scale_mode_encoded;
	PtrToArg<int64_t>::encode(p_fixed_size_scale_mode, &p_fixed_size_scale_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fixed_size_scale_mode_encoded);
}

TextServer::FixedSizeScaleMode FontFile::get_fixed_size_scale_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_fixed_size_scale_mode")._native_ptr(), 753873478);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::FixedSizeScaleMode(0)));
	return (TextServer::FixedSizeScaleMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FontFile::set_allow_system_fallback(bool p_allow_system_fallback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_allow_system_fallback")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_system_fallback_encoded;
	PtrToArg<bool>::encode(p_allow_system_fallback, &p_allow_system_fallback_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allow_system_fallback_encoded);
}

bool FontFile::is_allow_system_fallback() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("is_allow_system_fallback")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void FontFile::set_force_autohinter(bool p_force_autohinter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_force_autohinter")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_force_autohinter_encoded;
	PtrToArg<bool>::encode(p_force_autohinter, &p_force_autohinter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force_autohinter_encoded);
}

bool FontFile::is_force_autohinter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("is_force_autohinter")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void FontFile::set_modulate_color_glyphs(bool p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_modulate_color_glyphs")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_modulate_encoded;
	PtrToArg<bool>::encode(p_modulate, &p_modulate_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_modulate_encoded);
}

bool FontFile::is_modulate_color_glyphs() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("is_modulate_color_glyphs")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void FontFile::set_hinting(TextServer::Hinting p_hinting) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_hinting")._native_ptr(), 1827459492);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_hinting_encoded;
	PtrToArg<int64_t>::encode(p_hinting, &p_hinting_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hinting_encoded);
}

TextServer::Hinting FontFile::get_hinting() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_hinting")._native_ptr(), 3683214614);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::Hinting(0)));
	return (TextServer::Hinting)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FontFile::set_subpixel_positioning(TextServer::SubpixelPositioning p_subpixel_positioning) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_subpixel_positioning")._native_ptr(), 4225742182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_subpixel_positioning_encoded;
	PtrToArg<int64_t>::encode(p_subpixel_positioning, &p_subpixel_positioning_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_subpixel_positioning_encoded);
}

TextServer::SubpixelPositioning FontFile::get_subpixel_positioning() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_subpixel_positioning")._native_ptr(), 1069238588);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::SubpixelPositioning(0)));
	return (TextServer::SubpixelPositioning)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FontFile::set_keep_rounding_remainders(bool p_keep_rounding_remainders) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_keep_rounding_remainders")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_rounding_remainders_encoded;
	PtrToArg<bool>::encode(p_keep_rounding_remainders, &p_keep_rounding_remainders_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_keep_rounding_remainders_encoded);
}

bool FontFile::get_keep_rounding_remainders() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_keep_rounding_remainders")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void FontFile::set_oversampling(float p_oversampling) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_oversampling")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_oversampling_encoded);
}

float FontFile::get_oversampling() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_oversampling")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

int32_t FontFile::get_cache_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_cache_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FontFile::clear_cache() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("clear_cache")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void FontFile::remove_cache(int32_t p_cache_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("remove_cache")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded);
}

TypedArray<Vector2i> FontFile::get_size_cache_list(int32_t p_cache_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_size_cache_list")._native_ptr(), 663333327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner, &p_cache_index_encoded);
}

void FontFile::clear_size_cache(int32_t p_cache_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("clear_size_cache")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded);
}

void FontFile::remove_size_cache(int32_t p_cache_index, const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("remove_size_cache")._native_ptr(), 2311374912);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size);
}

void FontFile::set_variation_coordinates(int32_t p_cache_index, const Dictionary &p_variation_coordinates) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_variation_coordinates")._native_ptr(), 64545446);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_variation_coordinates);
}

Dictionary FontFile::get_variation_coordinates(int32_t p_cache_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_variation_coordinates")._native_ptr(), 3485342025);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_cache_index_encoded);
}

void FontFile::set_embolden(int32_t p_cache_index, float p_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_embolden")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_strength_encoded);
}

float FontFile::get_embolden(int32_t p_cache_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_embolden")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_cache_index_encoded);
}

void FontFile::set_transform(int32_t p_cache_index, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_transform")._native_ptr(), 30160968);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_transform);
}

Transform2D FontFile::get_transform(int32_t p_cache_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_transform")._native_ptr(), 3836996910);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner, &p_cache_index_encoded);
}

void FontFile::set_extra_spacing(int32_t p_cache_index, TextServer::SpacingType p_spacing, int64_t p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_extra_spacing")._native_ptr(), 62942285);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_spacing_encoded;
	PtrToArg<int64_t>::encode(p_spacing, &p_spacing_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_spacing_encoded, &p_value_encoded);
}

int64_t FontFile::get_extra_spacing(int32_t p_cache_index, TextServer::SpacingType p_spacing) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_extra_spacing")._native_ptr(), 1924257185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_spacing_encoded;
	PtrToArg<int64_t>::encode(p_spacing, &p_spacing_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_spacing_encoded);
}

void FontFile::set_extra_baseline_offset(int32_t p_cache_index, float p_baseline_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_extra_baseline_offset")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	double p_baseline_offset_encoded;
	PtrToArg<double>::encode(p_baseline_offset, &p_baseline_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_baseline_offset_encoded);
}

float FontFile::get_extra_baseline_offset(int32_t p_cache_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_extra_baseline_offset")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_cache_index_encoded);
}

void FontFile::set_face_index(int32_t p_cache_index, int64_t p_face_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_face_index")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_face_index_encoded;
	PtrToArg<int64_t>::encode(p_face_index, &p_face_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_face_index_encoded);
}

int64_t FontFile::get_face_index(int32_t p_cache_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_face_index")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_cache_index_encoded);
}

void FontFile::set_cache_ascent(int32_t p_cache_index, int32_t p_size, float p_ascent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_cache_ascent")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_ascent_encoded;
	PtrToArg<double>::encode(p_ascent, &p_ascent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded, &p_ascent_encoded);
}

float FontFile::get_cache_ascent(int32_t p_cache_index, int32_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_cache_ascent")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded);
}

void FontFile::set_cache_descent(int32_t p_cache_index, int32_t p_size, float p_descent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_cache_descent")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_descent_encoded;
	PtrToArg<double>::encode(p_descent, &p_descent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded, &p_descent_encoded);
}

float FontFile::get_cache_descent(int32_t p_cache_index, int32_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_cache_descent")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded);
}

void FontFile::set_cache_underline_position(int32_t p_cache_index, int32_t p_size, float p_underline_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_cache_underline_position")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_underline_position_encoded;
	PtrToArg<double>::encode(p_underline_position, &p_underline_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded, &p_underline_position_encoded);
}

float FontFile::get_cache_underline_position(int32_t p_cache_index, int32_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_cache_underline_position")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded);
}

void FontFile::set_cache_underline_thickness(int32_t p_cache_index, int32_t p_size, float p_underline_thickness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_cache_underline_thickness")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_underline_thickness_encoded;
	PtrToArg<double>::encode(p_underline_thickness, &p_underline_thickness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded, &p_underline_thickness_encoded);
}

float FontFile::get_cache_underline_thickness(int32_t p_cache_index, int32_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_cache_underline_thickness")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded);
}

void FontFile::set_cache_scale(int32_t p_cache_index, int32_t p_size, float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_cache_scale")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded, &p_scale_encoded);
}

float FontFile::get_cache_scale(int32_t p_cache_index, int32_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_cache_scale")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded);
}

int32_t FontFile::get_texture_count(int32_t p_cache_index, const Vector2i &p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_texture_count")._native_ptr(), 1987661582);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size);
}

void FontFile::clear_textures(int32_t p_cache_index, const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("clear_textures")._native_ptr(), 2311374912);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size);
}

void FontFile::remove_texture(int32_t p_cache_index, const Vector2i &p_size, int32_t p_texture_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("remove_texture")._native_ptr(), 2328951467);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_texture_index_encoded;
	PtrToArg<int64_t>::encode(p_texture_index, &p_texture_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_texture_index_encoded);
}

void FontFile::set_texture_image(int32_t p_cache_index, const Vector2i &p_size, int32_t p_texture_index, const Ref<Image> &p_image) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_texture_image")._native_ptr(), 4157974066);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_texture_index_encoded;
	PtrToArg<int64_t>::encode(p_texture_index, &p_texture_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_texture_index_encoded, (p_image != nullptr ? &p_image->_owner : nullptr));
}

Ref<Image> FontFile::get_texture_image(int32_t p_cache_index, const Vector2i &p_size, int32_t p_texture_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_texture_image")._native_ptr(), 3878418953);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_texture_index_encoded;
	PtrToArg<int64_t>::encode(p_texture_index, &p_texture_index_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_texture_index_encoded));
}

void FontFile::set_texture_offsets(int32_t p_cache_index, const Vector2i &p_size, int32_t p_texture_index, const PackedInt32Array &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_texture_offsets")._native_ptr(), 2849993437);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_texture_index_encoded;
	PtrToArg<int64_t>::encode(p_texture_index, &p_texture_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_texture_index_encoded, &p_offset);
}

PackedInt32Array FontFile::get_texture_offsets(int32_t p_cache_index, const Vector2i &p_size, int32_t p_texture_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_texture_offsets")._native_ptr(), 3703444828);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_texture_index_encoded;
	PtrToArg<int64_t>::encode(p_texture_index, &p_texture_index_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_texture_index_encoded);
}

PackedInt32Array FontFile::get_glyph_list(int32_t p_cache_index, const Vector2i &p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_glyph_list")._native_ptr(), 681709689);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size);
}

void FontFile::clear_glyphs(int32_t p_cache_index, const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("clear_glyphs")._native_ptr(), 2311374912);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size);
}

void FontFile::remove_glyph(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("remove_glyph")._native_ptr(), 2328951467);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_glyph_encoded);
}

void FontFile::set_glyph_advance(int32_t p_cache_index, int32_t p_size, int32_t p_glyph, const Vector2 &p_advance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_glyph_advance")._native_ptr(), 947991729);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded, &p_glyph_encoded, &p_advance);
}

Vector2 FontFile::get_glyph_advance(int32_t p_cache_index, int32_t p_size, int32_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_glyph_advance")._native_ptr(), 1601573536);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded, &p_glyph_encoded);
}

void FontFile::set_glyph_offset(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_glyph_offset")._native_ptr(), 921719850);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_glyph_encoded, &p_offset);
}

Vector2 FontFile::get_glyph_offset(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_glyph_offset")._native_ptr(), 3205412300);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_glyph_encoded);
}

void FontFile::set_glyph_size(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_gl_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_glyph_size")._native_ptr(), 921719850);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_glyph_encoded, &p_gl_size);
}

Vector2 FontFile::get_glyph_size(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_glyph_size")._native_ptr(), 3205412300);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_glyph_encoded);
}

void FontFile::set_glyph_uv_rect(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Rect2 &p_uv_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_glyph_uv_rect")._native_ptr(), 3821620992);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_glyph_encoded, &p_uv_rect);
}

Rect2 FontFile::get_glyph_uv_rect(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_glyph_uv_rect")._native_ptr(), 3927917900);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_glyph_encoded);
}

void FontFile::set_glyph_texture_idx(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph, int32_t p_texture_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_glyph_texture_idx")._native_ptr(), 355564111);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	int64_t p_texture_idx_encoded;
	PtrToArg<int64_t>::encode(p_texture_idx, &p_texture_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_glyph_encoded, &p_texture_idx_encoded);
}

int32_t FontFile::get_glyph_texture_idx(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_glyph_texture_idx")._native_ptr(), 1629411054);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_glyph_encoded);
}

TypedArray<Vector2i> FontFile::get_kerning_list(int32_t p_cache_index, int32_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_kerning_list")._native_ptr(), 2345056839);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded);
}

void FontFile::clear_kerning_map(int32_t p_cache_index, int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("clear_kerning_map")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded);
}

void FontFile::remove_kerning(int32_t p_cache_index, int32_t p_size, const Vector2i &p_glyph_pair) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("remove_kerning")._native_ptr(), 3930204747);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded, &p_glyph_pair);
}

void FontFile::set_kerning(int32_t p_cache_index, int32_t p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_kerning")._native_ptr(), 3182200918);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded, &p_glyph_pair, &p_kerning);
}

Vector2 FontFile::get_kerning(int32_t p_cache_index, int32_t p_size, const Vector2i &p_glyph_pair) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_kerning")._native_ptr(), 1611912865);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size_encoded, &p_glyph_pair);
}

void FontFile::render_range(int32_t p_cache_index, const Vector2i &p_size, char32_t p_start, char32_t p_end) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("render_range")._native_ptr(), 355564111);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_start_encoded;
	PtrToArg<int64_t>::encode(p_start, &p_start_encoded);
	int64_t p_end_encoded;
	PtrToArg<int64_t>::encode(p_end, &p_end_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_start_encoded, &p_end_encoded);
}

void FontFile::render_glyph(int32_t p_cache_index, const Vector2i &p_size, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("render_glyph")._native_ptr(), 2328951467);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cache_index_encoded;
	PtrToArg<int64_t>::encode(p_cache_index, &p_cache_index_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cache_index_encoded, &p_size, &p_index_encoded);
}

void FontFile::set_language_support_override(const String &p_language, bool p_supported) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_language_support_override")._native_ptr(), 2678287736);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_supported_encoded;
	PtrToArg<bool>::encode(p_supported, &p_supported_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_language, &p_supported_encoded);
}

bool FontFile::get_language_support_override(const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_language_support_override")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_language);
}

void FontFile::remove_language_support_override(const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("remove_language_support_override")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_language);
}

PackedStringArray FontFile::get_language_support_overrides() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_language_support_overrides")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void FontFile::set_script_support_override(const String &p_script, bool p_supported) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_script_support_override")._native_ptr(), 2678287736);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_supported_encoded;
	PtrToArg<bool>::encode(p_supported, &p_supported_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_script, &p_supported_encoded);
}

bool FontFile::get_script_support_override(const String &p_script) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_script_support_override")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_script);
}

void FontFile::remove_script_support_override(const String &p_script) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("remove_script_support_override")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_script);
}

PackedStringArray FontFile::get_script_support_overrides() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_script_support_overrides")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void FontFile::set_opentype_feature_overrides(const Dictionary &p_overrides) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("set_opentype_feature_overrides")._native_ptr(), 4155329257);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_overrides);
}

Dictionary FontFile::get_opentype_feature_overrides() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_opentype_feature_overrides")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

int32_t FontFile::get_glyph_index(int32_t p_size, char32_t p_char, char32_t p_variation_selector) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_glyph_index")._native_ptr(), 864943070);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_char_encoded;
	PtrToArg<int64_t>::encode(p_char, &p_char_encoded);
	int64_t p_variation_selector_encoded;
	PtrToArg<int64_t>::encode(p_variation_selector, &p_variation_selector_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_size_encoded, &p_char_encoded, &p_variation_selector_encoded);
}

char32_t FontFile::get_char_from_glyph_index(int32_t p_size, int32_t p_glyph_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FontFile::get_class_static()._native_ptr(), StringName("get_char_from_glyph_index")._native_ptr(), 3175239445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_glyph_index_encoded;
	PtrToArg<int64_t>::encode(p_glyph_index, &p_glyph_index_encoded);
	return ::godot::internal::_call_native_mb_ret<char32_t>(_gde_method_bind, _owner, &p_size_encoded, &p_glyph_index_encoded);
}

} // namespace godot
