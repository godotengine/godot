/**************************************************************************/
/*  system_font.cpp                                                       */
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

#include <godot_cpp/classes/system_font.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void SystemFont::set_antialiasing(TextServer::FontAntialiasing p_antialiasing) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_antialiasing")._native_ptr(), 1669900);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_antialiasing_encoded;
	PtrToArg<int64_t>::encode(p_antialiasing, &p_antialiasing_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_antialiasing_encoded);
}

TextServer::FontAntialiasing SystemFont::get_antialiasing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("get_antialiasing")._native_ptr(), 4262718649);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::FontAntialiasing(0)));
	return (TextServer::FontAntialiasing)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SystemFont::set_disable_embedded_bitmaps(bool p_disable_embedded_bitmaps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_disable_embedded_bitmaps")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_embedded_bitmaps_encoded;
	PtrToArg<bool>::encode(p_disable_embedded_bitmaps, &p_disable_embedded_bitmaps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_disable_embedded_bitmaps_encoded);
}

bool SystemFont::get_disable_embedded_bitmaps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("get_disable_embedded_bitmaps")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SystemFont::set_generate_mipmaps(bool p_generate_mipmaps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_generate_mipmaps")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_generate_mipmaps_encoded;
	PtrToArg<bool>::encode(p_generate_mipmaps, &p_generate_mipmaps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_generate_mipmaps_encoded);
}

bool SystemFont::get_generate_mipmaps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("get_generate_mipmaps")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SystemFont::set_allow_system_fallback(bool p_allow_system_fallback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_allow_system_fallback")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_system_fallback_encoded;
	PtrToArg<bool>::encode(p_allow_system_fallback, &p_allow_system_fallback_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allow_system_fallback_encoded);
}

bool SystemFont::is_allow_system_fallback() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("is_allow_system_fallback")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SystemFont::set_force_autohinter(bool p_force_autohinter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_force_autohinter")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_force_autohinter_encoded;
	PtrToArg<bool>::encode(p_force_autohinter, &p_force_autohinter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force_autohinter_encoded);
}

bool SystemFont::is_force_autohinter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("is_force_autohinter")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SystemFont::set_modulate_color_glyphs(bool p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_modulate_color_glyphs")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_modulate_encoded;
	PtrToArg<bool>::encode(p_modulate, &p_modulate_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_modulate_encoded);
}

bool SystemFont::is_modulate_color_glyphs() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("is_modulate_color_glyphs")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SystemFont::set_hinting(TextServer::Hinting p_hinting) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_hinting")._native_ptr(), 1827459492);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_hinting_encoded;
	PtrToArg<int64_t>::encode(p_hinting, &p_hinting_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hinting_encoded);
}

TextServer::Hinting SystemFont::get_hinting() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("get_hinting")._native_ptr(), 3683214614);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::Hinting(0)));
	return (TextServer::Hinting)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SystemFont::set_subpixel_positioning(TextServer::SubpixelPositioning p_subpixel_positioning) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_subpixel_positioning")._native_ptr(), 4225742182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_subpixel_positioning_encoded;
	PtrToArg<int64_t>::encode(p_subpixel_positioning, &p_subpixel_positioning_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_subpixel_positioning_encoded);
}

TextServer::SubpixelPositioning SystemFont::get_subpixel_positioning() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("get_subpixel_positioning")._native_ptr(), 1069238588);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::SubpixelPositioning(0)));
	return (TextServer::SubpixelPositioning)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SystemFont::set_keep_rounding_remainders(bool p_keep_rounding_remainders) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_keep_rounding_remainders")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_rounding_remainders_encoded;
	PtrToArg<bool>::encode(p_keep_rounding_remainders, &p_keep_rounding_remainders_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_keep_rounding_remainders_encoded);
}

bool SystemFont::get_keep_rounding_remainders() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("get_keep_rounding_remainders")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SystemFont::set_multichannel_signed_distance_field(bool p_msdf) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_multichannel_signed_distance_field")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_msdf_encoded;
	PtrToArg<bool>::encode(p_msdf, &p_msdf_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msdf_encoded);
}

bool SystemFont::is_multichannel_signed_distance_field() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("is_multichannel_signed_distance_field")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SystemFont::set_msdf_pixel_range(int32_t p_msdf_pixel_range) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_msdf_pixel_range")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msdf_pixel_range_encoded;
	PtrToArg<int64_t>::encode(p_msdf_pixel_range, &p_msdf_pixel_range_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msdf_pixel_range_encoded);
}

int32_t SystemFont::get_msdf_pixel_range() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("get_msdf_pixel_range")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SystemFont::set_msdf_size(int32_t p_msdf_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_msdf_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msdf_size_encoded;
	PtrToArg<int64_t>::encode(p_msdf_size, &p_msdf_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msdf_size_encoded);
}

int32_t SystemFont::get_msdf_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("get_msdf_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SystemFont::set_oversampling(float p_oversampling) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_oversampling")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_oversampling_encoded);
}

float SystemFont::get_oversampling() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("get_oversampling")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

PackedStringArray SystemFont::get_font_names() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("get_font_names")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void SystemFont::set_font_names(const PackedStringArray &p_names) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_font_names")._native_ptr(), 4015028928);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_names);
}

bool SystemFont::get_font_italic() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("get_font_italic")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SystemFont::set_font_italic(bool p_italic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_font_italic")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_italic_encoded;
	PtrToArg<bool>::encode(p_italic, &p_italic_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_italic_encoded);
}

void SystemFont::set_font_weight(int32_t p_weight) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_font_weight")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_weight_encoded;
	PtrToArg<int64_t>::encode(p_weight, &p_weight_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_weight_encoded);
}

void SystemFont::set_font_stretch(int32_t p_stretch) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SystemFont::get_class_static()._native_ptr(), StringName("set_font_stretch")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stretch_encoded;
	PtrToArg<int64_t>::encode(p_stretch, &p_stretch_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stretch_encoded);
}

} // namespace godot
