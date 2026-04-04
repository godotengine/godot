/**************************************************************************/
/*  text_server.cpp                                                       */
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

#include <godot_cpp/classes/text_server.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

namespace godot {

bool TextServer::has_feature(TextServer::Feature p_feature) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("has_feature")._native_ptr(), 3967367083);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_feature_encoded;
	PtrToArg<int64_t>::encode(p_feature, &p_feature_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_feature_encoded);
}

String TextServer::get_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("get_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

int64_t TextServer::get_features() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("get_features")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool TextServer::load_support_data(const String &p_filename) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("load_support_data")._native_ptr(), 2323990056);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_filename);
}

String TextServer::get_support_data_filename() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("get_support_data_filename")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String TextServer::get_support_data_info() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("get_support_data_info")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool TextServer::save_support_data(const String &p_filename) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("save_support_data")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_filename);
}

PackedByteArray TextServer::get_support_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("get_support_data")._native_ptr(), 2362200018);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner);
}

bool TextServer::is_locale_using_support_data(const String &p_locale) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("is_locale_using_support_data")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_locale);
}

bool TextServer::is_locale_right_to_left(const String &p_locale) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("is_locale_right_to_left")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_locale);
}

int64_t TextServer::name_to_tag(const String &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("name_to_tag")._native_ptr(), 1321353865);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_name);
}

String TextServer::tag_to_name(int64_t p_tag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("tag_to_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_tag_encoded;
	PtrToArg<int64_t>::encode(p_tag, &p_tag_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_tag_encoded);
}

bool TextServer::has(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("has")._native_ptr(), 3521089500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_rid);
}

void TextServer::free_rid(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("free_rid")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid);
}

RID TextServer::create_font() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("create_font")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID TextServer::create_font_linked_variation(const RID &p_font_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("create_font_linked_variation")._native_ptr(), 41030802);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_data(const RID &p_font_rid, const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_data")._native_ptr(), 1355495400);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_data);
}

void TextServer::font_set_face_index(const RID &p_font_rid, int64_t p_face_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_face_index")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_face_index_encoded;
	PtrToArg<int64_t>::encode(p_face_index, &p_face_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_face_index_encoded);
}

int64_t TextServer::font_get_face_index(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_face_index")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

int64_t TextServer::font_get_face_count(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_face_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_style(const RID &p_font_rid, BitField<TextServer::FontStyle> p_style) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_style")._native_ptr(), 898466325);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_style);
}

BitField<TextServer::FontStyle> TextServer::font_get_style(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_style")._native_ptr(), 3082502592);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<TextServer::FontStyle>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_name(const RID &p_font_rid, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_name")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_name);
}

String TextServer::font_get_name(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_name")._native_ptr(), 642473191);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_font_rid);
}

Dictionary TextServer::font_get_ot_name_strings(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_ot_name_strings")._native_ptr(), 1882737106);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_style_name(const RID &p_font_rid, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_style_name")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_name);
}

String TextServer::font_get_style_name(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_style_name")._native_ptr(), 642473191);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_weight(const RID &p_font_rid, int64_t p_weight) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_weight")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_weight_encoded;
	PtrToArg<int64_t>::encode(p_weight, &p_weight_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_weight_encoded);
}

int64_t TextServer::font_get_weight(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_weight")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_stretch(const RID &p_font_rid, int64_t p_weight) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_stretch")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_weight_encoded;
	PtrToArg<int64_t>::encode(p_weight, &p_weight_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_weight_encoded);
}

int64_t TextServer::font_get_stretch(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_stretch")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_antialiasing(const RID &p_font_rid, TextServer::FontAntialiasing p_antialiasing) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_antialiasing")._native_ptr(), 958337235);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_antialiasing_encoded;
	PtrToArg<int64_t>::encode(p_antialiasing, &p_antialiasing_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_antialiasing_encoded);
}

TextServer::FontAntialiasing TextServer::font_get_antialiasing(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_antialiasing")._native_ptr(), 3389420495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::FontAntialiasing(0)));
	return (TextServer::FontAntialiasing)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_disable_embedded_bitmaps(const RID &p_font_rid, bool p_disable_embedded_bitmaps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_disable_embedded_bitmaps")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_embedded_bitmaps_encoded;
	PtrToArg<bool>::encode(p_disable_embedded_bitmaps, &p_disable_embedded_bitmaps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_disable_embedded_bitmaps_encoded);
}

bool TextServer::font_get_disable_embedded_bitmaps(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_disable_embedded_bitmaps")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_generate_mipmaps(const RID &p_font_rid, bool p_generate_mipmaps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_generate_mipmaps")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_generate_mipmaps_encoded;
	PtrToArg<bool>::encode(p_generate_mipmaps, &p_generate_mipmaps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_generate_mipmaps_encoded);
}

bool TextServer::font_get_generate_mipmaps(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_generate_mipmaps")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_multichannel_signed_distance_field(const RID &p_font_rid, bool p_msdf) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_multichannel_signed_distance_field")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_msdf_encoded;
	PtrToArg<bool>::encode(p_msdf, &p_msdf_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_msdf_encoded);
}

bool TextServer::font_is_multichannel_signed_distance_field(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_is_multichannel_signed_distance_field")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_msdf_pixel_range(const RID &p_font_rid, int64_t p_msdf_pixel_range) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_msdf_pixel_range")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msdf_pixel_range_encoded;
	PtrToArg<int64_t>::encode(p_msdf_pixel_range, &p_msdf_pixel_range_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_msdf_pixel_range_encoded);
}

int64_t TextServer::font_get_msdf_pixel_range(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_msdf_pixel_range")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_msdf_size(const RID &p_font_rid, int64_t p_msdf_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_msdf_size")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msdf_size_encoded;
	PtrToArg<int64_t>::encode(p_msdf_size, &p_msdf_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_msdf_size_encoded);
}

int64_t TextServer::font_get_msdf_size(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_msdf_size")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_fixed_size(const RID &p_font_rid, int64_t p_fixed_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_fixed_size")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_fixed_size_encoded;
	PtrToArg<int64_t>::encode(p_fixed_size, &p_fixed_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_fixed_size_encoded);
}

int64_t TextServer::font_get_fixed_size(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_fixed_size")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_fixed_size_scale_mode(const RID &p_font_rid, TextServer::FixedSizeScaleMode p_fixed_size_scale_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_fixed_size_scale_mode")._native_ptr(), 1029390307);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_fixed_size_scale_mode_encoded;
	PtrToArg<int64_t>::encode(p_fixed_size_scale_mode, &p_fixed_size_scale_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_fixed_size_scale_mode_encoded);
}

TextServer::FixedSizeScaleMode TextServer::font_get_fixed_size_scale_mode(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_fixed_size_scale_mode")._native_ptr(), 4113120379);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::FixedSizeScaleMode(0)));
	return (TextServer::FixedSizeScaleMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_allow_system_fallback(const RID &p_font_rid, bool p_allow_system_fallback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_allow_system_fallback")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_system_fallback_encoded;
	PtrToArg<bool>::encode(p_allow_system_fallback, &p_allow_system_fallback_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_allow_system_fallback_encoded);
}

bool TextServer::font_is_allow_system_fallback(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_is_allow_system_fallback")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_clear_system_fallback_cache() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_clear_system_fallback_cache")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextServer::font_set_force_autohinter(const RID &p_font_rid, bool p_force_autohinter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_force_autohinter")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_force_autohinter_encoded;
	PtrToArg<bool>::encode(p_force_autohinter, &p_force_autohinter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_force_autohinter_encoded);
}

bool TextServer::font_is_force_autohinter(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_is_force_autohinter")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_modulate_color_glyphs(const RID &p_font_rid, bool p_force_autohinter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_modulate_color_glyphs")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_force_autohinter_encoded;
	PtrToArg<bool>::encode(p_force_autohinter, &p_force_autohinter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_force_autohinter_encoded);
}

bool TextServer::font_is_modulate_color_glyphs(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_is_modulate_color_glyphs")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_hinting(const RID &p_font_rid, TextServer::Hinting p_hinting) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_hinting")._native_ptr(), 1520010864);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_hinting_encoded;
	PtrToArg<int64_t>::encode(p_hinting, &p_hinting_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_hinting_encoded);
}

TextServer::Hinting TextServer::font_get_hinting(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_hinting")._native_ptr(), 3971592737);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::Hinting(0)));
	return (TextServer::Hinting)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_subpixel_positioning(const RID &p_font_rid, TextServer::SubpixelPositioning p_subpixel_positioning) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_subpixel_positioning")._native_ptr(), 3830459669);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_subpixel_positioning_encoded;
	PtrToArg<int64_t>::encode(p_subpixel_positioning, &p_subpixel_positioning_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_subpixel_positioning_encoded);
}

TextServer::SubpixelPositioning TextServer::font_get_subpixel_positioning(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_subpixel_positioning")._native_ptr(), 2752233671);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::SubpixelPositioning(0)));
	return (TextServer::SubpixelPositioning)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_keep_rounding_remainders(const RID &p_font_rid, bool p_keep_rounding_remainders) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_keep_rounding_remainders")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_rounding_remainders_encoded;
	PtrToArg<bool>::encode(p_keep_rounding_remainders, &p_keep_rounding_remainders_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_keep_rounding_remainders_encoded);
}

bool TextServer::font_get_keep_rounding_remainders(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_keep_rounding_remainders")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_embolden(const RID &p_font_rid, double p_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_embolden")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_strength_encoded);
}

double TextServer::font_get_embolden(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_embolden")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_spacing(const RID &p_font_rid, TextServer::SpacingType p_spacing, int64_t p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_spacing")._native_ptr(), 1307259930);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_spacing_encoded;
	PtrToArg<int64_t>::encode(p_spacing, &p_spacing_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_spacing_encoded, &p_value_encoded);
}

int64_t TextServer::font_get_spacing(const RID &p_font_rid, TextServer::SpacingType p_spacing) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_spacing")._native_ptr(), 1213653558);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_spacing_encoded;
	PtrToArg<int64_t>::encode(p_spacing, &p_spacing_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid, &p_spacing_encoded);
}

void TextServer::font_set_baseline_offset(const RID &p_font_rid, double p_baseline_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_baseline_offset")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_baseline_offset_encoded;
	PtrToArg<double>::encode(p_baseline_offset, &p_baseline_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_baseline_offset_encoded);
}

double TextServer::font_get_baseline_offset(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_baseline_offset")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_transform(const RID &p_font_rid, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_transform")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_transform);
}

Transform2D TextServer::font_get_transform(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_transform")._native_ptr(), 213527486);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_variation_coordinates(const RID &p_font_rid, const Dictionary &p_variation_coordinates) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_variation_coordinates")._native_ptr(), 1217542888);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_variation_coordinates);
}

Dictionary TextServer::font_get_variation_coordinates(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_variation_coordinates")._native_ptr(), 1882737106);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_oversampling(const RID &p_font_rid, double p_oversampling) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_oversampling")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_oversampling_encoded);
}

double TextServer::font_get_oversampling(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_oversampling")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_rid);
}

TypedArray<Vector2i> TextServer::font_get_size_cache_list(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_size_cache_list")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_clear_size_cache(const RID &p_font_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_clear_size_cache")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_remove_size_cache(const RID &p_font_rid, const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_remove_size_cache")._native_ptr(), 2450610377);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size);
}

TypedArray<Dictionary> TextServer::font_get_size_cache_info(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_size_cache_info")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_ascent(const RID &p_font_rid, int64_t p_size, double p_ascent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_ascent")._native_ptr(), 1892459533);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_ascent_encoded;
	PtrToArg<double>::encode(p_ascent, &p_ascent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_ascent_encoded);
}

double TextServer::font_get_ascent(const RID &p_font_rid, int64_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_ascent")._native_ptr(), 755457166);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded);
}

void TextServer::font_set_descent(const RID &p_font_rid, int64_t p_size, double p_descent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_descent")._native_ptr(), 1892459533);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_descent_encoded;
	PtrToArg<double>::encode(p_descent, &p_descent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_descent_encoded);
}

double TextServer::font_get_descent(const RID &p_font_rid, int64_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_descent")._native_ptr(), 755457166);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded);
}

void TextServer::font_set_underline_position(const RID &p_font_rid, int64_t p_size, double p_underline_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_underline_position")._native_ptr(), 1892459533);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_underline_position_encoded;
	PtrToArg<double>::encode(p_underline_position, &p_underline_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_underline_position_encoded);
}

double TextServer::font_get_underline_position(const RID &p_font_rid, int64_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_underline_position")._native_ptr(), 755457166);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded);
}

void TextServer::font_set_underline_thickness(const RID &p_font_rid, int64_t p_size, double p_underline_thickness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_underline_thickness")._native_ptr(), 1892459533);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_underline_thickness_encoded;
	PtrToArg<double>::encode(p_underline_thickness, &p_underline_thickness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_underline_thickness_encoded);
}

double TextServer::font_get_underline_thickness(const RID &p_font_rid, int64_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_underline_thickness")._native_ptr(), 755457166);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded);
}

void TextServer::font_set_scale(const RID &p_font_rid, int64_t p_size, double p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_scale")._native_ptr(), 1892459533);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_scale_encoded);
}

double TextServer::font_get_scale(const RID &p_font_rid, int64_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_scale")._native_ptr(), 755457166);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded);
}

int64_t TextServer::font_get_texture_count(const RID &p_font_rid, const Vector2i &p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_texture_count")._native_ptr(), 1311001310);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid, &p_size);
}

void TextServer::font_clear_textures(const RID &p_font_rid, const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_clear_textures")._native_ptr(), 2450610377);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size);
}

void TextServer::font_remove_texture(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_remove_texture")._native_ptr(), 3810512262);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_texture_index_encoded;
	PtrToArg<int64_t>::encode(p_texture_index, &p_texture_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_texture_index_encoded);
}

void TextServer::font_set_texture_image(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index, const Ref<Image> &p_image) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_texture_image")._native_ptr(), 2354485091);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_texture_index_encoded;
	PtrToArg<int64_t>::encode(p_texture_index, &p_texture_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_texture_index_encoded, (p_image != nullptr ? &p_image->_owner : nullptr));
}

Ref<Image> TextServer::font_get_texture_image(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_texture_image")._native_ptr(), 2451761155);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	int64_t p_texture_index_encoded;
	PtrToArg<int64_t>::encode(p_texture_index, &p_texture_index_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_texture_index_encoded));
}

void TextServer::font_set_texture_offsets(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index, const PackedInt32Array &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_texture_offsets")._native_ptr(), 3005398047);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_texture_index_encoded;
	PtrToArg<int64_t>::encode(p_texture_index, &p_texture_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_texture_index_encoded, &p_offset);
}

PackedInt32Array TextServer::font_get_texture_offsets(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_texture_offsets")._native_ptr(), 3420028887);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	int64_t p_texture_index_encoded;
	PtrToArg<int64_t>::encode(p_texture_index, &p_texture_index_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_texture_index_encoded);
}

PackedInt32Array TextServer::font_get_glyph_list(const RID &p_font_rid, const Vector2i &p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_glyph_list")._native_ptr(), 46086620);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_font_rid, &p_size);
}

void TextServer::font_clear_glyphs(const RID &p_font_rid, const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_clear_glyphs")._native_ptr(), 2450610377);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size);
}

void TextServer::font_remove_glyph(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_remove_glyph")._native_ptr(), 3810512262);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_glyph_encoded);
}

Vector2 TextServer::font_get_glyph_advance(const RID &p_font_rid, int64_t p_size, int64_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_glyph_advance")._native_ptr(), 2555689501);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_glyph_encoded);
}

void TextServer::font_set_glyph_advance(const RID &p_font_rid, int64_t p_size, int64_t p_glyph, const Vector2 &p_advance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_glyph_advance")._native_ptr(), 3219397315);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_glyph_encoded, &p_advance);
}

Vector2 TextServer::font_get_glyph_offset(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_glyph_offset")._native_ptr(), 513728628);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_glyph_encoded);
}

void TextServer::font_set_glyph_offset(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_glyph_offset")._native_ptr(), 1812632090);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_glyph_encoded, &p_offset);
}

Vector2 TextServer::font_get_glyph_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_glyph_size")._native_ptr(), 513728628);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_glyph_encoded);
}

void TextServer::font_set_glyph_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Vector2 &p_gl_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_glyph_size")._native_ptr(), 1812632090);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_glyph_encoded, &p_gl_size);
}

Rect2 TextServer::font_get_glyph_uv_rect(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_glyph_uv_rect")._native_ptr(), 2274268786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_glyph_encoded);
}

void TextServer::font_set_glyph_uv_rect(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Rect2 &p_uv_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_glyph_uv_rect")._native_ptr(), 1973324081);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_glyph_encoded, &p_uv_rect);
}

int64_t TextServer::font_get_glyph_texture_idx(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_glyph_texture_idx")._native_ptr(), 4292800474);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_glyph_encoded);
}

void TextServer::font_set_glyph_texture_idx(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, int64_t p_texture_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_glyph_texture_idx")._native_ptr(), 4254580980);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	int64_t p_texture_idx_encoded;
	PtrToArg<int64_t>::encode(p_texture_idx, &p_texture_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_glyph_encoded, &p_texture_idx_encoded);
}

RID TextServer::font_get_glyph_texture_rid(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_glyph_texture_rid")._native_ptr(), 1451696141);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_glyph_encoded);
}

Vector2 TextServer::font_get_glyph_texture_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_glyph_texture_size")._native_ptr(), 513728628);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_glyph_encoded;
	PtrToArg<int64_t>::encode(p_glyph, &p_glyph_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_glyph_encoded);
}

Dictionary TextServer::font_get_glyph_contours(const RID &p_font, int64_t p_size, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_glyph_contours")._native_ptr(), 2903964473);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_font, &p_size_encoded, &p_index_encoded);
}

TypedArray<Vector2i> TextServer::font_get_kerning_list(const RID &p_font_rid, int64_t p_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_kerning_list")._native_ptr(), 1778388067);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded);
}

void TextServer::font_clear_kerning_map(const RID &p_font_rid, int64_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_clear_kerning_map")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded);
}

void TextServer::font_remove_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_remove_kerning")._native_ptr(), 2141860016);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_glyph_pair);
}

void TextServer::font_set_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_kerning")._native_ptr(), 3630965883);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_glyph_pair, &p_kerning);
}

Vector2 TextServer::font_get_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_kerning")._native_ptr(), 1019980169);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_glyph_pair);
}

int64_t TextServer::font_get_glyph_index(const RID &p_font_rid, int64_t p_size, int64_t p_char, int64_t p_variation_selector) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_glyph_index")._native_ptr(), 1765635060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_char_encoded;
	PtrToArg<int64_t>::encode(p_char, &p_char_encoded);
	int64_t p_variation_selector_encoded;
	PtrToArg<int64_t>::encode(p_variation_selector, &p_variation_selector_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_char_encoded, &p_variation_selector_encoded);
}

int64_t TextServer::font_get_char_from_glyph_index(const RID &p_font_rid, int64_t p_size, int64_t p_glyph_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_char_from_glyph_index")._native_ptr(), 2156738276);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_glyph_index_encoded;
	PtrToArg<int64_t>::encode(p_glyph_index, &p_glyph_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_font_rid, &p_size_encoded, &p_glyph_index_encoded);
}

bool TextServer::font_has_char(const RID &p_font_rid, int64_t p_char) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_has_char")._native_ptr(), 3120086654);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_char_encoded;
	PtrToArg<int64_t>::encode(p_char, &p_char_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid, &p_char_encoded);
}

String TextServer::font_get_supported_chars(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_supported_chars")._native_ptr(), 642473191);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_font_rid);
}

PackedInt32Array TextServer::font_get_supported_glyphs(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_supported_glyphs")._native_ptr(), 788230395);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_render_range(const RID &p_font_rid, const Vector2i &p_size, int64_t p_start, int64_t p_end) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_render_range")._native_ptr(), 4254580980);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_start_encoded;
	PtrToArg<int64_t>::encode(p_start, &p_start_encoded);
	int64_t p_end_encoded;
	PtrToArg<int64_t>::encode(p_end, &p_end_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_start_encoded, &p_end_encoded);
}

void TextServer::font_render_glyph(const RID &p_font_rid, const Vector2i &p_size, int64_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_render_glyph")._native_ptr(), 3810512262);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_size, &p_index_encoded);
}

void TextServer::font_draw_glyph(const RID &p_font_rid, const RID &p_canvas, int64_t p_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_draw_glyph")._native_ptr(), 3103234926);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_canvas, &p_size_encoded, &p_pos, &p_index_encoded, &p_color, &p_oversampling_encoded);
}

void TextServer::font_draw_glyph_outline(const RID &p_font_rid, const RID &p_canvas, int64_t p_size, int64_t p_outline_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_draw_glyph_outline")._native_ptr(), 1976041553);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_outline_size_encoded;
	PtrToArg<int64_t>::encode(p_outline_size, &p_outline_size_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_canvas, &p_size_encoded, &p_outline_size_encoded, &p_pos, &p_index_encoded, &p_color, &p_oversampling_encoded);
}

bool TextServer::font_is_language_supported(const RID &p_font_rid, const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_is_language_supported")._native_ptr(), 3199320846);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid, &p_language);
}

void TextServer::font_set_language_support_override(const RID &p_font_rid, const String &p_language, bool p_supported) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_language_support_override")._native_ptr(), 2313957094);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_supported_encoded;
	PtrToArg<bool>::encode(p_supported, &p_supported_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_language, &p_supported_encoded);
}

bool TextServer::font_get_language_support_override(const RID &p_font_rid, const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_language_support_override")._native_ptr(), 2829184646);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid, &p_language);
}

void TextServer::font_remove_language_support_override(const RID &p_font_rid, const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_remove_language_support_override")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_language);
}

PackedStringArray TextServer::font_get_language_support_overrides(const RID &p_font_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_language_support_overrides")._native_ptr(), 2801473409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_font_rid);
}

bool TextServer::font_is_script_supported(const RID &p_font_rid, const String &p_script) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_is_script_supported")._native_ptr(), 3199320846);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid, &p_script);
}

void TextServer::font_set_script_support_override(const RID &p_font_rid, const String &p_script, bool p_supported) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_script_support_override")._native_ptr(), 2313957094);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_supported_encoded;
	PtrToArg<bool>::encode(p_supported, &p_supported_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_script, &p_supported_encoded);
}

bool TextServer::font_get_script_support_override(const RID &p_font_rid, const String &p_script) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_script_support_override")._native_ptr(), 2829184646);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_font_rid, &p_script);
}

void TextServer::font_remove_script_support_override(const RID &p_font_rid, const String &p_script) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_remove_script_support_override")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_script);
}

PackedStringArray TextServer::font_get_script_support_overrides(const RID &p_font_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_script_support_overrides")._native_ptr(), 2801473409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_font_rid);
}

void TextServer::font_set_opentype_feature_overrides(const RID &p_font_rid, const Dictionary &p_overrides) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_opentype_feature_overrides")._native_ptr(), 1217542888);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_rid, &p_overrides);
}

Dictionary TextServer::font_get_opentype_feature_overrides(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_opentype_feature_overrides")._native_ptr(), 1882737106);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_font_rid);
}

Dictionary TextServer::font_supported_feature_list(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_supported_feature_list")._native_ptr(), 1882737106);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_font_rid);
}

Dictionary TextServer::font_supported_variation_list(const RID &p_font_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_supported_variation_list")._native_ptr(), 1882737106);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_font_rid);
}

double TextServer::font_get_global_oversampling() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_get_global_oversampling")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void TextServer::font_set_global_oversampling(double p_oversampling) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("font_set_global_oversampling")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_oversampling_encoded);
}

Vector2 TextServer::get_hex_code_box_size(int64_t p_size, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("get_hex_code_box_size")._native_ptr(), 3016396712);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_size_encoded, &p_index_encoded);
}

void TextServer::draw_hex_code_box(const RID &p_canvas, int64_t p_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("draw_hex_code_box")._native_ptr(), 1602046441);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas, &p_size_encoded, &p_pos, &p_index_encoded, &p_color);
}

RID TextServer::create_shaped_text(TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("create_shaped_text")._native_ptr(), 1231398698);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_direction_encoded, &p_orientation_encoded);
}

void TextServer::shaped_text_clear(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_clear")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid);
}

RID TextServer::shaped_text_duplicate(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_duplicate")._native_ptr(), 41030802);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_rid);
}

void TextServer::shaped_text_set_direction(const RID &p_shaped, TextServer::Direction p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_set_direction")._native_ptr(), 1551430183);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_direction_encoded);
}

TextServer::Direction TextServer::shaped_text_get_direction(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_direction")._native_ptr(), 3065904362);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::Direction(0)));
	return (TextServer::Direction)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped);
}

TextServer::Direction TextServer::shaped_text_get_inferred_direction(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_inferred_direction")._native_ptr(), 3065904362);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::Direction(0)));
	return (TextServer::Direction)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped);
}

void TextServer::shaped_text_set_bidi_override(const RID &p_shaped, const Array &p_override) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_set_bidi_override")._native_ptr(), 684822712);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_override);
}

void TextServer::shaped_text_set_custom_punctuation(const RID &p_shaped, const String &p_punct) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_set_custom_punctuation")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_punct);
}

String TextServer::shaped_text_get_custom_punctuation(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_custom_punctuation")._native_ptr(), 642473191);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_shaped);
}

void TextServer::shaped_text_set_custom_ellipsis(const RID &p_shaped, int64_t p_char) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_set_custom_ellipsis")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_char_encoded;
	PtrToArg<int64_t>::encode(p_char, &p_char_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_char_encoded);
}

int64_t TextServer::shaped_text_get_custom_ellipsis(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_custom_ellipsis")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped);
}

void TextServer::shaped_text_set_orientation(const RID &p_shaped, TextServer::Orientation p_orientation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_set_orientation")._native_ptr(), 3019609126);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_orientation_encoded);
}

TextServer::Orientation TextServer::shaped_text_get_orientation(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_orientation")._native_ptr(), 3142708106);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::Orientation(0)));
	return (TextServer::Orientation)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped);
}

void TextServer::shaped_text_set_preserve_invalid(const RID &p_shaped, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_set_preserve_invalid")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_enabled_encoded);
}

bool TextServer::shaped_text_get_preserve_invalid(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_preserve_invalid")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_shaped);
}

void TextServer::shaped_text_set_preserve_control(const RID &p_shaped, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_set_preserve_control")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_enabled_encoded);
}

bool TextServer::shaped_text_get_preserve_control(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_preserve_control")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_shaped);
}

void TextServer::shaped_text_set_spacing(const RID &p_shaped, TextServer::SpacingType p_spacing, int64_t p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_set_spacing")._native_ptr(), 1307259930);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_spacing_encoded;
	PtrToArg<int64_t>::encode(p_spacing, &p_spacing_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_spacing_encoded, &p_value_encoded);
}

int64_t TextServer::shaped_text_get_spacing(const RID &p_shaped, TextServer::SpacingType p_spacing) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_spacing")._native_ptr(), 1213653558);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_spacing_encoded;
	PtrToArg<int64_t>::encode(p_spacing, &p_spacing_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_spacing_encoded);
}

bool TextServer::shaped_text_add_string(const RID &p_shaped, const String &p_text, const TypedArray<RID> &p_fonts, int64_t p_size, const Dictionary &p_opentype_features, const String &p_language, const Variant &p_meta) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_add_string")._native_ptr(), 623473029);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_shaped, &p_text, &p_fonts, &p_size_encoded, &p_opentype_features, &p_language, &p_meta);
}

bool TextServer::shaped_text_add_object(const RID &p_shaped, const Variant &p_key, const Vector2 &p_size, InlineAlignment p_inline_align, int64_t p_length, double p_baseline) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_add_object")._native_ptr(), 3664424789);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_inline_align_encoded;
	PtrToArg<int64_t>::encode(p_inline_align, &p_inline_align_encoded);
	int64_t p_length_encoded;
	PtrToArg<int64_t>::encode(p_length, &p_length_encoded);
	double p_baseline_encoded;
	PtrToArg<double>::encode(p_baseline, &p_baseline_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_shaped, &p_key, &p_size, &p_inline_align_encoded, &p_length_encoded, &p_baseline_encoded);
}

bool TextServer::shaped_text_resize_object(const RID &p_shaped, const Variant &p_key, const Vector2 &p_size, InlineAlignment p_inline_align, double p_baseline) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_resize_object")._native_ptr(), 790361552);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_inline_align_encoded;
	PtrToArg<int64_t>::encode(p_inline_align, &p_inline_align_encoded);
	double p_baseline_encoded;
	PtrToArg<double>::encode(p_baseline, &p_baseline_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_shaped, &p_key, &p_size, &p_inline_align_encoded, &p_baseline_encoded);
}

bool TextServer::shaped_text_has_object(const RID &p_shaped, const Variant &p_key) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_has_object")._native_ptr(), 2360964694);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_shaped, &p_key);
}

String TextServer::shaped_get_text(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_text")._native_ptr(), 642473191);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_shaped);
}

int64_t TextServer::shaped_get_span_count(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_span_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped);
}

Variant TextServer::shaped_get_span_meta(const RID &p_shaped, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_span_meta")._native_ptr(), 4069510997);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_shaped, &p_index_encoded);
}

Variant TextServer::shaped_get_span_embedded_object(const RID &p_shaped, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_span_embedded_object")._native_ptr(), 4069510997);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_shaped, &p_index_encoded);
}

String TextServer::shaped_get_span_text(const RID &p_shaped, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_span_text")._native_ptr(), 1464764419);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_shaped, &p_index_encoded);
}

Variant TextServer::shaped_get_span_object(const RID &p_shaped, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_span_object")._native_ptr(), 4069510997);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_shaped, &p_index_encoded);
}

void TextServer::shaped_set_span_update_font(const RID &p_shaped, int64_t p_index, const TypedArray<RID> &p_fonts, int64_t p_size, const Dictionary &p_opentype_features) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_set_span_update_font")._native_ptr(), 2022725822);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_index_encoded, &p_fonts, &p_size_encoded, &p_opentype_features);
}

int64_t TextServer::shaped_get_run_count(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_run_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped);
}

String TextServer::shaped_get_run_text(const RID &p_shaped, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_run_text")._native_ptr(), 1464764419);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_shaped, &p_index_encoded);
}

Vector2i TextServer::shaped_get_run_range(const RID &p_shaped, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_run_range")._native_ptr(), 4069534484);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_shaped, &p_index_encoded);
}

RID TextServer::shaped_get_run_font_rid(const RID &p_shaped, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_run_font_rid")._native_ptr(), 1066463050);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_shaped, &p_index_encoded);
}

int32_t TextServer::shaped_get_run_font_size(const RID &p_shaped, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_run_font_size")._native_ptr(), 1120910005);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_index_encoded);
}

String TextServer::shaped_get_run_language(const RID &p_shaped, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_run_language")._native_ptr(), 1464764419);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_shaped, &p_index_encoded);
}

TextServer::Direction TextServer::shaped_get_run_direction(const RID &p_shaped, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_run_direction")._native_ptr(), 2413896864);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::Direction(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (TextServer::Direction)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_index_encoded);
}

Variant TextServer::shaped_get_run_object(const RID &p_shaped, int64_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_get_run_object")._native_ptr(), 4069510997);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_shaped, &p_index_encoded);
}

RID TextServer::shaped_text_substr(const RID &p_shaped, int64_t p_start, int64_t p_length) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_substr")._native_ptr(), 1937682086);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_start_encoded;
	PtrToArg<int64_t>::encode(p_start, &p_start_encoded);
	int64_t p_length_encoded;
	PtrToArg<int64_t>::encode(p_length, &p_length_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_shaped, &p_start_encoded, &p_length_encoded);
}

RID TextServer::shaped_text_get_parent(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_parent")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_shaped);
}

double TextServer::shaped_text_fit_to_width(const RID &p_shaped, double p_width, BitField<TextServer::JustificationFlag> p_justification_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_fit_to_width")._native_ptr(), 530670926);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_shaped, &p_width_encoded, &p_justification_flags);
}

double TextServer::shaped_text_tab_align(const RID &p_shaped, const PackedFloat32Array &p_tab_stops) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_tab_align")._native_ptr(), 1283669550);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_shaped, &p_tab_stops);
}

bool TextServer::shaped_text_shape(const RID &p_shaped) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_shape")._native_ptr(), 3521089500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_shaped);
}

bool TextServer::shaped_text_is_ready(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_is_ready")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_shaped);
}

bool TextServer::shaped_text_has_visible_chars(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_has_visible_chars")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_shaped);
}

TypedArray<Dictionary> TextServer::shaped_text_get_glyphs(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_glyphs")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, &p_shaped);
}

TypedArray<Dictionary> TextServer::shaped_text_sort_logical(const RID &p_shaped) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_sort_logical")._native_ptr(), 2670461153);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, &p_shaped);
}

int64_t TextServer::shaped_text_get_glyph_count(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_glyph_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped);
}

Vector2i TextServer::shaped_text_get_range(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_range")._native_ptr(), 733700038);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_shaped);
}

PackedInt32Array TextServer::shaped_text_get_line_breaks_adv(const RID &p_shaped, const PackedFloat32Array &p_width, int64_t p_start, bool p_once, BitField<TextServer::LineBreakFlag> p_break_flags) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_line_breaks_adv")._native_ptr(), 2376991424);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	int64_t p_start_encoded;
	PtrToArg<int64_t>::encode(p_start, &p_start_encoded);
	int8_t p_once_encoded;
	PtrToArg<bool>::encode(p_once, &p_once_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_shaped, &p_width, &p_start_encoded, &p_once_encoded, &p_break_flags);
}

PackedInt32Array TextServer::shaped_text_get_line_breaks(const RID &p_shaped, double p_width, int64_t p_start, BitField<TextServer::LineBreakFlag> p_break_flags) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_line_breaks")._native_ptr(), 2651359741);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int64_t p_start_encoded;
	PtrToArg<int64_t>::encode(p_start, &p_start_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_shaped, &p_width_encoded, &p_start_encoded, &p_break_flags);
}

PackedInt32Array TextServer::shaped_text_get_word_breaks(const RID &p_shaped, BitField<TextServer::GraphemeFlag> p_grapheme_flags, BitField<TextServer::GraphemeFlag> p_skip_grapheme_flags) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_word_breaks")._native_ptr(), 4099476853);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_shaped, &p_grapheme_flags, &p_skip_grapheme_flags);
}

int64_t TextServer::shaped_text_get_trim_pos(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_trim_pos")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped);
}

int64_t TextServer::shaped_text_get_ellipsis_pos(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_ellipsis_pos")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped);
}

TypedArray<Dictionary> TextServer::shaped_text_get_ellipsis_glyphs(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_ellipsis_glyphs")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, &p_shaped);
}

int64_t TextServer::shaped_text_get_ellipsis_glyph_count(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_ellipsis_glyph_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped);
}

void TextServer::shaped_text_overrun_trim_to_width(const RID &p_shaped, double p_width, BitField<TextServer::TextOverrunFlag> p_overrun_trim_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_overrun_trim_to_width")._native_ptr(), 2723146520);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_width_encoded, &p_overrun_trim_flags);
}

Array TextServer::shaped_text_get_objects(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_objects")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_shaped);
}

Rect2 TextServer::shaped_text_get_object_rect(const RID &p_shaped, const Variant &p_key) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_object_rect")._native_ptr(), 447978354);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner, &p_shaped, &p_key);
}

Vector2i TextServer::shaped_text_get_object_range(const RID &p_shaped, const Variant &p_key) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_object_range")._native_ptr(), 2524675647);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_shaped, &p_key);
}

int64_t TextServer::shaped_text_get_object_glyph(const RID &p_shaped, const Variant &p_key) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_object_glyph")._native_ptr(), 1260085030);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_key);
}

Vector2 TextServer::shaped_text_get_size(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_size")._native_ptr(), 2440833711);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_shaped);
}

double TextServer::shaped_text_get_ascent(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_ascent")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_shaped);
}

double TextServer::shaped_text_get_descent(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_descent")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_shaped);
}

double TextServer::shaped_text_get_width(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_width")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_shaped);
}

double TextServer::shaped_text_get_underline_position(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_underline_position")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_shaped);
}

double TextServer::shaped_text_get_underline_thickness(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_underline_thickness")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_shaped);
}

Dictionary TextServer::shaped_text_get_carets(const RID &p_shaped, int64_t p_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_carets")._native_ptr(), 1574219346);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_position_encoded;
	PtrToArg<int64_t>::encode(p_position, &p_position_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_shaped, &p_position_encoded);
}

PackedVector2Array TextServer::shaped_text_get_selection(const RID &p_shaped, int64_t p_start, int64_t p_end) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_selection")._native_ptr(), 3714187733);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	int64_t p_start_encoded;
	PtrToArg<int64_t>::encode(p_start, &p_start_encoded);
	int64_t p_end_encoded;
	PtrToArg<int64_t>::encode(p_end, &p_end_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner, &p_shaped, &p_start_encoded, &p_end_encoded);
}

int64_t TextServer::shaped_text_hit_test_grapheme(const RID &p_shaped, double p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_hit_test_grapheme")._native_ptr(), 3149310417);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	double p_coords_encoded;
	PtrToArg<double>::encode(p_coords, &p_coords_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_coords_encoded);
}

int64_t TextServer::shaped_text_hit_test_position(const RID &p_shaped, double p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_hit_test_position")._native_ptr(), 3149310417);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	double p_coords_encoded;
	PtrToArg<double>::encode(p_coords, &p_coords_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_coords_encoded);
}

Vector2 TextServer::shaped_text_get_grapheme_bounds(const RID &p_shaped, int64_t p_pos) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_grapheme_bounds")._native_ptr(), 2546185844);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_pos_encoded;
	PtrToArg<int64_t>::encode(p_pos, &p_pos_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_shaped, &p_pos_encoded);
}

int64_t TextServer::shaped_text_next_grapheme_pos(const RID &p_shaped, int64_t p_pos) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_next_grapheme_pos")._native_ptr(), 1120910005);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_pos_encoded;
	PtrToArg<int64_t>::encode(p_pos, &p_pos_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_pos_encoded);
}

int64_t TextServer::shaped_text_prev_grapheme_pos(const RID &p_shaped, int64_t p_pos) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_prev_grapheme_pos")._native_ptr(), 1120910005);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_pos_encoded;
	PtrToArg<int64_t>::encode(p_pos, &p_pos_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_pos_encoded);
}

PackedInt32Array TextServer::shaped_text_get_character_breaks(const RID &p_shaped) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_character_breaks")._native_ptr(), 788230395);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_shaped);
}

int64_t TextServer::shaped_text_next_character_pos(const RID &p_shaped, int64_t p_pos) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_next_character_pos")._native_ptr(), 1120910005);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_pos_encoded;
	PtrToArg<int64_t>::encode(p_pos, &p_pos_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_pos_encoded);
}

int64_t TextServer::shaped_text_prev_character_pos(const RID &p_shaped, int64_t p_pos) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_prev_character_pos")._native_ptr(), 1120910005);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_pos_encoded;
	PtrToArg<int64_t>::encode(p_pos, &p_pos_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_pos_encoded);
}

int64_t TextServer::shaped_text_closest_character_pos(const RID &p_shaped, int64_t p_pos) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_closest_character_pos")._native_ptr(), 1120910005);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_pos_encoded;
	PtrToArg<int64_t>::encode(p_pos, &p_pos_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_pos_encoded);
}

void TextServer::shaped_text_draw(const RID &p_shaped, const RID &p_canvas, const Vector2 &p_pos, double p_clip_l, double p_clip_r, const Color &p_color, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_draw")._native_ptr(), 1647687596);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_clip_l_encoded;
	PtrToArg<double>::encode(p_clip_l, &p_clip_l_encoded);
	double p_clip_r_encoded;
	PtrToArg<double>::encode(p_clip_r, &p_clip_r_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_canvas, &p_pos, &p_clip_l_encoded, &p_clip_r_encoded, &p_color, &p_oversampling_encoded);
}

void TextServer::shaped_text_draw_outline(const RID &p_shaped, const RID &p_canvas, const Vector2 &p_pos, double p_clip_l, double p_clip_r, int64_t p_outline_size, const Color &p_color, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_draw_outline")._native_ptr(), 1217146601);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_clip_l_encoded;
	PtrToArg<double>::encode(p_clip_l, &p_clip_l_encoded);
	double p_clip_r_encoded;
	PtrToArg<double>::encode(p_clip_r, &p_clip_r_encoded);
	int64_t p_outline_size_encoded;
	PtrToArg<int64_t>::encode(p_outline_size, &p_outline_size_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shaped, &p_canvas, &p_pos, &p_clip_l_encoded, &p_clip_r_encoded, &p_outline_size_encoded, &p_color, &p_oversampling_encoded);
}

TextServer::Direction TextServer::shaped_text_get_dominant_direction_in_range(const RID &p_shaped, int64_t p_start, int64_t p_end) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("shaped_text_get_dominant_direction_in_range")._native_ptr(), 3326907668);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::Direction(0)));
	int64_t p_start_encoded;
	PtrToArg<int64_t>::encode(p_start, &p_start_encoded);
	int64_t p_end_encoded;
	PtrToArg<int64_t>::encode(p_end, &p_end_encoded);
	return (TextServer::Direction)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shaped, &p_start_encoded, &p_end_encoded);
}

String TextServer::format_number(const String &p_number, const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("format_number")._native_ptr(), 2664628024);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_number, &p_language);
}

String TextServer::parse_number(const String &p_number, const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("parse_number")._native_ptr(), 2664628024);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_number, &p_language);
}

String TextServer::percent_sign(const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("percent_sign")._native_ptr(), 993269549);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_language);
}

PackedInt32Array TextServer::string_get_word_breaks(const String &p_string, const String &p_language, int64_t p_chars_per_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("string_get_word_breaks")._native_ptr(), 581857818);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	int64_t p_chars_per_line_encoded;
	PtrToArg<int64_t>::encode(p_chars_per_line, &p_chars_per_line_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_string, &p_language, &p_chars_per_line_encoded);
}

PackedInt32Array TextServer::string_get_character_breaks(const String &p_string, const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("string_get_character_breaks")._native_ptr(), 2333794773);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_string, &p_language);
}

int64_t TextServer::is_confusable(const String &p_string, const PackedStringArray &p_dict) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("is_confusable")._native_ptr(), 1433197768);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_string, &p_dict);
}

bool TextServer::spoof_check(const String &p_string) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("spoof_check")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_string);
}

String TextServer::strip_diacritics(const String &p_string) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("strip_diacritics")._native_ptr(), 3135753539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_string);
}

bool TextServer::is_valid_identifier(const String &p_string) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("is_valid_identifier")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_string);
}

bool TextServer::is_valid_letter(uint64_t p_unicode) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("is_valid_letter")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_unicode_encoded;
	PtrToArg<int64_t>::encode(p_unicode, &p_unicode_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_unicode_encoded);
}

String TextServer::string_to_upper(const String &p_string, const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("string_to_upper")._native_ptr(), 2664628024);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_string, &p_language);
}

String TextServer::string_to_lower(const String &p_string, const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("string_to_lower")._native_ptr(), 2664628024);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_string, &p_language);
}

String TextServer::string_to_title(const String &p_string, const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("string_to_title")._native_ptr(), 2664628024);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_string, &p_language);
}

TypedArray<Vector3i> TextServer::parse_structured_text(TextServer::StructuredTextParser p_parser_type, const Array &p_args, const String &p_text) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServer::get_class_static()._native_ptr(), StringName("parse_structured_text")._native_ptr(), 3310685015);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector3i>()));
	int64_t p_parser_type_encoded;
	PtrToArg<int64_t>::encode(p_parser_type, &p_parser_type_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector3i>>(_gde_method_bind, _owner, &p_parser_type_encoded, &p_args, &p_text);
}

} // namespace godot
