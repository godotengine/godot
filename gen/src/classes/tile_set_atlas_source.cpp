/**************************************************************************/
/*  tile_set_atlas_source.cpp                                             */
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

#include <godot_cpp/classes/tile_set_atlas_source.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/tile_data.hpp>

namespace godot {

void TileSetAtlasSource::set_texture(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_texture")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> TileSetAtlasSource::get_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void TileSetAtlasSource::set_margins(const Vector2i &p_margins) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_margins")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margins);
}

Vector2i TileSetAtlasSource::get_margins() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_margins")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void TileSetAtlasSource::set_separation(const Vector2i &p_separation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_separation")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_separation);
}

Vector2i TileSetAtlasSource::get_separation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_separation")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void TileSetAtlasSource::set_texture_region_size(const Vector2i &p_texture_region_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_texture_region_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture_region_size);
}

Vector2i TileSetAtlasSource::get_texture_region_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_texture_region_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void TileSetAtlasSource::set_use_texture_padding(bool p_use_texture_padding) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_use_texture_padding")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_texture_padding_encoded;
	PtrToArg<bool>::encode(p_use_texture_padding, &p_use_texture_padding_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_use_texture_padding_encoded);
}

bool TileSetAtlasSource::get_use_texture_padding() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_use_texture_padding")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TileSetAtlasSource::create_tile(const Vector2i &p_atlas_coords, const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("create_tile")._native_ptr(), 190528769);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_atlas_coords, &p_size);
}

void TileSetAtlasSource::remove_tile(const Vector2i &p_atlas_coords) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("remove_tile")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_atlas_coords);
}

void TileSetAtlasSource::move_tile_in_atlas(const Vector2i &p_atlas_coords, const Vector2i &p_new_atlas_coords, const Vector2i &p_new_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("move_tile_in_atlas")._native_ptr(), 3870111920);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_atlas_coords, &p_new_atlas_coords, &p_new_size);
}

Vector2i TileSetAtlasSource::get_tile_size_in_atlas(const Vector2i &p_atlas_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tile_size_in_atlas")._native_ptr(), 3050897911);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_atlas_coords);
}

bool TileSetAtlasSource::has_room_for_tile(const Vector2i &p_atlas_coords, const Vector2i &p_size, int32_t p_animation_columns, const Vector2i &p_animation_separation, int32_t p_frames_count, const Vector2i &p_ignored_tile) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("has_room_for_tile")._native_ptr(), 3018597268);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_animation_columns_encoded;
	PtrToArg<int64_t>::encode(p_animation_columns, &p_animation_columns_encoded);
	int64_t p_frames_count_encoded;
	PtrToArg<int64_t>::encode(p_frames_count, &p_frames_count_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_atlas_coords, &p_size, &p_animation_columns_encoded, &p_animation_separation, &p_frames_count_encoded, &p_ignored_tile);
}

PackedVector2Array TileSetAtlasSource::get_tiles_to_be_removed_on_change(const Ref<Texture2D> &p_texture, const Vector2i &p_margins, const Vector2i &p_separation, const Vector2i &p_texture_region_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tiles_to_be_removed_on_change")._native_ptr(), 1240378054);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_margins, &p_separation, &p_texture_region_size);
}

Vector2i TileSetAtlasSource::get_tile_at_coords(const Vector2i &p_atlas_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tile_at_coords")._native_ptr(), 3050897911);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_atlas_coords);
}

bool TileSetAtlasSource::has_tiles_outside_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("has_tiles_outside_texture")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TileSetAtlasSource::clear_tiles_outside_texture() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("clear_tiles_outside_texture")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TileSetAtlasSource::set_tile_animation_columns(const Vector2i &p_atlas_coords, int32_t p_frame_columns) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_tile_animation_columns")._native_ptr(), 3200960707);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_frame_columns_encoded;
	PtrToArg<int64_t>::encode(p_frame_columns, &p_frame_columns_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_atlas_coords, &p_frame_columns_encoded);
}

int32_t TileSetAtlasSource::get_tile_animation_columns(const Vector2i &p_atlas_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tile_animation_columns")._native_ptr(), 2485466453);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_atlas_coords);
}

void TileSetAtlasSource::set_tile_animation_separation(const Vector2i &p_atlas_coords, const Vector2i &p_separation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_tile_animation_separation")._native_ptr(), 1941061099);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_atlas_coords, &p_separation);
}

Vector2i TileSetAtlasSource::get_tile_animation_separation(const Vector2i &p_atlas_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tile_animation_separation")._native_ptr(), 3050897911);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_atlas_coords);
}

void TileSetAtlasSource::set_tile_animation_speed(const Vector2i &p_atlas_coords, float p_speed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_tile_animation_speed")._native_ptr(), 2262553149);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_speed_encoded;
	PtrToArg<double>::encode(p_speed, &p_speed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_atlas_coords, &p_speed_encoded);
}

float TileSetAtlasSource::get_tile_animation_speed(const Vector2i &p_atlas_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tile_animation_speed")._native_ptr(), 719993801);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_atlas_coords);
}

void TileSetAtlasSource::set_tile_animation_mode(const Vector2i &p_atlas_coords, TileSetAtlasSource::TileAnimationMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_tile_animation_mode")._native_ptr(), 3192753483);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_atlas_coords, &p_mode_encoded);
}

TileSetAtlasSource::TileAnimationMode TileSetAtlasSource::get_tile_animation_mode(const Vector2i &p_atlas_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tile_animation_mode")._native_ptr(), 4025349959);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TileSetAtlasSource::TileAnimationMode(0)));
	return (TileSetAtlasSource::TileAnimationMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_atlas_coords);
}

void TileSetAtlasSource::set_tile_animation_frames_count(const Vector2i &p_atlas_coords, int32_t p_frames_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_tile_animation_frames_count")._native_ptr(), 3200960707);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_frames_count_encoded;
	PtrToArg<int64_t>::encode(p_frames_count, &p_frames_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_atlas_coords, &p_frames_count_encoded);
}

int32_t TileSetAtlasSource::get_tile_animation_frames_count(const Vector2i &p_atlas_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tile_animation_frames_count")._native_ptr(), 2485466453);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_atlas_coords);
}

void TileSetAtlasSource::set_tile_animation_frame_duration(const Vector2i &p_atlas_coords, int32_t p_frame_index, float p_duration) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_tile_animation_frame_duration")._native_ptr(), 2843487787);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_frame_index_encoded;
	PtrToArg<int64_t>::encode(p_frame_index, &p_frame_index_encoded);
	double p_duration_encoded;
	PtrToArg<double>::encode(p_duration, &p_duration_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_atlas_coords, &p_frame_index_encoded, &p_duration_encoded);
}

float TileSetAtlasSource::get_tile_animation_frame_duration(const Vector2i &p_atlas_coords, int32_t p_frame_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tile_animation_frame_duration")._native_ptr(), 1802448425);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_frame_index_encoded;
	PtrToArg<int64_t>::encode(p_frame_index, &p_frame_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_atlas_coords, &p_frame_index_encoded);
}

float TileSetAtlasSource::get_tile_animation_total_duration(const Vector2i &p_atlas_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tile_animation_total_duration")._native_ptr(), 719993801);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_atlas_coords);
}

int32_t TileSetAtlasSource::create_alternative_tile(const Vector2i &p_atlas_coords, int32_t p_alternative_id_override) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("create_alternative_tile")._native_ptr(), 2226298068);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_alternative_id_override_encoded;
	PtrToArg<int64_t>::encode(p_alternative_id_override, &p_alternative_id_override_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_atlas_coords, &p_alternative_id_override_encoded);
}

void TileSetAtlasSource::remove_alternative_tile(const Vector2i &p_atlas_coords, int32_t p_alternative_tile) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("remove_alternative_tile")._native_ptr(), 3200960707);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alternative_tile_encoded;
	PtrToArg<int64_t>::encode(p_alternative_tile, &p_alternative_tile_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_atlas_coords, &p_alternative_tile_encoded);
}

void TileSetAtlasSource::set_alternative_tile_id(const Vector2i &p_atlas_coords, int32_t p_alternative_tile, int32_t p_new_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("set_alternative_tile_id")._native_ptr(), 1499785778);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alternative_tile_encoded;
	PtrToArg<int64_t>::encode(p_alternative_tile, &p_alternative_tile_encoded);
	int64_t p_new_id_encoded;
	PtrToArg<int64_t>::encode(p_new_id, &p_new_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_atlas_coords, &p_alternative_tile_encoded, &p_new_id_encoded);
}

int32_t TileSetAtlasSource::get_next_alternative_tile_id(const Vector2i &p_atlas_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_next_alternative_tile_id")._native_ptr(), 2485466453);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_atlas_coords);
}

TileData *TileSetAtlasSource::get_tile_data(const Vector2i &p_atlas_coords, int32_t p_alternative_tile) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tile_data")._native_ptr(), 3534028207);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_alternative_tile_encoded;
	PtrToArg<int64_t>::encode(p_alternative_tile, &p_alternative_tile_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<TileData>(_gde_method_bind, _owner, &p_atlas_coords, &p_alternative_tile_encoded);
}

Vector2i TileSetAtlasSource::get_atlas_grid_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_atlas_grid_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

Rect2i TileSetAtlasSource::get_tile_texture_region(const Vector2i &p_atlas_coords, int32_t p_frame) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_tile_texture_region")._native_ptr(), 241857547);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	int64_t p_frame_encoded;
	PtrToArg<int64_t>::encode(p_frame, &p_frame_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner, &p_atlas_coords, &p_frame_encoded);
}

Ref<Texture2D> TileSetAtlasSource::get_runtime_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_runtime_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

Rect2i TileSetAtlasSource::get_runtime_tile_texture_region(const Vector2i &p_atlas_coords, int32_t p_frame) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSetAtlasSource::get_class_static()._native_ptr(), StringName("get_runtime_tile_texture_region")._native_ptr(), 104874263);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	int64_t p_frame_encoded;
	PtrToArg<int64_t>::encode(p_frame, &p_frame_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner, &p_atlas_coords, &p_frame_encoded);
}

} // namespace godot
