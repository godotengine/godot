/**************************************************************************/
/*  tile_map.cpp                                                          */
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

#include <godot_cpp/classes/tile_map.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/tile_data.hpp>
#include <godot_cpp/classes/tile_map_pattern.hpp>

namespace godot {

void TileMap::set_navigation_map(int32_t p_layer, const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_navigation_map")._native_ptr(), 4040184819);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_map);
}

RID TileMap::get_navigation_map(int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_navigation_map")._native_ptr(), 495598643);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::force_update(int32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("force_update")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::set_tileset(const Ref<TileSet> &p_tileset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_tileset")._native_ptr(), 774531446);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_tileset != nullptr ? &p_tileset->_owner : nullptr));
}

Ref<TileSet> TileMap::get_tileset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_tileset")._native_ptr(), 2678226422);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TileSet>()));
	return Ref<TileSet>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TileSet>(_gde_method_bind, _owner));
}

void TileMap::set_rendering_quadrant_size(int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_rendering_quadrant_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

int32_t TileMap::get_rendering_quadrant_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_rendering_quadrant_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TileMap::get_layers_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_layers_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileMap::add_layer(int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("add_layer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_to_position_encoded);
}

void TileMap::move_layer(int32_t p_layer, int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("move_layer")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_to_position_encoded);
}

void TileMap::remove_layer(int32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("remove_layer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::set_layer_name(int32_t p_layer, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_layer_name")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_name);
}

String TileMap::get_layer_name(int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_layer_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::set_layer_enabled(int32_t p_layer, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_layer_enabled")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_enabled_encoded);
}

bool TileMap::is_layer_enabled(int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("is_layer_enabled")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::set_layer_modulate(int32_t p_layer, const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_layer_modulate")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_modulate);
}

Color TileMap::get_layer_modulate(int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_layer_modulate")._native_ptr(), 3457211756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::set_layer_y_sort_enabled(int32_t p_layer, bool p_y_sort_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_layer_y_sort_enabled")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_y_sort_enabled_encoded;
	PtrToArg<bool>::encode(p_y_sort_enabled, &p_y_sort_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_y_sort_enabled_encoded);
}

bool TileMap::is_layer_y_sort_enabled(int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("is_layer_y_sort_enabled")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::set_layer_y_sort_origin(int32_t p_layer, int32_t p_y_sort_origin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_layer_y_sort_origin")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int64_t p_y_sort_origin_encoded;
	PtrToArg<int64_t>::encode(p_y_sort_origin, &p_y_sort_origin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_y_sort_origin_encoded);
}

int32_t TileMap::get_layer_y_sort_origin(int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_layer_y_sort_origin")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::set_layer_z_index(int32_t p_layer, int32_t p_z_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_layer_z_index")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int64_t p_z_index_encoded;
	PtrToArg<int64_t>::encode(p_z_index, &p_z_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_z_index_encoded);
}

int32_t TileMap::get_layer_z_index(int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_layer_z_index")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::set_layer_navigation_enabled(int32_t p_layer, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_layer_navigation_enabled")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_enabled_encoded);
}

bool TileMap::is_layer_navigation_enabled(int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("is_layer_navigation_enabled")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::set_layer_navigation_map(int32_t p_layer, const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_layer_navigation_map")._native_ptr(), 4040184819);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_map);
}

RID TileMap::get_layer_navigation_map(int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_layer_navigation_map")._native_ptr(), 495598643);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::set_collision_animatable(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_collision_animatable")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TileMap::is_collision_animatable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("is_collision_animatable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TileMap::set_collision_visibility_mode(TileMap::VisibilityMode p_collision_visibility_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_collision_visibility_mode")._native_ptr(), 3193440636);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_collision_visibility_mode_encoded;
	PtrToArg<int64_t>::encode(p_collision_visibility_mode, &p_collision_visibility_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_collision_visibility_mode_encoded);
}

TileMap::VisibilityMode TileMap::get_collision_visibility_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_collision_visibility_mode")._native_ptr(), 1697018252);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TileMap::VisibilityMode(0)));
	return (TileMap::VisibilityMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileMap::set_navigation_visibility_mode(TileMap::VisibilityMode p_navigation_visibility_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_navigation_visibility_mode")._native_ptr(), 3193440636);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_navigation_visibility_mode_encoded;
	PtrToArg<int64_t>::encode(p_navigation_visibility_mode, &p_navigation_visibility_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_navigation_visibility_mode_encoded);
}

TileMap::VisibilityMode TileMap::get_navigation_visibility_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_navigation_visibility_mode")._native_ptr(), 1697018252);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TileMap::VisibilityMode(0)));
	return (TileMap::VisibilityMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileMap::set_cell(int32_t p_layer, const Vector2i &p_coords, int32_t p_source_id, const Vector2i &p_atlas_coords, int32_t p_alternative_tile) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_cell")._native_ptr(), 966713560);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int64_t p_source_id_encoded;
	PtrToArg<int64_t>::encode(p_source_id, &p_source_id_encoded);
	int64_t p_alternative_tile_encoded;
	PtrToArg<int64_t>::encode(p_alternative_tile, &p_alternative_tile_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_coords, &p_source_id_encoded, &p_atlas_coords, &p_alternative_tile_encoded);
}

void TileMap::erase_cell(int32_t p_layer, const Vector2i &p_coords) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("erase_cell")._native_ptr(), 2311374912);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_coords);
}

int32_t TileMap::get_cell_source_id(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_cell_source_id")._native_ptr(), 551761942);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_use_proxies_encoded;
	PtrToArg<bool>::encode(p_use_proxies, &p_use_proxies_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_layer_encoded, &p_coords, &p_use_proxies_encoded);
}

Vector2i TileMap::get_cell_atlas_coords(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_cell_atlas_coords")._native_ptr(), 1869815066);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_use_proxies_encoded;
	PtrToArg<bool>::encode(p_use_proxies, &p_use_proxies_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_layer_encoded, &p_coords, &p_use_proxies_encoded);
}

int32_t TileMap::get_cell_alternative_tile(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_cell_alternative_tile")._native_ptr(), 551761942);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_use_proxies_encoded;
	PtrToArg<bool>::encode(p_use_proxies, &p_use_proxies_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_layer_encoded, &p_coords, &p_use_proxies_encoded);
}

TileData *TileMap::get_cell_tile_data(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_cell_tile_data")._native_ptr(), 2849631287);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_use_proxies_encoded;
	PtrToArg<bool>::encode(p_use_proxies, &p_use_proxies_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<TileData>(_gde_method_bind, _owner, &p_layer_encoded, &p_coords, &p_use_proxies_encoded);
}

bool TileMap::is_cell_flipped_h(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("is_cell_flipped_h")._native_ptr(), 2908343862);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_use_proxies_encoded;
	PtrToArg<bool>::encode(p_use_proxies, &p_use_proxies_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_encoded, &p_coords, &p_use_proxies_encoded);
}

bool TileMap::is_cell_flipped_v(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("is_cell_flipped_v")._native_ptr(), 2908343862);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_use_proxies_encoded;
	PtrToArg<bool>::encode(p_use_proxies, &p_use_proxies_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_encoded, &p_coords, &p_use_proxies_encoded);
}

bool TileMap::is_cell_transposed(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("is_cell_transposed")._native_ptr(), 2908343862);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_use_proxies_encoded;
	PtrToArg<bool>::encode(p_use_proxies, &p_use_proxies_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_encoded, &p_coords, &p_use_proxies_encoded);
}

Vector2i TileMap::get_coords_for_body_rid(const RID &p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_coords_for_body_rid")._native_ptr(), 291584212);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_body);
}

int32_t TileMap::get_layer_for_body_rid(const RID &p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_layer_for_body_rid")._native_ptr(), 3917799429);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

Ref<TileMapPattern> TileMap::get_pattern(int32_t p_layer, const TypedArray<Vector2i> &p_coords_array) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_pattern")._native_ptr(), 2833570986);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TileMapPattern>()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return Ref<TileMapPattern>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TileMapPattern>(_gde_method_bind, _owner, &p_layer_encoded, &p_coords_array));
}

Vector2i TileMap::map_pattern(const Vector2i &p_position_in_tilemap, const Vector2i &p_coords_in_pattern, const Ref<TileMapPattern> &p_pattern) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("map_pattern")._native_ptr(), 1864516957);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_position_in_tilemap, &p_coords_in_pattern, (p_pattern != nullptr ? &p_pattern->_owner : nullptr));
}

void TileMap::set_pattern(int32_t p_layer, const Vector2i &p_position, const Ref<TileMapPattern> &p_pattern) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_pattern")._native_ptr(), 1195853946);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_position, (p_pattern != nullptr ? &p_pattern->_owner : nullptr));
}

void TileMap::set_cells_terrain_connect(int32_t p_layer, const TypedArray<Vector2i> &p_cells, int32_t p_terrain_set, int32_t p_terrain, bool p_ignore_empty_terrains) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_cells_terrain_connect")._native_ptr(), 3578627656);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_terrain_encoded;
	PtrToArg<int64_t>::encode(p_terrain, &p_terrain_encoded);
	int8_t p_ignore_empty_terrains_encoded;
	PtrToArg<bool>::encode(p_ignore_empty_terrains, &p_ignore_empty_terrains_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_cells, &p_terrain_set_encoded, &p_terrain_encoded, &p_ignore_empty_terrains_encoded);
}

void TileMap::set_cells_terrain_path(int32_t p_layer, const TypedArray<Vector2i> &p_path, int32_t p_terrain_set, int32_t p_terrain, bool p_ignore_empty_terrains) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("set_cells_terrain_path")._native_ptr(), 3578627656);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_terrain_encoded;
	PtrToArg<int64_t>::encode(p_terrain, &p_terrain_encoded);
	int8_t p_ignore_empty_terrains_encoded;
	PtrToArg<bool>::encode(p_ignore_empty_terrains, &p_ignore_empty_terrains_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_path, &p_terrain_set_encoded, &p_terrain_encoded, &p_ignore_empty_terrains_encoded);
}

void TileMap::fix_invalid_tiles() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("fix_invalid_tiles")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TileMap::clear_layer(int32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("clear_layer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded);
}

void TileMap::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TileMap::update_internals() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("update_internals")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TileMap::notify_runtime_tile_data_update(int32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("notify_runtime_tile_data_update")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded);
}

TypedArray<Vector2i> TileMap::get_surrounding_cells(const Vector2i &p_coords) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_surrounding_cells")._native_ptr(), 2673526557);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner, &p_coords);
}

TypedArray<Vector2i> TileMap::get_used_cells(int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_used_cells")._native_ptr(), 663333327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner, &p_layer_encoded);
}

TypedArray<Vector2i> TileMap::get_used_cells_by_id(int32_t p_layer, int32_t p_source_id, const Vector2i &p_atlas_coords, int32_t p_alternative_tile) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_used_cells_by_id")._native_ptr(), 2931012785);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int64_t p_source_id_encoded;
	PtrToArg<int64_t>::encode(p_source_id, &p_source_id_encoded);
	int64_t p_alternative_tile_encoded;
	PtrToArg<int64_t>::encode(p_alternative_tile, &p_alternative_tile_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner, &p_layer_encoded, &p_source_id_encoded, &p_atlas_coords, &p_alternative_tile_encoded);
}

Rect2i TileMap::get_used_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_used_rect")._native_ptr(), 410525958);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner);
}

Vector2 TileMap::map_to_local(const Vector2i &p_map_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("map_to_local")._native_ptr(), 108438297);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_map_position);
}

Vector2i TileMap::local_to_map(const Vector2 &p_local_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("local_to_map")._native_ptr(), 837806996);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_local_position);
}

Vector2i TileMap::get_neighbor_cell(const Vector2i &p_coords, TileSet::CellNeighbor p_neighbor) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMap::get_class_static()._native_ptr(), StringName("get_neighbor_cell")._native_ptr(), 986575103);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_neighbor_encoded;
	PtrToArg<int64_t>::encode(p_neighbor, &p_neighbor_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_coords, &p_neighbor_encoded);
}

bool TileMap::_use_tile_data_runtime_update(int32_t p_layer, const Vector2i &p_coords) {
	return false;
}

void TileMap::_tile_data_runtime_update(int32_t p_layer, const Vector2i &p_coords, TileData *p_tile_data) {}

} // namespace godot
