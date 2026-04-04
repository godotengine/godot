/**************************************************************************/
/*  tile_map_layer.cpp                                                    */
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

#include <godot_cpp/classes/tile_map_layer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/tile_data.hpp>
#include <godot_cpp/classes/tile_map_pattern.hpp>

namespace godot {

void TileMapLayer::set_cell(const Vector2i &p_coords, int32_t p_source_id, const Vector2i &p_atlas_coords, int32_t p_alternative_tile) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_cell")._native_ptr(), 2428518503);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_id_encoded;
	PtrToArg<int64_t>::encode(p_source_id, &p_source_id_encoded);
	int64_t p_alternative_tile_encoded;
	PtrToArg<int64_t>::encode(p_alternative_tile, &p_alternative_tile_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_coords, &p_source_id_encoded, &p_atlas_coords, &p_alternative_tile_encoded);
}

void TileMapLayer::erase_cell(const Vector2i &p_coords) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("erase_cell")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_coords);
}

void TileMapLayer::fix_invalid_tiles() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("fix_invalid_tiles")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TileMapLayer::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

int32_t TileMapLayer::get_cell_source_id(const Vector2i &p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_cell_source_id")._native_ptr(), 2485466453);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_coords);
}

Vector2i TileMapLayer::get_cell_atlas_coords(const Vector2i &p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_cell_atlas_coords")._native_ptr(), 3050897911);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_coords);
}

int32_t TileMapLayer::get_cell_alternative_tile(const Vector2i &p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_cell_alternative_tile")._native_ptr(), 2485466453);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_coords);
}

TileData *TileMapLayer::get_cell_tile_data(const Vector2i &p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_cell_tile_data")._native_ptr(), 205084707);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<TileData>(_gde_method_bind, _owner, &p_coords);
}

bool TileMapLayer::is_cell_flipped_h(const Vector2i &p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("is_cell_flipped_h")._native_ptr(), 3900751641);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_coords);
}

bool TileMapLayer::is_cell_flipped_v(const Vector2i &p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("is_cell_flipped_v")._native_ptr(), 3900751641);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_coords);
}

bool TileMapLayer::is_cell_transposed(const Vector2i &p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("is_cell_transposed")._native_ptr(), 3900751641);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_coords);
}

TypedArray<Vector2i> TileMapLayer::get_used_cells() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_used_cells")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner);
}

TypedArray<Vector2i> TileMapLayer::get_used_cells_by_id(int32_t p_source_id, const Vector2i &p_atlas_coords, int32_t p_alternative_tile) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_used_cells_by_id")._native_ptr(), 4175304538);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	int64_t p_source_id_encoded;
	PtrToArg<int64_t>::encode(p_source_id, &p_source_id_encoded);
	int64_t p_alternative_tile_encoded;
	PtrToArg<int64_t>::encode(p_alternative_tile, &p_alternative_tile_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner, &p_source_id_encoded, &p_atlas_coords, &p_alternative_tile_encoded);
}

Rect2i TileMapLayer::get_used_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_used_rect")._native_ptr(), 410525958);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner);
}

Ref<TileMapPattern> TileMapLayer::get_pattern(const TypedArray<Vector2i> &p_coords_array) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_pattern")._native_ptr(), 3820813253);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TileMapPattern>()));
	return Ref<TileMapPattern>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TileMapPattern>(_gde_method_bind, _owner, &p_coords_array));
}

void TileMapLayer::set_pattern(const Vector2i &p_position, const Ref<TileMapPattern> &p_pattern) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_pattern")._native_ptr(), 1491151770);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position, (p_pattern != nullptr ? &p_pattern->_owner : nullptr));
}

void TileMapLayer::set_cells_terrain_connect(const TypedArray<Vector2i> &p_cells, int32_t p_terrain_set, int32_t p_terrain, bool p_ignore_empty_terrains) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_cells_terrain_connect")._native_ptr(), 748968311);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_terrain_encoded;
	PtrToArg<int64_t>::encode(p_terrain, &p_terrain_encoded);
	int8_t p_ignore_empty_terrains_encoded;
	PtrToArg<bool>::encode(p_ignore_empty_terrains, &p_ignore_empty_terrains_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cells, &p_terrain_set_encoded, &p_terrain_encoded, &p_ignore_empty_terrains_encoded);
}

void TileMapLayer::set_cells_terrain_path(const TypedArray<Vector2i> &p_path, int32_t p_terrain_set, int32_t p_terrain, bool p_ignore_empty_terrains) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_cells_terrain_path")._native_ptr(), 748968311);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_terrain_encoded;
	PtrToArg<int64_t>::encode(p_terrain, &p_terrain_encoded);
	int8_t p_ignore_empty_terrains_encoded;
	PtrToArg<bool>::encode(p_ignore_empty_terrains, &p_ignore_empty_terrains_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_terrain_set_encoded, &p_terrain_encoded, &p_ignore_empty_terrains_encoded);
}

bool TileMapLayer::has_body_rid(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("has_body_rid")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_body);
}

Vector2i TileMapLayer::get_coords_for_body_rid(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_coords_for_body_rid")._native_ptr(), 733700038);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_body);
}

void TileMapLayer::update_internals() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("update_internals")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TileMapLayer::notify_runtime_tile_data_update() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("notify_runtime_tile_data_update")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Vector2i TileMapLayer::map_pattern(const Vector2i &p_position_in_tilemap, const Vector2i &p_coords_in_pattern, const Ref<TileMapPattern> &p_pattern) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("map_pattern")._native_ptr(), 1864516957);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_position_in_tilemap, &p_coords_in_pattern, (p_pattern != nullptr ? &p_pattern->_owner : nullptr));
}

TypedArray<Vector2i> TileMapLayer::get_surrounding_cells(const Vector2i &p_coords) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_surrounding_cells")._native_ptr(), 2673526557);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner, &p_coords);
}

Vector2i TileMapLayer::get_neighbor_cell(const Vector2i &p_coords, TileSet::CellNeighbor p_neighbor) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_neighbor_cell")._native_ptr(), 986575103);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_neighbor_encoded;
	PtrToArg<int64_t>::encode(p_neighbor, &p_neighbor_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_coords, &p_neighbor_encoded);
}

Vector2 TileMapLayer::map_to_local(const Vector2i &p_map_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("map_to_local")._native_ptr(), 108438297);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_map_position);
}

Vector2i TileMapLayer::local_to_map(const Vector2 &p_local_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("local_to_map")._native_ptr(), 837806996);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_local_position);
}

void TileMapLayer::set_tile_map_data_from_array(const PackedByteArray &p_tile_map_layer_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_tile_map_data_from_array")._native_ptr(), 2971499966);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tile_map_layer_data);
}

PackedByteArray TileMapLayer::get_tile_map_data_as_array() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_tile_map_data_as_array")._native_ptr(), 2362200018);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner);
}

void TileMapLayer::set_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TileMapLayer::is_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("is_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TileMapLayer::set_tile_set(const Ref<TileSet> &p_tile_set) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_tile_set")._native_ptr(), 774531446);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_tile_set != nullptr ? &p_tile_set->_owner : nullptr));
}

Ref<TileSet> TileMapLayer::get_tile_set() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_tile_set")._native_ptr(), 2678226422);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TileSet>()));
	return Ref<TileSet>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TileSet>(_gde_method_bind, _owner));
}

void TileMapLayer::set_y_sort_origin(int32_t p_y_sort_origin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_y_sort_origin")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_y_sort_origin_encoded;
	PtrToArg<int64_t>::encode(p_y_sort_origin, &p_y_sort_origin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_y_sort_origin_encoded);
}

int32_t TileMapLayer::get_y_sort_origin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_y_sort_origin")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileMapLayer::set_x_draw_order_reversed(bool p_x_draw_order_reversed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_x_draw_order_reversed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_x_draw_order_reversed_encoded;
	PtrToArg<bool>::encode(p_x_draw_order_reversed, &p_x_draw_order_reversed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_x_draw_order_reversed_encoded);
}

bool TileMapLayer::is_x_draw_order_reversed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("is_x_draw_order_reversed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TileMapLayer::set_rendering_quadrant_size(int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_rendering_quadrant_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

int32_t TileMapLayer::get_rendering_quadrant_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_rendering_quadrant_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileMapLayer::set_collision_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_collision_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TileMapLayer::is_collision_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("is_collision_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TileMapLayer::set_use_kinematic_bodies(bool p_use_kinematic_bodies) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_use_kinematic_bodies")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_kinematic_bodies_encoded;
	PtrToArg<bool>::encode(p_use_kinematic_bodies, &p_use_kinematic_bodies_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_use_kinematic_bodies_encoded);
}

bool TileMapLayer::is_using_kinematic_bodies() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("is_using_kinematic_bodies")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TileMapLayer::set_collision_visibility_mode(TileMapLayer::DebugVisibilityMode p_visibility_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_collision_visibility_mode")._native_ptr(), 3508099847);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_visibility_mode_encoded;
	PtrToArg<int64_t>::encode(p_visibility_mode, &p_visibility_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visibility_mode_encoded);
}

TileMapLayer::DebugVisibilityMode TileMapLayer::get_collision_visibility_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_collision_visibility_mode")._native_ptr(), 338220793);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TileMapLayer::DebugVisibilityMode(0)));
	return (TileMapLayer::DebugVisibilityMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileMapLayer::set_physics_quadrant_size(int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_physics_quadrant_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

int32_t TileMapLayer::get_physics_quadrant_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_physics_quadrant_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileMapLayer::set_occlusion_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_occlusion_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TileMapLayer::is_occlusion_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("is_occlusion_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TileMapLayer::set_navigation_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_navigation_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TileMapLayer::is_navigation_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("is_navigation_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TileMapLayer::set_navigation_map(const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_navigation_map")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map);
}

RID TileMapLayer::get_navigation_map() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_navigation_map")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void TileMapLayer::set_navigation_visibility_mode(TileMapLayer::DebugVisibilityMode p_show_navigation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("set_navigation_visibility_mode")._native_ptr(), 3508099847);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_show_navigation_encoded;
	PtrToArg<int64_t>::encode(p_show_navigation, &p_show_navigation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_show_navigation_encoded);
}

TileMapLayer::DebugVisibilityMode TileMapLayer::get_navigation_visibility_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapLayer::get_class_static()._native_ptr(), StringName("get_navigation_visibility_mode")._native_ptr(), 338220793);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TileMapLayer::DebugVisibilityMode(0)));
	return (TileMapLayer::DebugVisibilityMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool TileMapLayer::_use_tile_data_runtime_update(const Vector2i &p_coords) {
	return false;
}

void TileMapLayer::_tile_data_runtime_update(const Vector2i &p_coords, TileData *p_tile_data) {}

void TileMapLayer::_update_cells(const TypedArray<Vector2i> &p_coords, bool p_forced_cleanup) {}

} // namespace godot
