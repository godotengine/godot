/**************************************************************************/
/*  tile_map.hpp                                                          */
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

#pragma once

#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/tile_set.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/rect2i.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class TileData;
class TileMapPattern;

class TileMap : public Node2D {
	GDEXTENSION_CLASS(TileMap, Node2D)

public:
	enum VisibilityMode {
		VISIBILITY_MODE_DEFAULT = 0,
		VISIBILITY_MODE_FORCE_HIDE = 2,
		VISIBILITY_MODE_FORCE_SHOW = 1,
	};

	void set_navigation_map(int32_t p_layer, const RID &p_map);
	RID get_navigation_map(int32_t p_layer) const;
	void force_update(int32_t p_layer = -1);
	void set_tileset(const Ref<TileSet> &p_tileset);
	Ref<TileSet> get_tileset() const;
	void set_rendering_quadrant_size(int32_t p_size);
	int32_t get_rendering_quadrant_size() const;
	int32_t get_layers_count() const;
	void add_layer(int32_t p_to_position);
	void move_layer(int32_t p_layer, int32_t p_to_position);
	void remove_layer(int32_t p_layer);
	void set_layer_name(int32_t p_layer, const String &p_name);
	String get_layer_name(int32_t p_layer) const;
	void set_layer_enabled(int32_t p_layer, bool p_enabled);
	bool is_layer_enabled(int32_t p_layer) const;
	void set_layer_modulate(int32_t p_layer, const Color &p_modulate);
	Color get_layer_modulate(int32_t p_layer) const;
	void set_layer_y_sort_enabled(int32_t p_layer, bool p_y_sort_enabled);
	bool is_layer_y_sort_enabled(int32_t p_layer) const;
	void set_layer_y_sort_origin(int32_t p_layer, int32_t p_y_sort_origin);
	int32_t get_layer_y_sort_origin(int32_t p_layer) const;
	void set_layer_z_index(int32_t p_layer, int32_t p_z_index);
	int32_t get_layer_z_index(int32_t p_layer) const;
	void set_layer_navigation_enabled(int32_t p_layer, bool p_enabled);
	bool is_layer_navigation_enabled(int32_t p_layer) const;
	void set_layer_navigation_map(int32_t p_layer, const RID &p_map);
	RID get_layer_navigation_map(int32_t p_layer) const;
	void set_collision_animatable(bool p_enabled);
	bool is_collision_animatable() const;
	void set_collision_visibility_mode(TileMap::VisibilityMode p_collision_visibility_mode);
	TileMap::VisibilityMode get_collision_visibility_mode() const;
	void set_navigation_visibility_mode(TileMap::VisibilityMode p_navigation_visibility_mode);
	TileMap::VisibilityMode get_navigation_visibility_mode() const;
	void set_cell(int32_t p_layer, const Vector2i &p_coords, int32_t p_source_id = -1, const Vector2i &p_atlas_coords = Vector2i(-1, -1), int32_t p_alternative_tile = 0);
	void erase_cell(int32_t p_layer, const Vector2i &p_coords);
	int32_t get_cell_source_id(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	Vector2i get_cell_atlas_coords(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	int32_t get_cell_alternative_tile(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	TileData *get_cell_tile_data(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	bool is_cell_flipped_h(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	bool is_cell_flipped_v(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	bool is_cell_transposed(int32_t p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	Vector2i get_coords_for_body_rid(const RID &p_body);
	int32_t get_layer_for_body_rid(const RID &p_body);
	Ref<TileMapPattern> get_pattern(int32_t p_layer, const TypedArray<Vector2i> &p_coords_array);
	Vector2i map_pattern(const Vector2i &p_position_in_tilemap, const Vector2i &p_coords_in_pattern, const Ref<TileMapPattern> &p_pattern);
	void set_pattern(int32_t p_layer, const Vector2i &p_position, const Ref<TileMapPattern> &p_pattern);
	void set_cells_terrain_connect(int32_t p_layer, const TypedArray<Vector2i> &p_cells, int32_t p_terrain_set, int32_t p_terrain, bool p_ignore_empty_terrains = true);
	void set_cells_terrain_path(int32_t p_layer, const TypedArray<Vector2i> &p_path, int32_t p_terrain_set, int32_t p_terrain, bool p_ignore_empty_terrains = true);
	void fix_invalid_tiles();
	void clear_layer(int32_t p_layer);
	void clear();
	void update_internals();
	void notify_runtime_tile_data_update(int32_t p_layer = -1);
	TypedArray<Vector2i> get_surrounding_cells(const Vector2i &p_coords);
	TypedArray<Vector2i> get_used_cells(int32_t p_layer) const;
	TypedArray<Vector2i> get_used_cells_by_id(int32_t p_layer, int32_t p_source_id = -1, const Vector2i &p_atlas_coords = Vector2i(-1, -1), int32_t p_alternative_tile = -1) const;
	Rect2i get_used_rect() const;
	Vector2 map_to_local(const Vector2i &p_map_position) const;
	Vector2i local_to_map(const Vector2 &p_local_position) const;
	Vector2i get_neighbor_cell(const Vector2i &p_coords, TileSet::CellNeighbor p_neighbor) const;
	virtual bool _use_tile_data_runtime_update(int32_t p_layer, const Vector2i &p_coords);
	virtual void _tile_data_runtime_update(int32_t p_layer, const Vector2i &p_coords, TileData *p_tile_data);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node2D::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_use_tile_data_runtime_update), decltype(&T::_use_tile_data_runtime_update)>) {
			BIND_VIRTUAL_METHOD(T, _use_tile_data_runtime_update, 3957903770);
		}
		if constexpr (!std::is_same_v<decltype(&B::_tile_data_runtime_update), decltype(&T::_tile_data_runtime_update)>) {
			BIND_VIRTUAL_METHOD(T, _tile_data_runtime_update, 4223434291);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TileMap::VisibilityMode);

