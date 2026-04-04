/**************************************************************************/
/*  tile_map_layer.hpp                                                    */
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
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/rect2i.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class TileData;
class TileMapPattern;

class TileMapLayer : public Node2D {
	GDEXTENSION_CLASS(TileMapLayer, Node2D)

public:
	enum DebugVisibilityMode {
		DEBUG_VISIBILITY_MODE_DEFAULT = 0,
		DEBUG_VISIBILITY_MODE_FORCE_HIDE = 2,
		DEBUG_VISIBILITY_MODE_FORCE_SHOW = 1,
	};

	void set_cell(const Vector2i &p_coords, int32_t p_source_id = -1, const Vector2i &p_atlas_coords = Vector2i(-1, -1), int32_t p_alternative_tile = 0);
	void erase_cell(const Vector2i &p_coords);
	void fix_invalid_tiles();
	void clear();
	int32_t get_cell_source_id(const Vector2i &p_coords) const;
	Vector2i get_cell_atlas_coords(const Vector2i &p_coords) const;
	int32_t get_cell_alternative_tile(const Vector2i &p_coords) const;
	TileData *get_cell_tile_data(const Vector2i &p_coords) const;
	bool is_cell_flipped_h(const Vector2i &p_coords) const;
	bool is_cell_flipped_v(const Vector2i &p_coords) const;
	bool is_cell_transposed(const Vector2i &p_coords) const;
	TypedArray<Vector2i> get_used_cells() const;
	TypedArray<Vector2i> get_used_cells_by_id(int32_t p_source_id = -1, const Vector2i &p_atlas_coords = Vector2i(-1, -1), int32_t p_alternative_tile = -1) const;
	Rect2i get_used_rect() const;
	Ref<TileMapPattern> get_pattern(const TypedArray<Vector2i> &p_coords_array);
	void set_pattern(const Vector2i &p_position, const Ref<TileMapPattern> &p_pattern);
	void set_cells_terrain_connect(const TypedArray<Vector2i> &p_cells, int32_t p_terrain_set, int32_t p_terrain, bool p_ignore_empty_terrains = true);
	void set_cells_terrain_path(const TypedArray<Vector2i> &p_path, int32_t p_terrain_set, int32_t p_terrain, bool p_ignore_empty_terrains = true);
	bool has_body_rid(const RID &p_body) const;
	Vector2i get_coords_for_body_rid(const RID &p_body) const;
	void update_internals();
	void notify_runtime_tile_data_update();
	Vector2i map_pattern(const Vector2i &p_position_in_tilemap, const Vector2i &p_coords_in_pattern, const Ref<TileMapPattern> &p_pattern);
	TypedArray<Vector2i> get_surrounding_cells(const Vector2i &p_coords);
	Vector2i get_neighbor_cell(const Vector2i &p_coords, TileSet::CellNeighbor p_neighbor) const;
	Vector2 map_to_local(const Vector2i &p_map_position) const;
	Vector2i local_to_map(const Vector2 &p_local_position) const;
	void set_tile_map_data_from_array(const PackedByteArray &p_tile_map_layer_data);
	PackedByteArray get_tile_map_data_as_array() const;
	void set_enabled(bool p_enabled);
	bool is_enabled() const;
	void set_tile_set(const Ref<TileSet> &p_tile_set);
	Ref<TileSet> get_tile_set() const;
	void set_y_sort_origin(int32_t p_y_sort_origin);
	int32_t get_y_sort_origin() const;
	void set_x_draw_order_reversed(bool p_x_draw_order_reversed);
	bool is_x_draw_order_reversed() const;
	void set_rendering_quadrant_size(int32_t p_size);
	int32_t get_rendering_quadrant_size() const;
	void set_collision_enabled(bool p_enabled);
	bool is_collision_enabled() const;
	void set_use_kinematic_bodies(bool p_use_kinematic_bodies);
	bool is_using_kinematic_bodies() const;
	void set_collision_visibility_mode(TileMapLayer::DebugVisibilityMode p_visibility_mode);
	TileMapLayer::DebugVisibilityMode get_collision_visibility_mode() const;
	void set_physics_quadrant_size(int32_t p_size);
	int32_t get_physics_quadrant_size() const;
	void set_occlusion_enabled(bool p_enabled);
	bool is_occlusion_enabled() const;
	void set_navigation_enabled(bool p_enabled);
	bool is_navigation_enabled() const;
	void set_navigation_map(const RID &p_map);
	RID get_navigation_map() const;
	void set_navigation_visibility_mode(TileMapLayer::DebugVisibilityMode p_show_navigation);
	TileMapLayer::DebugVisibilityMode get_navigation_visibility_mode() const;
	virtual bool _use_tile_data_runtime_update(const Vector2i &p_coords);
	virtual void _tile_data_runtime_update(const Vector2i &p_coords, TileData *p_tile_data);
	virtual void _update_cells(const TypedArray<Vector2i> &p_coords, bool p_forced_cleanup);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node2D::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_use_tile_data_runtime_update), decltype(&T::_use_tile_data_runtime_update)>) {
			BIND_VIRTUAL_METHOD(T, _use_tile_data_runtime_update, 3715736492);
		}
		if constexpr (!std::is_same_v<decltype(&B::_tile_data_runtime_update), decltype(&T::_tile_data_runtime_update)>) {
			BIND_VIRTUAL_METHOD(T, _tile_data_runtime_update, 1627322126);
		}
		if constexpr (!std::is_same_v<decltype(&B::_update_cells), decltype(&T::_update_cells)>) {
			BIND_VIRTUAL_METHOD(T, _update_cells, 3156113851);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TileMapLayer::DebugVisibilityMode);

