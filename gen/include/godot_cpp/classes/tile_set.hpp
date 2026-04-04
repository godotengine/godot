/**************************************************************************/
/*  tile_set.hpp                                                          */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PhysicsMaterial;
class TileMapPattern;
class TileSetSource;

class TileSet : public Resource {
	GDEXTENSION_CLASS(TileSet, Resource)

public:
	enum TileShape {
		TILE_SHAPE_SQUARE = 0,
		TILE_SHAPE_ISOMETRIC = 1,
		TILE_SHAPE_HALF_OFFSET_SQUARE = 2,
		TILE_SHAPE_HEXAGON = 3,
	};

	enum TileLayout {
		TILE_LAYOUT_STACKED = 0,
		TILE_LAYOUT_STACKED_OFFSET = 1,
		TILE_LAYOUT_STAIRS_RIGHT = 2,
		TILE_LAYOUT_STAIRS_DOWN = 3,
		TILE_LAYOUT_DIAMOND_RIGHT = 4,
		TILE_LAYOUT_DIAMOND_DOWN = 5,
	};

	enum TileOffsetAxis {
		TILE_OFFSET_AXIS_HORIZONTAL = 0,
		TILE_OFFSET_AXIS_VERTICAL = 1,
	};

	enum CellNeighbor {
		CELL_NEIGHBOR_RIGHT_SIDE = 0,
		CELL_NEIGHBOR_RIGHT_CORNER = 1,
		CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE = 2,
		CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER = 3,
		CELL_NEIGHBOR_BOTTOM_SIDE = 4,
		CELL_NEIGHBOR_BOTTOM_CORNER = 5,
		CELL_NEIGHBOR_BOTTOM_LEFT_SIDE = 6,
		CELL_NEIGHBOR_BOTTOM_LEFT_CORNER = 7,
		CELL_NEIGHBOR_LEFT_SIDE = 8,
		CELL_NEIGHBOR_LEFT_CORNER = 9,
		CELL_NEIGHBOR_TOP_LEFT_SIDE = 10,
		CELL_NEIGHBOR_TOP_LEFT_CORNER = 11,
		CELL_NEIGHBOR_TOP_SIDE = 12,
		CELL_NEIGHBOR_TOP_CORNER = 13,
		CELL_NEIGHBOR_TOP_RIGHT_SIDE = 14,
		CELL_NEIGHBOR_TOP_RIGHT_CORNER = 15,
	};

	enum TerrainMode {
		TERRAIN_MODE_MATCH_CORNERS_AND_SIDES = 0,
		TERRAIN_MODE_MATCH_CORNERS = 1,
		TERRAIN_MODE_MATCH_SIDES = 2,
	};

	int32_t get_next_source_id() const;
	int32_t add_source(const Ref<TileSetSource> &p_source, int32_t p_atlas_source_id_override = -1);
	void remove_source(int32_t p_source_id);
	void set_source_id(int32_t p_source_id, int32_t p_new_source_id);
	int32_t get_source_count() const;
	int32_t get_source_id(int32_t p_index) const;
	bool has_source(int32_t p_source_id) const;
	Ref<TileSetSource> get_source(int32_t p_source_id) const;
	void set_tile_shape(TileSet::TileShape p_shape);
	TileSet::TileShape get_tile_shape() const;
	void set_tile_layout(TileSet::TileLayout p_layout);
	TileSet::TileLayout get_tile_layout() const;
	void set_tile_offset_axis(TileSet::TileOffsetAxis p_alignment);
	TileSet::TileOffsetAxis get_tile_offset_axis() const;
	void set_tile_size(const Vector2i &p_size);
	Vector2i get_tile_size() const;
	void set_uv_clipping(bool p_uv_clipping);
	bool is_uv_clipping() const;
	int32_t get_occlusion_layers_count() const;
	void add_occlusion_layer(int32_t p_to_position = -1);
	void move_occlusion_layer(int32_t p_layer_index, int32_t p_to_position);
	void remove_occlusion_layer(int32_t p_layer_index);
	void set_occlusion_layer_light_mask(int32_t p_layer_index, int32_t p_light_mask);
	int32_t get_occlusion_layer_light_mask(int32_t p_layer_index) const;
	void set_occlusion_layer_sdf_collision(int32_t p_layer_index, bool p_sdf_collision);
	bool get_occlusion_layer_sdf_collision(int32_t p_layer_index) const;
	int32_t get_physics_layers_count() const;
	void add_physics_layer(int32_t p_to_position = -1);
	void move_physics_layer(int32_t p_layer_index, int32_t p_to_position);
	void remove_physics_layer(int32_t p_layer_index);
	void set_physics_layer_collision_layer(int32_t p_layer_index, uint32_t p_layer);
	uint32_t get_physics_layer_collision_layer(int32_t p_layer_index) const;
	void set_physics_layer_collision_mask(int32_t p_layer_index, uint32_t p_mask);
	uint32_t get_physics_layer_collision_mask(int32_t p_layer_index) const;
	void set_physics_layer_collision_priority(int32_t p_layer_index, float p_priority);
	float get_physics_layer_collision_priority(int32_t p_layer_index) const;
	void set_physics_layer_physics_material(int32_t p_layer_index, const Ref<PhysicsMaterial> &p_physics_material);
	Ref<PhysicsMaterial> get_physics_layer_physics_material(int32_t p_layer_index) const;
	int32_t get_terrain_sets_count() const;
	void add_terrain_set(int32_t p_to_position = -1);
	void move_terrain_set(int32_t p_terrain_set, int32_t p_to_position);
	void remove_terrain_set(int32_t p_terrain_set);
	void set_terrain_set_mode(int32_t p_terrain_set, TileSet::TerrainMode p_mode);
	TileSet::TerrainMode get_terrain_set_mode(int32_t p_terrain_set) const;
	int32_t get_terrains_count(int32_t p_terrain_set) const;
	void add_terrain(int32_t p_terrain_set, int32_t p_to_position = -1);
	void move_terrain(int32_t p_terrain_set, int32_t p_terrain_index, int32_t p_to_position);
	void remove_terrain(int32_t p_terrain_set, int32_t p_terrain_index);
	void set_terrain_name(int32_t p_terrain_set, int32_t p_terrain_index, const String &p_name);
	String get_terrain_name(int32_t p_terrain_set, int32_t p_terrain_index) const;
	void set_terrain_color(int32_t p_terrain_set, int32_t p_terrain_index, const Color &p_color);
	Color get_terrain_color(int32_t p_terrain_set, int32_t p_terrain_index) const;
	int32_t get_navigation_layers_count() const;
	void add_navigation_layer(int32_t p_to_position = -1);
	void move_navigation_layer(int32_t p_layer_index, int32_t p_to_position);
	void remove_navigation_layer(int32_t p_layer_index);
	void set_navigation_layer_layers(int32_t p_layer_index, uint32_t p_layers);
	uint32_t get_navigation_layer_layers(int32_t p_layer_index) const;
	void set_navigation_layer_layer_value(int32_t p_layer_index, int32_t p_layer_number, bool p_value);
	bool get_navigation_layer_layer_value(int32_t p_layer_index, int32_t p_layer_number) const;
	int32_t get_custom_data_layers_count() const;
	void add_custom_data_layer(int32_t p_to_position = -1);
	void move_custom_data_layer(int32_t p_layer_index, int32_t p_to_position);
	void remove_custom_data_layer(int32_t p_layer_index);
	int32_t get_custom_data_layer_by_name(const String &p_layer_name) const;
	void set_custom_data_layer_name(int32_t p_layer_index, const String &p_layer_name);
	bool has_custom_data_layer_by_name(const String &p_layer_name) const;
	String get_custom_data_layer_name(int32_t p_layer_index) const;
	void set_custom_data_layer_type(int32_t p_layer_index, Variant::Type p_layer_type);
	Variant::Type get_custom_data_layer_type(int32_t p_layer_index) const;
	void set_source_level_tile_proxy(int32_t p_source_from, int32_t p_source_to);
	int32_t get_source_level_tile_proxy(int32_t p_source_from);
	bool has_source_level_tile_proxy(int32_t p_source_from);
	void remove_source_level_tile_proxy(int32_t p_source_from);
	void set_coords_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_source_to, const Vector2i &p_coords_to);
	Array get_coords_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from);
	bool has_coords_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from);
	void remove_coords_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from);
	void set_alternative_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_alternative_from, int32_t p_source_to, const Vector2i &p_coords_to, int32_t p_alternative_to);
	Array get_alternative_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_alternative_from);
	bool has_alternative_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_alternative_from);
	void remove_alternative_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_alternative_from);
	Array map_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_alternative_from) const;
	void cleanup_invalid_tile_proxies();
	void clear_tile_proxies();
	int32_t add_pattern(const Ref<TileMapPattern> &p_pattern, int32_t p_index = -1);
	Ref<TileMapPattern> get_pattern(int32_t p_index = -1);
	void remove_pattern(int32_t p_index);
	int32_t get_patterns_count();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TileSet::TileShape);
VARIANT_ENUM_CAST(TileSet::TileLayout);
VARIANT_ENUM_CAST(TileSet::TileOffsetAxis);
VARIANT_ENUM_CAST(TileSet::CellNeighbor);
VARIANT_ENUM_CAST(TileSet::TerrainMode);

