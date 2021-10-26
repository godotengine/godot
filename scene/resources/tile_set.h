/*************************************************************************/
/*  tile_set.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TILE_SET_H
#define TILE_SET_H

#include "core/io/resource.h"
#include "core/object/object.h"
#include "core/templates/local_vector.h"
#include "scene/2d/light_occluder_2d.h"
#include "scene/2d/navigation_region_2d.h"
#include "scene/main/canvas_item.h"
#include "scene/resources/concave_polygon_shape_2d.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/physics_material.h"
#include "scene/resources/shape_2d.h"

#ifndef DISABLE_DEPRECATED
#include "scene/resources/shader.h"
#include "scene/resources/texture.h"
#endif

class TileMap;
struct TileMapQuadrant;
class TileSetSource;
class TileSetAtlasSource;
class TileData;

// Forward-declare the plugins.
class TileSetPlugin;
class TileSetPluginAtlasRendering;
class TileSetPluginAtlasPhysics;
class TileSetPluginAtlasNavigation;

union TileMapCell {
	struct {
		int32_t source_id : 16;
		int16_t coord_x : 16;
		int16_t coord_y : 16;
		int32_t alternative_tile : 16;
	};

	uint64_t _u64t;
	TileMapCell(int p_source_id = -1, Vector2i p_atlas_coords = Vector2i(-1, -1), int p_alternative_tile = -1) { // default are INVALID_SOURCE, INVALID_ATLAS_COORDS, INVALID_TILE_ALTERNATIVE
		source_id = p_source_id;
		set_atlas_coords(p_atlas_coords);
		alternative_tile = p_alternative_tile;
	}

	Vector2i get_atlas_coords() const {
		return Vector2i(coord_x, coord_y);
	}

	void set_atlas_coords(const Vector2i &r_coords) {
		coord_x = r_coords.x;
		coord_y = r_coords.y;
	}

	bool operator<(const TileMapCell &p_other) const {
		if (source_id == p_other.source_id) {
			if (coord_x == p_other.coord_x) {
				if (coord_y == p_other.coord_y) {
					return alternative_tile < p_other.alternative_tile;
				} else {
					return coord_y < p_other.coord_y;
				}
			} else {
				return coord_x < p_other.coord_x;
			}
		} else {
			return source_id < p_other.source_id;
		}
	}

	bool operator!=(const TileMapCell &p_other) const {
		return !(source_id == p_other.source_id && coord_x == p_other.coord_x && coord_y == p_other.coord_y && alternative_tile == p_other.alternative_tile);
	}
};

class TileMapPattern : public Resource {
	GDCLASS(TileMapPattern, Resource);

	Vector2i size;
	Map<Vector2i, TileMapCell> pattern;

	void _set_tile_data(const Vector<int> &p_data);
	Vector<int> _get_tile_data() const;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	void set_cell(const Vector2i &p_coords, int p_source_id, const Vector2i p_atlas_coords, int p_alternative_tile = 0);
	bool has_cell(const Vector2i &p_coords) const;
	void remove_cell(const Vector2i &p_coords, bool p_update_size = true);
	int get_cell_source_id(const Vector2i &p_coords) const;
	Vector2i get_cell_atlas_coords(const Vector2i &p_coords) const;
	int get_cell_alternative_tile(const Vector2i &p_coords) const;

	TypedArray<Vector2i> get_used_cells() const;

	Vector2i get_size() const;
	void set_size(const Vector2i &p_size);
	bool is_empty() const;

	void clear();
};

class TileSet : public Resource {
	GDCLASS(TileSet, Resource);

#ifndef DISABLE_DEPRECATED
private:
	struct CompatibilityShapeData {
		Vector2i autotile_coords;
		bool one_way;
		float one_way_margin;
		Ref<Shape2D> shape;
		Transform2D transform;
	};

	struct CompatibilityTileData {
		String name;
		Ref<Texture2D> texture;
		Vector2 tex_offset;
		Ref<ShaderMaterial> material;
		Rect2 region;
		int tile_mode = 0;
		Color modulate = Color(1, 1, 1);

		// Atlas or autotiles data
		int autotile_bitmask_mode = 0;
		Vector2 autotile_icon_coordinate;
		Size2i autotile_tile_size = Size2i(16, 16);

		int autotile_spacing = 0;
		Map<Vector2i, int> autotile_bitmask_flags;
		Map<Vector2i, Ref<OccluderPolygon2D>> autotile_occluder_map;
		Map<Vector2i, Ref<NavigationPolygon>> autotile_navpoly_map;
		Map<Vector2i, int> autotile_priority_map;
		Map<Vector2i, int> autotile_z_index_map;

		Vector<CompatibilityShapeData> shapes;
		Ref<OccluderPolygon2D> occluder;
		Vector2 occluder_offset;
		Ref<NavigationPolygon> navigation;
		Vector2 navigation_offset;
		int z_index = 0;
	};

	enum CompatibilityTileMode {
		COMPATIBILITY_TILE_MODE_SINGLE_TILE = 0,
		COMPATIBILITY_TILE_MODE_AUTO_TILE,
		COMPATIBILITY_TILE_MODE_ATLAS_TILE,
	};

	Map<int, CompatibilityTileData *> compatibility_data;
	Map<int, int> compatibility_tilemap_mapping_tile_modes;
	Map<int, Map<Array, Array>> compatibility_tilemap_mapping;

	void _compatibility_conversion();

public:
	// Format of output array [source_id, atlas_coords, alternative]
	Array compatibility_tilemap_map(int p_tile_id, Vector2i p_coords, bool p_flip_h, bool p_flip_v, bool p_transpose);
#endif // DISABLE_DEPRECATED

public:
	static const int INVALID_SOURCE; // -1;

	enum CellNeighbor {
		CELL_NEIGHBOR_RIGHT_SIDE = 0,
		CELL_NEIGHBOR_RIGHT_CORNER,
		CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE,
		CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER,
		CELL_NEIGHBOR_BOTTOM_SIDE,
		CELL_NEIGHBOR_BOTTOM_CORNER,
		CELL_NEIGHBOR_BOTTOM_LEFT_SIDE,
		CELL_NEIGHBOR_BOTTOM_LEFT_CORNER,
		CELL_NEIGHBOR_LEFT_SIDE,
		CELL_NEIGHBOR_LEFT_CORNER,
		CELL_NEIGHBOR_TOP_LEFT_SIDE,
		CELL_NEIGHBOR_TOP_LEFT_CORNER,
		CELL_NEIGHBOR_TOP_SIDE,
		CELL_NEIGHBOR_TOP_CORNER,
		CELL_NEIGHBOR_TOP_RIGHT_SIDE,
		CELL_NEIGHBOR_TOP_RIGHT_CORNER,
		CELL_NEIGHBOR_MAX,
	};

	static const char *CELL_NEIGHBOR_ENUM_TO_TEXT[];

	enum TerrainMode {
		TERRAIN_MODE_MATCH_CORNERS_AND_SIDES = 0,
		TERRAIN_MODE_MATCH_CORNERS,
		TERRAIN_MODE_MATCH_SIDES,
	};

	enum TileShape {
		TILE_SHAPE_SQUARE,
		TILE_SHAPE_ISOMETRIC,
		TILE_SHAPE_HALF_OFFSET_SQUARE,
		TILE_SHAPE_HEXAGON,
	};

	enum TileLayout {
		TILE_LAYOUT_STACKED,
		TILE_LAYOUT_STACKED_OFFSET,
		TILE_LAYOUT_STAIRS_RIGHT,
		TILE_LAYOUT_STAIRS_DOWN,
		TILE_LAYOUT_DIAMOND_RIGHT,
		TILE_LAYOUT_DIAMOND_DOWN,
	};

	enum TileOffsetAxis {
		TILE_OFFSET_AXIS_HORIZONTAL,
		TILE_OFFSET_AXIS_VERTICAL,
	};

	struct PackedSceneSource {
		Ref<PackedScene> scene;
		Vector2 offset;
	};
	typedef Array TerrainsPattern;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	virtual void _validate_property(PropertyInfo &property) const override;

private:
	// --- TileSet data ---
	// Basic shape and layout.
	TileShape tile_shape = TILE_SHAPE_SQUARE;
	TileLayout tile_layout = TILE_LAYOUT_STACKED;
	TileOffsetAxis tile_offset_axis = TILE_OFFSET_AXIS_HORIZONTAL;
	Size2i tile_size = Size2i(16, 16); //Size2(64, 64);
	Vector2 tile_skew = Vector2(0, 0);

	// Rendering.
	bool uv_clipping = false;
	struct OcclusionLayer {
		uint32_t light_mask = 1;
		bool sdf_collision = false;
	};
	Vector<OcclusionLayer> occlusion_layers;

	Ref<ArrayMesh> tile_lines_mesh;
	Ref<ArrayMesh> tile_filled_mesh;
	bool tile_meshes_dirty = true;

	// Physics
	struct PhysicsLayer {
		uint32_t collision_layer = 1;
		uint32_t collision_mask = 1;
		Ref<PhysicsMaterial> physics_material;
	};
	Vector<PhysicsLayer> physics_layers;

	// Terrains
	struct Terrain {
		String name;
		Color color;
	};
	struct TerrainSet {
		TerrainMode mode = TERRAIN_MODE_MATCH_CORNERS_AND_SIDES;
		Vector<Terrain> terrains;
	};
	Vector<TerrainSet> terrain_sets;

	Map<TerrainMode, Map<CellNeighbor, Ref<ArrayMesh>>> terrain_bits_meshes;
	bool terrain_bits_meshes_dirty = true;

	LocalVector<Map<TileSet::TerrainsPattern, Set<TileMapCell>>> per_terrain_pattern_tiles; // Cached data.
	bool terrains_cache_dirty = true;
	void _update_terrains_cache();

	// Navigation
	struct NavigationLayer {
		uint32_t layers = 1;
	};
	Vector<NavigationLayer> navigation_layers;

	// CustomData
	struct CustomDataLayer {
		String name;
		Variant::Type type = Variant::NIL;
	};
	Vector<CustomDataLayer> custom_data_layers;
	Map<String, int> custom_data_layers_by_name;

	// Per Atlas source data.
	Map<int, Ref<TileSetSource>> sources;
	Vector<int> source_ids;
	int next_source_id = 0;
	// ---------------------

	LocalVector<Ref<TileMapPattern>> patterns;

	void _compute_next_source_id();
	void _source_changed();

	// Tile proxies
	Map<int, int> source_level_proxies;
	Map<Array, Array> coords_level_proxies;
	Map<Array, Array> alternative_level_proxies;

	// Helpers
	Vector<Point2> _get_square_corner_or_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit);
	Vector<Point2> _get_square_corner_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit);
	Vector<Point2> _get_square_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit);

	Vector<Point2> _get_isometric_corner_or_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit);
	Vector<Point2> _get_isometric_corner_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit);
	Vector<Point2> _get_isometric_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit);

	Vector<Point2> _get_half_offset_corner_or_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis);
	Vector<Point2> _get_half_offset_corner_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis);
	Vector<Point2> _get_half_offset_side_terrain_bit_polygon(Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis);

protected:
	static void _bind_methods();

public:
	// --- Plugins ---
	Vector<TileSetPlugin *> get_tile_set_atlas_plugins() const;

	// --- Accessors for TileSet data ---

	// -- Shape and layout --
	void set_tile_shape(TileShape p_shape);
	TileShape get_tile_shape() const;
	void set_tile_layout(TileLayout p_layout);
	TileLayout get_tile_layout() const;
	void set_tile_offset_axis(TileOffsetAxis p_alignment);
	TileOffsetAxis get_tile_offset_axis() const;
	void set_tile_size(Size2i p_size);
	Size2i get_tile_size() const;

	// -- Sources management --
	int get_next_source_id() const;
	int get_source_count() const;
	int get_source_id(int p_index) const;
	int add_source(Ref<TileSetSource> p_tile_set_source, int p_source_id_override = -1);
	void set_source_id(int p_source_id, int p_new_id);
	void remove_source(int p_source_id);
	bool has_source(int p_source_id) const;
	Ref<TileSetSource> get_source(int p_source_id) const;

	// Rendering
	void set_uv_clipping(bool p_uv_clipping);
	bool is_uv_clipping() const;

	int get_occlusion_layers_count() const;
	void add_occlusion_layer(int p_index = -1);
	void move_occlusion_layer(int p_from_index, int p_to_pos);
	void remove_occlusion_layer(int p_index);
	void set_occlusion_layer_light_mask(int p_layer_index, int p_light_mask);
	int get_occlusion_layer_light_mask(int p_layer_index) const;
	void set_occlusion_layer_sdf_collision(int p_layer_index, bool p_sdf_collision);
	bool get_occlusion_layer_sdf_collision(int p_layer_index) const;

	// Physics
	int get_physics_layers_count() const;
	void add_physics_layer(int p_index = -1);
	void move_physics_layer(int p_from_index, int p_to_pos);
	void remove_physics_layer(int p_index);
	void set_physics_layer_collision_layer(int p_layer_index, uint32_t p_layer);
	uint32_t get_physics_layer_collision_layer(int p_layer_index) const;
	void set_physics_layer_collision_mask(int p_layer_index, uint32_t p_mask);
	uint32_t get_physics_layer_collision_mask(int p_layer_index) const;
	void set_physics_layer_physics_material(int p_layer_index, Ref<PhysicsMaterial> p_physics_material);
	Ref<PhysicsMaterial> get_physics_layer_physics_material(int p_layer_index) const;

	// Terrain sets
	int get_terrain_sets_count() const;
	void add_terrain_set(int p_index = -1);
	void move_terrain_set(int p_from_index, int p_to_pos);
	void remove_terrain_set(int p_index);
	void set_terrain_set_mode(int p_terrain_set, TerrainMode p_terrain_mode);
	TerrainMode get_terrain_set_mode(int p_terrain_set) const;

	// Terrains
	int get_terrains_count(int p_terrain_set) const;
	void add_terrain(int p_terrain_set, int p_index = -1);
	void move_terrain(int p_terrain_set, int p_from_index, int p_to_pos);
	void remove_terrain(int p_terrain_set, int p_index);
	void set_terrain_name(int p_terrain_set, int p_terrain_index, String p_name);
	String get_terrain_name(int p_terrain_set, int p_terrain_index) const;
	void set_terrain_color(int p_terrain_set, int p_terrain_index, Color p_color);
	Color get_terrain_color(int p_terrain_set, int p_terrain_index) const;
	bool is_valid_peering_bit_for_mode(TileSet::TerrainMode p_terrain_mode, TileSet::CellNeighbor p_peering_bit) const;
	bool is_valid_peering_bit_terrain(int p_terrain_set, TileSet::CellNeighbor p_peering_bit) const;

	// Navigation
	int get_navigation_layers_count() const;
	void add_navigation_layer(int p_index = -1);
	void move_navigation_layer(int p_from_index, int p_to_pos);
	void remove_navigation_layer(int p_index);
	void set_navigation_layer_layers(int p_layer_index, uint32_t p_layers);
	uint32_t get_navigation_layer_layers(int p_layer_index) const;

	// Custom data
	int get_custom_data_layers_count() const;
	void add_custom_data_layer(int p_index = -1);
	void move_custom_data_layer(int p_from_index, int p_to_pos);
	void remove_custom_data_layer(int p_index);
	int get_custom_data_layer_by_name(String p_value) const;
	void set_custom_data_name(int p_layer_id, String p_value);
	String get_custom_data_name(int p_layer_id) const;
	void set_custom_data_type(int p_layer_id, Variant::Type p_value);
	Variant::Type get_custom_data_type(int p_layer_id) const;

	// Tiles proxies.
	void set_source_level_tile_proxy(int p_source_from, int p_source_to);
	int get_source_level_tile_proxy(int p_source_from);
	bool has_source_level_tile_proxy(int p_source_from);
	void remove_source_level_tile_proxy(int p_source_from);

	void set_coords_level_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_source_to, Vector2i p_coords_to);
	Array get_coords_level_tile_proxy(int p_source_from, Vector2i p_coords_from);
	bool has_coords_level_tile_proxy(int p_source_from, Vector2i p_coords_from);
	void remove_coords_level_tile_proxy(int p_source_from, Vector2i p_coords_from);

	void set_alternative_level_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_alternative_from, int p_source_to, Vector2i p_coords_to, int p_alternative_to);
	Array get_alternative_level_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_alternative_from);
	bool has_alternative_level_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_alternative_from);
	void remove_alternative_level_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_alternative_from);

	Array get_source_level_tile_proxies() const;
	Array get_coords_level_tile_proxies() const;
	Array get_alternative_level_tile_proxies() const;

	Array map_tile_proxy(int p_source_from, Vector2i p_coords_from, int p_alternative_from) const;

	void cleanup_invalid_tile_proxies();
	void clear_tile_proxies();

	// Patterns.
	int add_pattern(Ref<TileMapPattern> p_pattern, int p_index = -1);
	Ref<TileMapPattern> get_pattern(int p_index);
	void remove_pattern(int p_index);
	int get_patterns_count();

	// Terrains.
	Set<TerrainsPattern> get_terrains_pattern_set(int p_terrain_set);
	Set<TileMapCell> get_tiles_for_terrains_pattern(int p_terrain_set, TerrainsPattern p_terrain_tile_pattern);
	TileMapCell get_random_tile_from_pattern(int p_terrain_set, TerrainsPattern p_terrain_tile_pattern);

	// Helpers
	Vector<Vector2> get_tile_shape_polygon();
	void draw_tile_shape(CanvasItem *p_canvas_item, Transform2D p_transform, Color p_color, bool p_filled = false, Ref<Texture2D> p_texture = Ref<Texture2D>());

	Vector<Point2> get_terrain_bit_polygon(int p_terrain_set, TileSet::CellNeighbor p_bit);
	void draw_terrains(CanvasItem *p_canvas_item, Transform2D p_transform, const TileData *p_tile_data);
	Vector<Vector<Ref<Texture2D>>> generate_terrains_icons(Size2i p_size);

	// Resource management
	virtual void reset_state() override;

	TileSet();
	~TileSet();
};

class TileSetSource : public Resource {
	GDCLASS(TileSetSource, Resource);

protected:
	const TileSet *tile_set = nullptr;

	static void _bind_methods();

public:
	static const Vector2i INVALID_ATLAS_COORDS; // Vector2i(-1, -1);
	static const int INVALID_TILE_ALTERNATIVE; // -1;

	// Not exposed.
	virtual void set_tile_set(const TileSet *p_tile_set);
	virtual void notify_tile_data_properties_should_change(){};
	virtual void add_occlusion_layer(int p_index){};
	virtual void move_occlusion_layer(int p_from_index, int p_to_pos){};
	virtual void remove_occlusion_layer(int p_index){};
	virtual void add_physics_layer(int p_index){};
	virtual void move_physics_layer(int p_from_index, int p_to_pos){};
	virtual void remove_physics_layer(int p_index){};
	virtual void add_terrain_set(int p_index){};
	virtual void move_terrain_set(int p_from_index, int p_to_pos){};
	virtual void remove_terrain_set(int p_index){};
	virtual void add_terrain(int p_terrain_set, int p_index){};
	virtual void move_terrain(int p_terrain_set, int p_from_index, int p_to_pos){};
	virtual void remove_terrain(int p_terrain_set, int p_index){};
	virtual void add_navigation_layer(int p_index){};
	virtual void move_navigation_layer(int p_from_index, int p_to_pos){};
	virtual void remove_navigation_layer(int p_index){};
	virtual void add_custom_data_layer(int p_index){};
	virtual void move_custom_data_layer(int p_from_index, int p_to_pos){};
	virtual void remove_custom_data_layer(int p_index){};
	virtual void reset_state() override{};

	// Tiles.
	virtual int get_tiles_count() const = 0;
	virtual Vector2i get_tile_id(int tile_index) const = 0;
	virtual bool has_tile(Vector2i p_atlas_coords) const = 0;

	// Alternative tiles.
	virtual int get_alternative_tiles_count(const Vector2i p_atlas_coords) const = 0;
	virtual int get_alternative_tile_id(const Vector2i p_atlas_coords, int p_index) const = 0;
	virtual bool has_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_tile) const = 0;
};

class TileSetAtlasSource : public TileSetSource {
	GDCLASS(TileSetAtlasSource, TileSetSource);

private:
	struct TileAlternativesData {
		Vector2i size_in_atlas = Vector2i(1, 1);
		Vector2i texture_offset;

		// Animation
		int animation_columns = 0;
		Vector2i animation_separation;
		real_t animation_speed = 1.0;
		LocalVector<real_t> animation_frames_durations;

		// Alternatives
		Map<int, TileData *> alternatives;
		Vector<int> alternatives_ids;
		int next_alternative_id = 1;
	};

	Ref<Texture2D> texture;
	Vector2i margins;
	Vector2i separation;
	Size2i texture_region_size = Size2i(16, 16);

	Map<Vector2i, TileAlternativesData> tiles;
	Vector<Vector2i> tiles_ids;
	Map<Vector2i, Vector2i> _coords_mapping_cache; // Maps any coordinate to the including tile

	TileData *_get_atlas_tile_data(Vector2i p_atlas_coords, int p_alternative_tile);
	const TileData *_get_atlas_tile_data(Vector2i p_atlas_coords, int p_alternative_tile) const;

	void _compute_next_alternative_id(const Vector2i p_atlas_coords);

	void _clear_coords_mapping_cache(Vector2i p_atlas_coords);
	void _create_coords_mapping_cache(Vector2i p_atlas_coords);

	void _clear_tiles_outside_texture();

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	// Not exposed.
	virtual void set_tile_set(const TileSet *p_tile_set) override;
	virtual void notify_tile_data_properties_should_change() override;
	virtual void add_occlusion_layer(int p_index) override;
	virtual void move_occlusion_layer(int p_from_index, int p_to_pos) override;
	virtual void remove_occlusion_layer(int p_index) override;
	virtual void add_physics_layer(int p_index) override;
	virtual void move_physics_layer(int p_from_index, int p_to_pos) override;
	virtual void remove_physics_layer(int p_index) override;
	virtual void add_terrain_set(int p_index) override;
	virtual void move_terrain_set(int p_from_index, int p_to_pos) override;
	virtual void remove_terrain_set(int p_index) override;
	virtual void add_terrain(int p_terrain_set, int p_index) override;
	virtual void move_terrain(int p_terrain_set, int p_from_index, int p_to_pos) override;
	virtual void remove_terrain(int p_terrain_set, int p_index) override;
	virtual void add_navigation_layer(int p_index) override;
	virtual void move_navigation_layer(int p_from_index, int p_to_pos) override;
	virtual void remove_navigation_layer(int p_index) override;
	virtual void add_custom_data_layer(int p_index) override;
	virtual void move_custom_data_layer(int p_from_index, int p_to_pos) override;
	virtual void remove_custom_data_layer(int p_index) override;
	virtual void reset_state() override;

	// Base properties.
	void set_texture(Ref<Texture2D> p_texture);
	Ref<Texture2D> get_texture() const;
	void set_margins(Vector2i p_margins);
	Vector2i get_margins() const;
	void set_separation(Vector2i p_separation);
	Vector2i get_separation() const;
	void set_texture_region_size(Vector2i p_tile_size);
	Vector2i get_texture_region_size() const;

	// Base tiles.
	void create_tile(const Vector2i p_atlas_coords, const Vector2i p_size = Vector2i(1, 1));
	void remove_tile(Vector2i p_atlas_coords);
	virtual bool has_tile(Vector2i p_atlas_coords) const override;
	void move_tile_in_atlas(Vector2i p_atlas_coords, Vector2i p_new_atlas_coords = INVALID_ATLAS_COORDS, Vector2i p_new_size = Vector2i(-1, -1));
	Vector2i get_tile_size_in_atlas(Vector2i p_atlas_coords) const;

	virtual int get_tiles_count() const override;
	virtual Vector2i get_tile_id(int p_index) const override;

	bool has_room_for_tile(Vector2i p_atlas_coords, Vector2i p_size, int p_animation_columns, Vector2i p_animation_separation, int p_frames_count, Vector2i p_ignored_tile = INVALID_ATLAS_COORDS) const;
	PackedVector2Array get_tiles_to_be_removed_on_change(Ref<Texture2D> p_texture, Vector2i p_margins, Vector2i p_separation, Vector2i p_texture_region_size);
	Vector2i get_tile_at_coords(Vector2i p_atlas_coords) const;

	// Animation.
	void set_tile_animation_columns(const Vector2i p_atlas_coords, int p_frame_columns);
	int get_tile_animation_columns(const Vector2i p_atlas_coords) const;
	void set_tile_animation_separation(const Vector2i p_atlas_coords, const Vector2i p_separation);
	Vector2i get_tile_animation_separation(const Vector2i p_atlas_coords) const;
	void set_tile_animation_speed(const Vector2i p_atlas_coords, real_t p_speed);
	real_t get_tile_animation_speed(const Vector2i p_atlas_coords) const;
	void set_tile_animation_frames_count(const Vector2i p_atlas_coords, int p_frames_count);
	int get_tile_animation_frames_count(const Vector2i p_atlas_coords) const;
	void set_tile_animation_frame_duration(const Vector2i p_atlas_coords, int p_frame_index, real_t p_duration);
	real_t get_tile_animation_frame_duration(const Vector2i p_atlas_coords, int p_frame_index) const;
	real_t get_tile_animation_total_duration(const Vector2i p_atlas_coords) const;

	// Alternative tiles.
	int create_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_id_override = -1);
	void remove_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_tile);
	void set_alternative_tile_id(const Vector2i p_atlas_coords, int p_alternative_tile, int p_new_id);
	virtual bool has_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_tile) const override;
	int get_next_alternative_tile_id(const Vector2i p_atlas_coords) const;

	virtual int get_alternative_tiles_count(const Vector2i p_atlas_coords) const override;
	virtual int get_alternative_tile_id(const Vector2i p_atlas_coords, int p_index) const override;

	// Get data associated to a tile.
	Object *get_tile_data(const Vector2i p_atlas_coords, int p_alternative_tile) const;

	// Helpers.
	Vector2i get_atlas_grid_size() const;
	Rect2i get_tile_texture_region(Vector2i p_atlas_coords, int p_frame = 0) const;
	Vector2i get_tile_effective_texture_offset(Vector2i p_atlas_coords, int p_alternative_tile) const;

	~TileSetAtlasSource();
};

class TileSetScenesCollectionSource : public TileSetSource {
	GDCLASS(TileSetScenesCollectionSource, TileSetSource);

private:
	struct SceneData {
		Ref<PackedScene> scene;
		bool display_placeholder = false;
	};
	Vector<int> scenes_ids;
	Map<int, SceneData> scenes;
	int next_scene_id = 1;

	void _compute_next_alternative_id();

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	// Tiles.
	int get_tiles_count() const override;
	Vector2i get_tile_id(int p_tile_index) const override;
	bool has_tile(Vector2i p_atlas_coords) const override;

	// Alternative tiles.
	int get_alternative_tiles_count(const Vector2i p_atlas_coords) const override;
	int get_alternative_tile_id(const Vector2i p_atlas_coords, int p_index) const override;
	bool has_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_tile) const override;

	// Scenes accessors. Lot are similar to "Alternative tiles".
	int get_scene_tiles_count() { return get_alternative_tiles_count(Vector2i()); }
	int get_scene_tile_id(int p_index) { return get_alternative_tile_id(Vector2i(), p_index); };
	bool has_scene_tile_id(int p_id) { return has_alternative_tile(Vector2i(), p_id); };
	int create_scene_tile(Ref<PackedScene> p_packed_scene = Ref<PackedScene>(), int p_id_override = -1);
	void set_scene_tile_id(int p_id, int p_new_id);
	void set_scene_tile_scene(int p_id, Ref<PackedScene> p_packed_scene);
	Ref<PackedScene> get_scene_tile_scene(int p_id) const;
	void set_scene_tile_display_placeholder(int p_id, bool p_packed_scene);
	bool get_scene_tile_display_placeholder(int p_id) const;
	void remove_scene_tile(int p_id);
	int get_next_scene_tile_id() const;
};

class TileData : public Object {
	GDCLASS(TileData, Object);

private:
	const TileSet *tile_set = nullptr;
	bool allow_transform = true;

	// Rendering
	bool flip_h = false;
	bool flip_v = false;
	bool transpose = false;
	Vector2i tex_offset = Vector2i();
	Ref<ShaderMaterial> material = Ref<ShaderMaterial>();
	Color modulate = Color(1.0, 1.0, 1.0, 1.0);
	int z_index = 0;
	int y_sort_origin = 0;
	Vector<Ref<OccluderPolygon2D>> occluders;

	// Physics
	struct PhysicsLayerTileData {
		struct PolygonShapeTileData {
			LocalVector<Vector2> polygon;
			LocalVector<Ref<ConvexPolygonShape2D>> shapes;
			bool one_way = false;
			float one_way_margin = 1.0;
		};

		Vector2 linear_velocity;
		float angular_velocity = 0.0;
		Vector<PolygonShapeTileData> polygons;
	};
	Vector<PhysicsLayerTileData> physics;
	// TODO add support for areas.

	// Terrain
	int terrain_set = -1;
	int terrain_peering_bits[16] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

	// Navigation
	Vector<Ref<NavigationPolygon>> navigation;

	// Misc
	double probability = 1.0;

	// Custom data
	Vector<Variant> custom_data;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	static void _bind_methods();

public:
	// Not exposed.
	void set_tile_set(const TileSet *p_tile_set);
	void notify_tile_data_properties_should_change();
	void add_occlusion_layer(int p_index);
	void move_occlusion_layer(int p_from_index, int p_to_pos);
	void remove_occlusion_layer(int p_index);
	void add_physics_layer(int p_index);
	void move_physics_layer(int p_from_index, int p_to_pos);
	void remove_physics_layer(int p_index);
	void add_terrain_set(int p_index);
	void move_terrain_set(int p_from_index, int p_to_pos);
	void remove_terrain_set(int p_index);
	void add_terrain(int p_terrain_set, int p_index);
	void move_terrain(int p_terrain_set, int p_from_index, int p_to_pos);
	void remove_terrain(int p_terrain_set, int p_index);
	void add_navigation_layer(int p_index);
	void move_navigation_layer(int p_from_index, int p_to_pos);
	void remove_navigation_layer(int p_index);
	void add_custom_data_layer(int p_index);
	void move_custom_data_layer(int p_from_index, int p_to_pos);
	void remove_custom_data_layer(int p_index);
	void reset_state();
	void set_allow_transform(bool p_allow_transform);
	bool is_allowing_transform() const;

	// Rendering
	void set_flip_h(bool p_flip_h);
	bool get_flip_h() const;
	void set_flip_v(bool p_flip_v);
	bool get_flip_v() const;
	void set_transpose(bool p_transpose);
	bool get_transpose() const;

	void set_texture_offset(Vector2i p_texture_offset);
	Vector2i get_texture_offset() const;
	void set_material(Ref<ShaderMaterial> p_material);
	Ref<ShaderMaterial> get_material() const;
	void set_modulate(Color p_modulate);
	Color get_modulate() const;
	void set_z_index(int p_z_index);
	int get_z_index() const;
	void set_y_sort_origin(int p_y_sort_origin);
	int get_y_sort_origin() const;

	void set_occluder(int p_layer_id, Ref<OccluderPolygon2D> p_occluder_polygon);
	Ref<OccluderPolygon2D> get_occluder(int p_layer_id) const;

	// Physics
	void set_constant_linear_velocity(int p_layer_id, const Vector2 &p_velocity);
	Vector2 get_constant_linear_velocity(int p_layer_id) const;
	void set_constant_angular_velocity(int p_layer_id, real_t p_velocity);
	real_t get_constant_angular_velocity(int p_layer_id) const;
	void set_collision_polygons_count(int p_layer_id, int p_shapes_count);
	int get_collision_polygons_count(int p_layer_id) const;
	void add_collision_polygon(int p_layer_id);
	void remove_collision_polygon(int p_layer_id, int p_polygon_index);
	void set_collision_polygon_points(int p_layer_id, int p_polygon_index, Vector<Vector2> p_polygon);
	Vector<Vector2> get_collision_polygon_points(int p_layer_id, int p_polygon_index) const;
	void set_collision_polygon_one_way(int p_layer_id, int p_polygon_index, bool p_one_way);
	bool is_collision_polygon_one_way(int p_layer_id, int p_polygon_index) const;
	void set_collision_polygon_one_way_margin(int p_layer_id, int p_polygon_index, float p_one_way_margin);
	float get_collision_polygon_one_way_margin(int p_layer_id, int p_polygon_index) const;
	int get_collision_polygon_shapes_count(int p_layer_id, int p_polygon_index) const;
	Ref<ConvexPolygonShape2D> get_collision_polygon_shape(int p_layer_id, int p_polygon_index, int shape_index) const;

	// Terrain
	void set_terrain_set(int p_terrain_id);
	int get_terrain_set() const;
	void set_peering_bit_terrain(TileSet::CellNeighbor p_peering_bit, int p_terrain_id);
	int get_peering_bit_terrain(TileSet::CellNeighbor p_peering_bit) const;
	bool is_valid_peering_bit_terrain(TileSet::CellNeighbor p_peering_bit) const;

	TileSet::TerrainsPattern get_terrains_pattern() const; // Not exposed.

	// Navigation
	void set_navigation_polygon(int p_layer_id, Ref<NavigationPolygon> p_navigation_polygon);
	Ref<NavigationPolygon> get_navigation_polygon(int p_layer_id) const;

	// Misc
	void set_probability(float p_probability);
	float get_probability() const;

	// Custom data.
	void set_custom_data(String p_layer_name, Variant p_value);
	Variant get_custom_data(String p_layer_name) const;
	void set_custom_data_by_layer_id(int p_layer_id, Variant p_value);
	Variant get_custom_data_by_layer_id(int p_layer_id) const;
};

VARIANT_ENUM_CAST(TileSet::CellNeighbor);
VARIANT_ENUM_CAST(TileSet::TerrainMode);
VARIANT_ENUM_CAST(TileSet::TileShape);
VARIANT_ENUM_CAST(TileSet::TileLayout);
VARIANT_ENUM_CAST(TileSet::TileOffsetAxis);

#endif // TILE_SET_H
