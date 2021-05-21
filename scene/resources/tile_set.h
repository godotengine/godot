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
#include "scene/2d/light_occluder_2d.h"
#include "scene/2d/navigation_region_2d.h"
#include "scene/main/canvas_item.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/physics_material.h"
#include "scene/resources/shape_2d.h"

#ifndef DISABLE_DEPRECATED
#include "scene/2d/light_occluder_2d.h"
#include "scene/2d/navigation_region_2d.h"
#include "scene/resources/shader.h"
#include "scene/resources/shape_2d.h"
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
class TileSetPluginAtlasTerrain;

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
		int tile_mode;
		Color modulate;

		// Atlas or autotiles data
		int autotile_bitmask_mode;
		Vector2 autotile_icon_coordinate;
		Size2i autotile_tile_size = Size2i(16, 16);

		int autotile_spacing;
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
		int z_index;
	};

	Map<int, CompatibilityTileData *> compatibility_data = Map<int, CompatibilityTileData *>();
	Map<int, int> compatibility_source_mapping = Map<int, int>();

private:
	void compatibility_conversion();

public:
	int compatibility_get_source_for_tile_id(int p_old_source) {
		return compatibility_source_mapping[p_old_source];
	};

#endif // DISABLE_DEPRECATED

public:
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

public:
	struct PackedSceneSource {
		Ref<PackedScene> scene;
		Vector2 offset;
	};

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

private:
	// --- TileSet data ---
	// Basic shape and layout.
	TileShape tile_shape = TILE_SHAPE_SQUARE;
	TileLayout tile_layout = TILE_LAYOUT_STACKED;
	TileOffsetAxis tile_offset_axis = TILE_OFFSET_AXIS_HORIZONTAL;
	Size2i tile_size = Size2i(16, 16); //Size2(64, 64);
	Vector2 tile_skew = Vector2(0, 0);

	// Rendering.
	bool y_sorting = false;
	bool uv_clipping = false;
	struct OcclusionLayer {
		uint32_t light_mask = 1;
		bool sdf_collision = false;
	};
	Vector<OcclusionLayer> occlusion_layers;

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

	// Navigation
	struct Navigationlayer {
		uint32_t layers = 1;
	};
	Vector<Navigationlayer> navigation_layers;

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

	// Plugins themselves.
	Vector<TileSetPlugin *> tile_set_plugins_vector;

	void _compute_next_source_id();
	void _source_changed();

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
	void set_tile_skew(Vector2 p_skew);
	Vector2 get_tile_skew() const;

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
	void set_y_sorting(bool p_y_sort);
	bool is_y_sorting() const;

	void set_uv_clipping(bool p_uv_clipping);
	bool is_uv_clipping() const;

	void set_occlusion_layers_count(int p_occlusion_layers_count);
	int get_occlusion_layers_count() const;
	void set_occlusion_layer_light_mask(int p_layer_index, int p_light_mask);
	int get_occlusion_layer_light_mask(int p_layer_index) const;
	void set_occlusion_layer_sdf_collision(int p_layer_index, int p_sdf_collision);
	bool get_occlusion_layer_sdf_collision(int p_layer_index) const;

	// Physics
	void set_physics_layers_count(int p_physics_layers_count);
	int get_physics_layers_count() const;
	void set_physics_layer_collision_layer(int p_layer_index, uint32_t p_layer);
	uint32_t get_physics_layer_collision_layer(int p_layer_index) const;
	void set_physics_layer_collision_mask(int p_layer_index, uint32_t p_mask);
	uint32_t get_physics_layer_collision_mask(int p_layer_index) const;
	void set_physics_layer_physics_material(int p_layer_index, Ref<PhysicsMaterial> p_physics_material);
	Ref<PhysicsMaterial> get_physics_layer_physics_material(int p_layer_index) const;

	// Terrains
	void set_terrain_sets_count(int p_terrains_sets_count);
	int get_terrain_sets_count() const;
	void set_terrain_set_mode(int p_terrain_set, TerrainMode p_terrain_mode);
	TerrainMode get_terrain_set_mode(int p_terrain_set) const;
	void set_terrains_count(int p_terrain_set, int p_terrains_count);
	int get_terrains_count(int p_terrain_set) const;
	void set_terrain_name(int p_terrain_set, int p_terrain_index, String p_name);
	String get_terrain_name(int p_terrain_set, int p_terrain_index) const;
	void set_terrain_color(int p_terrain_set, int p_terrain_index, Color p_color);
	Color get_terrain_color(int p_terrain_set, int p_terrain_index) const;
	bool is_valid_peering_bit_terrain(int p_terrain_set, TileSet::CellNeighbor p_peering_bit) const;

	// Navigation
	void set_navigation_layers_count(int p_navigation_layers_count);
	int get_navigation_layers_count() const;
	void set_navigation_layer_layers(int p_layer_index, uint32_t p_layers);
	uint32_t get_navigation_layer_layers(int p_layer_index) const;

	// Custom data
	void set_custom_data_layers_count(int p_custom_data_layers_count);
	int get_custom_data_layers_count() const;
	int get_custom_data_layer_by_name(String p_value) const;
	void set_custom_data_name(int p_layer_id, String p_value);
	String get_custom_data_name(int p_layer_id) const;
	void set_custom_data_type(int p_layer_id, Variant::Type p_value);
	Variant::Type get_custom_data_type(int p_layer_id) const;

	// Helpers
	void draw_tile_shape(CanvasItem *p_canvas_item, Rect2 p_region, Color p_color, bool p_filled = false, Ref<Texture2D> p_texture = Ref<Texture2D>());

	virtual void reset_state() override;

	TileSet();
	~TileSet();
};

class TileSetSource : public Resource {
	GDCLASS(TileSetSource, Resource);

protected:
	const TileSet *tile_set = nullptr;

public:
	static const Vector2i INVALID_ATLAS_COORDS; // Vector2i(-1, -1);
	static const int INVALID_TILE_ALTERNATIVE; // -1;

	// Not exposed.
	virtual void set_tile_set(const TileSet *p_tile_set);
	virtual void notify_tile_data_properties_should_change(){};
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

public:
	struct TileAlternativesData {
		Vector2i size_in_atlas = Vector2i(1, 1);
		Vector2i texture_offset;
		Map<int, TileData *> alternatives;
		Vector<int> alternatives_ids;
		int next_alternative_id = 1;
	};

private:
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

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	// Not exposed.
	virtual void set_tile_set(const TileSet *p_tile_set) override;
	virtual void notify_tile_data_properties_should_change() override;
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
	void create_tile(const Vector2i p_atlas_coords, const Vector2i p_size = Vector2i(1, 1)); // Create a tile if it does not exists, or add alternative tile if it does.
	void remove_tile(Vector2i p_atlas_coords); // Remove a tile. If p_tile_key.alternative_tile if different from 0, remove the alternative
	virtual bool has_tile(Vector2i p_atlas_coords) const override;
	bool can_move_tile_in_atlas(Vector2i p_atlas_coords, Vector2i p_new_atlas_coords = INVALID_ATLAS_COORDS, Vector2i p_new_size = Vector2i(-1, -1)) const;
	void move_tile_in_atlas(Vector2i p_atlas_coords, Vector2i p_new_atlas_coords = INVALID_ATLAS_COORDS, Vector2i p_new_size = Vector2i(-1, -1));
	Vector2i get_tile_size_in_atlas(Vector2i p_atlas_coords) const;

	virtual int get_tiles_count() const override;
	virtual Vector2i get_tile_id(int p_index) const override;

	Vector2i get_tile_at_coords(Vector2i p_atlas_coords) const;

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
	bool has_tiles_outside_texture();
	void clear_tiles_outside_texture();
	Rect2i get_tile_texture_region(Vector2i p_atlas_coords) const;
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

	// Scenes sccessors. Lot are similar to "Alternative tiles".
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
		struct ShapeTileData {
			Ref<Shape2D> shape = Ref<Shape2D>();
			bool one_way = false;
			float one_way_margin = 1.0;
		};

		Vector<ShapeTileData> shapes;
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
	void tile_set_material(Ref<ShaderMaterial> p_material);
	Ref<ShaderMaterial> tile_get_material() const;
	void set_modulate(Color p_modulate);
	Color get_modulate() const;
	void set_z_index(int p_z_index);
	int get_z_index() const;
	void set_y_sort_origin(int p_y_sort_origin);
	int get_y_sort_origin() const;

	void set_occluder(int p_layer_id, Ref<OccluderPolygon2D> p_occluder_polygon);
	Ref<OccluderPolygon2D> get_occluder(int p_layer_id) const;

	// Physics
	int get_collision_shapes_count(int p_layer_id) const;
	void set_collision_shapes_count(int p_layer_id, int p_shapes_count);
	void add_collision_shape(int p_layer_id);
	void remove_collision_shape(int p_layer_id, int p_shape_index);
	void set_collision_shape_shape(int p_layer_id, int p_shape_index, Ref<Shape2D> p_shape);
	Ref<Shape2D> get_collision_shape_shape(int p_layer_id, int p_shape_index) const;
	void set_collision_shape_one_way(int p_layer_id, int p_shape_index, bool p_one_way);
	bool is_collision_shape_one_way(int p_layer_id, int p_shape_index) const;
	void set_collision_shape_one_way_margin(int p_layer_id, int p_shape_index, float p_one_way_margin);
	float get_collision_shape_one_way_margin(int p_layer_id, int p_shape_index) const;

	// Terrain
	void set_terrain_set(int p_terrain_id);
	int get_terrain_set() const;
	void set_peering_bit_terrain(TileSet::CellNeighbor p_peering_bit, int p_terrain_id);
	int get_peering_bit_terrain(TileSet::CellNeighbor p_peering_bit) const;
	bool is_valid_peering_bit_terrain(TileSet::CellNeighbor p_peering_bit) const;

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

#include "scene/2d/tile_map.h"

class TileSetPlugin : public Object {
	GDCLASS(TileSetPlugin, Object);

public:
	// Tilemap updates.
	virtual void tilemap_notification(TileMap *p_tile_map, int p_what){};
	virtual void update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list){};
	virtual void create_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant){};
	virtual void cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant){};

	virtual void draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant){};
};

class TileSetPluginAtlasRendering : public TileSetPlugin {
	GDCLASS(TileSetPluginAtlasRendering, TileSetPlugin);

private:
	static constexpr float fp_adjust = 0.00001;
	bool quadrant_order_dirty = false;

public:
	// Tilemap updates
	virtual void tilemap_notification(TileMap *p_tile_map, int p_what) override;
	virtual void update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) override;
	virtual void create_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
	virtual void cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
	virtual void draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;

	// Other.
	static void draw_tile(RID p_canvas_item, Vector2i p_position, const Ref<TileSet> p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, Color p_modulation = Color(1.0, 1.0, 1.0, 1.0));
};

class TileSetPluginAtlasTerrain : public TileSetPlugin {
	GDCLASS(TileSetPluginAtlasTerrain, TileSetPlugin);

private:
	static void _draw_square_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);
	static void _draw_square_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);
	static void _draw_square_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);

	static void _draw_isometric_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);
	static void _draw_isometric_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);
	static void _draw_isometric_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);

	static void _draw_half_offset_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis);
	static void _draw_half_offset_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis);
	static void _draw_half_offset_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis);

public:
	static void draw_terrains(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, const TileData *p_tile_data);
};

class TileSetPluginAtlasPhysics : public TileSetPlugin {
	GDCLASS(TileSetPluginAtlasPhysics, TileSetPlugin);

public:
	// Tilemap updates
	virtual void tilemap_notification(TileMap *p_tile_map, int p_what) override;
	virtual void update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) override;
	virtual void create_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
	virtual void cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
	virtual void draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
};

class TileSetPluginAtlasNavigation : public TileSetPlugin {
	GDCLASS(TileSetPluginAtlasNavigation, TileSetPlugin);

public:
	// Tilemap updates
	virtual void tilemap_notification(TileMap *p_tile_map, int p_what) override;
	virtual void update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) override;
	virtual void cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
	virtual void draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
};

class TileSetPluginScenesCollections : public TileSetPlugin {
	GDCLASS(TileSetPluginScenesCollections, TileSetPlugin);

public:
	// Tilemap updates
	virtual void update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) override;
	virtual void cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
	virtual void draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
};

VARIANT_ENUM_CAST(TileSet::CellNeighbor);
VARIANT_ENUM_CAST(TileSet::TerrainMode);
VARIANT_ENUM_CAST(TileSet::TileShape);
VARIANT_ENUM_CAST(TileSet::TileLayout);
VARIANT_ENUM_CAST(TileSet::TileOffsetAxis);

#endif // TILE_SET_H
