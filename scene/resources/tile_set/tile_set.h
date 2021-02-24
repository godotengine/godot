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

#include "scene/main/canvas_item.h"
#include "scene/resources/packed_scene.h"

#include "tile_set_atlas_plugin.h"
#include "tile_set_atlas_plugin_navigation.h"
#include "tile_set_atlas_plugin_physics.h"
#include "tile_set_atlas_plugin_rendering.h"
#include "tile_set_atlas_plugin_terrain.h"

class TileMap;

#ifndef DISABLE_DEPRECATED
#include "scene/2d/light_occluder_2d.h"
#include "scene/2d/navigation_region_2d.h"
#include "scene/resources/shader.h"
#include "scene/resources/shape_2d.h"
#include "scene/resources/texture.h"
#endif

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
	Vector2i y_sort_origin = Vector2i();
	Vector<Ref<OccluderPolygon2D>> occluders;

	//TerrainTileData terrains;
	//PhysicsTileData physics;
	//NavigationData navigation;

	// Misc
	double probability = 1.0;

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
	void tile_set_allow_transform(bool p_allow_transform);
	bool tile_is_allowing_transform() const;

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
	void set_y_sort_origin(Vector2i p_y_sort_origin);
	Vector2i get_y_sort_origin() const;

	void set_occluder(int p_layer_id, Ref<OccluderPolygon2D> p_occluder_polygon);
	Ref<OccluderPolygon2D> get_occluder(int p_layer_id) const;

	// Terrain
	/*
	void tile_get_terrain(int p_layer_id, flaot p_terrain) const;
	float tile_get_terrain( int p_layer_id) const;
	*/

	// Collision
	/*
	int tile_get_collision_shape_count(int p_layer_id) const;
	float tile_get_collision_shape(int p_layer_id, int p_shape_id) const;
	bool tile_get_collision_shape_one_way(int p_layer_id, int p_shape_id) const;
	float tile_get_collision_shape_one_way_margin(int p_layer_id, int p_shape_id) const;
	Transform2D tile_get_collision_shape_transform(int p_layer_id, int p_shape_id) const;
	*/

	// Navigation
	/*
	Ref<NavigationPolygon> tile_get_navigation(int p_layer_id) const;
	*/

	// Misc
	void tile_set_probability(float p_probability);
	float tile_get_probability() const;
};

class TileSetSource : public Resource {
	GDCLASS(TileSetSource, Resource);

protected:
	const TileSet *tile_set = nullptr;

public:
	// Not exposed.
	virtual void set_tile_set(const TileSet *p_tile_set);
	virtual void notify_tile_data_properties_should_change(){};
	virtual void reset_state() override{};

	// Tiles checks.
	virtual bool has_tile(Vector2i p_atlas_coords) const = 0;
	virtual bool has_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_tile) const = 0;
};

class TileSetAtlasSource : public TileSetSource {
	GDCLASS(TileSetAtlasSource, TileSetSource);

public:
	static const Vector2i INVALID_ATLAS_COORDS; // Vector2i(-1, -1);
	static const int INVALID_TILE_ALTERNATIVE; // -1;

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

	// TODO: maybe move to plugin structures ?
	Vector2i base_texture_offset;

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
	void set_base_texture_offset(Vector2i p_base_texture_offset);
	Vector2i get_base_texture_offset() const;

	// Base tiles.
	void create_tile(const Vector2i p_atlas_coords, const Vector2i p_size = Vector2i(1, 1)); // Create a tile if it does not exists, or add alternative tile if it does.
	void remove_tile(Vector2i p_atlas_coords); // Remove a tile. If p_tile_key.alternative_tile if different from 0, remove the alternative
	virtual bool has_tile(Vector2i p_atlas_coords) const override;
	bool can_move_tile_in_atlas(Vector2i p_atlas_coords, Vector2i p_new_atlas_coords = INVALID_ATLAS_COORDS, Vector2i p_new_size = Vector2i(-1, -1)) const;
	void move_tile_in_atlas(Vector2i p_atlas_coords, Vector2i p_new_atlas_coords = INVALID_ATLAS_COORDS, Vector2i p_new_size = Vector2i(-1, -1));
	Vector2i get_tile_size_in_atlas(Vector2i p_atlas_coords) const;

	int get_tiles_count() const;
	Vector2i get_tile_id(int p_index) const;

	Vector2i get_tile_at_coords(Vector2i p_atlas_coords) const;

	// Alternative tiles.
	int create_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_id_override = -1);
	void remove_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_tile);
	void set_alternative_tile_id(const Vector2i p_atlas_coords, int p_alternative_tile, int p_new_id);
	virtual bool has_alternative_tile(const Vector2i p_atlas_coords, int p_alternative_tile) const override;
	int get_next_alternative_tile_id(const Vector2i p_atlas_coords) const;

	int get_alternative_tiles_count(const Vector2i p_atlas_coords) const;
	int get_alternative_tile_id(const Vector2i p_atlas_coords, int p_index) const;

	// Get data associated to a tile.
	TileData *get_tile_data(const Vector2i p_atlas_coords, int p_alternative_tile) const;

	// Helpers.
	Vector2i get_atlas_grid_size() const;
	bool has_tiles_outside_texture();
	void clear_tiles_outside_texture();
	Rect2i get_tile_texture_region(Vector2i p_atlas_coords) const;

	~TileSetAtlasSource();
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
		int tile_mode;
		Color modulate;

		// Atlas or autotiles data
		int autotile_bitmask_mode;
		Vector2 autotile_icon_coordinate;
		Size2i autotile_tile_size = Size2i(16, 16);
		;
		int autotile_spacing;
		Map<Vector2i, int> autotile_bitmask_flags;
		Map<Vector2i, Ref<OccluderPolygon2D>> autotile_occluder_map;
		Map<Vector2i, Ref<NavigationPolygon>> autotile_navpoly_map;
		Map<Vector2i, int> autotile_priority_map;
		Map<Vector2i, int> autotile_z_index_map;

		Vector<CompatibilityShapeData *> shapes = Vector<CompatibilityShapeData *>();
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
		int light_mask = 0;
		bool sdf_collision = false;
	};
	Vector<OcclusionLayer> occlusion_layers;

	//TerrainsTileSetData terrains;
	//PhysicsTileSetData physics;
	//NavigationTileSetData navigation;

	// Per Atlas source data.
	Map<int, Ref<TileSetSource>> sources;
	Vector<int> source_ids;
	int next_source_id = 0;
	// ---------------------

	// Plugins themselves.
	TileSetAtlasPluginRendering tile_set_plugin_rendering;
	//TileSetAtlasPluginTerrain tile_set_plugin_terrain;
	//TileSetAtlasPluginPhysics tile_set_plugin_physics;
	//TileSetAtlasPluginNavigation tile_set_plugin_navigation;

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
	int add_source(Ref<TileSetAtlasSource> p_tile_atlas_source, int p_source_id_override = -1);
	void set_source_id(int p_source_id, int p_new_id);
	void remove_source(int p_source_id);
	bool has_source(int p_source_id) const;
	Ref<TileSetSource> get_source(int p_source_id) const;

	/*
	// -- Plugins data accessors --
	// Terrain
	uint32_t get_terrain_bitmask_mode(int p_layer_id) const;
	int get_terrain_types_count(int p_layer_id) const;
	String get_terrain_type_name(int p_layer_id, int p_terrain_type) const;
	Ref<Texture2D> get_terrain_type_icon(int p_layer_id, int p_terrain_type) const;
	Rect2i get_terrain_type_icon_region(int p_layer_id, int p_terrain_type) const;
	*/

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

	/*
	// Physics
	int get_physics_layers_count();
	uint32_t get_physics_layer_collision_layer(int p_collision_layer) const;
	uint32_t get_physics_layer_collision_mask(int p_collision_layer) const;
	bool get_physics_layer_use_kinematic(int p_collision_layer) const;
	float get_physics_layer_friction(int p_collision_layer) const;
	float get_physics_layer_bounce(int p_collision_layer) const;

	// Navigation
	// Nothing for now
	*/

	// Helpers
	void draw_tile_shape(CanvasItem *p_canvas_item, Rect2 p_region, Color p_color, bool p_filled = false, Ref<Texture2D> p_texture = Ref<Texture2D>());
	Vector2i get_tile_effective_texture_offset(int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile) const;

	virtual void reset_state();

	TileSet();
	~TileSet();

public:
	static void _append_property_list_with_prefix(const StringName &p_name, List<PropertyInfo> *p_to_prepend, List<PropertyInfo> *p_list);
};
VARIANT_ENUM_CAST(TileSet::TileShape);
VARIANT_ENUM_CAST(TileSet::TileLayout);
VARIANT_ENUM_CAST(TileSet::TileOffsetAxis);

#endif // TILE_SET_H
