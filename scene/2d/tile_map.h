/*************************************************************************/
/*  tile_map.h                                                           */
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

#ifndef TILE_MAP_H
#define TILE_MAP_H

#include "scene/2d/node_2d.h"
#include "scene/gui/control.h"
#include "scene/resources/tile_set.h"

class TileSetAtlasSource;

struct TileMapQuadrant {
	struct CoordsWorldComparator {
		_ALWAYS_INLINE_ bool operator()(const Vector2i &p_a, const Vector2i &p_b) const {
			// We sort the cells by their world coords, as it is needed by rendering.
			if (p_a.y == p_b.y) {
				return p_a.x > p_b.x;
			} else {
				return p_a.y < p_b.y;
			}
		}
	};

	// Dirty list element
	SelfList<TileMapQuadrant> dirty_list_element;

	// Quadrant layer and coords.
	int layer = -1;
	Vector2i coords;

	// TileMapCells
	Set<Vector2i> cells;
	// We need those two maps to sort by world position for rendering
	// This is kind of workaround, it would be better to sort the cells directly in the "cells" set instead.
	Map<Vector2i, Vector2i> map_to_world;
	Map<Vector2i, Vector2i, CoordsWorldComparator> world_to_map;

	// Debug.
	RID debug_canvas_item;

	// Rendering.
	List<RID> canvas_items;
	List<RID> occluders;

	// Physics.
	List<RID> bodies;

	// Navigation.
	Map<Vector2i, Vector<RID>> navigation_regions;

	// Scenes.
	Map<Vector2i, String> scenes;

	// Runtime TileData cache.
	Map<Vector2i, TileData *> runtime_tile_data_cache;

	void operator=(const TileMapQuadrant &q) {
		layer = q.layer;
		coords = q.coords;
		debug_canvas_item = q.debug_canvas_item;
		canvas_items = q.canvas_items;
		occluders = q.occluders;
		bodies = q.bodies;
		navigation_regions = q.navigation_regions;
	}

	TileMapQuadrant(const TileMapQuadrant &q) :
			dirty_list_element(this) {
		layer = q.layer;
		coords = q.coords;
		debug_canvas_item = q.debug_canvas_item;
		canvas_items = q.canvas_items;
		occluders = q.occluders;
		bodies = q.bodies;
		navigation_regions = q.navigation_regions;
	}

	TileMapQuadrant() :
			dirty_list_element(this) {
	}
};

class TileMap : public Node2D {
	GDCLASS(TileMap, Node2D);

public:
	class TerrainConstraint {
	private:
		const TileMap *tile_map;
		Vector2i base_cell_coords = Vector2i();
		int bit = -1;
		int terrain = -1;

	public:
		bool operator<(const TerrainConstraint &p_other) const {
			if (base_cell_coords == p_other.base_cell_coords) {
				return bit < p_other.bit;
			}
			return base_cell_coords < p_other.base_cell_coords;
		}

		String to_string() const {
			return vformat("Constraint {pos:%s, bit:%d, terrain:%d}", base_cell_coords, bit, terrain);
		}

		Vector2i get_base_cell_coords() const {
			return base_cell_coords;
		}

		Map<Vector2i, TileSet::CellNeighbor> get_overlapping_coords_and_peering_bits() const;

		void set_terrain(int p_terrain) {
			terrain = p_terrain;
		}

		int get_terrain() const {
			return terrain;
		}

		TerrainConstraint(const TileMap *p_tile_map, const Vector2i &p_position, const TileSet::CellNeighbor &p_bit, int p_terrain);
		TerrainConstraint() {}
	};

	enum VisibilityMode {
		VISIBILITY_MODE_DEFAULT,
		VISIBILITY_MODE_FORCE_SHOW,
		VISIBILITY_MODE_FORCE_HIDE,
	};

private:
	friend class TileSetPlugin;

	// A compatibility enum to specify how is the data if formatted.
	enum DataFormat {
		FORMAT_1 = 0,
		FORMAT_2,
		FORMAT_3
	};
	mutable DataFormat format = FORMAT_1; // Assume lowest possible format if none is present;

	static constexpr float FP_ADJUST = 0.00001;

	// Properties.
	Ref<TileSet> tile_set;
	int quadrant_size = 16;
	bool collision_animatable = false;
	VisibilityMode collision_visibility_mode = VISIBILITY_MODE_DEFAULT;
	VisibilityMode navigation_visibility_mode = VISIBILITY_MODE_DEFAULT;

	// Updates.
	bool pending_update = false;

	// Rect.
	Rect2 rect_cache;
	bool rect_cache_dirty = true;
	Rect2i used_rect_cache;
	bool used_rect_cache_dirty = true;

	// TileMap layers.
	struct TileMapLayer {
		String name;
		bool enabled = true;
		Color modulate = Color(1, 1, 1, 1);
		bool y_sort_enabled = false;
		int y_sort_origin = 0;
		int z_index = 0;
		RID canvas_item;
		Map<Vector2i, TileMapCell> tile_map;
		Map<Vector2i, TileMapQuadrant> quadrant_map;
		SelfList<TileMapQuadrant>::List dirty_quadrant_list;
	};
	LocalVector<TileMapLayer> layers;
	int selected_layer = -1;

	// Mapping for RID to coords.
	Map<RID, Vector2i> bodies_coords;

	// Quadrants and internals management.
	Vector2i _coords_to_quadrant_coords(int p_layer, const Vector2i &p_coords) const;

	Map<Vector2i, TileMapQuadrant>::Element *_create_quadrant(int p_layer, const Vector2i &p_qk);

	void _make_quadrant_dirty(Map<Vector2i, TileMapQuadrant>::Element *Q);
	void _make_all_quadrants_dirty();
	void _queue_update_dirty_quadrants();

	void _update_dirty_quadrants();

	void _recreate_layer_internals(int p_layer);
	void _recreate_internals();

	void _erase_quadrant(Map<Vector2i, TileMapQuadrant>::Element *Q);
	void _clear_layer_internals(int p_layer);
	void _clear_internals();

	// Rect caching.
	void _recompute_rect_cache();

	// Per-system methods.
	bool _rendering_quadrant_order_dirty = false;
	void _rendering_notification(int p_what);
	void _rendering_update_layer(int p_layer);
	void _rendering_cleanup_layer(int p_layer);
	void _rendering_update_dirty_quadrants(SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list);
	void _rendering_create_quadrant(TileMapQuadrant *p_quadrant);
	void _rendering_cleanup_quadrant(TileMapQuadrant *p_quadrant);
	void _rendering_draw_quadrant_debug(TileMapQuadrant *p_quadrant);

	Transform2D last_valid_transform;
	Transform2D new_transform;
	void _physics_notification(int p_what);
	void _physics_update_dirty_quadrants(SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list);
	void _physics_cleanup_quadrant(TileMapQuadrant *p_quadrant);
	void _physics_draw_quadrant_debug(TileMapQuadrant *p_quadrant);

	void _navigation_notification(int p_what);
	void _navigation_update_dirty_quadrants(SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list);
	void _navigation_cleanup_quadrant(TileMapQuadrant *p_quadrant);
	void _navigation_draw_quadrant_debug(TileMapQuadrant *p_quadrant);

	void _scenes_update_dirty_quadrants(SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list);
	void _scenes_cleanup_quadrant(TileMapQuadrant *p_quadrant);
	void _scenes_draw_quadrant_debug(TileMapQuadrant *p_quadrant);

	// Terrains.
	Set<TileSet::TerrainsPattern> _get_valid_terrains_patterns_for_constraints(int p_terrain_set, const Vector2i &p_position, Set<TerrainConstraint> p_constraints);

	// Set and get tiles from data arrays.
	void _set_tile_data(int p_layer, const Vector<int> &p_data);
	Vector<int> _get_tile_data(int p_layer) const;

	void _build_runtime_update_tile_data(SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list);

	void _tile_set_changed();
	bool _tile_set_changed_deferred_update_needed = false;
	void _tile_set_changed_deferred_update();

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _notification(int p_what);
	static void _bind_methods();

public:
	static Vector2i transform_coords_layout(Vector2i p_coords, TileSet::TileOffsetAxis p_offset_axis, TileSet::TileLayout p_from_layout, TileSet::TileLayout p_to_layout);

	enum {
		INVALID_CELL = -1
	};

#ifdef TOOLS_ENABLED
	virtual Rect2 _edit_get_rect() const override;
#endif

	void set_tileset(const Ref<TileSet> &p_tileset);
	Ref<TileSet> get_tileset() const;

	void set_quadrant_size(int p_size);
	int get_quadrant_size() const;

	static void draw_tile(RID p_canvas_item, Vector2i p_position, const Ref<TileSet> p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, int p_frame = -1, Color p_modulation = Color(1.0, 1.0, 1.0, 1.0), const TileData *p_tile_data_override = nullptr);

	// Layers management.
	int get_layers_count() const;
	void add_layer(int p_to_pos);
	void move_layer(int p_layer, int p_to_pos);
	void remove_layer(int p_layer);
	void set_layer_name(int p_layer, String p_name);
	String get_layer_name(int p_layer) const;
	void set_layer_enabled(int p_layer, bool p_visible);
	bool is_layer_enabled(int p_layer) const;
	void set_layer_modulate(int p_layer, Color p_modulate);
	Color get_layer_modulate(int p_layer) const;
	void set_layer_y_sort_enabled(int p_layer, bool p_enabled);
	bool is_layer_y_sort_enabled(int p_layer) const;
	void set_layer_y_sort_origin(int p_layer, int p_y_sort_origin);
	int get_layer_y_sort_origin(int p_layer) const;
	void set_layer_z_index(int p_layer, int p_z_index);
	int get_layer_z_index(int p_layer) const;
	void set_selected_layer(int p_layer_id); // For editor use.
	int get_selected_layer() const;

	void set_collision_animatable(bool p_enabled);
	bool is_collision_animatable() const;

	// Debug visibility modes.
	void set_collision_visibility_mode(VisibilityMode p_show_collision);
	VisibilityMode get_collision_visibility_mode();

	void set_navigation_visibility_mode(VisibilityMode p_show_navigation);
	VisibilityMode get_navigation_visibility_mode();

	// Cells accessors.
	void set_cell(int p_layer, const Vector2i &p_coords, int p_source_id = -1, const Vector2i p_atlas_coords = TileSetSource::INVALID_ATLAS_COORDS, int p_alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE);
	int get_cell_source_id(int p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	Vector2i get_cell_atlas_coords(int p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	int get_cell_alternative_tile(int p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;

	// Patterns.
	Ref<TileMapPattern> get_pattern(int p_layer, TypedArray<Vector2i> p_coords_array);
	Vector2i map_pattern(Vector2i p_position_in_tilemap, Vector2i p_coords_in_pattern, Ref<TileMapPattern> p_pattern);
	void set_pattern(int p_layer, Vector2i p_position, const Ref<TileMapPattern> p_pattern);

	// Terrains.
	Set<TerrainConstraint> get_terrain_constraints_from_removed_cells_list(int p_layer, const Set<Vector2i> &p_to_replace, int p_terrain_set, bool p_ignore_empty_terrains = true) const; // Not exposed.
	Set<TerrainConstraint> get_terrain_constraints_from_added_tile(Vector2i p_position, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern) const; // Not exposed.
	Map<Vector2i, TileSet::TerrainsPattern> terrain_wave_function_collapse(const Set<Vector2i> &p_to_replace, int p_terrain_set, const Set<TerrainConstraint> p_constraints); // Not exposed.
	void set_cells_from_surrounding_terrains(int p_layer, TypedArray<Vector2i> p_coords_array, int p_terrain_set, bool p_ignore_empty_terrains = true);

	// Not exposed to users
	TileMapCell get_cell(int p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	Map<Vector2i, TileMapQuadrant> *get_quadrant_map(int p_layer);
	int get_effective_quadrant_size(int p_layer) const;
	//---

	virtual void set_y_sort_enabled(bool p_enable) override;

	Vector2 map_to_world(const Vector2i &p_pos) const;
	Vector2i world_to_map(const Vector2 &p_pos) const;

	bool is_existing_neighbor(TileSet::CellNeighbor p_cell_neighbor) const;
	Vector2i get_neighbor_cell(const Vector2i &p_coords, TileSet::CellNeighbor p_cell_neighbor) const;

	TypedArray<Vector2i> get_used_cells(int p_layer) const;
	Rect2 get_used_rect(); // Not const because of cache

	// Override some methods of the CanvasItem class to pass the changes to the quadrants CanvasItems
	virtual void set_light_mask(int p_light_mask) override;
	virtual void set_material(const Ref<Material> &p_material) override;
	virtual void set_use_parent_material(bool p_use_parent_material) override;
	virtual void set_texture_filter(CanvasItem::TextureFilter p_texture_filter) override;
	virtual void set_texture_repeat(CanvasItem::TextureRepeat p_texture_repeat) override;

	// For finding tiles from collision.
	Vector2i get_coords_for_body_rid(RID p_physics_body);

	// Fixing a nclearing methods.
	void fix_invalid_tiles();

	// Clears tiles from a given layer
	void clear_layer(int p_layer);
	void clear();

	// Force a TileMap update
	void force_update(int p_layer = -1);

	// Helpers?
	TypedArray<Vector2i> get_surrounding_tiles(Vector2i coords);
	void draw_cells_outline(Control *p_control, Set<Vector2i> p_cells, Color p_color, Transform2D p_transform = Transform2D());

	// Virtual function to modify the TileData at runtime
	GDVIRTUAL2R(bool, _use_tile_data_runtime_update, int, Vector2i);
	GDVIRTUAL3(_tile_data_runtime_update, int, Vector2i, TileData *);

	// Configuration warnings.
	TypedArray<String> get_configuration_warnings() const override;

	TileMap();
	~TileMap();
};

VARIANT_ENUM_CAST(TileMap::VisibilityMode);

#endif // TILE_MAP_H
