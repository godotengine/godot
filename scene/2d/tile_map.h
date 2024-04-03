/**************************************************************************/
/*  tile_map.h                                                            */
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

#ifndef TILE_MAP_H
#define TILE_MAP_H

#include "scene/2d/node_2d.h"
#include "scene/gui/control.h"
#include "scene/resources/tile_set.h"

class TileSetAtlasSource;

class TerrainConstraint {
private:
	const TileMap *tile_map = nullptr;
	Vector2i base_cell_coords;
	int bit = -1;
	int terrain = -1;

	int priority = 1;

public:
	bool operator<(const TerrainConstraint &p_other) const {
		if (base_cell_coords == p_other.base_cell_coords) {
			return bit < p_other.bit;
		}
		return base_cell_coords < p_other.base_cell_coords;
	}

	String to_string() const {
		return vformat("Constraint {pos:%s, bit:%d, terrain:%d, priority:%d}", base_cell_coords, bit, terrain, priority);
	}

	Vector2i get_base_cell_coords() const {
		return base_cell_coords;
	}

	bool is_center_bit() const {
		return bit == 0;
	}

	HashMap<Vector2i, TileSet::CellNeighbor> get_overlapping_coords_and_peering_bits() const;

	void set_terrain(int p_terrain) {
		terrain = p_terrain;
	}

	int get_terrain() const {
		return terrain;
	}

	void set_priority(int p_priority) {
		priority = p_priority;
	}

	int get_priority() const {
		return priority;
	}

	TerrainConstraint(const TileMap *p_tile_map, const Vector2i &p_position, int p_terrain); // For the center terrain bit
	TerrainConstraint(const TileMap *p_tile_map, const Vector2i &p_position, const TileSet::CellNeighbor &p_bit, int p_terrain); // For peering bits
	TerrainConstraint(){};
};

#ifdef DEBUG_ENABLED
class DebugQuadrant;
#endif // DEBUG_ENABLED
class RenderingQuadrant;

struct CellData {
	Vector2i coords;
	TileMapCell cell;

	// Debug.
	SelfList<CellData> debug_quadrant_list_element;

	// Rendering.
	Ref<RenderingQuadrant> rendering_quadrant;
	SelfList<CellData> rendering_quadrant_list_element;
	LocalVector<RID> occluders;

	// Physics.
	LocalVector<RID> bodies;

	// Navigation.
	LocalVector<RID> navigation_regions;

	// Scenes.
	String scene;

	// Runtime TileData cache.
	TileData *runtime_tile_data_cache = nullptr;

	// List elements.
	SelfList<CellData> dirty_list_element;

	bool operator<(const CellData &p_other) const {
		return coords < p_other.coords;
	}

	// For those, copy everything but SelfList elements.
	void operator=(const CellData &p_other) {
		coords = p_other.coords;
		cell = p_other.cell;
		occluders = p_other.occluders;
		bodies = p_other.bodies;
		navigation_regions = p_other.navigation_regions;
		scene = p_other.scene;
		runtime_tile_data_cache = p_other.runtime_tile_data_cache;
	}

	CellData(const CellData &p_other) :
			debug_quadrant_list_element(this),
			rendering_quadrant_list_element(this),
			dirty_list_element(this) {
		coords = p_other.coords;
		cell = p_other.cell;
		occluders = p_other.occluders;
		bodies = p_other.bodies;
		navigation_regions = p_other.navigation_regions;
		scene = p_other.scene;
		runtime_tile_data_cache = p_other.runtime_tile_data_cache;
	}

	CellData() :
			debug_quadrant_list_element(this),
			rendering_quadrant_list_element(this),
			dirty_list_element(this) {
	}
};

// For compatibility reasons, we use another comparator for Y-sorted layers.
struct CellDataYSortedComparator {
	_FORCE_INLINE_ bool operator()(const CellData &p_a, const CellData &p_b) const {
		return p_a.coords.x == p_b.coords.x ? (p_a.coords.y < p_b.coords.y) : (p_a.coords.x > p_b.coords.x);
	}
};

#ifdef DEBUG_ENABLED
class DebugQuadrant : public RefCounted {
	GDCLASS(DebugQuadrant, RefCounted);

public:
	Vector2i quadrant_coords;
	SelfList<CellData>::List cells;
	RID canvas_item;

	SelfList<DebugQuadrant> dirty_quadrant_list_element;

	// For those, copy everything but SelfList elements.
	DebugQuadrant(const DebugQuadrant &p_other) :
			dirty_quadrant_list_element(this) {
		quadrant_coords = p_other.quadrant_coords;
		cells = p_other.cells;
		canvas_item = p_other.canvas_item;
	}

	DebugQuadrant() :
			dirty_quadrant_list_element(this) {
	}

	~DebugQuadrant() {
		cells.clear();
	}
};
#endif // DEBUG_ENABLED

class RenderingQuadrant : public RefCounted {
	GDCLASS(RenderingQuadrant, RefCounted);

public:
	struct CoordsWorldComparator {
		_ALWAYS_INLINE_ bool operator()(const Vector2 &p_a, const Vector2 &p_b) const {
			// We sort the cells by their local coords, as it is needed by rendering.
			if (p_a.y == p_b.y) {
				return p_a.x > p_b.x;
			} else {
				return p_a.y < p_b.y;
			}
		}
	};

	Vector2i quadrant_coords;
	SelfList<CellData>::List cells;
	List<RID> canvas_items;
	Vector2 canvas_items_position;

	SelfList<RenderingQuadrant> dirty_quadrant_list_element;

	// For those, copy everything but SelfList elements.
	RenderingQuadrant(const RenderingQuadrant &p_other) :
			dirty_quadrant_list_element(this) {
		quadrant_coords = p_other.quadrant_coords;
		cells = p_other.cells;
		canvas_items = p_other.canvas_items;
	}

	RenderingQuadrant() :
			dirty_quadrant_list_element(this) {
	}

	~RenderingQuadrant() {
		cells.clear();
	}
};

class TileMapLayer : public RefCounted {
	GDCLASS(TileMapLayer, RefCounted);

public:
	enum DataFormat {
		FORMAT_1 = 0,
		FORMAT_2,
		FORMAT_3,
		FORMAT_MAX,
	};

	enum DirtyFlags {
		DIRTY_FLAGS_LAYER_ENABLED = 0,
		DIRTY_FLAGS_LAYER_MODULATE,
		DIRTY_FLAGS_LAYER_Y_SORT_ENABLED,
		DIRTY_FLAGS_LAYER_Y_SORT_ORIGIN,
		DIRTY_FLAGS_LAYER_Z_INDEX,
		DIRTY_FLAGS_LAYER_NAVIGATION_ENABLED,
		DIRTY_FLAGS_LAYER_INDEX_IN_TILE_MAP_NODE,
		DIRTY_FLAGS_TILE_MAP_IN_TREE,
		DIRTY_FLAGS_TILE_MAP_IN_CANVAS,
		DIRTY_FLAGS_TILE_MAP_VISIBILITY,
		DIRTY_FLAGS_TILE_MAP_XFORM,
		DIRTY_FLAGS_TILE_MAP_LOCAL_XFORM,
		DIRTY_FLAGS_TILE_MAP_SELECTED_LAYER,
		DIRTY_FLAGS_TILE_MAP_LIGHT_MASK,
		DIRTY_FLAGS_TILE_MAP_MATERIAL,
		DIRTY_FLAGS_TILE_MAP_USE_PARENT_MATERIAL,
		DIRTY_FLAGS_TILE_MAP_TEXTURE_FILTER,
		DIRTY_FLAGS_TILE_MAP_TEXTURE_REPEAT,
		DIRTY_FLAGS_TILE_MAP_TILE_SET,
		DIRTY_FLAGS_TILE_MAP_QUADRANT_SIZE,
		DIRTY_FLAGS_TILE_MAP_COLLISION_ANIMATABLE,
		DIRTY_FLAGS_TILE_MAP_COLLISION_VISIBILITY_MODE,
		DIRTY_FLAGS_TILE_MAP_NAVIGATION_VISIBILITY_MODE,
		DIRTY_FLAGS_TILE_MAP_Y_SORT_ENABLED,
		DIRTY_FLAGS_TILE_MAP_RUNTIME_UPDATE,
		DIRTY_FLAGS_MAX,
	};

private:
	// Exposed properties.
	String name;
	bool enabled = true;
	Color modulate = Color(1, 1, 1, 1);
	bool y_sort_enabled = false;
	int y_sort_origin = 0;
	int z_index = 0;
	bool navigation_enabled = true;
	RID navigation_map;
	bool uses_world_navigation_map = false;

	// Internal.
	TileMap *tile_map_node = nullptr;
	int layer_index_in_tile_map_node = -1;
	RID canvas_item;
	HashMap<Vector2i, CellData> tile_map;

	// Dirty flag. Allows knowing what was modified since the last update.
	struct {
		bool flags[DIRTY_FLAGS_MAX] = { false };
		SelfList<CellData>::List cell_list;
	} dirty;

	// Rect cache.
	mutable Rect2 rect_cache;
	mutable bool rect_cache_dirty = true;
	mutable Rect2i used_rect_cache;
	mutable bool used_rect_cache_dirty = true;

	// Runtime tile data.
	bool _runtime_update_tile_data_was_cleaned_up = false;
	void _build_runtime_update_tile_data(bool p_force_cleanup);
	void _build_runtime_update_tile_data_for_cell(CellData &r_cell_data, bool p_auto_add_to_dirty_list = false);
	void _clear_runtime_update_tile_data();

	// Per-system methods.
#ifdef DEBUG_ENABLED
	HashMap<Vector2i, Ref<DebugQuadrant>> debug_quadrant_map;
	Vector2i _coords_to_debug_quadrant_coords(const Vector2i &p_coords) const;
	bool _debug_was_cleaned_up = false;
	void _debug_update(bool p_force_cleanup);
	void _debug_quadrants_update_cell(CellData &r_cell_data, SelfList<DebugQuadrant>::List &r_dirty_debug_quadrant_list);
#endif // DEBUG_ENABLED

	HashMap<Vector2i, Ref<RenderingQuadrant>> rendering_quadrant_map;
	bool _rendering_was_cleaned_up = false;
	void _rendering_update(bool p_force_cleanup);
	void _rendering_quadrants_update_cell(CellData &r_cell_data, SelfList<RenderingQuadrant>::List &r_dirty_rendering_quadrant_list);
	void _rendering_occluders_clear_cell(CellData &r_cell_data);
	void _rendering_occluders_update_cell(CellData &r_cell_data);
#ifdef DEBUG_ENABLED
	void _rendering_draw_cell_debug(const RID &p_canvas_item, const Vector2i &p_quadrant_pos, const CellData &r_cell_data);
#endif // DEBUG_ENABLED

	HashMap<RID, Vector2i> bodies_coords; // Mapping for RID to coords.
	bool _physics_was_cleaned_up = false;
	void _physics_update(bool p_force_cleanup);
	void _physics_notify_tilemap_change(DirtyFlags p_what);
	void _physics_clear_cell(CellData &r_cell_data);
	void _physics_update_cell(CellData &r_cell_data);
#ifdef DEBUG_ENABLED
	void _physics_draw_cell_debug(const RID &p_canvas_item, const Vector2i &p_quadrant_pos, const CellData &r_cell_data);
#endif // DEBUG_ENABLED

	bool _navigation_was_cleaned_up = false;
	void _navigation_update(bool p_force_cleanup);
	void _navigation_clear_cell(CellData &r_cell_data);
	void _navigation_update_cell(CellData &r_cell_data);
#ifdef DEBUG_ENABLED
	void _navigation_draw_cell_debug(const RID &p_canvas_item, const Vector2i &p_quadrant_pos, const CellData &r_cell_data);
#endif // DEBUG_ENABLED

	bool _scenes_was_cleaned_up = false;
	void _scenes_update(bool p_force_cleanup);
	void _scenes_clear_cell(CellData &r_cell_data);
	void _scenes_update_cell(CellData &r_cell_data);
#ifdef DEBUG_ENABLED
	void _scenes_draw_cell_debug(const RID &p_canvas_item, const Vector2i &p_quadrant_pos, const CellData &r_cell_data);
#endif // DEBUG_ENABLED

	// Terrains.
	TileSet::TerrainsPattern _get_best_terrain_pattern_for_constraints(int p_terrain_set, const Vector2i &p_position, const RBSet<TerrainConstraint> &p_constraints, TileSet::TerrainsPattern p_current_pattern);
	RBSet<TerrainConstraint> _get_terrain_constraints_from_added_pattern(const Vector2i &p_position, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern) const;
	RBSet<TerrainConstraint> _get_terrain_constraints_from_painted_cells_list(const RBSet<Vector2i> &p_painted, int p_terrain_set, bool p_ignore_empty_terrains) const;

public:
	// TileMap node.
	void set_tile_map(TileMap *p_tile_map);
	void set_layer_index_in_tile_map_node(int p_index);

	// Rect caching.
	Rect2 get_rect(bool &r_changed) const;

	// Terrains.
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_constraints(const Vector<Vector2i> &p_to_replace, int p_terrain_set, const RBSet<TerrainConstraint> &p_constraints); // Not exposed.
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_connect(const Vector<Vector2i> &p_coords_array, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true); // Not exposed.
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_path(const Vector<Vector2i> &p_coords_array, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true); // Not exposed.
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_pattern(const Vector<Vector2i> &p_coords_array, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern, bool p_ignore_empty_terrains = true); // Not exposed.

	// Not exposed to users.
	TileMapCell get_cell(const Vector2i &p_coords, bool p_use_proxies = false) const;

	// For TileMap node's use.
	void set_tile_data(DataFormat p_format, const Vector<int> &p_data);
	Vector<int> get_tile_data() const;
	void notify_tile_map_change(DirtyFlags p_what);
	void internal_update(bool p_force_cleanup);

	// --- Exposed in TileMap ---

	// Cells manipulation.
	void set_cell(const Vector2i &p_coords, int p_source_id = TileSet::INVALID_SOURCE, const Vector2i p_atlas_coords = TileSetSource::INVALID_ATLAS_COORDS, int p_alternative_tile = 0);
	void erase_cell(const Vector2i &p_coords);

	int get_cell_source_id(const Vector2i &p_coords, bool p_use_proxies = false) const;
	Vector2i get_cell_atlas_coords(const Vector2i &p_coords, bool p_use_proxies = false) const;
	int get_cell_alternative_tile(const Vector2i &p_coords, bool p_use_proxies = false) const;
	TileData *get_cell_tile_data(const Vector2i &p_coords, bool p_use_proxies = false) const; // Helper method to make accessing the data easier.
	void clear();

	// Patterns.
	Ref<TileMapPattern> get_pattern(TypedArray<Vector2i> p_coords_array);
	void set_pattern(const Vector2i &p_position, const Ref<TileMapPattern> p_pattern);

	// Terrains.
	void set_cells_terrain_connect(TypedArray<Vector2i> p_cells, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true);
	void set_cells_terrain_path(TypedArray<Vector2i> p_path, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true);

	// Cells usage.
	TypedArray<Vector2i> get_used_cells() const;
	TypedArray<Vector2i> get_used_cells_by_id(int p_source_id = TileSet::INVALID_SOURCE, const Vector2i p_atlas_coords = TileSetSource::INVALID_ATLAS_COORDS, int p_alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE) const;
	Rect2i get_used_rect() const;

	// Layer properties.
	void set_name(String p_name);
	String get_name() const;
	void set_enabled(bool p_enabled);
	bool is_enabled() const;
	void set_modulate(Color p_modulate);
	Color get_modulate() const;
	void set_y_sort_enabled(bool p_y_sort_enabled);
	bool is_y_sort_enabled() const;
	void set_y_sort_origin(int p_y_sort_origin);
	int get_y_sort_origin() const;
	void set_z_index(int p_z_index);
	int get_z_index() const;
	void set_navigation_enabled(bool p_enabled);
	bool is_navigation_enabled() const;
	void set_navigation_map(RID p_map);
	RID get_navigation_map() const;

	// Fixing and clearing methods.
	void fix_invalid_tiles();

	// Find coords for body.
	bool has_body_rid(RID p_physics_body) const;
	Vector2i get_coords_for_body_rid(RID p_physics_body) const; // For finding tiles from collision.

	~TileMapLayer();
};

class TileMap : public Node2D {
	GDCLASS(TileMap, Node2D);

public:
	enum VisibilityMode {
		VISIBILITY_MODE_DEFAULT,
		VISIBILITY_MODE_FORCE_SHOW,
		VISIBILITY_MODE_FORCE_HIDE,
	};

private:
	friend class TileSetPlugin;

	// A compatibility enum to specify how is the data if formatted.
	mutable TileMapLayer::DataFormat format = TileMapLayer::FORMAT_3;

	static constexpr float FP_ADJUST = 0.00001;

	// Properties.
	Ref<TileSet> tile_set;
	int rendering_quadrant_size = 16;
	bool collision_animatable = false;
	VisibilityMode collision_visibility_mode = VISIBILITY_MODE_DEFAULT;
	VisibilityMode navigation_visibility_mode = VISIBILITY_MODE_DEFAULT;

	// Layers.
	LocalVector<Ref<TileMapLayer>> layers;
	Ref<TileMapLayer> default_layer; // Dummy layer to fetch default values.
	int selected_layer = -1;
	bool pending_update = false;

	Transform2D last_valid_transform;
	Transform2D new_transform;

	void _tile_set_changed();

	void _update_notify_local_transform();

	// Polygons.
	HashMap<Pair<Ref<Resource>, int>, Ref<Resource>, PairHash<Ref<Resource>, int>> polygon_cache;
	PackedVector2Array _get_transformed_vertices(const PackedVector2Array &p_vertices, int p_alternative_id);

	void _deferred_internal_update();
	void _internal_update(bool p_force_cleanup);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	bool _property_can_revert(const StringName &p_name) const;
	bool _property_get_revert(const StringName &p_name, Variant &r_property) const;

	void _notification(int p_what);
	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	Rect2i _get_used_rect_bind_compat_78328();
	void _set_quadrant_size_compat_81070(int p_quadrant_size);
	int _get_quadrant_size_compat_81070() const;

	static void _bind_compatibility_methods();
#endif

public:
	static Vector2i transform_coords_layout(const Vector2i &p_coords, TileSet::TileOffsetAxis p_offset_axis, TileSet::TileLayout p_from_layout, TileSet::TileLayout p_to_layout);

#ifdef TOOLS_ENABLED
	virtual Rect2 _edit_get_rect() const override;
#endif

#ifndef DISABLE_DEPRECATED
	void force_update(int p_layer);
#endif

	// Called by TileMapLayers.
	void queue_internal_update();

	void set_tileset(const Ref<TileSet> &p_tileset);
	Ref<TileSet> get_tileset() const;

	void set_rendering_quadrant_size(int p_size);
	int get_rendering_quadrant_size() const;

	static void draw_tile(RID p_canvas_item, const Vector2 &p_position, const Ref<TileSet> p_tile_set, int p_atlas_source_id, const Vector2i &p_atlas_coords, int p_alternative_tile, int p_frame = -1, Color p_modulation = Color(1.0, 1.0, 1.0, 1.0), const TileData *p_tile_data_override = nullptr, real_t p_normalized_animation_offset = 0.0);

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
	void set_layer_navigation_enabled(int p_layer, bool p_enabled);
	bool is_layer_navigation_enabled(int p_layer) const;
	void set_layer_navigation_map(int p_layer, RID p_map);
	RID get_layer_navigation_map(int p_layer) const;

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
	void set_cell(int p_layer, const Vector2i &p_coords, int p_source_id = TileSet::INVALID_SOURCE, const Vector2i p_atlas_coords = TileSetSource::INVALID_ATLAS_COORDS, int p_alternative_tile = 0);
	void erase_cell(int p_layer, const Vector2i &p_coords);
	int get_cell_source_id(int p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	Vector2i get_cell_atlas_coords(int p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	int get_cell_alternative_tile(int p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	// Helper method to make accessing the data easier.
	TileData *get_cell_tile_data(int p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;

	// Patterns.
	Ref<TileMapPattern> get_pattern(int p_layer, TypedArray<Vector2i> p_coords_array);
	Vector2i map_pattern(const Vector2i &p_position_in_tilemap, const Vector2i &p_coords_in_pattern, Ref<TileMapPattern> p_pattern);
	void set_pattern(int p_layer, const Vector2i &p_position, const Ref<TileMapPattern> p_pattern);

	// Terrains (Not exposed).
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_constraints(int p_layer, const Vector<Vector2i> &p_to_replace, int p_terrain_set, const RBSet<TerrainConstraint> &p_constraints);
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_connect(int p_layer, const Vector<Vector2i> &p_coords_array, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true);
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_path(int p_layer, const Vector<Vector2i> &p_coords_array, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true);
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_pattern(int p_layer, const Vector<Vector2i> &p_coords_array, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern, bool p_ignore_empty_terrains = true);

	// Terrains (exposed).
	void set_cells_terrain_connect(int p_layer, TypedArray<Vector2i> p_cells, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true);
	void set_cells_terrain_path(int p_layer, TypedArray<Vector2i> p_path, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true);

	// Not exposed to users.
	TileMapCell get_cell(int p_layer, const Vector2i &p_coords, bool p_use_proxies = false) const;
	int get_effective_quadrant_size(int p_layer) const;

	virtual void set_y_sort_enabled(bool p_enable) override;

	Vector2 map_to_local(const Vector2i &p_pos) const;
	Vector2i local_to_map(const Vector2 &p_pos) const;

	bool is_existing_neighbor(TileSet::CellNeighbor p_cell_neighbor) const;
	Vector2i get_neighbor_cell(const Vector2i &p_coords, TileSet::CellNeighbor p_cell_neighbor) const;

	TypedArray<Vector2i> get_used_cells(int p_layer) const;
	TypedArray<Vector2i> get_used_cells_by_id(int p_layer, int p_source_id = TileSet::INVALID_SOURCE, const Vector2i p_atlas_coords = TileSetSource::INVALID_ATLAS_COORDS, int p_alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE) const;
	Rect2i get_used_rect() const;

	// Override some methods of the CanvasItem class to pass the changes to the quadrants CanvasItems.
	virtual void set_light_mask(int p_light_mask) override;
	virtual void set_material(const Ref<Material> &p_material) override;
	virtual void set_use_parent_material(bool p_use_parent_material) override;
	virtual void set_texture_filter(CanvasItem::TextureFilter p_texture_filter) override;
	virtual void set_texture_repeat(CanvasItem::TextureRepeat p_texture_repeat) override;

	// For finding tiles from collision.
	Vector2i get_coords_for_body_rid(RID p_physics_body);
	// For getting their layers as well.
	int get_layer_for_body_rid(RID p_physics_body);

	// Fixing and clearing methods.
	void fix_invalid_tiles();

	// Clears tiles from a given layer.
	void clear_layer(int p_layer);
	void clear();

	// Force a TileMap update.
	void update_internals();
	void notify_runtime_tile_data_update(int p_layer = -1);

	// Helpers?
	TypedArray<Vector2i> get_surrounding_cells(const Vector2i &coords);
	void draw_cells_outline(Control *p_control, const RBSet<Vector2i> &p_cells, Color p_color, Transform2D p_transform = Transform2D());
	Ref<Resource> get_transformed_polygon(Ref<Resource> p_polygon, int p_alternative_id);

	// Virtual function to modify the TileData at runtime.
	GDVIRTUAL2R(bool, _use_tile_data_runtime_update, int, Vector2i);
	GDVIRTUAL3(_tile_data_runtime_update, int, Vector2i, TileData *);

	// Configuration warnings.
	PackedStringArray get_configuration_warnings() const override;

	TileMap();
	~TileMap();
};

VARIANT_ENUM_CAST(TileMap::VisibilityMode);

#endif // TILE_MAP_H
