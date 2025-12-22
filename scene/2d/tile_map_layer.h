/**************************************************************************/
/*  tile_map_layer.h                                                      */
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

#pragma once

#include "scene/resources/2d/tile_set.h"

#ifndef NAVIGATION_2D_DISABLED
class NavigationMeshSourceGeometryData2D;
#endif // NAVIGATION_2D_DISABLED
class TileSetAtlasSource;
class TileMap;

enum TileMapLayerDataFormat {
	TILE_MAP_LAYER_DATA_FORMAT_0 = 0,
	TILE_MAP_LAYER_DATA_FORMAT_MAX,
};

class TerrainConstraint {
private:
	Ref<TileSet> tile_set;
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

	TerrainConstraint(Ref<TileSet> p_tile_set, const Vector2i &p_position, int p_terrain); // For the center terrain bit
	TerrainConstraint(Ref<TileSet> p_tile_set, const Vector2i &p_position, const TileSet::CellNeighbor &p_bit, int p_terrain); // For peering bits
	TerrainConstraint() {}
};

#ifdef DEBUG_ENABLED
class DebugQuadrant;
#endif // DEBUG_ENABLED
class RenderingQuadrant;
class PhysicsQuadrant;

struct CellData {
	Vector2i coords;
	TileMapCell cell;

	// Debug
	SelfList<CellData> debug_quadrant_list_element;

	// Rendering.
	Ref<RenderingQuadrant> rendering_quadrant;
	SelfList<CellData> rendering_quadrant_list_element;
	LocalVector<LocalVector<RID>> occluders;

#ifndef PHYSICS_2D_DISABLED
	// Physics.
	Ref<PhysicsQuadrant> physics_quadrant;
	SelfList<CellData> physics_quadrant_list_element;
#endif // PHYSICS_2D_DISABLED

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
		navigation_regions = p_other.navigation_regions;
		scene = p_other.scene;
		runtime_tile_data_cache = p_other.runtime_tile_data_cache;
	}

	CellData(const CellData &p_other) :
			debug_quadrant_list_element(this),
			rendering_quadrant_list_element(this),
#ifndef PHYSICS_2D_DISABLED
			physics_quadrant_list_element(this),
#endif // PHYSICS_2D_DISABLED
			dirty_list_element(this) {
		coords = p_other.coords;
		cell = p_other.cell;
		occluders = p_other.occluders;
		navigation_regions = p_other.navigation_regions;
		scene = p_other.scene;
		runtime_tile_data_cache = p_other.runtime_tile_data_cache;
	}

	CellData() :
			debug_quadrant_list_element(this),
			rendering_quadrant_list_element(this),
#ifndef PHYSICS_2D_DISABLED
			physics_quadrant_list_element(this),
#endif // PHYSICS_2D_DISABLED
			dirty_list_element(this) {
	}
};

// We use another comparator for Y-sorted layers with reversed X drawing order.
struct CellDataYSortedXReversedComparator {
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

	RID physics_mesh;

	// Used to deleted unused quadrants.
	bool drawn_to = false;

	SelfList<DebugQuadrant> dirty_quadrant_list_element;

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

	RenderingQuadrant() :
			dirty_quadrant_list_element(this) {
	}

	~RenderingQuadrant() {
		cells.clear();
	}
};

#ifndef PHYSICS_2D_DISABLED
class PhysicsQuadrant : public RefCounted {
	GDCLASS(PhysicsQuadrant, RefCounted);

public:
	struct PhysicsBodyKey {
		int physics_layer = 0;
		Vector2 linear_velocity;
		real_t angular_velocity = 0.0;

		bool one_way_collision = false;
		real_t one_way_collision_margin = 0.0;

		int64_t y_origin = 0; // This is only used if one_way_collision is on, to avoid merging polygons vertically in that case.

		bool operator<(const PhysicsBodyKey &p_other) const {
			if (physics_layer == p_other.physics_layer) {
				if (linear_velocity == p_other.linear_velocity) {
					if (angular_velocity == p_other.angular_velocity) {
						if (one_way_collision == p_other.one_way_collision) {
							if (one_way_collision && y_origin != p_other.y_origin) {
								return y_origin < p_other.y_origin;
							}
							return one_way_collision_margin < p_other.one_way_collision_margin;
						}
						return one_way_collision < p_other.one_way_collision;
					}
					return angular_velocity < p_other.angular_velocity;
				}
				return linear_velocity < p_other.linear_velocity;
			}
			return physics_layer < p_other.physics_layer;
		}

		bool operator!=(const PhysicsBodyKey &p_other) const {
			return !this->operator==(p_other);
		}
		bool operator==(const PhysicsBodyKey &p_other) const {
			return physics_layer == p_other.physics_layer &&
					linear_velocity == p_other.linear_velocity &&
					angular_velocity == p_other.angular_velocity &&
					one_way_collision == p_other.one_way_collision &&
					one_way_collision_margin == p_other.one_way_collision_margin &&
					(!one_way_collision || y_origin == p_other.y_origin);
		}
	};

	struct PhysicsBodyKeyHasher {
		static uint32_t hash(const PhysicsBodyKey &p_hash) {
			uint32_t h = hash_murmur3_one_32(p_hash.physics_layer);
			h = hash_murmur3_one_real(p_hash.linear_velocity.x);
			h = hash_murmur3_one_real(p_hash.linear_velocity.y, h);
			h = hash_murmur3_one_real(p_hash.angular_velocity, h);
			if (p_hash.one_way_collision) {
				h = hash_murmur3_one_32(p_hash.y_origin, h);
			}
			return h;
		}
	};

	struct PhysicsBodyValue {
		RID body;
		Vector<Vector<Vector2>> polygons;
	};

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

	HashMap<PhysicsBodyKey, PhysicsBodyValue, PhysicsBodyKeyHasher> bodies;
	LocalVector<Ref<ConvexPolygonShape2D>> shapes;

	SelfList<PhysicsQuadrant> dirty_quadrant_list_element;

	PhysicsQuadrant() :
			dirty_quadrant_list_element(this) {
	}

	~PhysicsQuadrant() {
		cells.clear();
	}
};
#endif // PHYSICS_2D_DISABLED

class TileMapLayer : public Node2D {
	GDCLASS(TileMapLayer, Node2D);

public:
	enum HighlightMode {
		HIGHLIGHT_MODE_DEFAULT,
		HIGHLIGHT_MODE_ABOVE,
		HIGHLIGHT_MODE_BELOW,
	};

	enum DebugVisibilityMode {
		DEBUG_VISIBILITY_MODE_DEFAULT,
		DEBUG_VISIBILITY_MODE_FORCE_SHOW,
		DEBUG_VISIBILITY_MODE_FORCE_HIDE,
	};

	enum DirtyFlags {
		DIRTY_FLAGS_LAYER_ENABLED = 0,

		DIRTY_FLAGS_LAYER_IN_TREE,
		DIRTY_FLAGS_LAYER_IN_CANVAS,
		DIRTY_FLAGS_LAYER_LOCAL_TRANSFORM,
		DIRTY_FLAGS_LAYER_VISIBILITY,
		DIRTY_FLAGS_LAYER_SELF_MODULATE,
		DIRTY_FLAGS_LAYER_Y_SORT_ENABLED,
		DIRTY_FLAGS_LAYER_Y_SORT_ORIGIN,
		DIRTY_FLAGS_LAYER_X_DRAW_ORDER_REVERSED,
		DIRTY_FLAGS_LAYER_Z_INDEX,
		DIRTY_FLAGS_LAYER_LIGHT_MASK,
		DIRTY_FLAGS_LAYER_TEXTURE_FILTER,
		DIRTY_FLAGS_LAYER_TEXTURE_REPEAT,
		DIRTY_FLAGS_LAYER_RENDERING_QUADRANT_SIZE,
		DIRTY_FLAGS_LAYER_COLLISION_ENABLED,
		DIRTY_FLAGS_LAYER_USE_KINEMATIC_BODIES,
		DIRTY_FLAGS_LAYER_PHYSICS_QUADRANT_SIZE,
		DIRTY_FLAGS_LAYER_COLLISION_VISIBILITY_MODE,
		DIRTY_FLAGS_LAYER_OCCLUSION_ENABLED,
		DIRTY_FLAGS_LAYER_NAVIGATION_ENABLED,
		DIRTY_FLAGS_LAYER_NAVIGATION_MAP,
		DIRTY_FLAGS_LAYER_NAVIGATION_VISIBILITY_MODE,
		DIRTY_FLAGS_LAYER_RUNTIME_UPDATE,
		DIRTY_FLAGS_LAYER_HIGHLIGHT_MODE,

		DIRTY_FLAGS_LAYER_INDEX_IN_TILE_MAP_NODE, // For compatibility.

		DIRTY_FLAGS_LAYER_GROUP_SELECTED_LAYERS,
		DIRTY_FLAGS_LAYER_GROUP_HIGHLIGHT_SELECTED,

		DIRTY_FLAGS_TILE_SET,

		DIRTY_FLAGS_MAX,
	};

private:
	static constexpr float FP_ADJUST = 0.00001;

	// Properties.
	HashMap<Vector2i, CellData> tile_map_layer_data;

	bool enabled = true;
	Ref<TileSet> tile_set;

	HighlightMode highlight_mode = HIGHLIGHT_MODE_DEFAULT;

	int y_sort_origin = 0;
	bool x_draw_order_reversed = false;
	int rendering_quadrant_size = 16;

	bool collision_enabled = true;
	bool use_kinematic_bodies = false;
	int physics_quadrant_size = 16;
	DebugVisibilityMode collision_visibility_mode = DEBUG_VISIBILITY_MODE_DEFAULT;

	bool occlusion_enabled = true;

	bool navigation_enabled = true;
	RID navigation_map_override;
	DebugVisibilityMode navigation_visibility_mode = DEBUG_VISIBILITY_MODE_DEFAULT;

	// Internal.
	bool pending_update = false;

	// For keeping compatibility with TileMap.
	TileMap *tile_map_node = nullptr;
	int layer_index_in_tile_map_node = -1;

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
	void _build_runtime_update_tile_data_for_cell(CellData &r_cell_data, bool p_use_tilemap_for_runtime, bool p_auto_add_to_dirty_list = false);
	bool _runtime_update_needs_all_cells_cleaned_up = false;
	void _clear_runtime_update_tile_data();
	void _clear_runtime_update_tile_data_for_cell(CellData &r_cell_data);
	void _update_cells_callback(bool p_force_cleanup);

	// Coords to quadrant coords
	Vector2i _coords_to_quadrant_coords(const Vector2i &p_coords, const int p_quadrant_size) const;

	// Per-system methods.
#ifdef DEBUG_ENABLED
	HashMap<Vector2i, Ref<DebugQuadrant>> debug_quadrant_map;
	bool _debug_was_cleaned_up = true;
	void _debug_update(bool p_force_cleanup);
	void _debug_quadrants_update_cell(CellData &r_cell_data);
	void _get_debug_quadrant_for_cell(const Vector2i &p_coords);
#endif // DEBUG_ENABLED

	HashMap<Vector2i, Ref<RenderingQuadrant>> rendering_quadrant_map;
	bool _rendering_was_cleaned_up = true;
	bool _occlusion_was_cleaned_up = true;
	void _rendering_update(bool p_force_cleanup);
	void _rendering_notification(int p_what);
	Color _highlight_color(const Color &p_modulate) const;
	void _rendering_quadrants_update_cell(CellData &r_cell_data, SelfList<RenderingQuadrant>::List &r_dirty_rendering_quadrant_list);
	void _rendering_occluders_clear_cell(CellData &r_cell_data);
	void _rendering_occluders_update_cell(CellData &r_cell_data);
#ifdef DEBUG_ENABLED
	void _rendering_draw_cell_debug(const RID &p_canvas_item, const Vector2 &p_quadrant_pos, const CellData &r_cell_data);
#endif // DEBUG_ENABLED

#ifndef PHYSICS_2D_DISABLED
	HashMap<Vector2i, Ref<PhysicsQuadrant>> physics_quadrant_map;
	HashMap<RID, Vector2i> bodies_coords; // Mapping for RID to coords.
	bool _physics_was_cleaned_up = true;
	void _physics_update(bool p_force_cleanup);
	void _physics_notification(int p_what);
	void _physics_quadrants_update_cell(CellData &r_cell_data, SelfList<PhysicsQuadrant>::List &r_dirty_physics_quadrant_list);
	void _physics_clear_cell(CellData &r_cell_data);
	void _physics_update_cell(CellData &r_cell_data);
#ifdef DEBUG_ENABLED
	void _physics_draw_quadrant_debug(const RID &p_canvas_item, DebugQuadrant &r_debug_quadrant);
#endif // DEBUG_ENABLED
#endif // PHYSICS_2D_DISABLED

#ifndef NAVIGATION_2D_DISABLED
	bool _navigation_was_cleaned_up = true;
	void _navigation_update(bool p_force_cleanup);
	void _navigation_notification(int p_what);
	void _navigation_clear_cell(CellData &r_cell_data);
	void _navigation_update_cell(CellData &r_cell_data);
#ifdef DEBUG_ENABLED
	void _navigation_draw_cell_debug(const RID &p_canvas_item, const Vector2 &p_quadrant_pos, const CellData &r_cell_data);
#endif // DEBUG_ENABLED
#endif // NAVIGATION_2D_DISABLED

	bool _scenes_was_cleaned_up = true;
	void _scenes_update(bool p_force_cleanup);
	void _scenes_clear_cell(CellData &r_cell_data);
	void _scenes_update_cell(CellData &r_cell_data);
#ifdef DEBUG_ENABLED
	void _scenes_draw_cell_debug(const RID &p_canvas_item, const Vector2 &p_quadrant_pos, const CellData &r_cell_data);
#endif // DEBUG_ENABLED
	void _set_scene_transform_with_alternative(Node2D *p_scene, const Vector2 &p_cell_position, const int p_alternative_id);

	// Terrains.
	TileSet::TerrainsPattern _get_best_terrain_pattern_for_constraints(int p_terrain_set, const Vector2i &p_position, const RBSet<TerrainConstraint> &p_constraints, TileSet::TerrainsPattern p_current_pattern) const;
	RBSet<TerrainConstraint> _get_terrain_constraints_from_added_pattern(const Vector2i &p_position, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern) const;
	RBSet<TerrainConstraint> _get_terrain_constraints_from_painted_cells_list(const RBSet<Vector2i> &p_painted, int p_terrain_set, bool p_ignore_empty_terrains) const;

	void _tile_set_changed();

	void _renamed();
	void _update_notify_local_transform();

	// Internal updates.
	void _queue_internal_update();
	void _deferred_internal_update();
	void _internal_update(bool p_force_cleanup);

	virtual void _physics_interpolated_changed() override;

protected:
	void _notification(int p_what);

	static void _bind_methods();
	void _validate_property(PropertyInfo &p_property) const;

	virtual void _update_self_texture_filter(RS::CanvasItemTextureFilter p_texture_filter) override;
	virtual void _update_self_texture_repeat(RS::CanvasItemTextureRepeat p_texture_repeat) override;

public:
#ifdef TOOLS_ENABLED
	virtual bool _edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const override;
#endif

	// TileMap node.
	void set_as_tile_map_internal_node(int p_index);
	int get_index_in_tile_map() const {
		return layer_index_in_tile_map_node;
	}
	const HashMap<Vector2i, CellData> &get_tile_map_layer_data() const {
		return tile_map_layer_data;
	}

	// Rect caching.
	Rect2 get_rect(bool &r_changed) const;

	// Terrains.
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_constraints(const Vector<Vector2i> &p_to_replace, int p_terrain_set, const RBSet<TerrainConstraint> &p_constraints) const; // Not exposed.
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_connect(const Vector<Vector2i> &p_coords_array, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true) const; // Not exposed.
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_path(const Vector<Vector2i> &p_coords_array, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true) const; // Not exposed.
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_pattern(const Vector<Vector2i> &p_coords_array, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern, bool p_ignore_empty_terrains = true) const; // Not exposed.

	// Not exposed to users.
	TileMapCell get_cell(const Vector2i &p_coords) const;

	static void compute_transformed_tile_dest_rect(Rect2 &r_dest_rect, bool &r_transpose, const Vector2 &p_position, const Vector2 &p_dest_rect_size, const TileData *p_tile_data, int p_alternative_tile);
	static void draw_tile(RID p_canvas_item, const Vector2 &p_position, const Ref<TileSet> p_tile_set, int p_atlas_source_id, const Vector2i &p_atlas_coords, int p_alternative_tile, int p_frame = -1, const TileData *p_tile_data_override = nullptr, real_t p_normalized_animation_offset = 0.0);

	////////////// Exposed functions //////////////

	// --- Cells manipulation ---
	// Generic cells manipulations and data access.
	void set_cell(const Vector2i &p_coords, int p_source_id = TileSet::INVALID_SOURCE, const Vector2i &p_atlas_coords = TileSetSource::INVALID_ATLAS_COORDS, int p_alternative_tile = 0);
	void place_random_tiles(Array atlas_coords_array, Rect2i region, float scattering_probability, bool replace_tile);
	void erase_cell(const Vector2i &p_coords);
	void fix_invalid_tiles();
	void clear();

	int get_cell_source_id(const Vector2i &p_coords) const;
	Vector2i get_cell_atlas_coords(const Vector2i &p_coords) const;
	int get_cell_alternative_tile(const Vector2i &p_coords) const;
	TileData *get_cell_tile_data(const Vector2i &p_coords) const; // Helper method to make accessing the data easier.

	TypedArray<Vector2i> get_used_cells() const;
	TypedArray<Vector2i> get_used_cells_by_id(int p_source_id = TileSet::INVALID_SOURCE, const Vector2i &p_atlas_coords = TileSetSource::INVALID_ATLAS_COORDS, int p_alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE) const;
	Rect2i get_used_rect() const;

	bool is_cell_flipped_h(const Vector2i &p_coords) const;
	bool is_cell_flipped_v(const Vector2i &p_coords) const;
	bool is_cell_transposed(const Vector2i &p_coords) const;

	// Patterns.
	Ref<TileMapPattern> get_pattern(TypedArray<Vector2i> p_coords_array);
	void set_pattern(const Vector2i &p_position, const Ref<TileMapPattern> p_pattern);

	// Terrains.
	void set_cells_terrain_connect(TypedArray<Vector2i> p_cells, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true);
	void set_cells_terrain_path(TypedArray<Vector2i> p_path, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains = true);

#ifndef PHYSICS_2D_DISABLED
	// --- Physics helpers ---
	bool has_body_rid(RID p_physics_body) const;
	Vector2i get_coords_for_body_rid(RID p_physics_body) const; // For finding tiles from collision.
#endif // PHYSICS_2D_DISABLED

	// --- Runtime ---
	void update_internals();
	void notify_runtime_tile_data_update();
	GDVIRTUAL1R(bool, _use_tile_data_runtime_update, Vector2i);
	GDVIRTUAL2(_tile_data_runtime_update, Vector2i, TileData *);
	GDVIRTUAL2(_update_cells, TypedArray<Vector2i>, bool);

	// --- Shortcuts to methods defined in TileSet ---
	Vector2i map_pattern(const Vector2i &p_position_in_tilemap, const Vector2i &p_coords_in_pattern, Ref<TileMapPattern> p_pattern);
	TypedArray<Vector2i> get_surrounding_cells(const Vector2i &p_coords);
	Vector2i get_neighbor_cell(const Vector2i &p_coords, TileSet::CellNeighbor p_cell_neighbor) const;
	Vector2 map_to_local(const Vector2i &p_pos) const;
	Vector2i local_to_map(const Vector2 &p_pos) const;

	// --- Accessors ---
	void set_tile_map_data_from_array(const Vector<uint8_t> &p_data);
	Vector<uint8_t> get_tile_map_data_as_array() const;

	void set_enabled(bool p_enabled);
	bool is_enabled() const;
	void set_tile_set(const Ref<TileSet> &p_tile_set);
	Ref<TileSet> get_tile_set() const;

	void set_highlight_mode(HighlightMode p_highlight_mode);
	HighlightMode get_highlight_mode() const;

	virtual void set_self_modulate(const Color &p_self_modulate) override;
	virtual void set_y_sort_enabled(bool p_y_sort_enabled) override;
	void set_y_sort_origin(int p_y_sort_origin);
	int get_y_sort_origin() const;
	void set_x_draw_order_reversed(bool p_x_draw_order_reversed);
	bool is_x_draw_order_reversed() const;
	virtual void set_z_index(int p_z_index) override;
	virtual void set_light_mask(int p_light_mask) override;
	void set_rendering_quadrant_size(int p_size);
	int get_rendering_quadrant_size() const;

	void set_collision_enabled(bool p_enabled);
	bool is_collision_enabled() const;
	void set_use_kinematic_bodies(bool p_use_kinematic_bodies);
	bool is_using_kinematic_bodies() const;
	void set_collision_visibility_mode(DebugVisibilityMode p_show_collision);
	DebugVisibilityMode get_collision_visibility_mode() const;
	void set_physics_quadrant_size(int p_size);
	int get_physics_quadrant_size() const;

	void set_occlusion_enabled(bool p_enabled);
	bool is_occlusion_enabled() const;

	void set_navigation_enabled(bool p_enabled);
	bool is_navigation_enabled() const;
	void set_navigation_map(RID p_map);
	RID get_navigation_map() const;
	void set_navigation_visibility_mode(DebugVisibilityMode p_show_navigation);
	DebugVisibilityMode get_navigation_visibility_mode() const;

private:
#ifndef NAVIGATION_2D_DISABLED
	static Callable _navmesh_source_geometry_parsing_callback;
	static RID _navmesh_source_geometry_parser;
#endif // NAVIGATION_2D_DISABLED

public:
#ifndef NAVIGATION_2D_DISABLED
	static void navmesh_parse_init();
	static void navmesh_parse_source_geometry(const Ref<NavigationPolygon> &p_navigation_mesh, Ref<NavigationMeshSourceGeometryData2D> p_source_geometry_data, Node *p_node);
#endif // NAVIGATION_2D_DISABLED

	TileMapLayer();
	~TileMapLayer();
};

VARIANT_ENUM_CAST(TileMapLayer::DebugVisibilityMode);
