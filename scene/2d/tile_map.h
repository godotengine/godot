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

#include "core/templates/self_list.h"
#include "core/templates/vset.h"
#include "scene/2d/node_2d.h"
#include "scene/gui/control.h"
#include "scene/resources/tile_set.h"

class TileSetAtlasSource;

union TileMapCell {
	struct {
		int32_t source_id : 16;
		int16_t coord_x : 16;
		int16_t coord_y : 16;
		int32_t alternative_tile : 16;
	};

	uint64_t _u64t;
	TileMapCell(int p_source_id = -1, Vector2i p_atlas_coords = TileSetSource::INVALID_ATLAS_COORDS, int p_alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE) {
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

	// Quadrant coords.
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

	void operator=(const TileMapQuadrant &q) {
		coords = q.coords;
		debug_canvas_item = q.debug_canvas_item;
		canvas_items = q.canvas_items;
		occluders = q.occluders;
		bodies = q.bodies;
		navigation_regions = q.navigation_regions;
	}

	TileMapQuadrant(const TileMapQuadrant &q) :
			dirty_list_element(this) {
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

class TileMapPattern : public Object {
	GDCLASS(TileMapPattern, Object);

	Vector2i size;
	Map<Vector2i, TileMapCell> pattern;

protected:
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

class TileMap : public Node2D {
	GDCLASS(TileMap, Node2D);

public:
private:
	friend class TileSetPlugin;

	enum DataFormat {
		FORMAT_1 = 0,
		FORMAT_2,
		FORMAT_3
	};

	Ref<TileSet> tile_set;
	int quadrant_size;
	Transform2D custom_transform;

	// Map of cells
	Map<Vector2i, TileMapCell> tile_map;

	Vector2i _coords_to_quadrant_coords(const Vector2i &p_coords) const;

	Map<Vector2i, TileMapQuadrant> quadrant_map;

	SelfList<TileMapQuadrant>::List dirty_quadrant_list;

	bool pending_update = false;

	Rect2 rect_cache;
	bool rect_cache_dirty = true;
	Rect2 used_size_cache;
	bool used_size_cache_dirty;
	mutable DataFormat format;

	void _fix_cell_transform(Transform2D &xform, const TileMapCell &p_cell, const Vector2 &p_offset, const Size2 &p_sc);

	Map<Vector2i, TileMapQuadrant>::Element *_create_quadrant(const Vector2i &p_qk);
	void _erase_quadrant(Map<Vector2i, TileMapQuadrant>::Element *Q);
	void _make_all_quadrants_dirty(bool p_update = true);
	void _make_quadrant_dirty(Map<Vector2i, TileMapQuadrant>::Element *Q, bool p_update = true);
	void _recreate_quadrants();
	void _clear_quadrants();
	void _recompute_rect_cache();

	void _update_all_items_material_state();

	void _set_tile_data(const Vector<int> &p_data);
	Vector<int> _get_tile_data() const;

	void _tile_set_changed();

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

	void set_cell(const Vector2i &p_coords, int p_source_id = -1, const Vector2i p_atlas_coords = TileSetSource::INVALID_ATLAS_COORDS, int p_alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE);
	int get_cell_source_id(const Vector2i &p_coords) const;
	Vector2i get_cell_atlas_coords(const Vector2i &p_coords) const;
	int get_cell_alternative_tile(const Vector2i &p_coords) const;

	TileMapPattern *get_pattern(TypedArray<Vector2i> p_coords_array);
	Vector2i map_pattern(Vector2i p_position_in_tilemap, Vector2i p_coords_in_pattern, const TileMapPattern *p_pattern);
	void set_pattern(Vector2i p_position, const TileMapPattern *p_pattern);

	// Not exposed to users
	TileMapCell get_cell(const Vector2i &p_coords) const;
	Map<Vector2i, TileMapQuadrant> &get_quadrant_map();
	int get_effective_quadrant_size() const;

	void update_dirty_quadrants();

	Vector2 map_to_world(const Vector2i &p_pos) const;
	Vector2i world_to_map(const Vector2 &p_pos) const;

	bool is_existing_neighbor(TileSet::CellNeighbor p_cell_neighbor) const;
	Vector2i get_neighbor_cell(const Vector2i &p_coords, TileSet::CellNeighbor p_cell_neighbor) const;

	TypedArray<Vector2i> get_used_cells() const;
	Rect2 get_used_rect(); // Not const because of cache

	// Override some methods of the CanvasItem class to pass the changes to the quadrants CanvasItems
	virtual void set_light_mask(int p_light_mask) override;
	virtual void set_material(const Ref<Material> &p_material) override;
	virtual void set_use_parent_material(bool p_use_parent_material) override;
	virtual void set_texture_filter(CanvasItem::TextureFilter p_texture_filter) override;
	virtual void set_texture_repeat(CanvasItem::TextureRepeat p_texture_repeat) override;

	void fix_invalid_tiles();
	void clear();

	// Helpers
	TypedArray<Vector2i> get_surrounding_tiles(Vector2i coords);
	void draw_cells_outline(Control *p_control, Set<Vector2i> p_cells, Color p_color, Transform2D p_transform = Transform2D());

	TileMap();
	~TileMap();
};
#endif // TILE_MAP_H
