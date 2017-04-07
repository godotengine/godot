/*************************************************************************/
/*  tile_map.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "scene/2d/navigation2d.h"
#include "scene/2d/node_2d.h"
#include "scene/resources/tile_set.h"
#include "self_list.h"
#include "vset.h"

class TileMap : public Node2D {

	GDCLASS(TileMap, Node2D);

public:
	enum Mode {
		MODE_SQUARE,
		MODE_ISOMETRIC,
		MODE_CUSTOM
	};

	enum HalfOffset {
		HALF_OFFSET_X,
		HALF_OFFSET_Y,
		HALF_OFFSET_DISABLED,
	};

	enum TileOrigin {
		TILE_ORIGIN_TOP_LEFT,
		TILE_ORIGIN_CENTER,
		TILE_ORIGIN_BOTTOM_LEFT
	};

private:
	Ref<TileSet> tile_set;
	Size2i cell_size;
	int quadrant_size;
	bool center_x, center_y;
	Mode mode;
	Transform2D custom_transform;
	HalfOffset half_offset;
	bool use_kinematic;
	Navigation2D *navigation;

	union PosKey {

		struct {
			int16_t x;
			int16_t y;
		};
		uint32_t key;

		//using a more precise comparison so the regions can be sorted later
		bool operator<(const PosKey &p_k) const { return (y == p_k.y) ? x < p_k.x : y < p_k.y; }

		PosKey(int16_t p_x, int16_t p_y) {
			x = p_x;
			y = p_y;
		}
		PosKey() {
			x = 0;
			y = 0;
		}
	};

	union Cell {

		struct {
			int32_t id : 24;
			bool flip_h : 1;
			bool flip_v : 1;
			bool transpose : 1;
		};

		uint32_t _u32t;
		Cell() { _u32t = 0; }
	};

	Map<PosKey, Cell> tile_map;
	struct Quadrant {

		Vector2 pos;
		List<RID> canvas_items;
		RID body;

		SelfList<Quadrant> dirty_list;

		struct NavPoly {
			int id;
			Transform2D xform;
		};

		struct Occluder {
			RID id;
			Transform2D xform;
		};

		Map<PosKey, NavPoly> navpoly_ids;
		Map<PosKey, Occluder> occluder_instances;

		VSet<PosKey> cells;

		void operator=(const Quadrant &q) {
			pos = q.pos;
			canvas_items = q.canvas_items;
			body = q.body;
			cells = q.cells;
			navpoly_ids = q.navpoly_ids;
			occluder_instances = q.occluder_instances;
		}
		Quadrant(const Quadrant &q)
			: dirty_list(this) {
			pos = q.pos;
			canvas_items = q.canvas_items;
			body = q.body;
			cells = q.cells;
			occluder_instances = q.occluder_instances;
			navpoly_ids = q.navpoly_ids;
		}
		Quadrant()
			: dirty_list(this) {}
	};

	Map<PosKey, Quadrant> quadrant_map;

	SelfList<Quadrant>::List dirty_quadrant_list;

	bool pending_update;

	Rect2 rect_cache;
	bool rect_cache_dirty;
	Rect2 used_size_cache;
	bool used_size_cache_dirty;
	bool quadrant_order_dirty;
	bool y_sort_mode;
	float fp_adjust;
	float friction;
	float bounce;
	uint32_t collision_layer;
	uint32_t collision_mask;

	TileOrigin tile_origin;

	int occluder_light_mask;

	void _fix_cell_transform(Transform2D &xform, const Cell &p_cell, const Vector2 &p_offset, const Size2 &p_sc);

	Map<PosKey, Quadrant>::Element *_create_quadrant(const PosKey &p_qk);
	void _erase_quadrant(Map<PosKey, Quadrant>::Element *Q);
	void _make_quadrant_dirty(Map<PosKey, Quadrant>::Element *Q);
	void _recreate_quadrants();
	void _clear_quadrants();
	void _update_dirty_quadrants();
	void _update_quadrant_space(const RID &p_space);
	void _update_quadrant_transform();
	void _recompute_rect_cache();

	_FORCE_INLINE_ int _get_quadrant_size() const;

	void _set_tile_data(const PoolVector<int> &p_data);
	PoolVector<int> _get_tile_data() const;

	void _set_old_cell_size(int p_size) { set_cell_size(Size2(p_size, p_size)); }
	int _get_old_cell_size() const { return cell_size.x; }

	_FORCE_INLINE_ Vector2 _map_to_world(int p_x, int p_y, bool p_ignore_ofs = false) const;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	enum {
		INVALID_CELL = -1
	};

	void set_tileset(const Ref<TileSet> &p_tileset);
	Ref<TileSet> get_tileset() const;

	void set_cell_size(Size2 p_size);
	Size2 get_cell_size() const;

	void set_quadrant_size(int p_size);
	int get_quadrant_size() const;

	void set_center_x(bool p_enable);
	bool get_center_x() const;
	void set_center_y(bool p_enable);
	bool get_center_y() const;

	void set_cell(int p_x, int p_y, int p_tile, bool p_flip_x = false, bool p_flip_y = false, bool p_transpose = false);
	int get_cell(int p_x, int p_y) const;
	bool is_cell_x_flipped(int p_x, int p_y) const;
	bool is_cell_y_flipped(int p_x, int p_y) const;
	bool is_cell_transposed(int p_x, int p_y) const;

	void set_cellv(const Vector2 &p_pos, int p_tile, bool p_flip_x = false, bool p_flip_y = false, bool p_transpose = false);
	int get_cellv(const Vector2 &p_pos) const;

	Rect2 get_item_rect() const;

	void set_collision_layer(uint32_t p_layer);
	uint32_t get_collision_layer() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_collision_use_kinematic(bool p_use_kinematic);
	bool get_collision_use_kinematic() const;

	void set_collision_friction(float p_friction);
	float get_collision_friction() const;

	void set_collision_bounce(float p_bounce);
	float get_collision_bounce() const;

	void set_mode(Mode p_mode);
	Mode get_mode() const;

	void set_half_offset(HalfOffset p_half_offset);
	HalfOffset get_half_offset() const;

	void set_tile_origin(TileOrigin p_tile_origin);
	TileOrigin get_tile_origin() const;

	void set_custom_transform(const Transform2D &p_xform);
	Transform2D get_custom_transform() const;

	Transform2D get_cell_transform() const;
	Vector2 get_cell_draw_offset() const;

	Vector2 map_to_world(const Vector2 &p_pos, bool p_ignore_ofs = false) const;
	Vector2 world_to_map(const Vector2 &p_pos) const;

	void set_y_sort_mode(bool p_enable);
	bool is_y_sort_mode_enabled() const;

	Array get_used_cells() const;
	Rect2 get_used_rect(); // Not const because of cache

	void set_occluder_light_mask(int p_mask);
	int get_occluder_light_mask() const;

	virtual void set_light_mask(int p_light_mask);

	void clear();

	TileMap();
	~TileMap();
};

VARIANT_ENUM_CAST(TileMap::Mode);
VARIANT_ENUM_CAST(TileMap::HalfOffset);
VARIANT_ENUM_CAST(TileMap::TileOrigin);

#endif // TILE_MAP_H
