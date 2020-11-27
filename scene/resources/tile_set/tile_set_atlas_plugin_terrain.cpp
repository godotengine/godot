/*************************************************************************/
/*  tile_set_atlas_plugin_terrain.cpp                                    */
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

/*List<Vector2i> dirty_bitmask;

void TileMap::make_bitmask_area_dirty(const Vector2 &p_pos) {
	// Autotiles: trigger bitmask update making them dirty
	for (int x = p_pos.x - 1; x <= p_pos.x + 1; x++) {
		for (int y = p_pos.y - 1; y <= p_pos.y + 1; y++) {
			Vector2i p(x, y);
			if (dirty_bitmask.find(p) == nullptr) {
				dirty_bitmask.push_back(p);
			}
		}
	}
}

void TileMap::update_bitmask_area(const Vector2 &p_pos) {
	// Autotiles: update the cells because of a bitmask change
	for (int x = p_pos.x - 1; x <= p_pos.x + 1; x++) {
		for (int y = p_pos.y - 1; y <= p_pos.y + 1; y++) {
			update_cell_bitmask(x, y);
		}
	}
}

void TileMap::update_bitmask_region(const Vector2 &p_start, const Vector2 &p_end) {
	// Autotiles: update the cells because of a bitmask change in the given region
	if ((p_end.x < p_start.x || p_end.y < p_start.y) || (p_end.x == p_start.x && p_end.y == p_start.y)) {
		// Update everything
		Array a = get_used_cells();
		for (int i = 0; i < a.size(); i++) {
			Vector2 vector = (Vector2)a[i];
			update_cell_bitmask(vector.x, vector.y);
		}
		return;
	}
	// Update cells in the region
	for (int x = p_start.x - 1; x <= p_end.x + 1; x++) {
		for (int y = p_start.y - 1; y <= p_end.y + 1; y++) {
			update_cell_bitmask(x, y);
		}
	}
}

void TileMap::update_cell_bitmask(int p_x, int p_y) {
	// Autotiles: Run the autotiling on a given cell
	ERR_FAIL_COND_MSG(tile_set.is_null(), "Cannot update cell bitmask if Tileset is not open.");
	Vector2i p(p_x, p_y);
	Map<Vector2i, Cell>::Element *E = tile_map.find(p);
	if (E != nullptr) {
		int id = get_cell_source_id(Vector2i(p_x, p_y));
		if (tile_set->tile_get_tile_mode(id) == TileSet::AUTO_TILE) {
			uint16_t mask = 0;
			int top_left = get_cell_source_id(Vector2i(p_x - 1, p_y - 1));
			int top = get_cell_source_id(Vector2i(p_x, p_y - 1));
			int top_right = get_cell_source_id(Vector2i(p_x + 1, p_y - 1));
			int right = get_cell_source_id(Vector2i(p_x + 1, p_y));
			int bottom_right = get_cell_source_id(Vector2i(p_x + 1, p_y + 1));
			int bottom = get_cell_source_id(Vector2i(p_x, p_y + 1));
			int bottom_left = get_cell_source_id(Vector2i(p_x - 1, p_y + 1));
			int left = get_cell_source_id(Vector2i(p_x - 1, p_y));

			if (tile_set->autotile_get_bitmask_mode(id) == TileSet::BITMASK_2X2) {
				if (tile_set->is_tile_bound(id, top_left) && tile_set->is_tile_bound(id, top) && tile_set->is_tile_bound(id, left)) {
					mask |= TileSet::BIND_TOPLEFT;
				}
				if (tile_set->is_tile_bound(id, top_right) && tile_set->is_tile_bound(id, top) && tile_set->is_tile_bound(id, right)) {
					mask |= TileSet::BIND_TOPRIGHT;
				}
				if (tile_set->is_tile_bound(id, bottom_left) && tile_set->is_tile_bound(id, bottom) && tile_set->is_tile_bound(id, left)) {
					mask |= TileSet::BIND_BOTTOMLEFT;
				}
				if (tile_set->is_tile_bound(id, bottom_right) && tile_set->is_tile_bound(id, bottom) && tile_set->is_tile_bound(id, right)) {
					mask |= TileSet::BIND_BOTTOMRIGHT;
				}
			} else {
				if (tile_set->autotile_get_bitmask_mode(id) == TileSet::BITMASK_3X3_MINIMAL) {
					if (tile_set->is_tile_bound(id, top_left) && tile_set->is_tile_bound(id, top) && tile_set->is_tile_bound(id, left)) {
						mask |= TileSet::BIND_TOPLEFT;
					}
					if (tile_set->is_tile_bound(id, top_right) && tile_set->is_tile_bound(id, top) && tile_set->is_tile_bound(id, right)) {
						mask |= TileSet::BIND_TOPRIGHT;
					}
					if (tile_set->is_tile_bound(id, bottom_left) && tile_set->is_tile_bound(id, bottom) && tile_set->is_tile_bound(id, left)) {
						mask |= TileSet::BIND_BOTTOMLEFT;
					}
					if (tile_set->is_tile_bound(id, bottom_right) && tile_set->is_tile_bound(id, bottom) && tile_set->is_tile_bound(id, right)) {
						mask |= TileSet::BIND_BOTTOMRIGHT;
					}
				} else {
					if (tile_set->is_tile_bound(id, top_left)) {
						mask |= TileSet::BIND_TOPLEFT;
					}
					if (tile_set->is_tile_bound(id, top_right)) {
						mask |= TileSet::BIND_TOPRIGHT;
					}
					if (tile_set->is_tile_bound(id, bottom_left)) {
						mask |= TileSet::BIND_BOTTOMLEFT;
					}
					if (tile_set->is_tile_bound(id, bottom_right)) {
						mask |= TileSet::BIND_BOTTOMRIGHT;
					}
				}
				if (tile_set->is_tile_bound(id, top)) {
					mask |= TileSet::BIND_TOP;
				}
				if (tile_set->is_tile_bound(id, left)) {
					mask |= TileSet::BIND_LEFT;
				}
				mask |= TileSet::BIND_CENTER;
				if (tile_set->is_tile_bound(id, right)) {
					mask |= TileSet::BIND_RIGHT;
				}
				if (tile_set->is_tile_bound(id, bottom)) {
					mask |= TileSet::BIND_BOTTOM;
				}
			}
			Vector2 coord = tile_set->autotile_get_subtile_for_bitmask(id, mask, this, Vector2(p_x, p_y));
			E->get().coord_x = (int)coord.x;
			E->get().coord_y = (int)coord.y;

			Vector2i qk = p.to_quadrant(_get_quadrant_size());
			Map<Vector2i, Quadrant>::Element *Q = quadrant_map.find(qk);
			_make_quadrant_dirty(Q);

		} else if (tile_set->tile_get_tile_mode(id) == TileSet::SINGLE_TILE) {
			E->get().coord_x = 0;
			E->get().coord_y = 0;
		} else if (tile_set->tile_get_tile_mode(id) == TileSet::ATLAS_TILE) {
			if (tile_set->autotile_get_bitmask(id, Vector2(p_x, p_y)) == TileSet::BIND_CENTER) {
				Vector2 coord = tile_set->atlastile_get_subtile_by_priority(id, this, Vector2(p_x, p_y));

				E->get().coord_x = (int)coord.x;
				E->get().coord_y = (int)coord.y;
			}
		}
	}
}

void TileMap::update_dirty_bitmask() {
	// Autotiles: Update the dirty bitmasks.
	while (dirty_bitmask.size() > 0) {
		update_cell_bitmask(dirty_bitmask[0].x, dirty_bitmask[0].y);
		dirty_bitmask.pop_front();
	}
}
*/
