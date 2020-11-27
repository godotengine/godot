/*************************************************************************/
/*  tile_map.cpp                                                         */
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

#include "tile_map.h"

#include "core/io/marshalls.h"
#include "core/math/geometry_2d.h"
#include "core/os/os.h"

#include "scene/resources/tile_set/tile_set_atlas_plugin_rendering.h"

int TileMap::get_effective_quadrant_size() const {
	// When using YSort, the quadrant size is reduced to 1 to have one CanvasItem per quadrant
	if (tile_set.is_valid() && tile_set->is_y_sorting()) {
		return 1;
	} else {
		return quadrant_size;
	}
}

Vector2i TileMap::_coords_to_quadrant_coords(const Vector2i &p_coords) const {
	int quadrant_size = get_effective_quadrant_size();

	// Rounding down, instead of simply rounding towards zero (truncating)
	return Vector2i(
			p_coords.x > 0 ? p_coords.x / quadrant_size : (p_coords.x - (quadrant_size - 1)) / quadrant_size,
			p_coords.y > 0 ? p_coords.y / quadrant_size : (p_coords.y - (quadrant_size - 1)) / quadrant_size);
}

void TileMap::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			pending_update = true;
			_recreate_quadrants();
			update_dirty_quadrants();
		} break;
	}

	// Transfers the notification to tileset plugins.
	if (tile_set.is_valid()) {
		for (int i = 0; i < tile_set->get_tile_set_atlas_plugins().size(); i++) {
			tile_set->get_tile_set_atlas_plugins()[i]->tilemap_notification(this, p_what);
		}
	}
}

Ref<TileSet> TileMap::get_tileset() const {
	return tile_set;
}

void TileMap::set_tileset(const Ref<TileSet> &p_tileset) {
	// Set the tileset, registering to its changes.
	if (tile_set.is_valid()) {
		tile_set->disconnect("changed", callable_mp(this, &TileMap::_make_all_quadrants_dirty));
		tile_set->disconnect("changed", callable_mp(this, &TileMap::_tile_set_changed));
	}

	tile_set = p_tileset;

	if (tile_set.is_valid()) {
		tile_set->connect("changed", callable_mp(this, &TileMap::_make_all_quadrants_dirty), varray(true));
		tile_set->connect("changed", callable_mp(this, &TileMap::_tile_set_changed));
	} else {
		clear();
	}

	_recreate_quadrants();
	emit_signal("changed");
}

int TileMap::get_quadrant_size() const {
	return quadrant_size;
}

void TileMap::set_quadrant_size(int p_size) {
	ERR_FAIL_COND_MSG(p_size < 1, "TileMapQuadrant size cannot be smaller than 1.");

	quadrant_size = p_size;
	_recreate_quadrants();
	emit_signal("changed");
}

void TileMap::_fix_cell_transform(Transform2D &xform, const TileMapCell &p_cell, const Vector2 &p_offset, const Size2 &p_sc) {
	Size2 s = p_sc;
	Vector2 offset = p_offset;

	// Flip/transpose: update the tile transform.
	TileData *tile_data = tile_set->get_atlas_source(p_cell.source_id)->get_tile_data(p_cell.get_atlas_coords(), p_cell.alternative_tile);
	if (tile_data->tile_get_transpose()) {
		SWAP(xform.elements[0].x, xform.elements[0].y);
		SWAP(xform.elements[1].x, xform.elements[1].y);
		SWAP(offset.x, offset.y);
		SWAP(s.x, s.y);
	}

	if (tile_data->tile_get_flip_h()) {
		xform.elements[0].x = -xform.elements[0].x;
		xform.elements[1].x = -xform.elements[1].x;
		offset.x = s.x - offset.x;
	}

	if (tile_data->tile_get_flip_v()) {
		xform.elements[0].y = -xform.elements[0].y;
		xform.elements[1].y = -xform.elements[1].y;
		offset.y = s.y - offset.y;
	}

	xform.elements[2] += offset;
}

void TileMap::update_dirty_quadrants() {
	if (!pending_update) {
		return;
	}
	if (!is_inside_tree() || !tile_set.is_valid()) {
		pending_update = false;
		return;
	}

	// Call the update_dirty_quadrant method on plugins
	for (int i = 0; i < tile_set->get_tile_set_atlas_plugins().size(); i++) {
		tile_set->get_tile_set_atlas_plugins()[i]->update_dirty_quadrants(this, dirty_quadrant_list);
	}

	// Clear the list
	while (dirty_quadrant_list.first()) {
		dirty_quadrant_list.remove(dirty_quadrant_list.first());
	}

	pending_update = false;

	_recompute_rect_cache();
}

void TileMap::_recompute_rect_cache() {
	// Compute the displayed area of the tilemap.
#ifdef DEBUG_ENABLED

	if (!rect_cache_dirty) {
		return;
	}

	Rect2 r_total;
	for (Map<Vector2i, TileMapQuadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		Rect2 r;
		r.position = map_to_world(E->key() * get_effective_quadrant_size());
		r.expand_to(map_to_world((E->key() + Vector2i(1, 0)) * get_effective_quadrant_size()));
		r.expand_to(map_to_world((E->key() + Vector2i(1, 1)) * get_effective_quadrant_size()));
		r.expand_to(map_to_world((E->key() + Vector2i(0, 1)) * get_effective_quadrant_size()));
		if (E == quadrant_map.front()) {
			r_total = r;
		} else {
			r_total = r_total.merge(r);
		}
	}

	rect_cache = r_total;

	item_rect_changed();

	rect_cache_dirty = false;
#endif
}

Map<Vector2i, TileMapQuadrant>::Element *TileMap::_create_quadrant(const Vector2i &p_qk) {
	ERR_FAIL_COND_V(!tile_set.is_valid(), nullptr);

	TileMapQuadrant q;

	rect_cache_dirty = true;

	// Call the update_dirty_quadrant method on plugins
	for (int i = 0; i < tile_set->get_tile_set_atlas_plugins().size(); i++) {
		tile_set->get_tile_set_atlas_plugins()[i]->create_quadrant(this, p_qk, &q);
	}

	return quadrant_map.insert(p_qk, q);
}

void TileMap::_erase_quadrant(Map<Vector2i, TileMapQuadrant>::Element *Q) {
	// Remove a quadrant.
	TileMapQuadrant &q = Q->get();

	// Free the canvas item..
	for (List<RID>::Element *E = q.canvas_items.front(); E; E = E->next()) {
		RenderingServer::get_singleton()->free(E->get());
	}
	q.canvas_items.clear();

	// Remove the quadrant from the dirty_list if it is there
	if (q.dirty_list_element.in_list()) {
		dirty_quadrant_list.remove(&q.dirty_list_element);
	}

	quadrant_map.erase(Q);
	rect_cache_dirty = true;
}

void TileMap::_make_all_quadrants_dirty(bool p_update) {
	// Make all quandrants dirty, then trigger an update later.
	for (Map<Vector2i, TileMapQuadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		if (!E->value().dirty_list_element.in_list()) {
			dirty_quadrant_list.add(&E->value().dirty_list_element);
		}
	}

	if (pending_update) {
		return;
	}
	pending_update = true;
	if (!is_inside_tree()) {
		return;
	}
	if (p_update) {
		call_deferred("update_dirty_quadrants");
	}
}

void TileMap::_make_quadrant_dirty(Map<Vector2i, TileMapQuadrant>::Element *Q, bool p_update) {
	// Make the given quadrant dirty, then trigger an update later.
	TileMapQuadrant &q = Q->get();
	if (!q.dirty_list_element.in_list()) {
		dirty_quadrant_list.add(&q.dirty_list_element);
	}

	if (pending_update) {
		return;
	}
	pending_update = true;
	if (!is_inside_tree()) {
		return;
	}

	if (p_update) {
		call_deferred("update_dirty_quadrants");
	}
}

void TileMap::set_cell(const Vector2i &p_coords, int p_source_id, const Vector2i p_atlas_coords, int p_alternative_tile) {
	print_line(vformat("Setting tile at position: (%s). Tile set ids: %d (%s) %d", p_coords, p_source_id, p_atlas_coords, p_alternative_tile));

	// Set the current cell tile (using integer position)
	Vector2i pk(p_coords);
	Map<Vector2i, TileMapCell>::Element *E = tile_map.find(pk);

	if (!E && p_source_id == TileSet::SOURCE_TYPE_INVALID) {
		return; // Nothing to do, the tile is already empty.
	}

	// Get the quadrant
	Vector2i qk = _coords_to_quadrant_coords(pk);

	Map<Vector2i, TileMapQuadrant>::Element *Q = quadrant_map.find(qk);

	if (p_source_id == TileSet::SOURCE_TYPE_INVALID) {
		// Erase existing cell in the tile map.
		tile_map.erase(pk);

		// Erase existing cell in the quadrant.
		ERR_FAIL_COND(!Q);
		TileMapQuadrant &q = Q->get();
		Vector2i pk_world_coords = q.map_to_world[pk];
		q.cells.erase(pk);
		q.map_to_world.erase(pk);
		q.world_to_map.erase(pk_world_coords);

		// Remove or make the quadrant dirty.
		if (q.cells.size() == 0) {
			_erase_quadrant(Q);
		} else {
			_make_quadrant_dirty(Q);
		}

		used_size_cache_dirty = true;
	} else {
		if (!E) {
			// Insert a new cell in the tile map.
			E = tile_map.insert(pk, TileMapCell());

			// Create a new quadrant if needed, then insert the cell if needed.
			if (!Q) {
				Q = _create_quadrant(qk);
			}
			TileMapQuadrant &q = Q->get();
			q.cells.insert(pk);
			Vector2i pk_world_coords = map_to_world(pk);
			q.map_to_world[pk] = pk_world_coords;
			q.world_to_map[pk_world_coords] = pk;
		} else {
			ERR_FAIL_COND(!Q); // TileMapQuadrant should exist...

			if (E->get().source_id == p_source_id && E->get().get_atlas_coords() == p_atlas_coords && E->get().alternative_tile == p_alternative_tile) {
				return; // Nothing changed.
			}
		}

		TileMapCell &c = E->get();

		c.source_id = p_source_id;
		c.set_atlas_coords(p_atlas_coords);
		c.alternative_tile = p_alternative_tile;

		_make_quadrant_dirty(Q);
		used_size_cache_dirty = true;
	}
}

int TileMap::get_cell_source_id(const Vector2i &p_coords) const {
	// Get a cell source id from position
	const Map<Vector2i, TileMapCell>::Element *E = tile_map.find(p_coords);

	if (!E) {
		return TileSet::SOURCE_TYPE_INVALID;
	}

	return E->get().source_id;
}

Vector2i TileMap::get_cell_atlas_coords(const Vector2i &p_coords) const {
	// Get a cell source id from position
	const Map<Vector2i, TileMapCell>::Element *E = tile_map.find(p_coords);

	if (!E) {
		return Vector2i();
	}

	return E->get().get_atlas_coords();
}

int TileMap::get_cell_alternative_tile(const Vector2i &p_coords) const {
	// Get a cell source id from position
	const Map<Vector2i, TileMapCell>::Element *E = tile_map.find(p_coords);

	if (!E) {
		return 0;
	}

	return E->get().alternative_tile;
}

TileMapCell &TileMap::get_cell(const Vector2i &p_coords) {
	return tile_map.find(p_coords)->get();
}

Map<Vector2i, TileMapQuadrant> &TileMap::get_quadrant_map() {
	return quadrant_map;
}

void TileMap::fix_invalid_tiles() {
	ERR_FAIL_COND_MSG(tile_set.is_null(), "Cannot fix invalid tiles if Tileset is not open.");
	for (Map<Vector2i, TileMapCell>::Element *E = tile_map.front(); E; E = E->next()) {
		// TODO: this may need to be updated to handle missing scenes, or coordinates outside the tilemap area ?
		if (tile_set->get_source_type(get_cell_source_id(E->key())) != TileSet::SOURCE_TYPE_INVALID) {
			set_cell(E->key(), TileSet::SOURCE_TYPE_INVALID, Vector2i(), 0);
		}
	}
}

void TileMap::_recreate_quadrants() {
	// Clear then recreate all quadrants
	_clear_quadrants();

	for (Map<Vector2i, TileMapCell>::Element *E = tile_map.front(); E; E = E->next()) {
		Vector2i qk = _coords_to_quadrant_coords(Vector2i(E->key().x, E->key().y));

		Map<Vector2i, TileMapQuadrant>::Element *Q = quadrant_map.find(qk);
		if (!Q) {
			Q = _create_quadrant(qk);
			dirty_quadrant_list.add(&Q->get().dirty_list_element);
		}

		Vector2i pk = E->key();
		Q->get().cells.insert(pk);
		Vector2i pk_world_coords = map_to_world(pk);
		Q->get().map_to_world[pk] = pk_world_coords;
		Q->get().world_to_map[pk_world_coords] = pk;
		_make_quadrant_dirty(Q, false);
	}

	update_dirty_quadrants();
}

void TileMap::_clear_quadrants() {
	// Clear quadrants.
	while (quadrant_map.size()) {
		_erase_quadrant(quadrant_map.front());
	}
}

void TileMap::clear() {
	// Remove all tiles.
	_clear_quadrants();
	tile_map.clear();
	used_size_cache_dirty = true;
}

void TileMap::_set_tile_data(const Vector<int> &p_data) {
	// Set data for a given tile from raw data.
	ERR_FAIL_COND(format > FORMAT_3);

	int c = p_data.size();
	const int *r = p_data.ptr();

	int offset = (format >= FORMAT_2) ? 3 : 2;

	clear();
	for (int i = 0; i < c; i += offset) {
		const uint8_t *ptr = (const uint8_t *)&r[i];
		uint8_t local[12];
		for (int j = 0; j < ((format >= FORMAT_2) ? 12 : 8); j++) {
			local[j] = ptr[j];
		}

#ifdef BIG_ENDIAN_ENABLED

		SWAP(local[0], local[3]);
		SWAP(local[1], local[2]);
		SWAP(local[4], local[7]);
		SWAP(local[5], local[6]);
		//TODO: ask someone to check this...
		if (FORMAT >= FORMAT_2) {
			SWAP(local[8], local[11]);
			SWAP(local[9], local[10]);
		}
#endif
		int16_t x = decode_uint16(&local[0]);
		int16_t y = decode_uint16(&local[2]);

		if (format == FORMAT_3) {
			uint16_t source_id = decode_uint16(&local[4]);
			uint16_t atlas_coords_x = decode_uint16(&local[6]);
			uint16_t atlas_coords_y = decode_uint32(&local[8]);
			uint16_t alternative_tile = decode_uint16(&local[10]);
			set_cell(Vector2i(x, y), source_id, Vector2i(atlas_coords_x, atlas_coords_y), alternative_tile);
		} else {
			uint32_t v = decode_uint32(&local[4]);
			v &= (1 << 29) - 1;

			// We generate an alternative tile number out of the the flags
			// An option should create the alternative in the tileset for compatibility
			bool flip_h = v & (1 << 29);
			bool flip_v = v & (1 << 30);
			bool transpose = v & (1 << 31);
			int16_t coord_x = 0;
			int16_t coord_y = 0;
			if (format == FORMAT_2) {
				coord_x = decode_uint16(&local[8]);
				coord_y = decode_uint16(&local[10]);
			}

			int compatibility_alternative_tile = ((int)flip_h) + ((int)flip_v << 1) + ((int)transpose << 2);

			if (tile_set.is_valid()) {
				v = tile_set->compatibility_get_source_for_tile_id(v);
			}

			set_cell(Vector2i(x, y), v, Vector2i(coord_x, coord_y), compatibility_alternative_tile);
		}
	}
}

Vector<int> TileMap::_get_tile_data() const {
	// Export tile data to raw format
	Vector<int> data;
	data.resize(tile_map.size() * 3);
	int *w = data.ptrw();

	// Save in highest format

	int idx = 0;
	for (const Map<Vector2i, TileMapCell>::Element *E = tile_map.front(); E; E = E->next()) {
		uint8_t *ptr = (uint8_t *)&w[idx];
		encode_uint16((int16_t)(E->key().x), &ptr[0]);
		encode_uint16((int16_t)(E->key().y), &ptr[2]);
		encode_uint16(E->get().source_id, &ptr[4]);
		encode_uint16(E->get().coord_x, &ptr[6]);
		encode_uint16(E->get().coord_y, &ptr[8]);
		encode_uint16(E->get().alternative_tile, &ptr[10]);
		idx += 3;
	}

	return data;
}

#ifdef TOOLS_ENABLED
Rect2 TileMap::_edit_get_rect() const {
	// Return the visible rect of the tilemap
	if (pending_update) {
		const_cast<TileMap *>(this)->update_dirty_quadrants();
	} else {
		const_cast<TileMap *>(this)->_recompute_rect_cache();
	}
	return rect_cache;
}
#endif

bool TileMap::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "format") {
		if (p_value.get_type() == Variant::INT) {
			format = (DataFormat)(p_value.operator int64_t()); // Set format used for loading
			return true;
		}
	} else if (p_name == "tile_data") {
		if (p_value.is_array()) {
			_set_tile_data(p_value);
			return true;
		}
		return false;
	}
	return false;
}

bool TileMap::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "format") {
		r_ret = FORMAT_3; // When saving, always save highest format
		return true;
	} else if (p_name == "tile_data") {
		r_ret = _get_tile_data();
		return true;
	}
	return false;
}

void TileMap::_get_property_list(List<PropertyInfo> *p_list) const {
	PropertyInfo p(Variant::INT, "format", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL);
	p_list->push_back(p);

	p = PropertyInfo(Variant::OBJECT, "tile_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL);
	p_list->push_back(p);
}

Vector2 TileMap::map_to_world(const Vector2 &p_pos) const {
	// SHOULD RETURN THE CENTER OF THE TILE
	ERR_FAIL_COND_V(!tile_set.is_valid(), Vector2());

	Vector2 ret = p_pos;
	TileSet::TileShape tile_shape = tile_set->get_tile_shape();
	TileSet::TileOffsetAxis tile_offset_axis = tile_set->get_tile_offset_axis();

	if (tile_shape == TileSet::TILE_SHAPE_HALF_OFFSET_SQUARE || tile_shape == TileSet::TILE_SHAPE_HEXAGON || tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		// Technically, those 3 shapes are equivalent, as they are basically half-offset, but with different levels or overlap.
		// square = no overlap, hexagon = 0.25 overlap, isometric = 0.5 overlap
		if (tile_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			switch (tile_set->get_tile_layout()) {
				case TileSet::TILE_LAYOUT_STACKED:
					ret = Vector2(ret.x + (Math::posmod(ret.y, 2) == 0 ? 0.0 : 0.5), ret.y);
					break;
				case TileSet::TILE_LAYOUT_STACKED_OFFSET:
					ret = Vector2(ret.x + (Math::posmod(ret.y, 2) == 1 ? 0.0 : 0.5), ret.y);
					break;
				case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
					ret = Vector2(ret.x + ret.y / 2, ret.y);
					break;
				case TileSet::TILE_LAYOUT_STAIRS_DOWN:
					ret = Vector2(ret.x / 2, ret.y * 2 + ret.x);
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
					ret = Vector2((ret.x + ret.y) / 2, ret.y - ret.x);
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
					ret = Vector2((ret.x - ret.y) / 2, ret.y + ret.x);
					break;
			}
		} else { // TILE_OFFSET_AXIS_VERTICAL
			switch (tile_set->get_tile_layout()) {
				case TileSet::TILE_LAYOUT_STACKED:
					ret = Vector2(ret.x, ret.y + (Math::posmod(ret.x, 2) == 0 ? 0.0 : 0.5));
					break;
				case TileSet::TILE_LAYOUT_STACKED_OFFSET:
					ret = Vector2(ret.x, ret.y + (Math::posmod(ret.x, 2) == 1 ? 0.0 : 0.5));
					break;
				case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
					ret = Vector2(ret.x * 2 + ret.y, ret.y / 2);
					break;
				case TileSet::TILE_LAYOUT_STAIRS_DOWN:
					ret = Vector2(ret.x, ret.y + ret.x / 2);
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
					ret = Vector2(ret.x + ret.y, (ret.y - ret.x) / 2);
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
					ret = Vector2(ret.x - ret.y, (ret.y + ret.x) / 2);
					break;
			}
		}
	}

	// Multiply by the overlapping ratio
	double overlapping_ratio = 1.0;
	if (tile_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
		if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
			overlapping_ratio = 0.5;
		} else if (tile_shape == TileSet::TILE_SHAPE_HEXAGON) {
			overlapping_ratio = 0.75;
		}
		ret.y *= overlapping_ratio;
	} else { // TILE_OFFSET_AXIS_VERTICAL
		if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
			overlapping_ratio = 0.5;
		} else if (tile_shape == TileSet::TILE_SHAPE_HEXAGON) {
			overlapping_ratio = 0.75;
		}
		ret.x *= overlapping_ratio;
	}

	return (ret + Vector2(0.5, 0.5)) * tile_set->get_tile_size();
}

Vector2i TileMap::world_to_map(const Vector2 &p_pos) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), Vector2());

	Vector2 ret = p_pos;
	ret /= tile_set->get_tile_size();

	TileSet::TileShape tile_shape = tile_set->get_tile_shape();
	TileSet::TileOffsetAxis tile_offset_axis = tile_set->get_tile_offset_axis();
	TileSet::TileLayout tile_layout = tile_set->get_tile_layout();

	// Divide by the overlapping ratio
	double overlapping_ratio = 1.0;
	if (tile_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
		if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
			overlapping_ratio = 0.5;
		} else if (tile_shape == TileSet::TILE_SHAPE_HEXAGON) {
			overlapping_ratio = 0.75;
		}
		ret.y /= overlapping_ratio;
	} else { // TILE_OFFSET_AXIS_VERTICAL
		if (tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
			overlapping_ratio = 0.5;
		} else if (tile_shape == TileSet::TILE_SHAPE_HEXAGON) {
			overlapping_ratio = 0.75;
		}
		ret.x /= overlapping_ratio;
	}

	// For each half-offset shape, we check if we are in the corner of the tile, and thus should correct the world position accordingly.
	if (tile_shape == TileSet::TILE_SHAPE_HALF_OFFSET_SQUARE || tile_shape == TileSet::TILE_SHAPE_HEXAGON || tile_shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		// Technically, those 3 shapes are equivalent, as they are basically half-offset, but with different levels or overlap.
		// square = no overlap, hexagon = 0.25 overlap, isometric = 0.5 overlap
		if (tile_offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			// Smart floor of the position
			Vector2 raw_pos = ret;
			if (Math::posmod(Math::floor(ret.y), 2) ^ (tile_layout == TileSet::TILE_LAYOUT_STACKED_OFFSET)) {
				ret = Vector2(Math::floor(ret.x + 0.5) - 0.5, Math::floor(ret.y));
			} else {
				ret = ret.floor();
			}

			// Compute the tile offset, and if we might the output for a neighbour top tile
			Vector2 in_tile_pos = raw_pos - ret;
			bool in_top_left_triangle = (in_tile_pos - Vector2(0.5, 0.0)).cross(Vector2(-0.5, 1.0 / overlapping_ratio - 1)) <= 0;
			bool in_top_right_triangle = (in_tile_pos - Vector2(0.5, 0.0)).cross(Vector2(0.5, 1.0 / overlapping_ratio - 1)) > 0;

			switch (tile_layout) {
				case TileSet::TILE_LAYOUT_STACKED:
					ret = ret.floor();
					if (in_top_left_triangle) {
						ret += Vector2i(Math::posmod(Math::floor(ret.y), 2) ? 0 : -1, -1);
					} else if (in_top_right_triangle) {
						ret += Vector2i(Math::posmod(Math::floor(ret.y), 2) ? 1 : 0, -1);
					}
					break;
				case TileSet::TILE_LAYOUT_STACKED_OFFSET:
					ret = ret.floor();
					if (in_top_left_triangle) {
						ret += Vector2i(Math::posmod(Math::floor(ret.y), 2) ? -1 : 0, -1);
					} else if (in_top_right_triangle) {
						ret += Vector2i(Math::posmod(Math::floor(ret.y), 2) ? 0 : 1, -1);
					}
					break;
				case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
					ret = Vector2(ret.x - ret.y / 2, ret.y).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(0, -1);
					} else if (in_top_right_triangle) {
						ret += Vector2i(1, -1);
					}
					break;
				case TileSet::TILE_LAYOUT_STAIRS_DOWN:
					ret = Vector2(ret.x * 2, ret.y / 2 - ret.x).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, 0);
					} else if (in_top_right_triangle) {
						ret += Vector2i(1, -1);
					}
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
					ret = Vector2(ret.x - ret.y / 2, ret.y / 2 + ret.x).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(0, -1);
					} else if (in_top_right_triangle) {
						ret += Vector2i(1, 0);
					}
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
					ret = Vector2(ret.x + ret.y / 2, ret.y / 2 - ret.x).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, 0);
					} else if (in_top_right_triangle) {
						ret += Vector2i(0, -1);
					}
					break;
			}
		} else { // TILE_OFFSET_AXIS_VERTICAL
			// Smart floor of the position
			Vector2 raw_pos = ret;
			if (Math::posmod(Math::floor(ret.x), 2) ^ (tile_layout == TileSet::TILE_LAYOUT_STACKED_OFFSET)) {
				ret = Vector2(Math::floor(ret.x), Math::floor(ret.y + 0.5) - 0.5);
			} else {
				ret = ret.floor();
			}

			// Compute the tile offset, and if we might the output for a neighbour top tile
			Vector2 in_tile_pos = raw_pos - ret;
			bool in_top_left_triangle = (in_tile_pos - Vector2(0.0, 0.5)).cross(Vector2(1.0 / overlapping_ratio - 1, -0.5)) > 0;
			bool in_bottom_left_triangle = (in_tile_pos - Vector2(0.0, 0.5)).cross(Vector2(1.0 / overlapping_ratio - 1, 0.5)) <= 0;

			switch (tile_layout) {
				case TileSet::TILE_LAYOUT_STACKED:
					ret = ret.floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, Math::posmod(Math::floor(ret.x), 2) ? 0 : -1);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(-1, Math::posmod(Math::floor(ret.x), 2) ? 1 : 0);
					}
					break;
				case TileSet::TILE_LAYOUT_STACKED_OFFSET:
					ret = ret.floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, Math::posmod(Math::floor(ret.x), 2) ? -1 : 0);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(-1, Math::posmod(Math::floor(ret.x), 2) ? 0 : 1);
					}
					break;
				case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
					ret = Vector2(ret.x / 2 - ret.y, ret.y * 2).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(0, -1);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(-1, 1);
					}
					break;
				case TileSet::TILE_LAYOUT_STAIRS_DOWN:
					ret = Vector2(ret.x, ret.y - ret.x / 2).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, 0);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(-1, 1);
					}
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
					ret = Vector2(ret.x / 2 - ret.y, ret.y + ret.x / 2).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(0, -1);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(-1, 0);
					}
					break;
				case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
					ret = Vector2(ret.x / 2 + ret.y, ret.y - ret.x / 2).floor();
					if (in_top_left_triangle) {
						ret += Vector2i(-1, 0);
					} else if (in_bottom_left_triangle) {
						ret += Vector2i(0, 1);
					}
					break;
			}
		}
	} else {
		ret = (ret + Vector2(0.00005, 0.00005)).floor();
	}
	return ret;
}

TypedArray<Vector2i> TileMap::get_used_cells() const {
	// Returns the cells used in the tilemap.
	TypedArray<Vector2i> a;
	a.resize(tile_map.size());
	int i = 0;
	for (Map<Vector2i, TileMapCell>::Element *E = tile_map.front(); E; E = E->next()) {
		Vector2i p(E->key().x, E->key().y);
		a[i++] = p;
	}

	return a;
}

TypedArray<Vector2i> TileMap::get_used_cells_by_index(int p_source_id, const Vector2i p_atlas_coords, int p_alternative_tile) const {
	// Return all cells matching a given ID.
	TypedArray<Vector2i> a;
	for (Map<Vector2i, TileMapCell>::Element *E = tile_map.front(); E; E = E->next()) {
		if (E->value().source_id == p_source_id && E->value().get_atlas_coords() == p_atlas_coords && E->value().alternative_tile == p_alternative_tile) {
			Vector2i p(E->key().x, E->key().y);
			a.push_back(p);
		}
	}

	return a;
}

Rect2 TileMap::get_used_rect() { // Not const because of cache
	// Return the rect of the currently used area
	if (used_size_cache_dirty) {
		if (tile_map.size() > 0) {
			used_size_cache = Rect2(tile_map.front()->key().x, tile_map.front()->key().y, 0, 0);

			for (Map<Vector2i, TileMapCell>::Element *E = tile_map.front(); E; E = E->next()) {
				used_size_cache.expand_to(Vector2(E->key().x, E->key().y));
			}

			used_size_cache.size += Vector2(1, 1);
		} else {
			used_size_cache = Rect2();
		}

		used_size_cache_dirty = false;
	}

	return used_size_cache;
}

// --- Override some methods of the CanvasItem class to pass the changes to the quadrants CanvasItems ---

void TileMap::set_light_mask(int p_light_mask) {
	// Occlusion: set light mask.
	CanvasItem::set_light_mask(p_light_mask);
	for (Map<Vector2i, TileMapQuadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		for (List<RID>::Element *F = E->get().canvas_items.front(); F; F = F->next()) {
			RenderingServer::get_singleton()->canvas_item_set_light_mask(F->get(), get_light_mask());
		}
	}
}

void TileMap::set_material(const Ref<Material> &p_material) {
	// Set material for the whole tilemap.
	CanvasItem::set_material(p_material);

	// Update material for the whole tilemap.
	for (Map<Vector2i, TileMapQuadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		TileMapQuadrant &q = E->get();
		for (List<RID>::Element *F = q.canvas_items.front(); F; F = F->next()) {
			RS::get_singleton()->canvas_item_set_use_parent_material(F->get(), get_use_parent_material() || get_material().is_valid());
		}
	}
}

void TileMap::set_use_parent_material(bool p_use_parent_material) {
	// Set use_parent_material for the whole tilemap.
	CanvasItem::set_use_parent_material(p_use_parent_material);

	// Update use_parent_material for the whole tilemap.
	for (Map<Vector2i, TileMapQuadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		TileMapQuadrant &q = E->get();
		for (List<RID>::Element *F = q.canvas_items.front(); F; F = F->next()) {
			RS::get_singleton()->canvas_item_set_use_parent_material(F->get(), get_use_parent_material() || get_material().is_valid());
		}
	}
}

void TileMap::set_texture_filter(TextureFilter p_texture_filter) {
	// Set a default texture filter for the whole tilemap
	CanvasItem::set_texture_filter(p_texture_filter);
	for (Map<Vector2i, TileMapQuadrant>::Element *F = quadrant_map.front(); F; F = F->next()) {
		TileMapQuadrant &q = F->get();
		for (List<RID>::Element *E = q.canvas_items.front(); E; E = E->next()) {
			RenderingServer::get_singleton()->canvas_item_set_default_texture_filter(E->get(), RS::CanvasItemTextureFilter(p_texture_filter));
			_make_quadrant_dirty(F);
		}
	}
}

void TileMap::set_texture_repeat(CanvasItem::TextureRepeat p_texture_repeat) {
	// Set a default texture repeat for the whole tilemap
	CanvasItem::set_texture_repeat(p_texture_repeat);
	for (Map<Vector2i, TileMapQuadrant>::Element *F = quadrant_map.front(); F; F = F->next()) {
		TileMapQuadrant &q = F->get();
		for (List<RID>::Element *E = q.canvas_items.front(); E; E = E->next()) {
			RenderingServer::get_singleton()->canvas_item_set_default_texture_repeat(E->get(), RS::CanvasItemTextureRepeat(p_texture_repeat));
			_make_quadrant_dirty(F);
		}
	}
}

void TileMap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_tileset", "tileset"), &TileMap::set_tileset);
	ClassDB::bind_method(D_METHOD("get_tileset"), &TileMap::get_tileset);

	ClassDB::bind_method(D_METHOD("set_quadrant_size", "size"), &TileMap::set_quadrant_size);
	ClassDB::bind_method(D_METHOD("get_quadrant_size"), &TileMap::get_quadrant_size);

	ClassDB::bind_method(D_METHOD("set_cell", "coords", "source_id", "atlas_coords", "alternative_tile"), &TileMap::set_cell, DEFVAL(0), DEFVAL(Vector2i()), DEFVAL(TileSet::SOURCE_TYPE_INVALID));
	ClassDB::bind_method(D_METHOD("get_cell_source_id", "coords"), &TileMap::get_cell_source_id);
	ClassDB::bind_method(D_METHOD("get_cell_atlas_coords", "coords"), &TileMap::get_cell_atlas_coords);
	ClassDB::bind_method(D_METHOD("get_cell_alternative_tile", "coords"), &TileMap::get_cell_alternative_tile);

	ClassDB::bind_method(D_METHOD("fix_invalid_tiles"), &TileMap::fix_invalid_tiles);
	ClassDB::bind_method(D_METHOD("clear"), &TileMap::clear);

	ClassDB::bind_method(D_METHOD("get_used_cells"), &TileMap::get_used_cells);
	ClassDB::bind_method(D_METHOD("get_used_cells_by_index", "source_id", "tileset_coords", "alternative_tile"), &TileMap::get_used_cells_by_index);
	ClassDB::bind_method(D_METHOD("get_used_rect"), &TileMap::get_used_rect);

	ClassDB::bind_method(D_METHOD("map_to_world", "map_position"), &TileMap::map_to_world);
	ClassDB::bind_method(D_METHOD("world_to_map", "world_position"), &TileMap::world_to_map);

	ClassDB::bind_method(D_METHOD("update_dirty_quadrants"), &TileMap::update_dirty_quadrants);

	ClassDB::bind_method(D_METHOD("_set_tile_data"), &TileMap::_set_tile_data);
	ClassDB::bind_method(D_METHOD("_get_tile_data"), &TileMap::_get_tile_data);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "tile_set", PROPERTY_HINT_RESOURCE_TYPE, "TileSet"), "set_tileset", "get_tileset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cell_quadrant_size", PROPERTY_HINT_RANGE, "1,128,1"), "set_quadrant_size", "get_quadrant_size");

	ADD_GROUP("Navigation", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bake_navigation"), "set_bake_navigation", "is_baking_navigation");

	ADD_PROPERTY_DEFAULT("format", FORMAT_1);

	ADD_SIGNAL(MethodInfo("changed"));
}

void TileMap::_tile_set_changed() {
	emit_signal("changed");
}

TileMap::TileMap() {
	rect_cache_dirty = true;
	used_size_cache_dirty = true;
	pending_update = false;
	quadrant_size = 16;
	format = FORMAT_1; // Assume lowest possible format if none is present

	set_notify_transform(true);
	set_notify_local_transform(false);
}

TileMap::~TileMap() {
	clear();
}
