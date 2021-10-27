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

#include "servers/navigation_server_2d.h"

Map<Vector2i, TileSet::CellNeighbor> TileMap::TerrainConstraint::get_overlapping_coords_and_peering_bits() const {
	Map<Vector2i, TileSet::CellNeighbor> output;
	Ref<TileSet> tile_set = tile_map->get_tileset();
	ERR_FAIL_COND_V(!tile_set.is_valid(), output);

	TileSet::TileShape shape = tile_set->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		switch (bit) {
			case 0:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_RIGHT_SIDE;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_LEFT_SIDE;
				break;
			case 1:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER;
				break;
			case 2:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_SIDE;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_SIDE;
				break;
			case 3:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER;
				break;
			default:
				ERR_FAIL_V(output);
		}
	} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		switch (bit) {
			case 0:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_RIGHT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_BOTTOM_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_CORNER)] = TileSet::CELL_NEIGHBOR_LEFT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_CORNER;
				break;
			case 1:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE;
				break;
			case 2:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_LEFT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_CORNER)] = TileSet::CELL_NEIGHBOR_TOP_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_RIGHT_CORNER;
				break;
			case 3:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
				break;
			default:
				ERR_FAIL_V(output);
		}
	} else {
		// Half offset shapes.
		TileSet::TileOffsetAxis offset_axis = tile_set->get_tile_offset_axis();
		if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			switch (bit) {
				case 0:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_RIGHT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_LEFT_SIDE;
					break;
				case 1:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_CORNER;
					break;
				case 2:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE;
					break;
				case 3:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER;
					break;
				case 4:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
					break;
				default:
					ERR_FAIL_V(output);
			}
		} else {
			switch (bit) {
				case 0:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_RIGHT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
					break;
				case 1:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE;
					break;
				case 2:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
					break;
				case 3:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_SIDE;
					break;
				case 4:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
					break;
				default:
					ERR_FAIL_V(output);
			}
		}
	}
	return output;
}

TileMap::TerrainConstraint::TerrainConstraint(const TileMap *p_tile_map, const Vector2i &p_position, const TileSet::CellNeighbor &p_bit, int p_terrain) {
	// The way we build the constraint make it easy to detect conflicting constraints.
	tile_map = p_tile_map;

	Ref<TileSet> tile_set = tile_map->get_tileset();
	ERR_FAIL_COND(!tile_set.is_valid());

	TileSet::TileShape shape = tile_set->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
				bit = 0;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				bit = 1;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
				bit = 2;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				bit = 1;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
				bit = 0;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				bit = 1;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_SIDE:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
				bit = 1;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
				break;
			default:
				ERR_FAIL();
				break;
		}
	} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
				bit = 1;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
				bit = 0;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
				bit = 1;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
				bit = 2;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
				bit = 1;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
				bit = 0;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_CORNER:
				bit = 1;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_CORNER);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
				break;
			default:
				ERR_FAIL();
				break;
		}
	} else {
		// Half-offset shapes
		TileSet::TileOffsetAxis offset_axis = tile_set->get_tile_offset_axis();
		if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			switch (p_bit) {
				case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
					bit = 0;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
					bit = 1;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
					bit = 2;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
					bit = 3;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
					bit = 4;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
					bit = 1;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
					bit = 0;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
					bit = 3;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
					bit = 2;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_CORNER:
					bit = 1;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
					bit = 4;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
					bit = 3;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
					break;
				default:
					ERR_FAIL();
					break;
			}
		} else {
			switch (p_bit) {
				case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
					bit = 0;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
					bit = 1;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
					bit = 2;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
					bit = 3;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
					bit = 0;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
					bit = 4;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
					bit = 2;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
					bit = 1;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
					bit = 0;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_SIDE:
					bit = 3;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
					bit = 2;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
					bit = 4;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
					break;
				default:
					ERR_FAIL();
					break;
			}
		}
	}
	terrain = p_terrain;
}

Vector2i TileMap::transform_coords_layout(Vector2i p_coords, TileSet::TileOffsetAxis p_offset_axis, TileSet::TileLayout p_from_layout, TileSet::TileLayout p_to_layout) {
	// Transform to stacked layout.
	Vector2i output = p_coords;
	if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
		SWAP(output.x, output.y);
	}
	switch (p_from_layout) {
		case TileSet::TILE_LAYOUT_STACKED:
			break;
		case TileSet::TILE_LAYOUT_STACKED_OFFSET:
			if (output.y % 2) {
				output.x -= 1;
			}
			break;
		case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
		case TileSet::TILE_LAYOUT_STAIRS_DOWN:
			if ((p_from_layout == TileSet::TILE_LAYOUT_STAIRS_RIGHT) ^ (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
				if (output.y < 0 && bool(output.y % 2)) {
					output = Vector2i(output.x + output.y / 2 - 1, output.y);
				} else {
					output = Vector2i(output.x + output.y / 2, output.y);
				}
			} else {
				if (output.x < 0 && bool(output.x % 2)) {
					output = Vector2i(output.x / 2 - 1, output.x + output.y * 2);
				} else {
					output = Vector2i(output.x / 2, output.x + output.y * 2);
				}
			}
			break;
		case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
		case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
			if ((p_from_layout == TileSet::TILE_LAYOUT_DIAMOND_RIGHT) ^ (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
				if ((output.x + output.y) < 0 && (output.x - output.y) % 2) {
					output = Vector2i((output.x + output.y) / 2 - 1, output.y - output.x);
				} else {
					output = Vector2i((output.x + output.y) / 2, -output.x + output.y);
				}
			} else {
				if ((output.x - output.y) < 0 && (output.x + output.y) % 2) {
					output = Vector2i((output.x - output.y) / 2 - 1, output.x + output.y);
				} else {
					output = Vector2i((output.x - output.y) / 2, output.x + output.y);
				}
			}
			break;
	}

	switch (p_to_layout) {
		case TileSet::TILE_LAYOUT_STACKED:
			break;
		case TileSet::TILE_LAYOUT_STACKED_OFFSET:
			if (output.y % 2) {
				output.x += 1;
			}
			break;
		case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
		case TileSet::TILE_LAYOUT_STAIRS_DOWN:
			if ((p_to_layout == TileSet::TILE_LAYOUT_STAIRS_RIGHT) ^ (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
				if (output.y < 0 && (output.y % 2)) {
					output = Vector2i(output.x - output.y / 2 + 1, output.y);
				} else {
					output = Vector2i(output.x - output.y / 2, output.y);
				}
			} else {
				if (output.y % 2) {
					if (output.y < 0) {
						output = Vector2i(2 * output.x + 1, -output.x + output.y / 2 - 1);
					} else {
						output = Vector2i(2 * output.x + 1, -output.x + output.y / 2);
					}
				} else {
					output = Vector2i(2 * output.x, -output.x + output.y / 2);
				}
			}
			break;
		case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
		case TileSet::TILE_LAYOUT_DIAMOND_DOWN:
			if ((p_to_layout == TileSet::TILE_LAYOUT_DIAMOND_RIGHT) ^ (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
				if (output.y % 2) {
					if (output.y > 0) {
						output = Vector2i(output.x - output.y / 2, output.x + output.y / 2 + 1);
					} else {
						output = Vector2i(output.x - output.y / 2 + 1, output.x + output.y / 2);
					}
				} else {
					output = Vector2i(output.x - output.y / 2, output.x + output.y / 2);
				}
			} else {
				if (output.y % 2) {
					if (output.y < 0) {
						output = Vector2i(output.x + output.y / 2, -output.x + output.y / 2 - 1);
					} else {
						output = Vector2i(output.x + output.y / 2 + 1, -output.x + output.y / 2);
					}
				} else {
					output = Vector2i(output.x + output.y / 2, -output.x + output.y / 2);
				}
			}
			break;
	}

	if (p_offset_axis == TileSet::TILE_OFFSET_AXIS_VERTICAL) {
		SWAP(output.x, output.y);
	}

	return output;
}

int TileMap::get_effective_quadrant_size(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), 1);

	// When using YSort, the quadrant size is reduced to 1 to have one CanvasItem per quadrant
	if (is_y_sort_enabled() && layers[p_layer].y_sort_enabled) {
		return 1;
	} else {
		return quadrant_size;
	}
}

void TileMap::set_selected_layer(int p_layer_id) {
	ERR_FAIL_COND(p_layer_id < -1 || p_layer_id >= (int)layers.size());
	selected_layer = p_layer_id;
	emit_signal(SNAME("changed"));
	_make_all_quadrants_dirty();
}

int TileMap::get_selected_layer() const {
	return selected_layer;
}

void TileMap::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_clear_internals();
			_recreate_internals();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			_clear_internals();
		} break;
	}

	// Transfers the notification to tileset plugins.
	if (tile_set.is_valid()) {
		_rendering_notification(p_what);
		_physics_notification(p_what);
		_navigation_notification(p_what);
	}
}

Ref<TileSet> TileMap::get_tileset() const {
	return tile_set;
}

void TileMap::set_tileset(const Ref<TileSet> &p_tileset) {
	if (p_tileset == tile_set) {
		return;
	}

	// Set the tileset, registering to its changes.
	if (tile_set.is_valid()) {
		tile_set->disconnect("changed", callable_mp(this, &TileMap::_tile_set_changed));
	}

	if (!p_tileset.is_valid()) {
		_clear_internals();
	}

	tile_set = p_tileset;

	if (tile_set.is_valid()) {
		tile_set->connect("changed", callable_mp(this, &TileMap::_tile_set_changed));
		_clear_internals();
		_recreate_internals();
	}

	emit_signal(SNAME("changed"));
}

void TileMap::set_quadrant_size(int p_size) {
	ERR_FAIL_COND_MSG(p_size < 1, "TileMapQuadrant size cannot be smaller than 1.");

	quadrant_size = p_size;
	_clear_internals();
	_recreate_internals();
	emit_signal(SNAME("changed"));
}

int TileMap::get_quadrant_size() const {
	return quadrant_size;
}

int TileMap::get_layers_count() const {
	return layers.size();
}

void TileMap::add_layer(int p_to_pos) {
	if (p_to_pos < 0) {
		p_to_pos = layers.size();
	}

	ERR_FAIL_INDEX(p_to_pos, (int)layers.size() + 1);

	// Must clear before adding the layer.
	_clear_internals();

	layers.insert(p_to_pos, TileMapLayer());
	_recreate_internals();
	notify_property_list_changed();

	emit_signal(SNAME("changed"));

	update_configuration_warnings();
}

void TileMap::move_layer(int p_layer, int p_to_pos) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	ERR_FAIL_INDEX(p_to_pos, (int)layers.size() + 1);

	// Clear before shuffling layers.
	_clear_internals();

	TileMapLayer tl = layers[p_layer];
	layers.insert(p_to_pos, tl);
	layers.remove(p_to_pos < p_layer ? p_layer + 1 : p_layer);
	_recreate_internals();
	notify_property_list_changed();

	if (selected_layer == p_layer) {
		selected_layer = p_to_pos < p_layer ? p_to_pos - 1 : p_to_pos;
	}

	emit_signal(SNAME("changed"));

	update_configuration_warnings();
}

void TileMap::remove_layer(int p_layer) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	// Clear before removing the layer.
	_clear_internals();

	layers.remove(p_layer);
	_recreate_internals();
	notify_property_list_changed();

	if (selected_layer >= p_layer) {
		selected_layer -= 1;
	}

	emit_signal(SNAME("changed"));

	update_configuration_warnings();
}

void TileMap::set_layer_name(int p_layer, String p_name) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	layers[p_layer].name = p_name;
	emit_signal(SNAME("changed"));
}

String TileMap::get_layer_name(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), String());
	return layers[p_layer].name;
}

void TileMap::set_layer_enabled(int p_layer, bool p_enabled) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	layers[p_layer].enabled = p_enabled;
	_clear_layer_internals(p_layer);
	_recreate_layer_internals(p_layer);
	emit_signal(SNAME("changed"));

	update_configuration_warnings();
}

bool TileMap::is_layer_enabled(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), false);
	return layers[p_layer].enabled;
}

void TileMap::set_layer_modulate(int p_layer, Color p_modulate) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	layers[p_layer].modulate = p_modulate;
	_clear_layer_internals(p_layer);
	_recreate_layer_internals(p_layer);
	emit_signal(SNAME("changed"));
}

Color TileMap::get_layer_modulate(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), Color());
	return layers[p_layer].modulate;
}

void TileMap::set_layer_y_sort_enabled(int p_layer, bool p_y_sort_enabled) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	layers[p_layer].y_sort_enabled = p_y_sort_enabled;
	_clear_layer_internals(p_layer);
	_recreate_layer_internals(p_layer);
	emit_signal(SNAME("changed"));

	update_configuration_warnings();
}

bool TileMap::is_layer_y_sort_enabled(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), false);
	return layers[p_layer].y_sort_enabled;
}

void TileMap::set_layer_y_sort_origin(int p_layer, int p_y_sort_origin) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	layers[p_layer].y_sort_origin = p_y_sort_origin;
	_clear_layer_internals(p_layer);
	_recreate_layer_internals(p_layer);
	emit_signal(SNAME("changed"));
}

int TileMap::get_layer_y_sort_origin(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), false);
	return layers[p_layer].y_sort_origin;
}

void TileMap::set_layer_z_index(int p_layer, int p_z_index) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	layers[p_layer].z_index = p_z_index;
	_clear_layer_internals(p_layer);
	_recreate_layer_internals(p_layer);
	emit_signal(SNAME("changed"));

	update_configuration_warnings();
}

int TileMap::get_layer_z_index(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), false);
	return layers[p_layer].z_index;
}

void TileMap::set_collision_animatable(bool p_enabled) {
	collision_animatable = p_enabled;
	_clear_internals();
	set_notify_local_transform(p_enabled);
	set_physics_process_internal(p_enabled);
	_recreate_internals();
	emit_signal(SNAME("changed"));
}

bool TileMap::is_collision_animatable() const {
	return collision_animatable;
}

void TileMap::set_collision_visibility_mode(TileMap::VisibilityMode p_show_collision) {
	collision_visibility_mode = p_show_collision;
	_clear_internals();
	_recreate_internals();
	emit_signal(SNAME("changed"));
}

TileMap::VisibilityMode TileMap::get_collision_visibility_mode() {
	return collision_visibility_mode;
}

void TileMap::set_navigation_visibility_mode(TileMap::VisibilityMode p_show_navigation) {
	navigation_visibility_mode = p_show_navigation;
	_clear_internals();
	_recreate_internals();
	emit_signal(SNAME("changed"));
}

TileMap::VisibilityMode TileMap::get_navigation_visibility_mode() {
	return navigation_visibility_mode;
}

void TileMap::set_y_sort_enabled(bool p_enable) {
	Node2D::set_y_sort_enabled(p_enable);
	_clear_internals();
	_recreate_internals();
	emit_signal(SNAME("changed"));
}

Vector2i TileMap::_coords_to_quadrant_coords(int p_layer, const Vector2i &p_coords) const {
	int quadrant_size = get_effective_quadrant_size(p_layer);

	// Rounding down, instead of simply rounding towards zero (truncating)
	return Vector2i(
			p_coords.x > 0 ? p_coords.x / quadrant_size : (p_coords.x - (quadrant_size - 1)) / quadrant_size,
			p_coords.y > 0 ? p_coords.y / quadrant_size : (p_coords.y - (quadrant_size - 1)) / quadrant_size);
}

Map<Vector2i, TileMapQuadrant>::Element *TileMap::_create_quadrant(int p_layer, const Vector2i &p_qk) {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), nullptr);

	TileMapQuadrant q;
	q.layer = p_layer;
	q.coords = p_qk;

	rect_cache_dirty = true;

	// Create the debug canvas item.
	RenderingServer *rs = RenderingServer::get_singleton();
	q.debug_canvas_item = rs->canvas_item_create();
	rs->canvas_item_set_z_index(q.debug_canvas_item, RS::CANVAS_ITEM_Z_MAX - 1);
	rs->canvas_item_set_parent(q.debug_canvas_item, get_canvas_item());

	// Call the create_quadrant method on plugins
	if (tile_set.is_valid()) {
		_rendering_create_quadrant(&q);
	}

	return layers[p_layer].quadrant_map.insert(p_qk, q);
}

void TileMap::_make_quadrant_dirty(Map<Vector2i, TileMapQuadrant>::Element *Q) {
	// Make the given quadrant dirty, then trigger an update later.
	TileMapQuadrant &q = Q->get();
	if (!q.dirty_list_element.in_list()) {
		layers[q.layer].dirty_quadrant_list.add(&q.dirty_list_element);
	}
	_queue_update_dirty_quadrants();
}

void TileMap::_make_all_quadrants_dirty() {
	// Make all quandrants dirty, then trigger an update later.
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		for (KeyValue<Vector2i, TileMapQuadrant> &E : layers[layer].quadrant_map) {
			if (!E.value.dirty_list_element.in_list()) {
				layers[layer].dirty_quadrant_list.add(&E.value.dirty_list_element);
			}
		}
	}
	_queue_update_dirty_quadrants();
}

void TileMap::_queue_update_dirty_quadrants() {
	if (pending_update || !is_inside_tree()) {
		return;
	}
	pending_update = true;
	call_deferred(SNAME("_update_dirty_quadrants"));
}

void TileMap::_update_dirty_quadrants() {
	if (!pending_update) {
		return;
	}
	if (!is_inside_tree() || !tile_set.is_valid()) {
		pending_update = false;
		return;
	}

	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		// Update the coords cache.
		for (SelfList<TileMapQuadrant> *q = layers[layer].dirty_quadrant_list.first(); q; q = q->next()) {
			q->self()->map_to_world.clear();
			q->self()->world_to_map.clear();
			for (Set<Vector2i>::Element *E = q->self()->cells.front(); E; E = E->next()) {
				Vector2i pk = E->get();
				Vector2i pk_world_coords = map_to_world(pk);
				q->self()->map_to_world[pk] = pk_world_coords;
				q->self()->world_to_map[pk_world_coords] = pk;
			}
		}

		// Call the update_dirty_quadrant method on plugins.
		_rendering_update_dirty_quadrants(layers[layer].dirty_quadrant_list);
		_physics_update_dirty_quadrants(layers[layer].dirty_quadrant_list);
		_navigation_update_dirty_quadrants(layers[layer].dirty_quadrant_list);
		_scenes_update_dirty_quadrants(layers[layer].dirty_quadrant_list);

		// Redraw the debug canvas_items.
		RenderingServer *rs = RenderingServer::get_singleton();
		for (SelfList<TileMapQuadrant> *q = layers[layer].dirty_quadrant_list.first(); q; q = q->next()) {
			rs->canvas_item_clear(q->self()->debug_canvas_item);
			Transform2D xform;
			xform.set_origin(map_to_world(q->self()->coords * get_effective_quadrant_size(layer)));
			rs->canvas_item_set_transform(q->self()->debug_canvas_item, xform);

			_rendering_draw_quadrant_debug(q->self());
			_physics_draw_quadrant_debug(q->self());
			_navigation_draw_quadrant_debug(q->self());
			_scenes_draw_quadrant_debug(q->self());
		}

		// Clear the list
		while (layers[layer].dirty_quadrant_list.first()) {
			layers[layer].dirty_quadrant_list.remove(layers[layer].dirty_quadrant_list.first());
		}
	}

	pending_update = false;

	_recompute_rect_cache();
}

void TileMap::_recreate_layer_internals(int p_layer) {
	// Make sure that _clear_internals() was called prior.
	ERR_FAIL_COND_MSG(layers[p_layer].quadrant_map.size() > 0, "TileMap layer " + itos(p_layer) + " had a non-empty quadrant map.");

	if (!layers[p_layer].enabled) {
		return;
	}

	// Upadate the layer internals.
	_rendering_update_layer(p_layer);

	// Recreate the quadrants.
	const Map<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	for (const KeyValue<Vector2i, TileMapCell> &E : tile_map) {
		Vector2i qk = _coords_to_quadrant_coords(p_layer, Vector2i(E.key.x, E.key.y));

		Map<Vector2i, TileMapQuadrant>::Element *Q = layers[p_layer].quadrant_map.find(qk);
		if (!Q) {
			Q = _create_quadrant(p_layer, qk);
			layers[p_layer].dirty_quadrant_list.add(&Q->get().dirty_list_element);
		}

		Vector2i pk = E.key;
		Q->get().cells.insert(pk);

		_make_quadrant_dirty(Q);
	}

	_queue_update_dirty_quadrants();
}

void TileMap::_recreate_internals() {
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		_recreate_layer_internals(layer);
	}
}

void TileMap::_erase_quadrant(Map<Vector2i, TileMapQuadrant>::Element *Q) {
	// Remove a quadrant.
	TileMapQuadrant *q = &(Q->get());

	// Call the cleanup_quadrant method on plugins.
	if (tile_set.is_valid()) {
		_rendering_cleanup_quadrant(q);
		_physics_cleanup_quadrant(q);
		_navigation_cleanup_quadrant(q);
		_scenes_cleanup_quadrant(q);
	}

	// Remove the quadrant from the dirty_list if it is there.
	if (q->dirty_list_element.in_list()) {
		layers[q->layer].dirty_quadrant_list.remove(&(q->dirty_list_element));
	}

	// Free the debug canvas item.
	RenderingServer *rs = RenderingServer::get_singleton();
	rs->free(q->debug_canvas_item);

	layers[q->layer].quadrant_map.erase(Q);
	rect_cache_dirty = true;
}

void TileMap::_clear_layer_internals(int p_layer) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	// Clear quadrants.
	while (layers[p_layer].quadrant_map.size()) {
		_erase_quadrant(layers[p_layer].quadrant_map.front());
	}

	// Clear the layers internals.
	_rendering_cleanup_layer(p_layer);

	// Clear the dirty quadrants list.
	while (layers[p_layer].dirty_quadrant_list.first()) {
		layers[p_layer].dirty_quadrant_list.remove(layers[p_layer].dirty_quadrant_list.first());
	}
}

void TileMap::_clear_internals() {
	// Clear quadrants.
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		_clear_layer_internals(layer);
	}
}

void TileMap::_recompute_rect_cache() {
	// Compute the displayed area of the tilemap.
#ifdef DEBUG_ENABLED

	if (!rect_cache_dirty) {
		return;
	}

	Rect2 r_total;
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		for (const Map<Vector2i, TileMapQuadrant>::Element *E = layers[layer].quadrant_map.front(); E; E = E->next()) {
			Rect2 r;
			r.position = map_to_world(E->key() * get_effective_quadrant_size(layer));
			r.expand_to(map_to_world((E->key() + Vector2i(1, 0)) * get_effective_quadrant_size(layer)));
			r.expand_to(map_to_world((E->key() + Vector2i(1, 1)) * get_effective_quadrant_size(layer)));
			r.expand_to(map_to_world((E->key() + Vector2i(0, 1)) * get_effective_quadrant_size(layer)));
			if (E == layers[layer].quadrant_map.front()) {
				r_total = r;
			} else {
				r_total = r_total.merge(r);
			}
		}
	}

	rect_cache = r_total;

	item_rect_changed();

	rect_cache_dirty = false;
#endif
}

/////////////////////////////// Rendering //////////////////////////////////////

void TileMap::_rendering_notification(int p_what) {
	switch (p_what) {
		case CanvasItem::NOTIFICATION_VISIBILITY_CHANGED: {
			bool visible = is_visible_in_tree();
			for (int layer = 0; layer < (int)layers.size(); layer++) {
				for (KeyValue<Vector2i, TileMapQuadrant> &E_quadrant : layers[layer].quadrant_map) {
					TileMapQuadrant &q = E_quadrant.value;

					// Update occluders transform.
					for (const KeyValue<Vector2i, Vector2i> &E_cell : q.world_to_map) {
						Transform2D xform;
						xform.set_origin(E_cell.key);
						for (const RID &occluder : q.occluders) {
							RS::get_singleton()->canvas_light_occluder_set_enabled(occluder, visible);
						}
					}
				}
			}
		} break;
		case CanvasItem::NOTIFICATION_TRANSFORM_CHANGED: {
			if (!is_inside_tree()) {
				return;
			}
			for (int layer = 0; layer < (int)layers.size(); layer++) {
				for (KeyValue<Vector2i, TileMapQuadrant> &E_quadrant : layers[layer].quadrant_map) {
					TileMapQuadrant &q = E_quadrant.value;

					// Update occluders transform.
					for (const KeyValue<Vector2i, Vector2i> &E_cell : q.world_to_map) {
						Transform2D xform;
						xform.set_origin(E_cell.key);
						for (const RID &occluder : q.occluders) {
							RS::get_singleton()->canvas_light_occluder_set_transform(occluder, get_global_transform() * xform);
						}
					}
				}
			}
		} break;
		case CanvasItem::NOTIFICATION_DRAW: {
			if (tile_set.is_valid()) {
				RenderingServer::get_singleton()->canvas_item_set_sort_children_by_y(get_canvas_item(), is_y_sort_enabled());
			}
		} break;
	}
}

void TileMap::_rendering_update_layer(int p_layer) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	RenderingServer *rs = RenderingServer::get_singleton();
	if (!layers[p_layer].canvas_item.is_valid()) {
		RID ci = rs->canvas_item_create();
		rs->canvas_item_set_parent(ci, get_canvas_item());

		/*Transform2D xform;
		xform.set_origin(Vector2(0, p_layer));
		rs->canvas_item_set_transform(ci, xform);*/
		rs->canvas_item_set_draw_index(ci, p_layer);

		layers[p_layer].canvas_item = ci;
	}
	RID &ci = layers[p_layer].canvas_item;
	rs->canvas_item_set_sort_children_by_y(ci, layers[p_layer].y_sort_enabled);
	rs->canvas_item_set_use_parent_material(ci, get_use_parent_material() || get_material().is_valid());
	rs->canvas_item_set_z_index(ci, layers[p_layer].z_index);
	rs->canvas_item_set_default_texture_filter(ci, RS::CanvasItemTextureFilter(get_texture_filter()));
	rs->canvas_item_set_default_texture_repeat(ci, RS::CanvasItemTextureRepeat(get_texture_repeat()));
	rs->canvas_item_set_light_mask(ci, get_light_mask());
}

void TileMap::_rendering_cleanup_layer(int p_layer) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	RenderingServer *rs = RenderingServer::get_singleton();
	if (layers[p_layer].canvas_item.is_valid()) {
		rs->free(layers[p_layer].canvas_item);
		layers[p_layer].canvas_item = RID();
	}
}

void TileMap::_rendering_update_dirty_quadrants(SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(!tile_set.is_valid());

	bool visible = is_visible_in_tree();

	SelfList<TileMapQuadrant> *q_list_element = r_dirty_quadrant_list.first();
	while (q_list_element) {
		TileMapQuadrant &q = *q_list_element->self();

		RenderingServer *rs = RenderingServer::get_singleton();

		// Free the canvas items.
		for (const RID &ci : q.canvas_items) {
			rs->free(ci);
		}
		q.canvas_items.clear();

		// Free the occluders.
		for (const RID &occluder : q.occluders) {
			rs->free(occluder);
		}
		q.occluders.clear();

		// Those allow to group cell per material or z-index.
		Ref<ShaderMaterial> prev_material;
		int prev_z_index = 0;
		RID prev_canvas_item;

		Color modulate = get_self_modulate();
		modulate *= get_layer_modulate(q.layer);
		if (selected_layer >= 0) {
			int z1 = get_layer_z_index(q.layer);
			int z2 = get_layer_z_index(selected_layer);
			if (z1 < z2 || (z1 == z2 && q.layer < selected_layer)) {
				modulate = modulate.darkened(0.5);
			} else if (z1 > z2 || (z1 == z2 && q.layer > selected_layer)) {
				modulate = modulate.darkened(0.5);
				modulate.a *= 0.3;
			}
		}

		// Iterate over the cells of the quadrant.
		for (const KeyValue<Vector2i, Vector2i> &E_cell : q.world_to_map) {
			TileMapCell c = get_cell(q.layer, E_cell.value, true);

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					// Get the tile data.
					TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile));
					Ref<ShaderMaterial> mat = tile_data->get_material();
					int z_index = tile_data->get_z_index();

					// Quandrant pos.
					Vector2 position = map_to_world(q.coords * get_effective_quadrant_size(q.layer));
					if (is_y_sort_enabled() && layers[q.layer].y_sort_enabled) {
						// When Y-sorting, the quandrant size is sure to be 1, we can thus offset the CanvasItem.
						position.y += layers[q.layer].y_sort_origin + tile_data->get_y_sort_origin();
					}

					// --- CanvasItems ---
					// Create two canvas items, for rendering and debug.
					RID canvas_item;

					// Check if the material or the z_index changed.
					if (prev_canvas_item == RID() || prev_material != mat || prev_z_index != z_index) {
						// If so, create a new CanvasItem.
						canvas_item = rs->canvas_item_create();
						if (mat.is_valid()) {
							rs->canvas_item_set_material(canvas_item, mat->get_rid());
						}
						rs->canvas_item_set_parent(canvas_item, layers[q.layer].canvas_item);
						rs->canvas_item_set_use_parent_material(canvas_item, get_use_parent_material() || get_material().is_valid());

						Transform2D xform;
						xform.set_origin(position);
						rs->canvas_item_set_transform(canvas_item, xform);

						rs->canvas_item_set_light_mask(canvas_item, get_light_mask());
						rs->canvas_item_set_z_index(canvas_item, z_index);

						rs->canvas_item_set_default_texture_filter(canvas_item, RS::CanvasItemTextureFilter(get_texture_filter()));
						rs->canvas_item_set_default_texture_repeat(canvas_item, RS::CanvasItemTextureRepeat(get_texture_repeat()));

						q.canvas_items.push_back(canvas_item);

						prev_canvas_item = canvas_item;
						prev_material = mat;
						prev_z_index = z_index;

					} else {
						// Keep the same canvas_item to draw on.
						canvas_item = prev_canvas_item;
					}

					// Drawing the tile in the canvas item.
					draw_tile(canvas_item, E_cell.key - position, tile_set, c.source_id, c.get_atlas_coords(), c.alternative_tile, -1, modulate);

					// --- Occluders ---
					for (int i = 0; i < tile_set->get_occlusion_layers_count(); i++) {
						Transform2D xform;
						xform.set_origin(E_cell.key);
						if (tile_data->get_occluder(i).is_valid()) {
							RID occluder_id = rs->canvas_light_occluder_create();
							rs->canvas_light_occluder_set_enabled(occluder_id, visible);
							rs->canvas_light_occluder_set_transform(occluder_id, get_global_transform() * xform);
							rs->canvas_light_occluder_set_polygon(occluder_id, tile_data->get_occluder(i)->get_rid());
							rs->canvas_light_occluder_attach_to_canvas(occluder_id, get_canvas());
							rs->canvas_light_occluder_set_light_mask(occluder_id, tile_set->get_occlusion_layer_light_mask(i));
							q.occluders.push_back(occluder_id);
						}
					}
				}
			}
		}

		_rendering_quadrant_order_dirty = true;
		q_list_element = q_list_element->next();
	}

	// Reset the drawing indices
	if (_rendering_quadrant_order_dirty) {
		int index = -(int64_t)0x80000000; //always must be drawn below children.

		for (int layer = 0; layer < (int)layers.size(); layer++) {
			// Sort the quadrants coords per world coordinates
			Map<Vector2i, Vector2i, TileMapQuadrant::CoordsWorldComparator> world_to_map;
			for (const KeyValue<Vector2i, TileMapQuadrant> &E : layers[layer].quadrant_map) {
				world_to_map[map_to_world(E.key)] = E.key;
			}

			// Sort the quadrants
			for (const KeyValue<Vector2i, Vector2i> &E : world_to_map) {
				TileMapQuadrant &q = layers[layer].quadrant_map[E.value];
				for (const RID &ci : q.canvas_items) {
					RS::get_singleton()->canvas_item_set_draw_index(ci, index++);
				}
			}
		}
		_rendering_quadrant_order_dirty = false;
	}
}

void TileMap::_rendering_create_quadrant(TileMapQuadrant *p_quadrant) {
	ERR_FAIL_COND(!tile_set.is_valid());

	_rendering_quadrant_order_dirty = true;
}

void TileMap::_rendering_cleanup_quadrant(TileMapQuadrant *p_quadrant) {
	// Free the canvas items.
	for (const RID &ci : p_quadrant->canvas_items) {
		RenderingServer::get_singleton()->free(ci);
	}
	p_quadrant->canvas_items.clear();

	// Free the occluders.
	for (const RID &occluder : p_quadrant->occluders) {
		RenderingServer::get_singleton()->free(occluder);
	}
	p_quadrant->occluders.clear();
}

void TileMap::_rendering_draw_quadrant_debug(TileMapQuadrant *p_quadrant) {
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	// Draw a placeholder for scenes needing one.
	RenderingServer *rs = RenderingServer::get_singleton();
	Vector2 quadrant_pos = map_to_world(p_quadrant->coords * get_effective_quadrant_size(p_quadrant->layer));
	for (Set<Vector2i>::Element *E_cell = p_quadrant->cells.front(); E_cell; E_cell = E_cell->next()) {
		const TileMapCell &c = get_cell(p_quadrant->layer, E_cell->get(), true);

		TileSetSource *source;
		if (tile_set->has_source(c.source_id)) {
			source = *tile_set->get_source(c.source_id);

			if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
				continue;
			}

			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				Vector2i grid_size = atlas_source->get_atlas_grid_size();
				if (!atlas_source->get_texture().is_valid() || c.get_atlas_coords().x >= grid_size.x || c.get_atlas_coords().y >= grid_size.y) {
					// Generate a random color from the hashed values of the tiles.
					Array to_hash;
					to_hash.push_back(c.source_id);
					to_hash.push_back(c.get_atlas_coords());
					to_hash.push_back(c.alternative_tile);
					uint32_t hash = RandomPCG(to_hash.hash()).rand();

					Color color;
					color = color.from_hsv(
							(float)((hash >> 24) & 0xFF) / 256.0,
							Math::lerp(0.5, 1.0, (float)((hash >> 16) & 0xFF) / 256.0),
							Math::lerp(0.5, 1.0, (float)((hash >> 8) & 0xFF) / 256.0),
							0.8);

					// Draw a placeholder tile.
					Transform2D xform;
					xform.set_origin(map_to_world(E_cell->get()) - quadrant_pos);
					rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, xform);
					rs->canvas_item_add_circle(p_quadrant->debug_canvas_item, Vector2(), MIN(tile_set->get_tile_size().x, tile_set->get_tile_size().y) / 4.0, color);
				}
			}
		}
	}
}

void TileMap::draw_tile(RID p_canvas_item, Vector2i p_position, const Ref<TileSet> p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, int p_frame, Color p_modulation) {
	ERR_FAIL_COND(!p_tile_set.is_valid());
	ERR_FAIL_COND(!p_tile_set->has_source(p_atlas_source_id));
	ERR_FAIL_COND(!p_tile_set->get_source(p_atlas_source_id)->has_tile(p_atlas_coords));
	ERR_FAIL_COND(!p_tile_set->get_source(p_atlas_source_id)->has_alternative_tile(p_atlas_coords, p_alternative_tile));
	TileSetSource *source = *p_tile_set->get_source(p_atlas_source_id);
	TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
	if (atlas_source) {
		// Check for the frame.
		if (p_frame >= 0) {
			ERR_FAIL_INDEX(p_frame, atlas_source->get_tile_animation_frames_count(p_atlas_coords));
		}

		// Get the texture.
		Ref<Texture2D> tex = atlas_source->get_texture();
		if (!tex.is_valid()) {
			return;
		}

		// Check if we are in the texture, return otherwise.
		Vector2i grid_size = atlas_source->get_atlas_grid_size();
		if (p_atlas_coords.x >= grid_size.x || p_atlas_coords.y >= grid_size.y) {
			return;
		}

		// Get tile data.
		TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(p_atlas_coords, p_alternative_tile));

		// Get the tile modulation.
		Color modulate = tile_data->get_modulate() * p_modulation;

		// Compute the offset.
		Vector2i tile_offset = atlas_source->get_tile_effective_texture_offset(p_atlas_coords, p_alternative_tile);

		// Get destination rect.
		Rect2 dest_rect;
		dest_rect.size = atlas_source->get_tile_texture_region(p_atlas_coords).size;
		dest_rect.size.x += FP_ADJUST;
		dest_rect.size.y += FP_ADJUST;

		bool transpose = tile_data->get_transpose();
		if (transpose) {
			dest_rect.position = (p_position - Vector2(dest_rect.size.y, dest_rect.size.x) / 2 - tile_offset);
		} else {
			dest_rect.position = (p_position - dest_rect.size / 2 - tile_offset);
		}

		if (tile_data->get_flip_h()) {
			dest_rect.size.x = -dest_rect.size.x;
		}

		if (tile_data->get_flip_v()) {
			dest_rect.size.y = -dest_rect.size.y;
		}

		// Draw the tile.
		if (p_frame >= 0) {
			Rect2i source_rect = atlas_source->get_tile_texture_region(p_atlas_coords, p_frame);
			tex->draw_rect_region(p_canvas_item, dest_rect, source_rect, modulate, transpose, p_tile_set->is_uv_clipping());
		} else if (atlas_source->get_tile_animation_frames_count(p_atlas_coords) == 1) {
			Rect2i source_rect = atlas_source->get_tile_texture_region(p_atlas_coords, 0);
			tex->draw_rect_region(p_canvas_item, dest_rect, source_rect, modulate, transpose, p_tile_set->is_uv_clipping());
		} else {
			real_t speed = atlas_source->get_tile_animation_speed(p_atlas_coords);
			real_t animation_duration = atlas_source->get_tile_animation_total_duration(p_atlas_coords) / speed;
			real_t time = 0.0;
			for (int frame = 0; frame < atlas_source->get_tile_animation_frames_count(p_atlas_coords); frame++) {
				real_t frame_duration = atlas_source->get_tile_animation_frame_duration(p_atlas_coords, frame) / speed;
				RenderingServer::get_singleton()->canvas_item_add_animation_slice(p_canvas_item, animation_duration, time, time + frame_duration, 0.0);

				Rect2i source_rect = atlas_source->get_tile_texture_region(p_atlas_coords, frame);
				tex->draw_rect_region(p_canvas_item, dest_rect, source_rect, modulate, transpose, p_tile_set->is_uv_clipping());

				time += frame_duration;
			}
			RenderingServer::get_singleton()->canvas_item_add_animation_slice(p_canvas_item, 1.0, 0.0, 1.0, 0.0);
		}
	}
}

/////////////////////////////// Physics //////////////////////////////////////

void TileMap::_physics_notification(int p_what) {
	switch (p_what) {
		case CanvasItem::NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			bool in_editor = false;
#ifdef TOOLS_ENABLED
			in_editor = Engine::get_singleton()->is_editor_hint();
#endif
			if (is_inside_tree() && collision_animatable && !in_editor) {
				// Update tranform on the physics tick when in animatable mode.
				last_valid_transform = new_transform;
				set_notify_local_transform(false);
				set_global_transform(new_transform);
				set_notify_local_transform(true);
			}
		} break;
		case CanvasItem::NOTIFICATION_TRANSFORM_CHANGED: {
			bool in_editor = false;
#ifdef TOOLS_ENABLED
			in_editor = Engine::get_singleton()->is_editor_hint();
#endif
			if (is_inside_tree() && (!collision_animatable || in_editor)) {
				// Update the new transform directly if we are not in animatable mode.
				Transform2D global_transform = get_global_transform();
				for (int layer = 0; layer < (int)layers.size(); layer++) {
					for (KeyValue<Vector2i, TileMapQuadrant> &E : layers[layer].quadrant_map) {
						TileMapQuadrant &q = E.value;

						for (RID body : q.bodies) {
							Transform2D xform;
							xform.set_origin(map_to_world(bodies_coords[body]));
							xform = global_transform * xform;
							PhysicsServer2D::get_singleton()->body_set_state(body, PhysicsServer2D::BODY_STATE_TRANSFORM, xform);
						}
					}
				}
			}
		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			bool in_editor = false;
#ifdef TOOLS_ENABLED
			in_editor = Engine::get_singleton()->is_editor_hint();
#endif
			if (is_inside_tree() && !in_editor && collision_animatable) {
				// Only active when animatable. Send the new transform to the physics...
				new_transform = get_global_transform();
				for (int layer = 0; layer < (int)layers.size(); layer++) {
					for (KeyValue<Vector2i, TileMapQuadrant> &E : layers[layer].quadrant_map) {
						TileMapQuadrant &q = E.value;

						for (RID body : q.bodies) {
							Transform2D xform;
							xform.set_origin(map_to_world(bodies_coords[body]));
							xform = new_transform * xform;

							PhysicsServer2D::get_singleton()->body_set_state(body, PhysicsServer2D::BODY_STATE_TRANSFORM, xform);
						}
					}
				}

				// ... but then revert changes.
				set_notify_local_transform(false);
				set_global_transform(last_valid_transform);
				set_notify_local_transform(true);
			}
		} break;
	}
}

void TileMap::_physics_update_dirty_quadrants(SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(!tile_set.is_valid());

	Transform2D global_transform = get_global_transform();
	last_valid_transform = global_transform;
	new_transform = global_transform;
	PhysicsServer2D *ps = PhysicsServer2D::get_singleton();
	RID space = get_world_2d()->get_space();

	SelfList<TileMapQuadrant> *q_list_element = r_dirty_quadrant_list.first();
	while (q_list_element) {
		TileMapQuadrant &q = *q_list_element->self();

		// Clear bodies.
		for (RID body : q.bodies) {
			bodies_coords.erase(body);
			ps->free(body);
		}
		q.bodies.clear();

		// Recreate bodies and shapes.
		for (Set<Vector2i>::Element *E_cell = q.cells.front(); E_cell; E_cell = E_cell->next()) {
			TileMapCell c = get_cell(q.layer, E_cell->get(), true);

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile));

					for (int tile_set_physics_layer = 0; tile_set_physics_layer < tile_set->get_physics_layers_count(); tile_set_physics_layer++) {
						Ref<PhysicsMaterial> physics_material = tile_set->get_physics_layer_physics_material(tile_set_physics_layer);
						uint32_t physics_layer = tile_set->get_physics_layer_collision_layer(tile_set_physics_layer);
						uint32_t physics_mask = tile_set->get_physics_layer_collision_mask(tile_set_physics_layer);

						// Create the body.
						RID body = ps->body_create();
						bodies_coords[body] = E_cell->get();
						ps->body_set_mode(body, collision_animatable ? PhysicsServer2D::BODY_MODE_KINEMATIC : PhysicsServer2D::BODY_MODE_STATIC);
						ps->body_set_space(body, space);

						Transform2D xform;
						xform.set_origin(map_to_world(E_cell->get()));
						xform = global_transform * xform;
						ps->body_set_state(body, PhysicsServer2D::BODY_STATE_TRANSFORM, xform);

						ps->body_attach_object_instance_id(body, get_instance_id());
						ps->body_set_collision_layer(body, physics_layer);
						ps->body_set_collision_mask(body, physics_mask);
						ps->body_set_pickable(body, false);
						ps->body_set_state(body, PhysicsServer2D::BODY_STATE_LINEAR_VELOCITY, tile_data->get_constant_linear_velocity(tile_set_physics_layer));
						ps->body_set_state(body, PhysicsServer2D::BODY_STATE_ANGULAR_VELOCITY, tile_data->get_constant_angular_velocity(tile_set_physics_layer));

						if (!physics_material.is_valid()) {
							ps->body_set_param(body, PhysicsServer2D::BODY_PARAM_BOUNCE, 0);
							ps->body_set_param(body, PhysicsServer2D::BODY_PARAM_FRICTION, 1);
						} else {
							ps->body_set_param(body, PhysicsServer2D::BODY_PARAM_BOUNCE, physics_material->computed_bounce());
							ps->body_set_param(body, PhysicsServer2D::BODY_PARAM_FRICTION, physics_material->computed_friction());
						}

						q.bodies.push_back(body);

						// Add the shapes to the body.
						int body_shape_index = 0;
						for (int polygon_index = 0; polygon_index < tile_data->get_collision_polygons_count(tile_set_physics_layer); polygon_index++) {
							// Iterate over the polygons.
							bool one_way_collision = tile_data->is_collision_polygon_one_way(tile_set_physics_layer, polygon_index);
							float one_way_collision_margin = tile_data->get_collision_polygon_one_way_margin(tile_set_physics_layer, polygon_index);
							int shapes_count = tile_data->get_collision_polygon_shapes_count(tile_set_physics_layer, polygon_index);
							for (int shape_index = 0; shape_index < shapes_count; shape_index++) {
								// Add decomposed convex shapes.
								Ref<ConvexPolygonShape2D> shape = tile_data->get_collision_polygon_shape(tile_set_physics_layer, polygon_index, shape_index);
								ps->body_add_shape(body, shape->get_rid());
								ps->body_set_shape_as_one_way_collision(body, body_shape_index, one_way_collision, one_way_collision_margin);

								body_shape_index++;
							}
						}
					}
				}
			}
		}

		q_list_element = q_list_element->next();
	}
}

void TileMap::_physics_cleanup_quadrant(TileMapQuadrant *p_quadrant) {
	// Remove a quadrant.
	for (RID body : p_quadrant->bodies) {
		bodies_coords.erase(body);
		PhysicsServer2D::get_singleton()->free(body);
	}
	p_quadrant->bodies.clear();
}

void TileMap::_physics_draw_quadrant_debug(TileMapQuadrant *p_quadrant) {
	// Draw the debug collision shapes.
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!get_tree()) {
		return;
	}

	bool show_collision = false;
	switch (collision_visibility_mode) {
		case TileMap::VISIBILITY_MODE_DEFAULT:
			show_collision = !Engine::get_singleton()->is_editor_hint() && (get_tree() && get_tree()->is_debugging_collisions_hint());
			break;
		case TileMap::VISIBILITY_MODE_FORCE_HIDE:
			show_collision = false;
			break;
		case TileMap::VISIBILITY_MODE_FORCE_SHOW:
			show_collision = true;
			break;
	}
	if (!show_collision) {
		return;
	}

	RenderingServer *rs = RenderingServer::get_singleton();
	PhysicsServer2D *ps = PhysicsServer2D::get_singleton();

	Color debug_collision_color = get_tree()->get_debug_collisions_color();
	Vector<Color> color;
	color.push_back(debug_collision_color);

	Vector2 quadrant_pos = map_to_world(p_quadrant->coords * get_effective_quadrant_size(p_quadrant->layer));
	Transform2D qudrant_xform;
	qudrant_xform.set_origin(quadrant_pos);
	Transform2D global_transform_inv = (get_global_transform() * qudrant_xform).affine_inverse();

	for (RID body : p_quadrant->bodies) {
		Transform2D xform = Transform2D(ps->body_get_state(body, PhysicsServer2D::BODY_STATE_TRANSFORM)) * global_transform_inv;
		rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, xform);
		for (int shape_index = 0; shape_index < ps->body_get_shape_count(body); shape_index++) {
			const RID &shape = ps->body_get_shape(body, shape_index);
			PhysicsServer2D::ShapeType type = ps->shape_get_type(shape);
			if (type == PhysicsServer2D::SHAPE_CONVEX_POLYGON) {
				Vector<Vector2> polygon = ps->shape_get_data(shape);
				rs->canvas_item_add_polygon(p_quadrant->debug_canvas_item, polygon, color);
			} else {
				WARN_PRINT("Wrong shape type for a tile, should be SHAPE_CONVEX_POLYGON.");
			}
		}
		rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, Transform2D());
	}
};

/////////////////////////////// Navigation //////////////////////////////////////

void TileMap::_navigation_notification(int p_what) {
	switch (p_what) {
		case CanvasItem::NOTIFICATION_TRANSFORM_CHANGED: {
			if (is_inside_tree()) {
				for (int layer = 0; layer < (int)layers.size(); layer++) {
					Transform2D tilemap_xform = get_global_transform();
					for (KeyValue<Vector2i, TileMapQuadrant> &E_quadrant : layers[layer].quadrant_map) {
						TileMapQuadrant &q = E_quadrant.value;
						for (const KeyValue<Vector2i, Vector<RID>> &E_region : q.navigation_regions) {
							for (int layer_index = 0; layer_index < E_region.value.size(); layer_index++) {
								RID region = E_region.value[layer_index];
								if (!region.is_valid()) {
									continue;
								}
								Transform2D tile_transform;
								tile_transform.set_origin(map_to_world(E_region.key));
								NavigationServer2D::get_singleton()->region_set_transform(region, tilemap_xform * tile_transform);
							}
						}
					}
				}
			}
		} break;
	}
}

void TileMap::_navigation_update_dirty_quadrants(SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(!tile_set.is_valid());

	// Get colors for debug.
	SceneTree *st = SceneTree::get_singleton();
	Color debug_navigation_color;
	bool debug_navigation = st && st->is_debugging_navigation_hint();
	if (debug_navigation) {
		debug_navigation_color = st->get_debug_navigation_color();
	}

	Transform2D tilemap_xform = get_global_transform();
	SelfList<TileMapQuadrant> *q_list_element = r_dirty_quadrant_list.first();
	while (q_list_element) {
		TileMapQuadrant &q = *q_list_element->self();

		// Clear navigation shapes in the quadrant.
		for (const KeyValue<Vector2i, Vector<RID>> &E : q.navigation_regions) {
			for (int i = 0; i < E.value.size(); i++) {
				RID region = E.value[i];
				if (!region.is_valid()) {
					continue;
				}
				NavigationServer2D::get_singleton()->region_set_map(region, RID());
			}
		}
		q.navigation_regions.clear();

		// Get the navigation polygons and create regions.
		for (Set<Vector2i>::Element *E_cell = q.cells.front(); E_cell; E_cell = E_cell->next()) {
			TileMapCell c = get_cell(q.layer, E_cell->get(), true);

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile));
					q.navigation_regions[E_cell->get()].resize(tile_set->get_navigation_layers_count());

					for (int layer_index = 0; layer_index < tile_set->get_navigation_layers_count(); layer_index++) {
						Ref<NavigationPolygon> navpoly;
						navpoly = tile_data->get_navigation_polygon(layer_index);

						if (navpoly.is_valid()) {
							Transform2D tile_transform;
							tile_transform.set_origin(map_to_world(E_cell->get()));

							RID region = NavigationServer2D::get_singleton()->region_create();
							NavigationServer2D::get_singleton()->region_set_map(region, get_world_2d()->get_navigation_map());
							NavigationServer2D::get_singleton()->region_set_transform(region, tilemap_xform * tile_transform);
							NavigationServer2D::get_singleton()->region_set_navpoly(region, navpoly);
							q.navigation_regions[E_cell->get()].write[layer_index] = region;
						}
					}
				}
			}
		}

		q_list_element = q_list_element->next();
	}
}

void TileMap::_navigation_cleanup_quadrant(TileMapQuadrant *p_quadrant) {
	// Clear navigation shapes in the quadrant.
	for (const KeyValue<Vector2i, Vector<RID>> &E : p_quadrant->navigation_regions) {
		for (int i = 0; i < E.value.size(); i++) {
			RID region = E.value[i];
			if (!region.is_valid()) {
				continue;
			}
			NavigationServer2D::get_singleton()->free(region);
		}
	}
	p_quadrant->navigation_regions.clear();
}

void TileMap::_navigation_draw_quadrant_debug(TileMapQuadrant *p_quadrant) {
	// Draw the debug collision shapes.
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!get_tree()) {
		return;
	}

	bool show_navigation = false;
	switch (navigation_visibility_mode) {
		case TileMap::VISIBILITY_MODE_DEFAULT:
			show_navigation = !Engine::get_singleton()->is_editor_hint() && (get_tree() && get_tree()->is_debugging_navigation_hint());
			break;
		case TileMap::VISIBILITY_MODE_FORCE_HIDE:
			show_navigation = false;
			break;
		case TileMap::VISIBILITY_MODE_FORCE_SHOW:
			show_navigation = true;
			break;
	}
	if (!show_navigation) {
		return;
	}

	RenderingServer *rs = RenderingServer::get_singleton();

	Color color = get_tree()->get_debug_navigation_color();
	RandomPCG rand;

	Vector2 quadrant_pos = map_to_world(p_quadrant->coords * get_effective_quadrant_size(p_quadrant->layer));

	for (Set<Vector2i>::Element *E_cell = p_quadrant->cells.front(); E_cell; E_cell = E_cell->next()) {
		TileMapCell c = get_cell(p_quadrant->layer, E_cell->get(), true);

		TileSetSource *source;
		if (tile_set->has_source(c.source_id)) {
			source = *tile_set->get_source(c.source_id);

			if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
				continue;
			}

			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile));

				Transform2D xform;
				xform.set_origin(map_to_world(E_cell->get()) - quadrant_pos);
				rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, xform);

				for (int layer_index = 0; layer_index < tile_set->get_navigation_layers_count(); layer_index++) {
					Ref<NavigationPolygon> navpoly = tile_data->get_navigation_polygon(layer_index);
					if (navpoly.is_valid()) {
						PackedVector2Array navigation_polygon_vertices = navpoly->get_vertices();

						for (int i = 0; i < navpoly->get_polygon_count(); i++) {
							// An array of vertices for this polygon.
							Vector<int> polygon = navpoly->get_polygon(i);
							Vector<Vector2> vertices;
							vertices.resize(polygon.size());
							for (int j = 0; j < polygon.size(); j++) {
								ERR_FAIL_INDEX(polygon[j], navigation_polygon_vertices.size());
								vertices.write[j] = navigation_polygon_vertices[polygon[j]];
							}

							// Generate the polygon color, slightly randomly modified from the settings one.
							Color random_variation_color;
							random_variation_color.set_hsv(color.get_h() + rand.random(-1.0, 1.0) * 0.05, color.get_s(), color.get_v() + rand.random(-1.0, 1.0) * 0.1);
							random_variation_color.a = color.a;
							Vector<Color> colors;
							colors.push_back(random_variation_color);

							rs->canvas_item_add_polygon(p_quadrant->debug_canvas_item, vertices, colors);
						}
					}
				}
			}
		}
	}
}

/////////////////////////////// Scenes //////////////////////////////////////

void TileMap::_scenes_update_dirty_quadrants(SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
	ERR_FAIL_COND(!tile_set.is_valid());

	SelfList<TileMapQuadrant> *q_list_element = r_dirty_quadrant_list.first();
	while (q_list_element) {
		TileMapQuadrant &q = *q_list_element->self();

		// Clear the scenes.
		for (const KeyValue<Vector2i, String> &E : q.scenes) {
			Node *node = get_node(E.value);
			if (node) {
				node->queue_delete();
			}
		}

		q.scenes.clear();

		// Recreate the scenes.
		for (Set<Vector2i>::Element *E_cell = q.cells.front(); E_cell; E_cell = E_cell->next()) {
			const TileMapCell &c = get_cell(q.layer, E_cell->get(), true);

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetScenesCollectionSource *scenes_collection_source = Object::cast_to<TileSetScenesCollectionSource>(source);
				if (scenes_collection_source) {
					Ref<PackedScene> packed_scene = scenes_collection_source->get_scene_tile_scene(c.alternative_tile);
					if (packed_scene.is_valid()) {
						Node *scene = packed_scene->instantiate();
						add_child(scene);
						Control *scene_as_control = Object::cast_to<Control>(scene);
						Node2D *scene_as_node2d = Object::cast_to<Node2D>(scene);
						if (scene_as_control) {
							scene_as_control->set_position(map_to_world(E_cell->get()) + scene_as_control->get_position());
						} else if (scene_as_node2d) {
							Transform2D xform;
							xform.set_origin(map_to_world(E_cell->get()));
							scene_as_node2d->set_transform(xform * scene_as_node2d->get_transform());
						}
						q.scenes[E_cell->get()] = scene->get_name();
					}
				}
			}
		}

		q_list_element = q_list_element->next();
	}
}

void TileMap::_scenes_cleanup_quadrant(TileMapQuadrant *p_quadrant) {
	// Clear the scenes.
	for (const KeyValue<Vector2i, String> &E : p_quadrant->scenes) {
		Node *node = get_node(E.value);
		if (node) {
			node->queue_delete();
		}
	}

	p_quadrant->scenes.clear();
}

void TileMap::_scenes_draw_quadrant_debug(TileMapQuadrant *p_quadrant) {
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	// Draw a placeholder for scenes needing one.
	RenderingServer *rs = RenderingServer::get_singleton();
	Vector2 quadrant_pos = map_to_world(p_quadrant->coords * get_effective_quadrant_size(p_quadrant->layer));
	for (Set<Vector2i>::Element *E_cell = p_quadrant->cells.front(); E_cell; E_cell = E_cell->next()) {
		const TileMapCell &c = get_cell(p_quadrant->layer, E_cell->get(), true);

		TileSetSource *source;
		if (tile_set->has_source(c.source_id)) {
			source = *tile_set->get_source(c.source_id);

			if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
				continue;
			}

			TileSetScenesCollectionSource *scenes_collection_source = Object::cast_to<TileSetScenesCollectionSource>(source);
			if (scenes_collection_source) {
				if (!scenes_collection_source->get_scene_tile_scene(c.alternative_tile).is_valid() || scenes_collection_source->get_scene_tile_display_placeholder(c.alternative_tile)) {
					// Generate a random color from the hashed values of the tiles.
					Array to_hash;
					to_hash.push_back(c.source_id);
					to_hash.push_back(c.alternative_tile);
					uint32_t hash = RandomPCG(to_hash.hash()).rand();

					Color color;
					color = color.from_hsv(
							(float)((hash >> 24) & 0xFF) / 256.0,
							Math::lerp(0.5, 1.0, (float)((hash >> 16) & 0xFF) / 256.0),
							Math::lerp(0.5, 1.0, (float)((hash >> 8) & 0xFF) / 256.0),
							0.8);

					// Draw a placeholder tile.
					Transform2D xform;
					xform.set_origin(map_to_world(E_cell->get()) - quadrant_pos);
					rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, xform);
					rs->canvas_item_add_circle(p_quadrant->debug_canvas_item, Vector2(), MIN(tile_set->get_tile_size().x, tile_set->get_tile_size().y) / 4.0, color);
				}
			}
		}
	}
}

void TileMap::set_cell(int p_layer, const Vector2i &p_coords, int p_source_id, const Vector2i p_atlas_coords, int p_alternative_tile) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	// Set the current cell tile (using integer position).
	Map<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	Vector2i pk(p_coords);
	Map<Vector2i, TileMapCell>::Element *E = tile_map.find(pk);

	int source_id = p_source_id;
	Vector2i atlas_coords = p_atlas_coords;
	int alternative_tile = p_alternative_tile;

	if ((source_id == TileSet::INVALID_SOURCE || atlas_coords == TileSetSource::INVALID_ATLAS_COORDS || alternative_tile == TileSetSource::INVALID_TILE_ALTERNATIVE) &&
			(source_id != TileSet::INVALID_SOURCE || atlas_coords != TileSetSource::INVALID_ATLAS_COORDS || alternative_tile != TileSetSource::INVALID_TILE_ALTERNATIVE)) {
		WARN_PRINT("Setting a cell a cell as empty requires both source_id, atlas_coord and alternative_tile to be set to their respective \"invalid\" values. Values were thus changes accordingly.");
		source_id = TileSet::INVALID_SOURCE;
		atlas_coords = TileSetSource::INVALID_ATLAS_COORDS;
		alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE;
	}

	if (!E && source_id == TileSet::INVALID_SOURCE) {
		return; // Nothing to do, the tile is already empty.
	}

	// Get the quadrant
	Vector2i qk = _coords_to_quadrant_coords(p_layer, pk);

	Map<Vector2i, TileMapQuadrant>::Element *Q = layers[p_layer].quadrant_map.find(qk);

	if (source_id == TileSet::INVALID_SOURCE) {
		// Erase existing cell in the tile map.
		tile_map.erase(pk);

		// Erase existing cell in the quadrant.
		ERR_FAIL_COND(!Q);
		TileMapQuadrant &q = Q->get();

		q.cells.erase(pk);

		// Remove or make the quadrant dirty.
		if (q.cells.size() == 0) {
			_erase_quadrant(Q);
		} else {
			_make_quadrant_dirty(Q);
		}

		used_rect_cache_dirty = true;
	} else {
		if (!E) {
			// Insert a new cell in the tile map.
			E = tile_map.insert(pk, TileMapCell());

			// Create a new quadrant if needed, then insert the cell if needed.
			if (!Q) {
				Q = _create_quadrant(p_layer, qk);
			}
			TileMapQuadrant &q = Q->get();
			q.cells.insert(pk);

		} else {
			ERR_FAIL_COND(!Q); // TileMapQuadrant should exist...

			if (E->get().source_id == source_id && E->get().get_atlas_coords() == atlas_coords && E->get().alternative_tile == alternative_tile) {
				return; // Nothing changed.
			}
		}

		TileMapCell &c = E->get();

		c.source_id = source_id;
		c.set_atlas_coords(atlas_coords);
		c.alternative_tile = alternative_tile;

		_make_quadrant_dirty(Q);
		used_rect_cache_dirty = true;
	}
}

int TileMap::get_cell_source_id(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), TileSet::INVALID_SOURCE);

	// Get a cell source id from position
	const Map<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	const Map<Vector2i, TileMapCell>::Element *E = tile_map.find(p_coords);

	if (!E) {
		return TileSet::INVALID_SOURCE;
	}

	if (p_use_proxies && tile_set.is_valid()) {
		Array proxyed = tile_set->map_tile_proxy(E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
		return proxyed[0];
	}

	return E->get().source_id;
}

Vector2i TileMap::get_cell_atlas_coords(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), TileSetSource::INVALID_ATLAS_COORDS);

	// Get a cell source id from position
	const Map<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	const Map<Vector2i, TileMapCell>::Element *E = tile_map.find(p_coords);

	if (!E) {
		return TileSetSource::INVALID_ATLAS_COORDS;
	}

	if (p_use_proxies && tile_set.is_valid()) {
		Array proxyed = tile_set->map_tile_proxy(E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
		return proxyed[1];
	}

	return E->get().get_atlas_coords();
}

int TileMap::get_cell_alternative_tile(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), TileSetSource::INVALID_TILE_ALTERNATIVE);

	// Get a cell source id from position
	const Map<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	const Map<Vector2i, TileMapCell>::Element *E = tile_map.find(p_coords);

	if (!E) {
		return TileSetSource::INVALID_TILE_ALTERNATIVE;
	}

	if (p_use_proxies && tile_set.is_valid()) {
		Array proxyed = tile_set->map_tile_proxy(E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
		return proxyed[2];
	}

	return E->get().alternative_tile;
}

Ref<TileMapPattern> TileMap::get_pattern(int p_layer, TypedArray<Vector2i> p_coords_array) {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), nullptr);
	ERR_FAIL_COND_V(!tile_set.is_valid(), nullptr);

	Ref<TileMapPattern> output;
	output.instantiate();
	if (p_coords_array.is_empty()) {
		return output;
	}

	Vector2i min = Vector2i(p_coords_array[0]);
	for (int i = 1; i < p_coords_array.size(); i++) {
		min = min.min(p_coords_array[i]);
	}

	Vector<Vector2i> coords_in_pattern_array;
	coords_in_pattern_array.resize(p_coords_array.size());
	Vector2i ensure_positive_offset;
	for (int i = 0; i < p_coords_array.size(); i++) {
		Vector2i coords = p_coords_array[i];
		Vector2i coords_in_pattern = coords - min;
		if (tile_set->get_tile_shape() != TileSet::TILE_SHAPE_SQUARE) {
			if (tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_STACKED) {
				if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL && bool(min.y % 2) && bool(coords_in_pattern.y % 2)) {
					coords_in_pattern.x -= 1;
					if (coords_in_pattern.x < 0) {
						ensure_positive_offset.x = 1;
					}
				} else if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL && bool(min.x % 2) && bool(coords_in_pattern.x % 2)) {
					coords_in_pattern.y -= 1;
					if (coords_in_pattern.y < 0) {
						ensure_positive_offset.y = 1;
					}
				}
			} else if (tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_STACKED_OFFSET) {
				if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL && bool(min.y % 2) && bool(coords_in_pattern.y % 2)) {
					coords_in_pattern.x += 1;
				} else if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL && bool(min.x % 2) && bool(coords_in_pattern.x % 2)) {
					coords_in_pattern.y += 1;
				}
			}
		}
		coords_in_pattern_array.write[i] = coords_in_pattern;
	}

	for (int i = 0; i < coords_in_pattern_array.size(); i++) {
		Vector2i coords = p_coords_array[i];
		Vector2i coords_in_pattern = coords_in_pattern_array[i];
		output->set_cell(coords_in_pattern + ensure_positive_offset, get_cell_source_id(p_layer, coords), get_cell_atlas_coords(p_layer, coords), get_cell_alternative_tile(p_layer, coords));
	}

	return output;
}

Vector2i TileMap::map_pattern(Vector2i p_position_in_tilemap, Vector2i p_coords_in_pattern, Ref<TileMapPattern> p_pattern) {
	ERR_FAIL_COND_V(!p_pattern->has_cell(p_coords_in_pattern), Vector2i());

	Vector2i output = p_position_in_tilemap + p_coords_in_pattern;
	if (tile_set->get_tile_shape() != TileSet::TILE_SHAPE_SQUARE) {
		if (tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_STACKED) {
			if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL && bool(p_position_in_tilemap.y % 2) && bool(p_coords_in_pattern.y % 2)) {
				output.x += 1;
			} else if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL && bool(p_position_in_tilemap.x % 2) && bool(p_coords_in_pattern.x % 2)) {
				output.y += 1;
			}
		} else if (tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_STACKED_OFFSET) {
			if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL && bool(p_position_in_tilemap.y % 2) && bool(p_coords_in_pattern.y % 2)) {
				output.x -= 1;
			} else if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL && bool(p_position_in_tilemap.x % 2) && bool(p_coords_in_pattern.x % 2)) {
				output.y -= 1;
			}
		}
	}

	return output;
}

void TileMap::set_pattern(int p_layer, Vector2i p_position, const Ref<TileMapPattern> p_pattern) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	ERR_FAIL_COND(!tile_set.is_valid());

	TypedArray<Vector2i> used_cells = p_pattern->get_used_cells();
	for (int i = 0; i < used_cells.size(); i++) {
		Vector2i coords = map_pattern(p_position, used_cells[i], p_pattern);
		set_cell(p_layer, coords, p_pattern->get_cell_source_id(coords), p_pattern->get_cell_atlas_coords(coords), p_pattern->get_cell_alternative_tile(coords));
	}
}

Set<TileSet::TerrainsPattern> TileMap::_get_valid_terrains_patterns_for_constraints(int p_terrain_set, const Vector2i &p_position, Set<TerrainConstraint> p_constraints) {
	if (!tile_set.is_valid()) {
		return Set<TileSet::TerrainsPattern>();
	}

	// Returns all tiles compatible with the given constraints.
	Set<TileSet::TerrainsPattern> compatible_terrain_tile_patterns;
	for (TileSet::TerrainsPattern &terrain_pattern : tile_set->get_terrains_pattern_set(p_terrain_set)) {
		int valid = true;
		int in_pattern_count = 0;
		for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(i);
			if (tile_set->is_valid_peering_bit_terrain(p_terrain_set, bit)) {
				// Check if the bit is compatible with the constraints.
				TerrainConstraint terrain_bit_constraint = TerrainConstraint(this, p_position, bit, terrain_pattern[in_pattern_count]);
				Set<TerrainConstraint>::Element *in_set_constraint_element = p_constraints.find(terrain_bit_constraint);
				if (in_set_constraint_element && in_set_constraint_element->get().get_terrain() != terrain_bit_constraint.get_terrain()) {
					valid = false;
					break;
				}
				in_pattern_count++;
			}
		}

		if (valid) {
			compatible_terrain_tile_patterns.insert(terrain_pattern);
		}
	}

	return compatible_terrain_tile_patterns;
}

Set<TileMap::TerrainConstraint> TileMap::get_terrain_constraints_from_removed_cells_list(int p_layer, const Set<Vector2i> &p_to_replace, int p_terrain_set, bool p_ignore_empty_terrains) const {
	if (!tile_set.is_valid()) {
		return Set<TerrainConstraint>();
	}

	ERR_FAIL_INDEX_V(p_terrain_set, tile_set->get_terrain_sets_count(), Set<TerrainConstraint>());
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), Set<TerrainConstraint>());

	// Build a set of dummy constraints get the constrained points.
	Set<TerrainConstraint> dummy_constraints;
	for (Set<Vector2i>::Element *E = p_to_replace.front(); E; E = E->next()) {
		for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) { // Iterates over sides.
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(i);
			if (tile_set->is_valid_peering_bit_terrain(p_terrain_set, bit)) {
				dummy_constraints.insert(TerrainConstraint(this, E->get(), bit, -1));
			}
		}
	}

	// For each constrained point, we get all overlapping tiles, and select the most adequate terrain for it.
	Set<TerrainConstraint> constraints;
	for (Set<TerrainConstraint>::Element *E = dummy_constraints.front(); E; E = E->next()) {
		TerrainConstraint c = E->get();

		Map<int, int> terrain_count;

		// Count the number of occurrences per terrain.
		Map<Vector2i, TileSet::CellNeighbor> overlapping_terrain_bits = c.get_overlapping_coords_and_peering_bits();
		for (const KeyValue<Vector2i, TileSet::CellNeighbor> &E_overlapping : overlapping_terrain_bits) {
			if (!p_to_replace.has(E_overlapping.key)) {
				TileData *neighbor_tile_data = nullptr;
				TileMapCell neighbor_cell = get_cell(p_layer, E_overlapping.key);
				if (neighbor_cell.source_id != TileSet::INVALID_SOURCE) {
					Ref<TileSetSource> source = tile_set->get_source(neighbor_cell.source_id);
					Ref<TileSetAtlasSource> atlas_source = source;
					if (atlas_source.is_valid()) {
						TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(neighbor_cell.get_atlas_coords(), neighbor_cell.alternative_tile));
						if (tile_data && tile_data->get_terrain_set() == p_terrain_set) {
							neighbor_tile_data = tile_data;
						}
					}
				}

				int terrain = neighbor_tile_data ? neighbor_tile_data->get_peering_bit_terrain(TileSet::CellNeighbor(E_overlapping.value)) : -1;
				if (!p_ignore_empty_terrains || terrain >= 0) {
					if (!terrain_count.has(terrain)) {
						terrain_count[terrain] = 0;
					}
					terrain_count[terrain] += 1;
				}
			}
		}

		// Get the terrain with the max number of occurrences.
		int max = 0;
		int max_terrain = -1;
		for (const KeyValue<int, int> &E_terrain_count : terrain_count) {
			if (E_terrain_count.value > max) {
				max = E_terrain_count.value;
				max_terrain = E_terrain_count.key;
			}
		}

		// Set the adequate terrain.
		if (max > 0) {
			c.set_terrain(max_terrain);
			constraints.insert(c);
		}
	}

	return constraints;
}

Set<TileMap::TerrainConstraint> TileMap::get_terrain_constraints_from_added_tile(Vector2i p_position, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern) const {
	if (!tile_set.is_valid()) {
		return Set<TerrainConstraint>();
	}

	// Compute the constraints needed from the surrounding tiles.
	Set<TerrainConstraint> output;
	int in_pattern_count = 0;
	for (uint32_t i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
		TileSet::CellNeighbor side = TileSet::CellNeighbor(i);
		if (tile_set->is_valid_peering_bit_terrain(p_terrain_set, side)) {
			TerrainConstraint c = TerrainConstraint(this, p_position, side, p_terrains_pattern[in_pattern_count]);
			output.insert(c);
			in_pattern_count++;
		}
	}

	return output;
}

Map<Vector2i, TileSet::TerrainsPattern> TileMap::terrain_wave_function_collapse(const Set<Vector2i> &p_to_replace, int p_terrain_set, const Set<TerrainConstraint> p_constraints) {
	if (!tile_set.is_valid()) {
		return Map<Vector2i, TileSet::TerrainsPattern>();
	}

	// Copy the constraints set.
	Set<TerrainConstraint> constraints = p_constraints;

	// Compute all acceptable patterns for each cell.
	Map<Vector2i, Set<TileSet::TerrainsPattern>> per_cell_acceptable_tiles;
	for (Vector2i cell : p_to_replace) {
		per_cell_acceptable_tiles[cell] = _get_valid_terrains_patterns_for_constraints(p_terrain_set, cell, constraints);
	}

	// Output map.
	Map<Vector2i, TileSet::TerrainsPattern> output;

	// Add all positions to a set.
	Set<Vector2i> to_replace = Set<Vector2i>(p_to_replace);
	while (!to_replace.is_empty()) {
		// Compute the minimum number of tile possibilities for each cell.
		int min_nb_possibilities = 100000000;
		for (const KeyValue<Vector2i, Set<TileSet::TerrainsPattern>> &E : per_cell_acceptable_tiles) {
			min_nb_possibilities = MIN(min_nb_possibilities, E.value.size());
		}

		// Get the set of possible cells to fill, out of the most constrained ones.
		LocalVector<Vector2i> to_choose_from;
		for (const KeyValue<Vector2i, Set<TileSet::TerrainsPattern>> &E : per_cell_acceptable_tiles) {
			if (E.value.size() == min_nb_possibilities) {
				to_choose_from.push_back(E.key);
			}
		}

		// Randomly a cell to fill out of the most constrained.
		Vector2i selected_cell_to_replace = to_choose_from[Math::random(0, to_choose_from.size() - 1)];

		// Get the list of acceptable pattens for the given cell.
		Set<TileSet::TerrainsPattern> valid_tiles = per_cell_acceptable_tiles[selected_cell_to_replace];
		if (valid_tiles.is_empty()) {
			break; // No possibilities :/
		}

		// Out of the possible patterns, prioritize the one which have the least amount of different terrains.
		LocalVector<TileSet::TerrainsPattern> valid_tiles_with_least_amount_of_terrains;
		int min_terrain_count = 10000;
		LocalVector<int> terrains_counts;
		int pattern_index = 0;
		for (const TileSet::TerrainsPattern &pattern : valid_tiles) {
			Set<int> terrains;
			for (int i = 0; i < pattern.size(); i++) {
				terrains.insert(pattern[i]);
			}
			min_terrain_count = MIN(min_terrain_count, terrains.size());
			terrains_counts.push_back(terrains.size());
			pattern_index++;
		}
		pattern_index = 0;
		for (const TileSet::TerrainsPattern &pattern : valid_tiles) {
			if (terrains_counts[pattern_index] == min_terrain_count) {
				valid_tiles_with_least_amount_of_terrains.push_back(pattern);
			}
			pattern_index++;
		}

		// Randomly select a pattern out of the remaining ones.
		TileSet::TerrainsPattern selected_terrain_tile_pattern = valid_tiles_with_least_amount_of_terrains[Math::random(0, valid_tiles_with_least_amount_of_terrains.size() - 1)];

		// Set the selected cell into the output.
		output[selected_cell_to_replace] = selected_terrain_tile_pattern;
		to_replace.erase(selected_cell_to_replace);
		per_cell_acceptable_tiles.erase(selected_cell_to_replace);

		// Add the new constraints from the added tiles.
		Set<TerrainConstraint> new_constraints = get_terrain_constraints_from_added_tile(selected_cell_to_replace, p_terrain_set, selected_terrain_tile_pattern);
		for (Set<TerrainConstraint>::Element *E_constraint = new_constraints.front(); E_constraint; E_constraint = E_constraint->next()) {
			constraints.insert(E_constraint->get());
		}

		// Compute valid tiles again for neighbors.
		for (uint32_t i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
			TileSet::CellNeighbor side = TileSet::CellNeighbor(i);
			if (is_existing_neighbor(side)) {
				Vector2i neighbor = get_neighbor_cell(selected_cell_to_replace, side);
				if (to_replace.has(neighbor)) {
					per_cell_acceptable_tiles[neighbor] = _get_valid_terrains_patterns_for_constraints(p_terrain_set, neighbor, constraints);
				}
			}
		}
	}
	return output;
}

void TileMap::set_cells_from_surrounding_terrains(int p_layer, TypedArray<Vector2i> p_coords_array, int p_terrain_set, bool p_ignore_empty_terrains) {
	ERR_FAIL_COND(!tile_set.is_valid());
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	ERR_FAIL_INDEX(p_terrain_set, tile_set->get_terrain_sets_count());

	Set<Vector2i> coords_set;
	for (int i = 0; i < p_coords_array.size(); i++) {
		coords_set.insert(p_coords_array[i]);
	}

	Set<TileMap::TerrainConstraint> constraints = get_terrain_constraints_from_removed_cells_list(p_layer, coords_set, p_terrain_set, p_ignore_empty_terrains);

	Map<Vector2i, TileSet::TerrainsPattern> wfc_output = terrain_wave_function_collapse(coords_set, p_terrain_set, constraints);
	for (const KeyValue<Vector2i, TileSet::TerrainsPattern> &kv : wfc_output) {
		TileMapCell cell = tile_set->get_random_tile_from_pattern(p_terrain_set, kv.value);
		set_cell(p_layer, kv.key, cell.source_id, cell.get_atlas_coords(), cell.alternative_tile);
	}
}

TileMapCell TileMap::get_cell(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), TileMapCell());
	const Map<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	if (!tile_map.has(p_coords)) {
		return TileMapCell();
	} else {
		TileMapCell c = tile_map.find(p_coords)->get();
		if (p_use_proxies && tile_set.is_valid()) {
			Array proxyed = tile_set->map_tile_proxy(c.source_id, c.get_atlas_coords(), c.alternative_tile);
			c.source_id = proxyed[0];
			c.set_atlas_coords(proxyed[1]);
			c.alternative_tile = proxyed[2];
		}
		return c;
	}
}

Map<Vector2i, TileMapQuadrant> *TileMap::get_quadrant_map(int p_layer) {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), nullptr);

	return &layers[p_layer].quadrant_map;
}

Vector2i TileMap::get_coords_for_body_rid(RID p_physics_body) {
	ERR_FAIL_COND_V_MSG(!bodies_coords.has(p_physics_body), Vector2i(), vformat("No tiles for the given body RID %d.", p_physics_body));
	return bodies_coords[p_physics_body];
}

void TileMap::fix_invalid_tiles() {
	ERR_FAIL_COND_MSG(tile_set.is_null(), "Cannot fix invalid tiles if Tileset is not open.");

	for (unsigned int i = 0; i < layers.size(); i++) {
		const Map<Vector2i, TileMapCell> &tile_map = layers[i].tile_map;
		Set<Vector2i> coords;
		for (const KeyValue<Vector2i, TileMapCell> &E : tile_map) {
			TileSetSource *source = *tile_set->get_source(E.value.source_id);
			if (!source || !source->has_tile(E.value.get_atlas_coords()) || !source->has_alternative_tile(E.value.get_atlas_coords(), E.value.alternative_tile)) {
				coords.insert(E.key);
			}
		}
		for (Set<Vector2i>::Element *E = coords.front(); E; E = E->next()) {
			set_cell(i, E->get(), TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
		}
	}
}

void TileMap::clear_layer(int p_layer) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	// Remove all tiles.
	_clear_layer_internals(p_layer);
	layers[p_layer].tile_map.clear();

	used_rect_cache_dirty = true;
}

void TileMap::clear() {
	// Remove all tiles.
	_clear_internals();
	for (unsigned int i = 0; i < layers.size(); i++) {
		layers[i].tile_map.clear();
	}
	used_rect_cache_dirty = true;
}

void TileMap::_set_tile_data(int p_layer, const Vector<int> &p_data) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	ERR_FAIL_COND(format > FORMAT_3);

	// Set data for a given tile from raw data.

	int c = p_data.size();
	const int *r = p_data.ptr();

	int offset = (format >= FORMAT_2) ? 3 : 2;
	ERR_FAIL_COND_MSG(c % offset != 0, "Corrupted tile data.");

	clear_layer(p_layer);

#ifdef DISABLE_DEPRECATED
	ERR_FAIL_COND_MSG(format != FORMAT_3, vformat("Cannot handle deprecated TileMap data format version %d. This Godot version was compiled with no support for deprecated data.", format));
#endif

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
		// Extracts position in TileMap.
		int16_t x = decode_uint16(&local[0]);
		int16_t y = decode_uint16(&local[2]);

		if (format == FORMAT_3) {
			uint16_t source_id = decode_uint16(&local[4]);
			uint16_t atlas_coords_x = decode_uint16(&local[6]);
			uint16_t atlas_coords_y = decode_uint16(&local[8]);
			uint16_t alternative_tile = decode_uint16(&local[10]);
			set_cell(p_layer, Vector2i(x, y), source_id, Vector2i(atlas_coords_x, atlas_coords_y), alternative_tile);
		} else {
#ifndef DISABLE_DEPRECATED
			// Previous decated format.

			uint32_t v = decode_uint32(&local[4]);
			// Extract the transform flags that used to be in the tilemap.
			bool flip_h = v & (1 << 29);
			bool flip_v = v & (1 << 30);
			bool transpose = v & (1 << 31);
			v &= (1 << 29) - 1;

			// Extract autotile/atlas coords.
			int16_t coord_x = 0;
			int16_t coord_y = 0;
			if (format == FORMAT_2) {
				coord_x = decode_uint16(&local[8]);
				coord_y = decode_uint16(&local[10]);
			}

			if (tile_set.is_valid()) {
				Array a = tile_set->compatibility_tilemap_map(v, Vector2i(coord_x, coord_y), flip_h, flip_v, transpose);
				if (a.size() == 3) {
					set_cell(p_layer, Vector2i(x, y), a[0], a[1], a[2]);
				} else {
					ERR_PRINT(vformat("No valid tile in Tileset for: tile:%s coords:%s flip_h:%s flip_v:%s transpose:%s", v, Vector2i(coord_x, coord_y), flip_h, flip_v, transpose));
				}
			} else {
				int compatibility_alternative_tile = ((int)flip_h) + ((int)flip_v << 1) + ((int)transpose << 2);
				set_cell(p_layer, Vector2i(x, y), v, Vector2i(coord_x, coord_y), compatibility_alternative_tile);
			}
#endif
		}
	}
	emit_signal(SNAME("changed"));
}

Vector<int> TileMap::_get_tile_data(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), Vector<int>());

	// Export tile data to raw format
	const Map<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	Vector<int> data;
	data.resize(tile_map.size() * 3);
	int *w = data.ptrw();

	// Save in highest format

	int idx = 0;
	for (const KeyValue<Vector2i, TileMapCell> &E : tile_map) {
		uint8_t *ptr = (uint8_t *)&w[idx];
		encode_uint16((int16_t)(E.key.x), &ptr[0]);
		encode_uint16((int16_t)(E.key.y), &ptr[2]);
		encode_uint16(E.value.source_id, &ptr[4]);
		encode_uint16(E.value.coord_x, &ptr[6]);
		encode_uint16(E.value.coord_y, &ptr[8]);
		encode_uint16(E.value.alternative_tile, &ptr[10]);
		idx += 3;
	}

	return data;
}

#ifdef TOOLS_ENABLED
Rect2 TileMap::_edit_get_rect() const {
	// Return the visible rect of the tilemap
	if (pending_update) {
		const_cast<TileMap *>(this)->_update_dirty_quadrants();
	} else {
		const_cast<TileMap *>(this)->_recompute_rect_cache();
	}
	return rect_cache;
}
#endif

bool TileMap::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (p_name == "format") {
		if (p_value.get_type() == Variant::INT) {
			format = (DataFormat)(p_value.operator int64_t()); // Set format used for loading
			return true;
		}
	} else if (p_name == "tile_data") { // Kept for compatibility reasons.
		if (p_value.is_array()) {
			if (layers.size() < 1) {
				layers.resize(1);
			}
			_set_tile_data(0, p_value);
			return true;
		}
		return false;
	} else if (components.size() == 2 && components[0].begins_with("layer_") && components[0].trim_prefix("layer_").is_valid_int()) {
		int index = components[0].trim_prefix("layer_").to_int();
		if (index < 0) {
			return false;
		}

		if (index >= (int)layers.size()) {
			_clear_internals();
			while (index >= (int)layers.size()) {
				layers.push_back(TileMapLayer());
			}
			_recreate_internals();

			notify_property_list_changed();
			emit_signal(SNAME("changed"));
			update_configuration_warnings();
		}

		if (components[1] == "name") {
			set_layer_name(index, p_value);
			return true;
		} else if (components[1] == "enabled") {
			set_layer_enabled(index, p_value);
			return true;
		} else if (components[1] == "modulate") {
			set_layer_modulate(index, p_value);
			return true;
		} else if (components[1] == "y_sort_enabled") {
			set_layer_y_sort_enabled(index, p_value);
			return true;
		} else if (components[1] == "y_sort_origin") {
			set_layer_y_sort_origin(index, p_value);
			return true;
		} else if (components[1] == "z_index") {
			set_layer_z_index(index, p_value);
			return true;
		} else if (components[1] == "tile_data") {
			_set_tile_data(index, p_value);
			return true;
		} else {
			return false;
		}
	}
	return false;
}

bool TileMap::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (p_name == "format") {
		r_ret = FORMAT_3; // When saving, always save highest format
		return true;
	} else if (components.size() == 2 && components[0].begins_with("layer_") && components[0].trim_prefix("layer_").is_valid_int()) {
		int index = components[0].trim_prefix("layer_").to_int();
		if (index < 0 || index >= (int)layers.size()) {
			return false;
		}

		if (components[1] == "name") {
			r_ret = get_layer_name(index);
			return true;
		} else if (components[1] == "enabled") {
			r_ret = is_layer_enabled(index);
			return true;
		} else if (components[1] == "modulate") {
			r_ret = get_layer_modulate(index);
			return true;
		} else if (components[1] == "y_sort_enabled") {
			r_ret = is_layer_y_sort_enabled(index);
			return true;
		} else if (components[1] == "y_sort_origin") {
			r_ret = get_layer_y_sort_origin(index);
			return true;
		} else if (components[1] == "z_index") {
			r_ret = get_layer_z_index(index);
			return true;
		} else if (components[1] == "tile_data") {
			r_ret = _get_tile_data(index);
			return true;
		} else {
			return false;
		}
	}
	return false;
}

void TileMap::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::INT, "format", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
	p_list->push_back(PropertyInfo(Variant::NIL, "Layers", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
	for (unsigned int i = 0; i < layers.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::STRING, vformat("layer_%d/name", i), PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::BOOL, vformat("layer_%d/enabled", i), PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::COLOR, vformat("layer_%d/modulate", i), PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::BOOL, vformat("layer_%d/y_sort_enabled", i), PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::INT, vformat("layer_%d/y_sort_origin", i), PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::INT, vformat("layer_%d/z_index", i), PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::OBJECT, vformat("layer_%d/tile_data", i), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	}
}

Vector2 TileMap::map_to_world(const Vector2i &p_pos) const {
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
	ERR_FAIL_COND_V(!tile_set.is_valid(), Vector2i());

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
	return Vector2i(ret);
}

bool TileMap::is_existing_neighbor(TileSet::CellNeighbor p_cell_neighbor) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), false);

	TileSet::TileShape shape = tile_set->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		return p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER;

	} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		return p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER ||
			   p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
	} else {
		if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			return p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE ||
				   p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
				   p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
				   p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE ||
				   p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
				   p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
		} else {
			return p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE ||
				   p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE ||
				   p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE ||
				   p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE ||
				   p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE ||
				   p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
		}
	}
}

Vector2i TileMap::get_neighbor_cell(const Vector2i &p_coords, TileSet::CellNeighbor p_cell_neighbor) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), p_coords);

	TileSet::TileShape shape = tile_set->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		switch (p_cell_neighbor) {
			case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
				return p_coords + Vector2i(1, 0);
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
				return p_coords + Vector2i(1, 1);
			case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
				return p_coords + Vector2i(0, 1);
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
				return p_coords + Vector2i(-1, 1);
			case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
				return p_coords + Vector2i(-1, 0);
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				return p_coords + Vector2i(-1, -1);
			case TileSet::CELL_NEIGHBOR_TOP_SIDE:
				return p_coords + Vector2i(0, -1);
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
				return p_coords + Vector2i(1, -1);
			default:
				ERR_FAIL_V(p_coords);
		}
	} else { // Half-offset shapes (square and hexagon)
		switch (tile_set->get_tile_layout()) {
			case TileSet::TILE_LAYOUT_STACKED: {
				if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
					bool is_offset = p_coords.y % 2;
					if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
						return p_coords + Vector2i(1, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
						return p_coords + Vector2i(is_offset ? 1 : 0, 1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
						return p_coords + Vector2i(0, 2);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
						return p_coords + Vector2i(is_offset ? 0 : -1, 1);
					} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
							   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
						return p_coords + Vector2i(-1, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
						return p_coords + Vector2i(is_offset ? 0 : -1, -1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
						return p_coords + Vector2i(0, -2);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
						return p_coords + Vector2i(is_offset ? 1 : 0, -1);
					} else {
						ERR_FAIL_V(p_coords);
					}
				} else {
					bool is_offset = p_coords.x % 2;

					if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
						return p_coords + Vector2i(0, 1);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
						return p_coords + Vector2i(1, is_offset ? 1 : 0);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
						return p_coords + Vector2i(2, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
						return p_coords + Vector2i(1, is_offset ? 0 : -1);
					} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
							   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
						return p_coords + Vector2i(0, -1);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
						return p_coords + Vector2i(-1, is_offset ? 0 : -1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
						return p_coords + Vector2i(-2, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
						return p_coords + Vector2i(-1, is_offset ? 1 : 0);
					} else {
						ERR_FAIL_V(p_coords);
					}
				}
			} break;
			case TileSet::TILE_LAYOUT_STACKED_OFFSET: {
				if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
					bool is_offset = p_coords.y % 2;

					if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
						return p_coords + Vector2i(1, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
						return p_coords + Vector2i(is_offset ? 0 : 1, 1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
						return p_coords + Vector2i(0, 2);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
						return p_coords + Vector2i(is_offset ? -1 : 0, 1);
					} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
							   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
						return p_coords + Vector2i(-1, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
						return p_coords + Vector2i(is_offset ? -1 : 0, -1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
						return p_coords + Vector2i(0, -2);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
						return p_coords + Vector2i(is_offset ? 0 : 1, -1);
					} else {
						ERR_FAIL_V(p_coords);
					}
				} else {
					bool is_offset = p_coords.x % 2;

					if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
							(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
						return p_coords + Vector2i(0, 1);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
						return p_coords + Vector2i(1, is_offset ? 0 : 1);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
						return p_coords + Vector2i(2, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
						return p_coords + Vector2i(1, is_offset ? -1 : 0);
					} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
							   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
						return p_coords + Vector2i(0, -1);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
						return p_coords + Vector2i(-1, is_offset ? -1 : 0);
					} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
						return p_coords + Vector2i(-2, 0);
					} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
						return p_coords + Vector2i(-1, is_offset ? 0 : 1);
					} else {
						ERR_FAIL_V(p_coords);
					}
				}
			} break;
			case TileSet::TILE_LAYOUT_STAIRS_RIGHT:
			case TileSet::TILE_LAYOUT_STAIRS_DOWN: {
				if ((tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_STAIRS_RIGHT) ^ (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
					if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
							return p_coords + Vector2i(1, 0);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
							return p_coords + Vector2i(-1, 2);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
								   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
							return p_coords + Vector2i(-1, 0);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
							return p_coords + Vector2i(1, -2);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, -1);
						} else {
							ERR_FAIL_V(p_coords);
						}

					} else {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
							return p_coords + Vector2i(0, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
							return p_coords + Vector2i(2, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, -1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
								   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
							return p_coords + Vector2i(0, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
							return p_coords + Vector2i(-2, 1);

						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 1);
						} else {
							ERR_FAIL_V(p_coords);
						}
					}
				} else {
					if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
							return p_coords + Vector2i(2, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
							return p_coords + Vector2i(0, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
								   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
							return p_coords + Vector2i(-2, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
							return p_coords + Vector2i(0, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, -1);
						} else {
							ERR_FAIL_V(p_coords);
						}

					} else {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
							return p_coords + Vector2i(-1, 2);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
							return p_coords + Vector2i(1, 0);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, -1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
								   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
							return p_coords + Vector2i(1, -2);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
							return p_coords + Vector2i(-1, 0);

						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 1);
						} else {
							ERR_FAIL_V(p_coords);
						}
					}
				}
			} break;
			case TileSet::TILE_LAYOUT_DIAMOND_RIGHT:
			case TileSet::TILE_LAYOUT_DIAMOND_DOWN: {
				if ((tile_set->get_tile_layout() == TileSet::TILE_LAYOUT_DIAMOND_RIGHT) ^ (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL)) {
					if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
							return p_coords + Vector2i(1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
							return p_coords + Vector2i(-1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
								   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
							return p_coords + Vector2i(-1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
							return p_coords + Vector2i(1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else {
							ERR_FAIL_V(p_coords);
						}

					} else {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
							return p_coords + Vector2i(1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
							return p_coords + Vector2i(1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
								   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
							return p_coords + Vector2i(-1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
							return p_coords + Vector2i(-1, 1);

						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else {
							ERR_FAIL_V(p_coords);
						}
					}
				} else {
					if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
							return p_coords + Vector2i(1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) {
							return p_coords + Vector2i(1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) ||
								   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
							return p_coords + Vector2i(-1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) {
							return p_coords + Vector2i(-1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else {
							ERR_FAIL_V(p_coords);
						}

					} else {
						if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_CORNER) ||
								(shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
							return p_coords + Vector2i(-1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) {
							return p_coords + Vector2i(0, 1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_RIGHT_CORNER) {
							return p_coords + Vector2i(1, 1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE) {
							return p_coords + Vector2i(1, 0);
						} else if ((shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_CORNER) ||
								   (shape != TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
							return p_coords + Vector2i(1, -1);
						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE) {
							return p_coords + Vector2i(0, -1);
						} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC && p_cell_neighbor == TileSet::CELL_NEIGHBOR_LEFT_CORNER) {
							return p_coords + Vector2i(-1, -1);

						} else if (p_cell_neighbor == TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) {
							return p_coords + Vector2i(-1, 0);
						} else {
							ERR_FAIL_V(p_coords);
						}
					}
				}
			} break;
			default:
				ERR_FAIL_V(p_coords);
		}
	}
}

TypedArray<Vector2i> TileMap::get_used_cells(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), TypedArray<Vector2i>());

	// Returns the cells used in the tilemap.
	TypedArray<Vector2i> a;
	a.resize(layers[p_layer].tile_map.size());
	int i = 0;
	for (const KeyValue<Vector2i, TileMapCell> &E : layers[p_layer].tile_map) {
		Vector2i p(E.key.x, E.key.y);
		a[i++] = p;
	}

	return a;
}

Rect2 TileMap::get_used_rect() { // Not const because of cache
	// Return the rect of the currently used area
	if (used_rect_cache_dirty) {
		bool first = true;
		used_rect_cache = Rect2i();

		for (unsigned int i = 0; i < layers.size(); i++) {
			const Map<Vector2i, TileMapCell> &tile_map = layers[i].tile_map;
			if (tile_map.size() > 0) {
				if (first) {
					used_rect_cache = Rect2i(tile_map.front()->key().x, tile_map.front()->key().y, 0, 0);
					first = false;
				}

				for (const KeyValue<Vector2i, TileMapCell> &E : tile_map) {
					used_rect_cache.expand_to(Vector2i(E.key.x, E.key.y));
				}
			}
		}

		if (!first) { // first is true if every layer is empty.
			used_rect_cache.size += Vector2i(1, 1); // The cache expands to top-left coordinate, so we add one full tile.
		}
		used_rect_cache_dirty = false;
	}

	return used_rect_cache;
}

// --- Override some methods of the CanvasItem class to pass the changes to the quadrants CanvasItems ---

void TileMap::set_light_mask(int p_light_mask) {
	// Occlusion: set light mask.
	CanvasItem::set_light_mask(p_light_mask);
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		for (const KeyValue<Vector2i, TileMapQuadrant> &E : layers[layer].quadrant_map) {
			for (const RID &ci : E.value.canvas_items) {
				RenderingServer::get_singleton()->canvas_item_set_light_mask(ci, get_light_mask());
			}
		}
		_rendering_update_layer(layer);
	}
}

void TileMap::set_material(const Ref<Material> &p_material) {
	// Set material for the whole tilemap.
	CanvasItem::set_material(p_material);

	// Update material for the whole tilemap.
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		for (KeyValue<Vector2i, TileMapQuadrant> &E : layers[layer].quadrant_map) {
			TileMapQuadrant &q = E.value;
			for (const RID &ci : q.canvas_items) {
				RS::get_singleton()->canvas_item_set_use_parent_material(ci, get_use_parent_material() || get_material().is_valid());
			}
		}
		_rendering_update_layer(layer);
	}
}

void TileMap::set_use_parent_material(bool p_use_parent_material) {
	// Set use_parent_material for the whole tilemap.
	CanvasItem::set_use_parent_material(p_use_parent_material);

	// Update use_parent_material for the whole tilemap.
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		for (KeyValue<Vector2i, TileMapQuadrant> &E : layers[layer].quadrant_map) {
			TileMapQuadrant &q = E.value;
			for (const RID &ci : q.canvas_items) {
				RS::get_singleton()->canvas_item_set_use_parent_material(ci, get_use_parent_material() || get_material().is_valid());
			}
		}
		_rendering_update_layer(layer);
	}
}

void TileMap::set_texture_filter(TextureFilter p_texture_filter) {
	// Set a default texture filter for the whole tilemap
	CanvasItem::set_texture_filter(p_texture_filter);
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		for (Map<Vector2i, TileMapQuadrant>::Element *F = layers[layer].quadrant_map.front(); F; F = F->next()) {
			TileMapQuadrant &q = F->get();
			for (const RID &ci : q.canvas_items) {
				RenderingServer::get_singleton()->canvas_item_set_default_texture_filter(ci, RS::CanvasItemTextureFilter(p_texture_filter));
				_make_quadrant_dirty(F);
			}
		}
		_rendering_update_layer(layer);
	}
}

void TileMap::set_texture_repeat(CanvasItem::TextureRepeat p_texture_repeat) {
	// Set a default texture repeat for the whole tilemap
	CanvasItem::set_texture_repeat(p_texture_repeat);
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		for (Map<Vector2i, TileMapQuadrant>::Element *F = layers[layer].quadrant_map.front(); F; F = F->next()) {
			TileMapQuadrant &q = F->get();
			for (const RID &ci : q.canvas_items) {
				RenderingServer::get_singleton()->canvas_item_set_default_texture_repeat(ci, RS::CanvasItemTextureRepeat(p_texture_repeat));
				_make_quadrant_dirty(F);
			}
		}
		_rendering_update_layer(layer);
	}
}

TypedArray<Vector2i> TileMap::get_surrounding_tiles(Vector2i coords) {
	if (!tile_set.is_valid()) {
		return TypedArray<Vector2i>();
	}

	TypedArray<Vector2i> around;
	TileSet::TileShape shape = tile_set->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_LEFT_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_SIDE));
	} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE));
		around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE));
	} else {
		if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_LEFT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE));
		} else {
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_SIDE));
			around.push_back(get_neighbor_cell(coords, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE));
		}
	}

	return around;
}

void TileMap::draw_cells_outline(Control *p_control, Set<Vector2i> p_cells, Color p_color, Transform2D p_transform) {
	if (!tile_set.is_valid()) {
		return;
	}

	// Create a set.
	Vector2i tile_size = tile_set->get_tile_size();
	Vector<Vector2> polygon = tile_set->get_tile_shape_polygon();
	TileSet::TileShape shape = tile_set->get_tile_shape();

	for (Set<Vector2i>::Element *E = p_cells.front(); E; E = E->next()) {
		Vector2 center = map_to_world(E->get());

#define DRAW_SIDE_IF_NEEDED(side, polygon_index_from, polygon_index_to)                     \
	if (!p_cells.has(get_neighbor_cell(E->get(), side))) {                                  \
		Vector2 from = p_transform.xform(center + polygon[polygon_index_from] * tile_size); \
		Vector2 to = p_transform.xform(center + polygon[polygon_index_to] * tile_size);     \
		p_control->draw_line(from, to, p_color);                                            \
	}

		if (shape == TileSet::TILE_SHAPE_SQUARE) {
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_RIGHT_SIDE, 1, 2);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, 2, 3);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_LEFT_SIDE, 3, 0);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_SIDE, 0, 1);
		} else {
			if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
				if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, 3, 4);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, 2, 3);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, 0, 1);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, 5, 0);
				} else {
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, 3, 4);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, 2, 3);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_LEFT_SIDE, 1, 2);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, 0, 1);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, 5, 0);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_RIGHT_SIDE, 4, 5);
				}
			} else {
				if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, 3, 4);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, 5, 0);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, 0, 1);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, 2, 3);
				} else {
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, 3, 4);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, 4, 5);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, 5, 0);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, 0, 1);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_SIDE, 1, 2);
					DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, 2, 3);
				}
			}
		}
	}
#undef DRAW_SIDE_IF_NEEDED
}

TypedArray<String> TileMap::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	// Retrieve the set of Z index values with a Y-sorted layer.
	Set<int> y_sorted_z_index;
	for (int layer = 0; layer < (int)layers.size(); layer++) {
		if (layers[layer].y_sort_enabled) {
			y_sorted_z_index.insert(layers[layer].z_index);
		}
	}

	// Check if we have a non-sorted layer in a Z-index with a Y-sorted layer.
	for (int layer = 0; layer < (int)layers.size(); layer++) {
		if (!layers[layer].y_sort_enabled && y_sorted_z_index.has(layers[layer].z_index)) {
			warnings.push_back(TTR("A Y-sorted layer has the same Z-index value as a not Y-sorted layer.\nThis may lead to unwanted behaviors, as a layer that is not Y-sorted will be Y-sorted as a whole with tiles from Y-sorted layers."));
			break;
		}
	}

	return warnings;
}

void TileMap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_tileset", "tileset"), &TileMap::set_tileset);
	ClassDB::bind_method(D_METHOD("get_tileset"), &TileMap::get_tileset);

	ClassDB::bind_method(D_METHOD("set_quadrant_size", "size"), &TileMap::set_quadrant_size);
	ClassDB::bind_method(D_METHOD("get_quadrant_size"), &TileMap::get_quadrant_size);

	ClassDB::bind_method(D_METHOD("get_layers_count"), &TileMap::get_layers_count);
	ClassDB::bind_method(D_METHOD("add_layer", "to_position"), &TileMap::add_layer);
	ClassDB::bind_method(D_METHOD("move_layer", "layer", "to_position"), &TileMap::move_layer);
	ClassDB::bind_method(D_METHOD("remove_layer", "layer"), &TileMap::remove_layer);
	ClassDB::bind_method(D_METHOD("set_layer_name", "layer", "name"), &TileMap::set_layer_name);
	ClassDB::bind_method(D_METHOD("get_layer_name", "layer"), &TileMap::get_layer_name);
	ClassDB::bind_method(D_METHOD("set_layer_enabled", "layer", "enabled"), &TileMap::set_layer_enabled);
	ClassDB::bind_method(D_METHOD("is_layer_enabled", "layer"), &TileMap::is_layer_enabled);
	ClassDB::bind_method(D_METHOD("set_layer_modulate", "layer", "enabled"), &TileMap::set_layer_modulate);
	ClassDB::bind_method(D_METHOD("get_layer_modulate", "layer"), &TileMap::get_layer_modulate);
	ClassDB::bind_method(D_METHOD("set_layer_y_sort_enabled", "layer", "y_sort_enabled"), &TileMap::set_layer_y_sort_enabled);
	ClassDB::bind_method(D_METHOD("is_layer_y_sort_enabled", "layer"), &TileMap::is_layer_y_sort_enabled);
	ClassDB::bind_method(D_METHOD("set_layer_y_sort_origin", "layer", "y_sort_origin"), &TileMap::set_layer_y_sort_origin);
	ClassDB::bind_method(D_METHOD("get_layer_y_sort_origin", "layer"), &TileMap::get_layer_y_sort_origin);
	ClassDB::bind_method(D_METHOD("set_layer_z_index", "layer", "z_index"), &TileMap::set_layer_z_index);
	ClassDB::bind_method(D_METHOD("get_layer_z_index", "layer"), &TileMap::get_layer_z_index);

	ClassDB::bind_method(D_METHOD("set_collision_animatable", "enabled"), &TileMap::set_collision_animatable);
	ClassDB::bind_method(D_METHOD("is_collision_animatable"), &TileMap::is_collision_animatable);
	ClassDB::bind_method(D_METHOD("set_collision_visibility_mode", "collision_visibility_mode"), &TileMap::set_collision_visibility_mode);
	ClassDB::bind_method(D_METHOD("get_collision_visibility_mode"), &TileMap::get_collision_visibility_mode);

	ClassDB::bind_method(D_METHOD("set_navigation_visibility_mode", "navigation_visibility_mode"), &TileMap::set_navigation_visibility_mode);
	ClassDB::bind_method(D_METHOD("get_navigation_visibility_mode"), &TileMap::get_navigation_visibility_mode);

	ClassDB::bind_method(D_METHOD("set_cell", "layer", "coords", "source_id", "atlas_coords", "alternative_tile"), &TileMap::set_cell, DEFVAL(TileSet::INVALID_SOURCE), DEFVAL(TileSetSource::INVALID_ATLAS_COORDS), DEFVAL(TileSetSource::INVALID_TILE_ALTERNATIVE));
	ClassDB::bind_method(D_METHOD("get_cell_source_id", "layer", "coords", "use_proxies"), &TileMap::get_cell_source_id);
	ClassDB::bind_method(D_METHOD("get_cell_atlas_coords", "layer", "coords", "use_proxies"), &TileMap::get_cell_atlas_coords);
	ClassDB::bind_method(D_METHOD("get_cell_alternative_tile", "layer", "coords", "use_proxies"), &TileMap::get_cell_alternative_tile);

	ClassDB::bind_method(D_METHOD("get_coords_for_body_rid", "body"), &TileMap::get_coords_for_body_rid);

	ClassDB::bind_method(D_METHOD("get_pattern", "layer", "coords_array"), &TileMap::get_pattern);
	ClassDB::bind_method(D_METHOD("map_pattern", "position_in_tilemap", "coords_in_pattern", "pattern"), &TileMap::map_pattern);
	ClassDB::bind_method(D_METHOD("set_pattern", "layer", "position", "pattern"), &TileMap::set_pattern);

	ClassDB::bind_method(D_METHOD("set_cells_from_surrounding_terrains", "layer", "cells", "terrain_set", "ignore_empty_terrains"), &TileMap::set_cells_from_surrounding_terrains, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("fix_invalid_tiles"), &TileMap::fix_invalid_tiles);
	ClassDB::bind_method(D_METHOD("clear_layer", "layer"), &TileMap::clear_layer);
	ClassDB::bind_method(D_METHOD("clear"), &TileMap::clear);

	ClassDB::bind_method(D_METHOD("get_surrounding_tiles", "coords"), &TileMap::get_surrounding_tiles);

	ClassDB::bind_method(D_METHOD("get_used_cells", "layer"), &TileMap::get_used_cells);
	ClassDB::bind_method(D_METHOD("get_used_rect"), &TileMap::get_used_rect);

	ClassDB::bind_method(D_METHOD("map_to_world", "map_position"), &TileMap::map_to_world);
	ClassDB::bind_method(D_METHOD("world_to_map", "world_position"), &TileMap::world_to_map);

	ClassDB::bind_method(D_METHOD("get_neighbor_cell", "coords", "neighbor"), &TileMap::get_neighbor_cell);

	ClassDB::bind_method(D_METHOD("_update_dirty_quadrants"), &TileMap::_update_dirty_quadrants);

	ClassDB::bind_method(D_METHOD("_set_tile_data", "layer", "data"), &TileMap::_set_tile_data);
	ClassDB::bind_method(D_METHOD("_get_tile_data", "layer"), &TileMap::_get_tile_data);

	ClassDB::bind_method(D_METHOD("_tile_set_changed_deferred_update"), &TileMap::_tile_set_changed_deferred_update);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "tile_set", PROPERTY_HINT_RESOURCE_TYPE, "TileSet"), "set_tileset", "get_tileset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cell_quadrant_size", PROPERTY_HINT_RANGE, "1,128,1"), "set_quadrant_size", "get_quadrant_size");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collision_animatable"), "set_collision_animatable", "is_collision_animatable");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_visibility_mode", PROPERTY_HINT_ENUM, "Default,Force Show,Force Hide"), "set_collision_visibility_mode", "get_collision_visibility_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_visibility_mode", PROPERTY_HINT_ENUM, "Default,Force Show,Force Hide"), "set_navigation_visibility_mode", "get_navigation_visibility_mode");

	ADD_ARRAY("layers", "layer_");

	ADD_PROPERTY_DEFAULT("format", FORMAT_1);

	ADD_SIGNAL(MethodInfo("changed"));

	BIND_ENUM_CONSTANT(VISIBILITY_MODE_DEFAULT);
	BIND_ENUM_CONSTANT(VISIBILITY_MODE_FORCE_HIDE);
	BIND_ENUM_CONSTANT(VISIBILITY_MODE_FORCE_SHOW);
}

void TileMap::_tile_set_changed() {
	emit_signal(SNAME("changed"));
	_tile_set_changed_deferred_update_needed = true;
	call_deferred("_tile_set_changed_deferred_update");
}

void TileMap::_tile_set_changed_deferred_update() {
	if (_tile_set_changed_deferred_update_needed) {
		_clear_internals();
		_recreate_internals();
		_tile_set_changed_deferred_update_needed = false;
	}
}

TileMap::TileMap() {
	set_notify_transform(true);
	set_notify_local_transform(false);

	layers.resize(1);
}

TileMap::~TileMap() {
	if (tile_set.is_valid()) {
		tile_set->disconnect("changed", callable_mp(this, &TileMap::_tile_set_changed));
	}
	_clear_internals();
}
