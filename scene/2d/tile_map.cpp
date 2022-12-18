/*************************************************************************/
/*  tile_map.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "scene/resources/world_2d.h"
#include "servers/navigation_server_2d.h"

#ifdef DEBUG_ENABLED
#include "servers/navigation_server_3d.h"
#endif // DEBUG_ENABLED

HashMap<Vector2i, TileSet::CellNeighbor> TileMap::TerrainConstraint::get_overlapping_coords_and_peering_bits() const {
	HashMap<Vector2i, TileSet::CellNeighbor> output;

	ERR_FAIL_COND_V(is_center_bit(), output);

	Ref<TileSet> ts = tile_map->get_tileset();
	ERR_FAIL_COND_V(!ts.is_valid(), output);

	TileSet::TileShape shape = ts->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		switch (bit) {
			case 1:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_RIGHT_SIDE;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_LEFT_SIDE;
				break;
			case 2:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER;
				break;
			case 3:
				output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_SIDE;
				output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_SIDE;
				break;
			default:
				ERR_FAIL_V(output);
		}
	} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		switch (bit) {
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
		TileSet::TileOffsetAxis offset_axis = ts->get_tile_offset_axis();
		if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			switch (bit) {
				case 1:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_RIGHT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_LEFT_SIDE;
					break;
				case 2:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_CORNER;
					break;
				case 3:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE;
					break;
				case 4:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER;
					break;
				case 5:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE;
					break;
				default:
					ERR_FAIL_V(output);
			}
		} else {
			switch (bit) {
				case 1:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_RIGHT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
					break;
				case 2:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE;
					break;
				case 3:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)] = TileSet::CELL_NEIGHBOR_LEFT_CORNER;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER;
					break;
				case 4:
					output[base_cell_coords] = TileSet::CELL_NEIGHBOR_BOTTOM_SIDE;
					output[tile_map->get_neighbor_cell(base_cell_coords, TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)] = TileSet::CELL_NEIGHBOR_TOP_SIDE;
					break;
				case 5:
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

TileMap::TerrainConstraint::TerrainConstraint(const TileMap *p_tile_map, const Vector2i &p_position, int p_terrain) {
	tile_map = p_tile_map;

	Ref<TileSet> ts = tile_map->get_tileset();
	ERR_FAIL_COND(!ts.is_valid());

	bit = 0;
	base_cell_coords = p_position;
	terrain = p_terrain;
}

TileMap::TerrainConstraint::TerrainConstraint(const TileMap *p_tile_map, const Vector2i &p_position, const TileSet::CellNeighbor &p_bit, int p_terrain) {
	// The way we build the constraint make it easy to detect conflicting constraints.
	tile_map = p_tile_map;

	Ref<TileSet> ts = tile_map->get_tileset();
	ERR_FAIL_COND(!ts.is_valid());

	TileSet::TileShape shape = ts->get_tile_shape();
	if (shape == TileSet::TILE_SHAPE_SQUARE) {
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
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
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
				bit = 1;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_SIDE:
				bit = 3;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
				break;
			default:
				ERR_FAIL();
				break;
		}
	} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
		switch (p_bit) {
			case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
				bit = 1;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
				bit = 2;
				base_cell_coords = p_position;
				break;
			case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
				bit = 3;
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
			case TileSet::CELL_NEIGHBOR_TOP_CORNER:
				bit = 2;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_CORNER);
				break;
			case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
				bit = 3;
				base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
				break;
			default:
				ERR_FAIL();
				break;
		}
	} else {
		// Half-offset shapes
		TileSet::TileOffsetAxis offset_axis = ts->get_tile_offset_axis();
		if (offset_axis == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
			switch (p_bit) {
				case TileSet::CELL_NEIGHBOR_RIGHT_SIDE:
					bit = 1;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
					bit = 2;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
					bit = 3;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_CORNER:
					bit = 4;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
					bit = 5;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
					bit = 2;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_LEFT_SIDE:
					bit = 1;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
					bit = 4;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
					bit = 3;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_CORNER:
					bit = 2;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
					bit = 5;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
					bit = 4;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE);
					break;
				default:
					ERR_FAIL();
					break;
			}
		} else {
			switch (p_bit) {
				case TileSet::CELL_NEIGHBOR_RIGHT_CORNER:
					bit = 1;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE:
					bit = 2;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER:
					bit = 3;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_SIDE:
					bit = 4;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER:
					bit = 1;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE:
					bit = 5;
					base_cell_coords = p_position;
					break;
				case TileSet::CELL_NEIGHBOR_LEFT_CORNER:
					bit = 3;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE:
					bit = 2;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER:
					bit = 1;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_SIDE:
					bit = 4;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER:
					bit = 3;
					base_cell_coords = p_tile_map->get_neighbor_cell(p_position, TileSet::CELL_NEIGHBOR_TOP_SIDE);
					break;
				case TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE:
					bit = 5;
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

Vector2i TileMap::transform_coords_layout(const Vector2i &p_coords, TileSet::TileOffsetAxis p_offset_axis, TileSet::TileLayout p_from_layout, TileSet::TileLayout p_to_layout) {
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
		p_to_pos = layers.size() + p_to_pos + 1;
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
	layers.remove_at(p_to_pos < p_layer ? p_layer + 1 : p_layer);
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

	layers.remove_at(p_layer);
	_recreate_internals();
	notify_property_list_changed();

	if (selected_layer >= p_layer) {
		selected_layer -= 1;
	}

	emit_signal(SNAME("changed"));

	update_configuration_warnings();
}

void TileMap::set_layer_name(int p_layer, String p_name) {
	if (p_layer < 0) {
		p_layer = layers.size() + p_layer;
	}
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	layers[p_layer].name = p_name;
	emit_signal(SNAME("changed"));
}

String TileMap::get_layer_name(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), String());
	return layers[p_layer].name;
}

void TileMap::set_layer_enabled(int p_layer, bool p_enabled) {
	if (p_layer < 0) {
		p_layer = layers.size() + p_layer;
	}
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
	if (p_layer < 0) {
		p_layer = layers.size() + p_layer;
	}
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
	if (p_layer < 0) {
		p_layer = layers.size() + p_layer;
	}
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
	if (p_layer < 0) {
		p_layer = layers.size() + p_layer;
	}
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
	if (p_layer < 0) {
		p_layer = layers.size() + p_layer;
	}
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
	update_configuration_warnings();
}

Vector2i TileMap::_coords_to_quadrant_coords(int p_layer, const Vector2i &p_coords) const {
	int quad_size = get_effective_quadrant_size(p_layer);

	// Rounding down, instead of simply rounding towards zero (truncating)
	return Vector2i(
			p_coords.x > 0 ? p_coords.x / quad_size : (p_coords.x - (quad_size - 1)) / quad_size,
			p_coords.y > 0 ? p_coords.y / quad_size : (p_coords.y - (quad_size - 1)) / quad_size);
}

HashMap<Vector2i, TileMapQuadrant>::Iterator TileMap::_create_quadrant(int p_layer, const Vector2i &p_qk) {
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

void TileMap::_make_quadrant_dirty(HashMap<Vector2i, TileMapQuadrant>::Iterator Q) {
	// Make the given quadrant dirty, then trigger an update later.
	TileMapQuadrant &q = Q->value;
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
		SelfList<TileMapQuadrant>::List &dirty_quadrant_list = layers[layer].dirty_quadrant_list;

		// Update the coords cache.
		for (SelfList<TileMapQuadrant> *q = dirty_quadrant_list.first(); q; q = q->next()) {
			q->self()->map_to_local.clear();
			q->self()->local_to_map.clear();
			for (const Vector2i &E : q->self()->cells) {
				Vector2i pk = E;
				Vector2i pk_local_coords = map_to_local(pk);
				q->self()->map_to_local[pk] = pk_local_coords;
				q->self()->local_to_map[pk_local_coords] = pk;
			}
		}

		// Find TileData that need a runtime modification.
		_build_runtime_update_tile_data(dirty_quadrant_list);

		// Call the update_dirty_quadrant method on plugins.
		_rendering_update_dirty_quadrants(dirty_quadrant_list);
		_physics_update_dirty_quadrants(dirty_quadrant_list);
		_navigation_update_dirty_quadrants(dirty_quadrant_list);
		_scenes_update_dirty_quadrants(dirty_quadrant_list);

		// Redraw the debug canvas_items.
		RenderingServer *rs = RenderingServer::get_singleton();
		for (SelfList<TileMapQuadrant> *q = dirty_quadrant_list.first(); q; q = q->next()) {
			rs->canvas_item_clear(q->self()->debug_canvas_item);
			Transform2D xform;
			xform.set_origin(map_to_local(q->self()->coords * get_effective_quadrant_size(layer)));
			rs->canvas_item_set_transform(q->self()->debug_canvas_item, xform);

			_rendering_draw_quadrant_debug(q->self());
			_physics_draw_quadrant_debug(q->self());
			_navigation_draw_quadrant_debug(q->self());
			_scenes_draw_quadrant_debug(q->self());
		}

		// Clear the list
		while (dirty_quadrant_list.first()) {
			// Clear the runtime tile data.
			for (const KeyValue<Vector2i, TileData *> &kv : dirty_quadrant_list.first()->self()->runtime_tile_data_cache) {
				memdelete(kv.value);
			}

			dirty_quadrant_list.remove(dirty_quadrant_list.first());
		}
	}

	pending_update = false;

	_recompute_rect_cache();
}

void TileMap::_recreate_layer_internals(int p_layer) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	// Make sure that _clear_internals() was called prior.
	ERR_FAIL_COND_MSG(layers[p_layer].quadrant_map.size() > 0, "TileMap layer " + itos(p_layer) + " had a non-empty quadrant map.");

	if (!layers[p_layer].enabled) {
		return;
	}

	// Update the layer internals.
	_rendering_update_layer(p_layer);

	// Recreate the quadrants.
	const HashMap<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	for (const KeyValue<Vector2i, TileMapCell> &E : tile_map) {
		Vector2i qk = _coords_to_quadrant_coords(p_layer, Vector2i(E.key.x, E.key.y));

		HashMap<Vector2i, TileMapQuadrant>::Iterator Q = layers[p_layer].quadrant_map.find(qk);
		if (!Q) {
			Q = _create_quadrant(p_layer, qk);
			layers[p_layer].dirty_quadrant_list.add(&Q->value.dirty_list_element);
		}

		Vector2i pk = E.key;
		Q->value.cells.insert(pk);

		_make_quadrant_dirty(Q);
	}

	_queue_update_dirty_quadrants();
}

void TileMap::_recreate_internals() {
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		_recreate_layer_internals(layer);
	}
}

void TileMap::_erase_quadrant(HashMap<Vector2i, TileMapQuadrant>::Iterator Q) {
	// Remove a quadrant.
	TileMapQuadrant *q = &(Q->value);

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

	layers[q->layer].quadrant_map.remove(Q);
	rect_cache_dirty = true;
}

void TileMap::_clear_layer_internals(int p_layer) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	// Clear quadrants.
	while (layers[p_layer].quadrant_map.size()) {
		_erase_quadrant(layers[p_layer].quadrant_map.begin());
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
	bool first = true;
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		for (const KeyValue<Vector2i, TileMapQuadrant> &E : layers[layer].quadrant_map) {
			Rect2 r;
			r.position = map_to_local(E.key * get_effective_quadrant_size(layer));
			r.expand_to(map_to_local((E.key + Vector2i(1, 0)) * get_effective_quadrant_size(layer)));
			r.expand_to(map_to_local((E.key + Vector2i(1, 1)) * get_effective_quadrant_size(layer)));
			r.expand_to(map_to_local((E.key + Vector2i(0, 1)) * get_effective_quadrant_size(layer)));
			if (first) {
				r_total = r;
				first = false;
			} else {
				r_total = r_total.merge(r);
			}
		}
	}

	bool changed = rect_cache != r_total;

	rect_cache = r_total;

	item_rect_changed(changed);

	rect_cache_dirty = false;
#endif
}

/////////////////////////////// Rendering //////////////////////////////////////

void TileMap::_rendering_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_CANVAS: {
			bool node_visible = is_visible_in_tree();
			for (int layer = 0; layer < (int)layers.size(); layer++) {
				for (KeyValue<Vector2i, TileMapQuadrant> &E_quadrant : layers[layer].quadrant_map) {
					TileMapQuadrant &q = E_quadrant.value;
					for (const KeyValue<Vector2i, RID> &kv : q.occluders) {
						Transform2D xform;
						xform.set_origin(map_to_local(kv.key));
						RS::get_singleton()->canvas_light_occluder_attach_to_canvas(kv.value, get_canvas());
						RS::get_singleton()->canvas_light_occluder_set_transform(kv.value, get_global_transform() * xform);
						RS::get_singleton()->canvas_light_occluder_set_enabled(kv.value, node_visible);
					}
				}
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			bool node_visible = is_visible_in_tree();
			for (int layer = 0; layer < (int)layers.size(); layer++) {
				for (KeyValue<Vector2i, TileMapQuadrant> &E_quadrant : layers[layer].quadrant_map) {
					TileMapQuadrant &q = E_quadrant.value;

					// Update occluders transform.
					for (const KeyValue<Vector2i, Vector2i> &E_cell : q.local_to_map) {
						Transform2D xform;
						xform.set_origin(E_cell.key);
						for (const KeyValue<Vector2i, RID> &kv : q.occluders) {
							RS::get_singleton()->canvas_light_occluder_set_enabled(kv.value, node_visible);
						}
					}
				}
			}
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (!is_inside_tree()) {
				return;
			}
			for (int layer = 0; layer < (int)layers.size(); layer++) {
				for (KeyValue<Vector2i, TileMapQuadrant> &E_quadrant : layers[layer].quadrant_map) {
					TileMapQuadrant &q = E_quadrant.value;

					// Update occluders transform.
					for (const KeyValue<Vector2i, RID> &kv : q.occluders) {
						Transform2D xform;
						xform.set_origin(map_to_local(kv.key));
						RenderingServer::get_singleton()->canvas_light_occluder_set_transform(kv.value, get_global_transform() * xform);
					}
				}
			}
		} break;

		case NOTIFICATION_DRAW: {
			if (tile_set.is_valid()) {
				RenderingServer::get_singleton()->canvas_item_set_sort_children_by_y(get_canvas_item(), is_y_sort_enabled());
			}
		} break;

		case NOTIFICATION_EXIT_CANVAS: {
			for (int layer = 0; layer < (int)layers.size(); layer++) {
				for (KeyValue<Vector2i, TileMapQuadrant> &E_quadrant : layers[layer].quadrant_map) {
					TileMapQuadrant &q = E_quadrant.value;
					for (const KeyValue<Vector2i, RID> &kv : q.occluders) {
						RS::get_singleton()->canvas_light_occluder_attach_to_canvas(kv.value, RID());
					}
				}
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
		rs->canvas_item_set_draw_index(ci, p_layer - (int64_t)0x80000000);

		layers[p_layer].canvas_item = ci;
	}
	RID &ci = layers[p_layer].canvas_item;
	rs->canvas_item_set_sort_children_by_y(ci, layers[p_layer].y_sort_enabled);
	rs->canvas_item_set_use_parent_material(ci, get_use_parent_material() || get_material().is_valid());
	rs->canvas_item_set_z_index(ci, layers[p_layer].z_index);
	rs->canvas_item_set_default_texture_filter(ci, RS::CanvasItemTextureFilter(get_texture_filter_in_tree()));
	rs->canvas_item_set_default_texture_repeat(ci, RS::CanvasItemTextureRepeat(get_texture_repeat_in_tree()));
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

	bool node_visible = is_visible_in_tree();

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
		for (const KeyValue<Vector2i, RID> &kv : q.occluders) {
			rs->free(kv.value);
		}
		q.occluders.clear();

		// Those allow to group cell per material or z-index.
		Ref<Material> prev_material;
		int prev_z_index = 0;
		RID prev_ci;

		Color tile_modulate = get_self_modulate();
		tile_modulate *= get_layer_modulate(q.layer);
		if (selected_layer >= 0) {
			int z1 = get_layer_z_index(q.layer);
			int z2 = get_layer_z_index(selected_layer);
			if (z1 < z2 || (z1 == z2 && q.layer < selected_layer)) {
				tile_modulate = tile_modulate.darkened(0.5);
			} else if (z1 > z2 || (z1 == z2 && q.layer > selected_layer)) {
				tile_modulate = tile_modulate.darkened(0.5);
				tile_modulate.a *= 0.3;
			}
		}

		// Iterate over the cells of the quadrant.
		for (const KeyValue<Vector2i, Vector2i> &E_cell : q.local_to_map) {
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
					const TileData *tile_data;
					if (q.runtime_tile_data_cache.has(E_cell.value)) {
						tile_data = q.runtime_tile_data_cache[E_cell.value];
					} else {
						tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);
					}

					Ref<Material> mat = tile_data->get_material();
					int tile_z_index = tile_data->get_z_index();

					// Quandrant pos.
					Vector2 tile_position = map_to_local(q.coords * get_effective_quadrant_size(q.layer));
					if (is_y_sort_enabled() && layers[q.layer].y_sort_enabled) {
						// When Y-sorting, the quandrant size is sure to be 1, we can thus offset the CanvasItem.
						tile_position.y += layers[q.layer].y_sort_origin + tile_data->get_y_sort_origin();
					}

					// --- CanvasItems ---
					// Create two canvas items, for rendering and debug.
					RID ci;

					// Check if the material or the z_index changed.
					if (prev_ci == RID() || prev_material != mat || prev_z_index != tile_z_index) {
						// If so, create a new CanvasItem.
						ci = rs->canvas_item_create();
						if (mat.is_valid()) {
							rs->canvas_item_set_material(ci, mat->get_rid());
						}
						rs->canvas_item_set_parent(ci, layers[q.layer].canvas_item);
						rs->canvas_item_set_use_parent_material(ci, get_use_parent_material() || get_material().is_valid());

						Transform2D xform;
						xform.set_origin(tile_position);
						rs->canvas_item_set_transform(ci, xform);

						rs->canvas_item_set_light_mask(ci, get_light_mask());
						rs->canvas_item_set_z_as_relative_to_parent(ci, true);
						rs->canvas_item_set_z_index(ci, tile_z_index);

						rs->canvas_item_set_default_texture_filter(ci, RS::CanvasItemTextureFilter(get_texture_filter_in_tree()));
						rs->canvas_item_set_default_texture_repeat(ci, RS::CanvasItemTextureRepeat(get_texture_repeat_in_tree()));

						q.canvas_items.push_back(ci);

						prev_ci = ci;
						prev_material = mat;
						prev_z_index = tile_z_index;

					} else {
						// Keep the same canvas_item to draw on.
						ci = prev_ci;
					}

					// Drawing the tile in the canvas item.
					draw_tile(ci, E_cell.key - tile_position, tile_set, c.source_id, c.get_atlas_coords(), c.alternative_tile, -1, tile_modulate, tile_data);

					// --- Occluders ---
					for (int i = 0; i < tile_set->get_occlusion_layers_count(); i++) {
						Transform2D xform;
						xform.set_origin(E_cell.key);
						if (tile_data->get_occluder(i).is_valid()) {
							RID occluder_id = rs->canvas_light_occluder_create();
							rs->canvas_light_occluder_set_enabled(occluder_id, node_visible);
							rs->canvas_light_occluder_set_transform(occluder_id, get_global_transform() * xform);
							rs->canvas_light_occluder_set_polygon(occluder_id, tile_data->get_occluder(i)->get_rid());
							rs->canvas_light_occluder_attach_to_canvas(occluder_id, get_canvas());
							rs->canvas_light_occluder_set_light_mask(occluder_id, tile_set->get_occlusion_layer_light_mask(i));
							q.occluders[E_cell.value] = occluder_id;
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
			// Sort the quadrants coords per local coordinates.
			RBMap<Vector2i, Vector2i, TileMapQuadrant::CoordsWorldComparator> local_to_map;
			for (const KeyValue<Vector2i, TileMapQuadrant> &E : layers[layer].quadrant_map) {
				local_to_map[map_to_local(E.key)] = E.key;
			}

			// Sort the quadrants.
			for (const KeyValue<Vector2i, Vector2i> &E : local_to_map) {
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
	for (const KeyValue<Vector2i, RID> &kv : p_quadrant->occluders) {
		RenderingServer::get_singleton()->free(kv.value);
	}
	p_quadrant->occluders.clear();
}

void TileMap::_rendering_draw_quadrant_debug(TileMapQuadrant *p_quadrant) {
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	// Draw a placeholder for tiles needing one.
	RenderingServer *rs = RenderingServer::get_singleton();
	Vector2 quadrant_pos = map_to_local(p_quadrant->coords * get_effective_quadrant_size(p_quadrant->layer));
	for (const Vector2i &E_cell : p_quadrant->cells) {
		const TileMapCell &c = get_cell(p_quadrant->layer, E_cell, true);

		TileSetSource *source;
		if (tile_set->has_source(c.source_id)) {
			source = *tile_set->get_source(c.source_id);

			if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
				continue;
			}

			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				Vector2i grid_size = atlas_source->get_atlas_grid_size();
				if (!atlas_source->get_runtime_texture().is_valid() || c.get_atlas_coords().x >= grid_size.x || c.get_atlas_coords().y >= grid_size.y) {
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
					Transform2D cell_to_quadrant;
					cell_to_quadrant.set_origin(map_to_local(E_cell) - quadrant_pos);
					rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, cell_to_quadrant);
					rs->canvas_item_add_circle(p_quadrant->debug_canvas_item, Vector2(), MIN(tile_set->get_tile_size().x, tile_set->get_tile_size().y) / 4.0, color);
				}
			}
		}
	}
}

void TileMap::draw_tile(RID p_canvas_item, const Vector2i &p_position, const Ref<TileSet> p_tile_set, int p_atlas_source_id, const Vector2i &p_atlas_coords, int p_alternative_tile, int p_frame, Color p_modulation, const TileData *p_tile_data_override) {
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
		Ref<Texture2D> tex = atlas_source->get_runtime_texture();
		if (!tex.is_valid()) {
			return;
		}

		// Check if we are in the texture, return otherwise.
		Vector2i grid_size = atlas_source->get_atlas_grid_size();
		if (p_atlas_coords.x >= grid_size.x || p_atlas_coords.y >= grid_size.y) {
			return;
		}

		// Get tile data.
		const TileData *tile_data = p_tile_data_override ? p_tile_data_override : atlas_source->get_tile_data(p_atlas_coords, p_alternative_tile);

		// Get the tile modulation.
		Color modulate = tile_data->get_modulate() * p_modulation;

		// Compute the offset.
		Vector2i tile_offset = atlas_source->get_tile_effective_texture_offset(p_atlas_coords, p_alternative_tile);

		// Get destination rect.
		Rect2 dest_rect;
		dest_rect.size = atlas_source->get_runtime_tile_texture_region(p_atlas_coords).size;
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
			Rect2i source_rect = atlas_source->get_runtime_tile_texture_region(p_atlas_coords, p_frame);
			tex->draw_rect_region(p_canvas_item, dest_rect, source_rect, modulate, transpose, p_tile_set->is_uv_clipping());
		} else if (atlas_source->get_tile_animation_frames_count(p_atlas_coords) == 1) {
			Rect2i source_rect = atlas_source->get_runtime_tile_texture_region(p_atlas_coords, 0);
			tex->draw_rect_region(p_canvas_item, dest_rect, source_rect, modulate, transpose, p_tile_set->is_uv_clipping());
		} else {
			real_t speed = atlas_source->get_tile_animation_speed(p_atlas_coords);
			real_t animation_duration = atlas_source->get_tile_animation_total_duration(p_atlas_coords) / speed;
			real_t time = 0.0;
			for (int frame = 0; frame < atlas_source->get_tile_animation_frames_count(p_atlas_coords); frame++) {
				real_t frame_duration = atlas_source->get_tile_animation_frame_duration(p_atlas_coords, frame) / speed;
				RenderingServer::get_singleton()->canvas_item_add_animation_slice(p_canvas_item, animation_duration, time, time + frame_duration, 0.0);

				Rect2i source_rect = atlas_source->get_runtime_tile_texture_region(p_atlas_coords, frame);
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
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			bool in_editor = false;
#ifdef TOOLS_ENABLED
			in_editor = Engine::get_singleton()->is_editor_hint();
#endif
			if (is_inside_tree() && collision_animatable && !in_editor) {
				// Update transform on the physics tick when in animatable mode.
				last_valid_transform = new_transform;
				set_notify_local_transform(false);
				set_global_transform(new_transform);
				set_notify_local_transform(true);
			}
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			bool in_editor = false;
#ifdef TOOLS_ENABLED
			in_editor = Engine::get_singleton()->is_editor_hint();
#endif
			if (is_inside_tree() && (!collision_animatable || in_editor)) {
				// Update the new transform directly if we are not in animatable mode.
				Transform2D gl_transform = get_global_transform();
				for (int layer = 0; layer < (int)layers.size(); layer++) {
					for (KeyValue<Vector2i, TileMapQuadrant> &E : layers[layer].quadrant_map) {
						TileMapQuadrant &q = E.value;

						for (RID body : q.bodies) {
							Transform2D xform;
							xform.set_origin(map_to_local(bodies_coords[body]));
							xform = gl_transform * xform;
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
							xform.set_origin(map_to_local(bodies_coords[body]));
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

	Transform2D gl_transform = get_global_transform();
	last_valid_transform = gl_transform;
	new_transform = gl_transform;
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
		for (const Vector2i &E_cell : q.cells) {
			TileMapCell c = get_cell(q.layer, E_cell, true);

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					const TileData *tile_data;
					if (q.runtime_tile_data_cache.has(E_cell)) {
						tile_data = q.runtime_tile_data_cache[E_cell];
					} else {
						tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);
					}
					for (int tile_set_physics_layer = 0; tile_set_physics_layer < tile_set->get_physics_layers_count(); tile_set_physics_layer++) {
						Ref<PhysicsMaterial> physics_material = tile_set->get_physics_layer_physics_material(tile_set_physics_layer);
						uint32_t physics_layer = tile_set->get_physics_layer_collision_layer(tile_set_physics_layer);
						uint32_t physics_mask = tile_set->get_physics_layer_collision_mask(tile_set_physics_layer);

						// Create the body.
						RID body = ps->body_create();
						bodies_coords[body] = E_cell;
						ps->body_set_mode(body, collision_animatable ? PhysicsServer2D::BODY_MODE_KINEMATIC : PhysicsServer2D::BODY_MODE_STATIC);
						ps->body_set_space(body, space);

						Transform2D xform;
						xform.set_origin(map_to_local(E_cell));
						xform = gl_transform * xform;
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

	Vector2 quadrant_pos = map_to_local(p_quadrant->coords * get_effective_quadrant_size(p_quadrant->layer));
	Transform2D quadrant_to_local;
	quadrant_to_local.set_origin(quadrant_pos);
	Transform2D global_to_quadrant = (get_global_transform() * quadrant_to_local).affine_inverse();

	for (RID body : p_quadrant->bodies) {
		Transform2D body_to_quadrant = global_to_quadrant * Transform2D(ps->body_get_state(body, PhysicsServer2D::BODY_STATE_TRANSFORM));
		rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, body_to_quadrant);
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
		case NOTIFICATION_TRANSFORM_CHANGED: {
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
								tile_transform.set_origin(map_to_local(E_region.key));
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
		for (const Vector2i &E_cell : q.cells) {
			TileMapCell c = get_cell(q.layer, E_cell, true);

			TileSetSource *source;
			if (tile_set->has_source(c.source_id)) {
				source = *tile_set->get_source(c.source_id);

				if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
					continue;
				}

				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					const TileData *tile_data;
					if (q.runtime_tile_data_cache.has(E_cell)) {
						tile_data = q.runtime_tile_data_cache[E_cell];
					} else {
						tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);
					}
					q.navigation_regions[E_cell].resize(tile_set->get_navigation_layers_count());

					for (int layer_index = 0; layer_index < tile_set->get_navigation_layers_count(); layer_index++) {
						Ref<NavigationPolygon> navigation_polygon;
						navigation_polygon = tile_data->get_navigation_polygon(layer_index);

						if (navigation_polygon.is_valid()) {
							Transform2D tile_transform;
							tile_transform.set_origin(map_to_local(E_cell));

							RID region = NavigationServer2D::get_singleton()->region_create();
							NavigationServer2D::get_singleton()->region_set_owner_id(region, get_instance_id());
							NavigationServer2D::get_singleton()->region_set_map(region, get_world_2d()->get_navigation_map());
							NavigationServer2D::get_singleton()->region_set_transform(region, tilemap_xform * tile_transform);
							NavigationServer2D::get_singleton()->region_set_navigation_polygon(region, navigation_polygon);
							q.navigation_regions[E_cell].write[layer_index] = region;
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

	Color color = Color(0.5, 1.0, 1.0, 1.0);
#ifdef DEBUG_ENABLED
	color = NavigationServer3D::get_singleton()->get_debug_navigation_geometry_face_color();
#endif // DEBUG_ENABLED
	RandomPCG rand;

	Vector2 quadrant_pos = map_to_local(p_quadrant->coords * get_effective_quadrant_size(p_quadrant->layer));

	for (const Vector2i &E_cell : p_quadrant->cells) {
		TileMapCell c = get_cell(p_quadrant->layer, E_cell, true);

		TileSetSource *source;
		if (tile_set->has_source(c.source_id)) {
			source = *tile_set->get_source(c.source_id);

			if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
				continue;
			}

			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				const TileData *tile_data;
				if (p_quadrant->runtime_tile_data_cache.has(E_cell)) {
					tile_data = p_quadrant->runtime_tile_data_cache[E_cell];
				} else {
					tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);
				}

				Transform2D cell_to_quadrant;
				cell_to_quadrant.set_origin(map_to_local(E_cell) - quadrant_pos);
				rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, cell_to_quadrant);

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

		// Clear the scenes if instance cache was cleared.
		if (instantiated_scenes.is_empty()) {
			for (const KeyValue<Vector2i, String> &E : q.scenes) {
				Node *node = get_node_or_null(E.value);
				if (node) {
					node->queue_free();
				}
			}
		}

		q.scenes.clear();

		// Recreate the scenes.
		for (const Vector2i &E_cell : q.cells) {
			Vector3i cell_coords = Vector3i(q.layer, E_cell.x, E_cell.y);
			if (instantiated_scenes.has(cell_coords)) {
				// Skip scene if the instance was cached (to avoid recreating scenes unnecessarily).
				continue;
			}
			if (!Engine::get_singleton()->is_editor_hint()) {
				instantiated_scenes.insert(cell_coords);
			}

			const TileMapCell &c = get_cell(q.layer, E_cell, true);

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
						Control *scene_as_control = Object::cast_to<Control>(scene);
						Node2D *scene_as_node2d = Object::cast_to<Node2D>(scene);
						if (scene_as_control) {
							scene_as_control->set_position(map_to_local(E_cell) + scene_as_control->get_position());
						} else if (scene_as_node2d) {
							Transform2D xform;
							xform.set_origin(map_to_local(E_cell));
							scene_as_node2d->set_transform(xform * scene_as_node2d->get_transform());
						}
						add_child(scene);
						q.scenes[E_cell] = scene->get_name();
					}
				}
			}
		}

		q_list_element = q_list_element->next();
	}
}

void TileMap::_scenes_cleanup_quadrant(TileMapQuadrant *p_quadrant) {
	// Clear the scenes if instance cache was cleared.
	if (instantiated_scenes.is_empty()) {
		for (const KeyValue<Vector2i, String> &E : p_quadrant->scenes) {
			Node *node = get_node_or_null(E.value);
			if (node) {
				node->queue_free();
			}
		}
		p_quadrant->scenes.clear();
	}
}

void TileMap::_scenes_draw_quadrant_debug(TileMapQuadrant *p_quadrant) {
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	// Draw a placeholder for scenes needing one.
	RenderingServer *rs = RenderingServer::get_singleton();
	Vector2 quadrant_pos = map_to_local(p_quadrant->coords * get_effective_quadrant_size(p_quadrant->layer));
	for (const Vector2i &E_cell : p_quadrant->cells) {
		const TileMapCell &c = get_cell(p_quadrant->layer, E_cell, true);

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
					Transform2D cell_to_quadrant;
					cell_to_quadrant.set_origin(map_to_local(E_cell) - quadrant_pos);
					rs->canvas_item_add_set_transform(p_quadrant->debug_canvas_item, cell_to_quadrant);
					rs->canvas_item_add_circle(p_quadrant->debug_canvas_item, Vector2(), MIN(tile_set->get_tile_size().x, tile_set->get_tile_size().y) / 4.0, color);
				}
			}
		}
	}
}

void TileMap::set_cell(int p_layer, const Vector2i &p_coords, int p_source_id, const Vector2i p_atlas_coords, int p_alternative_tile) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	// Set the current cell tile (using integer position).
	HashMap<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	Vector2i pk(p_coords);
	HashMap<Vector2i, TileMapCell>::Iterator E = tile_map.find(pk);

	int source_id = p_source_id;
	Vector2i atlas_coords = p_atlas_coords;
	int alternative_tile = p_alternative_tile;

	if ((source_id == TileSet::INVALID_SOURCE || atlas_coords == TileSetSource::INVALID_ATLAS_COORDS || alternative_tile == TileSetSource::INVALID_TILE_ALTERNATIVE) &&
			(source_id != TileSet::INVALID_SOURCE || atlas_coords != TileSetSource::INVALID_ATLAS_COORDS || alternative_tile != TileSetSource::INVALID_TILE_ALTERNATIVE)) {
		source_id = TileSet::INVALID_SOURCE;
		atlas_coords = TileSetSource::INVALID_ATLAS_COORDS;
		alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE;
	}

	if (!E && source_id == TileSet::INVALID_SOURCE) {
		return; // Nothing to do, the tile is already empty.
	}

	// Get the quadrant
	Vector2i qk = _coords_to_quadrant_coords(p_layer, pk);

	HashMap<Vector2i, TileMapQuadrant>::Iterator Q = layers[p_layer].quadrant_map.find(qk);

	if (source_id == TileSet::INVALID_SOURCE) {
		// Erase existing cell in the tile map.
		tile_map.erase(pk);

		// Erase existing cell in the quadrant.
		ERR_FAIL_COND(!Q);
		TileMapQuadrant &q = Q->value;

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
			TileMapQuadrant &q = Q->value;
			q.cells.insert(pk);

		} else {
			ERR_FAIL_COND(!Q); // TileMapQuadrant should exist...

			if (E->value.source_id == source_id && E->value.get_atlas_coords() == atlas_coords && E->value.alternative_tile == alternative_tile) {
				return; // Nothing changed.
			}
		}

		TileMapCell &c = E->value;

		c.source_id = source_id;
		c.set_atlas_coords(atlas_coords);
		c.alternative_tile = alternative_tile;

		_make_quadrant_dirty(Q);
		used_rect_cache_dirty = true;
	}
}

void TileMap::erase_cell(int p_layer, const Vector2i &p_coords) {
	set_cell(p_layer, p_coords, TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
}

int TileMap::get_cell_source_id(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), TileSet::INVALID_SOURCE);

	// Get a cell source id from position
	const HashMap<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	HashMap<Vector2i, TileMapCell>::ConstIterator E = tile_map.find(p_coords);

	if (!E) {
		return TileSet::INVALID_SOURCE;
	}

	if (p_use_proxies && tile_set.is_valid()) {
		Array proxyed = tile_set->map_tile_proxy(E->value.source_id, E->value.get_atlas_coords(), E->value.alternative_tile);
		return proxyed[0];
	}

	return E->value.source_id;
}

Vector2i TileMap::get_cell_atlas_coords(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), TileSetSource::INVALID_ATLAS_COORDS);

	// Get a cell source id from position
	const HashMap<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	HashMap<Vector2i, TileMapCell>::ConstIterator E = tile_map.find(p_coords);

	if (!E) {
		return TileSetSource::INVALID_ATLAS_COORDS;
	}

	if (p_use_proxies && tile_set.is_valid()) {
		Array proxyed = tile_set->map_tile_proxy(E->value.source_id, E->value.get_atlas_coords(), E->value.alternative_tile);
		return proxyed[1];
	}

	return E->value.get_atlas_coords();
}

int TileMap::get_cell_alternative_tile(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), TileSetSource::INVALID_TILE_ALTERNATIVE);

	// Get a cell source id from position
	const HashMap<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	HashMap<Vector2i, TileMapCell>::ConstIterator E = tile_map.find(p_coords);

	if (!E) {
		return TileSetSource::INVALID_TILE_ALTERNATIVE;
	}

	if (p_use_proxies && tile_set.is_valid()) {
		Array proxyed = tile_set->map_tile_proxy(E->value.source_id, E->value.get_atlas_coords(), E->value.alternative_tile);
		return proxyed[2];
	}

	return E->value.alternative_tile;
}

TileData *TileMap::get_cell_tile_data(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	int source_id = get_cell_source_id(p_layer, p_coords, p_use_proxies);
	ERR_FAIL_COND_V_MSG(source_id == TileSet::INVALID_SOURCE, nullptr, vformat("Invalid TileSetSource at cell %s. Make sure a tile exists at this cell.", p_coords));

	Ref<TileSetAtlasSource> source = tile_set->get_source(source_id);
	if (source.is_valid()) {
		return source->get_tile_data(get_cell_atlas_coords(p_layer, p_coords, p_use_proxies), get_cell_alternative_tile(p_layer, p_coords, p_use_proxies));
	}

	return nullptr;
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

Vector2i TileMap::map_pattern(const Vector2i &p_position_in_tilemap, const Vector2i &p_coords_in_pattern, Ref<TileMapPattern> p_pattern) {
	ERR_FAIL_COND_V(p_pattern.is_null(), Vector2i());
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

void TileMap::set_pattern(int p_layer, const Vector2i &p_position, const Ref<TileMapPattern> p_pattern) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	ERR_FAIL_COND(!tile_set.is_valid());

	TypedArray<Vector2i> used_cells = p_pattern->get_used_cells();
	for (int i = 0; i < used_cells.size(); i++) {
		Vector2i coords = map_pattern(p_position, used_cells[i], p_pattern);
		set_cell(p_layer, coords, p_pattern->get_cell_source_id(used_cells[i]), p_pattern->get_cell_atlas_coords(used_cells[i]), p_pattern->get_cell_alternative_tile(used_cells[i]));
	}
}

TileSet::TerrainsPattern TileMap::_get_best_terrain_pattern_for_constraints(int p_terrain_set, const Vector2i &p_position, const RBSet<TerrainConstraint> &p_constraints, TileSet::TerrainsPattern p_current_pattern) {
	if (!tile_set.is_valid()) {
		return TileSet::TerrainsPattern();
	}
	// Returns all tiles compatible with the given constraints.
	RBMap<TileSet::TerrainsPattern, int> terrain_pattern_score;
	RBSet<TileSet::TerrainsPattern> pattern_set = tile_set->get_terrains_pattern_set(p_terrain_set);
	ERR_FAIL_COND_V(pattern_set.is_empty(), TileSet::TerrainsPattern());
	for (TileSet::TerrainsPattern &terrain_pattern : pattern_set) {
		int score = 0;

		// Check the center bit constraint
		TerrainConstraint terrain_constraint = TerrainConstraint(this, p_position, terrain_pattern.get_terrain());
		const RBSet<TerrainConstraint>::Element *in_set_constraint_element = p_constraints.find(terrain_constraint);
		if (in_set_constraint_element) {
			if (in_set_constraint_element->get().get_terrain() != terrain_constraint.get_terrain()) {
				score += in_set_constraint_element->get().get_priority();
			}
		} else if (p_current_pattern.get_terrain() != terrain_pattern.get_terrain()) {
			continue; // Ignore a pattern that cannot keep bits without constraints unmodified.
		}

		// Check the surrounding bits
		bool invalid_pattern = false;
		for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(i);
			if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, bit)) {
				// Check if the bit is compatible with the constraints.
				TerrainConstraint terrain_bit_constraint = TerrainConstraint(this, p_position, bit, terrain_pattern.get_terrain_peering_bit(bit));
				in_set_constraint_element = p_constraints.find(terrain_bit_constraint);
				if (in_set_constraint_element) {
					if (in_set_constraint_element->get().get_terrain() != terrain_bit_constraint.get_terrain()) {
						score += in_set_constraint_element->get().get_priority();
					}
				} else if (p_current_pattern.get_terrain_peering_bit(bit) != terrain_pattern.get_terrain_peering_bit(bit)) {
					invalid_pattern = true; // Ignore a pattern that cannot keep bits without constraints unmodified.
					break;
				}
			}
		}
		if (invalid_pattern) {
			continue;
		}

		terrain_pattern_score[terrain_pattern] = score;
	}

	// Compute the minimum score
	TileSet::TerrainsPattern min_score_pattern = p_current_pattern;
	int min_score = INT32_MAX;
	for (KeyValue<TileSet::TerrainsPattern, int> E : terrain_pattern_score) {
		if (E.value < min_score) {
			min_score_pattern = E.key;
			min_score = E.value;
		}
	}

	return min_score_pattern;
}

RBSet<TileMap::TerrainConstraint> TileMap::_get_terrain_constraints_from_added_pattern(const Vector2i &p_position, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern) const {
	if (!tile_set.is_valid()) {
		return RBSet<TerrainConstraint>();
	}

	// Compute the constraints needed from the surrounding tiles.
	RBSet<TerrainConstraint> output;
	output.insert(TerrainConstraint(this, p_position, p_terrains_pattern.get_terrain()));

	for (uint32_t i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
		TileSet::CellNeighbor side = TileSet::CellNeighbor(i);
		if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, side)) {
			TerrainConstraint c = TerrainConstraint(this, p_position, side, p_terrains_pattern.get_terrain_peering_bit(side));
			output.insert(c);
		}
	}

	return output;
}

RBSet<TileMap::TerrainConstraint> TileMap::_get_terrain_constraints_from_painted_cells_list(int p_layer, const RBSet<Vector2i> &p_painted, int p_terrain_set, bool p_ignore_empty_terrains) const {
	if (!tile_set.is_valid()) {
		return RBSet<TerrainConstraint>();
	}

	ERR_FAIL_INDEX_V(p_terrain_set, tile_set->get_terrain_sets_count(), RBSet<TerrainConstraint>());
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), RBSet<TerrainConstraint>());

	// Build a set of dummy constraints to get the constrained points.
	RBSet<TerrainConstraint> dummy_constraints;
	for (const Vector2i &E : p_painted) {
		for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) { // Iterates over neighbor bits.
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(i);
			if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, bit)) {
				dummy_constraints.insert(TerrainConstraint(this, E, bit, -1));
			}
		}
	}

	// For each constrained point, we get all overlapping tiles, and select the most adequate terrain for it.
	RBSet<TerrainConstraint> constraints;
	for (const TerrainConstraint &E_constraint : dummy_constraints) {
		HashMap<int, int> terrain_count;

		// Count the number of occurrences per terrain.
		HashMap<Vector2i, TileSet::CellNeighbor> overlapping_terrain_bits = E_constraint.get_overlapping_coords_and_peering_bits();
		for (const KeyValue<Vector2i, TileSet::CellNeighbor> &E_overlapping : overlapping_terrain_bits) {
			TileData *neighbor_tile_data = nullptr;
			TileMapCell neighbor_cell = get_cell(p_layer, E_overlapping.key);
			if (neighbor_cell.source_id != TileSet::INVALID_SOURCE) {
				Ref<TileSetSource> source = tile_set->get_source(neighbor_cell.source_id);
				Ref<TileSetAtlasSource> atlas_source = source;
				if (atlas_source.is_valid()) {
					TileData *tile_data = atlas_source->get_tile_data(neighbor_cell.get_atlas_coords(), neighbor_cell.alternative_tile);
					if (tile_data && tile_data->get_terrain_set() == p_terrain_set) {
						neighbor_tile_data = tile_data;
					}
				}
			}

			int terrain = neighbor_tile_data ? neighbor_tile_data->get_terrain_peering_bit(TileSet::CellNeighbor(E_overlapping.value)) : -1;
			if (!p_ignore_empty_terrains || terrain >= 0) {
				if (!terrain_count.has(terrain)) {
					terrain_count[terrain] = 0;
				}
				terrain_count[terrain] += 1;
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
			TerrainConstraint c = E_constraint;
			c.set_terrain(max_terrain);
			constraints.insert(c);
		}
	}

	// Add the centers as constraints
	for (Vector2i E_coords : p_painted) {
		TileData *tile_data = nullptr;
		TileMapCell cell = get_cell(p_layer, E_coords);
		if (cell.source_id != TileSet::INVALID_SOURCE) {
			Ref<TileSetSource> source = tile_set->get_source(cell.source_id);
			Ref<TileSetAtlasSource> atlas_source = source;
			if (atlas_source.is_valid()) {
				tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
			}
		}

		int terrain = (tile_data && tile_data->get_terrain_set() == p_terrain_set) ? tile_data->get_terrain() : -1;
		if (!p_ignore_empty_terrains || terrain >= 0) {
			constraints.insert(TerrainConstraint(this, E_coords, terrain));
		}
	}

	return constraints;
}

HashMap<Vector2i, TileSet::TerrainsPattern> TileMap::terrain_fill_constraints(int p_layer, const Vector<Vector2i> &p_to_replace, int p_terrain_set, const RBSet<TerrainConstraint> &p_constraints) {
	if (!tile_set.is_valid()) {
		return HashMap<Vector2i, TileSet::TerrainsPattern>();
	}

	// Copy the constraints set.
	RBSet<TerrainConstraint> constraints = p_constraints;

	// Output map.
	HashMap<Vector2i, TileSet::TerrainsPattern> output;

	// Add all positions to a set.
	for (int i = 0; i < p_to_replace.size(); i++) {
		const Vector2i &coords = p_to_replace[i];

		// Select the best pattern for the given constraints
		TileSet::TerrainsPattern current_pattern = TileSet::TerrainsPattern(*tile_set, p_terrain_set);
		TileMapCell cell = get_cell(p_layer, coords);
		if (cell.source_id != TileSet::INVALID_SOURCE) {
			TileSetSource *source = *tile_set->get_source(cell.source_id);
			TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
			if (atlas_source) {
				// Get tile data.
				TileData *tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
				if (tile_data && tile_data->get_terrain_set() == p_terrain_set) {
					current_pattern = tile_data->get_terrains_pattern();
				}
			}
		}
		TileSet::TerrainsPattern pattern = _get_best_terrain_pattern_for_constraints(p_terrain_set, coords, constraints, current_pattern);

		// Update the constraint set with the new ones
		RBSet<TerrainConstraint> new_constraints = _get_terrain_constraints_from_added_pattern(coords, p_terrain_set, pattern);
		for (const TerrainConstraint &E_constraint : new_constraints) {
			if (constraints.has(E_constraint)) {
				constraints.erase(E_constraint);
			}
			TerrainConstraint c = E_constraint;
			c.set_priority(5);
			constraints.insert(c);
		}

		output[coords] = pattern;
	}
	return output;
}

HashMap<Vector2i, TileSet::TerrainsPattern> TileMap::terrain_fill_connect(int p_layer, const Vector<Vector2i> &p_coords_array, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	HashMap<Vector2i, TileSet::TerrainsPattern> output;
	ERR_FAIL_COND_V(!tile_set.is_valid(), output);
	ERR_FAIL_INDEX_V(p_terrain_set, tile_set->get_terrain_sets_count(), output);

	// Build list and set of tiles that can be modified (painted and their surroundings)
	Vector<Vector2i> can_modify_list;
	RBSet<Vector2i> can_modify_set;
	RBSet<Vector2i> painted_set;
	for (int i = p_coords_array.size() - 1; i >= 0; i--) {
		const Vector2i &coords = p_coords_array[i];
		can_modify_list.push_back(coords);
		can_modify_set.insert(coords);
		painted_set.insert(coords);
	}
	for (Vector2i coords : p_coords_array) {
		// Find the adequate neighbor
		for (int j = 0; j < TileSet::CELL_NEIGHBOR_MAX; j++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(j);
			if (is_existing_neighbor(bit)) {
				Vector2i neighbor = get_neighbor_cell(coords, bit);
				if (!can_modify_set.has(neighbor)) {
					can_modify_list.push_back(neighbor);
					can_modify_set.insert(neighbor);
				}
			}
		}
	}

	// Build a set, out of the possibly modified tiles, of the one with a center bit that is set (or will be) to the painted terrain
	RBSet<Vector2i> cells_with_terrain_center_bit;
	for (Vector2i coords : can_modify_set) {
		bool connect = false;
		if (painted_set.has(coords)) {
			connect = true;
		} else {
			// Get the center bit of the cell
			TileData *tile_data = nullptr;
			TileMapCell cell = get_cell(p_layer, coords);
			if (cell.source_id != TileSet::INVALID_SOURCE) {
				Ref<TileSetSource> source = tile_set->get_source(cell.source_id);
				Ref<TileSetAtlasSource> atlas_source = source;
				if (atlas_source.is_valid()) {
					tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
				}
			}

			if (tile_data && tile_data->get_terrain_set() == p_terrain_set && tile_data->get_terrain() == p_terrain) {
				connect = true;
			}
		}
		if (connect) {
			cells_with_terrain_center_bit.insert(coords);
		}
	}

	RBSet<TerrainConstraint> constraints;

	// Add new constraints from the path drawn.
	for (Vector2i coords : p_coords_array) {
		// Constraints on the center bit.
		TerrainConstraint c = TerrainConstraint(this, coords, p_terrain);
		c.set_priority(10);
		constraints.insert(c);

		// Constraints on the connecting bits.
		for (int j = 0; j < TileSet::CELL_NEIGHBOR_MAX; j++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(j);
			if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, bit)) {
				c = TerrainConstraint(this, coords, bit, p_terrain);
				c.set_priority(10);
				if ((int(bit) % 2) == 0) {
					// Side peering bits: add the constraint if the center is of the same terrain
					Vector2i neighbor = get_neighbor_cell(coords, bit);
					if (cells_with_terrain_center_bit.has(neighbor)) {
						constraints.insert(c);
					}
				} else {
					// Corner peering bits: add the constraint if all tiles on the constraint has the same center bit
					HashMap<Vector2i, TileSet::CellNeighbor> overlapping_terrain_bits = c.get_overlapping_coords_and_peering_bits();
					bool valid = true;
					for (KeyValue<Vector2i, TileSet::CellNeighbor> kv : overlapping_terrain_bits) {
						if (!cells_with_terrain_center_bit.has(kv.key)) {
							valid = false;
							break;
						}
					}
					if (valid) {
						constraints.insert(c);
					}
				}
			}
		}
	}

	// Fills in the constraint list from existing tiles.
	for (TerrainConstraint c : _get_terrain_constraints_from_painted_cells_list(p_layer, painted_set, p_terrain_set, p_ignore_empty_terrains)) {
		constraints.insert(c);
	}

	// Fill the terrains.
	output = terrain_fill_constraints(p_layer, can_modify_list, p_terrain_set, constraints);
	return output;
}

HashMap<Vector2i, TileSet::TerrainsPattern> TileMap::terrain_fill_path(int p_layer, const Vector<Vector2i> &p_path, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	HashMap<Vector2i, TileSet::TerrainsPattern> output;
	ERR_FAIL_COND_V(!tile_set.is_valid(), output);
	ERR_FAIL_INDEX_V(p_terrain_set, tile_set->get_terrain_sets_count(), output);

	// Make sure the path is correct and build the peering bit list while doing it.
	Vector<TileSet::CellNeighbor> neighbor_list;
	for (int i = 0; i < p_path.size() - 1; i++) {
		// Find the adequate neighbor
		TileSet::CellNeighbor found_bit = TileSet::CELL_NEIGHBOR_MAX;
		for (int j = 0; j < TileSet::CELL_NEIGHBOR_MAX; j++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(j);
			if (is_existing_neighbor(bit)) {
				if (get_neighbor_cell(p_path[i], bit) == p_path[i + 1]) {
					found_bit = bit;
					break;
				}
			}
		}
		ERR_FAIL_COND_V_MSG(found_bit == TileSet::CELL_NEIGHBOR_MAX, output, vformat("Invalid terrain path, %s is not a neighbouring tile of %s", p_path[i + 1], p_path[i]));
		neighbor_list.push_back(found_bit);
	}

	// Build list and set of tiles that can be modified (painted and their surroundings)
	Vector<Vector2i> can_modify_list;
	RBSet<Vector2i> can_modify_set;
	RBSet<Vector2i> painted_set;
	for (int i = p_path.size() - 1; i >= 0; i--) {
		const Vector2i &coords = p_path[i];
		can_modify_list.push_back(coords);
		can_modify_set.insert(coords);
		painted_set.insert(coords);
	}
	for (Vector2i coords : p_path) {
		// Find the adequate neighbor
		for (int j = 0; j < TileSet::CELL_NEIGHBOR_MAX; j++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(j);
			if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, bit)) {
				Vector2i neighbor = get_neighbor_cell(coords, bit);
				if (!can_modify_set.has(neighbor)) {
					can_modify_list.push_back(neighbor);
					can_modify_set.insert(neighbor);
				}
			}
		}
	}

	RBSet<TerrainConstraint> constraints;

	// Add new constraints from the path drawn.
	for (Vector2i coords : p_path) {
		// Constraints on the center bit
		TerrainConstraint c = TerrainConstraint(this, coords, p_terrain);
		c.set_priority(10);
		constraints.insert(c);
	}
	for (int i = 0; i < p_path.size() - 1; i++) {
		// Constraints on the peering bits.
		TerrainConstraint c = TerrainConstraint(this, p_path[i], neighbor_list[i], p_terrain);
		c.set_priority(10);
		constraints.insert(c);
	}

	// Fills in the constraint list from existing tiles.
	for (TerrainConstraint c : _get_terrain_constraints_from_painted_cells_list(p_layer, painted_set, p_terrain_set, p_ignore_empty_terrains)) {
		constraints.insert(c);
	}

	// Fill the terrains.
	output = terrain_fill_constraints(p_layer, can_modify_list, p_terrain_set, constraints);
	return output;
}

HashMap<Vector2i, TileSet::TerrainsPattern> TileMap::terrain_fill_pattern(int p_layer, const Vector<Vector2i> &p_coords_array, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern, bool p_ignore_empty_terrains) {
	HashMap<Vector2i, TileSet::TerrainsPattern> output;
	ERR_FAIL_COND_V(!tile_set.is_valid(), output);
	ERR_FAIL_INDEX_V(p_terrain_set, tile_set->get_terrain_sets_count(), output);

	// Build list and set of tiles that can be modified (painted and their surroundings).
	Vector<Vector2i> can_modify_list;
	RBSet<Vector2i> can_modify_set;
	RBSet<Vector2i> painted_set;
	for (int i = p_coords_array.size() - 1; i >= 0; i--) {
		const Vector2i &coords = p_coords_array[i];
		can_modify_list.push_back(coords);
		can_modify_set.insert(coords);
		painted_set.insert(coords);
	}
	for (Vector2i coords : p_coords_array) {
		// Find the adequate neighbor
		for (int j = 0; j < TileSet::CELL_NEIGHBOR_MAX; j++) {
			TileSet::CellNeighbor bit = TileSet::CellNeighbor(j);
			if (tile_set->is_valid_terrain_peering_bit(p_terrain_set, bit)) {
				Vector2i neighbor = get_neighbor_cell(coords, bit);
				if (!can_modify_set.has(neighbor)) {
					can_modify_list.push_back(neighbor);
					can_modify_set.insert(neighbor);
				}
			}
		}
	}

	// Add constraint by the new ones.
	RBSet<TerrainConstraint> constraints;

	// Add new constraints from the path drawn.
	for (Vector2i coords : p_coords_array) {
		// Constraints on the center bit
		RBSet<TerrainConstraint> added_constraints = _get_terrain_constraints_from_added_pattern(coords, p_terrain_set, p_terrains_pattern);
		for (TerrainConstraint c : added_constraints) {
			c.set_priority(10);
			constraints.insert(c);
		}
	}

	// Fills in the constraint list from modified tiles border.
	for (TerrainConstraint c : _get_terrain_constraints_from_painted_cells_list(p_layer, painted_set, p_terrain_set, p_ignore_empty_terrains)) {
		constraints.insert(c);
	}

	// Fill the terrains.
	output = terrain_fill_constraints(p_layer, can_modify_list, p_terrain_set, constraints);
	return output;
}

void TileMap::set_cells_terrain_connect(int p_layer, TypedArray<Vector2i> p_cells, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	ERR_FAIL_COND(!tile_set.is_valid());
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	ERR_FAIL_INDEX(p_terrain_set, tile_set->get_terrain_sets_count());

	Vector<Vector2i> cells_vector;
	HashSet<Vector2i> painted_set;
	for (int i = 0; i < p_cells.size(); i++) {
		cells_vector.push_back(p_cells[i]);
		painted_set.insert(p_cells[i]);
	}
	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_output = terrain_fill_connect(p_layer, cells_vector, p_terrain_set, p_terrain, p_ignore_empty_terrains);
	for (const KeyValue<Vector2i, TileSet::TerrainsPattern> &kv : terrain_fill_output) {
		if (painted_set.has(kv.key)) {
			// Paint a random tile with the correct terrain for the painted path.
			TileMapCell c = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
			set_cell(p_layer, kv.key, c.source_id, c.get_atlas_coords(), c.alternative_tile);
		} else {
			// Avoids updating the painted path from the output if the new pattern is the same as before.
			TileSet::TerrainsPattern in_map_terrain_pattern = TileSet::TerrainsPattern(*tile_set, p_terrain_set);
			TileMapCell cell = get_cell(p_layer, kv.key);
			if (cell.source_id != TileSet::INVALID_SOURCE) {
				TileSetSource *source = *tile_set->get_source(cell.source_id);
				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					// Get tile data.
					TileData *tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
					if (tile_data && tile_data->get_terrain_set() == p_terrain_set) {
						in_map_terrain_pattern = tile_data->get_terrains_pattern();
					}
				}
			}
			if (in_map_terrain_pattern != kv.value) {
				TileMapCell c = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
				set_cell(p_layer, kv.key, c.source_id, c.get_atlas_coords(), c.alternative_tile);
			}
		}
	}
}

void TileMap::set_cells_terrain_path(int p_layer, TypedArray<Vector2i> p_path, int p_terrain_set, int p_terrain, bool p_ignore_empty_terrains) {
	ERR_FAIL_COND(!tile_set.is_valid());
	ERR_FAIL_INDEX(p_layer, (int)layers.size());
	ERR_FAIL_INDEX(p_terrain_set, tile_set->get_terrain_sets_count());

	Vector<Vector2i> vector_path;
	HashSet<Vector2i> painted_set;
	for (int i = 0; i < p_path.size(); i++) {
		vector_path.push_back(p_path[i]);
		painted_set.insert(p_path[i]);
	}

	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_output = terrain_fill_path(p_layer, vector_path, p_terrain_set, p_terrain, p_ignore_empty_terrains);
	for (const KeyValue<Vector2i, TileSet::TerrainsPattern> &kv : terrain_fill_output) {
		if (painted_set.has(kv.key)) {
			// Paint a random tile with the correct terrain for the painted path.
			TileMapCell c = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
			set_cell(p_layer, kv.key, c.source_id, c.get_atlas_coords(), c.alternative_tile);
		} else {
			// Avoids updating the painted path from the output if the new pattern is the same as before.
			TileSet::TerrainsPattern in_map_terrain_pattern = TileSet::TerrainsPattern(*tile_set, p_terrain_set);
			TileMapCell cell = get_cell(p_layer, kv.key);
			if (cell.source_id != TileSet::INVALID_SOURCE) {
				TileSetSource *source = *tile_set->get_source(cell.source_id);
				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					// Get tile data.
					TileData *tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
					if (tile_data && tile_data->get_terrain_set() == p_terrain_set) {
						in_map_terrain_pattern = tile_data->get_terrains_pattern();
					}
				}
			}
			if (in_map_terrain_pattern != kv.value) {
				TileMapCell c = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
				set_cell(p_layer, kv.key, c.source_id, c.get_atlas_coords(), c.alternative_tile);
			}
		}
	}
}

TileMapCell TileMap::get_cell(int p_layer, const Vector2i &p_coords, bool p_use_proxies) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers.size(), TileMapCell());
	const HashMap<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	if (!tile_map.has(p_coords)) {
		return TileMapCell();
	} else {
		TileMapCell c = tile_map.find(p_coords)->value;
		if (p_use_proxies && tile_set.is_valid()) {
			Array proxyed = tile_set->map_tile_proxy(c.source_id, c.get_atlas_coords(), c.alternative_tile);
			c.source_id = proxyed[0];
			c.set_atlas_coords(proxyed[1]);
			c.alternative_tile = proxyed[2];
		}
		return c;
	}
}

HashMap<Vector2i, TileMapQuadrant> *TileMap::get_quadrant_map(int p_layer) {
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
		const HashMap<Vector2i, TileMapCell> &tile_map = layers[i].tile_map;
		RBSet<Vector2i> coords;
		for (const KeyValue<Vector2i, TileMapCell> &E : tile_map) {
			TileSetSource *source = *tile_set->get_source(E.value.source_id);
			if (!source || !source->has_tile(E.value.get_atlas_coords()) || !source->has_alternative_tile(E.value.get_atlas_coords(), E.value.alternative_tile)) {
				coords.insert(E.key);
			}
		}
		for (const Vector2i &E : coords) {
			set_cell(i, E, TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
		}
	}
}

void TileMap::clear_layer(int p_layer) {
	ERR_FAIL_INDEX(p_layer, (int)layers.size());

	// Remove all tiles.
	_clear_layer_internals(p_layer);
	layers[p_layer].tile_map.clear();
	_recreate_layer_internals(p_layer);
	used_rect_cache_dirty = true;
}

void TileMap::clear() {
	// Remove all tiles.
	_clear_internals();
	for (unsigned int i = 0; i < layers.size(); i++) {
		layers[i].tile_map.clear();
	}
	_recreate_internals();
	used_rect_cache_dirty = true;
}

void TileMap::force_update(int p_layer) {
	if (p_layer >= 0) {
		ERR_FAIL_INDEX(p_layer, (int)layers.size());
		_clear_layer_internals(p_layer);
		_recreate_layer_internals(p_layer);
	} else {
		_clear_internals();
		_recreate_internals();
	}
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
			bool flip_h = v & (1UL << 29);
			bool flip_v = v & (1UL << 30);
			bool transpose = v & (1UL << 31);
			v &= (1UL << 29) - 1;

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
	const HashMap<Vector2i, TileMapCell> &tile_map = layers[p_layer].tile_map;
	Vector<int> tile_data;
	tile_data.resize(tile_map.size() * 3);
	int *w = tile_data.ptrw();

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

	return tile_data;
}

void TileMap::_build_runtime_update_tile_data(SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) {
	if (GDVIRTUAL_IS_OVERRIDDEN(_use_tile_data_runtime_update) && GDVIRTUAL_IS_OVERRIDDEN(_tile_data_runtime_update)) {
		SelfList<TileMapQuadrant> *q_list_element = r_dirty_quadrant_list.first();
		while (q_list_element) {
			TileMapQuadrant &q = *q_list_element->self();
			// Iterate over the cells of the quadrant.
			for (const KeyValue<Vector2i, Vector2i> &E_cell : q.local_to_map) {
				TileMapCell c = get_cell(q.layer, E_cell.value, true);

				TileSetSource *source;
				if (tile_set->has_source(c.source_id)) {
					source = *tile_set->get_source(c.source_id);

					if (!source->has_tile(c.get_atlas_coords()) || !source->has_alternative_tile(c.get_atlas_coords(), c.alternative_tile)) {
						continue;
					}

					TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
					if (atlas_source) {
						bool ret = false;
						if (GDVIRTUAL_CALL(_use_tile_data_runtime_update, q.layer, E_cell.value, ret) && ret) {
							TileData *tile_data = atlas_source->get_tile_data(c.get_atlas_coords(), c.alternative_tile);

							// Create the runtime TileData.
							TileData *tile_data_runtime_use = tile_data->duplicate();
							tile_data->set_allow_transform(true);
							q.runtime_tile_data_cache[E_cell.value] = tile_data_runtime_use;

							GDVIRTUAL_CALL(_tile_data_runtime_update, q.layer, E_cell.value, tile_data_runtime_use);
						}
					}
				}
			}
			q_list_element = q_list_element->next();
		}
	}
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
	p_list->push_back(PropertyInfo(Variant::INT, "format", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
	p_list->push_back(PropertyInfo(Variant::NIL, "Layers", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
	for (unsigned int i = 0; i < layers.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::STRING, vformat("layer_%d/name", i), PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::BOOL, vformat("layer_%d/enabled", i), PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::COLOR, vformat("layer_%d/modulate", i), PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::BOOL, vformat("layer_%d/y_sort_enabled", i), PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::INT, vformat("layer_%d/y_sort_origin", i), PROPERTY_HINT_NONE, "suffix:px"));
		p_list->push_back(PropertyInfo(Variant::INT, vformat("layer_%d/z_index", i), PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::OBJECT, vformat("layer_%d/tile_data", i), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	}
}

Vector2 TileMap::map_to_local(const Vector2i &p_pos) const {
	// SHOULD RETURN THE CENTER OF THE CELL
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

Vector2i TileMap::local_to_map(const Vector2 &p_local_position) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), Vector2i());

	Vector2 ret = p_local_position;
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

	// For each half-offset shape, we check if we are in the corner of the tile, and thus should correct the local position accordingly.
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

Rect2i TileMap::get_used_rect() { // Not const because of cache
	// Return the rect of the currently used area
	if (used_rect_cache_dirty) {
		bool first = true;
		used_rect_cache = Rect2i();

		for (unsigned int i = 0; i < layers.size(); i++) {
			const HashMap<Vector2i, TileMapCell> &tile_map = layers[i].tile_map;
			if (tile_map.size() > 0) {
				if (first) {
					used_rect_cache = Rect2i(tile_map.begin()->key.x, tile_map.begin()->key.y, 0, 0);
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
	// Set a default texture filter for the whole tilemap.
	CanvasItem::set_texture_filter(p_texture_filter);
	TextureFilter target_filter = get_texture_filter_in_tree();
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		for (HashMap<Vector2i, TileMapQuadrant>::Iterator F = layers[layer].quadrant_map.begin(); F; ++F) {
			TileMapQuadrant &q = F->value;
			for (const RID &ci : q.canvas_items) {
				RenderingServer::get_singleton()->canvas_item_set_default_texture_filter(ci, RS::CanvasItemTextureFilter(target_filter));
				_make_quadrant_dirty(F);
			}
		}
		_rendering_update_layer(layer);
	}
}

void TileMap::set_texture_repeat(CanvasItem::TextureRepeat p_texture_repeat) {
	// Set a default texture repeat for the whole tilemap.
	CanvasItem::set_texture_repeat(p_texture_repeat);
	TextureRepeat target_repeat = get_texture_repeat_in_tree();
	for (unsigned int layer = 0; layer < layers.size(); layer++) {
		for (HashMap<Vector2i, TileMapQuadrant>::Iterator F = layers[layer].quadrant_map.begin(); F; ++F) {
			TileMapQuadrant &q = F->value;
			for (const RID &ci : q.canvas_items) {
				RenderingServer::get_singleton()->canvas_item_set_default_texture_repeat(ci, RS::CanvasItemTextureRepeat(target_repeat));
				_make_quadrant_dirty(F);
			}
		}
		_rendering_update_layer(layer);
	}
}

TypedArray<Vector2i> TileMap::get_surrounding_cells(const Vector2i &coords) {
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

void TileMap::draw_cells_outline(Control *p_control, const RBSet<Vector2i> &p_cells, Color p_color, Transform2D p_transform) {
	if (!tile_set.is_valid()) {
		return;
	}

	// Create a set.
	Vector2i tile_size = tile_set->get_tile_size();
	Vector<Vector2> polygon = tile_set->get_tile_shape_polygon();
	TileSet::TileShape shape = tile_set->get_tile_shape();

	for (const Vector2i &E : p_cells) {
		Vector2 center = map_to_local(E);

#define DRAW_SIDE_IF_NEEDED(side, polygon_index_from, polygon_index_to)                     \
	if (!p_cells.has(get_neighbor_cell(E, side))) {                                         \
		Vector2 from = p_transform.xform(center + polygon[polygon_index_from] * tile_size); \
		Vector2 to = p_transform.xform(center + polygon[polygon_index_to] * tile_size);     \
		p_control->draw_line(from, to, p_color);                                            \
	}

		if (shape == TileSet::TILE_SHAPE_SQUARE) {
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_RIGHT_SIDE, 1, 2);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, 2, 3);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_LEFT_SIDE, 3, 0);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_SIDE, 0, 1);
		} else if (shape == TileSet::TILE_SHAPE_ISOMETRIC) {
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, 2, 3);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, 1, 2);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, 0, 1);
			DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, 3, 0);
		} else {
			if (tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, 3, 4);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, 2, 3);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_LEFT_SIDE, 1, 2);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, 0, 1);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, 5, 0);
				DRAW_SIDE_IF_NEEDED(TileSet::CELL_NEIGHBOR_RIGHT_SIDE, 4, 5);
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
#undef DRAW_SIDE_IF_NEEDED
}

PackedStringArray TileMap::get_configuration_warnings() const {
	PackedStringArray warnings = Node::get_configuration_warnings();

	// Retrieve the set of Z index values with a Y-sorted layer.
	RBSet<int> y_sorted_z_index;
	for (int layer = 0; layer < (int)layers.size(); layer++) {
		if (layers[layer].y_sort_enabled) {
			y_sorted_z_index.insert(layers[layer].z_index);
		}
	}

	// Check if we have a non-sorted layer in a Z-index with a Y-sorted layer.
	for (int layer = 0; layer < (int)layers.size(); layer++) {
		if (!layers[layer].y_sort_enabled && y_sorted_z_index.has(layers[layer].z_index)) {
			warnings.push_back(RTR("A Y-sorted layer has the same Z-index value as a not Y-sorted layer.\nThis may lead to unwanted behaviors, as a layer that is not Y-sorted will be Y-sorted as a whole with tiles from Y-sorted layers."));
			break;
		}
	}

	if (tile_set.is_valid() && tile_set->get_tile_shape() == TileSet::TILE_SHAPE_ISOMETRIC) {
		bool warn = !is_y_sort_enabled();
		if (!warn) {
			for (int layer = 0; layer < (int)layers.size(); layer++) {
				if (!layers[layer].y_sort_enabled) {
					warn = true;
					break;
				}
			}
		}

		if (warn) {
			warnings.push_back(RTR("Isometric TileSet will likely not look as intended without Y-sort enabled for the TileMap and all of its layers."));
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
	ClassDB::bind_method(D_METHOD("set_layer_modulate", "layer", "modulate"), &TileMap::set_layer_modulate);
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

	ClassDB::bind_method(D_METHOD("set_cell", "layer", "coords", "source_id", "atlas_coords", "alternative_tile"), &TileMap::set_cell, DEFVAL(TileSet::INVALID_SOURCE), DEFVAL(TileSetSource::INVALID_ATLAS_COORDS), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("erase_cell", "layer", "coords"), &TileMap::erase_cell);
	ClassDB::bind_method(D_METHOD("get_cell_source_id", "layer", "coords", "use_proxies"), &TileMap::get_cell_source_id, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_cell_atlas_coords", "layer", "coords", "use_proxies"), &TileMap::get_cell_atlas_coords, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_cell_alternative_tile", "layer", "coords", "use_proxies"), &TileMap::get_cell_alternative_tile, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_cell_tile_data", "layer", "coords", "use_proxies"), &TileMap::get_cell_tile_data, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("get_coords_for_body_rid", "body"), &TileMap::get_coords_for_body_rid);

	ClassDB::bind_method(D_METHOD("get_pattern", "layer", "coords_array"), &TileMap::get_pattern);
	ClassDB::bind_method(D_METHOD("map_pattern", "position_in_tilemap", "coords_in_pattern", "pattern"), &TileMap::map_pattern);
	ClassDB::bind_method(D_METHOD("set_pattern", "layer", "position", "pattern"), &TileMap::set_pattern);

	ClassDB::bind_method(D_METHOD("set_cells_terrain_connect", "layer", "cells", "terrain_set", "terrain", "ignore_empty_terrains"), &TileMap::set_cells_terrain_connect, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("set_cells_terrain_path", "layer", "path", "terrain_set", "terrain", "ignore_empty_terrains"), &TileMap::set_cells_terrain_path, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("fix_invalid_tiles"), &TileMap::fix_invalid_tiles);
	ClassDB::bind_method(D_METHOD("clear_layer", "layer"), &TileMap::clear_layer);
	ClassDB::bind_method(D_METHOD("clear"), &TileMap::clear);

	ClassDB::bind_method(D_METHOD("force_update", "layer"), &TileMap::force_update, DEFVAL(-1));

	ClassDB::bind_method(D_METHOD("get_surrounding_cells", "coords"), &TileMap::get_surrounding_cells);

	ClassDB::bind_method(D_METHOD("get_used_cells", "layer"), &TileMap::get_used_cells);
	ClassDB::bind_method(D_METHOD("get_used_rect"), &TileMap::get_used_rect);

	ClassDB::bind_method(D_METHOD("map_to_local", "map_position"), &TileMap::map_to_local);
	ClassDB::bind_method(D_METHOD("local_to_map", "local_position"), &TileMap::local_to_map);

	ClassDB::bind_method(D_METHOD("get_neighbor_cell", "coords", "neighbor"), &TileMap::get_neighbor_cell);

	ClassDB::bind_method(D_METHOD("_update_dirty_quadrants"), &TileMap::_update_dirty_quadrants);

	ClassDB::bind_method(D_METHOD("_tile_set_changed_deferred_update"), &TileMap::_tile_set_changed_deferred_update);

	GDVIRTUAL_BIND(_use_tile_data_runtime_update, "layer", "coords");
	GDVIRTUAL_BIND(_tile_data_runtime_update, "layer", "coords", "tile_data");

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
	instantiated_scenes.clear();
	call_deferred(SNAME("_tile_set_changed_deferred_update"));
	update_configuration_warnings();
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
