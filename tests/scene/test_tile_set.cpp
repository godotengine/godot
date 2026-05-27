/**************************************************************************/
/*  test_tile_set.cpp                                                     */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_tile_set)

#include "scene/resources/2d/tile_set.h"

namespace TestTileSet {

TEST_CASE("[TileSet] get_neighbor_cell: square shape") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_SQUARE);

	Vector2i center_cell(0, 0);

	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(0, -1));
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_CORNER) == Vector2i(1, -1));
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 0));
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER) == Vector2i(1, 1));
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(0, 1));
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER) == Vector2i(-1, 1));
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 0));
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_CORNER) == Vector2i(-1, -1));
}

TEST_CASE("[TileSet] get_neighbor_cell: stacked layout, horizontal offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STACKED);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	// CELL_NEIGHBOR_RIGHT_SIDE (hex) / CELL_NEIGHBOR_RIGHT_CORNER (isometric) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 0));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));

	// CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE — shifts x+1 on odd rows
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1)); // even row
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 1), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 2)); // odd row

	// CELL_NEIGHBOR_BOTTOM_CORNER (isometric only) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 2));

	// CELL_NEIGHBOR_BOTTOM_LEFT_SIDE — shifts x-1 on even rows
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1)); // even row
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 1), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 2));  // odd row

	// CELL_NEIGHBOR_LEFT_SIDE (hex) / CELL_NEIGHBOR_LEFT_CORNER (isometric) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 0));

	// CELL_NEIGHBOR_TOP_LEFT_SIDE — shifts x-1 on even rows
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, -1)); // even row
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 1), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, 0));   // odd row

	// CELL_NEIGHBOR_TOP_CORNER (isometric only) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -2));

	// CELL_NEIGHBOR_TOP_RIGHT_SIDE — shifts x+1 on odd rows
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1)); // even row
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 1), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));  // odd row

	// Invalid neighbor for this shape
	ERR_PRINT_OFF;
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;
}

TEST_CASE("[TileSet] get_neighbor_cell: stacked layout, vertical offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STACKED);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	// CELL_NEIGHBOR_BOTTOM_SIDE (hex) / CELL_NEIGHBOR_BOTTOM_CORNER (isometric) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(0, 1));

	// CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE — shifts y+1 on odd columns
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0)); // even column
	CHECK(tile_set->get_neighbor_cell(Vector2i(1, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(2, 1)); // odd column

	// CELL_NEIGHBOR_RIGHT_CORNER (isometric only) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, 0));

	// CELL_NEIGHBOR_TOP_RIGHT_SIDE — shifts y-1 on even columns; note: shares branch with RIGHT_CORNER for isometric
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));  // even column
	CHECK(tile_set->get_neighbor_cell(Vector2i(1, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(2, -1)); // odd column

	// CELL_NEIGHBOR_TOP_SIDE (hex) / CELL_NEIGHBOR_TOP_CORNER (isometric) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(0, -1));

	// CELL_NEIGHBOR_TOP_LEFT_SIDE — shifts y-1 on even columns
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, -1)); // even column
	CHECK(tile_set->get_neighbor_cell(Vector2i(1, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, 0));   // odd column

	// CELL_NEIGHBOR_LEFT_CORNER (isometric only) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 0));

	// CELL_NEIGHBOR_BOTTOM_LEFT_SIDE — shifts y+1 on odd columns
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0)); // even column
	CHECK(tile_set->get_neighbor_cell(Vector2i(1, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));  // odd column

	// Invalid neighbor for this shape
	ERR_PRINT_OFF;
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;
}

TEST_CASE("[TileSet] get_neighbor_cell: stacked offset layout, horizontal offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STACKED_OFFSET);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	// CELL_NEIGHBOR_RIGHT_SIDE (hex) / CELL_NEIGHBOR_RIGHT_CORNER (isometric) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 0));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));

	// CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE — shifts x+1 on even rows (inverted vs stacked)
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 1)); // even row
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 1), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 2)); // odd row

	// CELL_NEIGHBOR_BOTTOM_CORNER (isometric only) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 2));

	// CELL_NEIGHBOR_BOTTOM_LEFT_SIDE — shifts x-1 on odd rows (inverted vs stacked)
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));  // even row
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 1), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 2)); // odd row

	// CELL_NEIGHBOR_LEFT_SIDE (hex) / CELL_NEIGHBOR_LEFT_CORNER (isometric) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 0));

	// CELL_NEIGHBOR_TOP_LEFT_SIDE — shifts x-1 on odd rows (inverted vs stacked)
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));  // even row
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 1), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0)); // odd row

	// CELL_NEIGHBOR_TOP_CORNER (isometric only) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -2));

	// CELL_NEIGHBOR_TOP_RIGHT_SIDE — shifts x+1 on even rows (inverted vs stacked)
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1)); // even row
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 1), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, 0));  // odd row

	// Invalid neighbor for this shape
	ERR_PRINT_OFF;
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;
}

TEST_CASE("[TileSet] get_neighbor_cell: stacked offset layout, vertical offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STACKED_OFFSET);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	// CELL_NEIGHBOR_BOTTOM_SIDE (hex) / CELL_NEIGHBOR_BOTTOM_CORNER (isometric) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(0, 1));

	// CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE — shifts y+1 on even columns (inverted vs stacked)
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 1)); // even column
	CHECK(tile_set->get_neighbor_cell(Vector2i(1, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(2, 0)); // odd column

	// CELL_NEIGHBOR_RIGHT_CORNER (isometric only) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, 0));

	// CELL_NEIGHBOR_TOP_RIGHT_SIDE — shifts y-1 on odd columns (inverted vs stacked)
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));  // even column
	CHECK(tile_set->get_neighbor_cell(Vector2i(1, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(2, -1)); // odd column

	// CELL_NEIGHBOR_TOP_SIDE (hex) / CELL_NEIGHBOR_TOP_CORNER (isometric) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(0, -1));

	// CELL_NEIGHBOR_TOP_LEFT_SIDE — shifts y-1 on odd columns (inverted vs stacked)
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));  // even column
	CHECK(tile_set->get_neighbor_cell(Vector2i(1, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1)); // odd column

	// CELL_NEIGHBOR_LEFT_CORNER (isometric only) — no offset effect
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 0));

	// CELL_NEIGHBOR_BOTTOM_LEFT_SIDE — shifts y+1 on even columns (inverted vs stacked)
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1)); // even column
	CHECK(tile_set->get_neighbor_cell(Vector2i(1, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 0));  // odd column

	// Invalid neighbor for this shape
	ERR_PRINT_OFF;
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;
}

TEST_CASE("[TileSet] get_neighbor_cell: stairs right layout, horizontal offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_RIGHT);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	// No offset effect in stairs layouts — all neighbors are fixed offsets from center
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 0));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 2));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 0));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -2));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
}

TEST_CASE("[TileSet] get_neighbor_cell: stairs down layout, vertical offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_DOWN);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(0, 1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(0, -1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
}

TEST_CASE("[TileSet] get_neighbor_cell: stairs down layout, horizontal offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_DOWN);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(2, -1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-2, 1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
}

TEST_CASE("[TileSet] get_neighbor_cell: stairs right layout, vertical offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_RIGHT);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(-1, 2));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 2));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(1, -2));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -2));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
}

TEST_CASE("[TileSet] get_neighbor_cell: diamond right layout, horizontal offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_RIGHT);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, -1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));

	// Invalid neighbor for this shape
	ERR_PRINT_OFF;
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;
}

TEST_CASE("[TileSet] get_neighbor_cell: diamond down layout, vertical offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_DOWN);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(1, 1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(-1, -1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(-1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));

	// Invalid neighbor for this shape
	ERR_PRINT_OFF;
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;
}

TEST_CASE("[TileSet] get_neighbor_cell: diamond down layout, horizontal offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_DOWN);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, -1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(-1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1));

	// Invalid neighbor for this shape
	ERR_PRINT_OFF;
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;
}

TEST_CASE("[TileSet] get_neighbor_cell: diamond right layout, vertical offset axis") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_RIGHT);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(-1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -1));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0));

	// Invalid neighbor for this shape
	ERR_PRINT_OFF;
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(Vector2i(0, 0), TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;
}

} // namespace TestTileSet
