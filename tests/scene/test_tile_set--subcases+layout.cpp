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

TEST_CASE("[TileSet] get_neighbor_cell on a square shaped tile") {
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

TEST_CASE("[TileSet] get_neighbor_cell on a non-square shaped tile for the stacked layout") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STACKED);

	Vector2i center_cell(0, 0);

	SUBCASE("hexagon") {
		tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);

		SUBCASE("horizontal offset axis") {
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 0));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));
			center_cell = Vector2i(0, 1); // UGLY
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 2));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 2));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 0));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, -1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, 0));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));

			ERR_PRINT_OFF;
			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
			ERR_PRINT_ON;
		}

		SUBCASE("vertical offset axis") {
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(0, 1));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(2, 1));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(0, -1));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, -1));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, 0));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));

			ERR_PRINT_OFF;
			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
			ERR_PRINT_ON;
		}
	}

	SUBCASE("isometric") {
		tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);

		SUBCASE("horizontal offset axis") {
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));
			center_cell = Vector2i(0, 1); // UGLY
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 2));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 2));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 2));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, -1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, 0));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -2));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));
		}

		SUBCASE("vertical offset axis") {
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(2, 1));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, 0));

			// Fourth if (buggy in original — tests RIGHT_CORNER instead of TOP_RIGHT_SIDE)
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, 0));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(3, 0));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, -1));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, 0));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 0));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));
		}
	}
}

TEST_CASE("[TileSet] get_neighbor_cell on a non-square shaped tile for the stacked offset layout") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STACKED_OFFSET);

	Vector2i center_cell(0, 0);

	SUBCASE("hexagon") {
		tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);

		SUBCASE("horizontal offset axis") {
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 0));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 2));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 2));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 0));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, 0));

			ERR_PRINT_OFF;
			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
			ERR_PRINT_ON;
		}

		SUBCASE("vertical offset axis") {
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(0, 1));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 1));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(2, 0));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(0, -1));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 0));

			ERR_PRINT_OFF;
			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
			ERR_PRINT_ON;
		}
	}

	SUBCASE("isometric") {
		tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);

		SUBCASE("horizontal offset axis") {
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 2));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 2));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 2));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -2));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
			center_cell = Vector2i(0, 1);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, 0));
		}

		SUBCASE("vertical offset axis") {
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 1));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(2, 0));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, 0));

			// Fourth if (buggy in original — tests RIGHT_CORNER instead of TOP_RIGHT_SIDE)
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, 0));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(3, 0));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));

			center_cell = Vector2i(0, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 0));

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
			center_cell = Vector2i(1, 0);
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 0));
		}
	}
}

TEST_CASE("[TileSet] get_neighbor_cell on a non-square shaped tile for the stairs layout") {
	Ref<TileSet> tile_set = memnew(TileSet);

	Vector2i center_cell(0, 0);

	SUBCASE("hexagon") {
		tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);

		SUBCASE("stairs right, horizontal offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_RIGHT);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
		}

		SUBCASE("stairs down, vertical offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_DOWN);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
		}

		SUBCASE("stairs down, horizontal offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_DOWN);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(2, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-2, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
		}

		SUBCASE("stairs right, vertical offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_RIGHT);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(-1, 2));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(1, -2));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
		}
	}

	SUBCASE("isometric") {
		tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);

		SUBCASE("stairs right, horizontal offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_RIGHT);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 2));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -2));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
		}

		SUBCASE("stairs down, vertical offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_DOWN);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
		}

		SUBCASE("stairs down, horizontal offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_DOWN);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
		}

		SUBCASE("stairs right, vertical offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_RIGHT);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 2));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -2));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
		}
	}
}

TEST_CASE("[TileSet] get_neighbor_cell on a non-square shaped tile for the diamond layout") {
	Ref<TileSet> tile_set = memnew(TileSet);

	Vector2i center_cell(0, 0);

	SUBCASE("hexagon") {
		tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);

		SUBCASE("diamond right, horizontal offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_RIGHT);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));

			ERR_PRINT_OFF;
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
			ERR_PRINT_ON;
		}

		SUBCASE("diamond down, vertical offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_DOWN);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(-1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));

			ERR_PRINT_OFF;
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
			ERR_PRINT_ON;
		}

		SUBCASE("diamond down, horizontal offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_DOWN);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1));

			ERR_PRINT_OFF;
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
			ERR_PRINT_ON;
		}

		SUBCASE("diamond right, vertical offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_RIGHT);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(-1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0));

			ERR_PRINT_OFF;
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
			ERR_PRINT_ON;
		}
	}

	SUBCASE("isometric") {
		tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);

		SUBCASE("diamond right, horizontal offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_RIGHT);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));
		}

		SUBCASE("diamond down, vertical offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_DOWN);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(-1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));
		}

		SUBCASE("diamond down, horizontal offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_DOWN);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(-1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1));
		}

		SUBCASE("diamond right, vertical offset axis") {
			tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_RIGHT);
			tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, -1));
			CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0));
		}
	}
}

} // namespace TestTileSet
