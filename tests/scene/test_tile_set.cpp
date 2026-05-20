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
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	Vector2i center_cell(0, 0);

	// First if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));

	// Second if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));

	center_cell = Vector2i(0, 1); // UGLY
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 2));

	// Third if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 2));

	// Fourth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));

	center_cell = Vector2i(0, 1);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 2));

	// Fifth if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 0));

	// Sixth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, -1));
	center_cell = Vector2i(0, 1);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, 0));

	// Seventh if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -2));

	// Eighth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1));
	center_cell = Vector2i(0, 1);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));

	// Error cases
	ERR_PRINT_OFF;
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;

	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================

	// TILE_OFFSET_AXIS_VERTICAL
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	// First if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(0, 1));

	// Second if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));

	center_cell = Vector2i(1, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(2, 1));

	// Third if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, 0));

	// Fourth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, 0));

	center_cell = Vector2i(1, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(3, 0));

	// Fifth if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));

	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(0, -1));

	// Sixth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, -1));

	center_cell = Vector2i(1, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, 0));

	// Seventh if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 0));

	// Eighth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0));

	center_cell = Vector2i(1, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));

	// Error cases
	ERR_PRINT_OFF;
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;
}

TEST_CASE("[TileSet] get_neighbor_cell on a non-square shaped tile for the stacked offset layout") {
	Ref<TileSet> tile_set = memnew(TileSet);
	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STACKED_OFFSET);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	Vector2i center_cell(0, 0);

	// First if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 0));

	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));

	// Second if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 1));

	center_cell = Vector2i(0, 1);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 2));

	// Third if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 2));

	// Fourth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));

	center_cell = Vector2i(0, 1);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 2));

	// Fifth if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 0));

	// Sixth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));
	center_cell = Vector2i(0, 1);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

	// Seventh if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -2));

	// Eighth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));
	center_cell = Vector2i(0, 1);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, 0));

	// Error cases
	ERR_PRINT_OFF;
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;

	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================

	// TILE_OFFSET_AXIS_VERTICAL
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	// First if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(0, 1));

	// Second if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 1));

	center_cell = Vector2i(1, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(2, 0));

	// Third if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, 0));

	// Fourth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, 0));

	center_cell = Vector2i(1, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(3, 0));

	// Fifth if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));

	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(0, -1));

	// Sixth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

	center_cell = Vector2i(1, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));

	// Seventh if
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 0));

	// Eighth if
	center_cell = Vector2i(0, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));

	center_cell = Vector2i(1, 0);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 0));

	// Error cases
	ERR_PRINT_OFF;
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;
}

TEST_CASE("[TileSet] get_neighbor_cell on a non-square shaped tile for the stairs layout") {
	Ref<TileSet> tile_set = memnew(TileSet);

	Vector2i center_cell(0, 0);

	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_RIGHT);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	// First if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));

	// Second if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));

	// Third if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 2));

	// Fourth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));

	// Fifth if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 0));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));

	// Sixth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));

	// Seventh if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -2));

	// Eighth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));

	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================

	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_DOWN);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	// First if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(0, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));

	// Second if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));

	// Third if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, -1));

	// Fourth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));

	// Fifth if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(0, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));

	// Sixth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

	// Seventh if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 1));

	// Eighth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));

	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================

	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_DOWN);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	// First if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(2, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(2, -1));

	// Second if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));

	// Third if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(0, 1));

	// Fourth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));

	// Fifth if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-2, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-2, 1));

	// Sixth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

	// Seventh if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, -1));

	// Eighth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));

	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================

	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_STAIRS_RIGHT);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	// First if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(-1, 2));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 2));

	// Second if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));

	// Third if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 0));

	// Fourth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, -1));

	// Fifth if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(1, -2));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -2));

	// Sixth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));

	// Seventh if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 0));

	// Eighth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 1));
}

TEST_CASE("[TileSet] get_neighbor_cell on a non-square shaped tile for the diamond layout") {
	Ref<TileSet> tile_set = memnew(TileSet);
	Vector2i center_cell(0, 0);

	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_RIGHT);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	// First if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, 1));

	// Second if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));

	// Third if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 1));

	// Fourth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0));

	// Fifth if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, -1));

	// Sixth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));

	// Seventh if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -1));

	// Eighth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));

	// Error cases
	ERR_PRINT_OFF;
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;

	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================

	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_DOWN);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	// First if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(1, 1));

	// Second if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));

	// Third if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, -1));

	// Fourth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1));

	// Fifth if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(-1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(-1, -1));

	// Sixth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

	// Seventh if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 1));

	// Eighth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));

	// Error cases
	ERR_PRINT_OFF;
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;

	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================

	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_DOWN);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_HORIZONTAL);

	// First if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_SIDE) == Vector2i(1, -1));

	// Second if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(1, 0));

	// Third if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(1, 1));

	// Fourth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(0, 1));

	// Fifth if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_SIDE) == Vector2i(-1, 1));

	// Sixth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(-1, 0));

	// Seventh if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(-1, -1));

	// Eighth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(0, -1));

	// Error cases
	ERR_PRINT_OFF;
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;

	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================

	tile_set->set_tile_layout(TileSet::TileLayout::TILE_LAYOUT_DIAMOND_RIGHT);
	tile_set->set_tile_offset_axis(TileSet::TileOffsetAxis::TILE_OFFSET_AXIS_VERTICAL);

	// First if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_CORNER) == Vector2i(-1, 1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_SIDE) == Vector2i(-1, 1));

	// Second if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE) == Vector2i(0, 1));

	// Third if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_RIGHT_CORNER) == Vector2i(1, 1));

	// Fourth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_RIGHT_SIDE) == Vector2i(1, 0));

	// Fifth if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_CORNER) == Vector2i(1, -1));

	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_SIDE) == Vector2i(1, -1));

	// Sixth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_TOP_LEFT_SIDE) == Vector2i(0, -1));

	// Seventh if
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(-1, -1));

	// Eighth if
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE) == Vector2i(-1, 0));

	// Error cases
	ERR_PRINT_OFF;
	center_cell = Vector2i(0, 0);
	tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);
	CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::CELL_NEIGHBOR_LEFT_CORNER) == Vector2i(0, 0));
	ERR_PRINT_ON;
}

} // namespace TestTileSet
