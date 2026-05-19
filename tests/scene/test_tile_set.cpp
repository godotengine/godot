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

TEST_CASE("[TileSet] get_neighbor_cell") {
    TileSet *tile_set = memnew(TileSet);
    tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_SQUARE);

    Vector2i center_cell(0, 0);

    CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::TOP) == Vector2i(0, -1));
    CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::TOP_RIGHT) == Vector2i(1, -1));
    CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::RIGHT) == Vector2i(1, 0));
    CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::BOTTOM_RIGHT) == Vector2i(1, 1));
    CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::BOTTOM) == Vector2i(0, 1));
    CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::BOTTOM_LEFT) == Vector2i(-1, 1));
    CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::LEFT) == Vector2i(-1, 0));
    CHECK(tile_set->get_neighbor_cell(center_cell, TileSet::CellNeighbor::TOP_LEFT) == Vector2i(-1, -1));
}

} // namespace TestTileSet