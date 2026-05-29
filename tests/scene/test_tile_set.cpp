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

struct NeighborTestCase {
	TileSet::CellNeighbor direction;
	Vector2i expected_position;
};

namespace TestTileSet {

class NeighborCellTester {
private:
	Ref<TileSet> tile_set;

public:
	NeighborCellTester() : tile_set(memnew(TileSet)) {}

	void configure(TileSet::TileShape shape,
			TileSet::TileLayout layout = TileSet::TILE_LAYOUT_STACKED,
			TileSet::TileOffsetAxis offset = TileSet::TILE_OFFSET_AXIS_HORIZONTAL) {
		tile_set->set_tile_shape(shape);
		tile_set->set_tile_layout(layout);
		tile_set->set_tile_offset_axis(offset);
	}

	void test_neighbors(const Vector2i &center, const Vector<NeighborTestCase> &cases) {
		for (const auto &test_case : cases) {
			Vector2i result = tile_set->get_neighbor_cell(center, test_case.direction);
			CHECK(result == test_case.expected_position);
		}
	}

	void test_error_case(const Vector2i &center, TileSet::CellNeighbor direction) {
		ERR_PRINT_OFF;
		Vector2i result = tile_set->get_neighbor_cell(center, direction);
		CHECK(result == center);
		ERR_PRINT_ON;
	}
};

TEST_CASE("[TileSet] get_neighbor_cell on a square shaped tile") {
	NeighborCellTester tester;
	tester.configure(TileSet::TILE_SHAPE_SQUARE);

	Vector2i center_cell(0, 0);

	SUBCASE("All neighbors of a square tile") {
		Vector<NeighborTestCase> square_neighbors = {
			{ TileSet::CELL_NEIGHBOR_TOP_SIDE, Vector2i(0, -1) },
			{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER, Vector2i(1, -1) },
			{ TileSet::CELL_NEIGHBOR_RIGHT_SIDE, Vector2i(1, 0) },
			{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER, Vector2i(1, 1) },
			{ TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, Vector2i(0, 1) },
			{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER, Vector2i(-1, 1) },
			{ TileSet::CELL_NEIGHBOR_LEFT_SIDE, Vector2i(-1, 0) },
			{ TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER, Vector2i(-1, -1) }
		};

		tester.test_neighbors(center_cell, square_neighbors);
	}
}

TEST_CASE("[TileSet] get_neighbor_cell on a non-square shaped tile for the stacked layout") {
	NeighborCellTester tester;

	SUBCASE("Hexagon") {
		SUBCASE("horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_STACKED,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors_even = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(0, -1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors_even);

			Vector<NeighborTestCase> neighbors_odd = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 2) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 2) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, 0) }
			};
			tester.test_neighbors(Vector2i(0, 1), neighbors_odd);

			tester.test_error_case(Vector2i(0, 0), TileSet::CELL_NEIGHBOR_TOP_CORNER);
		}

		SUBCASE("vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_STACKED,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors_even = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, -1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 0) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors_even);

			Vector<NeighborTestCase> neighbors_odd = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(2, 1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 1) }
			};
			tester.test_neighbors(Vector2i(1, 0), neighbors_odd);

			tester.test_error_case(Vector2i(0, 0), TileSet::CELL_NEIGHBOR_LEFT_CORNER);
		}
	}

	SUBCASE("isometric") {
		SUBCASE("horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_STACKED,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors_even = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(0, 2) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(0, -2) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(0, -1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors_even);

			Vector<NeighborTestCase> neighbors_odd = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 2) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 2) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, 0) }
			};
			tester.test_neighbors(Vector2i(0, 1), neighbors_odd);
		}

		SUBCASE("vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_STACKED,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors_even = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(2, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, -1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-2, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 0) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors_even);

			Vector<NeighborTestCase> neighbors_odd = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(2, 1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(2, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 1) }
			};
			tester.test_neighbors(Vector2i(1, 0), neighbors_odd);
		}
	}
}

TEST_CASE("[TileSet] get_neighbor_cell on a non-square shaped tile for the stacked offset layout") {
	NeighborCellTester tester;

	SUBCASE("hexagon") {
		SUBCASE("horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_STACKED_OFFSET,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors_even = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, -1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors_even);

			Vector<NeighborTestCase> neighbors_odd = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 2) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 2) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(0, 0) }
			};
			tester.test_neighbors(Vector2i(0, 1), neighbors_odd);

			tester.test_error_case(Vector2i(0, 0), TileSet::CELL_NEIGHBOR_TOP_CORNER);
		}

		SUBCASE("vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_STACKED_OFFSET,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors_even = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 1) },
				{ TileSet::CELL_NEIGHBOR_TOP_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors_even);

			Vector<NeighborTestCase> neighbors_odd = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(2, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 0) }
			};
			tester.test_neighbors(Vector2i(1, 0), neighbors_odd);

			tester.test_error_case(Vector2i(0, 0), TileSet::CELL_NEIGHBOR_LEFT_CORNER);
		}
	}

	SUBCASE("isometric") {
		SUBCASE("horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_STACKED_OFFSET,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors_even = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(0, 2) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(0, -2) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, -1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors_even);

			Vector<NeighborTestCase> neighbors_odd = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 2) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 2) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(0, 0) }
			};
			tester.test_neighbors(Vector2i(0, 1), neighbors_odd);
		}

		SUBCASE("vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_STACKED_OFFSET,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors_even = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 1) },
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(2, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-2, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors_even);

			Vector<NeighborTestCase> neighbors_odd = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(2, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(2, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 0) }
			};
			tester.test_neighbors(Vector2i(1, 0), neighbors_odd);
		}
	}
}

TEST_CASE("[TileSet] get_neighbor_cell on a non-square shaped tile for the stairs layout") {
	NeighborCellTester tester;

	SUBCASE("hexagon") {
		SUBCASE("stairs right, horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_STAIRS_RIGHT,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, -1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}

		SUBCASE("stairs down, vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_STAIRS_DOWN,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}

		SUBCASE("stairs down, horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_STAIRS_DOWN,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_SIDE, Vector2i(2, -1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_SIDE, Vector2i(-2, 1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, -1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}

		SUBCASE("stairs right, vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_STAIRS_RIGHT,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, Vector2i(-1, 2) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_SIDE, Vector2i(1, -2) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}
	}

	SUBCASE("isometric") {
		SUBCASE("stairs right, horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_STAIRS_RIGHT,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(-1, 2) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(1, -2) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, -1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}

		SUBCASE("stairs down, vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_STAIRS_DOWN,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(2, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-2, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}

		SUBCASE("stairs down, horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_STAIRS_DOWN,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(2, -1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-2, 1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, -1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}

		SUBCASE("stairs right, vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_STAIRS_RIGHT,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(-1, 2) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(1, -2) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}
	}
}

TEST_CASE("[TileSet] get_neighbor_cell on a non-square shaped tile for the diamond layout") {
	NeighborCellTester tester;

	Ref<TileSet> tile_set = memnew(TileSet);

	SUBCASE("hexagon") {
		tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_HEXAGON);

		SUBCASE("diamond right, horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_DIAMOND_RIGHT,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_SIDE, Vector2i(1, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_LEFT_SIDE, Vector2i(-1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, 0) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);

			tester.test_error_case(Vector2i(0, 0), TileSet::CELL_NEIGHBOR_TOP_CORNER);
		}

		SUBCASE("diamond down, vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_DIAMOND_DOWN,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, Vector2i(1, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_SIDE, Vector2i(-1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);

			tester.test_error_case(Vector2i(0, 0), TileSet::CELL_NEIGHBOR_LEFT_CORNER);
		}

		SUBCASE("diamond down, horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_DIAMOND_DOWN,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_SIDE, Vector2i(1, -1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_SIDE, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(0, -1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);

			tester.test_error_case(Vector2i(0, 0), TileSet::CELL_NEIGHBOR_TOP_CORNER);
		}

		SUBCASE("diamond right, vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_HEXAGON,
					TileSet::TILE_LAYOUT_DIAMOND_RIGHT,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_SIDE, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_SIDE, Vector2i(1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 0) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);

			tester.test_error_case(Vector2i(0, 0), TileSet::CELL_NEIGHBOR_LEFT_CORNER);
		}
	}

	SUBCASE("isometric") {
		tile_set->set_tile_shape(TileSet::TileShape::TILE_SHAPE_ISOMETRIC);

		SUBCASE("diamond right, horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_DIAMOND_RIGHT,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(1, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, 0) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}

		SUBCASE("diamond down, vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_DIAMOND_DOWN,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(1, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(-1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}

		SUBCASE("diamond down, horizontal offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_DIAMOND_DOWN,
					TileSet::TILE_OFFSET_AXIS_HORIZONTAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(1, -1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(1, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(-1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(-1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(0, -1) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}

		SUBCASE("diamond right, vertical offset axis") {
			tester.configure(TileSet::TILE_SHAPE_ISOMETRIC,
					TileSet::TILE_LAYOUT_DIAMOND_RIGHT,
					TileSet::TILE_OFFSET_AXIS_VERTICAL);

			Vector<NeighborTestCase> neighbors = {
				{ TileSet::CELL_NEIGHBOR_BOTTOM_CORNER, Vector2i(-1, 1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE, Vector2i(0, 1) },
				{ TileSet::CELL_NEIGHBOR_RIGHT_CORNER, Vector2i(1, 1) },
				{ TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE, Vector2i(1, 0) },
				{ TileSet::CELL_NEIGHBOR_TOP_CORNER, Vector2i(1, -1) },
				{ TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE, Vector2i(0, -1) },
				{ TileSet::CELL_NEIGHBOR_LEFT_CORNER, Vector2i(-1, -1) },
				{ TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE, Vector2i(-1, 0) }
			};
			tester.test_neighbors(Vector2i(0, 0), neighbors);
		}
	}
}

} // namespace TestTileSet
