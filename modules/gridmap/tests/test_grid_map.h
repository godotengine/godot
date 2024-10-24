/**************************************************************************/
/*  test_grid_map.h                                                       */
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

#ifndef TEST_GRID_MAP_H
#define TEST_GRID_MAP_H

#include "../grid_map.h"
#include "core/math/math_defs.h"
#include "core/math/vector3i.h"
#include "core/string/print_string.h"
#include "tests/core/math/test_vector4.h"
#include "tests/test_macros.h"

namespace TestGridMap {

TEST_CASE("[SceneTree][GridMap] hex cell set_cell_size()") {
	GridMap map;
	map.set_cell_shape(GridMap::CELL_SHAPE_HEXAGON);
	map.set_cell_size(Vector3(1.0, 3.0, 200.0));
	CHECK(map.get_cell_size().x == 1.0);
	CHECK(map.get_cell_size().y == 3.0);
	CHECK_MESSAGE(map.get_cell_size().z == 1.0,
			"set_cell_size() hex cell, z value should be overwritten with x");
}

#define CHECK_HEX_BOUNDARIES(p, c)                                                     \
	{                                                                                  \
		Vector3i index = map.local_to_map((p));                                        \
		CHECK_MESSAGE(index == (c), #p " center not in expected cell");                \
		index = map.local_to_map((p) + Vector3(0, 0, 0.499));                          \
		CHECK_MESSAGE(index == (c), #p " east edge not in expected cell");             \
		index = map.local_to_map((p) + Vector3(Math_SQRT3 / 2.0 - 0.001, 0, 0.499));   \
		CHECK_MESSAGE(index == (c), #p " northeast vertex not in expected cell");      \
		index = map.local_to_map((p) + Vector3(0, 0, 0.999));                          \
		CHECK_MESSAGE(index == (c), #p " north vertex not in expected cell");          \
		index = map.local_to_map((p) + Vector3(-Math_SQRT3 / 2.0 + 0.001, 0, 0.499));  \
		CHECK_MESSAGE(index == (c), #p " northwest vertex not in expected cell");      \
		index = map.local_to_map((p) + Vector3(0, 0, -0.499));                         \
		CHECK_MESSAGE(index == (c), #p " west edge not in expected cell");             \
		index = map.local_to_map((p) + Vector3(-Math_SQRT3 / 2.0 + 0.001, 0, -0.499)); \
		CHECK_MESSAGE(index == (c), #p " southwest vertex not in expected cell");      \
		index = map.local_to_map((p) + Vector3(0, 0, -0.99));                          \
		CHECK_MESSAGE(index == (c), #p " south vertex not in expected cell");          \
		index = map.local_to_map((p) + Vector3(Math_SQRT3 / 2.0 - 0.001, 0, -0.49));   \
		CHECK_MESSAGE(index == (c), #p " southeast vertex not in expected cell");      \
	}

TEST_CASE("[SceneTree][GridMap] local_to_map() for hex cells") {
	GridMap map;
	map.set_cell_shape(GridMap::CELL_SHAPE_HEXAGON);
	map.set_cell_size(Vector3(1.0, 1.0, 1.0));

	// verify origin
	CHECK_HEX_BOUNDARIES(Vector3(0, 0, 0), Vector3i(0, 0, 0));
	// north two rows
	CHECK_HEX_BOUNDARIES(Vector3(0, 0, 3), Vector3i(-1, 0, 2));
	// south two rows
	CHECK_HEX_BOUNDARIES(Vector3(0, 0, -3), Vector3i(1, 0, -2));
	// east one cell
	CHECK_HEX_BOUNDARIES(Vector3(Math_SQRT3, 0, 0), Vector3i(1, 0, 0));
	// west one cell
	CHECK_HEX_BOUNDARIES(Vector3(-Math_SQRT3, 0, 0), Vector3i(-1, 0, 0));
	// northeast one cell
	CHECK_HEX_BOUNDARIES(Vector3(Math_SQRT3 / 2, 0, 1.5), Vector3i(0, 0, 1));
	// northwest one cell
	CHECK_HEX_BOUNDARIES(Vector3(-Math_SQRT3 / 2, 0, 1.5), Vector3i(-1, 0, 1));
	// southwest one cell
	CHECK_HEX_BOUNDARIES(Vector3(-Math_SQRT3 / 2, 0, -1.5), Vector3i(0, 0, -1));
}

TEST_CASE("[SceneTree][GridMap] map_to_local() for hex cells") {
	GridMap map;
	map.set_cell_shape(GridMap::CELL_SHAPE_HEXAGON);
	map.set_cell_size(Vector3(1.0, 1.0, 1.0));

	CHECK(map.map_to_local(Vector3i(0, 0, 0)) == Vector3(0, 0.5, 0));
	CHECK(map.map_to_local(Vector3i(-1, 0, 2)) == Vector3(0, 0.5, 3));
	CHECK(map.map_to_local(Vector3i(1, 0, -2)) == Vector3(0, 0.5, -3));
	CHECK(map.map_to_local(Vector3i(1, 0, 0)) == Vector3(Math_SQRT3, 0.5, 0));
	CHECK(map.map_to_local(Vector3i(-1, 0, 0)) == Vector3(-Math_SQRT3, 0.5, 0));
	CHECK(map.map_to_local(Vector3i(0, 0, 1)) == Vector3(Math_SQRT3 / 2, 0.5, 1.5));
	CHECK(map.map_to_local(Vector3i(-1, 0, 1)) == Vector3(-Math_SQRT3 / 2, 0.5, 1.5));
	CHECK(map.map_to_local(Vector3i(-1, 0, 1)) == Vector3(-Math_SQRT3 / 2, 0.5, 1.5));
	CHECK(map.map_to_local(Vector3i(0, 0, -1)) == Vector3(-Math_SQRT3 / 2, 0.5, -1.5));

	CHECK(map.map_to_local(Vector3i(0, 1, 0)) == Vector3(0, 1.5, 0));
	CHECK(map.map_to_local(Vector3i(0, -1, 0)) == Vector3(0, -0.5, 0));
}

TEST_CASE("[SceneTree][GridMap] map_to_local() with square cells") {
	GridMap map;
	map.set_cell_shape(GridMap::CELL_SHAPE_SQUARE);
	map.set_cell_size(Vector3(1.0, 1.0, 1.0));

	// even with zero volume, we'll include the single cell we are within.
	TypedArray<Vector3i> cells = map.local_region_to_map(Vector3(0, 0, 0), Vector3(0, 0, 0));
	CHECK(cells.has(Vector3i(0, 0, 0)));
	CHECK(cells.size() == 1);

	// three cells along the x-axis
	cells = map.local_region_to_map(Vector3(0, 0, 0), Vector3(2, 0, 0));
	CHECK(cells.has(Vector3i(0, 0, 0)));
	CHECK(cells.has(Vector3i(1, 0, 0)));
	CHECK(cells.has(Vector3i(2, 0, 0)));
	CHECK(cells.size() == 3);

	// three cells along the y-axis
	cells = map.local_region_to_map(Vector3(0, 0, 0), Vector3(0, 2, 0));
	CHECK(cells.has(Vector3i(0, 0, 0)));
	CHECK(cells.has(Vector3i(0, 1, 0)));
	CHECK(cells.has(Vector3i(0, 2, 0)));
	CHECK(cells.size() == 3);

	// three cells along the z-axis
	cells = map.local_region_to_map(Vector3(0, 0, 0), Vector3(0, 0, 2));
	CHECK(cells.has(Vector3i(0, 0, 0)));
	CHECK(cells.has(Vector3i(0, 0, 1)));
	CHECK(cells.has(Vector3i(0, 0, 2)));
	CHECK(cells.size() == 3);

	// three by three by three region starting at origin
	cells = map.local_region_to_map(Vector3(0, 0, 0), Vector3(2, 2, 2));
	CHECK(cells.size() == 3 * 3 * 3);
	for (int z = 0; z <= 2; z++) {
		for (int y = 0; y <= 2; y++) {
			for (int x = 0; x <= 2; x++) {
				CHECK_MESSAGE(cells.has(Vector3i(x, y, z)),
						"cells should contain (", x, ", ", y, ", ", z, ")");
			}
		}
	}

	// two by three by four region starting at (-10, -100, -1000)
	cells = map.local_region_to_map(Vector3(-10, -100, -1000), Vector3(-11, -102, -1003));
	CHECK(cells.size() == 2 * 3 * 4);
	for (int z = -1003; z <= -1000; z++) {
		for (int y = -102; y <= -100; y++) {
			for (int x = -11; x <= -10; x++) {
				CHECK_MESSAGE(cells.has(Vector3i(x, y, z)),
						"cells should contain (", x, ", ", y, ", ", z, ")");
			}
		}
	}

	// 5x2x1 passing through the origin
	cells = map.local_region_to_map(Vector3(-2, -1, 0), Vector3(2, 0, 0));
	CHECK(cells.size() == 5 * 2 * 1);
	for (int z = 0; z <= 0; z++) {
		for (int y = -1; y <= 0; y++) {
			for (int x = -2; x <= -2; x++) {
				CHECK_MESSAGE(cells.has(Vector3i(x, y, z)),
						"cells should contain (", x, ", ", y, ", ", z, ")");
			}
		}
	}

	SUBCASE("2x2x2 cube tiles") {
		GridMap test_map;
		test_map.set_cell_shape(GridMap::CELL_SHAPE_SQUARE);
		test_map.set_cell_size(Vector3(2.0, 2.0, 2.0));

		cells = test_map.local_region_to_map(Vector3(0, 0, 0), Vector3(3.9, 0, 3.9));
		CHECK(cells.has(Vector3i(0, 0, 0)));
		CHECK(cells.has(Vector3i(0, 0, 1)));
		CHECK(cells.has(Vector3i(1, 0, 0)));
		CHECK(cells.has(Vector3i(1, 0, 1)));
		CHECK(cells.size() == 4);
	}
}

TEST_CASE("[SceneTree][GridMap] map_to_local() with hex cells") {
	GridMap map;
	map.set_cell_shape(GridMap::CELL_SHAPE_HEXAGON);
	map.set_cell_size(Vector3(1.0, 1.0, 1.0));

	// even with zero volume, we'll include the single cell we are within.
	TypedArray<Vector3i> cells = map.local_region_to_map(Vector3(0, 0, 0), Vector3(0, 0, 0));
	CHECK(cells.has(Vector3i(0, 0, 0)));
	CHECK(cells.size() == 1);

	// three cells along the x-axis
	cells = map.local_region_to_map(Vector3(0, 0, 0), Vector3(Math_SQRT3 * 2, 0, 0));
	CHECK(cells.has(Vector3i(0, 0, 0)));
	CHECK(cells.has(Vector3i(1, 0, 0)));
	CHECK(cells.has(Vector3i(2, 0, 0)));
	CHECK(cells.size() == 3);

	// three cells along the y-axis
	cells = map.local_region_to_map(Vector3(0, 0, 0), Vector3(0, 2, 0));
	CHECK(cells.has(Vector3i(0, 0, 0)));
	CHECK(cells.has(Vector3i(0, 1, 0)));
	CHECK(cells.has(Vector3i(0, 2, 0)));
	CHECK(cells.size() == 3);

	// cells along the z-axis, slightly to the left of center to ensure we add
	// the southwest zigzag cells
	cells = map.local_region_to_map(Vector3(-0.1, 0, 0), Vector3(-0.1, 0, 3));
	CHECK(cells.has(Vector3i(0, 0, 0)));
	CHECK(cells.has(Vector3i(-1, 0, 1)));
	CHECK(cells.has(Vector3i(-1, 0, 2)));
	CHECK(cells.size() == 3);

	// Same as above, but go southeast instead
	cells = map.local_region_to_map(Vector3(0.1, 0, 0), Vector3(0.1, 0, 3));
	CHECK(cells.has(Vector3i(0, 0, 0)));
	CHECK(cells.has(Vector3i(0, 0, 1)));
	CHECK(cells.has(Vector3i(-1, 0, 2)));
	CHECK(cells.size() == 3);

	// square region around the origin
	cells = map.local_region_to_map(Vector3(-Math_SQRT3, 0, -1), Vector3(Math_SQRT3, 0, 1));
	CHECK(cells.has(Vector3i(0, 0, 0)));
	CHECK(cells.has(Vector3i(0, 0, 1)));
	CHECK(cells.has(Vector3i(0, 0, -1)));
	CHECK(cells.has(Vector3i(1, 0, -1)));
	CHECK(cells.has(Vector3i(1, 0, 0)));
	CHECK(cells.has(Vector3i(-1, 0, 0)));
	CHECK(cells.has(Vector3i(-1, 0, 1)));
	CHECK(cells.size() == 7);

	// similar to above, but bring in the X coordinates to exclude east & west
	// cells
	cells = map.local_region_to_map(Vector3(-SQRT3_2 + 0.1, 0, -1), Vector3(SQRT3_2 - 0.1, 0, 1));
	CHECK(cells.has(Vector3i(0, 0, 0)));
	CHECK(cells.has(Vector3i(0, 0, 1)));
	CHECK(cells.has(Vector3i(0, 0, -1)));
	CHECK(cells.has(Vector3i(1, 0, -1)));
	CHECK(cells.has(Vector3i(-1, 0, 1)));
	CHECK(cells.size() == 5);
}

TEST_CASE("[SceneTree][GridMap] get_cell_neighbors() square cells") {
	GridMap map;
	map.set_cell_shape(GridMap::CELL_SHAPE_SQUARE);

	SUBCASE("cell (0,0,0)") {
		TypedArray<Vector3i> cells = map.get_cell_neighbors(Vector3i(0, 0, 0));
		CHECK(cells.size() == 6);
		CHECK(cells.has(Vector3i(1, 0, 0)));
		CHECK(cells.has(Vector3i(0, 1, 0)));
		CHECK(cells.has(Vector3i(0, 0, 1)));
		CHECK(cells.has(Vector3i(-1, 0, 0)));
		CHECK(cells.has(Vector3i(0, -1, 0)));
		CHECK(cells.has(Vector3i(0, 0, -1)));
	}

	SUBCASE("cell (-12,10,100)") {
		Vector3i cell = Vector3i(-12, 10, 100);
		TypedArray<Vector3i> cells = map.get_cell_neighbors(cell);
		CHECK(cells.size() == 6);
		CHECK(cells.has(cell + Vector3i(1, 0, 0)));
		CHECK(cells.has(cell + Vector3i(0, 1, 0)));
		CHECK(cells.has(cell + Vector3i(0, 0, 1)));
		CHECK(cells.has(cell + Vector3i(-1, 0, 0)));
		CHECK(cells.has(cell + Vector3i(0, -1, 0)));
		CHECK(cells.has(cell + Vector3i(0, 0, -1)));
	}
}

TEST_CASE("[SceneTree][GridMap] get_cell_neighbors() hex cells") {
	GridMap map;
	map.set_cell_shape(GridMap::CELL_SHAPE_HEXAGON);

	SUBCASE("cell (0,0,0)") {
		TypedArray<Vector3i> cells = map.get_cell_neighbors(Vector3i(0, 0, 0));
		CHECK(cells.size() == 8);
		CHECK(cells.has(Vector3i(1, 0, 0)));
		CHECK(cells.has(Vector3i(1, 0, -1)));
		CHECK(cells.has(Vector3i(0, 0, -1)));
		CHECK(cells.has(Vector3i(-1, 0, 0)));
		CHECK(cells.has(Vector3i(-1, 0, 1)));
		CHECK(cells.has(Vector3i(0, 0, 1)));
		CHECK(cells.has(Vector3i(0, -1, 0)));
		CHECK(cells.has(Vector3i(0, 1, 0)));
	}

	SUBCASE("cell (-12,10,100)") {
		Vector3i cell = Vector3i(-12, 10, 100);
		TypedArray<Vector3i> cells = map.get_cell_neighbors(cell);
		CHECK(cells.size() == 8);
		CHECK(cells.has(cell + Vector3i(1, 0, 0)));
		CHECK(cells.has(cell + Vector3i(1, 0, -1)));
		CHECK(cells.has(cell + Vector3i(0, 0, -1)));
		CHECK(cells.has(cell + Vector3i(-1, 0, 0)));
		CHECK(cells.has(cell + Vector3i(-1, 0, 1)));
		CHECK(cells.has(cell + Vector3i(0, 0, 1)));
		CHECK(cells.has(cell + Vector3i(0, -1, 0)));
		CHECK(cells.has(cell + Vector3i(0, 1, 0)));
	}
}
} // namespace TestGridMap

#endif // TEST_GRID_MAP_H
