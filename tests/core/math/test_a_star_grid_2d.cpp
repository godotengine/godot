/**************************************************************************/
/*  test_a_star_grid_2d.cpp                                               */
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

TEST_FORCE_LINK(test_a_star_grid_2d)

#include "core/math/a_star_grid_2d.h"
#include "core/variant/typed_array.h"

namespace TestAStarGrid2D {

TEST_CASE("[AStarGrid2D] Initialization") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();

	CHECK(grid->get_region() == Rect2i());
	CHECK(grid->get_offset() == Vector2());
	CHECK(grid->get_cell_size() == Size2(1, 1));
	CHECK(grid->get_cell_shape() == AStarGrid2D::CELL_SHAPE_SQUARE);
	CHECK(grid->get_diagonal_mode() == AStarGrid2D::DIAGONAL_MODE_ALWAYS);
	CHECK(grid->get_default_compute_heuristic() == AStarGrid2D::HEURISTIC_EUCLIDEAN);
	CHECK(grid->get_default_estimate_heuristic() == AStarGrid2D::HEURISTIC_EUCLIDEAN);
	CHECK(grid->is_jumping_enabled() == false);
	CHECK(grid->is_dirty() == false);
}

TEST_CASE("[AStarGrid2D] Region and update") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();

	grid->set_region(Rect2i(0, 0, 4, 4));
	CHECK(grid->get_region() == Rect2i(0, 0, 4, 4));
	CHECK(grid->is_dirty() == true);

	grid->update();
	CHECK(grid->is_dirty() == false);

	SUBCASE("Setting the same region does not mark dirty") {
		grid->set_region(Rect2i(0, 0, 4, 4));
		CHECK(grid->is_dirty() == false);
	}

	SUBCASE("Setting a different region marks dirty") {
		grid->set_region(Rect2i(0, 0, 8, 8));
		CHECK(grid->is_dirty() == true);
	}

	SUBCASE("Negative region size is rejected") {
		ERR_PRINT_OFF;
		grid->set_region(Rect2i(0, 0, -1, 4));
		ERR_PRINT_ON;
		// Region should remain unchanged.
		CHECK(grid->get_region() == Rect2i(0, 0, 4, 4));
	}
}

TEST_CASE("[AStarGrid2D] Cell size and offset") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();

	grid->set_cell_size(Size2(2, 3));
	CHECK(grid->get_cell_size() == Size2(2, 3));

	grid->set_offset(Vector2(5, 10));
	CHECK(grid->get_offset() == Vector2(5, 10));
}

TEST_CASE("[AStarGrid2D] Bounds checking") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 4, 4));
	grid->update();

	CHECK(grid->is_in_bounds(0, 0) == true);
	CHECK(grid->is_in_bounds(3, 3) == true);
	CHECK(grid->is_in_boundsv(Vector2i(2, 1)) == true);

	// Out of bounds.
	CHECK(grid->is_in_bounds(4, 4) == false);
	CHECK(grid->is_in_bounds(-1, 0) == false);
	CHECK(grid->is_in_boundsv(Vector2i(5, 0)) == false);
}

TEST_CASE("[AStarGrid2D] Solid points") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 4, 4));
	grid->update();

	// All points start as non-solid.
	CHECK(grid->is_point_solid(Vector2i(1, 1)) == false);

	grid->set_point_solid(Vector2i(1, 1));
	CHECK(grid->is_point_solid(Vector2i(1, 1)) == true);

	// Unset solid.
	grid->set_point_solid(Vector2i(1, 1), false);
	CHECK(grid->is_point_solid(Vector2i(1, 1)) == false);

	SUBCASE("Setting solid on out-of-bounds point fails") {
		ERR_PRINT_OFF;
		grid->set_point_solid(Vector2i(10, 10));
		ERR_PRINT_ON;
	}
}

TEST_CASE("[AStarGrid2D] Fill solid region") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 4, 4));
	grid->update();

	grid->fill_solid_region(Rect2i(1, 1, 2, 2));
	CHECK(grid->is_point_solid(Vector2i(1, 1)) == true);
	CHECK(grid->is_point_solid(Vector2i(2, 2)) == true);
	CHECK(grid->is_point_solid(Vector2i(0, 0)) == false);
	CHECK(grid->is_point_solid(Vector2i(3, 3)) == false);

	// Unfill.
	grid->fill_solid_region(Rect2i(1, 1, 2, 2), false);
	CHECK(grid->is_point_solid(Vector2i(1, 1)) == false);
	CHECK(grid->is_point_solid(Vector2i(2, 2)) == false);
}

TEST_CASE("[AStarGrid2D] Weight scale") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 4, 4));
	grid->update();

	// Default weight scale is 1.0.
	CHECK(grid->get_point_weight_scale(Vector2i(0, 0)) == doctest::Approx(1.0));

	grid->set_point_weight_scale(Vector2i(1, 1), 5.0);
	CHECK(grid->get_point_weight_scale(Vector2i(1, 1)) == doctest::Approx(5.0));

	SUBCASE("Negative weight scale is rejected") {
		ERR_PRINT_OFF;
		grid->set_point_weight_scale(Vector2i(0, 0), -1.0);
		ERR_PRINT_ON;
		CHECK(grid->get_point_weight_scale(Vector2i(0, 0)) == doctest::Approx(1.0));
	}
}

TEST_CASE("[AStarGrid2D] Fill weight scale region") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 4, 4));
	grid->update();

	grid->fill_weight_scale_region(Rect2i(0, 0, 2, 2), 3.0);
	CHECK(grid->get_point_weight_scale(Vector2i(0, 0)) == doctest::Approx(3.0));
	CHECK(grid->get_point_weight_scale(Vector2i(1, 1)) == doctest::Approx(3.0));
	// Outside the fill region.
	CHECK(grid->get_point_weight_scale(Vector2i(3, 3)) == doctest::Approx(1.0));
}

TEST_CASE("[AStarGrid2D] Point position with default cell size") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 4, 4));
	grid->update();

	// With cell_size (1,1), offset (0,0), square shape: position = (x, y) * cell_size.
	CHECK(grid->get_point_position(Vector2i(0, 0)) == Vector2(0, 0));
	CHECK(grid->get_point_position(Vector2i(2, 3)) == Vector2(2, 3));
}

TEST_CASE("[AStarGrid2D] Point position with custom cell size and offset") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_cell_size(Size2(2, 2));
	grid->set_offset(Vector2(10, 20));
	grid->set_region(Rect2i(0, 0, 4, 4));
	grid->update();

	// position = offset + (x, y) * cell_size = (10, 20) + (0, 0) = (10, 20).
	CHECK(grid->get_point_position(Vector2i(0, 0)) == Vector2(10, 20));
	// position = (10, 20) + (1, 1) * (2, 2) = (12, 22).
	CHECK(grid->get_point_position(Vector2i(1, 1)) == Vector2(12, 22));
}

TEST_CASE("[AStarGrid2D] Simple pathfinding") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 4, 4));
	grid->set_diagonal_mode(AStarGrid2D::DIAGONAL_MODE_NEVER);
	grid->update();

	// Straight path with no obstacles, no diagonals.
	// From (0,0) to (3,0) should go right.
	TypedArray<Vector2i> path = grid->get_id_path(Vector2i(0, 0), Vector2i(3, 0));
	REQUIRE(path.size() == 4);
	CHECK(Vector2i(path[0]) == Vector2i(0, 0));
	CHECK(Vector2i(path[3]) == Vector2i(3, 0));
}

TEST_CASE("[AStarGrid2D] Path to same point") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 4, 4));
	grid->update();

	TypedArray<Vector2i> path = grid->get_id_path(Vector2i(1, 1), Vector2i(1, 1));
	REQUIRE(path.size() == 1);
	CHECK(Vector2i(path[0]) == Vector2i(1, 1));
}

TEST_CASE("[AStarGrid2D] Path around obstacle") {
	// Grid layout (S=start, E=end, X=solid):
	//   0 1 2 3
	// 0 S . . .
	// 1 . X X .
	// 2 . . . E
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 4, 3));
	grid->set_diagonal_mode(AStarGrid2D::DIAGONAL_MODE_NEVER);
	grid->update();

	grid->set_point_solid(Vector2i(1, 1));
	grid->set_point_solid(Vector2i(2, 1));

	TypedArray<Vector2i> path = grid->get_id_path(Vector2i(0, 0), Vector2i(3, 2));
	REQUIRE(path.size() > 0);
	CHECK(Vector2i(path[0]) == Vector2i(0, 0));
	CHECK(Vector2i(path[path.size() - 1]) == Vector2i(3, 2));

	// The path should not go through solid points.
	for (int i = 0; i < path.size(); i++) {
		CHECK(grid->is_point_solid(Vector2i(path[i])) == false);
	}
}

TEST_CASE("[AStarGrid2D] No path when blocked") {
	// Grid layout:
	//   0 1 2
	// 0 S X .
	// 1 X . E
	// Start (0,0) is completely walled off.
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 3, 2));
	grid->set_diagonal_mode(AStarGrid2D::DIAGONAL_MODE_NEVER);
	grid->update();

	grid->set_point_solid(Vector2i(1, 0));
	grid->set_point_solid(Vector2i(0, 1));

	ERR_PRINT_OFF;
	TypedArray<Vector2i> path = grid->get_id_path(Vector2i(0, 0), Vector2i(2, 1));
	ERR_PRINT_ON;
	CHECK(path.size() == 0);
}

TEST_CASE("[AStarGrid2D] Partial path") {
	// When fully blocked but partial path is allowed, we should get a path
	// to the closest reachable point.
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 5, 1));
	grid->set_diagonal_mode(AStarGrid2D::DIAGONAL_MODE_NEVER);
	grid->update();

	// Wall at x=2, blocking passage.
	grid->set_point_solid(Vector2i(2, 0));

	TypedArray<Vector2i> partial = grid->get_id_path(Vector2i(0, 0), Vector2i(4, 0), true);
	REQUIRE(partial.size() > 0);
	CHECK(Vector2i(partial[0]) == Vector2i(0, 0));
	// Should end at the closest reachable point (1, 0).
	CHECK(Vector2i(partial[partial.size() - 1]) == Vector2i(1, 0));
}

TEST_CASE("[AStarGrid2D] Diagonal modes") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 3, 3));

	SUBCASE("DIAGONAL_MODE_ALWAYS allows diagonal movement") {
		grid->set_diagonal_mode(AStarGrid2D::DIAGONAL_MODE_ALWAYS);
		grid->update();

		TypedArray<Vector2i> path = grid->get_id_path(Vector2i(0, 0), Vector2i(2, 2));
		// With diagonals, the path should be shorter than without.
		// Direct diagonal: (0,0) -> (1,1) -> (2,2) = 3 points.
		CHECK(path.size() == 3);
	}

	SUBCASE("DIAGONAL_MODE_NEVER disables diagonal movement") {
		grid->set_diagonal_mode(AStarGrid2D::DIAGONAL_MODE_NEVER);
		grid->update();

		TypedArray<Vector2i> path = grid->get_id_path(Vector2i(0, 0), Vector2i(2, 2));
		// Without diagonals, must take Manhattan path: 5 points (2 right + 2 down + 1 start).
		CHECK(path.size() == 5);
	}
}

TEST_CASE("[AStarGrid2D] Weight scale affects pathfinding") {
	// Grid layout:
	//   0 1 2
	// 0 S . E
	// 1 . . .
	// Make the top row expensive so the path goes through the bottom.
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 3, 2));
	grid->set_diagonal_mode(AStarGrid2D::DIAGONAL_MODE_NEVER);
	grid->update();

	grid->set_point_weight_scale(Vector2i(1, 0), 100.0);

	TypedArray<Vector2i> path = grid->get_id_path(Vector2i(0, 0), Vector2i(2, 0));
	REQUIRE(path.size() > 0);
	CHECK(Vector2i(path[0]) == Vector2i(0, 0));
	CHECK(Vector2i(path[path.size() - 1]) == Vector2i(2, 0));

	// The path should avoid (1, 0) due to high weight.
	bool goes_through_heavy = false;
	for (int i = 0; i < path.size(); i++) {
		if (Vector2i(path[i]) == Vector2i(1, 0)) {
			goes_through_heavy = true;
		}
	}
	CHECK(goes_through_heavy == false);
}

TEST_CASE("[AStarGrid2D] Heuristic setters") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();

	grid->set_default_compute_heuristic(AStarGrid2D::HEURISTIC_MANHATTAN);
	CHECK(grid->get_default_compute_heuristic() == AStarGrid2D::HEURISTIC_MANHATTAN);

	grid->set_default_estimate_heuristic(AStarGrid2D::HEURISTIC_CHEBYSHEV);
	CHECK(grid->get_default_estimate_heuristic() == AStarGrid2D::HEURISTIC_CHEBYSHEV);

	grid->set_default_compute_heuristic(AStarGrid2D::HEURISTIC_OCTILE);
	CHECK(grid->get_default_compute_heuristic() == AStarGrid2D::HEURISTIC_OCTILE);
}

TEST_CASE("[AStarGrid2D] Cell shape") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();

	grid->set_cell_shape(AStarGrid2D::CELL_SHAPE_ISOMETRIC_RIGHT);
	CHECK(grid->get_cell_shape() == AStarGrid2D::CELL_SHAPE_ISOMETRIC_RIGHT);

	grid->set_cell_shape(AStarGrid2D::CELL_SHAPE_ISOMETRIC_DOWN);
	CHECK(grid->get_cell_shape() == AStarGrid2D::CELL_SHAPE_ISOMETRIC_DOWN);

	SUBCASE("Invalid cell shape is rejected") {
		ERR_PRINT_OFF;
		grid->set_cell_shape((AStarGrid2D::CellShape)99);
		ERR_PRINT_ON;
		CHECK(grid->get_cell_shape() == AStarGrid2D::CELL_SHAPE_ISOMETRIC_DOWN);
	}
}

TEST_CASE("[AStarGrid2D] Jumping enabled") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();

	CHECK(grid->is_jumping_enabled() == false);
	grid->set_jumping_enabled(true);
	CHECK(grid->is_jumping_enabled() == true);
}

TEST_CASE("[AStarGrid2D] Clear resets the grid") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 4, 4));
	grid->update();

	grid->clear();
	CHECK(grid->get_region() == Rect2i());
}

TEST_CASE("[AStarGrid2D] Operations on dirty grid fail") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(0, 0, 4, 4));
	// Grid is dirty (not updated yet).

	ERR_PRINT_OFF;
	CHECK(grid->is_point_solid(Vector2i(0, 0)) == false);
	CHECK(grid->get_point_weight_scale(Vector2i(0, 0)) == doctest::Approx(0.0));
	CHECK(grid->get_point_position(Vector2i(0, 0)) == Vector2());

	TypedArray<Vector2i> path = grid->get_id_path(Vector2i(0, 0), Vector2i(1, 1));
	CHECK(path.size() == 0);
	ERR_PRINT_ON;
}

TEST_CASE("[AStarGrid2D] Negative region offset") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_region(Rect2i(-2, -2, 4, 4));
	grid->update();

	// Points in the negative region should be valid.
	CHECK(grid->is_in_bounds(-2, -2) == true);
	CHECK(grid->is_in_bounds(1, 1) == true);
	CHECK(grid->is_in_bounds(2, 2) == false);

	grid->set_point_solid(Vector2i(-1, -1));
	CHECK(grid->is_point_solid(Vector2i(-1, -1)) == true);

	// Pathfinding should work in negative coordinates.
	grid->set_diagonal_mode(AStarGrid2D::DIAGONAL_MODE_NEVER);
	TypedArray<Vector2i> path = grid->get_id_path(Vector2i(-2, -2), Vector2i(1, 1));
	REQUIRE(path.size() > 0);
	CHECK(Vector2i(path[0]) == Vector2i(-2, -2));
	CHECK(Vector2i(path[path.size() - 1]) == Vector2i(1, 1));
}

TEST_CASE("[AStarGrid2D] get_point_path returns world positions") {
	Ref<AStarGrid2D> grid;
	grid.instantiate();
	grid->set_cell_size(Size2(10, 10));
	grid->set_region(Rect2i(0, 0, 3, 1));
	grid->set_diagonal_mode(AStarGrid2D::DIAGONAL_MODE_NEVER);
	grid->update();

	Vector<Vector2> path = grid->get_point_path(Vector2i(0, 0), Vector2i(2, 0));
	REQUIRE(path.size() == 3);
	CHECK(path[0] == Vector2(0, 0));
	CHECK(path[1] == Vector2(10, 0));
	CHECK(path[2] == Vector2(20, 0));
}

} // namespace TestAStarGrid2D
