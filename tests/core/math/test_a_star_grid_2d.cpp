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

// Checks that `p_path` is a contiguous, walkable chain of cells from `p_from` to `p_to`.
static bool is_valid_path(AStarGrid2D &p_grid, const TypedArray<Vector2i> &p_path, const Vector2i &p_from, const Vector2i &p_to) {
	if (p_path.is_empty()) {
		return false;
	}
	if ((Vector2i)p_path[0] != p_from) {
		return false;
	}
	if ((Vector2i)p_path[p_path.size() - 1] != p_to) {
		return false;
	}
	for (int i = 0; i < p_path.size(); i++) {
		const Vector2i cell = p_path[i];
		if (!p_grid.is_in_boundsv(cell) || p_grid.is_point_solid(cell)) {
			return false;
		}
		if (i > 0) {
			const Vector2i prev = p_path[i - 1];
			const int dx = Math::abs(cell.x - prev.x);
			const int dy = Math::abs(cell.y - prev.y);
			// Each step must move to an 8-connected neighbor (and actually move).
			if (MAX(dx, dy) != 1) {
				return false;
			}
		}
	}
	return true;
}

TEST_CASE("[AStarGrid2D] HPA* returns a valid path on an open grid") {
	AStarGrid2D grid;
	grid.set_region(Rect2i(0, 0, 32, 32));
	grid.update();
	grid.set_hpa_enabled(true);

	const Vector2i from(0, 0);
	const Vector2i to(31, 31);

	TypedArray<Vector2i> path = grid.get_id_path(from, to);
	CHECK(is_valid_path(grid, path, from, to));

	// The hierarchical path may be slightly longer than optimal, but not wildly so.
	AStarGrid2D reference;
	reference.set_region(Rect2i(0, 0, 32, 32));
	reference.update();
	TypedArray<Vector2i> reference_path = reference.get_id_path(from, to);
	CHECK(reference_path.size() > 0);
	// Any valid path cannot be shorter than the optimal one A* finds.
	CHECK(path.size() >= reference_path.size());
	// HPA* trades a little path quality for speed, but should stay in the same ballpark.
	CHECK(path.size() <= reference_path.size() * 2 + 16);
}

TEST_CASE("[AStarGrid2D] HPA* and regular A* agree on reachability with obstacles") {
	// A vertical wall splitting the grid, with a single one-cell gap near the bottom.
	const Rect2i region(0, 0, 30, 30);
	const Vector2i from(2, 2);
	const Vector2i to(27, 2);

	AStarGrid2D reference;
	reference.set_region(region);
	reference.update();
	reference.fill_solid_region(Rect2i(15, 0, 1, 29)); // Leaves (15, 29) walkable.
	TypedArray<Vector2i> reference_path = reference.get_id_path(from, to);

	AStarGrid2D grid;
	grid.set_region(region);
	grid.update();
	grid.set_hpa_enabled(true);
	grid.fill_solid_region(Rect2i(15, 0, 1, 29));
	TypedArray<Vector2i> path = grid.get_id_path(from, to);

	// Both should find a route (around the gap), and the HPA* one must be valid.
	CHECK(reference_path.size() > 0);
	CHECK(path.size() > 0);
	CHECK(is_valid_path(grid, path, from, to));
}

TEST_CASE("[AStarGrid2D] HPA* handles identical start and goal") {
	AStarGrid2D grid;
	grid.set_region(Rect2i(0, 0, 16, 16));
	grid.update();
	grid.set_hpa_enabled(true);

	TypedArray<Vector2i> path = grid.get_id_path(Vector2i(4, 4), Vector2i(4, 4));
	REQUIRE(path.size() == 1);
	CHECK((Vector2i)path[0] == Vector2i(4, 4));
}

TEST_CASE("[AStarGrid2D] HPA* returns no path when the goal is walled off") {
	AStarGrid2D grid;
	grid.set_region(Rect2i(0, 0, 20, 20));
	grid.update();
	grid.set_hpa_enabled(true);

	// Completely enclose the goal cell (10, 10).
	grid.fill_solid_region(Rect2i(9, 9, 3, 3));
	grid.set_point_solid(Vector2i(10, 10), false); // Goal itself is walkable but surrounded.

	TypedArray<Vector2i> path = grid.get_id_path(Vector2i(0, 0), Vector2i(10, 10));
	CHECK(path.is_empty());

	// When a partial path is requested, the fallback to regular A* must still produce one.
	TypedArray<Vector2i> partial = grid.get_id_path(Vector2i(0, 0), Vector2i(10, 10), true);
	CHECK(partial.size() > 0);
}

TEST_CASE("[AStarGrid2D] HPA* rebuilds when solid cells change") {
	AStarGrid2D grid;
	grid.set_region(Rect2i(0, 0, 24, 24));
	grid.update();
	grid.set_hpa_enabled(true);

	const Vector2i from(0, 12);
	const Vector2i to(23, 12);

	TypedArray<Vector2i> first = grid.get_id_path(from, to);
	CHECK(is_valid_path(grid, first, from, to));
	CHECK_FALSE(grid.is_hpa_dirty());

	// Block a full vertical wall (no gap): the path must now be impossible.
	grid.fill_solid_region(Rect2i(12, 0, 1, 24));
	CHECK(grid.is_hpa_dirty()); // Changing solid cells invalidates the abstract graph.

	TypedArray<Vector2i> blocked = grid.get_id_path(from, to);
	CHECK(blocked.is_empty());

	// Re-open a gap and confirm a valid path is found again.
	grid.set_point_solid(Vector2i(12, 12), false);
	TypedArray<Vector2i> reopened = grid.get_id_path(from, to);
	CHECK(is_valid_path(grid, reopened, from, to));
}

TEST_CASE("[AStarGrid2D] Toggling HPA* keeps pathfinding working") {
	AStarGrid2D grid;
	grid.set_region(Rect2i(0, 0, 20, 20));
	grid.update();

	const Vector2i from(0, 0);
	const Vector2i to(19, 19);

	// Regular A* baseline.
	TypedArray<Vector2i> plain = grid.get_id_path(from, to);
	CHECK(is_valid_path(grid, plain, from, to));

	// Enable, then disable HPA*: results must remain valid in both cases.
	grid.set_hpa_enabled(true);
	TypedArray<Vector2i> hpa = grid.get_id_path(from, to);
	CHECK(is_valid_path(grid, hpa, from, to));

	grid.set_hpa_enabled(false);
	CHECK_FALSE(grid.is_hpa_enabled());
	TypedArray<Vector2i> plain_again = grid.get_id_path(from, to);
	CHECK(is_valid_path(grid, plain_again, from, to));
}

TEST_CASE("[AStarGrid2D] HPA* point path matches id path positions") {
	AStarGrid2D grid;
	grid.set_region(Rect2i(0, 0, 24, 24));
	grid.update();
	grid.set_hpa_enabled(true);

	const Vector2i from(1, 1);
	const Vector2i to(22, 20);

	TypedArray<Vector2i> id_path = grid.get_id_path(from, to);
	Vector<Vector2> point_path = grid.get_point_path(from, to);

	REQUIRE(id_path.size() > 0);
	REQUIRE(id_path.size() == point_path.size());

	// The point path should be the world positions of the id path cells.
	for (int i = 0; i < id_path.size(); i++) {
		CHECK(point_path[i].is_equal_approx(grid.get_point_position((Vector2i)id_path[i])));
	}
}

} // namespace TestAStarGrid2D
