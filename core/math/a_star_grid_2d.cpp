/**************************************************************************/
/*  a_star_grid_2d.cpp                                                    */
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

#include "a_star_grid_2d.h"
#include "a_star_grid_2d.compat.inc"

#include "core/object/class_db.h"
#include "core/variant/typed_array.h"

static real_t heuristic_euclidean(const Vector2i &p_from, const Vector2i &p_to) {
	real_t dx = (real_t)Math::abs(p_to.x - p_from.x);
	real_t dy = (real_t)Math::abs(p_to.y - p_from.y);
	return (real_t)Math::sqrt(dx * dx + dy * dy);
}

static real_t heuristic_manhattan(const Vector2i &p_from, const Vector2i &p_to) {
	real_t dx = (real_t)Math::abs(p_to.x - p_from.x);
	real_t dy = (real_t)Math::abs(p_to.y - p_from.y);
	return dx + dy;
}

static real_t heuristic_octile(const Vector2i &p_from, const Vector2i &p_to) {
	real_t dx = (real_t)Math::abs(p_to.x - p_from.x);
	real_t dy = (real_t)Math::abs(p_to.y - p_from.y);
	real_t F = Math::SQRT2 - 1;
	return (dx < dy) ? F * dx + dy : F * dy + dx;
}

static real_t heuristic_chebyshev(const Vector2i &p_from, const Vector2i &p_to) {
	real_t dx = (real_t)Math::abs(p_to.x - p_from.x);
	real_t dy = (real_t)Math::abs(p_to.y - p_from.y);
	return MAX(dx, dy);
}

static real_t (*heuristics[AStarGrid2D::HEURISTIC_MAX])(const Vector2i &, const Vector2i &) = { heuristic_euclidean, heuristic_manhattan, heuristic_octile, heuristic_chebyshev };

void AStarGrid2D::set_region(const Rect2i &p_region) {
	ERR_FAIL_COND(p_region.size.x < 0 || p_region.size.y < 0);
	if (p_region != region) {
		region = p_region;
		dirty = true;
	}
}

Rect2i AStarGrid2D::get_region() const {
	return region;
}

void AStarGrid2D::set_size(const Size2i &p_size) {
	WARN_DEPRECATED_MSG(R"(The "size" property is deprecated, use "region" instead.)");
	ERR_FAIL_COND(p_size.x < 0 || p_size.y < 0);
	if (p_size != region.size) {
		region.size = p_size;
		dirty = true;
	}
}

Size2i AStarGrid2D::get_size() const {
	return region.size;
}

void AStarGrid2D::set_offset(const Vector2 &p_offset) {
	if (!offset.is_equal_approx(p_offset)) {
		offset = p_offset;
		dirty = true;
	}
}

Vector2 AStarGrid2D::get_offset() const {
	return offset;
}

void AStarGrid2D::set_cell_size(const Size2 &p_cell_size) {
	if (!cell_size.is_equal_approx(p_cell_size)) {
		cell_size = p_cell_size;
		dirty = true;
	}
}

void AStarGrid2D::set_hpa_cluster_size(int32_t p_cluster_size) {
	ERR_FAIL_COND(p_cluster_size <= 0);
	if (p_cluster_size != hpa_cluster_size) {
		hpa_cluster_size = p_cluster_size;
		hpa_dirty = true;
	}
}

int32_t AStarGrid2D::get_hpa_cluster_size() const {
	return hpa_cluster_size;
}

Size2 AStarGrid2D::get_cell_size() const {
	return cell_size;
}

void AStarGrid2D::set_cell_shape(CellShape p_cell_shape) {
	if (cell_shape == p_cell_shape) {
		return;
	}

	ERR_FAIL_INDEX(p_cell_shape, CellShape::CELL_SHAPE_MAX);
	cell_shape = p_cell_shape;
	dirty = true;
}

AStarGrid2D::CellShape AStarGrid2D::get_cell_shape() const {
	return cell_shape;
}

void AStarGrid2D::update() {
	if (!dirty) {
		return;
	}

	points.clear();
	solid_mask.clear();

	const int32_t end_x = region.get_end().x;
	const int32_t end_y = region.get_end().y;
	const Vector2 half_cell_size = cell_size / 2;
	for (int32_t x = region.position.x; x < end_x + 2; x++) {
		solid_mask.push_back(true);
	}

	for (int32_t y = region.position.y; y < end_y; y++) {
		LocalVector<Point> line;
		solid_mask.push_back(true);
		for (int32_t x = region.position.x; x < end_x; x++) {
			Vector2 v = offset;
			switch (cell_shape) {
				case CELL_SHAPE_ISOMETRIC_RIGHT:
					v += half_cell_size + Vector2(x + y, y - x) * half_cell_size;
					break;
				case CELL_SHAPE_ISOMETRIC_DOWN:
					v += half_cell_size + Vector2(x - y, x + y) * half_cell_size;
					break;
				case CELL_SHAPE_SQUARE:
					v += Vector2(x, y) * cell_size;
					break;
				default:
					break;
			}
			line.push_back(Point(Vector2i(x, y), v));
			solid_mask.push_back(false);
		}
		solid_mask.push_back(true);
		points.push_back(std::move(line));
	}

	for (int32_t x = region.position.x; x < end_x + 2; x++) {
		solid_mask.push_back(true);
	}

	dirty = false;
	// The grid was rebuilt, so any previously built abstract graph is now invalid.
	hpa_dirty = true;
}

bool AStarGrid2D::is_hpa_enabled() const {
	return hpa_enabled;
}

void AStarGrid2D::set_hpa_enabled(bool p_enabled) {
	if (hpa_enabled != p_enabled) {
		hpa_enabled = p_enabled;
		hpa_dirty = true;

		if (!p_enabled) {
			// Free the abstract graph immediately so disabling HPA* reclaims its memory.
			hpa_clusters.reset();
			hpa_entrances.reset();
			hpa_nodes.reset();
			hpa_cluster_cols = 0;
			hpa_cluster_rows = 0;
		}
	}
}

bool AStarGrid2D::is_in_bounds(int32_t p_x, int32_t p_y) const {
	return region.has_point(Vector2i(p_x, p_y));
}

bool AStarGrid2D::is_in_boundsv(const Vector2i &p_id) const {
	return region.has_point(p_id);
}

bool AStarGrid2D::is_dirty() const {
	return dirty;
}

bool AStarGrid2D::is_hpa_dirty() const {
	return hpa_dirty;
}

void AStarGrid2D::set_jumping_enabled(bool p_enabled) {
	jumping_enabled = p_enabled;
}

bool AStarGrid2D::is_jumping_enabled() const {
	return jumping_enabled;
}

void AStarGrid2D::set_diagonal_mode(DiagonalMode p_diagonal_mode) {
	ERR_FAIL_INDEX((int)p_diagonal_mode, (int)DIAGONAL_MODE_MAX);
	diagonal_mode = p_diagonal_mode;
}

AStarGrid2D::DiagonalMode AStarGrid2D::get_diagonal_mode() const {
	return diagonal_mode;
}

void AStarGrid2D::set_default_compute_heuristic(Heuristic p_heuristic) {
	ERR_FAIL_INDEX((int)p_heuristic, (int)HEURISTIC_MAX);
	default_compute_heuristic = p_heuristic;
}

AStarGrid2D::Heuristic AStarGrid2D::get_default_compute_heuristic() const {
	return default_compute_heuristic;
}

void AStarGrid2D::set_default_estimate_heuristic(Heuristic p_heuristic) {
	ERR_FAIL_INDEX((int)p_heuristic, (int)HEURISTIC_MAX);
	default_estimate_heuristic = p_heuristic;
}

AStarGrid2D::Heuristic AStarGrid2D::get_default_estimate_heuristic() const {
	return default_estimate_heuristic;
}

void AStarGrid2D::set_point_solid(const Vector2i &p_id, bool p_solid) {
	ERR_FAIL_COND_MSG(dirty, "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_MSG(!is_in_boundsv(p_id), vformat("Can't set if point is disabled. Point %s out of bounds %s.", p_id, region));
	if (_get_solid_unchecked(p_id) != p_solid) {
		_set_solid_unchecked(p_id, p_solid);
		// The abstract graph depends on which cells are solid, so it must be rebuilt.
		hpa_dirty = true;
	}
}

bool AStarGrid2D::is_point_solid(const Vector2i &p_id) const {
	ERR_FAIL_COND_V_MSG(dirty, false, "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_id), false, vformat("Can't get if point is disabled. Point %s out of bounds %s.", p_id, region));
	return _get_solid_unchecked(p_id);
}

void AStarGrid2D::set_point_weight_scale(const Vector2i &p_id, real_t p_weight_scale) {
	ERR_FAIL_COND_MSG(dirty, "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_MSG(!is_in_boundsv(p_id), vformat("Can't set point's weight scale. Point %s out of bounds %s.", p_id, region));
	ERR_FAIL_COND_MSG(p_weight_scale < 0.0, vformat("Can't set point's weight scale less than 0.0: %f.", p_weight_scale));
	_get_point_unchecked(p_id)->weight_scale = p_weight_scale;
}

real_t AStarGrid2D::get_point_weight_scale(const Vector2i &p_id) const {
	ERR_FAIL_COND_V_MSG(dirty, 0, "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_id), 0, vformat("Can't get point's weight scale. Point %s out of bounds %s.", p_id, region));
	return _get_point_unchecked(p_id)->weight_scale;
}

void AStarGrid2D::fill_solid_region(const Rect2i &p_region, bool p_solid) {
	ERR_FAIL_COND_MSG(dirty, "Grid is not initialized. Call the update method.");

	const Rect2i safe_region = p_region.intersection(region);
	const int32_t end_x = safe_region.get_end().x;
	const int32_t end_y = safe_region.get_end().y;

	for (int32_t y = safe_region.position.y; y < end_y; y++) {
		for (int32_t x = safe_region.position.x; x < end_x; x++) {
			_set_solid_unchecked(x, y, p_solid);
		}
	}

	if (safe_region.has_area()) {
		// The abstract graph depends on which cells are solid, so it must be rebuilt.
		hpa_dirty = true;
	}
}

void AStarGrid2D::fill_weight_scale_region(const Rect2i &p_region, real_t p_weight_scale) {
	ERR_FAIL_COND_MSG(dirty, "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_MSG(p_weight_scale < 0.0, vformat("Can't set point's weight scale less than 0.0: %f.", p_weight_scale));

	const Rect2i safe_region = p_region.intersection(region);
	const int32_t end_x = safe_region.get_end().x;
	const int32_t end_y = safe_region.get_end().y;

	for (int32_t y = safe_region.position.y; y < end_y; y++) {
		for (int32_t x = safe_region.position.x; x < end_x; x++) {
			_get_point_unchecked(x, y)->weight_scale = p_weight_scale;
		}
	}
}

AStarGrid2D::Point *AStarGrid2D::_jump(Point *p_from, Point *p_to) {
	int32_t from_x = p_from->id.x;
	int32_t from_y = p_from->id.y;

	int32_t to_x = p_to->id.x;
	int32_t to_y = p_to->id.y;

	int32_t dx = to_x - from_x;
	int32_t dy = to_y - from_y;

	if (diagonal_mode == DIAGONAL_MODE_ALWAYS || diagonal_mode == DIAGONAL_MODE_AT_LEAST_ONE_WALKABLE) {
		if (dx == 0 || dy == 0) {
			return _forced_successor(to_x, to_y, dx, dy);
		}

		while (_is_walkable(to_x, to_y) && (diagonal_mode == DIAGONAL_MODE_ALWAYS || _is_walkable(to_x, to_y - dy) || _is_walkable(to_x - dx, to_y))) {
			if (end->id.x == to_x && end->id.y == to_y) {
				return end;
			}

			if ((_is_walkable(to_x - dx, to_y + dy) && !_is_walkable(to_x - dx, to_y)) || (_is_walkable(to_x + dx, to_y - dy) && !_is_walkable(to_x, to_y - dy))) {
				return _get_point_unchecked(to_x, to_y);
			}

			if (_forced_successor(to_x + dx, to_y, dx, 0) != nullptr || _forced_successor(to_x, to_y + dy, 0, dy) != nullptr) {
				return _get_point_unchecked(to_x, to_y);
			}

			to_x += dx;
			to_y += dy;
		}

	} else if (diagonal_mode == DIAGONAL_MODE_ONLY_IF_NO_OBSTACLES) {
		if (dx == 0 || dy == 0) {
			return _forced_successor(from_x, from_y, dx, dy, true);
		}

		while (_is_walkable(to_x, to_y) && _is_walkable(to_x, to_y - dy) && _is_walkable(to_x - dx, to_y)) {
			if (end->id.x == to_x && end->id.y == to_y) {
				return end;
			}

			if ((_is_walkable(to_x + dx, to_y + dy) && !_is_walkable(to_x, to_y + dy)) || !_is_walkable(to_x + dx, to_y)) {
				return _get_point_unchecked(to_x, to_y);
			}

			if (_forced_successor(to_x, to_y, dx, 0) != nullptr || _forced_successor(to_x, to_y, 0, dy) != nullptr) {
				return _get_point_unchecked(to_x, to_y);
			}

			to_x += dx;
			to_y += dy;
		}

	} else { // DIAGONAL_MODE_NEVER
		if (dy == 0) {
			return _forced_successor(from_x, from_y, dx, 0, true);
		}

		while (_is_walkable(to_x, to_y)) {
			if (end->id.x == to_x && end->id.y == to_y) {
				return end;
			}

			if ((_is_walkable(to_x - 1, to_y) && !_is_walkable(to_x - 1, to_y - dy)) || (_is_walkable(to_x + 1, to_y) && !_is_walkable(to_x + 1, to_y - dy))) {
				return _get_point_unchecked(to_x, to_y);
			}

			if (_forced_successor(to_x, to_y, 1, 0, true) != nullptr || _forced_successor(to_x, to_y, -1, 0, true) != nullptr) {
				return _get_point_unchecked(to_x, to_y);
			}

			to_y += dy;
		}
	}

	return nullptr;
}

AStarGrid2D::Point *AStarGrid2D::_forced_successor(int32_t p_x, int32_t p_y, int32_t p_dx, int32_t p_dy, bool p_inclusive) {
	// Remembering previous results can improve performance.
	bool l_prev = false, r_prev = false, l = false, r = false;

	int32_t o_x = p_x, o_y = p_y;
	if (p_inclusive) {
		o_x += p_dx;
		o_y += p_dy;
	}

	int32_t l_x = p_x - p_dy, l_y = p_y - p_dx;
	int32_t r_x = p_x + p_dy, r_y = p_y + p_dx;

	while (_is_walkable(o_x, o_y)) {
		if (end->id.x == o_x && end->id.y == o_y) {
			return end;
		}

		l_prev = l || _is_walkable(l_x, l_y);
		r_prev = r || _is_walkable(r_x, r_y);

		l_x += p_dx;
		l_y += p_dy;
		r_x += p_dx;
		r_y += p_dy;

		l = _is_walkable(l_x, l_y);
		r = _is_walkable(r_x, r_y);

		if ((l && !l_prev) || (r && !r_prev)) {
			return _get_point_unchecked(o_x, o_y);
		}

		o_x += p_dx;
		o_y += p_dy;
	}
	return nullptr;
}

void AStarGrid2D::_get_nbors(Point *p_point, LocalVector<Point *> &r_nbors) {
	bool ts0 = false, td0 = false,
		 ts1 = false, td1 = false,
		 ts2 = false, td2 = false,
		 ts3 = false, td3 = false;

	Point *left = nullptr;
	Point *right = nullptr;
	Point *top = nullptr;
	Point *bottom = nullptr;

	Point *top_left = nullptr;
	Point *top_right = nullptr;
	Point *bottom_left = nullptr;
	Point *bottom_right = nullptr;

	{
		bool has_left = false;
		bool has_right = false;

		if (p_point->id.x - 1 >= region.position.x) {
			left = _get_point_unchecked(p_point->id.x - 1, p_point->id.y);
			has_left = true;
		}
		if (p_point->id.x + 1 < region.position.x + region.size.width) {
			right = _get_point_unchecked(p_point->id.x + 1, p_point->id.y);
			has_right = true;
		}
		if (p_point->id.y - 1 >= region.position.y) {
			top = _get_point_unchecked(p_point->id.x, p_point->id.y - 1);
			if (has_left) {
				top_left = _get_point_unchecked(p_point->id.x - 1, p_point->id.y - 1);
			}
			if (has_right) {
				top_right = _get_point_unchecked(p_point->id.x + 1, p_point->id.y - 1);
			}
		}
		if (p_point->id.y + 1 < region.position.y + region.size.height) {
			bottom = _get_point_unchecked(p_point->id.x, p_point->id.y + 1);
			if (has_left) {
				bottom_left = _get_point_unchecked(p_point->id.x - 1, p_point->id.y + 1);
			}
			if (has_right) {
				bottom_right = _get_point_unchecked(p_point->id.x + 1, p_point->id.y + 1);
			}
		}
	}

	if (top && !_get_solid_unchecked(top->id)) {
		r_nbors.push_back(top);
		ts0 = true;
	}
	if (right && !_get_solid_unchecked(right->id)) {
		r_nbors.push_back(right);
		ts1 = true;
	}
	if (bottom && !_get_solid_unchecked(bottom->id)) {
		r_nbors.push_back(bottom);
		ts2 = true;
	}
	if (left && !_get_solid_unchecked(left->id)) {
		r_nbors.push_back(left);
		ts3 = true;
	}

	switch (diagonal_mode) {
		case DIAGONAL_MODE_ALWAYS: {
			td0 = true;
			td1 = true;
			td2 = true;
			td3 = true;
		} break;
		case DIAGONAL_MODE_NEVER: {
		} break;
		case DIAGONAL_MODE_AT_LEAST_ONE_WALKABLE: {
			td0 = ts3 || ts0;
			td1 = ts0 || ts1;
			td2 = ts1 || ts2;
			td3 = ts2 || ts3;
		} break;
		case DIAGONAL_MODE_ONLY_IF_NO_OBSTACLES: {
			td0 = ts3 && ts0;
			td1 = ts0 && ts1;
			td2 = ts1 && ts2;
			td3 = ts2 && ts3;
		} break;
		default:
			break;
	}

	if (td0 && (top_left && !_get_solid_unchecked(top_left->id))) {
		r_nbors.push_back(top_left);
	}
	if (td1 && (top_right && !_get_solid_unchecked(top_right->id))) {
		r_nbors.push_back(top_right);
	}
	if (td2 && (bottom_right && !_get_solid_unchecked(bottom_right->id))) {
		r_nbors.push_back(bottom_right);
	}
	if (td3 && (bottom_left && !_get_solid_unchecked(bottom_left->id))) {
		r_nbors.push_back(bottom_left);
	}
}

bool AStarGrid2D::_solve(Point *p_begin_point, Point *p_end_point, bool p_allow_partial_path) {
	last_closest_point = nullptr;
	pass++;

	if (_get_solid_unchecked(p_begin_point->id)) {
		return false;
	}
	if (p_begin_point == p_end_point) {
		return true;
	}
	if (_get_solid_unchecked(p_end_point->id) && !p_allow_partial_path) {
		return false;
	}

	bool found_route = false;

	LocalVector<Point *> open_list;
	SortArray<Point *, SortPoints> sorter;
	LocalVector<Point *> nbors;

	p_begin_point->g_score = 0;
	p_begin_point->f_score = _estimate_cost(p_begin_point->id, p_end_point->id);
	p_begin_point->abs_g_score = 0;
	p_begin_point->abs_f_score = _estimate_cost(p_begin_point->id, p_end_point->id);
	open_list.push_back(p_begin_point);
	end = p_end_point;

	while (!open_list.is_empty()) {
		Point *p = open_list[0]; // The currently processed point.

		// Find point closer to end_point, or same distance to end_point but closer to begin_point.
		if (last_closest_point == nullptr || last_closest_point->abs_f_score > p->abs_f_score || (last_closest_point->abs_f_score >= p->abs_f_score && last_closest_point->abs_g_score > p->abs_g_score)) {
			last_closest_point = p;
		}

		if (p == p_end_point) {
			found_route = true;
			break;
		}

		sorter.pop_heap(0, open_list.size(), open_list.ptr()); // Remove the current point from the open list.
		open_list.remove_at(open_list.size() - 1);
		p->closed_pass = pass; // Mark the point as closed.

		nbors.clear();
		_get_nbors(p, nbors);

		for (Point *e : nbors) {
			real_t weight_scale = 1.0;

			if (jumping_enabled) {
				// TODO: Make it works with weight_scale.
				e = _jump(p, e);
				if (!e || e->closed_pass == pass) {
					continue;
				}
			} else {
				if (_get_solid_unchecked(e->id) || e->closed_pass == pass) {
					continue;
				}
				weight_scale = e->weight_scale;
			}

			real_t tentative_g_score = p->g_score + _compute_cost(p->id, e->id) * weight_scale;
			bool new_point = false;

			if (e->open_pass != pass) { // The point wasn't inside the open list.
				e->open_pass = pass;
				open_list.push_back(e);
				new_point = true;
			} else if (tentative_g_score >= e->g_score) { // The new path is worse than the previous.
				continue;
			}

			e->prev_point = p;
			e->g_score = tentative_g_score;
			e->f_score = e->g_score + _estimate_cost(e->id, p_end_point->id);

			e->abs_g_score = tentative_g_score;
			e->abs_f_score = e->f_score - e->g_score;

			if (new_point) { // The position of the new points is already known.
				sorter.push_heap(0, open_list.size() - 1, 0, e, open_list.ptr());
			} else {
				sorter.push_heap(0, open_list.find(e), 0, e, open_list.ptr());
			}
		}
	}

	return found_route;
}

real_t AStarGrid2D::_estimate_cost(const Vector2i &p_from_id, const Vector2i &p_end_id) {
	real_t scost;
	if (GDVIRTUAL_CALL(_estimate_cost, p_from_id, p_end_id, scost)) {
		return scost;
	}
	return heuristics[default_estimate_heuristic](p_from_id, p_end_id);
}

real_t AStarGrid2D::_compute_cost(const Vector2i &p_from_id, const Vector2i &p_to_id) {
	real_t scost;
	if (GDVIRTUAL_CALL(_compute_cost, p_from_id, p_to_id, scost)) {
		return scost;
	}
	return heuristics[default_compute_heuristic](p_from_id, p_to_id);
}

void AStarGrid2D::clear() {
	points.clear();
	region = Rect2i();

	hpa_clusters.reset();
	hpa_entrances.reset();
	hpa_nodes.reset();
	hpa_cluster_cols = 0;
	hpa_cluster_rows = 0;
	hpa_dirty = true;
}

Vector2 AStarGrid2D::get_point_position(const Vector2i &p_id) const {
	ERR_FAIL_COND_V_MSG(dirty, Vector2(), "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_id), Vector2(), vformat("Can't get point's position. Point %s out of bounds %s.", p_id, region));
	return _get_point_unchecked(p_id)->pos;
}

TypedArray<Dictionary> AStarGrid2D::get_point_data_in_region(const Rect2i &p_region) const {
	ERR_FAIL_COND_V_MSG(dirty, TypedArray<Dictionary>(), "Grid is not initialized. Call the update method.");
	const Rect2i inter_region = region.intersection(p_region);

	const int32_t start_x = inter_region.position.x - region.position.x;
	const int32_t start_y = inter_region.position.y - region.position.y;
	const int32_t end_x = inter_region.get_end().x - region.position.x;
	const int32_t end_y = inter_region.get_end().y - region.position.y;

	TypedArray<Dictionary> data;

	for (int32_t y = start_y; y < end_y; y++) {
		for (int32_t x = start_x; x < end_x; x++) {
			const Point &p = points[y][x];

			Dictionary dict;
			dict["id"] = p.id;
			dict["position"] = p.pos;
			dict["solid"] = _get_solid_unchecked(p.id);
			dict["weight_scale"] = p.weight_scale;
			data.push_back(dict);
		}
	}

	return data;
}

Vector<Vector2> AStarGrid2D::get_point_path(const Vector2i &p_from_id, const Vector2i &p_to_id, bool p_allow_partial_path) {
	ERR_FAIL_COND_V_MSG(dirty, Vector<Vector2>(), "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_from_id), Vector<Vector2>(), vformat("Can't get id path. Point %s out of bounds %s.", p_from_id, region));
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_to_id), Vector<Vector2>(), vformat("Can't get id path. Point %s out of bounds %s.", p_to_id, region));

	if (hpa_enabled) {
		if (hpa_dirty) {
			update_hpa();
		}
		LocalVector<Vector2i> cells;
		const HPAPathResult result = _hpa_get_cell_path(p_from_id, p_to_id, cells);
		if (result == HPA_PATH_RESULT_FOUND) {
			Vector<Vector2> path;
			path.resize(cells.size());
			Vector2 *w = path.ptrw();
			for (uint32_t i = 0; i < cells.size(); i++) {
				w[i] = _get_point_unchecked(cells[i])->pos;
			}
			return path;
		}
		if (result == HPA_PATH_RESULT_UNREACHABLE && !p_allow_partial_path) {
			// The abstract graph encodes all inter-cluster connectivity, so an unreachable
			// goal can be reported without running a full search on the cell grid.
			return Vector<Vector2>();
		}
		// The abstract path could not be refined (or a partial path was requested);
		// fall back to regular A*.
	}

	Point *begin_point = _get_point(p_from_id.x, p_from_id.y);
	Point *end_point = _get_point(p_to_id.x, p_to_id.y);

	bool found_route = _solve(begin_point, end_point, p_allow_partial_path);
	if (!found_route) {
		if (!p_allow_partial_path || last_closest_point == nullptr) {
			return Vector<Vector2>();
		}

		// Use closest point instead.
		end_point = last_closest_point;
	}

	Point *p = end_point;
	int32_t pc = 1;
	while (p != begin_point) {
		pc++;
		p = p->prev_point;
	}

	Vector<Vector2> path;
	path.resize(pc);

	{
		Vector2 *w = path.ptrw();

		p = end_point;
		int32_t idx = pc - 1;
		while (p != begin_point) {
			w[idx--] = p->pos;
			p = p->prev_point;
		}

		w[0] = p->pos;
	}

	return path;
}

TypedArray<Vector2i> AStarGrid2D::get_id_path(const Vector2i &p_from_id, const Vector2i &p_to_id, bool p_allow_partial_path) {
	ERR_FAIL_COND_V_MSG(dirty, TypedArray<Vector2i>(), "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_from_id), TypedArray<Vector2i>(), vformat("Can't get id path. Point %s out of bounds %s.", p_from_id, region));
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_to_id), TypedArray<Vector2i>(), vformat("Can't get id path. Point %s out of bounds %s.", p_to_id, region));

	if (hpa_enabled) {
		if (hpa_dirty) {
			update_hpa();
		}
		LocalVector<Vector2i> cells;
		const HPAPathResult result = _hpa_get_cell_path(p_from_id, p_to_id, cells);
		if (result == HPA_PATH_RESULT_FOUND) {
			TypedArray<Vector2i> path;
			path.resize(cells.size());
			for (uint32_t i = 0; i < cells.size(); i++) {
				path[i] = cells[i];
			}
			return path;
		}
		if (result == HPA_PATH_RESULT_UNREACHABLE && !p_allow_partial_path) {
			// The abstract graph encodes all inter-cluster connectivity, so an unreachable
			// goal can be reported without running a full search on the cell grid.
			return TypedArray<Vector2i>();
		}
		// The abstract path could not be refined (or a partial path was requested);
		// fall back to regular A*.
	}

	Point *begin_point = _get_point(p_from_id.x, p_from_id.y);
	Point *end_point = _get_point(p_to_id.x, p_to_id.y);

	bool found_route = _solve(begin_point, end_point, p_allow_partial_path);
	if (!found_route) {
		if (!p_allow_partial_path || last_closest_point == nullptr) {
			return TypedArray<Vector2i>();
		}

		// Use closest point instead.
		end_point = last_closest_point;
	}

	Point *p = end_point;
	int32_t pc = 1;
	while (p != begin_point) {
		pc++;
		p = p->prev_point;
	}

	TypedArray<Vector2i> path;
	path.resize(pc);

	{
		p = end_point;
		int32_t idx = pc - 1;
		while (p != begin_point) {
			path[idx--] = p->id;
			p = p->prev_point;
		}

		path[0] = p->id;
	}

	return path;
}

AStarGrid2D::ClusterAdjacency AStarGrid2D::_clusters_are_adjacent(const Cluster *p_c1, const Cluster *p_c2) const {
	const int32_t c1_left = p_c1->cluster_pos.x;
	const int32_t c1_bottom = p_c1->cluster_pos.y;
	const int32_t c1_right = c1_left + p_c1->cluster_width;
	const int32_t c1_top = c1_bottom + p_c1->cluster_height;

	const int32_t c2_left = p_c2->cluster_pos.x;
	const int32_t c2_bottom = p_c2->cluster_pos.y;
	const int32_t c2_right = c2_left + p_c2->cluster_width;
	const int32_t c2_top = c2_bottom + p_c2->cluster_height;

	const bool overlap_x = c1_left == c2_left;
	const bool overlap_y = c1_bottom == c2_bottom;

	if (overlap_x) {
		if (c1_top == c2_bottom || c2_top == c1_bottom) {
			return CLUSTER_ADJACENCY_VERTICAL;
		}
	}

	if (overlap_y) {
		if (c1_right == c2_left || c2_right == c1_left) {
			return CLUSTER_ADJACENCY_LATERAL;
		}
	}

	if (diagonal_mode != DIAGONAL_MODE_NEVER) {
		if ((c1_right == c2_left || c2_right == c1_left) && (c1_bottom == c2_top || c2_bottom == c1_top)) {
			return CLUSTER_ADJACENCY_DIAGONAL;
		}
	}

	return CLUSTER_ADJACENCY_NONE;
}

void AStarGrid2D::_build_clusters() {
	hpa_clusters.clear();

	const int32_t start_x = region.position.x;
	const int32_t start_y = region.position.y;
	const int32_t end_x = region.get_end().x;
	const int32_t end_y = region.get_end().y;

	// Cache the cluster grid dimensions so a cell can be mapped to its cluster in O(1).
	// Clusters are stored in column-major order (see the loops below), so the index of a
	// cell's cluster is `column * hpa_cluster_rows + row`.
	hpa_cluster_cols = (end_x - start_x + hpa_cluster_size - 1) / hpa_cluster_size;
	hpa_cluster_rows = (end_y - start_y + hpa_cluster_size - 1) / hpa_cluster_size;

	Cluster c;

	// We create the clusters in column-major order.
	for (int32_t x = start_x; x < end_x; x += hpa_cluster_size) {
		// Check if the cluster would extend beyond the map.
		if (x + hpa_cluster_size > end_x) {
			c.cluster_width = end_x - x;
		} else {
			c.cluster_width = hpa_cluster_size;
		}

		for (int32_t y = start_y; y < end_y; y += hpa_cluster_size) {
			// Check if the cluster would extend beyond the map.
			if (y + hpa_cluster_size > end_y) {
				c.cluster_height = end_y - y;
			} else {
				c.cluster_height = hpa_cluster_size;
			}

			c.cluster_pos = Vector2i(x, y);
			hpa_clusters.push_back(c);
		}
	}
}

void AStarGrid2D::_build_entrances() {
	hpa_entrances.clear();

	const uint32_t num_clusters = hpa_clusters.size();

	// TODO: If more levels of hierarchy are added, it would be wise to store multiple entrances between clusters.
	for (uint32_t i = 0; i < num_clusters; i++) {
		for (uint32_t k = i + 1; k < num_clusters; k++) {
			const ClusterAdjacency adjacent_dir = _clusters_are_adjacent(&hpa_clusters[i], &hpa_clusters[k]);

			if (adjacent_dir != CLUSTER_ADJACENCY_NONE) {
				_create_entrances(i, k, adjacent_dir);
			}
		}
	}
}

void AStarGrid2D::_add_entrance(int32_t p_c1_idx, int32_t p_c2_idx, const Vector2i &p_c1_pos, const Vector2i &p_c2_pos, int32_t p_width, int32_t p_height, ClusterAdjacency p_dir) {
	hpa_entrances.push_back(Entrance{ p_c1_idx, p_c2_idx, p_c1_pos, p_c2_pos, p_width, p_height, p_dir });
}

void AStarGrid2D::_create_entrances(int32_t p_c1_idx, int32_t p_c2_idx, ClusterAdjacency p_adjacent_dir) {
	ERR_FAIL_INDEX(p_c1_idx, (int32_t)hpa_clusters.size());
	ERR_FAIL_INDEX(p_c2_idx, (int32_t)hpa_clusters.size());
	ERR_FAIL_COND_MSG(dirty, "Grid is not initialized. Call the update method.");

	const Cluster *c1 = &hpa_clusters[p_c1_idx];
	const Cluster *c2 = &hpa_clusters[p_c2_idx];

	switch (p_adjacent_dir) {
		case CLUSTER_ADJACENCY_LATERAL: {
			const bool c1_is_left = c1->cluster_pos.x <= c2->cluster_pos.x;
			const int32_t c1_x = c1_is_left ? (c1->cluster_pos.x + c1->cluster_width - 1) : c1->cluster_pos.x;
			const int32_t c2_x = c1_is_left ? c2->cluster_pos.x : (c2->cluster_pos.x + c2->cluster_width - 1);
			const int32_t start_y = MAX(c1->cluster_pos.y, c2->cluster_pos.y);
			const int32_t end_y = MIN(c1->cluster_pos.y + c1->cluster_height, c2->cluster_pos.y + c2->cluster_height);

			int32_t entrance_start = -1;
			for (int32_t y = start_y; y < end_y; y++) {
				const bool walkable = !is_point_solid(Vector2i(c1_x, y)) && !is_point_solid(Vector2i(c2_x, y));

				if (walkable) {
					if (entrance_start == -1) {
						entrance_start = y;
					}
				} else if (entrance_start != -1) {
					_add_entrance(p_c1_idx, p_c2_idx, Vector2i(c1_x, entrance_start), Vector2i(c2_x, entrance_start), 1, y - entrance_start, p_adjacent_dir);
					entrance_start = -1;
				}
			}

			if (entrance_start != -1) {
				_add_entrance(p_c1_idx, p_c2_idx, Vector2i(c1_x, entrance_start), Vector2i(c2_x, entrance_start), 1, end_y - entrance_start, p_adjacent_dir);
			}
		} break;

		case CLUSTER_ADJACENCY_VERTICAL: {
			const bool c1_is_bottom = c1->cluster_pos.y <= c2->cluster_pos.y;
			const int32_t c1_y = c1_is_bottom ? (c1->cluster_pos.y + c1->cluster_height - 1) : c1->cluster_pos.y;
			const int32_t c2_y = c1_is_bottom ? c2->cluster_pos.y : (c2->cluster_pos.y + c2->cluster_height - 1);
			const int32_t start_x = MAX(c1->cluster_pos.x, c2->cluster_pos.x);
			const int32_t end_x = MIN(c1->cluster_pos.x + c1->cluster_width, c2->cluster_pos.x + c2->cluster_width);

			int32_t entrance_start = -1;
			for (int32_t x = start_x; x < end_x; x++) {
				const bool walkable = !is_point_solid(Vector2i(x, c1_y)) && !is_point_solid(Vector2i(x, c2_y));

				if (walkable) {
					if (entrance_start == -1) {
						entrance_start = x;
					}
				} else if (entrance_start != -1) {
					_add_entrance(p_c1_idx, p_c2_idx, Vector2i(entrance_start, c1_y), Vector2i(entrance_start, c2_y), x - entrance_start, 1, p_adjacent_dir);
					entrance_start = -1;
				}
			}

			if (entrance_start != -1) {
				_add_entrance(p_c1_idx, p_c2_idx, Vector2i(entrance_start, c1_y), Vector2i(entrance_start, c2_y), end_x - entrance_start, 1, p_adjacent_dir);
			}
		} break;

		case CLUSTER_ADJACENCY_DIAGONAL: {
			Vector2i c1_pos;
			Vector2i c2_pos;

			if (c1->cluster_pos.x + c1->cluster_width == c2->cluster_pos.x) {
				c1_pos.x = c1->cluster_pos.x + c1->cluster_width - 1;
				c2_pos.x = c2->cluster_pos.x;
			} else {
				c1_pos.x = c1->cluster_pos.x;
				c2_pos.x = c2->cluster_pos.x + c2->cluster_width - 1;
			}

			if (c1->cluster_pos.y + c1->cluster_height == c2->cluster_pos.y) {
				c1_pos.y = c1->cluster_pos.y + c1->cluster_height - 1;
				c2_pos.y = c2->cluster_pos.y;
			} else {
				c1_pos.y = c1->cluster_pos.y;
				c2_pos.y = c2->cluster_pos.y + c2->cluster_height - 1;
			}

			if (!is_point_solid(c1_pos) && !is_point_solid(c2_pos)) {
				_add_entrance(p_c1_idx, p_c2_idx, c1_pos, c2_pos, 1, 1, p_adjacent_dir);
			}
		} break;

		default:
			break;
	}
}

void AStarGrid2D::_create_hpa_nodes() {
	const int32_t NODE_MIN_DISTANCE = 6;

	hpa_nodes.clear();

	for (uint32_t i = 0; i < hpa_entrances.size(); i++) {
		const Entrance &entrance = hpa_entrances[i];
		const int32_t c1_idx = entrance.cluster_1_idx;
		const int32_t c2_idx = entrance.cluster_2_idx;

		switch (entrance.dir) {
			case CLUSTER_ADJACENCY_LATERAL: {
				// The entrance is a vertical strip, so we place transition nodes along the y axis.
				const int32_t end_y = entrance.c1_pos.y + entrance.height;

				for (int32_t y = entrance.c1_pos.y; y < end_y - 1; y += NODE_MIN_DISTANCE) {
					_build_node_and_inter_edges(Vector2i(entrance.c1_pos.x, y), c1_idx, Vector2i(entrance.c2_pos.x, y), c2_idx);
				}

				// Always place a node at the far end of the entrance.
				const int32_t last_y = end_y - 1;
				_build_node_and_inter_edges(Vector2i(entrance.c1_pos.x, last_y), c1_idx, Vector2i(entrance.c2_pos.x, last_y), c2_idx);
			} break;

			case CLUSTER_ADJACENCY_VERTICAL: {
				// The entrance is a horizontal strip, so we place transition nodes along the x axis.
				const int32_t end_x = entrance.c1_pos.x + entrance.width;

				for (int32_t x = entrance.c1_pos.x; x < end_x - 1; x += NODE_MIN_DISTANCE) {
					_build_node_and_inter_edges(Vector2i(x, entrance.c1_pos.y), c1_idx, Vector2i(x, entrance.c2_pos.y), c2_idx);
				}

				// Always place a node at the far end of the entrance.
				const int32_t last_x = end_x - 1;
				_build_node_and_inter_edges(Vector2i(last_x, entrance.c1_pos.y), c1_idx, Vector2i(last_x, entrance.c2_pos.y), c2_idx);
			} break;

			case CLUSTER_ADJACENCY_DIAGONAL: {
				_build_node_and_inter_edges(entrance.c1_pos, c1_idx, entrance.c2_pos, c2_idx);
			} break;

			default:
				break;
		}
	}
}

void AStarGrid2D::_build_node_and_inter_edges(const Vector2i &p_pos1, int32_t p_c1_idx, const Vector2i &p_pos2, int32_t p_c2_idx) {
	const int32_t n1_idx = hpa_nodes.size();
	const int32_t n2_idx = n1_idx + 1;
	const real_t cost = _compute_cost(p_pos1, p_pos2);

	HierNode n1;
	n1.id = n1_idx;
	n1.pos = p_pos1;
	n1.cluster_idx = p_c1_idx;

	HierNode n2;
	n2.id = n2_idx;
	n2.pos = p_pos2;
	n2.cluster_idx = p_c2_idx;

	n1.edges.push_back(HierEdge{ n2_idx, cost, true });
	n2.edges.push_back(HierEdge{ n1_idx, cost, true });

	hpa_nodes.push_back(n1);
	hpa_nodes.push_back(n2);

	hpa_clusters[p_c1_idx].node_ids.push_back(n1_idx);
	hpa_clusters[p_c2_idx].node_ids.push_back(n2_idx);
}

void AStarGrid2D::_build_intra_edges() {
	for (uint32_t i = 0; i < hpa_clusters.size(); i++) {
		Cluster &cluster = hpa_clusters[i];

		if (cluster.node_ids.size() < 2) {
			continue;
		}

		for (uint32_t j = 0; j < cluster.node_ids.size(); j++) {
			for (uint32_t k = j + 1; k < cluster.node_ids.size(); k++) {
				const int32_t n1_idx = cluster.node_ids[j];
				const int32_t n2_idx = cluster.node_ids[k];

				HierNode &n1 = hpa_nodes[n1_idx];
				HierNode &n2 = hpa_nodes[n2_idx];

				const real_t path_cost = _solve_bounded(n1.pos, n2.pos, _cluster_rect(&cluster));

				if (path_cost >= 0.0) {
					n1.edges.push_back(HierEdge{ n2_idx, path_cost, false });
					n2.edges.push_back(HierEdge{ n1_idx, path_cost, false });
				}
			}
		}
	}
}

int32_t AStarGrid2D::_get_cluster_index(const Vector2i &p_pos) const {
	if (hpa_cluster_rows <= 0 || !region.has_point(p_pos)) {
		return -1;
	}
	const int32_t col = (p_pos.x - region.position.x) / hpa_cluster_size;
	const int32_t row = (p_pos.y - region.position.y) / hpa_cluster_size;
	const int32_t idx = col * hpa_cluster_rows + row;
	if (idx < 0 || idx >= (int32_t)hpa_clusters.size()) {
		return -1;
	}
	return idx;
}

// Bounded A* over the cell grid: expansion is restricted to `p_bounds`. Returns the path
// cost, or -1.0 if no route exists. When `r_path` is given it is filled with the cells from
// `p_from` to `p_to` (inclusive). Used both to weight intra-cluster edges and to refine the
// abstract path into concrete cells.
real_t AStarGrid2D::_solve_bounded(const Vector2i &p_from, const Vector2i &p_to, const Rect2i &p_bounds, LocalVector<Vector2i> *r_path) {
	pass++;

	Point *begin_point = _get_point_unchecked(p_from);
	Point *end_point = _get_point_unchecked(p_to);

	if (_get_solid_unchecked(begin_point->id) || _get_solid_unchecked(end_point->id)) {
		return -1.0;
	}

	begin_point->g_score = 0;
	begin_point->f_score = _estimate_cost(begin_point->id, end_point->id);
	begin_point->open_pass = pass;
	begin_point->prev_point = nullptr;

	LocalVector<Point *> open_list;
	SortArray<Point *, SortPoints> sorter;
	LocalVector<Point *> nbors;

	open_list.push_back(begin_point);

	bool found = false;
	while (!open_list.is_empty()) {
		Point *p = open_list[0];

		if (p == end_point) {
			found = true;
			break;
		}

		sorter.pop_heap(0, open_list.size(), open_list.ptr());
		open_list.remove_at(open_list.size() - 1);
		p->closed_pass = pass;

		nbors.clear();
		_get_nbors(p, nbors);

		for (Point *e : nbors) {
			// Restrict the search to the bounded region.
			if (!p_bounds.has_point(e->id)) {
				continue;
			}
			if (e->closed_pass == pass || _get_solid_unchecked(e->id)) {
				continue;
			}

			const real_t tentative_g_score = p->g_score + _compute_cost(p->id, e->id) * e->weight_scale;
			bool new_point = false;

			if (e->open_pass != pass) {
				e->open_pass = pass;
				open_list.push_back(e);
				new_point = true;
			} else if (tentative_g_score >= e->g_score) {
				continue;
			}

			e->prev_point = p;
			e->g_score = tentative_g_score;
			e->f_score = tentative_g_score + _estimate_cost(e->id, end_point->id);

			if (new_point) {
				sorter.push_heap(0, open_list.size() - 1, 0, e, open_list.ptr());
			} else {
				sorter.push_heap(0, open_list.find(e), 0, e, open_list.ptr());
			}
		}
	}

	if (!found) {
		return -1.0;
	}

	if (r_path) {
		r_path->clear();
		for (Point *p = end_point; p != nullptr; p = p->prev_point) {
			r_path->push_back(p->id);
		}
		const uint32_t count = r_path->size();
		for (uint32_t i = 0; i < count / 2; i++) {
			SWAP((*r_path)[i], (*r_path)[count - 1 - i]);
		}
	}

	return end_point->g_score;
}

// A* over the abstract graph. Operates on node indices and uses each node's edge list as the
// adjacency. Fills `r_node_path` with the sequence of node positions from start to goal.
bool AStarGrid2D::_abstract_astar(int32_t p_start_idx, int32_t p_goal_idx, LocalVector<Vector2i> &r_node_path) {
	hpa_pass++;

	const Vector2i goal_pos = hpa_nodes[p_goal_idx].pos;

	hpa_nodes[p_start_idx].g_score = 0;
	hpa_nodes[p_start_idx].f_score = _estimate_cost(hpa_nodes[p_start_idx].pos, goal_pos);
	hpa_nodes[p_start_idx].prev_node_idx = -1;
	hpa_nodes[p_start_idx].open_pass = hpa_pass;

	LocalVector<int32_t> open;
	open.push_back(p_start_idx);

	while (!open.is_empty()) {
		// Pick the open node with the lowest f_score (linear scan; the abstract graph is small).
		uint32_t best_i = 0;
		for (uint32_t i = 1; i < open.size(); i++) {
			if (hpa_nodes[open[i]].f_score < hpa_nodes[open[best_i]].f_score) {
				best_i = i;
			}
		}

		const int32_t curr = open[best_i];
		if (curr == p_goal_idx) {
			for (int32_t n = p_goal_idx; n != -1; n = hpa_nodes[n].prev_node_idx) {
				r_node_path.push_back(hpa_nodes[n].pos);
			}
			const uint32_t count = r_node_path.size();
			for (uint32_t i = 0; i < count / 2; i++) {
				SWAP(r_node_path[i], r_node_path[count - 1 - i]);
			}
			return true;
		}

		open.remove_at(best_i);
		hpa_nodes[curr].closed_pass = hpa_pass;

		const LocalVector<HierEdge> &edges = hpa_nodes[curr].edges;
		for (uint32_t i = 0; i < edges.size(); i++) {
			const int32_t nb = edges[i].to_node_idx;
			if (hpa_nodes[nb].closed_pass == hpa_pass) {
				continue;
			}

			const real_t tentative_g = hpa_nodes[curr].g_score + edges[i].distance;
			if (hpa_nodes[nb].open_pass != hpa_pass) {
				hpa_nodes[nb].open_pass = hpa_pass;
				hpa_nodes[nb].g_score = tentative_g;
				hpa_nodes[nb].f_score = tentative_g + _estimate_cost(hpa_nodes[nb].pos, goal_pos);
				hpa_nodes[nb].prev_node_idx = curr;
				open.push_back(nb);
			} else if (tentative_g < hpa_nodes[nb].g_score) {
				hpa_nodes[nb].g_score = tentative_g;
				hpa_nodes[nb].f_score = tentative_g + _estimate_cost(hpa_nodes[nb].pos, goal_pos);
				hpa_nodes[nb].prev_node_idx = curr;
			}
		}
	}

	return false;
}

// Inserts the start and goal as temporary nodes, connects them to their clusters, runs the
// abstract search, then removes the temporary nodes (and the edges that referenced them) so the
// static graph is left untouched. The abstract graph encodes all inter-cluster connectivity,
// so a failed search means the goal is genuinely unreachable.
AStarGrid2D::HPAPathResult AStarGrid2D::_hpa_solve_abstract(const Vector2i &p_from, const Vector2i &p_to, LocalVector<Vector2i> &r_node_path) {
	const int32_t from_cluster = _get_cluster_index(p_from);
	const int32_t to_cluster = _get_cluster_index(p_to);
	if (from_cluster < 0 || to_cluster < 0) {
		return HPA_PATH_RESULT_FAILED;
	}

	const int32_t static_count = (int32_t)hpa_nodes.size();
	const int32_t start_idx = static_count;
	const int32_t goal_idx = static_count + 1;

	// Append the two temporary nodes. No further pushes to `hpa_nodes` happen afterwards, so
	// indices into it remain stable for the rest of this function.
	HierNode start_node;
	start_node.id = start_idx;
	start_node.pos = p_from;
	start_node.cluster_idx = from_cluster;
	hpa_nodes.push_back(start_node);

	HierNode goal_node;
	goal_node.id = goal_idx;
	goal_node.pos = p_to;
	goal_node.cluster_idx = to_cluster;
	hpa_nodes.push_back(goal_node);

	// Start: outgoing edges to the static nodes of its cluster (stored on the temp node only).
	const Rect2i from_rect = _cluster_rect(&hpa_clusters[from_cluster]);
	for (uint32_t i = 0; i < hpa_clusters[from_cluster].node_ids.size(); i++) {
		const int32_t sn = hpa_clusters[from_cluster].node_ids[i];
		const real_t cost = _solve_bounded(p_from, hpa_nodes[sn].pos, from_rect);
		if (cost >= 0.0) {
			hpa_nodes[start_idx].edges.push_back(HierEdge{ sn, cost, false });
		}
	}

	// Goal: incoming edges from the static nodes of its cluster (temporarily added to those
	// static nodes, removed during cleanup below).
	const Rect2i to_rect = _cluster_rect(&hpa_clusters[to_cluster]);
	for (uint32_t i = 0; i < hpa_clusters[to_cluster].node_ids.size(); i++) {
		const int32_t sn = hpa_clusters[to_cluster].node_ids[i];
		const real_t cost = _solve_bounded(hpa_nodes[sn].pos, p_to, to_rect);
		if (cost >= 0.0) {
			hpa_nodes[sn].edges.push_back(HierEdge{ goal_idx, cost, false });
		}
	}

	// If start and goal share a cluster, connect them directly as well.
	if (from_cluster == to_cluster) {
		const real_t cost = _solve_bounded(p_from, p_to, from_rect);
		if (cost >= 0.0) {
			hpa_nodes[start_idx].edges.push_back(HierEdge{ goal_idx, cost, false });
		}
	}

	const bool found = _abstract_astar(start_idx, goal_idx, r_node_path);

	// Cleanup: drop the temporary goal edges from the static nodes, then drop the temp nodes.
	for (uint32_t i = 0; i < hpa_clusters[to_cluster].node_ids.size(); i++) {
		const int32_t sn = hpa_clusters[to_cluster].node_ids[i];
		LocalVector<HierEdge> &edges = hpa_nodes[sn].edges;
		while (!edges.is_empty() && edges[edges.size() - 1].to_node_idx == goal_idx) {
			edges.remove_at(edges.size() - 1);
		}
	}
	hpa_nodes.resize(static_count);

	return found ? HPA_PATH_RESULT_FOUND : HPA_PATH_RESULT_UNREACHABLE;
}

// Produces the full concrete cell path by refining each leg of the abstract path with a bounded
// low-level A* restricted to the clusters the two abstract nodes belong to.
AStarGrid2D::HPAPathResult AStarGrid2D::_hpa_get_cell_path(const Vector2i &p_from, const Vector2i &p_to, LocalVector<Vector2i> &r_cells) {
	LocalVector<Vector2i> node_path;
	const HPAPathResult abstract_result = _hpa_solve_abstract(p_from, p_to, node_path);
	if (abstract_result != HPA_PATH_RESULT_FOUND) {
		return abstract_result;
	}

	r_cells.clear();

	if (node_path.size() == 1) {
		r_cells.push_back(node_path[0]);
		return HPA_PATH_RESULT_FOUND;
	}

	LocalVector<Vector2i> segment;
	for (uint32_t i = 0; i + 1 < node_path.size(); i++) {
		const Vector2i a = node_path[i];
		const Vector2i b = node_path[i + 1];

		Rect2i bounds = _cluster_rect(&hpa_clusters[_get_cluster_index(a)]);
		const int32_t b_cluster = _get_cluster_index(b);
		if (b_cluster >= 0) {
			bounds = bounds.merge(_cluster_rect(&hpa_clusters[b_cluster]));
		}

		segment.clear();
		const real_t cost = _solve_bounded(a, b, bounds, &segment);
		if (cost < 0.0 || segment.is_empty()) {
			return HPA_PATH_RESULT_FAILED;
		}

		// Append the segment, skipping its first cell when it would duplicate the last appended one.
		const uint32_t first = r_cells.is_empty() ? 0 : 1;
		for (uint32_t j = first; j < segment.size(); j++) {
			r_cells.push_back(segment[j]);
		}
	}

	return HPA_PATH_RESULT_FOUND;
}

void AStarGrid2D::update_hpa() {
	hpa_clusters.clear();
	hpa_entrances.clear();
	hpa_nodes.clear();
	hpa_cluster_cols = 0;
	hpa_cluster_rows = 0;

	if (!hpa_enabled) {
		hpa_dirty = false;
		return;
	}

	ERR_FAIL_COND_MSG(dirty, "Grid is not initialized. Call the update method.");

	_build_clusters();
	_build_entrances();
	_create_hpa_nodes();
	_build_intra_edges();

	hpa_dirty = false;
}

void AStarGrid2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_region", "region"), &AStarGrid2D::set_region);
	ClassDB::bind_method(D_METHOD("get_region"), &AStarGrid2D::get_region);
	ClassDB::bind_method(D_METHOD("set_size", "size"), &AStarGrid2D::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &AStarGrid2D::get_size);
	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &AStarGrid2D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &AStarGrid2D::get_offset);
	ClassDB::bind_method(D_METHOD("set_cell_size", "cell_size"), &AStarGrid2D::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &AStarGrid2D::get_cell_size);
	ClassDB::bind_method(D_METHOD("set_cell_shape", "cell_shape"), &AStarGrid2D::set_cell_shape);
	ClassDB::bind_method(D_METHOD("get_cell_shape"), &AStarGrid2D::get_cell_shape);
	ClassDB::bind_method(D_METHOD("is_in_bounds", "x", "y"), &AStarGrid2D::is_in_bounds);
	ClassDB::bind_method(D_METHOD("is_in_boundsv", "id"), &AStarGrid2D::is_in_boundsv);
	ClassDB::bind_method(D_METHOD("is_dirty"), &AStarGrid2D::is_dirty);
	ClassDB::bind_method(D_METHOD("update"), &AStarGrid2D::update);
	ClassDB::bind_method(D_METHOD("set_jumping_enabled", "enabled"), &AStarGrid2D::set_jumping_enabled);
	ClassDB::bind_method(D_METHOD("is_jumping_enabled"), &AStarGrid2D::is_jumping_enabled);
	ClassDB::bind_method(D_METHOD("set_diagonal_mode", "mode"), &AStarGrid2D::set_diagonal_mode);
	ClassDB::bind_method(D_METHOD("get_diagonal_mode"), &AStarGrid2D::get_diagonal_mode);
	ClassDB::bind_method(D_METHOD("set_default_compute_heuristic", "heuristic"), &AStarGrid2D::set_default_compute_heuristic);
	ClassDB::bind_method(D_METHOD("get_default_compute_heuristic"), &AStarGrid2D::get_default_compute_heuristic);
	ClassDB::bind_method(D_METHOD("set_default_estimate_heuristic", "heuristic"), &AStarGrid2D::set_default_estimate_heuristic);
	ClassDB::bind_method(D_METHOD("get_default_estimate_heuristic"), &AStarGrid2D::get_default_estimate_heuristic);
	ClassDB::bind_method(D_METHOD("set_point_solid", "id", "solid"), &AStarGrid2D::set_point_solid, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("is_point_solid", "id"), &AStarGrid2D::is_point_solid);
	ClassDB::bind_method(D_METHOD("set_point_weight_scale", "id", "weight_scale"), &AStarGrid2D::set_point_weight_scale);
	ClassDB::bind_method(D_METHOD("get_point_weight_scale", "id"), &AStarGrid2D::get_point_weight_scale);
	ClassDB::bind_method(D_METHOD("fill_solid_region", "region", "solid"), &AStarGrid2D::fill_solid_region, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("fill_weight_scale_region", "region", "weight_scale"), &AStarGrid2D::fill_weight_scale_region);
	ClassDB::bind_method(D_METHOD("clear"), &AStarGrid2D::clear);

	ClassDB::bind_method(D_METHOD("get_point_position", "id"), &AStarGrid2D::get_point_position);
	ClassDB::bind_method(D_METHOD("get_point_data_in_region", "region"), &AStarGrid2D::get_point_data_in_region);
	ClassDB::bind_method(D_METHOD("get_point_path", "from_id", "to_id", "allow_partial_path"), &AStarGrid2D::get_point_path, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_id_path", "from_id", "to_id", "allow_partial_path"), &AStarGrid2D::get_id_path, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("set_hpa_enabled", "enabled"), &AStarGrid2D::set_hpa_enabled);
	ClassDB::bind_method(D_METHOD("is_hpa_enabled"), &AStarGrid2D::is_hpa_enabled);
	ClassDB::bind_method(D_METHOD("is_hpa_dirty"), &AStarGrid2D::is_hpa_dirty);
	ClassDB::bind_method(D_METHOD("set_hpa_cluster_size", "cluster_size"), &AStarGrid2D::set_hpa_cluster_size);
	ClassDB::bind_method(D_METHOD("get_hpa_cluster_size"), &AStarGrid2D::get_hpa_cluster_size);
	ClassDB::bind_method(D_METHOD("update_hpa"), &AStarGrid2D::update_hpa);

	GDVIRTUAL_BIND(_estimate_cost, "from_id", "end_id")
	GDVIRTUAL_BIND(_compute_cost, "from_id", "to_id")

	ADD_PROPERTY(PropertyInfo(Variant::RECT2I, "region"), "set_region", "get_region");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "cell_size"), "set_cell_size", "get_cell_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cell_shape", PROPERTY_HINT_ENUM, "Square,IsometricRight,IsometricDown"), "set_cell_shape", "get_cell_shape");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "jumping_enabled"), "set_jumping_enabled", "is_jumping_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "default_compute_heuristic", PROPERTY_HINT_ENUM, "Euclidean,Manhattan,Octile,Chebyshev"), "set_default_compute_heuristic", "get_default_compute_heuristic");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "default_estimate_heuristic", PROPERTY_HINT_ENUM, "Euclidean,Manhattan,Octile,Chebyshev"), "set_default_estimate_heuristic", "get_default_estimate_heuristic");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "diagonal_mode", PROPERTY_HINT_ENUM, "Always,Never,At Least One Walkable,Only If No Obstacles"), "set_diagonal_mode", "get_diagonal_mode");

	ADD_GROUP("HPA*", "hpa_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hpa_enabled"), "set_hpa_enabled", "is_hpa_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "hpa_cluster_size", PROPERTY_HINT_RANGE, "1,256,1,or_greater"), "set_hpa_cluster_size", "get_hpa_cluster_size");

	BIND_ENUM_CONSTANT(HEURISTIC_EUCLIDEAN);
	BIND_ENUM_CONSTANT(HEURISTIC_MANHATTAN);
	BIND_ENUM_CONSTANT(HEURISTIC_OCTILE);
	BIND_ENUM_CONSTANT(HEURISTIC_CHEBYSHEV);
	BIND_ENUM_CONSTANT(HEURISTIC_MAX);

	BIND_ENUM_CONSTANT(DIAGONAL_MODE_ALWAYS);
	BIND_ENUM_CONSTANT(DIAGONAL_MODE_NEVER);
	BIND_ENUM_CONSTANT(DIAGONAL_MODE_AT_LEAST_ONE_WALKABLE);
	BIND_ENUM_CONSTANT(DIAGONAL_MODE_ONLY_IF_NO_OBSTACLES);
	BIND_ENUM_CONSTANT(DIAGONAL_MODE_MAX);

	BIND_ENUM_CONSTANT(CELL_SHAPE_SQUARE);
	BIND_ENUM_CONSTANT(CELL_SHAPE_ISOMETRIC_RIGHT);
	BIND_ENUM_CONSTANT(CELL_SHAPE_ISOMETRIC_DOWN);
	BIND_ENUM_CONSTANT(CELL_SHAPE_MAX);
}
