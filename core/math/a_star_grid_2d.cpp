/*************************************************************************/
/*  a_star_grid_2d.cpp                                                   */
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

#include "a_star_grid_2d.h"

static real_t heuristic_manhattan(const Vector2i &p_from, const Vector2i &p_to) {
	real_t dx = (real_t)ABS(p_to.x - p_from.x);
	real_t dy = (real_t)ABS(p_to.y - p_from.y);
	return dx + dy;
}

static real_t heuristic_euclidian(const Vector2i &p_from, const Vector2i &p_to) {
	real_t dx = (real_t)ABS(p_to.x - p_from.x);
	real_t dy = (real_t)ABS(p_to.y - p_from.y);
	return (real_t)Math::sqrt(dx * dx + dy * dy);
}

static real_t heuristic_octile(const Vector2i &p_from, const Vector2i &p_to) {
	real_t dx = (real_t)ABS(p_to.x - p_from.x);
	real_t dy = (real_t)ABS(p_to.y - p_from.y);
	real_t F = Math_SQRT2 - 1;
	return (dx < dy) ? F * dx + dy : F * dy + dx;
}

static real_t heuristic_chebyshev(const Vector2i &p_from, const Vector2i &p_to) {
	real_t dx = (real_t)ABS(p_to.x - p_from.x);
	real_t dy = (real_t)ABS(p_to.y - p_from.y);
	return MAX(dx, dy);
}

static real_t (*heuristics[AStarGrid2D::HEURISTIC_MAX])(const Vector2i &, const Vector2i &) = { heuristic_manhattan, heuristic_euclidian, heuristic_octile, heuristic_chebyshev };

void AStarGrid2D::set_size(const Size2i &p_size) {
	ERR_FAIL_COND(p_size.x < 0 || p_size.y < 0);
	if (p_size != size) {
		size = p_size;
		dirty = true;
	}
}

Size2i AStarGrid2D::get_size() const {
	return size;
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

Size2 AStarGrid2D::get_cell_size() const {
	return cell_size;
}

void AStarGrid2D::update() {
	points.clear();
	for (int64_t y = 0; y < size.y; y++) {
		LocalVector<Point> line;
		for (int64_t x = 0; x < size.x; x++) {
			line.push_back(Point(Vector2i(x, y), offset + Vector2(x, y) * cell_size));
		}
		points.push_back(line);
	}
	dirty = false;
}

bool AStarGrid2D::is_in_bounds(int p_x, int p_y) const {
	return p_x >= 0 && p_x < size.width && p_y >= 0 && p_y < size.height;
}

bool AStarGrid2D::is_in_boundsv(const Vector2i &p_id) const {
	return p_id.x >= 0 && p_id.x < size.width && p_id.y >= 0 && p_id.y < size.height;
}

bool AStarGrid2D::is_dirty() const {
	return dirty;
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

void AStarGrid2D::set_default_heuristic(Heuristic p_heuristic) {
	ERR_FAIL_INDEX((int)p_heuristic, (int)HEURISTIC_MAX);
	default_heuristic = p_heuristic;
}

AStarGrid2D::Heuristic AStarGrid2D::get_default_heuristic() const {
	return default_heuristic;
}

void AStarGrid2D::set_point_solid(const Vector2i &p_id, bool p_solid) {
	ERR_FAIL_COND_MSG(dirty, "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_MSG(!is_in_boundsv(p_id), vformat("Can't set if point is disabled. Point out of bounds (%s/%s, %s/%s).", p_id.x, size.width, p_id.y, size.height));
	points[p_id.y][p_id.x].solid = p_solid;
}

bool AStarGrid2D::is_point_solid(const Vector2i &p_id) const {
	ERR_FAIL_COND_V_MSG(dirty, false, "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_id), false, vformat("Can't get if point is disabled. Point out of bounds (%s/%s, %s/%s).", p_id.x, size.width, p_id.y, size.height));
	return points[p_id.y][p_id.x].solid;
}

AStarGrid2D::Point *AStarGrid2D::_jump(Point *p_from, Point *p_to) {
	if (!p_to || p_to->solid) {
		return nullptr;
	}
	if (p_to == end) {
		return p_to;
	}

	int64_t from_x = p_from->id.x;
	int64_t from_y = p_from->id.y;

	int64_t to_x = p_to->id.x;
	int64_t to_y = p_to->id.y;

	int64_t dx = to_x - from_x;
	int64_t dy = to_y - from_y;

	if (diagonal_mode == DIAGONAL_MODE_ALWAYS || diagonal_mode == DIAGONAL_MODE_AT_LEAST_ONE_WALKABLE) {
		if (dx != 0 && dy != 0) {
			if ((_is_walkable(to_x - dx, to_y + dy) && !_is_walkable(to_x - dx, to_y)) || (_is_walkable(to_x + dx, to_y - dy) && !_is_walkable(to_x, to_y - dy))) {
				return p_to;
			}
			if (_jump(p_to, _get_point(to_x + dx, to_y)) != nullptr) {
				return p_to;
			}
			if (_jump(p_to, _get_point(to_x, to_y + dy)) != nullptr) {
				return p_to;
			}
		} else {
			if (dx != 0) {
				if ((_is_walkable(to_x + dx, to_y + 1) && !_is_walkable(to_x, to_y + 1)) || (_is_walkable(to_x + dx, to_y - 1) && !_is_walkable(to_x, to_y - 1))) {
					return p_to;
				}
			} else {
				if ((_is_walkable(to_x + 1, to_y + dy) && !_is_walkable(to_x + 1, to_y)) || (_is_walkable(to_x - 1, to_y + dy) && !_is_walkable(to_x - 1, to_y))) {
					return p_to;
				}
			}
		}
		if (_is_walkable(to_x + dx, to_y + dy) && (diagonal_mode == DIAGONAL_MODE_ALWAYS || (_is_walkable(to_x + dx, to_y) || _is_walkable(to_x, to_y + dy)))) {
			return _jump(p_to, _get_point(to_x + dx, to_y + dy));
		}
	} else if (diagonal_mode == DIAGONAL_MODE_ONLY_IF_NO_OBSTACLES) {
		if (dx != 0 && dy != 0) {
			if ((_is_walkable(to_x + dx, to_y + dy) && !_is_walkable(to_x, to_y + dy)) || !_is_walkable(to_x + dx, to_y)) {
				return p_to;
			}
			if (_jump(p_to, _get_point(to_x + dx, to_y)) != nullptr) {
				return p_to;
			}
			if (_jump(p_to, _get_point(to_x, to_y + dy)) != nullptr) {
				return p_to;
			}
		} else {
			if (dx != 0) {
				if ((_is_walkable(to_x, to_y + 1) && !_is_walkable(to_x - dx, to_y + 1)) || (_is_walkable(to_x, to_y - 1) && !_is_walkable(to_x - dx, to_y - 1))) {
					return p_to;
				}
			} else {
				if ((_is_walkable(to_x + 1, to_y) && !_is_walkable(to_x + 1, to_y - dy)) || (_is_walkable(to_x - 1, to_y) && !_is_walkable(to_x - 1, to_y - dy))) {
					return p_to;
				}
			}
		}
		if (_is_walkable(to_x + dx, to_y + dy) && _is_walkable(to_x + dx, to_y) && _is_walkable(to_x, to_y + dy)) {
			return _jump(p_to, _get_point(to_x + dx, to_y + dy));
		}
	} else { // DIAGONAL_MODE_NEVER
		if (dx != 0) {
			if (!_is_walkable(to_x + dx, to_y)) {
				return p_to;
			}
			if (_jump(p_to, _get_point(to_x, to_y + 1)) != nullptr) {
				return p_to;
			}
			if (_jump(p_to, _get_point(to_x, to_y - 1)) != nullptr) {
				return p_to;
			}
		} else {
			if (!_is_walkable(to_x, to_y + dy)) {
				return p_to;
			}
			if (_jump(p_to, _get_point(to_x + 1, to_y)) != nullptr) {
				return p_to;
			}
			if (_jump(p_to, _get_point(to_x - 1, to_y)) != nullptr) {
				return p_to;
			}
		}
		if (_is_walkable(to_x + dx, to_y + dy) && _is_walkable(to_x + dx, to_y) && _is_walkable(to_x, to_y + dy)) {
			return _jump(p_to, _get_point(to_x + dx, to_y + dy));
		}
	}
	return nullptr;
}

void AStarGrid2D::_get_nbors(Point *p_point, List<Point *> &r_nbors) {
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

		if (p_point->id.x - 1 >= 0) {
			left = _get_point_unchecked(p_point->id.x - 1, p_point->id.y);
			has_left = true;
		}
		if (p_point->id.x + 1 < size.width) {
			right = _get_point_unchecked(p_point->id.x + 1, p_point->id.y);
			has_right = true;
		}
		if (p_point->id.y - 1 >= 0) {
			top = _get_point_unchecked(p_point->id.x, p_point->id.y - 1);
			if (has_left) {
				top_left = _get_point_unchecked(p_point->id.x - 1, p_point->id.y - 1);
			}
			if (has_right) {
				top_right = _get_point_unchecked(p_point->id.x + 1, p_point->id.y - 1);
			}
		}
		if (p_point->id.y + 1 < size.height) {
			bottom = _get_point_unchecked(p_point->id.x, p_point->id.y + 1);
			if (has_left) {
				bottom_left = _get_point_unchecked(p_point->id.x - 1, p_point->id.y + 1);
			}
			if (has_right) {
				bottom_right = _get_point_unchecked(p_point->id.x + 1, p_point->id.y + 1);
			}
		}
	}

	if (top && !top->solid) {
		r_nbors.push_back(top);
		ts0 = true;
	}
	if (right && !right->solid) {
		r_nbors.push_back(right);
		ts1 = true;
	}
	if (bottom && !bottom->solid) {
		r_nbors.push_back(bottom);
		ts2 = true;
	}
	if (left && !left->solid) {
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

	if (td0 && (top_left && !top_left->solid)) {
		r_nbors.push_back(top_left);
	}
	if (td1 && (top_right && !top_right->solid)) {
		r_nbors.push_back(top_right);
	}
	if (td2 && (bottom_right && !bottom_right->solid)) {
		r_nbors.push_back(bottom_right);
	}
	if (td3 && (bottom_left && !bottom_left->solid)) {
		r_nbors.push_back(bottom_left);
	}
}

bool AStarGrid2D::_solve(Point *p_begin_point, Point *p_end_point) {
	pass++;

	if (p_end_point->solid) {
		return false;
	}

	bool found_route = false;

	Vector<Point *> open_list;
	SortArray<Point *, SortPoints> sorter;

	p_begin_point->g_score = 0;
	p_begin_point->f_score = _estimate_cost(p_begin_point->id, p_end_point->id);
	open_list.push_back(p_begin_point);
	end = p_end_point;

	while (!open_list.is_empty()) {
		Point *p = open_list[0]; // The currently processed point.

		if (p == p_end_point) {
			found_route = true;
			break;
		}

		sorter.pop_heap(0, open_list.size(), open_list.ptrw()); // Remove the current point from the open list.
		open_list.remove_at(open_list.size() - 1);
		p->closed_pass = pass; // Mark the point as closed.

		List<Point *> nbors;
		_get_nbors(p, nbors);
		for (List<Point *>::Element *E = nbors.front(); E; E = E->next()) {
			Point *e = E->get(); // The neighbour point.
			if (jumping_enabled) {
				e = _jump(p, e);
				if (!e || e->closed_pass == pass) {
					continue;
				}
			} else {
				if (e->solid || e->closed_pass == pass) {
					continue;
				}
			}

			real_t tentative_g_score = p->g_score + _compute_cost(p->id, e->id);
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

			if (new_point) { // The position of the new points is already known.
				sorter.push_heap(0, open_list.size() - 1, 0, e, open_list.ptrw());
			} else {
				sorter.push_heap(0, open_list.find(e), 0, e, open_list.ptrw());
			}
		}
	}

	return found_route;
}

real_t AStarGrid2D::_estimate_cost(const Vector2i &p_from_id, const Vector2i &p_to_id) {
	real_t scost;
	if (GDVIRTUAL_CALL(_estimate_cost, p_from_id, p_to_id, scost)) {
		return scost;
	}
	return heuristics[default_heuristic](p_from_id, p_to_id);
}

real_t AStarGrid2D::_compute_cost(const Vector2i &p_from_id, const Vector2i &p_to_id) {
	real_t scost;
	if (GDVIRTUAL_CALL(_compute_cost, p_from_id, p_to_id, scost)) {
		return scost;
	}
	return heuristics[default_heuristic](p_from_id, p_to_id);
}

void AStarGrid2D::clear() {
	points.clear();
	size = Vector2i();
}

Vector<Vector2> AStarGrid2D::get_point_path(const Vector2i &p_from_id, const Vector2i &p_to_id) {
	ERR_FAIL_COND_V_MSG(dirty, Vector<Vector2>(), "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_from_id), Vector<Vector2>(), vformat("Can't get id path. Point out of bounds (%s/%s, %s/%s)", p_from_id.x, size.width, p_from_id.y, size.height));
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_to_id), Vector<Vector2>(), vformat("Can't get id path. Point out of bounds (%s/%s, %s/%s)", p_to_id.x, size.width, p_to_id.y, size.height));

	Point *a = _get_point(p_from_id.x, p_from_id.y);
	Point *b = _get_point(p_to_id.x, p_to_id.y);

	if (a == b) {
		Vector<Vector2> ret;
		ret.push_back(a->pos);
		return ret;
	}

	Point *begin_point = a;
	Point *end_point = b;

	bool found_route = _solve(begin_point, end_point);
	if (!found_route) {
		return Vector<Vector2>();
	}

	Point *p = end_point;
	int64_t pc = 1;
	while (p != begin_point) {
		pc++;
		p = p->prev_point;
	}

	Vector<Vector2> path;
	path.resize(pc);

	{
		Vector2 *w = path.ptrw();

		p = end_point;
		int64_t idx = pc - 1;
		while (p != begin_point) {
			w[idx--] = p->pos;
			p = p->prev_point;
		}

		w[0] = p->pos;
	}

	return path;
}

Vector<Vector2> AStarGrid2D::get_id_path(const Vector2i &p_from_id, const Vector2i &p_to_id) {
	ERR_FAIL_COND_V_MSG(dirty, Vector<Vector2>(), "Grid is not initialized. Call the update method.");
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_from_id), Vector<Vector2>(), vformat("Can't get id path. Point out of bounds (%s/%s, %s/%s)", p_from_id.x, size.width, p_from_id.y, size.height));
	ERR_FAIL_COND_V_MSG(!is_in_boundsv(p_to_id), Vector<Vector2>(), vformat("Can't get id path. Point out of bounds (%s/%s, %s/%s)", p_to_id.x, size.width, p_to_id.y, size.height));

	Point *a = _get_point(p_from_id.x, p_from_id.y);
	Point *b = _get_point(p_to_id.x, p_to_id.y);

	if (a == b) {
		Vector<Vector2> ret;
		ret.push_back(Vector2((float)a->id.x, (float)a->id.y));
		return ret;
	}

	Point *begin_point = a;
	Point *end_point = b;

	bool found_route = _solve(begin_point, end_point);
	if (!found_route) {
		return Vector<Vector2>();
	}

	Point *p = end_point;
	int64_t pc = 1;
	while (p != begin_point) {
		pc++;
		p = p->prev_point;
	}

	Vector<Vector2> path;
	path.resize(pc);

	{
		Vector2 *w = path.ptrw();

		p = end_point;
		int64_t idx = pc - 1;
		while (p != begin_point) {
			w[idx--] = Vector2((float)p->id.x, (float)p->id.y);
			p = p->prev_point;
		}

		w[0] = p->id;
	}

	return path;
}

void AStarGrid2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &AStarGrid2D::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &AStarGrid2D::get_size);
	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &AStarGrid2D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &AStarGrid2D::get_offset);
	ClassDB::bind_method(D_METHOD("set_cell_size", "cell_size"), &AStarGrid2D::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &AStarGrid2D::get_cell_size);
	ClassDB::bind_method(D_METHOD("is_in_bounds", "x", "y"), &AStarGrid2D::is_in_bounds);
	ClassDB::bind_method(D_METHOD("is_in_boundsv", "id"), &AStarGrid2D::is_in_boundsv);
	ClassDB::bind_method(D_METHOD("is_dirty"), &AStarGrid2D::is_dirty);
	ClassDB::bind_method(D_METHOD("update"), &AStarGrid2D::update);
	ClassDB::bind_method(D_METHOD("set_jumping_enabled", "enabled"), &AStarGrid2D::set_jumping_enabled);
	ClassDB::bind_method(D_METHOD("is_jumping_enabled"), &AStarGrid2D::is_jumping_enabled);
	ClassDB::bind_method(D_METHOD("set_diagonal_mode", "mode"), &AStarGrid2D::set_diagonal_mode);
	ClassDB::bind_method(D_METHOD("get_diagonal_mode"), &AStarGrid2D::get_diagonal_mode);
	ClassDB::bind_method(D_METHOD("set_default_heuristic", "heuristic"), &AStarGrid2D::set_default_heuristic);
	ClassDB::bind_method(D_METHOD("get_default_heuristic"), &AStarGrid2D::get_default_heuristic);
	ClassDB::bind_method(D_METHOD("set_point_solid", "id", "solid"), &AStarGrid2D::set_point_solid, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("is_point_solid", "id"), &AStarGrid2D::is_point_solid);
	ClassDB::bind_method(D_METHOD("clear"), &AStarGrid2D::clear);

	ClassDB::bind_method(D_METHOD("get_point_path", "from_id", "to_id"), &AStarGrid2D::get_point_path);
	ClassDB::bind_method(D_METHOD("get_id_path", "from_id", "to_id"), &AStarGrid2D::get_id_path);

	GDVIRTUAL_BIND(_estimate_cost, "from_id", "to_id")
	GDVIRTUAL_BIND(_compute_cost, "from_id", "to_id")

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "cell_size"), "set_cell_size", "get_cell_size");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "jumping_enabled"), "set_jumping_enabled", "is_jumping_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "default_heuristic", PROPERTY_HINT_ENUM, "Manhattan,Euclidean,Octile,Chebyshev,Max"), "set_default_heuristic", "get_default_heuristic");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "diagonal_mode", PROPERTY_HINT_ENUM, "Never,Always,At Least One Walkable,Only If No Obstacles,Max"), "set_diagonal_mode", "get_diagonal_mode");

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
}
