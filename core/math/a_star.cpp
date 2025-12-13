/**************************************************************************/
/*  a_star.cpp                                                            */
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

#include "a_star.h"
#include "a_star.compat.inc"

#include "core/math/geometry_3d.h"

int64_t AStar3D::get_available_point_id() const {
	if (points.has(last_free_id)) {
		int64_t cur_new_id = last_free_id + 1;
		while (points.has(cur_new_id)) {
			cur_new_id++;
		}
		last_free_id = cur_new_id;
	}

	return last_free_id;
}

void AStar3D::add_point(int64_t p_id, const Vector3 &p_pos, real_t p_weight_scale) {
	ERR_FAIL_COND_MSG(p_id < 0, vformat("Can't add a point with negative id: %d.", p_id));
	ERR_FAIL_COND_MSG(p_weight_scale < 0.0, vformat("Can't add a point with weight scale less than 0.0: %f.", p_weight_scale));

	Point **point_entry = points.getptr(p_id);

	if (!point_entry) {
		Point *pt = memnew(Point);
		pt->id = p_id;
		pt->pos = p_pos;
		pt->weight_scale = p_weight_scale;
		pt->prev_point = nullptr;
		pt->open_pass = 0;
		pt->closed_pass = 0;
		pt->enabled = true;
		points.insert_new(p_id, pt);
	} else {
		Point *found_pt = *point_entry;
		found_pt->pos = p_pos;
		found_pt->weight_scale = p_weight_scale;
	}
}

Vector3 AStar3D::get_point_position(int64_t p_id) const {
	Point *const *point_entry = points.getptr(p_id);
	ERR_FAIL_COND_V_MSG(!point_entry, Vector3(), vformat("Can't get point's position. Point with id: %d doesn't exist.", p_id));

	return (*point_entry)->pos;
}

void AStar3D::set_point_position(int64_t p_id, const Vector3 &p_pos) {
	Point **point_entry = points.getptr(p_id);
	ERR_FAIL_COND_MSG(!point_entry, vformat("Can't set point's position. Point with id: %d doesn't exist.", p_id));

	(*point_entry)->pos = p_pos;
}

real_t AStar3D::get_point_weight_scale(int64_t p_id) const {
	Point *const *point_entry = points.getptr(p_id);
	ERR_FAIL_COND_V_MSG(!point_entry, 0, vformat("Can't get point's weight scale. Point with id: %d doesn't exist.", p_id));

	return (*point_entry)->weight_scale;
}

void AStar3D::set_point_weight_scale(int64_t p_id, real_t p_weight_scale) {
	Point **point_entry = points.getptr(p_id);
	ERR_FAIL_COND_MSG(!point_entry, vformat("Can't set point's weight scale. Point with id: %d doesn't exist.", p_id));
	ERR_FAIL_COND_MSG(p_weight_scale < 0.0, vformat("Can't set point's weight scale less than 0.0: %f.", p_weight_scale));

	(*point_entry)->weight_scale = p_weight_scale;
}

void AStar3D::remove_point(int64_t p_id) {
	Point **point_entry = points.getptr(p_id);
	ERR_FAIL_COND_MSG(!point_entry, vformat("Can't remove point. Point with id: %d doesn't exist.", p_id));
	Point *p = *point_entry;

	for (KeyValue<int64_t, Point *> &kv : p->neighbors) {
		Segment s(p_id, kv.key);
		segments.erase(s);

		kv.value->neighbors.erase(p->id);
		kv.value->unlinked_neighbours.erase(p->id);
	}

	for (KeyValue<int64_t, Point *> &kv : p->unlinked_neighbours) {
		Segment s(p_id, kv.key);
		segments.erase(s);

		kv.value->neighbors.erase(p->id);
		kv.value->unlinked_neighbours.erase(p->id);
	}

	memdelete(p);
	points.erase(p_id);
	last_free_id = p_id;
}

void AStar3D::connect_points(int64_t p_id, int64_t p_with_id, bool bidirectional) {
	ERR_FAIL_COND_MSG(p_id == p_with_id, vformat("Can't connect point with id: %d to itself.", p_id));

	Point **a_entry = points.getptr(p_id);
	ERR_FAIL_COND_MSG(!a_entry, vformat("Can't connect points. Point with id: %d doesn't exist.", p_id));
	Point *a = *a_entry;

	Point **b_entry = points.getptr(p_with_id);
	ERR_FAIL_COND_MSG(!b_entry, vformat("Can't connect points. Point with id: %d doesn't exist.", p_with_id));
	Point *b = *b_entry;

	a->neighbors.insert(b->id, b);

	if (bidirectional) {
		b->neighbors.insert(a->id, a);
	} else {
		b->unlinked_neighbours.insert(a->id, a);
	}

	Segment s(p_id, p_with_id);
	if (bidirectional) {
		s.direction = Segment::BIDIRECTIONAL;
	}

	HashSet<Segment, Segment>::Iterator element = segments.find(s);
	if (element) {
		s.direction |= element->direction;
		if (s.direction == Segment::BIDIRECTIONAL) {
			// Both are neighbors of each other now
			a->unlinked_neighbours.erase(b->id);
			b->unlinked_neighbours.erase(a->id);
		}
		segments.remove(element);
	}

	segments.insert(s);
}

void AStar3D::disconnect_points(int64_t p_id, int64_t p_with_id, bool bidirectional) {
	Point **a_entry = points.getptr(p_id);
	ERR_FAIL_COND_MSG(!a_entry, vformat("Can't disconnect points. Point with id: %d doesn't exist.", p_id));
	Point *a = *a_entry;

	Point **b_entry = points.getptr(p_with_id);
	ERR_FAIL_COND_MSG(!b_entry, vformat("Can't disconnect points. Point with id: %d doesn't exist.", p_with_id));
	Point *b = *b_entry;

	Segment s(p_id, p_with_id);
	int remove_direction = bidirectional ? (int)Segment::BIDIRECTIONAL : (int)s.direction;

	HashSet<Segment, Segment>::Iterator element = segments.find(s);
	if (element) {
		// s is the new segment
		// Erase the directions to be removed
		s.direction = (element->direction & ~remove_direction);

		a->neighbors.erase(b->id);
		if (bidirectional) {
			b->neighbors.erase(a->id);
			if (element->direction != Segment::BIDIRECTIONAL) {
				a->unlinked_neighbours.erase(b->id);
				b->unlinked_neighbours.erase(a->id);
			}
		} else {
			if (s.direction == Segment::NONE) {
				b->unlinked_neighbours.erase(a->id);
			} else {
				a->unlinked_neighbours.insert(b->id, b);
			}
		}

		segments.remove(element);
		if (s.direction != Segment::NONE) {
			segments.insert(s);
		}
	}
}

bool AStar3D::has_point(int64_t p_id) const {
	return points.has(p_id);
}

PackedInt64Array AStar3D::get_point_ids() {
	PackedInt64Array point_list;

	for (KeyValue<int64_t, Point *> &kv : points) {
		point_list.push_back(kv.key);
	}

	return point_list;
}

Vector<int64_t> AStar3D::get_point_connections(int64_t p_id) {
	Point **p_entry = points.getptr(p_id);
	ERR_FAIL_COND_V_MSG(!p_entry, Vector<int64_t>(), vformat("Can't get point's connections. Point with id: %d doesn't exist.", p_id));

	Vector<int64_t> point_list;

	for (KeyValue<int64_t, Point *> &kv : (*p_entry)->neighbors) {
		point_list.push_back(kv.key);
	}

	return point_list;
}

bool AStar3D::are_points_connected(int64_t p_id, int64_t p_with_id, bool bidirectional) const {
	Segment s(p_id, p_with_id);
	const HashSet<Segment, Segment>::Iterator element = segments.find(s);

	return element &&
			(bidirectional || (element->direction & s.direction) == s.direction);
}

void AStar3D::clear() {
	last_free_id = 0;
	for (KeyValue<int64_t, Point *> &kv : points) {
		memdelete(kv.value);
	}
	segments.clear();
	points.clear();
}

int64_t AStar3D::get_point_count() const {
	return points.size();
}

int64_t AStar3D::get_point_capacity() const {
	return points.get_capacity();
}

void AStar3D::reserve_space(int64_t p_num_nodes) {
	ERR_FAIL_COND_MSG(p_num_nodes <= 0, vformat("New capacity must be greater than 0, new was: %d.", p_num_nodes));
	points.reserve(p_num_nodes);
}

int64_t AStar3D::get_closest_point(const Vector3 &p_point, bool p_include_disabled) const {
	int64_t closest_id = -1;
	real_t closest_dist = 1e20;

	for (const KeyValue<int64_t, Point *> &kv : points) {
		if (!p_include_disabled && !kv.value->enabled) {
			continue; // Disabled points should not be considered.
		}

		// Keep the closest point's ID, and in case of multiple closest IDs,
		// the smallest one (makes it deterministic).
		real_t d = p_point.distance_squared_to(kv.value->pos);
		int64_t id = kv.key;
		if (d <= closest_dist) {
			if (d == closest_dist && id > closest_id) { // Keep lowest ID.
				continue;
			}
			closest_dist = d;
			closest_id = id;
		}
	}

	return closest_id;
}

Vector3 AStar3D::get_closest_position_in_segment(const Vector3 &p_point) const {
	real_t closest_dist = 1e20;
	Vector3 closest_point;

	for (const Segment &E : segments) {
		const Point *from_point = *points.getptr(E.key.first);
		const Point *to_point = *points.getptr(E.key.second);

		if (!(from_point->enabled && to_point->enabled)) {
			continue;
		}

		Vector3 p = Geometry3D::get_closest_point_to_segment(p_point, from_point->pos, to_point->pos);
		real_t d = p_point.distance_squared_to(p);
		if (d < closest_dist) {
			closest_point = p;
			closest_dist = d;
		}
	}

	return closest_point;
}

bool AStar3D::_solve(Point *p_begin_point, Point *p_end_point, bool p_allow_partial_path) {
	last_closest_point = nullptr;
	pass++;

	if (!p_begin_point->enabled) {
		return false;
	}
	if (p_begin_point == p_end_point) {
		return true;
	}
	if (!p_end_point->enabled && !p_allow_partial_path) {
		return false;
	}

	bool found_route = false;

	LocalVector<Point *> open_list;
	SortArray<Point *, SortPoints> sorter;

	p_begin_point->g_score = 0;
	p_begin_point->f_score = _estimate_cost(p_begin_point->id, p_end_point->id);
	p_begin_point->abs_g_score = 0;
	p_begin_point->abs_f_score = _estimate_cost(p_begin_point->id, p_end_point->id);
	open_list.push_back(p_begin_point);

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

		for (const KeyValue<int64_t, Point *> &kv : p->neighbors) {
			Point *e = kv.value; // The neighbor point.

			if (!e->enabled || e->closed_pass == pass) {
				continue;
			}

			if (neighbor_filter_enabled) {
				bool filtered;
				if (GDVIRTUAL_CALL(_filter_neighbor, p->id, e->id, filtered) && filtered) {
					continue;
				}
			}

			real_t tentative_g_score = p->g_score + _compute_cost(p->id, e->id) * e->weight_scale;

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

real_t AStar3D::_estimate_cost(int64_t p_from_id, int64_t p_end_id) {
	real_t scost;
	if (GDVIRTUAL_CALL(_estimate_cost, p_from_id, p_end_id, scost)) {
		return scost;
	}

	Point **from_entry = points.getptr(p_from_id);
	ERR_FAIL_COND_V_MSG(!from_entry, 0, vformat("Can't estimate cost. Point with id: %d doesn't exist.", p_from_id));
	Point *from_point = *from_entry;

	Point **end_entry = points.getptr(p_end_id);
	ERR_FAIL_COND_V_MSG(!end_entry, 0, vformat("Can't estimate cost. Point with id: %d doesn't exist.", p_end_id));
	Point *end_point = *end_entry;

	return from_point->pos.distance_to(end_point->pos);
}

real_t AStar3D::_compute_cost(int64_t p_from_id, int64_t p_to_id) {
	real_t scost;
	if (GDVIRTUAL_CALL(_compute_cost, p_from_id, p_to_id, scost)) {
		return scost;
	}

	Point **from_entry = points.getptr(p_from_id);
	ERR_FAIL_COND_V_MSG(!from_entry, 0, vformat("Can't compute cost. Point with id: %d doesn't exist.", p_from_id));
	Point *from_point = *from_entry;

	Point **to_entry = points.getptr(p_to_id);
	ERR_FAIL_COND_V_MSG(!to_entry, 0, vformat("Can't compute cost. Point with id: %d doesn't exist.", p_to_id));
	Point *to_point = *to_entry;

	return from_point->pos.distance_to(to_point->pos);
}

Vector<Vector3> AStar3D::get_point_path(int64_t p_from_id, int64_t p_to_id, bool p_allow_partial_path) {
	Point **a_entry = points.getptr(p_from_id);
	ERR_FAIL_COND_V_MSG(!a_entry, Vector<Vector3>(), vformat("Can't get point path. Point with id: %d doesn't exist.", p_from_id));
	Point *a = *a_entry;

	Point **b_entry = points.getptr(p_to_id);
	ERR_FAIL_COND_V_MSG(!b_entry, Vector<Vector3>(), vformat("Can't get point path. Point with id: %d doesn't exist.", p_to_id));
	Point *b = *b_entry;

	Point *begin_point = a;
	Point *end_point = b;

	bool found_route = _solve(begin_point, end_point, p_allow_partial_path);
	if (!found_route) {
		if (!p_allow_partial_path || last_closest_point == nullptr) {
			return Vector<Vector3>();
		}

		// Use closest point instead.
		end_point = last_closest_point;
	}

	Point *p = end_point;
	int64_t pc = 1; // Begin point
	while (p != begin_point) {
		pc++;
		p = p->prev_point;
	}

	Vector<Vector3> path;
	path.resize(pc);

	{
		Vector3 *w = path.ptrw();

		Point *p2 = end_point;
		int64_t idx = pc - 1;
		while (p2 != begin_point) {
			w[idx--] = p2->pos;
			p2 = p2->prev_point;
		}

		w[0] = p2->pos; // Assign first
	}

	return path;
}

Vector<int64_t> AStar3D::get_id_path(int64_t p_from_id, int64_t p_to_id, bool p_allow_partial_path) {
	Point **a_entry = points.getptr(p_from_id);
	ERR_FAIL_COND_V_MSG(!a_entry, Vector<int64_t>(), vformat("Can't get id path. Point with id: %d doesn't exist.", p_from_id));
	Point *a = *a_entry;

	Point **b_entry = points.getptr(p_to_id);
	ERR_FAIL_COND_V_MSG(!b_entry, Vector<int64_t>(), vformat("Can't get id path. Point with id: %d doesn't exist.", p_to_id));
	Point *b = *b_entry;

	Point *begin_point = a;
	Point *end_point = b;

	bool found_route = _solve(begin_point, end_point, p_allow_partial_path);
	if (!found_route) {
		if (!p_allow_partial_path || last_closest_point == nullptr) {
			return Vector<int64_t>();
		}

		// Use closest point instead.
		end_point = last_closest_point;
	}

	Point *p = end_point;
	int64_t pc = 1; // Begin point
	while (p != begin_point) {
		pc++;
		p = p->prev_point;
	}

	Vector<int64_t> path;
	path.resize(pc);

	{
		int64_t *w = path.ptrw();

		p = end_point;
		int64_t idx = pc - 1;
		while (p != begin_point) {
			w[idx--] = p->id;
			p = p->prev_point;
		}

		w[0] = p->id; // Assign first
	}

	return path;
}

bool AStar3D::is_neighbor_filter_enabled() const {
	return neighbor_filter_enabled;
}

void AStar3D::set_neighbor_filter_enabled(bool p_enabled) {
	neighbor_filter_enabled = p_enabled;
}

void AStar3D::set_point_disabled(int64_t p_id, bool p_disabled) {
	Point **p_entry = points.getptr(p_id);
	ERR_FAIL_COND_MSG(!p_entry, vformat("Can't set if point is disabled. Point with id: %d doesn't exist.", p_id));
	Point *p = *p_entry;

	p->enabled = !p_disabled;
}

bool AStar3D::is_point_disabled(int64_t p_id) const {
	Point *const *p_entry = points.getptr(p_id);
	ERR_FAIL_COND_V_MSG(!p_entry, false, vformat("Can't get if point is disabled. Point with id: %d doesn't exist.", p_id));
	Point *p = *p_entry;

	return !p->enabled;
}

void AStar3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_available_point_id"), &AStar3D::get_available_point_id);
	ClassDB::bind_method(D_METHOD("add_point", "id", "position", "weight_scale"), &AStar3D::add_point, DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("get_point_position", "id"), &AStar3D::get_point_position);
	ClassDB::bind_method(D_METHOD("set_point_position", "id", "position"), &AStar3D::set_point_position);
	ClassDB::bind_method(D_METHOD("get_point_weight_scale", "id"), &AStar3D::get_point_weight_scale);
	ClassDB::bind_method(D_METHOD("set_point_weight_scale", "id", "weight_scale"), &AStar3D::set_point_weight_scale);
	ClassDB::bind_method(D_METHOD("remove_point", "id"), &AStar3D::remove_point);
	ClassDB::bind_method(D_METHOD("has_point", "id"), &AStar3D::has_point);
	ClassDB::bind_method(D_METHOD("get_point_connections", "id"), &AStar3D::get_point_connections);
	ClassDB::bind_method(D_METHOD("get_point_ids"), &AStar3D::get_point_ids);

	ClassDB::bind_method(D_METHOD("set_point_disabled", "id", "disabled"), &AStar3D::set_point_disabled, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("is_point_disabled", "id"), &AStar3D::is_point_disabled);

	ClassDB::bind_method(D_METHOD("set_neighbor_filter_enabled", "enabled"), &AStar3D::set_neighbor_filter_enabled);
	ClassDB::bind_method(D_METHOD("is_neighbor_filter_enabled"), &AStar3D::is_neighbor_filter_enabled);

	ClassDB::bind_method(D_METHOD("connect_points", "id", "to_id", "bidirectional"), &AStar3D::connect_points, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("disconnect_points", "id", "to_id", "bidirectional"), &AStar3D::disconnect_points, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("are_points_connected", "id", "to_id", "bidirectional"), &AStar3D::are_points_connected, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("get_point_count"), &AStar3D::get_point_count);
	ClassDB::bind_method(D_METHOD("get_point_capacity"), &AStar3D::get_point_capacity);
	ClassDB::bind_method(D_METHOD("reserve_space", "num_nodes"), &AStar3D::reserve_space);
	ClassDB::bind_method(D_METHOD("clear"), &AStar3D::clear);

	ClassDB::bind_method(D_METHOD("get_closest_point", "to_position", "include_disabled"), &AStar3D::get_closest_point, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_closest_position_in_segment", "to_position"), &AStar3D::get_closest_position_in_segment);

	ClassDB::bind_method(D_METHOD("get_point_path", "from_id", "to_id", "allow_partial_path"), &AStar3D::get_point_path, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_id_path", "from_id", "to_id", "allow_partial_path"), &AStar3D::get_id_path, DEFVAL(false));

	GDVIRTUAL_BIND(_filter_neighbor, "from_id", "neighbor_id")
	GDVIRTUAL_BIND(_estimate_cost, "from_id", "end_id")
	GDVIRTUAL_BIND(_compute_cost, "from_id", "to_id")

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "neighbor_filter_enabled"), "set_neighbor_filter_enabled", "is_neighbor_filter_enabled");
}

AStar3D::~AStar3D() {
	clear();
}

/////////////////////////////////////////////////////////////

int64_t AStar2D::get_available_point_id() const {
	return astar.get_available_point_id();
}

void AStar2D::add_point(int64_t p_id, const Vector2 &p_pos, real_t p_weight_scale) {
	astar.add_point(p_id, Vector3(p_pos.x, p_pos.y, 0), p_weight_scale);
}

Vector2 AStar2D::get_point_position(int64_t p_id) const {
	Vector3 p = astar.get_point_position(p_id);
	return Vector2(p.x, p.y);
}

void AStar2D::set_point_position(int64_t p_id, const Vector2 &p_pos) {
	astar.set_point_position(p_id, Vector3(p_pos.x, p_pos.y, 0));
}

real_t AStar2D::get_point_weight_scale(int64_t p_id) const {
	return astar.get_point_weight_scale(p_id);
}

void AStar2D::set_point_weight_scale(int64_t p_id, real_t p_weight_scale) {
	astar.set_point_weight_scale(p_id, p_weight_scale);
}

void AStar2D::remove_point(int64_t p_id) {
	astar.remove_point(p_id);
}

bool AStar2D::has_point(int64_t p_id) const {
	return astar.has_point(p_id);
}

Vector<int64_t> AStar2D::get_point_connections(int64_t p_id) {
	return astar.get_point_connections(p_id);
}

PackedInt64Array AStar2D::get_point_ids() {
	return astar.get_point_ids();
}

bool AStar2D::is_neighbor_filter_enabled() const {
	return astar.neighbor_filter_enabled;
}

void AStar2D::set_neighbor_filter_enabled(bool p_enabled) {
	astar.neighbor_filter_enabled = p_enabled;
}

void AStar2D::set_point_disabled(int64_t p_id, bool p_disabled) {
	astar.set_point_disabled(p_id, p_disabled);
}

bool AStar2D::is_point_disabled(int64_t p_id) const {
	return astar.is_point_disabled(p_id);
}

void AStar2D::connect_points(int64_t p_id, int64_t p_with_id, bool p_bidirectional) {
	astar.connect_points(p_id, p_with_id, p_bidirectional);
}

void AStar2D::disconnect_points(int64_t p_id, int64_t p_with_id, bool p_bidirectional) {
	astar.disconnect_points(p_id, p_with_id, p_bidirectional);
}

bool AStar2D::are_points_connected(int64_t p_id, int64_t p_with_id, bool p_bidirectional) const {
	return astar.are_points_connected(p_id, p_with_id, p_bidirectional);
}

int64_t AStar2D::get_point_count() const {
	return astar.get_point_count();
}

int64_t AStar2D::get_point_capacity() const {
	return astar.get_point_capacity();
}

void AStar2D::clear() {
	astar.clear();
}

void AStar2D::reserve_space(int64_t p_num_nodes) {
	astar.reserve_space(p_num_nodes);
}

int64_t AStar2D::get_closest_point(const Vector2 &p_point, bool p_include_disabled) const {
	return astar.get_closest_point(Vector3(p_point.x, p_point.y, 0), p_include_disabled);
}

Vector2 AStar2D::get_closest_position_in_segment(const Vector2 &p_point) const {
	Vector3 p = astar.get_closest_position_in_segment(Vector3(p_point.x, p_point.y, 0));
	return Vector2(p.x, p.y);
}

real_t AStar2D::_estimate_cost(int64_t p_from_id, int64_t p_end_id) {
	real_t scost;
	if (GDVIRTUAL_CALL(_estimate_cost, p_from_id, p_end_id, scost)) {
		return scost;
	}

	AStar3D::Point **from_entry = astar.points.getptr(p_from_id);
	ERR_FAIL_COND_V_MSG(!from_entry, 0, vformat("Can't estimate cost. Point with id: %d doesn't exist.", p_from_id));
	AStar3D::Point *from_point = *from_entry;

	AStar3D::Point **end_entry = astar.points.getptr(p_end_id);
	ERR_FAIL_COND_V_MSG(!end_entry, 0, vformat("Can't estimate cost. Point with id: %d doesn't exist.", p_end_id));
	AStar3D::Point *end_point = *end_entry;

	return from_point->pos.distance_to(end_point->pos);
}

real_t AStar2D::_compute_cost(int64_t p_from_id, int64_t p_to_id) {
	real_t scost;
	if (GDVIRTUAL_CALL(_compute_cost, p_from_id, p_to_id, scost)) {
		return scost;
	}

	AStar3D::Point **from_entry = astar.points.getptr(p_from_id);
	ERR_FAIL_COND_V_MSG(!from_entry, 0, vformat("Can't compute cost. Point with id: %d doesn't exist.", p_from_id));
	AStar3D::Point *from_point = *from_entry;

	AStar3D::Point **to_entry = astar.points.getptr(p_to_id);
	ERR_FAIL_COND_V_MSG(!to_entry, 0, vformat("Can't compute cost. Point with id: %d doesn't exist.", p_to_id));
	AStar3D::Point *to_point = *to_entry;

	return from_point->pos.distance_to(to_point->pos);
}

Vector<Vector2> AStar2D::get_point_path(int64_t p_from_id, int64_t p_to_id, bool p_allow_partial_path) {
	AStar3D::Point **a_entry = astar.points.getptr(p_from_id);
	ERR_FAIL_COND_V_MSG(!a_entry, Vector<Vector2>(), vformat("Can't get point path. Point with id: %d doesn't exist.", p_from_id));
	AStar3D::Point *a = *a_entry;

	AStar3D::Point **b_entry = astar.points.getptr(p_to_id);
	ERR_FAIL_COND_V_MSG(!b_entry, Vector<Vector2>(), vformat("Can't get point path. Point with id: %d doesn't exist.", p_to_id));
	AStar3D::Point *b = *b_entry;

	AStar3D::Point *begin_point = a;
	AStar3D::Point *end_point = b;

	bool found_route = _solve(begin_point, end_point, p_allow_partial_path);
	if (!found_route) {
		if (!p_allow_partial_path || astar.last_closest_point == nullptr) {
			return Vector<Vector2>();
		}

		// Use closest point instead.
		end_point = astar.last_closest_point;
	}

	AStar3D::Point *p = end_point;
	int64_t pc = 1; // Begin point
	while (p != begin_point) {
		pc++;
		p = p->prev_point;
	}

	Vector<Vector2> path;
	path.resize(pc);

	{
		Vector2 *w = path.ptrw();

		AStar3D::Point *p2 = end_point;
		int64_t idx = pc - 1;
		while (p2 != begin_point) {
			w[idx--] = Vector2(p2->pos.x, p2->pos.y);
			p2 = p2->prev_point;
		}

		w[0] = Vector2(p2->pos.x, p2->pos.y); // Assign first
	}

	return path;
}

Vector<int64_t> AStar2D::get_id_path(int64_t p_from_id, int64_t p_to_id, bool p_allow_partial_path) {
	AStar3D::Point **a_entry = astar.points.getptr(p_from_id);
	ERR_FAIL_COND_V_MSG(!a_entry, Vector<int64_t>(), vformat("Can't get id path. Point with id: %d doesn't exist.", p_from_id));
	AStar3D::Point *a = *a_entry;

	AStar3D::Point **to_entry = astar.points.getptr(p_to_id);
	ERR_FAIL_COND_V_MSG(!to_entry, Vector<int64_t>(), vformat("Can't get id path. Point with id: %d doesn't exist.", p_to_id));
	AStar3D::Point *b = *to_entry;

	AStar3D::Point *begin_point = a;
	AStar3D::Point *end_point = b;

	bool found_route = _solve(begin_point, end_point, p_allow_partial_path);
	if (!found_route) {
		if (!p_allow_partial_path || astar.last_closest_point == nullptr) {
			return Vector<int64_t>();
		}

		// Use closest point instead.
		end_point = astar.last_closest_point;
	}

	AStar3D::Point *p = end_point;
	int64_t pc = 1; // Begin point
	while (p != begin_point) {
		pc++;
		p = p->prev_point;
	}

	Vector<int64_t> path;
	path.resize(pc);

	{
		int64_t *w = path.ptrw();

		p = end_point;
		int64_t idx = pc - 1;
		while (p != begin_point) {
			w[idx--] = p->id;
			p = p->prev_point;
		}

		w[0] = p->id; // Assign first
	}

	return path;
}

bool AStar2D::_solve(AStar3D::Point *p_begin_point, AStar3D::Point *p_end_point, bool p_allow_partial_path) {
	astar.last_closest_point = nullptr;
	astar.pass++;

	if (!p_begin_point->enabled) {
		return false;
	}
	if (p_begin_point == p_end_point) {
		return true;
	}
	if (!p_end_point->enabled && !p_allow_partial_path) {
		return false;
	}

	bool found_route = false;

	LocalVector<AStar3D::Point *> open_list;
	SortArray<AStar3D::Point *, AStar3D::SortPoints> sorter;

	p_begin_point->g_score = 0;
	p_begin_point->f_score = _estimate_cost(p_begin_point->id, p_end_point->id);
	p_begin_point->abs_g_score = 0;
	p_begin_point->abs_f_score = _estimate_cost(p_begin_point->id, p_end_point->id);
	open_list.push_back(p_begin_point);

	while (!open_list.is_empty()) {
		AStar3D::Point *p = open_list[0]; // The currently processed point.

		// Find point closer to end_point, or same distance to end_point but closer to begin_point.
		if (astar.last_closest_point == nullptr || astar.last_closest_point->abs_f_score > p->abs_f_score || (astar.last_closest_point->abs_f_score >= p->abs_f_score && astar.last_closest_point->abs_g_score > p->abs_g_score)) {
			astar.last_closest_point = p;
		}

		if (p == p_end_point) {
			found_route = true;
			break;
		}

		sorter.pop_heap(0, open_list.size(), open_list.ptr()); // Remove the current point from the open list.
		open_list.remove_at(open_list.size() - 1);
		p->closed_pass = astar.pass; // Mark the point as closed.

		for (KeyValue<int64_t, AStar3D::Point *> &kv : p->neighbors) {
			AStar3D::Point *e = kv.value; // The neighbor point.

			if (!e->enabled || e->closed_pass == astar.pass) {
				continue;
			}

			if (astar.neighbor_filter_enabled) {
				bool filtered;
				if (GDVIRTUAL_CALL(_filter_neighbor, p->id, e->id, filtered) && filtered) {
					continue;
				}
			}

			real_t tentative_g_score = p->g_score + _compute_cost(p->id, e->id) * e->weight_scale;

			bool new_point = false;

			if (e->open_pass != astar.pass) { // The point wasn't inside the open list.
				e->open_pass = astar.pass;
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

void AStar2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_available_point_id"), &AStar2D::get_available_point_id);
	ClassDB::bind_method(D_METHOD("add_point", "id", "position", "weight_scale"), &AStar2D::add_point, DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("get_point_position", "id"), &AStar2D::get_point_position);
	ClassDB::bind_method(D_METHOD("set_point_position", "id", "position"), &AStar2D::set_point_position);
	ClassDB::bind_method(D_METHOD("get_point_weight_scale", "id"), &AStar2D::get_point_weight_scale);
	ClassDB::bind_method(D_METHOD("set_point_weight_scale", "id", "weight_scale"), &AStar2D::set_point_weight_scale);
	ClassDB::bind_method(D_METHOD("remove_point", "id"), &AStar2D::remove_point);
	ClassDB::bind_method(D_METHOD("has_point", "id"), &AStar2D::has_point);
	ClassDB::bind_method(D_METHOD("get_point_connections", "id"), &AStar2D::get_point_connections);
	ClassDB::bind_method(D_METHOD("get_point_ids"), &AStar2D::get_point_ids);

	ClassDB::bind_method(D_METHOD("set_neighbor_filter_enabled", "enabled"), &AStar2D::set_neighbor_filter_enabled);
	ClassDB::bind_method(D_METHOD("is_neighbor_filter_enabled"), &AStar2D::is_neighbor_filter_enabled);

	ClassDB::bind_method(D_METHOD("set_point_disabled", "id", "disabled"), &AStar2D::set_point_disabled, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("is_point_disabled", "id"), &AStar2D::is_point_disabled);

	ClassDB::bind_method(D_METHOD("connect_points", "id", "to_id", "bidirectional"), &AStar2D::connect_points, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("disconnect_points", "id", "to_id", "bidirectional"), &AStar2D::disconnect_points, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("are_points_connected", "id", "to_id", "bidirectional"), &AStar2D::are_points_connected, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("get_point_count"), &AStar2D::get_point_count);
	ClassDB::bind_method(D_METHOD("get_point_capacity"), &AStar2D::get_point_capacity);
	ClassDB::bind_method(D_METHOD("reserve_space", "num_nodes"), &AStar2D::reserve_space);
	ClassDB::bind_method(D_METHOD("clear"), &AStar2D::clear);

	ClassDB::bind_method(D_METHOD("get_closest_point", "to_position", "include_disabled"), &AStar2D::get_closest_point, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_closest_position_in_segment", "to_position"), &AStar2D::get_closest_position_in_segment);

	ClassDB::bind_method(D_METHOD("get_point_path", "from_id", "to_id", "allow_partial_path"), &AStar2D::get_point_path, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_id_path", "from_id", "to_id", "allow_partial_path"), &AStar2D::get_id_path, DEFVAL(false));

	GDVIRTUAL_BIND(_filter_neighbor, "from_id", "neighbor_id")
	GDVIRTUAL_BIND(_estimate_cost, "from_id", "end_id")
	GDVIRTUAL_BIND(_compute_cost, "from_id", "to_id")

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "neighbor_filter_enabled"), "set_neighbor_filter_enabled", "is_neighbor_filter_enabled");
}
