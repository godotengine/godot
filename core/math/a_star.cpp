/*************************************************************************/
/*  a_star.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "a_star.h"
#include "geometry.h"
#include "scene/scene_string_names.h"
#include "script_language.h"

int AStar::get_available_point_id() const {

	if (points.empty()) {
		return 1;
	}

	return points.back()->key() + 1;
}

void AStar::add_point(int p_id, const Vector3 &p_pos, real_t p_weight_scale) {

	ERR_FAIL_COND(p_id < 0);
	ERR_FAIL_COND(p_weight_scale < 1);

	if (!points.has(p_id)) {
		Point *pt = memnew(Point);
		pt->id = p_id;
		pt->pos = p_pos;
		pt->weight_scale = p_weight_scale;
		pt->prev_point = NULL;
		pt->last_pass = 0;
		points[p_id] = pt;
	} else {
		points[p_id]->pos = p_pos;
		points[p_id]->weight_scale = p_weight_scale;
	}
}

Vector3 AStar::get_point_position(int p_id) const {

	ERR_FAIL_COND_V(!points.has(p_id), Vector3());

	return points[p_id]->pos;
}

void AStar::set_point_position(int p_id, const Vector3 &p_pos) {

	ERR_FAIL_COND(!points.has(p_id));

	points[p_id]->pos = p_pos;
}

real_t AStar::get_point_weight_scale(int p_id) const {

	ERR_FAIL_COND_V(!points.has(p_id), 0);

	return points[p_id]->weight_scale;
}

void AStar::set_point_weight_scale(int p_id, real_t p_weight_scale) {

	ERR_FAIL_COND(!points.has(p_id));
	ERR_FAIL_COND(p_weight_scale < 1);

	points[p_id]->weight_scale = p_weight_scale;
}

void AStar::remove_point(int p_id) {

	ERR_FAIL_COND(!points.has(p_id));

	Point *p = points[p_id];

	for (int i = 0; i < p->neighbours.size(); i++) {

		Segment s(p_id, p->neighbours[i]->id);
		segments.erase(s);
		p->neighbours[i]->neighbours.erase(p);
	}

	memdelete(p);
	points.erase(p_id);
}

void AStar::connect_points(int p_id, int p_with_id, bool bidirectional) {

	ERR_FAIL_COND(!points.has(p_id));
	ERR_FAIL_COND(!points.has(p_with_id));
	ERR_FAIL_COND(p_id == p_with_id);

	Point *a = points[p_id];
	Point *b = points[p_with_id];
	a->neighbours.push_back(b);

	if (bidirectional)
		b->neighbours.push_back(a);

	Segment s(p_id, p_with_id);
	if (s.from == p_id) {
		s.from_point = a;
		s.to_point = b;
	} else {
		s.from_point = b;
		s.to_point = a;
	}

	segments.insert(s);
}
void AStar::disconnect_points(int p_id, int p_with_id) {

	Segment s(p_id, p_with_id);
	ERR_FAIL_COND(!segments.has(s));

	segments.erase(s);

	Point *a = points[p_id];
	Point *b = points[p_with_id];
	a->neighbours.erase(b);
	b->neighbours.erase(a);
}

bool AStar::has_point(int p_id) const {

	return points.has(p_id);
}

Array AStar::get_points() {

	Array point_list;

	for (const Map<int, Point *>::Element *E = points.front(); E; E = E->next()) {
		point_list.push_back(E->key());
	}

	return point_list;
}

PoolVector<int> AStar::get_point_connections(int p_id) {

	ERR_FAIL_COND_V(!points.has(p_id), PoolVector<int>());

	PoolVector<int> point_list;

	Point *p = points[p_id];

	for (int i = 0; i < p->neighbours.size(); i++) {
		point_list.push_back(p->neighbours[i]->id);
	}

	return point_list;
}

bool AStar::are_points_connected(int p_id, int p_with_id) const {

	Segment s(p_id, p_with_id);
	return segments.has(s);
}

void AStar::clear() {

	for (const Map<int, Point *>::Element *E = points.front(); E; E = E->next()) {

		memdelete(E->get());
	}
	segments.clear();
	points.clear();
}

int AStar::get_closest_point(const Vector3 &p_point) const {

	int closest_id = -1;
	real_t closest_dist = 1e20;

	for (const Map<int, Point *>::Element *E = points.front(); E; E = E->next()) {

		real_t d = p_point.distance_squared_to(E->get()->pos);
		if (closest_id < 0 || d < closest_dist) {
			closest_dist = d;
			closest_id = E->key();
		}
	}

	return closest_id;
}

Vector3 AStar::get_closest_position_in_segment(const Vector3 &p_point) const {

	real_t closest_dist = 1e20;
	bool found = false;
	Vector3 closest_point;

	for (const Set<Segment>::Element *E = segments.front(); E; E = E->next()) {

		Vector3 segment[2] = {
			E->get().from_point->pos,
			E->get().to_point->pos,
		};

		Vector3 p = Geometry::get_closest_point_to_segment(p_point, segment);
		real_t d = p_point.distance_squared_to(p);
		if (!found || d < closest_dist) {

			closest_point = p;
			closest_dist = d;
			found = true;
		}
	}

	return closest_point;
}

bool AStar::_solve(Point *begin_point, Point *end_point) {

	pass++;

	SelfList<Point>::List open_list;

	bool found_route = false;

	for (int i = 0; i < begin_point->neighbours.size(); i++) {

		Point *n = begin_point->neighbours[i];
		n->prev_point = begin_point;
		n->distance = _compute_cost(begin_point->id, n->id) * n->weight_scale;
		n->last_pass = pass;
		open_list.add(&n->list);

		if (end_point == n) {
			found_route = true;
			break;
		}
	}

	while (!found_route) {

		if (open_list.first() == NULL) {
			// No path found
			break;
		}
		// Check open list

		SelfList<Point> *least_cost_point = NULL;
		real_t least_cost = 1e30;

		// TODO: Cache previous results
		for (SelfList<Point> *E = open_list.first(); E; E = E->next()) {

			Point *p = E->self();

			real_t cost = p->distance;
			cost += _estimate_cost(p->id, end_point->id);

			if (cost < least_cost) {

				least_cost_point = E;
				least_cost = cost;
			}
		}

		Point *p = least_cost_point->self();
		// Open the neighbours for search
		int es = p->neighbours.size();

		for (int i = 0; i < es; i++) {

			Point *e = p->neighbours[i];

			real_t distance = _compute_cost(p->id, e->id) * e->weight_scale + p->distance;

			if (e->last_pass == pass) {
				// Already visited, is this cheaper?

				if (e->distance > distance) {

					e->prev_point = p;
					e->distance = distance;
				}
			} else {
				// Add to open neighbours

				e->prev_point = p;
				e->distance = distance;
				e->last_pass = pass; // Mark as used
				open_list.add(&e->list);

				if (e == end_point) {
					// End reached; stop algorithm
					found_route = true;
					break;
				}
			}
		}

		if (found_route)
			break;

		open_list.remove(least_cost_point);
	}

	// Clear the openf list
	while (open_list.first()) {
		open_list.remove(open_list.first());
	}

	return found_route;
}

float AStar::_estimate_cost(int p_from_id, int p_to_id) {

	if (get_script_instance() && get_script_instance()->has_method(SceneStringNames::get_singleton()->_estimate_cost))
		return get_script_instance()->call(SceneStringNames::get_singleton()->_estimate_cost, p_from_id, p_to_id);

	return points[p_from_id]->pos.distance_to(points[p_to_id]->pos);
}

float AStar::_compute_cost(int p_from_id, int p_to_id) {

	if (get_script_instance() && get_script_instance()->has_method(SceneStringNames::get_singleton()->_compute_cost))
		return get_script_instance()->call(SceneStringNames::get_singleton()->_compute_cost, p_from_id, p_to_id);

	return points[p_from_id]->pos.distance_to(points[p_to_id]->pos);
}

PoolVector<Vector3> AStar::get_point_path(int p_from_id, int p_to_id) {

	ERR_FAIL_COND_V(!points.has(p_from_id), PoolVector<Vector3>());
	ERR_FAIL_COND_V(!points.has(p_to_id), PoolVector<Vector3>());

	pass++;

	Point *a = points[p_from_id];
	Point *b = points[p_to_id];

	if (a == b) {
		PoolVector<Vector3> ret;
		ret.push_back(a->pos);
		return ret;
	}

	Point *begin_point = a;
	Point *end_point = b;

	bool found_route = _solve(begin_point, end_point);

	if (!found_route)
		return PoolVector<Vector3>();

	// Midpoints
	Point *p = end_point;
	int pc = 1; // Begin point
	while (p != begin_point) {
		pc++;
		p = p->prev_point;
	}

	PoolVector<Vector3> path;
	path.resize(pc);

	{
		PoolVector<Vector3>::Write w = path.write();

		Point *p = end_point;
		int idx = pc - 1;
		while (p != begin_point) {
			w[idx--] = p->pos;
			p = p->prev_point;
		}

		w[0] = p->pos; // Assign first
	}

	return path;
}

PoolVector<int> AStar::get_id_path(int p_from_id, int p_to_id) {

	ERR_FAIL_COND_V(!points.has(p_from_id), PoolVector<int>());
	ERR_FAIL_COND_V(!points.has(p_to_id), PoolVector<int>());

	pass++;

	Point *a = points[p_from_id];
	Point *b = points[p_to_id];

	if (a == b) {
		PoolVector<int> ret;
		ret.push_back(a->id);
		return ret;
	}

	Point *begin_point = a;
	Point *end_point = b;

	bool found_route = _solve(begin_point, end_point);

	if (!found_route)
		return PoolVector<int>();

	// Midpoints
	Point *p = end_point;
	int pc = 1; // Begin point
	while (p != begin_point) {
		pc++;
		p = p->prev_point;
	}

	PoolVector<int> path;
	path.resize(pc);

	{
		PoolVector<int>::Write w = path.write();

		p = end_point;
		int idx = pc - 1;
		while (p != begin_point) {
			w[idx--] = p->id;
			p = p->prev_point;
		}

		w[0] = p->id; // Assign first
	}

	return path;
}

void AStar::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_available_point_id"), &AStar::get_available_point_id);
	ClassDB::bind_method(D_METHOD("add_point", "id", "position", "weight_scale"), &AStar::add_point, DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("get_point_position", "id"), &AStar::get_point_position);
	ClassDB::bind_method(D_METHOD("set_point_position", "id", "position"), &AStar::set_point_position);
	ClassDB::bind_method(D_METHOD("get_point_weight_scale", "id"), &AStar::get_point_weight_scale);
	ClassDB::bind_method(D_METHOD("set_point_weight_scale", "id", "weight_scale"), &AStar::set_point_weight_scale);
	ClassDB::bind_method(D_METHOD("remove_point", "id"), &AStar::remove_point);
	ClassDB::bind_method(D_METHOD("has_point", "id"), &AStar::has_point);
	ClassDB::bind_method(D_METHOD("get_points"), &AStar::get_points);

	ClassDB::bind_method(D_METHOD("get_point_connections"), &AStar::get_point_connections);

	ClassDB::bind_method(D_METHOD("connect_points", "id", "to_id", "bidirectional"), &AStar::connect_points, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("disconnect_points", "id", "to_id"), &AStar::disconnect_points);
	ClassDB::bind_method(D_METHOD("are_points_connected", "id", "to_id"), &AStar::are_points_connected);

	ClassDB::bind_method(D_METHOD("clear"), &AStar::clear);

	ClassDB::bind_method(D_METHOD("get_closest_point", "to_position"), &AStar::get_closest_point);
	ClassDB::bind_method(D_METHOD("get_closest_position_in_segment", "to_position"), &AStar::get_closest_position_in_segment);

	ClassDB::bind_method(D_METHOD("get_point_path", "from_id", "to_id"), &AStar::get_point_path);
	ClassDB::bind_method(D_METHOD("get_id_path", "from_id", "to_id"), &AStar::get_id_path);

	BIND_VMETHOD(MethodInfo(Variant::REAL, "_estimate_cost", PropertyInfo(Variant::INT, "from_id"), PropertyInfo(Variant::INT, "to_id")));
	BIND_VMETHOD(MethodInfo(Variant::REAL, "_compute_cost", PropertyInfo(Variant::INT, "from_id"), PropertyInfo(Variant::INT, "to_id")));
}

AStar::AStar() {

	pass = 1;
}

AStar::~AStar() {

	pass = 1;
	clear();
}
