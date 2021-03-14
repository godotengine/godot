/*************************************************************************/
/*  polygon_path_finder.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "polygon_path_finder.h"
#include "core/math/geometry_2d.h"

bool PolygonPathFinder::_is_point_inside(const Vector2 &p_point) const {
	int crosses = 0;

	for (Set<Edge>::Element *E = edges.front(); E; E = E->next()) {
		const Edge &e = E->get();

		Vector2 a = points[e.points[0]].pos;
		Vector2 b = points[e.points[1]].pos;

		if (Geometry2D::segment_intersects_segment(a, b, p_point, outside_point, nullptr)) {
			crosses++;
		}
	}

	return crosses & 1;
}

void PolygonPathFinder::setup(const Vector<Vector2> &p_points, const Vector<int> &p_connections) {
	ERR_FAIL_COND(p_connections.size() & 1);

	points.clear();
	edges.clear();

	//insert points

	int point_count = p_points.size();
	points.resize(point_count + 2);
	bounds = Rect2();

	for (int i = 0; i < p_points.size(); i++) {
		points.write[i].pos = p_points[i];
		points.write[i].penalty = 0;

		outside_point.x = i == 0 ? p_points[0].x : (MAX(p_points[i].x, outside_point.x));
		outside_point.y = i == 0 ? p_points[0].y : (MAX(p_points[i].y, outside_point.y));

		if (i == 0) {
			bounds.position = points[i].pos;
		} else {
			bounds.expand_to(points[i].pos);
		}
	}

	outside_point.x += 20.451 + Math::randf() * 10.2039;
	outside_point.y += 21.193 + Math::randf() * 12.5412;

	//insert edges (which are also connections)

	for (int i = 0; i < p_connections.size(); i += 2) {
		Edge e(p_connections[i], p_connections[i + 1]);
		ERR_FAIL_INDEX(e.points[0], point_count);
		ERR_FAIL_INDEX(e.points[1], point_count);
		points.write[p_connections[i]].connections.insert(p_connections[i + 1]);
		points.write[p_connections[i + 1]].connections.insert(p_connections[i]);
		edges.insert(e);
	}

	//fill the remaining connections based on visibility

	for (int i = 0; i < point_count; i++) {
		for (int j = i + 1; j < point_count; j++) {
			if (edges.has(Edge(i, j))) {
				continue; //if in edge ignore
			}

			Vector2 from = points[i].pos;
			Vector2 to = points[j].pos;

			if (!_is_point_inside(from * 0.5 + to * 0.5)) { //connection between points in inside space
				continue;
			}

			bool valid = true;

			for (Set<Edge>::Element *E = edges.front(); E; E = E->next()) {
				const Edge &e = E->get();
				if (e.points[0] == i || e.points[1] == i || e.points[0] == j || e.points[1] == j) {
					continue;
				}

				Vector2 a = points[e.points[0]].pos;
				Vector2 b = points[e.points[1]].pos;

				if (Geometry2D::segment_intersects_segment(a, b, from, to, nullptr)) {
					valid = false;
					break;
				}
			}

			if (valid) {
				points.write[i].connections.insert(j);
				points.write[j].connections.insert(i);
			}
		}
	}
}

Vector<Vector2> PolygonPathFinder::find_path(const Vector2 &p_from, const Vector2 &p_to) {
	Vector<Vector2> path;

	Vector2 from = p_from;
	Vector2 to = p_to;
	Edge ignore_from_edge(-1, -1);
	Edge ignore_to_edge(-1, -1);

	if (!_is_point_inside(from)) {
		float closest_dist = 1e20;
		Vector2 closest_point;

		for (Set<Edge>::Element *E = edges.front(); E; E = E->next()) {
			const Edge &e = E->get();
			Vector2 seg[2] = {
				points[e.points[0]].pos,
				points[e.points[1]].pos
			};

			Vector2 closest = Geometry2D::get_closest_point_to_segment(from, seg);
			float d = from.distance_squared_to(closest);

			if (d < closest_dist) {
				ignore_from_edge = E->get();
				closest_dist = d;
				closest_point = closest;
			}
		}

		from = closest_point;
	};

	if (!_is_point_inside(to)) {
		float closest_dist = 1e20;
		Vector2 closest_point;

		for (Set<Edge>::Element *E = edges.front(); E; E = E->next()) {
			const Edge &e = E->get();
			Vector2 seg[2] = {
				points[e.points[0]].pos,
				points[e.points[1]].pos
			};

			Vector2 closest = Geometry2D::get_closest_point_to_segment(to, seg);
			float d = to.distance_squared_to(closest);

			if (d < closest_dist) {
				ignore_to_edge = E->get();
				closest_dist = d;
				closest_point = closest;
			}
		}

		to = closest_point;
	};

	//test direct connection
	{
		bool can_see_eachother = true;

		for (Set<Edge>::Element *E = edges.front(); E; E = E->next()) {
			const Edge &e = E->get();
			if (e.points[0] == ignore_from_edge.points[0] && e.points[1] == ignore_from_edge.points[1]) {
				continue;
			}
			if (e.points[0] == ignore_to_edge.points[0] && e.points[1] == ignore_to_edge.points[1]) {
				continue;
			}

			Vector2 a = points[e.points[0]].pos;
			Vector2 b = points[e.points[1]].pos;

			if (Geometry2D::segment_intersects_segment(a, b, from, to, nullptr)) {
				can_see_eachother = false;
				break;
			}
		}

		if (can_see_eachother) {
			path.push_back(from);
			path.push_back(to);
			return path;
		}
	}

	//add to graph

	int aidx = points.size() - 2;
	int bidx = points.size() - 1;
	points.write[aidx].pos = from;
	points.write[bidx].pos = to;
	points.write[aidx].distance = 0;
	points.write[bidx].distance = 0;
	points.write[aidx].prev = -1;
	points.write[bidx].prev = -1;
	points.write[aidx].penalty = 0;
	points.write[bidx].penalty = 0;

	for (int i = 0; i < points.size() - 2; i++) {
		bool valid_a = true;
		bool valid_b = true;
		points.write[i].prev = -1;
		points.write[i].distance = 0;

		if (!_is_point_inside(from * 0.5 + points[i].pos * 0.5)) {
			valid_a = false;
		}

		if (!_is_point_inside(to * 0.5 + points[i].pos * 0.5)) {
			valid_b = false;
		}

		for (Set<Edge>::Element *E = edges.front(); E; E = E->next()) {
			const Edge &e = E->get();

			if (e.points[0] == i || e.points[1] == i) {
				continue;
			}

			Vector2 a = points[e.points[0]].pos;
			Vector2 b = points[e.points[1]].pos;

			if (valid_a) {
				if (e.points[0] != ignore_from_edge.points[1] &&
						e.points[1] != ignore_from_edge.points[1] &&
						e.points[0] != ignore_from_edge.points[0] &&
						e.points[1] != ignore_from_edge.points[0]) {
					if (Geometry2D::segment_intersects_segment(a, b, from, points[i].pos, nullptr)) {
						valid_a = false;
					}
				}
			}

			if (valid_b) {
				if (e.points[0] != ignore_to_edge.points[1] &&
						e.points[1] != ignore_to_edge.points[1] &&
						e.points[0] != ignore_to_edge.points[0] &&
						e.points[1] != ignore_to_edge.points[0]) {
					if (Geometry2D::segment_intersects_segment(a, b, to, points[i].pos, nullptr)) {
						valid_b = false;
					}
				}
			}

			if (!valid_a && !valid_b) {
				break;
			}
		}

		if (valid_a) {
			points.write[i].connections.insert(aidx);
			points.write[aidx].connections.insert(i);
		}

		if (valid_b) {
			points.write[i].connections.insert(bidx);
			points.write[bidx].connections.insert(i);
		}
	}
	//solve graph

	Set<int> open_list;

	points.write[aidx].distance = 0;
	points.write[aidx].prev = aidx;
	for (Set<int>::Element *E = points[aidx].connections.front(); E; E = E->next()) {
		open_list.insert(E->get());
		points.write[E->get()].distance = from.distance_to(points[E->get()].pos);
		points.write[E->get()].prev = aidx;
	}

	bool found_route = false;

	while (true) {
		if (open_list.size() == 0) {
			printf("open list empty\n");
			break;
		}
		//check open list

		int least_cost_point = -1;
		float least_cost = 1e30;

		//this could be faster (cache previous results)
		for (Set<int>::Element *E = open_list.front(); E; E = E->next()) {
			const Point &p = points[E->get()];
			float cost = p.distance;
			cost += p.pos.distance_to(to);
			cost += p.penalty;

			if (cost < least_cost) {
				least_cost_point = E->get();
				least_cost = cost;
			}
		}

		const Point &np = points[least_cost_point];
		//open the neighbours for search

		for (Set<int>::Element *E = np.connections.front(); E; E = E->next()) {
			Point &p = points.write[E->get()];
			float distance = np.pos.distance_to(p.pos) + np.distance;

			if (p.prev != -1) {
				//oh this was visited already, can we win the cost?

				if (p.distance > distance) {
					p.prev = least_cost_point; //reassign previous
					p.distance = distance;
				}
			} else {
				//add to open neighbours

				p.prev = least_cost_point;
				p.distance = distance;
				open_list.insert(E->get());

				if (E->get() == bidx) {
					//oh my reached end! stop algorithm
					found_route = true;
					break;
				}
			}
		}

		if (found_route) {
			break;
		}

		open_list.erase(least_cost_point);
	}

	if (found_route) {
		int at = bidx;
		path.push_back(points[at].pos);
		do {
			at = points[at].prev;
			path.push_back(points[at].pos);
		} while (at != aidx);

		path.reverse();
	}

	for (int i = 0; i < points.size() - 2; i++) {
		points.write[i].connections.erase(aidx);
		points.write[i].connections.erase(bidx);
		points.write[i].prev = -1;
		points.write[i].distance = 0;
	}

	points.write[aidx].connections.clear();
	points.write[aidx].prev = -1;
	points.write[aidx].distance = 0;
	points.write[bidx].connections.clear();
	points.write[bidx].prev = -1;
	points.write[bidx].distance = 0;

	return path;
}

void PolygonPathFinder::_set_data(const Dictionary &p_data) {
	ERR_FAIL_COND(!p_data.has("points"));
	ERR_FAIL_COND(!p_data.has("connections"));
	ERR_FAIL_COND(!p_data.has("segments"));
	ERR_FAIL_COND(!p_data.has("bounds"));

	Vector<Vector2> p = p_data["points"];
	Array c = p_data["connections"];

	ERR_FAIL_COND(c.size() != p.size());
	if (c.size()) {
		return;
	}

	int pc = p.size();
	points.resize(pc + 2);

	const Vector2 *pr = p.ptr();
	for (int i = 0; i < pc; i++) {
		points.write[i].pos = pr[i];
		Vector<int> con = c[i];
		const int *cr = con.ptr();
		int cc = con.size();
		for (int j = 0; j < cc; j++) {
			points.write[i].connections.insert(cr[j]);
		}
	}

	if (p_data.has("penalties")) {
		Vector<float> penalties = p_data["penalties"];
		if (penalties.size() == pc) {
			const float *pr2 = penalties.ptr();
			for (int i = 0; i < pc; i++) {
				points.write[i].penalty = pr2[i];
			}
		}
	}

	Vector<int> segs = p_data["segments"];
	int sc = segs.size();
	ERR_FAIL_COND(sc & 1);
	const int *sr = segs.ptr();
	for (int i = 0; i < sc; i += 2) {
		Edge e(sr[i], sr[i + 1]);
		edges.insert(e);
	}
	bounds = p_data["bounds"];
}

Dictionary PolygonPathFinder::_get_data() const {
	Dictionary d;
	Vector<Vector2> p;
	Vector<int> ind;
	Array connections;
	p.resize(MAX(0, points.size() - 2));
	connections.resize(MAX(0, points.size() - 2));
	ind.resize(edges.size() * 2);
	Vector<float> penalties;
	penalties.resize(MAX(0, points.size() - 2));
	{
		Vector2 *wp = p.ptrw();
		float *pw = penalties.ptrw();

		for (int i = 0; i < points.size() - 2; i++) {
			wp[i] = points[i].pos;
			pw[i] = points[i].penalty;
			Vector<int> c;
			c.resize(points[i].connections.size());
			{
				int *cw = c.ptrw();
				int idx = 0;
				for (Set<int>::Element *E = points[i].connections.front(); E; E = E->next()) {
					cw[idx++] = E->get();
				}
			}
			connections[i] = c;
		}
	}
	{
		int *iw = ind.ptrw();
		int idx = 0;
		for (Set<Edge>::Element *E = edges.front(); E; E = E->next()) {
			iw[idx++] = E->get().points[0];
			iw[idx++] = E->get().points[1];
		}
	}

	d["bounds"] = bounds;
	d["points"] = p;
	d["penalties"] = penalties;
	d["connections"] = connections;
	d["segments"] = ind;

	return d;
}

bool PolygonPathFinder::is_point_inside(const Vector2 &p_point) const {
	return _is_point_inside(p_point);
}

Vector2 PolygonPathFinder::get_closest_point(const Vector2 &p_point) const {
	float closest_dist = 1e20;
	Vector2 closest_point;

	for (Set<Edge>::Element *E = edges.front(); E; E = E->next()) {
		const Edge &e = E->get();
		Vector2 seg[2] = {
			points[e.points[0]].pos,
			points[e.points[1]].pos
		};

		Vector2 closest = Geometry2D::get_closest_point_to_segment(p_point, seg);
		float d = p_point.distance_squared_to(closest);

		if (d < closest_dist) {
			closest_dist = d;
			closest_point = closest;
		}
	}

	ERR_FAIL_COND_V(closest_dist == 1e20, Vector2());

	return closest_point;
}

Vector<Vector2> PolygonPathFinder::get_intersections(const Vector2 &p_from, const Vector2 &p_to) const {
	Vector<Vector2> inters;

	for (Set<Edge>::Element *E = edges.front(); E; E = E->next()) {
		Vector2 a = points[E->get().points[0]].pos;
		Vector2 b = points[E->get().points[1]].pos;

		Vector2 res;
		if (Geometry2D::segment_intersects_segment(a, b, p_from, p_to, &res)) {
			inters.push_back(res);
		}
	}

	return inters;
}

Rect2 PolygonPathFinder::get_bounds() const {
	return bounds;
}

void PolygonPathFinder::set_point_penalty(int p_point, float p_penalty) {
	ERR_FAIL_INDEX(p_point, points.size() - 2);
	points.write[p_point].penalty = p_penalty;
}

float PolygonPathFinder::get_point_penalty(int p_point) const {
	ERR_FAIL_INDEX_V(p_point, points.size() - 2, 0);
	return points[p_point].penalty;
}

void PolygonPathFinder::_bind_methods() {
	ClassDB::bind_method(D_METHOD("setup", "points", "connections"), &PolygonPathFinder::setup);
	ClassDB::bind_method(D_METHOD("find_path", "from", "to"), &PolygonPathFinder::find_path);
	ClassDB::bind_method(D_METHOD("get_intersections", "from", "to"), &PolygonPathFinder::get_intersections);
	ClassDB::bind_method(D_METHOD("get_closest_point", "point"), &PolygonPathFinder::get_closest_point);
	ClassDB::bind_method(D_METHOD("is_point_inside", "point"), &PolygonPathFinder::is_point_inside);
	ClassDB::bind_method(D_METHOD("set_point_penalty", "idx", "penalty"), &PolygonPathFinder::set_point_penalty);
	ClassDB::bind_method(D_METHOD("get_point_penalty", "idx"), &PolygonPathFinder::get_point_penalty);

	ClassDB::bind_method(D_METHOD("get_bounds"), &PolygonPathFinder::get_bounds);
	ClassDB::bind_method(D_METHOD("_set_data"), &PolygonPathFinder::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &PolygonPathFinder::_get_data);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");
}

PolygonPathFinder::PolygonPathFinder() {
}
