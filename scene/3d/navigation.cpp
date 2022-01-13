/*************************************************************************/
/*  navigation.cpp                                                       */
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

#include "navigation.h"

#define USE_ENTRY_POINT

void Navigation::_navmesh_link(int p_id) {
	ERR_FAIL_COND(!navmesh_map.has(p_id));
	NavMesh &nm = navmesh_map[p_id];
	ERR_FAIL_COND(nm.linked);
	ERR_FAIL_COND(nm.navmesh.is_null());

	PoolVector<Vector3> vertices = nm.navmesh->get_vertices();
	int len = vertices.size();
	if (len == 0) {
		return;
	}

	PoolVector<Vector3>::Read r = vertices.read();

	for (int i = 0; i < nm.navmesh->get_polygon_count(); i++) {
		//build

		List<Polygon>::Element *P = nm.polygons.push_back(Polygon());
		Polygon &p = P->get();
		p.owner = &nm;

		Vector<int> poly = nm.navmesh->get_polygon(i);
		int plen = poly.size();
		const int *indices = poly.ptr();
		bool valid = true;
		p.edges.resize(plen);

		Vector3 center;
		float sum = 0;

		for (int j = 0; j < plen; j++) {
			int idx = indices[j];
			if (idx < 0 || idx >= len) {
				valid = false;
				break;
			}

			Polygon::Edge e;
			Vector3 ep = nm.xform.xform(r[idx]);
			center += ep;
			e.point = _get_point(ep);
			p.edges.write[j] = e;

			if (j >= 2) {
				Vector3 epa = nm.xform.xform(r[indices[j - 2]]);
				Vector3 epb = nm.xform.xform(r[indices[j - 1]]);

				sum += up.dot((epb - epa).cross(ep - epa));
			}
		}

		p.clockwise = sum > 0;

		if (!valid) {
			nm.polygons.pop_back();
			ERR_CONTINUE(!valid);
		}

		p.center = center;
		if (plen != 0) {
			p.center /= plen;
		}

		//connect

		for (int j = 0; j < plen; j++) {
			int next = (j + 1) % plen;
			EdgeKey ek(p.edges[j].point, p.edges[next].point);

			Map<EdgeKey, Connection>::Element *C = connections.find(ek);
			if (!C) {
				Connection c;
				c.A = &p;
				c.A_edge = j;
				c.B = nullptr;
				c.B_edge = -1;
				connections[ek] = c;
			} else {
				if (C->get().B != nullptr) {
					ConnectionPending pending;
					pending.polygon = &p;
					pending.edge = j;
					p.edges.write[j].P = C->get().pending.push_back(pending);
					continue;
				}

				C->get().B = &p;
				C->get().B_edge = j;
				C->get().A->edges.write[C->get().A_edge].C = &p;
				C->get().A->edges.write[C->get().A_edge].C_edge = j;
				p.edges.write[j].C = C->get().A;
				p.edges.write[j].C_edge = C->get().A_edge;
				//connection successful.
			}
		}
	}

	nm.linked = true;
}

void Navigation::_navmesh_unlink(int p_id) {
	ERR_FAIL_COND(!navmesh_map.has(p_id));
	NavMesh &nm = navmesh_map[p_id];
	ERR_FAIL_COND(!nm.linked);

	for (List<Polygon>::Element *E = nm.polygons.front(); E; E = E->next()) {
		Polygon &p = E->get();

		int edge_count = p.edges.size();
		Polygon::Edge *edges = p.edges.ptrw();

		for (int i = 0; i < edge_count; i++) {
			int next = (i + 1) % edge_count;
			EdgeKey ek(edges[i].point, edges[next].point);
			Map<EdgeKey, Connection>::Element *C = connections.find(ek);

			ERR_CONTINUE(!C);

			if (edges[i].P) {
				C->get().pending.erase(edges[i].P);
				edges[i].P = nullptr;
			} else if (C->get().B) {
				//disconnect

				C->get().B->edges.write[C->get().B_edge].C = nullptr;
				C->get().B->edges.write[C->get().B_edge].C_edge = -1;
				C->get().A->edges.write[C->get().A_edge].C = nullptr;
				C->get().A->edges.write[C->get().A_edge].C_edge = -1;

				if (C->get().A == &E->get()) {
					C->get().A = C->get().B;
					C->get().A_edge = C->get().B_edge;
				}
				C->get().B = nullptr;
				C->get().B_edge = -1;

				if (C->get().pending.size()) {
					//reconnect if something is pending
					ConnectionPending cp = C->get().pending.front()->get();
					C->get().pending.pop_front();

					C->get().B = cp.polygon;
					C->get().B_edge = cp.edge;
					C->get().A->edges.write[C->get().A_edge].C = cp.polygon;
					C->get().A->edges.write[C->get().A_edge].C_edge = cp.edge;
					cp.polygon->edges.write[cp.edge].C = C->get().A;
					cp.polygon->edges.write[cp.edge].C_edge = C->get().A_edge;
					cp.polygon->edges.write[cp.edge].P = nullptr;
				}

			} else {
				connections.erase(C);
				//erase
			}
		}
	}

	nm.polygons.clear();

	nm.linked = false;
}

int Navigation::navmesh_add(const Ref<NavigationMesh> &p_mesh, const Transform &p_xform, Object *p_owner) {
	int id = last_id++;
	NavMesh nm;
	nm.linked = false;
	nm.navmesh = p_mesh;
	nm.xform = p_xform;
	nm.owner = p_owner;
	navmesh_map[id] = nm;

	_navmesh_link(id);

	return id;
}

void Navigation::navmesh_set_transform(int p_id, const Transform &p_xform) {
	ERR_FAIL_COND(!navmesh_map.has(p_id));
	NavMesh &nm = navmesh_map[p_id];
	if (nm.xform == p_xform) {
		return; //bleh
	}
	_navmesh_unlink(p_id);
	nm.xform = p_xform;
	_navmesh_link(p_id);
}
void Navigation::navmesh_remove(int p_id) {
	ERR_FAIL_COND_MSG(!navmesh_map.has(p_id), "Trying to remove nonexisting navmesh with id: " + itos(p_id));
	_navmesh_unlink(p_id);
	navmesh_map.erase(p_id);
}

void Navigation::_clip_path(Vector<Vector3> &path, Polygon *from_poly, const Vector3 &p_to_point, Polygon *p_to_poly) {
	Vector3 from = path[path.size() - 1];

	if (from.distance_to(p_to_point) < CMP_EPSILON) {
		return;
	}
	Plane cut_plane;
	cut_plane.normal = (from - p_to_point).cross(up);
	if (cut_plane.normal == Vector3()) {
		return;
	}
	cut_plane.normal.normalize();
	cut_plane.d = cut_plane.normal.dot(from);

	while (from_poly != p_to_poly) {
		int edge_count = from_poly->edges.size();
		ERR_FAIL_COND_MSG(edge_count == 0, "Polygon has no edges.");
		int pe = from_poly->prev_edge;
		int next = (pe + 1) % edge_count;
		Vector3 a = _get_vertex(from_poly->edges[pe].point);
		Vector3 b = _get_vertex(from_poly->edges[next].point);

		from_poly = from_poly->edges[pe].C;
		ERR_FAIL_COND(!from_poly);

		if (a.distance_to(b) > CMP_EPSILON) {
			Vector3 inters;
			if (cut_plane.intersects_segment(a, b, &inters)) {
				if (inters.distance_to(p_to_point) > CMP_EPSILON && inters.distance_to(path[path.size() - 1]) > CMP_EPSILON) {
					path.push_back(inters);
				}
			}
		}
	}
}

Vector<Vector3> Navigation::get_simple_path(const Vector3 &p_start, const Vector3 &p_end, bool p_optimize) {
	Polygon *begin_poly = nullptr;
	Polygon *end_poly = nullptr;
	Vector3 begin_point;
	Vector3 end_point;
	float begin_d = 1e20;
	float end_d = 1e20;

	for (Map<int, NavMesh>::Element *E = navmesh_map.front(); E; E = E->next()) {
		if (!E->get().linked) {
			continue;
		}
		for (List<Polygon>::Element *F = E->get().polygons.front(); F; F = F->next()) {
			Polygon &p = F->get();
			for (int i = 2; i < p.edges.size(); i++) {
				Face3 f(_get_vertex(p.edges[0].point), _get_vertex(p.edges[i - 1].point), _get_vertex(p.edges[i].point));
				Vector3 spoint = f.get_closest_point_to(p_start);
				float dpoint = spoint.distance_to(p_start);
				if (dpoint < begin_d) {
					begin_d = dpoint;
					begin_poly = &p;
					begin_point = spoint;
				}

				spoint = f.get_closest_point_to(p_end);
				dpoint = spoint.distance_to(p_end);
				if (dpoint < end_d) {
					end_d = dpoint;
					end_poly = &p;
					end_point = spoint;
				}
			}

			p.prev_edge = -1;
		}
	}

	if (!begin_poly || !end_poly) {
		return Vector<Vector3>(); //no path
	}

	if (begin_poly == end_poly) {
		Vector<Vector3> path;
		path.resize(2);
		path.write[0] = begin_point;
		path.write[1] = end_point;
		return path;
	}

	bool found_route = false;

	List<Polygon *> open_list;
	int begin_edge_count = begin_poly->edges.size();

	for (int i = 0; i < begin_edge_count; i++) {
		if (begin_poly->edges[i].C) {
			begin_poly->edges[i].C->prev_edge = begin_poly->edges[i].C_edge;
#ifdef USE_ENTRY_POINT
			int next = (i + 1) % begin_edge_count;
			Vector3 edge[2] = {
				_get_vertex(begin_poly->edges[i].point),
				_get_vertex(begin_poly->edges[next].point)
			};

			Vector3 entry = Geometry::get_closest_point_to_segment(begin_poly->entry, edge);
			begin_poly->edges[i].C->distance = begin_point.distance_to(entry);
			begin_poly->edges[i].C->entry = entry;
#else
			begin_poly->edges[i].C->distance = begin_poly->center.distance_to(begin_poly->edges[i].C->center);
#endif
			open_list.push_back(begin_poly->edges[i].C);
		}
	}

	while (!found_route) {
		if (open_list.size() == 0) {
			break;
		}
		//check open list

		List<Polygon *>::Element *least_cost_poly = nullptr;
		float least_cost = 1e30;

		//this could be faster (cache previous results)
		for (List<Polygon *>::Element *E = open_list.front(); E; E = E->next()) {
			Polygon *p = E->get();

			float cost = p->distance;
#ifdef USE_ENTRY_POINT
			cost += p->entry.distance_to(end_point);
#else
			cost += p->center.distance_to(end_point);
#endif
			if (cost < least_cost) {
				least_cost_poly = E;
				least_cost = cost;
			}
		}

		Polygon *p = least_cost_poly->get();
		//open the neighbours for search

		if (p == end_poly) {
			//oh my reached end! stop algorithm
			found_route = true;
			break;
		}

		int edge_count = p->edges.size();
		for (int i = 0; i < edge_count; i++) {
			Polygon::Edge &e = p->edges.write[i];

			if (!e.C) {
				continue;
			}

#ifdef USE_ENTRY_POINT
			int next = (i + 1) % edge_count;
			Vector3 edge[2] = {
				_get_vertex(p->edges[i].point),
				_get_vertex(p->edges[next].point)
			};

			Vector3 entry = Geometry::get_closest_point_to_segment(p->entry, edge);
			float distance = p->entry.distance_to(entry) + p->distance;
#else
			float distance = p->center.distance_to(e.C->center) + p->distance;
#endif

			if (e.C->prev_edge != -1) {
				//oh this was visited already, can we win the cost?

				if (e.C->distance > distance) {
					e.C->prev_edge = e.C_edge;
					e.C->distance = distance;
#ifdef USE_ENTRY_POINT
					e.C->entry = entry;
#endif
				}
			} else {
				//add to open neighbours

				e.C->prev_edge = e.C_edge;
				e.C->distance = distance;
#ifdef USE_ENTRY_POINT
				e.C->entry = entry;
#endif
				open_list.push_back(e.C);
			}
		}

		open_list.erase(least_cost_poly);
	}

	if (found_route) {
		Vector<Vector3> path;

		if (p_optimize) {
			//string pulling

			Polygon *apex_poly = end_poly;
			Vector3 apex_point = end_point;
			Vector3 portal_left = apex_point;
			Vector3 portal_right = apex_point;
			Polygon *left_poly = end_poly;
			Polygon *right_poly = end_poly;
			Polygon *p = end_poly;
			path.push_back(end_point);

			while (p) {
				Vector3 left;
				Vector3 right;

#define CLOCK_TANGENT(m_a, m_b, m_c) (((m_a) - (m_c)).cross((m_a) - (m_b)))

				if (p == begin_poly) {
					left = begin_point;
					right = begin_point;
				} else {
					int edge_count = p->edges.size();
					ERR_FAIL_COND_V_MSG(edge_count == 0, Vector<Vector3>(), "Polygon has no edges.");
					int prev = p->prev_edge;
					int prev_n = (p->prev_edge + 1) % edge_count;
					left = _get_vertex(p->edges[prev].point);
					right = _get_vertex(p->edges[prev_n].point);

					//if (CLOCK_TANGENT(apex_point,left,(left+right)*0.5).dot(up) < 0){
					if (p->clockwise) {
						SWAP(left, right);
					}
				}

				bool skip = false;

				if (CLOCK_TANGENT(apex_point, portal_left, left).dot(up) >= 0) {
					//process
					if (portal_left == apex_point || CLOCK_TANGENT(apex_point, left, portal_right).dot(up) > 0) {
						left_poly = p;
						portal_left = left;
					} else {
						_clip_path(path, apex_poly, portal_right, right_poly);

						apex_point = portal_right;
						p = right_poly;
						left_poly = p;
						apex_poly = p;
						portal_left = apex_point;
						portal_right = apex_point;
						path.push_back(apex_point);
						skip = true;
					}
				}

				if (!skip && CLOCK_TANGENT(apex_point, portal_right, right).dot(up) <= 0) {
					//process
					if (portal_right == apex_point || CLOCK_TANGENT(apex_point, right, portal_left).dot(up) < 0) {
						right_poly = p;
						portal_right = right;
					} else {
						_clip_path(path, apex_poly, portal_left, left_poly);

						apex_point = portal_left;
						p = left_poly;
						right_poly = p;
						apex_poly = p;
						portal_right = apex_point;
						portal_left = apex_point;
						path.push_back(apex_point);
					}
				}

				if (p != begin_poly) {
					p = p->edges[p->prev_edge].C;
				} else {
					p = nullptr;
				}
			}

			if (path[path.size() - 1] != begin_point) {
				path.push_back(begin_point);
			}

			path.invert();

		} else {
			//midpoints
			Polygon *p = end_poly;

			path.push_back(end_point);
			while (true) {
				int prev = p->prev_edge;
#ifdef USE_ENTRY_POINT
				Vector3 point = p->entry;
#else
				int edge_count = p->edges.size();
				ERR_FAIL_COND_V_MSG(edge_count == 0, Vector<Vector3>(), "Polygon has no edges.");
				int prev_n = (p->prev_edge + 1) % edge_count;
				Vector3 point = (_get_vertex(p->edges[prev].point) + _get_vertex(p->edges[prev_n].point)) * 0.5;
#endif
				path.push_back(point);
				p = p->edges[prev].C;
				if (p == begin_poly) {
					break;
				}
			}

			path.push_back(begin_point);

			path.invert();
		}

		return path;
	}

	return Vector<Vector3>();
}

Vector3 Navigation::get_closest_point_to_segment(const Vector3 &p_from, const Vector3 &p_to, const bool &p_use_collision) {
	bool use_collision = p_use_collision;
	Vector3 closest_point;
	float closest_point_d = 1e20;

	for (Map<int, NavMesh>::Element *E = navmesh_map.front(); E; E = E->next()) {
		if (!E->get().linked) {
			continue;
		}
		for (List<Polygon>::Element *F = E->get().polygons.front(); F; F = F->next()) {
			Polygon &p = F->get();
			for (int i = 2; i < p.edges.size(); i++) {
				Face3 f(_get_vertex(p.edges[0].point), _get_vertex(p.edges[i - 1].point), _get_vertex(p.edges[i].point));
				Vector3 inters;
				if (f.intersects_segment(p_from, p_to, &inters)) {
					if (!use_collision) {
						closest_point = inters;
						use_collision = true;
						closest_point_d = p_from.distance_to(inters);
					} else if (closest_point_d > inters.distance_to(p_from)) {
						closest_point = inters;
						closest_point_d = p_from.distance_to(inters);
					}
				}
			}

			if (!use_collision) {
				int edge_count = p.edges.size();
				for (int i = 0; i < edge_count; i++) {
					Vector3 a, b;
					int next = (i + 1) % edge_count;
					Geometry::get_closest_points_between_segments(p_from, p_to, _get_vertex(p.edges[i].point), _get_vertex(p.edges[next].point), a, b);

					float d = a.distance_to(b);
					if (d < closest_point_d) {
						closest_point_d = d;
						closest_point = b;
					}
				}
			}
		}
	}

	return closest_point;
}

Vector3 Navigation::get_closest_point(const Vector3 &p_point) {
	Vector3 closest_point;
	float closest_point_d = 1e20;

	for (Map<int, NavMesh>::Element *E = navmesh_map.front(); E; E = E->next()) {
		if (!E->get().linked) {
			continue;
		}
		for (List<Polygon>::Element *F = E->get().polygons.front(); F; F = F->next()) {
			Polygon &p = F->get();
			for (int i = 2; i < p.edges.size(); i++) {
				Face3 f(_get_vertex(p.edges[0].point), _get_vertex(p.edges[i - 1].point), _get_vertex(p.edges[i].point));
				Vector3 inters = f.get_closest_point_to(p_point);
				float d = inters.distance_to(p_point);
				if (d < closest_point_d) {
					closest_point = inters;
					closest_point_d = d;
				}
			}
		}
	}

	return closest_point;
}

Vector3 Navigation::get_closest_point_normal(const Vector3 &p_point) {
	Vector3 closest_point;
	Vector3 closest_normal;
	float closest_point_d = 1e20;

	for (Map<int, NavMesh>::Element *E = navmesh_map.front(); E; E = E->next()) {
		if (!E->get().linked) {
			continue;
		}
		for (List<Polygon>::Element *F = E->get().polygons.front(); F; F = F->next()) {
			Polygon &p = F->get();
			for (int i = 2; i < p.edges.size(); i++) {
				Face3 f(_get_vertex(p.edges[0].point), _get_vertex(p.edges[i - 1].point), _get_vertex(p.edges[i].point));
				Vector3 inters = f.get_closest_point_to(p_point);
				float d = inters.distance_to(p_point);
				if (d < closest_point_d) {
					closest_point = inters;
					closest_point_d = d;
					closest_normal = f.get_plane().normal;
				}
			}
		}
	}

	return closest_normal;
}

Object *Navigation::get_closest_point_owner(const Vector3 &p_point) {
	Vector3 closest_point;
	Object *owner = nullptr;
	float closest_point_d = 1e20;

	for (Map<int, NavMesh>::Element *E = navmesh_map.front(); E; E = E->next()) {
		if (!E->get().linked) {
			continue;
		}
		for (List<Polygon>::Element *F = E->get().polygons.front(); F; F = F->next()) {
			Polygon &p = F->get();
			for (int i = 2; i < p.edges.size(); i++) {
				Face3 f(_get_vertex(p.edges[0].point), _get_vertex(p.edges[i - 1].point), _get_vertex(p.edges[i].point));
				Vector3 inters = f.get_closest_point_to(p_point);
				float d = inters.distance_to(p_point);
				if (d < closest_point_d) {
					closest_point = inters;
					closest_point_d = d;
					owner = E->get().owner;
				}
			}
		}
	}

	return owner;
}

void Navigation::set_up_vector(const Vector3 &p_up) {
	up = p_up;
}

Vector3 Navigation::get_up_vector() const {
	return up;
}

void Navigation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("navmesh_add", "mesh", "xform", "owner"), &Navigation::navmesh_add, DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("navmesh_set_transform", "id", "xform"), &Navigation::navmesh_set_transform);
	ClassDB::bind_method(D_METHOD("navmesh_remove", "id"), &Navigation::navmesh_remove);

	ClassDB::bind_method(D_METHOD("get_simple_path", "start", "end", "optimize"), &Navigation::get_simple_path, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_closest_point_to_segment", "start", "end", "use_collision"), &Navigation::get_closest_point_to_segment, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_closest_point", "to_point"), &Navigation::get_closest_point);
	ClassDB::bind_method(D_METHOD("get_closest_point_normal", "to_point"), &Navigation::get_closest_point_normal);
	ClassDB::bind_method(D_METHOD("get_closest_point_owner", "to_point"), &Navigation::get_closest_point_owner);

	ClassDB::bind_method(D_METHOD("set_up_vector", "up"), &Navigation::set_up_vector);
	ClassDB::bind_method(D_METHOD("get_up_vector"), &Navigation::get_up_vector);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "up_vector"), "set_up_vector", "get_up_vector");
}

Navigation::Navigation() {
	ERR_FAIL_COND(sizeof(Point) != 8);
	cell_size = 0.01; //one centimeter
	last_id = 1;
	up = Vector3(0, 1, 0);
}
