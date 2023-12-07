/**************************************************************************/
/*  nav_map.cpp                                                           */
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

#include "nav_map.h"

#include "nav_agent.h"
#include "nav_link.h"
#include "nav_obstacle.h"
#include "nav_region.h"

#include "core/config/project_settings.h"
#include "core/object/worker_thread_pool.h"

#include <Obstacle2d.h>

#define THREE_POINTS_CROSS_PRODUCT(m_a, m_b, m_c) (((m_c) - (m_a)).cross((m_b) - (m_a)))

// Helper macro
#define APPEND_METADATA(poly)                                  \
	if (r_path_types) {                                        \
		r_path_types->push_back(poly->owner->get_type());      \
	}                                                          \
	if (r_path_rids) {                                         \
		r_path_rids->push_back(poly->owner->get_self());       \
	}                                                          \
	if (r_path_owners) {                                       \
		r_path_owners->push_back(poly->owner->get_owner_id()); \
	}

void NavMap::set_up(Vector3 p_up) {
	if (up == p_up) {
		return;
	}
	up = p_up;
	regenerate_polygons = true;
}

void NavMap::set_cell_size(real_t p_cell_size) {
	if (cell_size == p_cell_size) {
		return;
	}
	cell_size = p_cell_size;
	regenerate_polygons = true;
}

void NavMap::set_cell_height(real_t p_cell_height) {
	if (cell_height == p_cell_height) {
		return;
	}
	cell_height = p_cell_height;
	regenerate_polygons = true;
}

void NavMap::set_use_edge_connections(bool p_enabled) {
	if (use_edge_connections == p_enabled) {
		return;
	}
	use_edge_connections = p_enabled;
	regenerate_links = true;
}

void NavMap::set_edge_connection_margin(real_t p_edge_connection_margin) {
	if (edge_connection_margin == p_edge_connection_margin) {
		return;
	}
	edge_connection_margin = p_edge_connection_margin;
	regenerate_links = true;
}

void NavMap::set_link_connection_radius(real_t p_link_connection_radius) {
	if (link_connection_radius == p_link_connection_radius) {
		return;
	}
	link_connection_radius = p_link_connection_radius;
	regenerate_links = true;
}

gd::PointKey NavMap::get_point_key(const Vector3 &p_pos) const {
	const int x = static_cast<int>(Math::floor(p_pos.x / cell_size));
	const int y = static_cast<int>(Math::floor(p_pos.y / cell_height));
	const int z = static_cast<int>(Math::floor(p_pos.z / cell_size));

	gd::PointKey p;
	p.key = 0;
	p.x = x;
	p.y = y;
	p.z = z;
	return p;
}

Vector<Vector3> NavMap::get_path(Vector3 p_origin, Vector3 p_destination, bool p_optimize, uint32_t p_navigation_layers, Vector<int32_t> *r_path_types, TypedArray<RID> *r_path_rids, Vector<int64_t> *r_path_owners) const {
	ERR_FAIL_COND_V_MSG(map_update_id == 0, Vector<Vector3>(), "NavigationServer map query failed because it was made before first map synchronization.");
	// Clear metadata outputs.
	if (r_path_types) {
		r_path_types->clear();
	}
	if (r_path_rids) {
		r_path_rids->clear();
	}
	if (r_path_owners) {
		r_path_owners->clear();
	}

	// Find the start poly and the end poly on this map.
	const gd::Polygon *begin_poly = nullptr;
	const gd::Polygon *end_poly = nullptr;
	Vector3 begin_point;
	Vector3 end_point;
	real_t begin_d = FLT_MAX;
	real_t end_d = FLT_MAX;
	// Find the initial poly and the end poly on this map.
	for (const gd::Polygon &p : polygons) {
		// Only consider the polygon if it in a region with compatible layers.
		if ((p_navigation_layers & p.owner->get_navigation_layers()) == 0) {
			continue;
		}

		// For each face check the distance between the origin/destination
		for (size_t point_id = 2; point_id < p.points.size(); point_id++) {
			const Face3 face(p.points[0].pos, p.points[point_id - 1].pos, p.points[point_id].pos);

			Vector3 point = face.get_closest_point_to(p_origin);
			real_t distance_to_point = point.distance_to(p_origin);
			if (distance_to_point < begin_d) {
				begin_d = distance_to_point;
				begin_poly = &p;
				begin_point = point;
			}

			point = face.get_closest_point_to(p_destination);
			distance_to_point = point.distance_to(p_destination);
			if (distance_to_point < end_d) {
				end_d = distance_to_point;
				end_poly = &p;
				end_point = point;
			}
		}
	}

	// Check for trivial cases
	if (!begin_poly || !end_poly) {
		return Vector<Vector3>();
	}
	if (begin_poly == end_poly) {
		if (r_path_types) {
			r_path_types->resize(2);
			r_path_types->write[0] = begin_poly->owner->get_type();
			r_path_types->write[1] = end_poly->owner->get_type();
		}

		if (r_path_rids) {
			r_path_rids->resize(2);
			(*r_path_rids)[0] = begin_poly->owner->get_self();
			(*r_path_rids)[1] = end_poly->owner->get_self();
		}

		if (r_path_owners) {
			r_path_owners->resize(2);
			r_path_owners->write[0] = begin_poly->owner->get_owner_id();
			r_path_owners->write[1] = end_poly->owner->get_owner_id();
		}

		Vector<Vector3> path;
		path.resize(2);
		path.write[0] = begin_point;
		path.write[1] = end_point;
		return path;
	}

	// List of all reachable navigation polys.
	LocalVector<gd::NavigationPoly> navigation_polys;
	navigation_polys.reserve(polygons.size() * 0.75);

	// Add the start polygon to the reachable navigation polygons.
	gd::NavigationPoly begin_navigation_poly = gd::NavigationPoly(begin_poly);
	begin_navigation_poly.self_id = 0;
	begin_navigation_poly.entry = begin_point;
	begin_navigation_poly.back_navigation_edge_pathway_start = begin_point;
	begin_navigation_poly.back_navigation_edge_pathway_end = begin_point;
	navigation_polys.push_back(begin_navigation_poly);

	// List of polygon IDs to visit.
	List<uint32_t> to_visit;
	to_visit.push_back(0);

	// This is an implementation of the A* algorithm.
	int least_cost_id = 0;
	int prev_least_cost_id = -1;
	bool found_route = false;

	const gd::Polygon *reachable_end = nullptr;
	real_t reachable_d = FLT_MAX;
	bool is_reachable = true;

	while (true) {
		// Takes the current least_cost_poly neighbors (iterating over its edges) and compute the traveled_distance.
		for (const gd::Edge &edge : navigation_polys[least_cost_id].poly->edges) {
			// Iterate over connections in this edge, then compute the new optimized travel distance assigned to this polygon.
			for (int connection_index = 0; connection_index < edge.connections.size(); connection_index++) {
				const gd::Edge::Connection &connection = edge.connections[connection_index];

				// Only consider the connection to another polygon if this polygon is in a region with compatible layers.
				if ((p_navigation_layers & connection.polygon->owner->get_navigation_layers()) == 0) {
					continue;
				}

				const gd::NavigationPoly &least_cost_poly = navigation_polys[least_cost_id];
				real_t poly_enter_cost = 0.0;
				real_t poly_travel_cost = least_cost_poly.poly->owner->get_travel_cost();

				if (prev_least_cost_id != -1 && (navigation_polys[prev_least_cost_id].poly->owner->get_self() != least_cost_poly.poly->owner->get_self())) {
					poly_enter_cost = least_cost_poly.poly->owner->get_enter_cost();
				}
				prev_least_cost_id = least_cost_id;

				Vector3 pathway[2] = { connection.pathway_start, connection.pathway_end };
				const Vector3 new_entry = Geometry3D::get_closest_point_to_segment(least_cost_poly.entry, pathway);
				const real_t new_distance = (least_cost_poly.entry.distance_to(new_entry) * poly_travel_cost) + poly_enter_cost + least_cost_poly.traveled_distance;

				int64_t already_visited_polygon_index = navigation_polys.find(gd::NavigationPoly(connection.polygon));

				if (already_visited_polygon_index != -1) {
					// Polygon already visited, check if we can reduce the travel cost.
					gd::NavigationPoly &avp = navigation_polys[already_visited_polygon_index];
					if (new_distance < avp.traveled_distance) {
						avp.back_navigation_poly_id = least_cost_id;
						avp.back_navigation_edge = connection.edge;
						avp.back_navigation_edge_pathway_start = connection.pathway_start;
						avp.back_navigation_edge_pathway_end = connection.pathway_end;
						avp.traveled_distance = new_distance;
						avp.entry = new_entry;
					}
				} else {
					// Add the neighbor polygon to the reachable ones.
					gd::NavigationPoly new_navigation_poly = gd::NavigationPoly(connection.polygon);
					new_navigation_poly.self_id = navigation_polys.size();
					new_navigation_poly.back_navigation_poly_id = least_cost_id;
					new_navigation_poly.back_navigation_edge = connection.edge;
					new_navigation_poly.back_navigation_edge_pathway_start = connection.pathway_start;
					new_navigation_poly.back_navigation_edge_pathway_end = connection.pathway_end;
					new_navigation_poly.traveled_distance = new_distance;
					new_navigation_poly.entry = new_entry;
					navigation_polys.push_back(new_navigation_poly);

					// Add the neighbor polygon to the polygons to visit.
					to_visit.push_back(navigation_polys.size() - 1);
				}
			}
		}

		// Removes the least cost polygon from the list of polygons to visit so we can advance.
		to_visit.erase(least_cost_id);

		// When the list of polygons to visit is empty at this point it means the End Polygon is not reachable
		if (to_visit.size() == 0) {
			// Thus use the further reachable polygon
			ERR_BREAK_MSG(is_reachable == false, "It's not expect to not find the most reachable polygons");
			is_reachable = false;
			if (reachable_end == nullptr) {
				// The path is not found and there is not a way out.
				break;
			}

			// Set as end point the furthest reachable point.
			end_poly = reachable_end;
			end_d = FLT_MAX;
			for (size_t point_id = 2; point_id < end_poly->points.size(); point_id++) {
				Face3 f(end_poly->points[0].pos, end_poly->points[point_id - 1].pos, end_poly->points[point_id].pos);
				Vector3 spoint = f.get_closest_point_to(p_destination);
				real_t dpoint = spoint.distance_to(p_destination);
				if (dpoint < end_d) {
					end_point = spoint;
					end_d = dpoint;
				}
			}

			// Search all faces of start polygon as well.
			bool closest_point_on_start_poly = false;
			for (size_t point_id = 2; point_id < begin_poly->points.size(); point_id++) {
				Face3 f(begin_poly->points[0].pos, begin_poly->points[point_id - 1].pos, begin_poly->points[point_id].pos);
				Vector3 spoint = f.get_closest_point_to(p_destination);
				real_t dpoint = spoint.distance_to(p_destination);
				if (dpoint < end_d) {
					end_point = spoint;
					end_d = dpoint;
					closest_point_on_start_poly = true;
				}
			}

			if (closest_point_on_start_poly) {
				// No point to run PostProcessing when start and end convex polygon is the same.
				if (r_path_types) {
					r_path_types->resize(2);
					r_path_types->write[0] = begin_poly->owner->get_type();
					r_path_types->write[1] = begin_poly->owner->get_type();
				}

				if (r_path_rids) {
					r_path_rids->resize(2);
					(*r_path_rids)[0] = begin_poly->owner->get_self();
					(*r_path_rids)[1] = begin_poly->owner->get_self();
				}

				if (r_path_owners) {
					r_path_owners->resize(2);
					r_path_owners->write[0] = begin_poly->owner->get_owner_id();
					r_path_owners->write[1] = begin_poly->owner->get_owner_id();
				}

				Vector<Vector3> path;
				path.resize(2);
				path.write[0] = begin_point;
				path.write[1] = end_point;
				return path;
			}

			// Reset open and navigation_polys
			gd::NavigationPoly np = navigation_polys[0];
			navigation_polys.clear();
			navigation_polys.push_back(np);
			to_visit.clear();
			to_visit.push_back(0);
			least_cost_id = 0;
			prev_least_cost_id = -1;

			reachable_end = nullptr;

			continue;
		}

		// Find the polygon with the minimum cost from the list of polygons to visit.
		least_cost_id = -1;
		real_t least_cost = FLT_MAX;
		for (List<uint32_t>::Element *element = to_visit.front(); element != nullptr; element = element->next()) {
			gd::NavigationPoly *np = &navigation_polys[element->get()];
			real_t cost = np->traveled_distance;
			cost += (np->entry.distance_to(end_point) * np->poly->owner->get_travel_cost());
			if (cost < least_cost) {
				least_cost_id = np->self_id;
				least_cost = cost;
			}
		}

		ERR_BREAK(least_cost_id == -1);

		// Stores the further reachable end polygon, in case our goal is not reachable.
		if (is_reachable) {
			real_t d = navigation_polys[least_cost_id].entry.distance_to(p_destination) * navigation_polys[least_cost_id].poly->owner->get_travel_cost();
			if (reachable_d > d) {
				reachable_d = d;
				reachable_end = navigation_polys[least_cost_id].poly;
			}
		}

		// Check if we reached the end
		if (navigation_polys[least_cost_id].poly == end_poly) {
			found_route = true;
			break;
		}
	}

	// We did not find a route but we have both a start polygon and an end polygon at this point.
	// Usually this happens because there was not a single external or internal connected edge, e.g. our start polygon is an isolated, single convex polygon.
	if (!found_route) {
		end_d = FLT_MAX;
		// Search all faces of the start polygon for the closest point to our target position.
		for (size_t point_id = 2; point_id < begin_poly->points.size(); point_id++) {
			Face3 f(begin_poly->points[0].pos, begin_poly->points[point_id - 1].pos, begin_poly->points[point_id].pos);
			Vector3 spoint = f.get_closest_point_to(p_destination);
			real_t dpoint = spoint.distance_to(p_destination);
			if (dpoint < end_d) {
				end_point = spoint;
				end_d = dpoint;
			}
		}

		if (r_path_types) {
			r_path_types->resize(2);
			r_path_types->write[0] = begin_poly->owner->get_type();
			r_path_types->write[1] = begin_poly->owner->get_type();
		}

		if (r_path_rids) {
			r_path_rids->resize(2);
			(*r_path_rids)[0] = begin_poly->owner->get_self();
			(*r_path_rids)[1] = begin_poly->owner->get_self();
		}

		if (r_path_owners) {
			r_path_owners->resize(2);
			r_path_owners->write[0] = begin_poly->owner->get_owner_id();
			r_path_owners->write[1] = begin_poly->owner->get_owner_id();
		}

		Vector<Vector3> path;
		path.resize(2);
		path.write[0] = begin_point;
		path.write[1] = end_point;
		return path;
	}

	Vector<Vector3> path;
	// Optimize the path.
	if (p_optimize) {
		// Set the apex poly/point to the end point
		gd::NavigationPoly *apex_poly = &navigation_polys[least_cost_id];

		Vector3 back_pathway[2] = { apex_poly->back_navigation_edge_pathway_start, apex_poly->back_navigation_edge_pathway_end };
		const Vector3 back_edge_closest_point = Geometry3D::get_closest_point_to_segment(end_point, back_pathway);
		if (end_point.is_equal_approx(back_edge_closest_point)) {
			// The end point is basically on top of the last crossed edge, funneling around the corners would at best do nothing.
			// At worst it would add an unwanted path point before the last point due to precision issues so skip to the next polygon.
			if (apex_poly->back_navigation_poly_id != -1) {
				apex_poly = &navigation_polys[apex_poly->back_navigation_poly_id];
			}
		}

		Vector3 apex_point = end_point;

		gd::NavigationPoly *left_poly = apex_poly;
		Vector3 left_portal = apex_point;
		gd::NavigationPoly *right_poly = apex_poly;
		Vector3 right_portal = apex_point;

		gd::NavigationPoly *p = apex_poly;

		path.push_back(end_point);
		APPEND_METADATA(end_poly);

		while (p) {
			// Set left and right points of the pathway between polygons.
			Vector3 left = p->back_navigation_edge_pathway_start;
			Vector3 right = p->back_navigation_edge_pathway_end;
			if (THREE_POINTS_CROSS_PRODUCT(apex_point, left, right).dot(up) < 0) {
				SWAP(left, right);
			}

			bool skip = false;
			if (THREE_POINTS_CROSS_PRODUCT(apex_point, left_portal, left).dot(up) >= 0) {
				//process
				if (left_portal == apex_point || THREE_POINTS_CROSS_PRODUCT(apex_point, left, right_portal).dot(up) > 0) {
					left_poly = p;
					left_portal = left;
				} else {
					clip_path(navigation_polys, path, apex_poly, right_portal, right_poly, r_path_types, r_path_rids, r_path_owners);

					apex_point = right_portal;
					p = right_poly;
					left_poly = p;
					apex_poly = p;
					left_portal = apex_point;
					right_portal = apex_point;

					path.push_back(apex_point);
					APPEND_METADATA(apex_poly->poly);
					skip = true;
				}
			}

			if (!skip && THREE_POINTS_CROSS_PRODUCT(apex_point, right_portal, right).dot(up) <= 0) {
				//process
				if (right_portal == apex_point || THREE_POINTS_CROSS_PRODUCT(apex_point, right, left_portal).dot(up) < 0) {
					right_poly = p;
					right_portal = right;
				} else {
					clip_path(navigation_polys, path, apex_poly, left_portal, left_poly, r_path_types, r_path_rids, r_path_owners);

					apex_point = left_portal;
					p = left_poly;
					right_poly = p;
					apex_poly = p;
					right_portal = apex_point;
					left_portal = apex_point;

					path.push_back(apex_point);
					APPEND_METADATA(apex_poly->poly);
				}
			}

			// Go to the previous polygon.
			if (p->back_navigation_poly_id != -1) {
				p = &navigation_polys[p->back_navigation_poly_id];
			} else {
				// The end
				p = nullptr;
			}
		}

		// If the last point is not the begin point, add it to the list.
		if (path[path.size() - 1] != begin_point) {
			path.push_back(begin_point);
			APPEND_METADATA(begin_poly);
		}

		path.reverse();
		if (r_path_types) {
			r_path_types->reverse();
		}
		if (r_path_rids) {
			r_path_rids->reverse();
		}
		if (r_path_owners) {
			r_path_owners->reverse();
		}

	} else {
		path.push_back(end_point);
		APPEND_METADATA(end_poly);

		// Add mid points
		int np_id = least_cost_id;
		while (np_id != -1 && navigation_polys[np_id].back_navigation_poly_id != -1) {
			if (navigation_polys[np_id].back_navigation_edge != -1) {
				int prev = navigation_polys[np_id].back_navigation_edge;
				int prev_n = (navigation_polys[np_id].back_navigation_edge + 1) % navigation_polys[np_id].poly->points.size();
				Vector3 point = (navigation_polys[np_id].poly->points[prev].pos + navigation_polys[np_id].poly->points[prev_n].pos) * 0.5;

				path.push_back(point);
				APPEND_METADATA(navigation_polys[np_id].poly);
			} else {
				path.push_back(navigation_polys[np_id].entry);
				APPEND_METADATA(navigation_polys[np_id].poly);
			}

			np_id = navigation_polys[np_id].back_navigation_poly_id;
		}

		path.push_back(begin_point);
		APPEND_METADATA(begin_poly);

		path.reverse();
		if (r_path_types) {
			r_path_types->reverse();
		}
		if (r_path_rids) {
			r_path_rids->reverse();
		}
		if (r_path_owners) {
			r_path_owners->reverse();
		}
	}

	// Ensure post conditions (path arrays MUST match in size).
	CRASH_COND(r_path_types && path.size() != r_path_types->size());
	CRASH_COND(r_path_rids && path.size() != r_path_rids->size());
	CRASH_COND(r_path_owners && path.size() != r_path_owners->size());

	return path;
}

Vector3 NavMap::get_closest_point_to_segment(const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision) const {
	ERR_FAIL_COND_V_MSG(map_update_id == 0, Vector3(), "NavigationServer map query failed because it was made before first map synchronization.");
	bool use_collision = p_use_collision;
	Vector3 closest_point;
	real_t closest_point_d = FLT_MAX;

	for (const gd::Polygon &p : polygons) {
		// For each face check the distance to the segment
		for (size_t point_id = 2; point_id < p.points.size(); point_id += 1) {
			const Face3 f(p.points[0].pos, p.points[point_id - 1].pos, p.points[point_id].pos);
			Vector3 inters;
			if (f.intersects_segment(p_from, p_to, &inters)) {
				const real_t d = closest_point_d = p_from.distance_to(inters);
				if (use_collision == false) {
					closest_point = inters;
					use_collision = true;
					closest_point_d = d;
				} else if (closest_point_d > d) {
					closest_point = inters;
					closest_point_d = d;
				}
			}
		}

		if (use_collision == false) {
			for (size_t point_id = 0; point_id < p.points.size(); point_id += 1) {
				Vector3 a, b;

				Geometry3D::get_closest_points_between_segments(
						p_from,
						p_to,
						p.points[point_id].pos,
						p.points[(point_id + 1) % p.points.size()].pos,
						a,
						b);

				const real_t d = a.distance_to(b);
				if (d < closest_point_d) {
					closest_point_d = d;
					closest_point = b;
				}
			}
		}
	}

	return closest_point;
}

Vector3 NavMap::get_closest_point(const Vector3 &p_point) const {
	ERR_FAIL_COND_V_MSG(map_update_id == 0, Vector3(), "NavigationServer map query failed because it was made before first map synchronization.");
	gd::ClosestPointQueryResult cp = get_closest_point_info(p_point);
	return cp.point;
}

Vector3 NavMap::get_closest_point_normal(const Vector3 &p_point) const {
	ERR_FAIL_COND_V_MSG(map_update_id == 0, Vector3(), "NavigationServer map query failed because it was made before first map synchronization.");
	gd::ClosestPointQueryResult cp = get_closest_point_info(p_point);
	return cp.normal;
}

RID NavMap::get_closest_point_owner(const Vector3 &p_point) const {
	ERR_FAIL_COND_V_MSG(map_update_id == 0, RID(), "NavigationServer map query failed because it was made before first map synchronization.");
	gd::ClosestPointQueryResult cp = get_closest_point_info(p_point);
	return cp.owner;
}

gd::ClosestPointQueryResult NavMap::get_closest_point_info(const Vector3 &p_point) const {
	gd::ClosestPointQueryResult result;
	real_t closest_point_ds = FLT_MAX;

	for (const gd::Polygon &p : polygons) {
		// For each face check the distance to the point
		for (size_t point_id = 2; point_id < p.points.size(); point_id += 1) {
			const Face3 f(p.points[0].pos, p.points[point_id - 1].pos, p.points[point_id].pos);
			const Vector3 inters = f.get_closest_point_to(p_point);
			const real_t ds = inters.distance_squared_to(p_point);
			if (ds < closest_point_ds) {
				result.point = inters;
				result.normal = f.get_plane().normal;
				result.owner = p.owner->get_self();
				closest_point_ds = ds;
			}
		}
	}

	return result;
}

void NavMap::add_region(NavRegion *p_region) {
	regions.push_back(p_region);
	regenerate_links = true;
}

void NavMap::remove_region(NavRegion *p_region) {
	int64_t region_index = regions.find(p_region);
	if (region_index >= 0) {
		regions.remove_at_unordered(region_index);
		regenerate_links = true;
	}
}

void NavMap::add_link(NavLink *p_link) {
	links.push_back(p_link);
	regenerate_links = true;
}

void NavMap::remove_link(NavLink *p_link) {
	int64_t link_index = links.find(p_link);
	if (link_index >= 0) {
		links.remove_at_unordered(link_index);
		regenerate_links = true;
	}
}

bool NavMap::has_agent(NavAgent *agent) const {
	return (agents.find(agent) >= 0);
}

void NavMap::add_agent(NavAgent *agent) {
	if (!has_agent(agent)) {
		agents.push_back(agent);
		agents_dirty = true;
	}
}

void NavMap::remove_agent(NavAgent *agent) {
	remove_agent_as_controlled(agent);
	int64_t agent_index = agents.find(agent);
	if (agent_index >= 0) {
		agents.remove_at_unordered(agent_index);
		agents_dirty = true;
	}
}

bool NavMap::has_obstacle(NavObstacle *obstacle) const {
	return (obstacles.find(obstacle) >= 0);
}

void NavMap::add_obstacle(NavObstacle *obstacle) {
	if (obstacle->get_paused()) {
		// No point in adding a paused obstacle, it will add itself when unpaused again.
		return;
	}

	if (!has_obstacle(obstacle)) {
		obstacles.push_back(obstacle);
		obstacles_dirty = true;
	}
}

void NavMap::remove_obstacle(NavObstacle *obstacle) {
	int64_t obstacle_index = obstacles.find(obstacle);
	if (obstacle_index >= 0) {
		obstacles.remove_at_unordered(obstacle_index);
		obstacles_dirty = true;
	}
}

void NavMap::set_agent_as_controlled(NavAgent *agent) {
	remove_agent_as_controlled(agent);

	if (agent->get_paused()) {
		// No point in adding a paused agent, it will add itself when unpaused again.
		return;
	}

	if (agent->get_use_3d_avoidance()) {
		int64_t agent_3d_index = active_3d_avoidance_agents.find(agent);
		if (agent_3d_index < 0) {
			active_3d_avoidance_agents.push_back(agent);
			agents_dirty = true;
		}
	} else {
		int64_t agent_2d_index = active_2d_avoidance_agents.find(agent);
		if (agent_2d_index < 0) {
			active_2d_avoidance_agents.push_back(agent);
			agents_dirty = true;
		}
	}
}

void NavMap::remove_agent_as_controlled(NavAgent *agent) {
	int64_t agent_3d_index = active_3d_avoidance_agents.find(agent);
	if (agent_3d_index >= 0) {
		active_3d_avoidance_agents.remove_at_unordered(agent_3d_index);
		agents_dirty = true;
	}
	int64_t agent_2d_index = active_2d_avoidance_agents.find(agent);
	if (agent_2d_index >= 0) {
		active_2d_avoidance_agents.remove_at_unordered(agent_2d_index);
		agents_dirty = true;
	}
}

void NavMap::sync() {
	// Performance Monitor
	int _new_pm_region_count = regions.size();
	int _new_pm_agent_count = agents.size();
	int _new_pm_link_count = links.size();
	int _new_pm_polygon_count = pm_polygon_count;
	int _new_pm_edge_count = pm_edge_count;
	int _new_pm_edge_merge_count = pm_edge_merge_count;
	int _new_pm_edge_connection_count = pm_edge_connection_count;
	int _new_pm_edge_free_count = pm_edge_free_count;

	// Check if we need to update the links.
	if (regenerate_polygons) {
		for (NavRegion *region : regions) {
			region->scratch_polygons();
		}
		regenerate_links = true;
	}

	for (NavRegion *region : regions) {
		if (region->sync()) {
			regenerate_links = true;
		}
	}

	for (NavLink *link : links) {
		if (link->check_dirty()) {
			regenerate_links = true;
		}
	}

	if (regenerate_links) {
		_new_pm_polygon_count = 0;
		_new_pm_edge_count = 0;
		_new_pm_edge_merge_count = 0;
		_new_pm_edge_connection_count = 0;
		_new_pm_edge_free_count = 0;

		// Remove regions connections.
		for (NavRegion *region : regions) {
			region->get_connections().clear();
		}

		// Resize the polygon count.
		int count = 0;
		for (const NavRegion *region : regions) {
			if (!region->get_enabled()) {
				continue;
			}
			count += region->get_polygons().size();
		}
		polygons.resize(count);

		// Copy all region polygons in the map.
		count = 0;
		for (const NavRegion *region : regions) {
			if (!region->get_enabled()) {
				continue;
			}
			const LocalVector<gd::Polygon> &polygons_source = region->get_polygons();
			for (uint32_t n = 0; n < polygons_source.size(); n++) {
				polygons[count + n] = polygons_source[n];
			}
			count += region->get_polygons().size();
		}

		_new_pm_polygon_count = polygons.size();

		// Group all edges per key.
		HashMap<gd::EdgeKey, Vector<gd::Edge::Connection>, gd::EdgeKey> connections;
		for (gd::Polygon &poly : polygons) {
			for (uint32_t p = 0; p < poly.points.size(); p++) {
				int next_point = (p + 1) % poly.points.size();
				gd::EdgeKey ek(poly.points[p].key, poly.points[next_point].key);

				HashMap<gd::EdgeKey, Vector<gd::Edge::Connection>, gd::EdgeKey>::Iterator connection = connections.find(ek);
				if (!connection) {
					connections[ek] = Vector<gd::Edge::Connection>();
					_new_pm_edge_count += 1;
				}
				if (connections[ek].size() <= 1) {
					// Add the polygon/edge tuple to this key.
					gd::Edge::Connection new_connection;
					new_connection.polygon = &poly;
					new_connection.edge = p;
					new_connection.pathway_start = poly.points[p].pos;
					new_connection.pathway_end = poly.points[next_point].pos;
					connections[ek].push_back(new_connection);
				} else {
					// The edge is already connected with another edge, skip.
					ERR_PRINT_ONCE("Navigation map synchronization error. Attempted to merge a navigation mesh polygon edge with another already-merged edge. This is usually caused by crossing edges, overlapping polygons, or a mismatch of the NavigationMesh / NavigationPolygon baked 'cell_size' and navigation map 'cell_size'.");
				}
			}
		}

		Vector<gd::Edge::Connection> free_edges;
		for (KeyValue<gd::EdgeKey, Vector<gd::Edge::Connection>> &E : connections) {
			if (E.value.size() == 2) {
				// Connect edge that are shared in different polygons.
				gd::Edge::Connection &c1 = E.value.write[0];
				gd::Edge::Connection &c2 = E.value.write[1];
				c1.polygon->edges[c1.edge].connections.push_back(c2);
				c2.polygon->edges[c2.edge].connections.push_back(c1);
				// Note: The pathway_start/end are full for those connection and do not need to be modified.
				_new_pm_edge_merge_count += 1;
			} else {
				CRASH_COND_MSG(E.value.size() != 1, vformat("Number of connection != 1. Found: %d", E.value.size()));
				if (use_edge_connections && E.value[0].polygon->owner->get_use_edge_connections()) {
					free_edges.push_back(E.value[0]);
				}
			}
		}

		// Find the compatible near edges.
		//
		// Note:
		// Considering that the edges must be compatible (for obvious reasons)
		// to be connected, create new polygons to remove that small gap is
		// not really useful and would result in wasteful computation during
		// connection, integration and path finding.
		_new_pm_edge_free_count = free_edges.size();

		for (int i = 0; i < free_edges.size(); i++) {
			const gd::Edge::Connection &free_edge = free_edges[i];
			Vector3 edge_p1 = free_edge.polygon->points[free_edge.edge].pos;
			Vector3 edge_p2 = free_edge.polygon->points[(free_edge.edge + 1) % free_edge.polygon->points.size()].pos;

			for (int j = 0; j < free_edges.size(); j++) {
				const gd::Edge::Connection &other_edge = free_edges[j];
				if (i == j || free_edge.polygon->owner == other_edge.polygon->owner) {
					continue;
				}

				Vector3 other_edge_p1 = other_edge.polygon->points[other_edge.edge].pos;
				Vector3 other_edge_p2 = other_edge.polygon->points[(other_edge.edge + 1) % other_edge.polygon->points.size()].pos;

				// Compute the projection of the opposite edge on the current one
				Vector3 edge_vector = edge_p2 - edge_p1;
				real_t projected_p1_ratio = edge_vector.dot(other_edge_p1 - edge_p1) / (edge_vector.length_squared());
				real_t projected_p2_ratio = edge_vector.dot(other_edge_p2 - edge_p1) / (edge_vector.length_squared());
				if ((projected_p1_ratio < 0.0 && projected_p2_ratio < 0.0) || (projected_p1_ratio > 1.0 && projected_p2_ratio > 1.0)) {
					continue;
				}

				// Check if the two edges are close to each other enough and compute a pathway between the two regions.
				Vector3 self1 = edge_vector * CLAMP(projected_p1_ratio, 0.0, 1.0) + edge_p1;
				Vector3 other1;
				if (projected_p1_ratio >= 0.0 && projected_p1_ratio <= 1.0) {
					other1 = other_edge_p1;
				} else {
					other1 = other_edge_p1.lerp(other_edge_p2, (1.0 - projected_p1_ratio) / (projected_p2_ratio - projected_p1_ratio));
				}
				if (other1.distance_to(self1) > edge_connection_margin) {
					continue;
				}

				Vector3 self2 = edge_vector * CLAMP(projected_p2_ratio, 0.0, 1.0) + edge_p1;
				Vector3 other2;
				if (projected_p2_ratio >= 0.0 && projected_p2_ratio <= 1.0) {
					other2 = other_edge_p2;
				} else {
					other2 = other_edge_p1.lerp(other_edge_p2, (0.0 - projected_p1_ratio) / (projected_p2_ratio - projected_p1_ratio));
				}
				if (other2.distance_to(self2) > edge_connection_margin) {
					continue;
				}

				// The edges can now be connected.
				gd::Edge::Connection new_connection = other_edge;
				new_connection.pathway_start = (self1 + other1) / 2.0;
				new_connection.pathway_end = (self2 + other2) / 2.0;
				free_edge.polygon->edges[free_edge.edge].connections.push_back(new_connection);

				// Add the connection to the region_connection map.
				((NavRegion *)free_edge.polygon->owner)->get_connections().push_back(new_connection);
				_new_pm_edge_connection_count += 1;
			}
		}

		uint32_t link_poly_idx = 0;
		link_polygons.resize(links.size());

		// Search for polygons within range of a nav link.
		for (const NavLink *link : links) {
			if (!link->get_enabled()) {
				continue;
			}
			const Vector3 start = link->get_start_position();
			const Vector3 end = link->get_end_position();

			gd::Polygon *closest_start_polygon = nullptr;
			real_t closest_start_distance = link_connection_radius;
			Vector3 closest_start_point;

			gd::Polygon *closest_end_polygon = nullptr;
			real_t closest_end_distance = link_connection_radius;
			Vector3 closest_end_point;

			// Create link to any polygons within the search radius of the start point.
			for (uint32_t start_index = 0; start_index < polygons.size(); start_index++) {
				gd::Polygon &start_poly = polygons[start_index];

				// For each face check the distance to the start
				for (uint32_t start_point_id = 2; start_point_id < start_poly.points.size(); start_point_id += 1) {
					const Face3 start_face(start_poly.points[0].pos, start_poly.points[start_point_id - 1].pos, start_poly.points[start_point_id].pos);
					const Vector3 start_point = start_face.get_closest_point_to(start);
					const real_t start_distance = start_point.distance_to(start);

					// Pick the polygon that is within our radius and is closer than anything we've seen yet.
					if (start_distance <= link_connection_radius && start_distance < closest_start_distance) {
						closest_start_distance = start_distance;
						closest_start_point = start_point;
						closest_start_polygon = &start_poly;
					}
				}
			}

			// Find any polygons within the search radius of the end point.
			for (gd::Polygon &end_poly : polygons) {
				// For each face check the distance to the end
				for (uint32_t end_point_id = 2; end_point_id < end_poly.points.size(); end_point_id += 1) {
					const Face3 end_face(end_poly.points[0].pos, end_poly.points[end_point_id - 1].pos, end_poly.points[end_point_id].pos);
					const Vector3 end_point = end_face.get_closest_point_to(end);
					const real_t end_distance = end_point.distance_to(end);

					// Pick the polygon that is within our radius and is closer than anything we've seen yet.
					if (end_distance <= link_connection_radius && end_distance < closest_end_distance) {
						closest_end_distance = end_distance;
						closest_end_point = end_point;
						closest_end_polygon = &end_poly;
					}
				}
			}

			// If we have both a start and end point, then create a synthetic polygon to route through.
			if (closest_start_polygon && closest_end_polygon) {
				gd::Polygon &new_polygon = link_polygons[link_poly_idx++];
				new_polygon.owner = link;

				new_polygon.edges.clear();
				new_polygon.edges.resize(4);
				new_polygon.points.clear();
				new_polygon.points.reserve(4);

				// Build a set of vertices that create a thin polygon going from the start to the end point.
				new_polygon.points.push_back({ closest_start_point, get_point_key(closest_start_point) });
				new_polygon.points.push_back({ closest_start_point, get_point_key(closest_start_point) });
				new_polygon.points.push_back({ closest_end_point, get_point_key(closest_end_point) });
				new_polygon.points.push_back({ closest_end_point, get_point_key(closest_end_point) });

				Vector3 center;
				for (int p = 0; p < 4; ++p) {
					center += new_polygon.points[p].pos;
				}
				new_polygon.center = center / real_t(new_polygon.points.size());
				new_polygon.clockwise = true;

				// Setup connections to go forward in the link.
				{
					gd::Edge::Connection entry_connection;
					entry_connection.polygon = &new_polygon;
					entry_connection.edge = -1;
					entry_connection.pathway_start = new_polygon.points[0].pos;
					entry_connection.pathway_end = new_polygon.points[1].pos;
					closest_start_polygon->edges[0].connections.push_back(entry_connection);

					gd::Edge::Connection exit_connection;
					exit_connection.polygon = closest_end_polygon;
					exit_connection.edge = -1;
					exit_connection.pathway_start = new_polygon.points[2].pos;
					exit_connection.pathway_end = new_polygon.points[3].pos;
					new_polygon.edges[2].connections.push_back(exit_connection);
				}

				// If the link is bi-directional, create connections from the end to the start.
				if (link->is_bidirectional()) {
					gd::Edge::Connection entry_connection;
					entry_connection.polygon = &new_polygon;
					entry_connection.edge = -1;
					entry_connection.pathway_start = new_polygon.points[2].pos;
					entry_connection.pathway_end = new_polygon.points[3].pos;
					closest_end_polygon->edges[0].connections.push_back(entry_connection);

					gd::Edge::Connection exit_connection;
					exit_connection.polygon = closest_start_polygon;
					exit_connection.edge = -1;
					exit_connection.pathway_start = new_polygon.points[0].pos;
					exit_connection.pathway_end = new_polygon.points[1].pos;
					new_polygon.edges[0].connections.push_back(exit_connection);
				}
			}
		}

		// Update the update ID.
		// Some code treats 0 as a failure case, so we avoid returning 0.
		map_update_id = map_update_id % 9999999 + 1;
	}

	// Do we have modified obstacle positions?
	for (NavObstacle *obstacle : obstacles) {
		if (obstacle->check_dirty()) {
			obstacles_dirty = true;
		}
	}
	// Do we have modified agent arrays?
	for (NavAgent *agent : agents) {
		if (agent->check_dirty()) {
			agents_dirty = true;
		}
	}

	// Update avoidance worlds.
	if (obstacles_dirty || agents_dirty) {
		_update_rvo_simulation();
	}

	regenerate_polygons = false;
	regenerate_links = false;
	obstacles_dirty = false;
	agents_dirty = false;

	// Performance Monitor.
	pm_region_count = _new_pm_region_count;
	pm_agent_count = _new_pm_agent_count;
	pm_link_count = _new_pm_link_count;
	pm_polygon_count = _new_pm_polygon_count;
	pm_edge_count = _new_pm_edge_count;
	pm_edge_merge_count = _new_pm_edge_merge_count;
	pm_edge_connection_count = _new_pm_edge_connection_count;
	pm_edge_free_count = _new_pm_edge_free_count;
}

void NavMap::_update_rvo_obstacles_tree_2d() {
	int obstacle_vertex_count = 0;
	for (NavObstacle *obstacle : obstacles) {
		obstacle_vertex_count += obstacle->get_vertices().size();
	}

	// Cleaning old obstacles.
	for (size_t i = 0; i < rvo_simulation_2d.obstacles_.size(); ++i) {
		delete rvo_simulation_2d.obstacles_[i];
	}
	rvo_simulation_2d.obstacles_.clear();

	// Cannot use LocalVector here as RVO library expects std::vector to build KdTree
	std::vector<RVO2D::Obstacle2D *> &raw_obstacles = rvo_simulation_2d.obstacles_;
	raw_obstacles.reserve(obstacle_vertex_count);

	// The following block is modified copy from RVO2D::AddObstacle()
	// Obstacles are linked and depend on all other obstacles.
	for (NavObstacle *obstacle : obstacles) {
		const Vector3 &_obstacle_position = obstacle->get_position();
		const Vector<Vector3> &_obstacle_vertices = obstacle->get_vertices();

		if (_obstacle_vertices.size() < 2) {
			continue;
		}

		std::vector<RVO2D::Vector2> rvo_2d_vertices;
		rvo_2d_vertices.reserve(_obstacle_vertices.size());

		uint32_t _obstacle_avoidance_layers = obstacle->get_avoidance_layers();
		real_t _obstacle_height = obstacle->get_height();

		for (const Vector3 &_obstacle_vertex : _obstacle_vertices) {
#ifdef TOOLS_ENABLED
			if (_obstacle_vertex.y != 0) {
				WARN_PRINT_ONCE("Y coordinates of static obstacle vertices are ignored. Please use obstacle position Y to change elevation of obstacle.");
			}
#endif
			rvo_2d_vertices.push_back(RVO2D::Vector2(_obstacle_vertex.x + _obstacle_position.x, _obstacle_vertex.z + _obstacle_position.z));
		}

		const size_t obstacleNo = raw_obstacles.size();

		for (size_t i = 0; i < rvo_2d_vertices.size(); i++) {
			RVO2D::Obstacle2D *rvo_2d_obstacle = new RVO2D::Obstacle2D();
			rvo_2d_obstacle->point_ = rvo_2d_vertices[i];
			rvo_2d_obstacle->height_ = _obstacle_height;
			rvo_2d_obstacle->elevation_ = _obstacle_position.y;

			rvo_2d_obstacle->avoidance_layers_ = _obstacle_avoidance_layers;

			if (i != 0) {
				rvo_2d_obstacle->prevObstacle_ = raw_obstacles.back();
				rvo_2d_obstacle->prevObstacle_->nextObstacle_ = rvo_2d_obstacle;
			}

			if (i == rvo_2d_vertices.size() - 1) {
				rvo_2d_obstacle->nextObstacle_ = raw_obstacles[obstacleNo];
				rvo_2d_obstacle->nextObstacle_->prevObstacle_ = rvo_2d_obstacle;
			}

			rvo_2d_obstacle->unitDir_ = normalize(rvo_2d_vertices[(i == rvo_2d_vertices.size() - 1 ? 0 : i + 1)] - rvo_2d_vertices[i]);

			if (rvo_2d_vertices.size() == 2) {
				rvo_2d_obstacle->isConvex_ = true;
			} else {
				rvo_2d_obstacle->isConvex_ = (leftOf(rvo_2d_vertices[(i == 0 ? rvo_2d_vertices.size() - 1 : i - 1)], rvo_2d_vertices[i], rvo_2d_vertices[(i == rvo_2d_vertices.size() - 1 ? 0 : i + 1)]) >= 0.0f);
			}

			rvo_2d_obstacle->id_ = raw_obstacles.size();

			raw_obstacles.push_back(rvo_2d_obstacle);
		}
	}

	rvo_simulation_2d.kdTree_->buildObstacleTree(raw_obstacles);
}

void NavMap::_update_rvo_agents_tree_2d() {
	// Cannot use LocalVector here as RVO library expects std::vector to build KdTree.
	std::vector<RVO2D::Agent2D *> raw_agents;
	raw_agents.reserve(active_2d_avoidance_agents.size());
	for (NavAgent *agent : active_2d_avoidance_agents) {
		raw_agents.push_back(agent->get_rvo_agent_2d());
	}
	rvo_simulation_2d.kdTree_->buildAgentTree(raw_agents);
}

void NavMap::_update_rvo_agents_tree_3d() {
	// Cannot use LocalVector here as RVO library expects std::vector to build KdTree.
	std::vector<RVO3D::Agent3D *> raw_agents;
	raw_agents.reserve(active_3d_avoidance_agents.size());
	for (NavAgent *agent : active_3d_avoidance_agents) {
		raw_agents.push_back(agent->get_rvo_agent_3d());
	}
	rvo_simulation_3d.kdTree_->buildAgentTree(raw_agents);
}

void NavMap::_update_rvo_simulation() {
	if (obstacles_dirty) {
		_update_rvo_obstacles_tree_2d();
	}
	if (agents_dirty) {
		_update_rvo_agents_tree_2d();
		_update_rvo_agents_tree_3d();
	}
}

void NavMap::compute_single_avoidance_step_2d(uint32_t index, NavAgent **agent) {
	(*(agent + index))->get_rvo_agent_2d()->computeNeighbors(&rvo_simulation_2d);
	(*(agent + index))->get_rvo_agent_2d()->computeNewVelocity(&rvo_simulation_2d);
	(*(agent + index))->get_rvo_agent_2d()->update(&rvo_simulation_2d);
	(*(agent + index))->update();
}

void NavMap::compute_single_avoidance_step_3d(uint32_t index, NavAgent **agent) {
	(*(agent + index))->get_rvo_agent_3d()->computeNeighbors(&rvo_simulation_3d);
	(*(agent + index))->get_rvo_agent_3d()->computeNewVelocity(&rvo_simulation_3d);
	(*(agent + index))->get_rvo_agent_3d()->update(&rvo_simulation_3d);
	(*(agent + index))->update();
}

void NavMap::step(real_t p_deltatime) {
	deltatime = p_deltatime;

	rvo_simulation_2d.setTimeStep(float(deltatime));
	rvo_simulation_3d.setTimeStep(float(deltatime));

	if (active_2d_avoidance_agents.size() > 0) {
		if (use_threads && avoidance_use_multiple_threads) {
			WorkerThreadPool::GroupID group_task = WorkerThreadPool::get_singleton()->add_template_group_task(this, &NavMap::compute_single_avoidance_step_2d, active_2d_avoidance_agents.ptr(), active_2d_avoidance_agents.size(), -1, true, SNAME("RVOAvoidanceAgents2D"));
			WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group_task);
		} else {
			for (NavAgent *agent : active_2d_avoidance_agents) {
				agent->get_rvo_agent_2d()->computeNeighbors(&rvo_simulation_2d);
				agent->get_rvo_agent_2d()->computeNewVelocity(&rvo_simulation_2d);
				agent->get_rvo_agent_2d()->update(&rvo_simulation_2d);
				agent->update();
			}
		}
	}

	if (active_3d_avoidance_agents.size() > 0) {
		if (use_threads && avoidance_use_multiple_threads) {
			WorkerThreadPool::GroupID group_task = WorkerThreadPool::get_singleton()->add_template_group_task(this, &NavMap::compute_single_avoidance_step_3d, active_3d_avoidance_agents.ptr(), active_3d_avoidance_agents.size(), -1, true, SNAME("RVOAvoidanceAgents3D"));
			WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group_task);
		} else {
			for (NavAgent *agent : active_3d_avoidance_agents) {
				agent->get_rvo_agent_3d()->computeNeighbors(&rvo_simulation_3d);
				agent->get_rvo_agent_3d()->computeNewVelocity(&rvo_simulation_3d);
				agent->get_rvo_agent_3d()->update(&rvo_simulation_3d);
				agent->update();
			}
		}
	}
}

void NavMap::dispatch_callbacks() {
	for (NavAgent *agent : active_2d_avoidance_agents) {
		agent->dispatch_avoidance_callback();
	}

	for (NavAgent *agent : active_3d_avoidance_agents) {
		agent->dispatch_avoidance_callback();
	}
}

void NavMap::clip_path(const LocalVector<gd::NavigationPoly> &p_navigation_polys, Vector<Vector3> &path, const gd::NavigationPoly *from_poly, const Vector3 &p_to_point, const gd::NavigationPoly *p_to_poly, Vector<int32_t> *r_path_types, TypedArray<RID> *r_path_rids, Vector<int64_t> *r_path_owners) const {
	Vector3 from = path[path.size() - 1];

	if (from.is_equal_approx(p_to_point)) {
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
		Vector3 pathway_start = from_poly->back_navigation_edge_pathway_start;
		Vector3 pathway_end = from_poly->back_navigation_edge_pathway_end;

		ERR_FAIL_COND(from_poly->back_navigation_poly_id == -1);
		from_poly = &p_navigation_polys[from_poly->back_navigation_poly_id];

		if (!pathway_start.is_equal_approx(pathway_end)) {
			Vector3 inters;
			if (cut_plane.intersects_segment(pathway_start, pathway_end, &inters)) {
				if (!inters.is_equal_approx(p_to_point) && !inters.is_equal_approx(path[path.size() - 1])) {
					path.push_back(inters);
					APPEND_METADATA(from_poly->poly);
				}
			}
		}
	}
}

NavMap::NavMap() {
	avoidance_use_multiple_threads = GLOBAL_GET("navigation/avoidance/thread_model/avoidance_use_multiple_threads");
	avoidance_use_high_priority_threads = GLOBAL_GET("navigation/avoidance/thread_model/avoidance_use_high_priority_threads");
}

NavMap::~NavMap() {
}
