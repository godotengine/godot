/*************************************************************************/
/*  nav_map.cpp                                                          */
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

#include "nav_map.h"

#include "core/os/threaded_array_processor.h"
#include "nav_region.h"
#include "rvo_agent.h"

#include <algorithm>

/**
	@author AndreaCatania
*/

#define THREE_POINTS_CROSS_PRODUCT(m_a, m_b, m_c) (((m_c) - (m_a)).cross((m_b) - (m_a)))

void NavMap::set_up(Vector3 p_up) {
	up = p_up;
	regenerate_polygons = true;
}

void NavMap::set_cell_size(float p_cell_size) {
	cell_size = p_cell_size;
	regenerate_polygons = true;
}

void NavMap::set_edge_connection_margin(float p_edge_connection_margin) {
	edge_connection_margin = p_edge_connection_margin;
	regenerate_links = true;
}

gd::PointKey NavMap::get_point_key(const Vector3 &p_pos) const {
	const int x = int(Math::floor(p_pos.x / cell_size));
	const int y = int(Math::floor(p_pos.y / cell_size));
	const int z = int(Math::floor(p_pos.z / cell_size));

	gd::PointKey p;
	p.key = 0;
	p.x = x;
	p.y = y;
	p.z = z;
	return p;
}

Vector<Vector3> NavMap::get_path(Vector3 p_origin, Vector3 p_destination, bool p_optimize, uint32_t p_layers) const {
	// Find the start poly and the end poly on this map.
	const gd::Polygon *begin_poly = nullptr;
	const gd::Polygon *end_poly = nullptr;
	Vector3 begin_point;
	Vector3 end_point;
	float begin_d = 1e20;
	float end_d = 1e20;
	// Find the initial poly and the end poly on this map.
	for (size_t i(0); i < polygons.size(); i++) {
		const gd::Polygon &p = polygons[i];

		// Only consider the polygon if it in a region with compatible layers.
		if ((p_layers & p.owner->get_layers()) == 0) {
			continue;
		}

		// For each point cast a face and check the distance between the origin/destination
		for (size_t point_id = 0; point_id < p.points.size(); point_id++) {
			const Vector3 p1 = p.points[point_id].pos;
			const Vector3 p2 = p.points[(point_id + 1) % p.points.size()].pos;
			const Vector3 p3 = p.points[(point_id + 2) % p.points.size()].pos;
			const Face3 face(p1, p2, p3);

			Vector3 point = face.get_closest_point_to(p_origin);
			float distance_to_point = point.distance_to(p_origin);
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
		Vector<Vector3> path;
		path.resize(2);
		path.write[0] = begin_point;
		path.write[1] = end_point;
		return path;
	}

	// List of all reachable navigation polys.
	std::vector<gd::NavigationPoly> navigation_polys;
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
	bool found_route = false;

	const gd::Polygon *reachable_end = nullptr;
	float reachable_d = 1e30;
	bool is_reachable = true;

	while (true) {
		gd::NavigationPoly *least_cost_poly = &navigation_polys[least_cost_id];

		// Takes the current least_cost_poly neighbors (iterating over its edges) and compute the traveled_distance.
		for (size_t i = 0; i < least_cost_poly->poly->edges.size(); i++) {
			const gd::Edge &edge = least_cost_poly->poly->edges[i];

			// Iterate over connections in this edge, then compute the new optimized travel distance assigned to this polygon.
			for (int connection_index = 0; connection_index < edge.connections.size(); connection_index++) {
				const gd::Edge::Connection &connection = edge.connections[connection_index];

				// Only consider the connection to another polygon if this polygon is in a region with compatible layers.
				if ((p_layers & connection.polygon->owner->get_layers()) == 0) {
					continue;
				}

				Vector3 pathway[2] = { connection.pathway_start, connection.pathway_end };
				const Vector3 new_entry = Geometry3D::get_closest_point_to_segment(least_cost_poly->entry, pathway);
				const float new_distance = least_cost_poly->entry.distance_to(new_entry) + least_cost_poly->traveled_distance;

				const std::vector<gd::NavigationPoly>::iterator it = std::find(
						navigation_polys.begin(),
						navigation_polys.end(),
						gd::NavigationPoly(connection.polygon));

				if (it != navigation_polys.end()) {
					// Polygon already visited, check if we can reduce the travel cost.
					if (new_distance < it->traveled_distance) {
						it->back_navigation_poly_id = least_cost_id;
						it->back_navigation_edge = connection.edge;
						it->back_navigation_edge_pathway_start = connection.pathway_start;
						it->back_navigation_edge_pathway_end = connection.pathway_end;
						it->traveled_distance = new_distance;
						it->entry = new_entry;
					}
				} else {
					// Add the neighbour polygon to the reachable ones.
					gd::NavigationPoly new_navigation_poly = gd::NavigationPoly(connection.polygon);
					new_navigation_poly.self_id = navigation_polys.size();
					new_navigation_poly.back_navigation_poly_id = least_cost_id;
					new_navigation_poly.back_navigation_edge = connection.edge;
					new_navigation_poly.back_navigation_edge_pathway_start = connection.pathway_start;
					new_navigation_poly.back_navigation_edge_pathway_end = connection.pathway_end;
					new_navigation_poly.traveled_distance = new_distance;
					new_navigation_poly.entry = new_entry;
					navigation_polys.push_back(new_navigation_poly);

					// Add the neighbour polygon to the polygons to visit.
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
			end_d = 1e20;
			for (size_t point_id = 2; point_id < end_poly->points.size(); point_id++) {
				Face3 f(end_poly->points[point_id - 2].pos, end_poly->points[point_id - 1].pos, end_poly->points[point_id].pos);
				Vector3 spoint = f.get_closest_point_to(p_destination);
				float dpoint = spoint.distance_to(p_destination);
				if (dpoint < end_d) {
					end_point = spoint;
					end_d = dpoint;
				}
			}

			// Reset open and navigation_polys
			gd::NavigationPoly np = navigation_polys[0];
			navigation_polys.clear();
			navigation_polys.push_back(np);
			to_visit.clear();
			to_visit.push_back(0);

			reachable_end = nullptr;

			continue;
		}

		// Find the polygon with the minimum cost from the list of polygons to visit.
		least_cost_id = -1;
		float least_cost = 1e30;
		for (List<uint32_t>::Element *element = to_visit.front(); element != nullptr; element = element->next()) {
			gd::NavigationPoly *np = &navigation_polys[element->get()];
			float cost = np->traveled_distance;
			cost += np->entry.distance_to(end_point);
			if (cost < least_cost) {
				least_cost_id = np->self_id;
				least_cost = cost;
			}
		}

		// Stores the further reachable end polygon, in case our goal is not reachable.
		if (is_reachable) {
			float d = navigation_polys[least_cost_id].entry.distance_to(p_destination);
			if (reachable_d > d) {
				reachable_d = d;
				reachable_end = navigation_polys[least_cost_id].poly;
			}
		}

		ERR_BREAK(least_cost_id == -1);

		// Check if we reached the end
		if (navigation_polys[least_cost_id].poly == end_poly) {
			found_route = true;
			break;
		}
	}

	// If we did not find a route, return an empty path.
	if (!found_route) {
		return Vector<Vector3>();
	}

	Vector<Vector3> path;
	// Optimize the path.
	if (p_optimize) {
		// Set the apex poly/point to the end point
		gd::NavigationPoly *apex_poly = &navigation_polys[least_cost_id];
		Vector3 apex_point = end_point;

		gd::NavigationPoly *left_poly = apex_poly;
		Vector3 left_portal = apex_point;
		gd::NavigationPoly *right_poly = apex_poly;
		Vector3 right_portal = apex_point;

		gd::NavigationPoly *p = apex_poly;

		path.push_back(end_point);

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
					clip_path(navigation_polys, path, apex_poly, right_portal, right_poly);

					apex_point = right_portal;
					p = right_poly;
					left_poly = p;
					apex_poly = p;
					left_portal = apex_point;
					right_portal = apex_point;
					path.push_back(apex_point);
					skip = true;
				}
			}

			if (!skip && THREE_POINTS_CROSS_PRODUCT(apex_point, right_portal, right).dot(up) <= 0) {
				//process
				if (right_portal == apex_point || THREE_POINTS_CROSS_PRODUCT(apex_point, right, left_portal).dot(up) < 0) {
					right_poly = p;
					right_portal = right;
				} else {
					clip_path(navigation_polys, path, apex_poly, left_portal, left_poly);

					apex_point = left_portal;
					p = left_poly;
					right_poly = p;
					apex_poly = p;
					right_portal = apex_point;
					left_portal = apex_point;
					path.push_back(apex_point);
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
		}

		path.reverse();

	} else {
		path.push_back(end_point);

		// Add mid points
		int np_id = least_cost_id;
		while (np_id != -1) {
			path.push_back(navigation_polys[np_id].entry);
			np_id = navigation_polys[np_id].back_navigation_poly_id;
		}

		path.reverse();
	}

	return path;
}

Vector3 NavMap::get_closest_point_to_segment(const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision) const {
	bool use_collision = p_use_collision;
	Vector3 closest_point;
	real_t closest_point_d = 1e20;

	// Find the initial poly and the end poly on this map.
	for (size_t i(0); i < polygons.size(); i++) {
		const gd::Polygon &p = polygons[i];

		// For each point cast a face and check the distance to the segment
		for (size_t point_id = 2; point_id < p.points.size(); point_id += 1) {
			const Face3 f(p.points[point_id - 2].pos, p.points[point_id - 1].pos, p.points[point_id].pos);
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
	// TODO this is really not optimal, please redesign the API to directly return all this data

	Vector3 closest_point;
	real_t closest_point_d = 1e20;

	// Find the initial poly and the end poly on this map.
	for (size_t i(0); i < polygons.size(); i++) {
		const gd::Polygon &p = polygons[i];

		// For each point cast a face and check the distance to the point
		for (size_t point_id = 2; point_id < p.points.size(); point_id += 1) {
			const Face3 f(p.points[point_id - 2].pos, p.points[point_id - 1].pos, p.points[point_id].pos);
			const Vector3 inters = f.get_closest_point_to(p_point);
			const real_t d = inters.distance_to(p_point);
			if (d < closest_point_d) {
				closest_point = inters;
				closest_point_d = d;
			}
		}
	}

	return closest_point;
}

Vector3 NavMap::get_closest_point_normal(const Vector3 &p_point) const {
	// TODO this is really not optimal, please redesign the API to directly return all this data

	Vector3 closest_point;
	Vector3 closest_point_normal;
	real_t closest_point_d = 1e20;

	// Find the initial poly and the end poly on this map.
	for (size_t i(0); i < polygons.size(); i++) {
		const gd::Polygon &p = polygons[i];

		// For each point cast a face and check the distance to the point
		for (size_t point_id = 2; point_id < p.points.size(); point_id += 1) {
			const Face3 f(p.points[point_id - 2].pos, p.points[point_id - 1].pos, p.points[point_id].pos);
			const Vector3 inters = f.get_closest_point_to(p_point);
			const real_t d = inters.distance_to(p_point);
			if (d < closest_point_d) {
				closest_point = inters;
				closest_point_normal = f.get_plane().normal;
				closest_point_d = d;
			}
		}
	}

	return closest_point_normal;
}

RID NavMap::get_closest_point_owner(const Vector3 &p_point) const {
	// TODO this is really not optimal, please redesign the API to directly return all this data

	Vector3 closest_point;
	RID closest_point_owner;
	real_t closest_point_d = 1e20;

	// Find the initial poly and the end poly on this map.
	for (size_t i(0); i < polygons.size(); i++) {
		const gd::Polygon &p = polygons[i];

		// For each point cast a face and check the distance to the point
		for (size_t point_id = 2; point_id < p.points.size(); point_id += 1) {
			const Face3 f(p.points[point_id - 2].pos, p.points[point_id - 1].pos, p.points[point_id].pos);
			const Vector3 inters = f.get_closest_point_to(p_point);
			const real_t d = inters.distance_to(p_point);
			if (d < closest_point_d) {
				closest_point = inters;
				closest_point_owner = p.owner->get_self();
				closest_point_d = d;
			}
		}
	}

	return closest_point_owner;
}

void NavMap::add_region(NavRegion *p_region) {
	regions.push_back(p_region);
	regenerate_links = true;
}

void NavMap::remove_region(NavRegion *p_region) {
	const std::vector<NavRegion *>::iterator it = std::find(regions.begin(), regions.end(), p_region);
	if (it != regions.end()) {
		regions.erase(it);
		regenerate_links = true;
	}
}

bool NavMap::has_agent(RvoAgent *agent) const {
	return std::find(agents.begin(), agents.end(), agent) != agents.end();
}

void NavMap::add_agent(RvoAgent *agent) {
	if (!has_agent(agent)) {
		agents.push_back(agent);
		agents_dirty = true;
	}
}

void NavMap::remove_agent(RvoAgent *agent) {
	remove_agent_as_controlled(agent);
	const std::vector<RvoAgent *>::iterator it = std::find(agents.begin(), agents.end(), agent);
	if (it != agents.end()) {
		agents.erase(it);
		agents_dirty = true;
	}
}

void NavMap::set_agent_as_controlled(RvoAgent *agent) {
	const bool exist = std::find(controlled_agents.begin(), controlled_agents.end(), agent) != controlled_agents.end();
	if (!exist) {
		ERR_FAIL_COND(!has_agent(agent));
		controlled_agents.push_back(agent);
	}
}

void NavMap::remove_agent_as_controlled(RvoAgent *agent) {
	const std::vector<RvoAgent *>::iterator it = std::find(controlled_agents.begin(), controlled_agents.end(), agent);
	if (it != controlled_agents.end()) {
		controlled_agents.erase(it);
	}
}

void NavMap::sync() {
	// Check if we need to update the links.
	if (regenerate_polygons) {
		for (size_t r(0); r < regions.size(); r++) {
			regions[r]->scratch_polygons();
		}
		regenerate_links = true;
	}

	for (size_t r(0); r < regions.size(); r++) {
		if (regions[r]->sync()) {
			regenerate_links = true;
		}
	}

	if (regenerate_links) {
		// Remove regions connections.
		for (size_t r(0); r < regions.size(); r++) {
			regions[r]->get_connections().clear();
		}

		// Resize the polygon count.
		int count = 0;
		for (size_t r(0); r < regions.size(); r++) {
			count += regions[r]->get_polygons().size();
		}
		polygons.resize(count);

		// Copy all region polygons in the map.
		count = 0;
		for (size_t r(0); r < regions.size(); r++) {
			std::copy(
					regions[r]->get_polygons().data(),
					regions[r]->get_polygons().data() + regions[r]->get_polygons().size(),
					polygons.begin() + count);
			count += regions[r]->get_polygons().size();
		}

		// Group all edges per key.
		Map<gd::EdgeKey, Vector<gd::Edge::Connection>> connections;
		for (size_t poly_id(0); poly_id < polygons.size(); poly_id++) {
			gd::Polygon &poly(polygons[poly_id]);

			for (size_t p(0); p < poly.points.size(); p++) {
				int next_point = (p + 1) % poly.points.size();
				gd::EdgeKey ek(poly.points[p].key, poly.points[next_point].key);

				Map<gd::EdgeKey, Vector<gd::Edge::Connection>>::Element *connection = connections.find(ek);
				if (!connection) {
					connections[ek] = Vector<gd::Edge::Connection>();
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
					ERR_PRINT("Attempted to merge a navigation mesh triangle edge with another already-merged edge. This happens when the current `cell_size` is different from the one used to generate the navigation mesh. This will cause navigation problem.");
				}
			}
		}

		Vector<gd::Edge::Connection> free_edges;
		for (Map<gd::EdgeKey, Vector<gd::Edge::Connection>>::Element *E = connections.front(); E; E = E->next()) {
			if (E->get().size() == 2) {
				// Connect edge that are shared in different polygons.
				gd::Edge::Connection &c1 = E->get().write[0];
				gd::Edge::Connection &c2 = E->get().write[1];
				c1.polygon->edges[c1.edge].connections.push_back(c2);
				c2.polygon->edges[c2.edge].connections.push_back(c1);
				// Note: The pathway_start/end are full for those connection and do not need to be modified.
			} else {
				CRASH_COND_MSG(E->get().size() != 1, vformat("Number of connection != 1. Found: %d", E->get().size()));
				free_edges.push_back(E->get()[0]);
			}
		}

		// Find the compatible near edges.
		//
		// Note:
		// Considering that the edges must be compatible (for obvious reasons)
		// to be connected, create new polygons to remove that small gap is
		// not really useful and would result in wasteful computation during
		// connection, integration and path finding.
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
				float projected_p1_ratio = edge_vector.dot(other_edge_p1 - edge_p1) / (edge_vector.length_squared());
				float projected_p2_ratio = edge_vector.dot(other_edge_p2 - edge_p1) / (edge_vector.length_squared());
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
				if ((self1 - other1).length() > edge_connection_margin) {
					continue;
				}

				Vector3 self2 = edge_vector * CLAMP(projected_p2_ratio, 0.0, 1.0) + edge_p1;
				Vector3 other2;
				if (projected_p2_ratio >= 0.0 && projected_p2_ratio <= 1.0) {
					other2 = other_edge_p2;
				} else {
					other2 = other_edge_p1.lerp(other_edge_p2, (0.0 - projected_p1_ratio) / (projected_p2_ratio - projected_p1_ratio));
				}
				if ((self2 - other2).length() > edge_connection_margin) {
					continue;
				}

				// The edges can now be connected.
				gd::Edge::Connection new_connection = other_edge;
				new_connection.pathway_start = (self1 + other1) / 2.0;
				new_connection.pathway_end = (self2 + other2) / 2.0;
				free_edge.polygon->edges[free_edge.edge].connections.push_back(new_connection);

				// Add the connection to the region_connection map.
				free_edge.polygon->owner->get_connections().push_back(new_connection);
			}
		}

		// Update the update ID.
		map_update_id = (map_update_id + 1) % 9999999;
	}

	// Update agents tree.
	if (agents_dirty) {
		std::vector<RVO::Agent *> raw_agents;
		raw_agents.reserve(agents.size());
		for (size_t i(0); i < agents.size(); i++) {
			raw_agents.push_back(agents[i]->get_agent());
		}
		rvo.buildAgentTree(raw_agents);
	}

	regenerate_polygons = false;
	regenerate_links = false;
	agents_dirty = false;
}

void NavMap::compute_single_step(uint32_t index, RvoAgent **agent) {
	(*(agent + index))->get_agent()->computeNeighbors(&rvo);
	(*(agent + index))->get_agent()->computeNewVelocity(deltatime);
}

void NavMap::step(real_t p_deltatime) {
	deltatime = p_deltatime;
	if (controlled_agents.size() > 0) {
		thread_process_array(
				controlled_agents.size(),
				this,
				&NavMap::compute_single_step,
				controlled_agents.data());
	}
}

void NavMap::dispatch_callbacks() {
	for (int i(0); i < static_cast<int>(controlled_agents.size()); i++) {
		controlled_agents[i]->dispatch_callback();
	}
}

void NavMap::clip_path(const std::vector<gd::NavigationPoly> &p_navigation_polys, Vector<Vector3> &path, const gd::NavigationPoly *from_poly, const Vector3 &p_to_point, const gd::NavigationPoly *p_to_poly) const {
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
		Vector3 pathway_start = from_poly->back_navigation_edge_pathway_start;
		Vector3 pathway_end = from_poly->back_navigation_edge_pathway_end;

		ERR_FAIL_COND(from_poly->back_navigation_poly_id == -1);
		from_poly = &p_navigation_polys[from_poly->back_navigation_poly_id];

		if (pathway_start.distance_to(pathway_end) > CMP_EPSILON) {
			Vector3 inters;
			if (cut_plane.intersects_segment(pathway_start, pathway_end, &inters)) {
				if (inters.distance_to(p_to_point) > CMP_EPSILON && inters.distance_to(path[path.size() - 1]) > CMP_EPSILON) {
					path.push_back(inters);
				}
			}
		}
	}
}
