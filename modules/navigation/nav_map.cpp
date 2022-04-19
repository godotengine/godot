/*************************************************************************/
/*  nav_map.cpp                                                          */
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

#include "nav_map.h"

#include "nav_region.h"
#include "rvo_agent.h"
#include <algorithm>

/**
	@author AndreaCatania
*/

#define USE_ENTRY_POINT

NavMap::NavMap() :
		up(0, 1, 0),
		cell_size(0.3),
		cell_height(0.2),
		edge_connection_margin(5.0),
		regenerate_polygons(true),
		regenerate_links(true),
		agents_dirty(false),
		deltatime(0.0),
		map_update_id(0) {}

NavMap::~NavMap() {
	step_work_pool.finish();
}

void NavMap::set_up(Vector3 p_up) {
	up = p_up;
	regenerate_polygons = true;
}

void NavMap::set_cell_size(float p_cell_size) {
	cell_size = p_cell_size;
	regenerate_polygons = true;
}

void NavMap::set_cell_height(float p_cell_height) {
	cell_height = p_cell_height;
	regenerate_polygons = true;
}

void NavMap::set_edge_connection_margin(float p_edge_connection_margin) {
	edge_connection_margin = p_edge_connection_margin;
	regenerate_links = true;
}

gd::PointKey NavMap::get_point_key(const Vector3 &p_pos) const {
	const int x = static_cast<int>(Math::round(p_pos.x / cell_size));
	const int y = static_cast<int>(Math::round(p_pos.y / cell_height));
	const int z = static_cast<int>(Math::round(p_pos.z / cell_size));

	gd::PointKey p;
	p.key = 0;
	p.x = x;
	p.y = y;
	p.z = z;
	return p;
}

Vector<Vector3> NavMap::get_path(Vector3 p_origin, Vector3 p_destination, bool p_optimize) const {
	const gd::Polygon *begin_poly = NULL;
	const gd::Polygon *end_poly = NULL;
	Vector3 begin_point;
	Vector3 end_point;
	float begin_d = 1e20;
	float end_d = 1e20;

	// Find the initial poly and the end poly on this map.
	for (size_t i(0); i < polygons.size(); i++) {
		const gd::Polygon &p = polygons[i];

		// For each face check the distance between the origin/destination
		for (size_t point_id = 2; point_id < p.points.size(); point_id++) {
			const Face3 face(p.points[0].pos, p.points[point_id - 1].pos, p.points[point_id].pos);

			Vector3 spoint = face.get_closest_point_to(p_origin);
			float dpoint = spoint.distance_to(p_origin);
			if (dpoint < begin_d) {
				begin_d = dpoint;
				begin_poly = &p;
				begin_point = spoint;
			}

			spoint = face.get_closest_point_to(p_destination);
			dpoint = spoint.distance_to(p_destination);
			if (dpoint < end_d) {
				end_d = dpoint;
				end_poly = &p;
				end_point = spoint;
			}
		}
	}

	if (!begin_poly || !end_poly) {
		// No path
		return Vector<Vector3>();
	}

	if (begin_poly == end_poly) {
		Vector<Vector3> path;
		path.resize(2);
		path.write[0] = begin_point;
		path.write[1] = end_point;
		return path;
	}

	std::vector<gd::NavigationPoly> navigation_polys;
	navigation_polys.reserve(polygons.size() * 0.75);

	// The elements indices in the `navigation_polys`.
	int least_cost_id(-1);
	List<uint32_t> open_list;
	bool found_route = false;

	navigation_polys.push_back(gd::NavigationPoly(begin_poly));
	{
		least_cost_id = 0;
		gd::NavigationPoly *least_cost_poly = &navigation_polys[least_cost_id];
		least_cost_poly->self_id = least_cost_id;
		least_cost_poly->entry = begin_point;
	}

	open_list.push_back(0);

	const gd::Polygon *reachable_end = NULL;
	float reachable_d = 1e30;
	bool is_reachable = true;

	while (found_route == false) {
		{
			// Takes the current least_cost_poly neighbors and compute the traveled_distance of each
			for (size_t i = 0; i < navigation_polys[least_cost_id].poly->edges.size(); i++) {
				gd::NavigationPoly *least_cost_poly = &navigation_polys[least_cost_id];

				const gd::Edge &edge = least_cost_poly->poly->edges[i];
				if (!edge.other_polygon)
					continue;

#ifdef USE_ENTRY_POINT
				Vector3 edge_line[2] = {
					least_cost_poly->poly->points[i].pos,
					least_cost_poly->poly->points[(i + 1) % least_cost_poly->poly->points.size()].pos
				};

				const Vector3 new_entry = Geometry::get_closest_point_to_segment(least_cost_poly->entry, edge_line);
				const float new_distance = least_cost_poly->entry.distance_to(new_entry) + least_cost_poly->traveled_distance;
#else
				const float new_distance = least_cost_poly->poly->center.distance_to(edge.other_polygon->center) + least_cost_poly->traveled_distance;
#endif

				auto it = std::find(
						navigation_polys.begin(),
						navigation_polys.end(),
						gd::NavigationPoly(edge.other_polygon));

				if (it != navigation_polys.end()) {
					// Oh this was visited already, can we win the cost?
					if (it->traveled_distance > new_distance) {
						it->prev_navigation_poly_id = least_cost_id;
						it->back_navigation_edge = edge.other_edge;
						it->traveled_distance = new_distance;
#ifdef USE_ENTRY_POINT
						it->entry = new_entry;
#endif
					}
				} else {
					// Add to open neighbours

					navigation_polys.push_back(gd::NavigationPoly(edge.other_polygon));
					gd::NavigationPoly *np = &navigation_polys[navigation_polys.size() - 1];

					np->self_id = navigation_polys.size() - 1;
					np->prev_navigation_poly_id = least_cost_id;
					np->back_navigation_edge = edge.other_edge;
					np->traveled_distance = new_distance;
#ifdef USE_ENTRY_POINT
					np->entry = new_entry;
#endif
					open_list.push_back(navigation_polys.size() - 1);
				}
			}
		}

		// Removes the least cost polygon from the open list so we can advance.
		open_list.erase(least_cost_id);

		if (open_list.size() == 0) {
			// When the open list is empty at this point the End Polygon is not reachable
			// so use the further reachable polygon
			ERR_BREAK_MSG(is_reachable == false, "It's not expect to not find the most reachable polygons");
			is_reachable = false;
			if (reachable_end == NULL) {
				// The path is not found and there is not a way out.
				break;
			}

			// Set as end point the furthest reachable point.
			end_poly = reachable_end;
			end_d = 1e20;
			for (size_t point_id = 2; point_id < end_poly->points.size(); point_id++) {
				Face3 f(end_poly->points[0].pos, end_poly->points[point_id - 1].pos, end_poly->points[point_id].pos);
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
			open_list.clear();
			open_list.push_back(0);
			least_cost_id = 0;

			reachable_end = NULL;

			continue;
		}

		// Now take the new least_cost_poly from the open list.
		least_cost_id = -1;
		float least_cost = 1e30;

		for (auto element = open_list.front(); element != NULL; element = element->next()) {
			gd::NavigationPoly *np = &navigation_polys[element->get()];
			float cost = np->traveled_distance;
#ifdef USE_ENTRY_POINT
			cost += np->entry.distance_to(end_point);
#else
			cost += np->poly->center.distance_to(end_point);
#endif
			if (cost < least_cost) {
				least_cost_id = np->self_id;
				least_cost = cost;
			}
		}

		ERR_BREAK(least_cost_id == -1);

		// Stores the further reachable end polygon, in case our goal is not reachable.
		if (is_reachable) {
			float d = navigation_polys[least_cost_id].entry.distance_to(p_destination);
			if (reachable_d > d) {
				reachable_d = d;
				reachable_end = navigation_polys[least_cost_id].poly;
			}
		}

		// Check if we reached the end
		if (navigation_polys[least_cost_id].poly == end_poly) {
			// Yep, done!!
			found_route = true;
			break;
		}
	}

	if (found_route) {
		Vector<Vector3> path;
		if (p_optimize) {
			// String pulling

			gd::NavigationPoly *apex_poly = &navigation_polys[least_cost_id];
			Vector3 apex_point = end_point;
			Vector3 portal_left = apex_point;
			Vector3 portal_right = apex_point;
			gd::NavigationPoly *left_poly = apex_poly;
			gd::NavigationPoly *right_poly = apex_poly;
			gd::NavigationPoly *p = apex_poly;

			path.push_back(end_point);

			while (p) {
				Vector3 left;
				Vector3 right;

#define CLOCK_TANGENT(m_a, m_b, m_c) (((m_a) - (m_c)).cross((m_a) - (m_b)))

				if (p->poly == begin_poly) {
					left = begin_point;
					right = begin_point;
				} else {
					int prev = p->back_navigation_edge;
					int prev_n = (p->back_navigation_edge + 1) % p->poly->points.size();
					left = p->poly->points[prev].pos;
					right = p->poly->points[prev_n].pos;

					//if (CLOCK_TANGENT(apex_point,left,(left+right)*0.5).dot(up) < 0){
					if (p->poly->clockwise) {
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
						clip_path(navigation_polys, path, apex_poly, portal_right, right_poly);

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
						clip_path(navigation_polys, path, apex_poly, portal_left, left_poly);

						apex_point = portal_left;
						p = left_poly;
						right_poly = p;
						apex_poly = p;
						portal_right = apex_point;
						portal_left = apex_point;
						path.push_back(apex_point);
					}
				}

				if (p->prev_navigation_poly_id != -1)
					p = &navigation_polys[p->prev_navigation_poly_id];
				else
					// The end
					p = NULL;
			}

			if (path[path.size() - 1] != begin_point)
				path.push_back(begin_point);

			path.invert();

		} else {
			path.push_back(end_point);

			// Add mid points
			int np_id = least_cost_id;
			while (np_id != -1 && navigation_polys[np_id].prev_navigation_poly_id != -1) {
				int prev = navigation_polys[np_id].back_navigation_edge;
				int prev_n = (navigation_polys[np_id].back_navigation_edge + 1) % navigation_polys[np_id].poly->points.size();
				Vector3 point = (navigation_polys[np_id].poly->points[prev].pos + navigation_polys[np_id].poly->points[prev_n].pos) * 0.5;
				path.push_back(point);
				np_id = navigation_polys[np_id].prev_navigation_poly_id;
			}

			path.push_back(begin_point);

			path.invert();
		}

		return path;
	}
	return Vector<Vector3>();
}

Vector3 NavMap::get_closest_point_to_segment(const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision) const {
	bool use_collision = p_use_collision;
	Vector3 closest_point;
	real_t closest_point_d = 1e20;

	for (size_t i(0); i < polygons.size(); i++) {
		const gd::Polygon &p = polygons[i];

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

				Geometry::get_closest_points_between_segments(
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
	gd::ClosestPointQueryResult cp = get_closest_point_info(p_point);
	return cp.point;
}

Vector3 NavMap::get_closest_point_normal(const Vector3 &p_point) const {
	gd::ClosestPointQueryResult cp = get_closest_point_info(p_point);
	return cp.normal;
}

RID NavMap::get_closest_point_owner(const Vector3 &p_point) const {
	gd::ClosestPointQueryResult cp = get_closest_point_info(p_point);
	return cp.owner;
}

gd::ClosestPointQueryResult NavMap::get_closest_point_info(const Vector3 &p_point) const {
	gd::ClosestPointQueryResult result;
	real_t closest_point_ds = 1e20;

	for (size_t i(0); i < polygons.size(); i++) {
		const gd::Polygon &p = polygons[i];

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
	std::vector<NavRegion *>::iterator it = std::find(regions.begin(), regions.end(), p_region);
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
	auto it = std::find(agents.begin(), agents.end(), agent);
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
	auto it = std::find(controlled_agents.begin(), controlled_agents.end(), agent);
	if (it != controlled_agents.end()) {
		controlled_agents.erase(it);
	}
}

void NavMap::sync() {
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
		// Copy all region polygons in the map.
		int count = 0;
		for (size_t r(0); r < regions.size(); r++) {
			count += regions[r]->get_polygons().size();
		}

		polygons.resize(count);
		count = 0;

		for (size_t r(0); r < regions.size(); r++) {
			std::copy(
					regions[r]->get_polygons().data(),
					regions[r]->get_polygons().data() + regions[r]->get_polygons().size(),
					polygons.begin() + count);

			count += regions[r]->get_polygons().size();
		}

		// Connects the `Edges` of all the `Polygons` of all `Regions` each other.
		Map<gd::EdgeKey, gd::Connection> connections;

		for (size_t poly_id(0); poly_id < polygons.size(); poly_id++) {
			gd::Polygon &poly(polygons[poly_id]);

			for (size_t p(0); p < poly.points.size(); p++) {
				int next_point = (p + 1) % poly.points.size();
				gd::EdgeKey ek(poly.points[p].key, poly.points[next_point].key);

				Map<gd::EdgeKey, gd::Connection>::Element *connection = connections.find(ek);
				if (!connection) {
					// Nothing yet
					gd::Connection c;
					c.A = &poly;
					c.A_edge = p;
					c.B = NULL;
					c.B_edge = -1;
					connections[ek] = c;

				} else if (connection->get().B == NULL) {
					CRASH_COND(connection->get().A == NULL); // Unreachable

					// Connect the two Polygons by this edge
					connection->get().B = &poly;
					connection->get().B_edge = p;

					connection->get().A->edges[connection->get().A_edge].this_edge = connection->get().A_edge;
					connection->get().A->edges[connection->get().A_edge].other_polygon = connection->get().B;
					connection->get().A->edges[connection->get().A_edge].other_edge = connection->get().B_edge;

					connection->get().B->edges[connection->get().B_edge].this_edge = connection->get().B_edge;
					connection->get().B->edges[connection->get().B_edge].other_polygon = connection->get().A;
					connection->get().B->edges[connection->get().B_edge].other_edge = connection->get().A_edge;
				} else {
					// The edge is already connected with another edge, skip.
					ERR_PRINT("Attempted to merge a navigation mesh triangle edge with another already-merged edge. Either the Navigation's `cell_size` is different from the one used to generate the navigation mesh or `detail/sample_max_error` is too small. This will cause navigation problem.");
				}
			}
		}

		// Takes all the free edges.
		std::vector<gd::FreeEdge> free_edges;
		free_edges.reserve(connections.size());

		for (auto connection_element = connections.front(); connection_element; connection_element = connection_element->next()) {
			if (connection_element->get().B == NULL) {
				CRASH_COND(connection_element->get().A == NULL); // Unreachable
				CRASH_COND(connection_element->get().A_edge < 0); // Unreachable

				// This is a free edge
				uint32_t id(free_edges.size());
				free_edges.push_back(gd::FreeEdge());
				free_edges[id].is_free = true;
				free_edges[id].poly = connection_element->get().A;
				free_edges[id].edge_id = connection_element->get().A_edge;
				uint32_t point_0(free_edges[id].edge_id);
				uint32_t point_1((free_edges[id].edge_id + 1) % free_edges[id].poly->points.size());
				Vector3 pos_0 = free_edges[id].poly->points[point_0].pos;
				Vector3 pos_1 = free_edges[id].poly->points[point_1].pos;
				Vector3 relative = pos_1 - pos_0;
				free_edges[id].edge_center = (pos_0 + pos_1) / 2.0;
				free_edges[id].edge_dir = relative.normalized();
				free_edges[id].edge_len_squared = relative.length_squared();
			}
		}

		const float ecm_squared(edge_connection_margin * edge_connection_margin);
#define LEN_TOLERANCE 0.1
#define DIR_TOLERANCE 0.9
		// In front of tolerance
#define IFO_TOLERANCE 0.5

		// Find the compatible near edges.
		//
		// Note:
		// Considering that the edges must be compatible (for obvious reasons)
		// to be connected, create new polygons to remove that small gap is
		// not really useful and would result in wasteful computation during
		// connection, integration and path finding.
		for (size_t i(0); i < free_edges.size(); i++) {
			if (!free_edges[i].is_free) {
				continue;
			}
			gd::FreeEdge &edge = free_edges[i];
			for (size_t y(0); y < free_edges.size(); y++) {
				gd::FreeEdge &other_edge = free_edges[y];
				if (i == y || !other_edge.is_free || edge.poly->owner == other_edge.poly->owner) {
					continue;
				}

				Vector3 rel_centers = other_edge.edge_center - edge.edge_center;
				if (ecm_squared > rel_centers.length_squared() // Are enough closer?
						&& ABS(edge.edge_len_squared - other_edge.edge_len_squared) < LEN_TOLERANCE // Are the same length?
						&& ABS(edge.edge_dir.dot(other_edge.edge_dir)) > DIR_TOLERANCE // Are aligned?
						&& ABS(rel_centers.normalized().dot(edge.edge_dir)) < IFO_TOLERANCE // Are one in front the other?
				) {
					// The edges can be connected
					edge.is_free = false;
					other_edge.is_free = false;

					edge.poly->edges[edge.edge_id].this_edge = edge.edge_id;
					edge.poly->edges[edge.edge_id].other_edge = other_edge.edge_id;
					edge.poly->edges[edge.edge_id].other_polygon = other_edge.poly;

					other_edge.poly->edges[other_edge.edge_id].this_edge = other_edge.edge_id;
					other_edge.poly->edges[other_edge.edge_id].other_edge = edge.edge_id;
					other_edge.poly->edges[other_edge.edge_id].other_polygon = edge.poly;
				}
			}
		}
	}

	if (regenerate_links) {
		map_update_id = map_update_id + 1 % 9999999;
	}

	if (agents_dirty) {
		std::vector<RVO::Agent *> raw_agents;
		raw_agents.reserve(agents.size());
		for (size_t i(0); i < agents.size(); i++)
			raw_agents.push_back(agents[i]->get_agent());
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
		if (step_work_pool.get_thread_count() == 0) {
			step_work_pool.init();
		}
		step_work_pool.do_work(
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

	if (from.distance_to(p_to_point) < CMP_EPSILON)
		return;
	Plane cut_plane;
	cut_plane.normal = (from - p_to_point).cross(up);
	if (cut_plane.normal == Vector3())
		return;
	cut_plane.normal.normalize();
	cut_plane.d = cut_plane.normal.dot(from);

	while (from_poly != p_to_poly) {
		int back_nav_edge = from_poly->back_navigation_edge;
		Vector3 a = from_poly->poly->points[back_nav_edge].pos;
		Vector3 b = from_poly->poly->points[(back_nav_edge + 1) % from_poly->poly->points.size()].pos;

		ERR_FAIL_COND(from_poly->prev_navigation_poly_id == -1);
		from_poly = &p_navigation_polys[from_poly->prev_navigation_poly_id];

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
