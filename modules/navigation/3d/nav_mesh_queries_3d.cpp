/**************************************************************************/
/*  nav_mesh_queries_3d.cpp                                               */
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

#ifndef _3D_DISABLED

#include "nav_mesh_queries_3d.h"

#include "../nav_base.h"

#include "core/math/geometry_3d.h"

#define THREE_POINTS_CROSS_PRODUCT(m_a, m_b, m_c) (((m_c) - (m_a)).cross((m_b) - (m_a)))

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

Vector3 NavMeshQueries3D::polygons_get_random_point(const LocalVector<gd::Polygon> &p_polygons, uint32_t p_navigation_layers, bool p_uniformly) {
	const LocalVector<gd::Polygon> &region_polygons = p_polygons;

	if (region_polygons.is_empty()) {
		return Vector3();
	}

	if (p_uniformly) {
		real_t accumulated_area = 0;
		RBMap<real_t, uint32_t> region_area_map;

		for (uint32_t rp_index = 0; rp_index < region_polygons.size(); rp_index++) {
			const gd::Polygon &region_polygon = region_polygons[rp_index];
			real_t polyon_area = region_polygon.surface_area;

			if (polyon_area == 0.0) {
				continue;
			}
			region_area_map[accumulated_area] = rp_index;
			accumulated_area += polyon_area;
		}
		if (region_area_map.is_empty() || accumulated_area == 0) {
			// All polygons have no real surface / no area.
			return Vector3();
		}

		real_t region_area_map_pos = Math::random(real_t(0), accumulated_area);

		RBMap<real_t, uint32_t>::Iterator region_E = region_area_map.find_closest(region_area_map_pos);
		ERR_FAIL_COND_V(!region_E, Vector3());
		uint32_t rrp_polygon_index = region_E->value;
		ERR_FAIL_UNSIGNED_INDEX_V(rrp_polygon_index, region_polygons.size(), Vector3());

		const gd::Polygon &rr_polygon = region_polygons[rrp_polygon_index];

		real_t accumulated_polygon_area = 0;
		RBMap<real_t, uint32_t> polygon_area_map;

		for (uint32_t rpp_index = 2; rpp_index < rr_polygon.points.size(); rpp_index++) {
			real_t face_area = Face3(rr_polygon.points[0].pos, rr_polygon.points[rpp_index - 1].pos, rr_polygon.points[rpp_index].pos).get_area();

			if (face_area == 0.0) {
				continue;
			}
			polygon_area_map[accumulated_polygon_area] = rpp_index;
			accumulated_polygon_area += face_area;
		}
		if (polygon_area_map.is_empty() || accumulated_polygon_area == 0) {
			// All faces have no real surface / no area.
			return Vector3();
		}

		real_t polygon_area_map_pos = Math::random(real_t(0), accumulated_polygon_area);

		RBMap<real_t, uint32_t>::Iterator polygon_E = polygon_area_map.find_closest(polygon_area_map_pos);
		ERR_FAIL_COND_V(!polygon_E, Vector3());
		uint32_t rrp_face_index = polygon_E->value;
		ERR_FAIL_UNSIGNED_INDEX_V(rrp_face_index, rr_polygon.points.size(), Vector3());

		const Face3 face(rr_polygon.points[0].pos, rr_polygon.points[rrp_face_index - 1].pos, rr_polygon.points[rrp_face_index].pos);

		Vector3 face_random_position = face.get_random_point_inside();
		return face_random_position;

	} else {
		uint32_t rrp_polygon_index = Math::random(int(0), region_polygons.size() - 1);

		const gd::Polygon &rr_polygon = region_polygons[rrp_polygon_index];

		uint32_t rrp_face_index = Math::random(int(2), rr_polygon.points.size() - 1);

		const Face3 face(rr_polygon.points[0].pos, rr_polygon.points[rrp_face_index - 1].pos, rr_polygon.points[rrp_face_index].pos);

		Vector3 face_random_position = face.get_random_point_inside();
		return face_random_position;
	}
}

Vector<Vector3> NavMeshQueries3D::polygons_get_path(const LocalVector<gd::Polygon> &p_polygons, Vector3 p_origin, Vector3 p_destination, bool p_optimize, uint32_t p_navigation_layers, Vector<int32_t> *r_path_types, TypedArray<RID> *r_path_rids, Vector<int64_t> *r_path_owners, const Vector3 &p_map_up, uint32_t p_link_polygons_size) {
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
	for (const gd::Polygon &p : p_polygons) {
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
	navigation_polys.resize(p_polygons.size() + p_link_polygons_size);

	// Initialize the matching navigation polygon.
	gd::NavigationPoly &begin_navigation_poly = navigation_polys[begin_poly->id];
	begin_navigation_poly.poly = begin_poly;
	begin_navigation_poly.entry = begin_point;
	begin_navigation_poly.back_navigation_edge_pathway_start = begin_point;
	begin_navigation_poly.back_navigation_edge_pathway_end = begin_point;

	// Heap of polygons to travel next.
	gd::Heap<gd::NavigationPoly *, gd::NavPolyTravelCostGreaterThan, gd::NavPolyHeapIndexer>
			traversable_polys;
	traversable_polys.reserve(p_polygons.size() * 0.25);

	// This is an implementation of the A* algorithm.
	int least_cost_id = begin_poly->id;
	int prev_least_cost_id = -1;
	bool found_route = false;

	const gd::Polygon *reachable_end = nullptr;
	real_t distance_to_reachable_end = FLT_MAX;
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

				if (prev_least_cost_id != -1 && navigation_polys[prev_least_cost_id].poly->owner->get_self() != least_cost_poly.poly->owner->get_self()) {
					poly_enter_cost = least_cost_poly.poly->owner->get_enter_cost();
				}
				prev_least_cost_id = least_cost_id;

				Vector3 pathway[2] = { connection.pathway_start, connection.pathway_end };
				const Vector3 new_entry = Geometry3D::get_closest_point_to_segment(least_cost_poly.entry, pathway);
				const real_t new_traveled_distance = least_cost_poly.entry.distance_to(new_entry) * poly_travel_cost + poly_enter_cost + least_cost_poly.traveled_distance;

				// Check if the neighbor polygon has already been processed.
				gd::NavigationPoly &neighbor_poly = navigation_polys[connection.polygon->id];
				if (neighbor_poly.poly != nullptr) {
					// If the neighbor polygon hasn't been traversed yet and the new path leading to
					// it is shorter, update the polygon.
					if (neighbor_poly.traversable_poly_index < traversable_polys.size() &&
							new_traveled_distance < neighbor_poly.traveled_distance) {
						neighbor_poly.back_navigation_poly_id = least_cost_id;
						neighbor_poly.back_navigation_edge = connection.edge;
						neighbor_poly.back_navigation_edge_pathway_start = connection.pathway_start;
						neighbor_poly.back_navigation_edge_pathway_end = connection.pathway_end;
						neighbor_poly.traveled_distance = new_traveled_distance;
						neighbor_poly.distance_to_destination =
								new_entry.distance_to(end_point) *
								neighbor_poly.poly->owner->get_travel_cost();
						neighbor_poly.entry = new_entry;

						// Update the priority of the polygon in the heap.
						traversable_polys.shift(neighbor_poly.traversable_poly_index);
					}
				} else {
					// Initialize the matching navigation polygon.
					neighbor_poly.poly = connection.polygon;
					neighbor_poly.back_navigation_poly_id = least_cost_id;
					neighbor_poly.back_navigation_edge = connection.edge;
					neighbor_poly.back_navigation_edge_pathway_start = connection.pathway_start;
					neighbor_poly.back_navigation_edge_pathway_end = connection.pathway_end;
					neighbor_poly.traveled_distance = new_traveled_distance;
					neighbor_poly.distance_to_destination =
							new_entry.distance_to(end_point) *
							neighbor_poly.poly->owner->get_travel_cost();
					neighbor_poly.entry = new_entry;

					// Add the polygon to the heap of polygons to traverse next.
					traversable_polys.push(&neighbor_poly);
				}
			}
		}

		// When the heap of traversable polygons is empty at this point it means the end polygon is
		// unreachable.
		if (traversable_polys.is_empty()) {
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

			for (gd::NavigationPoly &nav_poly : navigation_polys) {
				nav_poly.poly = nullptr;
			}
			navigation_polys[begin_poly->id].poly = begin_poly;

			least_cost_id = begin_poly->id;
			prev_least_cost_id = -1;

			reachable_end = nullptr;

			continue;
		}

		// Pop the polygon with the lowest travel cost from the heap of traversable polygons.
		least_cost_id = traversable_polys.pop()->poly->id;

		// Store the farthest reachable end polygon in case our goal is not reachable.
		if (is_reachable) {
			real_t distance = navigation_polys[least_cost_id].entry.distance_to(p_destination);
			if (distance_to_reachable_end > distance) {
				distance_to_reachable_end = distance;
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
			if (THREE_POINTS_CROSS_PRODUCT(apex_point, left, right).dot(p_map_up) < 0) {
				SWAP(left, right);
			}

			bool skip = false;
			if (THREE_POINTS_CROSS_PRODUCT(apex_point, left_portal, left).dot(p_map_up) >= 0) {
				//process
				if (left_portal == apex_point || THREE_POINTS_CROSS_PRODUCT(apex_point, left, right_portal).dot(p_map_up) > 0) {
					left_poly = p;
					left_portal = left;
				} else {
					clip_path(navigation_polys, path, apex_poly, right_portal, right_poly, r_path_types, r_path_rids, r_path_owners, p_map_up);

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

			if (!skip && THREE_POINTS_CROSS_PRODUCT(apex_point, right_portal, right).dot(p_map_up) <= 0) {
				//process
				if (right_portal == apex_point || THREE_POINTS_CROSS_PRODUCT(apex_point, right, left_portal).dot(p_map_up) < 0) {
					right_poly = p;
					right_portal = right;
				} else {
					clip_path(navigation_polys, path, apex_poly, left_portal, left_poly, r_path_types, r_path_rids, r_path_owners, p_map_up);

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

Vector3 NavMeshQueries3D::polygons_get_closest_point_to_segment(const LocalVector<gd::Polygon> &p_polygons, const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision) {
	bool use_collision = p_use_collision;
	Vector3 closest_point;
	real_t closest_point_distance = FLT_MAX;

	for (const gd::Polygon &polygon : p_polygons) {
		// For each face check the distance to the segment.
		for (size_t point_id = 2; point_id < polygon.points.size(); point_id += 1) {
			const Face3 face(polygon.points[0].pos, polygon.points[point_id - 1].pos, polygon.points[point_id].pos);
			Vector3 intersection_point;
			if (face.intersects_segment(p_from, p_to, &intersection_point)) {
				const real_t d = p_from.distance_to(intersection_point);
				if (!use_collision) {
					closest_point = intersection_point;
					use_collision = true;
					closest_point_distance = d;
				} else if (closest_point_distance > d) {
					closest_point = intersection_point;
					closest_point_distance = d;
				}
			}
			// If segment does not itersect face, check the distance from segment's endpoints.
			else if (!use_collision) {
				const Vector3 p_from_closest = face.get_closest_point_to(p_from);
				const real_t d_p_from = p_from.distance_to(p_from_closest);
				if (closest_point_distance > d_p_from) {
					closest_point = p_from_closest;
					closest_point_distance = d_p_from;
				}

				const Vector3 p_to_closest = face.get_closest_point_to(p_to);
				const real_t d_p_to = p_to.distance_to(p_to_closest);
				if (closest_point_distance > d_p_to) {
					closest_point = p_to_closest;
					closest_point_distance = d_p_to;
				}
			}
		}
		// Finally, check for a case when shortest distance is between some point located on a face's edge and some point located on a line segment.
		if (!use_collision) {
			for (size_t point_id = 0; point_id < polygon.points.size(); point_id += 1) {
				Vector3 a, b;

				Geometry3D::get_closest_points_between_segments(
						p_from,
						p_to,
						polygon.points[point_id].pos,
						polygon.points[(point_id + 1) % polygon.points.size()].pos,
						a,
						b);

				const real_t d = a.distance_to(b);
				if (d < closest_point_distance) {
					closest_point_distance = d;
					closest_point = b;
				}
			}
		}
	}

	return closest_point;
}

Vector3 NavMeshQueries3D::polygons_get_closest_point(const LocalVector<gd::Polygon> &p_polygons, const Vector3 &p_point) {
	gd::ClosestPointQueryResult cp = polygons_get_closest_point_info(p_polygons, p_point);
	return cp.point;
}

Vector3 NavMeshQueries3D::polygons_get_closest_point_normal(const LocalVector<gd::Polygon> &p_polygons, const Vector3 &p_point) {
	gd::ClosestPointQueryResult cp = polygons_get_closest_point_info(p_polygons, p_point);
	return cp.normal;
}

gd::ClosestPointQueryResult NavMeshQueries3D::polygons_get_closest_point_info(const LocalVector<gd::Polygon> &p_polygons, const Vector3 &p_point) {
	gd::ClosestPointQueryResult result;
	real_t closest_point_distance_squared = FLT_MAX;

	for (const gd::Polygon &polygon : p_polygons) {
		for (size_t point_id = 2; point_id < polygon.points.size(); point_id += 1) {
			const Face3 face(polygon.points[0].pos, polygon.points[point_id - 1].pos, polygon.points[point_id].pos);
			const Vector3 closest_point_on_face = face.get_closest_point_to(p_point);
			const real_t distance_squared_to_point = closest_point_on_face.distance_squared_to(p_point);
			if (distance_squared_to_point < closest_point_distance_squared) {
				result.point = closest_point_on_face;
				result.normal = face.get_plane().normal;
				result.owner = polygon.owner->get_self();
				closest_point_distance_squared = distance_squared_to_point;
			}
		}
	}

	return result;
}

RID NavMeshQueries3D::polygons_get_closest_point_owner(const LocalVector<gd::Polygon> &p_polygons, const Vector3 &p_point) {
	gd::ClosestPointQueryResult cp = polygons_get_closest_point_info(p_polygons, p_point);
	return cp.owner;
}

void NavMeshQueries3D::clip_path(const LocalVector<gd::NavigationPoly> &p_navigation_polys, Vector<Vector3> &path, const gd::NavigationPoly *from_poly, const Vector3 &p_to_point, const gd::NavigationPoly *p_to_poly, Vector<int32_t> *r_path_types, TypedArray<RID> *r_path_rids, Vector<int64_t> *r_path_owners, const Vector3 &p_map_up) {
	Vector3 from = path[path.size() - 1];

	if (from.is_equal_approx(p_to_point)) {
		return;
	}

	Plane cut_plane;
	cut_plane.normal = (from - p_to_point).cross(p_map_up);
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

#endif // _3D_DISABLED
