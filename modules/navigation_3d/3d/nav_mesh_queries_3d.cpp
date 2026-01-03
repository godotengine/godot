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

#include "nav_mesh_queries_3d.h"

#include "../nav_base_3d.h"
#include "../nav_map_3d.h"
#include "nav_region_iteration_3d.h"

#include "core/math/geometry_2d.h"
#include "core/math/geometry_3d.h"

using namespace Nav3D;

#define THREE_POINTS_CROSS_PRODUCT(m_a, m_b, m_c) (((m_c) - (m_a)).cross((m_b) - (m_a)))

bool NavMeshQueries3D::emit_callback(const Callable &p_callback) {
	ERR_FAIL_COND_V(!p_callback.is_valid(), false);

	Callable::CallError ce;
	Variant result;
	p_callback.callp(nullptr, 0, result, ce);

	return ce.error == Callable::CallError::CALL_OK;
}

Vector3 NavMeshQueries3D::polygons_get_random_point(const LocalVector<Polygon> &p_polygons, uint32_t p_navigation_layers, bool p_uniformly) {
	const LocalVector<Polygon> &region_polygons = p_polygons;

	if (region_polygons.is_empty()) {
		return Vector3();
	}

	if (p_uniformly) {
		real_t accumulated_area = 0;
		RBMap<real_t, uint32_t> region_area_map;

		for (uint32_t rp_index = 0; rp_index < region_polygons.size(); rp_index++) {
			const Polygon &region_polygon = region_polygons[rp_index];
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

		const Polygon &rr_polygon = region_polygons[rrp_polygon_index];

		real_t accumulated_polygon_area = 0;
		RBMap<real_t, uint32_t> polygon_area_map;

		for (uint32_t rpp_index = 2; rpp_index < rr_polygon.vertices.size(); rpp_index++) {
			real_t face_area = Face3(rr_polygon.vertices[0], rr_polygon.vertices[rpp_index - 1], rr_polygon.vertices[rpp_index]).get_area();

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
		ERR_FAIL_UNSIGNED_INDEX_V(rrp_face_index, rr_polygon.vertices.size(), Vector3());

		const Face3 face(rr_polygon.vertices[0], rr_polygon.vertices[rrp_face_index - 1], rr_polygon.vertices[rrp_face_index]);

		Vector3 face_random_position = face.get_random_point_inside();
		return face_random_position;

	} else {
		uint32_t rrp_polygon_index = Math::random(int(0), region_polygons.size() - 1);

		const Polygon &rr_polygon = region_polygons[rrp_polygon_index];

		uint32_t rrp_face_index = Math::random(int(2), rr_polygon.vertices.size() - 1);

		const Face3 face(rr_polygon.vertices[0], rr_polygon.vertices[rrp_face_index - 1], rr_polygon.vertices[rrp_face_index]);

		Vector3 face_random_position = face.get_random_point_inside();
		return face_random_position;
	}
}

void NavMeshQueries3D::_query_task_push_back_point_with_metadata(NavMeshPathQueryTask3D &p_query_task, const Vector3 &p_point, const Polygon *p_point_polygon) {
	if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_TYPES)) {
		p_query_task.path_meta_point_types.push_back(p_point_polygon->owner->get_type());
	}

	if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_RIDS)) {
		p_query_task.path_meta_point_rids.push_back(p_point_polygon->owner->get_self());
	}

	if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_OWNERS)) {
		p_query_task.path_meta_point_owners.push_back(p_point_polygon->owner->get_owner_id());
	}

	p_query_task.path_points.push_back(p_point);
}

void NavMeshQueries3D::map_query_path(NavMap3D *map, const Ref<NavigationPathQueryParameters3D> &p_query_parameters, Ref<NavigationPathQueryResult3D> p_query_result, const Callable &p_callback) {
	ERR_FAIL_NULL(map);
	ERR_FAIL_COND(p_query_parameters.is_null());
	ERR_FAIL_COND(p_query_result.is_null());

	using namespace NavigationDefaults3D;

	NavMeshQueries3D::NavMeshPathQueryTask3D query_task;
	query_task.start_position = p_query_parameters->get_start_position();
	query_task.target_position = p_query_parameters->get_target_position();
	query_task.navigation_layers = p_query_parameters->get_navigation_layers();
	query_task.callback = p_callback;

	const TypedArray<RID> &_excluded_regions = p_query_parameters->get_excluded_regions();
	const TypedArray<RID> &_included_regions = p_query_parameters->get_included_regions();

	uint32_t _excluded_region_count = _excluded_regions.size();
	uint32_t _included_region_count = _included_regions.size();

	query_task.exclude_regions = _excluded_region_count > 0;
	query_task.include_regions = _included_region_count > 0;

	if (query_task.exclude_regions) {
		query_task.excluded_regions.resize(_excluded_region_count);
		for (uint32_t i = 0; i < _excluded_region_count; i++) {
			query_task.excluded_regions[i] = _excluded_regions[i];
		}
	}

	if (query_task.include_regions) {
		query_task.included_regions.resize(_included_region_count);
		for (uint32_t i = 0; i < _included_region_count; i++) {
			query_task.included_regions[i] = _included_regions[i];
		}
	}

	switch (p_query_parameters->get_pathfinding_algorithm()) {
		case NavigationPathQueryParameters3D::PathfindingAlgorithm::PATHFINDING_ALGORITHM_ASTAR: {
			query_task.pathfinding_algorithm = PathfindingAlgorithm::PATHFINDING_ALGORITHM_ASTAR;
		} break;
		default: {
			WARN_PRINT("No match for used PathfindingAlgorithm - fallback to default");
			query_task.pathfinding_algorithm = PathfindingAlgorithm::PATHFINDING_ALGORITHM_ASTAR;
		} break;
	}

	switch (p_query_parameters->get_path_postprocessing()) {
		case NavigationPathQueryParameters3D::PathPostProcessing::PATH_POSTPROCESSING_CORRIDORFUNNEL: {
			query_task.path_postprocessing = PathPostProcessing::PATH_POSTPROCESSING_CORRIDORFUNNEL;
		} break;
		case NavigationPathQueryParameters3D::PathPostProcessing::PATH_POSTPROCESSING_EDGECENTERED: {
			query_task.path_postprocessing = PathPostProcessing::PATH_POSTPROCESSING_EDGECENTERED;
		} break;
		case NavigationPathQueryParameters3D::PathPostProcessing::PATH_POSTPROCESSING_NONE: {
			query_task.path_postprocessing = PathPostProcessing::PATH_POSTPROCESSING_NONE;
		} break;
		default: {
			WARN_PRINT("No match for used PathPostProcessing - fallback to default");
			query_task.path_postprocessing = PathPostProcessing::PATH_POSTPROCESSING_CORRIDORFUNNEL;
		} break;
	}

	query_task.metadata_flags = (int64_t)p_query_parameters->get_metadata_flags();
	query_task.simplify_path = p_query_parameters->get_simplify_path();
	query_task.simplify_epsilon = p_query_parameters->get_simplify_epsilon();
	query_task.path_return_max_length = p_query_parameters->get_path_return_max_length();
	query_task.path_return_max_radius = p_query_parameters->get_path_return_max_radius();
	query_task.path_search_max_polygons = p_query_parameters->get_path_search_max_polygons();
	query_task.path_search_max_distance = p_query_parameters->get_path_search_max_distance();
	query_task.status = NavMeshPathQueryTask3D::TaskStatus::QUERY_STARTED;

	map->query_path(query_task);

	p_query_result->set_data(
			query_task.path_points,
			query_task.path_meta_point_types,
			query_task.path_meta_point_rids,
			query_task.path_meta_point_owners);
	p_query_result->set_path_length(query_task.path_length);

	if (query_task.callback.is_valid()) {
		if (emit_callback(query_task.callback)) {
			query_task.status = NavMeshPathQueryTask3D::TaskStatus::CALLBACK_DISPATCHED;
		} else {
			query_task.status = NavMeshPathQueryTask3D::TaskStatus::CALLBACK_FAILED;
		}
	}
}

void NavMeshQueries3D::_query_task_find_start_end_positions(NavMeshPathQueryTask3D &p_query_task, const NavMapIteration3D &p_map_iteration) {
	real_t begin_d_squared = FLT_MAX;
	real_t end_d_squared = FLT_MAX;

	const LocalVector<Ref<NavRegionIteration3D>> &regions = p_map_iteration.region_iterations;

	for (const Ref<NavRegionIteration3D> &region : regions) {
		if (!_query_task_is_connection_owner_usable(p_query_task, region.ptr())) {
			continue;
		}

		// Find the initial poly and the end poly on this map.
		for (const Polygon &p : region->get_navmesh_polygons()) {
			// Only consider the polygon if it in a region with compatible layers.
			if ((p_query_task.navigation_layers & p.owner->get_navigation_layers()) == 0) {
				continue;
			}

			// For each face check the distance between the origin/destination.
			for (uint32_t point_id = 2; point_id < p.vertices.size(); point_id++) {
				const Face3 face(p.vertices[0], p.vertices[point_id - 1], p.vertices[point_id]);

				Vector3 point = face.get_closest_point_to(p_query_task.start_position);
				real_t distance_to_point_squared = point.distance_squared_to(p_query_task.start_position);
				if (distance_to_point_squared < begin_d_squared) {
					begin_d_squared = distance_to_point_squared;
					p_query_task.begin_polygon = &p;
					p_query_task.begin_position = point;
				}

				point = face.get_closest_point_to(p_query_task.target_position);
				distance_to_point_squared = point.distance_squared_to(p_query_task.target_position);
				if (distance_to_point_squared < end_d_squared) {
					end_d_squared = distance_to_point_squared;
					p_query_task.end_polygon = &p;
					p_query_task.end_position = point;
				}
			}
		}
	}
}

void NavMeshQueries3D::_query_task_search_polygon_connections(NavMeshPathQueryTask3D &p_query_task, const Connection &p_connection, uint32_t p_least_cost_id, const NavigationPoly &p_least_cost_poly, real_t p_poly_enter_cost, const Vector3 &p_end_point) {
	const NavBaseIteration3D *connection_owner = p_connection.polygon->owner;
	ERR_FAIL_NULL(connection_owner);
	const bool owner_is_usable = _query_task_is_connection_owner_usable(p_query_task, connection_owner);
	if (!owner_is_usable) {
		return;
	}

	Heap<NavigationPoly *, NavPolyTravelCostGreaterThan, NavPolyHeapIndexer>
			&traversable_polys = p_query_task.path_query_slot->traversable_polys;
	LocalVector<NavigationPoly> &navigation_polys = p_query_task.path_query_slot->path_corridor;

	real_t poly_travel_cost = p_least_cost_poly.poly->owner->get_travel_cost();

	Vector3 new_entry = Geometry3D::get_closest_point_to_segment(p_least_cost_poly.entry, p_connection.pathway_start, p_connection.pathway_end);
	real_t new_traveled_distance = p_least_cost_poly.entry.distance_to(new_entry) * poly_travel_cost + p_poly_enter_cost + p_least_cost_poly.traveled_distance;

	// Check if the neighbor polygon has already been processed.
	NavigationPoly &neighbor_poly = navigation_polys[p_query_task.path_query_slot->poly_to_id[p_connection.polygon]];
	if (new_traveled_distance < neighbor_poly.traveled_distance) {
		// Add the polygon to the heap of polygons to traverse next.
		neighbor_poly.back_navigation_poly_id = p_least_cost_id;
		neighbor_poly.back_navigation_edge = p_connection.edge;
		neighbor_poly.back_navigation_edge_pathway_start = p_connection.pathway_start;
		neighbor_poly.back_navigation_edge_pathway_end = p_connection.pathway_end;
		neighbor_poly.traveled_distance = new_traveled_distance;
		neighbor_poly.distance_to_destination =
				new_entry.distance_to(p_end_point) *
				connection_owner->get_travel_cost();
		neighbor_poly.entry = new_entry;

		if (neighbor_poly.traversable_poly_index != traversable_polys.INVALID_INDEX) {
			traversable_polys.shift(neighbor_poly.traversable_poly_index);
		} else {
			neighbor_poly.poly = p_connection.polygon;
			traversable_polys.push(&neighbor_poly);
		}
	}
}

void NavMeshQueries3D::_query_task_build_path_corridor(NavMeshPathQueryTask3D &p_query_task, const NavMapIteration3D &p_map_iteration) {
	const Vector3 p_target_position = p_query_task.target_position;
	const Polygon *begin_poly = p_query_task.begin_polygon;
	const Polygon *end_poly = p_query_task.end_polygon;
	Vector3 begin_point = p_query_task.begin_position;
	Vector3 end_point = p_query_task.end_position;

	// Heap of polygons to travel next.
	Heap<NavigationPoly *, NavPolyTravelCostGreaterThan, NavPolyHeapIndexer>
			&traversable_polys = p_query_task.path_query_slot->traversable_polys;
	traversable_polys.clear();

	LocalVector<NavigationPoly> &navigation_polys = p_query_task.path_query_slot->path_corridor;
	for (NavigationPoly &polygon : navigation_polys) {
		polygon.reset();
	}

	// Initialize the matching navigation polygon.
	NavigationPoly &begin_navigation_poly = navigation_polys[p_query_task.path_query_slot->poly_to_id[begin_poly]];
	begin_navigation_poly.poly = begin_poly;
	begin_navigation_poly.entry = begin_point;
	begin_navigation_poly.back_navigation_edge_pathway_start = begin_point;
	begin_navigation_poly.back_navigation_edge_pathway_end = begin_point;
	begin_navigation_poly.traveled_distance = 0.f;

	// This is an implementation of the A* algorithm.
	uint32_t least_cost_id = p_query_task.path_query_slot->poly_to_id[begin_poly];
	bool found_route = false;

	const Polygon *reachable_end = nullptr;
	real_t distance_to_reachable_end = FLT_MAX;
	bool is_reachable = true;
	real_t poly_enter_cost = 0.0;

	const HashMap<const NavBaseIteration3D *, LocalVector<LocalVector<Nav3D::Connection>>> &navbases_polygons_external_connections = p_map_iteration.navbases_polygons_external_connections;

	// True if we reached the max polygon search count or distance from the begin position.
	bool path_search_max_reached = false;

	const float path_search_max_distance_sqr = p_query_task.path_search_max_distance * p_query_task.path_search_max_distance;
	bool has_path_search_max_distance = path_search_max_distance_sqr > 0.0;

	int processed_polygon_count = 0;
	bool has_path_search_max_polygons = p_query_task.path_search_max_polygons > 0;

	bool has_path_search_max = p_query_task.path_search_max_polygons > 0 || path_search_max_distance_sqr > 0.0;

	while (true) {
		const NavigationPoly &least_cost_poly = navigation_polys[least_cost_id];

		const NavBaseIteration3D *least_cost_navbase = least_cost_poly.poly->owner;

		processed_polygon_count += 1;

		const uint32_t navbase_local_polygon_id = least_cost_poly.poly->id;
		const LocalVector<LocalVector<Connection>> &navbase_polygons_to_connections = least_cost_poly.poly->owner->get_internal_connections();

		if (navbase_polygons_to_connections.size() > 0) {
			const LocalVector<Connection> &polygon_connections = navbase_polygons_to_connections[navbase_local_polygon_id];

			for (const Connection &connection : polygon_connections) {
				_query_task_search_polygon_connections(p_query_task, connection, least_cost_id, least_cost_poly, poly_enter_cost, end_point);
			}
		}

		// Search region external navmesh polygon connections, aka connections to other regions created by outline edge merge or links.
		for (const Connection &connection : navbases_polygons_external_connections[least_cost_navbase][navbase_local_polygon_id]) {
			_query_task_search_polygon_connections(p_query_task, connection, least_cost_id, least_cost_poly, poly_enter_cost, end_point);
		}

		if (has_path_search_max && !path_search_max_reached) {
			if (has_path_search_max_polygons && processed_polygon_count >= p_query_task.path_search_max_polygons) {
				path_search_max_reached = true;
				traversable_polys.clear();
			} else if (has_path_search_max_distance && begin_point.distance_squared_to(least_cost_poly.entry) > path_search_max_distance_sqr) {
				path_search_max_reached = true;
				traversable_polys.clear();
			}
		}

		poly_enter_cost = 0;
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
			real_t end_d = FLT_MAX;

			for (uint32_t point_id = 2; point_id < end_poly->vertices.size(); point_id++) {
				Face3 f(end_poly->vertices[0], end_poly->vertices[point_id - 1], end_poly->vertices[point_id]);
				Vector3 spoint = f.get_closest_point_to(p_target_position);
				real_t dpoint = spoint.distance_squared_to(p_target_position);
				if (dpoint < end_d) {
					end_point = spoint;
					end_d = dpoint;
				}
			}

			// Search all faces of start polygon as well.
			bool closest_point_on_start_poly = false;

			for (uint32_t point_id = 2; point_id < begin_poly->vertices.size(); point_id++) {
				Face3 f(begin_poly->vertices[0], begin_poly->vertices[point_id - 1], begin_poly->vertices[point_id]);
				Vector3 spoint = f.get_closest_point_to(p_target_position);
				real_t dpoint = spoint.distance_squared_to(p_target_position);
				if (dpoint < end_d) {
					end_point = spoint;
					end_d = dpoint;
					closest_point_on_start_poly = true;
				}
			}

			if (closest_point_on_start_poly) {
				// No point to run PostProcessing when start and end convex polygon is the same.
				p_query_task.path_clear();

				_query_task_push_back_point_with_metadata(p_query_task, begin_point, begin_poly);
				_query_task_push_back_point_with_metadata(p_query_task, end_point, begin_poly);
				p_query_task.status = NavMeshPathQueryTask3D::TaskStatus::QUERY_FINISHED;
				return;
			}

			for (NavigationPoly &nav_poly : navigation_polys) {
				nav_poly.poly = nullptr;
				nav_poly.traveled_distance = FLT_MAX;
			}
			uint32_t _bp_id = p_query_task.path_query_slot->poly_to_id[begin_poly];
			navigation_polys[_bp_id].poly = begin_poly;
			navigation_polys[_bp_id].traveled_distance = 0;
			least_cost_id = _bp_id;
			reachable_end = nullptr;
		} else {
			// Pop the polygon with the lowest travel cost from the heap of traversable polygons.
			least_cost_id = p_query_task.path_query_slot->poly_to_id[traversable_polys.pop()->poly];

			// Store the farthest reachable end polygon in case our goal is not reachable.
			if (is_reachable) {
				real_t distance = navigation_polys[least_cost_id].entry.distance_squared_to(p_target_position);
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

			if (navigation_polys[least_cost_id].poly->owner->get_self() != least_cost_poly.poly->owner->get_self()) {
				ERR_FAIL_NULL(least_cost_poly.poly->owner);
				poly_enter_cost = least_cost_poly.poly->owner->get_enter_cost();
			}
		}
	}

	// We did not find a route but we have both a start polygon and an end polygon at this point.
	// Usually this happens because there was not a single external or internal connected edge, e.g. our start polygon is an isolated, single convex polygon.
	if (!found_route) {
		real_t end_d = FLT_MAX;
		// Search all faces of the start polygon for the closest point to our target position.

		for (uint32_t point_id = 2; point_id < begin_poly->vertices.size(); point_id++) {
			Face3 f(begin_poly->vertices[0], begin_poly->vertices[point_id - 1], begin_poly->vertices[point_id]);
			Vector3 spoint = f.get_closest_point_to(p_target_position);
			real_t dpoint = spoint.distance_squared_to(p_target_position);
			if (dpoint < end_d) {
				end_point = spoint;
				end_d = dpoint;
			}
		}

		p_query_task.path_clear();

		_query_task_push_back_point_with_metadata(p_query_task, begin_point, begin_poly);
		_query_task_push_back_point_with_metadata(p_query_task, end_point, begin_poly);
		p_query_task.status = NavMeshPathQueryTask3D::TaskStatus::QUERY_FINISHED;
	} else {
		p_query_task.end_position = end_point;
		p_query_task.end_polygon = end_poly;
		p_query_task.begin_position = begin_point;
		p_query_task.begin_polygon = begin_poly;
		p_query_task.least_cost_id = least_cost_id;
	}
}

void NavMeshQueries3D::query_task_map_iteration_get_path(NavMeshPathQueryTask3D &p_query_task, const NavMapIteration3D &p_map_iteration) {
	p_query_task.path_clear();

	_query_task_find_start_end_positions(p_query_task, p_map_iteration);

	// Check for trivial cases.
	if (!p_query_task.begin_polygon || !p_query_task.end_polygon) {
		p_query_task.status = NavMeshPathQueryTask3D::TaskStatus::QUERY_FINISHED;
		return;
	}
	if (p_query_task.begin_polygon == p_query_task.end_polygon) {
		p_query_task.path_clear();
		_query_task_push_back_point_with_metadata(p_query_task, p_query_task.begin_position, p_query_task.begin_polygon);
		_query_task_push_back_point_with_metadata(p_query_task, p_query_task.end_position, p_query_task.end_polygon);
		_query_task_process_path_result_limits(p_query_task);
		p_query_task.status = NavMeshPathQueryTask3D::TaskStatus::QUERY_FINISHED;
		return;
	}

	_query_task_build_path_corridor(p_query_task, p_map_iteration);

	if (p_query_task.status == NavMeshPathQueryTask3D::TaskStatus::QUERY_FINISHED || p_query_task.status == NavMeshPathQueryTask3D::TaskStatus::QUERY_FAILED) {
		_query_task_process_path_result_limits(p_query_task);
		return;
	}

	// Post-Process path.
	switch (p_query_task.path_postprocessing) {
		case PathPostProcessing::PATH_POSTPROCESSING_CORRIDORFUNNEL: {
			_query_task_post_process_corridorfunnel(p_query_task);
		} break;
		case PathPostProcessing::PATH_POSTPROCESSING_EDGECENTERED: {
			_query_task_post_process_edgecentered(p_query_task);
		} break;
		case PathPostProcessing::PATH_POSTPROCESSING_NONE: {
			_query_task_post_process_nopostprocessing(p_query_task);
		} break;
		default: {
			WARN_PRINT("No match for used PathPostProcessing - fallback to default");
			_query_task_post_process_corridorfunnel(p_query_task);
		} break;
	}

	p_query_task.path_reverse();

	if (p_query_task.simplify_path) {
		_query_task_simplified_path_points(p_query_task);
	}

	_query_task_process_path_result_limits(p_query_task);

#ifdef DEBUG_ENABLED
	// Ensure post conditions as path meta arrays if used MUST match in array size with the path points.
	if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_TYPES)) {
		DEV_ASSERT(p_query_task.path_points.size() == p_query_task.path_meta_point_types.size());
	}

	if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_RIDS)) {
		DEV_ASSERT(p_query_task.path_points.size() == p_query_task.path_meta_point_rids.size());
	}

	if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_OWNERS)) {
		DEV_ASSERT(p_query_task.path_points.size() == p_query_task.path_meta_point_owners.size());
	}
#endif // DEBUG_ENABLED

	p_query_task.status = NavMeshPathQueryTask3D::TaskStatus::QUERY_FINISHED;
}

float NavMeshQueries3D::_calculate_path_length(const LocalVector<Vector3> &p_path, uint32_t p_start_index, uint32_t p_end_index) {
	const uint32_t path_size = p_path.size();
	if (path_size < 2) {
		return 0.0;
	}

	ERR_FAIL_COND_V(p_start_index >= p_end_index, 0.0);
	ERR_FAIL_COND_V(p_start_index >= path_size - 1, 0.0);
	ERR_FAIL_COND_V(p_end_index >= path_size, 0.0);

	const Vector3 *path_ptr = p_path.ptr();

	float path_length = 0.0;

	for (uint32_t i = p_start_index; i < p_end_index; i++) {
		const Vector3 &vertex1 = path_ptr[i];
		const Vector3 &vertex2 = path_ptr[i + 1];
		float edge_length = vertex1.distance_to(vertex2);
		path_length += edge_length;
	}

	return path_length;
}

void NavMeshQueries3D::_query_task_process_path_result_limits(NavMeshPathQueryTask3D &p_query_task) {
	if (p_query_task.path_points.size() < 2) {
		return;
	}

	bool check_max_length = p_query_task.path_return_max_length > 0.0;
	bool check_max_radius = p_query_task.path_return_max_radius > 0.0;

	if (!check_max_length && !check_max_radius) {
		p_query_task.path_length = _calculate_path_length(p_query_task.path_points, 0, p_query_task.path_points.size() - 1);
		return;
	}

	LocalVector<Vector3> &path = p_query_task.path_points;

	const float max_length = p_query_task.path_return_max_length;
	const float max_radius = p_query_task.path_return_max_radius;
	const float max_radius_sqr = max_radius * max_radius;

	const Vector3 &start_pos = path[0];

	float accumulated_path_length = 0.0;

	Vector3 *path_ptrw = path.ptr();

	uint32_t path_max_size = path.size();
	bool path_max_reached = false;

	for (uint32_t i = 0; i < path.size() - 1; i++) {
		uint32_t next_index = i + 1;
		const Vector3 &vertex1 = path_ptrw[i];
		Vector3 &vertex2 = path_ptrw[next_index];

		float edge_length = (vertex2 - vertex1).length();

		if (check_max_radius && start_pos.distance_squared_to(vertex2) > max_radius_sqr) {
			// Path point segment goes over max radius, clip it.
			Vector3 intersect_positon, intersect_normal;
			bool intersected = Geometry3D::segment_intersects_sphere(vertex2, vertex1, start_pos, max_radius, &intersect_positon, &intersect_normal);
			if (intersected) {
				edge_length = (intersect_positon - vertex1).length();

				path_ptrw[next_index] = intersect_positon;
				path_max_size = next_index + 1;
				path_max_reached = true;
			}
		}

		if (check_max_length && accumulated_path_length + edge_length > max_length) {
			// Path point segment goes over max length, clip it.
			edge_length = max_length - accumulated_path_length;
			Vector3 edge_direction = vertex1.direction_to(vertex2);

			path_ptrw[next_index] = vertex1 + (edge_direction * edge_length);
			path_max_size = next_index + 1;

			p_query_task.path_length = accumulated_path_length + edge_length;
			path_max_reached = true;
		}

		accumulated_path_length += edge_length;

		if (path_max_reached) {
			break;
		}
	}

	p_query_task.path_length = accumulated_path_length;

	if (path_max_size < path.size()) {
		p_query_task.path_points.resize(path_max_size);

		if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_TYPES)) {
			p_query_task.path_meta_point_types.resize(path_max_size);
		}

		if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_RIDS)) {
			p_query_task.path_meta_point_rids.resize(path_max_size);
		}

		if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_OWNERS)) {
			p_query_task.path_meta_point_owners.resize(path_max_size);
		}
	}
}

void NavMeshQueries3D::_query_task_simplified_path_points(NavMeshPathQueryTask3D &p_query_task) {
	if (!p_query_task.simplify_path || p_query_task.path_points.size() <= 2) {
		return;
	}

	const LocalVector<uint32_t> &simplified_path_indices = NavMeshQueries3D::get_simplified_path_indices(p_query_task.path_points, p_query_task.simplify_epsilon);

	uint32_t index_count = simplified_path_indices.size();

	{
		Vector3 *points_ptr = p_query_task.path_points.ptr();
		for (uint32_t i = 0; i < index_count; i++) {
			points_ptr[i] = points_ptr[simplified_path_indices[i]];
		}
		p_query_task.path_points.resize(index_count);
	}

	if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_TYPES)) {
		int32_t *types_ptr = p_query_task.path_meta_point_types.ptr();
		for (uint32_t i = 0; i < index_count; i++) {
			types_ptr[i] = types_ptr[simplified_path_indices[i]];
		}
		p_query_task.path_meta_point_types.resize(index_count);
	}

	if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_RIDS)) {
		RID *rids_ptr = p_query_task.path_meta_point_rids.ptr();
		for (uint32_t i = 0; i < index_count; i++) {
			rids_ptr[i] = rids_ptr[simplified_path_indices[i]];
		}
		p_query_task.path_meta_point_rids.resize(index_count);
	}

	if (p_query_task.metadata_flags.has_flag(PathMetadataFlags::PATH_INCLUDE_OWNERS)) {
		int64_t *owners_ptr = p_query_task.path_meta_point_owners.ptr();
		for (uint32_t i = 0; i < index_count; i++) {
			owners_ptr[i] = owners_ptr[simplified_path_indices[i]];
		}
		p_query_task.path_meta_point_owners.resize(index_count);
	}
}

void NavMeshQueries3D::_query_task_post_process_corridorfunnel(NavMeshPathQueryTask3D &p_query_task) {
	Vector3 end_point = p_query_task.end_position;
	const Polygon *end_poly = p_query_task.end_polygon;
	Vector3 begin_point = p_query_task.begin_position;
	const Polygon *begin_poly = p_query_task.begin_polygon;
	uint32_t least_cost_id = p_query_task.least_cost_id;
	LocalVector<NavigationPoly> &navigation_polys = p_query_task.path_query_slot->path_corridor;
	Vector3 p_map_up = p_query_task.map_up;

	// Set the apex poly/point to the end point
	NavigationPoly *apex_poly = &navigation_polys[least_cost_id];

	const Vector3 back_edge_closest_point = Geometry3D::get_closest_point_to_segment(end_point, apex_poly->back_navigation_edge_pathway_start, apex_poly->back_navigation_edge_pathway_end);
	if (end_point.is_equal_approx(back_edge_closest_point)) {
		// The end point is basically on top of the last crossed edge, funneling around the corners would at best do nothing.
		// At worst it would add an unwanted path point before the last point due to precision issues so skip to the next polygon.
		if (apex_poly->back_navigation_poly_id != -1) {
			apex_poly = &navigation_polys[apex_poly->back_navigation_poly_id];
		}
	}

	Vector3 apex_point = end_point;

	NavigationPoly *left_poly = apex_poly;
	Vector3 left_portal = apex_point;
	NavigationPoly *right_poly = apex_poly;
	Vector3 right_portal = apex_point;

	NavigationPoly *p = apex_poly;

	_query_task_push_back_point_with_metadata(p_query_task, end_point, end_poly);

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
				_query_task_clip_path(p_query_task, apex_poly, right_portal, right_poly);

				apex_point = right_portal;
				p = right_poly;
				left_poly = p;
				apex_poly = p;
				left_portal = apex_point;
				right_portal = apex_point;

				_query_task_push_back_point_with_metadata(p_query_task, apex_point, apex_poly->poly);
				skip = true;
			}
		}

		if (!skip && THREE_POINTS_CROSS_PRODUCT(apex_point, right_portal, right).dot(p_map_up) <= 0) {
			//process
			if (right_portal == apex_point || THREE_POINTS_CROSS_PRODUCT(apex_point, right, left_portal).dot(p_map_up) < 0) {
				right_poly = p;
				right_portal = right;
			} else {
				_query_task_clip_path(p_query_task, apex_poly, left_portal, left_poly);

				apex_point = left_portal;
				p = left_poly;
				right_poly = p;
				apex_poly = p;
				right_portal = apex_point;
				left_portal = apex_point;

				_query_task_push_back_point_with_metadata(p_query_task, apex_point, apex_poly->poly);
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
	if (p_query_task.path_points[p_query_task.path_points.size() - 1] != begin_point) {
		_query_task_push_back_point_with_metadata(p_query_task, begin_point, begin_poly);
	}
}

void NavMeshQueries3D::_query_task_post_process_edgecentered(NavMeshPathQueryTask3D &p_query_task) {
	Vector3 end_point = p_query_task.end_position;
	const Polygon *end_poly = p_query_task.end_polygon;
	Vector3 begin_point = p_query_task.begin_position;
	const Polygon *begin_poly = p_query_task.begin_polygon;
	uint32_t least_cost_id = p_query_task.least_cost_id;
	LocalVector<NavigationPoly> &navigation_polys = p_query_task.path_query_slot->path_corridor;

	_query_task_push_back_point_with_metadata(p_query_task, end_point, end_poly);

	// Add mid points
	int np_id = least_cost_id;
	while (np_id != -1 && navigation_polys[np_id].back_navigation_poly_id != -1) {
		if (navigation_polys[np_id].back_navigation_edge != -1) {
			int prev = navigation_polys[np_id].back_navigation_edge;
			int prev_n = (navigation_polys[np_id].back_navigation_edge + 1) % navigation_polys[np_id].poly->vertices.size();
			Vector3 point = (navigation_polys[np_id].poly->vertices[prev] + navigation_polys[np_id].poly->vertices[prev_n]) * 0.5;

			_query_task_push_back_point_with_metadata(p_query_task, point, navigation_polys[np_id].poly);
		} else {
			_query_task_push_back_point_with_metadata(p_query_task, navigation_polys[np_id].entry, navigation_polys[np_id].poly);
		}

		np_id = navigation_polys[np_id].back_navigation_poly_id;
	}

	_query_task_push_back_point_with_metadata(p_query_task, begin_point, begin_poly);
}

void NavMeshQueries3D::_query_task_post_process_nopostprocessing(NavMeshPathQueryTask3D &p_query_task) {
	Vector3 end_point = p_query_task.end_position;
	const Polygon *end_poly = p_query_task.end_polygon;
	Vector3 begin_point = p_query_task.begin_position;
	const Polygon *begin_poly = p_query_task.begin_polygon;
	uint32_t least_cost_id = p_query_task.least_cost_id;
	LocalVector<NavigationPoly> &navigation_polys = p_query_task.path_query_slot->path_corridor;

	_query_task_push_back_point_with_metadata(p_query_task, end_point, end_poly);

	// Add mid points
	int np_id = least_cost_id;
	while (np_id != -1 && navigation_polys[np_id].back_navigation_poly_id != -1) {
		_query_task_push_back_point_with_metadata(p_query_task, navigation_polys[np_id].entry, navigation_polys[np_id].poly);

		np_id = navigation_polys[np_id].back_navigation_poly_id;
	}

	_query_task_push_back_point_with_metadata(p_query_task, begin_point, begin_poly);
}

Vector3 NavMeshQueries3D::map_iteration_get_closest_point_to_segment(const NavMapIteration3D &p_map_iteration, const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision) {
	bool use_collision = p_use_collision;
	Vector3 closest_point;
	real_t closest_point_distance_squared = FLT_MAX;

	const LocalVector<Ref<NavRegionIteration3D>> &regions = p_map_iteration.region_iterations;
	for (const Ref<NavRegionIteration3D> &region : regions) {
		for (const Polygon &polygon : region->get_navmesh_polygons()) {
			// For each face check the distance to the segment.
			for (uint32_t point_id = 2; point_id < polygon.vertices.size(); point_id += 1) {
				const Face3 face(polygon.vertices[0], polygon.vertices[point_id - 1], polygon.vertices[point_id]);
				Vector3 intersection_point;
				if (face.intersects_segment(p_from, p_to, &intersection_point)) {
					const real_t d_squared = p_from.distance_squared_to(intersection_point);
					if (!use_collision) {
						closest_point = intersection_point;
						use_collision = true;
						closest_point_distance_squared = d_squared;
					} else if (closest_point_distance_squared > d_squared) {
						closest_point = intersection_point;
						closest_point_distance_squared = d_squared;
					}
				}
				// If segment does not itersect face, check the distance from segment's endpoints.
				else if (!use_collision) {
					const Vector3 p_from_closest = face.get_closest_point_to(p_from);
					const real_t d_p_from_squared = p_from.distance_squared_to(p_from_closest);
					if (closest_point_distance_squared > d_p_from_squared) {
						closest_point = p_from_closest;
						closest_point_distance_squared = d_p_from_squared;
					}

					const Vector3 p_to_closest = face.get_closest_point_to(p_to);
					const real_t d_p_to_squared = p_to.distance_squared_to(p_to_closest);
					if (closest_point_distance_squared > d_p_to_squared) {
						closest_point = p_to_closest;
						closest_point_distance_squared = d_p_to_squared;
					}
				}
			}
			// Finally, check for a case when shortest distance is between some point located on a face's edge and some point located on a line segment.
			if (!use_collision) {
				for (uint32_t point_id = 0; point_id < polygon.vertices.size(); point_id += 1) {
					Vector3 a, b;

					Geometry3D::get_closest_points_between_segments(
							p_from,
							p_to,
							polygon.vertices[point_id],
							polygon.vertices[(point_id + 1) % polygon.vertices.size()],
							a,
							b);

					const real_t d_squared = a.distance_squared_to(b);
					if (d_squared < closest_point_distance_squared) {
						closest_point_distance_squared = d_squared;
						closest_point = b;
					}
				}
			}
		}
	}

	return closest_point;
}

Vector3 NavMeshQueries3D::map_iteration_get_closest_point(const NavMapIteration3D &p_map_iteration, const Vector3 &p_point) {
	ClosestPointQueryResult cp = map_iteration_get_closest_point_info(p_map_iteration, p_point);
	return cp.point;
}

Vector3 NavMeshQueries3D::map_iteration_get_closest_point_normal(const NavMapIteration3D &p_map_iteration, const Vector3 &p_point) {
	ClosestPointQueryResult cp = map_iteration_get_closest_point_info(p_map_iteration, p_point);
	return cp.normal;
}

RID NavMeshQueries3D::map_iteration_get_closest_point_owner(const NavMapIteration3D &p_map_iteration, const Vector3 &p_point) {
	ClosestPointQueryResult cp = map_iteration_get_closest_point_info(p_map_iteration, p_point);
	return cp.owner;
}

ClosestPointQueryResult NavMeshQueries3D::map_iteration_get_closest_point_info(const NavMapIteration3D &p_map_iteration, const Vector3 &p_point) {
	ClosestPointQueryResult result;
	real_t closest_point_distance_squared = FLT_MAX;

	const LocalVector<Ref<NavRegionIteration3D>> &regions = p_map_iteration.region_iterations;
	for (const Ref<NavRegionIteration3D> &region : regions) {
		for (const Polygon &polygon : region->get_navmesh_polygons()) {
			Vector3 plane_normal = (polygon.vertices[1] - polygon.vertices[0]).cross(polygon.vertices[2] - polygon.vertices[0]);
			Vector3 closest_on_polygon;
			real_t closest = FLT_MAX;
			bool inside = true;
			Vector3 previous = polygon.vertices[polygon.vertices.size() - 1];
			for (uint32_t point_id = 0; point_id < polygon.vertices.size(); ++point_id) {
				Vector3 edge = polygon.vertices[point_id] - previous;
				Vector3 to_point = p_point - previous;
				Vector3 edge_to_point_pormal = edge.cross(to_point);
				bool clockwise = edge_to_point_pormal.dot(plane_normal) > 0;
				// If we are not clockwise, the point will never be inside the polygon and so the closest point will be on an edge.
				if (!clockwise) {
					inside = false;
					real_t point_projected_on_edge = edge.dot(to_point);
					real_t edge_square = edge.length_squared();

					if (point_projected_on_edge > edge_square) {
						real_t distance = polygon.vertices[point_id].distance_squared_to(p_point);
						if (distance < closest) {
							closest_on_polygon = polygon.vertices[point_id];
							closest = distance;
						}
					} else if (point_projected_on_edge < 0.f) {
						real_t distance = previous.distance_squared_to(p_point);
						if (distance < closest) {
							closest_on_polygon = previous;
							closest = distance;
						}
					} else {
						// If we project on this edge, this will be the closest point.
						real_t percent = point_projected_on_edge / edge_square;
						closest_on_polygon = previous + percent * edge;
						break;
					}
				}
				previous = polygon.vertices[point_id];
			}

			if (inside) {
				Vector3 plane_normalized = plane_normal.normalized();
				real_t distance = plane_normalized.dot(p_point - polygon.vertices[0]);
				real_t distance_squared = distance * distance;
				if (distance_squared < closest_point_distance_squared) {
					closest_point_distance_squared = distance_squared;
					result.point = p_point - plane_normalized * distance;
					result.normal = plane_normal;
					result.owner = polygon.owner->get_self();

					if (Math::is_zero_approx(distance)) {
						break;
					}
				}
			} else {
				real_t distance = closest_on_polygon.distance_squared_to(p_point);
				if (distance < closest_point_distance_squared) {
					closest_point_distance_squared = distance;
					result.point = closest_on_polygon;
					result.normal = plane_normal;
					result.owner = polygon.owner->get_self();
				}
			}
		}
	}

	return result;
}

Vector3 NavMeshQueries3D::map_iteration_get_random_point(const NavMapIteration3D &p_map_iteration, uint32_t p_navigation_layers, bool p_uniformly) {
	if (p_map_iteration.region_iterations.is_empty()) {
		return Vector3();
	}

	LocalVector<uint32_t> accessible_regions;
	accessible_regions.reserve(p_map_iteration.region_iterations.size());

	for (uint32_t i = 0; i < p_map_iteration.region_iterations.size(); i++) {
		const Ref<NavRegionIteration3D> &region = p_map_iteration.region_iterations[i];
		if (!region->get_enabled() || (p_navigation_layers & region->get_navigation_layers()) == 0) {
			continue;
		}
		accessible_regions.push_back(i);
	}

	if (accessible_regions.is_empty()) {
		// All existing region polygons are disabled.
		return Vector3();
	}

	if (p_uniformly) {
		real_t accumulated_region_surface_area = 0;
		RBMap<real_t, uint32_t> accessible_regions_area_map;

		for (uint32_t accessible_region_index = 0; accessible_region_index < accessible_regions.size(); accessible_region_index++) {
			const Ref<NavRegionIteration3D> &region = p_map_iteration.region_iterations[accessible_regions[accessible_region_index]];

			real_t region_surface_area = region->surface_area;

			if (region_surface_area == 0.0f) {
				continue;
			}

			accessible_regions_area_map[accumulated_region_surface_area] = accessible_region_index;
			accumulated_region_surface_area += region_surface_area;
		}
		if (accessible_regions_area_map.is_empty() || accumulated_region_surface_area == 0) {
			// All faces have no real surface / no area.
			return Vector3();
		}

		real_t random_accessible_regions_area_map = Math::random(real_t(0), accumulated_region_surface_area);

		RBMap<real_t, uint32_t>::Iterator E = accessible_regions_area_map.find_closest(random_accessible_regions_area_map);
		ERR_FAIL_COND_V(!E, Vector3());
		uint32_t random_region_index = E->value;
		ERR_FAIL_UNSIGNED_INDEX_V(random_region_index, accessible_regions.size(), Vector3());

		const Ref<NavRegionIteration3D> &random_region = p_map_iteration.region_iterations[accessible_regions[random_region_index]];

		return NavMeshQueries3D::polygons_get_random_point(random_region->navmesh_polygons, p_navigation_layers, p_uniformly);

	} else {
		uint32_t random_region_index = Math::random(int(0), accessible_regions.size() - 1);

		const Ref<NavRegionIteration3D> &random_region = p_map_iteration.region_iterations[accessible_regions[random_region_index]];

		return NavMeshQueries3D::polygons_get_random_point(random_region->navmesh_polygons, p_navigation_layers, p_uniformly);
	}
}

Vector3 NavMeshQueries3D::polygons_get_closest_point_to_segment(const LocalVector<Polygon> &p_polygons, const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision) {
	bool use_collision = p_use_collision;
	Vector3 closest_point;
	real_t closest_point_distance_squared = FLT_MAX;

	for (const Polygon &polygon : p_polygons) {
		// For each face check the distance to the segment.
		for (uint32_t point_id = 2; point_id < polygon.vertices.size(); point_id += 1) {
			const Face3 face(polygon.vertices[0], polygon.vertices[point_id - 1], polygon.vertices[point_id]);
			Vector3 intersection_point;
			if (face.intersects_segment(p_from, p_to, &intersection_point)) {
				const real_t d_squared = p_from.distance_squared_to(intersection_point);
				if (!use_collision) {
					closest_point = intersection_point;
					use_collision = true;
					closest_point_distance_squared = d_squared;
				} else if (closest_point_distance_squared > d_squared) {
					closest_point = intersection_point;
					closest_point_distance_squared = d_squared;
				}
			}
			// If segment does not itersect face, check the distance from segment's endpoints.
			else if (!use_collision) {
				const Vector3 p_from_closest = face.get_closest_point_to(p_from);
				const real_t d_p_from_squared = p_from.distance_squared_to(p_from_closest);
				if (closest_point_distance_squared > d_p_from_squared) {
					closest_point = p_from_closest;
					closest_point_distance_squared = d_p_from_squared;
				}

				const Vector3 p_to_closest = face.get_closest_point_to(p_to);
				const real_t d_p_to_squared = p_to.distance_squared_to(p_to_closest);
				if (closest_point_distance_squared > d_p_to_squared) {
					closest_point = p_to_closest;
					closest_point_distance_squared = d_p_to_squared;
				}
			}
		}
		// Finally, check for a case when shortest distance is between some point located on a face's edge and some point located on a line segment.
		if (!use_collision) {
			for (uint32_t point_id = 0; point_id < polygon.vertices.size(); point_id += 1) {
				Vector3 a, b;

				Geometry3D::get_closest_points_between_segments(
						p_from,
						p_to,
						polygon.vertices[point_id],
						polygon.vertices[(point_id + 1) % polygon.vertices.size()],
						a,
						b);

				const real_t d_squared = a.distance_squared_to(b);
				if (d_squared < closest_point_distance_squared) {
					closest_point_distance_squared = d_squared;
					closest_point = b;
				}
			}
		}
	}

	return closest_point;
}

Vector3 NavMeshQueries3D::polygons_get_closest_point(const LocalVector<Polygon> &p_polygons, const Vector3 &p_point) {
	ClosestPointQueryResult cp = polygons_get_closest_point_info(p_polygons, p_point);
	return cp.point;
}

Vector3 NavMeshQueries3D::polygons_get_closest_point_normal(const LocalVector<Polygon> &p_polygons, const Vector3 &p_point) {
	ClosestPointQueryResult cp = polygons_get_closest_point_info(p_polygons, p_point);
	return cp.normal;
}

ClosestPointQueryResult NavMeshQueries3D::polygons_get_closest_point_info(const LocalVector<Polygon> &p_polygons, const Vector3 &p_point) {
	ClosestPointQueryResult result;
	real_t closest_point_distance_squared = FLT_MAX;

	for (const Polygon &polygon : p_polygons) {
		Vector3 plane_normal = (polygon.vertices[1] - polygon.vertices[0]).cross(polygon.vertices[2] - polygon.vertices[0]);
		Vector3 closest_on_polygon;
		real_t closest = FLT_MAX;
		bool inside = true;
		Vector3 previous = polygon.vertices[polygon.vertices.size() - 1];
		for (uint32_t point_id = 0; point_id < polygon.vertices.size(); ++point_id) {
			Vector3 edge = polygon.vertices[point_id] - previous;
			Vector3 to_point = p_point - previous;
			Vector3 edge_to_point_pormal = edge.cross(to_point);
			bool clockwise = edge_to_point_pormal.dot(plane_normal) > 0;
			// If we are not clockwise, the point will never be inside the polygon and so the closest point will be on an edge.
			if (!clockwise) {
				inside = false;
				real_t point_projected_on_edge = edge.dot(to_point);
				real_t edge_square = edge.length_squared();

				if (point_projected_on_edge > edge_square) {
					real_t distance = polygon.vertices[point_id].distance_squared_to(p_point);
					if (distance < closest) {
						closest_on_polygon = polygon.vertices[point_id];
						closest = distance;
					}
				} else if (point_projected_on_edge < 0.f) {
					real_t distance = previous.distance_squared_to(p_point);
					if (distance < closest) {
						closest_on_polygon = previous;
						closest = distance;
					}
				} else {
					// If we project on this edge, this will be the closest point.
					real_t percent = point_projected_on_edge / edge_square;
					closest_on_polygon = previous + percent * edge;
					break;
				}
			}
			previous = polygon.vertices[point_id];
		}

		if (inside) {
			Vector3 plane_normalized = plane_normal.normalized();
			real_t distance = plane_normalized.dot(p_point - polygon.vertices[0]);
			real_t distance_squared = distance * distance;
			if (distance_squared < closest_point_distance_squared) {
				closest_point_distance_squared = distance_squared;
				result.point = p_point - plane_normalized * distance;
				result.normal = plane_normal;
				result.owner = polygon.owner->get_self();

				if (Math::is_zero_approx(distance)) {
					break;
				}
			}
		} else {
			real_t distance = closest_on_polygon.distance_squared_to(p_point);
			if (distance < closest_point_distance_squared) {
				closest_point_distance_squared = distance;
				result.point = closest_on_polygon;
				result.normal = plane_normal;
				result.owner = polygon.owner->get_self();
			}
		}
	}

	return result;
}

RID NavMeshQueries3D::polygons_get_closest_point_owner(const LocalVector<Polygon> &p_polygons, const Vector3 &p_point) {
	ClosestPointQueryResult cp = polygons_get_closest_point_info(p_polygons, p_point);
	return cp.owner;
}

void NavMeshQueries3D::_query_task_clip_path(NavMeshPathQueryTask3D &p_query_task, const NavigationPoly *from_poly, const Vector3 &p_to_point, const NavigationPoly *p_to_poly) {
	Vector3 from = p_query_task.path_points[p_query_task.path_points.size() - 1];
	const LocalVector<NavigationPoly> &p_navigation_polys = p_query_task.path_query_slot->path_corridor;
	const Vector3 p_map_up = p_query_task.map_up;

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
				if (!inters.is_equal_approx(p_to_point) && !inters.is_equal_approx(p_query_task.path_points[p_query_task.path_points.size() - 1])) {
					_query_task_push_back_point_with_metadata(p_query_task, inters, from_poly->poly);
				}
			}
		}
	}
}

bool NavMeshQueries3D::_query_task_is_connection_owner_usable(const NavMeshPathQueryTask3D &p_query_task, const NavBaseIteration3D *p_owner) {
	ERR_FAIL_NULL_V(p_owner, false);

	bool owner_usable = true;

	if (!p_owner->get_enabled()) {
		owner_usable = false;
		return owner_usable;
	}

	if ((p_query_task.navigation_layers & p_owner->get_navigation_layers()) == 0) {
		// Not usable. No matching bit between task filter bitmask and owner bitmask.
		owner_usable = false;
		return owner_usable;
	}

	if (p_query_task.exclude_regions || p_query_task.include_regions) {
		switch (p_owner->get_type()) {
			case NavigationEnums3D::PathSegmentType::PATH_SEGMENT_TYPE_REGION: {
				if (p_query_task.exclude_regions && p_query_task.excluded_regions.has(p_owner->get_self())) {
					// Not usable. Exclude region filter is active and this region is excluded.
					owner_usable = false;
				} else if (p_query_task.include_regions && !p_query_task.included_regions.has(p_owner->get_self())) {
					// Not usable. Include region filter is active and this region is not included.
					owner_usable = false;
				}
			} break;
			case NavigationEnums3D::PathSegmentType::PATH_SEGMENT_TYPE_LINK: {
				const LocalVector<Polygon> &link_polygons = p_owner->get_navmesh_polygons();
				if (link_polygons.size() != 2) {
					// Not usable. Whatever this is, it is not a valid connected link.
					owner_usable = false;
				} else {
					const RID link_start_region = link_polygons[0].owner->get_self();
					const RID link_end_region = link_polygons[1].owner->get_self();
					if (p_query_task.exclude_regions && (p_query_task.excluded_regions.has(link_start_region) || p_query_task.excluded_regions.has(link_end_region))) {
						// Not usable. Exclude region filter is active and at least one region of the link is excluded.
						owner_usable = false;
					}
					if (p_query_task.include_regions && (!p_query_task.included_regions.has(link_start_region) || !p_query_task.excluded_regions.has(link_end_region))) {
						// Not usable. Include region filter is active and not both regions of the links are included.
						owner_usable = false;
					}
				}
			} break;
		}
	}

	return owner_usable;
}

LocalVector<uint32_t> NavMeshQueries3D::get_simplified_path_indices(const LocalVector<Vector3> &p_path, real_t p_epsilon) {
	p_epsilon = MAX(0.0, p_epsilon);
	real_t squared_epsilon = p_epsilon * p_epsilon;

	LocalVector<uint32_t> simplified_path_indices;
	simplified_path_indices.reserve(p_path.size());
	simplified_path_indices.push_back(0);
	simplify_path_segment(0, p_path.size() - 1, p_path, squared_epsilon, simplified_path_indices);
	simplified_path_indices.push_back(p_path.size() - 1);

	return simplified_path_indices;
}

void NavMeshQueries3D::simplify_path_segment(int p_start_inx, int p_end_inx, const LocalVector<Vector3> &p_points, real_t p_epsilon, LocalVector<uint32_t> &r_simplified_path_indices) {
	const Vector3 path_segment_a = p_points[p_start_inx];
	const Vector3 path_segment_b = p_points[p_end_inx];

	real_t point_max_distance = 0.0;
	int point_max_index = 0;

	for (int i = p_start_inx; i < p_end_inx; i++) {
		const Vector3 &checked_point = p_points[i];

		const Vector3 closest_point = Geometry3D::get_closest_point_to_segment(checked_point, path_segment_a, path_segment_b);
		real_t distance_squared = closest_point.distance_squared_to(checked_point);

		if (distance_squared > point_max_distance) {
			point_max_index = i;
			point_max_distance = distance_squared;
		}
	}

	if (point_max_distance > p_epsilon) {
		simplify_path_segment(p_start_inx, point_max_index, p_points, p_epsilon, r_simplified_path_indices);
		r_simplified_path_indices.push_back(point_max_index);
		simplify_path_segment(point_max_index, p_end_inx, p_points, p_epsilon, r_simplified_path_indices);
	}
}
