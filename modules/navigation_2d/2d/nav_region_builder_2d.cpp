/**************************************************************************/
/*  nav_region_builder_2d.cpp                                             */
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

#include "nav_region_builder_2d.h"

#include "../nav_map_2d.h"
#include "../nav_region_2d.h"
#include "../triangle2.h"
#include "nav_region_iteration_2d.h"

#include "core/config/project_settings.h"

using namespace Nav2D;

void NavRegionBuilder2D::build_iteration(NavRegionIterationBuild2D &r_build) {
	PerformanceData &performance_data = r_build.performance_data;

	performance_data.pm_polygon_count = 0;
	performance_data.pm_edge_count = 0;
	performance_data.pm_edge_merge_count = 0;
	performance_data.pm_edge_connection_count = 0;
	performance_data.pm_edge_free_count = 0;

	_build_step_process_navmesh_data(r_build);

	_build_step_find_edge_connection_pairs(r_build);

	_build_step_merge_edge_connection_pairs(r_build);

	_build_update_iteration(r_build);
}

void NavRegionBuilder2D::_build_step_process_navmesh_data(NavRegionIterationBuild2D &r_build) {
	Vector<Vector2> _navmesh_vertices = r_build.navmesh_data.vertices;
	Vector<Vector<int>> _navmesh_polygons = r_build.navmesh_data.polygons;

	if (_navmesh_vertices.is_empty() || _navmesh_polygons.is_empty()) {
		return;
	}

	PerformanceData &performance_data = r_build.performance_data;
	Ref<NavRegionIteration2D> region_iteration = r_build.region_iteration;

	const Transform2D &region_transform = region_iteration->transform;
	LocalVector<Nav2D::Polygon> &navmesh_polygons = region_iteration->navmesh_polygons;

	const int vertex_count = _navmesh_vertices.size();

	const Vector2 *vertices_ptr = _navmesh_vertices.ptr();
	const Vector<int> *polygons_ptr = _navmesh_polygons.ptr();

	navmesh_polygons.resize(_navmesh_polygons.size());

	real_t _new_region_surface_area = 0.0;
	Rect2 _new_region_bounds;

	bool first_vertex = true;

	for (uint32_t i = 0; i < navmesh_polygons.size(); i++) {
		Polygon &polygon = navmesh_polygons[i];
		polygon.id = i;
		polygon.owner = region_iteration.ptr();
		polygon.surface_area = 0.0;

		Vector<int> polygon_indices = polygons_ptr[i];

		int polygon_size = polygon_indices.size();
		if (polygon_size < 3) {
			continue;
		}

		const int *indices_ptr = polygon_indices.ptr();

		bool polygon_valid = true;

		polygon.vertices.resize(polygon_size);

		{
			real_t _new_polygon_surface_area = 0.0;

			for (int j(2); j < polygon_size; j++) {
				const Triangle2 triangle = Triangle2(
						region_transform.xform(vertices_ptr[indices_ptr[0]]),
						region_transform.xform(vertices_ptr[indices_ptr[j - 1]]),
						region_transform.xform(vertices_ptr[indices_ptr[j]]));

				_new_polygon_surface_area += triangle.get_area();
			}

			polygon.surface_area = _new_polygon_surface_area;
			_new_region_surface_area += _new_polygon_surface_area;
		}

		for (int j(0); j < polygon_size; j++) {
			int vertex_index = indices_ptr[j];
			if (vertex_index < 0 || vertex_index >= vertex_count) {
				polygon_valid = false;
				break;
			}

			const Vector2 point_position = region_transform.xform(vertices_ptr[vertex_index]);
			polygon.vertices[j] = point_position;

			if (first_vertex) {
				first_vertex = false;
				_new_region_bounds.position = point_position;
			} else {
				_new_region_bounds.expand_to(point_position);
			}
		}

		if (!polygon_valid) {
			polygon.surface_area = 0.0;
			polygon.vertices.clear();
			ERR_FAIL_COND_MSG(!polygon_valid, "Corrupted navigation mesh set on region. The indices of a polygon are out of range.");
		}
	}

	region_iteration->surface_area = _new_region_surface_area;
	region_iteration->bounds = _new_region_bounds;

	performance_data.pm_polygon_count = navmesh_polygons.size();
}

Nav2D::PointKey NavRegionBuilder2D::get_point_key(const Vector2 &p_pos, const Vector2 &p_cell_size) {
	const int x = static_cast<int>(Math::floor(p_pos.x / p_cell_size.x));
	const int y = static_cast<int>(Math::floor(p_pos.y / p_cell_size.y));

	PointKey p;
	p.key = 0;
	p.x = x;
	p.y = y;
	return p;
}

Nav2D::EdgeKey NavRegionBuilder2D::get_edge_key(const Vector2 &p_vertex1, const Vector2 &p_vertex2, const Vector2 &p_cell_size) {
	EdgeKey ek(get_point_key(p_vertex1, p_cell_size), get_point_key(p_vertex2, p_cell_size));
	return ek;
}

void NavRegionBuilder2D::_build_step_find_edge_connection_pairs(NavRegionIterationBuild2D &r_build) {
	PerformanceData &performance_data = r_build.performance_data;

	const Vector2 &map_cell_size = r_build.map_cell_size;
	Ref<NavRegionIteration2D> region_iteration = r_build.region_iteration;
	LocalVector<Nav2D::Polygon> &navmesh_polygons = region_iteration->navmesh_polygons;

	HashMap<EdgeKey, EdgeConnectionPair, EdgeKey> &connection_pairs_map = r_build.iter_connection_pairs_map;
	connection_pairs_map.clear();

	region_iteration->internal_connections.clear();
	region_iteration->internal_connections.resize(navmesh_polygons.size());

	region_iteration->external_edges.clear();

	int free_edges_count = 0;
	int edge_merge_error_count = 0;

	for (Polygon &poly : region_iteration->navmesh_polygons) {
		for (uint32_t p = 0; p < poly.vertices.size(); p++) {
			const int next_point = (p + 1) % poly.vertices.size();
			const EdgeKey ek = get_edge_key(poly.vertices[p], poly.vertices[next_point], map_cell_size);

			HashMap<EdgeKey, EdgeConnectionPair, EdgeKey>::Iterator pair_it = connection_pairs_map.find(ek);
			if (!pair_it) {
				pair_it = connection_pairs_map.insert(ek, EdgeConnectionPair());
				performance_data.pm_edge_count += 1;
				++free_edges_count;
			}
			EdgeConnectionPair &pair = pair_it->value;
			if (pair.size < 2) {
				// Add the polygon/edge tuple to this key.
				Connection new_connection;
				new_connection.polygon = &poly;
				new_connection.edge = p;
				new_connection.pathway_start = poly.vertices[p];
				new_connection.pathway_end = poly.vertices[next_point];

				pair.connections[pair.size] = new_connection;
				++pair.size;
				if (pair.size == 2) {
					--free_edges_count;
				}

			} else {
				// The edge is already connected with another edge, skip.
				edge_merge_error_count++;
			}
		}
	}

	if (edge_merge_error_count > 0 && GLOBAL_GET_CACHED(bool, "navigation/2d/warnings/navmesh_edge_merge_errors")) {
		WARN_PRINT("Navigation region synchronization had " + itos(edge_merge_error_count) + " edge error(s).\nMore than 2 edges tried to occupy the same map rasterization space.\nThis causes a logical error in the navigation mesh geometry and is commonly caused by overlap or too densely placed edges.\nConsider baking with a higher 'cell_size', greater geometry margin, and less detailed bake objects to cause fewer edges.\nConsider lowering the 'navigation/2d/merge_rasterizer_cell_scale' in the project settings.\nThis warning can be toggled under 'navigation/2d/warnings/navmesh_edge_merge_errors' in the project settings.");
	}

	performance_data.pm_edge_free_count = free_edges_count;
}

void NavRegionBuilder2D::_build_step_merge_edge_connection_pairs(NavRegionIterationBuild2D &r_build) {
	PerformanceData &performance_data = r_build.performance_data;

	Ref<NavRegionIteration2D> region_iteration = r_build.region_iteration;

	HashMap<EdgeKey, EdgeConnectionPair, EdgeKey> &connection_pairs_map = r_build.iter_connection_pairs_map;

	for (const KeyValue<EdgeKey, EdgeConnectionPair> &pair_it : connection_pairs_map) {
		const EdgeConnectionPair &pair = pair_it.value;
		if (pair.size == 2) {
			// Connect edge that are shared in different polygons.
			const Connection &c1 = pair.connections[0];
			const Connection &c2 = pair.connections[1];
			region_iteration->internal_connections[c1.polygon->id].push_back(c2);
			region_iteration->internal_connections[c2.polygon->id].push_back(c1);
			performance_data.pm_edge_merge_count += 1;

		} else {
			ERR_FAIL_COND_MSG(pair.size != 1, vformat("Number of connection != 1. Found: %d", pair.size));

			const Connection &connection = pair.connections[0];

			ConnectableEdge ce;
			ce.ek = pair_it.key;
			ce.polygon_index = connection.polygon->id;
			ce.edge = connection.edge;
			ce.pathway_start = connection.pathway_start;
			ce.pathway_end = connection.pathway_end;

			region_iteration->external_edges.push_back(ce);
		}
	}
}

void NavRegionBuilder2D::_build_update_iteration(NavRegionIterationBuild2D &r_build) {
	ERR_FAIL_NULL(r_build.region);
	// Stub. End of the build.
}
