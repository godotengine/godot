/**************************************************************************/
/*  nav_region_2d.cpp                                                     */
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

#include "nav_region_2d.h"

#include "nav_map_2d.h"
#include "triangle2.h"

#include "2d/nav_map_builder_2d.h"
#include "2d/nav_mesh_queries_2d.h"
#include "2d/nav_region_iteration_2d.h"

using namespace nav_2d;

void NavRegion2D::set_map(NavMap2D *p_map) {
	if (map == p_map) {
		return;
	}

	cancel_sync_request();

	if (map) {
		map->remove_region(this);
	}

	map = p_map;
	polygons_dirty = true;

	if (map) {
		map->add_region(this);
		request_sync();
	}
}

void NavRegion2D::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}
	enabled = p_enabled;

	// TODO: This should not require a full rebuild as the region has not really changed.
	polygons_dirty = true;

	request_sync();
}

void NavRegion2D::set_use_edge_connections(bool p_enabled) {
	if (use_edge_connections != p_enabled) {
		use_edge_connections = p_enabled;
		polygons_dirty = true;
	}

	request_sync();
}

void NavRegion2D::set_transform(const Transform2D &p_transform) {
	if (transform == p_transform) {
		return;
	}
	transform = p_transform;
	polygons_dirty = true;

	request_sync();
}

void NavRegion2D::set_navigation_polygon(Ref<NavigationPolygon> p_navigation_polygon) {
#ifdef DEBUG_ENABLED
	if (map && p_navigation_polygon.is_valid() && !Math::is_equal_approx(double(map->get_cell_size()), double(p_navigation_polygon->get_cell_size()))) {
		ERR_PRINT_ONCE(vformat("Attempted to update a navigation region with a navigation mesh that uses a `cell_size` of %s while assigned to a navigation map set to a `cell_size` of %s. The cell size for navigation maps can be changed by using the NavigationServer map_set_cell_size() function. The cell size for default navigation maps can also be changed in the ProjectSettings.", double(p_navigation_polygon->get_cell_size()), double(map->get_cell_size())));
	}
#endif // DEBUG_ENABLED

	RWLockWrite write_lock(navmesh_rwlock);

	pending_navmesh_vertices.clear();
	pending_navmesh_polygons.clear();

	if (p_navigation_polygon.is_valid()) {
		p_navigation_polygon->get_data(pending_navmesh_vertices, pending_navmesh_polygons);
	}

	polygons_dirty = true;

	request_sync();
}

ClosestPointQueryResult NavRegion2D::get_closest_point_info(const Vector2 &p_point) const {
	RWLockRead read_lock(region_rwlock);

	return NavMeshQueries2D::polygons_get_closest_point_info(get_polygons(), p_point);
}

Vector2 NavRegion2D::get_random_point(uint32_t p_navigation_layers, bool p_uniformly) const {
	RWLockRead read_lock(region_rwlock);

	if (!get_enabled()) {
		return Vector2();
	}

	return NavMeshQueries2D::polygons_get_random_point(get_polygons(), p_navigation_layers, p_uniformly);
}

void NavRegion2D::set_navigation_layers(uint32_t p_navigation_layers) {
	if (navigation_layers == p_navigation_layers) {
		return;
	}
	navigation_layers = p_navigation_layers;
	region_dirty = true;

	request_sync();
}

void NavRegion2D::set_enter_cost(real_t p_enter_cost) {
	real_t new_enter_cost = MAX(p_enter_cost, 0.0);
	if (enter_cost == new_enter_cost) {
		return;
	}
	enter_cost = new_enter_cost;
	region_dirty = true;

	request_sync();
}

void NavRegion2D::set_travel_cost(real_t p_travel_cost) {
	real_t new_travel_cost = MAX(p_travel_cost, 0.0);
	if (travel_cost == new_travel_cost) {
		return;
	}
	travel_cost = new_travel_cost;
	region_dirty = true;

	request_sync();
}

void NavRegion2D::set_owner_id(ObjectID p_owner_id) {
	if (owner_id == p_owner_id) {
		return;
	}
	owner_id = p_owner_id;
	region_dirty = true;

	request_sync();
}

bool NavRegion2D::sync() {
	RWLockWrite write_lock(region_rwlock);

	bool something_changed = region_dirty || polygons_dirty;

	region_dirty = false;

	update_polygons();

	return something_changed;
}

void NavRegion2D::update_polygons() {
	if (!polygons_dirty) {
		return;
	}
	navmesh_polygons.clear();
	surface_area = 0.0;
	bounds = Rect2();
	polygons_dirty = false;

	if (map == nullptr) {
		return;
	}

	RWLockRead read_lock(navmesh_rwlock);

	if (pending_navmesh_vertices.is_empty() || pending_navmesh_polygons.is_empty()) {
		return;
	}

	int len = pending_navmesh_vertices.size();
	if (len == 0) {
		return;
	}

	const Vector2 *vertices_r = pending_navmesh_vertices.ptr();

	navmesh_polygons.resize(pending_navmesh_polygons.size());

	real_t _new_region_surface_area = 0.0;
	Rect2 _new_bounds;

	bool first_vertex = true;
	int navigation_mesh_polygon_index = 0;

	for (Polygon &polygon : navmesh_polygons) {
		polygon.surface_area = 0.0;

		Vector<int> navigation_mesh_polygon = pending_navmesh_polygons[navigation_mesh_polygon_index];
		navigation_mesh_polygon_index += 1;

		int navigation_mesh_polygon_size = navigation_mesh_polygon.size();
		if (navigation_mesh_polygon_size < 3) {
			continue;
		}

		const int *indices = navigation_mesh_polygon.ptr();
		bool valid(true);

		polygon.points.resize(navigation_mesh_polygon_size);
		polygon.edges.resize(navigation_mesh_polygon_size);

		real_t _new_polygon_surface_area = 0.0;

		for (int j(2); j < navigation_mesh_polygon_size; j++) {
			const Triangle2 triangle = Triangle2(
					transform.xform(vertices_r[indices[0]]),
					transform.xform(vertices_r[indices[j - 1]]),
					transform.xform(vertices_r[indices[j]]));

			_new_polygon_surface_area += triangle.get_area();
		}

		polygon.surface_area = _new_polygon_surface_area;
		_new_region_surface_area += _new_polygon_surface_area;

		for (int j(0); j < navigation_mesh_polygon_size; j++) {
			int idx = indices[j];
			if (idx < 0 || idx >= len) {
				valid = false;
				break;
			}

			Vector2 point_position = transform.xform(vertices_r[idx]);
			polygon.points[j].pos = point_position;
			polygon.points[j].key = NavMapBuilder2D::get_point_key(point_position, map->get_merge_rasterizer_cell_size());

			if (first_vertex) {
				first_vertex = false;
				_new_bounds.position = point_position;
			} else {
				_new_bounds.expand_to(point_position);
			}
		}

		if (!valid) {
			ERR_BREAK_MSG(!valid, "The navigation polygon set in this region is not valid!");
		}
	}

	surface_area = _new_region_surface_area;
	bounds = _new_bounds;
}

void NavRegion2D::get_iteration_update(NavRegionIteration2D &r_iteration) {
	r_iteration.navigation_layers = get_navigation_layers();
	r_iteration.enter_cost = get_enter_cost();
	r_iteration.travel_cost = get_travel_cost();
	r_iteration.owner_object_id = get_owner_id();
	r_iteration.owner_type = get_type();
	r_iteration.owner_rid = get_self();

	r_iteration.enabled = get_enabled();
	r_iteration.transform = get_transform();
	r_iteration.owner_use_edge_connections = get_use_edge_connections();
	r_iteration.bounds = get_bounds();
	r_iteration.surface_area = get_surface_area();

	r_iteration.navmesh_polygons.clear();
	r_iteration.navmesh_polygons.resize(navmesh_polygons.size());
	for (uint32_t i = 0; i < navmesh_polygons.size(); i++) {
		Polygon &navmesh_polygon = navmesh_polygons[i];
		navmesh_polygon.owner = &r_iteration;
		r_iteration.navmesh_polygons[i] = navmesh_polygon;
	}
}

void NavRegion2D::request_sync() {
	if (map && !sync_dirty_request_list_element.in_list()) {
		map->add_region_sync_dirty_request(&sync_dirty_request_list_element);
	}
}

void NavRegion2D::cancel_sync_request() {
	if (map && sync_dirty_request_list_element.in_list()) {
		map->remove_region_sync_dirty_request(&sync_dirty_request_list_element);
	}
}

NavRegion2D::NavRegion2D() :
		sync_dirty_request_list_element(this) {
	type = NavigationUtilities::PathSegmentType::PATH_SEGMENT_TYPE_REGION;
}

NavRegion2D::~NavRegion2D() {
	cancel_sync_request();
}
