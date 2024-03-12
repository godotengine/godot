/**************************************************************************/
/*  nav_region_3d.cpp                                                     */
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

#include "nav_region_3d.h"

#include "nav_map_3d.h"

#include "3d/nav_mesh_queries_3d.h"
#include "3d/nav_region_builder_3d.h"
#include "3d/nav_region_iteration_3d.h"
#include "core/config/project_settings.h"

using namespace Nav3D;

void NavRegion3D::set_map(NavMap3D *p_map) {
	if (map == p_map) {
		return;
	}

	cancel_async_thread_join();
	cancel_sync_request();

	if (map) {
		map->remove_region(this);
	}

	map = p_map;
	iteration_dirty = true;

	if (map) {
		map->add_region(this);
		request_sync();
		if (iteration_build_thread_task_id != WorkerThreadPool::INVALID_TASK_ID) {
			request_async_thread_join();
		}
	}
}

void NavRegion3D::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}
	enabled = p_enabled;
	iteration_dirty = true;

	request_sync();
}

void NavRegion3D::set_use_edge_connections(bool p_enabled) {
	if (use_edge_connections != p_enabled) {
		use_edge_connections = p_enabled;
		iteration_dirty = true;
	}

	request_sync();
}

void NavRegion3D::set_transform(Transform3D p_transform) {
	if (transform == p_transform) {
		return;
	}
	transform = p_transform;
	iteration_dirty = true;

	request_sync();

#ifdef DEBUG_ENABLED
	if (map && Math::rad_to_deg(map->get_up().angle_to(transform.basis.get_column(1))) >= 90.0f) {
		ERR_PRINT_ONCE("Attempted to update a navigation region transform rotated 90 degrees or more away from the current navigation map UP orientation.");
	}
#endif // DEBUG_ENABLED
}

void NavRegion3D::set_navigation_mesh(Ref<NavigationMesh> p_navigation_mesh) {
#ifdef DEBUG_ENABLED
	if (map && p_navigation_mesh.is_valid() && GLOBAL_GET_CACHED(bool, "navigation/3d/warnings/navmesh_cell_size_mismatch")) {
		const double map_cell_size = double(map->get_cell_size());
		const double map_cell_height = double(map->get_cell_height());
		const double navmesh_cell_size = double(p_navigation_mesh->get_cell_size());
		const double navmesh_cell_height = double(p_navigation_mesh->get_cell_height());

		if (map_cell_size > navmesh_cell_size) {
			WARN_PRINT(vformat("A navigation mesh that uses a `cell_size` of %s was assigned to a navigation map set to a larger `cell_size` of %s.\nThis mismatch in cell size can cause rasterization errors with navigation mesh edges on the navigation map.\nThe cell size for navigation maps can be changed by using the NavigationServer map_set_cell_size() function.\nThe cell size for default navigation maps can also be changed in the project settings.\nThis warning can be toggled under 'navigation/3d/warnings/navmesh_cell_size_mismatch' in the project settings.", navmesh_cell_size, map_cell_size));
		}
		if (map_cell_height > navmesh_cell_height) {
			WARN_PRINT(vformat("A navigation mesh that uses a `cell_height` of %s was assigned to a navigation map set to a larger `cell_height` of %s.\nThis mismatch in cell height can cause rasterization errors with navigation mesh edges on the navigation map.\nThe cell height for navigation maps can be changed by using the NavigationServer map_set_cell_height() function.\nThe cell height for default navigation maps can also be changed in the project settings.\nThis warning can be toggled under 'navigation/3d/warnings/navmesh_cell_size_mismatch' in the project settings.", navmesh_cell_height, map_cell_height));
		}
	}
#endif // DEBUG_ENABLED

	navmesh = p_navigation_mesh;

	iteration_dirty = true;

	request_sync();
}

Vector3 NavRegion3D::get_closest_point_to_segment(const Vector3 &p_from, const Vector3 &p_to, bool p_use_collision) const {
	RWLockRead read_lock(region_rwlock);

	return NavMeshQueries3D::polygons_get_closest_point_to_segment(
			get_polygons(), p_from, p_to, p_use_collision);
}

ClosestPointQueryResult NavRegion3D::get_closest_point_info(const Vector3 &p_point) const {
	RWLockRead read_lock(region_rwlock);

	return NavMeshQueries3D::polygons_get_closest_point_info(get_polygons(), p_point);
}

Vector3 NavRegion3D::get_random_point(uint32_t p_navigation_layers, bool p_uniformly) const {
	RWLockRead read_lock(region_rwlock);

	if (!get_enabled()) {
		return Vector3();
	}

	return NavMeshQueries3D::polygons_get_random_point(get_polygons(), p_navigation_layers, p_uniformly);
}

void NavRegion3D::set_navigation_layers(uint32_t p_navigation_layers) {
	if (navigation_layers == p_navigation_layers) {
		return;
	}
	navigation_layers = p_navigation_layers;
	iteration_dirty = true;

	request_sync();
}

void NavRegion3D::set_enter_cost(real_t p_enter_cost) {
	real_t new_enter_cost = MAX(p_enter_cost, 0.0);
	if (enter_cost == new_enter_cost) {
		return;
	}
	enter_cost = new_enter_cost;
	iteration_dirty = true;

	request_sync();
}

void NavRegion3D::set_travel_cost(real_t p_travel_cost) {
	real_t new_travel_cost = MAX(p_travel_cost, 0.0);
	if (travel_cost == new_travel_cost) {
		return;
	}
	travel_cost = new_travel_cost;
	iteration_dirty = true;

	request_sync();
}

void NavRegion3D::set_owner_id(ObjectID p_owner_id) {
	if (owner_id == p_owner_id) {
		return;
	}
	owner_id = p_owner_id;
	iteration_dirty = true;

	request_sync();
}

void NavRegion3D::scratch_polygons() {
	iteration_dirty = true;

	request_sync();
}

real_t NavRegion3D::get_surface_area() const {
	RWLockRead read_lock(iteration_rwlock);
	return iteration->get_surface_area();
}

AABB NavRegion3D::get_bounds() const {
	RWLockRead read_lock(iteration_rwlock);
	return iteration->get_bounds();
}

LocalVector<Nav3D::Polygon> const &NavRegion3D::get_polygons() const {
	RWLockRead read_lock(iteration_rwlock);
	return iteration->get_navmesh_polygons();
}

bool NavRegion3D::sync() {
	bool requires_map_update = false;
	if (!map) {
		return requires_map_update;
	}

	if (iteration_dirty && !iteration_building && !iteration_ready) {
		_build_iteration();
	}

	if (iteration_ready) {
		_sync_iteration();
		requires_map_update = true;
	}

	return requires_map_update;
}

void NavRegion3D::sync_async_tasks() {
	if (iteration_build_thread_task_id != WorkerThreadPool::INVALID_TASK_ID) {
		if (WorkerThreadPool::get_singleton()->is_task_completed(iteration_build_thread_task_id)) {
			WorkerThreadPool::get_singleton()->wait_for_task_completion(iteration_build_thread_task_id);

			iteration_build_thread_task_id = WorkerThreadPool::INVALID_TASK_ID;
			iteration_building = false;
			iteration_ready = true;
			request_sync();
		}
	}
}

void NavRegion3D::_build_iteration() {
	if (!iteration_dirty || iteration_building || iteration_ready) {
		return;
	}

	iteration_dirty = false;
	iteration_building = true;
	iteration_ready = false;

	iteration_build.reset();

	if (navmesh.is_valid()) {
		navmesh->get_data(iteration_build.navmesh_data.vertices, iteration_build.navmesh_data.polygons);
	}

	iteration_build.map_cell_size = map->get_merge_rasterizer_cell_size();

	Ref<NavRegionIteration3D> new_iteration;
	new_iteration.instantiate();

	new_iteration->navigation_layers = get_navigation_layers();
	new_iteration->enter_cost = get_enter_cost();
	new_iteration->travel_cost = get_travel_cost();
	new_iteration->owner_object_id = get_owner_id();
	new_iteration->owner_type = get_type();
	new_iteration->owner_rid = get_self();
	new_iteration->enabled = get_enabled();
	new_iteration->transform = get_transform();
	new_iteration->owner_use_edge_connections = get_use_edge_connections();

	iteration_build.region_iteration = new_iteration;

	if (use_async_iterations) {
		iteration_build_thread_task_id = WorkerThreadPool::get_singleton()->add_native_task(&NavRegion3D::_build_iteration_threaded, &iteration_build, true, SNAME("NavRegionBuilder3D"));
		request_async_thread_join();
	} else {
		NavRegionBuilder3D::build_iteration(iteration_build);

		iteration_building = false;
		iteration_ready = true;
	}
}

void NavRegion3D::_build_iteration_threaded(void *p_arg) {
	NavRegionIterationBuild3D *_iteration_build = static_cast<NavRegionIterationBuild3D *>(p_arg);

	NavRegionBuilder3D::build_iteration(*_iteration_build);
}

void NavRegion3D::_sync_iteration() {
	if (iteration_building || !iteration_ready) {
		return;
	}

	performance_data.pm_polygon_count = iteration_build.performance_data.pm_polygon_count;
	performance_data.pm_edge_count = iteration_build.performance_data.pm_edge_count;
	performance_data.pm_edge_merge_count = iteration_build.performance_data.pm_edge_merge_count;

	RWLockWrite write_lock(iteration_rwlock);
	ERR_FAIL_COND(iteration.is_null());
	iteration = Ref<NavRegionIteration3D>();
	DEV_ASSERT(iteration.is_null());
	iteration = iteration_build.region_iteration;
	iteration_build.region_iteration = Ref<NavRegionIteration3D>();
	DEV_ASSERT(iteration_build.region_iteration.is_null());
	iteration_id = iteration_id % UINT32_MAX + 1;

	iteration_ready = false;

	cancel_async_thread_join();
}

Ref<NavRegionIteration3D> NavRegion3D::get_iteration() {
	RWLockRead read_lock(iteration_rwlock);
	return iteration;
}

void NavRegion3D::request_async_thread_join() {
	DEV_ASSERT(map);
	if (map && !async_list_element.in_list()) {
		map->add_region_async_thread_join_request(&async_list_element);
	}
}

void NavRegion3D::cancel_async_thread_join() {
	if (map && async_list_element.in_list()) {
		map->remove_region_async_thread_join_request(&async_list_element);
	}
}

void NavRegion3D::request_sync() {
	if (map && !sync_dirty_request_list_element.in_list()) {
		map->add_region_sync_dirty_request(&sync_dirty_request_list_element);
	}
}

void NavRegion3D::cancel_sync_request() {
	if (map && sync_dirty_request_list_element.in_list()) {
		map->remove_region_sync_dirty_request(&sync_dirty_request_list_element);
	}
}

void NavRegion3D::set_use_async_iterations(bool p_enabled) {
	if (use_async_iterations == p_enabled) {
		return;
	}
#ifdef THREADS_ENABLED
	use_async_iterations = p_enabled;
#endif
}

bool NavRegion3D::get_use_async_iterations() const {
	return use_async_iterations;
}

NavRegion3D::NavRegion3D() :
		sync_dirty_request_list_element(this), async_list_element(this) {
	type = NavigationEnums3D::PathSegmentType::PATH_SEGMENT_TYPE_REGION;
	iteration_build.region = this;
	iteration.instantiate();

#ifdef THREADS_ENABLED
	use_async_iterations = GLOBAL_GET("navigation/world/region_use_async_iterations");
#else
	use_async_iterations = false;
#endif
}

NavRegion3D::~NavRegion3D() {
	cancel_async_thread_join();
	cancel_sync_request();

	if (iteration_build_thread_task_id != WorkerThreadPool::INVALID_TASK_ID) {
		WorkerThreadPool::get_singleton()->wait_for_task_completion(iteration_build_thread_task_id);
		iteration_build_thread_task_id = WorkerThreadPool::INVALID_TASK_ID;
	}

	iteration_build.region = nullptr;
	iteration_build.region_iteration = Ref<NavRegionIteration3D>();
	iteration = Ref<NavRegionIteration3D>();
}
