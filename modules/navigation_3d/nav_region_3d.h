/**************************************************************************/
/*  nav_region_3d.h                                                       */
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

#pragma once

#include "nav_base_3d.h"
#include "nav_utils_3d.h"

#include "core/os/rw_lock.h"
#include "scene/resources/navigation_mesh.h"

#include "3d/nav_region_iteration_3d.h"

class NavRegion3D : public NavBase3D {
	RWLock region_rwlock;

	NavMap3D *map = nullptr;
	Transform3D transform;
	bool enabled = true;

	bool use_edge_connections = true;

	AABB bounds;

	Ref<NavigationMesh> navmesh;

	Nav3D::PerformanceData performance_data;

	uint32_t iteration_id = 0;

	SelfList<NavRegion3D> sync_dirty_request_list_element;
	mutable RWLock iteration_rwlock;
	Ref<NavRegionIteration3D> iteration;

	NavRegionIterationBuild3D iteration_build;
	bool use_async_iterations = true;
	SelfList<NavRegion3D> async_list_element;
	WorkerThreadPool::TaskID iteration_build_thread_task_id = WorkerThreadPool::INVALID_TASK_ID;
	static void _build_iteration_threaded(void *p_arg);

	bool iteration_dirty = true;
	bool iteration_building = false;
	bool iteration_ready = false;

	void _build_iteration();
	void _sync_iteration();

public:
	NavRegion3D();
	~NavRegion3D();

	uint32_t get_iteration_id() const { return iteration_id; }

	void scratch_polygons();

	void set_enabled(bool p_enabled);
	bool get_enabled() const { return enabled; }

	void set_map(NavMap3D *p_map);
	NavMap3D *get_map() const {
		return map;
	}

	virtual void set_use_edge_connections(bool p_enabled) override;
	virtual bool get_use_edge_connections() const override { return use_edge_connections; }

	void set_transform(Transform3D transform);
	const Transform3D &get_transform() const {
		return transform;
	}

	void set_navigation_mesh(Ref<NavigationMesh> p_navigation_mesh);
	Ref<NavigationMesh> get_navigation_mesh() const { return navmesh; }

	LocalVector<Nav3D::Polygon> const &get_polygons() const;

	Vector3 get_closest_point_to_segment(const Vector3 &p_from, const Vector3 &p_to, bool p_use_collision) const;
	Nav3D::ClosestPointQueryResult get_closest_point_info(const Vector3 &p_point) const;
	Vector3 get_random_point(uint32_t p_navigation_layers, bool p_uniformly) const;

	real_t get_surface_area() const;
	AABB get_bounds() const;

	// NavBase properties.
	virtual void set_navigation_layers(uint32_t p_navigation_layers) override;
	virtual void set_enter_cost(real_t p_enter_cost) override;
	virtual void set_travel_cost(real_t p_travel_cost) override;
	virtual void set_owner_id(ObjectID p_owner_id) override;

	bool sync();
	void request_sync();
	void cancel_sync_request();

	void sync_async_tasks();
	void request_async_thread_join();
	void cancel_async_thread_join();

	Ref<NavRegionIteration3D> get_iteration();

	// Performance Monitor
	int get_pm_polygon_count() const { return performance_data.pm_polygon_count; }
	int get_pm_edge_count() const { return performance_data.pm_edge_count; }
	int get_pm_edge_merge_count() const { return performance_data.pm_edge_merge_count; }

	void set_use_async_iterations(bool p_enabled);
	bool get_use_async_iterations() const;
};
