/**************************************************************************/
/*  nav_region_2d.h                                                       */
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

#include "nav_base_2d.h"
#include "nav_utils_2d.h"

#include "core/os/rw_lock.h"
#include "scene/resources/2d/navigation_polygon.h"

#include "2d/nav_region_iteration_2d.h"

class NavRegion2D : public NavBase2D {
	RWLock region_rwlock;

	NavMap2D *map = nullptr;
	Transform2D transform;
	bool enabled = true;

	bool use_edge_connections = true;

	Rect2 bounds;

	Ref<NavigationPolygon> navmesh;

	Nav2D::PerformanceData performance_data;

	uint32_t iteration_id = 0;

	SelfList<NavRegion2D> sync_dirty_request_list_element;
	mutable RWLock iteration_rwlock;
	Ref<NavRegionIteration2D> iteration;

	NavRegionIterationBuild2D iteration_build;
	bool use_async_iterations = true;
	SelfList<NavRegion2D> async_list_element;
	WorkerThreadPool::TaskID iteration_build_thread_task_id = WorkerThreadPool::INVALID_TASK_ID;
	static void _build_iteration_threaded(void *p_arg);

	bool iteration_dirty = true;
	bool iteration_building = false;
	bool iteration_ready = false;

	void _build_iteration();
	void _sync_iteration();

public:
	NavRegion2D();
	~NavRegion2D();

	uint32_t get_iteration_id() const { return iteration_id; }

	void scratch_polygons();

	void set_enabled(bool p_enabled);
	bool get_enabled() const { return enabled; }

	void set_map(NavMap2D *p_map);
	NavMap2D *get_map() const {
		return map;
	}

	virtual void set_use_edge_connections(bool p_enabled) override;
	virtual bool get_use_edge_connections() const override { return use_edge_connections; }

	void set_transform(Transform2D transform);
	const Transform2D &get_transform() const {
		return transform;
	}

	void set_navigation_mesh(Ref<NavigationPolygon> p_navigation_mesh);
	Ref<NavigationPolygon> get_navigation_mesh() const { return navmesh; }

	LocalVector<Nav2D::Polygon> const &get_polygons() const;

	Nav2D::ClosestPointQueryResult get_closest_point_info(const Vector2 &p_point) const;
	Vector2 get_random_point(uint32_t p_navigation_layers, bool p_uniformly) const;

	real_t get_surface_area() const;
	Rect2 get_bounds() const;

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

	Ref<NavRegionIteration2D> get_iteration();

	// Performance Monitor
	int get_pm_polygon_count() const { return performance_data.pm_polygon_count; }
	int get_pm_edge_count() const { return performance_data.pm_edge_count; }
	int get_pm_edge_merge_count() const { return performance_data.pm_edge_merge_count; }

	void set_use_async_iterations(bool p_enabled);
	bool get_use_async_iterations() const;
};
