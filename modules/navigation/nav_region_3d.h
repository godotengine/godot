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

struct NavRegionIteration3D;

class NavRegion3D : public NavBase3D {
	RWLock region_rwlock;

	NavMap3D *map = nullptr;
	Transform3D transform;
	bool enabled = true;

	bool use_edge_connections = true;

	bool region_dirty = true;
	bool polygons_dirty = true;

	LocalVector<nav_3d::Polygon> navmesh_polygons;

	real_t surface_area = 0.0;
	AABB bounds;

	RWLock navmesh_rwlock;
	Vector<Vector3> pending_navmesh_vertices;
	Vector<Vector<int>> pending_navmesh_polygons;

	SelfList<NavRegion3D> sync_dirty_request_list_element;

public:
	NavRegion3D();
	~NavRegion3D();

	void scratch_polygons() {
		polygons_dirty = true;
	}

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

	LocalVector<nav_3d::Polygon> const &get_polygons() const {
		return navmesh_polygons;
	}

	Vector3 get_closest_point_to_segment(const Vector3 &p_from, const Vector3 &p_to, bool p_use_collision) const;
	nav_3d::ClosestPointQueryResult get_closest_point_info(const Vector3 &p_point) const;
	Vector3 get_random_point(uint32_t p_navigation_layers, bool p_uniformly) const;

	real_t get_surface_area() const { return surface_area; }
	AABB get_bounds() const { return bounds; }

	// NavBase properties.
	virtual void set_navigation_layers(uint32_t p_navigation_layers) override;
	virtual void set_enter_cost(real_t p_enter_cost) override;
	virtual void set_travel_cost(real_t p_travel_cost) override;
	virtual void set_owner_id(ObjectID p_owner_id) override;

	bool sync();
	void request_sync();
	void cancel_sync_request();

	void get_iteration_update(NavRegionIteration3D &r_iteration);

private:
	void update_polygons();
};
