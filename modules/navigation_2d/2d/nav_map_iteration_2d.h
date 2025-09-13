/**************************************************************************/
/*  nav_map_iteration_2d.h                                                */
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

#include "../nav_rid_2d.h"
#include "../nav_utils_2d.h"
#include "nav_mesh_queries_2d.h"

#include "core/math/math_defs.h"
#include "core/os/semaphore.h"

class NavLinkIteration2D;
class NavRegion2D;
class NavRegionIteration2D;
struct NavMapIteration2D;

struct NavMapIterationBuild2D {
	Vector2 merge_rasterizer_cell_size;
	bool use_edge_connections = true;
	real_t edge_connection_margin;
	real_t link_connection_radius;
	Nav2D::PerformanceData performance_data;
	int polygon_count = 0;
	int free_edge_count = 0;

	HashMap<Nav2D::EdgeKey, Nav2D::EdgeConnectionPair, Nav2D::EdgeKey> iter_connection_pairs_map;
	LocalVector<Nav2D::Connection> iter_free_edges;

	NavMapIteration2D *map_iteration = nullptr;

	int navmesh_polygon_count = 0;

	void reset() {
		performance_data.reset();

		iter_connection_pairs_map.clear();
		iter_free_edges.clear();
		polygon_count = 0;
		free_edge_count = 0;

		navmesh_polygon_count = 0;
	}
};

struct NavMapIteration2D {
	mutable SafeNumeric<uint32_t> users;
	RWLock rwlock;

	LocalVector<Ref<NavRegionIteration2D>> region_iterations;
	LocalVector<Ref<NavLinkIteration2D>> link_iterations;

	int navmesh_polygon_count = 0;

	// The edge connections that the map builds on top with the edge connection margin.
	HashMap<const NavBaseIteration2D *, LocalVector<Nav2D::Connection>> external_region_connections;
	HashMap<const NavBaseIteration2D *, LocalVector<LocalVector<Nav2D::Connection>>> navbases_polygons_external_connections;

	LocalVector<Nav2D::Polygon> navlink_polygons;

	HashMap<NavRegion2D *, Ref<NavRegionIteration2D>> region_ptr_to_region_iteration;

	LocalVector<NavMeshQueries2D::PathQuerySlot> path_query_slots;
	Mutex path_query_slots_mutex;
	Semaphore path_query_slots_semaphore;

	void clear() {
		navmesh_polygon_count = 0;

		region_iterations.clear();
		link_iterations.clear();
		external_region_connections.clear();
		navbases_polygons_external_connections.clear();
		navlink_polygons.clear();
		region_ptr_to_region_iteration.clear();
	}
};

class NavMapIterationRead2D {
	const NavMapIteration2D &map_iteration;

public:
	_ALWAYS_INLINE_ NavMapIterationRead2D(const NavMapIteration2D &p_iteration) :
			map_iteration(p_iteration) {
		map_iteration.rwlock.read_lock();
		map_iteration.users.increment();
	}
	_ALWAYS_INLINE_ ~NavMapIterationRead2D() {
		map_iteration.users.decrement();
		map_iteration.rwlock.read_unlock();
	}
};
