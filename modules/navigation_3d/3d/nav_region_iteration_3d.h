/**************************************************************************/
/*  nav_region_iteration_3d.h                                             */
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

#include "../nav_utils_3d.h"
#include "nav_base_iteration_3d.h"
#include "scene/resources/navigation_mesh.h"

#include "core/math/aabb.h"

class NavRegion3D;
class NavRegionIteration3D;

struct NavRegionIterationBuild3D {
	Nav3D::PerformanceData performance_data;

	NavRegion3D *region = nullptr;

	Vector3 map_cell_size;
	Transform3D region_transform;

	struct NavMeshData {
		Vector<Vector3> vertices;
		Vector<Vector<int>> polygons;

		void clear() {
			vertices.clear();
			polygons.clear();
		}
	} navmesh_data;

	Ref<NavRegionIteration3D> region_iteration;

	HashMap<Nav3D::EdgeKey, Nav3D::EdgeConnectionPair, Nav3D::EdgeKey> iter_connection_pairs_map;

	void reset() {
		performance_data.reset();

		navmesh_data.clear();
		region_iteration = Ref<NavRegionIteration3D>();
		iter_connection_pairs_map.clear();
	}
};

class NavRegionIteration3D : public NavBaseIteration3D {
	GDCLASS(NavRegionIteration3D, NavBaseIteration3D);

public:
	Transform3D transform;
	real_t surface_area = 0.0;
	AABB bounds;
	LocalVector<Nav3D::ConnectableEdge> external_edges;

	const Transform3D &get_transform() const { return transform; }
	real_t get_surface_area() const { return surface_area; }
	AABB get_bounds() const { return bounds; }
	const LocalVector<Nav3D::ConnectableEdge> &get_external_edges() const { return external_edges; }

	virtual ~NavRegionIteration3D() override {
		external_edges.clear();
		navmesh_polygons.clear();
		internal_connections.clear();
	}
};
