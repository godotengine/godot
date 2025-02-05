/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "scene/resources/mesh.h"

#define ENABLE_VHACD_IMPLEMENTATION 1
#include "thirdparty/vhacd/VHACD.h"

static Vector<Vector<Vector3>> convex_decompose(const real_t *p_vertices, int p_vertex_count, const uint32_t *p_triangles, int p_triangle_count, const Ref<MeshConvexDecompositionSettings> &p_settings, Vector<Vector<uint32_t>> *r_convex_indices) {
	VHACD::IVHACD::Parameters params;
	params.m_minimumVolumePercentErrorAllowed = p_settings->get_max_concavity();
	params.m_resolution = p_settings->get_resolution();
	params.m_maxNumVerticesPerCH = p_settings->get_max_num_vertices_per_convex_hull();
	params.m_maxConvexHulls = p_settings->get_max_convex_hulls();
	params.m_shrinkWrap = p_settings->get_project_hull_vertices();

	// Following are defaults
	// params.m_maxRecursionDepth = 10;        // The maximum recursion depth
	// params.m_fillMode = FillMode::FLOOD_FILL; // How to fill the interior of the voxelized mesh
	// params.m_asyncACD = true;             // Whether or not to run asynchronously, taking advantage of additional cores
	// params.m_minEdgeLength = 2;           // Once a voxel patch has an edge length of less than 4 on all 3 sides, we don't keep recursing
	// params.m_findBestPlane = false;       // Whether or not to attempt to split planes along the best location. Experimental feature. False by default.

	VHACD::IVHACD *decomposer = VHACD::CreateVHACD();
	decomposer->Compute(p_vertices, p_vertex_count, p_triangles, p_triangle_count, params);

	int hull_count = decomposer->GetNConvexHulls();

	Vector<Vector<Vector3>> ret;
	ret.resize(hull_count);

	if (r_convex_indices) {
		r_convex_indices->resize(hull_count);
	}

	for (int i = 0; i < hull_count; i++) {
		VHACD::IVHACD::ConvexHull hull;
		decomposer->GetConvexHull(i, hull);

		uint32_t m_nPoints = hull.m_points.size();
		Vector<Vector3> &points = ret.write[i];
		points.resize(m_nPoints);

		Vector3 *w = points.ptrw();
		for (uint32_t j = 0; j < m_nPoints; ++j) {
			w[j].x = hull.m_points[j].mX;
			w[j].y = hull.m_points[j].mY;
			w[j].z = hull.m_points[j].mZ;
		}

		if (r_convex_indices) {
			uint32_t m_nTriangles = hull.m_triangles.size();
			Vector<uint32_t> &indices = r_convex_indices->write[i];
			indices.resize(m_nTriangles * 3);

			uint32_t *ind = indices.ptrw();
			for (uint32_t j = 0; j < m_nTriangles; j = j + 3) {
				ind[j] = hull.m_triangles[j].mI0;
				ind[j + 1] = hull.m_triangles[j].mI1;
				ind[j + 2] = hull.m_triangles[j].mI2;
			}
		}
	}

	decomposer->Clean();
	decomposer->Release();

	return ret;
}

void initialize_vhacd_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	Mesh::convex_decomposition_function = convex_decompose;
}

void uninitialize_vhacd_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	Mesh::convex_decomposition_function = nullptr;
}
