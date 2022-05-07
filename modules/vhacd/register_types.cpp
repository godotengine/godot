/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "register_types.h"
#include "scene/resources/mesh.h"
#include "thirdparty/vhacd/public/VHACD.h"

static Vector<PoolVector<Vector3>> convex_decompose(const real_t *p_vertices, int p_vertex_count, const uint32_t *p_triangles, int p_triangle_count, int p_max_convex_hulls = -1, Vector<PoolVector<uint32_t>> *r_convex_indices = nullptr) {
	VHACD::IVHACD::Parameters params;
	if (p_max_convex_hulls > 0) {
		params.m_maxConvexHulls = p_max_convex_hulls;
	}

	VHACD::IVHACD *decomposer = VHACD::CreateVHACD();
	decomposer->Compute(p_vertices, p_vertex_count, p_triangles, p_triangle_count, params);

	int hull_count = decomposer->GetNConvexHulls();

	Vector<PoolVector<Vector3>> ret;
	ret.resize(hull_count);

	if (r_convex_indices) {
		r_convex_indices->resize(hull_count);
	}

	for (int i = 0; i < hull_count; i++) {
		VHACD::IVHACD::ConvexHull hull;
		decomposer->GetConvexHull(i, hull);

		PoolVector<Vector3> &points = ret.write[i];
		points.resize(hull.m_nPoints);

		PoolVector<Vector3>::Write w = points.write();
		for (uint32_t j = 0; j < hull.m_nPoints; ++j) {
			for (int k = 0; k < 3; k++) {
				w[j][k] = hull.m_points[j * 3 + k];
			}
		}

		if (r_convex_indices) {
			PoolVector<uint32_t> &indices = r_convex_indices->write[i];
			indices.resize(hull.m_nTriangles * 3);

			memcpy(indices.write().ptr(), hull.m_triangles, hull.m_nTriangles * 3 * sizeof(uint32_t));
		}
	}

	decomposer->Clean();
	decomposer->Release();

	return ret;
}

void register_vhacd_types() {
	Mesh::convex_decomposition_function = convex_decompose;
}

void unregister_vhacd_types() {
	Mesh::convex_decomposition_function = nullptr;
}
