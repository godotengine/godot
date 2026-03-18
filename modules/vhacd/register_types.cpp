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

#include "thirdparty/vhacd/public/VHACD.h"

static Vector<Vector<Vector3>> convex_decompose(const real_t *p_vertices, int p_vertex_count, const uint32_t *p_triangles, int p_triangle_count, const Ref<MeshConvexDecompositionSettings> &p_settings, Vector<Vector<uint32_t>> *r_convex_indices) {
	VHACD::IVHACD::Parameters params;
	params.m_concavity = p_settings->get_max_concavity();
	params.m_alpha = p_settings->get_symmetry_planes_clipping_bias();
	params.m_beta = p_settings->get_revolution_axes_clipping_bias();
	params.m_minVolumePerCH = p_settings->get_min_volume_per_convex_hull();
	params.m_resolution = p_settings->get_resolution();
	params.m_maxNumVerticesPerCH = p_settings->get_max_num_vertices_per_convex_hull();
	params.m_planeDownsampling = p_settings->get_plane_downsampling();
	params.m_convexhullDownsampling = p_settings->get_convex_hull_downsampling();
	params.m_pca = p_settings->get_normalize_mesh();
	params.m_mode = p_settings->get_mode();
	params.m_convexhullApproximation = p_settings->get_convex_hull_approximation();
	params.m_oclAcceleration = true;
	params.m_maxConvexHulls = p_settings->get_max_convex_hulls();
	params.m_projectHullVertices = p_settings->get_project_hull_vertices();

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

		Vector<Vector3> &points = ret.write[i];
		points.resize(hull.m_nPoints);

		Vector3 *w = points.ptrw();
		for (uint32_t j = 0; j < hull.m_nPoints; ++j) {
			for (int k = 0; k < 3; k++) {
				w[j][k] = hull.m_points[j * 3 + k];
			}
		}

		if (r_convex_indices) {
			Vector<uint32_t> &indices = r_convex_indices->write[i];
			indices.resize(hull.m_nTriangles * 3);

			memcpy(indices.ptrw(), hull.m_triangles, hull.m_nTriangles * 3 * sizeof(uint32_t));
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
