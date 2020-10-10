/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

static Vector<Vector<Face3>> convex_decompose(const Vector<Face3> &p_faces) {
	Vector<float> vertices;
	vertices.resize(p_faces.size() * 9);
	Vector<uint32_t> indices;
	indices.resize(p_faces.size() * 3);

	for (int i = 0; i < p_faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			vertices.write[i * 9 + j * 3 + 0] = p_faces[i].vertex[j].x;
			vertices.write[i * 9 + j * 3 + 1] = p_faces[i].vertex[j].y;
			vertices.write[i * 9 + j * 3 + 2] = p_faces[i].vertex[j].z;
			indices.write[i * 3 + j] = i * 3 + j;
		}
	}

	VHACD::IVHACD *decomposer = VHACD::CreateVHACD();
	VHACD::IVHACD::Parameters params;
	decomposer->Compute(vertices.ptr(), vertices.size() / 3, indices.ptr(), indices.size() / 3, params);

	int hull_count = decomposer->GetNConvexHulls();

	Vector<Vector<Face3>> ret;

	for (int i = 0; i < hull_count; i++) {
		Vector<Face3> triangles;
		VHACD::IVHACD::ConvexHull hull;
		decomposer->GetConvexHull(i, hull);
		triangles.resize(hull.m_nTriangles);
		for (uint32_t j = 0; j < hull.m_nTriangles; j++) {
			Face3 f;
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					f.vertex[k][l] = hull.m_points[hull.m_triangles[j * 3 + k] * 3 + l];
				}
			}
			triangles.write[j] = f;
		}
		ret.push_back(triangles);
	}

	decomposer->Clean();
	decomposer->Release();

	return ret;
}

void register_vhacd_types() {
	Mesh::convex_composition_function = convex_decompose;
}

void unregister_vhacd_types() {
	Mesh::convex_composition_function = nullptr;
}
