/**************************************************************************/
/*  csg.cpp                                                               */
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

#include "csg.h"

#include "core/math/geometry_2d.h"
#include "core/math/math_funcs.h"
#include "core/templates/sort_array.h"
#include "scene/resources/mesh_data_tool.h"
#include "scene/resources/surface_tool.h"

#include "thirdparty/manifold/include/manifold/manifold.h"

// CSGBrush

void CSGBrush::build_from_faces(const Vector<Vector3> &p_vertices, const Vector<Vector2> &p_uvs, const Vector<bool> &p_smooth, const Vector<Ref<Material>> &p_materials, const Vector<bool> &p_flip_faces) {
	faces.clear();

	int vc = p_vertices.size();

	ERR_FAIL_COND((vc % 3) != 0);

	const Vector3 *rv = p_vertices.ptr();
	int uvc = p_uvs.size();
	const Vector2 *ruv = p_uvs.ptr();
	int sc = p_smooth.size();
	const bool *rs = p_smooth.ptr();
	int mc = p_materials.size();
	const Ref<Material> *rm = p_materials.ptr();
	int ic = p_flip_faces.size();
	const bool *ri = p_flip_faces.ptr();

	HashMap<Ref<Material>, int> material_map;

	faces.resize(p_vertices.size() / 3);

	for (int i = 0; i < faces.size(); i++) {
		Face &f = faces.write[i];
		f.vertices[0] = rv[i * 3 + 0];
		f.vertices[1] = rv[i * 3 + 1];
		f.vertices[2] = rv[i * 3 + 2];

		if (uvc == vc) {
			f.uvs[0] = ruv[i * 3 + 0];
			f.uvs[1] = ruv[i * 3 + 1];
			f.uvs[2] = ruv[i * 3 + 2];
		}

		if (sc == vc / 3) {
			f.smooth = rs[i];
		} else {
			f.smooth = false;
		}

		if (ic == vc / 3) {
			f.invert = ri[i];
		} else {
			f.invert = false;
		}

		if (mc == vc / 3) {
			Ref<Material> mat = rm[i];
			if (mat.is_valid()) {
				HashMap<Ref<Material>, int>::ConstIterator E = material_map.find(mat);

				if (E) {
					f.material = E->value;
				} else {
					f.material = material_map.size();
					material_map[mat] = f.material;
				}

			} else {
				f.material = -1;
			}
		}
	}

	materials.resize(material_map.size());
	for (const KeyValue<Ref<Material>, int> &E : material_map) {
		materials.write[E.value] = E.key;
	}

	_regen_face_aabbs();
}

void CSGBrush::copy_from(const CSGBrush &p_brush, const Transform3D &p_xform) {
	faces = p_brush.faces;
	materials = p_brush.materials;

	for (int i = 0; i < faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			faces.write[i].vertices[j] = p_xform.xform(p_brush.faces[i].vertices[j]);
		}
	}

	_regen_face_aabbs();
}

enum {
	MANIFOLD_PROPERTY_POSITION_X = 0,
	MANIFOLD_PROPERTY_POSITION_Y,
	MANIFOLD_PROPERTY_POSITION_Z,
	MANIFOLD_PROPERTY_NORMAL_X,
	MANIFOLD_PROPERTY_NORMAL_Y,
	MANIFOLD_PROPERTY_NORMAL_Z,
	MANIFOLD_PROPERTY_INVERT,
	MANIFOLD_PROPERTY_SMOOTH_GROUP,
	MANIFOLD_PROPERTY_UV_X_0,
	MANIFOLD_PROPERTY_UV_Y_0,
	MANIFOLD_MAX
};

static void _pack_manifold(
		const CSGBrush *const p_mesh_merge,
		manifold::Manifold &r_manifold,
		HashMap<int32_t, Ref<Material>> &p_mesh_materials,
		Ref<Material> p_default_material,
		float p_snap) {
	ERR_FAIL_NULL_MSG(p_mesh_merge, "p_mesh_merge is null");

	HashMap<uint32_t, Vector<CSGBrush::Face>> faces_by_material;
	for (int face_i = 0; face_i < p_mesh_merge->faces.size(); face_i++) {
		const CSGBrush::Face &face = p_mesh_merge->faces[face_i];
		faces_by_material[face.material].push_back(face);
	}

	manifold::MeshGL64 mesh;
	mesh.numProp = MANIFOLD_MAX;
	mesh.runOriginalID.reserve(faces_by_material.size());
	mesh.runIndex.reserve(faces_by_material.size() + 1);
	mesh.vertProperties.reserve(p_mesh_merge->faces.size() * 3 * MANIFOLD_MAX);

	// Make a run of triangles for each material.
	for (const KeyValue<uint32_t, Vector<CSGBrush::Face>> &E : faces_by_material) {
		const uint32_t material_id = E.key;
		const Vector<CSGBrush::Face> &faces = E.value;
		mesh.runIndex.push_back(mesh.triVerts.size());

		// Associate the material with an ID.
		uint32_t reserved_id = r_manifold.ReserveIDs(1);
		mesh.runOriginalID.push_back(reserved_id);
		Ref<Material> material;
		if (material_id < p_mesh_merge->materials.size()) {
			material = p_mesh_merge->materials[material_id];
		}

		if (material.is_null()) {
			material = p_default_material;
		}

		p_mesh_materials.insert(reserved_id, material);

		for (const CSGBrush::Face &face : faces) {
			for (int32_t tri_order_i = 0; tri_order_i < 3; tri_order_i++) {
				constexpr int32_t order[3] = { 0, 2, 1 };
				int i = order[tri_order_i];

				mesh.triVerts.push_back(mesh.vertProperties.size() / MANIFOLD_MAX);

				size_t begin = mesh.vertProperties.size();
				mesh.vertProperties.resize(mesh.vertProperties.size() + MANIFOLD_MAX);
				// Add the vertex properties.
				// Use CSGBrush constants rather than push_back for clarity.
				double *vert = &mesh.vertProperties[begin];
				vert[MANIFOLD_PROPERTY_POSITION_X] = face.vertices[i].x;
				vert[MANIFOLD_PROPERTY_POSITION_Y] = face.vertices[i].y;
				vert[MANIFOLD_PROPERTY_POSITION_Z] = face.vertices[i].z;
				vert[MANIFOLD_PROPERTY_UV_X_0] = face.uvs[i].x;
				vert[MANIFOLD_PROPERTY_UV_Y_0] = face.uvs[i].y;
				vert[MANIFOLD_PROPERTY_SMOOTH_GROUP] = face.smooth ? 1.0f : 0.0f;
				vert[MANIFOLD_PROPERTY_INVERT] = face.invert ? 1.0f : 0.0f;
			}
		}
	}
	// runIndex needs an explicit end value.
	mesh.runIndex.push_back(mesh.triVerts.size());

	ERR_FAIL_COND_MSG(mesh.vertProperties.size() % mesh.numProp != 0, "Invalid vertex properties size.");

	mesh.precision = p_snap;

	/**
	 * MeshGL64::merge(): updates the mergeFromVert and mergeToVert vectors in order to create a
	 * manifold solid. If the MeshGL64 is already manifold, no change will occur, and
	 * the function will return false.
	 */
	if (mesh.Merge()) {
		std::vector<int32_t> index_map(mesh.vertProperties.size() / MANIFOLD_MAX, -1);
		const size_t vertices_count = mesh.mergeFromVert.size();
		for (size_t i = 0; i < vertices_count; ++i) {
			index_map[mesh.mergeFromVert[i]] = mesh.mergeToVert[i];
		}
		const size_t indices_count = mesh.triVerts.size();
		for (size_t i = 0; i < indices_count; ++i) {
			if (index_map[i] > -1) {
				mesh.triVerts[i] = index_map[i];
			}
		}
	}

	r_manifold = manifold::Manifold(mesh);
	manifold::Manifold::Error err = r_manifold.Status();
	if (err != manifold::Manifold::Error::NoError) {
		print_error(String("Manifold creation from mesh failed:" + itos((int)err)));
	}
}

static void _unpack_manifold(
		const manifold::Manifold &p_manifold,
		const HashMap<int32_t, Ref<Material>> &p_mesh_materials,
		CSGBrush *r_mesh_merge) {
	manifold::MeshGL64 mesh = p_manifold.GetMeshGL64();

	constexpr int32_t order[3] = { 0, 2, 1 };

	for (size_t run_i = 0; run_i < mesh.runIndex.size() - 1; run_i++) {
		uint32_t original_id = -1;
		if (run_i < mesh.runOriginalID.size()) {
			original_id = mesh.runOriginalID[run_i];
		}

		Ref<Material> material;
		if (p_mesh_materials.has(original_id)) {
			material = p_mesh_materials[original_id];
		}
		// Find or reserve a material ID in the brush.
		int run_material = 0;
		int32_t material_id = r_mesh_merge->materials.find(material);
		if (material_id != -1) {
			run_material = material_id;
		} else {
			run_material = r_mesh_merge->materials.size();
			r_mesh_merge->materials.push_back(material);
		}

		size_t begin = mesh.runIndex[run_i];
		size_t end = mesh.runIndex[run_i + 1];
		for (size_t vert_i = begin; vert_i < end; vert_i += 3) {
			CSGBrush::Face face;
			face.material = run_material;

			for (int32_t tri_order_i = 0; tri_order_i < 3; tri_order_i++) {
				int32_t property_i = mesh.triVerts[vert_i + order[tri_order_i]];

				ERR_FAIL_COND_MSG(property_i * mesh.numProp >= mesh.vertProperties.size(), "Invalid index into vertex properties");

				face.vertices[tri_order_i] = Vector3(
						mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_POSITION_X],
						mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_POSITION_Y],
						mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_POSITION_Z]);

				face.uvs[tri_order_i] = Vector2(
						mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_UV_X_0],
						mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_UV_Y_0]);

				face.smooth = mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_SMOOTH_GROUP] > 0.5f;
				face.invert = mesh.vertProperties[property_i * mesh.numProp + MANIFOLD_PROPERTY_INVERT] > 0.5f;
			}

			r_mesh_merge->faces.push_back(face);
		}
	}

	r_mesh_merge->_regen_face_aabbs();
}

// CSGBrushOperation

void CSGBrushOperation::merge_brushes(Operation p_operation, const CSGBrush &p_brush_a, const CSGBrush &p_brush_b, CSGBrush &r_merged_brush, float p_vertex_snap, Ref<Material> p_default_material) {
	HashMap<int32_t, Ref<Material>> mesh_materials;
	manifold::Manifold brush_a;
	_pack_manifold(&p_brush_a, brush_a, mesh_materials, p_default_material, p_vertex_snap);
	manifold::Manifold brush_b;
	_pack_manifold(&p_brush_b, brush_b, mesh_materials, p_default_material, p_vertex_snap);
	manifold::Manifold merged_brush;
	switch (p_operation) {
		case OPERATION_UNION:
			merged_brush = brush_a + brush_b;
			break;
		case OPERATION_INTERSECTION:
			merged_brush = brush_a ^ brush_b;
			break;
		case OPERATION_SUBTRACTION:
			merged_brush = brush_a - brush_b;
			break;
	}
	_unpack_manifold(merged_brush, mesh_materials, &r_merged_brush);
}
