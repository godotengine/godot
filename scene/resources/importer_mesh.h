/**************************************************************************/
/*  importer_mesh.h                                                       */
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

#ifndef IMPORTER_MESH_H
#define IMPORTER_MESH_H

#include "core/io/resource.h"
#include "core/templates/local_vector.h"
#include "scene/resources/concave_polygon_shape_3d.h"
#include "scene/resources/convex_polygon_shape_3d.h"
#include "scene/resources/mesh.h"
#include "scene/resources/navigation_mesh.h"

#include "thirdparty/misc/mikktspace.h"
#include <cstdint>

// The following classes are used by importers instead of ArrayMesh and MeshInstance3D
// so the data is not registered (hence, quality loss), importing happens faster and
// its easier to modify before saving

//mikktspace callbacks
namespace {
struct TangentGenerationContextUserData {
	struct Vertex {
		Vector3 vertex;
		Color color;
		Vector3 normal; // normal, binormal, tangent
		Vector3 binormal;
		Vector3 tangent;
		Vector2 uv;
		Vector2 uv2;
		Vector<int> bones;
		Vector<float> weights;
		Color custom[RS::ARRAY_CUSTOM_COUNT];
		uint32_t smooth_group = 0;

		bool operator==(const Vertex &p_vertex) const;

		Vertex() {}
	};
	LocalVector<Vertex> *vertices;
	LocalVector<int> *indices;
};
} // namespace

class ImporterMesh : public Resource {
	GDCLASS(ImporterMesh, Resource)

	struct Surface {
		Mesh::PrimitiveType primitive;
		Array arrays;
		struct BlendShape {
			Array arrays;
		};
		Vector<BlendShape> blend_shape_data;
		struct LOD {
			Vector<int> indices;
			float distance = 0.0f;
		};
		Vector<LOD> lods;
		Ref<Material> material;
		String name;
		uint64_t flags = 0;

		struct LODComparator {
			_FORCE_INLINE_ bool operator()(const LOD &l, const LOD &r) const {
				return l.distance < r.distance;
			}
		};

		void split_normals(const LocalVector<int> &p_indices, const LocalVector<Vector3> &p_normals);
		static void _split_normals(Array &r_arrays, const LocalVector<int> &p_indices, const LocalVector<Vector3> &p_normals);
	};
	Vector<Surface> surfaces;
	Vector<String> blend_shapes;
	Mesh::BlendShapeMode blend_shape_mode = Mesh::BLEND_SHAPE_MODE_NORMALIZED;

	Ref<ArrayMesh> mesh;

	Ref<ImporterMesh> shadow_mesh;

	Size2i lightmap_size_hint;

protected:
	void _set_data(const Dictionary &p_data);
	Dictionary _get_data() const;

	static void _bind_methods();

public:
	static int mikktGetNumFaces(const SMikkTSpaceContext *pContext) {
		TangentGenerationContextUserData &triangle_data = *reinterpret_cast<TangentGenerationContextUserData *>(pContext->m_pUserData);

		if (triangle_data.indices->size() > 0) {
			return triangle_data.indices->size() / 3;
		} else {
			return triangle_data.vertices->size() / 3;
		}
	}

	static int mikktGetNumVerticesOfFace(const SMikkTSpaceContext *pContext, const int iFace) {
		return 3; //always 3
	}

	static void mikktGetPosition(const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace, const int iVert) {
		TangentGenerationContextUserData &triangle_data = *reinterpret_cast<TangentGenerationContextUserData *>(pContext->m_pUserData);
		Vector3 v;
		if (triangle_data.indices->size() > 0) {
			uint32_t index = triangle_data.indices->operator[](iFace * 3 + iVert);
			if (index < triangle_data.vertices->size()) {
				v = triangle_data.vertices->operator[](index).vertex;
			}
		} else {
			v = triangle_data.vertices->operator[](iFace * 3 + iVert).vertex;
		}

		fvPosOut[0] = v.x;
		fvPosOut[1] = v.y;
		fvPosOut[2] = v.z;
	}

	static void mikktGetNormal(const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace, const int iVert) {
		TangentGenerationContextUserData &triangle_data = *reinterpret_cast<TangentGenerationContextUserData *>(pContext->m_pUserData);
		Vector3 v;
		if (triangle_data.indices->size() > 0) {
			uint32_t index = triangle_data.indices->operator[](iFace * 3 + iVert);
			if (index < triangle_data.vertices->size()) {
				v = triangle_data.vertices->operator[](index).normal;
			}
		} else {
			v = triangle_data.vertices->operator[](iFace * 3 + iVert).normal;
		}

		fvNormOut[0] = v.x;
		fvNormOut[1] = v.y;
		fvNormOut[2] = v.z;
	}

	static void mikktGetTexCoord(const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace, const int iVert) {
		TangentGenerationContextUserData &triangle_data = *reinterpret_cast<TangentGenerationContextUserData *>(pContext->m_pUserData);
		Vector2 v;
		if (triangle_data.indices->size() > 0) {
			uint32_t index = triangle_data.indices->operator[](iFace * 3 + iVert);
			if (index < triangle_data.vertices->size()) {
				v = triangle_data.vertices->operator[](index).uv;
			}
		} else {
			v = triangle_data.vertices->operator[](iFace * 3 + iVert).uv;
		}

		fvTexcOut[0] = v.x;
		fvTexcOut[1] = v.y;
	}

	static void mikktSetTSpaceDefault(const SMikkTSpaceContext *pContext, const float fvTangent[], const float fvBiTangent[], const float fMagS, const float fMagT,
			const tbool bIsOrientationPreserving, const int iFace, const int iVert) {
		TangentGenerationContextUserData &triangle_data = *reinterpret_cast<TangentGenerationContextUserData *>(pContext->m_pUserData);
		TangentGenerationContextUserData::Vertex *vtx = nullptr;
		if (triangle_data.indices->size() > 0) {
			uint32_t index = triangle_data.indices->operator[](iFace * 3 + iVert);
			if (index < triangle_data.vertices->size()) {
				vtx = &triangle_data.vertices->operator[](index);
			}
		} else {
			vtx = &triangle_data.vertices->operator[](iFace * 3 + iVert);
		}

		if (vtx != nullptr) {
			vtx->tangent = Vector3(fvTangent[0], fvTangent[1], fvTangent[2]);
			vtx->binormal = Vector3(-fvBiTangent[0], -fvBiTangent[1], -fvBiTangent[2]); // for some reason these are reversed, something with the coordinate system in Godot
		}
	}

	void generate_tangents();
	void add_blend_shape(const String &p_name);
	int get_blend_shape_count() const;
	int get_surface_blend_shape_count(int p_surface) const;
	String get_blend_shape_name(int p_blend_shape) const;

	void add_surface(Mesh::PrimitiveType p_primitive, const Array &p_arrays, const TypedArray<Array> &p_blend_shapes = Array(), const Dictionary &p_lods = Dictionary(), const Ref<Material> &p_material = Ref<Material>(), const String &p_name = String(), const uint64_t p_flags = 0);
	int get_surface_count() const;

	void set_blend_shape_mode(Mesh::BlendShapeMode p_blend_shape_mode);
	Mesh::BlendShapeMode get_blend_shape_mode() const;

	Mesh::PrimitiveType get_surface_primitive_type(int p_surface);
	String get_surface_name(int p_surface) const;
	void set_surface_name(int p_surface, const String &p_name);
	Array get_surface_arrays(int p_surface) const;
	Array get_surface_blend_shape_arrays(int p_surface, int p_blend_shape) const;
	int get_surface_lod_count(int p_surface) const;
	Vector<int> get_surface_lod_indices(int p_surface, int p_lod) const;
	float get_surface_lod_size(int p_surface, int p_lod) const;
	Ref<Material> get_surface_material(int p_surface) const;
	uint64_t get_surface_format(int p_surface) const;

	void set_surface_material(int p_surface, const Ref<Material> &p_material);

	void generate_lods(float p_normal_merge_angle, float p_normal_split_angle, Array p_skin_pose_transform_array);

	void create_shadow_mesh();
	Ref<ImporterMesh> get_shadow_mesh() const;

	Vector<Face3> get_faces() const;
	Vector<Ref<Shape3D>> convex_decompose(const Ref<MeshConvexDecompositionSettings> &p_settings) const;
	Ref<ConvexPolygonShape3D> create_convex_shape(bool p_clean = true, bool p_simplify = false) const;
	Ref<ConcavePolygonShape3D> create_trimesh_shape() const;
	Ref<NavigationMesh> create_navigation_mesh();
	Error lightmap_unwrap_cached(const Transform3D &p_base_transform, float p_texel_size, const Vector<uint8_t> &p_src_cache, Vector<uint8_t> &r_dst_cache);

	void set_lightmap_size_hint(const Size2i &p_size);
	Size2i get_lightmap_size_hint() const;

	bool has_mesh() const;
	Ref<ArrayMesh> get_mesh(const Ref<ArrayMesh> &p_base = Ref<ArrayMesh>());
	void clear();
};

#endif // IMPORTER_MESH_H
