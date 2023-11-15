/**************************************************************************/
/*  importer_mesh.cpp                                                     */
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

#include "importer_mesh.h"

#include "core/io/marshalls.h"
#include "core/math/convex_hull.h"
#include "core/math/random_pcg.h"
#include "core/math/static_raycaster.h"
#include "scene/resources/surface_tool.h"

#include <cstdint>

void ImporterMesh::Surface::split_normals(const LocalVector<int> &p_indices, const LocalVector<Vector3> &p_normals) {
	_split_normals(arrays, p_indices, p_normals);

	for (BlendShape &blend_shape : blend_shape_data) {
		_split_normals(blend_shape.arrays, p_indices, p_normals);
	}
}

void ImporterMesh::Surface::_split_normals(Array &r_arrays, const LocalVector<int> &p_indices, const LocalVector<Vector3> &p_normals) {
	ERR_FAIL_COND(r_arrays.size() != RS::ARRAY_MAX);

	const PackedVector3Array &vertices = r_arrays[RS::ARRAY_VERTEX];
	int current_vertex_count = vertices.size();
	int new_vertex_count = p_indices.size();
	int final_vertex_count = current_vertex_count + new_vertex_count;
	const int *indices_ptr = p_indices.ptr();

	for (int i = 0; i < r_arrays.size(); i++) {
		if (i == RS::ARRAY_INDEX) {
			continue;
		}

		if (r_arrays[i].get_type() == Variant::NIL) {
			continue;
		}

		switch (r_arrays[i].get_type()) {
			case Variant::PACKED_VECTOR3_ARRAY: {
				PackedVector3Array data = r_arrays[i];
				data.resize(final_vertex_count);
				Vector3 *data_ptr = data.ptrw();
				if (i == RS::ARRAY_NORMAL) {
					const Vector3 *normals_ptr = p_normals.ptr();
					memcpy(&data_ptr[current_vertex_count], normals_ptr, sizeof(Vector3) * new_vertex_count);
				} else {
					for (int j = 0; j < new_vertex_count; j++) {
						data_ptr[current_vertex_count + j] = data_ptr[indices_ptr[j]];
					}
				}
				r_arrays[i] = data;
			} break;
			case Variant::PACKED_VECTOR2_ARRAY: {
				PackedVector2Array data = r_arrays[i];
				data.resize(final_vertex_count);
				Vector2 *data_ptr = data.ptrw();
				for (int j = 0; j < new_vertex_count; j++) {
					data_ptr[current_vertex_count + j] = data_ptr[indices_ptr[j]];
				}
				r_arrays[i] = data;
			} break;
			case Variant::PACKED_FLOAT32_ARRAY: {
				PackedFloat32Array data = r_arrays[i];
				int elements = data.size() / current_vertex_count;
				data.resize(final_vertex_count * elements);
				float *data_ptr = data.ptrw();
				for (int j = 0; j < new_vertex_count; j++) {
					memcpy(&data_ptr[(current_vertex_count + j) * elements], &data_ptr[indices_ptr[j] * elements], sizeof(float) * elements);
				}
				r_arrays[i] = data;
			} break;
			case Variant::PACKED_INT32_ARRAY: {
				PackedInt32Array data = r_arrays[i];
				int elements = data.size() / current_vertex_count;
				data.resize(final_vertex_count * elements);
				int32_t *data_ptr = data.ptrw();
				for (int j = 0; j < new_vertex_count; j++) {
					memcpy(&data_ptr[(current_vertex_count + j) * elements], &data_ptr[indices_ptr[j] * elements], sizeof(int32_t) * elements);
				}
				r_arrays[i] = data;
			} break;
			case Variant::PACKED_BYTE_ARRAY: {
				PackedByteArray data = r_arrays[i];
				int elements = data.size() / current_vertex_count;
				data.resize(final_vertex_count * elements);
				uint8_t *data_ptr = data.ptrw();
				for (int j = 0; j < new_vertex_count; j++) {
					memcpy(&data_ptr[(current_vertex_count + j) * elements], &data_ptr[indices_ptr[j] * elements], sizeof(uint8_t) * elements);
				}
				r_arrays[i] = data;
			} break;
			case Variant::PACKED_COLOR_ARRAY: {
				PackedColorArray data = r_arrays[i];
				data.resize(final_vertex_count);
				Color *data_ptr = data.ptrw();
				for (int j = 0; j < new_vertex_count; j++) {
					data_ptr[current_vertex_count + j] = data_ptr[indices_ptr[j]];
				}
				r_arrays[i] = data;
			} break;
			default: {
				ERR_FAIL_MSG("Unhandled array type.");
			} break;
		}
	}
}

void ImporterMesh::add_blend_shape(const String &p_name) {
	ERR_FAIL_COND(surfaces.size() > 0);
	blend_shapes.push_back(p_name);
}

int ImporterMesh::get_blend_shape_count() const {
	return blend_shapes.size();
}

String ImporterMesh::get_blend_shape_name(int p_blend_shape) const {
	ERR_FAIL_INDEX_V(p_blend_shape, blend_shapes.size(), String());
	return blend_shapes[p_blend_shape];
}

void ImporterMesh::set_blend_shape_mode(Mesh::BlendShapeMode p_blend_shape_mode) {
	blend_shape_mode = p_blend_shape_mode;
}

Mesh::BlendShapeMode ImporterMesh::get_blend_shape_mode() const {
	return blend_shape_mode;
}

void ImporterMesh::add_surface(Mesh::PrimitiveType p_primitive, const Array &p_arrays, const TypedArray<Array> &p_blend_shapes, const Dictionary &p_lods, const Ref<Material> &p_material, const String &p_name, const uint64_t p_flags) {
	ERR_FAIL_COND(p_blend_shapes.size() != blend_shapes.size());
	ERR_FAIL_COND(p_arrays.size() != Mesh::ARRAY_MAX);
	Surface s;
	s.primitive = p_primitive;
	s.arrays = p_arrays;
	s.name = p_name;
	s.flags = p_flags;

	Vector<Vector3> vertex_array = p_arrays[Mesh::ARRAY_VERTEX];
	int vertex_count = vertex_array.size();
	ERR_FAIL_COND(vertex_count == 0);

	for (int i = 0; i < blend_shapes.size(); i++) {
		Array bsdata = p_blend_shapes[i];
		ERR_FAIL_COND(bsdata.size() != Mesh::ARRAY_MAX);
		Vector<Vector3> vertex_data = bsdata[Mesh::ARRAY_VERTEX];
		ERR_FAIL_COND(vertex_data.size() != vertex_count);
		Surface::BlendShape bs;
		bs.arrays = bsdata;
		s.blend_shape_data.push_back(bs);
	}

	List<Variant> lods;
	p_lods.get_key_list(&lods);
	for (const Variant &E : lods) {
		ERR_CONTINUE(!E.is_num());
		Surface::LOD lod;
		lod.distance = E;
		lod.indices = p_lods[E];
		ERR_CONTINUE(lod.indices.size() == 0);
		s.lods.push_back(lod);
	}

	s.material = p_material;

	surfaces.push_back(s);
	mesh.unref();
}

int ImporterMesh::get_surface_count() const {
	return surfaces.size();
}

Mesh::PrimitiveType ImporterMesh::get_surface_primitive_type(int p_surface) {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Mesh::PRIMITIVE_MAX);
	return surfaces[p_surface].primitive;
}
Array ImporterMesh::get_surface_arrays(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Array());
	return surfaces[p_surface].arrays;
}
String ImporterMesh::get_surface_name(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), String());
	return surfaces[p_surface].name;
}
void ImporterMesh::set_surface_name(int p_surface, const String &p_name) {
	ERR_FAIL_INDEX(p_surface, surfaces.size());
	surfaces.write[p_surface].name = p_name;
	mesh.unref();
}

Array ImporterMesh::get_surface_blend_shape_arrays(int p_surface, int p_blend_shape) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Array());
	ERR_FAIL_INDEX_V(p_blend_shape, surfaces[p_surface].blend_shape_data.size(), Array());
	return surfaces[p_surface].blend_shape_data[p_blend_shape].arrays;
}
int ImporterMesh::get_surface_lod_count(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), 0);
	return surfaces[p_surface].lods.size();
}
Vector<int> ImporterMesh::get_surface_lod_indices(int p_surface, int p_lod) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Vector<int>());
	ERR_FAIL_INDEX_V(p_lod, surfaces[p_surface].lods.size(), Vector<int>());

	return surfaces[p_surface].lods[p_lod].indices;
}

float ImporterMesh::get_surface_lod_size(int p_surface, int p_lod) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), 0);
	ERR_FAIL_INDEX_V(p_lod, surfaces[p_surface].lods.size(), 0);
	return surfaces[p_surface].lods[p_lod].distance;
}

uint64_t ImporterMesh::get_surface_format(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), 0);
	return surfaces[p_surface].flags;
}

Ref<Material> ImporterMesh::get_surface_material(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Ref<Material>());
	return surfaces[p_surface].material;
}

void ImporterMesh::set_surface_material(int p_surface, const Ref<Material> &p_material) {
	ERR_FAIL_INDEX(p_surface, surfaces.size());
	surfaces.write[p_surface].material = p_material;
	mesh.unref();
}

#define VERTEX_SKIN_FUNC(bone_count, vert_idx, read_array, write_array, transform_array, bone_array, weight_array) \
	Vector3 transformed_vert;                                                                                      \
	for (unsigned int weight_idx = 0; weight_idx < bone_count; weight_idx++) {                                     \
		int bone_idx = bone_array[vert_idx * bone_count + weight_idx];                                             \
		float w = weight_array[vert_idx * bone_count + weight_idx];                                                \
		if (w < FLT_EPSILON) {                                                                                     \
			continue;                                                                                              \
		}                                                                                                          \
		ERR_FAIL_INDEX(bone_idx, static_cast<int>(transform_array.size()));                                        \
		transformed_vert += transform_array[bone_idx].xform(read_array[vert_idx]) * w;                             \
	}                                                                                                              \
	write_array[vert_idx] = transformed_vert;

void ImporterMesh::generate_lods(float p_normal_merge_angle, float p_normal_split_angle, Array p_bone_transform_array) {
	if (!SurfaceTool::simplify_scale_func) {
		return;
	}
	if (!SurfaceTool::simplify_with_attrib_func) {
		return;
	}
	if (!SurfaceTool::optimize_vertex_cache_func) {
		return;
	}

	LocalVector<Transform3D> bone_transform_vector;
	for (int i = 0; i < p_bone_transform_array.size(); i++) {
		ERR_FAIL_COND(p_bone_transform_array[i].get_type() != Variant::TRANSFORM3D);
		bone_transform_vector.push_back(p_bone_transform_array[i]);
	}

	for (int i = 0; i < surfaces.size(); i++) {
		if (surfaces[i].primitive != Mesh::PRIMITIVE_TRIANGLES) {
			continue;
		}

		surfaces.write[i].lods.clear();
		Vector<Vector3> vertices = surfaces[i].arrays[RS::ARRAY_VERTEX];
		PackedInt32Array indices = surfaces[i].arrays[RS::ARRAY_INDEX];
		Vector<Vector3> normals = surfaces[i].arrays[RS::ARRAY_NORMAL];
		Vector<Vector2> uvs = surfaces[i].arrays[RS::ARRAY_TEX_UV];
		Vector<Vector2> uv2s = surfaces[i].arrays[RS::ARRAY_TEX_UV2];
		Vector<int> bones = surfaces[i].arrays[RS::ARRAY_BONES];
		Vector<float> weights = surfaces[i].arrays[RS::ARRAY_WEIGHTS];

		unsigned int index_count = indices.size();
		unsigned int vertex_count = vertices.size();

		if (index_count == 0) {
			continue; //no lods if no indices
		}

		const Vector3 *vertices_ptr = vertices.ptr();
		const int *indices_ptr = indices.ptr();

		if (normals.is_empty()) {
			normals.resize(index_count);
			Vector3 *n_ptr = normals.ptrw();
			for (unsigned int j = 0; j < index_count; j += 3) {
				const Vector3 &v0 = vertices_ptr[indices_ptr[j + 0]];
				const Vector3 &v1 = vertices_ptr[indices_ptr[j + 1]];
				const Vector3 &v2 = vertices_ptr[indices_ptr[j + 2]];
				Vector3 n = vec3_cross(v0 - v2, v0 - v1).normalized();
				n_ptr[j + 0] = n;
				n_ptr[j + 1] = n;
				n_ptr[j + 2] = n;
			}
		}

		if (bones.size() > 0 && weights.size() && bone_transform_vector.size() > 0) {
			Vector3 *vertices_ptrw = vertices.ptrw();

			// Apply bone transforms to regular surface.
			unsigned int bone_weight_length = surfaces[i].flags & Mesh::ARRAY_FLAG_USE_8_BONE_WEIGHTS ? 8 : 4;

			const int *bo = bones.ptr();
			const float *we = weights.ptr();

			for (unsigned int j = 0; j < vertex_count; j++) {
				VERTEX_SKIN_FUNC(bone_weight_length, j, vertices_ptr, vertices_ptrw, bone_transform_vector, bo, we)
			}

			vertices_ptr = vertices.ptr();
		}

		float normal_merge_threshold = Math::cos(Math::deg_to_rad(p_normal_merge_angle));
		float normal_pre_split_threshold = Math::cos(Math::deg_to_rad(MIN(180.0f, p_normal_split_angle * 2.0f)));
		float normal_split_threshold = Math::cos(Math::deg_to_rad(p_normal_split_angle));
		const Vector3 *normals_ptr = normals.ptr();

		HashMap<Vector3, LocalVector<Pair<int, int>>> unique_vertices;

		LocalVector<int> vertex_remap;
		LocalVector<int> vertex_inverse_remap;
		LocalVector<Vector3> merged_vertices;
		LocalVector<Vector3> merged_normals;
		LocalVector<int> merged_normals_counts;
		const Vector2 *uvs_ptr = uvs.ptr();
		const Vector2 *uv2s_ptr = uv2s.ptr();

		for (unsigned int j = 0; j < vertex_count; j++) {
			const Vector3 &v = vertices_ptr[j];
			const Vector3 &n = normals_ptr[j];

			HashMap<Vector3, LocalVector<Pair<int, int>>>::Iterator E = unique_vertices.find(v);

			if (E) {
				const LocalVector<Pair<int, int>> &close_verts = E->value;

				bool found = false;
				for (const Pair<int, int> &idx : close_verts) {
					bool is_uvs_close = (!uvs_ptr || uvs_ptr[j].distance_squared_to(uvs_ptr[idx.second]) < CMP_EPSILON2);
					bool is_uv2s_close = (!uv2s_ptr || uv2s_ptr[j].distance_squared_to(uv2s_ptr[idx.second]) < CMP_EPSILON2);
					ERR_FAIL_INDEX(idx.second, normals.size());
					bool is_normals_close = normals[idx.second].dot(n) > normal_merge_threshold;
					if (is_uvs_close && is_uv2s_close && is_normals_close) {
						vertex_remap.push_back(idx.first);
						merged_normals[idx.first] += normals[idx.second];
						merged_normals_counts[idx.first]++;
						found = true;
						break;
					}
				}

				if (!found) {
					int vcount = merged_vertices.size();
					unique_vertices[v].push_back(Pair<int, int>(vcount, j));
					vertex_inverse_remap.push_back(j);
					merged_vertices.push_back(v);
					vertex_remap.push_back(vcount);
					merged_normals.push_back(normals_ptr[j]);
					merged_normals_counts.push_back(1);
				}
			} else {
				int vcount = merged_vertices.size();
				unique_vertices[v] = LocalVector<Pair<int, int>>();
				unique_vertices[v].push_back(Pair<int, int>(vcount, j));
				vertex_inverse_remap.push_back(j);
				merged_vertices.push_back(v);
				vertex_remap.push_back(vcount);
				merged_normals.push_back(normals_ptr[j]);
				merged_normals_counts.push_back(1);
			}
		}

		LocalVector<int> merged_indices;
		merged_indices.resize(index_count);
		for (unsigned int j = 0; j < index_count; j++) {
			merged_indices[j] = vertex_remap[indices[j]];
		}

		unsigned int merged_vertex_count = merged_vertices.size();
		const Vector3 *merged_vertices_ptr = merged_vertices.ptr();
		const int32_t *merged_indices_ptr = merged_indices.ptr();

		{
			const int *counts_ptr = merged_normals_counts.ptr();
			Vector3 *merged_normals_ptrw = merged_normals.ptr();
			for (unsigned int j = 0; j < merged_vertex_count; j++) {
				merged_normals_ptrw[j] /= counts_ptr[j];
			}
		}

		LocalVector<float> normal_weights;
		normal_weights.resize(merged_vertex_count);
		for (unsigned int j = 0; j < merged_vertex_count; j++) {
			normal_weights[j] = 2.0; // Give some weight to normal preservation, may be worth exposing as an import setting
		}

		Vector<float> merged_vertices_f32 = vector3_to_float32_array(merged_vertices_ptr, merged_vertex_count);
		float scale = SurfaceTool::simplify_scale_func(merged_vertices_f32.ptr(), merged_vertex_count, sizeof(float) * 3);

		unsigned int index_target = 12; // Start with the smallest target, 4 triangles
		unsigned int last_index_count = 0;

		int split_vertex_count = vertex_count;
		LocalVector<Vector3> split_vertex_normals;
		LocalVector<int> split_vertex_indices;
		split_vertex_normals.reserve(index_count / 3);
		split_vertex_indices.reserve(index_count / 3);

		RandomPCG pcg;
		pcg.seed(123456789); // Keep seed constant across imports

		Ref<StaticRaycaster> raycaster = StaticRaycaster::create();
		if (raycaster.is_valid()) {
			raycaster->add_mesh(vertices, indices, 0);
			raycaster->commit();
		}

		const float max_mesh_error = FLT_MAX; // We don't want to limit by error, just by index target
		float mesh_error = 0.0f;

		while (index_target < index_count) {
			PackedInt32Array new_indices;
			new_indices.resize(index_count);

			Vector<float> merged_normals_f32 = vector3_to_float32_array(merged_normals.ptr(), merged_normals.size());
			const int simplify_options = SurfaceTool::SIMPLIFY_LOCK_BORDER;

			size_t new_index_count = SurfaceTool::simplify_with_attrib_func(
					(unsigned int *)new_indices.ptrw(),
					(const uint32_t *)merged_indices_ptr, index_count,
					merged_vertices_f32.ptr(), merged_vertex_count,
					sizeof(float) * 3, // Vertex stride
					index_target,
					max_mesh_error,
					simplify_options,
					&mesh_error,
					merged_normals_f32.ptr(),
					normal_weights.ptr(), 3);

			if (new_index_count < last_index_count * 1.5f) {
				index_target = index_target * 1.5f;
				continue;
			}

			if (new_index_count == 0 || (new_index_count >= (index_count * 0.75f))) {
				break;
			}
			if (new_index_count > 5000000) {
				// This limit theoretically shouldn't be needed, but it's here
				// as an ad-hoc fix to prevent a crash with complex meshes.
				// The crash still happens with limit of 6000000, but 5000000 works.
				// In the future, identify what's causing that crash and fix it.
				WARN_PRINT("Mesh LOD generation failed for mesh " + get_name() + " surface " + itos(i) + ", mesh is too complex. Some automatic LODs were not generated.");
				break;
			}

			new_indices.resize(new_index_count);

			LocalVector<LocalVector<int>> vertex_corners;
			vertex_corners.resize(vertex_count);
			{
				int *ptrw = new_indices.ptrw();
				for (unsigned int j = 0; j < new_index_count; j++) {
					const int &remapped = vertex_inverse_remap[ptrw[j]];
					vertex_corners[remapped].push_back(j);
					ptrw[j] = remapped;
				}
			}

			if (raycaster.is_valid()) {
				float error_factor = 1.0f / (scale * MAX(mesh_error, 0.15));
				const float ray_bias = 0.05;
				float ray_length = ray_bias + mesh_error * scale * 3.0f;

				Vector<StaticRaycaster::Ray> rays;
				LocalVector<Vector2> ray_uvs;

				int32_t *new_indices_ptr = new_indices.ptrw();

				int current_ray_count = 0;
				for (unsigned int j = 0; j < new_index_count; j += 3) {
					const Vector3 &v0 = vertices_ptr[new_indices_ptr[j + 0]];
					const Vector3 &v1 = vertices_ptr[new_indices_ptr[j + 1]];
					const Vector3 &v2 = vertices_ptr[new_indices_ptr[j + 2]];
					Vector3 face_normal = vec3_cross(v0 - v2, v0 - v1);
					float face_area = face_normal.length(); // Actually twice the face area, since it's the same error_factor on all faces, we don't care
					if (!Math::is_finite(face_area) || face_area == 0) {
						WARN_PRINT_ONCE("Ignoring face with non-finite normal in LOD generation.");
						continue;
					}

					Vector3 dir = face_normal / face_area;
					int ray_count = CLAMP(5.0 * face_area * error_factor, 16, 64);

					rays.resize(current_ray_count + ray_count);
					StaticRaycaster::Ray *rays_ptr = rays.ptrw();

					ray_uvs.resize(current_ray_count + ray_count);
					Vector2 *ray_uvs_ptr = ray_uvs.ptr();

					for (int k = 0; k < ray_count; k++) {
						float u = pcg.randf();
						float v = pcg.randf();

						if (u + v >= 1.0f) {
							u = 1.0f - u;
							v = 1.0f - v;
						}

						u = 0.9f * u + 0.05f / 3.0f; // Give barycentric coordinates some padding, we don't want to sample right on the edge
						v = 0.9f * v + 0.05f / 3.0f; // v = (v - one_third) * 0.95f + one_third;
						float w = 1.0f - u - v;

						Vector3 org = v0 * w + v1 * u + v2 * v;
						org -= dir * ray_bias;
						rays_ptr[current_ray_count + k] = StaticRaycaster::Ray(org, dir, 0.0f, ray_length);
						rays_ptr[current_ray_count + k].id = j / 3;
						ray_uvs_ptr[current_ray_count + k] = Vector2(u, v);
					}

					current_ray_count += ray_count;
				}

				raycaster->intersect(rays);

				LocalVector<Vector3> ray_normals;
				LocalVector<real_t> ray_normal_weights;

				ray_normals.resize(new_index_count);
				ray_normal_weights.resize(new_index_count);

				for (unsigned int j = 0; j < new_index_count; j++) {
					ray_normal_weights[j] = 0.0f;
				}

				const StaticRaycaster::Ray *rp = rays.ptr();
				for (int j = 0; j < rays.size(); j++) {
					if (rp[j].geomID != 0) { // Ray missed
						continue;
					}

					if (rp[j].normal.normalized().dot(rp[j].dir) > 0.0f) { // Hit a back face.
						continue;
					}

					const float &u = rp[j].u;
					const float &v = rp[j].v;
					const float w = 1.0f - u - v;

					const unsigned int &hit_tri_id = rp[j].primID;
					const unsigned int &orig_tri_id = rp[j].id;

					const Vector3 &n0 = normals_ptr[indices_ptr[hit_tri_id * 3 + 0]];
					const Vector3 &n1 = normals_ptr[indices_ptr[hit_tri_id * 3 + 1]];
					const Vector3 &n2 = normals_ptr[indices_ptr[hit_tri_id * 3 + 2]];
					Vector3 normal = n0 * w + n1 * u + n2 * v;

					Vector2 orig_uv = ray_uvs[j];
					const real_t orig_bary[3] = { 1.0f - orig_uv.x - orig_uv.y, orig_uv.x, orig_uv.y };
					for (int k = 0; k < 3; k++) {
						int idx = orig_tri_id * 3 + k;
						real_t weight = orig_bary[k];
						ray_normals[idx] += normal * weight;
						ray_normal_weights[idx] += weight;
					}
				}

				for (unsigned int j = 0; j < new_index_count; j++) {
					if (ray_normal_weights[j] < 1.0f) { // Not enough data, the new normal would be just a bad guess
						ray_normals[j] = Vector3();
					} else {
						ray_normals[j] /= ray_normal_weights[j];
					}
				}

				LocalVector<LocalVector<int>> normal_group_indices;
				LocalVector<Vector3> normal_group_averages;
				normal_group_indices.reserve(24);
				normal_group_averages.reserve(24);

				for (unsigned int j = 0; j < vertex_count; j++) {
					const LocalVector<int> &corners = vertex_corners[j];
					const Vector3 &vertex_normal = normals_ptr[j];

					for (const int &corner_idx : corners) {
						const Vector3 &ray_normal = ray_normals[corner_idx];

						if (ray_normal.length_squared() < CMP_EPSILON2) {
							continue;
						}

						bool found = false;
						for (unsigned int l = 0; l < normal_group_indices.size(); l++) {
							LocalVector<int> &group_indices = normal_group_indices[l];
							Vector3 n = normal_group_averages[l] / group_indices.size();
							if (n.dot(ray_normal) > normal_pre_split_threshold) {
								found = true;
								group_indices.push_back(corner_idx);
								normal_group_averages[l] += ray_normal;
								break;
							}
						}

						if (!found) {
							normal_group_indices.push_back({ corner_idx });
							normal_group_averages.push_back(ray_normal);
						}
					}

					for (unsigned int k = 0; k < normal_group_indices.size(); k++) {
						LocalVector<int> &group_indices = normal_group_indices[k];
						Vector3 n = normal_group_averages[k] / group_indices.size();

						if (vertex_normal.dot(n) < normal_split_threshold) {
							split_vertex_indices.push_back(j);
							split_vertex_normals.push_back(n);
							int new_idx = split_vertex_count++;
							for (const int &index : group_indices) {
								new_indices_ptr[index] = new_idx;
							}
						}
					}

					normal_group_indices.clear();
					normal_group_averages.clear();
				}
			}

			Surface::LOD lod;
			lod.distance = MAX(mesh_error * scale, CMP_EPSILON2);
			lod.indices = new_indices;
			surfaces.write[i].lods.push_back(lod);
			index_target = MAX(new_index_count, index_target) * 2;
			last_index_count = new_index_count;

			if (mesh_error == 0.0f) {
				break;
			}
		}

		surfaces.write[i].split_normals(split_vertex_indices, split_vertex_normals);
		surfaces.write[i].lods.sort_custom<Surface::LODComparator>();

		for (int j = 0; j < surfaces.write[i].lods.size(); j++) {
			Surface::LOD &lod = surfaces.write[i].lods.write[j];
			unsigned int *lod_indices_ptr = (unsigned int *)lod.indices.ptrw();
			SurfaceTool::optimize_vertex_cache_func(lod_indices_ptr, lod_indices_ptr, lod.indices.size(), split_vertex_count);
		}
	}
}

bool ImporterMesh::has_mesh() const {
	return mesh.is_valid();
}

Ref<ArrayMesh> ImporterMesh::get_mesh(const Ref<ArrayMesh> &p_base) {
	ERR_FAIL_COND_V(surfaces.size() == 0, Ref<ArrayMesh>());

	if (mesh.is_null()) {
		if (p_base.is_valid()) {
			mesh = p_base;
		}
		if (mesh.is_null()) {
			mesh.instantiate();
		}
		mesh->set_name(get_name());
		if (has_meta("import_id")) {
			mesh->set_meta("import_id", get_meta("import_id"));
		}
		for (int i = 0; i < blend_shapes.size(); i++) {
			mesh->add_blend_shape(blend_shapes[i]);
		}
		mesh->set_blend_shape_mode(blend_shape_mode);
		for (int i = 0; i < surfaces.size(); i++) {
			Array bs_data;
			if (surfaces[i].blend_shape_data.size()) {
				for (int j = 0; j < surfaces[i].blend_shape_data.size(); j++) {
					bs_data.push_back(surfaces[i].blend_shape_data[j].arrays);
				}
			}
			Dictionary lods;
			if (surfaces[i].lods.size()) {
				for (int j = 0; j < surfaces[i].lods.size(); j++) {
					lods[surfaces[i].lods[j].distance] = surfaces[i].lods[j].indices;
				}
			}

			mesh->add_surface_from_arrays(surfaces[i].primitive, surfaces[i].arrays, bs_data, lods, surfaces[i].flags);
			if (surfaces[i].material.is_valid()) {
				mesh->surface_set_material(mesh->get_surface_count() - 1, surfaces[i].material);
			}
			if (!surfaces[i].name.is_empty()) {
				mesh->surface_set_name(mesh->get_surface_count() - 1, surfaces[i].name);
			}
		}

		mesh->set_lightmap_size_hint(lightmap_size_hint);

		if (shadow_mesh.is_valid()) {
			Ref<ArrayMesh> shadow = shadow_mesh->get_mesh();
			mesh->set_shadow_mesh(shadow);
		}
	}

	return mesh;
}

void ImporterMesh::clear() {
	surfaces.clear();
	blend_shapes.clear();
	mesh.unref();
}

void ImporterMesh::create_shadow_mesh() {
	if (shadow_mesh.is_valid()) {
		shadow_mesh.unref();
	}

	//no shadow mesh for blendshapes
	if (blend_shapes.size() > 0) {
		return;
	}
	//no shadow mesh for skeletons
	for (int i = 0; i < surfaces.size(); i++) {
		if (surfaces[i].arrays[RS::ARRAY_BONES].get_type() != Variant::NIL) {
			return;
		}
		if (surfaces[i].arrays[RS::ARRAY_WEIGHTS].get_type() != Variant::NIL) {
			return;
		}
	}

	shadow_mesh.instantiate();

	for (int i = 0; i < surfaces.size(); i++) {
		LocalVector<int> vertex_remap;
		Vector<Vector3> new_vertices;
		Vector<Vector3> vertices = surfaces[i].arrays[RS::ARRAY_VERTEX];
		int vertex_count = vertices.size();
		{
			HashMap<Vector3, int> unique_vertices;
			const Vector3 *vptr = vertices.ptr();
			for (int j = 0; j < vertex_count; j++) {
				const Vector3 &v = vptr[j];

				HashMap<Vector3, int>::Iterator E = unique_vertices.find(v);

				if (E) {
					vertex_remap.push_back(E->value);
				} else {
					int vcount = unique_vertices.size();
					unique_vertices[v] = vcount;
					vertex_remap.push_back(vcount);
					new_vertices.push_back(v);
				}
			}
		}

		Array new_surface;
		new_surface.resize(RS::ARRAY_MAX);
		Dictionary lods;

		//		print_line("original vertex count: " + itos(vertices.size()) + " new vertex count: " + itos(new_vertices.size()));

		new_surface[RS::ARRAY_VERTEX] = new_vertices;

		Vector<int> indices = surfaces[i].arrays[RS::ARRAY_INDEX];
		if (indices.size()) {
			int index_count = indices.size();
			const int *index_rptr = indices.ptr();
			Vector<int> new_indices;
			new_indices.resize(indices.size());
			int *index_wptr = new_indices.ptrw();

			for (int j = 0; j < index_count; j++) {
				int index = index_rptr[j];
				ERR_FAIL_INDEX(index, vertex_count);
				index_wptr[j] = vertex_remap[index];
			}

			new_surface[RS::ARRAY_INDEX] = new_indices;

			// Make sure the same LODs as the full version are used.
			// This makes it more coherent between rendered model and its shadows.
			for (int j = 0; j < surfaces[i].lods.size(); j++) {
				indices = surfaces[i].lods[j].indices;

				index_count = indices.size();
				index_rptr = indices.ptr();
				new_indices.resize(indices.size());
				index_wptr = new_indices.ptrw();

				for (int k = 0; k < index_count; k++) {
					int index = index_rptr[k];
					ERR_FAIL_INDEX(index, vertex_count);
					index_wptr[k] = vertex_remap[index];
				}

				lods[surfaces[i].lods[j].distance] = new_indices;
			}
		}

		shadow_mesh->add_surface(surfaces[i].primitive, new_surface, Array(), lods, Ref<Material>(), surfaces[i].name, surfaces[i].flags);
	}
}

Ref<ImporterMesh> ImporterMesh::get_shadow_mesh() const {
	return shadow_mesh;
}

void ImporterMesh::_set_data(const Dictionary &p_data) {
	clear();
	if (p_data.has("blend_shape_names")) {
		blend_shapes = p_data["blend_shape_names"];
	}
	if (p_data.has("surfaces")) {
		Array surface_arr = p_data["surfaces"];
		for (int i = 0; i < surface_arr.size(); i++) {
			Dictionary s = surface_arr[i];
			ERR_CONTINUE(!s.has("primitive"));
			ERR_CONTINUE(!s.has("arrays"));
			Mesh::PrimitiveType prim = Mesh::PrimitiveType(int(s["primitive"]));
			ERR_CONTINUE(prim >= Mesh::PRIMITIVE_MAX);
			Array arr = s["arrays"];
			Dictionary lods;
			String surf_name;
			if (s.has("name")) {
				surf_name = s["name"];
			}
			if (s.has("lods")) {
				lods = s["lods"];
			}
			Array b_shapes;
			if (s.has("b_shapes")) {
				b_shapes = s["b_shapes"];
			}
			Ref<Material> material;
			if (s.has("material")) {
				material = s["material"];
			}
			uint64_t flags = 0;
			if (s.has("flags")) {
				flags = s["flags"];
			}
			add_surface(prim, arr, b_shapes, lods, material, surf_name, flags);
		}
	}
}
Dictionary ImporterMesh::_get_data() const {
	Dictionary data;
	if (blend_shapes.size()) {
		data["blend_shape_names"] = blend_shapes;
	}
	Array surface_arr;
	for (int i = 0; i < surfaces.size(); i++) {
		Dictionary d;
		d["primitive"] = surfaces[i].primitive;
		d["arrays"] = surfaces[i].arrays;
		if (surfaces[i].blend_shape_data.size()) {
			Array bs_data;
			for (int j = 0; j < surfaces[i].blend_shape_data.size(); j++) {
				bs_data.push_back(surfaces[i].blend_shape_data[j].arrays);
			}
			d["blend_shapes"] = bs_data;
		}
		if (surfaces[i].lods.size()) {
			Dictionary lods;
			for (int j = 0; j < surfaces[i].lods.size(); j++) {
				lods[surfaces[i].lods[j].distance] = surfaces[i].lods[j].indices;
			}
			d["lods"] = lods;
		}

		if (surfaces[i].material.is_valid()) {
			d["material"] = surfaces[i].material;
		}

		if (!surfaces[i].name.is_empty()) {
			d["name"] = surfaces[i].name;
		}

		d["flags"] = surfaces[i].flags;

		surface_arr.push_back(d);
	}
	data["surfaces"] = surface_arr;
	return data;
}

Vector<Face3> ImporterMesh::get_faces() const {
	Vector<Face3> faces;
	for (int i = 0; i < surfaces.size(); i++) {
		if (surfaces[i].primitive == Mesh::PRIMITIVE_TRIANGLES) {
			Vector<Vector3> vertices = surfaces[i].arrays[Mesh::ARRAY_VERTEX];
			Vector<int> indices = surfaces[i].arrays[Mesh::ARRAY_INDEX];
			if (indices.size()) {
				for (int j = 0; j < indices.size(); j += 3) {
					Face3 f;
					f.vertex[0] = vertices[indices[j + 0]];
					f.vertex[1] = vertices[indices[j + 1]];
					f.vertex[2] = vertices[indices[j + 2]];
					faces.push_back(f);
				}
			} else {
				for (int j = 0; j < vertices.size(); j += 3) {
					Face3 f;
					f.vertex[0] = vertices[j + 0];
					f.vertex[1] = vertices[j + 1];
					f.vertex[2] = vertices[j + 2];
					faces.push_back(f);
				}
			}
		}
	}

	return faces;
}

Vector<Ref<Shape3D>> ImporterMesh::convex_decompose(const Ref<MeshConvexDecompositionSettings> &p_settings) const {
	ERR_FAIL_NULL_V(Mesh::convex_decomposition_function, Vector<Ref<Shape3D>>());

	const Vector<Face3> faces = get_faces();
	int face_count = faces.size();

	Vector<Vector3> vertices;
	uint32_t vertex_count = 0;
	vertices.resize(face_count * 3);
	Vector<uint32_t> indices;
	indices.resize(face_count * 3);
	{
		HashMap<Vector3, uint32_t> vertex_map;
		Vector3 *vertex_w = vertices.ptrw();
		uint32_t *index_w = indices.ptrw();
		for (int i = 0; i < face_count; i++) {
			for (int j = 0; j < 3; j++) {
				const Vector3 &vertex = faces[i].vertex[j];
				HashMap<Vector3, uint32_t>::Iterator found_vertex = vertex_map.find(vertex);
				uint32_t index;
				if (found_vertex) {
					index = found_vertex->value;
				} else {
					index = ++vertex_count;
					vertex_map[vertex] = index;
					vertex_w[index] = vertex;
				}
				index_w[i * 3 + j] = index;
			}
		}
	}
	vertices.resize(vertex_count);

	Vector<Vector<Vector3>> decomposed = Mesh::convex_decomposition_function((real_t *)vertices.ptr(), vertex_count, indices.ptr(), face_count, p_settings, nullptr);

	Vector<Ref<Shape3D>> ret;

	for (int i = 0; i < decomposed.size(); i++) {
		Ref<ConvexPolygonShape3D> shape;
		shape.instantiate();
		shape->set_points(decomposed[i]);
		ret.push_back(shape);
	}

	return ret;
}

Ref<ConvexPolygonShape3D> ImporterMesh::create_convex_shape(bool p_clean, bool p_simplify) const {
	if (p_simplify) {
		Ref<MeshConvexDecompositionSettings> settings;
		settings.instantiate();
		settings->set_max_convex_hulls(1);
		Vector<Ref<Shape3D>> decomposed = convex_decompose(settings);
		if (decomposed.size() == 1) {
			return decomposed[0];
		} else {
			ERR_PRINT("Convex shape simplification failed, falling back to simpler process.");
		}
	}

	Vector<Vector3> vertices;
	for (int i = 0; i < get_surface_count(); i++) {
		Array a = get_surface_arrays(i);
		ERR_FAIL_COND_V(a.is_empty(), Ref<ConvexPolygonShape3D>());
		Vector<Vector3> v = a[Mesh::ARRAY_VERTEX];
		vertices.append_array(v);
	}

	Ref<ConvexPolygonShape3D> shape = memnew(ConvexPolygonShape3D);

	if (p_clean) {
		Geometry3D::MeshData md;
		Error err = ConvexHullComputer::convex_hull(vertices, md);
		if (err == OK) {
			shape->set_points(md.vertices);
			return shape;
		} else {
			ERR_PRINT("Convex shape cleaning failed, falling back to simpler process.");
		}
	}

	shape->set_points(vertices);
	return shape;
}

Ref<ConcavePolygonShape3D> ImporterMesh::create_trimesh_shape() const {
	Vector<Face3> faces = get_faces();
	if (faces.size() == 0) {
		return Ref<ConcavePolygonShape3D>();
	}

	Vector<Vector3> face_points;
	face_points.resize(faces.size() * 3);

	for (int i = 0; i < face_points.size(); i += 3) {
		Face3 f = faces.get(i / 3);
		face_points.set(i, f.vertex[0]);
		face_points.set(i + 1, f.vertex[1]);
		face_points.set(i + 2, f.vertex[2]);
	}

	Ref<ConcavePolygonShape3D> shape = memnew(ConcavePolygonShape3D);
	shape->set_faces(face_points);
	return shape;
}

Ref<NavigationMesh> ImporterMesh::create_navigation_mesh() {
	Vector<Face3> faces = get_faces();
	if (faces.size() == 0) {
		return Ref<NavigationMesh>();
	}

	HashMap<Vector3, int> unique_vertices;
	LocalVector<int> face_indices;

	for (int i = 0; i < faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			Vector3 v = faces[i].vertex[j];
			int idx;
			if (unique_vertices.has(v)) {
				idx = unique_vertices[v];
			} else {
				idx = unique_vertices.size();
				unique_vertices[v] = idx;
			}
			face_indices.push_back(idx);
		}
	}

	Vector<Vector3> vertices;
	vertices.resize(unique_vertices.size());
	for (const KeyValue<Vector3, int> &E : unique_vertices) {
		vertices.write[E.value] = E.key;
	}

	Ref<NavigationMesh> nm;
	nm.instantiate();
	nm->set_vertices(vertices);

	Vector<int> v3;
	v3.resize(3);
	for (uint32_t i = 0; i < face_indices.size(); i += 3) {
		v3.write[0] = face_indices[i + 0];
		v3.write[1] = face_indices[i + 1];
		v3.write[2] = face_indices[i + 2];
		nm->add_polygon(v3);
	}

	return nm;
}

extern bool (*array_mesh_lightmap_unwrap_callback)(float p_texel_size, const float *p_vertices, const float *p_normals, int p_vertex_count, const int *p_indices, int p_index_count, const uint8_t *p_cache_data, bool *r_use_cache, uint8_t **r_mesh_cache, int *r_mesh_cache_size, float **r_uv, int **r_vertex, int *r_vertex_count, int **r_index, int *r_index_count, int *r_size_hint_x, int *r_size_hint_y);

struct EditorSceneFormatImporterMeshLightmapSurface {
	Ref<Material> material;
	LocalVector<SurfaceTool::Vertex> vertices;
	Mesh::PrimitiveType primitive = Mesh::PrimitiveType::PRIMITIVE_MAX;
	uint64_t format = 0;
	String name;
};

static const uint32_t custom_shift[RS::ARRAY_CUSTOM_COUNT] = { Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT, Mesh::ARRAY_FORMAT_CUSTOM1_SHIFT, Mesh::ARRAY_FORMAT_CUSTOM2_SHIFT, Mesh::ARRAY_FORMAT_CUSTOM3_SHIFT };

Error ImporterMesh::lightmap_unwrap_cached(const Transform3D &p_base_transform, float p_texel_size, const Vector<uint8_t> &p_src_cache, Vector<uint8_t> &r_dst_cache) {
	ERR_FAIL_NULL_V(array_mesh_lightmap_unwrap_callback, ERR_UNCONFIGURED);
	ERR_FAIL_COND_V_MSG(blend_shapes.size() != 0, ERR_UNAVAILABLE, "Can't unwrap mesh with blend shapes.");

	LocalVector<float> vertices;
	LocalVector<float> normals;
	LocalVector<int> indices;
	LocalVector<float> uv;
	LocalVector<Pair<int, int>> uv_indices;

	Vector<EditorSceneFormatImporterMeshLightmapSurface> lightmap_surfaces;

	// Keep only the scale
	Basis basis = p_base_transform.get_basis();
	Vector3 scale = Vector3(basis.get_column(0).length(), basis.get_column(1).length(), basis.get_column(2).length());

	Transform3D transform;
	transform.scale(scale);

	Basis normal_basis = transform.basis.inverse().transposed();

	for (int i = 0; i < get_surface_count(); i++) {
		EditorSceneFormatImporterMeshLightmapSurface s;
		s.primitive = get_surface_primitive_type(i);

		ERR_FAIL_COND_V_MSG(s.primitive != Mesh::PRIMITIVE_TRIANGLES, ERR_UNAVAILABLE, "Only triangles are supported for lightmap unwrap.");
		Array arrays = get_surface_arrays(i);
		s.material = get_surface_material(i);
		s.name = get_surface_name(i);

		SurfaceTool::create_vertex_array_from_triangle_arrays(arrays, s.vertices, &s.format);

		PackedVector3Array rvertices = arrays[Mesh::ARRAY_VERTEX];
		int vc = rvertices.size();

		PackedVector3Array rnormals = arrays[Mesh::ARRAY_NORMAL];

		if (!rnormals.size()) {
			continue;
		}

		int vertex_ofs = vertices.size() / 3;

		vertices.resize((vertex_ofs + vc) * 3);
		normals.resize((vertex_ofs + vc) * 3);
		uv_indices.resize(vertex_ofs + vc);

		for (int j = 0; j < vc; j++) {
			Vector3 v = transform.xform(rvertices[j]);
			Vector3 n = normal_basis.xform(rnormals[j]).normalized();

			vertices[(j + vertex_ofs) * 3 + 0] = v.x;
			vertices[(j + vertex_ofs) * 3 + 1] = v.y;
			vertices[(j + vertex_ofs) * 3 + 2] = v.z;
			normals[(j + vertex_ofs) * 3 + 0] = n.x;
			normals[(j + vertex_ofs) * 3 + 1] = n.y;
			normals[(j + vertex_ofs) * 3 + 2] = n.z;
			uv_indices[j + vertex_ofs] = Pair<int, int>(i, j);
		}

		PackedInt32Array rindices = arrays[Mesh::ARRAY_INDEX];
		int ic = rindices.size();

		float eps = 1.19209290e-7F; // Taken from xatlas.h
		if (ic == 0) {
			for (int j = 0; j < vc / 3; j++) {
				Vector3 p0 = transform.xform(rvertices[j * 3 + 0]);
				Vector3 p1 = transform.xform(rvertices[j * 3 + 1]);
				Vector3 p2 = transform.xform(rvertices[j * 3 + 2]);

				if ((p0 - p1).length_squared() < eps || (p1 - p2).length_squared() < eps || (p2 - p0).length_squared() < eps) {
					continue;
				}

				indices.push_back(vertex_ofs + j * 3 + 0);
				indices.push_back(vertex_ofs + j * 3 + 1);
				indices.push_back(vertex_ofs + j * 3 + 2);
			}

		} else {
			for (int j = 0; j < ic / 3; j++) {
				ERR_FAIL_INDEX_V(rindices[j * 3 + 0], rvertices.size(), ERR_INVALID_DATA);
				ERR_FAIL_INDEX_V(rindices[j * 3 + 1], rvertices.size(), ERR_INVALID_DATA);
				ERR_FAIL_INDEX_V(rindices[j * 3 + 2], rvertices.size(), ERR_INVALID_DATA);
				Vector3 p0 = transform.xform(rvertices[rindices[j * 3 + 0]]);
				Vector3 p1 = transform.xform(rvertices[rindices[j * 3 + 1]]);
				Vector3 p2 = transform.xform(rvertices[rindices[j * 3 + 2]]);

				if ((p0 - p1).length_squared() < eps || (p1 - p2).length_squared() < eps || (p2 - p0).length_squared() < eps) {
					continue;
				}

				indices.push_back(vertex_ofs + rindices[j * 3 + 0]);
				indices.push_back(vertex_ofs + rindices[j * 3 + 1]);
				indices.push_back(vertex_ofs + rindices[j * 3 + 2]);
			}
		}

		lightmap_surfaces.push_back(s);
	}

	//unwrap

	bool use_cache = true; // Used to request cache generation and to know if cache was used
	uint8_t *gen_cache;
	int gen_cache_size;
	float *gen_uvs;
	int *gen_vertices;
	int *gen_indices;
	int gen_vertex_count;
	int gen_index_count;
	int size_x;
	int size_y;

	bool ok = array_mesh_lightmap_unwrap_callback(p_texel_size, vertices.ptr(), normals.ptr(), vertices.size() / 3, indices.ptr(), indices.size(), p_src_cache.ptr(), &use_cache, &gen_cache, &gen_cache_size, &gen_uvs, &gen_vertices, &gen_vertex_count, &gen_indices, &gen_index_count, &size_x, &size_y);

	if (!ok) {
		return ERR_CANT_CREATE;
	}

	//create surfacetools for each surface..
	LocalVector<Ref<SurfaceTool>> surfaces_tools;

	for (int i = 0; i < lightmap_surfaces.size(); i++) {
		Ref<SurfaceTool> st;
		st.instantiate();
		st->set_skin_weight_count((lightmap_surfaces[i].format & Mesh::ARRAY_FLAG_USE_8_BONE_WEIGHTS) ? SurfaceTool::SKIN_8_WEIGHTS : SurfaceTool::SKIN_4_WEIGHTS);
		st->begin(Mesh::PRIMITIVE_TRIANGLES);
		st->set_material(lightmap_surfaces[i].material);
		st->set_meta("name", lightmap_surfaces[i].name);

		for (int custom_i = 0; custom_i < RS::ARRAY_CUSTOM_COUNT; custom_i++) {
			st->set_custom_format(custom_i, (SurfaceTool::CustomFormat)((lightmap_surfaces[i].format >> custom_shift[custom_i]) & RS::ARRAY_FORMAT_CUSTOM_MASK));
		}
		surfaces_tools.push_back(st); //stay there
	}

	//remove surfaces
	clear();

	print_verbose("Mesh: Gen indices: " + itos(gen_index_count));

	//go through all indices
	for (int i = 0; i < gen_index_count; i += 3) {
		ERR_FAIL_INDEX_V(gen_vertices[gen_indices[i + 0]], (int)uv_indices.size(), ERR_BUG);
		ERR_FAIL_INDEX_V(gen_vertices[gen_indices[i + 1]], (int)uv_indices.size(), ERR_BUG);
		ERR_FAIL_INDEX_V(gen_vertices[gen_indices[i + 2]], (int)uv_indices.size(), ERR_BUG);

		ERR_FAIL_COND_V(uv_indices[gen_vertices[gen_indices[i + 0]]].first != uv_indices[gen_vertices[gen_indices[i + 1]]].first || uv_indices[gen_vertices[gen_indices[i + 0]]].first != uv_indices[gen_vertices[gen_indices[i + 2]]].first, ERR_BUG);

		int surface = uv_indices[gen_vertices[gen_indices[i + 0]]].first;

		for (int j = 0; j < 3; j++) {
			SurfaceTool::Vertex v = lightmap_surfaces[surface].vertices[uv_indices[gen_vertices[gen_indices[i + j]]].second];

			if (lightmap_surfaces[surface].format & Mesh::ARRAY_FORMAT_COLOR) {
				surfaces_tools[surface]->set_color(v.color);
			}
			if (lightmap_surfaces[surface].format & Mesh::ARRAY_FORMAT_TEX_UV) {
				surfaces_tools[surface]->set_uv(v.uv);
			}
			if (lightmap_surfaces[surface].format & Mesh::ARRAY_FORMAT_NORMAL) {
				surfaces_tools[surface]->set_normal(v.normal);
			}
			if (lightmap_surfaces[surface].format & Mesh::ARRAY_FORMAT_TANGENT) {
				Plane t;
				t.normal = v.tangent;
				t.d = v.binormal.dot(v.normal.cross(v.tangent)) < 0 ? -1 : 1;
				surfaces_tools[surface]->set_tangent(t);
			}
			if (lightmap_surfaces[surface].format & Mesh::ARRAY_FORMAT_BONES) {
				surfaces_tools[surface]->set_bones(v.bones);
			}
			if (lightmap_surfaces[surface].format & Mesh::ARRAY_FORMAT_WEIGHTS) {
				surfaces_tools[surface]->set_weights(v.weights);
			}
			for (int custom_i = 0; custom_i < RS::ARRAY_CUSTOM_COUNT; custom_i++) {
				if ((lightmap_surfaces[surface].format >> custom_shift[custom_i]) & RS::ARRAY_FORMAT_CUSTOM_MASK) {
					surfaces_tools[surface]->set_custom(custom_i, v.custom[custom_i]);
				}
			}

			Vector2 uv2(gen_uvs[gen_indices[i + j] * 2 + 0], gen_uvs[gen_indices[i + j] * 2 + 1]);
			surfaces_tools[surface]->set_uv2(uv2);

			surfaces_tools[surface]->add_vertex(v.vertex);
		}
	}

	//generate surfaces
	for (int i = 0; i < lightmap_surfaces.size(); i++) {
		Ref<SurfaceTool> &tool = surfaces_tools[i];
		tool->index();
		Array arrays = tool->commit_to_arrays();

		uint64_t format = lightmap_surfaces[i].format;
		if (tool->get_skin_weight_count() == SurfaceTool::SKIN_8_WEIGHTS) {
			format |= RS::ARRAY_FLAG_USE_8_BONE_WEIGHTS;
		} else {
			format &= ~RS::ARRAY_FLAG_USE_8_BONE_WEIGHTS;
		}

		add_surface(tool->get_primitive_type(), arrays, Array(), Dictionary(), tool->get_material(), tool->get_meta("name"), format);
	}

	set_lightmap_size_hint(Size2(size_x, size_y));

	if (gen_cache_size > 0) {
		r_dst_cache.resize(gen_cache_size);
		memcpy(r_dst_cache.ptrw(), gen_cache, gen_cache_size);
		memfree(gen_cache);
	}

	if (!use_cache) {
		// Cache was not used, free the buffers
		memfree(gen_vertices);
		memfree(gen_indices);
		memfree(gen_uvs);
	}

	return OK;
}

void ImporterMesh::set_lightmap_size_hint(const Size2i &p_size) {
	lightmap_size_hint = p_size;
}

Size2i ImporterMesh::get_lightmap_size_hint() const {
	return lightmap_size_hint;
}

void ImporterMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_blend_shape", "name"), &ImporterMesh::add_blend_shape);
	ClassDB::bind_method(D_METHOD("get_blend_shape_count"), &ImporterMesh::get_blend_shape_count);
	ClassDB::bind_method(D_METHOD("get_blend_shape_name", "blend_shape_idx"), &ImporterMesh::get_blend_shape_name);

	ClassDB::bind_method(D_METHOD("set_blend_shape_mode", "mode"), &ImporterMesh::set_blend_shape_mode);
	ClassDB::bind_method(D_METHOD("get_blend_shape_mode"), &ImporterMesh::get_blend_shape_mode);

	ClassDB::bind_method(D_METHOD("add_surface", "primitive", "arrays", "blend_shapes", "lods", "material", "name", "flags"), &ImporterMesh::add_surface, DEFVAL(TypedArray<Array>()), DEFVAL(Dictionary()), DEFVAL(Ref<Material>()), DEFVAL(String()), DEFVAL(0));

	ClassDB::bind_method(D_METHOD("get_surface_count"), &ImporterMesh::get_surface_count);
	ClassDB::bind_method(D_METHOD("get_surface_primitive_type", "surface_idx"), &ImporterMesh::get_surface_primitive_type);
	ClassDB::bind_method(D_METHOD("get_surface_name", "surface_idx"), &ImporterMesh::get_surface_name);
	ClassDB::bind_method(D_METHOD("get_surface_arrays", "surface_idx"), &ImporterMesh::get_surface_arrays);
	ClassDB::bind_method(D_METHOD("get_surface_blend_shape_arrays", "surface_idx", "blend_shape_idx"), &ImporterMesh::get_surface_blend_shape_arrays);
	ClassDB::bind_method(D_METHOD("get_surface_lod_count", "surface_idx"), &ImporterMesh::get_surface_lod_count);
	ClassDB::bind_method(D_METHOD("get_surface_lod_size", "surface_idx", "lod_idx"), &ImporterMesh::get_surface_lod_size);
	ClassDB::bind_method(D_METHOD("get_surface_lod_indices", "surface_idx", "lod_idx"), &ImporterMesh::get_surface_lod_indices);
	ClassDB::bind_method(D_METHOD("get_surface_material", "surface_idx"), &ImporterMesh::get_surface_material);
	ClassDB::bind_method(D_METHOD("get_surface_format", "surface_idx"), &ImporterMesh::get_surface_format);

	ClassDB::bind_method(D_METHOD("set_surface_name", "surface_idx", "name"), &ImporterMesh::set_surface_name);
	ClassDB::bind_method(D_METHOD("set_surface_material", "surface_idx", "material"), &ImporterMesh::set_surface_material);

	ClassDB::bind_method(D_METHOD("generate_lods", "normal_merge_angle", "normal_split_angle", "bone_transform_array"), &ImporterMesh::generate_lods);
	ClassDB::bind_method(D_METHOD("get_mesh", "base_mesh"), &ImporterMesh::get_mesh, DEFVAL(Ref<ArrayMesh>()));
	ClassDB::bind_method(D_METHOD("clear"), &ImporterMesh::clear);

	ClassDB::bind_method(D_METHOD("_set_data", "data"), &ImporterMesh::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &ImporterMesh::_get_data);

	ClassDB::bind_method(D_METHOD("set_lightmap_size_hint", "size"), &ImporterMesh::set_lightmap_size_hint);
	ClassDB::bind_method(D_METHOD("get_lightmap_size_hint"), &ImporterMesh::get_lightmap_size_hint);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "_set_data", "_get_data");
}
