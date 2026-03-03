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
#include "core/math/random_pcg.h"
#include "core/object/class_db.h"
#include "scene/resources/surface_tool.h"

#ifndef PHYSICS_3D_DISABLED
#include "core/math/convex_hull.h"
#endif // PHYSICS_3D_DISABLED

Ref<ImporterMesh> ImporterMesh::merge_importer_meshes(const TypedArray<ImporterMesh> &p_importer_meshes, const TypedArray<Transform3D> &p_relative_transforms, bool p_deduplicate_surfaces) {
	// Setup and safety checks.
	const int mesh_count = p_importer_meshes.size();
	Ref<ImporterMesh> merged_importer_mesh;
	ERR_FAIL_COND_V(mesh_count == 0, merged_importer_mesh);
	ERR_FAIL_COND_V(mesh_count != p_relative_transforms.size(), merged_importer_mesh);
	// Contains more than just the surface arrays, also contains some metadata to help with merging.
	HashMap<String, Array> names_to_surfaces;
	for (int mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
		const Ref<ImporterMesh> importer_mesh = p_importer_meshes[mesh_index];
		if (importer_mesh->get_blend_shape_count() > 0) {
			WARN_PRINT("ImporterMesh.merge_importer_meshes: Mesh " + itos(mesh_index) + " has blend shapes, which are not supported and will be discarded in the merged mesh.");
		}
		const Transform3D &relative_transform = p_relative_transforms[mesh_index];
		const bool is_determinant_negative = relative_transform.basis.determinant() < 0;
		for (int surface_index = 0; surface_index < importer_mesh->get_surface_count(); surface_index++) {
			if (importer_mesh->get_surface_lod_count(surface_index) > 0) {
				WARN_PRINT("ImporterMesh.merge_importer_meshes: Mesh " + itos(mesh_index) + " surface " + itos(surface_index) + " has LODs, which are not supported and will be discarded in the merged mesh.");
			}
			// Shallow-duplicate the surface arrays so that writing transformed data back doesn't mutate the original mesh.
			Array this_surface_arrays = importer_mesh->get_surface_arrays(surface_index).duplicate(false);
			ERR_FAIL_COND_V(this_surface_arrays.size() != Mesh::ARRAY_MAX, merged_importer_mesh);
			// Transform the data of the mesh by the instance's relative transform.
			{
				PackedVector3Array vertices = this_surface_arrays[Mesh::ARRAY_VERTEX];
				for (int vertex_index = 0; vertex_index < vertices.size(); vertex_index++) {
					vertices.ptrw()[vertex_index] = relative_transform.xform(vertices[vertex_index]);
				}
				PackedVector3Array normals = this_surface_arrays[Mesh::ARRAY_NORMAL];
				for (int normal_index = 0; normal_index < normals.size(); normal_index++) {
					normals.ptrw()[normal_index] = relative_transform.basis.xform(normals[normal_index]).normalized();
				}
				PackedFloat32Array tangents = this_surface_arrays[Mesh::ARRAY_TANGENT];
				for (int tangent_index = 0; tangent_index < tangents.size(); tangent_index += 4) {
					Vector3 tangent = Vector3(tangents[tangent_index], tangents[tangent_index + 1], tangents[tangent_index + 2]);
					tangent = relative_transform.basis.xform(tangent).normalized();
					tangents.ptrw()[tangent_index + 0] = tangent.x;
					tangents.ptrw()[tangent_index + 1] = tangent.y;
					tangents.ptrw()[tangent_index + 2] = tangent.z;
					// The tangent's W component is not transformed (the binormal direction sign), so we keep it as is.
				}
				// If the determinant is negative, we need to swap vertices to fix the winding order.
				if (is_determinant_negative) {
					PackedInt32Array this_indices = this_surface_arrays[Mesh::ARRAY_INDEX];
					if (this_indices.is_empty()) {
						// For non-indexed meshes, we need to swap the data in the arrays.
						PackedColorArray colors = this_surface_arrays[Mesh::ARRAY_COLOR];
						PackedVector2Array tex_uv1 = this_surface_arrays[Mesh::ARRAY_TEX_UV];
						PackedVector2Array tex_uv2 = this_surface_arrays[Mesh::ARRAY_TEX_UV2];
						Color temp_color;
						Vector4 temp_vec4;
						Vector3 temp_vec3;
						Vector2 temp_vec2;
						for (int i = 1; i < vertices.size() - 1; i += 3) {
							temp_vec3 = vertices[i];
							vertices.ptrw()[i] = vertices[i + 1];
							vertices.ptrw()[i + 1] = temp_vec3;
						}
						for (int i = 1; i < normals.size() - 1; i += 3) {
							temp_vec3 = normals[i];
							normals.ptrw()[i] = normals[i + 1];
							normals.ptrw()[i + 1] = temp_vec3;
						}
						for (int i = 4; i < tangents.size() - 1; i += 12) {
							temp_vec4 = Vector4(tangents[i + 0], tangents[i + 1], tangents[i + 2], tangents[i + 3]);
							tangents.ptrw()[i + 0] = tangents[i + 4];
							tangents.ptrw()[i + 1] = tangents[i + 5];
							tangents.ptrw()[i + 2] = tangents[i + 6];
							tangents.ptrw()[i + 3] = tangents[i + 7];
							tangents.ptrw()[i + 4] = temp_vec4.x;
							tangents.ptrw()[i + 5] = temp_vec4.y;
							tangents.ptrw()[i + 6] = temp_vec4.z;
							tangents.ptrw()[i + 7] = temp_vec4.w;
						}
						for (int i = 1; i < colors.size() - 1; i += 3) {
							temp_color = colors[i];
							colors.ptrw()[i] = colors[i + 1];
							colors.ptrw()[i + 1] = temp_color;
						}
						for (int i = 1; i < tex_uv1.size() - 1; i += 3) {
							temp_vec2 = tex_uv1[i];
							tex_uv1.ptrw()[i] = tex_uv1[i + 1];
							tex_uv1.ptrw()[i + 1] = temp_vec2;
						}
						for (int i = 1; i < tex_uv2.size() - 1; i += 3) {
							temp_vec2 = tex_uv2[i];
							tex_uv2.ptrw()[i] = tex_uv2[i + 1];
							tex_uv2.ptrw()[i + 1] = temp_vec2;
						}
						// Swap custom data channels.
						for (int custom_index = 0; custom_index < 4; custom_index++) {
							Variant custom_var = this_surface_arrays[Mesh::ARRAY_CUSTOM0 + custom_index];
							if (custom_var.get_type() == Variant::PACKED_BYTE_ARRAY) {
								PackedByteArray custom_bytes = custom_var;
								if (!custom_bytes.is_empty()) {
									// Each vertex may have multiple bytes associated with it, such as in a half precision float.
									const int byte_stride = custom_bytes.size() / vertices.size();
									for (int i = 1; i < vertices.size() - 1; i += 3) {
										for (int s = 0; s < byte_stride; s++) {
											const uint8_t temp_byte = custom_bytes[i * byte_stride + s];
											custom_bytes.ptrw()[i * byte_stride + s] = custom_bytes[(i + 1) * byte_stride + s];
											custom_bytes.ptrw()[(i + 1) * byte_stride + s] = temp_byte;
										}
									}
									this_surface_arrays[Mesh::ARRAY_CUSTOM0 + custom_index] = custom_bytes;
								}
							} else if (custom_var.get_type() == Variant::PACKED_FLOAT32_ARRAY) {
								PackedFloat32Array custom_floats = custom_var;
								if (!custom_floats.is_empty()) {
									// Each vertex may have multiple floats associated with it, such as in a vector or color.
									const int float_stride = custom_floats.size() / vertices.size();
									for (int i = 1; i < vertices.size() - 1; i += 3) {
										for (int s = 0; s < float_stride; s++) {
											const float temp_float = custom_floats[i * float_stride + s];
											custom_floats.ptrw()[i * float_stride + s] = custom_floats[(i + 1) * float_stride + s];
											custom_floats.ptrw()[(i + 1) * float_stride + s] = temp_float;
										}
									}
									this_surface_arrays[Mesh::ARRAY_CUSTOM0 + custom_index] = custom_floats;
								}
							} else {
								ERR_PRINT("Unsupported custom data format when merging ImporterMesh surfaces.");
							}
						}
						// Put the data back into the surface arrays.
						this_surface_arrays[Mesh::ARRAY_COLOR] = colors.is_empty() ? Variant() : Variant(colors);
						this_surface_arrays[Mesh::ARRAY_TEX_UV] = tex_uv1.is_empty() ? Variant() : Variant(tex_uv1);
						this_surface_arrays[Mesh::ARRAY_TEX_UV2] = tex_uv2.is_empty() ? Variant() : Variant(tex_uv2);
					} else {
						// For indexed meshes, we need to swap the indices.
						for (int i = 1; i < this_indices.size() - 1; i += 3) {
							int32_t temp = this_indices[i];
							this_indices.ptrw()[i] = this_indices[i + 1];
							this_indices.ptrw()[i + 1] = temp;
						}
						this_surface_arrays[Mesh::ARRAY_INDEX] = this_indices;
					}
				}
				// This data always needs to be put back into the surface arrays,
				// because it gets transformed even if the determinant is positive.
				this_surface_arrays[Mesh::ARRAY_VERTEX] = vertices;
				this_surface_arrays[Mesh::ARRAY_NORMAL] = normals.is_empty() ? Variant() : Variant(normals);
				this_surface_arrays[Mesh::ARRAY_TANGENT] = tangents.is_empty() ? Variant() : Variant(tangents);
			}
			// Insert the transformed data into the temporary HashMap.
			const Mesh::PrimitiveType mesh_prim_type = importer_mesh->get_surface_primitive_type(surface_index);
			const uint64_t mesh_flags = importer_mesh->get_surface_format(surface_index);
			String surface_name = importer_mesh->get_surface_name(surface_index);
			if (surface_name.is_empty()) {
				surface_name = String("surface_") + itos(surface_index);
			}
			// Check if the surface has bone data by inspecting the actual arrays.
			// NOTE: Unlike ArrayMesh, we can't use the mesh format flags, because those may not be set by ImporterMesh callers.
			const bool has_bones = this_surface_arrays[Mesh::ARRAY_BONES].get_type() != Variant::NIL;
			const bool has_weights = this_surface_arrays[Mesh::ARRAY_WEIGHTS].get_type() != Variant::NIL;
			const bool name_exists = names_to_surfaces.has(surface_name);
			if (name_exists) {
				// Only attempt to deduplicate surfaces if the mesh is not skinned.
				// Avoid deduplicating surfaces with bone weights.
				constexpr uint64_t skinning_flags = Mesh::ARRAY_FORMAT_BONES | Mesh::ARRAY_FORMAT_WEIGHTS;
				const bool is_skinned = has_bones || has_weights;
				if (p_deduplicate_surfaces && !is_skinned && (mesh_flags & skinning_flags) == 0) {
					Array &existing_surface = names_to_surfaces[surface_name];
					const Mesh::PrimitiveType existing_prim_type = (Mesh::PrimitiveType)(uint64_t)existing_surface[0];
					const uint64_t existing_flags = (uint64_t)existing_surface[3];
					if (existing_prim_type == mesh_prim_type && existing_flags == mesh_flags) {
						// Duplicate surface found, insert the data into the existing surface.
						Array merged_surface_arrays = existing_surface[1];
						PackedVector3Array merged_vertices = merged_surface_arrays[Mesh::ARRAY_VERTEX];
						const int32_t existing_vertex_count = merged_vertices.size();
						// Merge vertices (always present).
						merged_vertices.append_array(this_surface_arrays[Mesh::ARRAY_VERTEX]);
						merged_surface_arrays[Mesh::ARRAY_VERTEX] = merged_vertices;
						// Merge normals.
						{
							PackedVector3Array existing_normals = merged_surface_arrays[Mesh::ARRAY_NORMAL];
							const PackedVector3Array incoming_normals = this_surface_arrays[Mesh::ARRAY_NORMAL];
							if (!existing_normals.is_empty() || !incoming_normals.is_empty()) {
								existing_normals.append_array(incoming_normals);
								merged_surface_arrays[Mesh::ARRAY_NORMAL] = existing_normals;
							}
						}
						// Merge tangents.
						{
							PackedFloat32Array existing_tangents = merged_surface_arrays[Mesh::ARRAY_TANGENT];
							const PackedFloat32Array incoming_tangents = this_surface_arrays[Mesh::ARRAY_TANGENT];
							if (!existing_tangents.is_empty() || !incoming_tangents.is_empty()) {
								existing_tangents.append_array(incoming_tangents);
								merged_surface_arrays[Mesh::ARRAY_TANGENT] = existing_tangents;
							}
						}
						// Merge colors.
						{
							PackedColorArray existing_colors = merged_surface_arrays[Mesh::ARRAY_COLOR];
							const PackedColorArray incoming_colors = this_surface_arrays[Mesh::ARRAY_COLOR];
							if (!existing_colors.is_empty() || !incoming_colors.is_empty()) {
								existing_colors.append_array(incoming_colors);
								merged_surface_arrays[Mesh::ARRAY_COLOR] = existing_colors;
							}
						}
						// Merge UV1.
						{
							PackedVector2Array existing_uv = merged_surface_arrays[Mesh::ARRAY_TEX_UV];
							const PackedVector2Array incoming_uv = this_surface_arrays[Mesh::ARRAY_TEX_UV];
							if (!existing_uv.is_empty() || !incoming_uv.is_empty()) {
								existing_uv.append_array(incoming_uv);
								merged_surface_arrays[Mesh::ARRAY_TEX_UV] = existing_uv;
							}
						}
						// Merge UV2.
						{
							PackedVector2Array existing_uv2 = merged_surface_arrays[Mesh::ARRAY_TEX_UV2];
							const PackedVector2Array incoming_uv2 = this_surface_arrays[Mesh::ARRAY_TEX_UV2];
							if (!existing_uv2.is_empty() || !incoming_uv2.is_empty()) {
								existing_uv2.append_array(incoming_uv2);
								merged_surface_arrays[Mesh::ARRAY_TEX_UV2] = existing_uv2;
							}
						}
						// Merge custom data channels.
						for (int custom_index = 0; custom_index < 4; custom_index++) {
							const Variant existing_custom = merged_surface_arrays[Mesh::ARRAY_CUSTOM0 + custom_index];
							const Variant incoming_custom = this_surface_arrays[Mesh::ARRAY_CUSTOM0 + custom_index];
							if (existing_custom.get_type() == Variant::PACKED_BYTE_ARRAY || incoming_custom.get_type() == Variant::PACKED_BYTE_ARRAY) {
								PackedByteArray merged_custom = existing_custom;
								merged_custom.append_array(PackedByteArray(incoming_custom));
								merged_surface_arrays[Mesh::ARRAY_CUSTOM0 + custom_index] = merged_custom;
							} else if (existing_custom.get_type() == Variant::PACKED_FLOAT32_ARRAY || incoming_custom.get_type() == Variant::PACKED_FLOAT32_ARRAY) {
								PackedFloat32Array merged_custom = existing_custom;
								merged_custom.append_array(PackedFloat32Array(incoming_custom));
								merged_surface_arrays[Mesh::ARRAY_CUSTOM0 + custom_index] = merged_custom;
							}
						}
						// Merge indices and remap to account for the new vertex count.
						{
							PackedInt32Array existing_indices = merged_surface_arrays[Mesh::ARRAY_INDEX];
							PackedInt32Array incoming_indices = this_surface_arrays[Mesh::ARRAY_INDEX];
							if (!existing_indices.is_empty() || !incoming_indices.is_empty()) {
								for (int i = 0; i < incoming_indices.size(); i++) {
									incoming_indices.ptrw()[i] = incoming_indices[i] + existing_vertex_count;
								}
								existing_indices.append_array(incoming_indices);
								merged_surface_arrays[Mesh::ARRAY_INDEX] = existing_indices;
							}
						}
						existing_surface.set(1, merged_surface_arrays);
						continue; // Next surface.
					}
				}
				// If the name already exists but isn't a duplicate, we need a new name for the surface.
				const String original_name = surface_name;
				int64_t discriminator = 2;
				do {
					surface_name = original_name + "_" + itos(discriminator);
					discriminator++;
				} while (names_to_surfaces.has(surface_name));
			}
			// Add a new entry to the temporary HashMap. The indices are based on the arguments to add_surface.
			Array new_surface;
			new_surface.resize(4);
			new_surface[0] = mesh_prim_type;
			new_surface[1] = this_surface_arrays;
			new_surface[2] = importer_mesh->get_surface_material(surface_index);
			new_surface[3] = mesh_flags;
			names_to_surfaces[surface_name] = new_surface;
		}
	}
	// Actually put the merged surfaces into the merged ImporterMesh.
	merged_importer_mesh.instantiate();
	for (const KeyValue<String, Array> &surface_name_kvp : names_to_surfaces) {
		const Array surface = surface_name_kvp.value;
		const Mesh::PrimitiveType mesh_prim_type = (Mesh::PrimitiveType)(uint64_t)surface[0];
		const Ref<Material> material = surface[2];
		const uint64_t mesh_flags = (uint64_t)surface[3];
		merged_importer_mesh->add_surface(mesh_prim_type, surface[1], TypedArray<Array>(), Dictionary(), material, surface_name_kvp.key, mesh_flags);
	}
	return merged_importer_mesh;
}

String ImporterMesh::validate_blend_shape_name(const String &p_name) {
	return p_name.replace_char(':', '_');
}

void ImporterMesh::add_blend_shape(const String &p_name) {
	ERR_FAIL_COND(surfaces.size() > 0);
	blend_shapes.push_back(validate_blend_shape_name(p_name));
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

void ImporterMesh::add_surface(Mesh::PrimitiveType p_primitive, const Array &p_arrays, const TypedArray<Array> &p_blend_shapes, const Dictionary &p_lods, const Ref<Material> &p_material, const String &p_surface_name, const uint64_t p_flags) {
	ERR_FAIL_COND(p_blend_shapes.size() != blend_shapes.size());
	ERR_FAIL_COND(p_arrays.size() != Mesh::ARRAY_MAX);
	Surface s;
	s.primitive = p_primitive;
	s.arrays = p_arrays;
	s.name = p_surface_name;
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

	for (const KeyValue<Variant, Variant> &kv : p_lods) {
		ERR_CONTINUE(!kv.key.is_num());
		Surface::LOD lod;
		lod.distance = kv.key;
		lod.indices = kv.value;
		ERR_CONTINUE(lod.indices.is_empty());
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

template <typename T>
static Vector<T> _remap_array(Vector<T> p_array, const Vector<uint32_t> &p_remap, uint32_t p_vertex_count) {
	ERR_FAIL_COND_V(p_array.size() % p_remap.size() != 0, p_array);
	int num_elements = p_array.size() / p_remap.size();
	T *data = p_array.ptrw();
	SurfaceTool::remap_vertex_func(data, data, p_remap.size(), sizeof(T) * num_elements, p_remap.ptr());
	p_array.resize(p_vertex_count * num_elements);
	return p_array;
}

static void _remap_arrays(Array &r_arrays, const Vector<uint32_t> &p_remap, uint32_t p_vertex_count) {
	for (int i = 0; i < r_arrays.size(); i++) {
		if (i == RSE::ARRAY_INDEX) {
			continue;
		}

		switch (r_arrays[i].get_type()) {
			case Variant::NIL:
				break;
			case Variant::PACKED_VECTOR3_ARRAY:
				r_arrays[i] = _remap_array<Vector3>(r_arrays[i], p_remap, p_vertex_count);
				break;
			case Variant::PACKED_VECTOR2_ARRAY:
				r_arrays[i] = _remap_array<Vector2>(r_arrays[i], p_remap, p_vertex_count);
				break;
			case Variant::PACKED_FLOAT32_ARRAY:
				r_arrays[i] = _remap_array<float>(r_arrays[i], p_remap, p_vertex_count);
				break;
			case Variant::PACKED_INT32_ARRAY:
				r_arrays[i] = _remap_array<int32_t>(r_arrays[i], p_remap, p_vertex_count);
				break;
			case Variant::PACKED_BYTE_ARRAY:
				r_arrays[i] = _remap_array<uint8_t>(r_arrays[i], p_remap, p_vertex_count);
				break;
			case Variant::PACKED_COLOR_ARRAY:
				r_arrays[i] = _remap_array<Color>(r_arrays[i], p_remap, p_vertex_count);
				break;
			default:
				ERR_FAIL_MSG("Unhandled array type.");
		}
	}
}

void ImporterMesh::optimize_indices() {
	if (!SurfaceTool::optimize_vertex_cache_func) {
		return;
	}
	if (!SurfaceTool::optimize_vertex_fetch_remap_func || !SurfaceTool::remap_vertex_func || !SurfaceTool::remap_index_func) {
		return;
	}

	for (int i = 0; i < surfaces.size(); i++) {
		if (surfaces[i].primitive != Mesh::PRIMITIVE_TRIANGLES) {
			continue;
		}

		Vector<Vector3> vertices = surfaces[i].arrays[RSE::ARRAY_VERTEX];
		PackedInt32Array indices = surfaces[i].arrays[RSE::ARRAY_INDEX];

		unsigned int index_count = indices.size();
		unsigned int vertex_count = vertices.size();

		if (index_count == 0) {
			continue;
		}

		// Optimize indices for vertex cache to establish final triangle order.
		int *indices_ptr = indices.ptrw();
		SurfaceTool::optimize_vertex_cache_func((unsigned int *)indices_ptr, (const unsigned int *)indices_ptr, index_count, vertex_count);
		surfaces.write[i].arrays[RSE::ARRAY_INDEX] = indices;

		for (int j = 0; j < surfaces[i].lods.size(); ++j) {
			Surface::LOD &lod = surfaces.write[i].lods.write[j];
			int *lod_indices_ptr = lod.indices.ptrw();
			SurfaceTool::optimize_vertex_cache_func((unsigned int *)lod_indices_ptr, (const unsigned int *)lod_indices_ptr, lod.indices.size(), vertex_count);
		}

		// Concatenate indices for all LODs in the order of coarse->fine; this establishes the effective order of vertices,
		// and is important to optimize for vertex fetch (all GPUs) and shading (Mali GPUs)
		PackedInt32Array merged_indices;
		for (int j = surfaces[i].lods.size() - 1; j >= 0; --j) {
			merged_indices.append_array(surfaces[i].lods[j].indices);
		}
		merged_indices.append_array(indices);

		// Generate remap array that establishes optimal vertex order according to the order of indices above.
		Vector<uint32_t> remap;
		remap.resize(vertex_count);
		unsigned int new_vertex_count = SurfaceTool::optimize_vertex_fetch_remap_func(remap.ptrw(), (const unsigned int *)merged_indices.ptr(), merged_indices.size(), vertex_count);

		// We need to remap all vertex and index arrays in lockstep according to the remap.
		SurfaceTool::remap_index_func((unsigned int *)indices_ptr, (const unsigned int *)indices_ptr, index_count, remap.ptr());
		surfaces.write[i].arrays[RSE::ARRAY_INDEX] = indices;

		for (int j = 0; j < surfaces[i].lods.size(); ++j) {
			Surface::LOD &lod = surfaces.write[i].lods.write[j];
			int *lod_indices_ptr = lod.indices.ptrw();
			SurfaceTool::remap_index_func((unsigned int *)lod_indices_ptr, (const unsigned int *)lod_indices_ptr, lod.indices.size(), remap.ptr());
		}

		_remap_arrays(surfaces.write[i].arrays, remap, new_vertex_count);
		for (int j = 0; j < surfaces[i].blend_shape_data.size(); j++) {
			_remap_arrays(surfaces.write[i].blend_shape_data.write[j].arrays, remap, new_vertex_count);
		}
	}

	if (shadow_mesh.is_valid()) {
		shadow_mesh->optimize_indices();
	}
}

#define VERTEX_SKIN_FUNC(bone_count, vert_idx, read_array, write_array, transform_array, bone_array, weight_array) \
	Vector3 transformed_vert; \
	for (unsigned int weight_idx = 0; weight_idx < bone_count; weight_idx++) { \
		int bone_idx = bone_array[vert_idx * bone_count + weight_idx]; \
		float w = weight_array[vert_idx * bone_count + weight_idx]; \
		if (w < FLT_EPSILON) { \
			continue; \
		} \
		ERR_FAIL_INDEX(bone_idx, static_cast<int>(transform_array.size())); \
		transformed_vert += transform_array[bone_idx].xform(read_array[vert_idx]) * w; \
	} \
	write_array[vert_idx] = transformed_vert;

void ImporterMesh::generate_lods(float p_normal_merge_angle, Array p_bone_transform_array) {
	if (!SurfaceTool::simplify_scale_func) {
		return;
	}
	if (!SurfaceTool::simplify_with_attrib_func) {
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
		Vector<Vector3> vertices = surfaces[i].arrays[RSE::ARRAY_VERTEX];
		PackedInt32Array indices = surfaces[i].arrays[RSE::ARRAY_INDEX];
		Vector<Vector3> normals = surfaces[i].arrays[RSE::ARRAY_NORMAL];
		Vector<float> tangents = surfaces[i].arrays[RSE::ARRAY_TANGENT];
		Vector<Vector2> uvs = surfaces[i].arrays[RSE::ARRAY_TEX_UV];
		Vector<Vector2> uv2s = surfaces[i].arrays[RSE::ARRAY_TEX_UV2];
		Vector<int> bones = surfaces[i].arrays[RSE::ARRAY_BONES];
		Vector<float> weights = surfaces[i].arrays[RSE::ARRAY_WEIGHTS];
		Vector<Color> colors = surfaces[i].arrays[RSE::ARRAY_COLOR];

		unsigned int index_count = indices.size();
		unsigned int vertex_count = vertices.size();

		if (index_count == 0) {
			continue; //no lods if no indices
		}
		ERR_FAIL_COND_MSG(index_count % 3 != 0, "ImporterMesh::generate_lods: Indexed triangle meshes MUST have an index array with a size that is a multiple of 3, but got " + itos(index_count) + " indices. Cannot generate LODs for this invalid mesh.");

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

		bool deformable = bones.size() > 0 || blend_shapes.size() > 0;

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
		const Vector3 *normals_ptr = normals.ptr();

		HashMap<Vector3, LocalVector<Pair<int, int>>> unique_vertices;

		LocalVector<int> vertex_remap;
		LocalVector<int> vertex_inverse_remap;
		LocalVector<Vector3> merged_vertices;
		LocalVector<Vector3> merged_normals;
		LocalVector<int> merged_normals_counts;
		const Vector2 *uvs_ptr = uvs.ptr();
		const Vector2 *uv2s_ptr = uv2s.ptr();
		const float *tangents_ptr = tangents.ptr();
		const Color *colors_ptr = colors.ptr();

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
					bool is_tang_aligned = !tangents_ptr || (tangents_ptr[j * 4 + 3] < 0) == (tangents_ptr[idx.second * 4 + 3] < 0);
					ERR_FAIL_INDEX(idx.second, normals.size());
					bool is_normals_close = normals[idx.second].dot(n) > normal_merge_threshold;
					bool is_col_close = (!colors_ptr || colors_ptr[j].is_equal_approx(colors_ptr[idx.second]));
					if (is_uvs_close && is_uv2s_close && is_normals_close && is_tang_aligned && is_col_close) {
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
		Vector3 *merged_normals_ptr = merged_normals.ptr();

		{
			const int *counts_ptr = merged_normals_counts.ptr();
			for (unsigned int j = 0; j < merged_vertex_count; j++) {
				merged_normals_ptr[j] /= counts_ptr[j];
			}
		}

		Vector<float> merged_vertices_f32 = vector3_to_float32_array(merged_vertices_ptr, merged_vertex_count);
		float scale = SurfaceTool::simplify_scale_func(merged_vertices_f32.ptr(), merged_vertex_count, sizeof(float) * 3);

		const size_t attrib_count = 6; // 3 for normal + 3 for color (if present)

		float attrib_weights[attrib_count] = {};

		// Give some weight to normal preservation
		attrib_weights[0] = attrib_weights[1] = attrib_weights[2] = 1.0f;

		// Give some weight to colors but only if present to avoid redundant computations during simplification
		if (colors_ptr) {
			attrib_weights[3] = attrib_weights[4] = attrib_weights[5] = 1.0f;
		}

		LocalVector<float> merged_attribs;
		merged_attribs.resize(merged_vertex_count * attrib_count);
		float *merged_attribs_ptr = merged_attribs.ptr();

		memset(merged_attribs_ptr, 0, merged_attribs.size() * sizeof(float));

		for (unsigned int j = 0; j < merged_vertex_count; ++j) {
			merged_attribs_ptr[j * attrib_count + 0] = merged_normals_ptr[j].x;
			merged_attribs_ptr[j * attrib_count + 1] = merged_normals_ptr[j].y;
			merged_attribs_ptr[j * attrib_count + 2] = merged_normals_ptr[j].z;

			if (colors_ptr) {
				unsigned int rj = vertex_inverse_remap[j];

				merged_attribs_ptr[j * attrib_count + 3] = colors_ptr[rj].r;
				merged_attribs_ptr[j * attrib_count + 4] = colors_ptr[rj].g;
				merged_attribs_ptr[j * attrib_count + 5] = colors_ptr[rj].b;
			}
		}

		print_verbose("LOD Generation: Triangles " + itos(index_count / 3) + ", vertices " + itos(vertex_count) + " (merged " + itos(merged_vertex_count) + ")" + (deformable ? ", deformable" : ""));

		const float max_mesh_error = 1.0f; // We only need LODs that can be selected by error threshold.
		const unsigned min_target_indices = 12;

		LocalVector<int> current_indices(merged_indices);
		float current_error = 0.0f;
		bool allow_prune = true;

		while (current_indices.size() > min_target_indices * 2) {
			unsigned int current_index_count = current_indices.size();
			unsigned int target_index_count = MAX(((current_index_count / 3) / 2) * 3, min_target_indices);

			PackedInt32Array new_indices;
			new_indices.resize(current_index_count);

			int simplify_options = SurfaceTool::SIMPLIFY_SPARSE; // Does not change appearance, but speeds up subsequent iterations.

			// Lock geometric boundary in case the mesh is composed of multiple material subsets.
			simplify_options |= SurfaceTool::SIMPLIFY_LOCK_BORDER;

			if (allow_prune) {
				// Remove small disconnected components.
				simplify_options |= SurfaceTool::SIMPLIFY_PRUNE;
			}

			if (deformable) {
				// Improves appearance of deformable objects after deformation by using more regular tessellation.
				simplify_options |= SurfaceTool::SIMPLIFY_REGULARIZE;
			}

			float step_error = 0.0f;
			size_t new_index_count = SurfaceTool::simplify_with_attrib_func(
					(unsigned int *)new_indices.ptrw(),
					(const uint32_t *)current_indices.ptr(), current_index_count,
					merged_vertices_f32.ptr(), merged_vertex_count,
					sizeof(float) * 3, // Vertex stride
					merged_attribs_ptr,
					sizeof(float) * attrib_count, // Attribute stride
					attrib_weights, attrib_count,
					nullptr, // Vertex lock
					target_index_count,
					max_mesh_error,
					simplify_options,
					&step_error);

			if (new_index_count == 0 && allow_prune) {
				// If the best result the simplifier could arrive at with pruning enabled is 0 triangles, there might still be an opportunity
				// to reduce the number of triangles further *without* completely decimating the mesh. It will be impossible to reach the target
				// this way - if the target was reachable without going down to 0, the simplifier would have done it! - but we might still be able
				// to get one more slightly lower level if we retry without pruning.
				allow_prune = false;
				continue;
			}

			// Accumulate error over iterations. Usually, it's correct to use step_error as is; however, on coarse LODs, we may start
			// getting *smaller* relative error compared to the previous LOD. To make sure the error is monotonic and strictly increasing,
			// and to limit the switching (pop) distance, we ensure the error grows by an arbitrary factor each iteration.
			current_error = MAX(current_error * 1.5f, step_error);

			new_indices.resize(new_index_count);
			current_indices = new_indices;

			if (new_index_count == 0 || (new_index_count >= current_index_count * 0.75f)) {
				print_verbose("  LOD stop: got " + itos(new_index_count / 3) + " triangles when asking for " + itos(target_index_count / 3));
				break;
			}

			if (current_error > max_mesh_error) {
				print_verbose("  LOD stop: reached " + rtos(current_error) + " cumulative error (step error " + rtos(step_error) + ")");
				break;
			}

			// We need to remap the LOD indices back to the original vertex array; note that we already copied new_indices into current_indices for subsequent iteration.
			{
				int *ptrw = new_indices.ptrw();
				for (unsigned int j = 0; j < new_index_count; j++) {
					ptrw[j] = vertex_inverse_remap[ptrw[j]];
				}
			}

			Surface::LOD lod;
			lod.distance = MAX(current_error * scale, CMP_EPSILON2);
			lod.indices = new_indices;
			surfaces.write[i].lods.push_back(lod);

			print_verbose("  LOD " + itos(surfaces.write[i].lods.size()) + ": " + itos(new_index_count / 3) + " triangles, error " + rtos(current_error) + " (step error " + rtos(step_error) + ")");
		}

		surfaces.write[i].lods.sort_custom<Surface::LODComparator>();
	}
}

void ImporterMesh::_generate_lods_bind(float p_normal_merge_angle, float p_normal_split_angle, Array p_skin_pose_transform_array) {
	// p_normal_split_angle is unused, but kept for compatibility
	generate_lods(p_normal_merge_angle, p_skin_pose_transform_array);
}

bool ImporterMesh::has_mesh() const {
	return mesh.is_valid();
}

Ref<ArrayMesh> ImporterMesh::get_mesh(const Ref<ArrayMesh> &p_base) {
	ERR_FAIL_COND_V(surfaces.is_empty(), Ref<ArrayMesh>());

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

Ref<ImporterMesh> ImporterMesh::from_mesh(const Ref<Mesh> &p_mesh) {
	Ref<ImporterMesh> importer_mesh;
	importer_mesh.instantiate();
	if (p_mesh.is_null()) {
		return importer_mesh;
	}
	Ref<ArrayMesh> array_mesh = p_mesh;
	// Convert blend shape mode and names if any.
	if (p_mesh->get_blend_shape_count() > 0) {
		ArrayMesh::BlendShapeMode shape_mode = ArrayMesh::BLEND_SHAPE_MODE_NORMALIZED;
		if (array_mesh.is_valid()) {
			shape_mode = array_mesh->get_blend_shape_mode();
		}
		importer_mesh->set_blend_shape_mode(shape_mode);
		for (int morph_i = 0; morph_i < p_mesh->get_blend_shape_count(); morph_i++) {
			importer_mesh->add_blend_shape(p_mesh->get_blend_shape_name(morph_i));
		}
	}
	// Add surfaces one by one.
	for (int32_t surface_i = 0; surface_i < p_mesh->get_surface_count(); surface_i++) {
		Ref<Material> mat = p_mesh->surface_get_material(surface_i);
		String surface_name;
		if (array_mesh.is_valid()) {
			surface_name = array_mesh->surface_get_name(surface_i);
		}
		if (surface_name.is_empty() && mat.is_valid()) {
			surface_name = mat->get_name();
		}
		importer_mesh->add_surface(p_mesh->surface_get_primitive_type(surface_i), p_mesh->surface_get_arrays(surface_i),
				p_mesh->surface_get_blend_shape_arrays(surface_i), p_mesh->surface_get_lods(surface_i),
				mat, surface_name, p_mesh->surface_get_format(surface_i));
	}
	// Merge metadata.
	importer_mesh->merge_meta_from(*p_mesh);
	importer_mesh->set_name(p_mesh->get_name());
	return importer_mesh;
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
		if (surfaces[i].arrays[RSE::ARRAY_BONES].get_type() != Variant::NIL) {
			return;
		}
		if (surfaces[i].arrays[RSE::ARRAY_WEIGHTS].get_type() != Variant::NIL) {
			return;
		}
	}

	shadow_mesh.instantiate();

	for (int i = 0; i < surfaces.size(); i++) {
		LocalVector<int> vertex_remap;
		Vector<Vector3> new_vertices;
		Vector<Vector3> vertices = surfaces[i].arrays[RSE::ARRAY_VERTEX];
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
		new_surface.resize(RSE::ARRAY_MAX);
		Dictionary lods;

		//		print_line("original vertex count: " + itos(vertices.size()) + " new vertex count: " + itos(new_vertices.size()));

		new_surface[RSE::ARRAY_VERTEX] = new_vertices;

		Vector<int> indices = surfaces[i].arrays[RSE::ARRAY_INDEX];
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

			new_surface[RSE::ARRAY_INDEX] = new_indices;

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

#ifndef PHYSICS_3D_DISABLED
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
					index = vertex_count++;
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
			shape->set_points(Vector<Vector3>(md.vertices));
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
	if (faces.is_empty()) {
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
#endif // PHYSICS_3D_DISABLED

Ref<NavigationMesh> ImporterMesh::create_navigation_mesh() {
	Vector<Face3> faces = get_faces();
	if (faces.is_empty()) {
		return Ref<NavigationMesh>();
	}

	HashMap<Vector3, int> unique_vertices;
	Vector<Vector<int>> face_polygons;
	face_polygons.resize(faces.size());

	for (int i = 0; i < faces.size(); i++) {
		Vector<int> face_indices;
		face_indices.resize(3);
		for (int j = 0; j < 3; j++) {
			Vector3 v = faces[i].vertex[j];
			int idx;
			if (unique_vertices.has(v)) {
				idx = unique_vertices[v];
			} else {
				idx = unique_vertices.size();
				unique_vertices[v] = idx;
			}
			face_indices.write[j] = idx;
		}
		face_polygons.write[i] = face_indices;
	}

	Vector<Vector3> vertices;
	vertices.resize(unique_vertices.size());
	for (const KeyValue<Vector3, int> &E : unique_vertices) {
		vertices.write[E.value] = E.key;
	}

	Ref<NavigationMesh> nm;
	nm.instantiate();
	nm->set_data(vertices, face_polygons);

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

static const uint32_t custom_shift[RSE::ARRAY_CUSTOM_COUNT] = { Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT, Mesh::ARRAY_FORMAT_CUSTOM1_SHIFT, Mesh::ARRAY_FORMAT_CUSTOM2_SHIFT, Mesh::ARRAY_FORMAT_CUSTOM3_SHIFT };

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

		SurfaceTool::create_vertex_array_from_arrays(arrays, s.vertices, &s.format);

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

		for (int custom_i = 0; custom_i < RSE::ARRAY_CUSTOM_COUNT; custom_i++) {
			st->set_custom_format(custom_i, (SurfaceTool::CustomFormat)((lightmap_surfaces[i].format >> custom_shift[custom_i]) & RSE::ARRAY_FORMAT_CUSTOM_MASK));
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
			for (int custom_i = 0; custom_i < RSE::ARRAY_CUSTOM_COUNT; custom_i++) {
				if ((lightmap_surfaces[surface].format >> custom_shift[custom_i]) & RSE::ARRAY_FORMAT_CUSTOM_MASK) {
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
			format |= RSE::ARRAY_FLAG_USE_8_BONE_WEIGHTS;
		} else {
			format &= ~RSE::ARRAY_FLAG_USE_8_BONE_WEIGHTS;
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
	ClassDB::bind_static_method("ImporterMesh", D_METHOD("merge_importer_meshes", "importer_meshes", "relative_transforms", "deduplicate_surfaces"), &ImporterMesh::merge_importer_meshes, DEFVAL(true));
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

	ClassDB::bind_method(D_METHOD("generate_lods", "normal_merge_angle", "normal_split_angle", "bone_transform_array"), &ImporterMesh::_generate_lods_bind);
	ClassDB::bind_method(D_METHOD("get_mesh", "base_mesh"), &ImporterMesh::get_mesh, DEFVAL(Ref<ArrayMesh>()));
	ClassDB::bind_static_method("ImporterMesh", D_METHOD("from_mesh", "mesh"), &ImporterMesh::from_mesh);
	ClassDB::bind_method(D_METHOD("clear"), &ImporterMesh::clear);

	ClassDB::bind_method(D_METHOD("_set_data", "data"), &ImporterMesh::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &ImporterMesh::_get_data);

	ClassDB::bind_method(D_METHOD("set_lightmap_size_hint", "size"), &ImporterMesh::set_lightmap_size_hint);
	ClassDB::bind_method(D_METHOD("get_lightmap_size_hint"), &ImporterMesh::get_lightmap_size_hint);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");
}
