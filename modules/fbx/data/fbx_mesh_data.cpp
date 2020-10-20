/*************************************************************************/
/*  fbx_mesh_data.cpp                                                    */
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

#include "fbx_mesh_data.h"

#include "core/local_vector.h"
#include "scene/resources/mesh.h"
#include "scene/resources/surface_tool.h"

#include "thirdparty/misc/triangulator.h"

template <class T>
T collect_first(const Vector<VertexData<T> > *p_data, T p_fall_back) {
	if (p_data->empty()) {
		return p_fall_back;
	}

	return (*p_data)[0].data;
}

template <class T>
HashMap<int, T> collect_all(const Vector<VertexData<T> > *p_data, HashMap<int, T> p_fall_back) {
	if (p_data->empty()) {
		return p_fall_back;
	}

	HashMap<int, T> collection;
	for (int i = 0; i < p_data->size(); i += 1) {
		const VertexData<T> &vd = (*p_data)[i];
		collection[vd.polygon_index] = vd.data;
	}
	return collection;
}

template <class T>
T collect_average(const Vector<VertexData<T> > *p_data, T p_fall_back) {
	if (p_data->empty()) {
		return p_fall_back;
	}

	T combined = (*p_data)[0].data; // Make sure the data is always correctly initialized.
	print_verbose("size of data: " + itos(p_data->size()));
	for (int i = 1; i < p_data->size(); i += 1) {
		combined += (*p_data)[i].data;
	}
	combined = combined / real_t(p_data->size());

	return combined.normalized();
}

HashMap<int, Vector3> collect_normal(const Vector<VertexData<Vector3> > *p_data, HashMap<int, Vector3> p_fall_back) {
	if (p_data->empty()) {
		return p_fall_back;
	}

	HashMap<int, Vector3> collection;
	for (int i = 0; i < p_data->size(); i += 1) {
		const VertexData<Vector3> &vd = (*p_data)[i];
		collection[vd.polygon_index] = vd.data;
	}
	return collection;
}

HashMap<int, Vector2> collect_uv(const Vector<VertexData<Vector2> > *p_data, HashMap<int, Vector2> p_fall_back) {
	if (p_data->empty()) {
		return p_fall_back;
	}

	HashMap<int, Vector2> collection;
	for (int i = 0; i < p_data->size(); i += 1) {
		const VertexData<Vector2> &vd = (*p_data)[i];
		collection[vd.polygon_index] = vd.data;
	}
	return collection;
}

typedef int Vertex;
typedef int SurfaceId;
typedef int PolygonId;
typedef int DataIndex;

struct SurfaceData {
	Ref<SurfaceTool> surface_tool;
	OrderedHashMap<Vertex, int> lookup_table; // proposed fix is to replace lookup_table[vertex_id] to give the position of the vertices_map[int] index.
	LocalVector<Vertex> vertices_map; // this must be ordered the same as insertion <-- slow to do find() operation.
	Ref<SpatialMaterial> material;
	HashMap<PolygonId, Vector<DataIndex> > surface_polygon_vertex;
	Array morphs;
};

MeshInstance *FBXMeshData::create_fbx_mesh(const ImportState &state, const FBXDocParser::MeshGeometry *mesh_geometry, const FBXDocParser::Model *model) {

	// todo: make this just use a uint64_t FBX ID this is a copy of our original materials unfortunately.
	const std::vector<const FBXDocParser::Material *> &material_lookup = model->GetMaterials();

	std::vector<int> polygon_indices = mesh_geometry->get_polygon_indices();
	std::vector<Vector3> vertices = mesh_geometry->get_vertices();

	// Phase 1. Parse all FBX data.
	HashMap<int, Vector3> normals;
	HashMap<int, HashMap<int, Vector3> > normals_raw = extract_per_vertex_data(
			vertices.size(),
			mesh_geometry->get_edge_map(),
			polygon_indices,
			mesh_geometry->get_normals(),
			&collect_all,
			HashMap<int, Vector3>());

	//	List<int> keys;
	//	normals.get_key_list(&keys);
	//
	//	const std::vector<Assimp::FBX::MeshGeometry::Edge>& edges = mesh_geometry->get_edge_map();
	//	for (int index = 0; index < keys.size(); index++) {
	//		const int key = keys[index];
	//		const int v1 = edges[key].vertex_0;
	//		const int v2 = edges[key].vertex_1;
	//		const Vector3& n1 = normals.get(v1);
	//		const Vector3& n2 = normals.get(v2);
	//		print_verbose("[" + itos(v1) + "] n1: " + n1 + "\n[" + itos(v2) + "] n2: " + n2);
	//		//print_verbose("[" + itos(key) + "] n1: " + n1 + ", n2: " + n2) ;
	//		//print_verbose("vindex: " + itos(edges[key].vertex_0) + ", vindex2: " + itos(edges[key].vertex_1));
	//		//Vector3 ver1 = vertices[edges[key].vertex_0];
	//		//Vector3 ver2 = vertices[edges[key].vertex_1];
	//		/*real_t angle1 = Math::rad2deg(n1.angle_to(n2));
	//		real_t angle2 = Math::rad2deg(n2.angle_to(n1));
	//		print_verbose("angle of normals: " + rtos(angle1) + " angle 2" + rtos(angle2));*/
	//	}

	HashMap<int, Vector2> uvs_0;
	HashMap<int, HashMap<int, Vector2> > uvs_0_raw = extract_per_vertex_data(
			vertices.size(),
			mesh_geometry->get_edge_map(),
			polygon_indices,
			mesh_geometry->get_uv_0(),
			&collect_all,
			HashMap<int, Vector2>());

	HashMap<int, Vector2> uvs_1;
	HashMap<int, HashMap<int, Vector2> > uvs_1_raw = extract_per_vertex_data(
			vertices.size(),
			mesh_geometry->get_edge_map(),
			polygon_indices,
			mesh_geometry->get_uv_1(),
			&collect_all,
			HashMap<int, Vector2>());

	HashMap<int, Color> colors = extract_per_vertex_data(
			vertices.size(),
			mesh_geometry->get_edge_map(),
			polygon_indices,
			mesh_geometry->get_colors(),
			&collect_first,
			Color());

	// TODO what about tangents?
	// TODO what about bi-nomials?
	// TODO there is other?

	HashMap<int, SurfaceId> polygon_surfaces = extract_per_polygon(
			vertices.size(),
			polygon_indices,
			mesh_geometry->get_material_allocation_id(),
			-1);

	HashMap<String, MorphVertexData> morphs;
	extract_morphs(mesh_geometry, morphs);

	// TODO please add skinning.
	//mesh_id = mesh_geometry->ID();

	sanitize_vertex_weights();

	// Re organize polygon vertices to to correctly take into account strange
	// UVs.
	reorganize_vertices(
			polygon_indices,
			vertices,
			normals,
			uvs_0,
			uvs_1,
			colors,
			morphs,
			normals_raw,
			uvs_0_raw,
			uvs_1_raw);

	// Make sure that from this moment on the mesh_geometry is no used anymore.
	// This is a safety step, because the mesh_geometry data are no more valid
	// at this point.
	mesh_geometry = nullptr;

	const int vertex_count = vertices.size();

	// The map key is the material allocator id that is also used as surface id.
	HashMap<SurfaceId, SurfaceData> surfaces;

	// Phase 2. For each material create a surface tool (So a different mesh).
	{
		if (polygon_surfaces.empty()) {
			// No material, just use the default one with index -1.
			// Set -1 to all polygons.
			const int polygon_count = count_polygons(polygon_indices);
			for (int p = 0; p < polygon_count; p += 1) {
				polygon_surfaces[p] = -1;
			}
		}

		// Create the surface now.
		for (const int *polygon_id = polygon_surfaces.next(nullptr); polygon_id != nullptr; polygon_id = polygon_surfaces.next(polygon_id)) {
			const int surface_id = polygon_surfaces[*polygon_id];
			if (surfaces.has(surface_id) == false) {
				SurfaceData sd;
				sd.surface_tool.instance();
				sd.surface_tool->begin(Mesh::PRIMITIVE_TRIANGLES);

				if (surface_id < 0) {
					// nothing to do
				} else if (surface_id < (int)material_lookup.size()) {
					const FBXDocParser::Material *mat_mapping = material_lookup.at(surface_id);
					const uint64_t mapping_id = mat_mapping->ID();
					if (state.cached_materials.has(mapping_id)) {
						sd.material = state.cached_materials[mapping_id];
					}
				} else {
					WARN_PRINT("out of bounds surface detected, FBX file has corrupt material data");
				}

				surfaces.set(surface_id, sd);
			}
		}
	}

	// Phase 3. Map the vertices relative to each surface, in this way we can
	// just insert the vertices that we need per each surface.
	{
		PolygonId polygon_index = -1;
		SurfaceId surface_id = -1;
		SurfaceData *surface_data = nullptr;

		for (size_t polygon_vertex = 0; polygon_vertex < polygon_indices.size(); polygon_vertex += 1) {
			if (is_start_of_polygon(polygon_indices, polygon_vertex)) {
				polygon_index += 1;
				ERR_FAIL_COND_V_MSG(polygon_surfaces.has(polygon_index) == false, nullptr, "The FBX file is currupted, This surface_index is not expected.");
				surface_id = polygon_surfaces[polygon_index];
				surface_data = surfaces.getptr(surface_id);
				CRASH_COND(surface_data == nullptr); // Can't be null.
			}

			const int vertex = get_vertex_from_polygon_vertex(polygon_indices, polygon_vertex);

			// The vertex position in the surface
			// Uses a lookup table for speed with large scenes
			bool has_polygon_vertex_index = surface_data->lookup_table.has(vertex);
			int surface_polygon_vertex_index = -1;

			if (has_polygon_vertex_index) {
				surface_polygon_vertex_index = surface_data->lookup_table[vertex];
			} else {
				surface_polygon_vertex_index = surface_data->vertices_map.size();
				surface_data->lookup_table[vertex] = surface_polygon_vertex_index;
				surface_data->vertices_map.push_back(vertex);
			}

			surface_data->surface_polygon_vertex[polygon_index].push_back(surface_polygon_vertex_index);
		}
	}

	//print_verbose("[debug UV 1] UV1: " + itos(uvs_0.size()));
	//print_verbose("[debug UV 2] UV2: " + itos(uvs_1.size()));

	// Phase 4. Per each surface just insert the vertices and add the indices.
	for (const SurfaceId *surface_id = surfaces.next(nullptr); surface_id != nullptr; surface_id = surfaces.next(surface_id)) {
		SurfaceData *surface = surfaces.getptr(*surface_id);

		// Just add the vertices data.
		for (unsigned int i = 0; i < surface->vertices_map.size(); i += 1) {
			const Vertex vertex = surface->vertices_map[i];

			// This must be done before add_vertex because the surface tool is
			// expecting this before the st->add_vertex() call
			add_vertex(
					surface->surface_tool,
					state.scale,
					vertex,
					vertices,
					normals,
					uvs_0,
					uvs_1,
					colors);
		}

		// Triangulate the various polygons and add the indices.
		for (const PolygonId *polygon_id = surface->surface_polygon_vertex.next(nullptr); polygon_id != nullptr; polygon_id = surface->surface_polygon_vertex.next(polygon_id)) {
			const Vector<DataIndex> *indices = surface->surface_polygon_vertex.getptr(*polygon_id);

			triangulate_polygon(
					surface->surface_tool,
					*indices,
					surface->vertices_map,
					vertices);
		}
	}

	// Phase 5. Compose the morphs if any.
	for (const SurfaceId *surface_id = surfaces.next(nullptr); surface_id != nullptr; surface_id = surfaces.next(surface_id)) {
		SurfaceData *surface = surfaces.getptr(*surface_id);

		for (const String *morph_name = morphs.next(nullptr); morph_name != nullptr; morph_name = morphs.next(morph_name)) {
			MorphVertexData *morph_data = morphs.getptr(*morph_name);

			// As said by the docs, this is not supposed to be different than
			// vertex_count.
			CRASH_COND(morph_data->vertices.size() != vertex_count);
			CRASH_COND(morph_data->normals.size() != vertex_count);

			Vector3 *vertices_ptr = morph_data->vertices.ptrw();
			Vector3 *normals_ptr = morph_data->normals.ptrw();

			Ref<SurfaceTool> morph_st;
			morph_st.instance();
			morph_st->begin(Mesh::PRIMITIVE_TRIANGLES);

			for (unsigned int vi = 0; vi < surface->vertices_map.size(); vi += 1) {
				const Vertex vertex = surface->vertices_map[vi];
				add_vertex(
						morph_st,
						state.scale,
						vertex,
						vertices,
						normals,
						uvs_0,
						uvs_1,
						colors,
						vertices_ptr[vertex],
						normals_ptr[vertex]);
			}

			morph_st->generate_tangents();
			surface->morphs.push_back(morph_st->commit_to_arrays());
		}
	}

	// Phase 6. Compose the mesh and return it.
	Ref<ArrayMesh> mesh;
	mesh.instance();

	// Add blend shape info.
	for (const String *morph_name = morphs.next(nullptr); morph_name != nullptr; morph_name = morphs.next(morph_name)) {
		mesh->add_blend_shape(*morph_name);
	}

	// TODO always normalized, Why?
	mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_NORMALIZED);

	// Add surfaces.
	int in_mesh_surface_id = 0;
	for (const SurfaceId *surface_id = surfaces.next(nullptr); surface_id != nullptr; surface_id = surfaces.next(surface_id)) {
		SurfaceData *surface = surfaces.getptr(*surface_id);

		surface->surface_tool->generate_tangents();

		mesh->add_surface_from_arrays(
				Mesh::PRIMITIVE_TRIANGLES,
				surface->surface_tool->commit_to_arrays(),
				surface->morphs);

		if (surface->material.is_valid()) {
			mesh->surface_set_name(in_mesh_surface_id, surface->material->get_name());
			mesh->surface_set_material(in_mesh_surface_id, surface->material);
		}

		in_mesh_surface_id += 1;
	}

	MeshInstance *godot_mesh = memnew(MeshInstance);
	godot_mesh->set_mesh(mesh);
	return godot_mesh;
}

void FBXMeshData::sanitize_vertex_weights() {
	const int max_bones = VS::ARRAY_WEIGHTS_SIZE;

	for (const Vertex *v = vertex_weights.next(nullptr); v != nullptr; v = vertex_weights.next(v)) {
		VertexWeightMapping *vm = vertex_weights.getptr(*v);
		ERR_CONTINUE(vm->bones.size() != vm->weights.size()); // No message, already checked.
		ERR_CONTINUE(vm->bones_ref.size() != vm->weights.size()); // No message, already checked.

		const int initial_size = vm->weights.size();

		{
			// Init bone id
			int *bones_ptr = vm->bones.ptrw();
			Ref<FBXBone> *bones_ref_ptr = vm->bones_ref.ptrw();

			for (int i = 0; i < vm->weights.size(); i += 1) {
				// At this point this is not possible because the skeleton is already initialized.
				CRASH_COND(bones_ref_ptr[i]->godot_bone_id == -2);
				bones_ptr[i] = bones_ref_ptr[i]->godot_bone_id;
			}

			// From this point on the data is no more valid.
			vm->bones_ref.clear();
		}

		{
			// Sort
			real_t *weights_ptr = vm->weights.ptrw();
			int *bones_ptr = vm->bones.ptrw();
			for (int i = 0; i < vm->weights.size(); i += 1) {
				for (int x = i + 1; x < vm->weights.size(); x += 1) {
					if (weights_ptr[i] < weights_ptr[x]) {
						SWAP(weights_ptr[i], weights_ptr[x]);
						SWAP(bones_ptr[i], bones_ptr[x]);
					}
				}
			}
		}

		{
			// Resize
			vm->weights.resize(max_bones);
			vm->bones.resize(max_bones);
			real_t *weights_ptr = vm->weights.ptrw();
			int *bones_ptr = vm->bones.ptrw();
			for (int i = initial_size; i < max_bones; i += 1) {
				weights_ptr[i] = 0.0;
				bones_ptr[i] = 0;
			}

			// Normalize
			real_t sum = 0.0;
			for (int i = 0; i < max_bones; i += 1) {
				sum += weights_ptr[i];
			}
			if (sum > 0.0) {
				for (int i = 0; i < vm->weights.size(); i += 1) {
					weights_ptr[i] = weights_ptr[i] / sum;
				}
			}
		}
	}
}

void FBXMeshData::reorganize_vertices(
		std::vector<int> &r_polygon_indices,
		std::vector<Vector3> &r_vertices,
		HashMap<int, Vector3> &r_normals,
		HashMap<int, Vector2> &r_uv_1,
		HashMap<int, Vector2> &r_uv_2,
		HashMap<int, Color> &r_color,
		HashMap<String, MorphVertexData> &r_morphs,
		HashMap<int, HashMap<int, Vector3> > &r_normals_raw,
		HashMap<int, HashMap<int, Vector2> > &r_uv_1_raw,
		HashMap<int, HashMap<int, Vector2> > &r_uv_2_raw) {

	// Key: OldVertex; Value: [New vertices];
	HashMap<int, Vector<int> > duplicated_vertices;

	PolygonId polygon_index = -1;
	for (int pv = 0; pv < (int)r_polygon_indices.size(); pv += 1) {
		if (is_start_of_polygon(r_polygon_indices, pv)) {
			polygon_index += 1;
		}
		const Vertex index = get_vertex_from_polygon_vertex(r_polygon_indices, pv);

		bool need_duplication = false;
		Vector2 this_vert_poly_uv1 = Vector2();
		Vector2 this_vert_poly_uv2 = Vector2();
		Vector3 this_vert_poly_normal = Vector3();

		// Take the normal and see if we need to duplicate this polygon.
		if (r_normals_raw.has(index)) {
			const HashMap<PolygonId, Vector3> *nrml_arr = r_normals_raw.getptr(index);
			if (nrml_arr->has(polygon_index)) {
				this_vert_poly_normal = nrml_arr->get(polygon_index);
			}
			// Now, check if we need to duplicate it.
			for (const PolygonId *pid = nrml_arr->next(nullptr); pid != nullptr; pid = nrml_arr->next(pid)) {
				if (*pid == polygon_index) {
					continue;
				}

				const Vector3 vert_poly_normal = *nrml_arr->getptr(*pid);
				if ((this_vert_poly_normal - vert_poly_normal).length_squared() > CMP_EPSILON) {
					// Yes this polygon need duplication.
					need_duplication = true;
					break;
				}
			}
		}

		// Take the UV1 and UV2 and see if we need to duplicate this polygon.
		{
			HashMap<int, HashMap<int, Vector2> > *uv_raw = &r_uv_1_raw;
			Vector2 *this_vert_poly_uv = &this_vert_poly_uv1;
			for (int kk = 0; kk < 2; kk++) {

				if (uv_raw->has(index)) {
					const HashMap<PolygonId, Vector2> *uvs = uv_raw->getptr(index);

					if (uvs->has(polygon_index)) {
						// This Polygon has its own uv.
						(*this_vert_poly_uv) = *uvs->getptr(polygon_index);

						// Check if we need to duplicate it.
						for (const PolygonId *pid = uvs->next(nullptr); pid != nullptr; pid = uvs->next(pid)) {
							if (*pid == polygon_index) {
								continue;
							}
							const Vector2 vert_poly_uv = *uvs->getptr(*pid);
							if (((*this_vert_poly_uv) - vert_poly_uv).length_squared() > CMP_EPSILON) {
								// Yes this polygon need duplication.
								need_duplication = true;
								break;
							}
						}
					} else if (uvs->has(-1)) {
						// It has the default UV.
						(*this_vert_poly_uv) = *uvs->getptr(-1);
					} else if (uvs->size() > 0) {
						// No uv, this is strange, just take the first and duplicate.
						(*this_vert_poly_uv) = *uvs->getptr(*uvs->next(nullptr));
						WARN_PRINT("No UVs for this polygon, while there is no default and some other polygons have it. This FBX file may be corrupted.");
					}
				}
				uv_raw = &r_uv_2_raw;
				this_vert_poly_uv = &this_vert_poly_uv2;
			}
		}

		// If we want to duplicate it, Let's see if we already duplicated this
		// vertex.
		if (need_duplication) {
			if (duplicated_vertices.has(index)) {
				Vertex similar_vertex = -1;
				// Let's see if one of the new vertices has the same data of this.
				const Vector<int> *new_vertices = duplicated_vertices.getptr(index);
				for (int j = 0; j < new_vertices->size(); j += 1) {
					const Vertex new_vertex = (*new_vertices)[j];
					bool same_uv1 = false;
					bool same_uv2 = false;
					bool same_normal = false;

					if (r_uv_1.has(new_vertex)) {
						if ((this_vert_poly_uv1 - (*r_uv_1.getptr(new_vertex))).length_squared() <= CMP_EPSILON) {
							same_uv1 = true;
						}
					}

					if (r_uv_2.has(new_vertex)) {
						if ((this_vert_poly_uv2 - (*r_uv_2.getptr(new_vertex))).length_squared() <= CMP_EPSILON) {
							same_uv2 = true;
						}
					}

					if (r_normals.has(new_vertex)) {
						if ((this_vert_poly_normal - (*r_normals.getptr(new_vertex))).length_squared() <= CMP_EPSILON) {
							same_uv2 = true;
						}
					}

					if (same_uv1 && same_uv2 && same_normal) {
						similar_vertex = new_vertex;
						break;
					}
				}

				if (similar_vertex != -1) {
					// Update polygon.
					if (is_end_of_polygon(r_polygon_indices, pv)) {
						r_polygon_indices[pv] = ~similar_vertex;
					} else {
						r_polygon_indices[pv] = similar_vertex;
					}
					need_duplication = false;
				}
			}
		}

		if (need_duplication) {
			const Vertex old_index = index;
			const Vertex new_index = r_vertices.size();

			// Polygon index.
			if (is_end_of_polygon(r_polygon_indices, pv)) {
				r_polygon_indices[pv] = ~new_index;
			} else {
				r_polygon_indices[pv] = new_index;
			}

			// Vertex position.
			r_vertices.push_back(r_vertices[old_index]);

			// Normals
			if (r_normals_raw.has(old_index)) {
				r_normals.set(new_index, this_vert_poly_normal);
				r_normals_raw.getptr(old_index)->erase(polygon_index);
				r_normals_raw[new_index][polygon_index] = this_vert_poly_normal;
			}

			// UV 0
			if (r_uv_1_raw.has(old_index)) {
				r_uv_1.set(new_index, this_vert_poly_uv1);
				r_uv_1_raw.getptr(old_index)->erase(polygon_index);
				r_uv_1_raw[new_index][polygon_index] = this_vert_poly_uv1;
			}

			// UV 1
			if (r_uv_2_raw.has(old_index)) {
				r_uv_2.set(new_index, this_vert_poly_uv2);
				r_uv_2_raw.getptr(old_index)->erase(polygon_index);
				r_uv_2_raw[new_index][polygon_index] = this_vert_poly_uv2;
			}

			// Vertex color.
			if (r_color.has(old_index)) {
				r_color[new_index] = r_color[old_index];
			}

			// Morphs
			for (const String *mname = r_morphs.next(nullptr); mname != nullptr; mname = r_morphs.next(mname)) {
				MorphVertexData *d = r_morphs.getptr(*mname);
				// This can't never happen.
				CRASH_COND(d == nullptr);
				if (d->vertices.size() > old_index) {
					d->vertices.push_back(d->vertices[old_index]);
				}
				if (d->normals.size() > old_index) {
					d->normals.push_back(d->normals[old_index]);
				}
			}

			if (vertex_weights.has(old_index)) {
				vertex_weights.set(new_index, vertex_weights[old_index]);
			}

			duplicated_vertices[old_index].push_back(new_index);
		} else {
			if (r_normals_raw.has(index) &&
					r_normals.has(index) == false) {
				r_normals.set(index, this_vert_poly_normal);
			}

			if (r_uv_1_raw.has(index) &&
					r_uv_1.has(index) == false) {
				r_uv_1.set(index, this_vert_poly_uv1);
			}

			if (r_uv_2_raw.has(index) &&
					r_uv_2.has(index) == false) {
				r_uv_2.set(index, this_vert_poly_uv2);
			}
		}
	}
}

void FBXMeshData::add_vertex(
		Ref<SurfaceTool> p_surface_tool,
		real_t p_scale,
		Vertex p_vertex,
		const std::vector<Vector3> &p_vertices_position,
		const HashMap<int, Vector3> &p_normals,
		const HashMap<int, Vector2> &p_uvs_0,
		const HashMap<int, Vector2> &p_uvs_1,
		const HashMap<int, Color> &p_colors,
		const Vector3 &p_morph_value,
		const Vector3 &p_morph_normal) {

	ERR_FAIL_INDEX_MSG(p_vertex, (Vertex)p_vertices_position.size(), "FBX file is corrupted, the position of the vertex can't be retrieved.");

	if (p_normals.has(p_vertex)) {
		p_surface_tool->add_normal(p_normals[p_vertex] + p_morph_normal);
	}

	if (p_uvs_0.has(p_vertex)) {
		//print_verbose("uv1: [" + itos(p_vertex) + "] " + p_uvs_0[p_vertex]);
		// Inverts Y UV.
		p_surface_tool->add_uv(Vector2(p_uvs_0[p_vertex].x, 1 - p_uvs_0[p_vertex].y));
	}

	if (p_uvs_1.has(p_vertex)) {
		//print_verbose("uv2: [" + itos(p_vertex) + "] " + p_uvs_1[p_vertex]);
		// Inverts Y UV.
		p_surface_tool->add_uv2(Vector2(p_uvs_1[p_vertex].x, 1 - p_uvs_1[p_vertex].y));
	}

	if (p_colors.has(p_vertex)) {
		p_surface_tool->add_color(p_colors[p_vertex]);
	}

	// TODO what about binormals?
	// TODO there is other?

	gen_weight_info(p_surface_tool, p_vertex);

	// The surface tool want the vertex position as last thing.
	p_surface_tool->add_vertex((p_vertices_position[p_vertex] + p_morph_value) * p_scale);
}

void FBXMeshData::triangulate_polygon(Ref<SurfaceTool> st, Vector<int> p_polygon_vertex, const Vector<Vertex> p_surface_vertex_map, const std::vector<Vector3> &p_vertices) const {
	const int polygon_vertex_count = p_polygon_vertex.size();
	if (polygon_vertex_count == 1) {
		// point to triangle
		st->add_index(p_polygon_vertex[0]);
		st->add_index(p_polygon_vertex[0]);
		st->add_index(p_polygon_vertex[0]);
		return;
	} else if (polygon_vertex_count == 2) {
		// line to triangle
		st->add_index(p_polygon_vertex[1]);
		st->add_index(p_polygon_vertex[1]);
		st->add_index(p_polygon_vertex[0]);
		return;
	} else if (polygon_vertex_count == 3) {
		// triangle to triangle
		st->add_index(p_polygon_vertex[0]);
		st->add_index(p_polygon_vertex[2]);
		st->add_index(p_polygon_vertex[1]);
		return;
	} else if (polygon_vertex_count == 4) {
		// quad to triangle - this code is awesome for import times
		// it prevents triangles being generated slowly
		st->add_index(p_polygon_vertex[0]);
		st->add_index(p_polygon_vertex[2]);
		st->add_index(p_polygon_vertex[1]);
		st->add_index(p_polygon_vertex[2]);
		st->add_index(p_polygon_vertex[0]);
		st->add_index(p_polygon_vertex[3]);
		return;
	} else {
		// non triangulated - we must run the triangulation algorithm
		bool is_simple_convex = false;
		// this code is 'slow' but required it triangulates all the unsupported geometry.
		// Doesn't allow for bigger polygons because those are unlikely be convex
		if (polygon_vertex_count <= 6) {
			// Start from true, check if it's false.
			is_simple_convex = true;
			Vector3 first_vec;
			for (int i = 0; i < polygon_vertex_count; i += 1) {
				const Vector3 p1 = p_vertices[p_surface_vertex_map[p_polygon_vertex[i]]];
				const Vector3 p2 = p_vertices[p_surface_vertex_map[p_polygon_vertex[(i + 1) % polygon_vertex_count]]];
				const Vector3 p3 = p_vertices[p_surface_vertex_map[p_polygon_vertex[(i + 2) % polygon_vertex_count]]];

				const Vector3 edge1 = p1 - p2;
				const Vector3 edge2 = p3 - p2;

				const Vector3 res = edge1.normalized().cross(edge2.normalized()).normalized();
				if (i == 0) {
					first_vec = res;
				} else {
					if (first_vec.dot(res) < 0.0) {
						// Ok we found an angle that is not the same dir of the
						// others.
						is_simple_convex = false;
						break;
					}
				}
			}
		}

		if (is_simple_convex) {
			// This is a convex polygon, so just triangulate it.
			for (int i = 0; i < (polygon_vertex_count - 2); i += 1) {
				st->add_index(p_polygon_vertex[2 + i]);
				st->add_index(p_polygon_vertex[1 + i]);
				st->add_index(p_polygon_vertex[0]);
			}
			return;
		}
	}

	{
		// This is a concave polygon.

		std::vector<Vector3> poly_vertices(polygon_vertex_count);
		for (int i = 0; i < polygon_vertex_count; i += 1) {
			poly_vertices[i] = p_vertices[p_surface_vertex_map[p_polygon_vertex[i]]];
		}

		const Vector3 poly_norm = get_poly_normal(poly_vertices);
		if (poly_norm.length_squared() <= CMP_EPSILON) {
			ERR_FAIL_COND_MSG(poly_norm.length_squared() <= CMP_EPSILON, "The normal of this poly was not computed. Is this FBX file corrupted.");
		}

		// Select the plan coordinate.
		int axis_1_coord = 0;
		int axis_2_coord = 1;
		{
			real_t inv = poly_norm.z;

			const real_t axis_x = ABS(poly_norm.x);
			const real_t axis_y = ABS(poly_norm.y);
			const real_t axis_z = ABS(poly_norm.z);

			if (axis_x > axis_y) {
				if (axis_x > axis_z) {
					// For the most part the normal point toward X.
					axis_1_coord = 1;
					axis_2_coord = 2;
					inv = poly_norm.x;
				}
			} else if (axis_y > axis_z) {
				// For the most part the normal point toward Y.
				axis_1_coord = 2;
				axis_2_coord = 0;
				inv = poly_norm.y;
			}

			// Swap projection axes to take the negated projection vector into account
			if (inv < 0.0f) {
				SWAP(axis_1_coord, axis_2_coord);
			}
		}

		TriangulatorPoly triangulator_poly;
		triangulator_poly.Init(polygon_vertex_count);
		std::vector<Vector2> projected_vertices(polygon_vertex_count);
		for (int i = 0; i < polygon_vertex_count; i += 1) {
			const Vector2 pv(poly_vertices[i][axis_1_coord], poly_vertices[i][axis_2_coord]);
			projected_vertices[i] = pv;
			triangulator_poly.GetPoint(i) = pv;
		}
		triangulator_poly.SetOrientation(TRIANGULATOR_CCW);

		List<TriangulatorPoly> out_poly;

		TriangulatorPartition triangulator_partition;
		if (triangulator_partition.Triangulate_OPT(&triangulator_poly, &out_poly) == 0) { // Good result.
			if (triangulator_partition.Triangulate_EC(&triangulator_poly, &out_poly) == 0) { // Medium result.
				if (triangulator_partition.Triangulate_MONO(&triangulator_poly, &out_poly) == 0) { // Really poor result.
					ERR_FAIL_MSG("The triangulation of this polygon failed, please try to triangulate your mesh or check if it has broken polygons.");
				}
			}
		}

		std::vector<Vector2> tris(out_poly.size());
		for (List<TriangulatorPoly>::Element *I = out_poly.front(); I; I = I->next()) {
			TriangulatorPoly &tp = I->get();

			ERR_FAIL_COND_MSG(tp.GetNumPoints() != 3, "The triangulator retuned more points, how this is possible?");
			// Find Index
			for (int i = 2; i >= 0; i -= 1) {
				const Vector2 vertex = tp.GetPoint(i);
				bool done = false;
				// Find Index
				for (int y = 0; y < polygon_vertex_count; y += 1) {
					if ((projected_vertices[y] - vertex).length_squared() <= CMP_EPSILON) {
						// This seems the right vertex
						st->add_index(p_polygon_vertex[y]);
						done = true;
						break;
					}
				}
				ERR_FAIL_COND(done == false);
			}
		}
	}
}

void FBXMeshData::gen_weight_info(Ref<SurfaceTool> st, Vertex vertex_id) const {
	if (vertex_weights.empty()) {
		return;
	}

	if (vertex_weights.has(vertex_id)) {
		// Let's extract the weight info.
		const VertexWeightMapping *vm = vertex_weights.getptr(vertex_id);
		st->add_weights(vm->weights);
		st->add_bones(vm->bones);
		print_verbose("[doc] Triangle added weights to mesh for bones");
	} else {
		// This vertex doesn't have any bone info, while the model is using the
		// bones.
		// So nothing more to do.
	}

	print_verbose("[doc] Triangle added weights to mesh for bones");
}

int FBXMeshData::get_vertex_from_polygon_vertex(const std::vector<int> &p_polygon_indices, int p_index) const {
	if (p_index < 0 || p_index >= (int)p_polygon_indices.size()) {
		return -1;
	}

	const int vertex = p_polygon_indices[p_index];
	if (vertex >= 0) {
		return vertex;
	} else {
		// Negative numbers are the end of the face, reversing the bits is
		// possible to obtain the positive correct vertex number.
		return ~vertex;
	}
}

bool FBXMeshData::is_end_of_polygon(const std::vector<int> &p_polygon_indices, int p_index) const {
	if (p_index < 0 || p_index >= (int)p_polygon_indices.size()) {
		return false;
	}

	const int vertex = p_polygon_indices[p_index];

	// If the index is negative this is the end of the Polygon.
	return vertex < 0;
}

bool FBXMeshData::is_start_of_polygon(const std::vector<int> &p_polygon_indices, int p_index) const {
	if (p_index < 0 || p_index >= (int)p_polygon_indices.size()) {
		return false;
	}

	if (p_index == 0) {
		return true;
	}

	// If the previous indices is negative this is the begin of a new Polygon.
	return p_polygon_indices[p_index - 1] < 0;
}

int FBXMeshData::count_polygons(const std::vector<int> &p_polygon_indices) const {
	// The negative numbers define the end of the polygon. Counting the amount of
	// negatives the numbers of polygons are obtained.
	int count = 0;
	for (size_t i = 0; i < p_polygon_indices.size(); i += 1) {
		if (p_polygon_indices[i] < 0) {
			count += 1;
		}
	}
	return count;
}

template <class R, class T>
HashMap<int, R> FBXMeshData::extract_per_vertex_data(
		int p_vertex_count,
		const std::vector<FBXDocParser::MeshGeometry::Edge> &p_edge_map,
		const std::vector<int> &p_mesh_indices,
		const FBXDocParser::MeshGeometry::MappingData<T> &p_mapping_data,
		R (*collector_function)(const Vector<VertexData<T> > *p_vertex_data, R p_fall_back),
		R p_fall_back) const {

	/* When index_to_direct is set
	 * index size is 184 ( contains index for the data array [values 0, 96] )
	 * data size is 96 (contains uv coordinates)
	 * this means index is simple data reduction basically
	 */

	ERR_FAIL_COND_V_MSG(p_mapping_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::index_to_direct && p_mapping_data.index.size() == 0, (HashMap<int, R>()), "FBX file is missing indexing array");
	ERR_FAIL_COND_V_MSG(p_mapping_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::index && p_mapping_data.index.size() == 0, (HashMap<int, R>()), "The FBX seems corrupted");

	// Aggregate vertex data.
	HashMap<Vertex, Vector<VertexData<T> > > aggregate_vertex_data;

	switch (p_mapping_data.map_type) {
		case FBXDocParser::MeshGeometry::MapType::none: {
			// No data nothing to do.
			return (HashMap<int, R>());
		}
		case FBXDocParser::MeshGeometry::MapType::vertex: {
			ERR_FAIL_COND_V_MSG(p_mapping_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::index_to_direct, (HashMap<int, R>()), "We will support in future");

			if (p_mapping_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::direct) {
				// The data is mapped per vertex directly.
				ERR_FAIL_COND_V_MSG((int)p_mapping_data.data.size() != p_vertex_count, (HashMap<int, R>()), "FBX file corrupted: #ERR01");
				for (size_t vertex_index = 0; vertex_index < p_mapping_data.data.size(); vertex_index += 1) {
					aggregate_vertex_data[vertex_index].push_back({ -1, p_mapping_data.data[vertex_index] });
				}
			} else {
				// The data is mapped per vertex using a reference.
				// The indices array, contains a *reference_id for each vertex.
				// * Note that the reference_id is the id of data into the data array.
				//
				// https://help.autodesk.com/view/FBX/2017/ENU/?guid=__cpp_ref_class_fbx_layer_element_html
				ERR_FAIL_COND_V_MSG((int)p_mapping_data.index.size() != p_vertex_count, (HashMap<int, R>()), "FBX file corrupted: #ERR02");
				for (size_t vertex_index = 0; vertex_index < p_mapping_data.index.size(); vertex_index += 1) {
					ERR_FAIL_INDEX_V_MSG(p_mapping_data.index[vertex_index], (int)p_mapping_data.data.size(), (HashMap<int, R>()), "FBX file seems corrupted: #ERR03.")
					aggregate_vertex_data[vertex_index].push_back({ -1, p_mapping_data.data[p_mapping_data.index[vertex_index]] });
				}
			}
		} break;
		case FBXDocParser::MeshGeometry::MapType::polygon_vertex: {
			if (p_mapping_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::index_to_direct) {
				// The data is mapped using each index from the indexes array then direct to the data (data reduction algorithm)
				ERR_FAIL_COND_V_MSG((int)p_mesh_indices.size() != (int)p_mapping_data.index.size(), (HashMap<int, R>()), "FBX file seems corrupted: #ERR04");
				int polygon_id = -1;
				for (size_t polygon_vertex_index = 0; polygon_vertex_index < p_mapping_data.index.size(); polygon_vertex_index += 1) {
					if (is_start_of_polygon(p_mesh_indices, polygon_vertex_index)) {
						polygon_id += 1;
					}
					const int vertex_index = get_vertex_from_polygon_vertex(p_mesh_indices, polygon_vertex_index);
					ERR_FAIL_COND_V_MSG(vertex_index < 0, (HashMap<int, R>()), "FBX file corrupted: #ERR05");
					ERR_FAIL_COND_V_MSG(vertex_index >= p_vertex_count, (HashMap<int, R>()), "FBX file corrupted: #ERR06");
					const int index_to_direct = p_mapping_data.index[polygon_vertex_index];
					T value = p_mapping_data.data[index_to_direct];
					aggregate_vertex_data[vertex_index].push_back({ polygon_id, value });
				}
			} else if (p_mapping_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::direct) {
				// The data are mapped per polygon vertex directly.
				ERR_FAIL_COND_V_MSG((int)p_mesh_indices.size() != (int)p_mapping_data.data.size(), (HashMap<int, R>()), "FBX file seems corrupted: #ERR04");
				int polygon_id = -1;
				for (size_t polygon_vertex_index = 0; polygon_vertex_index < p_mapping_data.data.size(); polygon_vertex_index += 1) {
					if (is_start_of_polygon(p_mesh_indices, polygon_vertex_index)) {
						polygon_id += 1;
					}
					const int vertex_index = get_vertex_from_polygon_vertex(p_mesh_indices, polygon_vertex_index);
					ERR_FAIL_COND_V_MSG(vertex_index < 0, (HashMap<int, R>()), "FBX file corrupted: #ERR05");
					ERR_FAIL_COND_V_MSG(vertex_index >= p_vertex_count, (HashMap<int, R>()), "FBX file corrupted: #ERR06");

					aggregate_vertex_data[vertex_index].push_back({ polygon_id, p_mapping_data.data[polygon_vertex_index] });
				}
			} else {
				// The data is mapped per polygon_vertex using a reference.
				// The indices array, contains a *reference_id for each polygon_vertex.
				// * Note that the reference_id is the id of data into the data array.
				//
				// https://help.autodesk.com/view/FBX/2017/ENU/?guid=__cpp_ref_class_fbx_layer_element_html
				ERR_FAIL_COND_V_MSG(p_mesh_indices.size() != p_mapping_data.index.size(), (HashMap<int, R>()), "FBX file corrupted: #ERR7");
				int polygon_id = -1;
				for (size_t polygon_vertex_index = 0; polygon_vertex_index < p_mapping_data.index.size(); polygon_vertex_index += 1) {
					if (is_start_of_polygon(p_mesh_indices, polygon_vertex_index)) {
						polygon_id += 1;
					}
					const int vertex_index = get_vertex_from_polygon_vertex(p_mesh_indices, polygon_vertex_index);
					ERR_FAIL_COND_V_MSG(vertex_index < 0, (HashMap<int, R>()), "FBX file corrupted: #ERR8");
					ERR_FAIL_COND_V_MSG(vertex_index >= p_vertex_count, (HashMap<int, R>()), "FBX file seems  corrupted: #ERR9.")
					ERR_FAIL_COND_V_MSG(p_mapping_data.index[polygon_vertex_index] < 0, (HashMap<int, R>()), "FBX file seems  corrupted: #ERR10.")
					ERR_FAIL_COND_V_MSG(p_mapping_data.index[polygon_vertex_index] >= (int)p_mapping_data.data.size(), (HashMap<int, R>()), "FBX file seems corrupted: #ERR11.")
					aggregate_vertex_data[vertex_index].push_back({ polygon_id, p_mapping_data.data[p_mapping_data.index[polygon_vertex_index]] });
				}
			}
		} break;
		case FBXDocParser::MeshGeometry::MapType::polygon: {
			if (p_mapping_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::direct) {
				// The data are mapped per polygon directly.
				const int polygon_count = count_polygons(p_mesh_indices);
				ERR_FAIL_COND_V_MSG(polygon_count != (int)p_mapping_data.data.size(), (HashMap<int, R>()), "FBX file seems corrupted: #ERR12");

				// Advance each polygon vertex, each new polygon advance the polygon index.
				int polygon_index = -1;
				for (size_t polygon_vertex_index = 0;
						polygon_vertex_index < p_mesh_indices.size();
						polygon_vertex_index += 1) {

					if (is_start_of_polygon(p_mesh_indices, polygon_vertex_index)) {
						polygon_index += 1;
						ERR_FAIL_INDEX_V_MSG(polygon_index, (int)p_mapping_data.data.size(), (HashMap<int, R>()), "FBX file seems corrupted: #ERR13");
					}

					const int vertex_index = get_vertex_from_polygon_vertex(p_mesh_indices, polygon_vertex_index);
					ERR_FAIL_INDEX_V_MSG(vertex_index, p_vertex_count, (HashMap<int, R>()), "FBX file corrupted: #ERR14");

					aggregate_vertex_data[vertex_index].push_back({ polygon_index, p_mapping_data.data[polygon_index] });
				}
				ERR_FAIL_COND_V_MSG((polygon_index + 1) != polygon_count, (HashMap<int, R>()), "FBX file seems corrupted: #ERR16. Not all Polygons are present in the file.")
			} else {
				// The data is mapped per polygon using a reference.
				// The indices array, contains a *reference_id for each polygon.
				// * Note that the reference_id is the id of data into the data array.
				//
				// https://help.autodesk.com/view/FBX/2017/ENU/?guid=__cpp_ref_class_fbx_layer_element_html
				const int polygon_count = count_polygons(p_mesh_indices);
				ERR_FAIL_COND_V_MSG(polygon_count != (int)p_mapping_data.index.size(), (HashMap<int, R>()), "FBX file seems corrupted: #ERR17");

				// Advance each polygon vertex, each new polygon advance the polygon index.
				int polygon_index = -1;
				for (size_t polygon_vertex_index = 0;
						polygon_vertex_index < p_mesh_indices.size();
						polygon_vertex_index += 1) {

					if (is_start_of_polygon(p_mesh_indices, polygon_vertex_index)) {
						polygon_index += 1;
						ERR_FAIL_INDEX_V_MSG(polygon_index, (int)p_mapping_data.index.size(), (HashMap<int, R>()), "FBX file seems corrupted: #ERR18");
						ERR_FAIL_INDEX_V_MSG(p_mapping_data.index[polygon_index], (int)p_mapping_data.data.size(), (HashMap<int, R>()), "FBX file seems corrupted: #ERR19");
					}

					const int vertex_index = get_vertex_from_polygon_vertex(p_mesh_indices, polygon_vertex_index);
					ERR_FAIL_INDEX_V_MSG(vertex_index, p_vertex_count, (HashMap<int, R>()), "FBX file corrupted: #ERR20");

					aggregate_vertex_data[vertex_index].push_back({ polygon_index, p_mapping_data.data[p_mapping_data.index[polygon_index]] });
				}
				ERR_FAIL_COND_V_MSG((polygon_index + 1) != polygon_count, (HashMap<int, R>()), "FBX file seems corrupted: #ERR22. Not all Polygons are present in the file.")
			}
		} break;
		case FBXDocParser::MeshGeometry::MapType::edge: {
			if (p_mapping_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::direct) {
				// The data are mapped per edge directly.
				ERR_FAIL_COND_V_MSG(p_edge_map.size() != p_mapping_data.data.size(), (HashMap<int, R>()), "FBX file seems corrupted: #ERR23");
				for (size_t edge_index = 0; edge_index < p_mapping_data.data.size(); edge_index += 1) {
					const FBXDocParser::MeshGeometry::Edge edge = FBXDocParser::MeshGeometry::get_edge(p_edge_map, edge_index);
					ERR_FAIL_INDEX_V_MSG(edge.vertex_0, p_vertex_count, (HashMap<int, R>()), "FBX file corrupted: #ERR24");
					ERR_FAIL_INDEX_V_MSG(edge.vertex_1, p_vertex_count, (HashMap<int, R>()), "FBX file corrupted: #ERR25");
					ERR_FAIL_INDEX_V_MSG(edge.vertex_0, (int)p_mapping_data.data.size(), (HashMap<int, R>()), "FBX file corrupted: #ERR26");
					ERR_FAIL_INDEX_V_MSG(edge.vertex_1, (int)p_mapping_data.data.size(), (HashMap<int, R>()), "FBX file corrupted: #ERR27");
					aggregate_vertex_data[edge.vertex_0].push_back({ -1, p_mapping_data.data[edge_index] });
					aggregate_vertex_data[edge.vertex_1].push_back({ -1, p_mapping_data.data[edge_index] });
				}
			} else {
				// The data is mapped per edge using a reference.
				// The indices array, contains a *reference_id for each polygon.
				// * Note that the reference_id is the id of data into the data array.
				//
				// https://help.autodesk.com/view/FBX/2017/ENU/?guid=__cpp_ref_class_fbx_layer_element_html
				ERR_FAIL_COND_V_MSG(p_edge_map.size() != p_mapping_data.index.size(), (HashMap<int, R>()), "FBX file seems corrupted: #ERR28");
				for (size_t edge_index = 0; edge_index < p_mapping_data.data.size(); edge_index += 1) {
					const FBXDocParser::MeshGeometry::Edge edge = FBXDocParser::MeshGeometry::get_edge(p_edge_map, edge_index);
					ERR_FAIL_INDEX_V_MSG(edge.vertex_0, p_vertex_count, (HashMap<int, R>()), "FBX file corrupted: #ERR29");
					ERR_FAIL_INDEX_V_MSG(edge.vertex_1, p_vertex_count, (HashMap<int, R>()), "FBX file corrupted: #ERR30");
					ERR_FAIL_INDEX_V_MSG(edge.vertex_0, (int)p_mapping_data.index.size(), (HashMap<int, R>()), "FBX file corrupted: #ERR31");
					ERR_FAIL_INDEX_V_MSG(edge.vertex_1, (int)p_mapping_data.index.size(), (HashMap<int, R>()), "FBX file corrupted: #ERR32");
					ERR_FAIL_INDEX_V_MSG(p_mapping_data.index[edge.vertex_0], (int)p_mapping_data.data.size(), (HashMap<int, R>()), "FBX file corrupted: #ERR33");
					ERR_FAIL_INDEX_V_MSG(p_mapping_data.index[edge.vertex_1], (int)p_mapping_data.data.size(), (HashMap<int, R>()), "FBX file corrupted: #ERR34");
					aggregate_vertex_data[edge.vertex_0].push_back({ -1, p_mapping_data.data[p_mapping_data.index[edge_index]] });
					aggregate_vertex_data[edge.vertex_1].push_back({ -1, p_mapping_data.data[p_mapping_data.index[edge_index]] });
				}
			}
		} break;
		case FBXDocParser::MeshGeometry::MapType::all_the_same: {
			// No matter the mode, no matter the data size; The first always win
			// and is set to all the vertices.
			ERR_FAIL_COND_V_MSG(p_mapping_data.data.size() <= 0, (HashMap<int, R>()), "FBX file seems corrupted: #ERR35");
			if (p_mapping_data.data.size() > 0) {
				for (int vertex_index = 0; vertex_index < p_vertex_count; vertex_index += 1) {
					aggregate_vertex_data[vertex_index].push_back({ -1, p_mapping_data.data[0] });
				}
			}
		} break;
	}

	if (aggregate_vertex_data.size() == 0) {
		return (HashMap<int, R>());
	}

	// A map is used because turns out that the some FBX file are not well organized
	// with vertices well compacted. Using a map allows avoid those issues.
	HashMap<Vertex, R> result;

	// Aggregate the collected data.
	for (const Vertex *index = aggregate_vertex_data.next(nullptr); index != nullptr; index = aggregate_vertex_data.next(index)) {
		Vector<VertexData<T> > *aggregated_vertex = aggregate_vertex_data.getptr(*index);
		// This can't be null because we are just iterating.
		CRASH_COND(aggregated_vertex == nullptr);

		ERR_FAIL_INDEX_V_MSG(0, aggregated_vertex->size(), (HashMap<int, R>()), "The FBX file is corrupted, No valid data for this vertex index.");
		result[*index] = collector_function(aggregated_vertex, p_fall_back);
	}

	// Sanitize the data now, if the file is broken we can try import it anyway.
	bool problem_found = false;
	for (size_t i = 0; i < p_mesh_indices.size(); i += 1) {
		const Vertex vertex = get_vertex_from_polygon_vertex(p_mesh_indices, i);
		if (result.has(vertex) == false) {
			result[vertex] = p_fall_back;
			problem_found = true;
		}
	}
	if (problem_found) {
		WARN_PRINT("Some data is missing, this FBX file may be corrupted: #WARN0.");
	}

	return result;
}

template <class T>
HashMap<int, T> FBXMeshData::extract_per_polygon(
		int p_vertex_count,
		const std::vector<int> &p_polygon_indices,
		const FBXDocParser::MeshGeometry::MappingData<T> &p_fbx_data,
		T p_fallback_value) const {

	ERR_FAIL_COND_V_MSG(p_fbx_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::index_to_direct && p_fbx_data.data.size() == 0, (HashMap<int, T>()), "invalid index to direct array");
	ERR_FAIL_COND_V_MSG(p_fbx_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::index && p_fbx_data.index.size() == 0, (HashMap<int, T>()), "The FBX seems corrupted");

	const int polygon_count = count_polygons(p_polygon_indices);

	// Aggregate vertex data.
	HashMap<int, Vector<T> > aggregate_polygon_data;

	switch (p_fbx_data.map_type) {
		case FBXDocParser::MeshGeometry::MapType::none: {
			// No data nothing to do.
			return (HashMap<int, T>());
		}
		case FBXDocParser::MeshGeometry::MapType::vertex: {
			ERR_FAIL_V_MSG((HashMap<int, T>()), "This data can't be extracted and organized per polygon, since into the FBX is mapped per vertex. This should not happen.");
		} break;
		case FBXDocParser::MeshGeometry::MapType::polygon_vertex: {
			ERR_FAIL_V_MSG((HashMap<int, T>()), "This data can't be extracted and organized per polygon, since into the FBX is mapped per polygon vertex. This should not happen.");
		} break;
		case FBXDocParser::MeshGeometry::MapType::polygon: {
			if (p_fbx_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::index_to_direct) {
				// The data is stored efficiently index_to_direct allows less data in the FBX file.
				for (int polygon_index = 0;
						polygon_index < polygon_count;
						polygon_index += 1) {

					if (p_fbx_data.index.size() == 0) {
						ERR_FAIL_INDEX_V_MSG(polygon_index, (int)p_fbx_data.data.size(), (HashMap<int, T>()), "FBX file is corrupted: #ERR62");
						aggregate_polygon_data[polygon_index].push_back(p_fbx_data.data[polygon_index]);
					} else {
						ERR_FAIL_INDEX_V_MSG(polygon_index, (int)p_fbx_data.index.size(), (HashMap<int, T>()), "FBX file is corrupted: #ERR62");

						const int index_to_direct = p_fbx_data.index[polygon_index];
						T value = p_fbx_data.data[index_to_direct];
						aggregate_polygon_data[polygon_index].push_back(value);
					}
				}
			} else if (p_fbx_data.ref_type == FBXDocParser::MeshGeometry::ReferenceType::direct) {
				// The data are mapped per polygon directly.
				ERR_FAIL_COND_V_MSG(polygon_count != (int)p_fbx_data.data.size(), (HashMap<int, T>()), "FBX file is corrupted: #ERR51");

				// Advance each polygon vertex, each new polygon advance the polygon index.
				for (int polygon_index = 0;
						polygon_index < polygon_count;
						polygon_index += 1) {

					ERR_FAIL_INDEX_V_MSG(polygon_index, (int)p_fbx_data.data.size(), (HashMap<int, T>()), "FBX file is corrupted: #ERR52");
					aggregate_polygon_data[polygon_index].push_back(p_fbx_data.data[polygon_index]);
				}
			} else {
				// The data is mapped per polygon using a reference.
				// The indices array, contains a *reference_id for each polygon.
				// * Note that the reference_id is the id of data into the data array.
				//
				// https://help.autodesk.com/view/FBX/2017/ENU/?guid=__cpp_ref_class_fbx_layer_element_html
				ERR_FAIL_COND_V_MSG(polygon_count != (int)p_fbx_data.index.size(), (HashMap<int, T>()), "FBX file seems corrupted: #ERR52");

				// Advance each polygon vertex, each new polygon advance the polygon index.
				for (int polygon_index = 0;
						polygon_index < polygon_count;
						polygon_index += 1) {

					ERR_FAIL_INDEX_V_MSG(polygon_index, (int)p_fbx_data.index.size(), (HashMap<int, T>()), "FBX file is corrupted: #ERR53");
					ERR_FAIL_INDEX_V_MSG(p_fbx_data.index[polygon_index], (int)p_fbx_data.data.size(), (HashMap<int, T>()), "FBX file is corrupted: #ERR54");
					aggregate_polygon_data[polygon_index].push_back(p_fbx_data.data[p_fbx_data.index[polygon_index]]);
				}
			}
		} break;
		case FBXDocParser::MeshGeometry::MapType::edge: {
			ERR_FAIL_V_MSG((HashMap<int, T>()), "This data can't be extracted and organized per polygon, since into the FBX is mapped per edge. This should not happen.");
		} break;
		case FBXDocParser::MeshGeometry::MapType::all_the_same: {
			// No matter the mode, no matter the data size; The first always win
			// and is set to all the vertices.
			ERR_FAIL_COND_V_MSG(p_fbx_data.data.size() <= 0, (HashMap<int, T>()), "FBX file seems corrupted: #ERR55");
			if (p_fbx_data.data.size() > 0) {
				for (int polygon_index = 0; polygon_index < polygon_count; polygon_index += 1) {
					aggregate_polygon_data[polygon_index].push_back(p_fbx_data.data[0]);
				}
			}
		} break;
	}

	if (aggregate_polygon_data.size() == 0) {
		return (HashMap<int, T>());
	}

	// A map is used because turns out that the some FBX file are not well organized
	// with vertices well compacted. Using a map allows avoid those issues.
	HashMap<int, T> polygons;

	// Take the first value for each vertex.
	for (const Vertex *index = aggregate_polygon_data.next(nullptr); index != nullptr; index = aggregate_polygon_data.next(index)) {
		Vector<T> *aggregated_polygon = aggregate_polygon_data.getptr(*index);
		// This can't be null because we are just iterating.
		CRASH_COND(aggregated_polygon == nullptr);

		ERR_FAIL_INDEX_V_MSG(0, (int)aggregated_polygon->size(), (HashMap<int, T>()), "The FBX file is corrupted, No valid data for this polygon index.");

		// Validate the final value.
		polygons[*index] = (*aggregated_polygon)[0];
	}

	// Sanitize the data now, if the file is broken we can try import it anyway.
	bool problem_found = false;
	for (int polygon_i = 0; polygon_i < polygon_count; polygon_i += 1) {
		if (polygons.has(polygon_i) == false) {
			polygons[polygon_i] = p_fallback_value;
			problem_found = true;
		}
	}
	if (problem_found) {
		WARN_PRINT("Some data is missing, this FBX file may be corrupted: #WARN1.");
	}

	return polygons;
}

void FBXMeshData::extract_morphs(const FBXDocParser::MeshGeometry *mesh_geometry, HashMap<String, MorphVertexData> &r_data) {

	r_data.clear();

	const int vertex_count = mesh_geometry->get_vertices().size();

	for (const FBXDocParser::BlendShape *blend_shape : mesh_geometry->get_blend_shapes()) {
		for (const FBXDocParser::BlendShapeChannel *blend_shape_channel : blend_shape->BlendShapeChannels()) {
			const std::vector<const FBXDocParser::ShapeGeometry *> &shape_geometries = blend_shape_channel->GetShapeGeometries();
			for (const FBXDocParser::ShapeGeometry *shape_geometry : shape_geometries) {

				String morph_name = ImportUtils::FBXAnimMeshName(shape_geometry->Name()).c_str();
				if (morph_name.empty()) {
					morph_name = "morph";
				}

				// TODO we have only these??
				const std::vector<unsigned int> &morphs_vertex_indices = shape_geometry->GetIndices();
				const std::vector<Vector3> &morphs_vertices = shape_geometry->GetVertices();
				const std::vector<Vector3> &morphs_normals = shape_geometry->GetNormals();

				ERR_FAIL_COND_MSG((int)morphs_vertex_indices.size() > vertex_count, "The FBX file is corrupted: #ERR103");
				ERR_FAIL_COND_MSG(morphs_vertex_indices.size() != morphs_vertices.size(), "The FBX file is corrupted: #ERR104");
				ERR_FAIL_COND_MSG((int)morphs_vertices.size() > vertex_count, "The FBX file is corrupted: #ERR105");
				ERR_FAIL_COND_MSG(morphs_normals.size() != 0 && morphs_normals.size() != morphs_vertices.size(), "The FBX file is corrupted: #ERR106");

				if (r_data.has(morph_name) == false) {
					// This morph doesn't exist yet.
					// Create it.
					MorphVertexData md;
					md.vertices.resize(vertex_count);
					md.normals.resize(vertex_count);
					r_data.set(morph_name, md);
				}

				MorphVertexData *data = r_data.getptr(morph_name);
				Vector3 *data_vertices_ptr = data->vertices.ptrw();
				Vector3 *data_normals_ptr = data->normals.ptrw();

				for (int i = 0; i < (int)morphs_vertex_indices.size(); i += 1) {
					const Vertex vertex = morphs_vertex_indices[i];

					ERR_FAIL_INDEX_MSG(vertex, vertex_count, "The blend shapes of this FBX file are corrupted. It has a not valid vertex.");

					data_vertices_ptr[vertex] = morphs_vertices[i];

					if (morphs_normals.size() != 0) {
						data_normals_ptr[vertex] = morphs_normals[i];
					}
				}
			}
		}
	}
}
