/*************************************************************************/
/*  fbx_mesh_data.h                                                      */
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

#ifndef EDITOR_SCENE_FBX_MESH_DATA_H
#define EDITOR_SCENE_FBX_MESH_DATA_H

#include "core/hash_map.h"
#include "fbx_bone.h"
#include "import_state.h"
#include "modules/fbx/tools/import_utils.h"
#include "scene/3d/mesh_instance.h"
#include "scene/resources/surface_tool.h"
#include "thirdparty/assimp_fbx/FBXMeshGeometry.h"

struct FBXMeshData;
struct FBXBone;
struct ImportState;

struct FBXSplitBySurfaceVertexMapping {
	// Original Mesh Data
	Map<size_t, Vector3> vertex_with_id = Map<size_t, Vector3>();
	Vector<Vector2> uv_0, uv_1 = Vector<Vector2>();
	Vector<Vector3> normals = Vector<Vector3>();
	Vector<Color> colors = Vector<Color>();

	void add_uv_0(Vector2 vec) {
		vec.y = 1.0f - vec.y;
		//print_verbose("added uv_0 " + vec);
		uv_0.push_back(vec);
	}

	void add_uv_1(Vector2 vec) {
		vec.y = 1.0f - vec.y;
		uv_1.push_back(vec);
	}

	Vector3 get_normal(int vertex_id, bool &found) const {
		found = false;
		if (vertex_id < normals.size()) {
			found = true;
			return normals[vertex_id];
		}
		return Vector3();
	}

	Color get_colors(int vertex_id, bool &found) const {
		found = false;
		if (vertex_id < colors.size()) {
			found = true;
			return colors[vertex_id];
		}
		return Color();
	}

	Vector2 get_uv_0(int vertex_id, bool &found) const {
		found = false;
		if (vertex_id < uv_0.size()) {
			found = true;
			return uv_0[vertex_id];
		}
		return Vector2();
	}

	Vector2 get_uv_1(int vertex_id, bool &found) const {
		found = false;
		if (vertex_id < uv_1.size()) {
			found = true;
			return uv_1[vertex_id];
		}
		return Vector2();
	}

	void GenerateIndices(Ref<SurfaceTool> st, uint32_t mesh_face_count) const {
		// todo: can we remove the split by primitive type so it's only material
		// todo: implement the fbx poly mapping thing here
		// todo: convert indices to the godot format
		switch (mesh_face_count) {
			case 1: // todo: validate this
				for (int x = 0; x < vertex_with_id.size(); x += 1) {
					st->add_index(x);
					st->add_index(x);
					st->add_index(x);
				}
				break;
			case 2: // todo: validate this
				for (int x = 0; x < vertex_with_id.size(); x += 2) {
					st->add_index(x + 1);
					st->add_index(x + 1);
					st->add_index(x);
				}
				break;
			case 3: {
				// triangle only
				for (int x = 0; x < vertex_with_id.size(); x += 3) {
					st->add_index(x + 2);
					st->add_index(x + 1);
					st->add_index(x);
				}
			} break;
			case 4: {
				// quad conversion to triangle
				for (int x = 0; x < vertex_with_id.size(); x += 4) {
					// complete first side of triangle

					// todo: unfuck this
					st->add_index(x + 2);
					st->add_index(x + 1);
					st->add_index(x);

					// complete second side of triangle

					// top right
					// bottom right
					// top left

					// first triangle is
					// (x+2), (x+1), (x)
					// second triangle is
					// (x+2), (x), (x+3)

					st->add_index(x + 2);
					st->add_index(x);
					st->add_index(x + 3);

					// anti clockwise rotation in indices
					// note had to reverse right from left here
					// [0](x) bottom right (-1,-1)
					// [1](x+1) bottom left (1,-1)
					// [2](x+2) top left (1,1)
					// [3](x+3) top right (-1,1)

					// we have 4 points
					// we have 2 triangles
					// we have CCW
				}
			} break;
			default:
				print_error("number is not implemented!");
				break;
		}
	}

	void GenerateSurfaceMaterial(Ref<SurfaceTool> st, size_t vertex_id) const {
		bool uv_0 = false;
		bool uv_1 = false;
		bool normal_found = false;
		bool color_found = false;
		Vector2 uv_0_vec = get_uv_0(vertex_id, uv_0);
		Vector2 uv_1_vec = get_uv_1(vertex_id, uv_1);
		Vector3 normal = get_normal(vertex_id, normal_found);
		Color color = get_colors(vertex_id, color_found);
		if (uv_0) {
			//print_verbose("added uv_0 st " + uv_0_vec);
			st->add_uv(uv_0_vec);
		}
		if (uv_1) {
			//print_verbose("added uv_1 st " + uv_1_vec);
			st->add_uv2(uv_1_vec);
		}

		if (normal_found) {
			st->add_normal(normal);
		}

		if (color_found) {
			st->add_color(color);
		}
	}
};

// TODO reneme to VertexWeightMapping
struct VertexMapping {
	Vector<real_t> weights;
	Vector<int> bones;
	// This extra vector is used because the bone id is computed in a second step.
	// TODO Get rid of this extra step is a good idea.
	Vector<Ref<FBXBone> > bones_ref;
};

template <class T>
struct VertexData {
	int polygon_index;
	T data;
};

// Caches mesh information and instantiates meshes for you using helper functions.
struct FBXMeshData : Reference {
	struct MorphVertexData {
		// TODO we have only these??
		/// Each element is a vertex. Not supposed to be void.
		Vector<Vector3> vertices;
		/// Each element is a vertex. Not supposed to be void.
		Vector<Vector3> normals;
	};

	/// vertex id, Weight Info
	/// later: perf we can use array here
	HashMap<int, VertexMapping> vertex_weights;

	// translate fbx mesh data from document context to FBX Mesh Geometry Context
	bool valid_weight_indexes = false;

	MeshInstance *create_fbx_mesh(const ImportState &state, const Assimp::FBX::MeshGeometry *mesh_geometry, const Assimp::FBX::Model *model);

	void gen_weight_info(Ref<SurfaceTool> st, int vertex_id) const;

	/* mesh maximum weight count */
	bool valid_weight_count = false;
	int max_weight_count = 0;
	uint64_t armature_id = 0;
	bool valid_armature_id = false;
	MeshInstance *godot_mesh_instance = nullptr;

private:
	void sanitize_vertex_weights();

	/// Make sure to reorganize the vertices so that the correct UV is taken.
	/// This step is needed because differently from the normal, that can be
	/// combined, the UV may need its own triangle because sometimes they have
	/// really different UV for the same vertex but different polygon.
	/// This function make sure to add another vertex for those UVS.
	void reorganize_vertices(
			std::vector<int> &r_polygon_indices,
			std::vector<Vector3> &r_vertices,
			HashMap<int, Vector3> &r_normals,
			HashMap<int, Vector2> &r_uv_1,
			HashMap<int, Vector2> &r_uv_2,
			HashMap<int, Color> &r_color,
			HashMap<String, MorphVertexData> &r_morphs,
			HashMap<int, HashMap<int, Vector2> > &r_uv_1_raw,
			HashMap<int, HashMap<int, Vector2> > &r_uv_2_raw);

	void add_vertex(
			Ref<SurfaceTool> p_surface_tool,
			real_t p_scale,
			int p_vertex,
			const std::vector<Vector3> &p_vertices_position,
			const HashMap<int, Vector3> &p_normals,
			const HashMap<int, Vector2> &p_uvs_0,
			const HashMap<int, Vector2> &p_uvs_1,
			const HashMap<int, Color> &p_colors,
			const Vector3 &p_morph_value = Vector3(),
			const Vector3 &p_morph_normal = Vector3());

	void triangulate_polygon(Ref<SurfaceTool> st, Vector<int> p_polygon_vertex, Vector<int> p_surface_vertex_map, const std::vector<Vector3> &p_vertices) const;

	/// This function is responsible to convert the FBX polygon vertex to
	/// vertex index.
	/// The polygon vertices are stored in an array with some negative
	/// values. The negative values define the last face index.
	/// For example the following `face_array` contains two faces, the former
	/// with 3 vertices and the latter with a line:
	/// [0,2,-2,3,-5]
	/// Parsed as:
	/// [0, 2, 1, 3, 4]
	/// The negative values are computed using this formula: `(-value) - 1`
	///
	/// Returns the vertex index from the poligon vertex.
	/// Returns -1 if `p_index` is invalid.
	int get_vertex_from_polygon_vertex(const std::vector<int> &p_face_indices, int p_index) const;

	/// Retuns true if this polygon_vertex_index is the end of a new polygon.
	bool is_end_of_polygon(const std::vector<int> &p_face_indices, int p_index) const;

	/// Retuns true if this polygon_vertex_index is the begin of a new polygon.
	bool is_start_of_polygon(const std::vector<int> &p_face_indices, int p_index) const;

	/// Returns the number of polygons.
	int count_polygons(const std::vector<int> &p_face_indices) const;

	/// Used to extract data from the `MappingData` alligned with vertex.
	/// Useful to extract normal/uvs/colors/tangets/etc...
	/// If the function fails somehow, it returns an hollow vector and print an error.
	template <class R, class T>
	HashMap<int, R> extract_per_vertex_data(
			int p_vertex_count,
			const std::vector<Assimp::FBX::MeshGeometry::Edge> &p_edges,
			const std::vector<int> &p_mesh_indices,
			const Assimp::FBX::MeshGeometry::MappingData<T> &p_mapping_data,
			R (*collector_function)(const Vector<VertexData<T> > *p_vertex_data, R p_fall_back),
			R p_fall_back) const;

	/// Used to extract data from the `MappingData` organized per polygon.
	/// Useful to extract the materila
	/// If the function fails somehow, it returns an hollow vector and print an error.
	template <class T>
	HashMap<int, T> extract_per_polygon(
			int p_vertex_count,
			const std::vector<int> &p_face_indices,
			const Assimp::FBX::MeshGeometry::MappingData<T> &p_fbx_data,
			T p_fallback_value) const;

	/// Extracts the morph data and organizes it per vertices.
	/// The returned `MorphVertexData` arrays are never something different
	/// then the `vertex_count`.
	void extract_morphs(const Assimp::FBX::MeshGeometry *mesh_geometry, HashMap<String, MorphVertexData> &r_data);
};

#endif // EDITOR_SCENE_FBX_MESH_DATA_H
