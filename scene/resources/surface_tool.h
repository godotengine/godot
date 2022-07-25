/*************************************************************************/
/*  surface_tool.h                                                       */
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

#ifndef SURFACE_TOOL_H
#define SURFACE_TOOL_H

#include "scene/resources/mesh.h"

#include "thirdparty/misc/mikktspace.h"

class SurfaceTool : public Reference {
	GDCLASS(SurfaceTool, Reference);

public:
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

		bool operator==(const Vertex &p_vertex) const;

		Vertex() {}
	};

private:
	struct VertexHasher {
		static _FORCE_INLINE_ uint32_t hash(const Vertex &p_vtx);
	};

	struct WeightSort {
		int index;
		float weight;
		bool operator<(const WeightSort &p_right) const {
			return weight < p_right.weight;
		}
	};

	bool begun;
	bool first;
	Mesh::PrimitiveType primitive;
	int format;
	Ref<Material> material;
	//arrays
	List<Vertex> vertex_array;
	List<int> index_array;
	Map<int, bool> smooth_groups;

	//memory
	Color last_color;
	Vector3 last_normal;
	Vector2 last_uv;
	Vector2 last_uv2;
	Vector<int> last_bones;
	Vector<float> last_weights;
	Plane last_tangent;

	void _create_list_from_arrays(Array arr, List<Vertex> *r_vertex, List<int> *r_index, int &lformat);
	void _create_list(const Ref<Mesh> &p_existing, int p_surface, List<Vertex> *r_vertex, List<int> *r_index, int &lformat);

	//mikktspace callbacks
	static int mikktGetNumFaces(const SMikkTSpaceContext *pContext);
	static int mikktGetNumVerticesOfFace(const SMikkTSpaceContext *pContext, const int iFace);
	static void mikktGetPosition(const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace, const int iVert);
	static void mikktGetNormal(const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace, const int iVert);
	static void mikktGetTexCoord(const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace, const int iVert);
	static void mikktSetTSpaceDefault(const SMikkTSpaceContext *pContext, const float fvTangent[], const float fvBiTangent[], const float fMagS, const float fMagT,
			const tbool bIsOrientationPreserving, const int iFace, const int iVert);

protected:
	static void _bind_methods();

public:
	void begin(Mesh::PrimitiveType p_primitive);

	void add_vertex(const Vector3 &p_vertex);
	void add_color(Color p_color);
	void add_normal(const Vector3 &p_normal);
	void add_tangent(const Plane &p_tangent);
	void add_uv(const Vector2 &p_uv);
	void add_uv2(const Vector2 &p_uv2);
	void add_bones(const Vector<int> &p_bones);
	void add_weights(const Vector<float> &p_weights);
	void add_smooth_group(bool p_smooth);

	void add_triangle_fan(const Vector<Vector3> &p_vertices, const Vector<Vector2> &p_uvs = Vector<Vector2>(), const Vector<Color> &p_colors = Vector<Color>(), const Vector<Vector2> &p_uv2s = Vector<Vector2>(), const Vector<Vector3> &p_normals = Vector<Vector3>(), const Vector<Plane> &p_tangents = Vector<Plane>());

	void add_index(int p_index);

	void index();
	void deindex();
	void generate_normals(bool p_flip = false);
	void generate_tangents();

	void set_material(const Ref<Material> &p_material);

	void clear();

	List<Vertex> &get_vertex_array() { return vertex_array; }

	void create_from_triangle_arrays(const Array &p_arrays);
	static Vector<Vertex> create_vertex_array_from_triangle_arrays(const Array &p_arrays);
	Array commit_to_arrays();
	void create_from(const Ref<Mesh> &p_existing, int p_surface);
	void create_from_blend_shape(const Ref<Mesh> &p_existing, int p_surface, const String &p_blend_shape_name);
	void append_from(const Ref<Mesh> &p_existing, int p_surface, const Transform &p_xform);
	Ref<ArrayMesh> commit(const Ref<ArrayMesh> &p_existing = Ref<ArrayMesh>(), uint32_t p_flags = Mesh::ARRAY_COMPRESS_DEFAULT);

	SurfaceTool();
};

#endif // SURFACE_TOOL_H
