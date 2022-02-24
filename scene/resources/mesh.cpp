/*************************************************************************/
/*  mesh.cpp                                                             */
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

#include "mesh.h"

#include "core/crypto/crypto_core.h"
#include "core/local_vector.h"
#include "core/math/convex_hull.h"
#include "core/pair.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"
#include "surface_tool.h"

#include <stdlib.h>

Mesh::ConvexDecompositionFunc Mesh::convex_decomposition_function = nullptr;

Ref<TriangleMesh> Mesh::generate_triangle_mesh() const {
	if (triangle_mesh.is_valid()) {
		return triangle_mesh;
	}

	int facecount = 0;

	for (int i = 0; i < get_surface_count(); i++) {
		if (surface_get_primitive_type(i) != PRIMITIVE_TRIANGLES) {
			continue;
		}

		if (surface_get_format(i) & ARRAY_FORMAT_INDEX) {
			facecount += surface_get_array_index_len(i);
		} else {
			facecount += surface_get_array_len(i);
		}
	}

	if (facecount == 0 || (facecount % 3) != 0) {
		return triangle_mesh;
	}

	PoolVector<Vector3> faces;
	faces.resize(facecount);
	PoolVector<Vector3>::Write facesw = faces.write();

	int widx = 0;

	for (int i = 0; i < get_surface_count(); i++) {
		if (surface_get_primitive_type(i) != PRIMITIVE_TRIANGLES) {
			continue;
		}

		Array a = surface_get_arrays(i);
		ERR_FAIL_COND_V(a.empty(), Ref<TriangleMesh>());

		int vc = surface_get_array_len(i);
		PoolVector<Vector3> vertices = a[ARRAY_VERTEX];
		PoolVector<Vector3>::Read vr = vertices.read();

		if (surface_get_format(i) & ARRAY_FORMAT_INDEX) {
			int ic = surface_get_array_index_len(i);
			PoolVector<int> indices = a[ARRAY_INDEX];
			PoolVector<int>::Read ir = indices.read();

			for (int j = 0; j < ic; j++) {
				int index = ir[j];
				facesw[widx++] = vr[index];
			}

		} else {
			for (int j = 0; j < vc; j++) {
				facesw[widx++] = vr[j];
			}
		}
	}

	facesw.release();

	triangle_mesh = Ref<TriangleMesh>(memnew(TriangleMesh));
	triangle_mesh->create(faces);

	return triangle_mesh;
}

void Mesh::generate_debug_mesh_lines(Vector<Vector3> &r_lines) {
	if (debug_lines.size() > 0) {
		r_lines = debug_lines;
		return;
	}

	Ref<TriangleMesh> tm = generate_triangle_mesh();
	if (tm.is_null()) {
		return;
	}

	PoolVector<int> triangle_indices;
	tm->get_indices(&triangle_indices);
	const int triangles_num = tm->get_triangles().size();
	PoolVector<Vector3> vertices = tm->get_vertices();

	debug_lines.resize(tm->get_triangles().size() * 6); // 3 lines x 2 points each line

	PoolVector<int>::Read ind_r = triangle_indices.read();
	PoolVector<Vector3>::Read ver_r = vertices.read();
	for (int j = 0, x = 0, i = 0; i < triangles_num; j += 6, x += 3, ++i) {
		// Triangle line 1
		debug_lines.write[j + 0] = ver_r[ind_r[x + 0]];
		debug_lines.write[j + 1] = ver_r[ind_r[x + 1]];

		// Triangle line 2
		debug_lines.write[j + 2] = ver_r[ind_r[x + 1]];
		debug_lines.write[j + 3] = ver_r[ind_r[x + 2]];

		// Triangle line 3
		debug_lines.write[j + 4] = ver_r[ind_r[x + 2]];
		debug_lines.write[j + 5] = ver_r[ind_r[x + 0]];
	}

	r_lines = debug_lines;
}
void Mesh::generate_debug_mesh_indices(Vector<Vector3> &r_points) {
	Ref<TriangleMesh> tm = generate_triangle_mesh();
	if (tm.is_null()) {
		return;
	}

	PoolVector<Vector3> vertices = tm->get_vertices();

	int vertices_size = vertices.size();
	r_points.resize(vertices_size);
	for (int i = 0; i < vertices_size; ++i) {
		r_points.write[i] = vertices[i];
	}
}

bool Mesh::surface_is_softbody_friendly(int p_idx) const {
	const uint32_t surface_format = surface_get_format(p_idx);
	return (surface_format & Mesh::ARRAY_FLAG_USE_DYNAMIC_UPDATE && (!(surface_format & Mesh::ARRAY_COMPRESS_VERTEX)) && (!(surface_format & Mesh::ARRAY_COMPRESS_NORMAL)));
}

PoolVector<Face3> Mesh::get_faces() const {
	Ref<TriangleMesh> tm = generate_triangle_mesh();
	if (tm.is_valid()) {
		return tm->get_faces();
	}
	return PoolVector<Face3>();
}

Ref<Shape> Mesh::create_convex_shape(bool p_clean, bool p_simplify) const {
	if (p_simplify) {
		Vector<Ref<Shape>> decomposed = convex_decompose(1);
		if (decomposed.size() == 1) {
			return decomposed[0];
		} else {
			ERR_PRINT("Convex shape simplification failed, falling back to simpler process.");
		}
	}

	PoolVector<Vector3> vertices;
	for (int i = 0; i < get_surface_count(); i++) {
		Array a = surface_get_arrays(i);
		ERR_FAIL_COND_V(a.empty(), Ref<ConvexPolygonShape>());
		PoolVector<Vector3> v = a[ARRAY_VERTEX];
		vertices.append_array(v);
	}

	Ref<ConvexPolygonShape> shape = memnew(ConvexPolygonShape);

	if (p_clean) {
		Geometry::MeshData md;
		Error err = ConvexHullComputer::convex_hull(vertices, md);
		if (err == OK) {
			int vertex_count = md.vertices.size();
			vertices.resize(vertex_count);
			{
				PoolVector<Vector3>::Write w = vertices.write();
				for (int idx = 0; idx < vertex_count; ++idx) {
					w[idx] = md.vertices[idx];
				}
			}
		} else {
			ERR_PRINT("Convex shape cleaning failed, falling back to simpler process.");
		}
	}

	shape->set_points(vertices);
	return shape;
}

Ref<Shape> Mesh::create_trimesh_shape() const {
	PoolVector<Face3> faces = get_faces();
	if (faces.size() == 0) {
		return Ref<Shape>();
	}

	PoolVector<Vector3> face_points;
	face_points.resize(faces.size() * 3);

	for (int i = 0; i < face_points.size(); i += 3) {
		Face3 f = faces.get(i / 3);
		face_points.set(i, f.vertex[0]);
		face_points.set(i + 1, f.vertex[1]);
		face_points.set(i + 2, f.vertex[2]);
	}

	Ref<ConcavePolygonShape> shape = memnew(ConcavePolygonShape);
	shape->set_faces(face_points);
	return shape;
}

Ref<Mesh> Mesh::create_outline(float p_margin) const {
	Array arrays;
	int index_accum = 0;
	for (int i = 0; i < get_surface_count(); i++) {
		if (surface_get_primitive_type(i) != PRIMITIVE_TRIANGLES) {
			continue;
		}

		Array a = surface_get_arrays(i);
		ERR_FAIL_COND_V(a.empty(), Ref<ArrayMesh>());

		if (i == 0) {
			arrays = a;
			PoolVector<Vector3> v = a[ARRAY_VERTEX];
			index_accum += v.size();
		} else {
			int vcount = 0;
			for (int j = 0; j < arrays.size(); j++) {
				if (arrays[j].get_type() == Variant::NIL || a[j].get_type() == Variant::NIL) {
					//mismatch, do not use
					arrays[j] = Variant();
					continue;
				}

				switch (j) {
					case ARRAY_VERTEX:
					case ARRAY_NORMAL: {
						PoolVector<Vector3> dst = arrays[j];
						PoolVector<Vector3> src = a[j];
						if (j == ARRAY_VERTEX) {
							vcount = src.size();
						}
						if (dst.size() == 0 || src.size() == 0) {
							arrays[j] = Variant();
							continue;
						}
						dst.append_array(src);
						arrays[j] = dst;
					} break;
					case ARRAY_TANGENT:
					case ARRAY_BONES:
					case ARRAY_WEIGHTS: {
						PoolVector<real_t> dst = arrays[j];
						PoolVector<real_t> src = a[j];
						if (dst.size() == 0 || src.size() == 0) {
							arrays[j] = Variant();
							continue;
						}
						dst.append_array(src);
						arrays[j] = dst;

					} break;
					case ARRAY_COLOR: {
						PoolVector<Color> dst = arrays[j];
						PoolVector<Color> src = a[j];
						if (dst.size() == 0 || src.size() == 0) {
							arrays[j] = Variant();
							continue;
						}
						dst.append_array(src);
						arrays[j] = dst;

					} break;
					case ARRAY_TEX_UV:
					case ARRAY_TEX_UV2: {
						PoolVector<Vector2> dst = arrays[j];
						PoolVector<Vector2> src = a[j];
						if (dst.size() == 0 || src.size() == 0) {
							arrays[j] = Variant();
							continue;
						}
						dst.append_array(src);
						arrays[j] = dst;

					} break;
					case ARRAY_INDEX: {
						PoolVector<int> dst = arrays[j];
						PoolVector<int> src = a[j];
						if (dst.size() == 0 || src.size() == 0) {
							arrays[j] = Variant();
							continue;
						}
						{
							int ss = src.size();
							PoolVector<int>::Write w = src.write();
							for (int k = 0; k < ss; k++) {
								w[k] += index_accum;
							}
						}
						dst.append_array(src);
						arrays[j] = dst;
						index_accum += vcount;

					} break;
				}
			}
		}
	}

	ERR_FAIL_COND_V(arrays.size() != ARRAY_MAX, Ref<ArrayMesh>());

	{
		PoolVector<int>::Write ir;
		PoolVector<int> indices = arrays[ARRAY_INDEX];
		bool has_indices = false;
		PoolVector<Vector3> vertices = arrays[ARRAY_VERTEX];
		int vc = vertices.size();
		ERR_FAIL_COND_V(!vc, Ref<ArrayMesh>());
		PoolVector<Vector3>::Write r = vertices.write();

		if (indices.size()) {
			ERR_FAIL_COND_V(indices.size() % 3 != 0, Ref<ArrayMesh>());
			vc = indices.size();
			ir = indices.write();
			has_indices = true;
		}

		Map<Vector3, Vector3> normal_accum;

		//fill normals with triangle normals
		for (int i = 0; i < vc; i += 3) {
			Vector3 t[3];

			if (has_indices) {
				t[0] = r[ir[i + 0]];
				t[1] = r[ir[i + 1]];
				t[2] = r[ir[i + 2]];
			} else {
				t[0] = r[i + 0];
				t[1] = r[i + 1];
				t[2] = r[i + 2];
			}

			Vector3 n = Plane(t[0], t[1], t[2]).normal;

			for (int j = 0; j < 3; j++) {
				Map<Vector3, Vector3>::Element *E = normal_accum.find(t[j]);
				if (!E) {
					normal_accum[t[j]] = n;
				} else {
					float d = n.dot(E->get());
					if (d < 1.0) {
						E->get() += n * (1.0 - d);
					}
					//E->get()+=n;
				}
			}
		}

		//normalize

		for (Map<Vector3, Vector3>::Element *E = normal_accum.front(); E; E = E->next()) {
			E->get().normalize();
		}

		//displace normals
		int vc2 = vertices.size();

		for (int i = 0; i < vc2; i++) {
			Vector3 t = r[i];

			Map<Vector3, Vector3>::Element *E = normal_accum.find(t);
			ERR_CONTINUE(!E);

			t += E->get() * p_margin;
			r[i] = t;
		}

		r.release();
		arrays[ARRAY_VERTEX] = vertices;

		if (!has_indices) {
			PoolVector<int> new_indices;
			new_indices.resize(vertices.size());
			PoolVector<int>::Write iw = new_indices.write();

			for (int j = 0; j < vc2; j += 3) {
				iw[j] = j;
				iw[j + 1] = j + 2;
				iw[j + 2] = j + 1;
			}

			iw.release();
			arrays[ARRAY_INDEX] = new_indices;

		} else {
			for (int j = 0; j < vc; j += 3) {
				SWAP(ir[j + 1], ir[j + 2]);
			}
			ir.release();
			arrays[ARRAY_INDEX] = indices;
		}
	}

	Ref<ArrayMesh> newmesh = memnew(ArrayMesh);
	newmesh->add_surface_from_arrays(PRIMITIVE_TRIANGLES, arrays);
	return newmesh;
}

void Mesh::set_lightmap_size_hint(const Vector2 &p_size) {
	lightmap_size_hint = p_size;
}

Size2 Mesh::get_lightmap_size_hint() const {
	return lightmap_size_hint;
}

void Mesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_lightmap_size_hint", "size"), &Mesh::set_lightmap_size_hint);
	ClassDB::bind_method(D_METHOD("get_lightmap_size_hint"), &Mesh::get_lightmap_size_hint);
	ClassDB::bind_method(D_METHOD("get_aabb"), &Mesh::get_aabb);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "lightmap_size_hint"), "set_lightmap_size_hint", "get_lightmap_size_hint");

	ClassDB::bind_method(D_METHOD("get_surface_count"), &Mesh::get_surface_count);
	ClassDB::bind_method(D_METHOD("surface_get_arrays", "surf_idx"), &Mesh::surface_get_arrays);
	ClassDB::bind_method(D_METHOD("surface_get_blend_shape_arrays", "surf_idx"), &Mesh::surface_get_blend_shape_arrays);
	ClassDB::bind_method(D_METHOD("surface_set_material", "surf_idx", "material"), &Mesh::surface_set_material);
	ClassDB::bind_method(D_METHOD("surface_get_material", "surf_idx"), &Mesh::surface_get_material);

	BIND_ENUM_CONSTANT(PRIMITIVE_POINTS);
	BIND_ENUM_CONSTANT(PRIMITIVE_LINES);
	BIND_ENUM_CONSTANT(PRIMITIVE_LINE_STRIP);
	BIND_ENUM_CONSTANT(PRIMITIVE_LINE_LOOP);
	BIND_ENUM_CONSTANT(PRIMITIVE_TRIANGLES);
	BIND_ENUM_CONSTANT(PRIMITIVE_TRIANGLE_STRIP);
	BIND_ENUM_CONSTANT(PRIMITIVE_TRIANGLE_FAN);

	BIND_ENUM_CONSTANT(BLEND_SHAPE_MODE_NORMALIZED);
	BIND_ENUM_CONSTANT(BLEND_SHAPE_MODE_RELATIVE);

	BIND_ENUM_CONSTANT(ARRAY_FORMAT_VERTEX);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_NORMAL);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_TANGENT);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_COLOR);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_TEX_UV);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_TEX_UV2);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_BONES);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_WEIGHTS);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_INDEX);

	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_BASE);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_VERTEX);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_NORMAL);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_TANGENT);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_COLOR);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_TEX_UV);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_TEX_UV2);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_BONES);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_WEIGHTS);
	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_INDEX);

	BIND_ENUM_CONSTANT(ARRAY_FLAG_USE_2D_VERTICES);
	BIND_ENUM_CONSTANT(ARRAY_FLAG_USE_16_BIT_BONES);
	BIND_ENUM_CONSTANT(ARRAY_FLAG_USE_OCTAHEDRAL_COMPRESSION);

	BIND_ENUM_CONSTANT(ARRAY_COMPRESS_DEFAULT);

	BIND_ENUM_CONSTANT(ARRAY_VERTEX);
	BIND_ENUM_CONSTANT(ARRAY_NORMAL);
	BIND_ENUM_CONSTANT(ARRAY_TANGENT);
	BIND_ENUM_CONSTANT(ARRAY_COLOR);
	BIND_ENUM_CONSTANT(ARRAY_TEX_UV);
	BIND_ENUM_CONSTANT(ARRAY_TEX_UV2);
	BIND_ENUM_CONSTANT(ARRAY_BONES);
	BIND_ENUM_CONSTANT(ARRAY_WEIGHTS);
	BIND_ENUM_CONSTANT(ARRAY_INDEX);
	BIND_ENUM_CONSTANT(ARRAY_MAX);
}

void Mesh::clear_cache() const {
	triangle_mesh.unref();
	debug_lines.clear();
}

Vector<Ref<Shape>> Mesh::convex_decompose(int p_max_convex_hulls) const {
	ERR_FAIL_COND_V(!convex_decomposition_function, Vector<Ref<Shape>>());

	Ref<TriangleMesh> tm = generate_triangle_mesh();
	ERR_FAIL_COND_V(!tm.is_valid(), Vector<Ref<Shape>>());

	const PoolVector<TriangleMesh::Triangle> &triangles = tm->get_triangles();
	int triangle_count = triangles.size();

	PoolVector<uint32_t> indices;
	{
		indices.resize(triangle_count * 3);
		PoolVector<uint32_t>::Write w = indices.write();
		PoolVector<TriangleMesh::Triangle>::Read triangles_read = triangles.read();
		for (int i = 0; i < triangle_count; i++) {
			for (int j = 0; j < 3; j++) {
				w[i * 3 + j] = triangles_read[i].indices[j];
			}
		}
	}

	const PoolVector<Vector3> &vertices = tm->get_vertices();
	int vertex_count = vertices.size();

	Vector<PoolVector<Vector3>> decomposed = convex_decomposition_function((real_t *)vertices.read().ptr(), vertex_count, indices.read().ptr(), triangle_count, p_max_convex_hulls, nullptr);

	Vector<Ref<Shape>> ret;

	for (int i = 0; i < decomposed.size(); i++) {
		Ref<ConvexPolygonShape> shape;
		shape.instance();
		shape->set_points(decomposed[i]);
		ret.push_back(shape);
	}

	return ret;
}

Mesh::Mesh() {
}

bool ArrayMesh::_set(const StringName &p_name, const Variant &p_value) {
	String sname = p_name;

	if (p_name == "blend_shape/names") {
		PoolVector<String> sk = p_value;
		int sz = sk.size();
		PoolVector<String>::Read r = sk.read();
		for (int i = 0; i < sz; i++) {
			add_blend_shape(r[i]);
		}
		return true;
	}

	if (p_name == "blend_shape/mode") {
		set_blend_shape_mode(BlendShapeMode(int(p_value)));
		return true;
	}

	if (sname.begins_with("surface_")) {
		int sl = sname.find("/");
		if (sl == -1) {
			return false;
		}
		int idx = sname.substr(8, sl - 8).to_int() - 1;
		String what = sname.get_slicec('/', 1);
		if (what == "material") {
			surface_set_material(idx, p_value);
		} else if (what == "name") {
			surface_set_name(idx, p_value);
		}
		return true;
	}

	if (!sname.begins_with("surfaces")) {
		return false;
	}

	int idx = sname.get_slicec('/', 1).to_int();
	String what = sname.get_slicec('/', 2);

	if (idx == surfaces.size()) {
		//create
		Dictionary d = p_value;
		ERR_FAIL_COND_V(!d.has("primitive"), false);

		if (d.has("arrays")) {
			//old format
			ERR_FAIL_COND_V(!d.has("morph_arrays"), false);
			add_surface_from_arrays(PrimitiveType(int(d["primitive"])), d["arrays"], d["morph_arrays"]);

		} else if (d.has("array_data")) {
			PoolVector<uint8_t> array_data = d["array_data"];
			PoolVector<uint8_t> array_index_data;
			if (d.has("array_index_data")) {
				array_index_data = d["array_index_data"];
			}

			ERR_FAIL_COND_V(!d.has("format"), false);
			uint32_t format = d["format"];

			uint32_t primitive = d["primitive"];

			ERR_FAIL_COND_V(!d.has("vertex_count"), false);
			int vertex_count = d["vertex_count"];

			int index_count = 0;
			if (d.has("index_count")) {
				index_count = d["index_count"];
			}

			Vector<PoolVector<uint8_t>> blend_shapes;

			if (d.has("blend_shape_data")) {
				Array blend_shape_data = d["blend_shape_data"];
				for (int i = 0; i < blend_shape_data.size(); i++) {
					PoolVector<uint8_t> shape = blend_shape_data[i];
					blend_shapes.push_back(shape);
				}
			}

			ERR_FAIL_COND_V(!d.has("aabb"), false);
			AABB aabb = d["aabb"];

			Vector<AABB> bone_aabb;
			if (d.has("skeleton_aabb")) {
				Array baabb = d["skeleton_aabb"];
				bone_aabb.resize(baabb.size());

				for (int i = 0; i < baabb.size(); i++) {
					bone_aabb.write[i] = baabb[i];
				}
			}

			add_surface(format, PrimitiveType(primitive), array_data, vertex_count, array_index_data, index_count, aabb, blend_shapes, bone_aabb);
		} else {
			ERR_FAIL_V(false);
		}

		if (d.has("material")) {
			surface_set_material(idx, d["material"]);
		}
		if (d.has("name")) {
			surface_set_name(idx, d["name"]);
		}

		return true;
	}

	return false;
}

bool ArrayMesh::_get(const StringName &p_name, Variant &r_ret) const {
	if (_is_generated()) {
		return false;
	}

	String sname = p_name;

	if (p_name == "blend_shape/names") {
		PoolVector<String> sk;
		for (int i = 0; i < blend_shapes.size(); i++) {
			sk.push_back(blend_shapes[i]);
		}
		r_ret = sk;
		return true;
	} else if (p_name == "blend_shape/mode") {
		r_ret = get_blend_shape_mode();
		return true;
	} else if (sname.begins_with("surface_")) {
		int sl = sname.find("/");
		if (sl == -1) {
			return false;
		}
		int idx = sname.substr(8, sl - 8).to_int() - 1;
		String what = sname.get_slicec('/', 1);
		if (what == "material") {
			r_ret = surface_get_material(idx);
		} else if (what == "name") {
			r_ret = surface_get_name(idx);
		}
		return true;
	} else if (!sname.begins_with("surfaces")) {
		return false;
	}

	int idx = sname.get_slicec('/', 1).to_int();
	ERR_FAIL_INDEX_V(idx, surfaces.size(), false);

	Dictionary d;

	d["array_data"] = VS::get_singleton()->mesh_surface_get_array(mesh, idx);
	d["vertex_count"] = VS::get_singleton()->mesh_surface_get_array_len(mesh, idx);
	d["array_index_data"] = VS::get_singleton()->mesh_surface_get_index_array(mesh, idx);
	d["index_count"] = VS::get_singleton()->mesh_surface_get_array_index_len(mesh, idx);
	d["primitive"] = VS::get_singleton()->mesh_surface_get_primitive_type(mesh, idx);
	d["format"] = VS::get_singleton()->mesh_surface_get_format(mesh, idx);
	d["aabb"] = VS::get_singleton()->mesh_surface_get_aabb(mesh, idx);

	Vector<AABB> skel_aabb = VS::get_singleton()->mesh_surface_get_skeleton_aabb(mesh, idx);
	Array arr;
	arr.resize(skel_aabb.size());
	for (int i = 0; i < skel_aabb.size(); i++) {
		arr[i] = skel_aabb[i];
	}
	d["skeleton_aabb"] = arr;

	Vector<PoolVector<uint8_t>> blend_shape_data = VS::get_singleton()->mesh_surface_get_blend_shapes(mesh, idx);

	Array md;
	for (int i = 0; i < blend_shape_data.size(); i++) {
		md.push_back(blend_shape_data[i]);
	}

	d["blend_shape_data"] = md;

	Ref<Material> m = surface_get_material(idx);
	if (m.is_valid()) {
		d["material"] = m;
	}
	String n = surface_get_name(idx);
	if (n != "") {
		d["name"] = n;
	}

	r_ret = d;

	return true;
}

void ArrayMesh::_get_property_list(List<PropertyInfo> *p_list) const {
	if (_is_generated()) {
		return;
	}

	if (blend_shapes.size()) {
		p_list->push_back(PropertyInfo(Variant::POOL_STRING_ARRAY, "blend_shape/names", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::INT, "blend_shape/mode", PROPERTY_HINT_ENUM, "Normalized,Relative"));
	}

	for (int i = 0; i < surfaces.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::DICTIONARY, "surfaces/" + itos(i), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::STRING, "surface_" + itos(i + 1) + "/name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		if (surfaces[i].is_2d) {
			p_list->push_back(PropertyInfo(Variant::OBJECT, "surface_" + itos(i + 1) + "/material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,CanvasItemMaterial", PROPERTY_USAGE_EDITOR));
		} else {
			p_list->push_back(PropertyInfo(Variant::OBJECT, "surface_" + itos(i + 1) + "/material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,SpatialMaterial", PROPERTY_USAGE_EDITOR));
		}
	}
}

void ArrayMesh::_recompute_aabb() {
	// regenerate AABB
	aabb = AABB();

	for (int i = 0; i < surfaces.size(); i++) {
		if (i == 0) {
			aabb = surfaces[i].aabb;
		} else {
			aabb.merge_with(surfaces[i].aabb);
		}
	}
}

void ArrayMesh::add_surface(uint32_t p_format, PrimitiveType p_primitive, const PoolVector<uint8_t> &p_array, int p_vertex_count, const PoolVector<uint8_t> &p_index_array, int p_index_count, const AABB &p_aabb, const Vector<PoolVector<uint8_t>> &p_blend_shapes, const Vector<AABB> &p_bone_aabbs) {
	Surface s;
	s.aabb = p_aabb;
	s.is_2d = p_format & ARRAY_FLAG_USE_2D_VERTICES;
	surfaces.push_back(s);
	_recompute_aabb();

	VisualServer::get_singleton()->mesh_add_surface(mesh, p_format, (VS::PrimitiveType)p_primitive, p_array, p_vertex_count, p_index_array, p_index_count, p_aabb, p_blend_shapes, p_bone_aabbs);
}

void ArrayMesh::add_surface_from_arrays(PrimitiveType p_primitive, const Array &p_arrays, const Array &p_blend_shapes, uint32_t p_flags) {
	ERR_FAIL_COND(p_arrays.size() != ARRAY_MAX);

	Surface s;

	VisualServer::get_singleton()->mesh_add_surface_from_arrays(mesh, (VisualServer::PrimitiveType)p_primitive, p_arrays, p_blend_shapes, p_flags);

	/* make aABB? */ {
		Variant arr = p_arrays[ARRAY_VERTEX];
		PoolVector<Vector3> vertices = arr;
		int len = vertices.size();
		ERR_FAIL_COND(len == 0);
		PoolVector<Vector3>::Read r = vertices.read();
		const Vector3 *vtx = r.ptr();

		// check AABB
		AABB aabb;
		for (int i = 0; i < len; i++) {
			if (i == 0) {
				aabb.position = vtx[i];
			} else {
				aabb.expand_to(vtx[i]);
			}
		}

		s.aabb = aabb;
		s.is_2d = arr.get_type() == Variant::POOL_VECTOR2_ARRAY;
		surfaces.push_back(s);

		_recompute_aabb();
	}

	clear_cache();
	_change_notify();
	emit_changed();
}

Array ArrayMesh::surface_get_arrays(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Array());
	return VisualServer::get_singleton()->mesh_surface_get_arrays(mesh, p_surface);
}
Array ArrayMesh::surface_get_blend_shape_arrays(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Array());
	return VisualServer::get_singleton()->mesh_surface_get_blend_shape_arrays(mesh, p_surface);
}

int ArrayMesh::get_surface_count() const {
	return surfaces.size();
}

void ArrayMesh::add_blend_shape(const StringName &p_name) {
	ERR_FAIL_COND_MSG(surfaces.size(), "Can't add a shape key count if surfaces are already created.");

	StringName name = p_name;

	if (blend_shapes.find(name) != -1) {
		int count = 2;
		do {
			name = String(p_name) + " " + itos(count);
			count++;
		} while (blend_shapes.find(name) != -1);
	}

	blend_shapes.push_back(name);
	VS::get_singleton()->mesh_set_blend_shape_count(mesh, blend_shapes.size());
}

int ArrayMesh::get_blend_shape_count() const {
	return blend_shapes.size();
}
StringName ArrayMesh::get_blend_shape_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, blend_shapes.size(), StringName());
	return blend_shapes[p_index];
}

void ArrayMesh::set_blend_shape_name(int p_index, const StringName &p_name) {
	ERR_FAIL_INDEX(p_index, blend_shapes.size());

	StringName name = p_name;
	int found = blend_shapes.find(name);
	if (found != -1 && found != p_index) {
		int count = 2;
		do {
			name = String(p_name) + " " + itos(count);
			count++;
		} while (blend_shapes.find(name) != -1);
	}

	blend_shapes.write[p_index] = name;
}

void ArrayMesh::clear_blend_shapes() {
	ERR_FAIL_COND_MSG(surfaces.size(), "Can't set shape key count if surfaces are already created.");

	blend_shapes.clear();
}

void ArrayMesh::set_blend_shape_mode(BlendShapeMode p_mode) {
	blend_shape_mode = p_mode;
	VS::get_singleton()->mesh_set_blend_shape_mode(mesh, (VS::BlendShapeMode)p_mode);
}

ArrayMesh::BlendShapeMode ArrayMesh::get_blend_shape_mode() const {
	return blend_shape_mode;
}

void ArrayMesh::surface_remove(int p_idx) {
	ERR_FAIL_INDEX(p_idx, surfaces.size());
	VisualServer::get_singleton()->mesh_remove_surface(mesh, p_idx);
	surfaces.remove(p_idx);

	clear_cache();
	_recompute_aabb();
	_change_notify();
	emit_changed();
}

int ArrayMesh::surface_get_array_len(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), -1);
	return VisualServer::get_singleton()->mesh_surface_get_array_len(mesh, p_idx);
}

int ArrayMesh::surface_get_array_index_len(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), -1);
	return VisualServer::get_singleton()->mesh_surface_get_array_index_len(mesh, p_idx);
}

uint32_t ArrayMesh::surface_get_format(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), 0);
	return VisualServer::get_singleton()->mesh_surface_get_format(mesh, p_idx);
}

ArrayMesh::PrimitiveType ArrayMesh::surface_get_primitive_type(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), PRIMITIVE_LINES);
	return (PrimitiveType)VisualServer::get_singleton()->mesh_surface_get_primitive_type(mesh, p_idx);
}

void ArrayMesh::surface_set_material(int p_idx, const Ref<Material> &p_material) {
	ERR_FAIL_INDEX(p_idx, surfaces.size());
	if (surfaces[p_idx].material == p_material) {
		return;
	}
	surfaces.write[p_idx].material = p_material;
	VisualServer::get_singleton()->mesh_surface_set_material(mesh, p_idx, p_material.is_null() ? RID() : p_material->get_rid());

	_change_notify("material");
	emit_changed();
}

int ArrayMesh::surface_find_by_name(const String &p_name) const {
	for (int i = 0; i < surfaces.size(); i++) {
		if (surfaces[i].name == p_name) {
			return i;
		}
	}

	return -1;
}

void ArrayMesh::surface_set_name(int p_idx, const String &p_name) {
	ERR_FAIL_INDEX(p_idx, surfaces.size());

	surfaces.write[p_idx].name = p_name;
	emit_changed();
}

String ArrayMesh::surface_get_name(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), String());
	return surfaces[p_idx].name;
}

void ArrayMesh::surface_update_region(int p_surface, int p_offset, const PoolVector<uint8_t> &p_data) {
	ERR_FAIL_INDEX(p_surface, surfaces.size());
	VS::get_singleton()->mesh_surface_update_region(mesh, p_surface, p_offset, p_data);
	emit_changed();
}

void ArrayMesh::surface_set_custom_aabb(int p_idx, const AABB &p_aabb) {
	ERR_FAIL_INDEX(p_idx, surfaces.size());
	surfaces.write[p_idx].aabb = p_aabb;
	// set custom aabb too?
	emit_changed();
}

Ref<Material> ArrayMesh::surface_get_material(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), Ref<Material>());
	return surfaces[p_idx].material;
}

void ArrayMesh::add_surface_from_mesh_data(const Geometry::MeshData &p_mesh_data) {
	VisualServer::get_singleton()->mesh_add_surface_from_mesh_data(mesh, p_mesh_data);
	AABB aabb;
	for (int i = 0; i < p_mesh_data.vertices.size(); i++) {
		if (i == 0) {
			aabb.position = p_mesh_data.vertices[i];
		} else {
			aabb.expand_to(p_mesh_data.vertices[i]);
		}
	}

	Surface s;
	s.aabb = aabb;
	if (surfaces.size() == 0) {
		aabb = s.aabb;
	} else {
		aabb.merge_with(s.aabb);
	}

	clear_cache();

	surfaces.push_back(s);
	_change_notify();

	emit_changed();
}

RID ArrayMesh::get_rid() const {
	return mesh;
}
AABB ArrayMesh::get_aabb() const {
	return aabb;
}

void ArrayMesh::clear_surfaces() {
	if (!mesh.is_valid()) {
		return;
	}
	VS::get_singleton()->mesh_clear(mesh);
	surfaces.clear();
	aabb = AABB();
}

void ArrayMesh::set_custom_aabb(const AABB &p_custom) {
	custom_aabb = p_custom;
	VS::get_singleton()->mesh_set_custom_aabb(mesh, custom_aabb);
	emit_changed();
}

AABB ArrayMesh::get_custom_aabb() const {
	return custom_aabb;
}

void ArrayMesh::regen_normalmaps() {
	Vector<Ref<SurfaceTool>> surfs;
	for (int i = 0; i < get_surface_count(); i++) {
		Ref<SurfaceTool> st = memnew(SurfaceTool);
		st->create_from(Ref<ArrayMesh>(this), i);
		surfs.push_back(st);
	}

	while (get_surface_count()) {
		surface_remove(0);
	}

	for (int i = 0; i < surfs.size(); i++) {
		surfs.write[i]->generate_tangents();
		surfs.write[i]->commit(Ref<ArrayMesh>(this));
	}
}

//dirty hack
bool (*array_mesh_lightmap_unwrap_callback)(float p_texel_size, const float *p_vertices, const float *p_normals, int p_vertex_count, const int *p_indices, const int *p_face_materials, int p_index_count, float **r_uv, int **r_vertex, int *r_vertex_count, int **r_index, int *r_index_count, int *r_size_hint_x, int *r_size_hint_y) = nullptr;

struct ArrayMeshLightmapSurface {
	Ref<Material> material;
	Vector<SurfaceTool::Vertex> vertices;
	Mesh::PrimitiveType primitive;
	uint32_t format;
};

Error ArrayMesh::lightmap_unwrap(const Transform &p_base_transform, float p_texel_size) {
	int *cache_data = nullptr;
	unsigned int cache_size = 0;
	bool use_cache = false; // Don't use cache
	return lightmap_unwrap_cached(cache_data, cache_size, use_cache, p_base_transform, p_texel_size);
}

Error ArrayMesh::lightmap_unwrap_cached(int *&r_cache_data, unsigned int &r_cache_size, bool &r_used_cache, const Transform &p_base_transform, float p_texel_size) {
	ERR_FAIL_COND_V(!array_mesh_lightmap_unwrap_callback, ERR_UNCONFIGURED);
	ERR_FAIL_COND_V_MSG(blend_shapes.size() != 0, ERR_UNAVAILABLE, "Can't unwrap mesh with blend shapes.");
	ERR_FAIL_COND_V_MSG(p_texel_size <= 0.0f, ERR_PARAMETER_RANGE_ERROR, "Texel size must be greater than 0.");

	LocalVector<float> vertices;
	LocalVector<float> normals;
	LocalVector<int> indices;
	LocalVector<int> face_materials;
	LocalVector<float> uv;
	LocalVector<Pair<int, int>> uv_indices;

	Vector<ArrayMeshLightmapSurface> lightmap_surfaces;

	// Keep only the scale
	Basis basis = p_base_transform.get_basis();
	Vector3 scale = Vector3(basis.get_axis(0).length(), basis.get_axis(1).length(), basis.get_axis(2).length());

	Transform transform;
	transform.scale(scale);

	Basis normal_basis = transform.basis.inverse().transposed();

	for (int i = 0; i < get_surface_count(); i++) {
		ArrayMeshLightmapSurface s;
		s.primitive = surface_get_primitive_type(i);

		ERR_FAIL_COND_V_MSG(s.primitive != Mesh::PRIMITIVE_TRIANGLES, ERR_UNAVAILABLE, "Only triangles are supported for lightmap unwrap.");
		s.format = surface_get_format(i);
		ERR_FAIL_COND_V_MSG(!(s.format & ARRAY_FORMAT_NORMAL), ERR_UNAVAILABLE, "Normals are required for lightmap unwrap.");

		Array arrays = surface_get_arrays(i);
		s.material = surface_get_material(i);
		s.vertices = SurfaceTool::create_vertex_array_from_triangle_arrays(arrays);

		PoolVector<Vector3> rvertices = arrays[Mesh::ARRAY_VERTEX];
		int vc = rvertices.size();
		PoolVector<Vector3>::Read r = rvertices.read();

		PoolVector<Vector3> rnormals = arrays[Mesh::ARRAY_NORMAL];
		PoolVector<Vector3>::Read rn = rnormals.read();

		int vertex_ofs = vertices.size() / 3;

		vertices.resize((vertex_ofs + vc) * 3);
		normals.resize((vertex_ofs + vc) * 3);
		uv_indices.resize(vertex_ofs + vc);

		for (int j = 0; j < vc; j++) {
			Vector3 v = transform.xform(r[j]);
			Vector3 n = normal_basis.xform(rn[j]).normalized();

			vertices[(j + vertex_ofs) * 3 + 0] = v.x;
			vertices[(j + vertex_ofs) * 3 + 1] = v.y;
			vertices[(j + vertex_ofs) * 3 + 2] = v.z;
			normals[(j + vertex_ofs) * 3 + 0] = n.x;
			normals[(j + vertex_ofs) * 3 + 1] = n.y;
			normals[(j + vertex_ofs) * 3 + 2] = n.z;
			uv_indices[j + vertex_ofs] = Pair<int, int>(i, j);
		}

		PoolVector<int> rindices = arrays[Mesh::ARRAY_INDEX];
		int ic = rindices.size();

		float eps = 1.19209290e-7F; // Taken from xatlas.h
		if (ic == 0) {
			for (int j = 0; j < vc / 3; j++) {
				Vector3 p0 = transform.xform(r[j * 3 + 0]);
				Vector3 p1 = transform.xform(r[j * 3 + 1]);
				Vector3 p2 = transform.xform(r[j * 3 + 2]);

				if ((p0 - p1).length_squared() < eps || (p1 - p2).length_squared() < eps || (p2 - p0).length_squared() < eps) {
					continue;
				}

				indices.push_back(vertex_ofs + j * 3 + 0);
				indices.push_back(vertex_ofs + j * 3 + 1);
				indices.push_back(vertex_ofs + j * 3 + 2);
				face_materials.push_back(i);
			}

		} else {
			PoolVector<int>::Read ri = rindices.read();

			for (int j = 0; j < ic / 3; j++) {
				Vector3 p0 = transform.xform(r[ri[j * 3 + 0]]);
				Vector3 p1 = transform.xform(r[ri[j * 3 + 1]]);
				Vector3 p2 = transform.xform(r[ri[j * 3 + 2]]);

				if ((p0 - p1).length_squared() < eps || (p1 - p2).length_squared() < eps || (p2 - p0).length_squared() < eps) {
					continue;
				}

				indices.push_back(vertex_ofs + ri[j * 3 + 0]);
				indices.push_back(vertex_ofs + ri[j * 3 + 1]);
				indices.push_back(vertex_ofs + ri[j * 3 + 2]);
				face_materials.push_back(i);
			}
		}

		lightmap_surfaces.push_back(s);
	}

	CryptoCore::MD5Context ctx;
	ctx.start();

	ctx.update((unsigned char *)&p_texel_size, sizeof(float));
	ctx.update((unsigned char *)indices.ptr(), sizeof(int) * indices.size());
	ctx.update((unsigned char *)face_materials.ptr(), sizeof(int) * face_materials.size());
	ctx.update((unsigned char *)vertices.ptr(), sizeof(float) * vertices.size());
	ctx.update((unsigned char *)normals.ptr(), sizeof(float) * normals.size());

	unsigned char hash[16];
	ctx.finish(hash);

	bool cached = false;
	unsigned int cache_idx = 0;

	if (r_used_cache && r_cache_data) {
		//Check if hash is in cache data

		int *cache_data = r_cache_data;
		int n_entries = cache_data[0];
		unsigned int r_idx = 1;
		for (int i = 0; i < n_entries; ++i) {
			if (memcmp(&cache_data[r_idx], hash, 16) == 0) {
				cached = true;
				cache_idx = r_idx;
				break;
			}

			r_idx += 4; // hash
			r_idx += 2; // size hint

			int vertex_count = cache_data[r_idx];
			r_idx += 1; // vertex count
			r_idx += vertex_count; // vertex
			r_idx += vertex_count * 2; // uvs

			int index_count = cache_data[r_idx];
			r_idx += 1; // index count
			r_idx += index_count; // indices
		}
	}

	//unwrap

	float *gen_uvs;
	int *gen_vertices;
	int *gen_indices;
	int gen_vertex_count;
	int gen_index_count;
	int size_x;
	int size_y;

	if (r_used_cache && cached) {
		int *cache_data = r_cache_data;

		// Return cache data pointer to the caller
		r_cache_data = &cache_data[cache_idx];

		cache_idx += 4;

		// Load size
		size_x = ((int *)cache_data)[cache_idx];
		size_y = ((int *)cache_data)[cache_idx + 1];
		cache_idx += 2;

		// Load vertices
		gen_vertex_count = cache_data[cache_idx];
		cache_idx++;
		gen_vertices = &cache_data[cache_idx];
		cache_idx += gen_vertex_count;

		// Load UVs
		gen_uvs = (float *)&cache_data[cache_idx];
		cache_idx += gen_vertex_count * 2;

		// Load indices
		gen_index_count = cache_data[cache_idx];
		cache_idx++;
		gen_indices = &cache_data[cache_idx];

		// Return cache data size to the caller
		r_cache_size = sizeof(int) * (4 + 2 + 1 + gen_vertex_count + (gen_vertex_count * 2) + 1 + gen_index_count); // hash + size hint + vertex_count + vertices + uvs + index_count + indices
		r_used_cache = true;
	}

	if (!cached) {
		bool ok = array_mesh_lightmap_unwrap_callback(p_texel_size, vertices.ptr(), normals.ptr(), vertices.size() / 3, indices.ptr(), face_materials.ptr(), indices.size(), &gen_uvs, &gen_vertices, &gen_vertex_count, &gen_indices, &gen_index_count, &size_x, &size_y);

		if (!ok) {
			return ERR_CANT_CREATE;
		}

		if (r_used_cache) {
			unsigned int new_cache_size = 4 + 2 + 1 + gen_vertex_count + (gen_vertex_count * 2) + 1 + gen_index_count; // hash + size hint + vertex_count + vertices + uvs + index_count + indices
			new_cache_size *= sizeof(int);
			int *new_cache_data = (int *)memalloc(new_cache_size);
			unsigned int new_cache_idx = 0;

			// hash
			memcpy(&new_cache_data[new_cache_idx], hash, 16);
			new_cache_idx += 4;

			// size hint
			new_cache_data[new_cache_idx] = size_x;
			new_cache_data[new_cache_idx + 1] = size_y;
			new_cache_idx += 2;

			// vertex count
			new_cache_data[new_cache_idx] = gen_vertex_count;
			new_cache_idx++;

			// vertices
			memcpy(&new_cache_data[new_cache_idx], gen_vertices, sizeof(int) * gen_vertex_count);
			new_cache_idx += gen_vertex_count;

			// uvs
			memcpy(&new_cache_data[new_cache_idx], gen_uvs, sizeof(float) * gen_vertex_count * 2);
			new_cache_idx += gen_vertex_count * 2;

			// index count
			new_cache_data[new_cache_idx] = gen_index_count;
			new_cache_idx++;

			// indices
			memcpy(&new_cache_data[new_cache_idx], gen_indices, sizeof(int) * gen_index_count);
			new_cache_idx += gen_index_count;

			// Return cache data to the caller
			r_cache_data = new_cache_data;
			r_cache_size = new_cache_size;
			r_used_cache = false;
		}
	}

	//remove surfaces
	while (get_surface_count()) {
		surface_remove(0);
	}

	//create surfacetools for each surface..
	LocalVector<Ref<SurfaceTool>> surfaces_tools;

	for (int i = 0; i < lightmap_surfaces.size(); i++) {
		Ref<SurfaceTool> st;
		st.instance();
		st->begin(Mesh::PRIMITIVE_TRIANGLES);
		st->set_material(lightmap_surfaces[i].material);
		surfaces_tools.push_back(st); //stay there
	}

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

			if (lightmap_surfaces[surface].format & ARRAY_FORMAT_COLOR) {
				surfaces_tools[surface]->add_color(v.color);
			}
			if (lightmap_surfaces[surface].format & ARRAY_FORMAT_TEX_UV) {
				surfaces_tools[surface]->add_uv(v.uv);
			}
			if (lightmap_surfaces[surface].format & ARRAY_FORMAT_NORMAL) {
				surfaces_tools[surface]->add_normal(v.normal);
			}
			if (lightmap_surfaces[surface].format & ARRAY_FORMAT_TANGENT) {
				Plane t;
				t.normal = v.tangent;
				t.d = v.binormal.dot(v.normal.cross(v.tangent)) < 0 ? -1 : 1;
				surfaces_tools[surface]->add_tangent(t);
			}
			if (lightmap_surfaces[surface].format & ARRAY_FORMAT_BONES) {
				surfaces_tools[surface]->add_bones(v.bones);
			}
			if (lightmap_surfaces[surface].format & ARRAY_FORMAT_WEIGHTS) {
				surfaces_tools[surface]->add_weights(v.weights);
			}

			Vector2 uv2(gen_uvs[gen_indices[i + j] * 2 + 0], gen_uvs[gen_indices[i + j] * 2 + 1]);
			surfaces_tools[surface]->add_uv2(uv2);

			surfaces_tools[surface]->add_vertex(v.vertex);
		}
	}

	//generate surfaces
	for (unsigned int i = 0; i < surfaces_tools.size(); i++) {
		surfaces_tools[i]->index();
		surfaces_tools[i]->commit(Ref<ArrayMesh>((ArrayMesh *)this), lightmap_surfaces[i].format);
	}

	set_lightmap_size_hint(Size2(size_x, size_y));

	if (!cached) {
		//free stuff
		::free(gen_vertices);
		::free(gen_indices);
		::free(gen_uvs);
	}

	return OK;
}

void ArrayMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_blend_shape", "name"), &ArrayMesh::add_blend_shape);
	ClassDB::bind_method(D_METHOD("get_blend_shape_count"), &ArrayMesh::get_blend_shape_count);
	ClassDB::bind_method(D_METHOD("get_blend_shape_name", "index"), &ArrayMesh::get_blend_shape_name);
	ClassDB::bind_method(D_METHOD("set_blend_shape_name", "index", "name"), &ArrayMesh::set_blend_shape_name);
	ClassDB::bind_method(D_METHOD("clear_blend_shapes"), &ArrayMesh::clear_blend_shapes);
	ClassDB::bind_method(D_METHOD("set_blend_shape_mode", "mode"), &ArrayMesh::set_blend_shape_mode);
	ClassDB::bind_method(D_METHOD("get_blend_shape_mode"), &ArrayMesh::get_blend_shape_mode);

	ClassDB::bind_method(D_METHOD("add_surface_from_arrays", "primitive", "arrays", "blend_shapes", "compress_flags"), &ArrayMesh::add_surface_from_arrays, DEFVAL(Array()), DEFVAL(ARRAY_COMPRESS_DEFAULT));
	ClassDB::bind_method(D_METHOD("clear_surfaces"), &ArrayMesh::clear_surfaces);
	ClassDB::bind_method(D_METHOD("surface_remove", "surf_idx"), &ArrayMesh::surface_remove);
	ClassDB::bind_method(D_METHOD("surface_update_region", "surf_idx", "offset", "data"), &ArrayMesh::surface_update_region);
	ClassDB::bind_method(D_METHOD("surface_get_array_len", "surf_idx"), &ArrayMesh::surface_get_array_len);
	ClassDB::bind_method(D_METHOD("surface_get_array_index_len", "surf_idx"), &ArrayMesh::surface_get_array_index_len);
	ClassDB::bind_method(D_METHOD("surface_get_format", "surf_idx"), &ArrayMesh::surface_get_format);
	ClassDB::bind_method(D_METHOD("surface_get_primitive_type", "surf_idx"), &ArrayMesh::surface_get_primitive_type);
	ClassDB::bind_method(D_METHOD("surface_find_by_name", "name"), &ArrayMesh::surface_find_by_name);
	ClassDB::bind_method(D_METHOD("surface_set_name", "surf_idx", "name"), &ArrayMesh::surface_set_name);
	ClassDB::bind_method(D_METHOD("surface_get_name", "surf_idx"), &ArrayMesh::surface_get_name);
	ClassDB::bind_method(D_METHOD("create_trimesh_shape"), &ArrayMesh::create_trimesh_shape);
	ClassDB::bind_method(D_METHOD("create_convex_shape", "clean", "simplify"), &ArrayMesh::create_convex_shape, DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("create_outline", "margin"), &ArrayMesh::create_outline);
	ClassDB::bind_method(D_METHOD("regen_normalmaps"), &ArrayMesh::regen_normalmaps);
	ClassDB::set_method_flags(get_class_static(), _scs_create("regen_normalmaps"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);
	ClassDB::bind_method(D_METHOD("lightmap_unwrap", "transform", "texel_size"), &ArrayMesh::lightmap_unwrap);
	ClassDB::set_method_flags(get_class_static(), _scs_create("lightmap_unwrap"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);
	ClassDB::bind_method(D_METHOD("get_faces"), &ArrayMesh::get_faces);
	ClassDB::bind_method(D_METHOD("generate_triangle_mesh"), &ArrayMesh::generate_triangle_mesh);

	ClassDB::bind_method(D_METHOD("set_custom_aabb", "aabb"), &ArrayMesh::set_custom_aabb);
	ClassDB::bind_method(D_METHOD("get_custom_aabb"), &ArrayMesh::get_custom_aabb);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "blend_shape_mode", PROPERTY_HINT_ENUM, "Normalized,Relative", PROPERTY_USAGE_NOEDITOR), "set_blend_shape_mode", "get_blend_shape_mode");
	ADD_PROPERTY(PropertyInfo(Variant::AABB, "custom_aabb", PROPERTY_HINT_NONE, ""), "set_custom_aabb", "get_custom_aabb");

	BIND_CONSTANT(NO_INDEX_ARRAY);
	BIND_CONSTANT(ARRAY_WEIGHTS_SIZE);

	BIND_ENUM_CONSTANT(ARRAY_VERTEX);
	BIND_ENUM_CONSTANT(ARRAY_NORMAL);
	BIND_ENUM_CONSTANT(ARRAY_TANGENT);
	BIND_ENUM_CONSTANT(ARRAY_COLOR);
	BIND_ENUM_CONSTANT(ARRAY_TEX_UV);
	BIND_ENUM_CONSTANT(ARRAY_TEX_UV2);
	BIND_ENUM_CONSTANT(ARRAY_BONES);
	BIND_ENUM_CONSTANT(ARRAY_WEIGHTS);
	BIND_ENUM_CONSTANT(ARRAY_INDEX);
	BIND_ENUM_CONSTANT(ARRAY_MAX);

	BIND_ENUM_CONSTANT(ARRAY_FORMAT_VERTEX);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_NORMAL);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_TANGENT);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_COLOR);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_TEX_UV);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_TEX_UV2);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_BONES);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_WEIGHTS);
	BIND_ENUM_CONSTANT(ARRAY_FORMAT_INDEX);
}

void ArrayMesh::reload_from_file() {
	VisualServer::get_singleton()->mesh_clear(mesh);
	surfaces.clear();
	clear_blend_shapes();
	clear_cache();

	Resource::reload_from_file();

	_change_notify();
}

ArrayMesh::ArrayMesh() {
	mesh = VisualServer::get_singleton()->mesh_create();
	blend_shape_mode = BLEND_SHAPE_MODE_RELATIVE;
}

ArrayMesh::~ArrayMesh() {
	VisualServer::get_singleton()->free(mesh);
}
