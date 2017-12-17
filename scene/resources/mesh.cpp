/*************************************************************************/
/*  mesh.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "pair.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"
#include "surface_tool.h"

#include <stdlib.h>

void Mesh::_clear_triangle_mesh() const {

	triangle_mesh.unref();
	;
}

Ref<TriangleMesh> Mesh::generate_triangle_mesh() const {

	if (triangle_mesh.is_valid())
		return triangle_mesh;

	int facecount = 0;

	for (int i = 0; i < get_surface_count(); i++) {

		if (surface_get_primitive_type(i) != PRIMITIVE_TRIANGLES)
			continue;

		if (surface_get_format(i) & ARRAY_FORMAT_INDEX) {

			facecount += surface_get_array_index_len(i);
		} else {

			facecount += surface_get_array_len(i);
		}
	}

	if (facecount == 0 || (facecount % 3) != 0)
		return triangle_mesh;

	PoolVector<Vector3> faces;
	faces.resize(facecount);
	PoolVector<Vector3>::Write facesw = faces.write();

	int widx = 0;

	for (int i = 0; i < get_surface_count(); i++) {

		if (surface_get_primitive_type(i) != PRIMITIVE_TRIANGLES)
			continue;

		Array a = surface_get_arrays(i);

		int vc = surface_get_array_len(i);
		PoolVector<Vector3> vertices = a[ARRAY_VERTEX];
		PoolVector<Vector3>::Read vr = vertices.read();

		if (surface_get_format(i) & ARRAY_FORMAT_INDEX) {

			int ic = surface_get_array_index_len(i);
			PoolVector<int> indices = a[ARRAY_INDEX];
			PoolVector<int>::Read ir = indices.read();

			for (int i = 0; i < ic; i++) {
				int index = ir[i];
				facesw[widx++] = vr[index];
			}

		} else {

			for (int i = 0; i < vc; i++)
				facesw[widx++] = vr[i];
		}
	}

	facesw = PoolVector<Vector3>::Write();

	triangle_mesh = Ref<TriangleMesh>(memnew(TriangleMesh));
	triangle_mesh->create(faces);

	return triangle_mesh;
}

PoolVector<Face3> Mesh::get_faces() const {

	Ref<TriangleMesh> tm = generate_triangle_mesh();
	if (tm.is_valid())
		return tm->get_faces();
	return PoolVector<Face3>();
	/*
	for (int i=0;i<surfaces.size();i++) {

		if (VisualServer::get_singleton()->mesh_surface_get_primitive_type( mesh, i ) != VisualServer::PRIMITIVE_TRIANGLES )
			continue;

		PoolVector<int> indices;
		PoolVector<Vector3> vertices;

		vertices=VisualServer::get_singleton()->mesh_surface_get_array(mesh, i,VisualServer::ARRAY_VERTEX);

		int len=VisualServer::get_singleton()->mesh_surface_get_array_index_len(mesh, i);
		bool has_indices;

		if (len>0) {

			indices=VisualServer::get_singleton()->mesh_surface_get_array(mesh, i,VisualServer::ARRAY_INDEX);
			has_indices=true;

		} else {

			len=vertices.size();
			has_indices=false;
		}

		if (len<=0)
			continue;

		PoolVector<int>::Read indicesr = indices.read();
		const int *indicesptr = indicesr.ptr();

		PoolVector<Vector3>::Read verticesr = vertices.read();
		const Vector3 *verticesptr = verticesr.ptr();

		int old_faces=faces.size();
		int new_faces=old_faces+(len/3);

		faces.resize(new_faces);

		PoolVector<Face3>::Write facesw = faces.write();
		Face3 *facesptr=facesw.ptr();


		for (int i=0;i<len/3;i++) {

			Face3 face;

			for (int j=0;j<3;j++) {

				int idx=i*3+j;
				face.vertex[j] = has_indices ? verticesptr[ indicesptr[ idx ] ] : verticesptr[idx];
			}

			facesptr[i+old_faces]=face;
		}

	}
*/
}

Ref<Shape> Mesh::create_convex_shape() const {

	PoolVector<Vector3> vertices;

	for (int i = 0; i < get_surface_count(); i++) {

		Array a = surface_get_arrays(i);
		PoolVector<Vector3> v = a[ARRAY_VERTEX];
		vertices.append_array(v);
	}

	Ref<ConvexPolygonShape> shape = memnew(ConvexPolygonShape);
	shape->set_points(vertices);
	return shape;
}

Ref<Shape> Mesh::create_trimesh_shape() const {

	PoolVector<Face3> faces = get_faces();
	if (faces.size() == 0)
		return Ref<Shape>();

	PoolVector<Vector3> face_points;
	face_points.resize(faces.size() * 3);

	for (int i = 0; i < face_points.size(); i++) {

		Face3 f = faces.get(i / 3);
		face_points.set(i, f.vertex[i % 3]);
	}

	Ref<ConcavePolygonShape> shape = memnew(ConcavePolygonShape);
	shape->set_faces(face_points);
	return shape;
}

Ref<Mesh> Mesh::create_outline(float p_margin) const {

	Array arrays;
	int index_accum = 0;
	for (int i = 0; i < get_surface_count(); i++) {

		if (surface_get_primitive_type(i) != PRIMITIVE_TRIANGLES)
			continue;

		Array a = surface_get_arrays(i);
		int vcount = 0;

		if (i == 0) {
			arrays = a;
			PoolVector<Vector3> v = a[ARRAY_VERTEX];
			index_accum += v.size();
		} else {

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
						if (j == ARRAY_VERTEX)
							vcount = src.size();
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

	{
		PoolVector<int>::Write ir;
		PoolVector<int> indices = arrays[ARRAY_INDEX];
		bool has_indices = false;
		PoolVector<Vector3> vertices = arrays[ARRAY_VERTEX];
		int vc = vertices.size();
		ERR_FAIL_COND_V(!vc, Ref<ArrayMesh>());
		PoolVector<Vector3>::Write r = vertices.write();

		if (indices.size()) {
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
					if (d < 1.0)
						E->get() += n * (1.0 - d);
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

		r = PoolVector<Vector3>::Write();
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

			iw = PoolVector<int>::Write();
			arrays[ARRAY_INDEX] = new_indices;

		} else {

			for (int j = 0; j < vc; j += 3) {

				SWAP(ir[j + 1], ir[j + 2]);
			}
			ir = PoolVector<int>::Write();
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

	ADD_PROPERTYNZ(PropertyInfo(Variant::VECTOR2, "lightmap_size_hint"), "set_lightmap_size_hint", "get_lightmap_size_hint");

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

Mesh::Mesh() {
}

static const char *_array_name[] = {
	"vertex_array",
	"normal_array",
	"tangent_array",
	"color_array",
	"tex_uv_array",
	"tex_uv2_array",
	"bone_array",
	"weights_array",
	"index_array",
	NULL
};

static const ArrayMesh::ArrayType _array_types[] = {

	ArrayMesh::ARRAY_VERTEX,
	ArrayMesh::ARRAY_NORMAL,
	ArrayMesh::ARRAY_TANGENT,
	ArrayMesh::ARRAY_COLOR,
	ArrayMesh::ARRAY_TEX_UV,
	ArrayMesh::ARRAY_TEX_UV2,
	ArrayMesh::ARRAY_BONES,
	ArrayMesh::ARRAY_WEIGHTS,
	ArrayMesh::ARRAY_INDEX
};

/* compatibility */
static const int _format_translate[] = {

	ArrayMesh::ARRAY_FORMAT_VERTEX,
	ArrayMesh::ARRAY_FORMAT_NORMAL,
	ArrayMesh::ARRAY_FORMAT_TANGENT,
	ArrayMesh::ARRAY_FORMAT_COLOR,
	ArrayMesh::ARRAY_FORMAT_TEX_UV,
	ArrayMesh::ARRAY_FORMAT_TEX_UV2,
	ArrayMesh::ARRAY_FORMAT_BONES,
	ArrayMesh::ARRAY_FORMAT_WEIGHTS,
	ArrayMesh::ARRAY_FORMAT_INDEX,
};

bool ArrayMesh::_set(const StringName &p_name, const Variant &p_value) {

	String sname = p_name;

	if (p_name == "blend_shape/names") {

		PoolVector<String> sk = p_value;
		int sz = sk.size();
		PoolVector<String>::Read r = sk.read();
		for (int i = 0; i < sz; i++)
			add_blend_shape(r[i]);
		return true;
	}

	if (p_name == "blend_shape/mode") {

		set_blend_shape_mode(BlendShapeMode(int(p_value)));
		return true;
	}

	if (sname.begins_with("surface_")) {

		int sl = sname.find("/");
		if (sl == -1)
			return false;
		int idx = sname.substr(8, sl - 8).to_int() - 1;
		String what = sname.get_slicec('/', 1);
		if (what == "material")
			surface_set_material(idx, p_value);
		else if (what == "name")
			surface_set_name(idx, p_value);
		return true;
	}

	if (sname == "custom_aabb/custom_aabb") {

		set_custom_aabb(p_value);
		return true;
	}

	if (!sname.begins_with("surfaces"))
		return false;

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
			if (d.has("array_index_data"))
				array_index_data = d["array_index_data"];

			ERR_FAIL_COND_V(!d.has("format"), false);
			uint32_t format = d["format"];

			uint32_t primitive = d["primitive"];

			ERR_FAIL_COND_V(!d.has("vertex_count"), false);
			int vertex_count = d["vertex_count"];

			int index_count = 0;
			if (d.has("index_count"))
				index_count = d["index_count"];

			Vector<PoolVector<uint8_t> > blend_shapes;

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
					bone_aabb[i] = baabb[i];
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

	if (_is_generated())
		return false;

	String sname = p_name;

	if (p_name == "blend_shape/names") {

		PoolVector<String> sk;
		for (int i = 0; i < blend_shapes.size(); i++)
			sk.push_back(blend_shapes[i]);
		r_ret = sk;
		return true;
	} else if (p_name == "blend_shape/mode") {

		r_ret = get_blend_shape_mode();
		return true;
	} else if (sname.begins_with("surface_")) {

		int sl = sname.find("/");
		if (sl == -1)
			return false;
		int idx = sname.substr(8, sl - 8).to_int() - 1;
		String what = sname.get_slicec('/', 1);
		if (what == "material")
			r_ret = surface_get_material(idx);
		else if (what == "name")
			r_ret = surface_get_name(idx);
		return true;
	} else if (sname == "custom_aabb/custom_aabb") {

		r_ret = custom_aabb;
		return true;

	} else if (!sname.begins_with("surfaces"))
		return false;

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
	for (int i = 0; i < skel_aabb.size(); i++) {
		arr[i] = skel_aabb[i];
	}
	d["skeleton_aabb"] = arr;

	Vector<PoolVector<uint8_t> > blend_shape_data = VS::get_singleton()->mesh_surface_get_blend_shapes(mesh, idx);

	Array md;
	for (int i = 0; i < blend_shape_data.size(); i++) {
		md.push_back(blend_shape_data[i]);
	}

	d["blend_shape_data"] = md;

	Ref<Material> m = surface_get_material(idx);
	if (m.is_valid())
		d["material"] = m;
	String n = surface_get_name(idx);
	if (n != "")
		d["name"] = n;

	r_ret = d;

	return true;
}

void ArrayMesh::_get_property_list(List<PropertyInfo> *p_list) const {

	if (_is_generated())
		return;

	if (blend_shapes.size()) {
		p_list->push_back(PropertyInfo(Variant::POOL_STRING_ARRAY, "blend_shape/names", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::INT, "blend_shape/mode", PROPERTY_HINT_ENUM, "Normalized,Relative"));
	}

	for (int i = 0; i < surfaces.size(); i++) {

		p_list->push_back(PropertyInfo(Variant::DICTIONARY, "surfaces/" + itos(i), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::STRING, "surface_" + itos(i + 1) + "/name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		if (surfaces[i].is_2d) {
			p_list->push_back(PropertyInfo(Variant::OBJECT, "surface_" + itos(i + 1) + "/material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,CanvasItemMaterial", PROPERTY_USAGE_EDITOR));
		} else {
			p_list->push_back(PropertyInfo(Variant::OBJECT, "surface_" + itos(i + 1) + "/material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,SpatialMaterial", PROPERTY_USAGE_EDITOR));
		}
	}

	p_list->push_back(PropertyInfo(Variant::AABB, "custom_aabb/custom_aabb"));
}

void ArrayMesh::_recompute_aabb() {

	// regenerate AABB
	aabb = AABB();

	for (int i = 0; i < surfaces.size(); i++) {

		if (i == 0)
			aabb = surfaces[i].aabb;
		else
			aabb.merge_with(surfaces[i].aabb);
	}
}

void ArrayMesh::add_surface(uint32_t p_format, PrimitiveType p_primitive, const PoolVector<uint8_t> &p_array, int p_vertex_count, const PoolVector<uint8_t> &p_index_array, int p_index_count, const AABB &p_aabb, const Vector<PoolVector<uint8_t> > &p_blend_shapes, const Vector<AABB> &p_bone_aabbs) {

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
	surfaces.push_back(s);

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

			if (i == 0)
				aabb.position = vtx[i];
			else
				aabb.expand_to(vtx[i]);
		}

		surfaces[surfaces.size() - 1].aabb = aabb;
		surfaces[surfaces.size() - 1].is_2d = arr.get_type() == Variant::POOL_VECTOR2_ARRAY;

		_recompute_aabb();
	}

	_clear_triangle_mesh();
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

	if (surfaces.size()) {
		ERR_EXPLAIN("Can't add a shape key count if surfaces are already created.");
		ERR_FAIL_COND(surfaces.size());
	}

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
void ArrayMesh::clear_blend_shapes() {

	if (surfaces.size()) {
		ERR_EXPLAIN("Can't set shape key count if surfaces are already created.");
		ERR_FAIL_COND(surfaces.size());
	}

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

	_clear_triangle_mesh();
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
	if (surfaces[p_idx].material == p_material)
		return;
	surfaces[p_idx].material = p_material;
	VisualServer::get_singleton()->mesh_surface_set_material(mesh, p_idx, p_material.is_null() ? RID() : p_material->get_rid());

	_change_notify("material");
}

void ArrayMesh::surface_set_name(int p_idx, const String &p_name) {

	ERR_FAIL_INDEX(p_idx, surfaces.size());

	surfaces[p_idx].name = p_name;
}

String ArrayMesh::surface_get_name(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), String());
	return surfaces[p_idx].name;
}

void ArrayMesh::surface_update_region(int p_surface, int p_offset, const PoolVector<uint8_t> &p_data) {

	ERR_FAIL_INDEX(p_surface, surfaces.size());
	VS::get_singleton()->mesh_surface_update_region(mesh, p_surface, p_offset, p_data);
}

void ArrayMesh::surface_set_custom_aabb(int p_idx, const AABB &p_aabb) {

	ERR_FAIL_INDEX(p_idx, surfaces.size());
	surfaces[p_idx].aabb = p_aabb;
	// set custom aabb too?
}

Ref<Material> ArrayMesh::surface_get_material(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), Ref<Material>());
	return surfaces[p_idx].material;
}

void ArrayMesh::add_surface_from_mesh_data(const Geometry::MeshData &p_mesh_data) {

	VisualServer::get_singleton()->mesh_add_surface_from_mesh_data(mesh, p_mesh_data);
	AABB aabb;
	for (int i = 0; i < p_mesh_data.vertices.size(); i++) {

		if (i == 0)
			aabb.position = p_mesh_data.vertices[i];
		else
			aabb.expand_to(p_mesh_data.vertices[i]);
	}

	Surface s;
	s.aabb = aabb;
	if (surfaces.size() == 0)
		aabb = s.aabb;
	else
		aabb.merge_with(s.aabb);

	_clear_triangle_mesh();

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

void ArrayMesh::set_custom_aabb(const AABB &p_custom) {

	custom_aabb = p_custom;
	VS::get_singleton()->mesh_set_custom_aabb(mesh, custom_aabb);
}

AABB ArrayMesh::get_custom_aabb() const {

	return custom_aabb;
}

void ArrayMesh::center_geometry() {

	/*
	Vector3 ofs = aabb.pos+aabb.size*0.5;

	for(int i=0;i<get_surface_count();i++) {

		PoolVector<Vector3> geom = surface_get_array(i,ARRAY_VERTEX);
		int gc =geom.size();
		PoolVector<Vector3>::Write w = geom.write();
		surfaces[i].aabb.pos-=ofs;

		for(int i=0;i<gc;i++) {

			w[i]-=ofs;
		}

		w = PoolVector<Vector3>::Write();

		surface_set_array(i,ARRAY_VERTEX,geom);

	}

	aabb.pos-=ofs;

*/
}

void ArrayMesh::regen_normalmaps() {

	Vector<Ref<SurfaceTool> > surfs;
	for (int i = 0; i < get_surface_count(); i++) {

		Ref<SurfaceTool> st = memnew(SurfaceTool);
		st->create_from(Ref<ArrayMesh>(this), i);
		surfs.push_back(st);
	}

	while (get_surface_count()) {
		surface_remove(0);
	}

	for (int i = 0; i < surfs.size(); i++) {

		surfs[i]->generate_tangents();
		surfs[i]->commit(Ref<ArrayMesh>(this));
	}
}

//dirty hack
bool (*array_mesh_lightmap_unwrap_callback)(float p_texel_size, const float *p_vertices, const float *p_normals, int p_vertex_count, const int *p_indices, const int *p_face_materials, int p_index_count, float **r_uv, int **r_vertex, int *r_vertex_count, int **r_index, int *r_index_count, int *r_size_hint_x, int *r_size_hint_y) = NULL;

struct ArrayMeshLightmapSurface {

	Ref<Material> material;
	Vector<SurfaceTool::Vertex> vertices;
	Mesh::PrimitiveType primitive;
	uint32_t format;
};

Error ArrayMesh::lightmap_unwrap(const Transform &p_base_transform, float p_texel_size) {

	ERR_FAIL_COND_V(!array_mesh_lightmap_unwrap_callback, ERR_UNCONFIGURED);
	ERR_EXPLAIN("Can't unwrap mesh with blend shapes");
	ERR_FAIL_COND_V(blend_shapes.size() != 0, ERR_UNAVAILABLE);

	Vector<float> vertices;
	Vector<float> normals;
	Vector<int> indices;
	Vector<int> face_materials;
	Vector<float> uv;
	Vector<Pair<int, int> > uv_index;

	Vector<ArrayMeshLightmapSurface> surfaces;
	for (int i = 0; i < get_surface_count(); i++) {
		ArrayMeshLightmapSurface s;
		s.primitive = surface_get_primitive_type(i);

		if (s.primitive != Mesh::PRIMITIVE_TRIANGLES) {
			ERR_EXPLAIN("Only triangles are supported for lightmap unwrap");
			ERR_FAIL_V(ERR_UNAVAILABLE);
		}
		s.format = surface_get_format(i);
		if (!(s.format & ARRAY_FORMAT_NORMAL)) {
			ERR_EXPLAIN("Normals are required for lightmap unwrap");
			ERR_FAIL_V(ERR_UNAVAILABLE);
		}

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
		uv_index.resize(vertex_ofs + vc);

		for (int j = 0; j < vc; j++) {

			Vector3 v = p_base_transform.xform(r[j]);
			Vector3 n = p_base_transform.basis.xform(rn[j]).normalized();

			vertices[(j + vertex_ofs) * 3 + 0] = v.x;
			vertices[(j + vertex_ofs) * 3 + 1] = v.y;
			vertices[(j + vertex_ofs) * 3 + 2] = v.z;
			normals[(j + vertex_ofs) * 3 + 0] = n.x;
			normals[(j + vertex_ofs) * 3 + 1] = n.y;
			normals[(j + vertex_ofs) * 3 + 2] = n.z;
			uv_index[j + vertex_ofs] = Pair<int, int>(i, j);
		}

		PoolVector<int> rindices = arrays[Mesh::ARRAY_INDEX];
		int ic = rindices.size();

		if (ic == 0) {

			for (int j = 0; j < vc / 3; j++) {
				if (Face3(r[j * 3 + 0], r[j * 3 + 1], r[j * 3 + 2]).is_degenerate())
					continue;

				indices.push_back(vertex_ofs + j * 3 + 0);
				indices.push_back(vertex_ofs + j * 3 + 1);
				indices.push_back(vertex_ofs + j * 3 + 2);
				face_materials.push_back(i);
			}

		} else {
			PoolVector<int>::Read ri = rindices.read();

			for (int j = 0; j < ic / 3; j++) {
				if (Face3(r[ri[j * 3 + 0]], r[ri[j * 3 + 1]], r[ri[j * 3 + 2]]).is_degenerate())
					continue;
				indices.push_back(vertex_ofs + ri[j * 3 + 0]);
				indices.push_back(vertex_ofs + ri[j * 3 + 1]);
				indices.push_back(vertex_ofs + ri[j * 3 + 2]);
				face_materials.push_back(i);
			}
		}

		surfaces.push_back(s);
	}

	//unwrap

	float *gen_uvs;
	int *gen_vertices;
	int *gen_indices;
	int gen_vertex_count;
	int gen_index_count;
	int size_x;
	int size_y;

	bool ok = array_mesh_lightmap_unwrap_callback(p_texel_size, vertices.ptr(), normals.ptr(), vertices.size() / 3, indices.ptr(), face_materials.ptr(), indices.size(), &gen_uvs, &gen_vertices, &gen_vertex_count, &gen_indices, &gen_index_count, &size_x, &size_y);

	if (!ok) {
		return ERR_CANT_CREATE;
	}

	//remove surfaces
	while (get_surface_count()) {
		surface_remove(0);
	}

	//create surfacetools for each surface..
	Vector<Ref<SurfaceTool> > surfaces_tools;

	for (int i = 0; i < surfaces.size(); i++) {
		Ref<SurfaceTool> st;
		st.instance();
		st->begin(Mesh::PRIMITIVE_TRIANGLES);
		st->set_material(surfaces[i].material);
		surfaces_tools.push_back(st); //stay there
	}

	print_line("gen indices: " + itos(gen_index_count));
	//go through all indices
	for (int i = 0; i < gen_index_count; i += 3) {

		ERR_FAIL_INDEX_V(gen_vertices[gen_indices[i + 0]], uv_index.size(), ERR_BUG);
		ERR_FAIL_INDEX_V(gen_vertices[gen_indices[i + 1]], uv_index.size(), ERR_BUG);
		ERR_FAIL_INDEX_V(gen_vertices[gen_indices[i + 2]], uv_index.size(), ERR_BUG);

		ERR_FAIL_COND_V(uv_index[gen_vertices[gen_indices[i + 0]]].first != uv_index[gen_vertices[gen_indices[i + 1]]].first || uv_index[gen_vertices[gen_indices[i + 0]]].first != uv_index[gen_vertices[gen_indices[i + 2]]].first, ERR_BUG);

		int surface = uv_index[gen_vertices[gen_indices[i + 0]]].first;

		for (int j = 0; j < 3; j++) {

			int vertex_idx = gen_vertices[gen_indices[i + j]];

			SurfaceTool::Vertex v = surfaces[surface].vertices[uv_index[gen_vertices[gen_indices[i + j]]].second];

			if (surfaces[surface].format & ARRAY_FORMAT_COLOR) {
				surfaces_tools[surface]->add_color(v.color);
			}
			if (surfaces[surface].format & ARRAY_FORMAT_TEX_UV) {
				surfaces_tools[surface]->add_uv(v.uv);
			}
			if (surfaces[surface].format & ARRAY_FORMAT_NORMAL) {
				surfaces_tools[surface]->add_normal(v.normal);
			}
			if (surfaces[surface].format & ARRAY_FORMAT_TANGENT) {
				Plane t;
				t.normal = v.tangent;
				t.d = v.binormal.dot(v.normal.cross(v.tangent)) < 0 ? -1 : 1;
				surfaces_tools[surface]->add_tangent(t);
			}
			if (surfaces[surface].format & ARRAY_FORMAT_BONES) {
				surfaces_tools[surface]->add_bones(v.bones);
			}
			if (surfaces[surface].format & ARRAY_FORMAT_WEIGHTS) {
				surfaces_tools[surface]->add_weights(v.weights);
			}

			Vector2 uv2(gen_uvs[gen_indices[i + j] * 2 + 0], gen_uvs[gen_indices[i + j] * 2 + 1]);
			surfaces_tools[surface]->add_uv2(uv2);

			surfaces_tools[surface]->add_vertex(v.vertex);
		}
	}

	//free stuff
	::free(gen_vertices);
	::free(gen_indices);
	::free(gen_uvs);

	//generate surfaces

	for (int i = 0; i < surfaces_tools.size(); i++) {
		surfaces_tools[i]->index();
		surfaces_tools[i]->commit(Ref<ArrayMesh>((ArrayMesh *)this), surfaces[i].format);
	}

	set_lightmap_size_hint(Size2(size_x, size_y));

	return OK;
}

void ArrayMesh::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_blend_shape", "name"), &ArrayMesh::add_blend_shape);
	ClassDB::bind_method(D_METHOD("get_blend_shape_count"), &ArrayMesh::get_blend_shape_count);
	ClassDB::bind_method(D_METHOD("get_blend_shape_name", "index"), &ArrayMesh::get_blend_shape_name);
	ClassDB::bind_method(D_METHOD("clear_blend_shapes"), &ArrayMesh::clear_blend_shapes);
	ClassDB::bind_method(D_METHOD("set_blend_shape_mode", "mode"), &ArrayMesh::set_blend_shape_mode);
	ClassDB::bind_method(D_METHOD("get_blend_shape_mode"), &ArrayMesh::get_blend_shape_mode);

	ClassDB::bind_method(D_METHOD("add_surface_from_arrays", "primitive", "arrays", "blend_shapes", "compress_flags"), &ArrayMesh::add_surface_from_arrays, DEFVAL(Array()), DEFVAL(ARRAY_COMPRESS_DEFAULT));
	ClassDB::bind_method(D_METHOD("get_surface_count"), &ArrayMesh::get_surface_count);
	ClassDB::bind_method(D_METHOD("surface_remove", "surf_idx"), &ArrayMesh::surface_remove);
	ClassDB::bind_method(D_METHOD("surface_update_region", "surf_idx", "offset", "data"), &ArrayMesh::surface_update_region);
	ClassDB::bind_method(D_METHOD("surface_get_array_len", "surf_idx"), &ArrayMesh::surface_get_array_len);
	ClassDB::bind_method(D_METHOD("surface_get_array_index_len", "surf_idx"), &ArrayMesh::surface_get_array_index_len);
	ClassDB::bind_method(D_METHOD("surface_get_format", "surf_idx"), &ArrayMesh::surface_get_format);
	ClassDB::bind_method(D_METHOD("surface_get_primitive_type", "surf_idx"), &ArrayMesh::surface_get_primitive_type);
	ClassDB::bind_method(D_METHOD("surface_set_material", "surf_idx", "material"), &ArrayMesh::surface_set_material);
	ClassDB::bind_method(D_METHOD("surface_get_material", "surf_idx"), &ArrayMesh::surface_get_material);
	ClassDB::bind_method(D_METHOD("surface_set_name", "surf_idx", "name"), &ArrayMesh::surface_set_name);
	ClassDB::bind_method(D_METHOD("surface_get_name", "surf_idx"), &ArrayMesh::surface_get_name);
	ClassDB::bind_method(D_METHOD("surface_get_arrays", "surf_idx"), &ArrayMesh::surface_get_arrays);
	ClassDB::bind_method(D_METHOD("surface_get_blend_shape_arrays", "surf_idx"), &ArrayMesh::surface_get_blend_shape_arrays);
	ClassDB::bind_method(D_METHOD("create_trimesh_shape"), &ArrayMesh::create_trimesh_shape);
	ClassDB::bind_method(D_METHOD("create_convex_shape"), &ArrayMesh::create_convex_shape);
	ClassDB::bind_method(D_METHOD("create_outline", "margin"), &ArrayMesh::create_outline);
	ClassDB::bind_method(D_METHOD("center_geometry"), &ArrayMesh::center_geometry);
	ClassDB::set_method_flags(get_class_static(), _scs_create("center_geometry"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);
	ClassDB::bind_method(D_METHOD("regen_normalmaps"), &ArrayMesh::regen_normalmaps);
	ClassDB::set_method_flags(get_class_static(), _scs_create("regen_normalmaps"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);
	ClassDB::bind_method(D_METHOD("get_faces"), &ArrayMesh::get_faces);
	ClassDB::bind_method(D_METHOD("generate_triangle_mesh"), &ArrayMesh::generate_triangle_mesh);

	ClassDB::bind_method(D_METHOD("set_custom_aabb", "aabb"), &ArrayMesh::set_custom_aabb);
	ClassDB::bind_method(D_METHOD("get_custom_aabb"), &ArrayMesh::get_custom_aabb);

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
