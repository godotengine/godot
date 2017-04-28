/*************************************************************************/
/*  mesh.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"
#include "surface_tool.h"

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

static const Mesh::ArrayType _array_types[] = {

	Mesh::ARRAY_VERTEX,
	Mesh::ARRAY_NORMAL,
	Mesh::ARRAY_TANGENT,
	Mesh::ARRAY_COLOR,
	Mesh::ARRAY_TEX_UV,
	Mesh::ARRAY_TEX_UV2,
	Mesh::ARRAY_BONES,
	Mesh::ARRAY_WEIGHTS,
	Mesh::ARRAY_INDEX
};

/* compatibility */
static const int _format_translate[] = {

	Mesh::ARRAY_FORMAT_VERTEX,
	Mesh::ARRAY_FORMAT_NORMAL,
	Mesh::ARRAY_FORMAT_TANGENT,
	Mesh::ARRAY_FORMAT_COLOR,
	Mesh::ARRAY_FORMAT_TEX_UV,
	Mesh::ARRAY_FORMAT_TEX_UV2,
	Mesh::ARRAY_FORMAT_BONES,
	Mesh::ARRAY_FORMAT_WEIGHTS,
	Mesh::ARRAY_FORMAT_INDEX,
};

bool Mesh::_set(const StringName &p_name, const Variant &p_value) {

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

			ERR_FAIL_COND_V(!d.has("primitive"), false);
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
			Rect3 aabb = d["aabb"];

			Vector<Rect3> bone_aabb;
			if (d.has("bone_aabb")) {
				Array baabb = d["bone_aabb"];
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

bool Mesh::_get(const StringName &p_name, Variant &r_ret) const {

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

	Vector<Rect3> skel_aabb = VS::get_singleton()->mesh_surface_get_skeleton_aabb(mesh, idx);
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

void Mesh::_get_property_list(List<PropertyInfo> *p_list) const {

	if (_is_generated())
		return;

	if (blend_shapes.size()) {
		p_list->push_back(PropertyInfo(Variant::POOL_STRING_ARRAY, "blend_shape/names", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::INT, "blend_shape/mode", PROPERTY_HINT_ENUM, "Normalized,Relative"));
	}

	for (int i = 0; i < surfaces.size(); i++) {

		p_list->push_back(PropertyInfo(Variant::DICTIONARY, "surfaces/" + itos(i), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::STRING, "surface_" + itos(i + 1) + "/name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::OBJECT, "surface_" + itos(i + 1) + "/material", PROPERTY_HINT_RESOURCE_TYPE, "Material", PROPERTY_USAGE_EDITOR));
	}

	p_list->push_back(PropertyInfo(Variant::RECT3, "custom_aabb/custom_aabb"));
}

void Mesh::_recompute_aabb() {

	// regenerate AABB
	aabb = Rect3();

	for (int i = 0; i < surfaces.size(); i++) {

		if (i == 0)
			aabb = surfaces[i].aabb;
		else
			aabb.merge_with(surfaces[i].aabb);
	}
}

void Mesh::add_surface(uint32_t p_format, PrimitiveType p_primitive, const PoolVector<uint8_t> &p_array, int p_vertex_count, const PoolVector<uint8_t> &p_index_array, int p_index_count, const Rect3 &p_aabb, const Vector<PoolVector<uint8_t> > &p_blend_shapes, const Vector<Rect3> &p_bone_aabbs) {

	Surface s;
	s.aabb = p_aabb;
	surfaces.push_back(s);

	VisualServer::get_singleton()->mesh_add_surface(mesh, p_format, (VS::PrimitiveType)p_primitive, p_array, p_vertex_count, p_index_array, p_index_count, p_aabb, p_blend_shapes, p_bone_aabbs);
}

void Mesh::add_surface_from_arrays(PrimitiveType p_primitive, const Array &p_arrays, const Array &p_blend_shapes, uint32_t p_flags) {

	ERR_FAIL_COND(p_arrays.size() != ARRAY_MAX);

	Surface s;

	VisualServer::get_singleton()->mesh_add_surface_from_arrays(mesh, (VisualServer::PrimitiveType)p_primitive, p_arrays, p_blend_shapes, p_flags);
	surfaces.push_back(s);

	/* make aABB? */ {

		PoolVector<Vector3> vertices = p_arrays[ARRAY_VERTEX];
		int len = vertices.size();
		ERR_FAIL_COND(len == 0);
		PoolVector<Vector3>::Read r = vertices.read();
		const Vector3 *vtx = r.ptr();

		// check AABB
		Rect3 aabb;
		for (int i = 0; i < len; i++) {

			if (i == 0)
				aabb.pos = vtx[i];
			else
				aabb.expand_to(vtx[i]);
		}

		surfaces[surfaces.size() - 1].aabb = aabb;

		_recompute_aabb();
	}

	triangle_mesh = Ref<TriangleMesh>();
	_change_notify();
	emit_changed();
}

Array Mesh::surface_get_arrays(int p_surface) const {

	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Array());
	return VisualServer::get_singleton()->mesh_surface_get_arrays(mesh, p_surface);
}
Array Mesh::surface_get_blend_shape_arrays(int p_surface) const {

	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Array());
	return Array();
}

int Mesh::get_surface_count() const {

	return surfaces.size();
}

void Mesh::add_blend_shape(const StringName &p_name) {

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

int Mesh::get_blend_shape_count() const {

	return blend_shapes.size();
}
StringName Mesh::get_blend_shape_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, blend_shapes.size(), StringName());
	return blend_shapes[p_index];
}
void Mesh::clear_blend_shapes() {

	if (surfaces.size()) {
		ERR_EXPLAIN("Can't set shape key count if surfaces are already created.");
		ERR_FAIL_COND(surfaces.size());
	}

	blend_shapes.clear();
}

void Mesh::set_blend_shape_mode(BlendShapeMode p_mode) {

	blend_shape_mode = p_mode;
	VS::get_singleton()->mesh_set_blend_shape_mode(mesh, (VS::BlendShapeMode)p_mode);
}

Mesh::BlendShapeMode Mesh::get_blend_shape_mode() const {

	return blend_shape_mode;
}

void Mesh::surface_remove(int p_idx) {

	ERR_FAIL_INDEX(p_idx, surfaces.size());
	VisualServer::get_singleton()->mesh_remove_surface(mesh, p_idx);
	surfaces.remove(p_idx);

	triangle_mesh = Ref<TriangleMesh>();
	_recompute_aabb();
	_change_notify();
	emit_changed();
}

int Mesh::surface_get_array_len(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), -1);
	return VisualServer::get_singleton()->mesh_surface_get_array_len(mesh, p_idx);
}

int Mesh::surface_get_array_index_len(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), -1);
	return VisualServer::get_singleton()->mesh_surface_get_array_index_len(mesh, p_idx);
}

uint32_t Mesh::surface_get_format(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), 0);
	return VisualServer::get_singleton()->mesh_surface_get_format(mesh, p_idx);
}

Mesh::PrimitiveType Mesh::surface_get_primitive_type(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), PRIMITIVE_LINES);
	return (PrimitiveType)VisualServer::get_singleton()->mesh_surface_get_primitive_type(mesh, p_idx);
}

void Mesh::surface_set_material(int p_idx, const Ref<Material> &p_material) {

	ERR_FAIL_INDEX(p_idx, surfaces.size());
	if (surfaces[p_idx].material == p_material)
		return;
	surfaces[p_idx].material = p_material;
	VisualServer::get_singleton()->mesh_surface_set_material(mesh, p_idx, p_material.is_null() ? RID() : p_material->get_rid());

	_change_notify("material");
}

void Mesh::surface_set_name(int p_idx, const String &p_name) {

	ERR_FAIL_INDEX(p_idx, surfaces.size());

	surfaces[p_idx].name = p_name;
}

String Mesh::surface_get_name(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), String());
	return surfaces[p_idx].name;
}

void Mesh::surface_set_custom_aabb(int p_idx, const Rect3 &p_aabb) {

	ERR_FAIL_INDEX(p_idx, surfaces.size());
	surfaces[p_idx].aabb = p_aabb;
	// set custom aabb too?
}

Ref<Material> Mesh::surface_get_material(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), Ref<Material>());
	return surfaces[p_idx].material;
}

void Mesh::add_surface_from_mesh_data(const Geometry::MeshData &p_mesh_data) {

	VisualServer::get_singleton()->mesh_add_surface_from_mesh_data(mesh, p_mesh_data);
	Rect3 aabb;
	for (int i = 0; i < p_mesh_data.vertices.size(); i++) {

		if (i == 0)
			aabb.pos = p_mesh_data.vertices[i];
		else
			aabb.expand_to(p_mesh_data.vertices[i]);
	}

	Surface s;
	s.aabb = aabb;
	if (surfaces.size() == 0)
		aabb = s.aabb;
	else
		aabb.merge_with(s.aabb);

	triangle_mesh = Ref<TriangleMesh>();

	surfaces.push_back(s);
	_change_notify();

	emit_changed();
}

RID Mesh::get_rid() const {

	return mesh;
}
Rect3 Mesh::get_aabb() const {

	return aabb;
}

void Mesh::set_custom_aabb(const Rect3 &p_custom) {

	custom_aabb = p_custom;
	VS::get_singleton()->mesh_set_custom_aabb(mesh, custom_aabb);
}

Rect3 Mesh::get_custom_aabb() const {

	return custom_aabb;
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

void Mesh::center_geometry() {

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

void Mesh::regen_normalmaps() {

	Vector<Ref<SurfaceTool> > surfs;
	for (int i = 0; i < get_surface_count(); i++) {

		Ref<SurfaceTool> st = memnew(SurfaceTool);
		st->create_from(Ref<Mesh>(this), i);
		surfs.push_back(st);
	}

	while (get_surface_count()) {
		surface_remove(0);
	}

	for (int i = 0; i < surfs.size(); i++) {

		surfs[i]->generate_tangents();
		surfs[i]->commit(Ref<Mesh>(this));
	}
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
		ERR_FAIL_COND_V(!vc, Ref<Mesh>());
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

	Ref<Mesh> newmesh = memnew(Mesh);
	newmesh->add_surface_from_arrays(PRIMITIVE_TRIANGLES, arrays);
	return newmesh;
}

void Mesh::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_blend_shape", "name"), &Mesh::add_blend_shape);
	ClassDB::bind_method(D_METHOD("get_blend_shape_count"), &Mesh::get_blend_shape_count);
	ClassDB::bind_method(D_METHOD("get_blend_shape_name", "index"), &Mesh::get_blend_shape_name);
	ClassDB::bind_method(D_METHOD("clear_blend_shapes"), &Mesh::clear_blend_shapes);
	ClassDB::bind_method(D_METHOD("set_blend_shape_mode", "mode"), &Mesh::set_blend_shape_mode);
	ClassDB::bind_method(D_METHOD("get_blend_shape_mode"), &Mesh::get_blend_shape_mode);

	ClassDB::bind_method(D_METHOD("add_surface_from_arrays", "primitive", "arrays", "blend_shapes", "compress_flags"), &Mesh::add_surface_from_arrays, DEFVAL(Array()), DEFVAL(ARRAY_COMPRESS_DEFAULT));
	ClassDB::bind_method(D_METHOD("get_surface_count"), &Mesh::get_surface_count);
	ClassDB::bind_method(D_METHOD("surface_remove", "surf_idx"), &Mesh::surface_remove);
	ClassDB::bind_method(D_METHOD("surface_get_array_len", "surf_idx"), &Mesh::surface_get_array_len);
	ClassDB::bind_method(D_METHOD("surface_get_array_index_len", "surf_idx"), &Mesh::surface_get_array_index_len);
	ClassDB::bind_method(D_METHOD("surface_get_format", "surf_idx"), &Mesh::surface_get_format);
	ClassDB::bind_method(D_METHOD("surface_get_primitive_type", "surf_idx"), &Mesh::surface_get_primitive_type);
	ClassDB::bind_method(D_METHOD("surface_set_material", "surf_idx", "material:Material"), &Mesh::surface_set_material);
	ClassDB::bind_method(D_METHOD("surface_get_material:Material", "surf_idx"), &Mesh::surface_get_material);
	ClassDB::bind_method(D_METHOD("surface_set_name", "surf_idx", "name"), &Mesh::surface_set_name);
	ClassDB::bind_method(D_METHOD("surface_get_name", "surf_idx"), &Mesh::surface_get_name);
	ClassDB::bind_method(D_METHOD("center_geometry"), &Mesh::center_geometry);
	ClassDB::set_method_flags(get_class_static(), _scs_create("center_geometry"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);
	ClassDB::bind_method(D_METHOD("regen_normalmaps"), &Mesh::regen_normalmaps);
	ClassDB::set_method_flags(get_class_static(), _scs_create("regen_normalmaps"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ClassDB::bind_method(D_METHOD("set_custom_aabb", "aabb"), &Mesh::set_custom_aabb);
	ClassDB::bind_method(D_METHOD("get_custom_aabb"), &Mesh::get_custom_aabb);

	BIND_CONSTANT(NO_INDEX_ARRAY);
	BIND_CONSTANT(ARRAY_WEIGHTS_SIZE);

	BIND_CONSTANT(ARRAY_VERTEX);
	BIND_CONSTANT(ARRAY_NORMAL);
	BIND_CONSTANT(ARRAY_TANGENT);
	BIND_CONSTANT(ARRAY_COLOR);
	BIND_CONSTANT(ARRAY_TEX_UV);
	BIND_CONSTANT(ARRAY_TEX_UV2);
	BIND_CONSTANT(ARRAY_BONES);
	BIND_CONSTANT(ARRAY_WEIGHTS);
	BIND_CONSTANT(ARRAY_INDEX);

	BIND_CONSTANT(ARRAY_FORMAT_VERTEX);
	BIND_CONSTANT(ARRAY_FORMAT_NORMAL);
	BIND_CONSTANT(ARRAY_FORMAT_TANGENT);
	BIND_CONSTANT(ARRAY_FORMAT_COLOR);
	BIND_CONSTANT(ARRAY_FORMAT_TEX_UV);
	BIND_CONSTANT(ARRAY_FORMAT_TEX_UV2);
	BIND_CONSTANT(ARRAY_FORMAT_BONES);
	BIND_CONSTANT(ARRAY_FORMAT_WEIGHTS);
	BIND_CONSTANT(ARRAY_FORMAT_INDEX);

	BIND_CONSTANT(PRIMITIVE_POINTS);
	BIND_CONSTANT(PRIMITIVE_LINES);
	BIND_CONSTANT(PRIMITIVE_LINE_STRIP);
	BIND_CONSTANT(PRIMITIVE_LINE_LOOP);
	BIND_CONSTANT(PRIMITIVE_TRIANGLES);
	BIND_CONSTANT(PRIMITIVE_TRIANGLE_STRIP);
	BIND_CONSTANT(PRIMITIVE_TRIANGLE_FAN);
}

Mesh::Mesh() {

	mesh = VisualServer::get_singleton()->mesh_create();
	blend_shape_mode = BLEND_SHAPE_MODE_RELATIVE;
}

Mesh::~Mesh() {

	VisualServer::get_singleton()->free(mesh);
}

////////////////////////

void QuadMesh::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_material", "material:Material"), &QuadMesh::set_material);
	ClassDB::bind_method(D_METHOD("get_material:Material"), &QuadMesh::get_material);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "Material"), "set_material", "get_material");
}

void QuadMesh::set_material(const Ref<Material> &p_material) {

	surface_set_material(0, p_material);
}

Ref<Material> QuadMesh::get_material() const {

	return surface_get_material(0);
}

QuadMesh::QuadMesh() {

	PoolVector<Vector3> faces;
	PoolVector<Vector3> normals;
	PoolVector<float> tangents;
	PoolVector<Vector2> uvs;

	faces.resize(4);
	normals.resize(4);
	tangents.resize(4 * 4);
	uvs.resize(4);

	for (int i = 0; i < 4; i++) {

		static const Vector3 quad_faces[4] = {
			Vector3(-1, -1, 0),
			Vector3(-1, 1, 0),
			Vector3(1, 1, 0),
			Vector3(1, -1, 0),
		};

		faces.set(i, quad_faces[i]);
		normals.set(i, Vector3(0, 0, 1));
		tangents.set(i * 4 + 0, 1.0);
		tangents.set(i * 4 + 1, 0.0);
		tangents.set(i * 4 + 2, 0.0);
		tangents.set(i * 4 + 3, 1.0);

		static const Vector2 quad_uv[4] = {
			Vector2(0, 1),
			Vector2(0, 0),
			Vector2(1, 0),
			Vector2(1, 1),
		};

		uvs.set(i, quad_uv[i]);
	}

	Array arr;
	arr.resize(ARRAY_MAX);
	arr[ARRAY_VERTEX] = faces;
	arr[ARRAY_NORMAL] = normals;
	arr[ARRAY_TANGENT] = tangents;
	arr[ARRAY_TEX_UV] = uvs;

	add_surface_from_arrays(PRIMITIVE_TRIANGLE_FAN, arr);
}
