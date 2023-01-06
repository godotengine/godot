/**************************************************************************/
/*  mesh_data_tool.cpp                                                    */
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

#include "mesh_data_tool.h"

void MeshDataTool::clear() {
	vertices.clear();
	edges.clear();
	faces.clear();
	material = Ref<Material>();
	format = 0;
}

Error MeshDataTool::create_from_surface(const Ref<ArrayMesh> &p_mesh, int p_surface) {
	ERR_FAIL_COND_V(p_mesh.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_mesh->surface_get_primitive_type(p_surface) != Mesh::PRIMITIVE_TRIANGLES, ERR_INVALID_PARAMETER);

	uint32_t custom_fmt_byte_size[RS::ARRAY_CUSTOM_MAX] = { 4, 4, 4, 8, 4, 8, 12, 16 };

	Array arrays = p_mesh->surface_get_arrays(p_surface);
	ERR_FAIL_COND_V(arrays.is_empty(), ERR_INVALID_PARAMETER);

	Vector<Vector3> varray = arrays[Mesh::ARRAY_VERTEX];

	int vcount = varray.size();
	ERR_FAIL_COND_V(vcount == 0, ERR_INVALID_PARAMETER);

	Vector<int> indices;

	if (arrays[Mesh::ARRAY_INDEX].get_type() != Variant::NIL) {
		indices = arrays[Mesh::ARRAY_INDEX];
	} else {
		//make code simpler
		indices.resize(vcount);
		int *iw = indices.ptrw();
		for (int i = 0; i < vcount; i++) {
			iw[i] = i;
		}
	}

	int icount = indices.size();
	const int *r = indices.ptr();

	ERR_FAIL_COND_V(icount == 0, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(icount % 3, ERR_INVALID_PARAMETER);
	for (int i = 0; i < icount; i++) {
		ERR_FAIL_INDEX_V(r[i], vcount, ERR_INVALID_PARAMETER);
	}

	clear();
	format = p_mesh->surface_get_format(p_surface);
	material = p_mesh->surface_get_material(p_surface);

	const Vector3 *vr = varray.ptr();

	const Vector3 *nr = nullptr;
	if (arrays[Mesh::ARRAY_NORMAL].get_type() != Variant::NIL) {
		nr = arrays[Mesh::ARRAY_NORMAL].operator Vector<Vector3>().ptr();
	}

	const real_t *ta = nullptr;
	if (arrays[Mesh::ARRAY_TANGENT].get_type() != Variant::NIL) {
		ta = arrays[Mesh::ARRAY_TANGENT].operator Vector<real_t>().ptr();
	}

	const Vector2 *uv = nullptr;
	if (arrays[Mesh::ARRAY_TEX_UV].get_type() != Variant::NIL) {
		uv = arrays[Mesh::ARRAY_TEX_UV].operator Vector<Vector2>().ptr();
	}
	const Vector2 *uv2 = nullptr;
	if (arrays[Mesh::ARRAY_TEX_UV2].get_type() != Variant::NIL) {
		uv2 = arrays[Mesh::ARRAY_TEX_UV2].operator Vector<Vector2>().ptr();
	}

	const Color *col = nullptr;
	if (arrays[Mesh::ARRAY_COLOR].get_type() != Variant::NIL) {
		col = arrays[Mesh::ARRAY_COLOR].operator Vector<Color>().ptr();
	}

	const int *bo = nullptr;
	if (arrays[Mesh::ARRAY_BONES].get_type() != Variant::NIL) {
		bo = arrays[Mesh::ARRAY_BONES].operator Vector<int>().ptr();
	}

	const float *we = nullptr;
	if (arrays[Mesh::ARRAY_WEIGHTS].get_type() != Variant::NIL) {
		we = arrays[Mesh::ARRAY_WEIGHTS].operator Vector<float>().ptr();
	}

	const uint8_t *cu[Mesh::ARRAY_CUSTOM_COUNT] = {};
	uint32_t custom_type_byte_sizes[Mesh::ARRAY_CUSTOM_COUNT] = {};
	for (uint32_t i = 0; i < Mesh::ARRAY_CUSTOM_COUNT; ++i) {
		uint32_t arrType = arrays[Mesh::ARRAY_CUSTOM0 + i].get_type();
		if (arrType == Variant::PACKED_FLOAT32_ARRAY) {
			cu[i] = (const uint8_t *)arrays[Mesh::ARRAY_CUSTOM0 + i].operator Vector<float>().ptr();
		} else if (arrType == Variant::PACKED_BYTE_ARRAY) {
			cu[i] = arrays[Mesh::ARRAY_CUSTOM0 + i].operator Vector<uint8_t>().ptr();
		}

		uint32_t type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + i * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
		custom_type_byte_sizes[i] = custom_fmt_byte_size[type];
	}

	vertices.resize(vcount);

	for (int i = 0; i < vcount; i++) {
		Vertex v;
		v.vertex = vr[i];
		if (nr) {
			v.normal = nr[i];
		}
		if (ta) {
			v.tangent = Plane(ta[i * 4 + 0], ta[i * 4 + 1], ta[i * 4 + 2], ta[i * 4 + 3]);
		}
		if (uv) {
			v.uv = uv[i];
		}
		if (uv2) {
			v.uv2 = uv2[i];
		}
		if (col) {
			v.color = col[i];
		}

		if (we) {
			v.weights.push_back(we[i * 4 + 0]);
			v.weights.push_back(we[i * 4 + 1]);
			v.weights.push_back(we[i * 4 + 2]);
			v.weights.push_back(we[i * 4 + 3]);
		}

		if (bo) {
			v.bones.push_back(bo[i * 4 + 0]);
			v.bones.push_back(bo[i * 4 + 1]);
			v.bones.push_back(bo[i * 4 + 2]);
			v.bones.push_back(bo[i * 4 + 3]);
		}

		for (uint32_t j = 0; j < Mesh::ARRAY_CUSTOM_COUNT; ++j) {
			if (cu[j]) {
				memcpy(&v.custom[j], &(cu[j][i * custom_type_byte_sizes[j]]), custom_type_byte_sizes[j]);
			}
		}

		vertices.write[i] = v;
	}

	HashMap<Point2i, int> edge_indices;

	for (int i = 0; i < icount; i += 3) {
		Vertex *v[3] = { &vertices.write[r[i + 0]], &vertices.write[r[i + 1]], &vertices.write[r[i + 2]] };

		int fidx = faces.size();
		Face face;

		for (int j = 0; j < 3; j++) {
			face.v[j] = r[i + j];

			Point2i edge(r[i + j], r[i + (j + 1) % 3]);
			if (edge.x > edge.y) {
				SWAP(edge.x, edge.y);
			}

			if (edge_indices.has(edge)) {
				face.edges[j] = edge_indices[edge];

			} else {
				face.edges[j] = edge_indices.size();
				edge_indices[edge] = face.edges[j];
				Edge e;
				e.vertex[0] = edge.x;
				e.vertex[1] = edge.y;
				edges.push_back(e);
				v[j]->edges.push_back(face.edges[j]);
				v[(j + 1) % 3]->edges.push_back(face.edges[j]);
			}

			edges.write[face.edges[j]].faces.push_back(fidx);
			v[j]->faces.push_back(fidx);
		}

		faces.push_back(face);
	}

	return OK;
}

Error MeshDataTool::commit_to_surface(const Ref<ArrayMesh> &p_mesh) {
	ERR_FAIL_COND_V(p_mesh.is_null(), ERR_INVALID_PARAMETER);
	Array arr;
	arr.resize(Mesh::ARRAY_MAX);

	uint32_t custom_fmt_size[RS::ARRAY_CUSTOM_MAX] = { 4, 4, 4, 8, 1, 2, 3, 4 };
	uint32_t custom_fmt_byte_size[RS::ARRAY_CUSTOM_MAX] = { 4, 4, 4, 8, 4, 8, 12, 16 };

	int vcount = vertices.size();

	Vector<Vector3> v;
	Vector<Vector3> n;
	Vector<real_t> t;
	Vector<Vector2> u;
	Vector<Vector2> u2;
	Vector<Color> c;
	Vector<int> b;
	Vector<real_t> w;
	Vector<int> in;
	Vector<float> cuf[Mesh::ARRAY_CUSTOM_COUNT];
	Vector<uint8_t> cuu[Mesh::ARRAY_CUSTOM_COUNT];

	{
		uint8_t *cu_ptr[Mesh::ARRAY_CUSTOM_COUNT] = {};
		uint32_t custom_type_byte_sizes[Mesh::ARRAY_CUSTOM_COUNT] = {};
		for (uint32_t i = 0; i < Mesh::ARRAY_CUSTOM_COUNT; ++i) {
			uint32_t type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + i * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
			custom_type_byte_sizes[i] = custom_fmt_byte_size[type];

			if (format & (Mesh::ARRAY_FORMAT_CUSTOM0 << i)) {
				if (type > Mesh::ARRAY_CUSTOM_RGBA_HALF) {
					// floating point type
					cuf[i].resize(vcount * custom_fmt_size[type]);
					cu_ptr[i] = (uint8_t *)cuf[i].ptrw();
				} else {
					// non floating point type
					cuu[i].resize(vcount * custom_fmt_size[type]);
					cu_ptr[i] = (uint8_t *)cuu[i].ptrw();
				}
			}
		}

		v.resize(vcount);
		Vector3 *vr = v.ptrw();

		Vector3 *nr = nullptr;
		if (format & Mesh::ARRAY_FORMAT_NORMAL) {
			n.resize(vcount);
			nr = n.ptrw();
		}

		real_t *ta = nullptr;
		if (format & Mesh::ARRAY_FORMAT_TANGENT) {
			t.resize(vcount * 4);
			ta = t.ptrw();
		}

		Vector2 *uv = nullptr;
		if (format & Mesh::ARRAY_FORMAT_TEX_UV) {
			u.resize(vcount);
			uv = u.ptrw();
		}

		Vector2 *uv2 = nullptr;
		if (format & Mesh::ARRAY_FORMAT_TEX_UV2) {
			u2.resize(vcount);
			uv2 = u2.ptrw();
		}

		Color *col = nullptr;
		if (format & Mesh::ARRAY_FORMAT_COLOR) {
			c.resize(vcount);
			col = c.ptrw();
		}

		int *bo = nullptr;
		if (format & Mesh::ARRAY_FORMAT_BONES) {
			b.resize(vcount * 4);
			bo = b.ptrw();
		}

		real_t *we = nullptr;
		if (format & Mesh::ARRAY_FORMAT_WEIGHTS) {
			w.resize(vcount * 4);
			we = w.ptrw();
		}

		for (int i = 0; i < vcount; i++) {
			const Vertex &vtx = vertices[i];

			vr[i] = vtx.vertex;
			if (nr) {
				nr[i] = vtx.normal;
			}
			if (ta) {
				ta[i * 4 + 0] = vtx.tangent.normal.x;
				ta[i * 4 + 1] = vtx.tangent.normal.y;
				ta[i * 4 + 2] = vtx.tangent.normal.z;
				ta[i * 4 + 3] = vtx.tangent.d;
			}
			if (uv) {
				uv[i] = vtx.uv;
			}
			if (uv2) {
				uv2[i] = vtx.uv2;
			}
			if (col) {
				col[i] = vtx.color;
			}

			if (we) {
				we[i * 4 + 0] = vtx.weights[0];
				we[i * 4 + 1] = vtx.weights[1];
				we[i * 4 + 2] = vtx.weights[2];
				we[i * 4 + 3] = vtx.weights[3];
			}

			if (bo) {
				bo[i * 4 + 0] = vtx.bones[0];
				bo[i * 4 + 1] = vtx.bones[1];
				bo[i * 4 + 2] = vtx.bones[2];
				bo[i * 4 + 3] = vtx.bones[3];
			}

			for (uint32_t j = 0; j < Mesh::ARRAY_CUSTOM_COUNT; ++j) {
				if (cu_ptr[j]) {
					memcpy(cu_ptr[j] + i * custom_type_byte_sizes[j], &vtx.custom[j], custom_type_byte_sizes[j]);
				}
			}
		}

		int fc = faces.size();
		in.resize(fc * 3);
		int *iw = in.ptrw();
		for (int i = 0; i < fc; i++) {
			iw[i * 3 + 0] = faces[i].v[0];
			iw[i * 3 + 1] = faces[i].v[1];
			iw[i * 3 + 2] = faces[i].v[2];
		}
	}

	arr[Mesh::ARRAY_VERTEX] = v;
	arr[Mesh::ARRAY_INDEX] = in;
	if (n.size()) {
		arr[Mesh::ARRAY_NORMAL] = n;
	}
	if (c.size()) {
		arr[Mesh::ARRAY_COLOR] = c;
	}
	if (u.size()) {
		arr[Mesh::ARRAY_TEX_UV] = u;
	}
	if (u2.size()) {
		arr[Mesh::ARRAY_TEX_UV2] = u2;
	}
	if (t.size()) {
		arr[Mesh::ARRAY_TANGENT] = t;
	}
	if (b.size()) {
		arr[Mesh::ARRAY_BONES] = b;
	}
	if (w.size()) {
		arr[Mesh::ARRAY_WEIGHTS] = w;
	}
	for (uint32_t i = 0; i < Mesh::ARRAY_CUSTOM_COUNT; ++i) {
		if (cuf[i].size()) {
			arr[Mesh::ARRAY_CUSTOM0 + i] = cuf[i];
		} else if (cuu[i].size()) {
			arr[Mesh::ARRAY_CUSTOM0 + i] = cuu[i];
		}
	}

	Ref<ArrayMesh> ncmesh = p_mesh;
	int sc = ncmesh->get_surface_count();
	ncmesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arr, Array(), Dictionary(), format);
	ncmesh->surface_set_material(sc, material);

	return OK;
}

int MeshDataTool::get_format() const {
	return format;
}

int MeshDataTool::get_vertex_count() const {
	return vertices.size();
}

int MeshDataTool::get_edge_count() const {
	return edges.size();
}

int MeshDataTool::get_face_count() const {
	return faces.size();
}

Vector3 MeshDataTool::get_vertex(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Vector3());
	return vertices[p_idx].vertex;
}

void MeshDataTool::set_vertex(int p_idx, const Vector3 &p_vertex) {
	ERR_FAIL_INDEX(p_idx, vertices.size());
	vertices.write[p_idx].vertex = p_vertex;
}

Vector3 MeshDataTool::get_vertex_normal(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Vector3());
	return vertices[p_idx].normal;
}

void MeshDataTool::set_vertex_normal(int p_idx, const Vector3 &p_normal) {
	ERR_FAIL_INDEX(p_idx, vertices.size());
	vertices.write[p_idx].normal = p_normal;
	format |= Mesh::ARRAY_FORMAT_NORMAL;
}

Plane MeshDataTool::get_vertex_tangent(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Plane());
	return vertices[p_idx].tangent;
}

void MeshDataTool::set_vertex_tangent(int p_idx, const Plane &p_tangent) {
	ERR_FAIL_INDEX(p_idx, vertices.size());
	vertices.write[p_idx].tangent = p_tangent;
	format |= Mesh::ARRAY_FORMAT_TANGENT;
}

Vector2 MeshDataTool::get_vertex_uv(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Vector2());
	return vertices[p_idx].uv;
}

void MeshDataTool::set_vertex_uv(int p_idx, const Vector2 &p_uv) {
	ERR_FAIL_INDEX(p_idx, vertices.size());
	vertices.write[p_idx].uv = p_uv;
	format |= Mesh::ARRAY_FORMAT_TEX_UV;
}

Vector2 MeshDataTool::get_vertex_uv2(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Vector2());
	return vertices[p_idx].uv2;
}

void MeshDataTool::set_vertex_uv2(int p_idx, const Vector2 &p_uv2) {
	ERR_FAIL_INDEX(p_idx, vertices.size());
	vertices.write[p_idx].uv2 = p_uv2;
	format |= Mesh::ARRAY_FORMAT_TEX_UV2;
}

Color MeshDataTool::get_vertex_color(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Color());
	return vertices[p_idx].color;
}

void MeshDataTool::set_vertex_color(int p_idx, const Color &p_color) {
	ERR_FAIL_INDEX(p_idx, vertices.size());
	vertices.write[p_idx].color = p_color;
	format |= Mesh::ARRAY_FORMAT_COLOR;
}

Vector<int> MeshDataTool::get_vertex_bones(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Vector<int>());
	return vertices[p_idx].bones;
}

void MeshDataTool::set_vertex_bones(int p_idx, const Vector<int> &p_bones) {
	ERR_FAIL_INDEX(p_idx, vertices.size());
	ERR_FAIL_COND(p_bones.size() != 4);
	vertices.write[p_idx].bones = p_bones;
	format |= Mesh::ARRAY_FORMAT_BONES;
}

Vector<float> MeshDataTool::get_vertex_weights(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Vector<float>());
	return vertices[p_idx].weights;
}

void MeshDataTool::set_vertex_weights(int p_idx, const Vector<float> &p_weights) {
	ERR_FAIL_INDEX(p_idx, vertices.size());
	ERR_FAIL_COND(p_weights.size() != 4);
	vertices.write[p_idx].weights = p_weights;
	format |= Mesh::ARRAY_FORMAT_WEIGHTS;
}

Variant MeshDataTool::get_vertex_meta(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Variant());
	return vertices[p_idx].meta;
}

void MeshDataTool::set_vertex_meta(int p_idx, const Variant &p_meta) {
	ERR_FAIL_INDEX(p_idx, vertices.size());
	vertices.write[p_idx].meta = p_meta;
}

Color MeshDataTool::get_vertex_custom_float(int p_idx, int p_custom_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Color());
	ERR_FAIL_INDEX_V(p_custom_idx, Mesh::ARRAY_CUSTOM_COUNT, Color());
	uint32_t from_type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
	uint32_t to_type = Mesh::ARRAY_CUSTOM_RGBA_FLOAT;
	return _convert_custom_data(vertices[p_idx].custom[p_custom_idx], from_type, to_type);
}

PackedByteArray MeshDataTool::get_vertex_custom_hfloat(int p_idx, int p_custom_idx) const {
	Vector<uint8_t> ret;
	ret.resize(8);
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), (memset(ret.ptrw(), 0, 8), ret));
	ERR_FAIL_INDEX_V(p_custom_idx, Mesh::ARRAY_CUSTOM_COUNT, (memset(ret.ptrw(), 0, 8), ret));
	uint32_t from_type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
	uint32_t to_type = Mesh::ARRAY_CUSTOM_RGBA_HALF;
	Color c = _convert_custom_data(vertices[p_idx].custom[p_custom_idx], from_type, to_type);
	memcpy(ret.ptrw(), &c, 8);
	return ret;
}

PackedByteArray MeshDataTool::get_vertex_custom_unorm8(int p_idx, int p_custom_idx) const {
	Vector<uint8_t> ret;
	ret.resize(4);
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), (memset(ret.ptrw(), 0, 4), ret));
	ERR_FAIL_INDEX_V(p_custom_idx, Mesh::ARRAY_CUSTOM_COUNT, (memset(ret.ptrw(), 0, 4), ret));
	uint32_t from_type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
	uint32_t to_type = Mesh::ARRAY_CUSTOM_RGBA8_UNORM;
	Color c = _convert_custom_data(vertices[p_idx].custom[p_custom_idx], from_type, to_type);
	memcpy(ret.ptrw(), &c, 4);
	return ret;
}

PackedInt32Array MeshDataTool::get_vertex_custom_snorm8_32i(int p_idx, int p_custom_idx) const {
	Vector<int32_t> ret;
	ret.resize(4);
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), (memset(ret.ptrw(), 0, 4 * sizeof(int32_t)), ret));
	ERR_FAIL_INDEX_V(p_custom_idx, Mesh::ARRAY_CUSTOM_COUNT, (memset(ret.ptrw(), 0, 4 * sizeof(int32_t)), ret));
	uint32_t from_type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
	uint32_t to_type = Mesh::ARRAY_CUSTOM_RGBA8_SNORM;
	Color c = _convert_custom_data(vertices[p_idx].custom[p_custom_idx], from_type, to_type);
	for (int i = 0; i < 4; ++i) {
		ret.write[i] = ((uint8_t *)&c)[i];
	}
	return ret;
}

Error MeshDataTool::set_vertex_custom_float(int p_idx, int p_custom_idx, const Color &p_customf) {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_custom_idx, Mesh::ARRAY_CUSTOM_COUNT, ERR_INVALID_PARAMETER);
	Color in(p_customf);

	if (format & (Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx)) {
		uint32_t type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
		vertices.write[p_idx].custom[p_custom_idx] = _convert_custom_data(in, Mesh::ARRAY_CUSTOM_RGBA_FLOAT, type);
	} else {
		// default behaviour setting the format to rgba_float
		format |= Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx;
		format &= ~(Mesh::ARRAY_FORMAT_CUSTOM_BITS << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS));
		format |= Mesh::ARRAY_CUSTOM_RGBA_FLOAT << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS);
		vertices.write[p_idx].custom[p_custom_idx] = in;
	}

	return OK;
}

Error MeshDataTool::set_vertex_custom_hfloat(int p_idx, int p_custom_idx, const PackedByteArray &p_customu) {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_custom_idx, Mesh::ARRAY_CUSTOM_COUNT, ERR_INVALID_PARAMETER);
	uint32_t count = MIN(p_customu.size(), 8);
	Color in(Color(0, 0, 0, 0));
	memcpy(&in, p_customu.ptr(), count);

	if (format & (Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx)) {
		uint32_t type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
		vertices.write[p_idx].custom[p_custom_idx] = _convert_custom_data(in, Mesh::ARRAY_CUSTOM_RGBA_HALF, type);
	} else {
		// default behaviour setting the format to rgba_hfloat
		format |= Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx;
		format &= ~(Mesh::ARRAY_FORMAT_CUSTOM_BITS << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS));
		format |= Mesh::ARRAY_CUSTOM_RGBA_HALF << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS);
		vertices.write[p_idx].custom[p_custom_idx] = in;
	}

	return OK;
}

Error MeshDataTool::set_vertex_custom_unorm8(int p_idx, int p_custom_idx, const PackedByteArray &p_customu) {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_custom_idx, Mesh::ARRAY_CUSTOM_COUNT, ERR_INVALID_PARAMETER);
	uint32_t count = MIN(p_customu.size(), 4);
	Color in(Color(0, 0, 0, 0));
	memcpy(&in, p_customu.ptr(), count);

	if (format & (Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx)) {
		uint32_t type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
		vertices.write[p_idx].custom[p_custom_idx] = _convert_custom_data(in, Mesh::ARRAY_CUSTOM_RGBA8_UNORM, type);
	} else {
		// default behaviour setting the format to rgba_hfloat
		format |= Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx;
		format &= ~(Mesh::ARRAY_FORMAT_CUSTOM_BITS << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS));
		format |= Mesh::ARRAY_CUSTOM_RGBA8_UNORM << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS);
		vertices.write[p_idx].custom[p_custom_idx] = in;
	}

	return OK;
}

Error MeshDataTool::set_vertex_custom_snorm8_32i(int p_idx, int p_custom_idx, const PackedInt32Array &p_customu) {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_custom_idx, Mesh::ARRAY_CUSTOM_COUNT, ERR_INVALID_PARAMETER);
	uint32_t count = MIN(p_customu.size(), 4);
	Color in(Color(0, 0, 0, 0));
	for (uint32_t i = 0; i < count; ++i) {
		((uint8_t *)&in)[i] = CLAMP(p_customu[i], -127, 127);
	}

	if (format & (Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx)) {
		uint32_t type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
		vertices.write[p_idx].custom[p_custom_idx] = _convert_custom_data(in, Mesh::ARRAY_CUSTOM_RGBA8_SNORM, type);
	} else {
		// default behaviour setting the format to rgba_hfloat
		format |= Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx;
		format &= ~(Mesh::ARRAY_FORMAT_CUSTOM_BITS << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS));
		format |= Mesh::ARRAY_CUSTOM_RGBA8_SNORM << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS);
		vertices.write[p_idx].custom[p_custom_idx] = in;
	}

	return OK;
}

int MeshDataTool::has_custom(int p_custom_idx) const {
	ERR_FAIL_INDEX_V(p_custom_idx, Mesh::ARRAY_CUSTOM_COUNT, 0);
	if (format & (Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx)) {
		return 1;
	} else {
		return 0;
	}
}

int MeshDataTool::get_custom_type(int p_custom_idx) const {
	ERR_FAIL_INDEX_V(p_custom_idx, Mesh::ARRAY_CUSTOM_COUNT, -1);
	ERR_FAIL_COND_V((format & (Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx)) == 0, -1);
	uint32_t type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
	return (Mesh::ArrayCustomFormat)type;
}

Error MeshDataTool::set_custom_type(int p_custom_idx, Mesh::ArrayCustomFormat p_type, bool p_convert, bool has_custom) {
	ERR_FAIL_INDEX_V(p_custom_idx, Mesh::ARRAY_CUSTOM_COUNT, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_type >= Mesh::ARRAY_CUSTOM_MAX, ERR_INVALID_PARAMETER);
	if (has_custom) {
		if (format & (Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx)) {
			uint32_t prev_type = (format >> (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS)) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
			if (prev_type != p_type) {
				format &= ~(Mesh::ARRAY_FORMAT_CUSTOM_MASK << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS));
				if (p_convert) {
					_convert_custom_data_n(vertices.ptrw(), p_custom_idx, vertices.size(), prev_type, p_type);
				}
			}
		}

		format |= Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx;
		format &= ~(Mesh::ARRAY_FORMAT_CUSTOM_BITS << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS));
		format |= p_type << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS);
	} else {
		// remove custom type.
		format &= ~(Mesh::ARRAY_FORMAT_CUSTOM0 << p_custom_idx);
		format &= ~(Mesh::ARRAY_FORMAT_CUSTOM_BITS << (Mesh::ARRAY_FORMAT_CUSTOM_BASE + p_custom_idx * Mesh::ARRAY_FORMAT_CUSTOM_BITS));
	}

	return OK;
}

Vector<int> MeshDataTool::get_vertex_edges(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Vector<int>());
	return vertices[p_idx].edges;
}

Vector<int> MeshDataTool::get_vertex_faces(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, vertices.size(), Vector<int>());
	return vertices[p_idx].faces;
}

int MeshDataTool::get_edge_vertex(int p_edge, int p_vertex) const {
	ERR_FAIL_INDEX_V(p_edge, edges.size(), -1);
	ERR_FAIL_INDEX_V(p_vertex, 2, -1);
	return edges[p_edge].vertex[p_vertex];
}

Vector<int> MeshDataTool::get_edge_faces(int p_edge) const {
	ERR_FAIL_INDEX_V(p_edge, edges.size(), Vector<int>());
	return edges[p_edge].faces;
}

Variant MeshDataTool::get_edge_meta(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, edges.size(), Variant());
	return edges[p_idx].meta;
}

void MeshDataTool::set_edge_meta(int p_idx, const Variant &p_meta) {
	ERR_FAIL_INDEX(p_idx, edges.size());
	edges.write[p_idx].meta = p_meta;
}

int MeshDataTool::get_face_vertex(int p_face, int p_vertex) const {
	ERR_FAIL_INDEX_V(p_face, faces.size(), -1);
	ERR_FAIL_INDEX_V(p_vertex, 3, -1);
	return faces[p_face].v[p_vertex];
}

int MeshDataTool::get_face_edge(int p_face, int p_vertex) const {
	ERR_FAIL_INDEX_V(p_face, faces.size(), -1);
	ERR_FAIL_INDEX_V(p_vertex, 3, -1);
	return faces[p_face].edges[p_vertex];
}

Variant MeshDataTool::get_face_meta(int p_face) const {
	ERR_FAIL_INDEX_V(p_face, faces.size(), Variant());
	return faces[p_face].meta;
}

void MeshDataTool::set_face_meta(int p_face, const Variant &p_meta) {
	ERR_FAIL_INDEX(p_face, faces.size());
	faces.write[p_face].meta = p_meta;
}

Vector3 MeshDataTool::get_face_normal(int p_face) const {
	ERR_FAIL_INDEX_V(p_face, faces.size(), Vector3());
	Vector3 v0 = vertices[faces[p_face].v[0]].vertex;
	Vector3 v1 = vertices[faces[p_face].v[1]].vertex;
	Vector3 v2 = vertices[faces[p_face].v[2]].vertex;

	return Plane(v0, v1, v2).normal;
}

Ref<Material> MeshDataTool::get_material() const {
	return material;
}

void MeshDataTool::set_material(const Ref<Material> &p_material) {
	material = p_material;
}

void MeshDataTool::_bind_methods() {
	ClassDB::bind_method(D_METHOD("clear"), &MeshDataTool::clear);
	ClassDB::bind_method(D_METHOD("create_from_surface", "mesh", "surface"), &MeshDataTool::create_from_surface);
	ClassDB::bind_method(D_METHOD("commit_to_surface", "mesh"), &MeshDataTool::commit_to_surface);

	ClassDB::bind_method(D_METHOD("get_format"), &MeshDataTool::get_format);

	ClassDB::bind_method(D_METHOD("get_vertex_count"), &MeshDataTool::get_vertex_count);
	ClassDB::bind_method(D_METHOD("get_edge_count"), &MeshDataTool::get_edge_count);
	ClassDB::bind_method(D_METHOD("get_face_count"), &MeshDataTool::get_face_count);

	ClassDB::bind_method(D_METHOD("set_vertex", "idx", "vertex"), &MeshDataTool::set_vertex);
	ClassDB::bind_method(D_METHOD("get_vertex", "idx"), &MeshDataTool::get_vertex);

	ClassDB::bind_method(D_METHOD("set_vertex_normal", "idx", "normal"), &MeshDataTool::set_vertex_normal);
	ClassDB::bind_method(D_METHOD("get_vertex_normal", "idx"), &MeshDataTool::get_vertex_normal);

	ClassDB::bind_method(D_METHOD("set_vertex_tangent", "idx", "tangent"), &MeshDataTool::set_vertex_tangent);
	ClassDB::bind_method(D_METHOD("get_vertex_tangent", "idx"), &MeshDataTool::get_vertex_tangent);

	ClassDB::bind_method(D_METHOD("set_vertex_uv", "idx", "uv"), &MeshDataTool::set_vertex_uv);
	ClassDB::bind_method(D_METHOD("get_vertex_uv", "idx"), &MeshDataTool::get_vertex_uv);

	ClassDB::bind_method(D_METHOD("set_vertex_uv2", "idx", "uv2"), &MeshDataTool::set_vertex_uv2);
	ClassDB::bind_method(D_METHOD("get_vertex_uv2", "idx"), &MeshDataTool::get_vertex_uv2);

	ClassDB::bind_method(D_METHOD("set_vertex_color", "idx", "color"), &MeshDataTool::set_vertex_color);
	ClassDB::bind_method(D_METHOD("get_vertex_color", "idx"), &MeshDataTool::get_vertex_color);

	ClassDB::bind_method(D_METHOD("set_vertex_bones", "idx", "bones"), &MeshDataTool::set_vertex_bones);
	ClassDB::bind_method(D_METHOD("get_vertex_bones", "idx"), &MeshDataTool::get_vertex_bones);

	ClassDB::bind_method(D_METHOD("set_vertex_weights", "idx", "weights"), &MeshDataTool::set_vertex_weights);
	ClassDB::bind_method(D_METHOD("get_vertex_weights", "idx"), &MeshDataTool::get_vertex_weights);

	ClassDB::bind_method(D_METHOD("set_vertex_meta", "idx", "meta"), &MeshDataTool::set_vertex_meta);
	ClassDB::bind_method(D_METHOD("get_vertex_meta", "idx"), &MeshDataTool::get_vertex_meta);

	ClassDB::bind_method(D_METHOD("set_vertex_custom_float", "idx", "custom_idx", "customf"), &MeshDataTool::set_vertex_custom_float);
	ClassDB::bind_method(D_METHOD("set_vertex_custom_hfloat", "idx", "custom_idx", "customu"), &MeshDataTool::set_vertex_custom_hfloat);
	ClassDB::bind_method(D_METHOD("set_vertex_custom_unorm8", "idx", "custom_idx", "customu"), &MeshDataTool::set_vertex_custom_unorm8);
	ClassDB::bind_method(D_METHOD("set_vertex_custom_snorm8_32i", "idx", "custom_idx", "customf"), &MeshDataTool::set_vertex_custom_snorm8_32i);

	ClassDB::bind_method(D_METHOD("get_vertex_custom_float", "idx", "custom_idx"), &MeshDataTool::get_vertex_custom_float);
	ClassDB::bind_method(D_METHOD("get_vertex_custom_hfloat", "idx", "custom_idx"), &MeshDataTool::get_vertex_custom_hfloat);
	ClassDB::bind_method(D_METHOD("get_vertex_custom_unorm8", "idx", "custom_idx"), &MeshDataTool::get_vertex_custom_unorm8);
	ClassDB::bind_method(D_METHOD("get_vertex_custom_snorm8_32i", "idx", "custom_idx"), &MeshDataTool::get_vertex_custom_snorm8_32i);

	ClassDB::bind_method(D_METHOD("has_custom", "custom_idx"), &MeshDataTool::has_custom);
	ClassDB::bind_method(D_METHOD("get_custom_type", "custom_idx"), &MeshDataTool::get_custom_type);
	ClassDB::bind_method(D_METHOD("set_custom_type", "custom_idx", "type", "convert", "has_custom"), &MeshDataTool::set_custom_type, DEFVAL(false), DEFVAL(true));

	ClassDB::bind_method(D_METHOD("get_vertex_edges", "idx"), &MeshDataTool::get_vertex_edges);
	ClassDB::bind_method(D_METHOD("get_vertex_faces", "idx"), &MeshDataTool::get_vertex_faces);

	ClassDB::bind_method(D_METHOD("get_edge_vertex", "idx", "vertex"), &MeshDataTool::get_edge_vertex);
	ClassDB::bind_method(D_METHOD("get_edge_faces", "idx"), &MeshDataTool::get_edge_faces);

	ClassDB::bind_method(D_METHOD("set_edge_meta", "idx", "meta"), &MeshDataTool::set_edge_meta);
	ClassDB::bind_method(D_METHOD("get_edge_meta", "idx"), &MeshDataTool::get_edge_meta);

	ClassDB::bind_method(D_METHOD("get_face_vertex", "idx", "vertex"), &MeshDataTool::get_face_vertex);
	ClassDB::bind_method(D_METHOD("get_face_edge", "idx", "edge"), &MeshDataTool::get_face_edge);

	ClassDB::bind_method(D_METHOD("set_face_meta", "idx", "meta"), &MeshDataTool::set_face_meta);
	ClassDB::bind_method(D_METHOD("get_face_meta", "idx"), &MeshDataTool::get_face_meta);

	ClassDB::bind_method(D_METHOD("get_face_normal", "idx"), &MeshDataTool::get_face_normal);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &MeshDataTool::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &MeshDataTool::get_material);
}

Error MeshDataTool::_convert_custom_data_n(MeshDataTool::Vertex *p_data, uint32_t p_custom_idx, uint32_t n, uint32_t p_from_type, uint32_t p_to_type) {
	// signed normalized conversions are based on : https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html
	// [-127, 127] will be mapped to [-1.0, 1.0]
	// unsigned normalized conversions are based on : https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html
	// [0, 255] will be mapped to [0.0, 1.0]

	if (p_from_type == p_to_type) {
		return OK;
	}
	switch (p_from_type) {
		case Mesh::ARRAY_CUSTOM_RGBA8_UNORM: {
			// 4x [0,255] 8-bit numbers.
			switch (p_to_type) {
				case Mesh::ARRAY_CUSTOM_RGBA8_SNORM: {
					// 4x [-127,127] 8-bit numbers.
					for (size_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						const Color srcc = p_data[i].custom[p_custom_idx];
						for (uint32_t j = 0; j < 4; ++j) {
							((uint8_t *)&dstc)[j] = (float)((uint8_t *)&srcc)[j] * (127 / 255);
						}
					}
					return OK;
				} break;
				case Mesh::ARRAY_CUSTOM_RG_HALF:
					// 2x 16-bit floating point numbers
				case Mesh::ARRAY_CUSTOM_RGBA_HALF: {
					// 4x 16-bit floating point numbers.
					const uint32_t ct = (p_to_type - Mesh::ARRAY_CUSTOM_RG_HALF + 1) * 2;
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						const Color srcc = p_data[i].custom[p_custom_idx];
						uint16_t *wptr = (uint16_t *)&dstc;
						for (uint32_t j = 0; j < ct; ++j) {
							float value = (float)((uint8_t *)&srcc)[j] / 255;
							// converting to half floating point
							wptr[j] = Math::make_half_float(value);
						}
					}
				} break;
				case Mesh::ARRAY_CUSTOM_R_FLOAT:
				case Mesh::ARRAY_CUSTOM_RG_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGB_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGBA_FLOAT: {
					// 4x 32-bit floating point numbers.
					const uint32_t ct = p_to_type - Mesh::ARRAY_CUSTOM_R_FLOAT + 1;
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						const Color srcc = p_data[i].custom[p_custom_idx];
						for (uint32_t j = 0; j < ct; ++j) {
							dstc.components[j] = (float)((uint8_t *)&srcc)[j] / 255;
						}
					}
				} break;
				default: {
					ERR_FAIL_V(ERR_INVALID_PARAMETER);
				} break;
			}
		} break;

		case Mesh::ARRAY_CUSTOM_RGBA8_SNORM: {
			// 4x [-127,127] 8-bit numbers.
			switch (p_to_type) {
				case Mesh::ARRAY_CUSTOM_RGBA8_UNORM: {
					// 4x [0, 255] 8-bit numbers.
					for (size_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						const Color srcc = p_data[i].custom[p_custom_idx];
						for (uint32_t j = 0; j < 4; ++j) {
							int8_t value = (int8_t)((uint8_t *)&srcc)[j];
							((uint8_t *)&dstc)[j] = (float)(CLAMP(value, 0, 127)) * (255 / 127);
						}
					}
					return OK;
				} break;
				case Mesh::ARRAY_CUSTOM_RG_HALF:
					// 2x 16-bit floating point numbers.
				case Mesh::ARRAY_CUSTOM_RGBA_HALF: {
					// 4x 16-bit floating point numbers.
					const uint32_t ct = (p_to_type - Mesh::ARRAY_CUSTOM_RG_HALF + 1) * 2;
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						const Color srcc = p_data[i].custom[p_custom_idx];
						uint16_t *wptr = (uint16_t *)&dstc;
						for (uint32_t j = 0; j < ct; ++j) {
							float value = (float)((uint8_t *)&srcc)[j] / 127;
							value = value >= -1.0f ? value : -1.0f;
							// converting to half floating point
							wptr[j] = Math::make_half_float(value);
						}
					}
				} break;
				case Mesh::ARRAY_CUSTOM_R_FLOAT:
				case Mesh::ARRAY_CUSTOM_RG_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGB_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGBA_FLOAT: {
					// 4x 32-bit floating point numbers.
					const uint32_t ct = p_to_type - Mesh::ARRAY_CUSTOM_R_FLOAT + 1;
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						const Color srcc = p_data[i].custom[p_custom_idx];
						for (uint32_t j = 0; j < ct; ++j) {
							float value = (float)((uint8_t *)&srcc)[j] / 127;
							dstc.components[j] = value >= -1.0f ? value : -1.0f;
						}
					}
				} break;
				default: {
					ERR_FAIL_V(ERR_INVALID_PARAMETER);
				} break;
			}
		} break;

		case Mesh::ARRAY_CUSTOM_RG_HALF:
			// 2x 16-bit floating point numbers.
		case Mesh::ARRAY_CUSTOM_RGBA_HALF: {
			// 4x 16-bit floating point numbers.
			const uint32_t cf = (p_from_type - Mesh::ARRAY_CUSTOM_RG_HALF + 1) * 2;
			switch (p_to_type) {
				case Mesh::ARRAY_CUSTOM_RGBA8_UNORM: {
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						uint16_t rptr[8];
						Color *srccptr = (Color *)rptr; // Pointer used to silence wrong -Wmaybe-initialized.
						*srccptr = p_data[i].custom[p_custom_idx];
						// 4x [0, 255] 8-bit numbers.
						for (uint32_t j = 0; j < cf; ++j) {
							float value = Math::half_to_float(rptr[j]);
							// converting to 32-bit floating point.
							value = CLAMP(value, 0.0f, 1.0f);
							((uint8_t *)&dstc)[j] = value * 255;
						}
					}
				} break;
				case Mesh::ARRAY_CUSTOM_RGBA8_SNORM: {
					// 4x [-127,127] 8-bit numbers.
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						uint16_t rptr[8];
						Color *srccptr = (Color *)rptr; // Pointer used to silence wrong -Wmaybe-initialized.
						*srccptr = p_data[i].custom[p_custom_idx];
						for (uint32_t j = 0; j < cf; ++j) {
							float value = Math::half_to_float(rptr[j]);
							// converting to 32-bit floating point.
							value = CLAMP(value, -1.0f, 1.0f);
							((uint8_t *)&dstc)[j] = value * 127;
						}
					}
				} break;
				case Mesh::ARRAY_CUSTOM_RG_HALF: {
					// 2x 16-bit floating point numbers.
					// do nothing
					return OK;
				} break;
				case Mesh::ARRAY_CUSTOM_RGBA_HALF: {
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						uint16_t *wptr = (uint16_t *)&dstc;
						for (uint32_t j = 2; j < 4; ++j) {
							wptr[j] = 0;
						}
					}
					return OK;
				} break;
				case Mesh::ARRAY_CUSTOM_R_FLOAT:
				case Mesh::ARRAY_CUSTOM_RG_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGB_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGBA_FLOAT: {
					// 4x 32-bit floating point numbers.
					const uint32_t ct = p_to_type - Mesh::ARRAY_CUSTOM_R_FLOAT + 1;
					const uint32_t c = MIN(cf, ct);
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						uint16_t rptr[8];
						Color *srccptr = (Color *)rptr; // Pointer used to silence wrong -Wmaybe-initialized.
						*srccptr = p_data[i].custom[p_custom_idx];
						for (uint32_t j = 0; j < c; ++j) {
							// converting to 32-bit floating point.
							dstc.components[j] = Math::half_to_float(rptr[j]);
						}
						for (uint32_t j = c; j < ct; ++j) {
							dstc.components[j] = 0.0f;
						}
					}
				} break;
				default: {
					ERR_FAIL_V(ERR_INVALID_PARAMETER);
				} break;
			}
		} break;

		case Mesh::ARRAY_CUSTOM_R_FLOAT:
			// 1x 32-bit floating point numbers.
		case Mesh::ARRAY_CUSTOM_RG_FLOAT:
			// 2x 32-bit floating point numbers.
		case Mesh::ARRAY_CUSTOM_RGB_FLOAT:
			// 3x 32-bit floating point numbers.
		case Mesh::ARRAY_CUSTOM_RGBA_FLOAT: {
			// 4x 32-bit floating point numbers.
			const uint32_t cf = p_from_type - Mesh::ARRAY_CUSTOM_R_FLOAT + 1;
			switch (p_to_type) {
				case Mesh::ARRAY_CUSTOM_RGBA8_UNORM: {
					// 4x [0, 255] 8-bit numbers.
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						const Color srcc = p_data[i].custom[p_custom_idx];
						for (uint32_t j = 0; j < cf; ++j) {
							float value = CLAMP(srcc.components[j], 0.0f, 1.0f);
							((uint8_t *)&dstc)[j] = value * 255;
						}
						for (uint32_t j = cf; j < 4; ++j) {
							((uint8_t *)&dstc)[j] = 0;
						}
					}
				} break;
				case Mesh::ARRAY_CUSTOM_RGBA8_SNORM: {
					// 4x [-127,127] 8-bit numbers.
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						const Color srcc = p_data[i].custom[p_custom_idx];
						for (uint32_t j = 0; j < cf; ++j) {
							float value = CLAMP(srcc.components[j], -1.0f, 1.0f);
							((uint8_t *)&dstc)[j] = value * 127;
						}
						for (uint32_t j = cf; j < 4; ++j) {
							((uint8_t *)&dstc)[j] = 0;
						}
					}
				} break;
				case Mesh::ARRAY_CUSTOM_RG_HALF:
					// 2x 16-bit floating point numbers.
				case Mesh::ARRAY_CUSTOM_RGBA_HALF: {
					// 4x 16-bit floating point numbers.
					const uint32_t ct = (p_to_type - Mesh::ARRAY_CUSTOM_RGBA_HALF + 1) * 2;
					const uint32_t c = MIN(cf, ct);
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						const Color srcc = p_data[i].custom[p_custom_idx];
						uint16_t *wptr = (uint16_t *)&dstc;
						for (uint32_t j = 0; j < c; ++j) {
							// converting to 16-bit floating point.
							wptr[j] = Math::make_half_float(srcc.components[j]);
						}
						for (uint32_t j = c; j < ct; ++j) {
							dstc.components[j] = 0.0f;
						}
					}
				} break;
				case Mesh::ARRAY_CUSTOM_R_FLOAT:
				case Mesh::ARRAY_CUSTOM_RG_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGB_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGBA_FLOAT: {
					const uint32_t ct = (p_to_type - Mesh::ARRAY_CUSTOM_R_FLOAT + 1);
					for (uint32_t i = 0; i < n; ++i) {
						Color &dstc = p_data[i].custom[p_custom_idx];
						for (uint32_t j = cf; j < ct; ++j) {
							dstc.components[j] = 0.0f;
						}
					}
					return OK;
				}
				default: {
					ERR_FAIL_V(ERR_INVALID_PARAMETER);
				} break;
			}
		} break;
		default: {
			ERR_FAIL_V(ERR_INVALID_PARAMETER);
		} break;
	}

	return OK;
}

Color MeshDataTool::_convert_custom_data(const Color &srcc, uint32_t p_from_type, uint32_t p_to_type) {
	// signed normalized conversions are based on : https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html
	// [-127, 127] will be mapped to [-1.0, 1.0]
	// unsigned normalized conversions are based on : https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html
	// [0, 255] will be mapped to [0.0, 1.0]

	if (p_from_type == p_to_type) {
		return srcc;
	}

	Color dstc(Color(0, 0, 0, 0));

	switch (p_from_type) {
		case Mesh::ARRAY_CUSTOM_RGBA8_UNORM: {
			// 4x [0,255] 8-bit numbers.
			switch (p_to_type) {
				case Mesh::ARRAY_CUSTOM_RGBA8_SNORM: {
					// 4x [-127,127] 8-bit numbers.
					for (uint32_t j = 0; j < 4; ++j) {
						((uint8_t *)&dstc)[j] = CLAMP(((uint8_t *)&srcc)[j], 0, 127);
					}
				} break;
				case Mesh::ARRAY_CUSTOM_RG_HALF:
					// 2x 16-bit floating point numbers
				case Mesh::ARRAY_CUSTOM_RGBA_HALF: {
					// 4x 16-bit floating point numbers.
					const uint32_t ct = (p_to_type - Mesh::ARRAY_CUSTOM_RG_HALF + 1) * 2;
					uint16_t *wptr = (uint16_t *)&dstc;
					for (uint32_t j = 0; j < ct; ++j) {
						float value = (float)((uint8_t *)&srcc)[j] / 255;
						// converting to half floating point
						wptr[j] = Math::make_half_float(value);
					}
				} break;
				case Mesh::ARRAY_CUSTOM_R_FLOAT:
				case Mesh::ARRAY_CUSTOM_RG_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGB_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGBA_FLOAT: {
					// 4x 32-bit floating point numbers.
					const uint32_t ct = p_to_type - Mesh::ARRAY_CUSTOM_R_FLOAT + 1;
					for (uint32_t j = 0; j < ct; ++j) {
						dstc.components[j] = (float)((uint8_t *)&srcc)[j] / 255;
					}
				} break;
				default: {
					ERR_FAIL_V(Color(Color(0, 0, 0, 0)));
				} break;
			}
		} break;

		case Mesh::ARRAY_CUSTOM_RGBA8_SNORM: {
			// 4x [-127,127] 8-bit numbers.
			switch (p_to_type) {
				case Mesh::ARRAY_CUSTOM_RGBA8_UNORM: {
					// 4x [0, 255] 8-bit numbers.
					for (uint32_t j = 0; j < 4; ++j) {
						int8_t value = (int8_t)((uint8_t *)&srcc)[j];
						((uint8_t *)&dstc)[j] = (float)(CLAMP(value, 0, 127)) * (255 / 127);
					}
				} break;
				case Mesh::ARRAY_CUSTOM_RG_HALF:
					// 2x 16-bit floating point numbers.
				case Mesh::ARRAY_CUSTOM_RGBA_HALF: {
					// 4x 16-bit floating point numbers.
					const uint32_t ct = (p_to_type - Mesh::ARRAY_CUSTOM_RG_HALF + 1) * 2;
					uint16_t *wptr = (uint16_t *)&dstc;
					for (uint32_t j = 0; j < ct; ++j) {
						float value = (float)((uint8_t *)&srcc)[j] / 127;
						value = value >= -1.0f ? value : -1.0f;
						// converting to half floating point
						wptr[j] = Math::make_half_float(value);
					}
				} break;
				case Mesh::ARRAY_CUSTOM_R_FLOAT:
				case Mesh::ARRAY_CUSTOM_RG_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGB_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGBA_FLOAT: {
					// 4x 32-bit floating point numbers.
					const uint32_t ct = p_to_type - Mesh::ARRAY_CUSTOM_R_FLOAT + 1;
					for (uint32_t j = 0; j < ct; ++j) {
						float value = (float)((uint8_t *)&srcc)[j] / 127;
						dstc.components[j] = value >= -1.0f ? value : -1.0f;
					}
				} break;
				default: {
					ERR_FAIL_V(Color(Color(0, 0, 0, 0)));
				} break;
			}
		} break;

		case Mesh::ARRAY_CUSTOM_RG_HALF:
			// 2x 16-bit floating point numbers.
		case Mesh::ARRAY_CUSTOM_RGBA_HALF: {
			// 4x 16-bit floating point numbers.
			const uint32_t cf = (p_from_type - Mesh::ARRAY_CUSTOM_RG_HALF + 1) * 2;
			switch (p_to_type) {
				case Mesh::ARRAY_CUSTOM_RGBA8_UNORM: {
					const uint16_t *rptr = (const uint16_t *)&srcc;
					// 4x [0, 255] 8-bit numbers.
					for (uint32_t j = 0; j < cf; ++j) {
						float value = Math::half_to_float(rptr[j]);
						// converting to 32-bit floating point.
						value = CLAMP(value, 0.0f, 1.0f);
						((uint8_t *)&dstc)[j] = value * 255;
					}
				} break;
				case Mesh::ARRAY_CUSTOM_RGBA8_SNORM: {
					// 4x [-127,127] 8-bit numbers.
					const uint16_t *rptr = (const uint16_t *)&srcc;
					for (uint32_t j = 0; j < cf; ++j) {
						float value = Math::half_to_float(rptr[j]);
						// converting to 32-bit floating point.
						value = CLAMP(value, -1.0f, 1.0f);
						((uint8_t *)&dstc)[j] = value * 127;
					}
				} break;
				case Mesh::ARRAY_CUSTOM_RG_HALF:
					// 2x 16-bit floating point numbers.
				case Mesh::ARRAY_CUSTOM_RGBA_HALF: {
					// do nothing
					return srcc;
				} break;
				case Mesh::ARRAY_CUSTOM_R_FLOAT:
				case Mesh::ARRAY_CUSTOM_RG_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGB_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGBA_FLOAT: {
					// 4x 32-bit floating point numbers.
					const uint32_t ct = p_to_type - Mesh::ARRAY_CUSTOM_R_FLOAT + 1;
					const uint32_t c = MIN(cf, ct);
					const uint16_t *rptr = (const uint16_t *)&srcc;
					for (uint32_t j = 0; j < c; ++j) {
						// converting to 32-bit floating point.
						dstc.components[j] = Math::half_to_float(rptr[j]);
					}
				} break;
				default: {
					ERR_FAIL_V(Color(Color(0, 0, 0, 0)));
				} break;
			}
		} break;

		case Mesh::ARRAY_CUSTOM_R_FLOAT:
			// 1x 32-bit floating point numbers.
		case Mesh::ARRAY_CUSTOM_RG_FLOAT:
			// 2x 32-bit floating point numbers.
		case Mesh::ARRAY_CUSTOM_RGB_FLOAT:
			// 3x 32-bit floating point numbers.
		case Mesh::ARRAY_CUSTOM_RGBA_FLOAT: {
			// 4x 32-bit floating point numbers.
			const uint32_t cf = p_from_type - Mesh::ARRAY_CUSTOM_R_FLOAT + 1;
			switch (p_to_type) {
				case Mesh::ARRAY_CUSTOM_RGBA8_UNORM: {
					// 4x [0, 255] 8-bit numbers.
					for (uint32_t j = 0; j < cf; ++j) {
						float value = CLAMP(srcc.components[j], 0.0f, 1.0f);
						((uint8_t *)&dstc)[j] = value * 255;
					}
				} break;
				case Mesh::ARRAY_CUSTOM_RGBA8_SNORM: {
					// 4x [-127,127] 8-bit numbers.
					for (uint32_t j = 0; j < cf; ++j) {
						float value = CLAMP(srcc.components[j], -1.0f, 1.0f);
						((uint8_t *)&dstc)[j] = value * 127;
					}
				} break;
				case Mesh::ARRAY_CUSTOM_RG_HALF:
					// 2x 16-bit floating point numbers.
				case Mesh::ARRAY_CUSTOM_RGBA_HALF: {
					// 4x 16-bit floating point numbers.
					const uint32_t ct = (p_to_type - Mesh::ARRAY_CUSTOM_RGBA_HALF + 1) * 2;
					const uint32_t c = MIN(cf, ct);
					uint16_t *wptr = (uint16_t *)&dstc;
					for (uint32_t j = 0; j < c; ++j) {
						// converting to 16-bit floating point.
						wptr[j] = Math::make_half_float(srcc.components[j]);
					}
				} break;
				case Mesh::ARRAY_CUSTOM_R_FLOAT:
				case Mesh::ARRAY_CUSTOM_RG_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGB_FLOAT:
				case Mesh::ARRAY_CUSTOM_RGBA_FLOAT: {
					return srcc;
				}
				default: {
					ERR_FAIL_V(Color(Color(0, 0, 0, 0)));
				} break;
			}
		} break;
		default: {
			ERR_FAIL_V(Color(Color(0, 0, 0, 0)));
		} break;
	}

	return dstc;
}

MeshDataTool::MeshDataTool() {
	clear();
}
