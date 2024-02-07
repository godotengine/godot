/**************************************************************************/
/*  surface_tool.cpp                                                      */
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

#include "surface_tool.h"

#include "core/method_bind_ext.gen.inc"

#define _VERTEX_SNAP 0.0001
#define EQ_VERTEX_DIST 0.00001

bool SurfaceTool::Vertex::operator==(const Vertex &p_vertex) const {
	if (vertex != p_vertex.vertex) {
		return false;
	}

	if (uv != p_vertex.uv) {
		return false;
	}

	if (uv2 != p_vertex.uv2) {
		return false;
	}

	if (normal != p_vertex.normal) {
		return false;
	}

	if (binormal != p_vertex.binormal) {
		return false;
	}

	if (color != p_vertex.color) {
		return false;
	}

	if (num_bones != p_vertex.num_bones) {
		return false;
	}

	for (int i = 0; i < num_bones; i++) {
		if (bones[i] != p_vertex.bones[i]) {
			return false;
		}
	}

	for (int i = 0; i < num_bones; i++) {
		if (weights[i] != p_vertex.weights[i]) {
			return false;
		}
	}

	return true;
}

uint32_t SurfaceTool::VertexHasher::hash(const Vertex &p_vtx) {
	uint32_t h = hash_djb2_buffer((const uint8_t *)&p_vtx.vertex, sizeof(real_t) * 3);
	h = hash_djb2_buffer((const uint8_t *)&p_vtx.normal, sizeof(real_t) * 3, h);
	h = hash_djb2_buffer((const uint8_t *)&p_vtx.binormal, sizeof(real_t) * 3, h);
	h = hash_djb2_buffer((const uint8_t *)&p_vtx.tangent, sizeof(real_t) * 3, h);
	h = hash_djb2_buffer((const uint8_t *)&p_vtx.uv, sizeof(real_t) * 2, h);
	h = hash_djb2_buffer((const uint8_t *)&p_vtx.uv2, sizeof(real_t) * 2, h);
	h = hash_djb2_buffer((const uint8_t *)&p_vtx.color, sizeof(real_t) * 4, h);
	h = hash_djb2_buffer((const uint8_t *)p_vtx.bones, p_vtx.num_bones * sizeof(int16_t), h);
	h = hash_djb2_buffer((const uint8_t *)p_vtx.weights, p_vtx.num_bones * sizeof(float), h);
	return h;
}

void SurfaceTool::begin(Mesh::PrimitiveType p_primitive) {
	clear();

	primitive = p_primitive;
	begun = true;
	first = true;
}

bool SurfaceTool::_sanitize_last_bones_and_weights() {
	const int expected_vertices = Vertex::MAX_BONES;

	if ((last_bones.size() == expected_vertices) && (last_weights.size() == expected_vertices)) {
		// already ideal
		return true;
	}
	ERR_FAIL_COND_V(last_bones.size() != last_weights.size(), false);

	int num_orig = last_weights.size();

	if (num_orig < expected_vertices) {
		//less than required, fill
		for (int i = last_weights.size(); i < expected_vertices; i++) {
			last_weights.push_back(0);
			last_bones.push_back(0);
		}
	} else if (num_orig > expected_vertices) {
		//more than required, sort, cap and normalize.
		Vector<WeightSort> weights;
		for (int i = 0; i < num_orig; i++) {
			WeightSort ws;
			ws.index = last_bones[i];
			ws.weight = last_weights[i];
			weights.push_back(ws);
		}

		//sort
		weights.sort();
		//cap
		weights.resize(expected_vertices);
		//renormalize
		float total = 0;
		for (int i = 0; i < expected_vertices; i++) {
			total += weights[i].weight;
		}

		last_weights.resize(expected_vertices);
		last_bones.resize(expected_vertices);

		for (int i = 0; i < expected_vertices; i++) {
			if (total > 0) {
				last_weights.write[i] = weights[i].weight / total;
			} else {
				last_weights.write[i] = 0;
			}
			last_bones.write[i] = weights[i].index;
		}
	}

	return true;
}

void SurfaceTool::add_vertex(const Vector3 &p_vertex) {
	ERR_FAIL_COND(!begun);

	Vertex vtx;
	vtx.vertex = p_vertex;
	vtx.color = last_color;
	vtx.normal = last_normal;
	vtx.uv = last_uv;
	vtx.uv2 = last_uv2;
	vtx.tangent = last_tangent.normal;
	vtx.binormal = last_normal.cross(last_tangent.normal).normalized() * last_tangent.d;

	if (format & Mesh::ARRAY_FORMAT_WEIGHTS || format & Mesh::ARRAY_FORMAT_BONES) {
		ERR_FAIL_COND(!_sanitize_last_bones_and_weights());

		vtx.num_bones = last_bones.size();
		for (int n = 0; n < last_bones.size(); n++) {
			vtx.bones[n] = last_bones[n];
			vtx.weights[n] = last_weights[n];
		}
	}

	vertex_array.push_back(vtx);
	first = false;

	format |= Mesh::ARRAY_FORMAT_VERTEX;
}
void SurfaceTool::add_color(Color p_color) {
	ERR_FAIL_COND(!begun);

	ERR_FAIL_COND(!first && !(format & Mesh::ARRAY_FORMAT_COLOR));

	format |= Mesh::ARRAY_FORMAT_COLOR;
	last_color = p_color;
}
void SurfaceTool::add_normal(const Vector3 &p_normal) {
	ERR_FAIL_COND(!begun);

	ERR_FAIL_COND(!first && !(format & Mesh::ARRAY_FORMAT_NORMAL));

	format |= Mesh::ARRAY_FORMAT_NORMAL;
	last_normal = p_normal;
}

void SurfaceTool::add_tangent(const Plane &p_tangent) {
	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND(!first && !(format & Mesh::ARRAY_FORMAT_TANGENT));

	format |= Mesh::ARRAY_FORMAT_TANGENT;
	last_tangent = p_tangent;
}

void SurfaceTool::add_uv(const Vector2 &p_uv) {
	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND(!first && !(format & Mesh::ARRAY_FORMAT_TEX_UV));

	format |= Mesh::ARRAY_FORMAT_TEX_UV;
	last_uv = p_uv;
}

void SurfaceTool::add_uv2(const Vector2 &p_uv2) {
	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND(!first && !(format & Mesh::ARRAY_FORMAT_TEX_UV2));

	format |= Mesh::ARRAY_FORMAT_TEX_UV2;
	last_uv2 = p_uv2;
}

void SurfaceTool::add_bones(const Vector<int> &p_bones) {
	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND(!first && !(format & Mesh::ARRAY_FORMAT_BONES));

	format |= Mesh::ARRAY_FORMAT_BONES;
	last_bones = p_bones;
}

void SurfaceTool::add_weights(const Vector<float> &p_weights) {
	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND(!first && !(format & Mesh::ARRAY_FORMAT_WEIGHTS));

	format |= Mesh::ARRAY_FORMAT_WEIGHTS;
	last_weights = p_weights;
}

void SurfaceTool::add_smooth_group(bool p_smooth) {
	ERR_FAIL_COND(!begun);
	smooth_groups[get_num_draw_vertices()] = p_smooth;
}

void SurfaceTool::add_triangle_fan(const Vector<Vector3> &p_vertices, const Vector<Vector2> &p_uvs, const Vector<Color> &p_colors, const Vector<Vector2> &p_uv2s, const Vector<Vector3> &p_normals, const Vector<Plane> &p_tangents) {
	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND(primitive != Mesh::PRIMITIVE_TRIANGLES);
	ERR_FAIL_COND(p_vertices.size() < 3);

#define ADD_POINT(n)                    \
	{                                   \
		if (p_colors.size() > n)        \
			add_color(p_colors[n]);     \
		if (p_uvs.size() > n)           \
			add_uv(p_uvs[n]);           \
		if (p_uv2s.size() > n)          \
			add_uv2(p_uv2s[n]);         \
		if (p_normals.size() > n)       \
			add_normal(p_normals[n]);   \
		if (p_tangents.size() > n)      \
			add_tangent(p_tangents[n]); \
		add_vertex(p_vertices[n]);      \
	}

	for (int i = 0; i < p_vertices.size() - 2; i++) {
		ADD_POINT(0);
		ADD_POINT(i + 1);
		ADD_POINT(i + 2);
	}

#undef ADD_POINT
}

void SurfaceTool::add_index(int p_index) {
	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND(p_index < 0);

	format |= Mesh::ARRAY_FORMAT_INDEX;
	index_array.push_back(p_index);
}

Array SurfaceTool::commit_to_arrays() {
	int varr_len = vertex_array.size();

	Array a;
	a.resize(Mesh::ARRAY_MAX);

	for (int i = 0; i < Mesh::ARRAY_MAX; i++) {
		if (!(format & (1 << i))) {
			continue; //not in format
		}

		switch (i) {
			case Mesh::ARRAY_VERTEX:
			case Mesh::ARRAY_NORMAL: {
				PoolVector<Vector3> array;
				array.resize(varr_len);
				PoolVector<Vector3>::Write w = array.write();

				for (uint32_t n = 0; n < vertex_array.size(); n++) {
					const Vertex &v = vertex_array[n];

					switch (i) {
						case Mesh::ARRAY_VERTEX: {
							w[n] = v.vertex;
						} break;
						case Mesh::ARRAY_NORMAL: {
							w[n] = v.normal;
						} break;
					}
				}

				w.release();
				a[i] = array;

			} break;

			case Mesh::ARRAY_TEX_UV:
			case Mesh::ARRAY_TEX_UV2: {
				PoolVector<Vector2> array;
				array.resize(varr_len);
				PoolVector<Vector2>::Write w = array.write();

				int idx = 0;
				for (uint32_t n = 0; n < vertex_array.size(); n++, idx++) {
					const Vertex &v = vertex_array[n];

					switch (i) {
						case Mesh::ARRAY_TEX_UV: {
							w[idx] = v.uv;
						} break;
						case Mesh::ARRAY_TEX_UV2: {
							w[idx] = v.uv2;
						} break;
					}
				}

				w.release();
				a[i] = array;
			} break;
			case Mesh::ARRAY_TANGENT: {
				PoolVector<float> array;
				array.resize(varr_len * 4);
				PoolVector<float>::Write w = array.write();

				int idx = 0;
				for (uint32_t n = 0; n < vertex_array.size(); n++, idx += 4) {
					const Vertex &v = vertex_array[n];

					w[idx + 0] = v.tangent.x;
					w[idx + 1] = v.tangent.y;
					w[idx + 2] = v.tangent.z;

					//float d = v.tangent.dot(v.binormal,v.normal);
					float d = v.binormal.dot(v.normal.cross(v.tangent));
					w[idx + 3] = d < 0 ? -1 : 1;
				}

				w.release();
				a[i] = array;

			} break;
			case Mesh::ARRAY_COLOR: {
				PoolVector<Color> array;
				array.resize(varr_len);
				PoolVector<Color>::Write w = array.write();

				int idx = 0;

				for (uint32_t n = 0; n < vertex_array.size(); n++, idx++) {
					const Vertex &v = vertex_array[n];
					w[idx] = v.color;
				}

				w.release();
				a[i] = array;
			} break;
			case Mesh::ARRAY_BONES: {
				PoolVector<int> array;
				array.resize(varr_len * Vertex::MAX_BONES);
				PoolVector<int>::Write w = array.write();

				int idx = 0;
				for (uint32_t n = 0; n < vertex_array.size(); n++, idx += Vertex::MAX_BONES) {
					const Vertex &v = vertex_array[n];

					ERR_CONTINUE(v.num_bones != Vertex::MAX_BONES);
					for (int j = 0; j < Vertex::MAX_BONES; j++) {
						w[idx + j] = v.bones[j];
					}
				}

				w.release();
				a[i] = array;

			} break;
			case Mesh::ARRAY_WEIGHTS: {
				PoolVector<float> array;
				array.resize(varr_len * Vertex::MAX_BONES);
				PoolVector<float>::Write w = array.write();

				int idx = 0;
				for (uint32_t n = 0; n < vertex_array.size(); n++, idx += Vertex::MAX_BONES) {
					const Vertex &v = vertex_array[n];

					ERR_CONTINUE(v.num_bones != Vertex::MAX_BONES);

					for (int j = 0; j < Vertex::MAX_BONES; j++) {
						w[idx + j] = v.weights[j];
					}
				}

				w.release();
				a[i] = array;

			} break;
			case Mesh::ARRAY_INDEX: {
				ERR_CONTINUE(index_array.size() == 0);

				PoolVector<int> array;
				array.resize(index_array.size());
				PoolVector<int>::Write w = array.write();

				for (uint32_t n = 0; n < index_array.size(); n++) {
					w[n] = index_array[n];
				}

				w.release();

				a[i] = array;
			} break;

			default: {
			}
		}
	}

	return a;
}

Ref<ArrayMesh> SurfaceTool::commit(const Ref<ArrayMesh> &p_existing, uint32_t p_flags) {
	Ref<ArrayMesh> mesh;
	if (p_existing.is_valid()) {
		mesh = p_existing;
	} else {
		mesh.instance();
	}

	int varr_len = vertex_array.size();

	if (varr_len == 0) {
		return mesh;
	}

	int surface = mesh->get_surface_count();

	Array a = commit_to_arrays();

	mesh->add_surface_from_arrays(primitive, a, Array(), p_flags);

	if (material.is_valid()) {
		mesh->surface_set_material(surface, material);
	}

	return mesh;
}

void SurfaceTool::index() {
	if (index_array.size()) {
		return; //already indexed
	}

	HashMap<Vertex, int, VertexHasher> indices;
	LocalVector<Vertex> new_vertices;

	// probably will use less, but this prevents a bunch of resizing
	new_vertices.reserve(vertex_array.size());

	for (uint32_t n = 0; n < vertex_array.size(); n++) {
		const Vertex &v = vertex_array[n];

		int *idxptr = indices.getptr(v);

		int idx;
		if (!idxptr) {
			idx = indices.size();
			new_vertices.push_back(v);
			indices[v] = idx;
		} else {
			idx = *idxptr;
		}

		index_array.push_back(idx);
	}

	vertex_array.clear();
	vertex_array = new_vertices;

	format |= Mesh::ARRAY_FORMAT_INDEX;
}

void SurfaceTool::deindex() {
	if (index_array.size() == 0) {
		return; //nothing to deindex
	}

	// make a copy of source verts
	LocalVector<Vertex> varr = vertex_array;
	vertex_array.resize(index_array.size());
	Vertex *dest_vert = vertex_array.ptr();

	for (uint32_t n = 0; n < index_array.size(); n++) {
		int idx = index_array[n];
		ERR_FAIL_INDEX(idx, (int)varr.size());
		*dest_vert++ = varr[idx];
	}

	format &= ~Mesh::ARRAY_FORMAT_INDEX;
	index_array.clear();
}

void SurfaceTool::_create_list(const Ref<Mesh> &p_existing, int p_surface, LocalVector<Vertex> *r_vertex, LocalVector<int> *r_index, uint32_t &lformat) {
	ERR_FAIL_COND_MSG(p_existing.is_null(), "First argument in SurfaceTool::_create_list() must be a valid object of type Mesh");
	Array arr = p_existing->surface_get_arrays(p_surface);
	ERR_FAIL_COND(arr.size() != VS::ARRAY_MAX);
	_create_list_from_arrays(arr, r_vertex, r_index, lformat);
}

Vector<SurfaceTool::Vertex> SurfaceTool::create_vertex_array_from_triangle_arrays(const Array &p_arrays) {
	Vector<SurfaceTool::Vertex> ret;

	PoolVector<Vector3> varr = p_arrays[VS::ARRAY_VERTEX];
	PoolVector<Vector3> narr = p_arrays[VS::ARRAY_NORMAL];
	PoolVector<float> tarr = p_arrays[VS::ARRAY_TANGENT];
	PoolVector<Color> carr = p_arrays[VS::ARRAY_COLOR];
	PoolVector<Vector2> uvarr = p_arrays[VS::ARRAY_TEX_UV];
	PoolVector<Vector2> uv2arr = p_arrays[VS::ARRAY_TEX_UV2];
	PoolVector<int> barr = p_arrays[VS::ARRAY_BONES];
	PoolVector<float> warr = p_arrays[VS::ARRAY_WEIGHTS];

	int vc = varr.size();

	if (vc == 0) {
		return ret;
	}
	int lformat = 0;

	PoolVector<Vector3>::Read rv;
	if (varr.size()) {
		lformat |= VS::ARRAY_FORMAT_VERTEX;
		rv = varr.read();
	}
	PoolVector<Vector3>::Read rn;
	if (narr.size()) {
		lformat |= VS::ARRAY_FORMAT_NORMAL;
		rn = narr.read();
	}
	PoolVector<float>::Read rt;
	if (tarr.size()) {
		lformat |= VS::ARRAY_FORMAT_TANGENT;
		rt = tarr.read();
	}
	PoolVector<Color>::Read rc;
	if (carr.size()) {
		lformat |= VS::ARRAY_FORMAT_COLOR;
		rc = carr.read();
	}

	PoolVector<Vector2>::Read ruv;
	if (uvarr.size()) {
		lformat |= VS::ARRAY_FORMAT_TEX_UV;
		ruv = uvarr.read();
	}

	PoolVector<Vector2>::Read ruv2;
	if (uv2arr.size()) {
		lformat |= VS::ARRAY_FORMAT_TEX_UV2;
		ruv2 = uv2arr.read();
	}

	PoolVector<int>::Read rb;
	if (barr.size()) {
		lformat |= VS::ARRAY_FORMAT_BONES;
		rb = barr.read();
	}

	PoolVector<float>::Read rw;
	if (warr.size()) {
		lformat |= VS::ARRAY_FORMAT_WEIGHTS;
		rw = warr.read();
	}

	ret.resize(vc);
	Vertex *ret_dest = ret.ptrw();

	for (int i = 0; i < vc; i++) {
		Vertex &v = *ret_dest++;
		if (lformat & VS::ARRAY_FORMAT_VERTEX) {
			v.vertex = varr[i];
		}
		if (lformat & VS::ARRAY_FORMAT_NORMAL) {
			v.normal = narr[i];
		}
		if (lformat & VS::ARRAY_FORMAT_TANGENT) {
			Plane p(tarr[i * 4 + 0], tarr[i * 4 + 1], tarr[i * 4 + 2], tarr[i * 4 + 3]);
			v.tangent = p.normal;
			v.binormal = p.normal.cross(v.tangent).normalized() * p.d;
		}
		if (lformat & VS::ARRAY_FORMAT_COLOR) {
			v.color = carr[i];
		}
		if (lformat & VS::ARRAY_FORMAT_TEX_UV) {
			v.uv = uvarr[i];
		}
		if (lformat & VS::ARRAY_FORMAT_TEX_UV2) {
			v.uv2 = uv2arr[i];
		}
		if (lformat & VS::ARRAY_FORMAT_BONES) {
			v.num_bones = Vertex::MAX_BONES;
			for (int b = 0; b < Vertex::MAX_BONES; b++) {
				v.bones[b] = barr[i * Vertex::MAX_BONES + b];
			}
		}
		if (lformat & VS::ARRAY_FORMAT_WEIGHTS) {
			v.num_bones = Vertex::MAX_BONES;
			for (int b = 0; b < Vertex::MAX_BONES; b++) {
				v.weights[b] = warr[i * Vertex::MAX_BONES + b];
			}
		}
	}

	return ret;
}

void SurfaceTool::_create_list_from_arrays(Array arr, LocalVector<Vertex> *r_vertex, LocalVector<int> *r_index, uint32_t &lformat) {
	PoolVector<Vector3> varr = arr[VS::ARRAY_VERTEX];
	PoolVector<Vector3> narr = arr[VS::ARRAY_NORMAL];
	PoolVector<float> tarr = arr[VS::ARRAY_TANGENT];
	PoolVector<Color> carr = arr[VS::ARRAY_COLOR];
	PoolVector<Vector2> uvarr = arr[VS::ARRAY_TEX_UV];
	PoolVector<Vector2> uv2arr = arr[VS::ARRAY_TEX_UV2];
	PoolVector<int> barr = arr[VS::ARRAY_BONES];
	PoolVector<float> warr = arr[VS::ARRAY_WEIGHTS];

	int vc = varr.size();

	if (vc == 0) {
		return;
	}
	lformat = 0;

	PoolVector<Vector3>::Read rv;
	if (varr.size()) {
		lformat |= VS::ARRAY_FORMAT_VERTEX;
		rv = varr.read();
	}
	PoolVector<Vector3>::Read rn;
	if (narr.size()) {
		lformat |= VS::ARRAY_FORMAT_NORMAL;
		rn = narr.read();
	}
	PoolVector<float>::Read rt;
	if (tarr.size()) {
		lformat |= VS::ARRAY_FORMAT_TANGENT;
		rt = tarr.read();
	}
	PoolVector<Color>::Read rc;
	if (carr.size()) {
		lformat |= VS::ARRAY_FORMAT_COLOR;
		rc = carr.read();
	}

	PoolVector<Vector2>::Read ruv;
	if (uvarr.size()) {
		lformat |= VS::ARRAY_FORMAT_TEX_UV;
		ruv = uvarr.read();
	}

	PoolVector<Vector2>::Read ruv2;
	if (uv2arr.size()) {
		lformat |= VS::ARRAY_FORMAT_TEX_UV2;
		ruv2 = uv2arr.read();
	}

	PoolVector<int>::Read rb;
	if (barr.size()) {
		lformat |= VS::ARRAY_FORMAT_BONES;
		rb = barr.read();
	}

	PoolVector<float>::Read rw;
	if (warr.size()) {
		lformat |= VS::ARRAY_FORMAT_WEIGHTS;
		rw = warr.read();
	}

	DEV_ASSERT(vc);
	uint32_t start = r_vertex->size();
	r_vertex->resize(start + vc);
	Vertex *vert_dest = &r_vertex->operator[](start);

	for (int i = 0; i < vc; i++) {
		Vertex &v = *vert_dest++;
		if (lformat & VS::ARRAY_FORMAT_VERTEX) {
			v.vertex = varr[i];
		}
		if (lformat & VS::ARRAY_FORMAT_NORMAL) {
			v.normal = narr[i];
		}
		if (lformat & VS::ARRAY_FORMAT_TANGENT) {
			Plane p(tarr[i * 4 + 0], tarr[i * 4 + 1], tarr[i * 4 + 2], tarr[i * 4 + 3]);
			v.tangent = p.normal;
			v.binormal = p.normal.cross(v.tangent).normalized() * p.d;
		}
		if (lformat & VS::ARRAY_FORMAT_COLOR) {
			v.color = carr[i];
		}
		if (lformat & VS::ARRAY_FORMAT_TEX_UV) {
			v.uv = uvarr[i];
		}
		if (lformat & VS::ARRAY_FORMAT_TEX_UV2) {
			v.uv2 = uv2arr[i];
		}
		if (lformat & VS::ARRAY_FORMAT_BONES) {
			v.num_bones = Vertex::MAX_BONES;
			for (int b = 0; b < Vertex::MAX_BONES; b++) {
				v.bones[b] = barr[i * Vertex::MAX_BONES + b];
			}
		}
		if (lformat & VS::ARRAY_FORMAT_WEIGHTS) {
			v.num_bones = Vertex::MAX_BONES;
			for (int b = 0; b < Vertex::MAX_BONES; b++) {
				v.weights[b] = warr[i * Vertex::MAX_BONES + b];
			}
		}
	}

	//indices

	PoolVector<int> idx = arr[VS::ARRAY_INDEX];
	int is = idx.size();
	if (is) {
		lformat |= VS::ARRAY_FORMAT_INDEX;
		PoolVector<int>::Read iarr = idx.read();

		uint32_t ind_start = r_index->size();
		r_index->resize(ind_start + is);
		int *ind_dest = &r_index->operator[](ind_start);

		for (int i = 0; i < is; i++) {
			*ind_dest = iarr[i];
			ind_dest++;
		}
	}
}

void SurfaceTool::create_from_triangle_arrays(const Array &p_arrays) {
	clear();
	primitive = Mesh::PRIMITIVE_TRIANGLES;
	_create_list_from_arrays(p_arrays, &vertex_array, &index_array, format);
}

void SurfaceTool::create_from(const Ref<Mesh> &p_existing, int p_surface) {
	ERR_FAIL_COND_MSG(p_existing.is_null(), "First argument in SurfaceTool::create_from() must be a valid object of type Mesh");
	clear();
	primitive = p_existing->surface_get_primitive_type(p_surface);
	_create_list(p_existing, p_surface, &vertex_array, &index_array, format);
	material = p_existing->surface_get_material(p_surface);
}

void SurfaceTool::create_from_blend_shape(const Ref<Mesh> &p_existing, int p_surface, const String &p_blend_shape_name) {
	ERR_FAIL_COND_MSG(p_existing.is_null(), "First argument in SurfaceTool::create_from_blend_shape() must be a valid object of type Mesh");
	clear();
	primitive = p_existing->surface_get_primitive_type(p_surface);
	Array arr = p_existing->surface_get_blend_shape_arrays(p_surface);
	Array blend_shape_names;
	int32_t shape_idx = -1;
	for (int32_t i = 0; i < p_existing->get_blend_shape_count(); i++) {
		String name = p_existing->get_blend_shape_name(i);
		if (name == p_blend_shape_name) {
			shape_idx = i;
			break;
		}
	}
	ERR_FAIL_COND(shape_idx == -1);
	ERR_FAIL_COND(shape_idx >= arr.size());
	Array mesh = arr[shape_idx];
	ERR_FAIL_COND(mesh.size() != VS::ARRAY_MAX);
	_create_list_from_arrays(arr[shape_idx], &vertex_array, &index_array, format);
}

// returns number of indices found within the subset
int SurfaceTool::create_from_subset(const SurfaceTool &p_source, const LocalVector<uint32_t> &p_ids, uint32_t p_subset_id) {
	clear();

	bool was_indexed = p_source.index_array.size() != 0;

	// expecting deindexed input for now as easier to deal with
	ERR_FAIL_COND_V(was_indexed, 0);

	// only deals with triangles
	ERR_FAIL_COND_V(p_source.primitive != Mesh::PRIMITIVE_TRIANGLES, 0);
	primitive = p_source.primitive;

	uint32_t num_source_tris = p_source.vertex_array.size() / 3;
	DEV_ASSERT((p_source.vertex_array.size() % 3) == 0);

	ERR_FAIL_COND_V(num_source_tris != p_ids.size(), 0);

	const Vertex *v[3];
	const Vertex *input = p_source.vertex_array.ptr();

	HashMap<Vertex, int, VertexHasher> indices;

	for (uint32_t t = 0; t < num_source_tris; t++) {
		v[0] = input++;
		v[1] = input++;
		v[2] = input++;

		if (p_ids[t] == p_subset_id) {
			// we can use this triangle
			for (int i = 0; i < 3; i++) {
				const Vertex &vert = *v[i];

				int *idxptr = indices.getptr(vert);

				int idx;
				if (!idxptr) {
					idx = indices.size();
					vertex_array.push_back(vert);
					indices[vert] = idx;
				} else {
					idx = *idxptr;
				}

				index_array.push_back(idx);
			} // for i
		} // bound intersects
	}

	// steal the format from the source surface tool
	format = p_source.format;
	format |= Mesh::ARRAY_FORMAT_INDEX;

	return get_num_draw_vertices();
}

void SurfaceTool::append_from(const Ref<Mesh> &p_existing, int p_surface, const Transform &p_xform) {
	ERR_FAIL_COND_MSG(p_existing.is_null(), "First argument in SurfaceTool::append_from() must be a valid object of type Mesh");
	if (vertex_array.size() == 0) {
		primitive = p_existing->surface_get_primitive_type(p_surface);
		format = 0;
	}

	uint32_t nformat;
	LocalVector<Vertex> nvertices;
	LocalVector<int> nindices;
	_create_list(p_existing, p_surface, &nvertices, &nindices, nformat);
	format |= nformat;
	int vfrom = vertex_array.size();

	if (nvertices.size()) {
		vertex_array.resize(vfrom + nvertices.size());
		Vertex *dest_vert = &vertex_array[vfrom];

		for (uint32_t n = 0; n < nvertices.size(); n++) {
			Vertex &v = *dest_vert++;
			v = nvertices[n];
			v.vertex = p_xform.xform(v.vertex);
			if (nformat & VS::ARRAY_FORMAT_NORMAL) {
				v.normal = p_xform.basis.xform(v.normal);
			}
			if (nformat & VS::ARRAY_FORMAT_TANGENT) {
				v.tangent = p_xform.basis.xform(v.tangent);
				v.binormal = p_xform.basis.xform(v.binormal);
			}
		}
	} // if there were new vertices to add

	if (nindices.size()) {
		int ind_start = index_array.size();
		index_array.resize(ind_start + nindices.size());
		int *dest_ind = &index_array[ind_start];

		for (uint32_t n = 0; n < nindices.size(); n++) {
			int dst_index = nindices[n] + vfrom;
			*dest_ind++ = dst_index;
		}
	} // if there were new indices to add
	if (index_array.size() % 3) {
		WARN_PRINT("SurfaceTool: Index array not a multiple of 3.");
	}
}

//mikktspace callbacks
namespace {
struct TangentGenerationContextUserData {
	Vector<SurfaceTool::Vertex *> vertices;
	Vector<int> indices;
};
} // namespace

int SurfaceTool::mikktGetNumFaces(const SMikkTSpaceContext *pContext) {
	TangentGenerationContextUserData &triangle_data = *reinterpret_cast<TangentGenerationContextUserData *>(pContext->m_pUserData);

	if (triangle_data.indices.size() > 0) {
		return triangle_data.indices.size() / 3;
	} else {
		return triangle_data.vertices.size() / 3;
	}
}
int SurfaceTool::mikktGetNumVerticesOfFace(const SMikkTSpaceContext *pContext, const int iFace) {
	return 3; //always 3
}
void SurfaceTool::mikktGetPosition(const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace, const int iVert) {
	TangentGenerationContextUserData &triangle_data = *reinterpret_cast<TangentGenerationContextUserData *>(pContext->m_pUserData);
	Vector3 v;
	if (triangle_data.indices.size() > 0) {
		int index = triangle_data.indices[iFace * 3 + iVert];
		if (index < triangle_data.vertices.size()) {
			v = triangle_data.vertices[index]->vertex;
		}
	} else {
		v = triangle_data.vertices[iFace * 3 + iVert]->vertex;
	}

	fvPosOut[0] = v.x;
	fvPosOut[1] = v.y;
	fvPosOut[2] = v.z;
}

void SurfaceTool::mikktGetNormal(const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace, const int iVert) {
	TangentGenerationContextUserData &triangle_data = *reinterpret_cast<TangentGenerationContextUserData *>(pContext->m_pUserData);
	Vector3 v;
	if (triangle_data.indices.size() > 0) {
		int index = triangle_data.indices[iFace * 3 + iVert];
		if (index < triangle_data.vertices.size()) {
			v = triangle_data.vertices[index]->normal;
		}
	} else {
		v = triangle_data.vertices[iFace * 3 + iVert]->normal;
	}

	fvNormOut[0] = v.x;
	fvNormOut[1] = v.y;
	fvNormOut[2] = v.z;
}
void SurfaceTool::mikktGetTexCoord(const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace, const int iVert) {
	TangentGenerationContextUserData &triangle_data = *reinterpret_cast<TangentGenerationContextUserData *>(pContext->m_pUserData);
	Vector2 v;
	if (triangle_data.indices.size() > 0) {
		int index = triangle_data.indices[iFace * 3 + iVert];
		if (index < triangle_data.vertices.size()) {
			v = triangle_data.vertices[index]->uv;
		}
	} else {
		v = triangle_data.vertices[iFace * 3 + iVert]->uv;
	}

	fvTexcOut[0] = v.x;
	fvTexcOut[1] = v.y;
}

void SurfaceTool::mikktSetTSpaceDefault(const SMikkTSpaceContext *pContext, const float fvTangent[], const float fvBiTangent[], const float fMagS, const float fMagT,
		const tbool bIsOrientationPreserving, const int iFace, const int iVert) {
	TangentGenerationContextUserData &triangle_data = *reinterpret_cast<TangentGenerationContextUserData *>(pContext->m_pUserData);
	Vertex *vtx = nullptr;
	if (triangle_data.indices.size() > 0) {
		int index = triangle_data.indices[iFace * 3 + iVert];
		if (index < triangle_data.vertices.size()) {
			vtx = triangle_data.vertices[index];
		}
	} else {
		vtx = triangle_data.vertices[iFace * 3 + iVert];
	}

	if (vtx != nullptr) {
		vtx->tangent = Vector3(fvTangent[0], fvTangent[1], fvTangent[2]);
		vtx->binormal = Vector3(-fvBiTangent[0], -fvBiTangent[1], -fvBiTangent[2]); // for some reason these are reversed, something with the coordinate system in Godot
	}
}

void SurfaceTool::generate_tangents() {
	ERR_FAIL_COND(!(format & Mesh::ARRAY_FORMAT_TEX_UV));
	ERR_FAIL_COND(!(format & Mesh::ARRAY_FORMAT_NORMAL));

	SMikkTSpaceInterface mkif;
	mkif.m_getNormal = mikktGetNormal;
	mkif.m_getNumFaces = mikktGetNumFaces;
	mkif.m_getNumVerticesOfFace = mikktGetNumVerticesOfFace;
	mkif.m_getPosition = mikktGetPosition;
	mkif.m_getTexCoord = mikktGetTexCoord;
	mkif.m_setTSpace = mikktSetTSpaceDefault;
	mkif.m_setTSpaceBasic = nullptr;

	SMikkTSpaceContext msc;
	msc.m_pInterface = &mkif;

	TangentGenerationContextUserData triangle_data;
	triangle_data.vertices.resize(vertex_array.size());
	int idx = 0;
	for (uint32_t n = 0; n < vertex_array.size(); n++) {
		Vertex &v = vertex_array[n];
		triangle_data.vertices.write[idx++] = &v;
		v.binormal = Vector3();
		v.tangent = Vector3();
	}
	triangle_data.indices.resize(index_array.size());
	for (uint32_t n = 0; n < index_array.size(); n++) {
		triangle_data.indices.write[n] = index_array[n];
	}
	msc.m_pUserData = &triangle_data;

	bool res = genTangSpaceDefault(&msc);

	ERR_FAIL_COND(!res);
	format |= Mesh::ARRAY_FORMAT_TANGENT;
}

void SurfaceTool::generate_normals(bool p_flip) {
	ERR_FAIL_COND(primitive != Mesh::PRIMITIVE_TRIANGLES);

	bool was_indexed = index_array.size();

	deindex();

	HashMap<Vertex, Vector3, VertexHasher> vertex_hash;

	int count = 0;
	bool smooth = false;
	int smooth_group_start = 0;
	if (smooth_groups.has(0)) {
		smooth = smooth_groups[0];
	}

	for (uint32_t t = 0; t < vertex_array.size(); t += 3) {
		Vertex *v[3];
		v[0] = &vertex_array[t];
		v[1] = &vertex_array[t + 1];
		v[2] = &vertex_array[t + 2];

		Vector3 normal;
		if (!p_flip) {
			normal = Plane(v[0]->vertex, v[1]->vertex, v[2]->vertex).normal;
		} else {
			normal = Plane(v[2]->vertex, v[1]->vertex, v[0]->vertex).normal;
		}

		if (smooth) {
			for (int i = 0; i < 3; i++) {
				Vector3 *lv = vertex_hash.getptr(*v[i]);
				if (!lv) {
					vertex_hash.set(*v[i], normal);
				} else {
					(*lv) += normal;
				}
			}
		} else {
			for (int i = 0; i < 3; i++) {
				v[i]->normal = normal;
			}
		}
		count += 3;

		// terminating smooth group, bake the smoothing group
		if (smooth_groups.has(count)) {
			_apply_smoothing_group(vertex_hash, smooth_group_start, count, smooth);
			smooth_group_start = count;
		}
	}

	// always terminate the last smoothing group
	_apply_smoothing_group(vertex_hash, smooth_group_start, vertex_array.size(), smooth);

	format |= Mesh::ARRAY_FORMAT_NORMAL;

	if (was_indexed) {
		index();
		smooth_groups.clear();
	}
}

void SurfaceTool::_apply_smoothing_group(HashMap<Vertex, Vector3, VertexHasher> &r_vertex_hash, uint32_t p_from, uint32_t p_to, bool &r_smooth) {
	if (r_vertex_hash.size()) {
		for (uint32_t n = p_from; n < p_to; n++) {
			Vertex &v = vertex_array[n];

			Vector3 *lv = r_vertex_hash.getptr(v);
			if (lv) {
				v.normal = lv->normalized();
			}
		}
	}

	r_vertex_hash.clear();
	if (p_to < vertex_array.size()) {
		r_smooth = smooth_groups[p_to];
	}
}

void SurfaceTool::set_material(const Ref<Material> &p_material) {
	material = p_material;
}

void SurfaceTool::clear() {
	begun = false;
	primitive = Mesh::PRIMITIVE_LINES;
	format = 0;
	last_bones.clear();
	last_weights.clear();
	index_array.clear();
	vertex_array.clear();
	smooth_groups.clear();
	material.unref();
}

void SurfaceTool::_bind_methods() {
	ClassDB::bind_method(D_METHOD("begin", "primitive"), &SurfaceTool::begin);

	ClassDB::bind_method(D_METHOD("add_vertex", "vertex"), &SurfaceTool::add_vertex);
	ClassDB::bind_method(D_METHOD("add_color", "color"), &SurfaceTool::add_color);
	ClassDB::bind_method(D_METHOD("add_normal", "normal"), &SurfaceTool::add_normal);
	ClassDB::bind_method(D_METHOD("add_tangent", "tangent"), &SurfaceTool::add_tangent);
	ClassDB::bind_method(D_METHOD("add_uv", "uv"), &SurfaceTool::add_uv);
	ClassDB::bind_method(D_METHOD("add_uv2", "uv2"), &SurfaceTool::add_uv2);
	ClassDB::bind_method(D_METHOD("add_bones", "bones"), &SurfaceTool::add_bones);
	ClassDB::bind_method(D_METHOD("add_weights", "weights"), &SurfaceTool::add_weights);
	ClassDB::bind_method(D_METHOD("add_smooth_group", "smooth"), &SurfaceTool::add_smooth_group);

	ClassDB::bind_method(D_METHOD("add_triangle_fan", "vertices", "uvs", "colors", "uv2s", "normals", "tangents"), &SurfaceTool::add_triangle_fan, DEFVAL(Vector<Vector2>()), DEFVAL(Vector<Color>()), DEFVAL(Vector<Vector2>()), DEFVAL(Vector<Vector3>()), DEFVAL(Vector<Plane>()));

	ClassDB::bind_method(D_METHOD("add_index", "index"), &SurfaceTool::add_index);

	ClassDB::bind_method(D_METHOD("index"), &SurfaceTool::index);
	ClassDB::bind_method(D_METHOD("deindex"), &SurfaceTool::deindex);
	ClassDB::bind_method(D_METHOD("generate_normals", "flip"), &SurfaceTool::generate_normals, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("generate_tangents"), &SurfaceTool::generate_tangents);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &SurfaceTool::set_material);

	ClassDB::bind_method(D_METHOD("clear"), &SurfaceTool::clear);

	ClassDB::bind_method(D_METHOD("create_from", "existing", "surface"), &SurfaceTool::create_from);
	ClassDB::bind_method(D_METHOD("create_from_blend_shape", "existing", "surface", "blend_shape"), &SurfaceTool::create_from_blend_shape);
	ClassDB::bind_method(D_METHOD("append_from", "existing", "surface", "transform"), &SurfaceTool::append_from);
	ClassDB::bind_method(D_METHOD("commit", "existing", "flags"), &SurfaceTool::commit, DEFVAL(Variant()), DEFVAL(Mesh::ARRAY_COMPRESS_DEFAULT));
	ClassDB::bind_method(D_METHOD("commit_to_arrays"), &SurfaceTool::commit_to_arrays);
}

SurfaceTool::SurfaceTool() {
	first = false;
	begun = false;
	primitive = Mesh::PRIMITIVE_LINES;
	format = 0;
}
