/*************************************************************************/
/*  surface_tool.cpp                                                     */
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
#include "surface_tool.h"
#include "method_bind_ext.inc"

#define _VERTEX_SNAP 0.0001
#define EQ_VERTEX_DIST 0.00001

bool SurfaceTool::Vertex::operator==(const Vertex &p_b) const {

	if (vertex != p_b.vertex)
		return false;

	if (uv != p_b.uv)
		return false;

	if (uv2 != p_b.uv2)
		return false;

	if (normal != p_b.normal)
		return false;

	if (binormal != p_b.binormal)
		return false;

	if (color != p_b.color)
		return false;

	if (bones.size() != p_b.bones.size())
		return false;

	for (int i = 0; i < bones.size(); i++) {
		if (bones[i] != p_b.bones[i])
			return false;
	}

	for (int i = 0; i < weights.size(); i++) {
		if (weights[i] != p_b.weights[i])
			return false;
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
	h = hash_djb2_buffer((const uint8_t *)p_vtx.bones.ptr(), p_vtx.bones.size() * sizeof(int), h);
	h = hash_djb2_buffer((const uint8_t *)p_vtx.weights.ptr(), p_vtx.weights.size() * sizeof(float), h);
	return h;
}

void SurfaceTool::begin(Mesh::PrimitiveType p_primitive) {

	clear();

	primitive = p_primitive;
	begun = true;
	first = true;
}

void SurfaceTool::add_vertex(const Vector3 &p_vertex) {

	ERR_FAIL_COND(!begun);

	Vertex vtx;
	vtx.vertex = p_vertex;
	vtx.color = last_color;
	vtx.normal = last_normal;
	vtx.uv = last_uv;
	vtx.weights = last_weights;
	vtx.bones = last_bones;
	vtx.tangent = last_tangent.normal;
	vtx.binormal = last_normal.cross(last_tangent.normal).normalized() * last_tangent.d;
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
	ERR_FAIL_COND(p_bones.size() != 4);
	ERR_FAIL_COND(!first && !(format & Mesh::ARRAY_FORMAT_BONES));

	format |= Mesh::ARRAY_FORMAT_BONES;
	last_bones = p_bones;
}

void SurfaceTool::add_weights(const Vector<float> &p_weights) {

	ERR_FAIL_COND(!begun);

	ERR_FAIL_COND(p_weights.size() != 4);
	ERR_FAIL_COND(!first && !(format & Mesh::ARRAY_FORMAT_WEIGHTS));

	format |= Mesh::ARRAY_FORMAT_WEIGHTS;
	last_weights = p_weights;
}

void SurfaceTool::add_smooth_group(bool p_smooth) {

	ERR_FAIL_COND(!begun);
	if (index_array.size()) {
		smooth_groups[index_array.size()] = p_smooth;
	} else {

		smooth_groups[vertex_array.size()] = p_smooth;
	}
}

void SurfaceTool::add_triangle_fan(const Vector<Vector3> &p_vertexes, const Vector<Vector2> &p_uvs, const Vector<Color> &p_colors, const Vector<Vector2> &p_uv2s, const Vector<Vector3> &p_normals, const Vector<Plane> &p_tangents) {
	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND(primitive != Mesh::PRIMITIVE_TRIANGLES);
	ERR_FAIL_COND(p_vertexes.size() < 3);

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
		add_vertex(p_vertexes[n]);      \
	}

	for (int i = 0; i < p_vertexes.size() - 2; i++) {
		ADD_POINT(0);
		ADD_POINT(i + 1);
		ADD_POINT(i + 2);
	}

#undef ADD_POINT
}

void SurfaceTool::add_index(int p_index) {

	ERR_FAIL_COND(!begun);

	format |= Mesh::ARRAY_FORMAT_INDEX;
	index_array.push_back(p_index);
}

Ref<Mesh> SurfaceTool::commit(const Ref<Mesh> &p_existing) {

	Ref<Mesh> mesh;
	if (p_existing.is_valid())
		mesh = p_existing;
	else
		mesh = Ref<Mesh>(memnew(Mesh));

	int varr_len = vertex_array.size();

	if (varr_len == 0)
		return mesh;

	int surface = mesh->get_surface_count();

	Array a;
	a.resize(Mesh::ARRAY_MAX);

	for (int i = 0; i < Mesh::ARRAY_MAX; i++) {

		switch (format & (1 << i)) {

			case Mesh::ARRAY_FORMAT_VERTEX:
			case Mesh::ARRAY_FORMAT_NORMAL: {

				PoolVector<Vector3> array;
				array.resize(varr_len);
				PoolVector<Vector3>::Write w = array.write();

				int idx = 0;
				for (List<Vertex>::Element *E = vertex_array.front(); E; E = E->next(), idx++) {

					const Vertex &v = E->get();

					switch (i) {
						case Mesh::ARRAY_VERTEX: {
							w[idx] = v.vertex;
						} break;
						case Mesh::ARRAY_NORMAL: {
							w[idx] = v.normal;
						} break;
					}
				}

				w = PoolVector<Vector3>::Write();
				a[i] = array;

			} break;

			case Mesh::ARRAY_FORMAT_TEX_UV:
			case Mesh::ARRAY_FORMAT_TEX_UV2: {

				PoolVector<Vector2> array;
				array.resize(varr_len);
				PoolVector<Vector2>::Write w = array.write();

				int idx = 0;
				for (List<Vertex>::Element *E = vertex_array.front(); E; E = E->next(), idx++) {

					const Vertex &v = E->get();

					switch (i) {

						case Mesh::ARRAY_TEX_UV: {
							w[idx] = v.uv;
						} break;
						case Mesh::ARRAY_TEX_UV2: {
							w[idx] = v.uv2;
						} break;
					}
				}

				w = PoolVector<Vector2>::Write();
				a[i] = array;
			} break;
			case Mesh::ARRAY_FORMAT_TANGENT: {

				PoolVector<float> array;
				array.resize(varr_len * 4);
				PoolVector<float>::Write w = array.write();

				int idx = 0;
				for (List<Vertex>::Element *E = vertex_array.front(); E; E = E->next(), idx += 4) {

					const Vertex &v = E->get();

					w[idx + 0] = v.tangent.x;
					w[idx + 1] = v.tangent.y;
					w[idx + 2] = v.tangent.z;

					//float d = v.tangent.dot(v.binormal,v.normal);
					float d = v.binormal.dot(v.normal.cross(v.tangent));
					w[idx + 3] = d < 0 ? -1 : 1;
				}

				w = PoolVector<float>::Write();
				a[i] = array;

			} break;
			case Mesh::ARRAY_FORMAT_COLOR: {

				PoolVector<Color> array;
				array.resize(varr_len);
				PoolVector<Color>::Write w = array.write();

				int idx = 0;
				for (List<Vertex>::Element *E = vertex_array.front(); E; E = E->next(), idx++) {

					const Vertex &v = E->get();
					w[idx] = v.color;
				}

				w = PoolVector<Color>::Write();
				a[i] = array;
			} break;
			case Mesh::ARRAY_FORMAT_BONES: {

				PoolVector<int> array;
				array.resize(varr_len * 4);
				PoolVector<int>::Write w = array.write();

				int idx = 0;
				for (List<Vertex>::Element *E = vertex_array.front(); E; E = E->next(), idx += 4) {

					const Vertex &v = E->get();

					ERR_CONTINUE(v.bones.size() != 4);

					for (int j = 0; j < 4; j++) {
						w[idx + j] = v.bones[j];
					}
				}

				w = PoolVector<int>::Write();
				a[i] = array;

			} break;
			case Mesh::ARRAY_FORMAT_WEIGHTS: {

				PoolVector<float> array;
				array.resize(varr_len * 4);
				PoolVector<float>::Write w = array.write();

				int idx = 0;
				for (List<Vertex>::Element *E = vertex_array.front(); E; E = E->next(), idx += 4) {

					const Vertex &v = E->get();
					ERR_CONTINUE(v.weights.size() != 4);

					for (int j = 0; j < 4; j++) {

						w[idx + j] = v.weights[j];
					}
				}

				w = PoolVector<float>::Write();
				a[i] = array;

			} break;
			case Mesh::ARRAY_FORMAT_INDEX: {

				ERR_CONTINUE(index_array.size() == 0);

				PoolVector<int> array;
				array.resize(index_array.size());
				PoolVector<int>::Write w = array.write();

				int idx = 0;
				for (List<int>::Element *E = index_array.front(); E; E = E->next(), idx++) {

					w[idx] = E->get();
				}

				w = PoolVector<int>::Write();
				a[i] = array;
			} break;

			default: {}
		}
	}

	mesh->add_surface_from_arrays(primitive, a);
	if (material.is_valid())
		mesh->surface_set_material(surface, material);

	return mesh;
}

void SurfaceTool::index() {

	if (index_array.size())
		return; //already indexed

	HashMap<Vertex, int, VertexHasher> indices;
	List<Vertex> new_vertices;

	for (List<Vertex>::Element *E = vertex_array.front(); E; E = E->next()) {

		int *idxptr = indices.getptr(E->get());
		int idx;
		if (!idxptr) {
			idx = indices.size();
			new_vertices.push_back(E->get());
			indices[E->get()] = idx;
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

	if (index_array.size() == 0)
		return; //nothing to deindex
	Vector<Vertex> varr;
	varr.resize(vertex_array.size());
	int idx = 0;
	for (List<Vertex>::Element *E = vertex_array.front(); E; E = E->next()) {

		varr[idx++] = E->get();
	}
	vertex_array.clear();
	for (List<int>::Element *E = index_array.front(); E; E = E->next()) {

		ERR_FAIL_INDEX(E->get(), varr.size());
		vertex_array.push_back(varr[E->get()]);
	}
	format &= ~Mesh::ARRAY_FORMAT_INDEX;
}

void SurfaceTool::_create_list(const Ref<Mesh> &p_existing, int p_surface, List<Vertex> *r_vertex, List<int> *r_index, int &lformat) {

	Array arr = p_existing->surface_get_arrays(p_surface);
	ERR_FAIL_COND(arr.size() != VS::ARRAY_MAX);

	PoolVector<Vector3> varr = arr[VS::ARRAY_VERTEX];
	PoolVector<Vector3> narr = arr[VS::ARRAY_NORMAL];
	PoolVector<float> tarr = arr[VS::ARRAY_TANGENT];
	PoolVector<Color> carr = arr[VS::ARRAY_COLOR];
	PoolVector<Vector2> uvarr = arr[VS::ARRAY_TEX_UV];
	PoolVector<Vector2> uv2arr = arr[VS::ARRAY_TEX_UV2];
	PoolVector<int> barr = arr[VS::ARRAY_BONES];
	PoolVector<float> warr = arr[VS::ARRAY_WEIGHTS];

	int vc = varr.size();

	if (vc == 0)
		return;
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

	for (int i = 0; i < vc; i++) {

		Vertex v;
		if (lformat & VS::ARRAY_FORMAT_VERTEX)
			v.vertex = varr[i];
		if (lformat & VS::ARRAY_FORMAT_NORMAL)
			v.normal = narr[i];
		if (lformat & VS::ARRAY_FORMAT_TANGENT) {
			Plane p(tarr[i * 4 + 0], tarr[i * 4 + 1], tarr[i * 4 + 2], tarr[i * 4 + 3]);
			v.tangent = p.normal;
			v.binormal = p.normal.cross(last_normal).normalized() * p.d;
		}
		if (lformat & VS::ARRAY_FORMAT_COLOR)
			v.color = carr[i];
		if (lformat & VS::ARRAY_FORMAT_TEX_UV)
			v.uv = uvarr[i];
		if (lformat & VS::ARRAY_FORMAT_TEX_UV2)
			v.uv2 = uv2arr[i];
		if (lformat & VS::ARRAY_FORMAT_BONES) {
			Vector<int> b;
			b.resize(4);
			b[0] = barr[i * 4 + 0];
			b[1] = barr[i * 4 + 1];
			b[2] = barr[i * 4 + 2];
			b[3] = barr[i * 4 + 3];
			v.bones = b;
		}
		if (lformat & VS::ARRAY_FORMAT_WEIGHTS) {
			Vector<float> w;
			w.resize(4);
			w[0] = warr[i * 4 + 0];
			w[1] = warr[i * 4 + 1];
			w[2] = warr[i * 4 + 2];
			w[3] = warr[i * 4 + 3];
			v.weights = w;
		}

		r_vertex->push_back(v);
	}

	//indices

	PoolVector<int> idx = arr[VS::ARRAY_INDEX];
	int is = idx.size();
	if (is) {

		lformat |= VS::ARRAY_FORMAT_INDEX;
		PoolVector<int>::Read iarr = idx.read();
		for (int i = 0; i < is; i++) {
			r_index->push_back(iarr[i]);
		}
	}
}

void SurfaceTool::create_from(const Ref<Mesh> &p_existing, int p_surface) {

	clear();
	primitive = p_existing->surface_get_primitive_type(p_surface);
	_create_list(p_existing, p_surface, &vertex_array, &index_array, format);
	material = p_existing->surface_get_material(p_surface);
}

void SurfaceTool::append_from(const Ref<Mesh> &p_existing, int p_surface, const Transform &p_xform) {

	if (vertex_array.size() == 0) {
		primitive = p_existing->surface_get_primitive_type(p_surface);
		format = 0;
	}

	int nformat;
	List<Vertex> nvertices;
	List<int> nindices;
	_create_list(p_existing, p_surface, &nvertices, &nindices, nformat);
	format |= nformat;
	int vfrom = vertex_array.size();

	for (List<Vertex>::Element *E = nvertices.front(); E; E = E->next()) {

		Vertex v = E->get();
		v.vertex = p_xform.xform(v.vertex);
		if (nformat & VS::ARRAY_FORMAT_NORMAL) {
			v.normal = p_xform.basis.xform(v.normal);
		}
		if (nformat & VS::ARRAY_FORMAT_TANGENT) {
			v.tangent = p_xform.basis.xform(v.tangent);
			v.binormal = p_xform.basis.xform(v.binormal);
		}

		vertex_array.push_back(v);
	}

	for (List<int>::Element *E = nindices.front(); E; E = E->next()) {

		int dst_index = E->get() + vfrom;
		/*
		if (dst_index <0 || dst_index>=vertex_array.size()) {
			print_line("invalid index!");
		}
		*/
		index_array.push_back(dst_index);
	}
	if (index_array.size() % 3)
		print_line("IA not div of 3?");
}

//mikktspace callbacks
int SurfaceTool::mikktGetNumFaces(const SMikkTSpaceContext *pContext) {

	Vector<List<Vertex>::Element *> &varr = *((Vector<List<Vertex>::Element *> *)pContext->m_pUserData);
	return varr.size() / 3;
}
int SurfaceTool::mikktGetNumVerticesOfFace(const SMikkTSpaceContext *pContext, const int iFace) {

	return 3; //always 3
}
void SurfaceTool::mikktGetPosition(const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace, const int iVert) {

	Vector<List<Vertex>::Element *> &varr = *((Vector<List<Vertex>::Element *> *)pContext->m_pUserData);
	Vector3 v = varr[iFace * 3 + iVert]->get().vertex;
	fvPosOut[0] = v.x;
	fvPosOut[1] = v.y;
	fvPosOut[2] = v.z;
}

void SurfaceTool::mikktGetNormal(const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace, const int iVert) {

	Vector<List<Vertex>::Element *> &varr = *((Vector<List<Vertex>::Element *> *)pContext->m_pUserData);
	Vector3 v = varr[iFace * 3 + iVert]->get().normal;
	fvNormOut[0] = v.x;
	fvNormOut[1] = v.y;
	fvNormOut[2] = v.z;
}
void SurfaceTool::mikktGetTexCoord(const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace, const int iVert) {

	Vector<List<Vertex>::Element *> &varr = *((Vector<List<Vertex>::Element *> *)pContext->m_pUserData);
	Vector2 v = varr[iFace * 3 + iVert]->get().uv;
	fvTexcOut[0] = v.x;
	fvTexcOut[1] = v.y;
	//fvTexcOut[1]=1.0-v.y;
}
void SurfaceTool::mikktSetTSpaceBasic(const SMikkTSpaceContext *pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert) {

	Vector<List<Vertex>::Element *> &varr = *((Vector<List<Vertex>::Element *> *)pContext->m_pUserData);
	Vertex &vtx = varr[iFace * 3 + iVert]->get();

	vtx.tangent = Vector3(fvTangent[0], fvTangent[1], fvTangent[2]);
	vtx.binormal = vtx.normal.cross(vtx.tangent) * fSign;
}

void SurfaceTool::generate_tangents() {

	ERR_FAIL_COND(!(format & Mesh::ARRAY_FORMAT_TEX_UV));
	ERR_FAIL_COND(!(format & Mesh::ARRAY_FORMAT_NORMAL));

	bool indexed = index_array.size() > 0;
	if (indexed)
		deindex();

	SMikkTSpaceInterface mkif;
	mkif.m_getNormal = mikktGetNormal;
	mkif.m_getNumFaces = mikktGetNumFaces;
	mkif.m_getNumVerticesOfFace = mikktGetNumVerticesOfFace;
	mkif.m_getPosition = mikktGetPosition;
	mkif.m_getTexCoord = mikktGetTexCoord;
	mkif.m_setTSpaceBasic = mikktSetTSpaceBasic;
	mkif.m_setTSpace = NULL;

	SMikkTSpaceContext msc;
	msc.m_pInterface = &mkif;

	Vector<List<Vertex>::Element *> vtx;
	vtx.resize(vertex_array.size());
	int idx = 0;
	for (List<Vertex>::Element *E = vertex_array.front(); E; E = E->next()) {
		vtx[idx++] = E;
		E->get().binormal = Vector3();
		E->get().tangent = Vector3();
	}
	msc.m_pUserData = &vtx;

	bool res = genTangSpaceDefault(&msc);

	ERR_FAIL_COND(!res);
	format |= Mesh::ARRAY_FORMAT_TANGENT;

	if (indexed)
		index();
}

void SurfaceTool::generate_normals() {

	ERR_FAIL_COND(primitive != Mesh::PRIMITIVE_TRIANGLES);

	bool was_indexed = index_array.size();

	deindex();

	HashMap<Vertex, Vector3, VertexHasher> vertex_hash;

	int count = 0;
	bool smooth = false;
	if (smooth_groups.has(0))
		smooth = smooth_groups[0];

	List<Vertex>::Element *B = vertex_array.front();
	for (List<Vertex>::Element *E = B; E;) {

		List<Vertex>::Element *v[3];
		v[0] = E;
		v[1] = v[0]->next();
		ERR_FAIL_COND(!v[1]);
		v[2] = v[1]->next();
		ERR_FAIL_COND(!v[2]);
		E = v[2]->next();

		Vector3 normal = Plane(v[0]->get().vertex, v[1]->get().vertex, v[2]->get().vertex).normal;

		if (smooth) {

			for (int i = 0; i < 3; i++) {

				Vector3 *lv = vertex_hash.getptr(v[i]->get());
				if (!lv) {
					vertex_hash.set(v[i]->get(), normal);
				} else {
					(*lv) += normal;
				}
			}
		} else {

			for (int i = 0; i < 3; i++) {

				v[i]->get().normal = normal;
			}
		}
		count += 3;

		if (smooth_groups.has(count) || !E) {

			if (vertex_hash.size()) {

				while (B != E) {

					Vector3 *lv = vertex_hash.getptr(B->get());
					if (lv) {
						B->get().normal = lv->normalized();
					}

					B = B->next();
				}

			} else {
				B = E;
			}

			vertex_hash.clear();
			if (E) {
				smooth = smooth_groups[count];
				print_line("SMOOTH AT " + itos(count) + ": " + itos(smooth));
			}
		}
	}

	format |= Mesh::ARRAY_FORMAT_NORMAL;

	if (was_indexed) {
		index();
		smooth_groups.clear();
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
	ClassDB::bind_method(D_METHOD("add_triangle_fan", "vertexes", "uvs", "colors", "uv2s", "normals", "tangents"), &SurfaceTool::add_triangle_fan, DEFVAL(Vector<Vector2>()), DEFVAL(Vector<Color>()), DEFVAL(Vector<Vector2>()), DEFVAL(Vector<Vector3>()), DEFVAL(Vector<Plane>()));
	ClassDB::bind_method(D_METHOD("set_material", "material:Material"), &SurfaceTool::set_material);
	ClassDB::bind_method(D_METHOD("index"), &SurfaceTool::index);
	ClassDB::bind_method(D_METHOD("deindex"), &SurfaceTool::deindex);
	///ClassDB::bind_method(D_METHOD("generate_flat_normals"),&SurfaceTool::generate_flat_normals);
	ClassDB::bind_method(D_METHOD("generate_normals"), &SurfaceTool::generate_normals);
	ClassDB::bind_method(D_METHOD("add_index", "index"), &SurfaceTool::add_index);
	ClassDB::bind_method(D_METHOD("commit:Mesh", "existing:Mesh"), &SurfaceTool::commit, DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("clear"), &SurfaceTool::clear);
}

SurfaceTool::SurfaceTool() {

	first = false;
	begun = false;
	primitive = Mesh::PRIMITIVE_LINES;
	format = 0;
}
