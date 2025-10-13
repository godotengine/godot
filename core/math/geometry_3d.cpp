/**************************************************************************/
/*  geometry_3d.cpp                                                       */
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

#include "geometry_3d.h"

#include "core/math/aabb.h"
#include "core/math/color.h"
#include "core/math/delaunay_3d.h"
#include "core/math/face3.h"
#include "core/math/vector2.h"
#include "core/math/vector3i.h"
#include "core/templates/hash_map.h"

void Geometry3D::get_closest_points_between_segments(const Vector3 &p_p0, const Vector3 &p_p1, const Vector3 &p_q0, const Vector3 &p_q1, Vector3 &r_ps, Vector3 &r_qt) {
	// Based on David Eberly's Computation of Distance Between Line Segments algorithm.

	Vector3 p = p_p1 - p_p0;
	Vector3 q = p_q1 - p_q0;
	Vector3 r = p_p0 - p_q0;

	real_t a = p.dot(p);
	real_t b = p.dot(q);
	real_t c = q.dot(q);
	real_t d = p.dot(r);
	real_t e = q.dot(r);

	real_t s = 0.0f;
	real_t t = 0.0f;

	real_t det = a * c - b * b;
	if (det > CMP_EPSILON) {
		// Non-parallel segments
		real_t bte = b * e;
		real_t ctd = c * d;

		if (bte <= ctd) {
			// s <= 0.0f
			if (e <= 0.0f) {
				// t <= 0.0f
				s = (-d >= a ? 1 : (-d > 0.0f ? -d / a : 0.0f));
				t = 0.0f;
			} else if (e < c) {
				// 0.0f < t < 1
				s = 0.0f;
				t = e / c;
			} else {
				// t >= 1
				s = (b - d >= a ? 1 : (b - d > 0.0f ? (b - d) / a : 0.0f));
				t = 1;
			}
		} else {
			// s > 0.0f
			s = bte - ctd;
			if (s >= det) {
				// s >= 1
				if (b + e <= 0.0f) {
					// t <= 0.0f
					s = (-d <= 0.0f ? 0.0f : (-d < a ? -d / a : 1));
					t = 0.0f;
				} else if (b + e < c) {
					// 0.0f < t < 1
					s = 1;
					t = (b + e) / c;
				} else {
					// t >= 1
					s = (b - d <= 0.0f ? 0.0f : (b - d < a ? (b - d) / a : 1));
					t = 1;
				}
			} else {
				// 0.0f < s < 1
				real_t ate = a * e;
				real_t btd = b * d;

				if (ate <= btd) {
					// t <= 0.0f
					s = (-d <= 0.0f ? 0.0f : (-d >= a ? 1 : -d / a));
					t = 0.0f;
				} else {
					// t > 0.0f
					t = ate - btd;
					if (t >= det) {
						// t >= 1
						s = (b - d <= 0.0f ? 0.0f : (b - d >= a ? 1 : (b - d) / a));
						t = 1;
					} else {
						// 0.0f < t < 1
						s /= det;
						t /= det;
					}
				}
			}
		}
	} else {
		// Parallel segments
		if (e <= 0.0f) {
			s = (-d <= 0.0f ? 0.0f : (-d >= a ? 1 : -d / a));
			t = 0.0f;
		} else if (e >= c) {
			s = (b - d <= 0.0f ? 0.0f : (b - d >= a ? 1 : (b - d) / a));
			t = 1;
		} else {
			s = 0.0f;
			t = e / c;
		}
	}

	r_ps = (1 - s) * p_p0 + s * p_p1;
	r_qt = (1 - t) * p_q0 + t * p_q1;
}

real_t Geometry3D::get_closest_distance_between_segments(const Vector3 &p_p0, const Vector3 &p_p1, const Vector3 &p_q0, const Vector3 &p_q1) {
	Vector3 ps;
	Vector3 qt;
	get_closest_points_between_segments(p_p0, p_p1, p_q0, p_q1, ps, qt);
	Vector3 st = qt - ps;
	return st.length();
}

void Geometry3D::MeshData::optimize_vertices() {
	HashMap<int, int> vtx_remap;

	for (MeshData::Face &face : faces) {
		for (int &index : face.indices) {
			if (!vtx_remap.has(index)) {
				int ni = vtx_remap.size();
				vtx_remap[index] = ni;
			}
			index = vtx_remap[index];
		}
	}

	for (MeshData::Edge &edge : edges) {
		int a = edge.vertex_a;
		int b = edge.vertex_b;

		if (!vtx_remap.has(a)) {
			int ni = vtx_remap.size();
			vtx_remap[a] = ni;
		}
		if (!vtx_remap.has(b)) {
			int ni = vtx_remap.size();
			vtx_remap[b] = ni;
		}

		edge.vertex_a = vtx_remap[a];
		edge.vertex_b = vtx_remap[b];
	}

	LocalVector<Vector3> new_vertices;
	new_vertices.resize(vtx_remap.size());

	for (uint32_t i = 0; i < vertices.size(); i++) {
		if (vtx_remap.has(i)) {
			new_vertices[vtx_remap[i]] = vertices[i];
		}
	}
	vertices = new_vertices;
}

struct _FaceClassify {
	struct _Link {
		int face = -1;
		int edge = -1;
		void clear() {
			face = -1;
			edge = -1;
		}
		_Link() {}
	};
	bool valid = false;
	int group = -1;
	_Link links[3];
	Face3 face;
	_FaceClassify() {}
};

/*** GEOMETRY WRAPPER ***/

enum _CellFlags {
	_CELL_SOLID = 1,
	_CELL_EXTERIOR = 2,
	_CELL_STEP_MASK = 0x1C,
	_CELL_STEP_NONE = 0 << 2,
	_CELL_STEP_Y_POS = 1 << 2,
	_CELL_STEP_Y_NEG = 2 << 2,
	_CELL_STEP_X_POS = 3 << 2,
	_CELL_STEP_X_NEG = 4 << 2,
	_CELL_STEP_Z_POS = 5 << 2,
	_CELL_STEP_Z_NEG = 6 << 2,
	_CELL_STEP_DONE = 7 << 2,
	_CELL_PREV_MASK = 0xE0,
	_CELL_PREV_NONE = 0 << 5,
	_CELL_PREV_Y_POS = 1 << 5,
	_CELL_PREV_Y_NEG = 2 << 5,
	_CELL_PREV_X_POS = 3 << 5,
	_CELL_PREV_X_NEG = 4 << 5,
	_CELL_PREV_Z_POS = 5 << 5,
	_CELL_PREV_Z_NEG = 6 << 5,
	_CELL_PREV_FIRST = 7 << 5,
};

static inline void _plot_face(uint8_t ***p_cell_status, int x, int y, int z, int len_x, int len_y, int len_z, const Vector3 &voxelsize, const Face3 &p_face) {
	AABB aabb(Vector3(x, y, z), Vector3(len_x, len_y, len_z));
	aabb.position = aabb.position * voxelsize;
	aabb.size = aabb.size * voxelsize;

	if (!p_face.intersects_aabb(aabb)) {
		return;
	}

	if (len_x == 1 && len_y == 1 && len_z == 1) {
		p_cell_status[x][y][z] = _CELL_SOLID;
		return;
	}

	int div_x = len_x > 1 ? 2 : 1;
	int div_y = len_y > 1 ? 2 : 1;
	int div_z = len_z > 1 ? 2 : 1;

#define SPLIT_DIV(m_i, m_div, m_v, m_len_v, m_new_v, m_new_len_v) \
	if (m_div == 1) {                                             \
		m_new_v = m_v;                                            \
		m_new_len_v = 1;                                          \
	} else if (m_i == 0) {                                        \
		m_new_v = m_v;                                            \
		m_new_len_v = m_len_v / 2;                                \
	} else {                                                      \
		m_new_v = m_v + m_len_v / 2;                              \
		m_new_len_v = m_len_v - m_len_v / 2;                      \
	}

	int new_x;
	int new_len_x;
	int new_y;
	int new_len_y;
	int new_z;
	int new_len_z;

	for (int i = 0; i < div_x; i++) {
		SPLIT_DIV(i, div_x, x, len_x, new_x, new_len_x);

		for (int j = 0; j < div_y; j++) {
			SPLIT_DIV(j, div_y, y, len_y, new_y, new_len_y);

			for (int k = 0; k < div_z; k++) {
				SPLIT_DIV(k, div_z, z, len_z, new_z, new_len_z);

				_plot_face(p_cell_status, new_x, new_y, new_z, new_len_x, new_len_y, new_len_z, voxelsize, p_face);
			}
		}
	}

#undef SPLIT_DIV
}

static inline void _mark_outside(uint8_t ***p_cell_status, int x, int y, int z, int len_x, int len_y, int len_z) {
	if (p_cell_status[x][y][z] & 3) {
		return; // Nothing to do, already used and/or visited.
	}

	p_cell_status[x][y][z] = _CELL_PREV_FIRST;

	while (true) {
		uint8_t &c = p_cell_status[x][y][z];

		if ((c & _CELL_STEP_MASK) == _CELL_STEP_NONE) {
			// Haven't been in here, mark as outside.
			p_cell_status[x][y][z] |= _CELL_EXTERIOR;
		}

		if ((c & _CELL_STEP_MASK) != _CELL_STEP_DONE) {
			// If not done, increase step.
			c += 1 << 2;
		}

		if ((c & _CELL_STEP_MASK) == _CELL_STEP_DONE) {
			// Go back.
			switch (c & _CELL_PREV_MASK) {
				case _CELL_PREV_FIRST: {
					return;
				} break;
				case _CELL_PREV_Y_POS: {
					y++;
					ERR_FAIL_COND(y >= len_y);
				} break;
				case _CELL_PREV_Y_NEG: {
					y--;
					ERR_FAIL_COND(y < 0);
				} break;
				case _CELL_PREV_X_POS: {
					x++;
					ERR_FAIL_COND(x >= len_x);
				} break;
				case _CELL_PREV_X_NEG: {
					x--;
					ERR_FAIL_COND(x < 0);
				} break;
				case _CELL_PREV_Z_POS: {
					z++;
					ERR_FAIL_COND(z >= len_z);
				} break;
				case _CELL_PREV_Z_NEG: {
					z--;
					ERR_FAIL_COND(z < 0);
				} break;
				default: {
					ERR_FAIL();
				}
			}
			continue;
		}

		int next_x = x;
		int next_y = y;
		int next_z = z;
		uint8_t prev = 0;

		switch (c & _CELL_STEP_MASK) {
			case _CELL_STEP_Y_POS: {
				next_y++;
				prev = _CELL_PREV_Y_NEG;
			} break;
			case _CELL_STEP_Y_NEG: {
				next_y--;
				prev = _CELL_PREV_Y_POS;
			} break;
			case _CELL_STEP_X_POS: {
				next_x++;
				prev = _CELL_PREV_X_NEG;
			} break;
			case _CELL_STEP_X_NEG: {
				next_x--;
				prev = _CELL_PREV_X_POS;
			} break;
			case _CELL_STEP_Z_POS: {
				next_z++;
				prev = _CELL_PREV_Z_NEG;
			} break;
			case _CELL_STEP_Z_NEG: {
				next_z--;
				prev = _CELL_PREV_Z_POS;
			} break;
			default:
				ERR_FAIL();
		}

		if (next_x < 0 || next_x >= len_x) {
			continue;
		}
		if (next_y < 0 || next_y >= len_y) {
			continue;
		}
		if (next_z < 0 || next_z >= len_z) {
			continue;
		}

		if (p_cell_status[next_x][next_y][next_z] & 3) {
			continue;
		}

		x = next_x;
		y = next_y;
		z = next_z;
		p_cell_status[x][y][z] |= prev;
	}
}

static inline void _build_faces(uint8_t ***p_cell_status, int x, int y, int z, int len_x, int len_y, int len_z, Vector<Face3> &p_faces) {
	ERR_FAIL_INDEX(x, len_x);
	ERR_FAIL_INDEX(y, len_y);
	ERR_FAIL_INDEX(z, len_z);

	if (p_cell_status[x][y][z] & _CELL_EXTERIOR) {
		return;
	}

#define vert(m_idx) Vector3(((m_idx) & 4) >> 2, ((m_idx) & 2) >> 1, (m_idx) & 1)

	static const uint8_t indices[6][4] = {
		{ 7, 6, 4, 5 },
		{ 7, 3, 2, 6 },
		{ 7, 5, 1, 3 },
		{ 0, 2, 3, 1 },
		{ 0, 1, 5, 4 },
		{ 0, 4, 6, 2 },

	};

	for (int i = 0; i < 6; i++) {
		Vector3 face_points[4];
		int disp_x = x + ((i % 3) == 0 ? ((i < 3) ? 1 : -1) : 0);
		int disp_y = y + (((i - 1) % 3) == 0 ? ((i < 3) ? 1 : -1) : 0);
		int disp_z = z + (((i - 2) % 3) == 0 ? ((i < 3) ? 1 : -1) : 0);

		bool plot = false;

		if (disp_x < 0 || disp_x >= len_x) {
			plot = true;
		}
		if (disp_y < 0 || disp_y >= len_y) {
			plot = true;
		}
		if (disp_z < 0 || disp_z >= len_z) {
			plot = true;
		}

		if (!plot && (p_cell_status[disp_x][disp_y][disp_z] & _CELL_EXTERIOR)) {
			plot = true;
		}

		if (!plot) {
			continue;
		}

		for (int j = 0; j < 4; j++) {
			face_points[j] = vert(indices[i][j]) + Vector3(x, y, z);
		}

		p_faces.push_back(
				Face3(
						face_points[0],
						face_points[1],
						face_points[2]));

		p_faces.push_back(
				Face3(
						face_points[2],
						face_points[3],
						face_points[0]));
	}
}

// Create a "wrap" that encloses the given geometry.
Vector<Face3> Geometry3D::wrap_geometry(const Vector<Face3> &p_array, real_t *r_error) {
	int face_count = p_array.size();
	const Face3 *faces = p_array.ptr();
	constexpr double min_size = 1.0;
	constexpr int max_length = 20;

	AABB global_aabb;

	for (int i = 0; i < face_count; i++) {
		if (i == 0) {
			global_aabb = faces[i].get_aabb();
		} else {
			global_aabb.merge_with(faces[i].get_aabb());
		}
	}

	global_aabb.grow_by(0.01f); // Avoid numerical error.

	// Determine amount of cells in grid axis.
	int div_x, div_y, div_z;

	if (global_aabb.size.x / min_size < max_length) {
		div_x = (int)(global_aabb.size.x / min_size) + 1;
	} else {
		div_x = max_length;
	}

	if (global_aabb.size.y / min_size < max_length) {
		div_y = (int)(global_aabb.size.y / min_size) + 1;
	} else {
		div_y = max_length;
	}

	if (global_aabb.size.z / min_size < max_length) {
		div_z = (int)(global_aabb.size.z / min_size) + 1;
	} else {
		div_z = max_length;
	}

	Vector3 voxelsize = global_aabb.size;
	voxelsize.x /= div_x;
	voxelsize.y /= div_y;
	voxelsize.z /= div_z;

	// Create and initialize cells to zero.

	uint8_t ***cell_status = memnew_arr(uint8_t **, div_x);
	for (int i = 0; i < div_x; i++) {
		cell_status[i] = memnew_arr(uint8_t *, div_y);

		for (int j = 0; j < div_y; j++) {
			cell_status[i][j] = memnew_arr(uint8_t, div_z);

			for (int k = 0; k < div_z; k++) {
				cell_status[i][j][k] = 0;
			}
		}
	}

	// Plot faces into cells.

	for (int i = 0; i < face_count; i++) {
		Face3 f = faces[i];
		for (int j = 0; j < 3; j++) {
			f.vertex[j] -= global_aabb.position;
		}
		_plot_face(cell_status, 0, 0, 0, div_x, div_y, div_z, voxelsize, f);
	}

	// Determine which cells connect to the outside by traversing the outside and recursively flood-fill marking.

	for (int i = 0; i < div_x; i++) {
		for (int j = 0; j < div_y; j++) {
			_mark_outside(cell_status, i, j, 0, div_x, div_y, div_z);
			_mark_outside(cell_status, i, j, div_z - 1, div_x, div_y, div_z);
		}
	}

	for (int i = 0; i < div_z; i++) {
		for (int j = 0; j < div_y; j++) {
			_mark_outside(cell_status, 0, j, i, div_x, div_y, div_z);
			_mark_outside(cell_status, div_x - 1, j, i, div_x, div_y, div_z);
		}
	}

	for (int i = 0; i < div_x; i++) {
		for (int j = 0; j < div_z; j++) {
			_mark_outside(cell_status, i, 0, j, div_x, div_y, div_z);
			_mark_outside(cell_status, i, div_y - 1, j, div_x, div_y, div_z);
		}
	}

	// Build faces for the inside-outside cell divisors.

	Vector<Face3> wrapped_faces;

	for (int i = 0; i < div_x; i++) {
		for (int j = 0; j < div_y; j++) {
			for (int k = 0; k < div_z; k++) {
				_build_faces(cell_status, i, j, k, div_x, div_y, div_z, wrapped_faces);
			}
		}
	}

	// Transform face vertices to global coords.

	int wrapped_faces_count = wrapped_faces.size();
	Face3 *wrapped_faces_ptr = wrapped_faces.ptrw();

	for (int i = 0; i < wrapped_faces_count; i++) {
		for (int j = 0; j < 3; j++) {
			Vector3 &v = wrapped_faces_ptr[i].vertex[j];
			v = v * voxelsize;
			v += global_aabb.position;
		}
	}

	// clean up grid

	for (int i = 0; i < div_x; i++) {
		for (int j = 0; j < div_y; j++) {
			memdelete_arr(cell_status[i][j]);
		}

		memdelete_arr(cell_status[i]);
	}

	memdelete_arr(cell_status);
	if (r_error) {
		*r_error = voxelsize.length();
	}

	return wrapped_faces;
}

Geometry3D::MeshData Geometry3D::build_convex_mesh(const Vector<Plane> &p_planes) {
	MeshData mesh;

#define SUBPLANE_SIZE 1024.0

	real_t subplane_size = 1024.0; // Should compute this from the actual plane.
	for (int i = 0; i < p_planes.size(); i++) {
		Plane p = p_planes[i];

		Vector3 ref = Vector3(0.0, 1.0, 0.0);

		if (Math::abs(p.normal.dot(ref)) > 0.95f) {
			ref = Vector3(0.0, 0.0, 1.0); // Change axis.
		}

		Vector3 right = p.normal.cross(ref).normalized();
		Vector3 up = p.normal.cross(right).normalized();

		Vector3 center = p.get_center();

		// make a quad clockwise
		LocalVector<Vector3> vertices = {
			center - up * subplane_size + right * subplane_size,
			center - up * subplane_size - right * subplane_size,
			center + up * subplane_size - right * subplane_size,
			center + up * subplane_size + right * subplane_size
		};

		for (int j = 0; j < p_planes.size(); j++) {
			if (j == i) {
				continue;
			}

			LocalVector<Vector3> new_vertices;
			Plane clip = p_planes[j];

			if (clip.normal.dot(p.normal) > 0.95f) {
				continue;
			}

			if (vertices.size() < 3) {
				break;
			}

			for (uint32_t k = 0; k < vertices.size(); k++) {
				int k_n = (k + 1) % vertices.size();

				Vector3 edge0_A = vertices[k];
				Vector3 edge1_A = vertices[k_n];

				real_t dist0 = clip.distance_to(edge0_A);
				real_t dist1 = clip.distance_to(edge1_A);

				if (dist0 <= 0) { // Behind plane.

					new_vertices.push_back(vertices[k]);
				}

				// Check for different sides and non coplanar.
				if ((dist0 * dist1) < 0) {
					// Calculate intersection.
					Vector3 rel = edge1_A - edge0_A;

					real_t den = clip.normal.dot(rel);
					if (Math::is_zero_approx(den)) {
						continue; // Point too short.
					}

					real_t dist = -(clip.normal.dot(edge0_A) - clip.d) / den;
					Vector3 inters = edge0_A + rel * dist;
					new_vertices.push_back(inters);
				}
			}

			vertices = new_vertices;
		}

		if (vertices.size() < 3) {
			continue;
		}

		// Result is a clockwise face.

		MeshData::Face face;

		// Add face indices.
		for (const Vector3 &vertex : vertices) {
			int idx = -1;
			for (uint32_t k = 0; k < mesh.vertices.size(); k++) {
				if (mesh.vertices[k].distance_to(vertex) < 0.001f) {
					idx = k;
					break;
				}
			}

			if (idx == -1) {
				idx = mesh.vertices.size();
				mesh.vertices.push_back(vertex);
			}

			face.indices.push_back(idx);
		}
		face.plane = p;
		mesh.faces.push_back(face);

		// Add edge.

		for (uint32_t j = 0; j < face.indices.size(); j++) {
			int a = face.indices[j];
			int b = face.indices[(j + 1) % face.indices.size()];

			bool found = false;
			int found_idx = -1;
			for (uint32_t k = 0; k < mesh.edges.size(); k++) {
				if (mesh.edges[k].vertex_a == a && mesh.edges[k].vertex_b == b) {
					found = true;
					found_idx = k;
					break;
				}
				if (mesh.edges[k].vertex_b == a && mesh.edges[k].vertex_a == b) {
					found = true;
					found_idx = k;
					break;
				}
			}

			if (found) {
				mesh.edges[found_idx].face_b = j;
				continue;
			}
			MeshData::Edge edge;
			edge.vertex_a = a;
			edge.vertex_b = b;
			edge.face_a = j;
			edge.face_b = -1;
			mesh.edges.push_back(edge);
		}
	}

	return mesh;
}

Vector<Plane> Geometry3D::build_box_planes(const Vector3 &p_extents) {
	Vector<Plane> planes = {
		Plane(Vector3(1, 0, 0), p_extents.x),
		Plane(Vector3(-1, 0, 0), p_extents.x),
		Plane(Vector3(0, 1, 0), p_extents.y),
		Plane(Vector3(0, -1, 0), p_extents.y),
		Plane(Vector3(0, 0, 1), p_extents.z),
		Plane(Vector3(0, 0, -1), p_extents.z)
	};

	return planes;
}

Vector<Plane> Geometry3D::build_cylinder_planes(real_t p_radius, real_t p_height, int p_sides, Vector3::Axis p_axis) {
	ERR_FAIL_INDEX_V(p_axis, 3, Vector<Plane>());

	Vector<Plane> planes;

	const double sides_step = Math::TAU / p_sides;
	for (int i = 0; i < p_sides; i++) {
		Vector3 normal;
		normal[(p_axis + 1) % 3] = Math::cos(i * sides_step);
		normal[(p_axis + 2) % 3] = Math::sin(i * sides_step);

		planes.push_back(Plane(normal, p_radius));
	}

	Vector3 axis;
	axis[p_axis] = 1.0;

	planes.push_back(Plane(axis, p_height * 0.5f));
	planes.push_back(Plane(-axis, p_height * 0.5f));

	return planes;
}

Vector<Plane> Geometry3D::build_sphere_planes(real_t p_radius, int p_lats, int p_lons, Vector3::Axis p_axis) {
	ERR_FAIL_INDEX_V(p_axis, 3, Vector<Plane>());

	Vector<Plane> planes;

	Vector3 axis;
	axis[p_axis] = 1.0;

	Vector3 axis_neg;
	axis_neg[(p_axis + 1) % 3] = 1.0;
	axis_neg[(p_axis + 2) % 3] = 1.0;
	axis_neg[p_axis] = -1.0;

	const double lon_step = Math::TAU / p_lons;
	for (int i = 0; i < p_lons; i++) {
		Vector3 normal;
		normal[(p_axis + 1) % 3] = Math::cos(i * lon_step);
		normal[(p_axis + 2) % 3] = Math::sin(i * lon_step);

		planes.push_back(Plane(normal, p_radius));

		for (int j = 1; j <= p_lats; j++) {
			Vector3 plane_normal = normal.lerp(axis, j / (real_t)p_lats).normalized();
			planes.push_back(Plane(plane_normal, p_radius));
			planes.push_back(Plane(plane_normal * axis_neg, p_radius));
		}
	}

	return planes;
}

Vector<Plane> Geometry3D::build_capsule_planes(real_t p_radius, real_t p_height, int p_sides, int p_lats, Vector3::Axis p_axis) {
	ERR_FAIL_INDEX_V(p_axis, 3, Vector<Plane>());

	Vector<Plane> planes;

	Vector3 axis;
	axis[p_axis] = 1.0;

	Vector3 axis_neg;
	axis_neg[(p_axis + 1) % 3] = 1.0;
	axis_neg[(p_axis + 2) % 3] = 1.0;
	axis_neg[p_axis] = -1.0;

	const double sides_step = Math::TAU / p_sides;
	for (int i = 0; i < p_sides; i++) {
		Vector3 normal;
		normal[(p_axis + 1) % 3] = Math::cos(i * sides_step);
		normal[(p_axis + 2) % 3] = Math::sin(i * sides_step);

		planes.push_back(Plane(normal, p_radius));

		for (int j = 1; j <= p_lats; j++) {
			Vector3 plane_normal = normal.lerp(axis, j / (real_t)p_lats).normalized();
			Vector3 position = axis * p_height * 0.5f + plane_normal * p_radius;
			planes.push_back(Plane(plane_normal, position));
			planes.push_back(Plane(plane_normal * axis_neg, position * axis_neg));
		}
	}

	return planes;
}

Vector<Vector3> Geometry3D::compute_convex_mesh_points(const Plane *p_planes, int p_plane_count) {
	Vector<Vector3> points;

	// Iterate through every unique combination of any three planes.
	for (int i = p_plane_count - 1; i >= 0; i--) {
		for (int j = i - 1; j >= 0; j--) {
			for (int k = j - 1; k >= 0; k--) {
				// Find the point where these planes all cross over (if they
				// do at all).
				Vector3 convex_shape_point;
				if (p_planes[i].intersect_3(p_planes[j], p_planes[k], &convex_shape_point)) {
					// See if any *other* plane excludes this point because it's
					// on the wrong side.
					bool excluded = false;
					for (int n = 0; n < p_plane_count; n++) {
						if (n != i && n != j && n != k) {
							real_t dp = p_planes[n].normal.dot(convex_shape_point);
							if (dp - p_planes[n].d > (real_t)CMP_EPSILON) {
								excluded = true;
								break;
							}
						}
					}

					// Only add the point if it passed all tests.
					if (!excluded) {
						points.push_back(convex_shape_point);
					}
				}
			}
		}
	}

	return points;
}

#define square(m_s) ((m_s) * (m_s))
#define BIG_VAL 1e20

/* dt of 1d function using squared distance */
static void edt(float *f, int stride, int n) {
	float *d = (float *)alloca(sizeof(float) * n + sizeof(int) * n + sizeof(float) * (n + 1));
	int *v = reinterpret_cast<int *>(&(d[n]));
	float *z = reinterpret_cast<float *>(&v[n]);

	int k = 0;
	v[0] = 0;
	z[0] = -BIG_VAL;
	z[1] = +BIG_VAL;
	for (int q = 1; q <= n - 1; q++) {
		float s = ((f[q * stride] + square(q)) - (f[v[k] * stride] + square(v[k]))) / (2 * q - 2 * v[k]);
		while (s <= z[k]) {
			k--;
			s = ((f[q * stride] + square(q)) - (f[v[k] * stride] + square(v[k]))) / (2 * q - 2 * v[k]);
		}
		k++;
		v[k] = q;

		z[k] = s;
		z[k + 1] = +BIG_VAL;
	}

	k = 0;
	for (int q = 0; q <= n - 1; q++) {
		while (z[k + 1] < q) {
			k++;
		}
		d[q] = square(q - v[k]) + f[v[k] * stride];
	}

	for (int i = 0; i < n; i++) {
		f[i * stride] = d[i];
	}
}

#undef square

Vector<uint32_t> Geometry3D::generate_edf(const Vector<bool> &p_voxels, const Vector3i &p_size, bool p_negative) {
	uint32_t float_count = p_size.x * p_size.y * p_size.z;

	ERR_FAIL_COND_V((uint32_t)p_voxels.size() != float_count, Vector<uint32_t>());

	float *work_memory = memnew_arr(float, float_count);
	for (uint32_t i = 0; i < float_count; i++) {
		work_memory[i] = BIG_VAL;
	}

	uint32_t y_mult = p_size.x;
	uint32_t z_mult = y_mult * p_size.y;

	//plot solid cells
	{
		const bool *voxr = p_voxels.ptr();
		for (uint32_t i = 0; i < float_count; i++) {
			bool plot = voxr[i];
			if (p_negative) {
				plot = !plot;
			}
			if (plot) {
				work_memory[i] = 0;
			}
		}
	}

	//process in each direction

	//xy->z

	for (int i = 0; i < p_size.x; i++) {
		for (int j = 0; j < p_size.y; j++) {
			edt(&work_memory[i + j * y_mult], z_mult, p_size.z);
		}
	}

	//xz->y

	for (int i = 0; i < p_size.x; i++) {
		for (int j = 0; j < p_size.z; j++) {
			edt(&work_memory[i + j * z_mult], y_mult, p_size.y);
		}
	}

	//yz->x
	for (int i = 0; i < p_size.y; i++) {
		for (int j = 0; j < p_size.z; j++) {
			edt(&work_memory[i * y_mult + j * z_mult], 1, p_size.x);
		}
	}

	Vector<uint32_t> ret;
	ret.resize(float_count);
	{
		uint32_t *w = ret.ptrw();
		for (uint32_t i = 0; i < float_count; i++) {
			w[i] = uint32_t(Math::sqrt(work_memory[i]));
		}
	}

	memdelete_arr(work_memory);

	return ret;
}

#undef BIG_VAL

Vector<int8_t> Geometry3D::generate_sdf8(const Vector<uint32_t> &p_positive, const Vector<uint32_t> &p_negative) {
	ERR_FAIL_COND_V(p_positive.size() != p_negative.size(), Vector<int8_t>());
	Vector<int8_t> sdf8;
	int s = p_positive.size();
	sdf8.resize(s);

	const uint32_t *rpos = p_positive.ptr();
	const uint32_t *rneg = p_negative.ptr();
	int8_t *wsdf = sdf8.ptrw();
	for (int i = 0; i < s; i++) {
		int32_t diff = int32_t(rpos[i]) - int32_t(rneg[i]);
		wsdf[i] = CLAMP(diff, -128, 127);
	}
	return sdf8;
}

bool Geometry3D::ray_intersects_triangle(const Vector3 &p_from, const Vector3 &p_dir, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2, Vector3 *r_res) {
	Vector3 e1 = p_v1 - p_v0;
	Vector3 e2 = p_v2 - p_v0;
	Vector3 h = p_dir.cross(e2);
	real_t a = e1.dot(h);
	if (Math::is_zero_approx(a)) { // Parallel test.
		return false;
	}

	real_t f = 1.0f / a;

	Vector3 s = p_from - p_v0;
	real_t u = f * s.dot(h);

	if ((u < 0.0f) || (u > 1.0f)) {
		return false;
	}

	Vector3 q = s.cross(e1);

	real_t v = f * p_dir.dot(q);

	if ((v < 0.0f) || (u + v > 1.0f)) {
		return false;
	}

	// At this stage we can compute t to find out where
	// the intersection point is on the line.
	real_t t = f * e2.dot(q);

	if (t > 0.00001f) { // ray intersection
		if (r_res) {
			*r_res = p_from + p_dir * t;
		}
		return true;
	}
	// This means that there is a line intersection but not a ray intersection.
	return false;
}

bool Geometry3D::segment_intersects_triangle(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2, Vector3 *r_res) {
	Vector3 rel = p_to - p_from;
	Vector3 e1 = p_v1 - p_v0;
	Vector3 e2 = p_v2 - p_v0;
	Vector3 h = rel.cross(e2);
	real_t a = e1.dot(h);
	if (Math::is_zero_approx(a)) { // Parallel test.
		return false;
	}

	real_t f = 1.0f / a;

	Vector3 s = p_from - p_v0;
	real_t u = f * s.dot(h);

	if ((u < 0.0f) || (u > 1.0f)) {
		return false;
	}

	Vector3 q = s.cross(e1);

	real_t v = f * rel.dot(q);

	if ((v < 0.0f) || (u + v > 1.0f)) {
		return false;
	}

	// At this stage we can compute t to find out where
	// the intersection point is on the line.
	real_t t = f * e2.dot(q);

	if (t > (real_t)CMP_EPSILON && t <= 1.0f) { // Ray intersection.
		if (r_res) {
			*r_res = p_from + rel * t;
		}
		return true;
	}
	// This means that there is a line intersection but not a ray intersection.
	return false;
}

bool Geometry3D::segment_intersects_sphere(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_sphere_pos, real_t p_sphere_radius, Vector3 *r_res, Vector3 *r_norm) {
	Vector3 sphere_pos = p_sphere_pos - p_from;
	Vector3 rel = (p_to - p_from);
	real_t rel_l = rel.length();
	if (rel_l < (real_t)CMP_EPSILON) {
		return false; // Both points are the same.
	}
	Vector3 normal = rel / rel_l;

	real_t sphere_d = normal.dot(sphere_pos);

	real_t ray_distance = sphere_pos.distance_to(normal * sphere_d);

	if (ray_distance >= p_sphere_radius) {
		return false;
	}

	real_t inters_d2 = p_sphere_radius * p_sphere_radius - ray_distance * ray_distance;
	real_t inters_d = sphere_d;

	if (inters_d2 >= (real_t)CMP_EPSILON) {
		inters_d -= Math::sqrt(inters_d2);
	}

	// Check in segment.
	if (inters_d < 0 || inters_d > rel_l) {
		return false;
	}

	Vector3 result = p_from + normal * inters_d;

	if (r_res) {
		*r_res = result;
	}
	if (r_norm) {
		*r_norm = (result - p_sphere_pos).normalized();
	}

	return true;
}

bool Geometry3D::segment_intersects_cylinder(const Vector3 &p_from, const Vector3 &p_to, real_t p_height, real_t p_radius, Vector3 *r_res, Vector3 *r_norm, int p_cylinder_axis) {
	Vector3 rel = (p_to - p_from);
	real_t rel_l = rel.length();
	if (rel_l < (real_t)CMP_EPSILON) {
		return false; // Both points are the same.
	}

	ERR_FAIL_COND_V(p_cylinder_axis < 0, false);
	ERR_FAIL_COND_V(p_cylinder_axis > 2, false);
	Vector3 cylinder_axis;
	cylinder_axis[p_cylinder_axis] = 1.0f;

	// First check if they are parallel.
	Vector3 normal = (rel / rel_l);
	Vector3 crs = normal.cross(cylinder_axis);
	real_t crs_l = crs.length();

	Vector3 axis_dir;

	if (crs_l < (real_t)CMP_EPSILON) {
		Vector3 side_axis;
		side_axis[(p_cylinder_axis + 1) % 3] = 1.0f; // Any side axis OK.
		axis_dir = side_axis;
	} else {
		axis_dir = crs / crs_l;
	}

	real_t dist = axis_dir.dot(p_from);

	if (dist >= p_radius) {
		return false; // Too far away.
	}

	// Convert to 2D.
	real_t w2 = p_radius * p_radius - dist * dist;
	if (w2 < (real_t)CMP_EPSILON) {
		return false; // Avoid numerical error.
	}
	Size2 size(Math::sqrt(w2), p_height * 0.5f);

	Vector3 side_dir = axis_dir.cross(cylinder_axis).normalized();

	Vector2 from2D(side_dir.dot(p_from), p_from[p_cylinder_axis]);
	Vector2 to2D(side_dir.dot(p_to), p_to[p_cylinder_axis]);

	real_t min = 0, max = 1;

	int axis = -1;

	for (int i = 0; i < 2; i++) {
		real_t seg_from = from2D[i];
		real_t seg_to = to2D[i];
		real_t box_begin = -size[i];
		real_t box_end = size[i];
		real_t cmin, cmax;

		if (seg_from < seg_to) {
			if (seg_from > box_end || seg_to < box_begin) {
				return false;
			}
			real_t length = seg_to - seg_from;
			cmin = (seg_from < box_begin) ? ((box_begin - seg_from) / length) : 0;
			cmax = (seg_to > box_end) ? ((box_end - seg_from) / length) : 1;

		} else {
			if (seg_to > box_end || seg_from < box_begin) {
				return false;
			}
			real_t length = seg_to - seg_from;
			cmin = (seg_from > box_end) ? (box_end - seg_from) / length : 0;
			cmax = (seg_to < box_begin) ? (box_begin - seg_from) / length : 1;
		}

		if (cmin > min) {
			min = cmin;
			axis = i;
		}
		if (cmax < max) {
			max = cmax;
		}
		if (max < min) {
			return false;
		}
	}

	// Convert to 3D again.
	Vector3 result = p_from + (rel * min);
	Vector3 res_normal = result;

	if (axis == 0) {
		res_normal[p_cylinder_axis] = 0;
	} else {
		int axis_side = (p_cylinder_axis + 1) % 3;
		res_normal[axis_side] = 0;
		axis_side = (axis_side + 1) % 3;
		res_normal[axis_side] = 0;
	}

	res_normal.normalize();

	if (r_res) {
		*r_res = result;
	}
	if (r_norm) {
		*r_norm = res_normal;
	}

	return true;
}

bool Geometry3D::segment_intersects_convex(const Vector3 &p_from, const Vector3 &p_to, const Plane *p_planes, int p_plane_count, Vector3 *r_res, Vector3 *r_norm) {
	real_t min = -1e20;
	real_t max = 1e20;

	Vector3 rel = p_to - p_from;
	real_t rel_l = rel.length();

	if (rel_l < (real_t)CMP_EPSILON) {
		return false;
	}

	Vector3 dir = rel / rel_l;

	int min_index = -1;

	for (int i = 0; i < p_plane_count; i++) {
		const Plane &p = p_planes[i];

		real_t den = p.normal.dot(dir);

		if (Math::abs(den) <= (real_t)CMP_EPSILON) {
			if (p.is_point_over(p_from)) {
				// Separating plane.
				return false;
			}
			continue; // Ignore parallel plane.
		}

		real_t dist = -p.distance_to(p_from) / den;

		if (den > 0) {
			// Backwards facing plane.
			if (dist < max) {
				max = dist;
			}
		} else {
			// Front facing plane.
			if (dist > min) {
				min = dist;
				min_index = i;
			}
		}
	}

	if (max <= min || min < 0 || min > rel_l || min_index == -1) { // Exit conditions.
		return false; // No intersection.
	}

	if (r_res) {
		*r_res = p_from + dir * min;
	}
	if (r_norm) {
		*r_norm = p_planes[min_index].normal;
	}

	return true;
}

Vector3 Geometry3D::get_closest_point_to_segment(const Vector3 &p_point, const Vector3 &p_segment_a, const Vector3 &p_segment_b) {
	Vector3 p = p_point - p_segment_a;
	Vector3 n = p_segment_b - p_segment_a;
	real_t l2 = n.length_squared();
	if (l2 < 1e-20f) {
		return p_segment_a; // Both points are the same, just give any.
	}

	real_t d = n.dot(p) / l2;

	if (d <= 0.0f) {
		return p_segment_a; // Before first point.
	} else if (d >= 1.0f) {
		return p_segment_b; // After first point.
	}
	return p_segment_a + n * d; // Inside.
}

Vector3 Geometry3D::get_closest_point_to_segment_uncapped(const Vector3 &p_point, const Vector3 &p_segment_a, const Vector3 &p_segment_b) {
	Vector3 p = p_point - p_segment_a;
	Vector3 n = p_segment_b - p_segment_a;
	real_t l2 = n.length_squared();
	if (l2 < 1e-20f) {
		return p_segment_a; // Both points are the same, just give any.
	}

	real_t d = n.dot(p) / l2;

	return p_segment_a + n * d; // Inside.
}

bool Geometry3D::triangle_sphere_intersection_test(const Vector3 &p_triangle_a, const Vector3 &p_triangle_b, const Vector3 &p_triangle_c, const Vector3 &p_normal, const Vector3 &p_sphere_pos, real_t p_sphere_radius, Vector3 &r_triangle_contact, Vector3 &r_sphere_contact) {
	real_t d = p_normal.dot(p_sphere_pos) - p_normal.dot(p_triangle_a);

	if (d > p_sphere_radius || d < -p_sphere_radius) {
		// Not touching the plane of the face, return.
		return false;
	}

	Vector3 contact = p_sphere_pos - (p_normal * d);

	/** 2nd) TEST INSIDE TRIANGLE **/

	if (Geometry3D::point_in_projected_triangle(contact, p_triangle_a, p_triangle_b, p_triangle_c)) {
		r_triangle_contact = contact;
		r_sphere_contact = p_sphere_pos - p_normal * p_sphere_radius;
		//printf("solved inside triangle\n");
		return true;
	}

	/** 3rd TEST INSIDE EDGE CYLINDERS **/

	const Vector3 verts[4] = { p_triangle_a, p_triangle_b, p_triangle_c, p_triangle_a }; // for() friendly

	for (int i = 0; i < 3; i++) {
		// Check edge cylinder.

		Vector3 n1 = verts[i] - verts[i + 1];
		Vector3 n2 = p_sphere_pos - verts[i + 1];

		///@TODO Maybe discard by range here to make the algorithm quicker.

		// Check point within cylinder radius.
		Vector3 axis = n1.cross(n2).cross(n1);
		axis.normalize();

		real_t ad = axis.dot(n2);

		if (Math::abs(ad) > p_sphere_radius) {
			// No chance with this edge, too far away.
			continue;
		}

		// Check point within edge capsule cylinder.
		/** 4th TEST INSIDE EDGE POINTS **/

		real_t sphere_at = n1.dot(n2);

		if (sphere_at >= 0 && sphere_at < n1.dot(n1)) {
			r_triangle_contact = p_sphere_pos - axis * (axis.dot(n2));
			r_sphere_contact = p_sphere_pos - axis * p_sphere_radius;
			// Point inside here.
			return true;
		}

		real_t r2 = p_sphere_radius * p_sphere_radius;

		if (n2.length_squared() < r2) {
			Vector3 n = (p_sphere_pos - verts[i + 1]).normalized();

			r_triangle_contact = verts[i + 1];
			r_sphere_contact = p_sphere_pos - n * p_sphere_radius;
			return true;
		}

		if (n2.distance_squared_to(n1) < r2) {
			Vector3 n = (p_sphere_pos - verts[i]).normalized();

			r_triangle_contact = verts[i];
			r_sphere_contact = p_sphere_pos - n * p_sphere_radius;
			return true;
		}

		break; // It's pointless to continue at this point, so save some CPU cycles.
	}

	return false;
}

#ifndef DISABLE_DEPRECATED
Vector3 Geometry3D::get_closest_point_to_segment(const Vector3 &p_point, const Vector3 *p_segment) {
	return get_closest_point_to_segment(p_point, p_segment[0], p_segment[1]);
}

Vector3 Geometry3D::get_closest_point_to_segment_uncapped(const Vector3 &p_point, const Vector3 *p_segment) {
	return get_closest_point_to_segment_uncapped(p_point, p_segment[0], p_segment[1]);
}

bool Geometry3D::triangle_sphere_intersection_test(const Vector3 *p_triangle, const Vector3 &p_normal, const Vector3 &p_sphere_pos, real_t p_sphere_radius, Vector3 &r_triangle_contact, Vector3 &r_sphere_contact) {
	return triangle_sphere_intersection_test(p_triangle[0], p_triangle[1], p_triangle[2], p_normal, p_sphere_pos, p_sphere_radius, r_triangle_contact, r_sphere_contact);
}
#endif // DISABLE_DEPRECATED

bool Geometry3D::point_in_projected_triangle(const Vector3 &p_point, const Vector3 &p_v1, const Vector3 &p_v2, const Vector3 &p_v3) {
	Vector3 face_n = (p_v1 - p_v3).cross(p_v1 - p_v2);

	Vector3 n1 = (p_point - p_v3).cross(p_point - p_v2);

	if (face_n.dot(n1) < 0) {
		return false;
	}

	Vector3 n2 = (p_v1 - p_v3).cross(p_v1 - p_point);

	if (face_n.dot(n2) < 0) {
		return false;
	}

	Vector3 n3 = (p_v1 - p_point).cross(p_v1 - p_v2);

	if (face_n.dot(n3) < 0) {
		return false;
	}

	return true;
}

Vector<Vector3> Geometry3D::clip_polygon(const Vector<Vector3> &p_polygon, const Plane &p_plane) {
	enum LocationCache {
		LOC_INSIDE = 1,
		LOC_BOUNDARY = 0,
		LOC_OUTSIDE = -1,
	};

	if (p_polygon.is_empty()) {
		return p_polygon;
	}

	int *location_cache = (int *)alloca(sizeof(int) * p_polygon.size());
	int inside_count = 0;
	int outside_count = 0;

	for (int a = 0; a < p_polygon.size(); a++) {
		real_t dist = p_plane.distance_to(p_polygon[a]);
		if (dist < (real_t)-CMP_POINT_IN_PLANE_EPSILON) {
			location_cache[a] = LOC_INSIDE;
			inside_count++;
		} else if (dist > (real_t)CMP_POINT_IN_PLANE_EPSILON) {
			location_cache[a] = LOC_OUTSIDE;
			outside_count++;
		} else {
			location_cache[a] = LOC_BOUNDARY;
		}
	}

	if (outside_count == 0) {
		return p_polygon; // No changes.
	} else if (inside_count == 0) {
		return Vector<Vector3>(); // Empty.
	}

	long previous = p_polygon.size() - 1;
	Vector<Vector3> clipped;

	for (int index = 0; index < p_polygon.size(); index++) {
		int loc = location_cache[index];
		if (loc == LOC_OUTSIDE) {
			if (location_cache[previous] == LOC_INSIDE) {
				const Vector3 &v1 = p_polygon[previous];
				const Vector3 &v2 = p_polygon[index];

				Vector3 segment = v1 - v2;
				real_t den = p_plane.normal.dot(segment);
				real_t dist = p_plane.distance_to(v1) / den;
				dist = -dist;
				clipped.push_back(v1 + segment * dist);
			}
		} else {
			const Vector3 &v1 = p_polygon[index];
			if ((loc == LOC_INSIDE) && (location_cache[previous] == LOC_OUTSIDE)) {
				const Vector3 &v2 = p_polygon[previous];
				Vector3 segment = v1 - v2;
				real_t den = p_plane.normal.dot(segment);
				real_t dist = p_plane.distance_to(v1) / den;
				dist = -dist;
				clipped.push_back(v1 + segment * dist);
			}

			clipped.push_back(v1);
		}

		previous = index;
	}

	return clipped;
}

Vector<int32_t> Geometry3D::tetrahedralize_delaunay(const Vector<Vector3> &p_points) {
	Vector<Delaunay3D::OutputSimplex> tetr = Delaunay3D::tetrahedralize(p_points);
	Vector<int32_t> tetrahedrons;

	tetrahedrons.resize(4 * tetr.size());
	int32_t *ptr = tetrahedrons.ptrw();
	for (int i = 0; i < tetr.size(); i++) {
		*ptr++ = tetr[i].points[0];
		*ptr++ = tetr[i].points[1];
		*ptr++ = tetr[i].points[2];
		*ptr++ = tetr[i].points[3];
	}
	return tetrahedrons;
}

bool Geometry3D::planeBoxOverlap(Vector3 p_normal, real_t p_d, Vector3 p_max_box) {
	int q;
	Vector3 vmin, vmax;
	for (q = 0; q <= 2; q++) {
		if (p_normal[q] > 0.0f) {
			vmin[q] = -p_max_box[q];
			vmax[q] = p_max_box[q];
		} else {
			vmin[q] = p_max_box[q];
			vmax[q] = -p_max_box[q];
		}
	}
	if (p_normal.dot(vmin) + p_d > 0.0f) {
		return false;
	}
	if (p_normal.dot(vmax) + p_d >= 0.0f) {
		return true;
	}

	return false;
}

#define FIND_MIN_MAX(m_x0, m_x1, m_x2) \
	min = max = m_x0;                  \
	if (m_x1 < min) {                  \
		min = m_x1;                    \
	}                                  \
	if (m_x1 > max) {                  \
		max = m_x1;                    \
	}                                  \
	if (m_x2 < min) {                  \
		min = m_x2;                    \
	}                                  \
	if (m_x2 > max) {                  \
		max = m_x2;                    \
	}

/*======================== X-tests ========================*/
#define AXISTEST_X01(m_a, m_b, m_fa, m_fb)                     \
	p0 = m_a * v0.y - m_b * v0.z;                              \
	p2 = m_a * v2.y - m_b * v2.z;                              \
	if (p0 < p2) {                                             \
		min = p0;                                              \
		max = p2;                                              \
	} else {                                                   \
		min = p2;                                              \
		max = p0;                                              \
	}                                                          \
	rad = m_fa * p_box_half_size.y + m_fb * p_box_half_size.z; \
	if (min > rad || max < -rad) {                             \
		return false;                                          \
	}

#define AXISTEST_X2(m_a, m_b, m_fa, m_fb)                      \
	p0 = m_a * v0.y - m_b * v0.z;                              \
	p1 = m_a * v1.y - m_b * v1.z;                              \
	if (p0 < p1) {                                             \
		min = p0;                                              \
		max = p1;                                              \
	} else {                                                   \
		min = p1;                                              \
		max = p0;                                              \
	}                                                          \
	rad = m_fa * p_box_half_size.y + m_fb * p_box_half_size.z; \
	if (min > rad || max < -rad) {                             \
		return false;                                          \
	}

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(m_a, m_b, m_fa, m_fb)                     \
	p0 = -m_a * v0.x + m_b * v0.z;                             \
	p2 = -m_a * v2.x + m_b * v2.z;                             \
	if (p0 < p2) {                                             \
		min = p0;                                              \
		max = p2;                                              \
	} else {                                                   \
		min = p2;                                              \
		max = p0;                                              \
	}                                                          \
	rad = m_fa * p_box_half_size.x + m_fb * p_box_half_size.z; \
	if (min > rad || max < -rad) {                             \
		return false;                                          \
	}

#define AXISTEST_Y1(m_a, m_b, m_fa, m_fb)                      \
	p0 = -m_a * v0.x + m_b * v0.z;                             \
	p1 = -m_a * v1.x + m_b * v1.z;                             \
	if (p0 < p1) {                                             \
		min = p0;                                              \
		max = p1;                                              \
	} else {                                                   \
		min = p1;                                              \
		max = p0;                                              \
	}                                                          \
	rad = m_fa * p_box_half_size.x + m_fb * p_box_half_size.z; \
	if (min > rad || max < -rad) {                             \
		return false;                                          \
	}

/*======================== Z-tests ========================*/
#define AXISTEST_Z12(m_a, m_b, m_fa, m_fb)                     \
	p1 = m_a * v1.x - m_b * v1.y;                              \
	p2 = m_a * v2.x - m_b * v2.y;                              \
	if (p2 < p1) {                                             \
		min = p2;                                              \
		max = p1;                                              \
	} else {                                                   \
		min = p1;                                              \
		max = p2;                                              \
	}                                                          \
	rad = m_fa * p_box_half_size.x + m_fb * p_box_half_size.y; \
	if (min > rad || max < -rad) {                             \
		return false;                                          \
	}

#define AXISTEST_Z0(m_a, m_b, m_fa, m_fb)                      \
	p0 = m_a * v0.x - m_b * v0.y;                              \
	p1 = m_a * v1.x - m_b * v1.y;                              \
	if (p0 < p1) {                                             \
		min = p0;                                              \
		max = p1;                                              \
	} else {                                                   \
		min = p1;                                              \
		max = p0;                                              \
	}                                                          \
	rad = m_fa * p_box_half_size.x + m_fb * p_box_half_size.y; \
	if (min > rad || max < -rad) {                             \
		return false;                                          \
	}

bool Geometry3D::triangle_box_overlap(const Vector3 &p_box_center, const Vector3 p_box_half_size, const Vector3 *p_tri_verts) {
	/*    use separating axis theorem to test overlap between triangle and box */
	/*    need to test for overlap in these directions: */
	/*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */
	/*       we do not even need to test these) */
	/*    2) normal of the triangle */
	/*    3) crossproduct(edge from tri, {x,y,z}-directin) */
	/*       this gives 3x3=9 more tests */
	real_t min;
	real_t max;
	real_t p0;
	real_t p1;
	real_t p2;
	real_t rad;
	real_t fex;
	real_t fey;
	real_t fez;

	/* This is the fastest branch on Sun */
	/* move everything so that the boxcenter is in (0,0,0) */

	const Vector3 v0 = p_tri_verts[0] - p_box_center;
	const Vector3 v1 = p_tri_verts[1] - p_box_center;
	const Vector3 v2 = p_tri_verts[2] - p_box_center;

	/* compute triangle edges */
	const Vector3 e0 = v1 - v0; /* tri edge 0 */
	const Vector3 e1 = v2 - v1; /* tri edge 1 */
	const Vector3 e2 = v0 - v2; /* tri edge 2 */

	/* Bullet 3:  */
	/*  test the 9 tests first (this was faster) */
	fex = Math::abs(e0.x);
	fey = Math::abs(e0.y);
	fez = Math::abs(e0.z);
	AXISTEST_X01(e0.z, e0.y, fez, fey);
	AXISTEST_Y02(e0.z, e0.x, fez, fex);
	AXISTEST_Z12(e0.y, e0.x, fey, fex);

	fex = Math::abs(e1.x);
	fey = Math::abs(e1.y);
	fez = Math::abs(e1.z);
	AXISTEST_X01(e1.z, e1.y, fez, fey);
	AXISTEST_Y02(e1.z, e1.x, fez, fex);
	AXISTEST_Z0(e1.y, e1.x, fey, fex);

	fex = Math::abs(e2.x);
	fey = Math::abs(e2.y);
	fez = Math::abs(e2.z);
	AXISTEST_X2(e2.z, e2.y, fez, fey);
	AXISTEST_Y1(e2.z, e2.x, fez, fex);
	AXISTEST_Z12(e2.y, e2.x, fey, fex);

	/* Bullet 1: */
	/*  first test overlap in the {x,y,z}-directions */
	/*  find min, max of the triangle each direction, and test for overlap in */
	/*  that direction -- this is equivalent to testing a minimal AABB around */
	/*  the triangle against the AABB */

	/* test in X-direction */
	FIND_MIN_MAX(v0.x, v1.x, v2.x);
	if (min > p_box_half_size.x || max < -p_box_half_size.x) {
		return false;
	}

	/* test in Y-direction */
	FIND_MIN_MAX(v0.y, v1.y, v2.y);
	if (min > p_box_half_size.y || max < -p_box_half_size.y) {
		return false;
	}

	/* test in Z-direction */
	FIND_MIN_MAX(v0.z, v1.z, v2.z);
	if (min > p_box_half_size.z || max < -p_box_half_size.z) {
		return false;
	}

	/* Bullet 2: */
	/*  test if the box intersects the plane of the triangle */
	/*  compute plane equation of triangle: normal*x+d=0 */
	const Vector3 normal = e0.cross(e1);
	const real_t d = -normal.dot(v0); /* plane eq: normal.x+d=0 */
	return planeBoxOverlap(normal, d, p_box_half_size); /* if true, box and triangle overlaps */
}

Vector3 Geometry3D::triangle_get_barycentric_coords(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, const Vector3 &p_pos) {
	const Vector3 v0 = p_b - p_a;
	const Vector3 v1 = p_c - p_a;
	const Vector3 v2 = p_pos - p_a;

	const real_t d00 = v0.dot(v0);
	const real_t d01 = v0.dot(v1);
	const real_t d11 = v1.dot(v1);
	const real_t d20 = v2.dot(v0);
	const real_t d21 = v2.dot(v1);
	const real_t denom = (d00 * d11 - d01 * d01);
	if (denom == 0) {
		return Vector3(); //invalid triangle, return empty
	}
	const real_t v = (d11 * d20 - d01 * d21) / denom;
	const real_t w = (d00 * d21 - d01 * d20) / denom;
	const real_t u = 1.0f - v - w;
	return Vector3(u, v, w);
}

Color Geometry3D::tetrahedron_get_barycentric_coords(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, const Vector3 &p_d, const Vector3 &p_pos) {
	const Vector3 vap = p_pos - p_a;
	const Vector3 vbp = p_pos - p_b;

	const Vector3 vab = p_b - p_a;
	const Vector3 vac = p_c - p_a;
	const Vector3 vad = p_d - p_a;

	const Vector3 vbc = p_c - p_b;
	const Vector3 vbd = p_d - p_b;
	// ScTP computes the scalar triple product
#define STP(m_a, m_b, m_c) ((m_a).dot((m_b).cross((m_c))))
	const real_t va6 = STP(vbp, vbd, vbc);
	const real_t vb6 = STP(vap, vac, vad);
	const real_t vc6 = STP(vap, vad, vab);
	const real_t vd6 = STP(vap, vab, vac);
	const real_t v6 = 1 / STP(vab, vac, vad);
	return Color(va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6);
#undef STP
}

Vector3 Geometry3D::octahedron_map_decode(const Vector2 &p_uv) {
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	const Vector2 f = p_uv * 2.0f - Vector2(1.0f, 1.0f);
	Vector3 n = Vector3(f.x, f.y, 1.0f - Math::abs(f.x) - Math::abs(f.y));
	const real_t t = CLAMP(-n.z, 0.0f, 1.0f);
	n.x += n.x >= 0 ? -t : t;
	n.y += n.y >= 0 ? -t : t;
	return n.normalized();
}
