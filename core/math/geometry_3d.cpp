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

#include "minimization.h"
#include "thirdparty/misc/clipper.hpp"
#include "thirdparty/misc/polypartition.h"

constexpr real_t CIRCLE_CIRCLE_RELATIVE_TOLERANCE = 1e-6;
constexpr real_t LINE_CIRCLE_RELATIVE_TOLERANCE = 1e-6;

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

		int next_x = x, next_y = y, next_z = z;
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

#define vert(m_idx) Vector3(((m_idx)&4) >> 2, ((m_idx)&2) >> 1, (m_idx)&1)

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

Vector<Face3> Geometry3D::wrap_geometry(Vector<Face3> p_array, real_t *p_error) {
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
	if (p_error) {
		*p_error = voxelsize.length();
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

		if (ABS(p.normal.dot(ref)) > 0.95f) {
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

	const double sides_step = Math_TAU / p_sides;
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

	const double lon_step = Math_TAU / p_lons;
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

	const double sides_step = Math_TAU / p_sides;
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
#define INF 1e20

/* dt of 1d function using squared distance */
static void edt(float *f, int stride, int n) {
	float *d = (float *)alloca(sizeof(float) * n + sizeof(int) * n + sizeof(float) * (n + 1));
	int *v = reinterpret_cast<int *>(&(d[n]));
	float *z = reinterpret_cast<float *>(&v[n]);

	int k = 0;
	v[0] = 0;
	z[0] = -INF;
	z[1] = +INF;
	for (int q = 1; q <= n - 1; q++) {
		float s = ((f[q * stride] + square(q)) - (f[v[k] * stride] + square(v[k]))) / (2 * q - 2 * v[k]);
		while (s <= z[k]) {
			k--;
			s = ((f[q * stride] + square(q)) - (f[v[k] * stride] + square(v[k]))) / (2 * q - 2 * v[k]);
		}
		k++;
		v[k] = q;

		z[k] = s;
		z[k + 1] = +INF;
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
		work_memory[i] = INF;
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

/*
 * Point-Circle distance
 */

Vector3 get_closest_point_on_circle(const Vector3 &p_point, const Transform3D &p_circle_transform, const real_t p_circle_radius) {
	Vector3 point_local = p_point - p_circle_transform.origin;
	Vector3 U = p_circle_transform.basis.get_column(0);
	Vector3 V = p_circle_transform.basis.get_column(2);
	Vector3 closest_local = U.dot(point_local) * U + V.dot(point_local) * V;
	real_t closest_local_length = closest_local.length();
	if (closest_local_length > 0.0) {
		return p_circle_transform.origin + p_circle_radius * (closest_local / closest_local_length);
	} else {
		return p_circle_transform.origin + p_circle_radius * U.normalized();
	}
}

/*
 * Circle-Circle distance
 * Based on David Vranek's Fast and Accurate Circle-Circle and Circle-Line 3D Distance Computation.
 * Implementation based on the one in MinuteTorus by Sang Hyun Son, but significantly modified.
 * See https://github.com/SonSang/MinuteTorus/blob/1a11cacc8cea6f62d16ffc9919ba35fe998742c9/Circle/CircleDistance.cpp
 */

/**********************************************************************************
 * MinuteTorus, Copyright (c) [2021] [Sanghyun Son, Youngjin Park]                *
 * Permission is hereby granted, free of charge, to any person obtaining a copy   *
 * of this software and associated documentation files (the "Software"), to deal  *
 * in the Software without restriction, including without limitation the rights   *
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      *
 * copies of the Software, and to permit persons to whom the Software is          *
 * furnished to do so, subject to the following conditions:                       *
 *                                                                                *
 * The above copyright notice and this permission notice shall be included in all *
 * copies or substantial portions of the Software.                                *
 *                                                                                *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    *
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  *
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  *
 * SOFTWARE.                                                                      *
 **********************************************************************************/

struct CircleCircleDistanceData {
	real_t a[10];
	real_t t = INFINITY; // Ensure an update on the first call to update_t.
	real_t cost = 0.0;
	real_t sint = 0.0;
	real_t cost2 = 0.0;
	real_t sint2 = 0.0;
	real_t costsint = 0.0;
	real_t root = 0.0;

	_FORCE_INLINE_ void update_t(real_t p_t) {
		if (t == p_t) {
			return;
		}
		t = p_t;
		cost = cos(t);
		sint = sin(t);
		cost2 = cost * cost;
		sint2 = sint * sint;
		costsint = cost * sint;
		root = sqrt(a[5] * cost2 + a[4] * sint2 + a[3] * cost + a[2] * sint + a[1] * costsint + a[0]);
	}
};

inline real_t circle_circle_distance(void *p_data, real_t p_t) { // p_t is the angle on circle1
	CircleCircleDistanceData *d = static_cast<CircleCircleDistanceData *>(p_data);
	d->update_t(p_t);
	return d->a[9] * d->cost + d->a[8] * d->sint + d->a[7] + d->a[6] * d->root;
}

inline real_t circle_circle_distance_derivative(void *p_data, real_t p_t) { // first derivative of circle_circle_distance
	CircleCircleDistanceData *d = static_cast<CircleCircleDistanceData *>(p_data);
	d->update_t(p_t);
	return -d->a[9] * d->sint + d->a[8] * d->cost + d->a[6] * 0.5 * (1.0 / d->root) * (2.0 * d->costsint * (d->a[4] - d->a[5]) - d->a[3] * d->sint + d->a[2] * d->cost + d->a[1] * (d->cost2 - d->sint2));
}

bool real_roots_of_quadratic_polynomial(const real_t p_coeff[3], real_t *r_x) {
	const real_t a = p_coeff[2];
	const real_t b = p_coeff[1];
	const real_t c = p_coeff[0];
	if (a == 0.0) { // Not actually quadratic.
		return false;
	}
	if (c == 0.0) { // At least one root is zero.
		r_x[0] = 0.0;
		r_x[1] = -b / a;
		return true;
	}
	// Compute discriminant avoiding overflow.
	real_t l_e;
	real_t l_d;
	const real_t l_b = 0.5 * b;
	if (Math::abs(l_b) < Math::abs(c)) {
		l_e = (c < 0.0) ? -a : a;
		l_e = l_b * (l_b / Math::abs(c)) - l_e;
		l_d = Math::sqrt(Math::abs(l_e)) * Math::sqrt(Math::abs(c));
	} else {
		l_e = 1.0 - (a / l_b) * (c / l_b);
		l_d = Math::sqrt(Math::abs(l_e)) * Math::abs(l_b);
	}
	if (l_e >= 0.0) { // Real roots.
		// Compute largest root (in absolute value) by avoiding cancellation.
		if (l_b > 0.0) {
			l_d = -l_d;
		}
		r_x[0] = (-l_b + l_d) / a;
		r_x[1] = (r_x[0] != 0.0) ? (c / r_x[0]) / a : 0.0; // Smallest root.
		return true;
	} else { // Complex roots.
		return false;
	}
}

void Geometry3D::get_closest_points_between_circle_and_circle(const Transform3D &p_circle0_transform, const real_t p_circle0_radius, const Transform3D &p_circle1_transform, const real_t p_circle1_radius, Vector3 *r_closest_points, size_t &r_num_closest_pairs) {
	// Constants used in the squared distance function.
	Vector3 center_a = p_circle0_transform.origin;
	Vector3 center_b = p_circle1_transform.origin;
	Vector3 normal_a = p_circle0_transform.basis.get_column(1);
	Vector3 U = p_circle1_transform.basis.get_column(0);
	Vector3 V = p_circle1_transform.basis.get_column(2);
	real_t R_a = p_circle0_radius;
	real_t R_b = p_circle1_radius;
	real_t tmp[11];
	tmp[0] = center_a.dot(center_a);
	tmp[1] = center_a.dot(center_b);
	tmp[2] = center_b.dot(center_b);
	tmp[3] = normal_a.dot(center_a);
	tmp[4] = normal_a.dot(center_b);
	tmp[5] = normal_a.dot(U);
	tmp[6] = normal_a.dot(V);
	tmp[7] = center_a.dot(U);
	tmp[8] = center_a.dot(V);
	tmp[9] = center_b.dot(U);
	tmp[10] = center_b.dot(V);

	// Coefficients used in the squared distance function and its derivative; see circle_circle_distance and circle_circle_distance_derivative.
	CircleCircleDistanceData data;
	data.a[0] = R_b * R_b + tmp[2] + tmp[0] - 2.0 * tmp[1] - (tmp[4] - tmp[3]) * (tmp[4] - tmp[3]);
	data.a[1] = -2.0 * R_b * R_b * tmp[5] * tmp[6];
	data.a[2] = 2.0 * R_b * tmp[10] - 2.0 * R_b * tmp[8] - 2.0 * R_b * (tmp[4] - tmp[3]) * tmp[6];
	data.a[3] = 2.0 * R_b * tmp[9] - 2.0 * R_b * tmp[7] - 2.0 * R_b * (tmp[4] - tmp[3]) * tmp[5];
	data.a[4] = -R_b * R_b * tmp[6] * tmp[6];
	data.a[5] = -R_b * R_b * tmp[5] * tmp[5];
	data.a[6] = -2.0 * R_a;
	data.a[7] = R_a * R_a + R_b * R_b + tmp[0] + tmp[2] - 2.0 * tmp[1];
	data.a[8] = 2.0 * R_b * tmp[10] - 2.0 * R_b * tmp[8];
	data.a[9] = 2.0 * R_b * tmp[9] - 2.0 * R_b * tmp[7];

	// Evaluate the squared distance function at three equally spaced angles.
	real_t a = (-2.0 / 3.0) * Math_PI;
	real_t b = 0;
	real_t c = (2.0 / 3.0) * Math_PI;
	real_t fa = circle_circle_distance(&data, a);
	real_t fb = circle_circle_distance(&data, b);
	real_t fc = circle_circle_distance(&data, c);

	// TODO: Handle special cases like f(a) == f(b) == f(c) and the other three below, which do happen.

	// Re-order the three angles to form a bracketing triplet.
	if (fb > fc) {
		if (fa > fc) { // f(c) is smallest; swap b and c.
			b = (2.0 / 3.0) * Math_PI;
			c = Math_TAU;
			SWAP(fb, fc);
		} else if (fa < fc) { // f(a) is smallest; swap b and a.
			a = -Math_TAU;
			b = -(2.0 / 3.0) * Math_PI;
			SWAP(fb, fa);
		} else { // f(b) is greater than f(a) == f(c)
			ERR_FAIL_MSG("f(a) == f(c)");
		}
	} else if (fb < fc) {
		if (fa < fb) { // f(a) is smallest; swap a and b.
			a = -Math_TAU;
			b = -(2.0 / 3.0) * Math_PI;
			SWAP(fa, fb);
		} else if (fa == fb) { // f(c) is greater than f(a) == f(b)
			ERR_FAIL_MSG("f(a) == f(b)");
		}
		// else f(b) is smallest; nothing to do.
	} else { // f(b) == f(c)
		ERR_FAIL_MSG("f(b) == f(c)");
	}

	// Find a single local minimum of the distance function numerically, using a modification of Brent's minimization algorithm (making use of the derivative).
	real_t loc_min_t;
	real_t loc_min_dist = Minimization::get_local_minimum(&data, &circle_circle_distance, &circle_circle_distance_derivative, a, b, c, CIRCLE_CIRCLE_RELATIVE_TOLERANCE, &loc_min_t);

	// Shift the distance function down by the local minimum; setting that equal to zero leads to a polynomial equation of degree four in z = cos(t).
	real_t d[3] = { data.a[7] - loc_min_dist, data.a[6] * data.a[6], data.a[8] * data.a[8] };
	real_t c1 = 2.0 * data.a[8] * d[0] - d[1] * data.a[2];
	//real_t c2 = d[1] * data.a[4]; // NOTE: unused.
	real_t c3 = d[0] * d[0] - d[1] * data.a[0] + d[2] - d[1] * data.a[4];
	real_t c4 = 2.0 * data.a[9] * d[0] - d[1] * data.a[3];
	real_t c5 = 2.0 * data.a[9] * data.a[8] - d[1] * data.a[1];
	real_t c6 = -d[2] + d[1] * data.a[4] + data.a[9] * data.a[9] - d[1] * data.a[5];
	// Coefficients of the quartic:
	real_t quartic_coeff[5];
	//quartic_coeff[0] = c3 * c3 - c1 * c1; // NOTE: unused.
	//quartic_coeff[1] = -2.0 * c1 * c5 + 2.0 * c3 * c4; // NOTE: unused.
	quartic_coeff[2] = c1 * c1 - c5 * c5 + 2.0 * c3 * c6 + c4 * c4;
	quartic_coeff[3] = 2.0 * c1 * c5 + 2.0 * c4 * c6;
	quartic_coeff[4] = c6 * c6 + c5 * c5;

	// The polynomial of degree four in z = cos(t) has a double root at the previously found z0 = cos(loc_min_t); use Horner's method to deflate it to a quadratic polynomial.
	real_t z = cos(loc_min_t);
	real_t quadratic_coeff[3];
	quadratic_coeff[0] = (3.0 * quartic_coeff[4] * z + 2.0 * quartic_coeff[3]) * z + quartic_coeff[2];
	quadratic_coeff[1] = 2.0 * quartic_coeff[4] * z + quartic_coeff[3];
	quadratic_coeff[2] = quartic_coeff[4];
	real_t quadratic_roots[2];

	real_t gl_min_dist = loc_min_dist;
	real_t gl_min_t = loc_min_t;

	// If the quadratic has two distinct real roots, then the global minimum lies between the two roots of the quadratic.
	if (real_roots_of_quadratic_polynomial(quadratic_coeff, quadratic_roots)) {
		real_t cosa = CLAMP(quadratic_roots[0], -1.0, 1.0);
		real_t cosb = CLAMP(quadratic_roots[1], -1.0, 1.0);
		a = acos(cosa); // In the range [0, pi].
		b = acos(cosb); // In the range [0, pi].
		if (circle_circle_distance_derivative(&data, a) * circle_circle_distance_derivative(&data, b) < 0.0) {
			// Find a minimum in this interval.
			// TODO: Can there be a local maximum instead of a minimum in this interval?
			// TODO: If not, should we use a root finder here to find a zero of the derivative?
			loc_min_dist = Minimization::get_local_minimum(&data, &circle_circle_distance, &circle_circle_distance_derivative, a, b, b, CIRCLE_CIRCLE_RELATIVE_TOLERANCE, &loc_min_t);
			if (loc_min_dist < gl_min_dist) {
				gl_min_dist = loc_min_dist;
				gl_min_t = loc_min_t;
			}
		}
		a = Math_TAU - a;
		b = Math_TAU - b;
		if (circle_circle_distance_derivative(&data, a) * circle_circle_distance_derivative(&data, b) < 0.0) {
			loc_min_dist = Minimization::get_local_minimum(&data, &circle_circle_distance, &circle_circle_distance_derivative, a, b, b, CIRCLE_CIRCLE_RELATIVE_TOLERANCE, &loc_min_t);
			if (loc_min_dist < gl_min_dist) {
				gl_min_dist = loc_min_dist;
				gl_min_t = loc_min_t;
			}
		}
	}
	// NOTE: If the quadratic has just one real root, then the previously found local minimum is a global minimum, and so is the root of the quadratic.
	// If the quadratic has no real roots, then the previously found local minimum is the unique global minimum.

	r_num_closest_pairs = 1;
	r_closest_points[1] = p_circle1_transform.origin + R_b * Math::cos(gl_min_t) * U + R_b * Math::sin(gl_min_t) * V;
	r_closest_points[0] = get_closest_point_on_circle(r_closest_points[1], p_circle0_transform, R_a);
}

/*
 * Line-Circle distance
 * Based on David Vranek's Fast and Accurate Circle-Circle and Circle-Line 3D Distance Computation.
 * Implementation based on the one once included in ODE by Russell L. Smith, but heavily modified.
 * See https://github.com/cyberbotics/webots/blob/21d6eefd7a2f59fa3277788b552587e388dfe0a4/src/ode/ode/src/cylinder_revolution_data.cpp
 */

/*************************************************************************
 *                                                                       *
 * Open Dynamics Engine, Copyright (C) 2001-2003 Russell L. Smith.       *
 * All rights reserved.  Email: russ@q12.org   Web: www.q12.org          *
 *                                                                       *
 * This library is free software; you can redistribute it and/or         *
 * modify it under the terms of EITHER:                                  *
 *   (1) The GNU Lesser General Public License as published by the Free  *
 *       Software Foundation; either version 2.1 of the License, or (at  *
 *       your option) any later version. The text of the GNU Lesser      *
 *       General Public License is included with this library in the     *
 *       file LICENSE.TXT.                                               *
 *   (2) The BSD-style license that is included with this library in     *
 *       the file LICENSE-BSD.TXT.                                       *
 *                                                                       *
 * This library is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files    *
 * LICENSE.TXT and LICENSE-BSD.TXT for more details.                     *
 *                                                                       *
 *************************************************************************/

struct LineCircleDistanceData {
	real_t A[6]; // NOTE: Compared with Vranek's notation (a_k), we have: A[0] = a_4, A[1] = a_5, A[2] = a_3, A[3] = a_0, A[4] = a_1, A[5] = a_2, and a_6 = 1.0.
	real_t B[2];
	real_t t = INFINITY; // Ensure an update on the first call to update_t.
	real_t distCN = 0.0;
	_FORCE_INLINE_ void update_t(real_t p_t) {
		if (t == p_t) {
			return;
		}
		t = p_t;
		distCN = Math::sqrt(A[3] + t * (A[4] + t * A[5]));
	}
};

real_t line_circle_distance(void *p_data, real_t p_t) { // p_t is the line parameter
	LineCircleDistanceData *distance_data = static_cast<LineCircleDistanceData *>(p_data);
	distance_data->update_t(p_t);
	return distance_data->A[0] + p_t * (distance_data->A[1] + p_t) + distance_data->A[2] * distance_data->distCN;
}

real_t line_circle_distance_derivative(void *p_data, real_t p_t) { // first derivative of line_circle_distance
	LineCircleDistanceData *distance_data = static_cast<LineCircleDistanceData *>(p_data);
	distance_data->update_t(p_t);
	return distance_data->A[1] + 2.0 * p_t + (distance_data->B[0] + p_t * distance_data->B[1]) * (1.0 / distance_data->distCN);
}

void Geometry3D::get_closest_points_between_line_and_circle(const Vector3 &p_line_origin, const Vector3 &p_line_direction, const Transform3D &p_circle_transform, const real_t p_circle_radius, Vector3 *r_closest_points, size_t &r_num_closest_pairs) {
	// NOTE: This function assumes p_line_direction is a normalized vector.

	// TODO: Detect and handle degenerate cases early?

	// Constants used in the squared distance function.
	Vector3 D = p_line_origin - p_circle_transform.origin;
	Vector3 circle_normal = p_circle_transform.basis.get_column(1);
	Vector3 E = p_line_direction - (circle_normal.dot(p_line_direction)) * circle_normal;
	Vector3 F = D - (circle_normal.dot(D)) * circle_normal;
	real_t R_a = p_circle_radius;
	Vector3 M = p_line_direction;

	LineCircleDistanceData data;

	// Coefficients used in the squared distance function; see line_circle_distance.
	data.A[0] = D.dot(D) + R_a * R_a;
	data.A[1] = 2 * D.dot(M);
	data.A[2] = -2 * R_a;
	data.A[3] = F.dot(F);
	data.A[4] = 2 * E.dot(F);
	data.A[5] = E.dot(E);
	// Note: Here we have a6 = p_line_direction.length_squared() == 1.

	// Coefficients used in the derivative of the squared distance function; see line_circle_distance_derivative.
	data.B[0] = 0.5 * data.A[2] * data.A[4];
	data.B[1] = data.A[2] * data.A[5];

	// TODO: Improve minimum bracketing or starting interval;
	// see https://math.stackexchange.com/questions/4618258/minimum-distance-between-a-line-and-a-circle-in-3d-bracketing-triplet
	// For now, determine a bracketing triplet from an interval:
	real_t r1 = p_circle_radius;
	real_t start = (p_circle_transform.origin - p_line_origin).dot(p_line_direction);
	// NOTE: We widen the interval in case the minimum happens to be exactly (start - r1) or (start + r1).
	real_t a = start - r1 - 0.1;
	real_t b = start + r1 + 0.1;
	real_t c = 0.0;
	real_t fa = 0.0;
	real_t fb = 0.0;
	real_t fc = 0.0;
	Minimization::bracketing_triplet_from_interval(&data, &line_circle_distance, &a, &b, &c, &fa, &fb, &fc);

	// Find a single local minimum of the distance function numerically, using a modification of Brent's minimization algorithm (making use of the derivative).
	real_t loc_min_t = 0.0;
	real_t loc_min_dist = Minimization::get_local_minimum(&data, &line_circle_distance, &line_circle_distance_derivative, a, b, c, LINE_CIRCLE_RELATIVE_TOLERANCE, &loc_min_t);

	// Shift the distance function f(t) down by the local minimum D:
	// f(t) = t^2 + a5*t + a4 + a3*sqrt(a2*t^2 + a1*t + a0)
	// g(t) = f(t) - D, where D = f(loc_min_t)
	// Solving g(t) = 0 leads to a polynomial equation of degree four in t:
	// t^2 + a5*t + a4 - D = a3*sqrt(a2*t^2 + a1*t + a0)
	// (t^2 + a5*t + a4 - D)^2 = a3^2 * (a2*t^2 + a1*t + a0)
	// (t^2 + a5*t + a4 - D)^2 - a3^2 * (a2*t^2 + a1*t + a0) = 0
	// Expand in powers of t:
	// t^4 + 2*a5*t^3 + (a5^2 - a2*a3^2 + 2*(a4 - D))*t^2 + (2*a4*a5 - a1*a3^2 - 2*D*a5)*t - a0*a3^2 + D^2 - 2*D*a4 + a4^2 = 0
	real_t quartic_coeff3 = 2.0 * data.A[1];
	real_t quartic_coeff2 = data.A[1] * data.A[1] - data.A[2] * data.A[2] * data.A[5] + 2.0 * (data.A[0] - loc_min_dist);

	// The polynomial of degree four has a double root at the previously found t0 = loc_min_t; use Horner's method twice to deflate it to a quadratic polynomial:
	// Quartic: C1*t^4 + C2*t^3 + C3*t^2 + C4*t + C5
	// Deflate quartic -> cubic B1*t^3 + B2*t^2 + B3*t + B4 using the root t0:
	// - B1 = C1;
	// - B2 = B1*t0 + C2 = C1*t0 + C2;
	// - B3 = B2*t0 + C3 = C1*t0^2 + C2*t0 + C3;
	// - B4 = B3*t0 + C4 = C1*t0^3 + C2*t0^2 + C3*t0 + C4
	// Deflate cubic -> quadratic A1*t^2 + A2*t + A3 using the root t0:
	// - A1 = B1 = C1
	// - A2 = A1*t0 + B2 = 2*C1*t0 + C2
	// - A3 = A2*t0 + B3 = 3*C1*t0^2 + 2*C2*t0 + C3
	real_t quadratic_coeff[3];
	quadratic_coeff[0] = (3.0 * loc_min_t + 2.0 * quartic_coeff3) * loc_min_t + quartic_coeff2;
	quadratic_coeff[1] = 2.0 * loc_min_t + quartic_coeff3;
	quadratic_coeff[2] = 1.0;

	// If the quadratic has two distinct real roots, then the global minimum lies between the two roots of the quadratic.
	real_t quadratic_roots[2];
	if (real_roots_of_quadratic_polynomial(quadratic_coeff, quadratic_roots)) {
		a = quadratic_roots[0];
		b = quadratic_roots[1];
		// Find a minimum in this interval.
		// TODO: Can there be a local maximum instead of a minimum in this interval?
		// TODO: If not, should we use a root finder here to find a zero of the derivative?
		Minimization::bracketing_triplet_from_interval(&data, &line_circle_distance, &a, &b, &c, &fa, &fb, &fc);
		loc_min_dist = Minimization::get_local_minimum(&data, &line_circle_distance, &line_circle_distance_derivative, a, b, c, LINE_CIRCLE_RELATIVE_TOLERANCE, &loc_min_t);
	}
	// NOTE: If the quadratic has just one real root, then the previously found local minimum is a global minimum, and so is the root of the quadratic.
	// If the quadratic has no real roots, then the previously found local minimum is the unique global minimum.

	r_closest_points[0] = p_line_origin + loc_min_t * p_line_direction;
	r_closest_points[1] = get_closest_point_on_circle(r_closest_points[0], p_circle_transform, p_circle_radius);
	r_num_closest_pairs = 1;
}
