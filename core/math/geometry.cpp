/*************************************************************************/
/*  geometry.cpp                                                         */
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
#include "geometry.h"
#include "print_string.h"

void Geometry::MeshData::optimize_vertices() {

	Map<int, int> vtx_remap;

	for (int i = 0; i < faces.size(); i++) {

		for (int j = 0; j < faces[i].indices.size(); j++) {

			int idx = faces[i].indices[j];
			if (!vtx_remap.has(idx)) {
				int ni = vtx_remap.size();
				vtx_remap[idx] = ni;
			}

			faces[i].indices[j] = vtx_remap[idx];
		}
	}

	for (int i = 0; i < edges.size(); i++) {

		int a = edges[i].a;
		int b = edges[i].b;

		if (!vtx_remap.has(a)) {
			int ni = vtx_remap.size();
			vtx_remap[a] = ni;
		}
		if (!vtx_remap.has(b)) {
			int ni = vtx_remap.size();
			vtx_remap[b] = ni;
		}

		edges[i].a = vtx_remap[a];
		edges[i].b = vtx_remap[b];
	}

	Vector<Vector3> new_vertices;
	new_vertices.resize(vtx_remap.size());

	for (int i = 0; i < vertices.size(); i++) {

		if (vtx_remap.has(i))
			new_vertices[vtx_remap[i]] = vertices[i];
	}
	vertices = new_vertices;
}

Vector<Vector<Vector2> > (*Geometry::_decompose_func)(const Vector<Vector2> &p_polygon) = NULL;

struct _FaceClassify {

	struct _Link {

		int face;
		int edge;
		void clear() {
			face = -1;
			edge = -1;
		}
		_Link() {
			face = -1;
			edge = -1;
		}
	};
	bool valid;
	int group;
	_Link links[3];
	Face3 face;
	_FaceClassify() {
		group = -1;
		valid = false;
	};
};

static bool _connect_faces(_FaceClassify *p_faces, int len, int p_group) {
	/* connect faces, error will occur if an edge is shared between more than 2 faces */
	/* clear connections */

	bool error = false;

	for (int i = 0; i < len; i++) {

		for (int j = 0; j < 3; j++) {

			p_faces[i].links[j].clear();
		}
	}

	for (int i = 0; i < len; i++) {

		if (p_faces[i].group != p_group)
			continue;
		for (int j = i + 1; j < len; j++) {

			if (p_faces[j].group != p_group)
				continue;

			for (int k = 0; k < 3; k++) {

				Vector3 vi1 = p_faces[i].face.vertex[k];
				Vector3 vi2 = p_faces[i].face.vertex[(k + 1) % 3];

				for (int l = 0; l < 3; l++) {

					Vector3 vj2 = p_faces[j].face.vertex[l];
					Vector3 vj1 = p_faces[j].face.vertex[(l + 1) % 3];

					if (vi1.distance_to(vj1) < 0.00001 &&
							vi2.distance_to(vj2) < 0.00001) {
						if (p_faces[i].links[k].face != -1) {

							ERR_PRINT("already linked\n");
							error = true;
							break;
						}
						if (p_faces[j].links[l].face != -1) {

							ERR_PRINT("already linked\n");
							error = true;
							break;
						}

						p_faces[i].links[k].face = j;
						p_faces[i].links[k].edge = l;
						p_faces[j].links[l].face = i;
						p_faces[j].links[l].edge = k;
					}
				}
				if (error)
					break;
			}
			if (error)
				break;
		}
		if (error)
			break;
	}

	for (int i = 0; i < len; i++) {

		p_faces[i].valid = true;
		for (int j = 0; j < 3; j++) {

			if (p_faces[i].links[j].face == -1)
				p_faces[i].valid = false;
		}
		/*printf("face %i is valid: %i, group %i. connected to %i:%i,%i:%i,%i:%i\n",i,p_faces[i].valid,p_faces[i].group,
			p_faces[i].links[0].face,
			p_faces[i].links[0].edge,
			p_faces[i].links[1].face,
			p_faces[i].links[1].edge,
			p_faces[i].links[2].face,
			p_faces[i].links[2].edge);*/
	}
	return error;
}

static bool _group_face(_FaceClassify *p_faces, int len, int p_index, int p_group) {

	if (p_faces[p_index].group >= 0)
		return false;

	p_faces[p_index].group = p_group;

	for (int i = 0; i < 3; i++) {

		ERR_FAIL_INDEX_V(p_faces[p_index].links[i].face, len, true);
		_group_face(p_faces, len, p_faces[p_index].links[i].face, p_group);
	}

	return true;
}

PoolVector<PoolVector<Face3> > Geometry::separate_objects(PoolVector<Face3> p_array) {

	PoolVector<PoolVector<Face3> > objects;

	int len = p_array.size();

	PoolVector<Face3>::Read r = p_array.read();

	const Face3 *arrayptr = r.ptr();

	PoolVector<_FaceClassify> fc;

	fc.resize(len);

	PoolVector<_FaceClassify>::Write fcw = fc.write();

	_FaceClassify *_fcptr = fcw.ptr();

	for (int i = 0; i < len; i++) {

		_fcptr[i].face = arrayptr[i];
	}

	bool error = _connect_faces(_fcptr, len, -1);

	if (error) {

		ERR_FAIL_COND_V(error, PoolVector<PoolVector<Face3> >()); // invalid geometry
	}

	/* group connected faces in separate objects */

	int group = 0;
	for (int i = 0; i < len; i++) {

		if (!_fcptr[i].valid)
			continue;
		if (_group_face(_fcptr, len, i, group)) {
			group++;
		}
	}

	/* group connected faces in separate objects */

	for (int i = 0; i < len; i++) {

		_fcptr[i].face = arrayptr[i];
	}

	if (group >= 0) {

		objects.resize(group);
		PoolVector<PoolVector<Face3> >::Write obw = objects.write();
		PoolVector<Face3> *group_faces = obw.ptr();

		for (int i = 0; i < len; i++) {
			if (!_fcptr[i].valid)
				continue;
			if (_fcptr[i].group >= 0 && _fcptr[i].group < group) {

				group_faces[_fcptr[i].group].push_back(_fcptr[i].face);
			}
		}
	}

	return objects;
}

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

	Rect3 aabb(Vector3(x, y, z), Vector3(len_x, len_y, len_z));
	aabb.pos = aabb.pos * voxelsize;
	aabb.size = aabb.size * voxelsize;

	if (!p_face.intersects_aabb(aabb))
		return;

	if (len_x == 1 && len_y == 1 && len_z == 1) {

		p_cell_status[x][y][z] = _CELL_SOLID;
		return;
	}

	int div_x = len_x > 1 ? 2 : 1;
	int div_y = len_y > 1 ? 2 : 1;
	int div_z = len_z > 1 ? 2 : 1;

#define _SPLIT(m_i, m_div, m_v, m_len_v, m_new_v, m_new_len_v) \
	if (m_div == 1) {                                          \
		m_new_v = m_v;                                         \
		m_new_len_v = 1;                                       \
	} else if (m_i == 0) {                                     \
		m_new_v = m_v;                                         \
		m_new_len_v = m_len_v / 2;                             \
	} else {                                                   \
		m_new_v = m_v + m_len_v / 2;                           \
		m_new_len_v = m_len_v - m_len_v / 2;                   \
	}

	int new_x;
	int new_len_x;
	int new_y;
	int new_len_y;
	int new_z;
	int new_len_z;

	for (int i = 0; i < div_x; i++) {

		_SPLIT(i, div_x, x, len_x, new_x, new_len_x);

		for (int j = 0; j < div_y; j++) {

			_SPLIT(j, div_y, y, len_y, new_y, new_len_y);

			for (int k = 0; k < div_z; k++) {

				_SPLIT(k, div_z, z, len_z, new_z, new_len_z);

				_plot_face(p_cell_status, new_x, new_y, new_z, new_len_x, new_len_y, new_len_z, voxelsize, p_face);
			}
		}
	}
}

static inline void _mark_outside(uint8_t ***p_cell_status, int x, int y, int z, int len_x, int len_y, int len_z) {

	if (p_cell_status[x][y][z] & 3)
		return; // nothing to do, already used and/or visited

	p_cell_status[x][y][z] = _CELL_PREV_FIRST;

	while (true) {

		uint8_t &c = p_cell_status[x][y][z];

		//printf("at %i,%i,%i\n",x,y,z);

		if ((c & _CELL_STEP_MASK) == _CELL_STEP_NONE) {
			/* Haven't been in here, mark as outside */
			p_cell_status[x][y][z] |= _CELL_EXTERIOR;
			//printf("not marked as anything, marking exterior\n");
		}

		//printf("cell step is %i\n",(c&_CELL_STEP_MASK));

		if ((c & _CELL_STEP_MASK) != _CELL_STEP_DONE) {
			/* if not done, increase step */
			c += 1 << 2;
			//printf("incrementing cell step\n");
		}

		if ((c & _CELL_STEP_MASK) == _CELL_STEP_DONE) {
			/* Go back */
			//printf("done, going back a cell\n");

			switch (c & _CELL_PREV_MASK) {
				case _CELL_PREV_FIRST: {
					//printf("at end, finished marking\n");
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

		//printf("attempting new cell!\n");

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
			default: ERR_FAIL();
		}

		//printf("testing if new cell will be ok...!\n");

		if (next_x < 0 || next_x >= len_x)
			continue;
		if (next_y < 0 || next_y >= len_y)
			continue;
		if (next_z < 0 || next_z >= len_z)
			continue;

		//printf("testing if new cell is traversable\n");

		if (p_cell_status[next_x][next_y][next_z] & 3)
			continue;

		//printf("move to it\n");

		x = next_x;
		y = next_y;
		z = next_z;
		p_cell_status[x][y][z] |= prev;
	}
}

static inline void _build_faces(uint8_t ***p_cell_status, int x, int y, int z, int len_x, int len_y, int len_z, PoolVector<Face3> &p_faces) {

	ERR_FAIL_INDEX(x, len_x);
	ERR_FAIL_INDEX(y, len_y);
	ERR_FAIL_INDEX(z, len_z);

	if (p_cell_status[x][y][z] & _CELL_EXTERIOR)
		return;

/*	static const Vector3 vertices[8]={
		Vector3(0,0,0),
		Vector3(0,0,1),
		Vector3(0,1,0),
		Vector3(0,1,1),
		Vector3(1,0,0),
		Vector3(1,0,1),
		Vector3(1,1,0),
		Vector3(1,1,1),
	};
*/
#define vert(m_idx) Vector3((m_idx & 4) >> 2, (m_idx & 2) >> 1, m_idx & 1)

	static const uint8_t indices[6][4] = {
		{ 7, 6, 4, 5 },
		{ 7, 3, 2, 6 },
		{ 7, 5, 1, 3 },
		{ 0, 2, 3, 1 },
		{ 0, 1, 5, 4 },
		{ 0, 4, 6, 2 },

	};
	/*

		{0,1,2,3},
		{0,1,4,5},
		{0,2,4,6},
		{4,5,6,7},
		{2,3,7,6},
		{1,3,5,7},

		{0,2,3,1},
		{0,1,5,4},
		{0,4,6,2},
		{7,6,4,5},
		{7,3,2,6},
		{7,5,1,3},
*/

	for (int i = 0; i < 6; i++) {

		Vector3 face_points[4];
		int disp_x = x + ((i % 3) == 0 ? ((i < 3) ? 1 : -1) : 0);
		int disp_y = y + (((i - 1) % 3) == 0 ? ((i < 3) ? 1 : -1) : 0);
		int disp_z = z + (((i - 2) % 3) == 0 ? ((i < 3) ? 1 : -1) : 0);

		bool plot = false;

		if (disp_x < 0 || disp_x >= len_x)
			plot = true;
		if (disp_y < 0 || disp_y >= len_y)
			plot = true;
		if (disp_z < 0 || disp_z >= len_z)
			plot = true;

		if (!plot && (p_cell_status[disp_x][disp_y][disp_z] & _CELL_EXTERIOR))
			plot = true;

		if (!plot)
			continue;

		for (int j = 0; j < 4; j++)
			face_points[j] = vert(indices[i][j]) + Vector3(x, y, z);

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

PoolVector<Face3> Geometry::wrap_geometry(PoolVector<Face3> p_array, real_t *p_error) {

#define _MIN_SIZE 1.0
#define _MAX_LENGTH 20

	int face_count = p_array.size();
	PoolVector<Face3>::Read facesr = p_array.read();
	const Face3 *faces = facesr.ptr();

	Rect3 global_aabb;

	for (int i = 0; i < face_count; i++) {

		if (i == 0) {

			global_aabb = faces[i].get_aabb();
		} else {

			global_aabb.merge_with(faces[i].get_aabb());
		}
	}

	global_aabb.grow_by(0.01); // avoid numerical error

	// determine amount of cells in grid axis
	int div_x, div_y, div_z;

	if (global_aabb.size.x / _MIN_SIZE < _MAX_LENGTH)
		div_x = (int)(global_aabb.size.x / _MIN_SIZE) + 1;
	else
		div_x = _MAX_LENGTH;

	if (global_aabb.size.y / _MIN_SIZE < _MAX_LENGTH)
		div_y = (int)(global_aabb.size.y / _MIN_SIZE) + 1;
	else
		div_y = _MAX_LENGTH;

	if (global_aabb.size.z / _MIN_SIZE < _MAX_LENGTH)
		div_z = (int)(global_aabb.size.z / _MIN_SIZE) + 1;
	else
		div_z = _MAX_LENGTH;

	Vector3 voxelsize = global_aabb.size;
	voxelsize.x /= div_x;
	voxelsize.y /= div_y;
	voxelsize.z /= div_z;

	// create and initialize cells to zero
	//print_line("Wrapper: Initializing Cells");

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

	// plot faces into cells
	//print_line("Wrapper (1/6): Plotting Faces");

	for (int i = 0; i < face_count; i++) {

		Face3 f = faces[i];
		for (int j = 0; j < 3; j++) {

			f.vertex[j] -= global_aabb.pos;
		}
		_plot_face(cell_status, 0, 0, 0, div_x, div_y, div_z, voxelsize, f);
	}

	// determine which cells connect to the outside by traversing the outside and recursively flood-fill marking

	//print_line("Wrapper (2/6): Flood Filling");

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

	// build faces for the inside-outside cell divisors

	//print_line("Wrapper (3/6): Building Faces");

	PoolVector<Face3> wrapped_faces;

	for (int i = 0; i < div_x; i++) {

		for (int j = 0; j < div_y; j++) {

			for (int k = 0; k < div_z; k++) {

				_build_faces(cell_status, i, j, k, div_x, div_y, div_z, wrapped_faces);
			}
		}
	}

	//print_line("Wrapper (4/6): Transforming Back Vertices");

	// transform face vertices to global coords

	int wrapped_faces_count = wrapped_faces.size();
	PoolVector<Face3>::Write wrapped_facesw = wrapped_faces.write();
	Face3 *wrapped_faces_ptr = wrapped_facesw.ptr();

	for (int i = 0; i < wrapped_faces_count; i++) {

		for (int j = 0; j < 3; j++) {

			Vector3 &v = wrapped_faces_ptr[i].vertex[j];
			v = v * voxelsize;
			v += global_aabb.pos;
		}
	}

	// clean up grid
	//print_line("Wrapper (5/6): Grid Cleanup");

	for (int i = 0; i < div_x; i++) {

		for (int j = 0; j < div_y; j++) {

			memdelete_arr(cell_status[i][j]);
		}

		memdelete_arr(cell_status[i]);
	}

	memdelete_arr(cell_status);
	if (p_error)
		*p_error = voxelsize.length();

	//print_line("Wrapper (6/6): Finished.");
	return wrapped_faces;
}

Geometry::MeshData Geometry::build_convex_mesh(const PoolVector<Plane> &p_planes) {

	MeshData mesh;

#define SUBPLANE_SIZE 1024.0

	real_t subplane_size = 1024.0; // should compute this from the actual plane
	for (int i = 0; i < p_planes.size(); i++) {

		Plane p = p_planes[i];

		Vector3 ref = Vector3(0.0, 1.0, 0.0);

		if (ABS(p.normal.dot(ref)) > 0.95)
			ref = Vector3(0.0, 0.0, 1.0); // change axis

		Vector3 right = p.normal.cross(ref).normalized();
		Vector3 up = p.normal.cross(right).normalized();

		Vector<Vector3> vertices;

		Vector3 center = p.get_any_point();
		// make a quad clockwise
		vertices.push_back(center - up * subplane_size + right * subplane_size);
		vertices.push_back(center - up * subplane_size - right * subplane_size);
		vertices.push_back(center + up * subplane_size - right * subplane_size);
		vertices.push_back(center + up * subplane_size + right * subplane_size);

		for (int j = 0; j < p_planes.size(); j++) {

			if (j == i)
				continue;

			Vector<Vector3> new_vertices;
			Plane clip = p_planes[j];

			if (clip.normal.dot(p.normal) > 0.95)
				continue;

			if (vertices.size() < 3)
				break;

			for (int k = 0; k < vertices.size(); k++) {

				int k_n = (k + 1) % vertices.size();

				Vector3 edge0_A = vertices[k];
				Vector3 edge1_A = vertices[k_n];

				real_t dist0 = clip.distance_to(edge0_A);
				real_t dist1 = clip.distance_to(edge1_A);

				if (dist0 <= 0) { // behind plane

					new_vertices.push_back(vertices[k]);
				}

				// check for different sides and non coplanar
				if ((dist0 * dist1) < 0) {

					// calculate intersection
					Vector3 rel = edge1_A - edge0_A;

					real_t den = clip.normal.dot(rel);
					if (Math::abs(den) < CMP_EPSILON)
						continue; // point too short

					real_t dist = -(clip.normal.dot(edge0_A) - clip.d) / den;
					Vector3 inters = edge0_A + rel * dist;
					new_vertices.push_back(inters);
				}
			}

			vertices = new_vertices;
		}

		if (vertices.size() < 3)
			continue;

		//result is a clockwise face

		MeshData::Face face;

		// add face indices
		for (int j = 0; j < vertices.size(); j++) {

			int idx = -1;
			for (int k = 0; k < mesh.vertices.size(); k++) {

				if (mesh.vertices[k].distance_to(vertices[j]) < 0.001) {

					idx = k;
					break;
				}
			}

			if (idx == -1) {

				idx = mesh.vertices.size();
				mesh.vertices.push_back(vertices[j]);
			}

			face.indices.push_back(idx);
		}
		face.plane = p;
		mesh.faces.push_back(face);

		//add edge

		for (int j = 0; j < face.indices.size(); j++) {

			int a = face.indices[j];
			int b = face.indices[(j + 1) % face.indices.size()];

			bool found = false;
			for (int k = 0; k < mesh.edges.size(); k++) {

				if (mesh.edges[k].a == a && mesh.edges[k].b == b) {
					found = true;
					break;
				}
				if (mesh.edges[k].b == a && mesh.edges[k].a == b) {
					found = true;
					break;
				}
			}

			if (found)
				continue;
			MeshData::Edge edge;
			edge.a = a;
			edge.b = b;
			mesh.edges.push_back(edge);
		}
	}

	return mesh;
}

PoolVector<Plane> Geometry::build_box_planes(const Vector3 &p_extents) {

	PoolVector<Plane> planes;

	planes.push_back(Plane(Vector3(1, 0, 0), p_extents.x));
	planes.push_back(Plane(Vector3(-1, 0, 0), p_extents.x));
	planes.push_back(Plane(Vector3(0, 1, 0), p_extents.y));
	planes.push_back(Plane(Vector3(0, -1, 0), p_extents.y));
	planes.push_back(Plane(Vector3(0, 0, 1), p_extents.z));
	planes.push_back(Plane(Vector3(0, 0, -1), p_extents.z));

	return planes;
}

PoolVector<Plane> Geometry::build_cylinder_planes(real_t p_radius, real_t p_height, int p_sides, Vector3::Axis p_axis) {

	PoolVector<Plane> planes;

	for (int i = 0; i < p_sides; i++) {

		Vector3 normal;
		normal[(p_axis + 1) % 3] = Math::cos(i * (2.0 * Math_PI) / p_sides);
		normal[(p_axis + 2) % 3] = Math::sin(i * (2.0 * Math_PI) / p_sides);

		planes.push_back(Plane(normal, p_radius));
	}

	Vector3 axis;
	axis[p_axis] = 1.0;

	planes.push_back(Plane(axis, p_height * 0.5));
	planes.push_back(Plane(-axis, p_height * 0.5));

	return planes;
}

PoolVector<Plane> Geometry::build_sphere_planes(real_t p_radius, int p_lats, int p_lons, Vector3::Axis p_axis) {

	PoolVector<Plane> planes;

	Vector3 axis;
	axis[p_axis] = 1.0;

	Vector3 axis_neg;
	axis_neg[(p_axis + 1) % 3] = 1.0;
	axis_neg[(p_axis + 2) % 3] = 1.0;
	axis_neg[p_axis] = -1.0;

	for (int i = 0; i < p_lons; i++) {

		Vector3 normal;
		normal[(p_axis + 1) % 3] = Math::cos(i * (2.0 * Math_PI) / p_lons);
		normal[(p_axis + 2) % 3] = Math::sin(i * (2.0 * Math_PI) / p_lons);

		planes.push_back(Plane(normal, p_radius));

		for (int j = 1; j <= p_lats; j++) {

			//todo this is stupid, fix
			Vector3 angle = normal.linear_interpolate(axis, j / (real_t)p_lats).normalized();
			Vector3 pos = angle * p_radius;
			planes.push_back(Plane(pos, angle));
			planes.push_back(Plane(pos * axis_neg, angle * axis_neg));
		}
	}

	return planes;
}

PoolVector<Plane> Geometry::build_capsule_planes(real_t p_radius, real_t p_height, int p_sides, int p_lats, Vector3::Axis p_axis) {

	PoolVector<Plane> planes;

	Vector3 axis;
	axis[p_axis] = 1.0;

	Vector3 axis_neg;
	axis_neg[(p_axis + 1) % 3] = 1.0;
	axis_neg[(p_axis + 2) % 3] = 1.0;
	axis_neg[p_axis] = -1.0;

	for (int i = 0; i < p_sides; i++) {

		Vector3 normal;
		normal[(p_axis + 1) % 3] = Math::cos(i * (2.0 * Math_PI) / p_sides);
		normal[(p_axis + 2) % 3] = Math::sin(i * (2.0 * Math_PI) / p_sides);

		planes.push_back(Plane(normal, p_radius));

		for (int j = 1; j <= p_lats; j++) {

			Vector3 angle = normal.linear_interpolate(axis, j / (real_t)p_lats).normalized();
			Vector3 pos = axis * p_height * 0.5 + angle * p_radius;
			planes.push_back(Plane(pos, angle));
			planes.push_back(Plane(pos * axis_neg, angle * axis_neg));
		}
	}

	return planes;
}

struct _AtlasWorkRect {

	Size2i s;
	Point2i p;
	int idx;
	_FORCE_INLINE_ bool operator<(const _AtlasWorkRect &p_r) const { return s.width > p_r.s.width; };
};

struct _AtlasWorkRectResult {

	Vector<_AtlasWorkRect> result;
	int max_w;
	int max_h;
};

void Geometry::make_atlas(const Vector<Size2i> &p_rects, Vector<Point2i> &r_result, Size2i &r_size) {

	//super simple, almost brute force scanline stacking fitter
	//it's pretty basic for now, but it tries to make sure that the aspect ratio of the
	//resulting atlas is somehow square. This is necessary because video cards have limits
	//on texture size (usually 2048 or 4096), so the more square a texture, the more chances
	//it will work in every hardware.
	// for example, it will prioritize a 1024x1024 atlas (works everywhere) instead of a
	// 256x8192 atlas (won't work anywhere).

	ERR_FAIL_COND(p_rects.size() == 0);

	Vector<_AtlasWorkRect> wrects;
	wrects.resize(p_rects.size());
	for (int i = 0; i < p_rects.size(); i++) {
		wrects[i].s = p_rects[i];
		wrects[i].idx = i;
	}
	wrects.sort();
	int widest = wrects[0].s.width;

	Vector<_AtlasWorkRectResult> results;

	for (int i = 0; i <= 12; i++) {

		int w = 1 << i;
		int max_h = 0;
		int max_w = 0;
		if (w < widest)
			continue;

		Vector<int> hmax;
		hmax.resize(w);
		for (int j = 0; j < w; j++)
			hmax[j] = 0;

		//place them
		int ofs = 0;
		int limit_h = 0;
		for (int j = 0; j < wrects.size(); j++) {

			if (ofs + wrects[j].s.width > w) {

				ofs = 0;
			}

			int from_y = 0;
			for (int k = 0; k < wrects[j].s.width; k++) {

				if (hmax[ofs + k] > from_y)
					from_y = hmax[ofs + k];
			}

			wrects[j].p.x = ofs;
			wrects[j].p.y = from_y;
			int end_h = from_y + wrects[j].s.height;
			int end_w = ofs + wrects[j].s.width;
			if (ofs == 0)
				limit_h = end_h;

			for (int k = 0; k < wrects[j].s.width; k++) {

				hmax[ofs + k] = end_h;
			}

			if (end_h > max_h)
				max_h = end_h;

			if (end_w > max_w)
				max_w = end_w;

			if (ofs == 0 || end_h > limit_h) //while h limit not reached, keep stacking
				ofs += wrects[j].s.width;
		}

		_AtlasWorkRectResult result;
		result.result = wrects;
		result.max_h = max_h;
		result.max_w = max_w;
		results.push_back(result);
	}

	//find the result with the best aspect ratio

	int best = -1;
	real_t best_aspect = 1e20;

	for (int i = 0; i < results.size(); i++) {

		real_t h = nearest_power_of_2(results[i].max_h);
		real_t w = nearest_power_of_2(results[i].max_w);
		real_t aspect = h > w ? h / w : w / h;
		if (aspect < best_aspect) {
			best = i;
			best_aspect = aspect;
		}
	}

	r_result.resize(p_rects.size());

	for (int i = 0; i < p_rects.size(); i++) {

		r_result[results[best].result[i].idx] = results[best].result[i].p;
	}

	r_size = Size2(results[best].max_w, results[best].max_h);
}
