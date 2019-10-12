/*************************************************************************/
/*  csg.cpp                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "csg.h"
#include "core/math/face3.h"
#include "core/math/geometry.h"
#include "core/os/os.h"
#include "core/sort_array.h"
#include "thirdparty/misc/triangulator.h"

void CSGBrush::clear() {
	faces.clear();
}

void CSGBrush::build_from_faces(const PoolVector<Vector3> &p_vertices, const PoolVector<Vector2> &p_uvs, const PoolVector<bool> &p_smooth, const PoolVector<Ref<Material> > &p_materials, const PoolVector<bool> &p_invert_faces) {

	clear();

	int vc = p_vertices.size();

	ERR_FAIL_COND((vc % 3) != 0);

	PoolVector<Vector3>::Read rv = p_vertices.read();
	int uvc = p_uvs.size();
	PoolVector<Vector2>::Read ruv = p_uvs.read();
	int sc = p_smooth.size();
	PoolVector<bool>::Read rs = p_smooth.read();
	int mc = p_materials.size();
	PoolVector<Ref<Material> >::Read rm = p_materials.read();
	int ic = p_invert_faces.size();
	PoolVector<bool>::Read ri = p_invert_faces.read();

	Map<Ref<Material>, int> material_map;

	faces.resize(p_vertices.size() / 3);

	for (int i = 0; i < faces.size(); i++) {
		Face &f = faces.write[i];
		f.vertices[0] = rv[i * 3 + 0];
		f.vertices[1] = rv[i * 3 + 1];
		f.vertices[2] = rv[i * 3 + 2];
		if (uvc == vc) {
			f.uvs[0] = ruv[i * 3 + 0];
			f.uvs[1] = ruv[i * 3 + 1];
			f.uvs[2] = ruv[i * 3 + 2];
		}
		if (sc == vc / 3) {
			f.smooth = rs[i];
		} else {
			f.smooth = false;
		}

		if (ic == vc / 3) {
			f.invert = ri[i];
		} else {
			f.invert = false;
		}

		if (mc == vc / 3) {
			Ref<Material> mat = rm[i];
			if (mat.is_valid()) {
				const Map<Ref<Material>, int>::Element *E = material_map.find(mat);
				if (E) {
					f.material = E->get();
				} else {
					f.material = material_map.size();
					material_map[mat] = f.material;
				}
			} else {
				f.material = -1;
			}
		}
	}

	materials.resize(material_map.size());
	for (Map<Ref<Material>, int>::Element *E = material_map.front(); E; E = E->next()) {
		materials.write[E->get()] = E->key();
	}

	_regen_face_aabbs();
}

void CSGBrush::_regen_face_aabbs() {

	for (int i = 0; i < faces.size(); i++) {

		faces.write[i].aabb.position = faces[i].vertices[0];
		faces.write[i].aabb.expand_to(faces[i].vertices[1]);
		faces.write[i].aabb.expand_to(faces[i].vertices[2]);
		faces.write[i].aabb.grow_by(faces[i].aabb.get_longest_axis_size() * 0.001); //make it a tad bigger to avoid num precision errors
	}
}

void CSGBrush::copy_from(const CSGBrush &p_brush, const Transform &p_xform) {

	faces = p_brush.faces;
	materials = p_brush.materials;

	for (int i = 0; i < faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			faces.write[i].vertices[j] = p_xform.xform(p_brush.faces[i].vertices[j]);
		}
	}

	_regen_face_aabbs();
}

////////////////////////

void CSGBrushOperation::BuildPoly::create(const CSGBrush *p_brush, int p_face, MeshMerge &mesh_merge, bool p_for_B) {

	//creates the initial face that will be used for clipping against the other faces

	Vector3 va[3] = {
		p_brush->faces[p_face].vertices[0],
		p_brush->faces[p_face].vertices[1],
		p_brush->faces[p_face].vertices[2],
	};

	plane = Plane(va[0], va[1], va[2]);

	to_world.origin = va[0];

	to_world.basis.set_axis(2, plane.normal);
	to_world.basis.set_axis(0, (va[1] - va[2]).normalized());
	to_world.basis.set_axis(1, to_world.basis.get_axis(0).cross(to_world.basis.get_axis(2)).normalized());

	to_poly = to_world.affine_inverse();

	face_index = p_face;

	for (int i = 0; i < 3; i++) {

		Point p;
		Vector3 localp = to_poly.xform(va[i]);
		p.point.x = localp.x;
		p.point.y = localp.y;
		p.uv = p_brush->faces[p_face].uvs[i];

		points.push_back(p);

		///edge

		Edge e;
		e.points[0] = i;
		e.points[1] = (i + 1) % 3;
		e.outer = true;
		edges.push_back(e);
	}

	smooth = p_brush->faces[p_face].smooth;
	invert = p_brush->faces[p_face].invert;

	if (p_brush->faces[p_face].material != -1) {
		material = p_brush->materials[p_brush->faces[p_face].material];
	}

	base_edges = 3;
}

static Vector2 interpolate_uv(const Vector2 &p_vertex_a, const Vector2 &p_vertex_b, const Vector2 &p_vertex_c, const Vector2 &p_uv_a, const Vector2 &p_uv_c) {

	float len_a_c = (p_vertex_c - p_vertex_a).length();
	if (len_a_c < CMP_EPSILON) {
		return p_uv_a;
	}

	float len_a_b = (p_vertex_b - p_vertex_a).length();

	float c = len_a_b / len_a_c;

	return p_uv_a.linear_interpolate(p_uv_c, c);
}

static Vector2 interpolate_triangle_uv(const Vector2 &p_pos, const Vector2 *p_vtx, const Vector2 *p_uv) {

	if (p_pos.distance_squared_to(p_vtx[0]) < CMP_EPSILON2) {
		return p_uv[0];
	}
	if (p_pos.distance_squared_to(p_vtx[1]) < CMP_EPSILON2) {
		return p_uv[1];
	}
	if (p_pos.distance_squared_to(p_vtx[2]) < CMP_EPSILON2) {
		return p_uv[2];
	}

	Vector2 v0 = p_vtx[1] - p_vtx[0];
	Vector2 v1 = p_vtx[2] - p_vtx[0];
	Vector2 v2 = p_pos - p_vtx[0];

	float d00 = v0.dot(v0);
	float d01 = v0.dot(v1);
	float d11 = v1.dot(v1);
	float d20 = v2.dot(v0);
	float d21 = v2.dot(v1);
	float denom = (d00 * d11 - d01 * d01);
	if (denom == 0) {
		return p_uv[0];
	}
	float v = (d11 * d20 - d01 * d21) / denom;
	float w = (d00 * d21 - d01 * d20) / denom;
	float u = 1.0f - v - w;

	return p_uv[0] * u + p_uv[1] * v + p_uv[2] * w;
}

void CSGBrushOperation::BuildPoly::_clip_segment(const CSGBrush *p_brush, int p_face, const Vector2 *segment, MeshMerge &mesh_merge, bool p_for_B) {

	//keep track of what was inserted
	Vector<int> inserted_points;

	//keep track of point indices for what was inserted, allowing reuse of points.
	int segment_idx[2] = { -1, -1 };

	//check if edge and poly share a vertex, of so, assign it to segment_idx
	for (int i = 0; i < points.size(); i++) {
		for (int j = 0; j < 2; j++) {
			if (segment[j] == points[i].point) {
				segment_idx[j] = i;
				inserted_points.push_back(i);
				break;
			}
		}
	}

	//check if both segment points are shared with other vertices
	if (segment_idx[0] != -1 && segment_idx[1] != -1) {

		if (segment_idx[0] == segment_idx[1]) {
			return; //segment was too tiny, both mapped to same point
		}

		bool found = false;

		//check if the segment already exists
		for (int i = 0; i < edges.size(); i++) {

			if (
					(edges[i].points[0] == segment_idx[0] && edges[i].points[1] == segment_idx[1]) ||
					(edges[i].points[0] == segment_idx[1] && edges[i].points[1] == segment_idx[0])) {
				found = true;
				break;
			}
		}

		if (found) {
			//it does already exist, do nothing
			return;
		}

		//directly add the new segment
		Edge new_edge;
		new_edge.points[0] = segment_idx[0];
		new_edge.points[1] = segment_idx[1];
		edges.push_back(new_edge);
		return;
	}

	//check edge by edge against the segment points to see if intersects

	for (int i = 0; i < base_edges; i++) {

		//if a point is shared with one of the edge points, then this edge must not be tested, as it will result in a numerical precision error.
		bool edge_valid = true;
		for (int j = 0; j < 2; j++) {

			if (edges[i].points[0] == segment_idx[0] || edges[i].points[1] == segment_idx[1] || edges[i].points[0] == segment_idx[1] || edges[i].points[1] == segment_idx[0]) {
				edge_valid = false; //segment has this point, can't check against this
				break;
			}
		}

		if (!edge_valid) //already hit a point in this edge, so don't test it
			continue;

		//see if either points are within the edge isntead of crossing it
		Vector2 res;
		bool found = false;
		int assign_segment_id = -1;

		for (int j = 0; j < 2; j++) {

			Vector2 edgeseg[2] = { points[edges[i].points[0]].point, points[edges[i].points[1]].point };
			Vector2 closest = Geometry::get_closest_point_to_segment_2d(segment[j], edgeseg);

			if (closest == segment[j]) {
				//point rest of this edge
				res = closest;
				found = true;
				assign_segment_id = j;
			}
		}

		//test if the point crosses the edge
		if (!found && Geometry::segment_intersects_segment_2d(segment[0], segment[1], points[edges[i].points[0]].point, points[edges[i].points[1]].point, &res)) {
			//point does cross the edge
			found = true;
		}

		//check whether an intersection against the segment happened
		if (found) {

			//It did! so first, must slice the segment
			Point new_point;
			new_point.point = res;
			//make sure to interpolate UV too
			new_point.uv = interpolate_uv(points[edges[i].points[0]].point, new_point.point, points[edges[i].points[1]].point, points[edges[i].points[0]].uv, points[edges[i].points[1]].uv);

			int point_idx = points.size();
			points.push_back(new_point);

			//split the edge in 2
			Edge new_edge;
			new_edge.points[0] = edges[i].points[0];
			new_edge.points[1] = point_idx;
			new_edge.outer = edges[i].outer;
			edges.write[i].points[0] = point_idx;
			edges.insert(i, new_edge);
			i++; //skip newly inserted edge
			base_edges++; //will need an extra one in the base triangle
			if (assign_segment_id >= 0) {
				//point did split a segment, so make sure to remember this
				segment_idx[assign_segment_id] = point_idx;
			}
			inserted_points.push_back(point_idx);
		}
	}

	//final step: after cutting the original triangle, try to see if we can still insert
	//this segment

	//if already inserted two points, just use them for a segment

	if (inserted_points.size() >= 2) { //should never be >2 on non-manifold geometry, but cope with error
		//two points were inserted, create the new edge
		Edge new_edge;
		new_edge.points[0] = inserted_points[0];
		new_edge.points[1] = inserted_points[1];
		edges.push_back(new_edge);
		return;
	}

	// One or no points were inserted (besides splitting), so try to see if extra points can be placed inside the triangle.
	// This needs to be done here, after the previous tests were exhausted
	for (int i = 0; i < 2; i++) {

		if (segment_idx[i] != -1)
			continue; //already assigned to something, so skip

		//check whether one of the segment endpoints is inside the triangle. If it is, this points needs to be inserted
		if (Geometry::is_point_in_triangle(segment[i], points[0].point, points[1].point, points[2].point)) {

			Point new_point;
			new_point.point = segment[i];

			Vector2 point3[3] = { points[0].point, points[1].point, points[2].point };
			Vector2 uv3[3] = { points[0].uv, points[1].uv, points[2].uv };

			new_point.uv = interpolate_triangle_uv(new_point.point, point3, uv3);

			int point_idx = points.size();
			points.push_back(new_point);
			inserted_points.push_back(point_idx);
		}
	}

	//check again whether two points were inserted, if so then create the new edge
	if (inserted_points.size() >= 2) { //should never be >2 on non-manifold geometry, but cope with error
		Edge new_edge;
		new_edge.points[0] = inserted_points[0];
		new_edge.points[1] = inserted_points[1];
		edges.push_back(new_edge);
	}
}

void CSGBrushOperation::BuildPoly::clip(const CSGBrush *p_brush, int p_face, MeshMerge &mesh_merge, bool p_for_B) {

	//Clip function.. find triangle points that will be mapped to the plane and form a segment

	Vector2 segment[3]; //2D

	int src_points = 0;

	for (int i = 0; i < 3; i++) {
		Vector3 p = p_brush->faces[p_face].vertices[i];
		if (plane.has_point(p)) {
			Vector3 pp = plane.project(p);
			pp = to_poly.xform(pp);
			segment[src_points++] = Vector2(pp.x, pp.y);
		} else {
			Vector3 q = p_brush->faces[p_face].vertices[(i + 1) % 3];
			if (plane.has_point(q))
				continue; //next point is in plane, will be added eventually
			if (plane.is_point_over(p) == plane.is_point_over(q))
				continue; // both on same side of the plane, don't add

			Vector3 res;
			if (plane.intersects_segment(p, q, &res)) {
				res = to_poly.xform(res);
				segment[src_points++] = Vector2(res.x, res.y);
			}
		}
	}

	//all above or all below, nothing to do. Should not happen though since a precheck was done before.
	if (src_points == 0)
		return;

	//just one point in plane is not worth doing anything
	if (src_points == 1)
		return;

	//transform A points to 2D

	if (segment[0] == segment[1])
		return; //too small

	_clip_segment(p_brush, p_face, segment, mesh_merge, p_for_B);
}

void CSGBrushOperation::_collision_callback(const CSGBrush *A, int p_face_a, Map<int, BuildPoly> &build_polys_a, const CSGBrush *B, int p_face_b, Map<int, BuildPoly> &build_polys_b, MeshMerge &mesh_merge) {

	//construct a frame of reference for both transforms, in order to do intersection test
	Vector3 va[3] = {
		A->faces[p_face_a].vertices[0],
		A->faces[p_face_a].vertices[1],
		A->faces[p_face_a].vertices[2],
	};
	Vector3 vb[3] = {
		B->faces[p_face_b].vertices[0],
		B->faces[p_face_b].vertices[1],
		B->faces[p_face_b].vertices[2],
	};

	{
		//check if either is a degenerate
		if (va[0] == va[1] || va[0] == va[2] || va[1] == va[2])
			return;

		if (vb[0] == vb[1] || vb[0] == vb[2] || vb[1] == vb[2])
			return;
	}

	{
		//check if points are the same
		int equal_count = 0;

		for (int i = 0; i < 3; i++) {

			for (int j = 0; j < 3; j++) {
				if (va[i].distance_to(vb[j]) < mesh_merge.vertex_snap) {
					equal_count++;
					break;
				}
			}
		}

		//if 2 or 3 points are the same, there is no point in doing anything. They can't
		//be clipped either, so add both.
		if (equal_count == 2 || equal_count == 3) {
			return;
		}
	}

	// do a quick pre-check for no-intersection using the SAT theorem

	{

		//b under or over a plane
		int over_count = 0, in_plane_count = 0, under_count = 0;
		Plane plane_a(va[0], va[1], va[2]);
		if (plane_a.normal == Vector3()) {
			return; //degenerate
		}

		for (int i = 0; i < 3; i++) {
			if (plane_a.has_point(vb[i]))
				in_plane_count++;
			else if (plane_a.is_point_over(vb[i]))
				over_count++;
			else
				under_count++;
		}

		if (over_count == 0 || under_count == 0)
			return; //no intersection, something needs to be under AND over

		//a under or over b plane
		over_count = 0;
		under_count = 0;
		in_plane_count = 0;

		Plane plane_b(vb[0], vb[1], vb[2]);
		if (plane_b.normal == Vector3())
			return; //degenerate

		for (int i = 0; i < 3; i++) {
			if (plane_b.has_point(va[i]))
				in_plane_count++;
			else if (plane_b.is_point_over(va[i]))
				over_count++;
			else
				under_count++;
		}

		if (over_count == 0 || under_count == 0)
			return; //no intersection, something needs to be under AND over

		//edge pairs (cross product combinations), see SAT theorem

		for (int i = 0; i < 3; i++) {

			Vector3 axis_a = (va[i] - va[(i + 1) % 3]).normalized();

			for (int j = 0; j < 3; j++) {

				Vector3 axis_b = (vb[j] - vb[(j + 1) % 3]).normalized();

				Vector3 sep_axis = axis_a.cross(axis_b);
				if (sep_axis == Vector3())
					continue; //colineal
				sep_axis.normalize();

				real_t min_a = 1e20, max_a = -1e20;
				real_t min_b = 1e20, max_b = -1e20;

				for (int k = 0; k < 3; k++) {
					real_t d = sep_axis.dot(va[k]);
					min_a = MIN(min_a, d);
					max_a = MAX(max_a, d);
					d = sep_axis.dot(vb[k]);
					min_b = MIN(min_b, d);
					max_b = MAX(max_b, d);
				}

				min_b -= (max_a - min_a) * 0.5;
				max_b += (max_a - min_a) * 0.5;

				real_t dmin = min_b - (min_a + max_a) * 0.5;
				real_t dmax = max_b - (min_a + max_a) * 0.5;

				if (dmin > CMP_EPSILON || dmax < -CMP_EPSILON) {
					return; //does not contain zero, so they don't overlap
				}
			}
		}
	}

	//if we are still here, it means they most likely intersect, so create BuildPolys if they don't exist

	BuildPoly *poly_a = NULL;

	if (!build_polys_a.has(p_face_a)) {

		BuildPoly bp;
		bp.create(A, p_face_a, mesh_merge, false);
		build_polys_a[p_face_a] = bp;
	}

	poly_a = &build_polys_a[p_face_a];

	BuildPoly *poly_b = NULL;

	if (!build_polys_b.has(p_face_b)) {

		BuildPoly bp;
		bp.create(B, p_face_b, mesh_merge, true);
		build_polys_b[p_face_b] = bp;
	}

	poly_b = &build_polys_b[p_face_b];

	//clip each other, this could be improved by using vertex unique IDs (more vertices may be shared instead of using snap)
	poly_a->clip(B, p_face_b, mesh_merge, false);
	poly_b->clip(A, p_face_a, mesh_merge, true);
}

void CSGBrushOperation::_add_poly_points(const BuildPoly &p_poly, int p_edge, int p_from_point, int p_to_point, const Vector<Vector<int> > &vertex_process, Vector<bool> &edge_process, Vector<PolyPoints> &r_poly) {

	//this function follows the polygon points counter clockwise and adds them. It creates lists of unique polygons
	//every time an unused edge is found, it's pushed to a stack and continues from there.

	List<EdgeSort> edge_stack;

	{
		EdgeSort es;
		es.angle = 0; //won't be checked here
		es.edge = p_edge;
		es.prev_point = p_from_point;
		es.edge_point = p_to_point;

		edge_stack.push_back(es);
	}

	//attempt to empty the stack.
	while (edge_stack.size()) {

		EdgeSort e = edge_stack.front()->get();
		edge_stack.pop_front();

		if (edge_process[e.edge]) {
			//nothing to do here
			continue;
		}

		Vector<int> points;
		points.push_back(e.prev_point);

		int prev_point = e.prev_point;
		int to_point = e.edge_point;
		int current_edge = e.edge;

		edge_process.write[e.edge] = true; //mark as processed

		int limit = p_poly.points.size() * 4; //avoid infinite recursion

		while (to_point != e.prev_point && limit) {

			Vector2 segment[2] = { p_poly.points[prev_point].point, p_poly.points[to_point].point };

			//construct a basis transform from the segment, which will be used to check the angle
			Transform2D t2d;
			t2d[0] = (segment[1] - segment[0]).normalized(); //use as Y
			t2d[1] = Vector2(-t2d[0].y, t2d[0].x); // use as tangent
			t2d[2] = segment[1]; //origin

			if (t2d.basis_determinant() == 0)
				break; //abort poly

			t2d.affine_invert();

			//push all edges found here, they will be sorted by minimum angle later.
			Vector<EdgeSort> next_edges;

			for (int i = 0; i < vertex_process[to_point].size(); i++) {

				int edge = vertex_process[to_point][i];
				int opposite_point = p_poly.edges[edge].points[0] == to_point ? p_poly.edges[edge].points[1] : p_poly.edges[edge].points[0];
				if (opposite_point == prev_point)
					continue; //not going back

				EdgeSort e2;
				Vector2 local_vec = t2d.xform(p_poly.points[opposite_point].point);
				e2.angle = -local_vec.angle(); //negate so we can sort by minimum angle
				e2.edge = edge;
				e2.edge_point = opposite_point;
				e2.prev_point = to_point;

				next_edges.push_back(e2);
			}

			//finally, sort by minimum angle
			next_edges.sort();

			int next_point = -1;
			int next_edge = -1;

			for (int i = 0; i < next_edges.size(); i++) {

				if (i == 0) {
					//minimum angle found is the next point
					next_point = next_edges[i].edge_point;
					next_edge = next_edges[i].edge;

				} else {
					//the rest are pushed to the stack IF they were not processed yet.
					if (!edge_process[next_edges[i].edge]) {
						edge_stack.push_back(next_edges[i]);
					}
				}
			}

			if (next_edge == -1) {
				//did not find anything, may be a dead-end edge (this should normally not happen)
				//just flip the direction and go back
				next_point = prev_point;
				next_edge = current_edge;
			}

			points.push_back(to_point);

			prev_point = to_point;
			to_point = next_point;
			edge_process.write[next_edge] = true; //mark this edge as processed
			current_edge = next_edge;

			limit--;
		}

		//if more than 2 points were added to the polygon, add it to the list of polygons.
		if (points.size() > 2) {
			PolyPoints pp;
			pp.points = points;
			r_poly.push_back(pp);
		}
	}
}

void CSGBrushOperation::_add_poly_outline(const BuildPoly &p_poly, int p_from_point, int p_to_point, const Vector<Vector<int> > &vertex_process, Vector<int> &r_outline) {

	//this is the opposite of the function above. It adds polygon outlines instead.
	//this is used for triangulating holes.
	//no stack is used here because only the bigger outline is interesting.

	r_outline.push_back(p_from_point);

	int prev_point = p_from_point;
	int to_point = p_to_point;

	int limit = p_poly.points.size() * 4; //avoid infinite recursion

	while (to_point != p_from_point && limit) {

		Vector2 segment[2] = { p_poly.points[prev_point].point, p_poly.points[to_point].point };
		//again create a transform to compute the angle.
		Transform2D t2d;
		t2d[0] = (segment[1] - segment[0]).normalized(); //use as Y
		t2d[1] = Vector2(-t2d[0].y, t2d[0].x); // use as tangent
		t2d[2] = segment[1]; //origin

		if (t2d.basis_determinant() == 0)
			break; //abort poly

		t2d.affine_invert();

		float max_angle = 0;
		int next_point_angle = -1;

		for (int i = 0; i < vertex_process[to_point].size(); i++) {

			int edge = vertex_process[to_point][i];
			int opposite_point = p_poly.edges[edge].points[0] == to_point ? p_poly.edges[edge].points[1] : p_poly.edges[edge].points[0];
			if (opposite_point == prev_point)
				continue; //not going back

			float angle = -t2d.xform(p_poly.points[opposite_point].point).angle();
			if (next_point_angle == -1 || angle > max_angle) { //same as before but use greater to check.
				max_angle = angle;
				next_point_angle = opposite_point;
			}
		}

		if (next_point_angle == -1) {
			//go back because no route found
			next_point_angle = prev_point;
		}

		r_outline.push_back(to_point);
		prev_point = to_point;
		to_point = next_point_angle;

		limit--;
	}
}

void CSGBrushOperation::_merge_poly(MeshMerge &mesh, int p_face_idx, const BuildPoly &p_poly, bool p_from_b) {

	//finally, merge the 2D polygon back to 3D

	Vector<Vector<int> > vertex_process;
	Vector<bool> edge_process;

	vertex_process.resize(p_poly.points.size());
	edge_process.resize(p_poly.edges.size());

	//none processed by default
	for (int i = 0; i < edge_process.size(); i++) {
		edge_process.write[i] = false;
	}

	//put edges in points, so points can go through them
	for (int i = 0; i < p_poly.edges.size(); i++) {
		vertex_process.write[p_poly.edges[i].points[0]].push_back(i);
		vertex_process.write[p_poly.edges[i].points[1]].push_back(i);
	}

	Vector<PolyPoints> polys;

	//process points that were not processed
	for (int i = 0; i < edge_process.size(); i++) {
		if (edge_process[i])
			continue; //already processed

		int intersect_poly = -1;

		if (i > 0) {
			//this is disconnected, so it's clearly a hole. lets find where it belongs
			Vector2 ref_point = p_poly.points[p_poly.edges[i].points[0]].point;

			for (int j = 0; j < polys.size(); j++) {

				//find a point outside poly
				Vector2 out_point(-1e20, -1e20);

				const PolyPoints &pp = polys[j];

				for (int k = 0; k < pp.points.size(); k++) {
					Vector2 p = p_poly.points[pp.points[k]].point;
					out_point.x = MAX(out_point.x, p.x);
					out_point.y = MAX(out_point.y, p.y);
				}

				out_point += Vector2(0.12341234, 0.4123412); // move to a random place to avoid direct edge-point chances

				int intersections = 0;

				for (int k = 0; k < pp.points.size(); k++) {
					Vector2 p1 = p_poly.points[pp.points[k]].point;
					Vector2 p2 = p_poly.points[pp.points[(k + 1) % pp.points.size()]].point;

					if (Geometry::segment_intersects_segment_2d(ref_point, out_point, p1, p2, NULL)) {
						intersections++;
					}
				}

				if (intersections % 2 == 1) {
					//hole is inside this poly
					intersect_poly = j;
					break;
				}
			}
		}

		if (intersect_poly != -1) {
			//must add this as a hole
			Vector<int> outline;
			_add_poly_outline(p_poly, p_poly.edges[i].points[0], p_poly.edges[i].points[1], vertex_process, outline);

			if (outline.size() > 1) {
				polys.write[intersect_poly].holes.push_back(outline);
			}
		}
		_add_poly_points(p_poly, i, p_poly.edges[i].points[0], p_poly.edges[i].points[1], vertex_process, edge_process, polys);
	}

	//get rid of holes, not the most optiomal way, but also not a common case at all to be inoptimal
	for (int i = 0; i < polys.size(); i++) {

		if (!polys[i].holes.size())
			continue;

		//repeat until no more holes are left to be merged
		while (polys[i].holes.size()) {

			//try to merge a hole with the outline
			bool added_hole = false;

			for (int j = 0; j < polys[i].holes.size(); j++) {

				//try hole vertices
				int with_outline_vertex = -1;
				int from_hole_vertex = -1;

				bool found = false;

				for (int k = 0; k < polys[i].holes[j].size(); k++) {

					int from_idx = polys[i].holes[j][k];
					Vector2 from = p_poly.points[from_idx].point;

					//try a segment from hole vertex to outline vertices
					from_hole_vertex = k;

					bool valid = true;

					for (int l = 0; l < polys[i].points.size(); l++) {

						int to_idx = polys[i].points[l];
						Vector2 to = p_poly.points[to_idx].point;
						with_outline_vertex = l;

						//try against outline (other points) first

						valid = true;

						for (int m = 0; m < polys[i].points.size(); m++) {

							int m_next = (m + 1) % polys[i].points.size();
							if (m == with_outline_vertex || m_next == with_outline_vertex) //do not test with edges that share this point
								continue;

							if (Geometry::segment_intersects_segment_2d(from, to, p_poly.points[polys[i].points[m]].point, p_poly.points[polys[i].points[m_next]].point, NULL)) {
								valid = false;
								break;
							}
						}

						if (!valid)
							continue;

						//try against all holes including self

						for (int m = 0; m < polys[i].holes.size(); m++) {

							for (int n = 0; n < polys[i].holes[m].size(); n++) {

								int n_next = (n + 1) % polys[i].holes[m].size();
								if (m == j && (n == from_hole_vertex || n_next == from_hole_vertex)) //contains vertex being tested from current hole, skip
									continue;

								if (Geometry::segment_intersects_segment_2d(from, to, p_poly.points[polys[i].holes[m][n]].point, p_poly.points[polys[i].holes[m][n_next]].point, NULL)) {
									valid = false;
									break;
								}
							}

							if (!valid)
								break;
						}

						if (valid) //all passed! exit loop
							break;
						else
							continue; //something went wrong, go on.
					}

					if (valid) {
						found = true; //if in the end this was valid, use it
						break;
					}
				}

				if (found) {

					//hook this hole with outline, and remove from list of holes

					//duplicate point
					int insert_at = with_outline_vertex;
					int point = polys[i].points[insert_at];
					polys.write[i].points.insert(insert_at, point);
					insert_at++;
					//insert all others, outline should be backwards (must check)
					int holesize = polys[i].holes[j].size();
					for (int k = 0; k <= holesize; k++) {
						int idx = (from_hole_vertex + k) % holesize;
						int point2 = polys[i].holes[j][idx];
						polys.write[i].points.insert(insert_at, point2);
						insert_at++;
					}

					added_hole = true;
					polys.write[i].holes.remove(j);
					break; //got rid of hole, break and continue
				}
			}

			ERR_BREAK(!added_hole);
		}
	}

	//triangulate polygons

	for (int i = 0; i < polys.size(); i++) {

		Vector<Vector2> vertices;
		vertices.resize(polys[i].points.size());
		for (int j = 0; j < vertices.size(); j++) {
			vertices.write[j] = p_poly.points[polys[i].points[j]].point;
		}

		Vector<int> indices = Geometry::triangulate_polygon(vertices);

		for (int j = 0; j < indices.size(); j += 3) {

			//obtain the vertex

			Vector3 face[3];
			Vector2 uv[3];
			float cp = Geometry::vec2_cross(p_poly.points[polys[i].points[indices[j + 0]]].point, p_poly.points[polys[i].points[indices[j + 1]]].point, p_poly.points[polys[i].points[indices[j + 2]]].point);
			if (Math::abs(cp) < CMP_EPSILON)
				continue;

			for (int k = 0; k < 3; k++) {

				Vector2 p = p_poly.points[polys[i].points[indices[j + k]]].point;
				face[k] = p_poly.to_world.xform(Vector3(p.x, p.y, 0));
				uv[k] = p_poly.points[polys[i].points[indices[j + k]]].uv;
			}

			mesh.add_face(face[0], face[1], face[2], uv[0], uv[1], uv[2], p_poly.smooth, p_poly.invert, p_poly.material, p_from_b);
		}
	}
}

//use a limit to speed up bvh and limit the depth
#define BVH_LIMIT 8

int CSGBrushOperation::MeshMerge::_create_bvh(BVH *p_bvh, BVH **p_bb, int p_from, int p_size, int p_depth, int &max_depth, int &max_alloc) {

	if (p_depth > max_depth) {
		max_depth = p_depth;
	}

	if (p_size == 0) {

		return -1;
	} else if (p_size <= BVH_LIMIT) {

		for (int i = 0; i < p_size - 1; i++) {
			p_bb[p_from + i]->next = p_bb[p_from + i + 1] - p_bvh;
		}
		return p_bb[p_from] - p_bvh;
	}

	AABB aabb;
	aabb = p_bb[p_from]->aabb;
	for (int i = 1; i < p_size; i++) {

		aabb.merge_with(p_bb[p_from + i]->aabb);
	}

	int li = aabb.get_longest_axis_index();

	switch (li) {

		case Vector3::AXIS_X: {
			SortArray<BVH *, BVHCmpX> sort_x;
			sort_x.nth_element(0, p_size, p_size / 2, &p_bb[p_from]);
			//sort_x.sort(&p_bb[p_from],p_size);
		} break;
		case Vector3::AXIS_Y: {
			SortArray<BVH *, BVHCmpY> sort_y;
			sort_y.nth_element(0, p_size, p_size / 2, &p_bb[p_from]);
			//sort_y.sort(&p_bb[p_from],p_size);
		} break;
		case Vector3::AXIS_Z: {
			SortArray<BVH *, BVHCmpZ> sort_z;
			sort_z.nth_element(0, p_size, p_size / 2, &p_bb[p_from]);
			//sort_z.sort(&p_bb[p_from],p_size);

		} break;
	}

	int left = _create_bvh(p_bvh, p_bb, p_from, p_size / 2, p_depth + 1, max_depth, max_alloc);
	int right = _create_bvh(p_bvh, p_bb, p_from + p_size / 2, p_size - p_size / 2, p_depth + 1, max_depth, max_alloc);

	int index = max_alloc++;
	BVH *_new = &p_bvh[index];
	_new->aabb = aabb;
	_new->center = aabb.position + aabb.size * 0.5;
	_new->face = -1;
	_new->left = left;
	_new->right = right;
	_new->next = -1;

	return index;
}

int CSGBrushOperation::MeshMerge::_bvh_count_intersections(BVH *bvhptr, int p_max_depth, int p_bvh_first, const Vector3 &p_begin, const Vector3 &p_end, int p_exclude) const {

	uint32_t *stack = (uint32_t *)alloca(sizeof(int) * p_max_depth);

	enum {
		TEST_AABB_BIT = 0,
		VISIT_LEFT_BIT = 1,
		VISIT_RIGHT_BIT = 2,
		VISIT_DONE_BIT = 3,
		VISITED_BIT_SHIFT = 29,
		NODE_IDX_MASK = (1 << VISITED_BIT_SHIFT) - 1,
		VISITED_BIT_MASK = ~NODE_IDX_MASK,

	};

	int intersections = 0;

	int level = 0;

	const Vector3 *vertexptr = points.ptr();
	const Face *facesptr = faces.ptr();
	AABB segment_aabb;
	segment_aabb.position = p_begin;
	segment_aabb.expand_to(p_end);

	int pos = p_bvh_first;

	stack[0] = pos;
	while (true) {

		uint32_t node = stack[level] & NODE_IDX_MASK;
		const BVH &b = bvhptr[node];
		bool done = false;

		switch (stack[level] >> VISITED_BIT_SHIFT) {
			case TEST_AABB_BIT: {

				if (b.face >= 0) {

					const BVH *bp = &b;

					while (bp) {

						bool valid = segment_aabb.intersects(bp->aabb) && bp->aabb.intersects_segment(p_begin, p_end);

						if (valid && p_exclude != bp->face) {
							const Face &s = facesptr[bp->face];
							Face3 f3(vertexptr[s.points[0]], vertexptr[s.points[1]], vertexptr[s.points[2]]);

							Vector3 res;

							if (f3.intersects_segment(p_begin, p_end, &res)) {
								intersections++;
							}
						}
						if (bp->next != -1) {
							bp = &bvhptr[bp->next];
						} else {
							bp = NULL;
						}
					}

					stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;

				} else {

					bool valid = segment_aabb.intersects(b.aabb) && b.aabb.intersects_segment(p_begin, p_end);

					if (!valid) {

						stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;

					} else {
						stack[level] = (VISIT_LEFT_BIT << VISITED_BIT_SHIFT) | node;
					}
				}
				continue;
			}
			case VISIT_LEFT_BIT: {

				stack[level] = (VISIT_RIGHT_BIT << VISITED_BIT_SHIFT) | node;
				stack[level + 1] = b.left | TEST_AABB_BIT;
				level++;
				continue;
			}
			case VISIT_RIGHT_BIT: {

				stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;
				stack[level + 1] = b.right | TEST_AABB_BIT;
				level++;
				continue;
			}
			case VISIT_DONE_BIT: {

				if (level == 0) {
					done = true;
					break;
				} else
					level--;
				continue;
			}
		}

		if (done)
			break;
	}

	return intersections;
}

void CSGBrushOperation::MeshMerge::mark_inside_faces() {

	// mark faces that are inside. This helps later do the boolean ops when merging.
	// this approach is very brute force (with a bunch of optimizatios, such as BVH and pre AABB intersection test)

	AABB aabb;

	for (int i = 0; i < points.size(); i++) {
		if (i == 0) {
			aabb.position = points[i];
		} else {
			aabb.expand_to(points[i]);
		}
	}

	float max_distance = aabb.size.length() * 1.2;

	Vector<BVH> bvhvec;
	bvhvec.resize(faces.size() * 3); //will never be larger than this (todo make better)
	BVH *bvh = bvhvec.ptrw();

	AABB faces_a;
	AABB faces_b;

	bool first_a = true;
	bool first_b = true;

	for (int i = 0; i < faces.size(); i++) {
		bvh[i].left = -1;
		bvh[i].right = -1;
		bvh[i].face = i;
		bvh[i].aabb.position = points[faces[i].points[0]];
		bvh[i].aabb.expand_to(points[faces[i].points[1]]);
		bvh[i].aabb.expand_to(points[faces[i].points[2]]);
		bvh[i].center = bvh[i].aabb.position + bvh[i].aabb.size * 0.5;
		bvh[i].next = -1;
		if (faces[i].from_b) {
			if (first_b) {
				faces_b = bvh[i].aabb;
				first_b = false;
			} else {
				faces_b.merge_with(bvh[i].aabb);
			}
		} else {
			if (first_a) {
				faces_a = bvh[i].aabb;
				first_a = false;
			} else {
				faces_a.merge_with(bvh[i].aabb);
			}
		}
	}

	AABB intersection_aabb = faces_a.intersection(faces_b);
	intersection_aabb.grow_by(intersection_aabb.get_longest_axis_size() * 0.01); //grow a little, avoid numerical error

	if (intersection_aabb.size == Vector3()) //AABB do not intersect, so neither do shapes.
		return;

	Vector<BVH *> bvhtrvec;
	bvhtrvec.resize(faces.size());
	BVH **bvhptr = bvhtrvec.ptrw();
	for (int i = 0; i < faces.size(); i++) {

		bvhptr[i] = &bvh[i];
	}

	int max_depth = 0;
	int max_alloc = faces.size();
	_create_bvh(bvh, bvhptr, 0, faces.size(), 1, max_depth, max_alloc);

	for (int i = 0; i < faces.size(); i++) {

		if (!intersection_aabb.intersects(bvh[i].aabb))
			continue; //not in AABB intersection, so not in face intersection
		Vector3 center = points[faces[i].points[0]];
		center += points[faces[i].points[1]];
		center += points[faces[i].points[2]];
		center /= 3.0;

		Plane plane(points[faces[i].points[0]], points[faces[i].points[1]], points[faces[i].points[2]]);
		Vector3 target = center + plane.normal * max_distance + Vector3(0.0001234, 0.000512, 0.00013423); //reduce chance of edge hits by doing a small increment

		int intersections = _bvh_count_intersections(bvh, max_depth, max_alloc - 1, center, target, i);

		if (intersections & 1) {
			faces.write[i].inside = true;
		}
	}
}

void CSGBrushOperation::MeshMerge::add_face(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, const Vector2 &p_uv_a, const Vector2 &p_uv_b, const Vector2 &p_uv_c, bool p_smooth, bool p_invert, const Ref<Material> &p_material, bool p_from_b) {

	Vector3 src_points[3] = { p_a, p_b, p_c };
	Vector2 src_uvs[3] = { p_uv_a, p_uv_b, p_uv_c };
	int indices[3];
	for (int i = 0; i < 3; i++) {

		VertexKey vk;
		vk.x = int((double(src_points[i].x) + double(vertex_snap) * 0.31234) / double(vertex_snap));
		vk.y = int((double(src_points[i].y) + double(vertex_snap) * 0.31234) / double(vertex_snap));
		vk.z = int((double(src_points[i].z) + double(vertex_snap) * 0.31234) / double(vertex_snap));

		int res;
		if (snap_cache.lookup(vk, res)) {
			indices[i] = res;
		} else {
			indices[i] = points.size();
			points.push_back(src_points[i]);
			snap_cache.set(vk, indices[i]);
		}
	}

	if (indices[0] == indices[2] || indices[0] == indices[1] || indices[1] == indices[2])
		return; //not adding degenerate

	MeshMerge::Face face;
	face.from_b = p_from_b;
	face.inside = false;
	face.smooth = p_smooth;
	face.invert = p_invert;
	if (p_material.is_valid()) {
		if (!materials.has(p_material)) {
			face.material_idx = materials.size();
			materials[p_material] = face.material_idx;
		} else {
			face.material_idx = materials[p_material];
		}
	} else {
		face.material_idx = -1;
	}

	for (int k = 0; k < 3; k++) {

		face.points[k] = indices[k];
		face.uvs[k] = src_uvs[k];
		;
	}

	faces.push_back(face);
}

void CSGBrushOperation::merge_brushes(Operation p_operation, const CSGBrush &p_A, const CSGBrush &p_B, CSGBrush &result, float p_snap) {

	CallbackData cd;
	cd.self = this;
	cd.A = &p_A;
	cd.B = &p_B;

	MeshMerge mesh_merge;
	mesh_merge.vertex_snap = p_snap;

	//check intersections between faces. Use AABB to speed up precheck
	//this generates list of buildpolys and clips them.
	//this was originally BVH optimized, but its not really worth it.
	for (int i = 0; i < p_A.faces.size(); i++) {
		cd.face_a = i;
		for (int j = 0; j < p_B.faces.size(); j++) {
			if (p_A.faces[i].aabb.intersects(p_B.faces[j].aabb)) {
				_collision_callback(&p_A, i, cd.build_polys_A, &p_B, j, cd.build_polys_B, mesh_merge);
			}
		}
	}

	//merge the already cliped polys back to 3D
	for (Map<int, BuildPoly>::Element *E = cd.build_polys_A.front(); E; E = E->next()) {
		_merge_poly(mesh_merge, E->key(), E->get(), false);
	}

	for (Map<int, BuildPoly>::Element *E = cd.build_polys_B.front(); E; E = E->next()) {
		_merge_poly(mesh_merge, E->key(), E->get(), true);
	}

	//merge the non clipped faces back

	for (int i = 0; i < p_A.faces.size(); i++) {

		if (cd.build_polys_A.has(i))
			continue; //made from buildpoly, skipping

		Vector3 points[3];
		Vector2 uvs[3];
		for (int j = 0; j < 3; j++) {
			points[j] = p_A.faces[i].vertices[j];
			uvs[j] = p_A.faces[i].uvs[j];
		}
		Ref<Material> material;
		if (p_A.faces[i].material != -1) {
			material = p_A.materials[p_A.faces[i].material];
		}
		mesh_merge.add_face(points[0], points[1], points[2], uvs[0], uvs[1], uvs[2], p_A.faces[i].smooth, p_A.faces[i].invert, material, false);
	}

	for (int i = 0; i < p_B.faces.size(); i++) {

		if (cd.build_polys_B.has(i))
			continue; //made from buildpoly, skipping

		Vector3 points[3];
		Vector2 uvs[3];
		for (int j = 0; j < 3; j++) {
			points[j] = p_B.faces[i].vertices[j];
			uvs[j] = p_B.faces[i].uvs[j];
		}
		Ref<Material> material;
		if (p_B.faces[i].material != -1) {
			material = p_B.materials[p_B.faces[i].material];
		}
		mesh_merge.add_face(points[0], points[1], points[2], uvs[0], uvs[1], uvs[2], p_B.faces[i].smooth, p_B.faces[i].invert, material, true);
	}

	//mark faces that ended up inside the intersection
	mesh_merge.mark_inside_faces();

	//regen new brush to start filling it again
	result.clear();

	switch (p_operation) {

		case OPERATION_UNION: {

			int outside_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {
				if (mesh_merge.faces[i].inside)
					continue;

				outside_count++;
			}

			result.faces.resize(outside_count);

			outside_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {
				if (mesh_merge.faces[i].inside)
					continue;
				for (int j = 0; j < 3; j++) {
					result.faces.write[outside_count].vertices[j] = mesh_merge.points[mesh_merge.faces[i].points[j]];
					result.faces.write[outside_count].uvs[j] = mesh_merge.faces[i].uvs[j];
				}

				result.faces.write[outside_count].smooth = mesh_merge.faces[i].smooth;
				result.faces.write[outside_count].invert = mesh_merge.faces[i].invert;
				result.faces.write[outside_count].material = mesh_merge.faces[i].material_idx;
				outside_count++;
			}

			result._regen_face_aabbs();

		} break;
		case OPERATION_INTERSECTION: {

			int inside_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {
				if (!mesh_merge.faces[i].inside)
					continue;

				inside_count++;
			}

			result.faces.resize(inside_count);

			inside_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {
				if (!mesh_merge.faces[i].inside)
					continue;
				for (int j = 0; j < 3; j++) {
					result.faces.write[inside_count].vertices[j] = mesh_merge.points[mesh_merge.faces[i].points[j]];
					result.faces.write[inside_count].uvs[j] = mesh_merge.faces[i].uvs[j];
				}

				result.faces.write[inside_count].smooth = mesh_merge.faces[i].smooth;
				result.faces.write[inside_count].invert = mesh_merge.faces[i].invert;
				result.faces.write[inside_count].material = mesh_merge.faces[i].material_idx;
				inside_count++;
			}

			result._regen_face_aabbs();

		} break;
		case OPERATION_SUBSTRACTION: {

			int face_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {
				if (mesh_merge.faces[i].from_b && !mesh_merge.faces[i].inside)
					continue;
				if (!mesh_merge.faces[i].from_b && mesh_merge.faces[i].inside)
					continue;

				face_count++;
			}

			result.faces.resize(face_count);

			face_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {

				if (mesh_merge.faces[i].from_b && !mesh_merge.faces[i].inside)
					continue;
				if (!mesh_merge.faces[i].from_b && mesh_merge.faces[i].inside)
					continue;

				for (int j = 0; j < 3; j++) {
					result.faces.write[face_count].vertices[j] = mesh_merge.points[mesh_merge.faces[i].points[j]];
					result.faces.write[face_count].uvs[j] = mesh_merge.faces[i].uvs[j];
				}

				if (mesh_merge.faces[i].from_b) {
					//invert facing of insides of B
					SWAP(result.faces.write[face_count].vertices[1], result.faces.write[face_count].vertices[2]);
					SWAP(result.faces.write[face_count].uvs[1], result.faces.write[face_count].uvs[2]);
				}

				result.faces.write[face_count].smooth = mesh_merge.faces[i].smooth;
				result.faces.write[face_count].invert = mesh_merge.faces[i].invert;
				result.faces.write[face_count].material = mesh_merge.faces[i].material_idx;
				face_count++;
			}

			result._regen_face_aabbs();

		} break;
	}

	//updatelist of materials
	result.materials.resize(mesh_merge.materials.size());
	for (const Map<Ref<Material>, int>::Element *E = mesh_merge.materials.front(); E; E = E->next()) {
		result.materials.write[E->get()] = E->key();
	}
}
