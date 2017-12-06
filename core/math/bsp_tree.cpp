/*************************************************************************/
/*  bsp_tree.cpp                                                         */
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
#include "bsp_tree.h"
#include "error_macros.h"
#include "print_string.h"

void BSP_Tree::from_aabb(const AABB &p_aabb) {

	planes.clear();

	for (int i = 0; i < 3; i++) {

		Vector3 n;
		n[i] = 1;
		planes.push_back(Plane(n, p_aabb.position[i] + p_aabb.size[i]));
		planes.push_back(Plane(-n, -p_aabb.position[i]));
	}

	nodes.clear();

	for (int i = 0; i < 6; i++) {

		Node n;
		n.plane = i;
		n.under = (i == 0) ? UNDER_LEAF : i - 1;
		n.over = OVER_LEAF;
		nodes.push_back(n);
	}

	aabb = p_aabb;
	error_radius = 0;
}

Vector<BSP_Tree::Node> BSP_Tree::get_nodes() const {

	return nodes;
}
Vector<Plane> BSP_Tree::get_planes() const {

	return planes;
}

AABB BSP_Tree::get_aabb() const {

	return aabb;
}

int BSP_Tree::_get_points_inside(int p_node, const Vector3 *p_points, int *p_indices, const Vector3 &p_center, const Vector3 &p_half_extents, int p_indices_count) const {

	const Node *node = &nodes[p_node];
	const Plane &p = planes[node->plane];

	Vector3 min(
			(p.normal.x > 0) ? -p_half_extents.x : p_half_extents.x,
			(p.normal.y > 0) ? -p_half_extents.y : p_half_extents.y,
			(p.normal.z > 0) ? -p_half_extents.z : p_half_extents.z);
	Vector3 max = -min;
	max += p_center;
	min += p_center;

	real_t dist_min = p.distance_to(min);
	real_t dist_max = p.distance_to(max);

	if ((dist_min * dist_max) < CMP_EPSILON) { //intersection, test point by point

		int under_count = 0;

		//sort points, so the are under first, over last
		for (int i = 0; i < p_indices_count; i++) {

			int index = p_indices[i];

			if (p.is_point_over(p_points[index])) {

				// kind of slow (but cache friendly), should try something else,
				// but this is a corner case most of the time

				for (int j = index; j < p_indices_count - 1; j++)
					p_indices[j] = p_indices[j + 1];

				p_indices[p_indices_count - 1] = index;

			} else {
				under_count++;
			}
		}

		int total = 0;

		if (under_count > 0) {
			if (node->under == UNDER_LEAF) {
				total += under_count;
			} else {
				total += _get_points_inside(node->under, p_points, p_indices, p_center, p_half_extents, under_count);
			}
		}

		if (under_count != p_indices_count) {
			if (node->over == OVER_LEAF) {
				//total+=0 //if they are over an OVER_LEAF, they are outside the model
			} else {
				total += _get_points_inside(node->over, p_points, &p_indices[under_count], p_center, p_half_extents, p_indices_count - under_count);
			}
		}

		return total;

	} else if (dist_min > 0) { //all points over plane

		if (node->over == OVER_LEAF) {

			return 0; // all these points are not visible
		}

		return _get_points_inside(node->over, p_points, p_indices, p_center, p_half_extents, p_indices_count);
	} else if (dist_min <= 0) { //all points behind plane

		if (node->under == UNDER_LEAF) {

			return p_indices_count; // all these points are visible
		}
		return _get_points_inside(node->under, p_points, p_indices, p_center, p_half_extents, p_indices_count);
	}

	return 0;
}

int BSP_Tree::get_points_inside(const Vector3 *p_points, int p_point_count) const {

	if (nodes.size() == 0)
		return 0;

#if 1
	//this version is easier to debug, and and MUCH faster in real world cases

	int pass_count = 0;
	const Node *nodesptr = &nodes[0];
	const Plane *planesptr = &planes[0];
	int plane_count = planes.size();
	int node_count = nodes.size();

	if (node_count == 0) // no nodes!
		return 0;

	for (int i = 0; i < p_point_count; i++) {

		const Vector3 &point = p_points[i];
		if (!aabb.has_point(point)) {
			continue;
		}

		int idx = node_count - 1;

		bool pass = false;

		while (true) {

			if (idx == OVER_LEAF) {
				pass = false;
				break;
			} else if (idx == UNDER_LEAF) {
				pass = true;
				break;
			}

			uint16_t plane = nodesptr[idx].plane;
#ifdef DEBUG_ENABLED

			ERR_FAIL_INDEX_V(plane, plane_count, false);
#endif

			idx = planesptr[nodesptr[idx].plane].is_point_over(point) ? nodes[idx].over : nodes[idx].under;

#ifdef DEBUG_ENABLED

			ERR_FAIL_COND_V(idx < MAX_NODES && idx >= node_count, false);
#endif
		}

		if (pass)
			pass_count++;
	}

	return pass_count;

#else
	//this version scales better but it's slower for real world cases

	int *indices = (int *)alloca(p_point_count * sizeof(int));
	AABB bounds;

	for (int i = 0; i < p_point_count; i++) {

		indices[i] = i;
		if (i == 0)
			bounds.pos = p_points[i];
		else
			bounds.expand_to(p_points[i]);
	}

	Vector3 half_extents = bounds.size / 2.0;
	return _get_points_inside(nodes.size() + 1, p_points, indices, bounds.pos + half_extents, half_extents, p_point_count);
#endif
}

bool BSP_Tree::point_is_inside(const Vector3 &p_point) const {

	if (!aabb.has_point(p_point)) {
		return false;
	}

	int node_count = nodes.size();

	if (node_count == 0) // no nodes!
		return false;

	const Node *nodesptr = &nodes[0];
	const Plane *planesptr = &planes[0];
	int plane_count = planes.size();

	int idx = node_count - 1;
	int steps = 0;

	while (true) {

		if (idx == OVER_LEAF) {
			return false;
		}
		if (idx == UNDER_LEAF) {

			return true;
		}

		uint16_t plane = nodesptr[idx].plane;
#ifdef DEBUG_ENABLED

		ERR_FAIL_INDEX_V(plane, plane_count, false);
#endif
		bool over = planesptr[nodesptr[idx].plane].is_point_over(p_point);

		idx = over ? nodes[idx].over : nodes[idx].under;

#ifdef DEBUG_ENABLED

		ERR_FAIL_COND_V(idx < MAX_NODES && idx >= node_count, false);
#endif

		steps++;
	}

	return false;
}

static int _bsp_find_best_half_plane(const Face3 *p_faces, const Vector<int> &p_indices, real_t p_tolerance) {

	int ic = p_indices.size();
	const int *indices = p_indices.ptr();

	int best_plane = -1;
	real_t best_plane_cost = 1e20;

	// Loop to find the polygon that best divides the set.

	for (int i = 0; i < ic; i++) {

		const Face3 &f = p_faces[indices[i]];
		Plane p = f.get_plane();

		int num_over = 0, num_under = 0, num_spanning = 0;

		for (int j = 0; j < ic; j++) {

			if (i == j)
				continue;

			const Face3 &g = p_faces[indices[j]];
			int over = 0, under = 0;

			for (int k = 0; k < 3; k++) {

				real_t d = p.distance_to(g.vertex[j]);

				if (Math::abs(d) > p_tolerance) {

					if (d > 0)
						over++;
					else
						under++;
				}
			}

			if (over && under)
				num_spanning++;
			else if (over)
				num_over++;
			else
				num_under++;
		}

		//real_t split_cost = num_spanning / (real_t) face_count;
		real_t relation = Math::abs(num_over - num_under) / (real_t)ic;

		// being honest, i never found a way to add split cost to the mix in a meaninguful way
		// in this engine, also, will likely be ignored anyway

		real_t plane_cost = /*split_cost +*/ relation;

		//printf("plane %i, %i over, %i under, %i spanning, cost is %g\n",i,num_over,num_under,num_spanning,plane_cost);
		if (plane_cost < best_plane_cost) {

			best_plane = i;
			best_plane_cost = plane_cost;
		}
	}

	return best_plane;
}

static int _bsp_create_node(const Face3 *p_faces, const Vector<int> &p_indices, Vector<Plane> &p_planes, Vector<BSP_Tree::Node> &p_nodes, real_t p_tolerance) {

	ERR_FAIL_COND_V(p_nodes.size() == BSP_Tree::MAX_NODES, -1);

	// should not reach here
	ERR_FAIL_COND_V(p_indices.size() == 0, -1)

	int ic = p_indices.size();
	const int *indices = p_indices.ptr();

	int divisor_idx = _bsp_find_best_half_plane(p_faces, p_indices, p_tolerance);

	// returned error
	ERR_FAIL_COND_V(divisor_idx < 0, -1);

	Vector<int> faces_over;
	Vector<int> faces_under;

	Plane divisor_plane = p_faces[indices[divisor_idx]].get_plane();

	for (int i = 0; i < ic; i++) {

		if (i == divisor_idx)
			continue;

		const Face3 &f = p_faces[indices[i]];

		/*
		if (f.get_plane().is_almost_like(divisor_plane))
			continue;
		*/

		int over_count = 0;
		int under_count = 0;

		for (int j = 0; j < 3; j++) {

			real_t d = divisor_plane.distance_to(f.vertex[j]);
			if (Math::abs(d) > p_tolerance) {

				if (d > 0)
					over_count++;
				else
					under_count++;
			}
		}

		if (over_count)
			faces_over.push_back(indices[i]);
		if (under_count)
			faces_under.push_back(indices[i]);
	}

	uint16_t over_idx = BSP_Tree::OVER_LEAF, under_idx = BSP_Tree::UNDER_LEAF;

	if (faces_over.size() > 0) { //have facess above?

		int idx = _bsp_create_node(p_faces, faces_over, p_planes, p_nodes, p_tolerance);
		if (idx >= 0)
			over_idx = idx;
	}

	if (faces_under.size() > 0) { //have facess above?

		int idx = _bsp_create_node(p_faces, faces_under, p_planes, p_nodes, p_tolerance);
		if (idx >= 0)
			under_idx = idx;
	}

	/* Create the node */

	// find existing divisor plane
	int divisor_plane_idx = -1;

	for (int i = 0; i < p_planes.size(); i++) {

		if (p_planes[i].is_almost_like(divisor_plane)) {
			divisor_plane_idx = i;
			break;
		}
	}

	if (divisor_plane_idx == -1) {

		ERR_FAIL_COND_V(p_planes.size() == BSP_Tree::MAX_PLANES, -1);
		divisor_plane_idx = p_planes.size();
		p_planes.push_back(divisor_plane);
	}

	BSP_Tree::Node node;
	node.plane = divisor_plane_idx;
	node.under = under_idx;
	node.over = over_idx;

	p_nodes.push_back(node);

	return p_nodes.size() - 1;
}

BSP_Tree::operator Variant() const {

	Dictionary d;
	d["error_radius"] = error_radius;

	Vector<real_t> plane_values;
	plane_values.resize(planes.size() * 4);

	for (int i = 0; i < planes.size(); i++) {

		plane_values[i * 4 + 0] = planes[i].normal.x;
		plane_values[i * 4 + 1] = planes[i].normal.y;
		plane_values[i * 4 + 2] = planes[i].normal.z;
		plane_values[i * 4 + 3] = planes[i].d;
	}

	d["planes"] = plane_values;

	PoolVector<int> dst_nodes;
	dst_nodes.resize(nodes.size() * 3);

	for (int i = 0; i < nodes.size(); i++) {

		dst_nodes.set(i * 3 + 0, nodes[i].over);
		dst_nodes.set(i * 3 + 1, nodes[i].under);
		dst_nodes.set(i * 3 + 2, nodes[i].plane);
	}

	d["nodes"] = dst_nodes;
	d["aabb"] = aabb;

	return Variant(d);
}

BSP_Tree::BSP_Tree() {
}

BSP_Tree::BSP_Tree(const Variant &p_variant) {

	Dictionary d = p_variant;
	ERR_FAIL_COND(!d.has("nodes"));
	ERR_FAIL_COND(!d.has("planes"));
	ERR_FAIL_COND(!d.has("aabb"));
	ERR_FAIL_COND(!d.has("error_radius"));

	PoolVector<int> src_nodes = d["nodes"];
	ERR_FAIL_COND(src_nodes.size() % 3);

	if (d["planes"].get_type() == Variant::POOL_REAL_ARRAY) {

		PoolVector<real_t> src_planes = d["planes"];
		int plane_count = src_planes.size();
		ERR_FAIL_COND(plane_count % 4);
		planes.resize(plane_count / 4);

		if (plane_count) {
			PoolVector<real_t>::Read r = src_planes.read();
			for (int i = 0; i < plane_count / 4; i++) {

				planes[i].normal.x = r[i * 4 + 0];
				planes[i].normal.y = r[i * 4 + 1];
				planes[i].normal.z = r[i * 4 + 2];
				planes[i].d = r[i * 4 + 3];
			}
		}

	} else {

		planes = d["planes"];
	}

	error_radius = d["error"];
	aabb = d["aabb"];

	//int node_count = src_nodes.size();
	nodes.resize(src_nodes.size() / 3);

	PoolVector<int>::Read r = src_nodes.read();

	for (int i = 0; i < nodes.size(); i++) {

		nodes[i].over = r[i * 3 + 0];
		nodes[i].under = r[i * 3 + 1];
		nodes[i].plane = r[i * 3 + 2];
	}
}

BSP_Tree::BSP_Tree(const PoolVector<Face3> &p_faces, real_t p_error_radius) {

	// compute aabb

	int face_count = p_faces.size();
	PoolVector<Face3>::Read faces_r = p_faces.read();
	const Face3 *facesptr = faces_r.ptr();

	bool first = true;

	Vector<int> indices;

	for (int i = 0; i < face_count; i++) {

		const Face3 &f = facesptr[i];

		if (f.is_degenerate())
			continue;

		for (int j = 0; j < 3; j++) {

			if (first) {

				aabb.position = f.vertex[0];
				first = false;
			} else {

				aabb.expand_to(f.vertex[j]);
			}
		}

		indices.push_back(i);
	}

	ERR_FAIL_COND(aabb.has_no_area());

	int top = _bsp_create_node(faces_r.ptr(), indices, planes, nodes, aabb.get_longest_axis_size() * 0.0001);

	if (top < 0) {

		nodes.clear();
		planes.clear();
		ERR_FAIL_COND(top < 0);
	}

	error_radius = p_error_radius;
}

BSP_Tree::BSP_Tree(const Vector<Node> &p_nodes, const Vector<Plane> &p_planes, const AABB &p_aabb, real_t p_error_radius) :
		nodes(p_nodes),
		planes(p_planes),
		aabb(p_aabb),
		error_radius(p_error_radius) {
}

BSP_Tree::~BSP_Tree() {
}
