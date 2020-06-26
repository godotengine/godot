/*************************************************************************/
/*  collision_solver_3d_sat.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "collision_solver_3d_sat.h"
#include "core/math/geometry_3d.h"

#define _EDGE_IS_VALID_SUPPORT_THRESHOLD 0.02

struct _CollectorCallback {
	CollisionSolver3DSW::CallbackResult callback;
	void *userdata;
	bool swap;
	bool collided;
	Vector3 normal;
	Vector3 *prev_axis;

	_FORCE_INLINE_ void call(const Vector3 &p_point_A, const Vector3 &p_point_B) {
		if (swap) {
			callback(p_point_B, p_point_A, userdata);
		} else {
			callback(p_point_A, p_point_B, userdata);
		}
	}
};

typedef void (*GenerateContactsFunc)(const Vector3 *, int, const Vector3 *, int, _CollectorCallback *);

static void _generate_contacts_point_point(const Vector3 *p_points_A, int p_point_count_A, const Vector3 *p_points_B, int p_point_count_B, _CollectorCallback *p_callback) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(p_point_count_A != 1);
	ERR_FAIL_COND(p_point_count_B != 1);
#endif

	p_callback->call(*p_points_A, *p_points_B);
}

static void _generate_contacts_point_edge(const Vector3 *p_points_A, int p_point_count_A, const Vector3 *p_points_B, int p_point_count_B, _CollectorCallback *p_callback) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(p_point_count_A != 1);
	ERR_FAIL_COND(p_point_count_B != 2);
#endif

	Vector3 closest_B = Geometry3D::get_closest_point_to_segment_uncapped(*p_points_A, p_points_B);
	p_callback->call(*p_points_A, closest_B);
}

static void _generate_contacts_point_face(const Vector3 *p_points_A, int p_point_count_A, const Vector3 *p_points_B, int p_point_count_B, _CollectorCallback *p_callback) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(p_point_count_A != 1);
	ERR_FAIL_COND(p_point_count_B < 3);
#endif

	Vector3 closest_B = Plane(p_points_B[0], p_points_B[1], p_points_B[2]).project(*p_points_A);

	p_callback->call(*p_points_A, closest_B);
}

static void _generate_contacts_edge_edge(const Vector3 *p_points_A, int p_point_count_A, const Vector3 *p_points_B, int p_point_count_B, _CollectorCallback *p_callback) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(p_point_count_A != 2);
	ERR_FAIL_COND(p_point_count_B != 2); // circle is actually a 4x3 matrix
#endif

	Vector3 rel_A = p_points_A[1] - p_points_A[0];
	Vector3 rel_B = p_points_B[1] - p_points_B[0];

	Vector3 c = rel_A.cross(rel_B).cross(rel_B);

	if (Math::is_zero_approx(rel_A.dot(c))) {
		// should handle somehow..
		//ERR_PRINT("TODO FIX");
		//return;

		Vector3 axis = rel_A.normalized(); //make an axis
		Vector3 base_A = p_points_A[0] - axis * axis.dot(p_points_A[0]);
		Vector3 base_B = p_points_B[0] - axis * axis.dot(p_points_B[0]);

		//sort all 4 points in axis
		real_t dvec[4] = { axis.dot(p_points_A[0]), axis.dot(p_points_A[1]), axis.dot(p_points_B[0]), axis.dot(p_points_B[1]) };

		SortArray<real_t> sa;
		sa.sort(dvec, 4);

		//use the middle ones as contacts
		p_callback->call(base_A + axis * dvec[1], base_B + axis * dvec[1]);
		p_callback->call(base_A + axis * dvec[2], base_B + axis * dvec[2]);

		return;
	}

	real_t d = (c.dot(p_points_B[0]) - p_points_A[0].dot(c)) / rel_A.dot(c);

	if (d < 0.0) {
		d = 0.0;
	} else if (d > 1.0) {
		d = 1.0;
	}

	Vector3 closest_A = p_points_A[0] + rel_A * d;
	Vector3 closest_B = Geometry3D::get_closest_point_to_segment_uncapped(closest_A, p_points_B);
	p_callback->call(closest_A, closest_B);
}

static void _generate_contacts_face_face(const Vector3 *p_points_A, int p_point_count_A, const Vector3 *p_points_B, int p_point_count_B, _CollectorCallback *p_callback) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(p_point_count_A < 2);
	ERR_FAIL_COND(p_point_count_B < 3);
#endif

	static const int max_clip = 32;

	Vector3 _clipbuf1[max_clip];
	Vector3 _clipbuf2[max_clip];
	Vector3 *clipbuf_src = _clipbuf1;
	Vector3 *clipbuf_dst = _clipbuf2;
	int clipbuf_len = p_point_count_A;

	// copy A points to clipbuf_src
	for (int i = 0; i < p_point_count_A; i++) {
		clipbuf_src[i] = p_points_A[i];
	}

	Plane plane_B(p_points_B[0], p_points_B[1], p_points_B[2]);

	// go through all of B points
	for (int i = 0; i < p_point_count_B; i++) {
		int i_n = (i + 1) % p_point_count_B;

		Vector3 edge0_B = p_points_B[i];
		Vector3 edge1_B = p_points_B[i_n];

		Vector3 clip_normal = (edge0_B - edge1_B).cross(plane_B.normal).normalized();
		// make a clip plane

		Plane clip(edge0_B, clip_normal);
		// avoid double clip if A is edge
		int dst_idx = 0;
		bool edge = clipbuf_len == 2;
		for (int j = 0; j < clipbuf_len; j++) {
			int j_n = (j + 1) % clipbuf_len;

			Vector3 edge0_A = clipbuf_src[j];
			Vector3 edge1_A = clipbuf_src[j_n];

			real_t dist0 = clip.distance_to(edge0_A);
			real_t dist1 = clip.distance_to(edge1_A);

			if (dist0 <= 0) { // behind plane

				ERR_FAIL_COND(dst_idx >= max_clip);
				clipbuf_dst[dst_idx++] = clipbuf_src[j];
			}

			// check for different sides and non coplanar
			//if ( (dist0*dist1) < -CMP_EPSILON && !(edge && j)) {
			if ((dist0 * dist1) < 0 && !(edge && j)) {
				// calculate intersection
				Vector3 rel = edge1_A - edge0_A;
				real_t den = clip.normal.dot(rel);
				real_t dist = -(clip.normal.dot(edge0_A) - clip.d) / den;
				Vector3 inters = edge0_A + rel * dist;

				ERR_FAIL_COND(dst_idx >= max_clip);
				clipbuf_dst[dst_idx] = inters;
				dst_idx++;
			}
		}

		clipbuf_len = dst_idx;
		SWAP(clipbuf_src, clipbuf_dst);
	}

	// generate contacts
	//Plane plane_A(p_points_A[0],p_points_A[1],p_points_A[2]);

	for (int i = 0; i < clipbuf_len; i++) {
		real_t d = plane_B.distance_to(clipbuf_src[i]);
		/*
		if (d>CMP_EPSILON)
			continue;
		*/

		Vector3 closest_B = clipbuf_src[i] - plane_B.normal * d;

		if (p_callback->normal.dot(clipbuf_src[i]) >= p_callback->normal.dot(closest_B)) {
			continue;
		}

		p_callback->call(clipbuf_src[i], closest_B);
	}
}

static void _generate_contacts_from_supports(const Vector3 *p_points_A, int p_point_count_A, const Vector3 *p_points_B, int p_point_count_B, _CollectorCallback *p_callback) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(p_point_count_A < 1);
	ERR_FAIL_COND(p_point_count_B < 1);
#endif

	static const GenerateContactsFunc generate_contacts_func_table[3][3] = {
		{
				_generate_contacts_point_point,
				_generate_contacts_point_edge,
				_generate_contacts_point_face,
		},
		{
				nullptr,
				_generate_contacts_edge_edge,
				_generate_contacts_face_face,
		},
		{
				nullptr,
				nullptr,
				_generate_contacts_face_face,
		}
	};

	int pointcount_B;
	int pointcount_A;
	const Vector3 *points_A;
	const Vector3 *points_B;

	if (p_point_count_A > p_point_count_B) {
		//swap
		p_callback->swap = !p_callback->swap;
		p_callback->normal = -p_callback->normal;

		pointcount_B = p_point_count_A;
		pointcount_A = p_point_count_B;
		points_A = p_points_B;
		points_B = p_points_A;
	} else {
		pointcount_B = p_point_count_B;
		pointcount_A = p_point_count_A;
		points_A = p_points_A;
		points_B = p_points_B;
	}

	int version_A = (pointcount_A > 3 ? 3 : pointcount_A) - 1;
	int version_B = (pointcount_B > 3 ? 3 : pointcount_B) - 1;

	GenerateContactsFunc contacts_func = generate_contacts_func_table[version_A][version_B];
	ERR_FAIL_COND(!contacts_func);
	contacts_func(points_A, pointcount_A, points_B, pointcount_B, p_callback);
}

template <class ShapeA, class ShapeB, bool withMargin = false>
class SeparatorAxisTest {
	const ShapeA *shape_A;
	const ShapeB *shape_B;
	const Transform *transform_A;
	const Transform *transform_B;
	real_t best_depth;
	Vector3 best_axis;
	_CollectorCallback *callback;
	real_t margin_A;
	real_t margin_B;
	Vector3 separator_axis;

public:
	_FORCE_INLINE_ bool test_previous_axis() {
		if (callback && callback->prev_axis && *callback->prev_axis != Vector3()) {
			return test_axis(*callback->prev_axis);
		} else {
			return true;
		}
	}

	_FORCE_INLINE_ bool test_axis(const Vector3 &p_axis) {
		Vector3 axis = p_axis;

		if (Math::abs(axis.x) < CMP_EPSILON &&
				Math::abs(axis.y) < CMP_EPSILON &&
				Math::abs(axis.z) < CMP_EPSILON) {
			// strange case, try an upwards separator
			axis = Vector3(0.0, 1.0, 0.0);
		}

		real_t min_A, max_A, min_B, max_B;

		shape_A->project_range(axis, *transform_A, min_A, max_A);
		shape_B->project_range(axis, *transform_B, min_B, max_B);

		if (withMargin) {
			min_A -= margin_A;
			max_A += margin_A;
			min_B -= margin_B;
			max_B += margin_B;
		}

		min_B -= (max_A - min_A) * 0.5;
		max_B += (max_A - min_A) * 0.5;

		min_B -= (min_A + max_A) * 0.5;
		max_B -= (min_A + max_A) * 0.5;

		if (min_B > 0.0 || max_B < 0.0) {
			separator_axis = axis;
			return false; // doesn't contain 0
		}

		//use the smallest depth

		if (min_B < 0.0) { // could be +0.0, we don't want it to become -0.0
			min_B = -min_B;
		}

		if (max_B < min_B) {
			if (max_B < best_depth) {
				best_depth = max_B;
				best_axis = axis;
			}
		} else {
			if (min_B < best_depth) {
				best_depth = min_B;
				best_axis = -axis; // keep it as A axis
			}
		}

		return true;
	}

	_FORCE_INLINE_ void generate_contacts() {
		// nothing to do, don't generate
		if (best_axis == Vector3(0.0, 0.0, 0.0)) {
			return;
		}

		if (!callback->callback) {
			//just was checking intersection?
			callback->collided = true;
			if (callback->prev_axis) {
				*callback->prev_axis = best_axis;
			}
			return;
		}

		static const int max_supports = 16;

		Vector3 supports_A[max_supports];
		int support_count_A;
		shape_A->get_supports(transform_A->basis.xform_inv(-best_axis).normalized(), max_supports, supports_A, support_count_A);
		for (int i = 0; i < support_count_A; i++) {
			supports_A[i] = transform_A->xform(supports_A[i]);
		}

		if (withMargin) {
			for (int i = 0; i < support_count_A; i++) {
				supports_A[i] += -best_axis * margin_A;
			}
		}

		Vector3 supports_B[max_supports];
		int support_count_B;
		shape_B->get_supports(transform_B->basis.xform_inv(best_axis).normalized(), max_supports, supports_B, support_count_B);
		for (int i = 0; i < support_count_B; i++) {
			supports_B[i] = transform_B->xform(supports_B[i]);
		}

		if (withMargin) {
			for (int i = 0; i < support_count_B; i++) {
				supports_B[i] += best_axis * margin_B;
			}
		}

		callback->normal = best_axis;
		if (callback->prev_axis) {
			*callback->prev_axis = best_axis;
		}
		_generate_contacts_from_supports(supports_A, support_count_A, supports_B, support_count_B, callback);

		callback->collided = true;
	}

	_FORCE_INLINE_ SeparatorAxisTest(const ShapeA *p_shape_A, const Transform &p_transform_A, const ShapeB *p_shape_B, const Transform &p_transform_B, _CollectorCallback *p_callback, real_t p_margin_A = 0, real_t p_margin_B = 0) {
		best_depth = 1e15;
		shape_A = p_shape_A;
		shape_B = p_shape_B;
		transform_A = &p_transform_A;
		transform_B = &p_transform_B;
		callback = p_callback;
		margin_A = p_margin_A;
		margin_B = p_margin_B;
	}
};

/****** SAT TESTS *******/

typedef void (*CollisionFunc)(const Shape3DSW *, const Transform &, const Shape3DSW *, const Transform &, _CollectorCallback *p_callback, real_t, real_t);

template <bool withMargin>
static void _collision_sphere_sphere(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const SphereShape3DSW *sphere_A = static_cast<const SphereShape3DSW *>(p_a);
	const SphereShape3DSW *sphere_B = static_cast<const SphereShape3DSW *>(p_b);

	SeparatorAxisTest<SphereShape3DSW, SphereShape3DSW, withMargin> separator(sphere_A, p_transform_a, sphere_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	// previous axis

	if (!separator.test_previous_axis()) {
		return;
	}

	if (!separator.test_axis((p_transform_a.origin - p_transform_b.origin).normalized())) {
		return;
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_sphere_box(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const SphereShape3DSW *sphere_A = static_cast<const SphereShape3DSW *>(p_a);
	const BoxShape3DSW *box_B = static_cast<const BoxShape3DSW *>(p_b);

	SeparatorAxisTest<SphereShape3DSW, BoxShape3DSW, withMargin> separator(sphere_A, p_transform_a, box_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	if (!separator.test_previous_axis()) {
		return;
	}

	// test faces

	for (int i = 0; i < 3; i++) {
		Vector3 axis = p_transform_b.basis.get_axis(i).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// calculate closest point to sphere

	Vector3 cnormal = p_transform_b.xform_inv(p_transform_a.origin);

	Vector3 cpoint = p_transform_b.xform(Vector3(

			(cnormal.x < 0) ? -box_B->get_half_extents().x : box_B->get_half_extents().x,
			(cnormal.y < 0) ? -box_B->get_half_extents().y : box_B->get_half_extents().y,
			(cnormal.z < 0) ? -box_B->get_half_extents().z : box_B->get_half_extents().z));

	// use point to test axis
	Vector3 point_axis = (p_transform_a.origin - cpoint).normalized();

	if (!separator.test_axis(point_axis)) {
		return;
	}

	// test edges

	for (int i = 0; i < 3; i++) {
		Vector3 axis = point_axis.cross(p_transform_b.basis.get_axis(i)).cross(p_transform_b.basis.get_axis(i)).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_sphere_capsule(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const SphereShape3DSW *sphere_A = static_cast<const SphereShape3DSW *>(p_a);
	const CapsuleShape3DSW *capsule_B = static_cast<const CapsuleShape3DSW *>(p_b);

	SeparatorAxisTest<SphereShape3DSW, CapsuleShape3DSW, withMargin> separator(sphere_A, p_transform_a, capsule_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	if (!separator.test_previous_axis()) {
		return;
	}

	//capsule sphere 1, sphere

	Vector3 capsule_axis = p_transform_b.basis.get_axis(2) * (capsule_B->get_height() * 0.5);

	Vector3 capsule_ball_1 = p_transform_b.origin + capsule_axis;

	if (!separator.test_axis((capsule_ball_1 - p_transform_a.origin).normalized())) {
		return;
	}

	//capsule sphere 2, sphere

	Vector3 capsule_ball_2 = p_transform_b.origin - capsule_axis;

	if (!separator.test_axis((capsule_ball_2 - p_transform_a.origin).normalized())) {
		return;
	}

	//capsule edge, sphere

	Vector3 b2a = p_transform_a.origin - p_transform_b.origin;

	Vector3 axis = b2a.cross(capsule_axis).cross(capsule_axis).normalized();

	if (!separator.test_axis(axis)) {
		return;
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_sphere_cylinder(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
}

template <bool withMargin>
static void _collision_sphere_convex_polygon(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const SphereShape3DSW *sphere_A = static_cast<const SphereShape3DSW *>(p_a);
	const ConvexPolygonShape3DSW *convex_polygon_B = static_cast<const ConvexPolygonShape3DSW *>(p_b);

	SeparatorAxisTest<SphereShape3DSW, ConvexPolygonShape3DSW, withMargin> separator(sphere_A, p_transform_a, convex_polygon_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	if (!separator.test_previous_axis()) {
		return;
	}

	const Geometry3D::MeshData &mesh = convex_polygon_B->get_mesh();

	const Geometry3D::MeshData::Face *faces = mesh.faces.ptr();
	int face_count = mesh.faces.size();
	const Geometry3D::MeshData::Edge *edges = mesh.edges.ptr();
	int edge_count = mesh.edges.size();
	const Vector3 *vertices = mesh.vertices.ptr();
	int vertex_count = mesh.vertices.size();

	// faces of B
	for (int i = 0; i < face_count; i++) {
		Vector3 axis = p_transform_b.xform(faces[i].plane).normal;

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// edges of B
	for (int i = 0; i < edge_count; i++) {
		Vector3 v1 = p_transform_b.xform(vertices[edges[i].a]);
		Vector3 v2 = p_transform_b.xform(vertices[edges[i].b]);
		Vector3 v3 = p_transform_a.origin;

		Vector3 n1 = v2 - v1;
		Vector3 n2 = v2 - v3;

		Vector3 axis = n1.cross(n2).cross(n1).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// vertices of B
	for (int i = 0; i < vertex_count; i++) {
		Vector3 v1 = p_transform_b.xform(vertices[i]);
		Vector3 v2 = p_transform_a.origin;

		Vector3 axis = (v2 - v1).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_sphere_face(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const SphereShape3DSW *sphere_A = static_cast<const SphereShape3DSW *>(p_a);
	const FaceShape3DSW *face_B = static_cast<const FaceShape3DSW *>(p_b);

	SeparatorAxisTest<SphereShape3DSW, FaceShape3DSW, withMargin> separator(sphere_A, p_transform_a, face_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	Vector3 vertex[3] = {
		p_transform_b.xform(face_B->vertex[0]),
		p_transform_b.xform(face_B->vertex[1]),
		p_transform_b.xform(face_B->vertex[2]),
	};

	if (!separator.test_axis((vertex[0] - vertex[2]).cross(vertex[0] - vertex[1]).normalized())) {
		return;
	}

	// edges and points of B
	for (int i = 0; i < 3; i++) {
		Vector3 n1 = vertex[i] - p_transform_a.origin;

		if (!separator.test_axis(n1.normalized())) {
			return;
		}

		Vector3 n2 = vertex[(i + 1) % 3] - vertex[i];

		Vector3 axis = n1.cross(n2).cross(n2).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_box_box(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const BoxShape3DSW *box_A = static_cast<const BoxShape3DSW *>(p_a);
	const BoxShape3DSW *box_B = static_cast<const BoxShape3DSW *>(p_b);

	SeparatorAxisTest<BoxShape3DSW, BoxShape3DSW, withMargin> separator(box_A, p_transform_a, box_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	if (!separator.test_previous_axis()) {
		return;
	}

	// test faces of A

	for (int i = 0; i < 3; i++) {
		Vector3 axis = p_transform_a.basis.get_axis(i).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// test faces of B

	for (int i = 0; i < 3; i++) {
		Vector3 axis = p_transform_b.basis.get_axis(i).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// test combined edges
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			Vector3 axis = p_transform_a.basis.get_axis(i).cross(p_transform_b.basis.get_axis(j));

			if (Math::is_zero_approx(axis.length_squared())) {
				continue;
			}
			axis.normalize();

			if (!separator.test_axis(axis)) {
				return;
			}
		}
	}

	if (withMargin) {
		//add endpoint test between closest vertices and edges

		// calculate closest point to sphere

		Vector3 ab_vec = p_transform_b.origin - p_transform_a.origin;

		Vector3 cnormal_a = p_transform_a.basis.xform_inv(ab_vec);

		Vector3 support_a = p_transform_a.xform(Vector3(

				(cnormal_a.x < 0) ? -box_A->get_half_extents().x : box_A->get_half_extents().x,
				(cnormal_a.y < 0) ? -box_A->get_half_extents().y : box_A->get_half_extents().y,
				(cnormal_a.z < 0) ? -box_A->get_half_extents().z : box_A->get_half_extents().z));

		Vector3 cnormal_b = p_transform_b.basis.xform_inv(-ab_vec);

		Vector3 support_b = p_transform_b.xform(Vector3(

				(cnormal_b.x < 0) ? -box_B->get_half_extents().x : box_B->get_half_extents().x,
				(cnormal_b.y < 0) ? -box_B->get_half_extents().y : box_B->get_half_extents().y,
				(cnormal_b.z < 0) ? -box_B->get_half_extents().z : box_B->get_half_extents().z));

		Vector3 axis_ab = (support_a - support_b);

		if (!separator.test_axis(axis_ab.normalized())) {
			return;
		}

		//now try edges, which become cylinders!

		for (int i = 0; i < 3; i++) {
			//a ->b
			Vector3 axis_a = p_transform_a.basis.get_axis(i);

			if (!separator.test_axis(axis_ab.cross(axis_a).cross(axis_a).normalized())) {
				return;
			}

			//b ->a
			Vector3 axis_b = p_transform_b.basis.get_axis(i);

			if (!separator.test_axis(axis_ab.cross(axis_b).cross(axis_b).normalized())) {
				return;
			}
		}
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_box_capsule(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const BoxShape3DSW *box_A = static_cast<const BoxShape3DSW *>(p_a);
	const CapsuleShape3DSW *capsule_B = static_cast<const CapsuleShape3DSW *>(p_b);

	SeparatorAxisTest<BoxShape3DSW, CapsuleShape3DSW, withMargin> separator(box_A, p_transform_a, capsule_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	if (!separator.test_previous_axis()) {
		return;
	}

	// faces of A
	for (int i = 0; i < 3; i++) {
		Vector3 axis = p_transform_a.basis.get_axis(i);

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	Vector3 cyl_axis = p_transform_b.basis.get_axis(2).normalized();

	// edges of A, capsule cylinder

	for (int i = 0; i < 3; i++) {
		// cylinder
		Vector3 box_axis = p_transform_a.basis.get_axis(i);
		Vector3 axis = box_axis.cross(cyl_axis);
		if (Math::is_zero_approx(axis.length_squared())) {
			continue;
		}

		if (!separator.test_axis(axis.normalized())) {
			return;
		}
	}

	// points of A, capsule cylinder
	// this sure could be made faster somehow..

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				Vector3 he = box_A->get_half_extents();
				he.x *= (i * 2 - 1);
				he.y *= (j * 2 - 1);
				he.z *= (k * 2 - 1);
				Vector3 point = p_transform_a.origin;
				for (int l = 0; l < 3; l++) {
					point += p_transform_a.basis.get_axis(l) * he[l];
				}

				//Vector3 axis = (point - cyl_axis * cyl_axis.dot(point)).normalized();
				Vector3 axis = Plane(cyl_axis, 0).project(point).normalized();

				if (!separator.test_axis(axis)) {
					return;
				}
			}
		}
	}

	// capsule balls, edges of A

	for (int i = 0; i < 2; i++) {
		Vector3 capsule_axis = p_transform_b.basis.get_axis(2) * (capsule_B->get_height() * 0.5);

		Vector3 sphere_pos = p_transform_b.origin + ((i == 0) ? capsule_axis : -capsule_axis);

		Vector3 cnormal = p_transform_a.xform_inv(sphere_pos);

		Vector3 cpoint = p_transform_a.xform(Vector3(

				(cnormal.x < 0) ? -box_A->get_half_extents().x : box_A->get_half_extents().x,
				(cnormal.y < 0) ? -box_A->get_half_extents().y : box_A->get_half_extents().y,
				(cnormal.z < 0) ? -box_A->get_half_extents().z : box_A->get_half_extents().z));

		// use point to test axis
		Vector3 point_axis = (sphere_pos - cpoint).normalized();

		if (!separator.test_axis(point_axis)) {
			return;
		}

		// test edges of A

		for (int j = 0; j < 3; j++) {
			Vector3 axis = point_axis.cross(p_transform_a.basis.get_axis(j)).cross(p_transform_a.basis.get_axis(j)).normalized();

			if (!separator.test_axis(axis)) {
				return;
			}
		}
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_box_cylinder(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
}

template <bool withMargin>
static void _collision_box_convex_polygon(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const BoxShape3DSW *box_A = static_cast<const BoxShape3DSW *>(p_a);
	const ConvexPolygonShape3DSW *convex_polygon_B = static_cast<const ConvexPolygonShape3DSW *>(p_b);

	SeparatorAxisTest<BoxShape3DSW, ConvexPolygonShape3DSW, withMargin> separator(box_A, p_transform_a, convex_polygon_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	if (!separator.test_previous_axis()) {
		return;
	}

	const Geometry3D::MeshData &mesh = convex_polygon_B->get_mesh();

	const Geometry3D::MeshData::Face *faces = mesh.faces.ptr();
	int face_count = mesh.faces.size();
	const Geometry3D::MeshData::Edge *edges = mesh.edges.ptr();
	int edge_count = mesh.edges.size();
	const Vector3 *vertices = mesh.vertices.ptr();
	int vertex_count = mesh.vertices.size();

	// faces of A
	for (int i = 0; i < 3; i++) {
		Vector3 axis = p_transform_a.basis.get_axis(i).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// faces of B
	for (int i = 0; i < face_count; i++) {
		Vector3 axis = p_transform_b.xform(faces[i].plane).normal;

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// A<->B edges
	for (int i = 0; i < 3; i++) {
		Vector3 e1 = p_transform_a.basis.get_axis(i);

		for (int j = 0; j < edge_count; j++) {
			Vector3 e2 = p_transform_b.basis.xform(vertices[edges[j].a]) - p_transform_b.basis.xform(vertices[edges[j].b]);

			Vector3 axis = e1.cross(e2).normalized();

			if (!separator.test_axis(axis)) {
				return;
			}
		}
	}

	if (withMargin) {
		// calculate closest points between vertices and box edges
		for (int v = 0; v < vertex_count; v++) {
			Vector3 vtxb = p_transform_b.xform(vertices[v]);
			Vector3 ab_vec = vtxb - p_transform_a.origin;

			Vector3 cnormal_a = p_transform_a.basis.xform_inv(ab_vec);

			Vector3 support_a = p_transform_a.xform(Vector3(

					(cnormal_a.x < 0) ? -box_A->get_half_extents().x : box_A->get_half_extents().x,
					(cnormal_a.y < 0) ? -box_A->get_half_extents().y : box_A->get_half_extents().y,
					(cnormal_a.z < 0) ? -box_A->get_half_extents().z : box_A->get_half_extents().z));

			Vector3 axis_ab = support_a - vtxb;

			if (!separator.test_axis(axis_ab.normalized())) {
				return;
			}

			//now try edges, which become cylinders!

			for (int i = 0; i < 3; i++) {
				//a ->b
				Vector3 axis_a = p_transform_a.basis.get_axis(i);

				if (!separator.test_axis(axis_ab.cross(axis_a).cross(axis_a).normalized())) {
					return;
				}
			}
		}

		//convex edges and box points
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					Vector3 he = box_A->get_half_extents();
					he.x *= (i * 2 - 1);
					he.y *= (j * 2 - 1);
					he.z *= (k * 2 - 1);
					Vector3 point = p_transform_a.origin;
					for (int l = 0; l < 3; l++) {
						point += p_transform_a.basis.get_axis(l) * he[l];
					}

					for (int e = 0; e < edge_count; e++) {
						Vector3 p1 = p_transform_b.xform(vertices[edges[e].a]);
						Vector3 p2 = p_transform_b.xform(vertices[edges[e].b]);
						Vector3 n = (p2 - p1);

						if (!separator.test_axis((point - p2).cross(n).cross(n).normalized())) {
							return;
						}
					}
				}
			}
		}
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_box_face(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const BoxShape3DSW *box_A = static_cast<const BoxShape3DSW *>(p_a);
	const FaceShape3DSW *face_B = static_cast<const FaceShape3DSW *>(p_b);

	SeparatorAxisTest<BoxShape3DSW, FaceShape3DSW, withMargin> separator(box_A, p_transform_a, face_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	Vector3 vertex[3] = {
		p_transform_b.xform(face_B->vertex[0]),
		p_transform_b.xform(face_B->vertex[1]),
		p_transform_b.xform(face_B->vertex[2]),
	};

	if (!separator.test_axis((vertex[0] - vertex[2]).cross(vertex[0] - vertex[1]).normalized())) {
		return;
	}

	// faces of A
	for (int i = 0; i < 3; i++) {
		Vector3 axis = p_transform_a.basis.get_axis(i).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// combined edges

	for (int i = 0; i < 3; i++) {
		Vector3 e = vertex[i] - vertex[(i + 1) % 3];

		for (int j = 0; j < 3; j++) {
			Vector3 axis = p_transform_a.basis.get_axis(j);

			if (!separator.test_axis(e.cross(axis).normalized())) {
				return;
			}
		}
	}

	if (withMargin) {
		// calculate closest points between vertices and box edges
		for (int v = 0; v < 3; v++) {
			Vector3 ab_vec = vertex[v] - p_transform_a.origin;

			Vector3 cnormal_a = p_transform_a.basis.xform_inv(ab_vec);

			Vector3 support_a = p_transform_a.xform(Vector3(

					(cnormal_a.x < 0) ? -box_A->get_half_extents().x : box_A->get_half_extents().x,
					(cnormal_a.y < 0) ? -box_A->get_half_extents().y : box_A->get_half_extents().y,
					(cnormal_a.z < 0) ? -box_A->get_half_extents().z : box_A->get_half_extents().z));

			Vector3 axis_ab = support_a - vertex[v];

			if (!separator.test_axis(axis_ab.normalized())) {
				return;
			}

			//now try edges, which become cylinders!

			for (int i = 0; i < 3; i++) {
				//a ->b
				Vector3 axis_a = p_transform_a.basis.get_axis(i);

				if (!separator.test_axis(axis_ab.cross(axis_a).cross(axis_a).normalized())) {
					return;
				}
			}
		}

		//convex edges and box points, there has to be a way to speed up this (get closest point?)
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					Vector3 he = box_A->get_half_extents();
					he.x *= (i * 2 - 1);
					he.y *= (j * 2 - 1);
					he.z *= (k * 2 - 1);
					Vector3 point = p_transform_a.origin;
					for (int l = 0; l < 3; l++) {
						point += p_transform_a.basis.get_axis(l) * he[l];
					}

					for (int e = 0; e < 3; e++) {
						Vector3 p1 = vertex[e];
						Vector3 p2 = vertex[(e + 1) % 3];

						Vector3 n = (p2 - p1);

						if (!separator.test_axis((point - p2).cross(n).cross(n).normalized())) {
							return;
						}
					}
				}
			}
		}
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_capsule_capsule(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const CapsuleShape3DSW *capsule_A = static_cast<const CapsuleShape3DSW *>(p_a);
	const CapsuleShape3DSW *capsule_B = static_cast<const CapsuleShape3DSW *>(p_b);

	SeparatorAxisTest<CapsuleShape3DSW, CapsuleShape3DSW, withMargin> separator(capsule_A, p_transform_a, capsule_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	if (!separator.test_previous_axis()) {
		return;
	}

	// some values

	Vector3 capsule_A_axis = p_transform_a.basis.get_axis(2) * (capsule_A->get_height() * 0.5);
	Vector3 capsule_B_axis = p_transform_b.basis.get_axis(2) * (capsule_B->get_height() * 0.5);

	Vector3 capsule_A_ball_1 = p_transform_a.origin + capsule_A_axis;
	Vector3 capsule_A_ball_2 = p_transform_a.origin - capsule_A_axis;
	Vector3 capsule_B_ball_1 = p_transform_b.origin + capsule_B_axis;
	Vector3 capsule_B_ball_2 = p_transform_b.origin - capsule_B_axis;

	//balls-balls

	if (!separator.test_axis((capsule_A_ball_1 - capsule_B_ball_1).normalized())) {
		return;
	}
	if (!separator.test_axis((capsule_A_ball_1 - capsule_B_ball_2).normalized())) {
		return;
	}

	if (!separator.test_axis((capsule_A_ball_2 - capsule_B_ball_1).normalized())) {
		return;
	}
	if (!separator.test_axis((capsule_A_ball_2 - capsule_B_ball_2).normalized())) {
		return;
	}

	// edges-balls

	if (!separator.test_axis((capsule_A_ball_1 - capsule_B_ball_1).cross(capsule_A_axis).cross(capsule_A_axis).normalized())) {
		return;
	}

	if (!separator.test_axis((capsule_A_ball_1 - capsule_B_ball_2).cross(capsule_A_axis).cross(capsule_A_axis).normalized())) {
		return;
	}

	if (!separator.test_axis((capsule_B_ball_1 - capsule_A_ball_1).cross(capsule_B_axis).cross(capsule_B_axis).normalized())) {
		return;
	}

	if (!separator.test_axis((capsule_B_ball_1 - capsule_A_ball_2).cross(capsule_B_axis).cross(capsule_B_axis).normalized())) {
		return;
	}

	// edges

	if (!separator.test_axis(capsule_A_axis.cross(capsule_B_axis).normalized())) {
		return;
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_capsule_cylinder(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
}

template <bool withMargin>
static void _collision_capsule_convex_polygon(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const CapsuleShape3DSW *capsule_A = static_cast<const CapsuleShape3DSW *>(p_a);
	const ConvexPolygonShape3DSW *convex_polygon_B = static_cast<const ConvexPolygonShape3DSW *>(p_b);

	SeparatorAxisTest<CapsuleShape3DSW, ConvexPolygonShape3DSW, withMargin> separator(capsule_A, p_transform_a, convex_polygon_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	if (!separator.test_previous_axis()) {
		return;
	}

	const Geometry3D::MeshData &mesh = convex_polygon_B->get_mesh();

	const Geometry3D::MeshData::Face *faces = mesh.faces.ptr();
	int face_count = mesh.faces.size();
	const Geometry3D::MeshData::Edge *edges = mesh.edges.ptr();
	int edge_count = mesh.edges.size();
	const Vector3 *vertices = mesh.vertices.ptr();

	// faces of B
	for (int i = 0; i < face_count; i++) {
		Vector3 axis = p_transform_b.xform(faces[i].plane).normal;

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// edges of B, capsule cylinder

	for (int i = 0; i < edge_count; i++) {
		// cylinder
		Vector3 edge_axis = p_transform_b.basis.xform(vertices[edges[i].a]) - p_transform_b.basis.xform(vertices[edges[i].b]);
		Vector3 axis = edge_axis.cross(p_transform_a.basis.get_axis(2)).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// capsule balls, edges of B

	for (int i = 0; i < 2; i++) {
		// edges of B, capsule cylinder

		Vector3 capsule_axis = p_transform_a.basis.get_axis(2) * (capsule_A->get_height() * 0.5);

		Vector3 sphere_pos = p_transform_a.origin + ((i == 0) ? capsule_axis : -capsule_axis);

		for (int j = 0; j < edge_count; j++) {
			Vector3 n1 = sphere_pos - p_transform_b.xform(vertices[edges[j].a]);
			Vector3 n2 = p_transform_b.basis.xform(vertices[edges[j].a]) - p_transform_b.basis.xform(vertices[edges[j].b]);

			Vector3 axis = n1.cross(n2).cross(n2).normalized();

			if (!separator.test_axis(axis)) {
				return;
			}
		}
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_capsule_face(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const CapsuleShape3DSW *capsule_A = static_cast<const CapsuleShape3DSW *>(p_a);
	const FaceShape3DSW *face_B = static_cast<const FaceShape3DSW *>(p_b);

	SeparatorAxisTest<CapsuleShape3DSW, FaceShape3DSW, withMargin> separator(capsule_A, p_transform_a, face_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	Vector3 vertex[3] = {
		p_transform_b.xform(face_B->vertex[0]),
		p_transform_b.xform(face_B->vertex[1]),
		p_transform_b.xform(face_B->vertex[2]),
	};

	if (!separator.test_axis((vertex[0] - vertex[2]).cross(vertex[0] - vertex[1]).normalized())) {
		return;
	}

	// edges of B, capsule cylinder

	Vector3 capsule_axis = p_transform_a.basis.get_axis(2) * (capsule_A->get_height() * 0.5);

	for (int i = 0; i < 3; i++) {
		// edge-cylinder
		Vector3 edge_axis = vertex[i] - vertex[(i + 1) % 3];
		Vector3 axis = edge_axis.cross(capsule_axis).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}

		if (!separator.test_axis((p_transform_a.origin - vertex[i]).cross(capsule_axis).cross(capsule_axis).normalized())) {
			return;
		}

		for (int j = 0; j < 2; j++) {
			// point-spheres
			Vector3 sphere_pos = p_transform_a.origin + ((j == 0) ? capsule_axis : -capsule_axis);

			Vector3 n1 = sphere_pos - vertex[i];

			if (!separator.test_axis(n1.normalized())) {
				return;
			}

			Vector3 n2 = edge_axis;

			axis = n1.cross(n2).cross(n2);

			if (!separator.test_axis(axis.normalized())) {
				return;
			}
		}
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_cylinder_cylinder(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
}

template <bool withMargin>
static void _collision_cylinder_convex_polygon(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
}

template <bool withMargin>
static void _collision_cylinder_face(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
}

template <bool withMargin>
static void _collision_convex_polygon_convex_polygon(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const ConvexPolygonShape3DSW *convex_polygon_A = static_cast<const ConvexPolygonShape3DSW *>(p_a);
	const ConvexPolygonShape3DSW *convex_polygon_B = static_cast<const ConvexPolygonShape3DSW *>(p_b);

	SeparatorAxisTest<ConvexPolygonShape3DSW, ConvexPolygonShape3DSW, withMargin> separator(convex_polygon_A, p_transform_a, convex_polygon_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	if (!separator.test_previous_axis()) {
		return;
	}

	const Geometry3D::MeshData &mesh_A = convex_polygon_A->get_mesh();

	const Geometry3D::MeshData::Face *faces_A = mesh_A.faces.ptr();
	int face_count_A = mesh_A.faces.size();
	const Geometry3D::MeshData::Edge *edges_A = mesh_A.edges.ptr();
	int edge_count_A = mesh_A.edges.size();
	const Vector3 *vertices_A = mesh_A.vertices.ptr();
	int vertex_count_A = mesh_A.vertices.size();

	const Geometry3D::MeshData &mesh_B = convex_polygon_B->get_mesh();

	const Geometry3D::MeshData::Face *faces_B = mesh_B.faces.ptr();
	int face_count_B = mesh_B.faces.size();
	const Geometry3D::MeshData::Edge *edges_B = mesh_B.edges.ptr();
	int edge_count_B = mesh_B.edges.size();
	const Vector3 *vertices_B = mesh_B.vertices.ptr();
	int vertex_count_B = mesh_B.vertices.size();

	// faces of A
	for (int i = 0; i < face_count_A; i++) {
		Vector3 axis = p_transform_a.xform(faces_A[i].plane).normal;
		//Vector3 axis = p_transform_a.basis.xform( faces_A[i].plane.normal ).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// faces of B
	for (int i = 0; i < face_count_B; i++) {
		Vector3 axis = p_transform_b.xform(faces_B[i].plane).normal;
		//Vector3 axis = p_transform_b.basis.xform( faces_B[i].plane.normal ).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// A<->B edges
	for (int i = 0; i < edge_count_A; i++) {
		Vector3 e1 = p_transform_a.basis.xform(vertices_A[edges_A[i].a]) - p_transform_a.basis.xform(vertices_A[edges_A[i].b]);

		for (int j = 0; j < edge_count_B; j++) {
			Vector3 e2 = p_transform_b.basis.xform(vertices_B[edges_B[j].a]) - p_transform_b.basis.xform(vertices_B[edges_B[j].b]);

			Vector3 axis = e1.cross(e2).normalized();

			if (!separator.test_axis(axis)) {
				return;
			}
		}
	}

	if (withMargin) {
		//vertex-vertex
		for (int i = 0; i < vertex_count_A; i++) {
			Vector3 va = p_transform_a.xform(vertices_A[i]);

			for (int j = 0; j < vertex_count_B; j++) {
				if (!separator.test_axis((va - p_transform_b.xform(vertices_B[j])).normalized())) {
					return;
				}
			}
		}
		//edge-vertex (shell)

		for (int i = 0; i < edge_count_A; i++) {
			Vector3 e1 = p_transform_a.basis.xform(vertices_A[edges_A[i].a]);
			Vector3 e2 = p_transform_a.basis.xform(vertices_A[edges_A[i].b]);
			Vector3 n = (e2 - e1);

			for (int j = 0; j < vertex_count_B; j++) {
				Vector3 e3 = p_transform_b.xform(vertices_B[j]);

				if (!separator.test_axis((e1 - e3).cross(n).cross(n).normalized())) {
					return;
				}
			}
		}

		for (int i = 0; i < edge_count_B; i++) {
			Vector3 e1 = p_transform_b.basis.xform(vertices_B[edges_B[i].a]);
			Vector3 e2 = p_transform_b.basis.xform(vertices_B[edges_B[i].b]);
			Vector3 n = (e2 - e1);

			for (int j = 0; j < vertex_count_A; j++) {
				Vector3 e3 = p_transform_a.xform(vertices_A[j]);

				if (!separator.test_axis((e1 - e3).cross(n).cross(n).normalized())) {
					return;
				}
			}
		}
	}

	separator.generate_contacts();
}

template <bool withMargin>
static void _collision_convex_polygon_face(const Shape3DSW *p_a, const Transform &p_transform_a, const Shape3DSW *p_b, const Transform &p_transform_b, _CollectorCallback *p_collector, real_t p_margin_a, real_t p_margin_b) {
	const ConvexPolygonShape3DSW *convex_polygon_A = static_cast<const ConvexPolygonShape3DSW *>(p_a);
	const FaceShape3DSW *face_B = static_cast<const FaceShape3DSW *>(p_b);

	SeparatorAxisTest<ConvexPolygonShape3DSW, FaceShape3DSW, withMargin> separator(convex_polygon_A, p_transform_a, face_B, p_transform_b, p_collector, p_margin_a, p_margin_b);

	const Geometry3D::MeshData &mesh = convex_polygon_A->get_mesh();

	const Geometry3D::MeshData::Face *faces = mesh.faces.ptr();
	int face_count = mesh.faces.size();
	const Geometry3D::MeshData::Edge *edges = mesh.edges.ptr();
	int edge_count = mesh.edges.size();
	const Vector3 *vertices = mesh.vertices.ptr();
	int vertex_count = mesh.vertices.size();

	Vector3 vertex[3] = {
		p_transform_b.xform(face_B->vertex[0]),
		p_transform_b.xform(face_B->vertex[1]),
		p_transform_b.xform(face_B->vertex[2]),
	};

	if (!separator.test_axis((vertex[0] - vertex[2]).cross(vertex[0] - vertex[1]).normalized())) {
		return;
	}

	// faces of A
	for (int i = 0; i < face_count; i++) {
		//Vector3 axis = p_transform_a.xform( faces[i].plane ).normal;
		Vector3 axis = p_transform_a.basis.xform(faces[i].plane.normal).normalized();

		if (!separator.test_axis(axis)) {
			return;
		}
	}

	// A<->B edges
	for (int i = 0; i < edge_count; i++) {
		Vector3 e1 = p_transform_a.xform(vertices[edges[i].a]) - p_transform_a.xform(vertices[edges[i].b]);

		for (int j = 0; j < 3; j++) {
			Vector3 e2 = vertex[j] - vertex[(j + 1) % 3];

			Vector3 axis = e1.cross(e2).normalized();

			if (!separator.test_axis(axis)) {
				return;
			}
		}
	}

	if (withMargin) {
		//vertex-vertex
		for (int i = 0; i < vertex_count; i++) {
			Vector3 va = p_transform_a.xform(vertices[i]);

			for (int j = 0; j < 3; j++) {
				if (!separator.test_axis((va - vertex[j]).normalized())) {
					return;
				}
			}
		}
		//edge-vertex (shell)

		for (int i = 0; i < edge_count; i++) {
			Vector3 e1 = p_transform_a.basis.xform(vertices[edges[i].a]);
			Vector3 e2 = p_transform_a.basis.xform(vertices[edges[i].b]);
			Vector3 n = (e2 - e1);

			for (int j = 0; j < 3; j++) {
				Vector3 e3 = vertex[j];

				if (!separator.test_axis((e1 - e3).cross(n).cross(n).normalized())) {
					return;
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			Vector3 e1 = vertex[i];
			Vector3 e2 = vertex[(i + 1) % 3];
			Vector3 n = (e2 - e1);

			for (int j = 0; j < vertex_count; j++) {
				Vector3 e3 = p_transform_a.xform(vertices[j]);

				if (!separator.test_axis((e1 - e3).cross(n).cross(n).normalized())) {
					return;
				}
			}
		}
	}

	separator.generate_contacts();
}

bool sat_calculate_penetration(const Shape3DSW *p_shape_A, const Transform &p_transform_A, const Shape3DSW *p_shape_B, const Transform &p_transform_B, CollisionSolver3DSW::CallbackResult p_result_callback, void *p_userdata, bool p_swap, Vector3 *r_prev_axis, real_t p_margin_a, real_t p_margin_b) {
	PhysicsServer3D::ShapeType type_A = p_shape_A->get_type();

	ERR_FAIL_COND_V(type_A == PhysicsServer3D::SHAPE_PLANE, false);
	ERR_FAIL_COND_V(type_A == PhysicsServer3D::SHAPE_RAY, false);
	ERR_FAIL_COND_V(p_shape_A->is_concave(), false);

	PhysicsServer3D::ShapeType type_B = p_shape_B->get_type();

	ERR_FAIL_COND_V(type_B == PhysicsServer3D::SHAPE_PLANE, false);
	ERR_FAIL_COND_V(type_B == PhysicsServer3D::SHAPE_RAY, false);
	ERR_FAIL_COND_V(p_shape_B->is_concave(), false);

	static const CollisionFunc collision_table[6][6] = {
		{ _collision_sphere_sphere<false>,
				_collision_sphere_box<false>,
				_collision_sphere_capsule<false>,
				_collision_sphere_cylinder<false>,
				_collision_sphere_convex_polygon<false>,
				_collision_sphere_face<false> },
		{ nullptr,
				_collision_box_box<false>,
				_collision_box_capsule<false>,
				_collision_box_cylinder<false>,
				_collision_box_convex_polygon<false>,
				_collision_box_face<false> },
		{ nullptr,
				nullptr,
				_collision_capsule_capsule<false>,
				_collision_capsule_cylinder<false>,
				_collision_capsule_convex_polygon<false>,
				_collision_capsule_face<false> },
		{ nullptr,
				nullptr,
				nullptr,
				_collision_cylinder_cylinder<false>,
				_collision_cylinder_convex_polygon<false>,
				_collision_cylinder_face<false> },
		{ nullptr,
				nullptr,
				nullptr,
				nullptr,
				_collision_convex_polygon_convex_polygon<false>,
				_collision_convex_polygon_face<false> },
		{ nullptr,
				nullptr,
				nullptr,
				nullptr,
				nullptr,
				nullptr },
	};

	static const CollisionFunc collision_table_margin[6][6] = {
		{ _collision_sphere_sphere<true>,
				_collision_sphere_box<true>,
				_collision_sphere_capsule<true>,
				_collision_sphere_cylinder<true>,
				_collision_sphere_convex_polygon<true>,
				_collision_sphere_face<true> },
		{ nullptr,
				_collision_box_box<true>,
				_collision_box_capsule<true>,
				_collision_box_cylinder<true>,
				_collision_box_convex_polygon<true>,
				_collision_box_face<true> },
		{ nullptr,
				nullptr,
				_collision_capsule_capsule<true>,
				_collision_capsule_cylinder<true>,
				_collision_capsule_convex_polygon<true>,
				_collision_capsule_face<true> },
		{ nullptr,
				nullptr,
				nullptr,
				_collision_cylinder_cylinder<true>,
				_collision_cylinder_convex_polygon<true>,
				_collision_cylinder_face<true> },
		{ nullptr,
				nullptr,
				nullptr,
				nullptr,
				_collision_convex_polygon_convex_polygon<true>,
				_collision_convex_polygon_face<true> },
		{ nullptr,
				nullptr,
				nullptr,
				nullptr,
				nullptr,
				nullptr },
	};

	_CollectorCallback callback;
	callback.callback = p_result_callback;
	callback.swap = p_swap;
	callback.userdata = p_userdata;
	callback.collided = false;
	callback.prev_axis = r_prev_axis;

	const Shape3DSW *A = p_shape_A;
	const Shape3DSW *B = p_shape_B;
	const Transform *transform_A = &p_transform_A;
	const Transform *transform_B = &p_transform_B;
	real_t margin_A = p_margin_a;
	real_t margin_B = p_margin_b;

	if (type_A > type_B) {
		SWAP(A, B);
		SWAP(transform_A, transform_B);
		SWAP(type_A, type_B);
		SWAP(margin_A, margin_B);
		callback.swap = !callback.swap;
	}

	CollisionFunc collision_func;
	if (margin_A != 0.0 || margin_B != 0.0) {
		collision_func = collision_table_margin[type_A - 2][type_B - 2];

	} else {
		collision_func = collision_table[type_A - 2][type_B - 2];
	}
	ERR_FAIL_COND_V(!collision_func, false);

	collision_func(A, *transform_A, B, *transform_B, &callback, margin_A, margin_B);

	return callback.collided;
}
