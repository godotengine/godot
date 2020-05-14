/*************************************************************************/
/*  collision_solver_3d_sw.cpp                                           */
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

#include "collision_solver_3d_sw.h"
#include "collision_solver_3d_sat.h"

#include "gjk_epa.h"

#define collision_solver sat_calculate_penetration
//#define collision_solver gjk_epa_calculate_penetration

bool CollisionSolver3DSW::solve_static_plane(const Shape3DSW *p_shape_A, const Transform &p_transform_A, const Shape3DSW *p_shape_B, const Transform &p_transform_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result) {
	const PlaneShape3DSW *plane = static_cast<const PlaneShape3DSW *>(p_shape_A);
	if (p_shape_B->get_type() == PhysicsServer3D::SHAPE_PLANE) {
		return false;
	}
	Plane p = p_transform_A.xform(plane->get_plane());

	static const int max_supports = 16;
	Vector3 supports[max_supports];
	int support_count;

	p_shape_B->get_supports(p_transform_B.basis.xform_inv(-p.normal).normalized(), max_supports, supports, support_count);

	bool found = false;

	for (int i = 0; i < support_count; i++) {
		supports[i] = p_transform_B.xform(supports[i]);
		if (p.distance_to(supports[i]) >= 0) {
			continue;
		}
		found = true;

		Vector3 support_A = p.project(supports[i]);

		if (p_result_callback) {
			if (p_swap_result) {
				p_result_callback(supports[i], support_A, p_userdata);
			} else {
				p_result_callback(support_A, supports[i], p_userdata);
			}
		}
	}

	return found;
}

bool CollisionSolver3DSW::solve_ray(const Shape3DSW *p_shape_A, const Transform &p_transform_A, const Shape3DSW *p_shape_B, const Transform &p_transform_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result) {
	const RayShape3DSW *ray = static_cast<const RayShape3DSW *>(p_shape_A);

	Vector3 from = p_transform_A.origin;
	Vector3 to = from + p_transform_A.basis.get_axis(2) * ray->get_length();
	Vector3 support_A = to;

	Transform ai = p_transform_B.affine_inverse();

	from = ai.xform(from);
	to = ai.xform(to);

	Vector3 p, n;
	if (!p_shape_B->intersect_segment(from, to, p, n)) {
		return false;
	}

	Vector3 support_B = p_transform_B.xform(p);
	if (ray->get_slips_on_slope()) {
		Vector3 global_n = ai.basis.xform_inv(n).normalized();
		support_B = support_A + (support_B - support_A).length() * global_n;
	}

	if (p_result_callback) {
		if (p_swap_result) {
			p_result_callback(support_B, support_A, p_userdata);
		} else {
			p_result_callback(support_A, support_B, p_userdata);
		}
	}
	return true;
}

struct _ConcaveCollisionInfo {
	const Transform *transform_A;
	const Shape3DSW *shape_A;
	const Transform *transform_B;
	CollisionSolver3DSW::CallbackResult result_callback;
	void *userdata;
	bool swap_result;
	bool collided;
	int aabb_tests;
	int collisions;
	bool tested;
	real_t margin_A;
	real_t margin_B;
	Vector3 close_A, close_B;
};

void CollisionSolver3DSW::concave_callback(void *p_userdata, Shape3DSW *p_convex) {
	_ConcaveCollisionInfo &cinfo = *(_ConcaveCollisionInfo *)(p_userdata);
	cinfo.aabb_tests++;

	bool collided = collision_solver(cinfo.shape_A, *cinfo.transform_A, p_convex, *cinfo.transform_B, cinfo.result_callback, cinfo.userdata, cinfo.swap_result, nullptr, cinfo.margin_A, cinfo.margin_B);
	if (!collided) {
		return;
	}

	cinfo.collided = true;
	cinfo.collisions++;
}

bool CollisionSolver3DSW::solve_concave(const Shape3DSW *p_shape_A, const Transform &p_transform_A, const Shape3DSW *p_shape_B, const Transform &p_transform_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result, real_t p_margin_A, real_t p_margin_B) {
	const ConcaveShape3DSW *concave_B = static_cast<const ConcaveShape3DSW *>(p_shape_B);

	_ConcaveCollisionInfo cinfo;
	cinfo.transform_A = &p_transform_A;
	cinfo.shape_A = p_shape_A;
	cinfo.transform_B = &p_transform_B;
	cinfo.result_callback = p_result_callback;
	cinfo.userdata = p_userdata;
	cinfo.swap_result = p_swap_result;
	cinfo.collided = false;
	cinfo.collisions = 0;
	cinfo.margin_A = p_margin_A;
	cinfo.margin_B = p_margin_B;

	cinfo.aabb_tests = 0;

	Transform rel_transform = p_transform_A;
	rel_transform.origin -= p_transform_B.origin;

	//quickly compute a local AABB

	AABB local_aabb;
	for (int i = 0; i < 3; i++) {
		Vector3 axis(p_transform_B.basis.get_axis(i));
		real_t axis_scale = 1.0 / axis.length();
		axis *= axis_scale;

		real_t smin, smax;
		p_shape_A->project_range(axis, rel_transform, smin, smax);
		smin -= p_margin_A;
		smax += p_margin_A;
		smin *= axis_scale;
		smax *= axis_scale;

		local_aabb.position[i] = smin;
		local_aabb.size[i] = smax - smin;
	}

	concave_B->cull(local_aabb, concave_callback, &cinfo);

	return cinfo.collided;
}

bool CollisionSolver3DSW::solve_static(const Shape3DSW *p_shape_A, const Transform &p_transform_A, const Shape3DSW *p_shape_B, const Transform &p_transform_B, CallbackResult p_result_callback, void *p_userdata, Vector3 *r_sep_axis, real_t p_margin_A, real_t p_margin_B) {
	PhysicsServer3D::ShapeType type_A = p_shape_A->get_type();
	PhysicsServer3D::ShapeType type_B = p_shape_B->get_type();
	bool concave_A = p_shape_A->is_concave();
	bool concave_B = p_shape_B->is_concave();

	bool swap = false;

	if (type_A > type_B) {
		SWAP(type_A, type_B);
		SWAP(concave_A, concave_B);
		swap = true;
	}

	if (type_A == PhysicsServer3D::SHAPE_PLANE) {
		if (type_B == PhysicsServer3D::SHAPE_PLANE) {
			return false;
		}
		if (type_B == PhysicsServer3D::SHAPE_RAY) {
			return false;
		}

		if (swap) {
			return solve_static_plane(p_shape_B, p_transform_B, p_shape_A, p_transform_A, p_result_callback, p_userdata, true);
		} else {
			return solve_static_plane(p_shape_A, p_transform_A, p_shape_B, p_transform_B, p_result_callback, p_userdata, false);
		}

	} else if (type_A == PhysicsServer3D::SHAPE_RAY) {
		if (type_B == PhysicsServer3D::SHAPE_RAY) {
			return false;
		}

		if (swap) {
			return solve_ray(p_shape_B, p_transform_B, p_shape_A, p_transform_A, p_result_callback, p_userdata, true);
		} else {
			return solve_ray(p_shape_A, p_transform_A, p_shape_B, p_transform_B, p_result_callback, p_userdata, false);
		}

	} else if (concave_B) {
		if (concave_A) {
			return false;
		}

		if (!swap) {
			return solve_concave(p_shape_A, p_transform_A, p_shape_B, p_transform_B, p_result_callback, p_userdata, false, p_margin_A, p_margin_B);
		} else {
			return solve_concave(p_shape_B, p_transform_B, p_shape_A, p_transform_A, p_result_callback, p_userdata, true, p_margin_A, p_margin_B);
		}

	} else {
		return collision_solver(p_shape_A, p_transform_A, p_shape_B, p_transform_B, p_result_callback, p_userdata, false, r_sep_axis, p_margin_A, p_margin_B);
	}
}

void CollisionSolver3DSW::concave_distance_callback(void *p_userdata, Shape3DSW *p_convex) {
	_ConcaveCollisionInfo &cinfo = *(_ConcaveCollisionInfo *)(p_userdata);
	cinfo.aabb_tests++;
	if (cinfo.collided) {
		return;
	}

	Vector3 close_A, close_B;
	cinfo.collided = !gjk_epa_calculate_distance(cinfo.shape_A, *cinfo.transform_A, p_convex, *cinfo.transform_B, close_A, close_B);

	if (cinfo.collided) {
		return;
	}
	if (!cinfo.tested || close_A.distance_squared_to(close_B) < cinfo.close_A.distance_squared_to(cinfo.close_B)) {
		cinfo.close_A = close_A;
		cinfo.close_B = close_B;
		cinfo.tested = true;
	}

	cinfo.collisions++;
}

bool CollisionSolver3DSW::solve_distance_plane(const Shape3DSW *p_shape_A, const Transform &p_transform_A, const Shape3DSW *p_shape_B, const Transform &p_transform_B, Vector3 &r_point_A, Vector3 &r_point_B) {
	const PlaneShape3DSW *plane = static_cast<const PlaneShape3DSW *>(p_shape_A);
	if (p_shape_B->get_type() == PhysicsServer3D::SHAPE_PLANE) {
		return false;
	}
	Plane p = p_transform_A.xform(plane->get_plane());

	static const int max_supports = 16;
	Vector3 supports[max_supports];
	int support_count;

	p_shape_B->get_supports(p_transform_B.basis.xform_inv(-p.normal).normalized(), max_supports, supports, support_count);

	bool collided = false;
	Vector3 closest;
	real_t closest_d = 0;

	for (int i = 0; i < support_count; i++) {
		supports[i] = p_transform_B.xform(supports[i]);
		real_t d = p.distance_to(supports[i]);
		if (i == 0 || d < closest_d) {
			closest = supports[i];
			closest_d = d;
			if (d <= 0) {
				collided = true;
			}
		}
	}

	r_point_A = p.project(closest);
	r_point_B = closest;

	return collided;
}

bool CollisionSolver3DSW::solve_distance(const Shape3DSW *p_shape_A, const Transform &p_transform_A, const Shape3DSW *p_shape_B, const Transform &p_transform_B, Vector3 &r_point_A, Vector3 &r_point_B, const AABB &p_concave_hint, Vector3 *r_sep_axis) {
	if (p_shape_A->is_concave()) {
		return false;
	}

	if (p_shape_B->get_type() == PhysicsServer3D::SHAPE_PLANE) {
		Vector3 a, b;
		bool col = solve_distance_plane(p_shape_B, p_transform_B, p_shape_A, p_transform_A, a, b);
		r_point_A = b;
		r_point_B = a;
		return !col;

	} else if (p_shape_B->is_concave()) {
		if (p_shape_A->is_concave()) {
			return false;
		}

		const ConcaveShape3DSW *concave_B = static_cast<const ConcaveShape3DSW *>(p_shape_B);

		_ConcaveCollisionInfo cinfo;
		cinfo.transform_A = &p_transform_A;
		cinfo.shape_A = p_shape_A;
		cinfo.transform_B = &p_transform_B;
		cinfo.result_callback = nullptr;
		cinfo.userdata = nullptr;
		cinfo.swap_result = false;
		cinfo.collided = false;
		cinfo.collisions = 0;
		cinfo.aabb_tests = 0;
		cinfo.tested = false;

		Transform rel_transform = p_transform_A;
		rel_transform.origin -= p_transform_B.origin;

		//quickly compute a local AABB

		bool use_cc_hint = p_concave_hint != AABB();
		AABB cc_hint_aabb;
		if (use_cc_hint) {
			cc_hint_aabb = p_concave_hint;
			cc_hint_aabb.position -= p_transform_B.origin;
		}

		AABB local_aabb;
		for (int i = 0; i < 3; i++) {
			Vector3 axis(p_transform_B.basis.get_axis(i));
			real_t axis_scale = ((real_t)1.0) / axis.length();
			axis *= axis_scale;

			real_t smin, smax;

			if (use_cc_hint) {
				cc_hint_aabb.project_range_in_plane(Plane(axis, 0), smin, smax);
			} else {
				p_shape_A->project_range(axis, rel_transform, smin, smax);
			}

			smin *= axis_scale;
			smax *= axis_scale;

			local_aabb.position[i] = smin;
			local_aabb.size[i] = smax - smin;
		}

		concave_B->cull(local_aabb, concave_distance_callback, &cinfo);
		if (!cinfo.collided) {
			r_point_A = cinfo.close_A;
			r_point_B = cinfo.close_B;
		}

		return !cinfo.collided;
	} else {
		return gjk_epa_calculate_distance(p_shape_A, p_transform_A, p_shape_B, p_transform_B, r_point_A, r_point_B); //should pass sepaxis..
	}
}
