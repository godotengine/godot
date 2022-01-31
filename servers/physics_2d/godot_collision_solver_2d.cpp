/*************************************************************************/
/*  godot_collision_solver_2d.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "godot_collision_solver_2d.h"
#include "godot_collision_solver_2d_sat.h"

#define collision_solver sat_2d_calculate_penetration
//#define collision_solver gjk_epa_calculate_penetration

bool GodotCollisionSolver2D::solve_static_world_boundary(const GodotShape2D *p_shape_A, const Transform2D &p_transform_A, const GodotShape2D *p_shape_B, const Transform2D &p_transform_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result) {
	const GodotWorldBoundaryShape2D *world_boundary = static_cast<const GodotWorldBoundaryShape2D *>(p_shape_A);
	if (p_shape_B->get_type() == PhysicsServer2D::SHAPE_WORLD_BOUNDARY) {
		return false;
	}

	Vector2 n = p_transform_A.basis_xform(world_boundary->get_normal()).normalized();
	Vector2 p = p_transform_A.xform(world_boundary->get_normal() * world_boundary->get_d());
	real_t d = n.dot(p);

	Vector2 supports[2];
	int support_count;

	p_shape_B->get_supports(p_transform_B.affine_inverse().basis_xform(-n).normalized(), supports, support_count);

	bool found = false;

	for (int i = 0; i < support_count; i++) {
		supports[i] = p_transform_B.xform(supports[i]);
		real_t pd = n.dot(supports[i]);
		if (pd >= d) {
			continue;
		}
		found = true;

		Vector2 support_A = supports[i] - n * (pd - d);

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

bool GodotCollisionSolver2D::solve_separation_ray(const GodotShape2D *p_shape_A, const Vector2 &p_motion_A, const Transform2D &p_transform_A, const GodotShape2D *p_shape_B, const Transform2D &p_transform_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result, Vector2 *r_sep_axis, real_t p_margin) {
	const GodotSeparationRayShape2D *ray = static_cast<const GodotSeparationRayShape2D *>(p_shape_A);
	if (p_shape_B->get_type() == PhysicsServer2D::SHAPE_SEPARATION_RAY) {
		return false;
	}

	Vector2 from = p_transform_A.get_origin();
	Vector2 to = from + p_transform_A[1] * (ray->get_length() + p_margin);
	if (p_motion_A != Vector2()) {
		//not the best but should be enough
		Vector2 normal = (to - from).normalized();
		to += normal * MAX(0.0, normal.dot(p_motion_A));
	}
	Vector2 support_A = to;

	Transform2D invb = p_transform_B.affine_inverse();
	from = invb.xform(from);
	to = invb.xform(to);

	Vector2 p, n;
	if (!p_shape_B->intersect_segment(from, to, p, n)) {
		if (r_sep_axis) {
			*r_sep_axis = p_transform_A[1].normalized();
		}
		return false;
	}

	// Discard contacts when the ray is fully contained inside the shape.
	if (n == Vector2()) {
		if (r_sep_axis) {
			*r_sep_axis = p_transform_A[1].normalized();
		}
		return false;
	}

	// Discard contacts in the wrong direction.
	if (n.dot(from - to) < CMP_EPSILON) {
		if (r_sep_axis) {
			*r_sep_axis = p_transform_A[1].normalized();
		}
		return false;
	}

	Vector2 support_B = p_transform_B.xform(p);
	if (ray->get_slide_on_slope()) {
		Vector2 global_n = invb.basis_xform_inv(n).normalized();
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

struct _ConcaveCollisionInfo2D {
	const Transform2D *transform_A = nullptr;
	const GodotShape2D *shape_A = nullptr;
	const Transform2D *transform_B = nullptr;
	Vector2 motion_A;
	Vector2 motion_B;
	real_t margin_A = 0.0;
	real_t margin_B = 0.0;
	GodotCollisionSolver2D::CallbackResult result_callback;
	void *userdata = nullptr;
	bool swap_result = false;
	bool collided = false;
	int aabb_tests = 0;
	int collisions = 0;
	Vector2 *sep_axis = nullptr;
};

bool GodotCollisionSolver2D::concave_callback(void *p_userdata, GodotShape2D *p_convex) {
	_ConcaveCollisionInfo2D &cinfo = *(_ConcaveCollisionInfo2D *)(p_userdata);
	cinfo.aabb_tests++;

	bool collided = collision_solver(cinfo.shape_A, *cinfo.transform_A, cinfo.motion_A, p_convex, *cinfo.transform_B, cinfo.motion_B, cinfo.result_callback, cinfo.userdata, cinfo.swap_result, cinfo.sep_axis, cinfo.margin_A, cinfo.margin_B);
	if (!collided) {
		return false;
	}

	cinfo.collided = true;
	cinfo.collisions++;

	// Stop at first collision if contacts are not needed.
	return !cinfo.result_callback;
}

bool GodotCollisionSolver2D::solve_concave(const GodotShape2D *p_shape_A, const Transform2D &p_transform_A, const Vector2 &p_motion_A, const GodotShape2D *p_shape_B, const Transform2D &p_transform_B, const Vector2 &p_motion_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result, Vector2 *r_sep_axis, real_t p_margin_A, real_t p_margin_B) {
	const GodotConcaveShape2D *concave_B = static_cast<const GodotConcaveShape2D *>(p_shape_B);

	_ConcaveCollisionInfo2D cinfo;
	cinfo.transform_A = &p_transform_A;
	cinfo.shape_A = p_shape_A;
	cinfo.transform_B = &p_transform_B;
	cinfo.motion_A = p_motion_A;
	cinfo.result_callback = p_result_callback;
	cinfo.userdata = p_userdata;
	cinfo.swap_result = p_swap_result;
	cinfo.collided = false;
	cinfo.collisions = 0;
	cinfo.sep_axis = r_sep_axis;
	cinfo.margin_A = p_margin_A;
	cinfo.margin_B = p_margin_B;

	cinfo.aabb_tests = 0;

	Transform2D rel_transform = p_transform_A;
	rel_transform.elements[2] -= p_transform_B.get_origin();

	//quickly compute a local Rect2

	Rect2 local_aabb;
	for (int i = 0; i < 2; i++) {
		Vector2 axis(p_transform_B.elements[i]);
		real_t axis_scale = 1.0 / axis.length();
		axis *= axis_scale;

		real_t smin, smax;
		p_shape_A->project_rangev(axis, rel_transform, smin, smax);
		smin *= axis_scale;
		smax *= axis_scale;

		local_aabb.position[i] = smin;
		local_aabb.size[i] = smax - smin;
	}

	concave_B->cull(local_aabb, concave_callback, &cinfo);

	return cinfo.collided;
}

bool GodotCollisionSolver2D::solve(const GodotShape2D *p_shape_A, const Transform2D &p_transform_A, const Vector2 &p_motion_A, const GodotShape2D *p_shape_B, const Transform2D &p_transform_B, const Vector2 &p_motion_B, CallbackResult p_result_callback, void *p_userdata, Vector2 *r_sep_axis, real_t p_margin_A, real_t p_margin_B) {
	PhysicsServer2D::ShapeType type_A = p_shape_A->get_type();
	PhysicsServer2D::ShapeType type_B = p_shape_B->get_type();
	bool concave_A = p_shape_A->is_concave();
	bool concave_B = p_shape_B->is_concave();
	real_t margin_A = p_margin_A, margin_B = p_margin_B;

	bool swap = false;

	if (type_A > type_B) {
		SWAP(type_A, type_B);
		SWAP(concave_A, concave_B);
		SWAP(margin_A, margin_B);
		swap = true;
	}

	if (type_A == PhysicsServer2D::SHAPE_WORLD_BOUNDARY) {
		if (type_B == PhysicsServer2D::SHAPE_WORLD_BOUNDARY) {
			return false;
		}

		if (swap) {
			return solve_static_world_boundary(p_shape_B, p_transform_B, p_shape_A, p_transform_A, p_result_callback, p_userdata, true);
		} else {
			return solve_static_world_boundary(p_shape_A, p_transform_A, p_shape_B, p_transform_B, p_result_callback, p_userdata, false);
		}

	} else if (type_A == PhysicsServer2D::SHAPE_SEPARATION_RAY) {
		if (type_B == PhysicsServer2D::SHAPE_SEPARATION_RAY) {
			return false; //no ray-ray
		}

		if (swap) {
			return solve_separation_ray(p_shape_B, p_motion_B, p_transform_B, p_shape_A, p_transform_A, p_result_callback, p_userdata, true, r_sep_axis, p_margin_B);
		} else {
			return solve_separation_ray(p_shape_A, p_motion_A, p_transform_A, p_shape_B, p_transform_B, p_result_callback, p_userdata, false, r_sep_axis, p_margin_A);
		}

	} else if (concave_B) {
		if (concave_A) {
			return false;
		}

		if (!swap) {
			return solve_concave(p_shape_A, p_transform_A, p_motion_A, p_shape_B, p_transform_B, p_motion_B, p_result_callback, p_userdata, false, r_sep_axis, margin_A, margin_B);
		} else {
			return solve_concave(p_shape_B, p_transform_B, p_motion_B, p_shape_A, p_transform_A, p_motion_A, p_result_callback, p_userdata, true, r_sep_axis, margin_A, margin_B);
		}

	} else {
		return collision_solver(p_shape_A, p_transform_A, p_motion_A, p_shape_B, p_transform_B, p_motion_B, p_result_callback, p_userdata, false, r_sep_axis, margin_A, margin_B);
	}
}
