/*************************************************************************/
/*  collision_solver_2d_sw.cpp                                           */
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
#include "collision_solver_2d_sw.h"
#include "collision_solver_2d_sat.h"

#define collision_solver sat_2d_calculate_penetration
//#define collision_solver gjk_epa_calculate_penetration

bool CollisionSolver2DSW::solve_static_line(const Shape2DSW *p_shape_A, const Transform2D &p_transform_A, const Shape2DSW *p_shape_B, const Transform2D &p_transform_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result) {

	const LineShape2DSW *line = static_cast<const LineShape2DSW *>(p_shape_A);
	if (p_shape_B->get_type() == Physics2DServer::SHAPE_LINE)
		return false;

	Vector2 n = p_transform_A.basis_xform(line->get_normal()).normalized();
	Vector2 p = p_transform_A.xform(line->get_normal() * line->get_d());
	real_t d = n.dot(p);

	Vector2 supports[2];
	int support_count;

	p_shape_B->get_supports(p_transform_A.affine_inverse().basis_xform(-n).normalized(), supports, support_count);

	bool found = false;

	for (int i = 0; i < support_count; i++) {

		supports[i] = p_transform_B.xform(supports[i]);
		real_t pd = n.dot(supports[i]);
		if (pd >= d)
			continue;
		found = true;

		Vector2 support_A = supports[i] - n * (pd - d);

		if (p_result_callback) {
			if (p_swap_result)
				p_result_callback(supports[i], support_A, p_userdata);
			else
				p_result_callback(support_A, supports[i], p_userdata);
		}
	}

	return found;
}

bool CollisionSolver2DSW::solve_raycast(const Shape2DSW *p_shape_A, const Transform2D &p_transform_A, const Shape2DSW *p_shape_B, const Transform2D &p_transform_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result, Vector2 *sep_axis) {

	const RayShape2DSW *ray = static_cast<const RayShape2DSW *>(p_shape_A);
	if (p_shape_B->get_type() == Physics2DServer::SHAPE_RAY)
		return false;

	Vector2 from = p_transform_A.get_origin();
	Vector2 to = from + p_transform_A[1] * ray->get_length();
	Vector2 support_A = to;

	Transform2D invb = p_transform_B.affine_inverse();
	from = invb.xform(from);
	to = invb.xform(to);

	Vector2 p, n;
	if (!p_shape_B->intersect_segment(from, to, p, n)) {

		if (sep_axis)
			*sep_axis = p_transform_A[1].normalized();
		return false;
	}

	Vector2 support_B = p_transform_B.xform(p);

	if (p_result_callback) {
		if (p_swap_result)
			p_result_callback(support_B, support_A, p_userdata);
		else
			p_result_callback(support_A, support_B, p_userdata);
	}
	return true;
}

/*
bool CollisionSolver2DSW::solve_ray(const Shape2DSW *p_shape_A,const Matrix32& p_transform_A,const Shape2DSW *p_shape_B,const Matrix32& p_transform_B,const Matrix32& p_inverse_B,CallbackResult p_result_callback,void *p_userdata,bool p_swap_result) {


	const RayShape2DSW *ray = static_cast<const RayShape2DSW*>(p_shape_A);

	Vector2 from = p_transform_A.origin;
	Vector2 to = from+p_transform_A.basis.get_axis(2)*ray->get_length();
	Vector2 support_A=to;

	from = p_inverse_B.xform(from);
	to = p_inverse_B.xform(to);

	Vector2 p,n;
	if (!p_shape_B->intersect_segment(from,to,&p,&n))
		return false;

	Vector2 support_B=p_transform_B.xform(p);

	if (p_result_callback) {
		if (p_swap_result)
			p_result_callback(support_B,support_A,p_userdata);
		else
			p_result_callback(support_A,support_B,p_userdata);
	}
	return true;
}
*/

struct _ConcaveCollisionInfo2D {

	const Transform2D *transform_A;
	const Shape2DSW *shape_A;
	const Transform2D *transform_B;
	Vector2 motion_A;
	Vector2 motion_B;
	real_t margin_A;
	real_t margin_B;
	CollisionSolver2DSW::CallbackResult result_callback;
	void *userdata;
	bool swap_result;
	bool collided;
	int aabb_tests;
	int collisions;
	Vector2 *sep_axis;
};

void CollisionSolver2DSW::concave_callback(void *p_userdata, Shape2DSW *p_convex) {

	_ConcaveCollisionInfo2D &cinfo = *(_ConcaveCollisionInfo2D *)(p_userdata);
	cinfo.aabb_tests++;
	if (!cinfo.result_callback && cinfo.collided)
		return; //already collided and no contacts requested, don't test anymore

	bool collided = collision_solver(cinfo.shape_A, *cinfo.transform_A, cinfo.motion_A, p_convex, *cinfo.transform_B, cinfo.motion_B, cinfo.result_callback, cinfo.userdata, cinfo.swap_result, cinfo.sep_axis, cinfo.margin_A, cinfo.margin_B);
	if (!collided)
		return;

	cinfo.collided = true;
	cinfo.collisions++;
}

bool CollisionSolver2DSW::solve_concave(const Shape2DSW *p_shape_A, const Transform2D &p_transform_A, const Vector2 &p_motion_A, const Shape2DSW *p_shape_B, const Transform2D &p_transform_B, const Vector2 &p_motion_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result, Vector2 *sep_axis, real_t p_margin_A, real_t p_margin_B) {

	const ConcaveShape2DSW *concave_B = static_cast<const ConcaveShape2DSW *>(p_shape_B);

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
	cinfo.sep_axis = sep_axis;
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

		local_aabb.pos[i] = smin;
		local_aabb.size[i] = smax - smin;
	}

	concave_B->cull(local_aabb, concave_callback, &cinfo);

	//print_line("Rect2 TESTS: "+itos(cinfo.aabb_tests));
	return cinfo.collided;
}

bool CollisionSolver2DSW::solve(const Shape2DSW *p_shape_A, const Transform2D &p_transform_A, const Vector2 &p_motion_A, const Shape2DSW *p_shape_B, const Transform2D &p_transform_B, const Vector2 &p_motion_B, CallbackResult p_result_callback, void *p_userdata, Vector2 *sep_axis, real_t p_margin_A, real_t p_margin_B) {

	Physics2DServer::ShapeType type_A = p_shape_A->get_type();
	Physics2DServer::ShapeType type_B = p_shape_B->get_type();
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

	if (type_A == Physics2DServer::SHAPE_LINE) {

		if (type_B == Physics2DServer::SHAPE_LINE || type_B == Physics2DServer::SHAPE_RAY) {
			return false;
		}
		/*
		if (type_B==Physics2DServer::SHAPE_RAY) {
			return false;
		*/

		if (swap) {
			return solve_static_line(p_shape_B, p_transform_B, p_shape_A, p_transform_A, p_result_callback, p_userdata, true);
		} else {
			return solve_static_line(p_shape_A, p_transform_A, p_shape_B, p_transform_B, p_result_callback, p_userdata, false);
		}

		/*} else if (type_A==Physics2DServer::SHAPE_RAY) {

		if (type_B==Physics2DServer::SHAPE_RAY)
			return false;

		if (swap) {
			return solve_ray(p_shape_B,p_transform_B,p_shape_A,p_transform_A,p_inverse_A,p_result_callback,p_userdata,true);
		} else {
			return solve_ray(p_shape_A,p_transform_A,p_shape_B,p_transform_B,p_inverse_B,p_result_callback,p_userdata,false);
		}
*/
	} else if (type_A == Physics2DServer::SHAPE_RAY) {

		if (type_B == Physics2DServer::SHAPE_RAY) {

			return false; //no ray-ray
		}

		if (swap) {
			return solve_raycast(p_shape_B, p_transform_B, p_shape_A, p_transform_A, p_result_callback, p_userdata, true, sep_axis);
		} else {
			return solve_raycast(p_shape_A, p_transform_A, p_shape_B, p_transform_B, p_result_callback, p_userdata, false, sep_axis);
		}

	} else if (concave_B) {

		if (concave_A)
			return false;

		if (!swap)
			return solve_concave(p_shape_A, p_transform_A, p_motion_A, p_shape_B, p_transform_B, p_motion_B, p_result_callback, p_userdata, false, sep_axis, margin_A, margin_B);
		else
			return solve_concave(p_shape_B, p_transform_B, p_motion_B, p_shape_A, p_transform_A, p_motion_A, p_result_callback, p_userdata, true, sep_axis, margin_A, margin_B);

	} else {

		return collision_solver(p_shape_A, p_transform_A, p_motion_A, p_shape_B, p_transform_B, p_motion_B, p_result_callback, p_userdata, false, sep_axis, margin_A, margin_B);
	}

	return false;
}
