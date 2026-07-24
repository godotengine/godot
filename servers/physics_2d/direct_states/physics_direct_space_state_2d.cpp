/**************************************************************************/
/*  physics_direct_space_state_2d.cpp                                     */
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

#include "physics_direct_space_state_2d.h"

#include "core/object/class_db.h"
#include "core/variant/typed_array.h"

Dictionary PhysicsDirectSpaceState2D::_intersect_ray(RequiredParam<PhysicsRayQueryParameters2D> rp_ray_query) {
	EXTRACT_PARAM_OR_FAIL_V(p_ray_query, rp_ray_query, Dictionary());

	PS2DT::RayResult result;
	bool res = intersect_ray(p_ray_query->get_parameters(), result);

	if (!res) {
		return Dictionary();
	}

	Dictionary d;
	d["position"] = result.position;
	d["normal"] = result.normal;
	d["collider_id"] = result.collider_id;
	d["collider"] = result.collider;
	d["shape"] = result.shape;
	d["rid"] = result.rid;

	return d;
}

TypedArray<Dictionary> PhysicsDirectSpaceState2D::_intersect_point(RequiredParam<PhysicsPointQueryParameters2D> rp_point_query, int p_max_results) {
	EXTRACT_PARAM_OR_FAIL_V(p_point_query, rp_point_query, TypedArray<Dictionary>());

	Vector<PS2DT::ShapeResult> ret;
	ret.resize(p_max_results);

	int rc = intersect_point(p_point_query->get_parameters(), ret.ptrw(), ret.size());

	if (rc == 0) {
		return TypedArray<Dictionary>();
	}

	TypedArray<Dictionary> r;
	r.resize(rc);
	for (int i = 0; i < rc; i++) {
		Dictionary d;
		d["rid"] = ret[i].rid;
		d["collider_id"] = ret[i].collider_id;
		d["collider"] = ret[i].collider;
		d["shape"] = ret[i].shape;
		r[i] = d;
	}
	return r;
}

TypedArray<Dictionary> PhysicsDirectSpaceState2D::_intersect_shape(RequiredParam<PhysicsShapeQueryParameters2D> rp_shape_query, int p_max_results) {
	EXTRACT_PARAM_OR_FAIL_V(p_shape_query, rp_shape_query, TypedArray<Dictionary>());

	Vector<PS2DT::ShapeResult> sr;
	sr.resize(p_max_results);
	int rc = intersect_shape(p_shape_query->get_parameters(), sr.ptrw(), sr.size());
	TypedArray<Dictionary> ret;
	ret.resize(rc);
	for (int i = 0; i < rc; i++) {
		Dictionary d;
		d["rid"] = sr[i].rid;
		d["collider_id"] = sr[i].collider_id;
		d["collider"] = sr[i].collider;
		d["shape"] = sr[i].shape;
		ret[i] = d;
	}

	return ret;
}

Vector<real_t> PhysicsDirectSpaceState2D::_cast_motion(RequiredParam<PhysicsShapeQueryParameters2D> rp_shape_query) {
	EXTRACT_PARAM_OR_FAIL_V(p_shape_query, rp_shape_query, Vector<real_t>());

	real_t closest_safe, closest_unsafe;
	bool res = cast_motion(p_shape_query->get_parameters(), closest_safe, closest_unsafe);
	if (!res) {
		return Vector<real_t>();
	}
	Vector<real_t> ret;
	ret.resize(2);
	ret.write[0] = closest_safe;
	ret.write[1] = closest_unsafe;
	return ret;
}

TypedArray<Vector2> PhysicsDirectSpaceState2D::_collide_shape(RequiredParam<PhysicsShapeQueryParameters2D> rp_shape_query, int p_max_results) {
	EXTRACT_PARAM_OR_FAIL_V(p_shape_query, rp_shape_query, TypedArray<Vector2>());

	Vector<Vector2> ret;
	ret.resize(p_max_results * 2);
	int rc = 0;
	bool res = collide_shape(p_shape_query->get_parameters(), ret.ptrw(), p_max_results, rc);
	if (!res) {
		return TypedArray<Vector2>();
	}
	TypedArray<Vector2> r;
	r.resize(rc * 2);
	for (int i = 0; i < rc * 2; i++) {
		r[i] = ret[i];
	}
	return r;
}

Dictionary PhysicsDirectSpaceState2D::_get_rest_info(RequiredParam<PhysicsShapeQueryParameters2D> rp_shape_query) {
	EXTRACT_PARAM_OR_FAIL_V(p_shape_query, rp_shape_query, Dictionary());

	PS2DT::ShapeRestInfo sri;

	bool res = rest_info(p_shape_query->get_parameters(), &sri);
	Dictionary r;
	if (!res) {
		return r;
	}

	r["point"] = sri.point;
	r["normal"] = sri.normal;
	r["rid"] = sri.rid;
	r["collider_id"] = sri.collider_id;
	r["shape"] = sri.shape;
	r["linear_velocity"] = sri.linear_velocity;

	return r;
}

bool PhysicsDirectSpaceState2D::_intersect_ray_typed(RequiredParam<PhysicsRayQueryParameters2D> rp_ray_query, RequiredParam<PhysicsRayIntersectionResult2D> rp_result) {
	EXTRACT_PARAM_OR_FAIL_V(p_ray_query, rp_ray_query, false);
	EXTRACT_PARAM_OR_FAIL_V(p_result, rp_result, false);

	return intersect_ray(p_ray_query->get_parameters(), p_result->result);
}

bool PhysicsDirectSpaceState2D::_intersect_point_typed(RequiredParam<PhysicsPointQueryParameters2D> rp_point_query, RequiredParam<PhysicsPointIntersectionResult2D> rp_result) {
	EXTRACT_PARAM_OR_FAIL_V(p_point_query, rp_point_query, false);
	EXTRACT_PARAM_OR_FAIL_V(p_result, rp_result, false);

	p_result->collision_count = intersect_point(p_point_query->get_parameters(), p_result->result.ptrw(), p_result->get_max_collisions());

	return p_result->collision_count > 0;
}

bool PhysicsDirectSpaceState2D::_intersect_shape_typed(RequiredParam<PhysicsShapeQueryParameters2D> rp_shape_query, RequiredParam<PhysicsShapeIntersectionResult2D> rp_result) {
	EXTRACT_PARAM_OR_FAIL_V(p_shape_query, rp_shape_query, false);
	EXTRACT_PARAM_OR_FAIL_V(p_result, rp_result, false);

	p_result->collision_count = intersect_shape(p_shape_query->get_parameters(), p_result->result.ptrw(), p_result->get_max_collisions());

	return p_result->collision_count > 0;
}

bool PhysicsDirectSpaceState2D::_cast_motion_typed(RequiredParam<PhysicsShapeQueryParameters2D> rp_shape_query, RequiredParam<PhysicsCastMotionResult2D> rp_result) {
	EXTRACT_PARAM_OR_FAIL_V(p_shape_query, rp_shape_query, false);
	EXTRACT_PARAM_OR_FAIL_V(p_result, rp_result, false);

	return cast_motion(p_shape_query->get_parameters(), p_result->safe_fraction, p_result->unsafe_fraction);
}

bool PhysicsDirectSpaceState2D::_collide_shape_typed(RequiredParam<PhysicsShapeQueryParameters2D> rp_shape_query, RequiredParam<PhysicsShapeCollisionResult2D> rp_result) {
	EXTRACT_PARAM_OR_FAIL_V(p_shape_query, rp_shape_query, false);
	EXTRACT_PARAM_OR_FAIL_V(p_result, rp_result, false);

	return collide_shape(p_shape_query->get_parameters(), p_result->result.ptrw(), p_result->get_max_collisions(), p_result->collision_count);
}

bool PhysicsDirectSpaceState2D::_get_rest_info_typed(RequiredParam<PhysicsShapeQueryParameters2D> rp_shape_query, RequiredParam<PhysicsRestInfoResult2D> rp_result) {
	EXTRACT_PARAM_OR_FAIL_V(p_shape_query, rp_shape_query, false);
	EXTRACT_PARAM_OR_FAIL_V(p_result, rp_result, false);

	return rest_info(p_shape_query->get_parameters(), &p_result->result);
}

PhysicsDirectSpaceState2D::PhysicsDirectSpaceState2D() {
}

void PhysicsDirectSpaceState2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("intersect_ray", "parameters"), &PhysicsDirectSpaceState2D::_intersect_ray);
	ClassDB::bind_method(D_METHOD("intersect_point", "parameters", "max_results"), &PhysicsDirectSpaceState2D::_intersect_point, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("intersect_shape", "parameters", "max_results"), &PhysicsDirectSpaceState2D::_intersect_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("cast_motion", "parameters"), &PhysicsDirectSpaceState2D::_cast_motion);
	ClassDB::bind_method(D_METHOD("collide_shape", "parameters", "max_results"), &PhysicsDirectSpaceState2D::_collide_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("get_rest_info", "parameters"), &PhysicsDirectSpaceState2D::_get_rest_info);

	ClassDB::bind_method(D_METHOD("intersect_ray_typed", "parameters", "result"), &PhysicsDirectSpaceState2D::_intersect_ray_typed);
	ClassDB::bind_method(D_METHOD("intersect_point_typed", "parameters", "result"), &PhysicsDirectSpaceState2D::_intersect_point_typed);
	ClassDB::bind_method(D_METHOD("intersect_shape_typed", "parameters", "result"), &PhysicsDirectSpaceState2D::_intersect_shape_typed);
	ClassDB::bind_method(D_METHOD("cast_motion_typed", "parameters", "result"), &PhysicsDirectSpaceState2D::_cast_motion_typed);
	ClassDB::bind_method(D_METHOD("collide_shape_typed", "parameters", "result"), &PhysicsDirectSpaceState2D::_collide_shape_typed);
	ClassDB::bind_method(D_METHOD("get_rest_info_typed", "parameters", "result"), &PhysicsDirectSpaceState2D::_get_rest_info_typed);
}
