/**************************************************************************/
/*  physics_server_2d.compat.inc                                          */
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

#ifndef DISABLE_DEPRECATED

#include "core/variant/typed_array.h"

Dictionary PhysicsDirectSpaceState2D::_intersect_ray_bind_compat_113970(RequiredParam<PhysicsRayQueryParameters2D> rp_ray_query) {
	EXTRACT_PARAM_OR_FAIL_V(p_ray_query, rp_ray_query, Dictionary());

	Ref<PhysicsRayIntersectionResult2D> p_result;
	p_result.instantiate();

	Dictionary r;

	if (_intersect_ray(p_ray_query, p_result)) {
		r["position"] = p_result->get_position();
		r["normal"] = p_result->get_normal();
		r["collider_id"] = p_result->get_collider_id();
		r["collider"] = p_result->get_collider();
		r["shape"] = p_result->get_collider_shape();
		r["rid"] = p_result->get_collider_rid();
	}

	return r;
}

TypedArray<Dictionary> PhysicsDirectSpaceState2D::_intersect_point_bind_compat_113970(RequiredParam<PhysicsPointQueryParameters2D> rp_point_query, int p_max_results) {
	EXTRACT_PARAM_OR_FAIL_V(p_point_query, rp_point_query, TypedArray<Dictionary>());

	Ref<PhysicsPointIntersectionResult2D> p_result = memnew(PhysicsPointIntersectionResult2D(p_max_results));

	p_result->collision_count = _intersect_point(p_point_query, p_result);

	TypedArray<Dictionary> r;
	r.resize(p_result->collision_count);

	for (int i = 0; i < p_result->collision_count; i++) {
		Dictionary d;
		d["rid"] = p_result->get_collider_rid(i);
		d["collider_id"] = p_result->get_collider_id(i);
		d["collider"] = p_result->get_collider(i);
		d["shape"] = p_result->get_collider_shape(i);
		r[i] = d;
	}

	return r;
}

TypedArray<Dictionary> PhysicsDirectSpaceState2D::_intersect_shape_bind_compat_113970(RequiredParam<PhysicsShapeQueryParameters2D> rp_shape_query, int p_max_results) {
	EXTRACT_PARAM_OR_FAIL_V(p_shape_query, rp_shape_query, TypedArray<Dictionary>());

	Ref<PhysicsShapeIntersectionResult2D> p_result = memnew(PhysicsShapeIntersectionResult2D(p_max_results));

	p_result->collision_count = _intersect_shape(p_shape_query, p_result);

	TypedArray<Dictionary> r;
	r.resize(p_result->collision_count);

	for (int i = 0; i < p_result->collision_count; i++) {
		Dictionary d;
		d["rid"] = p_result->get_collider_rid(i);
		d["collider_id"] = p_result->get_collider_id(i);
		d["collider"] = p_result->get_collider(i);
		d["shape"] = p_result->get_collider_shape(i);
		r[i] = d;
	}

	return r;
}

void PhysicsDirectSpaceState2D::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("intersect_ray", "parameters"), &PhysicsDirectSpaceState2D::_intersect_ray_bind_compat_113970);
	ClassDB::bind_compatibility_method(D_METHOD("intersect_point", "parameters", "max_results"), &PhysicsDirectSpaceState2D::_intersect_point_bind_compat_113970, DEFVAL(32));
	ClassDB::bind_compatibility_method(D_METHOD("intersect_shape", "parameters", "max_results"), &PhysicsDirectSpaceState2D::_intersect_shape_bind_compat_113970, DEFVAL(32));
}

#endif
