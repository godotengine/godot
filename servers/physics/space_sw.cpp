/**************************************************************************/
/*  space_sw.cpp                                                          */
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

#include "space_sw.h"

#include "collision_solver_sw.h"
#include "core/project_settings.h"
#include "physics_server_sw.h"

#define TEST_MOTION_MARGIN_MIN_VALUE 0.0001
#define TEST_MOTION_MIN_CONTACT_DEPTH_FACTOR 0.05

_FORCE_INLINE_ static bool _can_collide_with(CollisionObjectSW *p_object, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	if (!(p_object->get_collision_layer() & p_collision_mask)) {
		return false;
	}

	if (p_object->get_type() == CollisionObjectSW::TYPE_AREA && !p_collide_with_areas) {
		return false;
	}

	if (p_object->get_type() == CollisionObjectSW::TYPE_BODY && !p_collide_with_bodies) {
		return false;
	}

	return true;
}

int PhysicsDirectSpaceStateSW::intersect_point(const Vector3 &p_point, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	ERR_FAIL_COND_V(space->locked, false);
	int amount = space->broadphase->cull_point(p_point, space->intersection_query_results, SpaceSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);
	int cc = 0;

	//Transform ai = p_xform.affine_inverse();

	for (int i = 0; i < amount; i++) {
		if (cc >= p_result_max) {
			break;
		}

		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		//area can't be picked by ray (default)

		if (p_exclude.has(space->intersection_query_results[i]->get_self())) {
			continue;
		}

		const CollisionObjectSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		Transform inv_xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);
		inv_xform.affine_invert();

		if (!col_obj->get_shape(shape_idx)->intersect_point(inv_xform.xform(p_point))) {
			continue;
		}

		r_results[cc].collider_id = col_obj->get_instance_id();
		if (r_results[cc].collider_id != 0) {
			r_results[cc].collider = ObjectDB::get_instance(r_results[cc].collider_id);
		} else {
			r_results[cc].collider = nullptr;
		}
		r_results[cc].rid = col_obj->get_self();
		r_results[cc].shape = shape_idx;

		cc++;
	}

	return cc;
}

bool PhysicsDirectSpaceStateSW::intersect_ray(const Vector3 &p_from, const Vector3 &p_to, RayResult &r_result, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_pick_ray) {
	ERR_FAIL_COND_V(space->locked, false);

	Vector3 begin, end;
	Vector3 normal;
	begin = p_from;
	end = p_to;
	normal = (end - begin).normalized();

	int amount = space->broadphase->cull_segment(begin, end, space->intersection_query_results, SpaceSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	//todo, create another array that references results, compute AABBs and check closest point to ray origin, sort, and stop evaluating results when beyond first collision

	bool collided = false;
	Vector3 res_point, res_normal;
	int res_shape = 0;
	const CollisionObjectSW *res_obj = nullptr;
	real_t min_d = 1e10;

	for (int i = 0; i < amount; i++) {
		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		if (p_pick_ray && !(space->intersection_query_results[i]->is_ray_pickable())) {
			continue;
		}

		if (p_exclude.has(space->intersection_query_results[i]->get_self())) {
			continue;
		}

		const CollisionObjectSW *col_obj = space->intersection_query_results[i];

		int shape_idx = space->intersection_query_subindex_results[i];
		Transform inv_xform = col_obj->get_shape_inv_transform(shape_idx) * col_obj->get_inv_transform();

		Vector3 local_from = inv_xform.xform(begin);
		Vector3 local_to = inv_xform.xform(end);

		const ShapeSW *shape = col_obj->get_shape(shape_idx);

		Vector3 shape_point, shape_normal;

		if (shape->intersect_segment(local_from, local_to, shape_point, shape_normal)) {
			Transform xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);
			shape_point = xform.xform(shape_point);

			real_t ld = normal.dot(shape_point);

			if (ld < min_d) {
				min_d = ld;
				res_point = shape_point;
				res_normal = inv_xform.basis.xform_inv(shape_normal).normalized();
				res_shape = shape_idx;
				res_obj = col_obj;
				collided = true;
			}
		}
	}

	if (!collided) {
		return false;
	}

	r_result.collider_id = res_obj->get_instance_id();
	if (r_result.collider_id != 0) {
		r_result.collider = ObjectDB::get_instance(r_result.collider_id);
	} else {
		r_result.collider = nullptr;
	}
	r_result.normal = res_normal;
	r_result.position = res_point;
	r_result.rid = res_obj->get_self();
	r_result.shape = res_shape;

	return true;
}

int PhysicsDirectSpaceStateSW::intersect_shape(const RID &p_shape, const Transform &p_xform, real_t p_margin, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	if (p_result_max <= 0) {
		return 0;
	}

	ShapeSW *shape = static_cast<PhysicsServerSW *>(PhysicsServer::get_singleton())->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	AABB aabb = p_xform.xform(shape->get_aabb());

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, SpaceSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	int cc = 0;

	//Transform ai = p_xform.affine_inverse();

	for (int i = 0; i < amount; i++) {
		if (cc >= p_result_max) {
			break;
		}

		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		//area can't be picked by ray (default)

		if (p_exclude.has(space->intersection_query_results[i]->get_self())) {
			continue;
		}

		const CollisionObjectSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		if (!CollisionSolverSW::solve_static(shape, p_xform, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), nullptr, nullptr, nullptr, p_margin, 0)) {
			continue;
		}

		if (r_results) {
			r_results[cc].collider_id = col_obj->get_instance_id();
			if (r_results[cc].collider_id != 0) {
				r_results[cc].collider = ObjectDB::get_instance(r_results[cc].collider_id);
			} else {
				r_results[cc].collider = nullptr;
			}
			r_results[cc].rid = col_obj->get_self();
			r_results[cc].shape = shape_idx;
		}

		cc++;
	}

	return cc;
}

bool PhysicsDirectSpaceStateSW::cast_motion(const RID &p_shape, const Transform &p_xform, const Vector3 &p_motion, real_t p_margin, real_t &p_closest_safe, real_t &p_closest_unsafe, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, ShapeRestInfo *r_info) {
	ShapeSW *shape = static_cast<PhysicsServerSW *>(PhysicsServer::get_singleton())->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, false);

	AABB aabb = p_xform.xform(shape->get_aabb());
	aabb = aabb.merge(AABB(aabb.position + p_motion, aabb.size)); //motion
	aabb = aabb.grow(p_margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, SpaceSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	real_t best_safe = 1;
	real_t best_unsafe = 1;

	Transform xform_inv = p_xform.affine_inverse();
	MotionShapeSW mshape;
	mshape.shape = shape;
	mshape.motion = xform_inv.basis.xform(p_motion);

	bool best_first = true;

	Vector3 motion_normal = p_motion.normalized();

	Vector3 closest_A, closest_B;

	for (int i = 0; i < amount; i++) {
		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		if (p_exclude.has(space->intersection_query_results[i]->get_self())) {
			continue; //ignore excluded
		}

		const CollisionObjectSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		Vector3 point_A, point_B;
		Vector3 sep_axis = motion_normal;

		Transform col_obj_xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);
		//test initial overlap, does it collide if going all the way?
		if (CollisionSolverSW::solve_distance(&mshape, p_xform, col_obj->get_shape(shape_idx), col_obj_xform, point_A, point_B, aabb, &sep_axis)) {
			continue;
		}

		//test initial overlap, ignore objects it's inside of.
		sep_axis = motion_normal;

		if (!CollisionSolverSW::solve_distance(shape, p_xform, col_obj->get_shape(shape_idx), col_obj_xform, point_A, point_B, aabb, &sep_axis)) {
			continue;
		}

		//just do kinematic solving
		real_t low = 0.0;
		real_t hi = 1.0;
		real_t fraction_coeff = 0.5;
		for (int j = 0; j < 8; j++) { //steps should be customizable..
			real_t fraction = low + (hi - low) * fraction_coeff;

			mshape.motion = xform_inv.basis.xform(p_motion * fraction);

			Vector3 lA, lB;
			Vector3 sep = motion_normal; //important optimization for this to work fast enough
			bool collided = !CollisionSolverSW::solve_distance(&mshape, p_xform, col_obj->get_shape(shape_idx), col_obj_xform, lA, lB, aabb, &sep);

			if (collided) {
				hi = fraction;
				if ((j == 0) || (low > 0.0)) { // Did it not collide before?
					// When alternating or first iteration, use dichotomy.
					fraction_coeff = 0.5;
				} else {
					// When colliding again, converge faster towards low fraction
					// for more accurate results with long motions that collide near the start.
					fraction_coeff = 0.25;
				}
			} else {
				point_A = lA;
				point_B = lB;
				low = fraction;
				if ((j == 0) || (hi < 1.0)) { // Did it collide before?
					// When alternating or first iteration, use dichotomy.
					fraction_coeff = 0.5;
				} else {
					// When not colliding again, converge faster towards high fraction
					// for more accurate results with long motions that collide near the end.
					fraction_coeff = 0.75;
				}
			}
		}

		if (low < best_safe) {
			best_first = true; //force reset
			best_safe = low;
			best_unsafe = hi;
		}

		if (r_info && (best_first || (point_A.distance_squared_to(point_B) < closest_A.distance_squared_to(closest_B) && low <= best_safe))) {
			closest_A = point_A;
			closest_B = point_B;
			r_info->collider_id = col_obj->get_instance_id();
			r_info->rid = col_obj->get_self();
			r_info->shape = shape_idx;
			r_info->point = closest_B;
			r_info->normal = (closest_A - closest_B).normalized();
			best_first = false;
			if (col_obj->get_type() == CollisionObjectSW::TYPE_BODY) {
				const BodySW *body = static_cast<const BodySW *>(col_obj);
				Vector3 rel_vec = closest_B - (body->get_transform().origin + body->get_center_of_mass());
				r_info->linear_velocity = body->get_linear_velocity() + (body->get_angular_velocity()).cross(rel_vec);
			}
		}
	}

	p_closest_safe = best_safe;
	p_closest_unsafe = best_unsafe;

	return true;
}

bool PhysicsDirectSpaceStateSW::collide_shape(RID p_shape, const Transform &p_shape_xform, real_t p_margin, Vector3 *r_results, int p_result_max, int &r_result_count, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	if (p_result_max <= 0) {
		return false;
	}

	ShapeSW *shape = static_cast<PhysicsServerSW *>(PhysicsServer::get_singleton())->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	AABB aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.grow(p_margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, SpaceSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	bool collided = false;
	r_result_count = 0;

	PhysicsServerSW::CollCbkData cbk;
	cbk.max = p_result_max;
	cbk.amount = 0;
	cbk.ptr = r_results;
	CollisionSolverSW::CallbackResult cbkres = PhysicsServerSW::_shape_col_cbk;

	PhysicsServerSW::CollCbkData *cbkptr = &cbk;

	for (int i = 0; i < amount; i++) {
		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		const CollisionObjectSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		if (p_exclude.has(col_obj->get_self())) {
			continue;
		}

		if (CollisionSolverSW::solve_static(shape, p_shape_xform, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), cbkres, cbkptr, nullptr, p_margin)) {
			collided = true;
		}
	}

	r_result_count = cbk.amount;

	return collided;
}

struct _RestCallbackData {
	const CollisionObjectSW *object;
	const CollisionObjectSW *best_object;
	int local_shape;
	int best_local_shape;
	int shape;
	int best_shape;
	Vector3 best_contact;
	Vector3 best_normal;
	real_t best_len;
	real_t min_allowed_depth;
};

static void _rest_cbk_result(const Vector3 &p_point_A, const Vector3 &p_point_B, void *p_userdata) {
	_RestCallbackData *rd = (_RestCallbackData *)p_userdata;

	Vector3 contact_rel = p_point_B - p_point_A;
	real_t len = contact_rel.length();
	if (len < rd->min_allowed_depth) {
		return;
	}
	if (len <= rd->best_len) {
		return;
	}

	rd->best_len = len;
	rd->best_contact = p_point_B;
	rd->best_normal = contact_rel / len;
	rd->best_object = rd->object;
	rd->best_shape = rd->shape;
	rd->best_local_shape = rd->local_shape;
}
bool PhysicsDirectSpaceStateSW::rest_info(RID p_shape, const Transform &p_shape_xform, real_t p_margin, ShapeRestInfo *r_info, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	ShapeSW *shape = static_cast<PhysicsServerSW *>(PhysicsServer::get_singleton())->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	real_t margin = MAX(p_margin, TEST_MOTION_MARGIN_MIN_VALUE);

	real_t min_contact_depth = margin * TEST_MOTION_MIN_CONTACT_DEPTH_FACTOR;

	AABB aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.grow(margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, SpaceSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	_RestCallbackData rcd;
	rcd.best_len = 0;
	rcd.best_object = nullptr;
	rcd.best_shape = 0;
	rcd.min_allowed_depth = min_contact_depth;

	for (int i = 0; i < amount; i++) {
		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		const CollisionObjectSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		if (p_exclude.has(col_obj->get_self())) {
			continue;
		}

		rcd.object = col_obj;
		rcd.shape = shape_idx;
		bool sc = CollisionSolverSW::solve_static(shape, p_shape_xform, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), _rest_cbk_result, &rcd, nullptr, margin);
		if (!sc) {
			continue;
		}
	}

	if (rcd.best_len == 0 || !rcd.best_object) {
		return false;
	}

	r_info->collider_id = rcd.best_object->get_instance_id();
	r_info->shape = rcd.best_shape;
	r_info->normal = rcd.best_normal;
	r_info->point = rcd.best_contact;
	r_info->rid = rcd.best_object->get_self();
	if (rcd.best_object->get_type() == CollisionObjectSW::TYPE_BODY) {
		const BodySW *body = static_cast<const BodySW *>(rcd.best_object);
		Vector3 rel_vec = rcd.best_contact - (body->get_transform().origin + body->get_center_of_mass());
		r_info->linear_velocity = body->get_linear_velocity() + (body->get_angular_velocity()).cross(rel_vec);

	} else {
		r_info->linear_velocity = Vector3();
	}

	return true;
}

Vector3 PhysicsDirectSpaceStateSW::get_closest_point_to_object_volume(RID p_object, const Vector3 p_point) const {
	CollisionObjectSW *obj = PhysicsServerSW::singleton->area_owner.getornull(p_object);
	if (!obj) {
		obj = PhysicsServerSW::singleton->body_owner.getornull(p_object);
	}
	ERR_FAIL_COND_V(!obj, Vector3());

	ERR_FAIL_COND_V(obj->get_space() != space, Vector3());

	float min_distance = 1e20;
	Vector3 min_point;

	bool shapes_found = false;

	for (int i = 0; i < obj->get_shape_count(); i++) {
		if (obj->is_shape_disabled(i)) {
			continue;
		}

		Transform shape_xform = obj->get_transform() * obj->get_shape_transform(i);
		ShapeSW *shape = obj->get_shape(i);

		Vector3 point = shape->get_closest_point_to(shape_xform.affine_inverse().xform(p_point));
		point = shape_xform.xform(point);

		float dist = point.distance_to(p_point);
		if (dist < min_distance) {
			min_distance = dist;
			min_point = point;
		}
		shapes_found = true;
	}

	if (!shapes_found) {
		return obj->get_transform().origin; //no shapes found, use distance to origin.
	} else {
		return min_point;
	}
}

PhysicsDirectSpaceStateSW::PhysicsDirectSpaceStateSW() {
	space = nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

int SpaceSW::_cull_aabb_for_body(BodySW *p_body, const AABB &p_aabb) {
	int amount = broadphase->cull_aabb(p_aabb, intersection_query_results, INTERSECTION_QUERY_MAX, intersection_query_subindex_results);

	for (int i = 0; i < amount; i++) {
		bool keep = true;

		if (intersection_query_results[i] == p_body) {
			keep = false;
		} else if (intersection_query_results[i]->get_type() == CollisionObjectSW::TYPE_AREA) {
			keep = false;
		} else if ((static_cast<BodySW *>(intersection_query_results[i])->test_collision_mask(p_body)) == 0) {
			keep = false;
		} else if (static_cast<BodySW *>(intersection_query_results[i])->has_exception(p_body->get_self()) || p_body->has_exception(intersection_query_results[i]->get_self())) {
			keep = false;
		}

		if (!keep) {
			if (i < amount - 1) {
				SWAP(intersection_query_results[i], intersection_query_results[amount - 1]);
				SWAP(intersection_query_subindex_results[i], intersection_query_subindex_results[amount - 1]);
			}

			amount--;
			i--;
		}
	}

	return amount;
}

int SpaceSW::test_body_ray_separation(BodySW *p_body, const Transform &p_transform, bool p_infinite_inertia, Vector3 &r_recover_motion, PhysicsServer::SeparationResult *r_results, int p_result_max, real_t p_margin) {
	AABB body_aabb;

	bool shapes_found = false;

	for (int i = 0; i < p_body->get_shape_count(); i++) {
		if (p_body->is_shape_disabled(i)) {
			continue;
		}

		if (!shapes_found) {
			body_aabb = p_body->get_shape_aabb(i);
			shapes_found = true;
		} else {
			body_aabb = body_aabb.merge(p_body->get_shape_aabb(i));
		}
	}

	if (!shapes_found) {
		return 0;
	}
	// Undo the currently transform the physics server is aware of and apply the provided one
	body_aabb = p_transform.xform(p_body->get_inv_transform().xform(body_aabb));
	body_aabb = body_aabb.grow(p_margin);

	Transform body_transform = p_transform;

	for (int i = 0; i < p_result_max; i++) {
		//reset results
		r_results[i].collision_depth = -1.0;
	}

	int rays_found = 0;

	{
		// raycast AND separate

		const int max_results = 32;
		int recover_attempts = 4;
		Vector3 sr[max_results * 2];
		PhysicsServerSW::CollCbkData cbk;
		cbk.max = max_results;
		PhysicsServerSW::CollCbkData *cbkptr = &cbk;
		CollisionSolverSW::CallbackResult cbkres = PhysicsServerSW::_shape_col_cbk;

		do {
			Vector3 recover_motion;

			bool collided = false;

			int amount = _cull_aabb_for_body(p_body, body_aabb);

			for (int j = 0; j < p_body->get_shape_count(); j++) {
				if (p_body->is_shape_disabled(j)) {
					continue;
				}

				ShapeSW *body_shape = p_body->get_shape(j);

				if (body_shape->get_type() != PhysicsServer::SHAPE_RAY) {
					continue;
				}

				Transform body_shape_xform = body_transform * p_body->get_shape_transform(j);

				for (int i = 0; i < amount; i++) {
					const CollisionObjectSW *col_obj = intersection_query_results[i];
					int shape_idx = intersection_query_subindex_results[i];

					cbk.amount = 0;
					cbk.ptr = sr;

					if (CollisionObjectSW::TYPE_BODY == col_obj->get_type()) {
						const BodySW *b = static_cast<const BodySW *>(col_obj);
						if (p_infinite_inertia && PhysicsServer::BODY_MODE_STATIC != b->get_mode() && PhysicsServer::BODY_MODE_KINEMATIC != b->get_mode()) {
							continue;
						}
					}

					ShapeSW *against_shape = col_obj->get_shape(shape_idx);
					if (CollisionSolverSW::solve_static(body_shape, body_shape_xform, against_shape, col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), cbkres, cbkptr, nullptr, p_margin)) {
						if (cbk.amount > 0) {
							collided = true;
						}

						int ray_index = -1; //reuse shape
						for (int k = 0; k < rays_found; k++) {
							if (r_results[k].collision_local_shape == j) {
								ray_index = k;
							}
						}

						if (ray_index == -1 && rays_found < p_result_max) {
							ray_index = rays_found;
							rays_found++;
						}

						if (ray_index != -1) {
							PhysicsServer::SeparationResult &result = r_results[ray_index];

							for (int k = 0; k < cbk.amount; k++) {
								Vector3 a = sr[k * 2 + 0];
								Vector3 b = sr[k * 2 + 1];

								// Compute plane on b towards a.
								Vector3 n = (a - b).normalized();
								float d = n.dot(b);

								// Compute depth on recovered motion.
								float depth = n.dot(a + recover_motion) - d;

								// Apply recovery without margin.
								float separation_depth = depth - p_margin;
								if (separation_depth > 0.0) {
									// Only recover if there is penetration.
									recover_motion -= n * separation_depth;
								}

								if (depth > result.collision_depth) {
									result.collision_depth = depth;
									result.collision_point = b;
									result.collision_normal = -n;
									result.collision_local_shape = j;
									result.collider = col_obj->get_self();
									result.collider_id = col_obj->get_instance_id();
									result.collider_shape = shape_idx;
									if (col_obj->get_type() == CollisionObjectSW::TYPE_BODY) {
										BodySW *body = (BodySW *)col_obj;

										Vector3 rel_vec = b - (body->get_transform().origin + body->get_center_of_mass());
										result.collider_velocity = body->get_linear_velocity() + (body->get_angular_velocity()).cross(rel_vec);
									}
								}
							}
						}
					}
				}
			}

			if (!collided || recover_motion == Vector3()) {
				break;
			}

			body_transform.origin += recover_motion;
			body_aabb.position += recover_motion;

			recover_attempts--;
		} while (recover_attempts);
	}

	r_recover_motion = body_transform.origin - p_transform.origin;
	return rays_found;
}

bool SpaceSW::test_body_motion(BodySW *p_body, const Transform &p_from, const Vector3 &p_motion, bool p_infinite_inertia, real_t p_margin, PhysicsServer::MotionResult *r_result, bool p_exclude_raycast_shapes, const Set<RID> &p_exclude) {
	//give me back regular physics engine logic
	//this is madness
	//and most people using this function will think
	//what it does is simpler than using physics
	//this took about a week to get right..
	//but is it right? who knows at this point..

	if (r_result) {
		r_result->collider_id = 0;
		r_result->collider_shape = 0;
	}

	AABB body_aabb;
	bool shapes_found = false;

	for (int i = 0; i < p_body->get_shape_count(); i++) {
		if (p_body->is_shape_disabled(i)) {
			continue;
		}

		if (!shapes_found) {
			body_aabb = p_body->get_shape_aabb(i);
			shapes_found = true;
		} else {
			body_aabb = body_aabb.merge(p_body->get_shape_aabb(i));
		}
	}

	if (!shapes_found) {
		if (r_result) {
			*r_result = PhysicsServer::MotionResult();
			r_result->motion = p_motion;
		}

		return false;
	}

	real_t margin = MAX(p_margin, TEST_MOTION_MARGIN_MIN_VALUE);

	// Undo the currently transform the physics server is aware of and apply the provided one
	body_aabb = p_from.xform(p_body->get_inv_transform().xform(body_aabb));
	body_aabb = body_aabb.grow(margin);

	real_t min_contact_depth = margin * TEST_MOTION_MIN_CONTACT_DEPTH_FACTOR;

	float motion_length = p_motion.length();
	Vector3 motion_normal = p_motion / motion_length;

	Transform body_transform = p_from;

	bool recovered = false;

	{
		//STEP 1, FREE BODY IF STUCK

		const int max_results = 32;
		int recover_attempts = 4;
		Vector3 sr[max_results * 2];

		do {
			PhysicsServerSW::CollCbkData cbk;
			cbk.max = max_results;
			cbk.amount = 0;
			cbk.ptr = sr;

			PhysicsServerSW::CollCbkData *cbkptr = &cbk;
			CollisionSolverSW::CallbackResult cbkres = PhysicsServerSW::_shape_col_cbk;

			bool collided = false;

			int amount = _cull_aabb_for_body(p_body, body_aabb);

			for (int j = 0; j < p_body->get_shape_count(); j++) {
				if (p_body->is_shape_disabled(j)) {
					continue;
				}

				Transform body_shape_xform = body_transform * p_body->get_shape_transform(j);
				ShapeSW *body_shape = p_body->get_shape(j);
				if (p_exclude_raycast_shapes && body_shape->get_type() == PhysicsServer::SHAPE_RAY) {
					continue;
				}

				for (int i = 0; i < amount; i++) {
					const CollisionObjectSW *col_obj = intersection_query_results[i];
					if (p_exclude.has(col_obj->get_self())) {
						continue;
					}
					int shape_idx = intersection_query_subindex_results[i];

					if (CollisionObjectSW::TYPE_BODY == col_obj->get_type()) {
						const BodySW *b = static_cast<const BodySW *>(col_obj);
						if (p_infinite_inertia && PhysicsServer::BODY_MODE_STATIC != b->get_mode() && PhysicsServer::BODY_MODE_KINEMATIC != b->get_mode()) {
							continue;
						}
					}

					if (CollisionSolverSW::solve_static(body_shape, body_shape_xform, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), cbkres, cbkptr, nullptr, margin)) {
						collided = cbk.amount > 0;
					}
				}
			}

			if (!collided) {
				break;
			}

			recovered = true;

			Vector3 recover_motion;
			for (int i = 0; i < cbk.amount; i++) {
				Vector3 a = sr[i * 2 + 0];
				Vector3 b = sr[i * 2 + 1];

				// Compute plane on b towards a.
				Vector3 n = (a - b).normalized();
				float d = n.dot(b);

				// Compute depth on recovered motion.
				float depth = n.dot(a + recover_motion) - d;
				if (depth > min_contact_depth + CMP_EPSILON) {
					// Only recover if there is penetration.
					recover_motion -= n * (depth - min_contact_depth) * 0.4;
				}
			}

			if (recover_motion == Vector3()) {
				collided = false;
				break;
			}

			body_transform.origin += recover_motion;
			body_aabb.position += recover_motion;

			recover_attempts--;

		} while (recover_attempts);
	}

	real_t safe = 1.0;
	real_t unsafe = 1.0;
	int best_shape = -1;

	{
		// STEP 2 ATTEMPT MOTION

		AABB motion_aabb = body_aabb;
		motion_aabb.position += p_motion;
		motion_aabb = motion_aabb.merge(body_aabb);

		int amount = _cull_aabb_for_body(p_body, motion_aabb);

		for (int j = 0; j < p_body->get_shape_count(); j++) {
			if (p_body->is_shape_disabled(j)) {
				continue;
			}

			Transform body_shape_xform = body_transform * p_body->get_shape_transform(j);
			ShapeSW *body_shape = p_body->get_shape(j);

			if (p_exclude_raycast_shapes && body_shape->get_type() == PhysicsServer::SHAPE_RAY) {
				continue;
			}

			Transform body_shape_xform_inv = body_shape_xform.affine_inverse();
			MotionShapeSW mshape;
			mshape.shape = body_shape;
			mshape.motion = body_shape_xform_inv.basis.xform(p_motion);

			bool stuck = false;

			real_t best_safe = 1;
			real_t best_unsafe = 1;

			for (int i = 0; i < amount; i++) {
				const CollisionObjectSW *col_obj = intersection_query_results[i];
				if (p_exclude.has(col_obj->get_self())) {
					continue;
				}
				int shape_idx = intersection_query_subindex_results[i];

				if (CollisionObjectSW::TYPE_BODY == col_obj->get_type()) {
					const BodySW *b = static_cast<const BodySW *>(col_obj);
					if (p_infinite_inertia && PhysicsServer::BODY_MODE_STATIC != b->get_mode() && PhysicsServer::BODY_MODE_KINEMATIC != b->get_mode()) {
						continue;
					}
				}

				//test initial overlap, does it collide if going all the way?
				Vector3 point_A, point_B;
				Vector3 sep_axis = motion_normal;

				Transform col_obj_xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);
				//test initial overlap, does it collide if going all the way?
				if (CollisionSolverSW::solve_distance(&mshape, body_shape_xform, col_obj->get_shape(shape_idx), col_obj_xform, point_A, point_B, motion_aabb, &sep_axis)) {
					continue;
				}
				sep_axis = motion_normal;

				if (!CollisionSolverSW::solve_distance(body_shape, body_shape_xform, col_obj->get_shape(shape_idx), col_obj_xform, point_A, point_B, motion_aabb, &sep_axis)) {
					stuck = true;
					break;
				}

				//just do kinematic solving
				real_t low = 0.0;
				real_t hi = 1.0;
				real_t fraction_coeff = 0.5;
				for (int k = 0; k < 8; k++) { //steps should be customizable..
					real_t fraction = low + (hi - low) * fraction_coeff;

					mshape.motion = body_shape_xform_inv.basis.xform(p_motion * fraction);

					Vector3 lA, lB;
					Vector3 sep = motion_normal; //important optimization for this to work fast enough
					bool collided = !CollisionSolverSW::solve_distance(&mshape, body_shape_xform, col_obj->get_shape(shape_idx), col_obj_xform, lA, lB, motion_aabb, &sep);

					if (collided) {
						hi = fraction;
						if ((k == 0) || (low > 0.0)) { // Did it not collide before?
							// When alternating or first iteration, use dichotomy.
							fraction_coeff = 0.5;
						} else {
							// When colliding again, converge faster towards low fraction
							// for more accurate results with long motions that collide near the start.
							fraction_coeff = 0.25;
						}
					} else {
						point_A = lA;
						point_B = lB;
						low = fraction;
						if ((k == 0) || (hi < 1.0)) { // Did it collide before?
							// When alternating or first iteration, use dichotomy.
							fraction_coeff = 0.5;
						} else {
							// When not colliding again, converge faster towards high fraction
							// for more accurate results with long motions that collide near the end.
							fraction_coeff = 0.75;
						}
					}
				}

				if (low < best_safe) {
					best_safe = low;
					best_unsafe = hi;
				}
			}

			if (stuck) {
				safe = 0;
				unsafe = 0;
				best_shape = j; //sadly it's the best
				break;
			}
			if (best_safe == 1.0) {
				continue;
			}
			if (best_safe < safe) {
				safe = best_safe;
				unsafe = best_unsafe;
				best_shape = j;
			}
		}
	}

	bool collided = false;

	if (recovered || (safe < 1)) {
		if (safe >= 1) {
			best_shape = -1; //no best shape with cast, reset to -1
		}

		//it collided, let's get the rest info in unsafe advance
		Transform ugt = body_transform;
		ugt.origin += p_motion * unsafe;

		_RestCallbackData rcd;
		rcd.best_len = 0;
		rcd.best_object = nullptr;
		rcd.best_shape = 0;

		// Allowed depth can't be lower than motion length, in order to handle contacts at low speed.
		rcd.min_allowed_depth = MIN(motion_length, min_contact_depth);

		body_aabb.position += p_motion * unsafe;
		int amount = _cull_aabb_for_body(p_body, body_aabb);

		int from_shape = best_shape != -1 ? best_shape : 0;
		int to_shape = best_shape != -1 ? best_shape + 1 : p_body->get_shape_count();

		for (int j = from_shape; j < to_shape; j++) {
			if (p_body->is_shape_disabled(j)) {
				continue;
			}

			Transform body_shape_xform = ugt * p_body->get_shape_transform(j);
			ShapeSW *body_shape = p_body->get_shape(j);

			if (p_exclude_raycast_shapes && body_shape->get_type() == PhysicsServer::SHAPE_RAY) {
				continue;
			}

			for (int i = 0; i < amount; i++) {
				const CollisionObjectSW *col_obj = intersection_query_results[i];
				if (p_exclude.has(col_obj->get_self())) {
					continue;
				}
				int shape_idx = intersection_query_subindex_results[i];

				if (CollisionObjectSW::TYPE_BODY == col_obj->get_type()) {
					const BodySW *b = static_cast<const BodySW *>(col_obj);
					if (p_infinite_inertia && PhysicsServer::BODY_MODE_STATIC != b->get_mode() && PhysicsServer::BODY_MODE_KINEMATIC != b->get_mode()) {
						continue;
					}
				}

				rcd.object = col_obj;
				rcd.shape = shape_idx;
				rcd.local_shape = j;
				bool sc = CollisionSolverSW::solve_static(body_shape, body_shape_xform, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), _rest_cbk_result, &rcd, nullptr, margin);
				if (!sc) {
					continue;
				}
			}
		}

		if (rcd.best_len != 0) {
			if (r_result) {
				r_result->collider = rcd.best_object->get_self();
				r_result->collider_id = rcd.best_object->get_instance_id();
				r_result->collider_shape = rcd.best_shape;
				r_result->collision_local_shape = rcd.best_local_shape;
				r_result->collision_normal = rcd.best_normal;
				r_result->collision_point = rcd.best_contact;
				r_result->collision_depth = rcd.best_len;
				r_result->collision_safe_fraction = safe;
				r_result->collision_unsafe_fraction = unsafe;
				//r_result->collider_metadata = rcd.best_object->get_shape_metadata(rcd.best_shape);

				const BodySW *body = static_cast<const BodySW *>(rcd.best_object);

				Vector3 rel_vec = rcd.best_contact - (body->get_transform().origin + body->get_center_of_mass());
				r_result->collider_velocity = body->get_linear_velocity() + (body->get_angular_velocity()).cross(rel_vec);

				r_result->motion = safe * p_motion;
				r_result->remainder = p_motion - safe * p_motion;
				r_result->motion += (body_transform.get_origin() - p_from.get_origin());
			}

			collided = true;
		}
	}

	if (!collided && r_result) {
		r_result->motion = p_motion;
		r_result->remainder = Vector3();
		r_result->motion += (body_transform.get_origin() - p_from.get_origin());
	}

	return collided;
}

// Assumes a valid collision pair, this should have been checked beforehand in the BVH or octree.
void *SpaceSW::_broadphase_pair(CollisionObjectSW *p_object_A, int p_subindex_A, CollisionObjectSW *p_object_B, int p_subindex_B, void *p_pair_data, void *p_self) {
	// An existing pair - nothing to do, pair is still valid.
	if (p_pair_data) {
		return p_pair_data;
	}

	// New pair
	CollisionObjectSW::Type type_A = p_object_A->get_type();
	CollisionObjectSW::Type type_B = p_object_B->get_type();
	if (type_A > type_B) {
		SWAP(p_object_A, p_object_B);
		SWAP(p_subindex_A, p_subindex_B);
		SWAP(type_A, type_B);
	}

	SpaceSW *self = (SpaceSW *)p_self;

	self->collision_pairs++;

	if (type_A == CollisionObjectSW::TYPE_AREA) {
		AreaSW *area_a = static_cast<AreaSW *>(p_object_A);
		if (type_B == CollisionObjectSW::TYPE_AREA) {
			AreaSW *area_b = static_cast<AreaSW *>(p_object_B);
			Area2PairSW *area2_pair = memnew(Area2PairSW(area_b, p_subindex_B, area_a, p_subindex_A));
			return area2_pair;
		} else {
			BodySW *body_b = static_cast<BodySW *>(p_object_B);
			AreaPairSW *area_pair = memnew(AreaPairSW(body_b, p_subindex_B, area_a, p_subindex_A));
			return area_pair;
		}
	} else {
		BodySW *body_a = static_cast<BodySW *>(p_object_A);
		BodySW *body_b = static_cast<BodySW *>(p_object_B);
		BodyPairSW *body_pair = memnew(BodyPairSW(body_a, p_subindex_A, body_b, p_subindex_B));
		return body_pair;
	}

	return nullptr;
}

void SpaceSW::_broadphase_unpair(CollisionObjectSW *p_object_A, int p_subindex_A, CollisionObjectSW *p_object_B, int p_subindex_B, void *p_pair_data, void *p_self) {
	if (!p_pair_data) {
		return;
	}

	SpaceSW *self = (SpaceSW *)p_self;
	self->collision_pairs--;
	ConstraintSW *c = (ConstraintSW *)p_pair_data;
	memdelete(c);
}

const SelfList<BodySW>::List &SpaceSW::get_active_body_list() const {
	return active_list;
}
void SpaceSW::body_add_to_active_list(SelfList<BodySW> *p_body) {
	active_list.add(p_body);
}
void SpaceSW::body_remove_from_active_list(SelfList<BodySW> *p_body) {
	active_list.remove(p_body);
}

void SpaceSW::body_add_to_inertia_update_list(SelfList<BodySW> *p_body) {
	inertia_update_list.add(p_body);
}

void SpaceSW::body_remove_from_inertia_update_list(SelfList<BodySW> *p_body) {
	inertia_update_list.remove(p_body);
}

BroadPhaseSW *SpaceSW::get_broadphase() {
	return broadphase;
}

void SpaceSW::add_object(CollisionObjectSW *p_object) {
	ERR_FAIL_COND(objects.has(p_object));
	objects.insert(p_object);
}

void SpaceSW::remove_object(CollisionObjectSW *p_object) {
	ERR_FAIL_COND(!objects.has(p_object));
	objects.erase(p_object);
}

const Set<CollisionObjectSW *> &SpaceSW::get_objects() const {
	return objects;
}

void SpaceSW::body_add_to_state_query_list(SelfList<BodySW> *p_body) {
	state_query_list.add(p_body);
}
void SpaceSW::body_remove_from_state_query_list(SelfList<BodySW> *p_body) {
	state_query_list.remove(p_body);
}

void SpaceSW::area_add_to_monitor_query_list(SelfList<AreaSW> *p_area) {
	monitor_query_list.add(p_area);
}
void SpaceSW::area_remove_from_monitor_query_list(SelfList<AreaSW> *p_area) {
	monitor_query_list.remove(p_area);
}

void SpaceSW::area_add_to_moved_list(SelfList<AreaSW> *p_area) {
	area_moved_list.add(p_area);
}

void SpaceSW::area_remove_from_moved_list(SelfList<AreaSW> *p_area) {
	area_moved_list.remove(p_area);
}

const SelfList<AreaSW>::List &SpaceSW::get_moved_area_list() const {
	return area_moved_list;
}

void SpaceSW::call_queries() {
	while (state_query_list.first()) {
		BodySW *b = state_query_list.first()->self();
		state_query_list.remove(state_query_list.first());
		b->call_queries();
	}

	while (monitor_query_list.first()) {
		AreaSW *a = monitor_query_list.first()->self();
		monitor_query_list.remove(monitor_query_list.first());
		a->call_queries();
	}
}

void SpaceSW::setup() {
	contact_debug_count = 0;
	while (inertia_update_list.first()) {
		inertia_update_list.first()->self()->update_inertias();
		inertia_update_list.remove(inertia_update_list.first());
	}
}

void SpaceSW::update() {
	broadphase->update();
}

void SpaceSW::set_param(PhysicsServer::SpaceParameter p_param, real_t p_value) {
	switch (p_param) {
		case PhysicsServer::SPACE_PARAM_CONTACT_RECYCLE_RADIUS:
			contact_recycle_radius = p_value;
			break;
		case PhysicsServer::SPACE_PARAM_CONTACT_MAX_SEPARATION:
			contact_max_separation = p_value;
			break;
		case PhysicsServer::SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION:
			contact_max_allowed_penetration = p_value;
			break;
		case PhysicsServer::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD:
			body_linear_velocity_sleep_threshold = p_value;
			break;
		case PhysicsServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD:
			body_angular_velocity_sleep_threshold = p_value;
			break;
		case PhysicsServer::SPACE_PARAM_BODY_TIME_TO_SLEEP:
			body_time_to_sleep = p_value;
			break;
		case PhysicsServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_DAMP_RATIO:
			body_angular_velocity_damp_ratio = p_value;
			break;
		case PhysicsServer::SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS:
			constraint_bias = p_value;
			break;
	}
}

real_t SpaceSW::get_param(PhysicsServer::SpaceParameter p_param) const {
	switch (p_param) {
		case PhysicsServer::SPACE_PARAM_CONTACT_RECYCLE_RADIUS:
			return contact_recycle_radius;
		case PhysicsServer::SPACE_PARAM_CONTACT_MAX_SEPARATION:
			return contact_max_separation;
		case PhysicsServer::SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION:
			return contact_max_allowed_penetration;
		case PhysicsServer::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD:
			return body_linear_velocity_sleep_threshold;
		case PhysicsServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD:
			return body_angular_velocity_sleep_threshold;
		case PhysicsServer::SPACE_PARAM_BODY_TIME_TO_SLEEP:
			return body_time_to_sleep;
		case PhysicsServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_DAMP_RATIO:
			return body_angular_velocity_damp_ratio;
		case PhysicsServer::SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS:
			return constraint_bias;
	}
	return 0;
}

void SpaceSW::lock() {
	locked = true;
}

void SpaceSW::unlock() {
	locked = false;
}

bool SpaceSW::is_locked() const {
	return locked;
}

PhysicsDirectSpaceStateSW *SpaceSW::get_direct_state() {
	return direct_access;
}

SpaceSW::SpaceSW() {
	collision_pairs = 0;
	active_objects = 0;
	island_count = 0;
	contact_debug_count = 0;

	locked = false;
	contact_recycle_radius = 0.01;
	contact_max_separation = 0.05;
	contact_max_allowed_penetration = 0.01;

	constraint_bias = 0.01;
	body_linear_velocity_sleep_threshold = GLOBAL_DEF("physics/3d/sleep_threshold_linear", 0.1);
	body_angular_velocity_sleep_threshold = GLOBAL_DEF("physics/3d/sleep_threshold_angular", (8.0 / 180.0 * Math_PI));
	body_time_to_sleep = GLOBAL_DEF("physics/3d/time_before_sleep", 0.5);
	ProjectSettings::get_singleton()->set_custom_property_info("physics/3d/time_before_sleep", PropertyInfo(Variant::REAL, "physics/3d/time_before_sleep", PROPERTY_HINT_RANGE, "0,5,0.01,or_greater"));
	body_angular_velocity_damp_ratio = 10;

	broadphase = BroadPhaseSW::create_func();
	broadphase->set_pair_callback(_broadphase_pair, this);
	broadphase->set_unpair_callback(_broadphase_unpair, this);
	area = nullptr;

	direct_access = memnew(PhysicsDirectSpaceStateSW);
	direct_access->space = this;

	for (int i = 0; i < ELAPSED_TIME_MAX; i++) {
		elapsed_time[i] = 0;
	}
}

SpaceSW::~SpaceSW() {
	memdelete(broadphase);
	memdelete(direct_access);
}
