/*************************************************************************/
/*  space_2d_sw.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "space_2d_sw.h"

#include "collision_solver_2d_sw.h"
#include "core/os/os.h"
#include "core/templates/pair.h"
#include "physics_server_2d_sw.h"
_FORCE_INLINE_ static bool _can_collide_with(CollisionObject2DSW *p_object, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	if (!(p_object->get_collision_layer() & p_collision_mask)) {
		return false;
	}

	if (p_object->get_type() == CollisionObject2DSW::TYPE_AREA && !p_collide_with_areas) {
		return false;
	}

	if (p_object->get_type() == CollisionObject2DSW::TYPE_BODY && !p_collide_with_bodies) {
		return false;
	}

	return true;
}

int PhysicsDirectSpaceState2DSW::_intersect_point_impl(const Vector2 &p_point, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_pick_point, bool p_filter_by_canvas, ObjectID p_canvas_instance_id) {
	if (p_result_max <= 0) {
		return 0;
	}

	Rect2 aabb;
	aabb.position = p_point - Vector2(0.00001, 0.00001);
	aabb.size = Vector2(0.00002, 0.00002);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	int cc = 0;

	for (int i = 0; i < amount; i++) {
		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		if (p_exclude.has(space->intersection_query_results[i]->get_self())) {
			continue;
		}

		const CollisionObject2DSW *col_obj = space->intersection_query_results[i];

		if (p_pick_point && !col_obj->is_pickable()) {
			continue;
		}

		if (p_filter_by_canvas && col_obj->get_canvas_instance_id() != p_canvas_instance_id) {
			continue;
		}

		int shape_idx = space->intersection_query_subindex_results[i];

		if (col_obj->is_shape_set_as_disabled(shape_idx)) {
			continue;
		}

		Shape2DSW *shape = col_obj->get_shape(shape_idx);

		Vector2 local_point = (col_obj->get_transform() * col_obj->get_shape_transform(shape_idx)).affine_inverse().xform(p_point);

		if (!shape->contains_point(local_point)) {
			continue;
		}

		if (cc >= p_result_max) {
			continue;
		}

		r_results[cc].collider_id = col_obj->get_instance_id();
		if (r_results[cc].collider_id.is_valid()) {
			r_results[cc].collider = ObjectDB::get_instance(r_results[cc].collider_id);
		}
		r_results[cc].rid = col_obj->get_self();
		r_results[cc].shape = shape_idx;
		r_results[cc].metadata = col_obj->get_shape_metadata(shape_idx);

		cc++;
	}

	return cc;
}

int PhysicsDirectSpaceState2DSW::intersect_point(const Vector2 &p_point, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_pick_point) {
	return _intersect_point_impl(p_point, r_results, p_result_max, p_exclude, p_collision_mask, p_collide_with_bodies, p_collide_with_areas, p_pick_point);
}

int PhysicsDirectSpaceState2DSW::intersect_point_on_canvas(const Vector2 &p_point, ObjectID p_canvas_instance_id, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_pick_point) {
	return _intersect_point_impl(p_point, r_results, p_result_max, p_exclude, p_collision_mask, p_collide_with_bodies, p_collide_with_areas, p_pick_point, true, p_canvas_instance_id);
}

bool PhysicsDirectSpaceState2DSW::intersect_ray(const Vector2 &p_from, const Vector2 &p_to, RayResult &r_result, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	ERR_FAIL_COND_V(space->locked, false);

	Vector2 begin, end;
	Vector2 normal;
	begin = p_from;
	end = p_to;
	normal = (end - begin).normalized();

	int amount = space->broadphase->cull_segment(begin, end, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	//todo, create another array that references results, compute AABBs and check closest point to ray origin, sort, and stop evaluating results when beyond first collision

	bool collided = false;
	Vector2 res_point, res_normal;
	int res_shape;
	const CollisionObject2DSW *res_obj;
	real_t min_d = 1e10;

	for (int i = 0; i < amount; i++) {
		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		if (p_exclude.has(space->intersection_query_results[i]->get_self())) {
			continue;
		}

		const CollisionObject2DSW *col_obj = space->intersection_query_results[i];

		int shape_idx = space->intersection_query_subindex_results[i];
		Transform2D inv_xform = col_obj->get_shape_inv_transform(shape_idx) * col_obj->get_inv_transform();

		Vector2 local_from = inv_xform.xform(begin);
		Vector2 local_to = inv_xform.xform(end);

		/*local_from = col_obj->get_inv_transform().xform(begin);
		local_from = col_obj->get_shape_inv_transform(shape_idx).xform(local_from);

		local_to = col_obj->get_inv_transform().xform(end);
		local_to = col_obj->get_shape_inv_transform(shape_idx).xform(local_to);*/

		const Shape2DSW *shape = col_obj->get_shape(shape_idx);

		Vector2 shape_point, shape_normal;

		if (shape->intersect_segment(local_from, local_to, shape_point, shape_normal)) {
			Transform2D xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);
			shape_point = xform.xform(shape_point);

			real_t ld = normal.dot(shape_point);

			if (ld < min_d) {
				min_d = ld;
				res_point = shape_point;
				res_normal = inv_xform.basis_xform_inv(shape_normal).normalized();
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
	if (r_result.collider_id.is_valid()) {
		r_result.collider = ObjectDB::get_instance(r_result.collider_id);
	}
	r_result.normal = res_normal;
	r_result.metadata = res_obj->get_shape_metadata(res_shape);
	r_result.position = res_point;
	r_result.rid = res_obj->get_self();
	r_result.shape = res_shape;

	return true;
}

int PhysicsDirectSpaceState2DSW::intersect_shape(const RID &p_shape, const Transform2D &p_xform, const Vector2 &p_motion, real_t p_margin, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	if (p_result_max <= 0) {
		return 0;
	}

	Shape2DSW *shape = PhysicsServer2DSW::singletonsw->shape_owner.getornull(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	Rect2 aabb = p_xform.xform(shape->get_aabb());
	aabb = aabb.grow(p_margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	int cc = 0;

	for (int i = 0; i < amount; i++) {
		if (cc >= p_result_max) {
			break;
		}

		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		if (p_exclude.has(space->intersection_query_results[i]->get_self())) {
			continue;
		}

		const CollisionObject2DSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		if (col_obj->is_shape_set_as_disabled(shape_idx)) {
			continue;
		}

		if (!CollisionSolver2DSW::solve(shape, p_xform, p_motion, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), Vector2(), nullptr, nullptr, nullptr, p_margin)) {
			continue;
		}

		r_results[cc].collider_id = col_obj->get_instance_id();
		if (r_results[cc].collider_id.is_valid()) {
			r_results[cc].collider = ObjectDB::get_instance(r_results[cc].collider_id);
		}
		r_results[cc].rid = col_obj->get_self();
		r_results[cc].shape = shape_idx;
		r_results[cc].metadata = col_obj->get_shape_metadata(shape_idx);

		cc++;
	}

	return cc;
}

bool PhysicsDirectSpaceState2DSW::cast_motion(const RID &p_shape, const Transform2D &p_xform, const Vector2 &p_motion, real_t p_margin, real_t &p_closest_safe, real_t &p_closest_unsafe, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	Shape2DSW *shape = PhysicsServer2DSW::singletonsw->shape_owner.getornull(p_shape);
	ERR_FAIL_COND_V(!shape, false);

	Rect2 aabb = p_xform.xform(shape->get_aabb());
	aabb = aabb.merge(Rect2(aabb.position + p_motion, aabb.size)); //motion
	aabb = aabb.grow(p_margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	real_t best_safe = 1;
	real_t best_unsafe = 1;

	for (int i = 0; i < amount; i++) {
		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		if (p_exclude.has(space->intersection_query_results[i]->get_self())) {
			continue; //ignore excluded
		}

		const CollisionObject2DSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		if (col_obj->is_shape_set_as_disabled(shape_idx)) {
			continue;
		}

		Transform2D col_obj_xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);
		//test initial overlap, does it collide if going all the way?
		if (!CollisionSolver2DSW::solve(shape, p_xform, p_motion, col_obj->get_shape(shape_idx), col_obj_xform, Vector2(), nullptr, nullptr, nullptr, p_margin)) {
			continue;
		}

		//test initial overlap, ignore objects it's inside of.
		if (CollisionSolver2DSW::solve(shape, p_xform, Vector2(), col_obj->get_shape(shape_idx), col_obj_xform, Vector2(), nullptr, nullptr, nullptr, p_margin)) {
			continue;
		}

		//just do kinematic solving
		real_t low = 0;
		real_t hi = 1;
		Vector2 mnormal = p_motion.normalized();

		for (int j = 0; j < 8; j++) { //steps should be customizable..

			real_t ofs = (low + hi) * 0.5;

			Vector2 sep = mnormal; //important optimization for this to work fast enough
			bool collided = CollisionSolver2DSW::solve(shape, p_xform, p_motion * ofs, col_obj->get_shape(shape_idx), col_obj_xform, Vector2(), nullptr, nullptr, &sep, p_margin);

			if (collided) {
				hi = ofs;
			} else {
				low = ofs;
			}
		}

		if (low < best_safe) {
			best_safe = low;
			best_unsafe = hi;
		}
	}

	p_closest_safe = best_safe;
	p_closest_unsafe = best_unsafe;

	return true;
}

bool PhysicsDirectSpaceState2DSW::collide_shape(RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, real_t p_margin, Vector2 *r_results, int p_result_max, int &r_result_count, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	if (p_result_max <= 0) {
		return false;
	}

	Shape2DSW *shape = PhysicsServer2DSW::singletonsw->shape_owner.getornull(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	Rect2 aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.merge(Rect2(aabb.position + p_motion, aabb.size)); //motion
	aabb = aabb.grow(p_margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	bool collided = false;
	r_result_count = 0;

	PhysicsServer2DSW::CollCbkData cbk;
	cbk.max = p_result_max;
	cbk.amount = 0;
	cbk.passed = 0;
	cbk.ptr = r_results;
	CollisionSolver2DSW::CallbackResult cbkres = PhysicsServer2DSW::_shape_col_cbk;

	PhysicsServer2DSW::CollCbkData *cbkptr = &cbk;

	for (int i = 0; i < amount; i++) {
		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		const CollisionObject2DSW *col_obj = space->intersection_query_results[i];

		if (p_exclude.has(col_obj->get_self())) {
			continue;
		}

		int shape_idx = space->intersection_query_subindex_results[i];

		if (col_obj->is_shape_set_as_disabled(shape_idx)) {
			continue;
		}

		cbk.valid_dir = Vector2();
		cbk.valid_depth = 0;

		if (CollisionSolver2DSW::solve(shape, p_shape_xform, p_motion, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), Vector2(), cbkres, cbkptr, nullptr, p_margin)) {
			collided = cbk.amount > 0;
		}
	}

	r_result_count = cbk.amount;

	return collided;
}

struct _RestCallbackData2D {
	const CollisionObject2DSW *object;
	const CollisionObject2DSW *best_object;
	int local_shape;
	int best_local_shape;
	int shape;
	int best_shape;
	Vector2 best_contact;
	Vector2 best_normal;
	real_t best_len;
	Vector2 valid_dir;
	real_t valid_depth;
	real_t min_allowed_depth;
};

static void _rest_cbk_result(const Vector2 &p_point_A, const Vector2 &p_point_B, void *p_userdata) {
	_RestCallbackData2D *rd = (_RestCallbackData2D *)p_userdata;

	Vector2 contact_rel = p_point_B - p_point_A;
	real_t len = contact_rel.length();

	if (len < rd->min_allowed_depth) {
		return;
	}

	if (len <= rd->best_len) {
		return;
	}

	Vector2 normal = contact_rel / len;

	if (rd->valid_dir != Vector2()) {
		if (len > rd->valid_depth) {
			return;
		}

		if (rd->valid_dir.dot(normal) > -CMP_EPSILON) {
			return;
		}
	}

	rd->best_len = len;
	rd->best_contact = p_point_B;
	rd->best_normal = normal;
	rd->best_object = rd->object;
	rd->best_shape = rd->shape;
	rd->best_local_shape = rd->local_shape;
}

bool PhysicsDirectSpaceState2DSW::rest_info(RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, real_t p_margin, ShapeRestInfo *r_info, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	Shape2DSW *shape = PhysicsServer2DSW::singletonsw->shape_owner.getornull(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	Rect2 aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.merge(Rect2(aabb.position + p_motion, aabb.size)); //motion
	aabb = aabb.grow(p_margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	_RestCallbackData2D rcd;
	rcd.best_len = 0;
	rcd.best_object = nullptr;
	rcd.best_shape = 0;
	rcd.min_allowed_depth = space->test_motion_min_contact_depth;

	for (int i = 0; i < amount; i++) {
		if (!_can_collide_with(space->intersection_query_results[i], p_collision_mask, p_collide_with_bodies, p_collide_with_areas)) {
			continue;
		}

		const CollisionObject2DSW *col_obj = space->intersection_query_results[i];

		if (p_exclude.has(col_obj->get_self())) {
			continue;
		}

		int shape_idx = space->intersection_query_subindex_results[i];

		if (col_obj->is_shape_set_as_disabled(shape_idx)) {
			continue;
		}

		rcd.valid_dir = Vector2();
		rcd.object = col_obj;
		rcd.shape = shape_idx;
		rcd.local_shape = 0;
		bool sc = CollisionSolver2DSW::solve(shape, p_shape_xform, p_motion, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), Vector2(), _rest_cbk_result, &rcd, nullptr, p_margin);
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
	r_info->metadata = rcd.best_object->get_shape_metadata(rcd.best_shape);
	if (rcd.best_object->get_type() == CollisionObject2DSW::TYPE_BODY) {
		const Body2DSW *body = static_cast<const Body2DSW *>(rcd.best_object);
		Vector2 rel_vec = r_info->point - body->get_transform().get_origin();
		r_info->linear_velocity = Vector2(-body->get_angular_velocity() * rel_vec.y, body->get_angular_velocity() * rel_vec.x) + body->get_linear_velocity();

	} else {
		r_info->linear_velocity = Vector2();
	}

	return true;
}

PhysicsDirectSpaceState2DSW::PhysicsDirectSpaceState2DSW() {
	space = nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

int Space2DSW::_cull_aabb_for_body(Body2DSW *p_body, const Rect2 &p_aabb) {
	int amount = broadphase->cull_aabb(p_aabb, intersection_query_results, INTERSECTION_QUERY_MAX, intersection_query_subindex_results);

	for (int i = 0; i < amount; i++) {
		bool keep = true;

		if (intersection_query_results[i] == p_body) {
			keep = false;
		} else if (intersection_query_results[i]->get_type() == CollisionObject2DSW::TYPE_AREA) {
			keep = false;
		} else if ((static_cast<Body2DSW *>(intersection_query_results[i])->test_collision_mask(p_body)) == 0) {
			keep = false;
		} else if (static_cast<Body2DSW *>(intersection_query_results[i])->has_exception(p_body->get_self()) || p_body->has_exception(intersection_query_results[i]->get_self())) {
			keep = false;
		} else if (static_cast<Body2DSW *>(intersection_query_results[i])->is_shape_set_as_disabled(intersection_query_subindex_results[i])) {
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

int Space2DSW::test_body_ray_separation(Body2DSW *p_body, const Transform2D &p_transform, bool p_infinite_inertia, Vector2 &r_recover_motion, PhysicsServer2D::SeparationResult *r_results, int p_result_max, real_t p_margin) {
	Rect2 body_aabb;

	bool shapes_found = false;

	for (int i = 0; i < p_body->get_shape_count(); i++) {
		if (p_body->is_shape_set_as_disabled(i)) {
			continue;
		}

		if (p_body->get_shape(i)->get_type() != PhysicsServer2D::SHAPE_RAY) {
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

	Transform2D body_transform = p_transform;

	for (int i = 0; i < p_result_max; i++) {
		//reset results
		r_results[i].collision_depth = 0;
	}

	int rays_found = 0;

	{
		// raycast AND separate

		const int max_results = 32;
		int recover_attempts = 4;
		Vector2 sr[max_results * 2];
		PhysicsServer2DSW::CollCbkData cbk;
		cbk.max = max_results;
		PhysicsServer2DSW::CollCbkData *cbkptr = &cbk;
		CollisionSolver2DSW::CallbackResult cbkres = PhysicsServer2DSW::_shape_col_cbk;

		do {
			Vector2 recover_motion;

			bool collided = false;

			int amount = _cull_aabb_for_body(p_body, body_aabb);

			for (int j = 0; j < p_body->get_shape_count(); j++) {
				if (p_body->is_shape_set_as_disabled(j)) {
					continue;
				}

				Shape2DSW *body_shape = p_body->get_shape(j);

				if (body_shape->get_type() != PhysicsServer2D::SHAPE_RAY) {
					continue;
				}

				Transform2D body_shape_xform = body_transform * p_body->get_shape_transform(j);

				for (int i = 0; i < amount; i++) {
					const CollisionObject2DSW *col_obj = intersection_query_results[i];
					int shape_idx = intersection_query_subindex_results[i];

					cbk.amount = 0;
					cbk.passed = 0;
					cbk.ptr = sr;
					cbk.invalid_by_dir = 0;

					if (CollisionObject2DSW::TYPE_BODY == col_obj->get_type()) {
						const Body2DSW *b = static_cast<const Body2DSW *>(col_obj);
						if (p_infinite_inertia && PhysicsServer2D::BODY_MODE_STATIC != b->get_mode() && PhysicsServer2D::BODY_MODE_KINEMATIC != b->get_mode()) {
							continue;
						}
					}

					Transform2D col_obj_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);

					/*
 * There is no point in supporting one way collisions with ray shapes, as they will always collide in the desired
 * direction. Use a short ray shape if you want to achieve a similar effect.
 *
					if (col_obj->is_shape_set_as_one_way_collision(shape_idx)) {
						cbk.valid_dir = col_obj_shape_xform.get_axis(1).normalized();
						cbk.valid_depth = p_margin; //only valid depth is the collision margin
						cbk.invalid_by_dir = 0;

					} else {
*/

					cbk.valid_dir = Vector2();
					cbk.valid_depth = 0;
					cbk.invalid_by_dir = 0;

					/*
					}
					*/

					Shape2DSW *against_shape = col_obj->get_shape(shape_idx);
					if (CollisionSolver2DSW::solve(body_shape, body_shape_xform, Vector2(), against_shape, col_obj_shape_xform, Vector2(), cbkres, cbkptr, nullptr, p_margin)) {
						if (cbk.amount > 0) {
							collided = true;
						}

						int ray_index = -1; //reuse shape
						for (int k = 0; k < rays_found; k++) {
							if (r_results[ray_index].collision_local_shape == j) {
								ray_index = k;
							}
						}

						if (ray_index == -1 && rays_found < p_result_max) {
							ray_index = rays_found;
							rays_found++;
						}

						if (ray_index != -1) {
							PhysicsServer2D::SeparationResult &result = r_results[ray_index];

							for (int k = 0; k < cbk.amount; k++) {
								Vector2 a = sr[k * 2 + 0];
								Vector2 b = sr[k * 2 + 1];

								recover_motion += (b - a) / cbk.amount;

								real_t depth = a.distance_to(b);
								if (depth > result.collision_depth) {
									result.collision_depth = depth;
									result.collision_point = b;
									result.collision_normal = (b - a).normalized();
									result.collision_local_shape = j;
									result.collider_shape = shape_idx;
									result.collider = col_obj->get_self();
									result.collider_id = col_obj->get_instance_id();
									result.collider_metadata = col_obj->get_shape_metadata(shape_idx);
									if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {
										Body2DSW *body = (Body2DSW *)col_obj;

										Vector2 rel_vec = b - body->get_transform().get_origin();
										result.collider_velocity = Vector2(-body->get_angular_velocity() * rel_vec.y, body->get_angular_velocity() * rel_vec.x) + body->get_linear_velocity();
									}
								}
							}
						}
					}
				}
			}

			if (!collided || recover_motion == Vector2()) {
				break;
			}

			body_transform.elements[2] += recover_motion;
			body_aabb.position += recover_motion;

			recover_attempts--;
		} while (recover_attempts);
	}

	//optimize results (remove non colliding)
	for (int i = 0; i < rays_found; i++) {
		if (r_results[i].collision_depth == 0) {
			rays_found--;
			SWAP(r_results[i], r_results[rays_found]);
		}
	}

	r_recover_motion = body_transform.elements[2] - p_transform.elements[2];
	return rays_found;
}

bool Space2DSW::test_body_motion(Body2DSW *p_body, const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia, real_t p_margin, PhysicsServer2D::MotionResult *r_result, bool p_exclude_raycast_shapes) {
	//give me back regular physics engine logic
	//this is madness
	//and most people using this function will think
	//what it does is simpler than using physics
	//this took about a week to get right..
	//but is it right? who knows at this point..

	if (r_result) {
		r_result->collider_id = ObjectID();
		r_result->collider_shape = 0;
	}
	Rect2 body_aabb;

	bool shapes_found = false;

	for (int i = 0; i < p_body->get_shape_count(); i++) {
		if (p_body->is_shape_set_as_disabled(i)) {
			continue;
		}

		if (p_exclude_raycast_shapes && p_body->get_shape(i)->get_type() == PhysicsServer2D::SHAPE_RAY) {
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
			*r_result = PhysicsServer2D::MotionResult();
			r_result->motion = p_motion;
		}
		return false;
	}

	// Undo the currently transform the physics server is aware of and apply the provided one
	body_aabb = p_from.xform(p_body->get_inv_transform().xform(body_aabb));
	body_aabb = body_aabb.grow(p_margin);

	static const int max_excluded_shape_pairs = 32;
	ExcludedShapeSW excluded_shape_pairs[max_excluded_shape_pairs];
	int excluded_shape_pair_count = 0;

	real_t motion_length = p_motion.length();
	Vector2 motion_normal = p_motion / motion_length;

	Transform2D body_transform = p_from;

	bool recovered = false;

	{
		//STEP 1, FREE BODY IF STUCK

		const int max_results = 32;
		int recover_attempts = 4;
		Vector2 sr[max_results * 2];

		do {
			PhysicsServer2DSW::CollCbkData cbk;
			cbk.max = max_results;
			cbk.amount = 0;
			cbk.passed = 0;
			cbk.ptr = sr;
			cbk.invalid_by_dir = 0;
			excluded_shape_pair_count = 0; //last step is the one valid

			PhysicsServer2DSW::CollCbkData *cbkptr = &cbk;
			CollisionSolver2DSW::CallbackResult cbkres = PhysicsServer2DSW::_shape_col_cbk;

			bool collided = false;

			int amount = _cull_aabb_for_body(p_body, body_aabb);

			for (int j = 0; j < p_body->get_shape_count(); j++) {
				if (p_body->is_shape_set_as_disabled(j)) {
					continue;
				}

				Shape2DSW *body_shape = p_body->get_shape(j);
				if (p_exclude_raycast_shapes && body_shape->get_type() == PhysicsServer2D::SHAPE_RAY) {
					continue;
				}

				Transform2D body_shape_xform = body_transform * p_body->get_shape_transform(j);
				for (int i = 0; i < amount; i++) {
					const CollisionObject2DSW *col_obj = intersection_query_results[i];
					int shape_idx = intersection_query_subindex_results[i];

					if (CollisionObject2DSW::TYPE_BODY == col_obj->get_type()) {
						const Body2DSW *b = static_cast<const Body2DSW *>(col_obj);
						if (p_infinite_inertia && PhysicsServer2D::BODY_MODE_STATIC != b->get_mode() && PhysicsServer2D::BODY_MODE_KINEMATIC != b->get_mode()) {
							continue;
						}
					}

					Transform2D col_obj_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);

					if (col_obj->is_shape_set_as_one_way_collision(shape_idx)) {
						cbk.valid_dir = col_obj_shape_xform.get_axis(1).normalized();

						real_t owc_margin = col_obj->get_shape_one_way_collision_margin(shape_idx);
						cbk.valid_depth = MAX(owc_margin, p_margin); //user specified, but never less than actual margin or it won't work
						cbk.invalid_by_dir = 0;

						if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {
							const Body2DSW *b = static_cast<const Body2DSW *>(col_obj);
							if (b->get_mode() == PhysicsServer2D::BODY_MODE_KINEMATIC || b->get_mode() == PhysicsServer2D::BODY_MODE_RIGID) {
								//fix for moving platforms (kinematic and dynamic), margin is increased by how much it moved in the given direction
								Vector2 lv = b->get_linear_velocity();
								//compute displacement from linear velocity
								Vector2 motion = lv * PhysicsDirectBodyState2DSW::singleton->step;
								real_t motion_len = motion.length();
								motion.normalize();
								cbk.valid_depth += motion_len * MAX(motion.dot(-cbk.valid_dir), 0.0);
							}
						}
					} else {
						cbk.valid_dir = Vector2();
						cbk.valid_depth = 0;
						cbk.invalid_by_dir = 0;
					}

					int current_passed = cbk.passed; //save how many points passed collision
					bool did_collide = false;

					Shape2DSW *against_shape = col_obj->get_shape(shape_idx);
					if (CollisionSolver2DSW::solve(body_shape, body_shape_xform, Vector2(), against_shape, col_obj_shape_xform, Vector2(), cbkres, cbkptr, nullptr, p_margin)) {
						did_collide = cbk.passed > current_passed; //more passed, so collision actually existed
					}

					if (!did_collide && cbk.invalid_by_dir > 0) {
						//this shape must be excluded
						if (excluded_shape_pair_count < max_excluded_shape_pairs) {
							ExcludedShapeSW esp;
							esp.local_shape = body_shape;
							esp.against_object = col_obj;
							esp.against_shape_index = shape_idx;
							excluded_shape_pairs[excluded_shape_pair_count++] = esp;
						}
					}

					if (did_collide) {
						collided = true;
					}
				}
			}

			if (!collided) {
				break;
			}

			Vector2 recover_motion;
			for (int i = 0; i < cbk.amount; i++) {
				Vector2 a = sr[i * 2 + 0];
				Vector2 b = sr[i * 2 + 1];

				// Compute plane on b towards a.
				Vector2 n = (a - b).normalized();
				real_t d = n.dot(b);

				// Compute depth on recovered motion.
				real_t depth = n.dot(a + recover_motion) - d;
				if (depth > 0.0) {
					// Only recover if there is penetration.
					recover_motion -= n * depth * 0.4;
				}
			}

			if (recover_motion == Vector2()) {
				collided = false;
				break;
			}

			recovered = true;

			body_transform.elements[2] += recover_motion;
			body_aabb.position += recover_motion;

			recover_attempts--;

		} while (recover_attempts);
	}

	real_t safe = 1.0;
	real_t unsafe = 1.0;
	int best_shape = -1;

	{
		// STEP 2 ATTEMPT MOTION

		Rect2 motion_aabb = body_aabb;
		motion_aabb.position += p_motion;
		motion_aabb = motion_aabb.merge(body_aabb);

		int amount = _cull_aabb_for_body(p_body, motion_aabb);

		for (int body_shape_idx = 0; body_shape_idx < p_body->get_shape_count(); body_shape_idx++) {
			if (p_body->is_shape_set_as_disabled(body_shape_idx)) {
				continue;
			}

			Shape2DSW *body_shape = p_body->get_shape(body_shape_idx);
			if (p_exclude_raycast_shapes && body_shape->get_type() == PhysicsServer2D::SHAPE_RAY) {
				continue;
			}

			Transform2D body_shape_xform = body_transform * p_body->get_shape_transform(body_shape_idx);

			bool stuck = false;

			real_t best_safe = 1;
			real_t best_unsafe = 1;

			for (int i = 0; i < amount; i++) {
				const CollisionObject2DSW *col_obj = intersection_query_results[i];
				int col_shape_idx = intersection_query_subindex_results[i];
				Shape2DSW *against_shape = col_obj->get_shape(col_shape_idx);

				if (CollisionObject2DSW::TYPE_BODY == col_obj->get_type()) {
					const Body2DSW *b = static_cast<const Body2DSW *>(col_obj);
					if (p_infinite_inertia && PhysicsServer2D::BODY_MODE_STATIC != b->get_mode() && PhysicsServer2D::BODY_MODE_KINEMATIC != b->get_mode()) {
						continue;
					}
				}

				bool excluded = false;

				for (int k = 0; k < excluded_shape_pair_count; k++) {
					if (excluded_shape_pairs[k].local_shape == body_shape && excluded_shape_pairs[k].against_object == col_obj && excluded_shape_pairs[k].against_shape_index == col_shape_idx) {
						excluded = true;
						break;
					}
				}

				if (excluded) {
					continue;
				}

				Transform2D col_obj_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(col_shape_idx);
				//test initial overlap, does it collide if going all the way?
				if (!CollisionSolver2DSW::solve(body_shape, body_shape_xform, p_motion, against_shape, col_obj_shape_xform, Vector2(), nullptr, nullptr, nullptr, 0)) {
					continue;
				}

				//test initial overlap
				if (CollisionSolver2DSW::solve(body_shape, body_shape_xform, Vector2(), against_shape, col_obj_shape_xform, Vector2(), nullptr, nullptr, nullptr, 0)) {
					if (col_obj->is_shape_set_as_one_way_collision(col_shape_idx)) {
						Vector2 direction = col_obj_shape_xform.get_axis(1).normalized();
						if (motion_normal.dot(direction) < 0) {
							continue;
						}
					}

					stuck = true;
					break;
				}

				//just do kinematic solving
				real_t low = 0;
				real_t hi = 1;

				for (int k = 0; k < 8; k++) { //steps should be customizable..

					real_t ofs = (low + hi) * 0.5;

					Vector2 sep = motion_normal; //important optimization for this to work fast enough
					bool collided = CollisionSolver2DSW::solve(body_shape, body_shape_xform, p_motion * ofs, against_shape, col_obj_shape_xform, Vector2(), nullptr, nullptr, &sep, 0);

					if (collided) {
						hi = ofs;
					} else {
						low = ofs;
					}
				}

				if (col_obj->is_shape_set_as_one_way_collision(col_shape_idx)) {
					Vector2 cd[2];
					PhysicsServer2DSW::CollCbkData cbk;
					cbk.max = 1;
					cbk.amount = 0;
					cbk.passed = 0;
					cbk.ptr = cd;
					cbk.valid_dir = col_obj_shape_xform.get_axis(1).normalized();

					cbk.valid_depth = 10e20;

					Vector2 sep = motion_normal; //important optimization for this to work fast enough
					bool collided = CollisionSolver2DSW::solve(body_shape, body_shape_xform, p_motion * (hi + contact_max_allowed_penetration), col_obj->get_shape(col_shape_idx), col_obj_shape_xform, Vector2(), PhysicsServer2DSW::_shape_col_cbk, &cbk, &sep, 0);
					if (!collided || cbk.amount == 0) {
						continue;
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
				best_shape = body_shape_idx; //sadly it's the best
				break;
			}
			if (best_safe == 1.0) {
				continue;
			}
			if (best_safe < safe) {
				safe = best_safe;
				unsafe = best_unsafe;
				best_shape = body_shape_idx;
			}
		}
	}

	bool collided = false;

	if (recovered || (safe < 1)) {
		if (safe >= 1) {
			best_shape = -1; //no best shape with cast, reset to -1
		}

		//it collided, let's get the rest info in unsafe advance
		Transform2D ugt = body_transform;
		ugt.elements[2] += p_motion * unsafe;

		_RestCallbackData2D rcd;
		rcd.best_len = 0;
		rcd.best_object = nullptr;
		rcd.best_shape = 0;

		// Allowed depth can't be lower than motion length, in order to handle contacts at low speed.
		rcd.min_allowed_depth = MIN(motion_length, test_motion_min_contact_depth);

		int from_shape = best_shape != -1 ? best_shape : 0;
		int to_shape = best_shape != -1 ? best_shape + 1 : p_body->get_shape_count();

		for (int j = from_shape; j < to_shape; j++) {
			if (p_body->is_shape_set_as_disabled(j)) {
				continue;
			}

			Transform2D body_shape_xform = ugt * p_body->get_shape_transform(j);
			Shape2DSW *body_shape = p_body->get_shape(j);

			if (p_exclude_raycast_shapes && body_shape->get_type() == PhysicsServer2D::SHAPE_RAY) {
				continue;
			}

			body_aabb.position += p_motion * unsafe;

			int amount = _cull_aabb_for_body(p_body, body_aabb);

			for (int i = 0; i < amount; i++) {
				const CollisionObject2DSW *col_obj = intersection_query_results[i];
				int shape_idx = intersection_query_subindex_results[i];

				if (CollisionObject2DSW::TYPE_BODY == col_obj->get_type()) {
					const Body2DSW *b = static_cast<const Body2DSW *>(col_obj);
					if (p_infinite_inertia && PhysicsServer2D::BODY_MODE_STATIC != b->get_mode() && PhysicsServer2D::BODY_MODE_KINEMATIC != b->get_mode()) {
						continue;
					}
				}

				Shape2DSW *against_shape = col_obj->get_shape(shape_idx);

				bool excluded = false;
				for (int k = 0; k < excluded_shape_pair_count; k++) {
					if (excluded_shape_pairs[k].local_shape == body_shape && excluded_shape_pairs[k].against_object == col_obj && excluded_shape_pairs[k].against_shape_index == shape_idx) {
						excluded = true;
						break;
					}
				}
				if (excluded) {
					continue;
				}

				Transform2D col_obj_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);

				if (col_obj->is_shape_set_as_one_way_collision(shape_idx)) {
					rcd.valid_dir = col_obj_shape_xform.get_axis(1).normalized();

					real_t owc_margin = col_obj->get_shape_one_way_collision_margin(shape_idx);
					rcd.valid_depth = MAX(owc_margin, p_margin); //user specified, but never less than actual margin or it won't work

					if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {
						const Body2DSW *b = static_cast<const Body2DSW *>(col_obj);
						if (b->get_mode() == PhysicsServer2D::BODY_MODE_KINEMATIC || b->get_mode() == PhysicsServer2D::BODY_MODE_RIGID) {
							//fix for moving platforms (kinematic and dynamic), margin is increased by how much it moved in the given direction
							Vector2 lv = b->get_linear_velocity();
							//compute displacement from linear velocity
							Vector2 motion = lv * PhysicsDirectBodyState2DSW::singleton->step;
							real_t motion_len = motion.length();
							motion.normalize();
							rcd.valid_depth += motion_len * MAX(motion.dot(-rcd.valid_dir), 0.0);
						}
					}
				} else {
					rcd.valid_dir = Vector2();
					rcd.valid_depth = 0;
				}

				rcd.object = col_obj;
				rcd.shape = shape_idx;
				rcd.local_shape = j;
				bool sc = CollisionSolver2DSW::solve(body_shape, body_shape_xform, Vector2(), against_shape, col_obj_shape_xform, Vector2(), _rest_cbk_result, &rcd, nullptr, p_margin);
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
				r_result->collider_metadata = rcd.best_object->get_shape_metadata(rcd.best_shape);

				const Body2DSW *body = static_cast<const Body2DSW *>(rcd.best_object);
				Vector2 rel_vec = r_result->collision_point - body->get_transform().get_origin();
				r_result->collider_velocity = Vector2(-body->get_angular_velocity() * rel_vec.y, body->get_angular_velocity() * rel_vec.x) + body->get_linear_velocity();

				r_result->motion = safe * p_motion;
				r_result->remainder = p_motion - safe * p_motion;
				r_result->motion += (body_transform.get_origin() - p_from.get_origin());
			}

			collided = true;
		}
	}

	if (!collided && r_result) {
		r_result->motion = p_motion;
		r_result->remainder = Vector2();
		r_result->motion += (body_transform.get_origin() - p_from.get_origin());
	}

	return collided;
}

void *Space2DSW::_broadphase_pair(CollisionObject2DSW *A, int p_subindex_A, CollisionObject2DSW *B, int p_subindex_B, void *p_self) {
	if (!A->test_collision_mask(B)) {
		return nullptr;
	}

	CollisionObject2DSW::Type type_A = A->get_type();
	CollisionObject2DSW::Type type_B = B->get_type();
	if (type_A > type_B) {
		SWAP(A, B);
		SWAP(p_subindex_A, p_subindex_B);
		SWAP(type_A, type_B);
	}

	Space2DSW *self = (Space2DSW *)p_self;
	self->collision_pairs++;

	if (type_A == CollisionObject2DSW::TYPE_AREA) {
		Area2DSW *area = static_cast<Area2DSW *>(A);
		if (type_B == CollisionObject2DSW::TYPE_AREA) {
			Area2DSW *area_b = static_cast<Area2DSW *>(B);
			Area2Pair2DSW *area2_pair = memnew(Area2Pair2DSW(area_b, p_subindex_B, area, p_subindex_A));
			return area2_pair;
		} else {
			Body2DSW *body = static_cast<Body2DSW *>(B);
			AreaPair2DSW *area_pair = memnew(AreaPair2DSW(body, p_subindex_B, area, p_subindex_A));
			return area_pair;
		}

	} else {
		BodyPair2DSW *b = memnew(BodyPair2DSW((Body2DSW *)A, p_subindex_A, (Body2DSW *)B, p_subindex_B));
		return b;
	}

	return nullptr;
}

void Space2DSW::_broadphase_unpair(CollisionObject2DSW *A, int p_subindex_A, CollisionObject2DSW *B, int p_subindex_B, void *p_data, void *p_self) {
	if (!p_data) {
		return;
	}

	Space2DSW *self = (Space2DSW *)p_self;
	self->collision_pairs--;
	Constraint2DSW *c = (Constraint2DSW *)p_data;
	memdelete(c);
}

const SelfList<Body2DSW>::List &Space2DSW::get_active_body_list() const {
	return active_list;
}

void Space2DSW::body_add_to_active_list(SelfList<Body2DSW> *p_body) {
	active_list.add(p_body);
}

void Space2DSW::body_remove_from_active_list(SelfList<Body2DSW> *p_body) {
	active_list.remove(p_body);
}

void Space2DSW::body_add_to_inertia_update_list(SelfList<Body2DSW> *p_body) {
	inertia_update_list.add(p_body);
}

void Space2DSW::body_remove_from_inertia_update_list(SelfList<Body2DSW> *p_body) {
	inertia_update_list.remove(p_body);
}

BroadPhase2DSW *Space2DSW::get_broadphase() {
	return broadphase;
}

void Space2DSW::add_object(CollisionObject2DSW *p_object) {
	ERR_FAIL_COND(objects.has(p_object));
	objects.insert(p_object);
}

void Space2DSW::remove_object(CollisionObject2DSW *p_object) {
	ERR_FAIL_COND(!objects.has(p_object));
	objects.erase(p_object);
}

const Set<CollisionObject2DSW *> &Space2DSW::get_objects() const {
	return objects;
}

void Space2DSW::body_add_to_state_query_list(SelfList<Body2DSW> *p_body) {
	state_query_list.add(p_body);
}

void Space2DSW::body_remove_from_state_query_list(SelfList<Body2DSW> *p_body) {
	state_query_list.remove(p_body);
}

void Space2DSW::area_add_to_monitor_query_list(SelfList<Area2DSW> *p_area) {
	monitor_query_list.add(p_area);
}

void Space2DSW::area_remove_from_monitor_query_list(SelfList<Area2DSW> *p_area) {
	monitor_query_list.remove(p_area);
}

void Space2DSW::area_add_to_moved_list(SelfList<Area2DSW> *p_area) {
	area_moved_list.add(p_area);
}

void Space2DSW::area_remove_from_moved_list(SelfList<Area2DSW> *p_area) {
	area_moved_list.remove(p_area);
}

const SelfList<Area2DSW>::List &Space2DSW::get_moved_area_list() const {
	return area_moved_list;
}

void Space2DSW::call_queries() {
	while (state_query_list.first()) {
		Body2DSW *b = state_query_list.first()->self();
		state_query_list.remove(state_query_list.first());
		b->call_queries();
	}

	while (monitor_query_list.first()) {
		Area2DSW *a = monitor_query_list.first()->self();
		monitor_query_list.remove(monitor_query_list.first());
		a->call_queries();
	}
}

void Space2DSW::setup() {
	contact_debug_count = 0;

	while (inertia_update_list.first()) {
		inertia_update_list.first()->self()->update_inertias();
		inertia_update_list.remove(inertia_update_list.first());
	}
}

void Space2DSW::update() {
	broadphase->update();
}

void Space2DSW::set_param(PhysicsServer2D::SpaceParameter p_param, real_t p_value) {
	switch (p_param) {
		case PhysicsServer2D::SPACE_PARAM_CONTACT_RECYCLE_RADIUS:
			contact_recycle_radius = p_value;
			break;
		case PhysicsServer2D::SPACE_PARAM_CONTACT_MAX_SEPARATION:
			contact_max_separation = p_value;
			break;
		case PhysicsServer2D::SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION:
			contact_max_allowed_penetration = p_value;
			break;
		case PhysicsServer2D::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD:
			body_linear_velocity_sleep_threshold = p_value;
			break;
		case PhysicsServer2D::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD:
			body_angular_velocity_sleep_threshold = p_value;
			break;
		case PhysicsServer2D::SPACE_PARAM_BODY_TIME_TO_SLEEP:
			body_time_to_sleep = p_value;
			break;
		case PhysicsServer2D::SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS:
			constraint_bias = p_value;
			break;
		case PhysicsServer2D::SPACE_PARAM_TEST_MOTION_MIN_CONTACT_DEPTH:
			test_motion_min_contact_depth = p_value;
			break;
	}
}

real_t Space2DSW::get_param(PhysicsServer2D::SpaceParameter p_param) const {
	switch (p_param) {
		case PhysicsServer2D::SPACE_PARAM_CONTACT_RECYCLE_RADIUS:
			return contact_recycle_radius;
		case PhysicsServer2D::SPACE_PARAM_CONTACT_MAX_SEPARATION:
			return contact_max_separation;
		case PhysicsServer2D::SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION:
			return contact_max_allowed_penetration;
		case PhysicsServer2D::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD:
			return body_linear_velocity_sleep_threshold;
		case PhysicsServer2D::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD:
			return body_angular_velocity_sleep_threshold;
		case PhysicsServer2D::SPACE_PARAM_BODY_TIME_TO_SLEEP:
			return body_time_to_sleep;
		case PhysicsServer2D::SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS:
			return constraint_bias;
		case PhysicsServer2D::SPACE_PARAM_TEST_MOTION_MIN_CONTACT_DEPTH:
			return test_motion_min_contact_depth;
	}
	return 0;
}

void Space2DSW::lock() {
	locked = true;
}

void Space2DSW::unlock() {
	locked = false;
}

bool Space2DSW::is_locked() const {
	return locked;
}

PhysicsDirectSpaceState2DSW *Space2DSW::get_direct_state() {
	return direct_access;
}

Space2DSW::Space2DSW() {
	collision_pairs = 0;
	active_objects = 0;
	island_count = 0;

	contact_debug_count = 0;

	locked = false;
	contact_recycle_radius = 1.0;
	contact_max_separation = 1.5;
	contact_max_allowed_penetration = 0.3;
	test_motion_min_contact_depth = 0.005;

	constraint_bias = 0.2;
	body_linear_velocity_sleep_threshold = GLOBAL_DEF("physics/2d/sleep_threshold_linear", 2.0);
	body_angular_velocity_sleep_threshold = GLOBAL_DEF("physics/2d/sleep_threshold_angular", Math::deg2rad(8.0));
	body_time_to_sleep = GLOBAL_DEF("physics/2d/time_before_sleep", 0.5);
	ProjectSettings::get_singleton()->set_custom_property_info("physics/2d/time_before_sleep", PropertyInfo(Variant::FLOAT, "physics/2d/time_before_sleep", PROPERTY_HINT_RANGE, "0,5,0.01,or_greater"));

	broadphase = BroadPhase2DSW::create_func();
	broadphase->set_pair_callback(_broadphase_pair, this);
	broadphase->set_unpair_callback(_broadphase_unpair, this);
	area = nullptr;

	direct_access = memnew(PhysicsDirectSpaceState2DSW);
	direct_access->space = this;

	for (int i = 0; i < ELAPSED_TIME_MAX; i++) {
		elapsed_time[i] = 0;
	}
}

Space2DSW::~Space2DSW() {
	memdelete(broadphase);
	memdelete(direct_access);
}
