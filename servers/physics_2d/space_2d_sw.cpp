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
#include "core/pair.h"
#include "physics_2d_server_sw.h"

_FORCE_INLINE_ static bool _can_collide_with(const CollisionObject2DSW *p_object, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {

	if (!(p_object->get_collision_layer() & p_collision_mask)) {
		return false;
	}

	if (p_object->get_type() == CollisionObject2DSW::TYPE_AREA && !p_collide_with_areas)
		return false;

	if (p_object->get_type() == CollisionObject2DSW::TYPE_BODY && !p_collide_with_bodies)
		return false;

	return true;
}

int Physics2DDirectSpaceStateSW::_intersect_point_impl(const Vector2 &p_point, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_pick_point, bool p_filter_by_canvas, ObjectID p_canvas_instance_id) {

	ERR_FAIL_COND_V_MSG(space->locked, 0, "Space is locked");

	Rect2 aabb;
	aabb.position = p_point - Vector2(0.00001, 0.00001);
	aabb.size = Vector2(0.00002, 0.00002);

	int pair_count = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	int col_count = 0;

	for (int pair_idx = 0; pair_idx < pair_count; pair_idx++) {

		if (col_count >= p_result_max)
			break;

		const CollisionObject2DSW *col_obj = space->intersection_query_results[pair_idx];

		// Check pickable
		if (p_pick_point && !col_obj->is_pickable())
			continue;

		// Check canvas
		if (p_filter_by_canvas && col_obj->get_canvas_instance_id() != p_canvas_instance_id)
			continue;

		// Check exclusions
		if (p_exclude.has(col_obj->get_self()))
			continue;

		// Check collision filters
		if (!_can_collide_with(col_obj, p_collision_mask, p_collide_with_bodies, p_collide_with_areas))
			continue;

		int col_shape_idx = space->intersection_query_subindex_results[pair_idx];
		const Shape2DSW *col_shape = col_obj->get_shape(col_shape_idx);
		const Transform2D col_shape_inv_xform = col_obj->get_shape_inv_transform(col_shape_idx) * col_obj->get_inv_transform();
		Vector2 local_point = col_shape_inv_xform.xform(p_point);

		if (!col_shape->contains_point(local_point))
			continue;

		r_results[col_count].rid = col_obj->get_self();
		r_results[col_count].collider_id = col_obj->get_instance_id();
		if (r_results[col_count].collider_id != 0)
			r_results[col_count].collider = ObjectDB::get_instance(r_results[col_count].collider_id);
		r_results[col_count].shape = col_shape_idx;
		r_results[col_count].metadata = col_obj->get_shape_metadata(col_shape_idx);

		col_count++;
	}

	return col_count;
}

int Physics2DDirectSpaceStateSW::intersect_point(const Vector2 &p_point, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_pick_point) {

	return _intersect_point_impl(p_point, r_results, p_result_max, p_exclude, p_collision_mask, p_collide_with_bodies, p_collide_with_areas, p_pick_point);
}

int Physics2DDirectSpaceStateSW::intersect_point_on_canvas(const Vector2 &p_point, ObjectID p_canvas_instance_id, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_pick_point) {

	return _intersect_point_impl(p_point, r_results, p_result_max, p_exclude, p_collision_mask, p_collide_with_bodies, p_collide_with_areas, p_pick_point, true, p_canvas_instance_id);
}

bool Physics2DDirectSpaceStateSW::intersect_ray(const Vector2 &p_from, const Vector2 &p_to, RayResult &r_result, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {

	ERR_FAIL_COND_V_MSG(space->locked, false, "Space is locked");

	Vector2 normal = (p_to - p_from).normalized();

	int pair_count = space->broadphase->cull_segment(p_from, p_to, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	//TODO Create another array that references results, compute AABBs and check closest point to ray origin, sort, and stop evaluating results when beyond first collision

	bool collided = false;
	real_t min_distance = 10e20;

	for (int pair_idx = 0; pair_idx < pair_count; pair_idx++) {

		const CollisionObject2DSW *col_obj = space->intersection_query_results[pair_idx];

		// Check exclusions
		if (p_exclude.has(col_obj->get_self()))
			continue;

		// Check collision filters
		if (!_can_collide_with(col_obj, p_collision_mask, p_collide_with_bodies, p_collide_with_areas))
			continue;

		int col_shape_idx = space->intersection_query_subindex_results[pair_idx];
		const Shape2DSW *col_shape = col_obj->get_shape(col_shape_idx);
		const Transform2D col_shape_inv_xform = col_obj->get_shape_inv_transform(col_shape_idx) * col_obj->get_inv_transform();

		Vector2 local_from = col_shape_inv_xform.xform(p_from);
		Vector2 local_to = col_shape_inv_xform.xform(p_to);
		Vector2 shape_point, shape_normal;

		if (!col_shape->intersect_segment(local_from, local_to, shape_point, shape_normal))
			continue;

		const Transform2D col_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(col_shape_idx);
		shape_point = col_shape_xform.xform(shape_point);

		real_t distance = normal.dot(shape_point);

		if (distance < min_distance) {

			collided = true;
			min_distance = distance;

			r_result.position = shape_point;
			r_result.normal = col_shape_inv_xform.basis_xform_inv(shape_normal).normalized();
			r_result.rid = col_obj->get_self();
			r_result.collider_id = col_obj->get_instance_id();
			if (r_result.collider_id != 0)
				r_result.collider = ObjectDB::get_instance(r_result.collider_id);
			r_result.shape = col_shape_idx;
			r_result.metadata = col_obj->get_shape_metadata(col_shape_idx);
		}
	}

	return collided;
}

int Physics2DDirectSpaceStateSW::intersect_shape(const RID &p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, real_t p_margin, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {

	ERR_FAIL_COND_V_MSG(space->locked, 0, "Space is locked");

	const Shape2DSW *shape = Physics2DServerSW::singletonsw->shape_owner.get(p_shape);
	ERR_FAIL_COND_V_MSG(!shape, 0, "RID is not a shape");

	Rect2 aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.grow(p_margin);

	int pair_count = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	int col_count = 0;

	for (int pair_idx = 0; pair_idx < pair_count; pair_idx++) {

		if (col_count >= p_result_max)
			break;

		const CollisionObject2DSW *col_obj = space->intersection_query_results[pair_idx];

		// Check exclusions
		if (p_exclude.has(col_obj->get_self()))
			continue;

		// Check collision filters
		if (!_can_collide_with(col_obj, p_collision_mask, p_collide_with_bodies, p_collide_with_areas))
			continue;

		int col_shape_idx = space->intersection_query_subindex_results[pair_idx];
		const Shape2DSW *col_shape = col_obj->get_shape(col_shape_idx);
		const Transform2D col_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(col_shape_idx);

		if (!CollisionSolver2DSW::solve(shape, p_shape_xform, p_motion, col_shape, col_shape_xform, Vector2(), nullptr, nullptr, nullptr, p_margin))
			continue;

		r_results[col_count].rid = col_obj->get_self();
		r_results[col_count].collider_id = col_obj->get_instance_id();
		if (r_results[col_count].collider_id != 0)
			r_results[col_count].collider = ObjectDB::get_instance(r_results[col_count].collider_id);
		r_results[col_count].shape = col_shape_idx;
		r_results[col_count].metadata = col_obj->get_shape_metadata(col_shape_idx);

		col_count++;
	}

	return col_count;
}

bool Physics2DDirectSpaceStateSW::cast_motion(const RID &p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, real_t p_margin, real_t &r_best_safe, real_t &r_best_unsafe, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {

	r_best_safe = 0;
	r_best_unsafe = 0;

	ERR_FAIL_COND_V_MSG(space->locked, false, "Space is locked");

	if (p_motion == Vector2()) {
		return false;
	}

	const Shape2DSW *shape = Physics2DServerSW::singletonsw->shape_owner.get(p_shape);
	ERR_FAIL_COND_V_MSG(!shape, false, "RID is not a shape");

	Rect2 aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.merge(Rect2(aabb.position + p_motion, aabb.size));
	aabb = aabb.grow(p_margin);
	Vector2 motion_normal = p_motion.normalized();

	int pair_count = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	for (int pair_idx = 0; pair_idx < pair_count; pair_idx++) {

		const CollisionObject2DSW *col_obj = space->intersection_query_results[pair_idx];

		// Check exclusions
		if (p_exclude.has(col_obj->get_self()))
			continue;

		// Check collision filters
		if (!_can_collide_with(col_obj, p_collision_mask, p_collide_with_bodies, p_collide_with_areas))
			continue;

		int col_shape_idx = space->intersection_query_subindex_results[pair_idx];
		const Shape2DSW *col_shape = col_obj->get_shape(col_shape_idx);
		const Transform2D col_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(col_shape_idx);

		// Does it collide if going all the way?
		Vector2 sep_axis = motion_normal;
		if (!CollisionSolver2DSW::solve(shape, p_shape_xform, p_motion, col_shape, col_shape_xform, Vector2(), nullptr, nullptr, &sep_axis, p_margin)) {
			continue;
		}

		// Ignore objects it's inside of.
		sep_axis = motion_normal;
		if (CollisionSolver2DSW::solve(shape, p_shape_xform, Vector2(), col_shape, col_shape_xform, Vector2(), nullptr, nullptr, &sep_axis, p_margin)) {
			continue;
		}

		// Do kinematic solving
		real_t low = 0.0;
		real_t hi = 1.0;

		for (int step = 0; step < 8; step++) { // Steps should be customizable.

			real_t ofs = (low + hi) * 0.5;

			sep_axis = motion_normal;
			if (CollisionSolver2DSW::solve(shape, p_shape_xform, p_motion * ofs, col_obj->get_shape(col_shape_idx), col_shape_xform, Vector2(), NULL, NULL, &sep_axis, p_margin)) {
				hi = ofs;
			} else {
				low = ofs;
			}
		}

		if (low < r_best_safe) {
			r_best_safe = low;
			r_best_unsafe = hi;
		}
	}

	return true;
}

bool Physics2DDirectSpaceStateSW::collide_shape(RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, real_t p_margin, Vector2 *r_results, int p_result_max, int &r_result_count, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {

	ERR_FAIL_COND_V_MSG(space->locked, false, "Space is locked");

	const Shape2DSW *shape = Physics2DServerSW::singletonsw->shape_owner.get(p_shape);
	ERR_FAIL_COND_V_MSG(!shape, false, "RID is not a shape");

	Rect2 aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.merge(Rect2(aabb.position + p_motion, aabb.size));
	aabb = aabb.grow(p_margin);

	int pair_count = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	r_result_count = 0;

	CollisionSolver2DSW::CallbackResult col_cbk = Physics2DServerSW::_shape_col_cbk;
	Physics2DServerSW::CollCbkData col_cbk_data;
	col_cbk_data.max = p_result_max;
	col_cbk_data.amount = 0;
	col_cbk_data.ptr = r_results;

	for (int pair_idx = 0; pair_idx < pair_count; pair_idx++) {

		const CollisionObject2DSW *col_obj = space->intersection_query_results[pair_idx];

		// Check exclusions
		if (p_exclude.has(col_obj->get_self()))
			continue;

		// Check collision filters
		if (!_can_collide_with(col_obj, p_collision_mask, p_collide_with_bodies, p_collide_with_areas))
			continue;

		int col_shape_idx = space->intersection_query_subindex_results[pair_idx];
		const Shape2DSW *col_shape = col_obj->get_shape(col_shape_idx);
		const Transform2D col_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(col_shape_idx);

		CollisionSolver2DSW::solve(shape, p_shape_xform, p_motion, col_shape, col_shape_xform, Vector2(), col_cbk, &col_cbk_data, nullptr, p_margin);
	}

	r_result_count = col_cbk_data.amount;

	return r_result_count > 0;
}

struct _RestCallbackData2D {

	const CollisionObject2DSW *object = nullptr;
	const CollisionObject2DSW *best_object = nullptr;
	int shape = 0;
	int best_shape = 0;
	Vector2 best_contact;
	Vector2 best_normal;
	real_t best_len = 0;
	Vector2 valid_dir;
};

static void _rest_cbk_result(const Vector2 &p_point_A, const Vector2 &p_point_B, void *p_userdata) {

	_RestCallbackData2D *rd = (_RestCallbackData2D *)p_userdata;

	Vector2 contact_rel = p_point_B - p_point_A;
	real_t len = contact_rel.length();

	if (len <= rd->best_len) {
		return;
	}

	Vector2 normal = contact_rel / len;

	if (rd->valid_dir != Vector2() && rd->valid_dir.dot(normal) > -CMP_EPSILON) {
		return;
	}

	rd->best_len = len;
	rd->best_contact = p_point_B;
	rd->best_normal = normal;
	rd->best_object = rd->object;
	rd->best_shape = rd->shape;
}

bool Physics2DDirectSpaceStateSW::rest_info(RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, real_t p_margin, ShapeRestInfo *r_info, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {

	ERR_FAIL_COND_V_MSG(space->locked, false, "Space is locked");

	const Shape2DSW *shape = Physics2DServerSW::singletonsw->shape_owner.get(p_shape);
	ERR_FAIL_COND_V_MSG(!shape, false, "RID is not a shape");

	Rect2 aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.merge(Rect2(aabb.position + p_motion, aabb.size)); //motion
	aabb = aabb.grow(p_margin);

	int pair_count = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	_RestCallbackData2D rcd;

	for (int pair_idx = 0; pair_idx < pair_count; pair_idx++) {

		const CollisionObject2DSW *col_obj = space->intersection_query_results[pair_idx];

		// Check exclusions
		if (p_exclude.has(col_obj->get_self()))
			continue;

		// Check collision filters
		if (!_can_collide_with(col_obj, p_collision_mask, p_collide_with_bodies, p_collide_with_areas))
			continue;

		int col_shape_idx = space->intersection_query_subindex_results[pair_idx];
		const Shape2DSW *col_shape = col_obj->get_shape(col_shape_idx);
		const Transform2D col_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(col_shape_idx);

		rcd.object = col_obj;
		rcd.shape = col_shape_idx;

		CollisionSolver2DSW::solve(shape, p_shape_xform, p_motion, col_shape, col_shape_xform, Vector2(), _rest_cbk_result, &rcd, nullptr, p_margin);
	}

	if (rcd.best_len == 0 || !rcd.best_object)
		return false;

	r_info->point = rcd.best_contact;
	r_info->normal = rcd.best_normal;
	r_info->rid = rcd.best_object->get_self();
	r_info->collider_id = rcd.best_object->get_instance_id();
	r_info->shape = rcd.best_shape;
	if (rcd.best_object->get_type() == CollisionObject2DSW::TYPE_BODY) {
		const Body2DSW *body = static_cast<const Body2DSW *>(rcd.best_object);
		Vector2 rel_vec = r_info->point - body->get_transform().get_origin();
		r_info->linear_velocity = Vector2(-body->get_angular_velocity() * rel_vec.y, body->get_angular_velocity() * rel_vec.x) + body->get_linear_velocity();
	}
	r_info->metadata = rcd.best_object->get_shape_metadata(rcd.best_shape);

	return true;
}

Physics2DDirectSpaceStateSW::Physics2DDirectSpaceStateSW() {

	space = nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

int Space2DSW::_cull_aabb_for_body(Body2DSW *p_body, const Rect2 &p_aabb) {

	int pair_count = broadphase->cull_aabb(p_aabb, intersection_query_results, INTERSECTION_QUERY_MAX, intersection_query_subindex_results);

	for (int pair_idx = 0; pair_idx < pair_count; pair_idx++) {
		bool keep = true;

		const CollisionObject2DSW *col_obj = intersection_query_results[pair_idx];
		if (col_obj == p_body)
			keep = false;
		else if (col_obj->get_type() == CollisionObject2DSW::TYPE_AREA)
			keep = false;
		else if (col_obj->is_shape_set_as_disabled(intersection_query_subindex_results[pair_idx]))
			keep = false;
		else if (!col_obj->test_collision_mask(p_body))
			keep = false;
		else if (static_cast<const Body2DSW *>(col_obj)->has_exception(p_body->get_self()) || p_body->has_exception(col_obj->get_self()))
			keep = false;

		if (!keep) {
			if (pair_idx < pair_count - 1) {
				SWAP(intersection_query_results[pair_idx], intersection_query_results[pair_count - 1]);
				SWAP(intersection_query_subindex_results[pair_idx], intersection_query_subindex_results[pair_count - 1]);
			}
			pair_count--;
			pair_idx--;
		}
	}

	return pair_count;
}

int Space2DSW::test_body_ray_separation(Body2DSW *p_body, const Transform2D &p_xform, bool p_infinite_inertia, Vector2 &r_recover_motion, Physics2DServer::SeparationResult *r_results, int p_result_max, real_t p_margin) {

	Rect2 body_aabb;
	bool ray_shapes_found = false;

	for (int body_shape_idx = 0; body_shape_idx < p_body->get_shape_count(); body_shape_idx++) {

		if (p_body->is_shape_set_as_disabled(body_shape_idx))
			continue;

		// Ignore all shapes except ray shapes
		if (p_body->get_shape(body_shape_idx)->get_type() != Physics2DServer::SHAPE_RAY)
			continue;

		if (!ray_shapes_found) {
			body_aabb = p_body->get_shape_aabb(body_shape_idx);
			ray_shapes_found = true;
		} else {
			body_aabb = body_aabb.merge(p_body->get_shape_aabb(body_shape_idx));
		}
	}

	if (!ray_shapes_found) {
		return 0;
	}

	// Undo the currently transform the physics server is aware of and apply the provided one
	body_aabb = p_xform.xform(p_body->get_inv_transform().xform(body_aabb));
	body_aabb = body_aabb.grow(p_margin);

	Transform2D body_transform = p_xform;

	// Initialise collision depths
	for (int i = 0; i < p_result_max; i++) {
		r_results[i].collision_depth = 0;
	}

	int rays_found = 0;

	// Raycast and separate
	CollisionSolver2DSW::CallbackResult col_cbk = Physics2DServerSW::_shape_col_cbk;
	Physics2DServerSW::CollCbkData col_cbk_data;
	Vector2 col_points[2];
	col_cbk_data.ptr = col_points;
	col_cbk_data.max = 1;

	for (int recover_attempt = 0; recover_attempt < 4; recover_attempt++) {

		Vector2 recover_motion;

		int pair_count = _cull_aabb_for_body(p_body, body_aabb);

		for (int body_shape_idx = 0; body_shape_idx < p_body->get_shape_count(); body_shape_idx++) {

			if (p_body->is_shape_set_as_disabled(body_shape_idx))
				continue;

			const Shape2DSW *body_shape = p_body->get_shape(body_shape_idx);

			// Ignore all shapes except ray shapes
			if (body_shape->get_type() != Physics2DServer::SHAPE_RAY)
				continue;

			const Transform2D body_shape_xform = body_transform * p_body->get_shape_transform(body_shape_idx);

			int ray_idx = rays_found;
			// If ray already has a result, reuse it.
			for (int result_idx = 0; result_idx < rays_found; result_idx++) {
				if (r_results[result_idx].collision_local_shape == body_shape_idx) {
					ray_idx = result_idx;
				}
			}

			for (int pair_idx = 0; pair_idx < pair_count; pair_idx++) {

				// Reset counter
				col_cbk_data.amount = 0;

				const CollisionObject2DSW *col_obj = intersection_query_results[pair_idx];

				// Ignore RigidBodies if infinite inertia is enabled
				if (CollisionObject2DSW::TYPE_BODY == col_obj->get_type()) {
					const Body2DSW *b = static_cast<const Body2DSW *>(col_obj);
					if (p_infinite_inertia && b->get_mode() != Physics2DServer::BODY_MODE_STATIC && b->get_mode() != Physics2DServer::BODY_MODE_KINEMATIC) {
						continue;
					}
				}

				int col_shape_idx = intersection_query_subindex_results[pair_idx];
				const Shape2DSW *col_shape = col_obj->get_shape(col_shape_idx);
				const Transform2D col_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(col_shape_idx);

				/*
				 * There is no point in supporting one way collisions with ray shapes,
				 * as they will always collide in the desired direction.
				 * Use a short ray shape if you want to achieve a similar effect.
				 *
				if (col_obj->is_shape_set_as_one_way_collision(shape_idx)) {

					col_cbk_data.valid_dir = col_obj_shape_xform.get_axis(1).normalized();
					col_cbk_data.valid_depth = p_margin; //only valid depth is the collision margin
					col_cbk_data.invalid_by_dir = 0;
				}
				*/

				if (CollisionSolver2DSW::solve(body_shape, body_shape_xform, Vector2(), col_shape, col_shape_xform, Vector2(), col_cbk, &col_cbk_data, nullptr, p_margin)) {

					Vector2 a = col_points[0];
					Vector2 b = col_points[1];
					recover_motion += (b - a) / col_cbk_data.amount;

					// Check whether this is the ray's first result.
					if (ray_idx == rays_found) {
						// Check whether or not the max results has been reached.
						if (rays_found < p_result_max) {
							rays_found++;
						} else {
							continue;
						}
					}

					Physics2DServer::SeparationResult &result = r_results[ray_idx];
					float depth = a.distance_to(b);

					if (depth > result.collision_depth) {

						result.collision_depth = depth;
						result.collision_point = b;
						result.collision_normal = (b - a).normalized();
						if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {
							Body2DSW *body = (Body2DSW *)col_obj;
							Vector2 rel_vec = b - body->get_transform().get_origin();
							result.collider_velocity = Vector2(-body->get_angular_velocity() * rel_vec.y, body->get_angular_velocity() * rel_vec.x) + body->get_linear_velocity();
						}
						result.collision_local_shape = body_shape_idx;
						result.collider_id = col_obj->get_instance_id();
						result.collider = col_obj->get_self();
						result.collider_shape = col_shape_idx;
						result.collider_metadata = col_obj->get_shape_metadata(col_shape_idx);
					}
				}
			}
		}

		if (recover_motion == Vector2()) {
			break;
		}

		body_transform.elements[2] += recover_motion;
		body_aabb.position += recover_motion;
	}

	r_recover_motion = body_transform.elements[2] - p_xform.elements[2];
	return rays_found;
}

bool Space2DSW::test_body_motion(Body2DSW *p_body, const Transform2D &p_xform, const Vector2 &p_motion, bool p_infinite_inertia, real_t p_margin, Physics2DServer::MotionResult *r_result, bool p_exclude_ray_shapes) {
	if (r_result) {
		r_result->motion = p_motion;
		r_result->remainder = Vector2();
	}

	Rect2 body_aabb;
	bool shapes_found = false;

	for (int body_shape_idx = 0; body_shape_idx < p_body->get_shape_count(); body_shape_idx++) {

		if (p_body->is_shape_set_as_disabled(body_shape_idx))
			continue;

		// Ignore ray shapes
		if (p_exclude_ray_shapes && p_body->get_shape(body_shape_idx)->get_type() == Physics2DServer::SHAPE_RAY) {
			continue;
		}

		if (!shapes_found) {
			body_aabb = p_body->get_shape_aabb(body_shape_idx);
			shapes_found = true;
		} else {
			body_aabb = body_aabb.merge(p_body->get_shape_aabb(body_shape_idx));
		}
	}

	if (!shapes_found) {
		return false;
	}

	// Undo the currently transform the physics server is aware of and apply the provided one
	body_aabb = p_xform.xform(p_body->get_inv_transform().xform(body_aabb));
	body_aabb = body_aabb.grow(p_margin);

	static const int max_excluded_shape_pairs = 32;
	ExcludedShapePairSW excluded_shape_pairs[max_excluded_shape_pairs];
	int excluded_shape_pair_count = 0;

	// Step 1: Try to free the body, if it is stuck

	Transform2D body_transform = p_xform;

	const int max_results = 32;
	Vector2 col_points[max_results * 2];

	CollisionSolver2DSW::CallbackResult col_cbk = Physics2DServerSW::_shape_col_cbk;
	Physics2DServerSW::CollCbkData col_cbk_data;
	col_cbk_data.max = max_results;
	col_cbk_data.ptr = col_points;

	// Don't separate by more than the intended motion
	float separation_margin = MIN(p_margin, MAX(0.0, p_motion.length() - CMP_EPSILON));

	for (int recover_attempt = 0; recover_attempt < 4; recover_attempt++) {

		// Reset counters
		col_cbk_data.amount = 0;
		excluded_shape_pair_count = 0; // Last step is the valid one

		int pair_count = _cull_aabb_for_body(p_body, body_aabb);

		for (int body_shape_idx = 0; body_shape_idx < p_body->get_shape_count(); body_shape_idx++) {

			if (p_body->is_shape_set_as_disabled(body_shape_idx))
				continue;

			const Shape2DSW *body_shape = p_body->get_shape(body_shape_idx);

			// Ignore ray shapes
			if (p_exclude_ray_shapes && body_shape->get_type() == Physics2DServer::SHAPE_RAY) {
				continue;
			}

			const Transform2D body_shape_xform = body_transform * p_body->get_shape_transform(body_shape_idx);

			for (int pair_idx = 0; pair_idx < pair_count; pair_idx++) {

				const CollisionObject2DSW *col_obj = intersection_query_results[pair_idx];

				// Ignore RigidBodies if infinite inertia is enabled
				if (CollisionObject2DSW::TYPE_BODY == col_obj->get_type()) {
					const Body2DSW *b = static_cast<const Body2DSW *>(col_obj);
					if (p_infinite_inertia && Physics2DServer::BODY_MODE_STATIC != b->get_mode() && Physics2DServer::BODY_MODE_KINEMATIC != b->get_mode()) {
						continue;
					}
				}

				int col_shape_idx = intersection_query_subindex_results[pair_idx];
				const Shape2DSW *col_shape = col_obj->get_shape(col_shape_idx);
				const Transform2D col_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(col_shape_idx);

				if (col_obj->is_shape_set_as_one_way_collision(col_shape_idx)) {

					col_cbk_data.valid_dir = col_shape_xform.get_axis(1).normalized();

					float owc_margin = col_obj->get_shape_one_way_collision_margin(col_shape_idx);
					col_cbk_data.valid_depth = MAX(owc_margin, p_margin); // User specified, but never less than actual margin or it won't work

					if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {
						const Body2DSW *b = static_cast<const Body2DSW *>(col_obj);
						if (b->get_mode() == Physics2DServer::BODY_MODE_KINEMATIC || b->get_mode() == Physics2DServer::BODY_MODE_RIGID) {
							// Fix for moving platforms (kinematic and dynamic), margin is increased by how much it moved in the given direction
							Vector2 lv = b->get_linear_velocity();
							// Compute displacement from linear velocity
							Vector2 motion = lv * Physics2DDirectBodyStateSW::singleton->step;
							float motion_len = motion.length();
							motion.normalize();
							col_cbk_data.valid_depth += motion_len * MAX(motion.dot(-col_cbk_data.valid_dir), 0.0);
						}
					}
				} else {
					col_cbk_data.valid_dir = Vector2();
					col_cbk_data.valid_depth = 0;
				}
				col_cbk_data.invalid_by_dir = false;

				if (CollisionSolver2DSW::solve(body_shape, body_shape_xform, Vector2(), col_shape, col_shape_xform, Vector2(), col_cbk, &col_cbk_data, NULL, separation_margin)) {
					if (col_cbk_data.invalid_by_dir) {
						// Add shape pair to excluded shape pairs.
						if (excluded_shape_pair_count < max_excluded_shape_pairs) {
							ExcludedShapePairSW esp;
							esp.local_shape = body_shape;
							esp.other_object = col_obj;
							esp.other_shape_index = col_shape_idx;
							excluded_shape_pairs[excluded_shape_pair_count++] = esp;
						}
					}
				}
			}
		}

		Vector2 recover_motion;

		for (int i = 0; i < col_cbk_data.amount; i++) {

			Vector2 a = col_points[i * 2 + 0];
			Vector2 b = col_points[i * 2 + 1];
			recover_motion += (b - a) / col_cbk_data.amount;
		}

		if (recover_motion == Vector2()) {
			break;
		}

		body_transform.elements[2] += recover_motion;
		body_aabb.position += recover_motion;
	}

	if (r_result) {
		r_result->motion += (body_transform.get_origin() - p_xform.get_origin());
	}

	// Step 2: Determine maximum possible motion without collision

	real_t safe = 1.0;
	real_t unsafe = 1.0;
	int best_shape = -1;

	Rect2 motion_aabb = body_aabb;
	motion_aabb.position += p_motion;
	motion_aabb = motion_aabb.merge(body_aabb);
	Vector2 motion_normal = p_motion.normalized();

	int pair_count = _cull_aabb_for_body(p_body, motion_aabb);

	for (int body_shape_idx = 0; body_shape_idx < p_body->get_shape_count(); body_shape_idx++) {

		if (p_body->is_shape_set_as_disabled(body_shape_idx))
			continue;

		const Shape2DSW *body_shape = p_body->get_shape(body_shape_idx);
		if (p_exclude_ray_shapes && body_shape->get_type() == Physics2DServer::SHAPE_RAY) {
			continue;
		}

		const Transform2D body_shape_xform = body_transform * p_body->get_shape_transform(body_shape_idx);

		bool stuck = false;

		real_t best_safe = 1.0;
		real_t best_unsafe = 1.0;

		for (int pair_idx = 0; pair_idx < pair_count; pair_idx++) {

			const CollisionObject2DSW *col_obj = intersection_query_results[pair_idx];

			// Ignore RigidBodies if infinite inertia is enabled
			if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {
				const Body2DSW *body = static_cast<const Body2DSW *>(col_obj);
				if (p_infinite_inertia && body->get_mode() != Physics2DServer::BODY_MODE_STATIC && body->get_mode() != Physics2DServer::BODY_MODE_KINEMATIC) {
					continue;
				}
			}

			int col_shape_idx = intersection_query_subindex_results[pair_idx];

			// Ignore previously excluded shape pairs.
			bool excluded = false;
			for (int excluded_pair = 0; excluded_pair < excluded_shape_pair_count; excluded_pair++) {
				if (excluded_shape_pairs[excluded_pair].local_shape == body_shape && excluded_shape_pairs[excluded_pair].other_object == col_obj && excluded_shape_pairs[excluded_pair].other_shape_index == col_shape_idx) {
					excluded = true;
					break;
				}
			}
			if (excluded) {
				continue;
			}

			const Shape2DSW *col_shape = col_obj->get_shape(col_shape_idx);
			const Transform2D col_obj_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(col_shape_idx);

			// Does it collide if going all the way?
			Vector2 sep_axis = motion_normal;
			if (!CollisionSolver2DSW::solve(body_shape, body_shape_xform, p_motion, col_shape, col_obj_shape_xform, Vector2(), nullptr, nullptr, &sep_axis)) {
				continue;
			}

			// Is it stuck?
			sep_axis = motion_normal;
			if (CollisionSolver2DSW::solve(body_shape, body_shape_xform, Vector2(), col_shape, col_obj_shape_xform, Vector2(), nullptr, nullptr, &sep_axis)) {
				if (col_obj->is_shape_set_as_one_way_collision(col_shape_idx)) {
					continue;
				}
				stuck = true;
				break;
			}

			// Do kinematic solving
			real_t low = 0;
			real_t hi = 1;

			for (int step = 0; step < 8; step++) { // Steps should be customizable.

				real_t ofs = (low + hi) * 0.5;

				sep_axis = motion_normal; // Important optimization for this to work fast enough.
				if (CollisionSolver2DSW::solve(body_shape, body_shape_xform, p_motion * ofs, col_shape, col_obj_shape_xform, Vector2(), nullptr, nullptr, &sep_axis)) {
					hi = ofs;
				} else {
					low = ofs;
				}
			}

			if (col_obj->is_shape_set_as_one_way_collision(col_shape_idx)) {

				col_cbk_data.amount = 0;
				col_cbk_data.valid_dir = col_obj_shape_xform.get_axis(1).normalized();
				col_cbk_data.valid_depth = 10e20;
				col_cbk_data.invalid_by_dir = false;

				sep_axis = motion_normal;
				CollisionSolver2DSW::solve(body_shape, body_shape_xform, p_motion * hi, col_shape, col_obj_shape_xform, Vector2(), col_cbk, &col_cbk_data, &sep_axis);
				if (col_cbk_data.invalid_by_dir) {
					// Add shape pair to excluded shape pairs.
					if (excluded_shape_pair_count < max_excluded_shape_pairs) {
						ExcludedShapePairSW esp;
						esp.local_shape = body_shape;
						esp.other_object = col_obj;
						esp.other_shape_index = col_shape_idx;
						excluded_shape_pairs[excluded_shape_pair_count++] = esp;
					}
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
			best_shape = body_shape_idx; // Sadly, it's the best
			break;
		}

		if (best_safe < safe) {
			safe = best_safe;
			unsafe = best_unsafe;
			best_shape = body_shape_idx;
		}
	}

	// Step 3: If it collided, retrieve collision information using the unsafe distance.

	if (safe < 1 && r_result) {
		Transform2D ugt = body_transform;
		ugt.elements[2] += p_motion * unsafe;

		_RestCallbackData2D rcd;

		const Transform2D body_shape_xform = ugt * p_body->get_shape_transform(best_shape);
		const Shape2DSW *body_shape = p_body->get_shape(best_shape);

		body_aabb.position += p_motion * unsafe;

		pair_count = _cull_aabb_for_body(p_body, body_aabb);

		for (int pair_idx = 0; pair_idx < pair_count; pair_idx++) {

			const CollisionObject2DSW *col_obj = intersection_query_results[pair_idx];
			int shape_idx = intersection_query_subindex_results[pair_idx];

			if (CollisionObject2DSW::TYPE_BODY == col_obj->get_type()) {
				const Body2DSW *b = static_cast<const Body2DSW *>(col_obj);
				if (p_infinite_inertia && Physics2DServer::BODY_MODE_STATIC != b->get_mode() && Physics2DServer::BODY_MODE_KINEMATIC != b->get_mode()) {
					continue;
				}
			}

			// Ignore previously excluded shape pairs.
			bool excluded = false;
			for (int excluded_pair = 0; excluded_pair < excluded_shape_pair_count; excluded_pair++) {
				if (excluded_shape_pairs[excluded_pair].local_shape == body_shape && excluded_shape_pairs[excluded_pair].other_object == col_obj && excluded_shape_pairs[excluded_pair].other_shape_index == shape_idx) {
					excluded = true;
					break;
				}
			}
			if (excluded) {
				continue;
			}

			const Transform2D col_obj_shape_xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);

			if (col_obj->is_shape_set_as_one_way_collision(shape_idx)) {
				rcd.valid_dir = col_obj_shape_xform.get_axis(1).normalized();
			} else {
				rcd.valid_dir = Vector2();
			}

			rcd.object = col_obj;
			rcd.shape = shape_idx;
			CollisionSolver2DSW::solve(body_shape, body_shape_xform, Vector2(), col_obj->get_shape(shape_idx), col_obj_shape_xform, Vector2(), _rest_cbk_result, &rcd, nullptr);
		}

		ERR_FAIL_COND_V_MSG(rcd.best_len == 0, true, "Failed to extract collision information");

		r_result->motion = safe * p_motion;
		r_result->remainder = p_motion - safe * p_motion;
		r_result->motion += (body_transform.get_origin() - p_xform.get_origin());

		r_result->collider = rcd.best_object->get_self();
		r_result->collider_id = rcd.best_object->get_instance_id();
		r_result->collider_shape = rcd.best_shape;
		r_result->collision_local_shape = best_shape;
		r_result->collision_normal = rcd.best_normal;
		r_result->collision_point = rcd.best_contact;
		r_result->collider_metadata = rcd.best_object->get_shape_metadata(rcd.best_shape);
		const Body2DSW *body = static_cast<const Body2DSW *>(rcd.best_object);
		Vector2 rel_vec = r_result->collision_point - body->get_transform().get_origin();
		r_result->collider_velocity = Vector2(-body->get_angular_velocity() * rel_vec.y, body->get_angular_velocity() * rel_vec.x) + body->get_linear_velocity();
	}

	return (safe < 1);
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

	ERR_FAIL_COND_MSG(objects.has(p_object), "Object already in space");
	objects.insert(p_object);
}

void Space2DSW::remove_object(CollisionObject2DSW *p_object) {

	ERR_FAIL_COND_MSG(!objects.has(p_object), "Object not in space");
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

void Space2DSW::set_param(Physics2DServer::SpaceParameter p_param, real_t p_value) {

	switch (p_param) {

		case Physics2DServer::SPACE_PARAM_CONTACT_RECYCLE_RADIUS: contact_recycle_radius = p_value; break;
		case Physics2DServer::SPACE_PARAM_CONTACT_MAX_SEPARATION: contact_max_separation = p_value; break;
		case Physics2DServer::SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION: contact_max_allowed_penetration = p_value; break;
		case Physics2DServer::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD: body_linear_velocity_sleep_threshold = p_value; break;
		case Physics2DServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD: body_angular_velocity_sleep_threshold = p_value; break;
		case Physics2DServer::SPACE_PARAM_BODY_TIME_TO_SLEEP: body_time_to_sleep = p_value; break;
		case Physics2DServer::SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS: constraint_bias = p_value; break;
	}
}

real_t Space2DSW::get_param(Physics2DServer::SpaceParameter p_param) const {

	switch (p_param) {

		case Physics2DServer::SPACE_PARAM_CONTACT_RECYCLE_RADIUS: return contact_recycle_radius;
		case Physics2DServer::SPACE_PARAM_CONTACT_MAX_SEPARATION: return contact_max_separation;
		case Physics2DServer::SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION: return contact_max_allowed_penetration;
		case Physics2DServer::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD: return body_linear_velocity_sleep_threshold;
		case Physics2DServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD: return body_angular_velocity_sleep_threshold;
		case Physics2DServer::SPACE_PARAM_BODY_TIME_TO_SLEEP: return body_time_to_sleep;
		case Physics2DServer::SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS: return constraint_bias;
	}

	ERR_FAIL_COND_V_MSG(true, 0, "Unknown space parameter");
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

Physics2DDirectSpaceStateSW *Space2DSW::get_direct_state() {

	return direct_access;
}

Space2DSW::Space2DSW() {

	ProjectSettings::get_singleton()->set_custom_property_info("physics/2d/time_before_sleep", PropertyInfo(Variant::REAL, "physics/2d/time_before_sleep", PROPERTY_HINT_RANGE, "0,5,0.01,or_greater"));

	broadphase = BroadPhase2DSW::create_func();
	broadphase->set_pair_callback(_broadphase_pair, this);
	broadphase->set_unpair_callback(_broadphase_unpair, this);

	direct_access = memnew(Physics2DDirectSpaceStateSW);
	direct_access->space = this;

	for (int i = 0; i < ELAPSED_TIME_MAX; i++)
		elapsed_time[i] = 0;
}

Space2DSW::~Space2DSW() {

	memdelete(broadphase);
	memdelete(direct_access);
}
