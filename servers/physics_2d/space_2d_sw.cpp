/*************************************************************************/
/*  space_2d_sw.cpp                                                      */
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
#include "space_2d_sw.h"
#include "collision_solver_2d_sw.h"
#include "physics_2d_server_sw.h"

_FORCE_INLINE_ static bool _match_object_type_query(CollisionObject2DSW *p_object, uint32_t p_layer_mask, uint32_t p_type_mask) {

	if ((p_object->get_layer_mask() & p_layer_mask) == 0)
		return false;

	if (p_object->get_type() == CollisionObject2DSW::TYPE_AREA)
		return p_type_mask & Physics2DDirectSpaceState::TYPE_MASK_AREA;

	Body2DSW *body = static_cast<Body2DSW *>(p_object);

	return (1 << body->get_mode()) & p_type_mask;
}

int Physics2DDirectSpaceStateSW::intersect_point(const Vector2 &p_point, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_layer_mask, uint32_t p_object_type_mask, bool p_pick_point) {

	if (p_result_max <= 0)
		return 0;

	Rect2 aabb;
	aabb.pos = p_point - Vector2(0.00001, 0.00001);
	aabb.size = Vector2(0.00002, 0.00002);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	int cc = 0;

	for (int i = 0; i < amount; i++) {

		if (!_match_object_type_query(space->intersection_query_results[i], p_layer_mask, p_object_type_mask))
			continue;

		if (p_exclude.has(space->intersection_query_results[i]->get_self()))
			continue;

		const CollisionObject2DSW *col_obj = space->intersection_query_results[i];

		if (p_pick_point && !col_obj->is_pickable())
			continue;

		int shape_idx = space->intersection_query_subindex_results[i];

		Shape2DSW *shape = col_obj->get_shape(shape_idx);

		Vector2 local_point = (col_obj->get_transform() * col_obj->get_shape_transform(shape_idx)).affine_inverse().xform(p_point);

		if (!shape->contains_point(local_point))
			continue;

		if (cc >= p_result_max)
			continue;

		r_results[cc].collider_id = col_obj->get_instance_id();
		if (r_results[cc].collider_id != 0)
			r_results[cc].collider = ObjectDB::get_instance(r_results[cc].collider_id);
		r_results[cc].rid = col_obj->get_self();
		r_results[cc].shape = shape_idx;
		r_results[cc].metadata = col_obj->get_shape_metadata(shape_idx);

		cc++;
	}

	return cc;
}

bool Physics2DDirectSpaceStateSW::intersect_ray(const Vector2 &p_from, const Vector2 &p_to, RayResult &r_result, const Set<RID> &p_exclude, uint32_t p_layer_mask, uint32_t p_object_type_mask) {

	ERR_FAIL_COND_V(space->locked, false);

	Vector2 begin, end;
	Vector2 normal;
	begin = p_from;
	end = p_to;
	normal = (end - begin).normalized();

	int amount = space->broadphase->cull_segment(begin, end, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	//todo, create another array tha references results, compute AABBs and check closest point to ray origin, sort, and stop evaluating results when beyond first collision

	bool collided = false;
	Vector2 res_point, res_normal;
	int res_shape;
	const CollisionObject2DSW *res_obj;
	real_t min_d = 1e10;

	for (int i = 0; i < amount; i++) {

		if (!_match_object_type_query(space->intersection_query_results[i], p_layer_mask, p_object_type_mask))
			continue;

		if (p_exclude.has(space->intersection_query_results[i]->get_self()))
			continue;

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

	if (!collided)
		return false;

	r_result.collider_id = res_obj->get_instance_id();
	if (r_result.collider_id != 0)
		r_result.collider = ObjectDB::get_instance(r_result.collider_id);
	r_result.normal = res_normal;
	r_result.metadata = res_obj->get_shape_metadata(res_shape);
	r_result.position = res_point;
	r_result.rid = res_obj->get_self();
	r_result.shape = res_shape;

	return true;
}

int Physics2DDirectSpaceStateSW::intersect_shape(const RID &p_shape, const Transform2D &p_xform, const Vector2 &p_motion, real_t p_margin, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_layer_mask, uint32_t p_object_type_mask) {

	if (p_result_max <= 0)
		return 0;

	Shape2DSW *shape = Physics2DServerSW::singletonsw->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	Rect2 aabb = p_xform.xform(shape->get_aabb());
	aabb = aabb.grow(p_margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, p_result_max, space->intersection_query_subindex_results);

	int cc = 0;

	for (int i = 0; i < amount; i++) {

		if (!_match_object_type_query(space->intersection_query_results[i], p_layer_mask, p_object_type_mask))
			continue;

		if (p_exclude.has(space->intersection_query_results[i]->get_self()))
			continue;

		const CollisionObject2DSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		if (!CollisionSolver2DSW::solve(shape, p_xform, p_motion, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), Vector2(), NULL, NULL, NULL, p_margin))
			continue;

		r_results[cc].collider_id = col_obj->get_instance_id();
		if (r_results[cc].collider_id != 0)
			r_results[cc].collider = ObjectDB::get_instance(r_results[cc].collider_id);
		r_results[cc].rid = col_obj->get_self();
		r_results[cc].shape = shape_idx;
		r_results[cc].metadata = col_obj->get_shape_metadata(shape_idx);

		cc++;
	}

	return cc;
}

bool Physics2DDirectSpaceStateSW::cast_motion(const RID &p_shape, const Transform2D &p_xform, const Vector2 &p_motion, real_t p_margin, real_t &p_closest_safe, real_t &p_closest_unsafe, const Set<RID> &p_exclude, uint32_t p_layer_mask, uint32_t p_object_type_mask) {

	Shape2DSW *shape = Physics2DServerSW::singletonsw->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, false);

	Rect2 aabb = p_xform.xform(shape->get_aabb());
	aabb = aabb.merge(Rect2(aabb.pos + p_motion, aabb.size)); //motion
	aabb = aabb.grow(p_margin);

	/*
	if (p_motion!=Vector2())
		print_line(p_motion);
	*/

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	real_t best_safe = 1;
	real_t best_unsafe = 1;

	for (int i = 0; i < amount; i++) {

		if (!_match_object_type_query(space->intersection_query_results[i], p_layer_mask, p_object_type_mask))
			continue;

		if (p_exclude.has(space->intersection_query_results[i]->get_self()))
			continue; //ignore excluded

		const CollisionObject2DSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		/*if (col_obj->get_type()==CollisionObject2DSW::TYPE_BODY) {

			const Body2DSW *body=static_cast<const Body2DSW*>(col_obj);
			if (body->get_one_way_collision_direction()!=Vector2() && p_motion.dot(body->get_one_way_collision_direction())<=CMP_EPSILON) {
				print_line("failed in motion dir");
				continue;
			}
		}*/

		Transform2D col_obj_xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);
		//test initial overlap, does it collide if going all the way?
		if (!CollisionSolver2DSW::solve(shape, p_xform, p_motion, col_obj->get_shape(shape_idx), col_obj_xform, Vector2(), NULL, NULL, NULL, p_margin)) {
			continue;
		}

		//test initial overlap
		if (CollisionSolver2DSW::solve(shape, p_xform, Vector2(), col_obj->get_shape(shape_idx), col_obj_xform, Vector2(), NULL, NULL, NULL, p_margin)) {

			if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {
				//if one way collision direction ignore initial overlap
				const Body2DSW *body = static_cast<const Body2DSW *>(col_obj);
				if (body->get_one_way_collision_direction() != Vector2()) {
					continue;
				}
			}

			return false;
		}

		//just do kinematic solving
		real_t low = 0;
		real_t hi = 1;
		Vector2 mnormal = p_motion.normalized();

		for (int i = 0; i < 8; i++) { //steps should be customizable..

			real_t ofs = (low + hi) * 0.5;

			Vector2 sep = mnormal; //important optimization for this to work fast enough
			bool collided = CollisionSolver2DSW::solve(shape, p_xform, p_motion * ofs, col_obj->get_shape(shape_idx), col_obj_xform, Vector2(), NULL, NULL, &sep, p_margin);

			if (collided) {

				hi = ofs;
			} else {

				low = ofs;
			}
		}

		if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {

			const Body2DSW *body = static_cast<const Body2DSW *>(col_obj);
			if (body->get_one_way_collision_direction() != Vector2()) {

				Vector2 cd[2];
				Physics2DServerSW::CollCbkData cbk;
				cbk.max = 1;
				cbk.amount = 0;
				cbk.ptr = cd;
				cbk.valid_dir = body->get_one_way_collision_direction();
				cbk.valid_depth = body->get_one_way_collision_max_depth();

				Vector2 sep = mnormal; //important optimization for this to work fast enough
				bool collided = CollisionSolver2DSW::solve(shape, p_xform, p_motion * (hi + space->contact_max_allowed_penetration), col_obj->get_shape(shape_idx), col_obj_xform, Vector2(), Physics2DServerSW::_shape_col_cbk, &cbk, &sep, p_margin);
				if (!collided || cbk.amount == 0) {
					continue;
				}
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

bool Physics2DDirectSpaceStateSW::collide_shape(RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, real_t p_margin, Vector2 *r_results, int p_result_max, int &r_result_count, const Set<RID> &p_exclude, uint32_t p_layer_mask, uint32_t p_object_type_mask) {

	if (p_result_max <= 0)
		return 0;

	Shape2DSW *shape = Physics2DServerSW::singletonsw->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	Rect2 aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.merge(Rect2(aabb.pos + p_motion, aabb.size)); //motion
	aabb = aabb.grow(p_margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	bool collided = false;
	r_result_count = 0;

	Physics2DServerSW::CollCbkData cbk;
	cbk.max = p_result_max;
	cbk.amount = 0;
	cbk.ptr = r_results;
	CollisionSolver2DSW::CallbackResult cbkres = NULL;

	Physics2DServerSW::CollCbkData *cbkptr = NULL;
	if (p_result_max > 0) {
		cbkptr = &cbk;
		cbkres = Physics2DServerSW::_shape_col_cbk;
	}

	for (int i = 0; i < amount; i++) {

		if (!_match_object_type_query(space->intersection_query_results[i], p_layer_mask, p_object_type_mask))
			continue;

		const CollisionObject2DSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		if (p_exclude.has(col_obj->get_self()))
			continue;
		if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {

			const Body2DSW *body = static_cast<const Body2DSW *>(col_obj);
			cbk.valid_dir = body->get_one_way_collision_direction();
			cbk.valid_depth = body->get_one_way_collision_max_depth();
		} else {
			cbk.valid_dir = Vector2();
			cbk.valid_depth = 0;
		}

		if (CollisionSolver2DSW::solve(shape, p_shape_xform, p_motion, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), Vector2(), cbkres, cbkptr, NULL, p_margin)) {
			collided = p_result_max == 0 || cbk.amount > 0;
		}
	}

	r_result_count = cbk.amount;

	return collided;
}

struct _RestCallbackData2D {

	const CollisionObject2DSW *object;
	const CollisionObject2DSW *best_object;
	int shape;
	int best_shape;
	Vector2 best_contact;
	Vector2 best_normal;
	real_t best_len;
	Vector2 valid_dir;
	real_t valid_depth;
};

static void _rest_cbk_result(const Vector2 &p_point_A, const Vector2 &p_point_B, void *p_userdata) {

	_RestCallbackData2D *rd = (_RestCallbackData2D *)p_userdata;

	if (rd->valid_dir != Vector2()) {

		if (rd->valid_dir != Vector2()) {
			if (p_point_A.distance_squared_to(p_point_B) > rd->valid_depth * rd->valid_depth)
				return;
			if (rd->valid_dir.dot((p_point_A - p_point_B).normalized()) < Math_PI * 0.25)
				return;
		}
	}

	Vector2 contact_rel = p_point_B - p_point_A;
	real_t len = contact_rel.length();
	if (len <= rd->best_len)
		return;

	rd->best_len = len;
	rd->best_contact = p_point_B;
	rd->best_normal = contact_rel / len;
	rd->best_object = rd->object;
	rd->best_shape = rd->shape;
}

bool Physics2DDirectSpaceStateSW::rest_info(RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, real_t p_margin, ShapeRestInfo *r_info, const Set<RID> &p_exclude, uint32_t p_layer_mask, uint32_t p_object_type_mask) {

	Shape2DSW *shape = Physics2DServerSW::singletonsw->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	Rect2 aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.merge(Rect2(aabb.pos + p_motion, aabb.size)); //motion
	aabb = aabb.grow(p_margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, Space2DSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	_RestCallbackData2D rcd;
	rcd.best_len = 0;
	rcd.best_object = NULL;
	rcd.best_shape = 0;

	for (int i = 0; i < amount; i++) {

		if (!_match_object_type_query(space->intersection_query_results[i], p_layer_mask, p_object_type_mask))
			continue;

		const CollisionObject2DSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		if (p_exclude.has(col_obj->get_self()))
			continue;

		if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {

			const Body2DSW *body = static_cast<const Body2DSW *>(col_obj);
			rcd.valid_dir = body->get_one_way_collision_direction();
			rcd.valid_depth = body->get_one_way_collision_max_depth();
		} else {
			rcd.valid_dir = Vector2();
			rcd.valid_depth = 0;
		}

		rcd.object = col_obj;
		rcd.shape = shape_idx;
		bool sc = CollisionSolver2DSW::solve(shape, p_shape_xform, p_motion, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), Vector2(), _rest_cbk_result, &rcd, NULL, p_margin);
		if (!sc)
			continue;
	}

	if (rcd.best_len == 0)
		return false;

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

Physics2DDirectSpaceStateSW::Physics2DDirectSpaceStateSW() {

	space = NULL;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

int Space2DSW::_cull_aabb_for_body(Body2DSW *p_body, const Rect2 &p_aabb) {

	int amount = broadphase->cull_aabb(p_aabb, intersection_query_results, INTERSECTION_QUERY_MAX, intersection_query_subindex_results);

	for (int i = 0; i < amount; i++) {

		bool keep = true;

		if (intersection_query_results[i] == p_body)
			keep = false;
		else if (intersection_query_results[i]->get_type() == CollisionObject2DSW::TYPE_AREA)
			keep = false;
		else if ((static_cast<Body2DSW *>(intersection_query_results[i])->test_collision_mask(p_body)) == 0)
			keep = false;
		else if (static_cast<Body2DSW *>(intersection_query_results[i])->has_exception(p_body->get_self()) || p_body->has_exception(intersection_query_results[i]->get_self()))
			keep = false;
		else if (static_cast<Body2DSW *>(intersection_query_results[i])->is_shape_set_as_trigger(intersection_query_subindex_results[i]))
			keep = false;

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

bool Space2DSW::test_body_motion(Body2DSW *p_body, const Transform2D &p_from, const Vector2 &p_motion, real_t p_margin, Physics2DServer::MotionResult *r_result) {

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
	Rect2 body_aabb;

	for (int i = 0; i < p_body->get_shape_count(); i++) {

		if (i == 0)
			body_aabb = p_body->get_shape_aabb(i);
		else
			body_aabb = body_aabb.merge(p_body->get_shape_aabb(i));
	}

	// Undo the currently transform the physics server is aware of and apply the provided one
	body_aabb = p_from.xform(p_body->get_inv_transform().xform(body_aabb));
	body_aabb = body_aabb.grow(p_margin);

	Transform2D body_transform = p_from;

	{
		//STEP 1, FREE BODY IF STUCK

		const int max_results = 32;
		int recover_attempts = 4;
		Vector2 sr[max_results * 2];

		do {

			Physics2DServerSW::CollCbkData cbk;
			cbk.max = max_results;
			cbk.amount = 0;
			cbk.ptr = sr;

			CollisionSolver2DSW::CallbackResult cbkres = NULL;

			Physics2DServerSW::CollCbkData *cbkptr = NULL;
			cbkptr = &cbk;
			cbkres = Physics2DServerSW::_shape_col_cbk;

			bool collided = false;

			int amount = _cull_aabb_for_body(p_body, body_aabb);

			for (int j = 0; j < p_body->get_shape_count(); j++) {
				if (p_body->is_shape_set_as_trigger(j))
					continue;

				Transform2D body_shape_xform = body_transform * p_body->get_shape_transform(j);
				Shape2DSW *body_shape = p_body->get_shape(j);
				for (int i = 0; i < amount; i++) {

					const CollisionObject2DSW *col_obj = intersection_query_results[i];
					int shape_idx = intersection_query_subindex_results[i];

					if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {

						const Body2DSW *body = static_cast<const Body2DSW *>(col_obj);

						Vector2 cdir = body->get_one_way_collision_direction();
						/*
						if (cdir!=Vector2() && p_motion.dot(cdir)<0)
							continue;
						*/

						cbk.valid_dir = cdir;
						cbk.valid_depth = body->get_one_way_collision_max_depth();
					} else {
						cbk.valid_dir = Vector2();
						cbk.valid_depth = 0;
					}

					if (CollisionSolver2DSW::solve(body_shape, body_shape_xform, Vector2(), col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), Vector2(), cbkres, cbkptr, NULL, p_margin)) {
						collided = cbk.amount > 0;
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

#if 0
				Vector2 rel = b-a;
				real_t d = rel.length();
				if (d==0)
					continue;

				Vector2 n = rel/d;
				real_t traveled = n.dot(recover_motion);
				a+=n*traveled;

				real_t d = a.distance_to(b);
				if (d<margin)
					continue;
#endif
				recover_motion += (b - a) * 0.4;
			}

			if (recover_motion == Vector2()) {
				collided = false;
				break;
			}

			body_transform.elements[2] += recover_motion;
			body_aabb.pos += recover_motion;

			recover_attempts--;

		} while (recover_attempts);
	}

	real_t safe = 1.0;
	real_t unsafe = 1.0;
	int best_shape = -1;

	{
		// STEP 2 ATTEMPT MOTION

		Rect2 motion_aabb = body_aabb;
		motion_aabb.pos += p_motion;
		motion_aabb = motion_aabb.merge(body_aabb);

		int amount = _cull_aabb_for_body(p_body, motion_aabb);

		for (int j = 0; j < p_body->get_shape_count(); j++) {

			if (p_body->is_shape_set_as_trigger(j))
				continue;

			Transform2D body_shape_xform = body_transform * p_body->get_shape_transform(j);
			Shape2DSW *body_shape = p_body->get_shape(j);

			bool stuck = false;

			real_t best_safe = 1;
			real_t best_unsafe = 1;

			for (int i = 0; i < amount; i++) {

				const CollisionObject2DSW *col_obj = intersection_query_results[i];
				int shape_idx = intersection_query_subindex_results[i];

				Transform2D col_obj_xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);
				//test initial overlap, does it collide if going all the way?
				if (!CollisionSolver2DSW::solve(body_shape, body_shape_xform, p_motion, col_obj->get_shape(shape_idx), col_obj_xform, Vector2(), NULL, NULL, NULL, 0)) {
					continue;
				}

				//test initial overlap
				if (CollisionSolver2DSW::solve(body_shape, body_shape_xform, Vector2(), col_obj->get_shape(shape_idx), col_obj_xform, Vector2(), NULL, NULL, NULL, 0)) {

					if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {
						//if one way collision direction ignore initial overlap
						const Body2DSW *body = static_cast<const Body2DSW *>(col_obj);
						if (body->get_one_way_collision_direction() != Vector2()) {
							continue;
						}
					}

					stuck = true;
					break;
				}

				//just do kinematic solving
				real_t low = 0;
				real_t hi = 1;
				Vector2 mnormal = p_motion.normalized();

				for (int i = 0; i < 8; i++) { //steps should be customizable..

					real_t ofs = (low + hi) * 0.5;

					Vector2 sep = mnormal; //important optimization for this to work fast enough
					bool collided = CollisionSolver2DSW::solve(body_shape, body_shape_xform, p_motion * ofs, col_obj->get_shape(shape_idx), col_obj_xform, Vector2(), NULL, NULL, &sep, 0);

					if (collided) {

						hi = ofs;
					} else {

						low = ofs;
					}
				}

				if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {

					const Body2DSW *body = static_cast<const Body2DSW *>(col_obj);
					if (body->get_one_way_collision_direction() != Vector2()) {

						Vector2 cd[2];
						Physics2DServerSW::CollCbkData cbk;
						cbk.max = 1;
						cbk.amount = 0;
						cbk.ptr = cd;
						cbk.valid_dir = body->get_one_way_collision_direction();
						cbk.valid_depth = body->get_one_way_collision_max_depth();

						Vector2 sep = mnormal; //important optimization for this to work fast enough
						bool collided = CollisionSolver2DSW::solve(body_shape, body_shape_xform, p_motion * (hi + contact_max_allowed_penetration), col_obj->get_shape(shape_idx), col_obj_xform, Vector2(), Physics2DServerSW::_shape_col_cbk, &cbk, &sep, 0);
						if (!collided || cbk.amount == 0) {
							continue;
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
	if (safe >= 1) {
		//not collided
		collided = false;
		if (r_result) {

			r_result->motion = p_motion;
			r_result->remainder = Vector2();
			r_result->motion += (body_transform.get_origin() - p_from.get_origin());
		}

	} else {

		//it collided, let's get the rest info in unsafe advance
		Transform2D ugt = body_transform;
		ugt.elements[2] += p_motion * unsafe;

		_RestCallbackData2D rcd;
		rcd.best_len = 0;
		rcd.best_object = NULL;
		rcd.best_shape = 0;

		Transform2D body_shape_xform = ugt * p_body->get_shape_transform(best_shape);
		Shape2DSW *body_shape = p_body->get_shape(best_shape);

		body_aabb.pos += p_motion * unsafe;

		int amount = _cull_aabb_for_body(p_body, body_aabb);

		for (int i = 0; i < amount; i++) {

			const CollisionObject2DSW *col_obj = intersection_query_results[i];
			int shape_idx = intersection_query_subindex_results[i];

			if (col_obj->get_type() == CollisionObject2DSW::TYPE_BODY) {

				const Body2DSW *body = static_cast<const Body2DSW *>(col_obj);
				rcd.valid_dir = body->get_one_way_collision_direction();
				rcd.valid_depth = body->get_one_way_collision_max_depth();
			} else {
				rcd.valid_dir = Vector2();
				rcd.valid_depth = 0;
			}

			rcd.object = col_obj;
			rcd.shape = shape_idx;
			bool sc = CollisionSolver2DSW::solve(body_shape, body_shape_xform, Vector2(), col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), Vector2(), _rest_cbk_result, &rcd, NULL, p_margin);
			if (!sc)
				continue;
		}

		if (rcd.best_len != 0) {

			if (r_result) {
				r_result->collider = rcd.best_object->get_self();
				r_result->collider_id = rcd.best_object->get_instance_id();
				r_result->collider_shape = rcd.best_shape;
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
		} else {
			if (r_result) {

				r_result->motion = p_motion;
				r_result->remainder = Vector2();
				r_result->motion += (body_transform.get_origin() - p_from.get_origin());
			}

			collided = false;
		}
	}

	return collided;

#if 0
	//give me back regular physics engine logic
	//this is madness
	//and most people using this function will think
	//what it does is simpler than using physics
	//this took about a week to get right..
	//but is it right? who knows at this point..


	colliding=false;
	ERR_FAIL_COND_V(!is_inside_tree(),Vector2());
	Physics2DDirectSpaceState *dss = Physics2DServer::get_singleton()->space_get_direct_state(get_world_2d()->get_space());
	ERR_FAIL_COND_V(!dss,Vector2());
	const int max_shapes=32;
	Vector2 sr[max_shapes*2];
	int res_shapes;

	Set<RID> exclude;
	exclude.insert(get_rid());


	//recover first
	int recover_attempts=4;

	bool collided=false;
	uint32_t mask=0;
	if (collide_static)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_STATIC_BODY;
	if (collide_kinematic)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_KINEMATIC_BODY;
	if (collide_rigid)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_RIGID_BODY;
	if (collide_character)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_CHARACTER_BODY;

	//print_line("motion: "+p_motion+" margin: "+rtos(margin));

	//print_line("margin: "+rtos(margin));
	do {

		//motion recover
		for(int i=0;i<get_shape_count();i++) {

			if (is_shape_set_as_trigger(i))
				continue;
			if (dss->collide_shape(get_shape(i)->get_rid(), get_global_transform() * get_shape_transform(i),Vector2(),margin,sr,max_shapes,res_shapes,exclude,get_layer_mask(),mask))
				collided=true;

		}

		if (!collided)
			break;

		Vector2 recover_motion;

		for(int i=0;i<res_shapes;i++) {

			Vector2 a = sr[i*2+0];
			Vector2 b = sr[i*2+1];

			real_t d = a.distance_to(b);

			/*
			if (d<margin)
				continue;
			*/
			recover_motion+=(b-a)*0.4;
		}

		if (recover_motion==Vector2()) {
			collided=false;
			break;
		}

		Matrix32 gt = get_global_transform();
		gt.elements[2]+=recover_motion;
		set_global_transform(gt);

		recover_attempts--;

	} while (recover_attempts);


	//move second
	real_t safe = 1.0;
	real_t unsafe = 1.0;
	int best_shape=-1;

	for(int i=0;i<get_shape_count();i++) {

		if (is_shape_set_as_trigger(i))
			continue;

		real_t lsafe,lunsafe;
		bool valid = dss->cast_motion(get_shape(i)->get_rid(), get_global_transform() * get_shape_transform(i), p_motion, 0,lsafe,lunsafe,exclude,get_layer_mask(),mask);
		//print_line("shape: "+itos(i)+" travel:"+rtos(ltravel));
		if (!valid) {

			safe=0;
			unsafe=0;
			best_shape=i; //sadly it's the best
			break;
		}
		if (lsafe==1.0) {
			continue;
		}
		if (lsafe < safe) {

			safe=lsafe;
			unsafe=lunsafe;
			best_shape=i;
		}
	}


	//print_line("best shape: "+itos(best_shape)+" motion "+p_motion);

	if (safe>=1) {
		//not collided
		colliding=false;
	} else {

		//it collided, let's get the rest info in unsafe advance
		Matrix32 ugt = get_global_transform();
		ugt.elements[2]+=p_motion*unsafe;
		Physics2DDirectSpaceState::ShapeRestInfo rest_info;
		bool c2 = dss->rest_info(get_shape(best_shape)->get_rid(), ugt*get_shape_transform(best_shape), Vector2(), margin,&rest_info,exclude,get_layer_mask(),mask);
		if (!c2) {
			//should not happen, but floating point precision is so weird..

			colliding=false;
		} else {


			//print_line("Travel: "+rtos(travel));
			colliding=true;
			collision=rest_info.point;
			normal=rest_info.normal;
			collider=rest_info.collider_id;
			collider_vel=rest_info.linear_velocity;
			collider_shape=rest_info.shape;
			collider_metadata=rest_info.metadata;
		}

	}

	Vector2 motion=p_motion*safe;
	Matrix32 gt = get_global_transform();
	gt.elements[2]+=motion;
	set_global_transform(gt);

	return p_motion-motion;

#endif
	return false;
}

void *Space2DSW::_broadphase_pair(CollisionObject2DSW *A, int p_subindex_A, CollisionObject2DSW *B, int p_subindex_B, void *p_self) {

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

	return NULL;
}

void Space2DSW::_broadphase_unpair(CollisionObject2DSW *A, int p_subindex_A, CollisionObject2DSW *B, int p_subindex_B, void *p_data, void *p_self) {

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
		b->call_queries();
		state_query_list.remove(state_query_list.first());
	}

	while (monitor_query_list.first()) {

		Area2DSW *a = monitor_query_list.first()->self();
		a->call_queries();
		monitor_query_list.remove(monitor_query_list.first());
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
		case Physics2DServer::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_TRESHOLD: body_linear_velocity_sleep_treshold = p_value; break;
		case Physics2DServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_TRESHOLD: body_angular_velocity_sleep_treshold = p_value; break;
		case Physics2DServer::SPACE_PARAM_BODY_TIME_TO_SLEEP: body_time_to_sleep = p_value; break;
		case Physics2DServer::SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS: constraint_bias = p_value; break;
	}
}

real_t Space2DSW::get_param(Physics2DServer::SpaceParameter p_param) const {

	switch (p_param) {

		case Physics2DServer::SPACE_PARAM_CONTACT_RECYCLE_RADIUS: return contact_recycle_radius;
		case Physics2DServer::SPACE_PARAM_CONTACT_MAX_SEPARATION: return contact_max_separation;
		case Physics2DServer::SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION: return contact_max_allowed_penetration;
		case Physics2DServer::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_TRESHOLD: return body_linear_velocity_sleep_treshold;
		case Physics2DServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_TRESHOLD: return body_angular_velocity_sleep_treshold;
		case Physics2DServer::SPACE_PARAM_BODY_TIME_TO_SLEEP: return body_time_to_sleep;
		case Physics2DServer::SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS: return constraint_bias;
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

Physics2DDirectSpaceStateSW *Space2DSW::get_direct_state() {

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

	constraint_bias = 0.2;
	body_linear_velocity_sleep_treshold = GLOBAL_DEF("physics/2d/sleep_threashold_linear", 2.0);
	body_angular_velocity_sleep_treshold = GLOBAL_DEF("physics/2d/sleep_threshold_angular", (8.0 / 180.0 * Math_PI));
	body_time_to_sleep = GLOBAL_DEF("physics/2d/time_before_sleep", 0.5);

	broadphase = BroadPhase2DSW::create_func();
	broadphase->set_pair_callback(_broadphase_pair, this);
	broadphase->set_unpair_callback(_broadphase_unpair, this);
	area = NULL;

	direct_access = memnew(Physics2DDirectSpaceStateSW);
	direct_access->space = this;

	for (int i = 0; i < ELAPSED_TIME_MAX; i++)
		elapsed_time[i] = 0;
}

Space2DSW::~Space2DSW() {

	memdelete(broadphase);
	memdelete(direct_access);
}
