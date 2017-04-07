/*************************************************************************/
/*  space_sw.cpp                                                         */
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
#include "space_sw.h"
#include "collision_solver_sw.h"
#include "global_config.h"
#include "physics_server_sw.h"

_FORCE_INLINE_ static bool _match_object_type_query(CollisionObjectSW *p_object, uint32_t p_layer_mask, uint32_t p_type_mask) {

	if (p_object->get_type() == CollisionObjectSW::TYPE_AREA)
		return p_type_mask & PhysicsDirectSpaceState::TYPE_MASK_AREA;

	if ((p_object->get_layer_mask() & p_layer_mask) == 0)
		return false;

	BodySW *body = static_cast<BodySW *>(p_object);

	return (1 << body->get_mode()) & p_type_mask;
}

bool PhysicsDirectSpaceStateSW::intersect_ray(const Vector3 &p_from, const Vector3 &p_to, RayResult &r_result, const Set<RID> &p_exclude, uint32_t p_layer_mask, uint32_t p_object_type_mask, bool p_pick_ray) {

	ERR_FAIL_COND_V(space->locked, false);

	Vector3 begin, end;
	Vector3 normal;
	begin = p_from;
	end = p_to;
	normal = (end - begin).normalized();

	int amount = space->broadphase->cull_segment(begin, end, space->intersection_query_results, SpaceSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	//todo, create another array tha references results, compute AABBs and check closest point to ray origin, sort, and stop evaluating results when beyond first collision

	bool collided = false;
	Vector3 res_point, res_normal;
	int res_shape;
	const CollisionObjectSW *res_obj;
	real_t min_d = 1e10;

	for (int i = 0; i < amount; i++) {

		if (!_match_object_type_query(space->intersection_query_results[i], p_layer_mask, p_object_type_mask))
			continue;

		if (p_pick_ray && !(static_cast<CollisionObjectSW *>(space->intersection_query_results[i])->is_ray_pickable()))
			continue;

		if (p_exclude.has(space->intersection_query_results[i]->get_self()))
			continue;

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

	if (!collided)
		return false;

	r_result.collider_id = res_obj->get_instance_id();
	if (r_result.collider_id != 0)
		r_result.collider = ObjectDB::get_instance(r_result.collider_id);
	else
		r_result.collider = NULL;
	r_result.normal = res_normal;
	r_result.position = res_point;
	r_result.rid = res_obj->get_self();
	r_result.shape = res_shape;

	return true;
}

int PhysicsDirectSpaceStateSW::intersect_shape(const RID &p_shape, const Transform &p_xform, real_t p_margin, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_layer_mask, uint32_t p_object_type_mask) {

	if (p_result_max <= 0)
		return 0;

	ShapeSW *shape = static_cast<PhysicsServerSW *>(PhysicsServer::get_singleton())->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	Rect3 aabb = p_xform.xform(shape->get_aabb());

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, SpaceSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	int cc = 0;

	//Transform ai = p_xform.affine_inverse();

	for (int i = 0; i < amount; i++) {

		if (cc >= p_result_max)
			break;

		if (!_match_object_type_query(space->intersection_query_results[i], p_layer_mask, p_object_type_mask))
			continue;

		//area can't be picked by ray (default)

		if (p_exclude.has(space->intersection_query_results[i]->get_self()))
			continue;

		const CollisionObjectSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		if (!CollisionSolverSW::solve_static(shape, p_xform, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), NULL, NULL, NULL, p_margin, 0))
			continue;

		if (r_results) {
			r_results[cc].collider_id = col_obj->get_instance_id();
			if (r_results[cc].collider_id != 0)
				r_results[cc].collider = ObjectDB::get_instance(r_results[cc].collider_id);
			else
				r_results[cc].collider = NULL;
			r_results[cc].rid = col_obj->get_self();
			r_results[cc].shape = shape_idx;
		}

		cc++;
	}

	return cc;
}

bool PhysicsDirectSpaceStateSW::cast_motion(const RID &p_shape, const Transform &p_xform, const Vector3 &p_motion, real_t p_margin, real_t &p_closest_safe, real_t &p_closest_unsafe, const Set<RID> &p_exclude, uint32_t p_layer_mask, uint32_t p_object_type_mask, ShapeRestInfo *r_info) {

	ShapeSW *shape = static_cast<PhysicsServerSW *>(PhysicsServer::get_singleton())->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, false);

	Rect3 aabb = p_xform.xform(shape->get_aabb());
	aabb = aabb.merge(Rect3(aabb.pos + p_motion, aabb.size)); //motion
	aabb = aabb.grow(p_margin);

	/*
	if (p_motion!=Vector3())
		print_line(p_motion);
	*/

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, SpaceSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	real_t best_safe = 1;
	real_t best_unsafe = 1;

	Transform xform_inv = p_xform.affine_inverse();
	MotionShapeSW mshape;
	mshape.shape = shape;
	mshape.motion = xform_inv.basis.xform(p_motion);

	bool best_first = true;

	Vector3 closest_A, closest_B;

	for (int i = 0; i < amount; i++) {

		if (!_match_object_type_query(space->intersection_query_results[i], p_layer_mask, p_object_type_mask))
			continue;

		if (p_exclude.has(space->intersection_query_results[i]->get_self()))
			continue; //ignore excluded

		const CollisionObjectSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		Vector3 point_A, point_B;
		Vector3 sep_axis = p_motion.normalized();

		Transform col_obj_xform = col_obj->get_transform() * col_obj->get_shape_transform(shape_idx);
		//test initial overlap, does it collide if going all the way?
		if (CollisionSolverSW::solve_distance(&mshape, p_xform, col_obj->get_shape(shape_idx), col_obj_xform, point_A, point_B, aabb, &sep_axis)) {
			//print_line("failed motion cast (no collision)");
			continue;
		}

//test initial overlap
#if 0
		if (CollisionSolverSW::solve_static(shape,p_xform,col_obj->get_shape(shape_idx),col_obj_xform,NULL,NULL,&sep_axis)) {
			print_line("failed initial cast (collision at beginning)");
			return false;
		}
#else
		sep_axis = p_motion.normalized();

		if (!CollisionSolverSW::solve_distance(shape, p_xform, col_obj->get_shape(shape_idx), col_obj_xform, point_A, point_B, aabb, &sep_axis)) {
			//print_line("failed motion cast (no collision)");
			return false;
		}
#endif

		//just do kinematic solving
		real_t low = 0;
		real_t hi = 1;
		Vector3 mnormal = p_motion.normalized();

		for (int i = 0; i < 8; i++) { //steps should be customizable..

			real_t ofs = (low + hi) * 0.5;

			Vector3 sep = mnormal; //important optimization for this to work fast enough

			mshape.motion = xform_inv.basis.xform(p_motion * ofs);

			Vector3 lA, lB;

			bool collided = !CollisionSolverSW::solve_distance(&mshape, p_xform, col_obj->get_shape(shape_idx), col_obj_xform, lA, lB, aabb, &sep);

			if (collided) {

				//print_line(itos(i)+": "+rtos(ofs));
				hi = ofs;
			} else {

				point_A = lA;
				point_B = lB;
				low = ofs;
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
				r_info->linear_velocity = body->get_linear_velocity() + (body->get_angular_velocity()).cross(body->get_transform().origin - closest_B);
			}
		}
	}

	p_closest_safe = best_safe;
	p_closest_unsafe = best_unsafe;

	return true;
}

bool PhysicsDirectSpaceStateSW::collide_shape(RID p_shape, const Transform &p_shape_xform, real_t p_margin, Vector3 *r_results, int p_result_max, int &r_result_count, const Set<RID> &p_exclude, uint32_t p_layer_mask, uint32_t p_object_type_mask) {

	if (p_result_max <= 0)
		return 0;

	ShapeSW *shape = static_cast<PhysicsServerSW *>(PhysicsServer::get_singleton())->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	Rect3 aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.grow(p_margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, SpaceSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	bool collided = false;
	r_result_count = 0;

	PhysicsServerSW::CollCbkData cbk;
	cbk.max = p_result_max;
	cbk.amount = 0;
	cbk.ptr = r_results;
	CollisionSolverSW::CallbackResult cbkres = NULL;

	PhysicsServerSW::CollCbkData *cbkptr = NULL;
	if (p_result_max > 0) {
		cbkptr = &cbk;
		cbkres = PhysicsServerSW::_shape_col_cbk;
	}

	for (int i = 0; i < amount; i++) {

		if (!_match_object_type_query(space->intersection_query_results[i], p_layer_mask, p_object_type_mask))
			continue;

		const CollisionObjectSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		if (p_exclude.has(col_obj->get_self())) {
			continue;
		}

		//print_line("AGAINST: "+itos(col_obj->get_self().get_id())+":"+itos(shape_idx));
		//print_line("THE ABBB: "+(col_obj->get_transform() * col_obj->get_shape_transform(shape_idx)).xform(col_obj->get_shape(shape_idx)->get_aabb()));

		if (CollisionSolverSW::solve_static(shape, p_shape_xform, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), cbkres, cbkptr, NULL, p_margin)) {
			collided = true;
		}
	}

	r_result_count = cbk.amount;

	return collided;
}

struct _RestCallbackData {

	const CollisionObjectSW *object;
	const CollisionObjectSW *best_object;
	int shape;
	int best_shape;
	Vector3 best_contact;
	Vector3 best_normal;
	real_t best_len;
};

static void _rest_cbk_result(const Vector3 &p_point_A, const Vector3 &p_point_B, void *p_userdata) {

	_RestCallbackData *rd = (_RestCallbackData *)p_userdata;

	Vector3 contact_rel = p_point_B - p_point_A;
	real_t len = contact_rel.length();
	if (len <= rd->best_len)
		return;

	rd->best_len = len;
	rd->best_contact = p_point_B;
	rd->best_normal = contact_rel / len;
	rd->best_object = rd->object;
	rd->best_shape = rd->shape;
}
bool PhysicsDirectSpaceStateSW::rest_info(RID p_shape, const Transform &p_shape_xform, real_t p_margin, ShapeRestInfo *r_info, const Set<RID> &p_exclude, uint32_t p_layer_mask, uint32_t p_object_type_mask) {

	ShapeSW *shape = static_cast<PhysicsServerSW *>(PhysicsServer::get_singleton())->shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, 0);

	Rect3 aabb = p_shape_xform.xform(shape->get_aabb());
	aabb = aabb.grow(p_margin);

	int amount = space->broadphase->cull_aabb(aabb, space->intersection_query_results, SpaceSW::INTERSECTION_QUERY_MAX, space->intersection_query_subindex_results);

	_RestCallbackData rcd;
	rcd.best_len = 0;
	rcd.best_object = NULL;
	rcd.best_shape = 0;

	for (int i = 0; i < amount; i++) {

		if (!_match_object_type_query(space->intersection_query_results[i], p_layer_mask, p_object_type_mask))
			continue;

		const CollisionObjectSW *col_obj = space->intersection_query_results[i];
		int shape_idx = space->intersection_query_subindex_results[i];

		if (p_exclude.has(col_obj->get_self()))
			continue;

		rcd.object = col_obj;
		rcd.shape = shape_idx;
		bool sc = CollisionSolverSW::solve_static(shape, p_shape_xform, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), _rest_cbk_result, &rcd, NULL, p_margin);
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
	if (rcd.best_object->get_type() == CollisionObjectSW::TYPE_BODY) {

		const BodySW *body = static_cast<const BodySW *>(rcd.best_object);
		r_info->linear_velocity = body->get_linear_velocity() +
								  (body->get_angular_velocity()).cross(body->get_transform().origin - rcd.best_contact); // * mPos);

	} else {
		r_info->linear_velocity = Vector3();
	}

	return true;
}

PhysicsDirectSpaceStateSW::PhysicsDirectSpaceStateSW() {

	space = NULL;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

void *SpaceSW::_broadphase_pair(CollisionObjectSW *A, int p_subindex_A, CollisionObjectSW *B, int p_subindex_B, void *p_self) {

	CollisionObjectSW::Type type_A = A->get_type();
	CollisionObjectSW::Type type_B = B->get_type();
	if (type_A > type_B) {

		SWAP(A, B);
		SWAP(p_subindex_A, p_subindex_B);
		SWAP(type_A, type_B);
	}

	SpaceSW *self = (SpaceSW *)p_self;

	self->collision_pairs++;

	if (type_A == CollisionObjectSW::TYPE_AREA) {

		AreaSW *area = static_cast<AreaSW *>(A);
		if (type_B == CollisionObjectSW::TYPE_AREA) {

			AreaSW *area_b = static_cast<AreaSW *>(B);
			Area2PairSW *area2_pair = memnew(Area2PairSW(area_b, p_subindex_B, area, p_subindex_A));
			return area2_pair;
		} else {

			BodySW *body = static_cast<BodySW *>(B);
			AreaPairSW *area_pair = memnew(AreaPairSW(body, p_subindex_B, area, p_subindex_A));
			return area_pair;
		}
	} else {

		BodyPairSW *b = memnew(BodyPairSW((BodySW *)A, p_subindex_A, (BodySW *)B, p_subindex_B));
		return b;
	}

	return NULL;
}

void SpaceSW::_broadphase_unpair(CollisionObjectSW *A, int p_subindex_A, CollisionObjectSW *B, int p_subindex_B, void *p_data, void *p_self) {

	SpaceSW *self = (SpaceSW *)p_self;
	self->collision_pairs--;
	ConstraintSW *c = (ConstraintSW *)p_data;
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
		b->call_queries();
		state_query_list.remove(state_query_list.first());
	}

	while (monitor_query_list.first()) {

		AreaSW *a = monitor_query_list.first()->self();
		a->call_queries();
		monitor_query_list.remove(monitor_query_list.first());
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

		case PhysicsServer::SPACE_PARAM_CONTACT_RECYCLE_RADIUS: contact_recycle_radius = p_value; break;
		case PhysicsServer::SPACE_PARAM_CONTACT_MAX_SEPARATION: contact_max_separation = p_value; break;
		case PhysicsServer::SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION: contact_max_allowed_penetration = p_value; break;
		case PhysicsServer::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_TRESHOLD: body_linear_velocity_sleep_threshold = p_value; break;
		case PhysicsServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_TRESHOLD: body_angular_velocity_sleep_threshold = p_value; break;
		case PhysicsServer::SPACE_PARAM_BODY_TIME_TO_SLEEP: body_time_to_sleep = p_value; break;
		case PhysicsServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_DAMP_RATIO: body_angular_velocity_damp_ratio = p_value; break;
		case PhysicsServer::SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS: constraint_bias = p_value; break;
	}
}

real_t SpaceSW::get_param(PhysicsServer::SpaceParameter p_param) const {

	switch (p_param) {

		case PhysicsServer::SPACE_PARAM_CONTACT_RECYCLE_RADIUS: return contact_recycle_radius;
		case PhysicsServer::SPACE_PARAM_CONTACT_MAX_SEPARATION: return contact_max_separation;
		case PhysicsServer::SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION: return contact_max_allowed_penetration;
		case PhysicsServer::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_TRESHOLD: return body_linear_velocity_sleep_threshold;
		case PhysicsServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_TRESHOLD: return body_angular_velocity_sleep_threshold;
		case PhysicsServer::SPACE_PARAM_BODY_TIME_TO_SLEEP: return body_time_to_sleep;
		case PhysicsServer::SPACE_PARAM_BODY_ANGULAR_VELOCITY_DAMP_RATIO: return body_angular_velocity_damp_ratio;
		case PhysicsServer::SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS: return constraint_bias;
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
	body_angular_velocity_damp_ratio = 10;

	broadphase = BroadPhaseSW::create_func();
	broadphase->set_pair_callback(_broadphase_pair, this);
	broadphase->set_unpair_callback(_broadphase_unpair, this);
	area = NULL;

	direct_access = memnew(PhysicsDirectSpaceStateSW);
	direct_access->space = this;

	for (int i = 0; i < ELAPSED_TIME_MAX; i++)
		elapsed_time[i] = 0;
}

SpaceSW::~SpaceSW() {

	memdelete(broadphase);
	memdelete(direct_access);
}
