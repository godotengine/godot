/*************************************************************************/
/*  space_2d_sw.h                                                        */
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

#ifndef SPACE_2D_SW_H
#define SPACE_2D_SW_H

#include "area_2d_sw.h"
#include "area_pair_2d_sw.h"
#include "body_2d_sw.h"
#include "body_pair_2d_sw.h"
#include "broad_phase_2d_sw.h"
#include "collision_object_2d_sw.h"
#include "core/hash_map.h"
#include "core/project_settings.h"
#include "core/typedefs.h"

class Physics2DDirectSpaceStateSW : public Physics2DDirectSpaceState {
	GDCLASS(Physics2DDirectSpaceStateSW, Physics2DDirectSpaceState);

	int _intersect_point_impl(const Vector2 &p_point, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_pick_point, bool p_filter_by_canvas = false, ObjectID p_canvas_instance_id = 0);

public:
	Space2DSW *space;

	virtual int intersect_point(const Vector2 &p_point, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false, bool p_pick_point = false);
	virtual int intersect_point_on_canvas(const Vector2 &p_point, ObjectID p_canvas_instance_id, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false, bool p_pick_point = false);
	virtual bool intersect_ray(const Vector2 &p_from, const Vector2 &p_to, RayResult &r_result, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false);
	virtual int intersect_shape(const RID &p_shape, const Transform2D &p_xform, const Vector2 &p_motion, real_t p_margin, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false);
	virtual bool cast_motion(const RID &p_shape, const Transform2D &p_xform, const Vector2 &p_motion, real_t p_margin, real_t &p_closest_safe, real_t &p_closest_unsafe, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false);
	virtual bool collide_shape(RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, real_t p_margin, Vector2 *r_results, int p_result_max, int &r_result_count, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false);
	virtual bool rest_info(RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, real_t p_margin, ShapeRestInfo *r_info, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false);

	Physics2DDirectSpaceStateSW();
};

class Space2DSW : public RID_Data {
public:
	enum ElapsedTime {
		ELAPSED_TIME_INTEGRATE_FORCES,
		ELAPSED_TIME_GENERATE_ISLANDS,
		ELAPSED_TIME_SETUP_CONSTRAINTS,
		ELAPSED_TIME_SOLVE_CONSTRAINTS,
		ELAPSED_TIME_INTEGRATE_VELOCITIES,
		ELAPSED_TIME_MAX

	};

private:
	struct ExcludedShapeSW {
		Shape2DSW *local_shape;
		const CollisionObject2DSW *against_object;
		int against_shape_index;
	};

	uint64_t elapsed_time[ELAPSED_TIME_MAX];

	Physics2DDirectSpaceStateSW *direct_access;
	RID self;

	BroadPhase2DSW *broadphase;
	SelfList<Body2DSW>::List active_list;
	SelfList<Body2DSW>::List inertia_update_list;
	SelfList<Body2DSW>::List state_query_list;
	SelfList<Area2DSW>::List monitor_query_list;
	SelfList<Area2DSW>::List area_moved_list;

	static void *_broadphase_pair(CollisionObject2DSW *p_object_A, int p_subindex_A, CollisionObject2DSW *p_object_B, int p_subindex_B, void *p_pair_data, void *p_self);
	static void _broadphase_unpair(CollisionObject2DSW *p_object_A, int p_subindex_A, CollisionObject2DSW *p_object_B, int p_subindex_B, void *p_pair_data, void *p_self);

	Set<CollisionObject2DSW *> objects;

	Area2DSW *area;

	real_t contact_recycle_radius;
	real_t contact_max_separation;
	real_t contact_max_allowed_penetration;
	real_t constraint_bias;

	enum {

		INTERSECTION_QUERY_MAX = 2048
	};

	CollisionObject2DSW *intersection_query_results[INTERSECTION_QUERY_MAX];
	int intersection_query_subindex_results[INTERSECTION_QUERY_MAX];

	real_t body_linear_velocity_sleep_threshold;
	real_t body_angular_velocity_sleep_threshold;
	real_t body_time_to_sleep;

	bool locked;

	real_t step;
	int island_count;
	int active_objects;
	int collision_pairs;

	int _cull_aabb_for_body(Body2DSW *p_body, const Rect2 &p_aabb);

	Vector<Vector2> contact_debug;
	int contact_debug_count;

	friend class Physics2DDirectSpaceStateSW;

public:
	_FORCE_INLINE_ void set_self(const RID &p_self) { self = p_self; }
	_FORCE_INLINE_ RID get_self() const { return self; }

	_FORCE_INLINE_ void set_step(const real_t &p_step) { step = p_step; }
	_FORCE_INLINE_ real_t get_step() const { return step; }

	void set_default_area(Area2DSW *p_area) { area = p_area; }
	Area2DSW *get_default_area() const { return area; }

	const SelfList<Body2DSW>::List &get_active_body_list() const;
	void body_add_to_active_list(SelfList<Body2DSW> *p_body);
	void body_remove_from_active_list(SelfList<Body2DSW> *p_body);
	void body_add_to_inertia_update_list(SelfList<Body2DSW> *p_body);
	void body_remove_from_inertia_update_list(SelfList<Body2DSW> *p_body);
	void area_add_to_moved_list(SelfList<Area2DSW> *p_area);
	void area_remove_from_moved_list(SelfList<Area2DSW> *p_area);
	const SelfList<Area2DSW>::List &get_moved_area_list() const;

	void body_add_to_state_query_list(SelfList<Body2DSW> *p_body);
	void body_remove_from_state_query_list(SelfList<Body2DSW> *p_body);

	void area_add_to_monitor_query_list(SelfList<Area2DSW> *p_area);
	void area_remove_from_monitor_query_list(SelfList<Area2DSW> *p_area);

	BroadPhase2DSW *get_broadphase();

	void add_object(CollisionObject2DSW *p_object);
	void remove_object(CollisionObject2DSW *p_object);
	const Set<CollisionObject2DSW *> &get_objects() const;

	_FORCE_INLINE_ real_t get_contact_recycle_radius() const { return contact_recycle_radius; }
	_FORCE_INLINE_ real_t get_contact_max_separation() const { return contact_max_separation; }
	_FORCE_INLINE_ real_t get_contact_max_allowed_penetration() const { return contact_max_allowed_penetration; }
	_FORCE_INLINE_ real_t get_constraint_bias() const { return constraint_bias; }
	_FORCE_INLINE_ real_t get_body_linear_velocity_sleep_threshold() const { return body_linear_velocity_sleep_threshold; }
	_FORCE_INLINE_ real_t get_body_angular_velocity_sleep_threshold() const { return body_angular_velocity_sleep_threshold; }
	_FORCE_INLINE_ real_t get_body_time_to_sleep() const { return body_time_to_sleep; }

	void update();
	void setup();
	void call_queries();

	bool is_locked() const;
	void lock();
	void unlock();

	void set_param(Physics2DServer::SpaceParameter p_param, real_t p_value);
	real_t get_param(Physics2DServer::SpaceParameter p_param) const;

	void set_island_count(int p_island_count) { island_count = p_island_count; }
	int get_island_count() const { return island_count; }

	void set_active_objects(int p_active_objects) { active_objects = p_active_objects; }
	int get_active_objects() const { return active_objects; }

	int get_collision_pairs() const { return collision_pairs; }

	bool test_body_motion(Body2DSW *p_body, const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia, real_t p_margin, Physics2DServer::MotionResult *r_result, bool p_exclude_raycast_shapes = true, const Set<RID> &p_exclude = Set<RID>());
	int test_body_ray_separation(Body2DSW *p_body, const Transform2D &p_transform, bool p_infinite_inertia, Vector2 &r_recover_motion, Physics2DServer::SeparationResult *r_results, int p_result_max, real_t p_margin);

	void set_debug_contacts(int p_amount) { contact_debug.resize(p_amount); }
	_FORCE_INLINE_ bool is_debugging_contacts() const { return !contact_debug.empty(); }
	_FORCE_INLINE_ void add_debug_contact(const Vector2 &p_contact) {
		if (contact_debug_count < contact_debug.size()) {
			contact_debug.write[contact_debug_count++] = p_contact;
		}
	}
	_FORCE_INLINE_ Vector<Vector2> get_debug_contacts() { return contact_debug; }
	_FORCE_INLINE_ int get_debug_contact_count() { return contact_debug_count; }

	Physics2DDirectSpaceStateSW *get_direct_state();

	void set_elapsed_time(ElapsedTime p_time, uint64_t p_msec) { elapsed_time[p_time] = p_msec; }
	uint64_t get_elapsed_time(ElapsedTime p_time) const { return elapsed_time[p_time]; }

	Space2DSW();
	~Space2DSW();
};

#endif // SPACE_2D_SW_H
