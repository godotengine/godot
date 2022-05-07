/*************************************************************************/
/*  physics_2d_server_sw.h                                               */
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

#ifndef PHYSICS_2D_SERVER_SW
#define PHYSICS_2D_SERVER_SW

#include "joints_2d_sw.h"
#include "servers/physics_2d_server.h"
#include "shape_2d_sw.h"
#include "space_2d_sw.h"
#include "step_2d_sw.h"

class Physics2DServerSW : public Physics2DServer {
	GDCLASS(Physics2DServerSW, Physics2DServer);

	friend class Physics2DDirectSpaceStateSW;
	friend class Physics2DDirectBodyStateSW;
	bool active;
	int iterations;
	bool doing_sync;

	int island_count;
	int active_objects;
	int collision_pairs;

	bool using_threads;

	bool flushing_queries;

	Step2DSW *stepper;
	Set<const Space2DSW *> active_spaces;

	mutable RID_Owner<Shape2DSW> shape_owner;
	mutable RID_Owner<Space2DSW> space_owner;
	mutable RID_Owner<Area2DSW> area_owner;
	mutable RID_Owner<Body2DSW> body_owner;
	mutable RID_Owner<Joint2DSW> joint_owner;

	static Physics2DServerSW *singletonsw;

	//void _clear_query(Query2DSW *p_query);
	friend class CollisionObject2DSW;
	SelfList<CollisionObject2DSW>::List pending_shape_update_list;
	void _update_shapes();

	RID _shape_create(ShapeType p_shape);

public:
	struct CollCbkData {
		Vector2 valid_dir;
		real_t valid_depth;
		int max;
		int amount;
		int passed;
		int invalid_by_dir;
		Vector2 *ptr;
	};

	virtual RID line_shape_create();
	virtual RID ray_shape_create();
	virtual RID segment_shape_create();
	virtual RID circle_shape_create();
	virtual RID rectangle_shape_create();
	virtual RID capsule_shape_create();
	virtual RID convex_polygon_shape_create();
	virtual RID concave_polygon_shape_create();

	static void _shape_col_cbk(const Vector2 &p_point_A, const Vector2 &p_point_B, void *p_userdata);

	virtual void shape_set_data(RID p_shape, const Variant &p_data);
	virtual void shape_set_custom_solver_bias(RID p_shape, real_t p_bias);

	virtual ShapeType shape_get_type(RID p_shape) const;
	virtual Variant shape_get_data(RID p_shape) const;
	virtual real_t shape_get_custom_solver_bias(RID p_shape) const;

	virtual bool shape_collide(RID p_shape_A, const Transform2D &p_xform_A, const Vector2 &p_motion_A, RID p_shape_B, const Transform2D &p_xform_B, const Vector2 &p_motion_B, Vector2 *r_results, int p_result_max, int &r_result_count);

	/* SPACE API */

	virtual RID space_create();
	virtual void space_set_active(RID p_space, bool p_active);
	virtual bool space_is_active(RID p_space) const;

	virtual void space_set_param(RID p_space, SpaceParameter p_param, real_t p_value);
	virtual real_t space_get_param(RID p_space, SpaceParameter p_param) const;

	virtual void space_set_debug_contacts(RID p_space, int p_max_contacts);
	virtual Vector<Vector2> space_get_contacts(RID p_space) const;
	virtual int space_get_contact_count(RID p_space) const;

	// this function only works on physics process, errors and returns null otherwise
	virtual Physics2DDirectSpaceState *space_get_direct_state(RID p_space);

	/* AREA API */

	virtual RID area_create();

	virtual void area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode);
	virtual AreaSpaceOverrideMode area_get_space_override_mode(RID p_area) const;

	virtual void area_set_space(RID p_area, RID p_space);
	virtual RID area_get_space(RID p_area) const;

	virtual void area_add_shape(RID p_area, RID p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false);
	virtual void area_set_shape(RID p_area, int p_shape_idx, RID p_shape);
	virtual void area_set_shape_transform(RID p_area, int p_shape_idx, const Transform2D &p_transform);

	virtual int area_get_shape_count(RID p_area) const;
	virtual RID area_get_shape(RID p_area, int p_shape_idx) const;
	virtual Transform2D area_get_shape_transform(RID p_area, int p_shape_idx) const;

	virtual void area_set_shape_disabled(RID p_area, int p_shape, bool p_disabled);

	virtual void area_remove_shape(RID p_area, int p_shape_idx);
	virtual void area_clear_shapes(RID p_area);

	virtual void area_attach_object_instance_id(RID p_area, ObjectID p_id);
	virtual ObjectID area_get_object_instance_id(RID p_area) const;

	virtual void area_attach_canvas_instance_id(RID p_area, ObjectID p_id);
	virtual ObjectID area_get_canvas_instance_id(RID p_area) const;

	virtual void area_set_param(RID p_area, AreaParameter p_param, const Variant &p_value);
	virtual void area_set_transform(RID p_area, const Transform2D &p_transform);

	virtual Variant area_get_param(RID p_area, AreaParameter p_param) const;
	virtual Transform2D area_get_transform(RID p_area) const;
	virtual void area_set_monitorable(RID p_area, bool p_monitorable);
	virtual void area_set_collision_mask(RID p_area, uint32_t p_mask);
	virtual void area_set_collision_layer(RID p_area, uint32_t p_layer);

	virtual void area_set_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method);
	virtual void area_set_area_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method);

	virtual void area_set_pickable(RID p_area, bool p_pickable);

	/* BODY API */

	// create a body of a given type
	virtual RID body_create();

	virtual void body_set_space(RID p_body, RID p_space);
	virtual RID body_get_space(RID p_body) const;

	virtual void body_set_mode(RID p_body, BodyMode p_mode);
	virtual BodyMode body_get_mode(RID p_body) const;

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false);
	virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape);
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform2D &p_transform);
	virtual void body_set_shape_metadata(RID p_body, int p_shape_idx, const Variant &p_metadata);

	virtual int body_get_shape_count(RID p_body) const;
	virtual RID body_get_shape(RID p_body, int p_shape_idx) const;
	virtual Transform2D body_get_shape_transform(RID p_body, int p_shape_idx) const;
	virtual Variant body_get_shape_metadata(RID p_body, int p_shape_idx) const;

	virtual void body_remove_shape(RID p_body, int p_shape_idx);
	virtual void body_clear_shapes(RID p_body);

	virtual void body_set_shape_disabled(RID p_body, int p_shape_idx, bool p_disabled);
	virtual void body_set_shape_as_one_way_collision(RID p_body, int p_shape_idx, bool p_enable, float p_margin);

	virtual void body_attach_object_instance_id(RID p_body, uint32_t p_id);
	virtual uint32_t body_get_object_instance_id(RID p_body) const;

	virtual void body_attach_canvas_instance_id(RID p_body, uint32_t p_id);
	virtual uint32_t body_get_canvas_instance_id(RID p_body) const;

	virtual void body_set_continuous_collision_detection_mode(RID p_body, CCDMode p_mode);
	virtual CCDMode body_get_continuous_collision_detection_mode(RID p_body) const;

	virtual void body_set_collision_layer(RID p_body, uint32_t p_layer);
	virtual uint32_t body_get_collision_layer(RID p_body) const;

	virtual void body_set_collision_mask(RID p_body, uint32_t p_mask);
	virtual uint32_t body_get_collision_mask(RID p_body) const;

	virtual void body_set_param(RID p_body, BodyParameter p_param, real_t p_value);
	virtual real_t body_get_param(RID p_body, BodyParameter p_param) const;

	virtual void body_set_state(RID p_body, BodyState p_state, const Variant &p_variant);
	virtual Variant body_get_state(RID p_body, BodyState p_state) const;

	virtual void body_set_applied_force(RID p_body, const Vector2 &p_force);
	virtual Vector2 body_get_applied_force(RID p_body) const;

	virtual void body_set_applied_torque(RID p_body, real_t p_torque);
	virtual real_t body_get_applied_torque(RID p_body) const;

	virtual void body_add_central_force(RID p_body, const Vector2 &p_force);
	virtual void body_add_force(RID p_body, const Vector2 &p_offset, const Vector2 &p_force);
	virtual void body_add_torque(RID p_body, real_t p_torque);

	virtual void body_apply_central_impulse(RID p_body, const Vector2 &p_impulse);
	virtual void body_apply_torque_impulse(RID p_body, real_t p_torque);
	virtual void body_apply_impulse(RID p_body, const Vector2 &p_pos, const Vector2 &p_impulse);
	virtual void body_set_axis_velocity(RID p_body, const Vector2 &p_axis_velocity);

	virtual void body_add_collision_exception(RID p_body, RID p_body_b);
	virtual void body_remove_collision_exception(RID p_body, RID p_body_b);
	virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions);

	virtual void body_set_contacts_reported_depth_threshold(RID p_body, real_t p_threshold);
	virtual real_t body_get_contacts_reported_depth_threshold(RID p_body) const;

	virtual void body_set_omit_force_integration(RID p_body, bool p_omit);
	virtual bool body_is_omitting_force_integration(RID p_body) const;

	virtual void body_set_max_contacts_reported(RID p_body, int p_contacts);
	virtual int body_get_max_contacts_reported(RID p_body) const;

	virtual void body_set_force_integration_callback(RID p_body, Object *p_receiver, const StringName &p_method, const Variant &p_udata = Variant());
	virtual bool body_collide_shape(RID p_body, int p_body_shape, RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, Vector2 *r_results, int p_result_max, int &r_result_count);

	virtual void body_set_pickable(RID p_body, bool p_pickable);

	virtual bool body_test_motion(RID p_body, const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia, real_t p_margin = 0.08, MotionResult *r_result = nullptr, bool p_exclude_raycast_shapes = true, const Set<RID> &p_exclude = Set<RID>());
	virtual int body_test_ray_separation(RID p_body, const Transform2D &p_transform, bool p_infinite_inertia, Vector2 &r_recover_motion, SeparationResult *r_results, int p_result_max, float p_margin = 0.08);

	// this function only works on physics process, errors and returns null otherwise
	virtual Physics2DDirectBodyState *body_get_direct_state(RID p_body);

	/* JOINT API */

	virtual void joint_set_param(RID p_joint, JointParam p_param, real_t p_value);
	virtual real_t joint_get_param(RID p_joint, JointParam p_param) const;

	virtual void joint_disable_collisions_between_bodies(RID p_joint, const bool p_disabled);
	virtual bool joint_is_disabled_collisions_between_bodies(RID p_joint) const;

	virtual RID pin_joint_create(const Vector2 &p_pos, RID p_body_a, RID p_body_b = RID());
	virtual RID groove_joint_create(const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, RID p_body_a, RID p_body_b);
	virtual RID damped_spring_joint_create(const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, RID p_body_a, RID p_body_b = RID());
	virtual void pin_joint_set_param(RID p_joint, PinJointParam p_param, real_t p_value);
	virtual real_t pin_joint_get_param(RID p_joint, PinJointParam p_param) const;
	virtual void damped_string_joint_set_param(RID p_joint, DampedStringParam p_param, real_t p_value);
	virtual real_t damped_string_joint_get_param(RID p_joint, DampedStringParam p_param) const;

	virtual JointType joint_get_type(RID p_joint) const;

	/* MISC */

	virtual void free(RID p_rid);

	virtual void set_active(bool p_active);
	virtual void init();
	virtual void step(real_t p_step);
	virtual void sync();
	virtual void flush_queries();
	virtual void end_sync();
	virtual void finish();

	virtual void set_collision_iterations(int p_iterations);

	virtual bool is_flushing_queries() const { return flushing_queries; }

	int get_process_info(ProcessInfo p_info);

	Physics2DServerSW();
	~Physics2DServerSW();
};

#endif
