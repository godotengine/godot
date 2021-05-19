/*************************************************************************/
/*  physics_server_2d_sw.h                                               */
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

#ifndef PHYSICS_2D_SERVER_SW
#define PHYSICS_2D_SERVER_SW

#include "core/templates/rid_owner.h"
#include "joints_2d_sw.h"
#include "servers/physics_server_2d.h"
#include "shape_2d_sw.h"
#include "space_2d_sw.h"
#include "step_2d_sw.h"

class PhysicsServer2DSW : public PhysicsServer2D {
	GDCLASS(PhysicsServer2DSW, PhysicsServer2D);

	friend class PhysicsDirectSpaceState2DSW;
	friend class PhysicsDirectBodyState2DSW;
	bool active;
	int iterations;
	bool doing_sync;
	real_t last_step;

	int island_count;
	int active_objects;
	int collision_pairs;

	bool using_threads;

	bool flushing_queries;

	Step2DSW *stepper;
	Set<const Space2DSW *> active_spaces;

	PhysicsDirectBodyState2DSW *direct_state;

	mutable RID_PtrOwner<Shape2DSW, true> shape_owner;
	mutable RID_PtrOwner<Space2DSW, true> space_owner;
	mutable RID_PtrOwner<Area2DSW, true> area_owner;
	mutable RID_PtrOwner<Body2DSW, true> body_owner;
	mutable RID_PtrOwner<Joint2DSW, true> joint_owner;

	static PhysicsServer2DSW *singletonsw;

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

	virtual RID line_shape_create() override;
	virtual RID ray_shape_create() override;
	virtual RID segment_shape_create() override;
	virtual RID circle_shape_create() override;
	virtual RID rectangle_shape_create() override;
	virtual RID capsule_shape_create() override;
	virtual RID convex_polygon_shape_create() override;
	virtual RID concave_polygon_shape_create() override;

	static void _shape_col_cbk(const Vector2 &p_point_A, const Vector2 &p_point_B, void *p_userdata);

	virtual void shape_set_data(RID p_shape, const Variant &p_data) override;
	virtual void shape_set_custom_solver_bias(RID p_shape, real_t p_bias) override;

	virtual ShapeType shape_get_type(RID p_shape) const override;
	virtual Variant shape_get_data(RID p_shape) const override;
	virtual real_t shape_get_custom_solver_bias(RID p_shape) const override;

	virtual bool shape_collide(RID p_shape_A, const Transform2D &p_xform_A, const Vector2 &p_motion_A, RID p_shape_B, const Transform2D &p_xform_B, const Vector2 &p_motion_B, Vector2 *r_results, int p_result_max, int &r_result_count) override;

	/* SPACE API */

	virtual RID space_create() override;
	virtual void space_set_active(RID p_space, bool p_active) override;
	virtual bool space_is_active(RID p_space) const override;

	virtual void space_set_param(RID p_space, SpaceParameter p_param, real_t p_value) override;
	virtual real_t space_get_param(RID p_space, SpaceParameter p_param) const override;

	virtual void space_set_debug_contacts(RID p_space, int p_max_contacts) override;
	virtual Vector<Vector2> space_get_contacts(RID p_space) const override;
	virtual int space_get_contact_count(RID p_space) const override;

	// this function only works on physics process, errors and returns null otherwise
	virtual PhysicsDirectSpaceState2D *space_get_direct_state(RID p_space) override;

	/* AREA API */

	virtual RID area_create() override;

	virtual void area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode) override;
	virtual AreaSpaceOverrideMode area_get_space_override_mode(RID p_area) const override;

	virtual void area_set_space(RID p_area, RID p_space) override;
	virtual RID area_get_space(RID p_area) const override;

	virtual void area_add_shape(RID p_area, RID p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false) override;
	virtual void area_set_shape(RID p_area, int p_shape_idx, RID p_shape) override;
	virtual void area_set_shape_transform(RID p_area, int p_shape_idx, const Transform2D &p_transform) override;

	virtual int area_get_shape_count(RID p_area) const override;
	virtual RID area_get_shape(RID p_area, int p_shape_idx) const override;
	virtual Transform2D area_get_shape_transform(RID p_area, int p_shape_idx) const override;

	virtual void area_set_shape_disabled(RID p_area, int p_shape, bool p_disabled) override;

	virtual void area_remove_shape(RID p_area, int p_shape_idx) override;
	virtual void area_clear_shapes(RID p_area) override;

	virtual void area_attach_object_instance_id(RID p_area, ObjectID p_id) override;
	virtual ObjectID area_get_object_instance_id(RID p_area) const override;

	virtual void area_attach_canvas_instance_id(RID p_area, ObjectID p_id) override;
	virtual ObjectID area_get_canvas_instance_id(RID p_area) const override;

	virtual void area_set_param(RID p_area, AreaParameter p_param, const Variant &p_value) override;
	virtual void area_set_transform(RID p_area, const Transform2D &p_transform) override;

	virtual Variant area_get_param(RID p_area, AreaParameter p_param) const override;
	virtual Transform2D area_get_transform(RID p_area) const override;
	virtual void area_set_monitorable(RID p_area, bool p_monitorable) override;
	virtual void area_set_collision_mask(RID p_area, uint32_t p_mask) override;
	virtual void area_set_collision_layer(RID p_area, uint32_t p_layer) override;

	virtual void area_set_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) override;
	virtual void area_set_area_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) override;

	virtual void area_set_pickable(RID p_area, bool p_pickable) override;

	/* BODY API */

	// create a body of a given type
	virtual RID body_create() override;

	virtual void body_set_space(RID p_body, RID p_space) override;
	virtual RID body_get_space(RID p_body) const override;

	virtual void body_set_mode(RID p_body, BodyMode p_mode) override;
	virtual BodyMode body_get_mode(RID p_body) const override;

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false) override;
	virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape) override;
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform2D &p_transform) override;
	virtual void body_set_shape_metadata(RID p_body, int p_shape_idx, const Variant &p_metadata) override;

	virtual int body_get_shape_count(RID p_body) const override;
	virtual RID body_get_shape(RID p_body, int p_shape_idx) const override;
	virtual Transform2D body_get_shape_transform(RID p_body, int p_shape_idx) const override;
	virtual Variant body_get_shape_metadata(RID p_body, int p_shape_idx) const override;

	virtual void body_remove_shape(RID p_body, int p_shape_idx) override;
	virtual void body_clear_shapes(RID p_body) override;

	virtual void body_set_shape_disabled(RID p_body, int p_shape_idx, bool p_disabled) override;
	virtual void body_set_shape_as_one_way_collision(RID p_body, int p_shape_idx, bool p_enable, real_t p_margin) override;

	virtual void body_attach_object_instance_id(RID p_body, ObjectID p_id) override;
	virtual ObjectID body_get_object_instance_id(RID p_body) const override;

	virtual void body_attach_canvas_instance_id(RID p_body, ObjectID p_id) override;
	virtual ObjectID body_get_canvas_instance_id(RID p_body) const override;

	virtual void body_set_continuous_collision_detection_mode(RID p_body, CCDMode p_mode) override;
	virtual CCDMode body_get_continuous_collision_detection_mode(RID p_body) const override;

	virtual void body_set_collision_layer(RID p_body, uint32_t p_layer) override;
	virtual uint32_t body_get_collision_layer(RID p_body) const override;

	virtual void body_set_collision_mask(RID p_body, uint32_t p_mask) override;
	virtual uint32_t body_get_collision_mask(RID p_body) const override;

	virtual void body_set_param(RID p_body, BodyParameter p_param, real_t p_value) override;
	virtual real_t body_get_param(RID p_body, BodyParameter p_param) const override;

	virtual void body_set_state(RID p_body, BodyState p_state, const Variant &p_variant) override;
	virtual Variant body_get_state(RID p_body, BodyState p_state) const override;

	virtual void body_set_applied_force(RID p_body, const Vector2 &p_force) override;
	virtual Vector2 body_get_applied_force(RID p_body) const override;

	virtual void body_set_applied_torque(RID p_body, real_t p_torque) override;
	virtual real_t body_get_applied_torque(RID p_body) const override;

	virtual void body_add_central_force(RID p_body, const Vector2 &p_force) override;
	virtual void body_add_force(RID p_body, const Vector2 &p_force, const Vector2 &p_position = Vector2()) override;
	virtual void body_add_torque(RID p_body, real_t p_torque) override;

	virtual void body_apply_central_impulse(RID p_body, const Vector2 &p_impulse) override;
	virtual void body_apply_torque_impulse(RID p_body, real_t p_torque) override;
	virtual void body_apply_impulse(RID p_body, const Vector2 &p_impulse, const Vector2 &p_position = Vector2()) override;
	virtual void body_set_axis_velocity(RID p_body, const Vector2 &p_axis_velocity) override;

	virtual void body_add_collision_exception(RID p_body, RID p_body_b) override;
	virtual void body_remove_collision_exception(RID p_body, RID p_body_b) override;
	virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) override;

	virtual void body_set_contacts_reported_depth_threshold(RID p_body, real_t p_threshold) override;
	virtual real_t body_get_contacts_reported_depth_threshold(RID p_body) const override;

	virtual void body_set_omit_force_integration(RID p_body, bool p_omit) override;
	virtual bool body_is_omitting_force_integration(RID p_body) const override;

	virtual void body_set_max_contacts_reported(RID p_body, int p_contacts) override;
	virtual int body_get_max_contacts_reported(RID p_body) const override;

	virtual void body_set_force_integration_callback(RID p_body, const Callable &p_callable, const Variant &p_udata = Variant()) override;
	virtual bool body_collide_shape(RID p_body, int p_body_shape, RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, Vector2 *r_results, int p_result_max, int &r_result_count) override;

	virtual void body_set_pickable(RID p_body, bool p_pickable) override;

	virtual bool body_test_motion(RID p_body, const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia, real_t p_margin = 0.001, MotionResult *r_result = nullptr, bool p_exclude_raycast_shapes = true) override;
	virtual int body_test_ray_separation(RID p_body, const Transform2D &p_transform, bool p_infinite_inertia, Vector2 &r_recover_motion, SeparationResult *r_results, int p_result_max, real_t p_margin = 0.001) override;

	// this function only works on physics process, errors and returns null otherwise
	virtual PhysicsDirectBodyState2D *body_get_direct_state(RID p_body) override;

	/* JOINT API */

	virtual RID joint_create() override;

	virtual void joint_clear(RID p_joint) override;

	virtual void joint_set_param(RID p_joint, JointParam p_param, real_t p_value) override;
	virtual real_t joint_get_param(RID p_joint, JointParam p_param) const override;

	virtual void joint_disable_collisions_between_bodies(RID p_joint, const bool p_disabled) override;
	virtual bool joint_is_disabled_collisions_between_bodies(RID p_joint) const override;

	virtual void joint_make_pin(RID p_joint, const Vector2 &p_anchor, RID p_body_a, RID p_body_b = RID()) override;
	virtual void joint_make_groove(RID p_joint, const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, RID p_body_a, RID p_body_b) override;
	virtual void joint_make_damped_spring(RID p_joint, const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, RID p_body_a, RID p_body_b = RID()) override;

	virtual void pin_joint_set_param(RID p_joint, PinJointParam p_param, real_t p_value) override;
	virtual real_t pin_joint_get_param(RID p_joint, PinJointParam p_param) const override;
	virtual void damped_spring_joint_set_param(RID p_joint, DampedSpringParam p_param, real_t p_value) override;
	virtual real_t damped_spring_joint_get_param(RID p_joint, DampedSpringParam p_param) const override;

	virtual JointType joint_get_type(RID p_joint) const override;

	/* MISC */

	virtual void free(RID p_rid) override;

	virtual void set_active(bool p_active) override;
	virtual void init() override;
	virtual void step(real_t p_step) override;
	virtual void sync() override;
	virtual void flush_queries() override;
	virtual void end_sync() override;
	virtual void finish() override;

	virtual void set_collision_iterations(int p_iterations) override;

	virtual bool is_flushing_queries() const override { return flushing_queries; }

	int get_process_info(ProcessInfo p_info) override;

	PhysicsServer2DSW(bool p_using_threads = false);
	~PhysicsServer2DSW() {}
};

#endif
