/*************************************************************************/
/*  physics_server_sw.h                                                  */
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

#ifndef PHYSICS_SERVER_SW_H
#define PHYSICS_SERVER_SW_H

#include "joints_sw.h"
#include "servers/physics_server.h"
#include "shape_sw.h"
#include "space_sw.h"
#include "step_sw.h"

class PhysicsServerSW : public PhysicsServer {
	GDCLASS(PhysicsServerSW, PhysicsServer);

	friend class PhysicsDirectSpaceStateSW;
	bool active;
	int iterations;

	int island_count;
	int active_objects;
	int collision_pairs;

	bool flushing_queries;

	StepSW *stepper;
	Set<const SpaceSW *> active_spaces;

	mutable RID_Owner<ShapeSW> shape_owner;
	mutable RID_Owner<SpaceSW> space_owner;
	mutable RID_Owner<AreaSW> area_owner;
	mutable RID_Owner<BodySW> body_owner;
	mutable RID_Owner<JointSW> joint_owner;

	//void _clear_query(QuerySW *p_query);
	friend class CollisionObjectSW;
	SelfList<CollisionObjectSW>::List pending_shape_update_list;
	void _update_shapes();

public:
	static PhysicsServerSW *singleton;

	struct CollCbkData {
		int max;
		int amount;
		Vector3 *ptr;
	};

	static void _shape_col_cbk(const Vector3 &p_point_A, const Vector3 &p_point_B, void *p_userdata);

	virtual RID shape_create(ShapeType p_shape);
	virtual void shape_set_data(RID p_shape, const Variant &p_data);
	virtual void shape_set_custom_solver_bias(RID p_shape, real_t p_bias);

	virtual ShapeType shape_get_type(RID p_shape) const;
	virtual Variant shape_get_data(RID p_shape) const;

	virtual void shape_set_margin(RID p_shape, real_t p_margin);
	virtual real_t shape_get_margin(RID p_shape) const;

	virtual real_t shape_get_custom_solver_bias(RID p_shape) const;

	/* SPACE API */

	virtual RID space_create();
	virtual void space_set_active(RID p_space, bool p_active);
	virtual bool space_is_active(RID p_space) const;

	virtual void space_set_param(RID p_space, SpaceParameter p_param, real_t p_value);
	virtual real_t space_get_param(RID p_space, SpaceParameter p_param) const;

	// this function only works on physics process, errors and returns null otherwise
	virtual PhysicsDirectSpaceState *space_get_direct_state(RID p_space);

	virtual void space_set_debug_contacts(RID p_space, int p_max_contacts);
	virtual Vector<Vector3> space_get_contacts(RID p_space) const;
	virtual int space_get_contact_count(RID p_space) const;

	/* AREA API */

	virtual RID area_create();

	virtual void area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode);
	virtual AreaSpaceOverrideMode area_get_space_override_mode(RID p_area) const;

	virtual void area_set_space(RID p_area, RID p_space);
	virtual RID area_get_space(RID p_area) const;

	virtual void area_add_shape(RID p_area, RID p_shape, const Transform &p_transform = Transform(), bool p_disabled = false);
	virtual void area_set_shape(RID p_area, int p_shape_idx, RID p_shape);
	virtual void area_set_shape_transform(RID p_area, int p_shape_idx, const Transform &p_transform);

	virtual int area_get_shape_count(RID p_area) const;
	virtual RID area_get_shape(RID p_area, int p_shape_idx) const;
	virtual Transform area_get_shape_transform(RID p_area, int p_shape_idx) const;

	virtual void area_remove_shape(RID p_area, int p_shape_idx);
	virtual void area_clear_shapes(RID p_area);

	virtual void area_set_shape_disabled(RID p_area, int p_shape_idx, bool p_disabled);

	virtual void area_attach_object_instance_id(RID p_area, ObjectID p_id);
	virtual ObjectID area_get_object_instance_id(RID p_area) const;

	virtual void area_set_param(RID p_area, AreaParameter p_param, const Variant &p_value);
	virtual void area_set_transform(RID p_area, const Transform &p_transform);

	virtual Variant area_get_param(RID p_area, AreaParameter p_param) const;
	virtual Transform area_get_transform(RID p_area) const;

	virtual void area_set_ray_pickable(RID p_area, bool p_enable);
	virtual bool area_is_ray_pickable(RID p_area) const;

	virtual void area_set_collision_mask(RID p_area, uint32_t p_mask);
	virtual void area_set_collision_layer(RID p_area, uint32_t p_layer);

	virtual void area_set_monitorable(RID p_area, bool p_monitorable);

	virtual void area_set_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method);
	virtual void area_set_area_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method);

	/* BODY API */

	// create a body of a given type
	virtual RID body_create(BodyMode p_mode = BODY_MODE_RIGID, bool p_init_sleeping = false);

	virtual void body_set_space(RID p_body, RID p_space);
	virtual RID body_get_space(RID p_body) const;

	virtual void body_set_mode(RID p_body, BodyMode p_mode);
	virtual BodyMode body_get_mode(RID p_body) const;

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform &p_transform = Transform(), bool p_disabled = false);
	virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape);
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform &p_transform);

	virtual int body_get_shape_count(RID p_body) const;
	virtual RID body_get_shape(RID p_body, int p_shape_idx) const;
	virtual Transform body_get_shape_transform(RID p_body, int p_shape_idx) const;

	virtual void body_set_shape_disabled(RID p_body, int p_shape_idx, bool p_disabled);

	virtual void body_remove_shape(RID p_body, int p_shape_idx);
	virtual void body_clear_shapes(RID p_body);

	virtual void body_attach_object_instance_id(RID p_body, uint32_t p_id);
	virtual uint32_t body_get_object_instance_id(RID p_body) const;

	virtual void body_set_enable_continuous_collision_detection(RID p_body, bool p_enable);
	virtual bool body_is_continuous_collision_detection_enabled(RID p_body) const;

	virtual void body_set_collision_layer(RID p_body, uint32_t p_layer);
	virtual uint32_t body_get_collision_layer(RID p_body) const;

	virtual void body_set_collision_mask(RID p_body, uint32_t p_mask);
	virtual uint32_t body_get_collision_mask(RID p_body) const;

	virtual void body_set_user_flags(RID p_body, uint32_t p_flags);
	virtual uint32_t body_get_user_flags(RID p_body) const;

	virtual void body_set_param(RID p_body, BodyParameter p_param, real_t p_value);
	virtual real_t body_get_param(RID p_body, BodyParameter p_param) const;

	virtual void body_set_kinematic_safe_margin(RID p_body, real_t p_margin);
	virtual real_t body_get_kinematic_safe_margin(RID p_body) const;

	virtual void body_set_state(RID p_body, BodyState p_state, const Variant &p_variant);
	virtual Variant body_get_state(RID p_body, BodyState p_state) const;

	virtual void body_set_applied_force(RID p_body, const Vector3 &p_force);
	virtual Vector3 body_get_applied_force(RID p_body) const;

	virtual void body_set_applied_torque(RID p_body, const Vector3 &p_torque);
	virtual Vector3 body_get_applied_torque(RID p_body) const;

	virtual void body_add_central_force(RID p_body, const Vector3 &p_force);
	virtual void body_add_force(RID p_body, const Vector3 &p_force, const Vector3 &p_pos);
	virtual void body_add_torque(RID p_body, const Vector3 &p_torque);

	virtual void body_apply_central_impulse(RID p_body, const Vector3 &p_impulse);
	virtual void body_apply_impulse(RID p_body, const Vector3 &p_pos, const Vector3 &p_impulse);
	virtual void body_apply_torque_impulse(RID p_body, const Vector3 &p_impulse);
	virtual void body_set_axis_velocity(RID p_body, const Vector3 &p_axis_velocity);

	virtual void body_set_axis_lock(RID p_body, BodyAxis p_axis, bool p_lock);
	virtual bool body_is_axis_locked(RID p_body, BodyAxis p_axis) const;

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

	virtual void body_set_ray_pickable(RID p_body, bool p_enable);
	virtual bool body_is_ray_pickable(RID p_body) const;

	virtual bool body_test_motion(RID p_body, const Transform &p_from, const Vector3 &p_motion, bool p_infinite_inertia, MotionResult *r_result = nullptr, bool p_exclude_raycast_shapes = true, const Set<RID> &p_exclude = Set<RID>());
	virtual int body_test_ray_separation(RID p_body, const Transform &p_transform, bool p_infinite_inertia, Vector3 &r_recover_motion, SeparationResult *r_results, int p_result_max, float p_margin = 0.001);

	// this function only works on physics process, errors and returns null otherwise
	virtual PhysicsDirectBodyState *body_get_direct_state(RID p_body);

	/* SOFT BODY */

	virtual RID soft_body_create(bool p_init_sleeping = false) { return RID(); }

	virtual void soft_body_update_visual_server(RID p_body, class SoftBodyVisualServerHandler *p_visual_server_handler) {}

	virtual void soft_body_set_space(RID p_body, RID p_space) {}
	virtual RID soft_body_get_space(RID p_body) const { return RID(); }

	virtual void soft_body_set_collision_layer(RID p_body, uint32_t p_layer) {}
	virtual uint32_t soft_body_get_collision_layer(RID p_body) const { return 0; }

	virtual void soft_body_set_collision_mask(RID p_body, uint32_t p_mask) {}
	virtual uint32_t soft_body_get_collision_mask(RID p_body) const { return 0; }

	virtual void soft_body_add_collision_exception(RID p_body, RID p_body_b) {}
	virtual void soft_body_remove_collision_exception(RID p_body, RID p_body_b) {}
	virtual void soft_body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) {}

	virtual void soft_body_set_state(RID p_body, BodyState p_state, const Variant &p_variant) {}
	virtual Variant soft_body_get_state(RID p_body, BodyState p_state) const { return Variant(); }

	virtual void soft_body_set_transform(RID p_body, const Transform &p_transform) {}
	virtual Vector3 soft_body_get_vertex_position(RID p_body, int vertex_index) const { return Vector3(); }

	virtual void soft_body_set_ray_pickable(RID p_body, bool p_enable) {}
	virtual bool soft_body_is_ray_pickable(RID p_body) const { return false; }

	virtual void soft_body_set_simulation_precision(RID p_body, int p_simulation_precision) {}
	virtual int soft_body_get_simulation_precision(RID p_body) { return 0; }

	virtual void soft_body_set_total_mass(RID p_body, real_t p_total_mass) {}
	virtual real_t soft_body_get_total_mass(RID p_body) { return 0.; }

	virtual void soft_body_set_linear_stiffness(RID p_body, real_t p_stiffness) {}
	virtual real_t soft_body_get_linear_stiffness(RID p_body) { return 0.; }

	virtual void soft_body_set_areaAngular_stiffness(RID p_body, real_t p_stiffness) {}
	virtual real_t soft_body_get_areaAngular_stiffness(RID p_body) { return 0.; }

	virtual void soft_body_set_volume_stiffness(RID p_body, real_t p_stiffness) {}
	virtual real_t soft_body_get_volume_stiffness(RID p_body) { return 0.; }

	virtual void soft_body_set_pressure_coefficient(RID p_body, real_t p_pressure_coefficient) {}
	virtual real_t soft_body_get_pressure_coefficient(RID p_body) { return 0.; }

	virtual void soft_body_set_pose_matching_coefficient(RID p_body, real_t p_pose_matching_coefficient) {}
	virtual real_t soft_body_get_pose_matching_coefficient(RID p_body) { return 0.; }

	virtual void soft_body_set_damping_coefficient(RID p_body, real_t p_damping_coefficient) {}
	virtual real_t soft_body_get_damping_coefficient(RID p_body) { return 0.; }

	virtual void soft_body_set_drag_coefficient(RID p_body, real_t p_drag_coefficient) {}
	virtual real_t soft_body_get_drag_coefficient(RID p_body) { return 0.; }

	virtual void soft_body_set_mesh(RID p_body, const REF &p_mesh) {}

	virtual void soft_body_move_point(RID p_body, int p_point_index, const Vector3 &p_global_position) {}
	virtual Vector3 soft_body_get_point_global_position(RID p_body, int p_point_index) { return Vector3(); }

	virtual Vector3 soft_body_get_point_offset(RID p_body, int p_point_index) const { return Vector3(); }

	virtual void soft_body_remove_all_pinned_points(RID p_body) {}
	virtual void soft_body_pin_point(RID p_body, int p_point_index, bool p_pin) {}
	virtual bool soft_body_is_point_pinned(RID p_body, int p_point_index) { return false; }

	/* JOINT API */

	virtual RID joint_create_pin(RID p_body_A, const Vector3 &p_local_A, RID p_body_B, const Vector3 &p_local_B);

	virtual void pin_joint_set_param(RID p_joint, PinJointParam p_param, real_t p_value);
	virtual real_t pin_joint_get_param(RID p_joint, PinJointParam p_param) const;

	virtual void pin_joint_set_local_a(RID p_joint, const Vector3 &p_A);
	virtual Vector3 pin_joint_get_local_a(RID p_joint) const;

	virtual void pin_joint_set_local_b(RID p_joint, const Vector3 &p_B);
	virtual Vector3 pin_joint_get_local_b(RID p_joint) const;

	virtual RID joint_create_hinge(RID p_body_A, const Transform &p_frame_A, RID p_body_B, const Transform &p_frame_B);
	virtual RID joint_create_hinge_simple(RID p_body_A, const Vector3 &p_pivot_A, const Vector3 &p_axis_A, RID p_body_B, const Vector3 &p_pivot_B, const Vector3 &p_axis_B);

	virtual void hinge_joint_set_param(RID p_joint, HingeJointParam p_param, real_t p_value);
	virtual real_t hinge_joint_get_param(RID p_joint, HingeJointParam p_param) const;

	virtual void hinge_joint_set_flag(RID p_joint, HingeJointFlag p_flag, bool p_value);
	virtual bool hinge_joint_get_flag(RID p_joint, HingeJointFlag p_flag) const;

	virtual RID joint_create_slider(RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B); //reference frame is A

	virtual void slider_joint_set_param(RID p_joint, SliderJointParam p_param, real_t p_value);
	virtual real_t slider_joint_get_param(RID p_joint, SliderJointParam p_param) const;

	virtual RID joint_create_cone_twist(RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B); //reference frame is A

	virtual void cone_twist_joint_set_param(RID p_joint, ConeTwistJointParam p_param, real_t p_value);
	virtual real_t cone_twist_joint_get_param(RID p_joint, ConeTwistJointParam p_param) const;

	virtual RID joint_create_generic_6dof(RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B); //reference frame is A

	virtual void generic_6dof_joint_set_param(RID p_joint, Vector3::Axis, G6DOFJointAxisParam p_param, real_t p_value);
	virtual real_t generic_6dof_joint_get_param(RID p_joint, Vector3::Axis, G6DOFJointAxisParam p_param);

	virtual void generic_6dof_joint_set_flag(RID p_joint, Vector3::Axis, G6DOFJointAxisFlag p_flag, bool p_enable);
	virtual bool generic_6dof_joint_get_flag(RID p_joint, Vector3::Axis, G6DOFJointAxisFlag p_flag);

	virtual JointType joint_get_type(RID p_joint) const;

	virtual void joint_set_solver_priority(RID p_joint, int p_priority);
	virtual int joint_get_solver_priority(RID p_joint) const;

	virtual void joint_disable_collisions_between_bodies(RID p_joint, const bool p_disable);
	virtual bool joint_is_disabled_collisions_between_bodies(RID p_joint) const;

	/* MISC */

	virtual void free(RID p_rid);

	virtual void set_active(bool p_active);
	virtual void init();
	virtual void step(real_t p_step);
	virtual void flush_queries();
	virtual void finish();

	virtual void set_collision_iterations(int p_iterations);

	virtual bool is_flushing_queries() const { return flushing_queries; }

	int get_process_info(ProcessInfo p_info);

	PhysicsServerSW();
	~PhysicsServerSW();
};

#endif // PHYSICS_SERVER_SW_H
