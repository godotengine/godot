/*************************************************************************/
/*  bullet_physics_server.h                                              */
/*  Author: AndreaCatania                                                */
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

#ifndef BULLET_PHYSICS_SERVER_H
#define BULLET_PHYSICS_SERVER_H

#include "area_bullet.h"
#include "joint_bullet.h"
#include "rid.h"
#include "rigid_body_bullet.h"
#include "servers/physics_server.h"
#include "shape_bullet.h"
#include "soft_body_bullet.h"
#include "space_bullet.h"

class BulletPhysicsServer : public PhysicsServer {
	GDCLASS(BulletPhysicsServer, PhysicsServer)

	friend class BulletPhysicsDirectSpaceState;

	bool active;
	char active_spaces_count;
	Vector<SpaceBullet *> active_spaces;

	mutable RID_Owner<SpaceBullet> space_owner;
	mutable RID_Owner<ShapeBullet> shape_owner;
	mutable RID_Owner<AreaBullet> area_owner;
	mutable RID_Owner<RigidBodyBullet> rigid_body_owner;
	mutable RID_Owner<SoftBodyBullet> soft_body_owner;
	mutable RID_Owner<JointBullet> joint_owner;

private:
	/// This is used when a collision shape is not active, so the bullet compound shapes index are always sync with godot index
	static btEmptyShape *emptyShape;

public:
	static btEmptyShape *get_empty_shape();

protected:
	static void _bind_methods();

public:
	BulletPhysicsServer();
	~BulletPhysicsServer();

	_FORCE_INLINE_ RID_Owner<SpaceBullet> *get_space_owner() {
		return &space_owner;
	}
	_FORCE_INLINE_ RID_Owner<ShapeBullet> *get_shape_owner() {
		return &shape_owner;
	}
	_FORCE_INLINE_ RID_Owner<AreaBullet> *get_area_owner() {
		return &area_owner;
	}
	_FORCE_INLINE_ RID_Owner<RigidBodyBullet> *get_rigid_body_owner() {
		return &rigid_body_owner;
	}
	_FORCE_INLINE_ RID_Owner<SoftBodyBullet> *get_soft_body_owner() {
		return &soft_body_owner;
	}
	_FORCE_INLINE_ RID_Owner<JointBullet> *get_joint_owner() {
		return &joint_owner;
	}

	/* SHAPE API */
	virtual RID shape_create(ShapeType p_shape);
	virtual void shape_set_data(RID p_shape, const Variant &p_data);
	virtual ShapeType shape_get_type(RID p_shape) const;
	virtual Variant shape_get_data(RID p_shape) const;

	/// Not supported
	virtual void shape_set_custom_solver_bias(RID p_shape, real_t p_bias);
	/// Not supported
	virtual real_t shape_get_custom_solver_bias(RID p_shape) const;

	/* SPACE API */

	virtual RID space_create();
	virtual void space_set_active(RID p_space, bool p_active);
	virtual bool space_is_active(RID p_space) const;

	/// Not supported
	virtual void space_set_param(RID p_space, SpaceParameter p_param, real_t p_value);
	/// Not supported
	virtual real_t space_get_param(RID p_space, SpaceParameter p_param) const;

	virtual PhysicsDirectSpaceState *space_get_direct_state(RID p_space);

	virtual void space_set_debug_contacts(RID p_space, int p_max_contacts);
	virtual Vector<Vector3> space_get_contacts(RID p_space) const;
	virtual int space_get_contact_count(RID p_space) const;

	/* AREA API */

	/// Bullet Physics Engine not support "Area", this must be handled by the game developer in another way.
	/// Since godot Physics use the concept of area even to define the main world, the API area_set_param is used to set initial physics world information.
	/// The API area_set_param is a bit hacky, and allow Godot to set some parameters on Bullet's world, a different use print a warning to console.
	/// All other APIs returns a warning message if used

	virtual RID area_create();

	virtual void area_set_space(RID p_area, RID p_space);

	virtual RID area_get_space(RID p_area) const;

	virtual void area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode);
	virtual AreaSpaceOverrideMode area_get_space_override_mode(RID p_area) const;

	virtual void area_add_shape(RID p_area, RID p_shape, const Transform &p_transform = Transform());
	virtual void area_set_shape(RID p_area, int p_shape_idx, RID p_shape);
	virtual void area_set_shape_transform(RID p_area, int p_shape_idx, const Transform &p_transform);
	virtual int area_get_shape_count(RID p_area) const;
	virtual RID area_get_shape(RID p_area, int p_shape_idx) const;
	virtual Transform area_get_shape_transform(RID p_area, int p_shape_idx) const;
	virtual void area_remove_shape(RID p_area, int p_shape_idx);
	virtual void area_clear_shapes(RID p_area);
	virtual void area_set_shape_disabled(RID p_area, int p_shape_idx, bool p_disabled);
	virtual void area_attach_object_instance_id(RID p_area, ObjectID p_ID);
	virtual ObjectID area_get_object_instance_id(RID p_area) const;

	/// If you pass as p_area the SpaceBullet you can set some parameters as specified below
	/// AREA_PARAM_GRAVITY
	/// AREA_PARAM_GRAVITY_VECTOR
	/// Otherwise you can set area parameters
	virtual void area_set_param(RID p_area, AreaParameter p_param, const Variant &p_value);
	virtual Variant area_get_param(RID p_parea, AreaParameter p_param) const;

	virtual void area_set_transform(RID p_area, const Transform &p_transform);
	virtual Transform area_get_transform(RID p_area) const;

	virtual void area_set_collision_mask(RID p_area, uint32_t p_mask);
	virtual void area_set_collision_layer(RID p_area, uint32_t p_layer);

	virtual void area_set_monitorable(RID p_area, bool p_monitorable);
	virtual void area_set_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method);
	virtual void area_set_area_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method);
	virtual void area_set_ray_pickable(RID p_area, bool p_enable);
	virtual bool area_is_ray_pickable(RID p_area) const;

	/* RIGID BODY API */

	virtual RID body_create(BodyMode p_mode = BODY_MODE_RIGID, bool p_init_sleeping = false);

	virtual void body_set_space(RID p_body, RID p_space);
	virtual RID body_get_space(RID p_body) const;

	virtual void body_set_mode(RID p_body, BodyMode p_mode);
	virtual BodyMode body_get_mode(RID p_body) const;

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform &p_transform = Transform());
	// Not supported, Please remove and add new shape
	virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape);
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform &p_transform);

	virtual int body_get_shape_count(RID p_body) const;
	virtual RID body_get_shape(RID p_body, int p_shape_idx) const;
	virtual Transform body_get_shape_transform(RID p_body, int p_shape_idx) const;

	virtual void body_set_shape_disabled(RID p_body, int p_shape_idx, bool p_disabled);

	virtual void body_remove_shape(RID p_body, int p_shape_idx);
	virtual void body_clear_shapes(RID p_body);

	// Used for Rigid and Soft Bodies
	virtual void body_attach_object_instance_id(RID p_body, uint32_t p_ID);
	virtual uint32_t body_get_object_instance_id(RID p_body) const;

	virtual void body_set_enable_continuous_collision_detection(RID p_body, bool p_enable);
	virtual bool body_is_continuous_collision_detection_enabled(RID p_body) const;

	virtual void body_set_collision_layer(RID p_body, uint32_t p_layer);
	virtual uint32_t body_get_collision_layer(RID p_body) const;

	virtual void body_set_collision_mask(RID p_body, uint32_t p_mask);
	virtual uint32_t body_get_collision_mask(RID p_body) const;

	/// This is not supported by physics server
	virtual void body_set_user_flags(RID p_body, uint32_t p_flags);
	/// This is not supported by physics server
	virtual uint32_t body_get_user_flags(RID p_body) const;

	virtual void body_set_param(RID p_body, BodyParameter p_param, float p_value);
	virtual float body_get_param(RID p_body, BodyParameter p_param) const;

	virtual void body_set_kinematic_safe_margin(RID p_body, real_t p_margin);
	virtual real_t body_get_kinematic_safe_margin(RID p_body) const;

	virtual void body_set_state(RID p_body, BodyState p_state, const Variant &p_variant);
	virtual Variant body_get_state(RID p_body, BodyState p_state) const;

	virtual void body_set_applied_force(RID p_body, const Vector3 &p_force);
	virtual Vector3 body_get_applied_force(RID p_body) const;

	virtual void body_set_applied_torque(RID p_body, const Vector3 &p_torque);
	virtual Vector3 body_get_applied_torque(RID p_body) const;

	virtual void body_apply_impulse(RID p_body, const Vector3 &p_pos, const Vector3 &p_impulse);
	virtual void body_apply_torque_impulse(RID p_body, const Vector3 &p_impulse);
	virtual void body_set_axis_velocity(RID p_body, const Vector3 &p_axis_velocity);

	virtual void body_set_axis_lock(RID p_body, BodyAxis p_axis, bool p_lock);
	virtual bool body_is_axis_locked(RID p_body, BodyAxis p_axis) const;

	virtual void body_add_collision_exception(RID p_body, RID p_body_b);
	virtual void body_remove_collision_exception(RID p_body, RID p_body_b);
	virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions);

	virtual void body_set_max_contacts_reported(RID p_body, int p_contacts);
	virtual int body_get_max_contacts_reported(RID p_body) const;

	virtual void body_set_contacts_reported_depth_threshold(RID p_body, float p_treshold);
	virtual float body_get_contacts_reported_depth_threshold(RID p_body) const;

	virtual void body_set_omit_force_integration(RID p_body, bool p_omit);
	virtual bool body_is_omitting_force_integration(RID p_body) const;

	virtual void body_set_force_integration_callback(RID p_body, Object *p_receiver, const StringName &p_method, const Variant &p_udata = Variant());

	virtual void body_set_ray_pickable(RID p_body, bool p_enable);
	virtual bool body_is_ray_pickable(RID p_body) const;

	// this function only works on physics process, errors and returns null otherwise
	virtual PhysicsDirectBodyState *body_get_direct_state(RID p_body);

	virtual bool body_test_motion(RID p_body, const Transform &p_from, const Vector3 &p_motion, MotionResult *r_result = NULL);

	/* SOFT BODY API */

	virtual RID soft_body_create(bool p_init_sleeping = false);

	virtual void soft_body_set_space(RID p_body, RID p_space);
	virtual RID soft_body_get_space(RID p_body) const;

	virtual void soft_body_set_trimesh_body_shape(RID p_body, PoolVector<int> p_indices, PoolVector<Vector3> p_vertices, int p_triangles_num);

	virtual void soft_body_set_collision_layer(RID p_body, uint32_t p_layer);
	virtual uint32_t soft_body_get_collision_layer(RID p_body) const;

	virtual void soft_body_set_collision_mask(RID p_body, uint32_t p_mask);
	virtual uint32_t soft_body_get_collision_mask(RID p_body) const;

	virtual void soft_body_add_collision_exception(RID p_body, RID p_body_b);
	virtual void soft_body_remove_collision_exception(RID p_body, RID p_body_b);
	virtual void soft_body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions);

	virtual void soft_body_set_state(RID p_body, BodyState p_state, const Variant &p_variant);
	virtual Variant soft_body_get_state(RID p_body, BodyState p_state) const;

	virtual void soft_body_set_transform(RID p_body, const Transform &p_transform);
	virtual Transform soft_body_get_transform(RID p_body) const;

	virtual void soft_body_set_ray_pickable(RID p_body, bool p_enable);
	virtual bool soft_body_is_ray_pickable(RID p_body) const;

	/* JOINT API */

	virtual JointType joint_get_type(RID p_joint) const;

	virtual void joint_set_solver_priority(RID p_joint, int p_priority);
	virtual int joint_get_solver_priority(RID p_joint) const;

	virtual RID joint_create_pin(RID p_body_A, const Vector3 &p_local_A, RID p_body_B, const Vector3 &p_local_B);

	virtual void pin_joint_set_param(RID p_joint, PinJointParam p_param, float p_value);
	virtual float pin_joint_get_param(RID p_joint, PinJointParam p_param) const;

	virtual void pin_joint_set_local_a(RID p_joint, const Vector3 &p_A);
	virtual Vector3 pin_joint_get_local_a(RID p_joint) const;

	virtual void pin_joint_set_local_b(RID p_joint, const Vector3 &p_B);
	virtual Vector3 pin_joint_get_local_b(RID p_joint) const;

	virtual RID joint_create_hinge(RID p_body_A, const Transform &p_frame_A, RID p_body_B, const Transform &p_frame_B);
	virtual RID joint_create_hinge_simple(RID p_body_A, const Vector3 &p_pivot_A, const Vector3 &p_axis_A, RID p_body_B, const Vector3 &p_pivot_B, const Vector3 &p_axis_B);

	virtual void hinge_joint_set_param(RID p_joint, HingeJointParam p_param, float p_value);
	virtual float hinge_joint_get_param(RID p_joint, HingeJointParam p_param) const;

	virtual void hinge_joint_set_flag(RID p_joint, HingeJointFlag p_flag, bool p_value);
	virtual bool hinge_joint_get_flag(RID p_joint, HingeJointFlag p_flag) const;

	/// Reference frame is A
	virtual RID joint_create_slider(RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B);

	virtual void slider_joint_set_param(RID p_joint, SliderJointParam p_param, float p_value);
	virtual float slider_joint_get_param(RID p_joint, SliderJointParam p_param) const;

	/// Reference frame is A
	virtual RID joint_create_cone_twist(RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B);

	virtual void cone_twist_joint_set_param(RID p_joint, ConeTwistJointParam p_param, float p_value);
	virtual float cone_twist_joint_get_param(RID p_joint, ConeTwistJointParam p_param) const;

	/// Reference frame is A
	virtual RID joint_create_generic_6dof(RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B);

	virtual void generic_6dof_joint_set_param(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisParam p_param, float p_value);
	virtual float generic_6dof_joint_get_param(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisParam p_param);

	virtual void generic_6dof_joint_set_flag(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisFlag p_flag, bool p_enable);
	virtual bool generic_6dof_joint_get_flag(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisFlag p_flag);

	/* MISC */

	virtual void free(RID p_rid);

	virtual void set_active(bool p_active) {
		active = p_active;
	}

	static bool singleton_isActive() {
		return static_cast<BulletPhysicsServer *>(get_singleton())->active;
	}

	bool isActive() {
		return active;
	}

	virtual void init();
	virtual void step(float p_deltaTime);
	virtual void sync();
	virtual void flush_queries();
	virtual void finish();

	virtual int get_process_info(ProcessInfo p_info);

	CollisionObjectBullet *get_collisin_object(RID p_object) const;
	RigidCollisionObjectBullet *get_rigid_collisin_object(RID p_object) const;

	/// Internal APIs
public:
};

#endif
