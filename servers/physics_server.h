/*************************************************************************/
/*  physics_server.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef PHYSICS_SERVER_H
#define PHYSICS_SERVER_H

#include "object.h"
#include "resource.h"

class PhysicsDirectSpaceState;

class PhysicsDirectBodyState : public Object {

	GDCLASS(PhysicsDirectBodyState, Object);

protected:
	static void _bind_methods();

public:
	virtual Vector3 get_total_gravity() const = 0;
	virtual float get_total_angular_damp() const = 0;
	virtual float get_total_linear_damp() const = 0;

	virtual Vector3 get_center_of_mass() const = 0;
	virtual Basis get_principal_inertia_axes() const = 0;
	virtual float get_inverse_mass() const = 0; // get the mass
	virtual Vector3 get_inverse_inertia() const = 0; // get density of this body space
	virtual Basis get_inverse_inertia_tensor() const = 0; // get density of this body space

	virtual void set_linear_velocity(const Vector3 &p_velocity) = 0;
	virtual Vector3 get_linear_velocity() const = 0;

	virtual void set_angular_velocity(const Vector3 &p_velocity) = 0;
	virtual Vector3 get_angular_velocity() const = 0;

	virtual void set_transform(const Transform &p_transform) = 0;
	virtual Transform get_transform() const = 0;

	virtual void add_force(const Vector3 &p_force, const Vector3 &p_pos) = 0;
	virtual void apply_impulse(const Vector3 &p_pos, const Vector3 &p_j) = 0;
	virtual void apply_torque_impulse(const Vector3 &p_j) = 0;

	virtual void set_sleep_state(bool p_enable) = 0;
	virtual bool is_sleeping() const = 0;

	virtual int get_contact_count() const = 0;

	virtual Vector3 get_contact_local_position(int p_contact_idx) const = 0;
	virtual Vector3 get_contact_local_normal(int p_contact_idx) const = 0;
	virtual int get_contact_local_shape(int p_contact_idx) const = 0;

	virtual RID get_contact_collider(int p_contact_idx) const = 0;
	virtual Vector3 get_contact_collider_position(int p_contact_idx) const = 0;
	virtual ObjectID get_contact_collider_id(int p_contact_idx) const = 0;
	virtual Object *get_contact_collider_object(int p_contact_idx) const;
	virtual int get_contact_collider_shape(int p_contact_idx) const = 0;
	virtual Vector3 get_contact_collider_velocity_at_position(int p_contact_idx) const = 0;

	virtual real_t get_step() const = 0;
	virtual void integrate_forces();

	virtual PhysicsDirectSpaceState *get_space_state() = 0;

	PhysicsDirectBodyState();
};

class PhysicsShapeQueryResult;

class PhysicsShapeQueryParameters : public Reference {

	GDCLASS(PhysicsShapeQueryParameters, Reference);
	friend class PhysicsDirectSpaceState;

	RID shape;
	Transform transform;
	float margin;
	Set<RID> exclude;
	uint32_t collision_layer;
	uint32_t object_type_mask;

protected:
	static void _bind_methods();

public:
	void set_shape(const RES &p_shape);
	void set_shape_rid(const RID &p_shape);
	RID get_shape_rid() const;

	void set_transform(const Transform &p_transform);
	Transform get_transform() const;

	void set_margin(float p_margin);
	float get_margin() const;

	void set_collision_layer(int p_collision_layer);
	int get_collision_layer() const;

	void set_object_type_mask(int p_object_type_mask);
	int get_object_type_mask() const;

	void set_exclude(const Vector<RID> &p_exclude);
	Vector<RID> get_exclude() const;

	PhysicsShapeQueryParameters();
};

class PhysicsDirectSpaceState : public Object {

	GDCLASS(PhysicsDirectSpaceState, Object);

public:
	enum ObjectTypeMask {
		TYPE_MASK_STATIC_BODY = 1 << 0,
		TYPE_MASK_KINEMATIC_BODY = 1 << 1,
		TYPE_MASK_RIGID_BODY = 1 << 2,
		TYPE_MASK_CHARACTER_BODY = 1 << 3,
		TYPE_MASK_AREA = 1 << 4,
		TYPE_MASK_COLLISION = TYPE_MASK_STATIC_BODY | TYPE_MASK_CHARACTER_BODY | TYPE_MASK_KINEMATIC_BODY | TYPE_MASK_RIGID_BODY
	};

private:
	Dictionary _intersect_ray(const Vector3 &p_from, const Vector3 &p_to, const Vector<RID> &p_exclude = Vector<RID>(), uint32_t p_layers = 0, uint32_t p_object_type_mask = TYPE_MASK_COLLISION);
	Array _intersect_shape(const Ref<PhysicsShapeQueryParameters> &p_shape_query, int p_max_results = 32);
	Array _cast_motion(const Ref<PhysicsShapeQueryParameters> &p_shape_query, const Vector3 &p_motion);
	Array _collide_shape(const Ref<PhysicsShapeQueryParameters> &p_shape_query, int p_max_results = 32);
	Dictionary _get_rest_info(const Ref<PhysicsShapeQueryParameters> &p_shape_query);

protected:
	static void _bind_methods();

public:
	struct ShapeResult {

		RID rid;
		ObjectID collider_id;
		Object *collider;
		int shape;
	};

	virtual int intersect_point(const Vector3 &p_point, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, uint32_t p_object_type_mask = TYPE_MASK_COLLISION) = 0;

	struct RayResult {

		Vector3 position;
		Vector3 normal;
		RID rid;
		ObjectID collider_id;
		Object *collider;
		int shape;
	};

	virtual bool intersect_ray(const Vector3 &p_from, const Vector3 &p_to, RayResult &r_result, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, uint32_t p_object_type_mask = TYPE_MASK_COLLISION, bool p_pick_ray = false) = 0;

	virtual int intersect_shape(const RID &p_shape, const Transform &p_xform, float p_margin, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, uint32_t p_object_type_mask = TYPE_MASK_COLLISION) = 0;

	struct ShapeRestInfo {

		Vector3 point;
		Vector3 normal;
		RID rid;
		ObjectID collider_id;
		int shape;
		Vector3 linear_velocity; //velocity at contact point
	};

	virtual bool cast_motion(const RID &p_shape, const Transform &p_xform, const Vector3 &p_motion, float p_margin, float &p_closest_safe, float &p_closest_unsafe, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, uint32_t p_object_type_mask = TYPE_MASK_COLLISION, ShapeRestInfo *r_info = NULL) = 0;

	virtual bool collide_shape(RID p_shape, const Transform &p_shape_xform, float p_margin, Vector3 *r_results, int p_result_max, int &r_result_count, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, uint32_t p_object_type_mask = TYPE_MASK_COLLISION) = 0;

	virtual bool rest_info(RID p_shape, const Transform &p_shape_xform, float p_margin, ShapeRestInfo *r_info, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, uint32_t p_object_type_mask = TYPE_MASK_COLLISION) = 0;

	virtual Vector3 get_closest_point_to_object_volume(RID p_object, const Vector3 p_point) const = 0;

	PhysicsDirectSpaceState();
};

VARIANT_ENUM_CAST(PhysicsDirectSpaceState::ObjectTypeMask);

class PhysicsShapeQueryResult : public Reference {

	GDCLASS(PhysicsShapeQueryResult, Reference);

	Vector<PhysicsDirectSpaceState::ShapeResult> result;

	friend class PhysicsDirectSpaceState;

protected:
	static void _bind_methods();

public:
	int get_result_count() const;
	RID get_result_rid(int p_idx) const;
	ObjectID get_result_object_id(int p_idx) const;
	Object *get_result_object(int p_idx) const;
	int get_result_object_shape(int p_idx) const;

	PhysicsShapeQueryResult();
};

class PhysicsServer : public Object {

	GDCLASS(PhysicsServer, Object);

	static PhysicsServer *singleton;

protected:
	static void _bind_methods();

public:
	static PhysicsServer *get_singleton();

	enum ShapeType {
		SHAPE_PLANE, ///< plane:"plane"
		SHAPE_RAY, ///< float:"length"
		SHAPE_SPHERE, ///< float:"radius"
		SHAPE_BOX, ///< vec3:"extents"
		SHAPE_CAPSULE, ///< dict( float:"radius", float:"height"):capsule
		SHAPE_CONVEX_POLYGON, ///< array of planes:"planes"
		SHAPE_CONCAVE_POLYGON, ///< vector3 array:"triangles" , or Dictionary with "indices" (int array) and "triangles" (Vector3 array)
		SHAPE_HEIGHTMAP, ///< dict( int:"width", int:"depth",float:"cell_size", float_array:"heights"
		SHAPE_CUSTOM, ///< Server-Implementation based custom shape, calling shape_create() with this value will result in an error
	};

	virtual RID shape_create(ShapeType p_shape) = 0;
	virtual void shape_set_data(RID p_shape, const Variant &p_data) = 0;
	virtual void shape_set_custom_solver_bias(RID p_shape, real_t p_bias) = 0;

	virtual ShapeType shape_get_type(RID p_shape) const = 0;
	virtual Variant shape_get_data(RID p_shape) const = 0;
	virtual real_t shape_get_custom_solver_bias(RID p_shape) const = 0;

	/* SPACE API */

	virtual RID space_create() = 0;
	virtual void space_set_active(RID p_space, bool p_active) = 0;
	virtual bool space_is_active(RID p_space) const = 0;

	enum SpaceParameter {

		SPACE_PARAM_CONTACT_RECYCLE_RADIUS,
		SPACE_PARAM_CONTACT_MAX_SEPARATION,
		SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION,
		SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD,
		SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD,
		SPACE_PARAM_BODY_TIME_TO_SLEEP,
		SPACE_PARAM_BODY_ANGULAR_VELOCITY_DAMP_RATIO,
		SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS,
	};

	virtual void space_set_param(RID p_space, SpaceParameter p_param, real_t p_value) = 0;
	virtual real_t space_get_param(RID p_space, SpaceParameter p_param) const = 0;

	// this function only works on physics process, errors and returns null otherwise
	virtual PhysicsDirectSpaceState *space_get_direct_state(RID p_space) = 0;

	virtual void space_set_debug_contacts(RID p_space, int p_max_contacts) = 0;
	virtual Vector<Vector3> space_get_contacts(RID p_space) const = 0;
	virtual int space_get_contact_count(RID p_space) const = 0;

	//missing space parameters

	/* AREA API */

	//missing attenuation? missing better override?

	enum AreaParameter {
		AREA_PARAM_GRAVITY,
		AREA_PARAM_GRAVITY_VECTOR,
		AREA_PARAM_GRAVITY_IS_POINT,
		AREA_PARAM_GRAVITY_DISTANCE_SCALE,
		AREA_PARAM_GRAVITY_POINT_ATTENUATION,
		AREA_PARAM_LINEAR_DAMP,
		AREA_PARAM_ANGULAR_DAMP,
		AREA_PARAM_PRIORITY
	};

	virtual RID area_create() = 0;

	virtual void area_set_space(RID p_area, RID p_space) = 0;
	virtual RID area_get_space(RID p_area) const = 0;

	enum AreaSpaceOverrideMode {
		AREA_SPACE_OVERRIDE_DISABLED,
		AREA_SPACE_OVERRIDE_COMBINE,
		AREA_SPACE_OVERRIDE_COMBINE_REPLACE,
		AREA_SPACE_OVERRIDE_REPLACE,
		AREA_SPACE_OVERRIDE_REPLACE_COMBINE
	};

	virtual void area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode) = 0;
	virtual AreaSpaceOverrideMode area_get_space_override_mode(RID p_area) const = 0;

	virtual void area_add_shape(RID p_area, RID p_shape, const Transform &p_transform = Transform()) = 0;
	virtual void area_set_shape(RID p_area, int p_shape_idx, RID p_shape) = 0;
	virtual void area_set_shape_transform(RID p_area, int p_shape_idx, const Transform &p_transform) = 0;

	virtual int area_get_shape_count(RID p_area) const = 0;
	virtual RID area_get_shape(RID p_area, int p_shape_idx) const = 0;
	virtual Transform area_get_shape_transform(RID p_area, int p_shape_idx) const = 0;

	virtual void area_remove_shape(RID p_area, int p_shape_idx) = 0;
	virtual void area_clear_shapes(RID p_area) = 0;

	virtual void area_set_shape_disabled(RID p_area, int p_shape_idx, bool p_disabled) = 0;

	virtual void area_attach_object_instance_id(RID p_area, ObjectID p_ID) = 0;
	virtual ObjectID area_get_object_instance_id(RID p_area) const = 0;

	virtual void area_set_param(RID p_area, AreaParameter p_param, const Variant &p_value) = 0;
	virtual void area_set_transform(RID p_area, const Transform &p_transform) = 0;

	virtual Variant area_get_param(RID p_parea, AreaParameter p_param) const = 0;
	virtual Transform area_get_transform(RID p_area) const = 0;

	virtual void area_set_collision_mask(RID p_area, uint32_t p_mask) = 0;
	virtual void area_set_collision_layer(RID p_area, uint32_t p_layer) = 0;

	virtual void area_set_monitorable(RID p_area, bool p_monitorable) = 0;

	virtual void area_set_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) = 0;
	virtual void area_set_area_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) = 0;

	virtual void area_set_ray_pickable(RID p_area, bool p_enable) = 0;
	virtual bool area_is_ray_pickable(RID p_area) const = 0;

	/* BODY API */

	//missing ccd?

	enum BodyMode {
		BODY_MODE_STATIC,
		BODY_MODE_KINEMATIC,
		BODY_MODE_RIGID,
		//BODY_MODE_SOFT
		BODY_MODE_CHARACTER
	};

	virtual RID body_create(BodyMode p_mode = BODY_MODE_RIGID, bool p_init_sleeping = false) = 0;

	virtual void body_set_space(RID p_body, RID p_space) = 0;
	virtual RID body_get_space(RID p_body) const = 0;

	virtual void body_set_mode(RID p_body, BodyMode p_mode) = 0;
	virtual BodyMode body_get_mode(RID p_body) const = 0;

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform &p_transform = Transform()) = 0;
	virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape) = 0;
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform &p_transform) = 0;

	virtual int body_get_shape_count(RID p_body) const = 0;
	virtual RID body_get_shape(RID p_body, int p_shape_idx) const = 0;
	virtual Transform body_get_shape_transform(RID p_body, int p_shape_idx) const = 0;

	virtual void body_remove_shape(RID p_body, int p_shape_idx) = 0;
	virtual void body_clear_shapes(RID p_body) = 0;

	virtual void body_set_shape_disabled(RID p_body, int p_shape_idx, bool p_disabled) = 0;

	virtual void body_attach_object_instance_id(RID p_body, uint32_t p_ID) = 0;
	virtual uint32_t body_get_object_instance_id(RID p_body) const = 0;

	virtual void body_set_enable_continuous_collision_detection(RID p_body, bool p_enable) = 0;
	virtual bool body_is_continuous_collision_detection_enabled(RID p_body) const = 0;

	virtual void body_set_collision_layer(RID p_body, uint32_t p_layer) = 0;
	virtual uint32_t body_get_collision_layer(RID p_body) const = 0;

	virtual void body_set_collision_mask(RID p_body, uint32_t p_mask) = 0;
	virtual uint32_t body_get_collision_mask(RID p_body) const = 0;

	virtual void body_set_user_flags(RID p_body, uint32_t p_flags) = 0;
	virtual uint32_t body_get_user_flags(RID p_body) const = 0;

	// common body variables
	enum BodyParameter {
		BODY_PARAM_BOUNCE,
		BODY_PARAM_FRICTION,
		BODY_PARAM_MASS, ///< unused for static, always infinite
		BODY_PARAM_GRAVITY_SCALE,
		BODY_PARAM_LINEAR_DAMP,
		BODY_PARAM_ANGULAR_DAMP,
		BODY_PARAM_MAX,
	};

	virtual void body_set_param(RID p_body, BodyParameter p_param, float p_value) = 0;
	virtual float body_get_param(RID p_body, BodyParameter p_param) const = 0;

	//state
	enum BodyState {
		BODY_STATE_TRANSFORM,
		BODY_STATE_LINEAR_VELOCITY,
		BODY_STATE_ANGULAR_VELOCITY,
		BODY_STATE_SLEEPING,
		BODY_STATE_CAN_SLEEP
	};

	virtual void body_set_state(RID p_body, BodyState p_state, const Variant &p_variant) = 0;
	virtual Variant body_get_state(RID p_body, BodyState p_state) const = 0;

	//do something about it
	virtual void body_set_applied_force(RID p_body, const Vector3 &p_force) = 0;
	virtual Vector3 body_get_applied_force(RID p_body) const = 0;

	virtual void body_set_applied_torque(RID p_body, const Vector3 &p_torque) = 0;
	virtual Vector3 body_get_applied_torque(RID p_body) const = 0;

	virtual void body_apply_impulse(RID p_body, const Vector3 &p_pos, const Vector3 &p_impulse) = 0;
	virtual void body_apply_torque_impulse(RID p_body, const Vector3 &p_impulse) = 0;
	virtual void body_set_axis_velocity(RID p_body, const Vector3 &p_axis_velocity) = 0;

	enum BodyAxisLock {
		BODY_AXIS_LOCK_DISABLED,
		BODY_AXIS_LOCK_X,
		BODY_AXIS_LOCK_Y,
		BODY_AXIS_LOCK_Z,
	};

	virtual void body_set_axis_lock(RID p_body, BodyAxisLock p_lock) = 0;
	virtual BodyAxisLock body_get_axis_lock(RID p_body) const = 0;

	//fix
	virtual void body_add_collision_exception(RID p_body, RID p_body_b) = 0;
	virtual void body_remove_collision_exception(RID p_body, RID p_body_b) = 0;
	virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) = 0;

	virtual void body_set_max_contacts_reported(RID p_body, int p_contacts) = 0;
	virtual int body_get_max_contacts_reported(RID p_body) const = 0;

	//missing remove
	virtual void body_set_contacts_reported_depth_threshold(RID p_body, float p_threshold) = 0;
	virtual float body_get_contacts_reported_depth_threshold(RID p_body) const = 0;

	virtual void body_set_omit_force_integration(RID p_body, bool p_omit) = 0;
	virtual bool body_is_omitting_force_integration(RID p_body) const = 0;

	virtual void body_set_force_integration_callback(RID p_body, Object *p_receiver, const StringName &p_method, const Variant &p_udata = Variant()) = 0;

	virtual void body_set_ray_pickable(RID p_body, bool p_enable) = 0;
	virtual bool body_is_ray_pickable(RID p_body) const = 0;

	// this function only works on physics process, errors and returns null otherwise
	virtual PhysicsDirectBodyState *body_get_direct_state(RID p_body) = 0;

	struct MotionResult {

		Vector3 motion;
		Vector3 remainder;

		Vector3 collision_point;
		Vector3 collision_normal;
		Vector3 collider_velocity;
		int collision_local_shape;
		ObjectID collider_id;
		RID collider;
		int collider_shape;
		Variant collider_metadata;
	};

	virtual bool body_test_motion(RID p_body, const Transform &p_from, const Vector3 &p_motion, float p_margin = 0.001, MotionResult *r_result = NULL) = 0;

	/* JOINT API */

	enum JointType {

		JOINT_PIN,
		JOINT_HINGE,
		JOINT_SLIDER,
		JOINT_CONE_TWIST,
		JOINT_6DOF

	};

	virtual JointType joint_get_type(RID p_joint) const = 0;

	virtual void joint_set_solver_priority(RID p_joint, int p_priority) = 0;
	virtual int joint_get_solver_priority(RID p_joint) const = 0;

	virtual RID joint_create_pin(RID p_body_A, const Vector3 &p_local_A, RID p_body_B, const Vector3 &p_local_B) = 0;

	enum PinJointParam {
		PIN_JOINT_BIAS,
		PIN_JOINT_DAMPING,
		PIN_JOINT_IMPULSE_CLAMP
	};

	virtual void pin_joint_set_param(RID p_joint, PinJointParam p_param, float p_value) = 0;
	virtual float pin_joint_get_param(RID p_joint, PinJointParam p_param) const = 0;

	virtual void pin_joint_set_local_a(RID p_joint, const Vector3 &p_A) = 0;
	virtual Vector3 pin_joint_get_local_a(RID p_joint) const = 0;

	virtual void pin_joint_set_local_b(RID p_joint, const Vector3 &p_B) = 0;
	virtual Vector3 pin_joint_get_local_b(RID p_joint) const = 0;

	enum HingeJointParam {

		HINGE_JOINT_BIAS,
		HINGE_JOINT_LIMIT_UPPER,
		HINGE_JOINT_LIMIT_LOWER,
		HINGE_JOINT_LIMIT_BIAS,
		HINGE_JOINT_LIMIT_SOFTNESS,
		HINGE_JOINT_LIMIT_RELAXATION,
		HINGE_JOINT_MOTOR_TARGET_VELOCITY,
		HINGE_JOINT_MOTOR_MAX_IMPULSE,
		HINGE_JOINT_MAX
	};

	enum HingeJointFlag {
		HINGE_JOINT_FLAG_USE_LIMIT,
		HINGE_JOINT_FLAG_ENABLE_MOTOR,
		HINGE_JOINT_FLAG_MAX
	};

	virtual RID joint_create_hinge(RID p_body_A, const Transform &p_hinge_A, RID p_body_B, const Transform &p_hinge_B) = 0;
	virtual RID joint_create_hinge_simple(RID p_body_A, const Vector3 &p_pivot_A, const Vector3 &p_axis_A, RID p_body_B, const Vector3 &p_pivot_B, const Vector3 &p_axis_B) = 0;

	virtual void hinge_joint_set_param(RID p_joint, HingeJointParam p_param, float p_value) = 0;
	virtual float hinge_joint_get_param(RID p_joint, HingeJointParam p_param) const = 0;

	virtual void hinge_joint_set_flag(RID p_joint, HingeJointFlag p_flag, bool p_value) = 0;
	virtual bool hinge_joint_get_flag(RID p_joint, HingeJointFlag p_flag) const = 0;

	enum SliderJointParam {
		SLIDER_JOINT_LINEAR_LIMIT_UPPER,
		SLIDER_JOINT_LINEAR_LIMIT_LOWER,
		SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS,
		SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION,
		SLIDER_JOINT_LINEAR_LIMIT_DAMPING,
		SLIDER_JOINT_LINEAR_MOTION_SOFTNESS,
		SLIDER_JOINT_LINEAR_MOTION_RESTITUTION,
		SLIDER_JOINT_LINEAR_MOTION_DAMPING,
		SLIDER_JOINT_LINEAR_ORTHOGONAL_SOFTNESS,
		SLIDER_JOINT_LINEAR_ORTHOGONAL_RESTITUTION,
		SLIDER_JOINT_LINEAR_ORTHOGONAL_DAMPING,

		SLIDER_JOINT_ANGULAR_LIMIT_UPPER,
		SLIDER_JOINT_ANGULAR_LIMIT_LOWER,
		SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS,
		SLIDER_JOINT_ANGULAR_LIMIT_RESTITUTION,
		SLIDER_JOINT_ANGULAR_LIMIT_DAMPING,
		SLIDER_JOINT_ANGULAR_MOTION_SOFTNESS,
		SLIDER_JOINT_ANGULAR_MOTION_RESTITUTION,
		SLIDER_JOINT_ANGULAR_MOTION_DAMPING,
		SLIDER_JOINT_ANGULAR_ORTHOGONAL_SOFTNESS,
		SLIDER_JOINT_ANGULAR_ORTHOGONAL_RESTITUTION,
		SLIDER_JOINT_ANGULAR_ORTHOGONAL_DAMPING,
		SLIDER_JOINT_MAX

	};

	virtual RID joint_create_slider(RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B) = 0; //reference frame is A

	virtual void slider_joint_set_param(RID p_joint, SliderJointParam p_param, float p_value) = 0;
	virtual float slider_joint_get_param(RID p_joint, SliderJointParam p_param) const = 0;

	enum ConeTwistJointParam {
		CONE_TWIST_JOINT_SWING_SPAN,
		CONE_TWIST_JOINT_TWIST_SPAN,
		CONE_TWIST_JOINT_BIAS,
		CONE_TWIST_JOINT_SOFTNESS,
		CONE_TWIST_JOINT_RELAXATION,
		CONE_TWIST_MAX
	};

	virtual RID joint_create_cone_twist(RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B) = 0; //reference frame is A

	virtual void cone_twist_joint_set_param(RID p_joint, ConeTwistJointParam p_param, float p_value) = 0;
	virtual float cone_twist_joint_get_param(RID p_joint, ConeTwistJointParam p_param) const = 0;

	enum G6DOFJointAxisParam {
		G6DOF_JOINT_LINEAR_LOWER_LIMIT,
		G6DOF_JOINT_LINEAR_UPPER_LIMIT,
		G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS,
		G6DOF_JOINT_LINEAR_RESTITUTION,
		G6DOF_JOINT_LINEAR_DAMPING,
		G6DOF_JOINT_ANGULAR_LOWER_LIMIT,
		G6DOF_JOINT_ANGULAR_UPPER_LIMIT,
		G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS,
		G6DOF_JOINT_ANGULAR_DAMPING,
		G6DOF_JOINT_ANGULAR_RESTITUTION,
		G6DOF_JOINT_ANGULAR_FORCE_LIMIT,
		G6DOF_JOINT_ANGULAR_ERP,
		G6DOF_JOINT_ANGULAR_MOTOR_TARGET_VELOCITY,
		G6DOF_JOINT_ANGULAR_MOTOR_FORCE_LIMIT,
		G6DOF_JOINT_MAX
	};

	enum G6DOFJointAxisFlag {

		G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT,
		G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT,
		G6DOF_JOINT_FLAG_ENABLE_MOTOR,
		G6DOF_JOINT_FLAG_MAX
	};

	virtual RID joint_create_generic_6dof(RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B) = 0; //reference frame is A

	virtual void generic_6dof_joint_set_param(RID p_joint, Vector3::Axis, G6DOFJointAxisParam p_param, float p_value) = 0;
	virtual float generic_6dof_joint_get_param(RID p_joint, Vector3::Axis, G6DOFJointAxisParam p_param) = 0;

	virtual void generic_6dof_joint_set_flag(RID p_joint, Vector3::Axis, G6DOFJointAxisFlag p_flag, bool p_enable) = 0;
	virtual bool generic_6dof_joint_get_flag(RID p_joint, Vector3::Axis, G6DOFJointAxisFlag p_flag) = 0;

	/* QUERY API */

	enum AreaBodyStatus {
		AREA_BODY_ADDED,
		AREA_BODY_REMOVED
	};

	/* MISC */

	virtual void free(RID p_rid) = 0;

	virtual void set_active(bool p_active) = 0;
	virtual void init() = 0;
	virtual void step(float p_step) = 0;
	virtual void sync() = 0;
	virtual void flush_queries() = 0;
	virtual void finish() = 0;

	enum ProcessInfo {

		INFO_ACTIVE_OBJECTS,
		INFO_COLLISION_PAIRS,
		INFO_ISLAND_COUNT
	};

	virtual int get_process_info(ProcessInfo p_info) = 0;

	PhysicsServer();
	~PhysicsServer();
};

VARIANT_ENUM_CAST(PhysicsServer::ShapeType);
VARIANT_ENUM_CAST(PhysicsServer::SpaceParameter);
VARIANT_ENUM_CAST(PhysicsServer::AreaParameter);
VARIANT_ENUM_CAST(PhysicsServer::AreaSpaceOverrideMode);
VARIANT_ENUM_CAST(PhysicsServer::BodyMode);
VARIANT_ENUM_CAST(PhysicsServer::BodyParameter);
VARIANT_ENUM_CAST(PhysicsServer::BodyState);
VARIANT_ENUM_CAST(PhysicsServer::BodyAxisLock);
VARIANT_ENUM_CAST(PhysicsServer::PinJointParam);
VARIANT_ENUM_CAST(PhysicsServer::JointType);
VARIANT_ENUM_CAST(PhysicsServer::HingeJointParam);
VARIANT_ENUM_CAST(PhysicsServer::HingeJointFlag);
VARIANT_ENUM_CAST(PhysicsServer::SliderJointParam);
VARIANT_ENUM_CAST(PhysicsServer::ConeTwistJointParam);
VARIANT_ENUM_CAST(PhysicsServer::G6DOFJointAxisParam);
VARIANT_ENUM_CAST(PhysicsServer::G6DOFJointAxisFlag);
VARIANT_ENUM_CAST(PhysicsServer::AreaBodyStatus);
VARIANT_ENUM_CAST(PhysicsServer::ProcessInfo);

#endif
