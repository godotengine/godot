/*************************************************************************/
/*  physics_server.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "reference.h"

class PhysicsDirectSpaceState;

class PhysicsDirectBodyState : public Object {

	OBJ_TYPE( PhysicsDirectBodyState, Object );
protected:
	static void _bind_methods();
public:

	virtual Vector3 get_total_gravity() const=0; // get gravity vector working on this body space/area
	virtual float get_total_density() const=0; // get density of this body space/area

	virtual float get_inverse_mass() const=0; // get the mass
	virtual Vector3 get_inverse_inertia() const=0; // get density of this body space
	virtual Matrix3 get_inverse_inertia_tensor() const=0; // get density of this body space

	virtual void set_linear_velocity(const Vector3& p_velocity)=0;
	virtual Vector3 get_linear_velocity() const=0;

	virtual void set_angular_velocity(const Vector3& p_velocity)=0;
	virtual Vector3 get_angular_velocity() const=0;

	virtual void set_transform(const Transform& p_transform)=0;
	virtual Transform get_transform() const=0;

	virtual void add_force(const Vector3& p_force, const Vector3& p_pos)=0;

	virtual void set_sleep_state(bool p_enable)=0;
	virtual bool is_sleeping() const=0;

	virtual int get_contact_count() const=0;

	virtual Vector3 get_contact_local_pos(int p_contact_idx) const=0;
	virtual Vector3 get_contact_local_normal(int p_contact_idx) const=0;
	virtual int get_contact_local_shape(int p_contact_idx) const=0;

	virtual RID get_contact_collider(int p_contact_idx) const=0;
	virtual Vector3 get_contact_collider_pos(int p_contact_idx) const=0;
	virtual ObjectID get_contact_collider_id(int p_contact_idx) const=0;
	virtual Object* get_contact_collider_object(int p_contact_idx) const;
	virtual int get_contact_collider_shape(int p_contact_idx) const=0;
	virtual Vector3 get_contact_collider_velocity_at_pos(int p_contact_idx) const=0;

	virtual real_t get_step() const=0;
	virtual void integrate_forces();

	virtual PhysicsDirectSpaceState* get_space_state()=0;

	PhysicsDirectBodyState();
};


class PhysicsShapeQueryResult;


class PhysicsDirectSpaceState : public Object {

	OBJ_TYPE( PhysicsDirectSpaceState, Object );

	Variant _intersect_ray(const Vector3& p_from, const Vector3& p_to,const Vector<RID>& p_exclude=Vector<RID>(),uint32_t p_user_mask=0);
	Variant _intersect_shape(const RID& p_shape, const Transform& p_xform,int p_result_max=64,const Vector<RID>& p_exclude=Vector<RID>(),uint32_t p_user_mask=0);


protected:
	static void _bind_methods();

public:

	struct RayResult {

		Vector3 position;
		Vector3 normal;
		RID rid;
		ObjectID collider_id;
		Object *collider;
		int shape;
	};

	virtual bool intersect_ray(const Vector3& p_from, const Vector3& p_to,RayResult &r_result,const Set<RID>& p_exclude=Set<RID>(),uint32_t p_user_mask=0)=0;

	struct ShapeResult {

		RID rid;
		ObjectID collider_id;
		Object *collider;
		int shape;

	};

	virtual int intersect_shape(const RID& p_shape, const Transform& p_xform,ShapeResult *r_results,int p_result_max,const Set<RID>& p_exclude=Set<RID>(),uint32_t p_user_mask=0)=0;

	PhysicsDirectSpaceState();
};


class PhysicsShapeQueryResult : public Reference {

	OBJ_TYPE( PhysicsShapeQueryResult, Reference );

	Vector<PhysicsDirectSpaceState::ShapeResult> result;

friend class PhysicsDirectSpaceState;

protected:
	static void _bind_methods();
public:

	int get_result_count() const;
	RID get_result_rid(int p_idx) const;
	ObjectID get_result_object_id(int p_idx) const;
	Object* get_result_object(int p_idx) const;
	int get_result_object_shape(int p_idx) const;

	PhysicsShapeQueryResult();
};


class PhysicsServer : public Object {

	OBJ_TYPE( PhysicsServer, Object );

	static PhysicsServer * singleton;

protected:
	static void _bind_methods();

public:

	static PhysicsServer * get_singleton();

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

	virtual RID shape_create(ShapeType p_shape)=0;
	virtual void shape_set_data(RID p_shape, const Variant& p_data)=0;
	virtual void shape_set_custom_solver_bias(RID p_shape, real_t p_bias)=0;

	virtual ShapeType shape_get_type(RID p_shape) const=0;
	virtual Variant shape_get_data(RID p_shape) const=0;
	virtual real_t shape_get_custom_solver_bias(RID p_shape) const=0;


	/* SPACE API */

	virtual RID space_create()=0;
	virtual void space_set_active(RID p_space,bool p_active)=0;
	virtual bool space_is_active(RID p_space) const=0;

	enum SpaceParameter {

		SPACE_PARAM_CONTACT_RECYCLE_RADIUS,
		SPACE_PARAM_CONTACT_MAX_SEPARATION,
		SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION,
		SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_TRESHOLD,
		SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_TRESHOLD,
		SPACE_PARAM_BODY_TIME_TO_SLEEP,
		SPACE_PARAM_BODY_ANGULAR_VELOCITY_DAMP_RATIO,
		SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS,
	};

	virtual void space_set_param(RID p_space,SpaceParameter p_param, real_t p_value)=0;
	virtual real_t space_get_param(RID p_space,SpaceParameter p_param) const=0;

	// this function only works on fixed process, errors and returns null otherwise
	virtual PhysicsDirectSpaceState* space_get_direct_state(RID p_space)=0;


	//missing space parameters

	/* AREA API */

	//missing attenuation? missing better override?



	enum AreaParameter {
		AREA_PARAM_GRAVITY,
		AREA_PARAM_GRAVITY_VECTOR,
		AREA_PARAM_GRAVITY_IS_POINT,
		AREA_PARAM_GRAVITY_POINT_ATTENUATION,
		AREA_PARAM_DENSITY,
		AREA_PARAM_PRIORITY
	};

	virtual RID area_create()=0;

	virtual void area_set_space(RID p_area, RID p_space)=0;
	virtual RID area_get_space(RID p_area) const=0;


	enum AreaSpaceOverrideMode {
		AREA_SPACE_OVERRIDE_DISABLED,
		AREA_SPACE_OVERRIDE_COMBINE,
		AREA_SPACE_OVERRIDE_REPLACE,
	};

	virtual void area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode)=0;
	virtual AreaSpaceOverrideMode area_get_space_override_mode(RID p_area) const=0;

	virtual void area_add_shape(RID p_area, RID p_shape, const Transform& p_transform=Transform())=0;
	virtual void area_set_shape(RID p_area, int p_shape_idx,RID p_shape)=0;
	virtual void area_set_shape_transform(RID p_area, int p_shape_idx, const Transform& p_transform)=0;

	virtual int area_get_shape_count(RID p_area) const=0;
	virtual RID area_get_shape(RID p_area, int p_shape_idx) const=0;
	virtual Transform area_get_shape_transform(RID p_area, int p_shape_idx) const=0;

	virtual void area_remove_shape(RID p_area, int p_shape_idx)=0;
	virtual void area_clear_shapes(RID p_area)=0;

	virtual void area_attach_object_instance_ID(RID p_area,ObjectID p_ID)=0;
	virtual ObjectID area_get_object_instance_ID(RID p_area) const=0;

	virtual void area_set_param(RID p_area,AreaParameter p_param,const Variant& p_value)=0;
	virtual void area_set_transform(RID p_area, const Transform& p_transform)=0;

	virtual Variant area_get_param(RID p_parea,AreaParameter p_param) const=0;
	virtual Transform area_get_transform(RID p_area) const=0;

	virtual void area_set_monitor_callback(RID p_area,Object *p_receiver,const StringName& p_method)=0;

	/* BODY API */

	//missing ccd?

	enum BodyMode {
		BODY_MODE_STATIC,
		BODY_MODE_KINEMATIC,
		BODY_MODE_RIGID,
		//BODY_MODE_SOFT
		BODY_MODE_CHARACTER
	};

	virtual RID body_create(BodyMode p_mode=BODY_MODE_RIGID,bool p_init_sleeping=false)=0;

	virtual void body_set_space(RID p_body, RID p_space)=0;
	virtual RID body_get_space(RID p_body) const=0;

	virtual void body_set_mode(RID p_body, BodyMode p_mode)=0;
	virtual BodyMode body_get_mode(RID p_body, BodyMode p_mode) const=0;

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform& p_transform=Transform())=0;
	virtual void body_set_shape(RID p_body, int p_shape_idx,RID p_shape)=0;
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform& p_transform)=0;

	virtual int body_get_shape_count(RID p_body) const=0;
	virtual RID body_get_shape(RID p_body, int p_shape_idx) const=0;
	virtual Transform body_get_shape_transform(RID p_body, int p_shape_idx) const=0;

	virtual void body_set_shape_as_trigger(RID p_body, int p_shape_idx,bool p_enable)=0;
	virtual bool body_is_shape_set_as_trigger(RID p_body, int p_shape_idx) const=0;

	virtual void body_remove_shape(RID p_body, int p_shape_idx)=0;
	virtual void body_clear_shapes(RID p_body)=0;

	virtual void body_attach_object_instance_ID(RID p_body,uint32_t p_ID)=0;
	virtual uint32_t body_get_object_instance_ID(RID p_body) const=0;

	virtual void body_set_enable_continuous_collision_detection(RID p_body,bool p_enable)=0;
	virtual bool body_is_continuous_collision_detection_enabled(RID p_body) const=0;

	virtual void body_set_user_flags(RID p_body, uint32_t p_flags)=0;
	virtual uint32_t body_get_user_flags(RID p_body, uint32_t p_flags) const=0;

	// common body variables
	enum BodyParameter {
		BODY_PARAM_BOUNCE,
		BODY_PARAM_FRICTION,
		BODY_PARAM_MASS, ///< unused for static, always infinite
		BODY_PARAM_MAX,
	};

	virtual void body_set_param(RID p_body, BodyParameter p_param, float p_value)=0;
	virtual float body_get_param(RID p_body, BodyParameter p_param) const=0;

	//advanced simulation
	virtual void body_static_simulate_motion(RID p_body,const Transform& p_new_transform)=0;

	//state
	enum BodyState {
		BODY_STATE_TRANSFORM,
		BODY_STATE_LINEAR_VELOCITY,
		BODY_STATE_ANGULAR_VELOCITY,
		BODY_STATE_SLEEPING,
		BODY_STATE_CAN_SLEEP
	};

	virtual void body_set_state(RID p_body, BodyState p_state, const Variant& p_variant)=0;
	virtual Variant body_get_state(RID p_body, BodyState p_state) const=0;

	//do something about it
	virtual void body_set_applied_force(RID p_body, const Vector3& p_force)=0;
	virtual Vector3 body_get_applied_force(RID p_body) const=0;

	virtual void body_set_applied_torque(RID p_body, const Vector3& p_torque)=0;
	virtual Vector3 body_get_applied_torque(RID p_body) const=0;

	virtual void body_apply_impulse(RID p_body, const Vector3& p_pos, const Vector3& p_impulse)=0;
	virtual void body_set_axis_velocity(RID p_body, const Vector3& p_axis_velocity)=0;

	enum BodyAxisLock {
		BODY_AXIS_LOCK_DISABLED,
		BODY_AXIS_LOCK_X,
		BODY_AXIS_LOCK_Y,
		BODY_AXIS_LOCK_Z,
	};

	virtual void body_set_axis_lock(RID p_body,BodyAxisLock p_lock)=0;
	virtual BodyAxisLock body_get_axis_lock(RID p_body) const=0;

	//fix
	virtual void body_add_collision_exception(RID p_body, RID p_body_b)=0;
	virtual void body_remove_collision_exception(RID p_body, RID p_body_b)=0;
	virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions)=0;

	virtual void body_set_max_contacts_reported(RID p_body, int p_contacts)=0;
	virtual int body_get_max_contacts_reported(RID p_body) const=0;

	//missing remove
	virtual void body_set_contacts_reported_depth_treshold(RID p_body, float p_treshold)=0;
	virtual float body_get_contacts_reported_depth_treshold(RID p_body) const=0;

	virtual void body_set_omit_force_integration(RID p_body,bool p_omit)=0;
	virtual bool body_is_omitting_force_integration(RID p_body) const=0;

	virtual void body_set_force_integration_callback(RID p_body,Object *p_receiver,const StringName& p_method,const Variant& p_udata=Variant())=0;

	/* JOINT API */
#if 0
	enum JointType {

		JOINT_PIN,
		JOINT_GROOVE,
		JOINT_DAMPED_SPRING
	};

	enum JointParam {
		JOINT_PARAM_BIAS,
		JOINT_PARAM_MAX_BIAS,
		JOINT_PARAM_MAX_FORCE,
	};

	virtual void joint_set_param(RID p_joint, JointParam p_param, real_t p_value)=0;
	virtual real_t joint_get_param(RID p_joint,JointParam p_param) const=0;

	virtual RID pin_joint_create(const Vector3& p_anchor,RID p_body_a,RID p_body_b=RID())=0;
	virtual RID groove_joint_create(const Vector3& p_a_groove1,const Vector3& p_a_groove2, const Vector3& p_b_anchor, RID p_body_a,RID p_body_b)=0;
	virtual RID damped_spring_joint_create(const Vector3& p_anchor_a,const Vector3& p_anchor_b,RID p_body_a,RID p_body_b=RID())=0;

	enum DampedStringParam {
		DAMPED_STRING_REST_LENGTH,
		DAMPED_STRING_STIFFNESS,
		DAMPED_STRING_DAMPING
	};
	virtual void damped_string_joint_set_param(RID p_joint, DampedStringParam p_param, real_t p_value)=0;
	virtual real_t damped_string_joint_get_param(RID p_joint, DampedStringParam p_param) const=0;

	virtual JointType joint_get_type(RID p_joint) const=0;	
#endif
	/* QUERY API */

	enum AreaBodyStatus {
		AREA_BODY_ADDED,
		AREA_BODY_REMOVED
	};


	/* MISC */

	virtual void free(RID p_rid)=0;

	virtual void set_active(bool p_active)=0;
	virtual void init()=0;
	virtual void step(float p_step)=0;
	virtual void sync()=0;
	virtual void flush_queries()=0;
	virtual void finish()=0;

	PhysicsServer();
	~PhysicsServer();
};

VARIANT_ENUM_CAST( PhysicsServer::ShapeType );
VARIANT_ENUM_CAST( PhysicsServer::SpaceParameter );
VARIANT_ENUM_CAST( PhysicsServer::AreaParameter );
VARIANT_ENUM_CAST( PhysicsServer::AreaSpaceOverrideMode );
VARIANT_ENUM_CAST( PhysicsServer::BodyMode );
VARIANT_ENUM_CAST( PhysicsServer::BodyParameter );
VARIANT_ENUM_CAST( PhysicsServer::BodyState );
VARIANT_ENUM_CAST( PhysicsServer::BodyAxisLock );
//VARIANT_ENUM_CAST( PhysicsServer::JointParam );
//VARIANT_ENUM_CAST( PhysicsServer::JointType );
//VARIANT_ENUM_CAST( PhysicsServer::DampedStringParam );
//VARIANT_ENUM_CAST( PhysicsServer::ObjectType );
VARIANT_ENUM_CAST( PhysicsServer::AreaBodyStatus );

#endif
