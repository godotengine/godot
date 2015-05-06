/*************************************************************************/
/*  physics_server_sw.h                                                  */
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
#ifndef PHYSICS_SERVER_SW
#define PHYSICS_SERVER_SW


#include "servers/physics_server.h"
#include "shape_sw.h"
#include "space_sw.h"
#include "step_sw.h"
#include "joints_sw.h"


class PhysicsServerSW : public PhysicsServer {

	OBJ_TYPE( PhysicsServerSW, PhysicsServer );

friend class PhysicsDirectSpaceStateSW;
	bool active;
	int iterations;
	bool doing_sync;
	real_t last_step;

	StepSW *stepper;
	Set<const SpaceSW*> active_spaces;

	PhysicsDirectBodyStateSW *direct_state;

	mutable RID_Owner<ShapeSW> shape_owner;
	mutable RID_Owner<SpaceSW> space_owner;
	mutable RID_Owner<AreaSW> area_owner;
	mutable RID_Owner<BodySW> body_owner;
	mutable RID_Owner<JointSW> joint_owner;

//	void _clear_query(QuerySW *p_query);
public:

	virtual RID shape_create(ShapeType p_shape);
	virtual void shape_set_data(RID p_shape, const Variant& p_data);
	virtual void shape_set_custom_solver_bias(RID p_shape, real_t p_bias);

	virtual ShapeType shape_get_type(RID p_shape) const;
	virtual Variant shape_get_data(RID p_shape) const;
	virtual real_t shape_get_custom_solver_bias(RID p_shape) const;

	/* SPACE API */

	virtual RID space_create();
	virtual void space_set_active(RID p_space,bool p_active);
	virtual bool space_is_active(RID p_space) const;

	virtual void space_set_param(RID p_space,SpaceParameter p_param, real_t p_value);
	virtual real_t space_get_param(RID p_space,SpaceParameter p_param) const;

	// this function only works on fixed process, errors and returns null otherwise
	virtual PhysicsDirectSpaceState* space_get_direct_state(RID p_space);


	/* AREA API */

	virtual RID area_create();

	virtual void area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode);
	virtual AreaSpaceOverrideMode area_get_space_override_mode(RID p_area) const;

	virtual void area_set_space(RID p_area, RID p_space);
	virtual RID area_get_space(RID p_area) const;

	virtual void area_add_shape(RID p_area, RID p_shape, const Transform& p_transform=Transform());
	virtual void area_set_shape(RID p_area, int p_shape_idx,RID p_shape);
	virtual void area_set_shape_transform(RID p_area, int p_shape_idx, const Transform& p_transform);

	virtual int area_get_shape_count(RID p_area) const;
	virtual RID area_get_shape(RID p_area, int p_shape_idx) const;
	virtual Transform area_get_shape_transform(RID p_area, int p_shape_idx) const;

	virtual void area_remove_shape(RID p_area, int p_shape_idx);
	virtual void area_clear_shapes(RID p_area);

	virtual void area_attach_object_instance_ID(RID p_area,ObjectID p_ID);
	virtual ObjectID area_get_object_instance_ID(RID p_area) const;

	virtual void area_set_param(RID p_area,AreaParameter p_param,const Variant& p_value);
	virtual void area_set_transform(RID p_area, const Transform& p_transform);

	virtual Variant area_get_param(RID p_parea,AreaParameter p_param) const;
	virtual Transform area_get_transform(RID p_area) const;

	virtual void area_set_monitor_callback(RID p_area,Object *p_receiver,const StringName& p_method);


	/* BODY API */

	// create a body of a given type
	virtual RID body_create(BodyMode p_mode=BODY_MODE_RIGID,bool p_init_sleeping=false);

	virtual void body_set_space(RID p_body, RID p_space);
	virtual RID body_get_space(RID p_body) const;

	virtual void body_set_mode(RID p_body, BodyMode p_mode);
	virtual BodyMode body_get_mode(RID p_body, BodyMode p_mode) const;

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform& p_transform=Transform());
	virtual void body_set_shape(RID p_body, int p_shape_idx,RID p_shape);
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform& p_transform);

	virtual int body_get_shape_count(RID p_body) const;
	virtual RID body_get_shape(RID p_body, int p_shape_idx) const;
	virtual Transform body_get_shape_transform(RID p_body, int p_shape_idx) const;

	virtual void body_set_shape_as_trigger(RID p_body, int p_shape_idx,bool p_enable);
	virtual bool body_is_shape_set_as_trigger(RID p_body, int p_shape_idx) const;

	virtual void body_remove_shape(RID p_body, int p_shape_idx);
	virtual void body_clear_shapes(RID p_body);

	virtual void body_attach_object_instance_ID(RID p_body,uint32_t p_ID);
	virtual uint32_t body_get_object_instance_ID(RID p_body) const;

	virtual void body_set_enable_continuous_collision_detection(RID p_body,bool p_enable);
	virtual bool body_is_continuous_collision_detection_enabled(RID p_body) const;

	virtual void body_set_user_flags(RID p_body, uint32_t p_flags);
	virtual uint32_t body_get_user_flags(RID p_body, uint32_t p_flags) const;

	virtual void body_set_param(RID p_body, BodyParameter p_param, float p_value);
	virtual float body_get_param(RID p_body, BodyParameter p_param) const;

	//advanced simulation
	virtual void body_static_simulate_motion(RID p_body,const Transform& p_new_transform);

	virtual void body_set_state(RID p_body, BodyState p_state, const Variant& p_variant);
	virtual Variant body_get_state(RID p_body, BodyState p_state) const;

	virtual void body_set_applied_force(RID p_body, const Vector3& p_force);
	virtual Vector3 body_get_applied_force(RID p_body) const;

	virtual void body_set_applied_torque(RID p_body, const Vector3& p_torque);
	virtual Vector3 body_get_applied_torque(RID p_body) const;

	virtual void body_apply_impulse(RID p_body, const Vector3& p_pos, const Vector3& p_impulse);
	virtual void body_set_axis_velocity(RID p_body, const Vector3& p_axis_velocity);

	virtual void body_set_axis_lock(RID p_body,BodyAxisLock p_lock);
	virtual BodyAxisLock body_get_axis_lock(RID p_body) const;

	virtual void body_add_collision_exception(RID p_body, RID p_body_b);
	virtual void body_remove_collision_exception(RID p_body, RID p_body_b);
	virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions);

	virtual void body_set_contacts_reported_depth_treshold(RID p_body, float p_treshold);
	virtual float body_get_contacts_reported_depth_treshold(RID p_body) const;

	virtual void body_set_omit_force_integration(RID p_body,bool p_omit);
	virtual bool body_is_omitting_force_integration(RID p_body) const;

	virtual void body_set_max_contacts_reported(RID p_body, int p_contacts);
	virtual int body_get_max_contacts_reported(RID p_body) const;

	virtual void body_set_force_integration_callback(RID p_body,Object *p_receiver,const StringName& p_method,const Variant& p_udata=Variant());

	/* JOINT API */
#if 0
	virtual void joint_set_param(RID p_joint, JointParam p_param, real_t p_value);
	virtual real_t joint_get_param(RID p_joint,JointParam p_param) const;

	virtual RID pin_joint_create(const Vector3& p_pos,RID p_body_a,RID p_body_b=RID());
	virtual RID groove_joint_create(const Vector3& p_a_groove1,const Vector3& p_a_groove2, const Vector3& p_b_anchor, RID p_body_a,RID p_body_b);
	virtual RID damped_spring_joint_create(const Vector3& p_anchor_a,const Vector3& p_anchor_b,RID p_body_a,RID p_body_b=RID());
	virtual void damped_string_joint_set_param(RID p_joint, DampedStringParam p_param, real_t p_value);
	virtual real_t damped_string_joint_get_param(RID p_joint, DampedStringParam p_param) const;

	virtual JointType joint_get_type(RID p_joint) const;
#endif
	/* MISC */

	virtual void free(RID p_rid);

	virtual void set_active(bool p_active);
	virtual void init();
	virtual void step(float p_step);
	virtual void sync();
	virtual void flush_queries();
	virtual void finish();

	PhysicsServerSW();
	~PhysicsServerSW();

};

#endif

