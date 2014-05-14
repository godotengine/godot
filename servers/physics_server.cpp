/*************************************************************************/
/*  physics_server.cpp                                                   */
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
#include "physics_server.h"
#include "print_string.h"
PhysicsServer * PhysicsServer::singleton=NULL;


void PhysicsDirectBodyState::integrate_forces() {

	real_t step = get_step();
	Vector3 lv = get_linear_velocity();
	lv+=get_total_gravity() * step;

	Vector3 av = get_angular_velocity();

	float damp = 1.0 - step * get_total_density();

	if (damp<0) // reached zero in the given time
		damp=0;

	lv*=damp;
	av*=damp;

	set_linear_velocity(lv);
	set_angular_velocity(av);




}

Object* PhysicsDirectBodyState::get_contact_collider_object(int p_contact_idx) const {

	ObjectID objid = get_contact_collider_id(p_contact_idx);
	Object *obj = ObjectDB::get_instance( objid );
	return obj;
}

PhysicsServer * PhysicsServer::get_singleton() {

	return singleton;
}

void PhysicsDirectBodyState::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_total_gravity"),&PhysicsDirectBodyState::get_total_gravity);
	ObjectTypeDB::bind_method(_MD("get_total_density"),&PhysicsDirectBodyState::get_total_density);

	ObjectTypeDB::bind_method(_MD("get_inverse_mass"),&PhysicsDirectBodyState::get_inverse_mass);
	ObjectTypeDB::bind_method(_MD("get_inverse_inertia"),&PhysicsDirectBodyState::get_inverse_inertia);

	ObjectTypeDB::bind_method(_MD("set_linear_velocity","velocity"),&PhysicsDirectBodyState::set_linear_velocity);
	ObjectTypeDB::bind_method(_MD("get_linear_velocity"),&PhysicsDirectBodyState::get_linear_velocity);

	ObjectTypeDB::bind_method(_MD("set_angular_velocity","velocity"),&PhysicsDirectBodyState::set_angular_velocity);
	ObjectTypeDB::bind_method(_MD("get_angular_velocity"),&PhysicsDirectBodyState::get_angular_velocity);

	ObjectTypeDB::bind_method(_MD("set_transform","transform"),&PhysicsDirectBodyState::set_transform);
	ObjectTypeDB::bind_method(_MD("get_transform"),&PhysicsDirectBodyState::get_transform);

	ObjectTypeDB::bind_method(_MD("add_force","force","pos"),&PhysicsDirectBodyState::add_force);

	ObjectTypeDB::bind_method(_MD("set_sleep_state","enabled"),&PhysicsDirectBodyState::set_sleep_state);
	ObjectTypeDB::bind_method(_MD("is_sleeping"),&PhysicsDirectBodyState::is_sleeping);

	ObjectTypeDB::bind_method(_MD("get_contact_count"),&PhysicsDirectBodyState::get_contact_count);

	ObjectTypeDB::bind_method(_MD("get_contact_local_pos","contact_idx"),&PhysicsDirectBodyState::get_contact_local_pos);
	ObjectTypeDB::bind_method(_MD("get_contact_local_normal","contact_idx"),&PhysicsDirectBodyState::get_contact_local_normal);
	ObjectTypeDB::bind_method(_MD("get_contact_local_shape","contact_idx"),&PhysicsDirectBodyState::get_contact_local_shape);
	ObjectTypeDB::bind_method(_MD("get_contact_collider","contact_idx"),&PhysicsDirectBodyState::get_contact_collider);
	ObjectTypeDB::bind_method(_MD("get_contact_collider_pos","contact_idx"),&PhysicsDirectBodyState::get_contact_collider_pos);
	ObjectTypeDB::bind_method(_MD("get_contact_collider_id","contact_idx"),&PhysicsDirectBodyState::get_contact_collider_id);
	ObjectTypeDB::bind_method(_MD("get_contact_collider_object","contact_idx"),&PhysicsDirectBodyState::get_contact_collider_object);
	ObjectTypeDB::bind_method(_MD("get_contact_collider_shape","contact_idx"),&PhysicsDirectBodyState::get_contact_collider_shape);
	ObjectTypeDB::bind_method(_MD("get_contact_collider_velocity_at_pos","contact_idx"),&PhysicsDirectBodyState::get_contact_collider_velocity_at_pos);
	ObjectTypeDB::bind_method(_MD("get_step"),&PhysicsDirectBodyState::get_step);
	ObjectTypeDB::bind_method(_MD("integrate_forces"),&PhysicsDirectBodyState::integrate_forces);
	ObjectTypeDB::bind_method(_MD("get_space_state:PhysicsDirectSpaceState"),&PhysicsDirectBodyState::get_space_state);

}

PhysicsDirectBodyState::PhysicsDirectBodyState() {}

///////////////////////////////////////////////////////



Variant PhysicsDirectSpaceState::_intersect_ray(const Vector3& p_from, const Vector3& p_to,const Vector<RID>& p_exclude,uint32_t p_user_mask) {

	RayResult inters;
	Set<RID> exclude;
	for(int i=0;i<p_exclude.size();i++)
		exclude.insert(p_exclude[i]);

	bool res = intersect_ray(p_from,p_to,inters,exclude,p_user_mask);

	if (!res)
		return Variant();

	Dictionary d;
	d["position"]=inters.position;
	d["normal"]=inters.normal;
	d["collider_id"]=inters.collider_id;
	d["collider"]=inters.collider;
	d["shape"]=inters.shape;
	d["rid"]=inters.rid;

	return d;
}

Variant PhysicsDirectSpaceState::_intersect_shape(const RID& p_shape, const Transform& p_xform,int p_result_max,const Vector<RID>& p_exclude,uint32_t p_user_mask) {



	ERR_FAIL_INDEX_V(p_result_max,4096,Variant());
	if (p_result_max<=0)
		return Variant();

	Set<RID> exclude;
	for(int i=0;i<p_exclude.size();i++)
		exclude.insert(p_exclude[i]);

	ShapeResult *res=(ShapeResult*)alloca(p_result_max*sizeof(ShapeResult));

	int rc = intersect_shape(p_shape,p_xform,res,p_result_max,exclude,p_user_mask);

	if (rc==0)
		return Variant();

	Ref<PhysicsShapeQueryResult>  result = memnew( PhysicsShapeQueryResult );
	result->result.resize(rc);
	for(int i=0;i<rc;i++)
		result->result[i]=res[i];

	return result;

}





PhysicsDirectSpaceState::PhysicsDirectSpaceState() {



}


void PhysicsDirectSpaceState::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("intersect_ray","from","to","exclude","umask"),&PhysicsDirectSpaceState::_intersect_ray,DEFVAL(Array()),DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("intersect_shape:PhysicsShapeQueryResult","shape","xform","result_max","exclude","umask"),&PhysicsDirectSpaceState::_intersect_shape,DEFVAL(Array()),DEFVAL(0));

}


int PhysicsShapeQueryResult::get_result_count() const {

	return result.size();
}
RID PhysicsShapeQueryResult::get_result_rid(int p_idx) const {

	return result[p_idx].rid;
}
ObjectID PhysicsShapeQueryResult::get_result_object_id(int p_idx) const {

	return result[p_idx].collider_id;
}
Object* PhysicsShapeQueryResult::get_result_object(int p_idx) const {

	return result[p_idx].collider;
}
int PhysicsShapeQueryResult::get_result_object_shape(int p_idx) const {

	return result[p_idx].shape;
}

PhysicsShapeQueryResult::PhysicsShapeQueryResult() {


}

void PhysicsShapeQueryResult::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_result_count"),&PhysicsShapeQueryResult::get_result_count);
	ObjectTypeDB::bind_method(_MD("get_result_rid","idx"),&PhysicsShapeQueryResult::get_result_rid);
	ObjectTypeDB::bind_method(_MD("get_result_object_id","idx"),&PhysicsShapeQueryResult::get_result_object_id);
	ObjectTypeDB::bind_method(_MD("get_result_object","idx"),&PhysicsShapeQueryResult::get_result_object);
	ObjectTypeDB::bind_method(_MD("get_result_object_shape","idx"),&PhysicsShapeQueryResult::get_result_object_shape);


}






///////////////////////////////////////

void PhysicsServer::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("shape_create","type"),&PhysicsServer::shape_create);
	ObjectTypeDB::bind_method(_MD("shape_set_data","shape","data"),&PhysicsServer::shape_set_data);

	ObjectTypeDB::bind_method(_MD("shape_get_type","shape"),&PhysicsServer::shape_get_type);
	ObjectTypeDB::bind_method(_MD("shape_get_data","shape"),&PhysicsServer::shape_get_data);


	ObjectTypeDB::bind_method(_MD("space_create"),&PhysicsServer::space_create);
	ObjectTypeDB::bind_method(_MD("space_set_active","space","active"),&PhysicsServer::space_set_active);
	ObjectTypeDB::bind_method(_MD("space_is_active","space"),&PhysicsServer::space_is_active);
	ObjectTypeDB::bind_method(_MD("space_set_param","space","param","value"),&PhysicsServer::space_set_param);
	ObjectTypeDB::bind_method(_MD("space_get_param","space","param"),&PhysicsServer::space_get_param);
	ObjectTypeDB::bind_method(_MD("space_get_direct_state:PhysicsDirectSpaceState","space"),&PhysicsServer::space_get_direct_state);

	ObjectTypeDB::bind_method(_MD("area_create"),&PhysicsServer::area_create);
	ObjectTypeDB::bind_method(_MD("area_set_space","area","space"),&PhysicsServer::area_set_space);
	ObjectTypeDB::bind_method(_MD("area_get_space","area"),&PhysicsServer::area_get_space);

	ObjectTypeDB::bind_method(_MD("area_set_space_override_mode","area","mode"),&PhysicsServer::area_set_space_override_mode);
	ObjectTypeDB::bind_method(_MD("area_get_space_override_mode","area"),&PhysicsServer::area_get_space_override_mode);

	ObjectTypeDB::bind_method(_MD("area_add_shape","area","shape","transform"),&PhysicsServer::area_set_shape,DEFVAL(Transform()));
	ObjectTypeDB::bind_method(_MD("area_set_shape","area","shape_idx","shape"),&PhysicsServer::area_get_shape);
	ObjectTypeDB::bind_method(_MD("area_set_shape_transform","area","shape_idx","transform"),&PhysicsServer::area_set_shape_transform);

	ObjectTypeDB::bind_method(_MD("area_get_shape_count","area"),&PhysicsServer::area_get_shape_count);
	ObjectTypeDB::bind_method(_MD("area_get_shape","area","shape_idx"),&PhysicsServer::area_get_shape);
	ObjectTypeDB::bind_method(_MD("area_get_shape_transform","area","shape_idx"),&PhysicsServer::area_get_shape_transform);

	ObjectTypeDB::bind_method(_MD("area_remove_shape","area","shape_idx"),&PhysicsServer::area_remove_shape);
	ObjectTypeDB::bind_method(_MD("area_clear_shapes","area"),&PhysicsServer::area_clear_shapes);


	ObjectTypeDB::bind_method(_MD("area_set_param","area","param","value"),&PhysicsServer::area_get_param);
	ObjectTypeDB::bind_method(_MD("area_set_transform","area","transform"),&PhysicsServer::area_get_transform);

	ObjectTypeDB::bind_method(_MD("area_get_param","area","param"),&PhysicsServer::area_get_param);
	ObjectTypeDB::bind_method(_MD("area_get_transform","area"),&PhysicsServer::area_get_transform);

	ObjectTypeDB::bind_method(_MD("area_attach_object_instance_ID","area","id"),&PhysicsServer::area_attach_object_instance_ID);
	ObjectTypeDB::bind_method(_MD("area_get_object_instance_ID","area"),&PhysicsServer::area_get_object_instance_ID);

	ObjectTypeDB::bind_method(_MD("area_set_monitor_callback","receiver","method"),&PhysicsServer::area_set_monitor_callback);

	ObjectTypeDB::bind_method(_MD("body_create","mode","init_sleeping"),&PhysicsServer::body_create,DEFVAL(BODY_MODE_RIGID),DEFVAL(false));

	ObjectTypeDB::bind_method(_MD("body_set_space","body","space"),&PhysicsServer::body_set_space);
	ObjectTypeDB::bind_method(_MD("body_get_space","body"),&PhysicsServer::body_get_space);

	ObjectTypeDB::bind_method(_MD("body_set_mode","body","mode"),&PhysicsServer::body_set_mode);
	ObjectTypeDB::bind_method(_MD("body_get_mode","body"),&PhysicsServer::body_get_mode);

	ObjectTypeDB::bind_method(_MD("body_add_shape","body","shape","transform"),&PhysicsServer::body_add_shape,DEFVAL(Transform()));
	ObjectTypeDB::bind_method(_MD("body_set_shape","body","shape_idx","shape"),&PhysicsServer::body_set_shape);
	ObjectTypeDB::bind_method(_MD("body_set_shape_transform","body","shape_idx","transform"),&PhysicsServer::body_set_shape_transform);

	ObjectTypeDB::bind_method(_MD("body_get_shape_count","body"),&PhysicsServer::body_get_shape_count);
	ObjectTypeDB::bind_method(_MD("body_get_shape","body","shape_idx"),&PhysicsServer::body_get_shape);
	ObjectTypeDB::bind_method(_MD("body_get_shape_transform","body","shape_idx"),&PhysicsServer::body_get_shape_transform);

	ObjectTypeDB::bind_method(_MD("body_remove_shape","body","shape_idx"),&PhysicsServer::body_remove_shape);
	ObjectTypeDB::bind_method(_MD("body_clear_shapes","body"),&PhysicsServer::body_clear_shapes);

	ObjectTypeDB::bind_method(_MD("body_attach_object_instance_ID","body","id"),&PhysicsServer::body_attach_object_instance_ID);
	ObjectTypeDB::bind_method(_MD("body_get_object_instance_ID","body"),&PhysicsServer::body_get_object_instance_ID);


	ObjectTypeDB::bind_method(_MD("body_set_enable_continuous_collision_detection","body","enable"),&PhysicsServer::body_set_enable_continuous_collision_detection);
	ObjectTypeDB::bind_method(_MD("body_is_continuous_collision_detection_enabled","body"),&PhysicsServer::body_is_continuous_collision_detection_enabled);


	//ObjectTypeDB::bind_method(_MD("body_set_user_flags","flags""),&PhysicsServer::body_set_shape,DEFVAL(Transform));
	//ObjectTypeDB::bind_method(_MD("body_get_user_flags","body","shape_idx","shape"),&PhysicsServer::body_get_shape);

	ObjectTypeDB::bind_method(_MD("body_set_param","body","param","value"),&PhysicsServer::body_set_param);
	ObjectTypeDB::bind_method(_MD("body_get_param","body","param"),&PhysicsServer::body_get_param);

	ObjectTypeDB::bind_method(_MD("body_static_simulate_motion","body","new_xform"),&PhysicsServer::body_static_simulate_motion);

	ObjectTypeDB::bind_method(_MD("body_set_state","body","state","value"),&PhysicsServer::body_set_state);
	ObjectTypeDB::bind_method(_MD("body_get_state","body","state"),&PhysicsServer::body_get_state);

	ObjectTypeDB::bind_method(_MD("body_apply_impulse","body","pos","impulse"),&PhysicsServer::body_apply_impulse);
	ObjectTypeDB::bind_method(_MD("body_set_axis_velocity","body","axis_velocity"),&PhysicsServer::body_set_axis_velocity);

	ObjectTypeDB::bind_method(_MD("body_set_axis_lock","body","axis"),&PhysicsServer::body_set_axis_lock);
	ObjectTypeDB::bind_method(_MD("body_get_axis_lock","body"),&PhysicsServer::body_set_axis_lock);

	ObjectTypeDB::bind_method(_MD("body_add_collision_exception","body","excepted_body"),&PhysicsServer::body_add_collision_exception);
	ObjectTypeDB::bind_method(_MD("body_remove_collision_exception","body","excepted_body"),&PhysicsServer::body_remove_collision_exception);
//	virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions)=0;

	ObjectTypeDB::bind_method(_MD("body_set_max_contacts_reported","body","amount"),&PhysicsServer::body_set_max_contacts_reported);
	ObjectTypeDB::bind_method(_MD("body_get_max_contacts_reported","body"),&PhysicsServer::body_get_max_contacts_reported);

	ObjectTypeDB::bind_method(_MD("body_set_omit_force_integration","body","enable"),&PhysicsServer::body_set_omit_force_integration);
	ObjectTypeDB::bind_method(_MD("body_is_omitting_force_integration","body"),&PhysicsServer::body_is_omitting_force_integration);

	ObjectTypeDB::bind_method(_MD("body_set_force_integration_callback","body","receiver","method","userdata"),&PhysicsServer::body_set_force_integration_callback,DEFVAL(Variant()));

	/* JOINT API */
/*
	ObjectTypeDB::bind_method(_MD("joint_set_param","joint","param","value"),&PhysicsServer::joint_set_param);
	ObjectTypeDB::bind_method(_MD("joint_get_param","joint","param"),&PhysicsServer::joint_get_param);

	ObjectTypeDB::bind_method(_MD("pin_joint_create","anchor","body_a","body_b"),&PhysicsServer::pin_joint_create,DEFVAL(RID()));
	ObjectTypeDB::bind_method(_MD("groove_joint_create","groove1_a","groove2_a","anchor_b","body_a","body_b"),&PhysicsServer::groove_joint_create,DEFVAL(RID()),DEFVAL(RID()));
	ObjectTypeDB::bind_method(_MD("damped_spring_joint_create","anchor_a","anchor_b","body_a","body_b"),&PhysicsServer::damped_spring_joint_create,DEFVAL(RID()));

	ObjectTypeDB::bind_method(_MD("damped_string_joint_set_param","joint","param","value"),&PhysicsServer::damped_string_joint_set_param,DEFVAL(RID()));
	ObjectTypeDB::bind_method(_MD("damped_string_joint_get_param","joint","param"),&PhysicsServer::damped_string_joint_get_param);

	ObjectTypeDB::bind_method(_MD("joint_get_type","joint"),&PhysicsServer::joint_get_type);
*/
	ObjectTypeDB::bind_method(_MD("free","rid"),&PhysicsServer::free);

	ObjectTypeDB::bind_method(_MD("set_active","active"),&PhysicsServer::set_active);

//	ObjectTypeDB::bind_method(_MD("init"),&PhysicsServer::init);
//	ObjectTypeDB::bind_method(_MD("step"),&PhysicsServer::step);
//	ObjectTypeDB::bind_method(_MD("sync"),&PhysicsServer::sync);
	//ObjectTypeDB::bind_method(_MD("flush_queries"),&PhysicsServer::flush_queries);


	BIND_CONSTANT( SHAPE_PLANE );
	BIND_CONSTANT( SHAPE_RAY );
	BIND_CONSTANT( SHAPE_SPHERE );
	BIND_CONSTANT( SHAPE_BOX );
	BIND_CONSTANT( SHAPE_CAPSULE );
	BIND_CONSTANT( SHAPE_CONVEX_POLYGON );
	BIND_CONSTANT( SHAPE_CONCAVE_POLYGON );
	BIND_CONSTANT( SHAPE_HEIGHTMAP );
	BIND_CONSTANT( SHAPE_CUSTOM );


	BIND_CONSTANT( AREA_PARAM_GRAVITY );
	BIND_CONSTANT( AREA_PARAM_GRAVITY_VECTOR );
	BIND_CONSTANT( AREA_PARAM_GRAVITY_IS_POINT );
	BIND_CONSTANT( AREA_PARAM_GRAVITY_POINT_ATTENUATION );
	BIND_CONSTANT( AREA_PARAM_DENSITY );
	BIND_CONSTANT( AREA_PARAM_PRIORITY );

	BIND_CONSTANT( AREA_SPACE_OVERRIDE_COMBINE );
	BIND_CONSTANT( AREA_SPACE_OVERRIDE_DISABLED );
	BIND_CONSTANT( AREA_SPACE_OVERRIDE_REPLACE );

	BIND_CONSTANT( BODY_MODE_STATIC );
	BIND_CONSTANT( BODY_MODE_KINEMATIC );
	BIND_CONSTANT( BODY_MODE_RIGID );
	BIND_CONSTANT( BODY_MODE_CHARACTER );

	BIND_CONSTANT( BODY_PARAM_BOUNCE );
	BIND_CONSTANT( BODY_PARAM_FRICTION );
	BIND_CONSTANT( BODY_PARAM_MASS );
	BIND_CONSTANT( BODY_PARAM_MAX );

	BIND_CONSTANT( BODY_STATE_TRANSFORM );
	BIND_CONSTANT( BODY_STATE_LINEAR_VELOCITY );
	BIND_CONSTANT( BODY_STATE_ANGULAR_VELOCITY );
	BIND_CONSTANT( BODY_STATE_SLEEPING );
	BIND_CONSTANT( BODY_STATE_CAN_SLEEP );
/*
	BIND_CONSTANT( JOINT_PIN );
	BIND_CONSTANT( JOINT_GROOVE );
	BIND_CONSTANT( JOINT_DAMPED_SPRING );

	BIND_CONSTANT( DAMPED_STRING_REST_LENGTH );
	BIND_CONSTANT( DAMPED_STRING_STIFFNESS );
	BIND_CONSTANT( DAMPED_STRING_DAMPING );
*/
//	BIND_CONSTANT( TYPE_BODY );
//	BIND_CONSTANT( TYPE_AREA );

	BIND_CONSTANT( AREA_BODY_ADDED );
	BIND_CONSTANT( AREA_BODY_REMOVED );


}


PhysicsServer::PhysicsServer() {

	ERR_FAIL_COND( singleton!=NULL );
	singleton=this;
}

PhysicsServer::~PhysicsServer() {

	singleton=NULL;
}

