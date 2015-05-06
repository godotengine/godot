/*************************************************************************/
/*  physics_2d_server.cpp                                                */
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
#include "physics_2d_server.h"
#include "print_string.h"
Physics2DServer * Physics2DServer::singleton=NULL;


void Physics2DDirectBodyState::integrate_forces() {

	real_t step = get_step();
	Vector2 lv = get_linear_velocity();
	lv+=get_total_gravity() * step;

	real_t av = get_angular_velocity();

	float damp = 1.0 - step * get_total_density();

	if (damp<0) // reached zero in the given time
		damp=0;

	lv*=damp;
	av*=damp;

	set_linear_velocity(lv);
	set_angular_velocity(av);




}

Object* Physics2DDirectBodyState::get_contact_collider_object(int p_contact_idx) const {

	ObjectID objid = get_contact_collider_id(p_contact_idx);
	Object *obj = ObjectDB::get_instance( objid );
	return obj;
}

Physics2DServer * Physics2DServer::get_singleton() {

	return singleton;
}

void Physics2DDirectBodyState::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_total_gravity"),&Physics2DDirectBodyState::get_total_gravity);
	ObjectTypeDB::bind_method(_MD("get_total_density"),&Physics2DDirectBodyState::get_total_density);

	ObjectTypeDB::bind_method(_MD("get_inverse_mass"),&Physics2DDirectBodyState::get_inverse_mass);
	ObjectTypeDB::bind_method(_MD("get_inverse_inertia"),&Physics2DDirectBodyState::get_inverse_inertia);

	ObjectTypeDB::bind_method(_MD("set_linear_velocity","velocity"),&Physics2DDirectBodyState::set_linear_velocity);
	ObjectTypeDB::bind_method(_MD("get_linear_velocity"),&Physics2DDirectBodyState::get_linear_velocity);

	ObjectTypeDB::bind_method(_MD("set_angular_velocity","velocity"),&Physics2DDirectBodyState::set_angular_velocity);
	ObjectTypeDB::bind_method(_MD("get_angular_velocity"),&Physics2DDirectBodyState::get_angular_velocity);

	ObjectTypeDB::bind_method(_MD("set_transform","transform"),&Physics2DDirectBodyState::set_transform);
	ObjectTypeDB::bind_method(_MD("get_transform"),&Physics2DDirectBodyState::get_transform);

	ObjectTypeDB::bind_method(_MD("set_sleep_state","enabled"),&Physics2DDirectBodyState::set_sleep_state);
	ObjectTypeDB::bind_method(_MD("is_sleeping"),&Physics2DDirectBodyState::is_sleeping);

	ObjectTypeDB::bind_method(_MD("get_contact_count"),&Physics2DDirectBodyState::get_contact_count);

	ObjectTypeDB::bind_method(_MD("get_contact_local_pos","contact_idx"),&Physics2DDirectBodyState::get_contact_local_pos);
	ObjectTypeDB::bind_method(_MD("get_contact_local_normal","contact_idx"),&Physics2DDirectBodyState::get_contact_local_normal);
	ObjectTypeDB::bind_method(_MD("get_contact_local_shape","contact_idx"),&Physics2DDirectBodyState::get_contact_local_shape);
	ObjectTypeDB::bind_method(_MD("get_contact_collider","contact_idx"),&Physics2DDirectBodyState::get_contact_collider);
	ObjectTypeDB::bind_method(_MD("get_contact_collider_pos","contact_idx"),&Physics2DDirectBodyState::get_contact_collider_pos);
	ObjectTypeDB::bind_method(_MD("get_contact_collider_id","contact_idx"),&Physics2DDirectBodyState::get_contact_collider_id);
	ObjectTypeDB::bind_method(_MD("get_contact_collider_object","contact_idx"),&Physics2DDirectBodyState::get_contact_collider_object);
	ObjectTypeDB::bind_method(_MD("get_contact_collider_shape","contact_idx"),&Physics2DDirectBodyState::get_contact_collider_shape);
	ObjectTypeDB::bind_method(_MD("get_contact_collider_velocity_at_pos","contact_idx"),&Physics2DDirectBodyState::get_contact_collider_velocity_at_pos);
	ObjectTypeDB::bind_method(_MD("get_step"),&Physics2DDirectBodyState::get_step);
	ObjectTypeDB::bind_method(_MD("integrate_forces"),&Physics2DDirectBodyState::integrate_forces);
	ObjectTypeDB::bind_method(_MD("get_space_state:Physics2DDirectSpaceState"),&Physics2DDirectBodyState::get_space_state);

}

Physics2DDirectBodyState::Physics2DDirectBodyState() {}

///////////////////////////////////////////////////////



Variant Physics2DDirectSpaceState::_intersect_ray(const Vector2& p_from, const Vector2& p_to,const Vector<RID>& p_exclude,uint32_t p_user_mask) {

	RayResult inters;
	Set<RID> exclude;
	for(int i=0;i<p_exclude.size();i++)
		exclude.insert(p_exclude[i]);

	bool res = intersect_ray(p_from,p_to,inters,exclude,p_user_mask);

	if (!res)
		return Variant();

	Dictionary d(true);
	d["position"]=inters.position;
	d["normal"]=inters.normal;
	d["collider_id"]=inters.collider_id;
	d["collider"]=inters.collider;
	d["shape"]=inters.shape;
	d["rid"]=inters.rid;

	return d;
}

Variant Physics2DDirectSpaceState::_intersect_shape(const RID& p_shape, const Matrix32& p_xform,int p_result_max,const Vector<RID>& p_exclude,uint32_t p_user_mask) {

	ERR_FAIL_INDEX_V(p_result_max,4096,Variant());
	if (p_result_max<=0)
		return Variant();

	Set<RID> exclude;
	for(int i=0;i<p_exclude.size();i++)
		exclude.insert(p_exclude[i]);

	ShapeResult *res=(ShapeResult*)alloca(p_result_max*sizeof(ShapeResult));

	int rc = intersect_shape(p_shape,p_xform,Vector2(),0,res,p_result_max,exclude,p_user_mask);

	if (rc==0)
		return Variant();

	Ref<Physics2DShapeQueryResult>  result = memnew( Physics2DShapeQueryResult );
	result->result.resize(rc);
	for(int i=0;i<rc;i++)
		result->result[i]=res[i];

	return result;

}


Variant Physics2DDirectSpaceState::_cast_motion(const RID& p_shape, const Matrix32& p_xform,const Vector2& p_motion,const Vector<RID>& p_exclude,uint32_t p_user_mask) {


#if 0
	Set<RID> exclude;
	for(int i=0;i<p_exclude.size();i++)
		exclude.insert(p_exclude[i]);


	bool result = cast_motion(p_shape,p_xform,p_motion,0,mcc,exclude,p_user_mask);

	if (!result)
		return Variant();

	Dictionary d(true);
	d["point"]=mcc.point;
	d["normal"]=mcc.normal;
	d["rid"]=mcc.rid;
	d["collider_id"]=mcc.collider_id;
	d["collider"]=mcc.collider;
	d["shape"]=mcc.shape;

	return d;
#endif
	return Variant();

}



Physics2DDirectSpaceState::Physics2DDirectSpaceState() {



}


void Physics2DDirectSpaceState::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("intersect_ray:Dictionary","from","to","exclude","umask"),&Physics2DDirectSpaceState::_intersect_ray,DEFVAL(Array()),DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("intersect_shape:Physics2DShapeQueryResult","shape","xform","result_max","exclude","umask"),&Physics2DDirectSpaceState::_intersect_shape,DEFVAL(Array()),DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("cast_motion","shape","xform","motion","exclude","umask"),&Physics2DDirectSpaceState::_intersect_shape,DEFVAL(Array()),DEFVAL(0));

}


int Physics2DShapeQueryResult::get_result_count() const {

	return result.size();
}
RID Physics2DShapeQueryResult::get_result_rid(int p_idx) const {

	return result[p_idx].rid;
}
ObjectID Physics2DShapeQueryResult::get_result_object_id(int p_idx) const {

	return result[p_idx].collider_id;
}
Object* Physics2DShapeQueryResult::get_result_object(int p_idx) const {

	return result[p_idx].collider;
}
int Physics2DShapeQueryResult::get_result_object_shape(int p_idx) const {

	return result[p_idx].shape;
}

Physics2DShapeQueryResult::Physics2DShapeQueryResult() {


}

void Physics2DShapeQueryResult::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_result_count"),&Physics2DShapeQueryResult::get_result_count);
	ObjectTypeDB::bind_method(_MD("get_result_rid","idx"),&Physics2DShapeQueryResult::get_result_rid);
	ObjectTypeDB::bind_method(_MD("get_result_object_id","idx"),&Physics2DShapeQueryResult::get_result_object_id);
	ObjectTypeDB::bind_method(_MD("get_result_object","idx"),&Physics2DShapeQueryResult::get_result_object);
	ObjectTypeDB::bind_method(_MD("get_result_object_shape","idx"),&Physics2DShapeQueryResult::get_result_object_shape);


}






///////////////////////////////////////

void Physics2DServer::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("shape_create","type"),&Physics2DServer::shape_create);
	ObjectTypeDB::bind_method(_MD("shape_set_data","shape","data"),&Physics2DServer::shape_set_data);

	ObjectTypeDB::bind_method(_MD("shape_get_type","shape"),&Physics2DServer::shape_get_type);
	ObjectTypeDB::bind_method(_MD("shape_get_data","shape"),&Physics2DServer::shape_get_data);


	ObjectTypeDB::bind_method(_MD("space_create"),&Physics2DServer::space_create);
	ObjectTypeDB::bind_method(_MD("space_set_active","space","active"),&Physics2DServer::space_set_active);
	ObjectTypeDB::bind_method(_MD("space_is_active","space"),&Physics2DServer::space_is_active);
	ObjectTypeDB::bind_method(_MD("space_set_param","space","param","value"),&Physics2DServer::space_set_param);
	ObjectTypeDB::bind_method(_MD("space_get_param","space","param"),&Physics2DServer::space_get_param);
	ObjectTypeDB::bind_method(_MD("space_get_direct_state:Physics2DDirectSpaceState","space"),&Physics2DServer::space_get_direct_state);

	ObjectTypeDB::bind_method(_MD("area_create"),&Physics2DServer::area_create);
	ObjectTypeDB::bind_method(_MD("area_set_space","area","space"),&Physics2DServer::area_set_space);
	ObjectTypeDB::bind_method(_MD("area_get_space","area"),&Physics2DServer::area_get_space);

	ObjectTypeDB::bind_method(_MD("area_set_space_override_mode","area","mode"),&Physics2DServer::area_set_space_override_mode);
	ObjectTypeDB::bind_method(_MD("area_get_space_override_mode","area"),&Physics2DServer::area_get_space_override_mode);

	ObjectTypeDB::bind_method(_MD("area_add_shape","area","shape","transform"),&Physics2DServer::area_set_shape,DEFVAL(Matrix32()));
	ObjectTypeDB::bind_method(_MD("area_set_shape","area","shape_idx","shape"),&Physics2DServer::area_get_shape);
	ObjectTypeDB::bind_method(_MD("area_set_shape_transform","area","shape_idx","transform"),&Physics2DServer::area_set_shape_transform);

	ObjectTypeDB::bind_method(_MD("area_get_shape_count","area"),&Physics2DServer::area_get_shape_count);
	ObjectTypeDB::bind_method(_MD("area_get_shape","area","shape_idx"),&Physics2DServer::area_get_shape);
	ObjectTypeDB::bind_method(_MD("area_get_shape_transform","area","shape_idx"),&Physics2DServer::area_get_shape_transform);

	ObjectTypeDB::bind_method(_MD("area_remove_shape","area","shape_idx"),&Physics2DServer::area_remove_shape);
	ObjectTypeDB::bind_method(_MD("area_clear_shapes","area"),&Physics2DServer::area_clear_shapes);


	ObjectTypeDB::bind_method(_MD("area_set_param","area","param","value"),&Physics2DServer::area_set_param);
	ObjectTypeDB::bind_method(_MD("area_set_transform","area","transform"),&Physics2DServer::area_set_transform);

	ObjectTypeDB::bind_method(_MD("area_get_param","area","param"),&Physics2DServer::area_get_param);
	ObjectTypeDB::bind_method(_MD("area_get_transform","area"),&Physics2DServer::area_get_transform);

	ObjectTypeDB::bind_method(_MD("area_attach_object_instance_ID","area","id"),&Physics2DServer::area_attach_object_instance_ID);
	ObjectTypeDB::bind_method(_MD("area_get_object_instance_ID","area"),&Physics2DServer::area_get_object_instance_ID);

	ObjectTypeDB::bind_method(_MD("area_set_monitor_callback","receiver","method"),&Physics2DServer::area_set_monitor_callback);

	ObjectTypeDB::bind_method(_MD("body_create","mode","init_sleeping"),&Physics2DServer::body_create,DEFVAL(BODY_MODE_RIGID),DEFVAL(false));

	ObjectTypeDB::bind_method(_MD("body_set_space","body","space"),&Physics2DServer::body_set_space);
	ObjectTypeDB::bind_method(_MD("body_get_space","body"),&Physics2DServer::body_get_space);

	ObjectTypeDB::bind_method(_MD("body_set_mode","body","mode"),&Physics2DServer::body_set_mode);
	ObjectTypeDB::bind_method(_MD("body_get_mode","body"),&Physics2DServer::body_get_mode);

	ObjectTypeDB::bind_method(_MD("body_add_shape","body","shape","transform"),&Physics2DServer::body_add_shape,DEFVAL(Matrix32()));
	ObjectTypeDB::bind_method(_MD("body_set_shape","body","shape_idx","shape"),&Physics2DServer::body_set_shape);
	ObjectTypeDB::bind_method(_MD("body_set_shape_transform","body","shape_idx","transform"),&Physics2DServer::body_set_shape_transform);

	ObjectTypeDB::bind_method(_MD("body_get_shape_count","body"),&Physics2DServer::body_get_shape_count);
	ObjectTypeDB::bind_method(_MD("body_get_shape","body","shape_idx"),&Physics2DServer::body_get_shape);
	ObjectTypeDB::bind_method(_MD("body_get_shape_transform","body","shape_idx"),&Physics2DServer::body_get_shape_transform);

	ObjectTypeDB::bind_method(_MD("body_remove_shape","body","shape_idx"),&Physics2DServer::body_remove_shape);
	ObjectTypeDB::bind_method(_MD("body_clear_shapes","body"),&Physics2DServer::body_clear_shapes);


	ObjectTypeDB::bind_method(_MD("body_set_shape_as_trigger","body","shape_idx","enable"),&Physics2DServer::body_set_shape_as_trigger);
	ObjectTypeDB::bind_method(_MD("body_is_shape_set_as_trigger","body","shape_idx"),&Physics2DServer::body_is_shape_set_as_trigger);

	ObjectTypeDB::bind_method(_MD("body_attach_object_instance_ID","body","id"),&Physics2DServer::body_attach_object_instance_ID);
	ObjectTypeDB::bind_method(_MD("body_get_object_instance_ID","body"),&Physics2DServer::body_get_object_instance_ID);


	ObjectTypeDB::bind_method(_MD("body_set_continuous_collision_detection_mode","body","mode"),&Physics2DServer::body_set_continuous_collision_detection_mode);
	ObjectTypeDB::bind_method(_MD("body_get_continuous_collision_detection_mode","body"),&Physics2DServer::body_get_continuous_collision_detection_mode);


	ObjectTypeDB::bind_method(_MD("body_set_layer_mask","body","mask"),&Physics2DServer::body_set_layer_mask);
	ObjectTypeDB::bind_method(_MD("body_get_layer_mask","body"),&Physics2DServer::body_get_layer_mask);

	ObjectTypeDB::bind_method(_MD("body_set_user_mask","body","mask"),&Physics2DServer::body_set_user_mask);
	ObjectTypeDB::bind_method(_MD("body_get_user_mask","body"),&Physics2DServer::body_get_user_mask);


	ObjectTypeDB::bind_method(_MD("body_set_param","body","param","value"),&Physics2DServer::body_set_param);
	ObjectTypeDB::bind_method(_MD("body_get_param","body","param"),&Physics2DServer::body_get_param);

	ObjectTypeDB::bind_method(_MD("body_set_state","body","state","value"),&Physics2DServer::body_set_state);
	ObjectTypeDB::bind_method(_MD("body_get_state","body","state"),&Physics2DServer::body_get_state);

	ObjectTypeDB::bind_method(_MD("body_apply_impulse","body","pos","impulse"),&Physics2DServer::body_apply_impulse);
	ObjectTypeDB::bind_method(_MD("body_set_axis_velocity","body","axis_velocity"),&Physics2DServer::body_set_axis_velocity);

	ObjectTypeDB::bind_method(_MD("body_add_collision_exception","body","excepted_body"),&Physics2DServer::body_add_collision_exception);
	ObjectTypeDB::bind_method(_MD("body_remove_collision_exception","body","excepted_body"),&Physics2DServer::body_remove_collision_exception);
//	virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions)=0;

	ObjectTypeDB::bind_method(_MD("body_set_max_contacts_reported","body","amount"),&Physics2DServer::body_set_max_contacts_reported);
	ObjectTypeDB::bind_method(_MD("body_get_max_contacts_reported","body"),&Physics2DServer::body_get_max_contacts_reported);

	ObjectTypeDB::bind_method(_MD("body_set_omit_force_integration","body","enable"),&Physics2DServer::body_set_omit_force_integration);
	ObjectTypeDB::bind_method(_MD("body_is_omitting_force_integration","body"),&Physics2DServer::body_is_omitting_force_integration);

	ObjectTypeDB::bind_method(_MD("body_set_force_integration_callback","body","receiver","method"),&Physics2DServer::body_set_force_integration_callback);

	/* JOINT API */

	ObjectTypeDB::bind_method(_MD("joint_set_param","joint","param","value"),&Physics2DServer::joint_set_param);
	ObjectTypeDB::bind_method(_MD("joint_get_param","joint","param"),&Physics2DServer::joint_get_param);

	ObjectTypeDB::bind_method(_MD("pin_joint_create","anchor","body_a","body_b"),&Physics2DServer::pin_joint_create,DEFVAL(RID()));
	ObjectTypeDB::bind_method(_MD("groove_joint_create","groove1_a","groove2_a","anchor_b","body_a","body_b"),&Physics2DServer::groove_joint_create,DEFVAL(RID()),DEFVAL(RID()));
	ObjectTypeDB::bind_method(_MD("damped_spring_joint_create","anchor_a","anchor_b","body_a","body_b"),&Physics2DServer::damped_spring_joint_create,DEFVAL(RID()));

	ObjectTypeDB::bind_method(_MD("damped_string_joint_set_param","joint","param","value"),&Physics2DServer::damped_string_joint_set_param,DEFVAL(RID()));
	ObjectTypeDB::bind_method(_MD("damped_string_joint_get_param","joint","param"),&Physics2DServer::damped_string_joint_get_param);

	ObjectTypeDB::bind_method(_MD("joint_get_type","joint"),&Physics2DServer::joint_get_type);

	ObjectTypeDB::bind_method(_MD("free","rid"),&Physics2DServer::free);

	ObjectTypeDB::bind_method(_MD("set_active","active"),&Physics2DServer::set_active);

//	ObjectTypeDB::bind_method(_MD("init"),&Physics2DServer::init);
//	ObjectTypeDB::bind_method(_MD("step"),&Physics2DServer::step);
//	ObjectTypeDB::bind_method(_MD("sync"),&Physics2DServer::sync);
	//ObjectTypeDB::bind_method(_MD("flush_queries"),&Physics2DServer::flush_queries);

	BIND_CONSTANT( SHAPE_LINE );
	BIND_CONSTANT( SHAPE_SEGMENT );
	BIND_CONSTANT( SHAPE_CIRCLE );
	BIND_CONSTANT( SHAPE_RECTANGLE );
	BIND_CONSTANT( SHAPE_CAPSULE );
	BIND_CONSTANT( SHAPE_CONVEX_POLYGON );
	BIND_CONSTANT( SHAPE_CONCAVE_POLYGON );
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

	BIND_CONSTANT( JOINT_PIN );
	BIND_CONSTANT( JOINT_GROOVE );
	BIND_CONSTANT( JOINT_DAMPED_SPRING );

	BIND_CONSTANT( DAMPED_STRING_REST_LENGTH );
	BIND_CONSTANT( DAMPED_STRING_STIFFNESS );
	BIND_CONSTANT( DAMPED_STRING_DAMPING );

	BIND_CONSTANT( CCD_MODE_DISABLED );
	BIND_CONSTANT( CCD_MODE_CAST_RAY );
	BIND_CONSTANT( CCD_MODE_CAST_SHAPE );

//	BIND_CONSTANT( TYPE_BODY );
//	BIND_CONSTANT( TYPE_AREA );

	BIND_CONSTANT( AREA_BODY_ADDED );
	BIND_CONSTANT( AREA_BODY_REMOVED );


}


Physics2DServer::Physics2DServer() {

	ERR_FAIL_COND( singleton!=NULL );
	singleton=this;
}

Physics2DServer::~Physics2DServer() {

	singleton=NULL;
}

