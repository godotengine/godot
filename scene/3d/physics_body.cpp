/*************************************************************************/
/*  physics_body.cpp                                                     */
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
#include "physics_body.h"
#include "scene/scene_string_names.h"

void PhysicsBody::_notification(int p_what) {

/*
	switch(p_what) {

		case NOTIFICATION_TRANSFORM_CHANGED: {

			PhysicsServer::get_singleton()->body_set_state(get_rid(),PhysicsServer::BODY_STATE_TRANSFORM,get_global_transform());

		} break;
	}
	*/
}

PhysicsBody::PhysicsBody(PhysicsServer::BodyMode p_mode) : CollisionObject( PhysicsServer::get_singleton()->body_create(p_mode), false) {



}

void StaticBody::set_constant_linear_velocity(const Vector3& p_vel) {

	constant_linear_velocity=p_vel;
	PhysicsServer::get_singleton()->body_set_state(get_rid(),PhysicsServer::BODY_STATE_LINEAR_VELOCITY,constant_linear_velocity);

}

void StaticBody::set_constant_angular_velocity(const Vector3& p_vel) {

	constant_angular_velocity=p_vel;
	PhysicsServer::get_singleton()->body_set_state(get_rid(),PhysicsServer::BODY_STATE_ANGULAR_VELOCITY,constant_angular_velocity);
}

Vector3 StaticBody::get_constant_linear_velocity() const {

	return constant_linear_velocity;
}
Vector3 StaticBody::get_constant_angular_velocity() const {

	return constant_angular_velocity;
}


void StaticBody::_state_notify(Object *p_object) {

	if (!pre_xform)
		return;

	PhysicsDirectBodyState *p2d = (PhysicsDirectBodyState*)p_object;
	setting=true;

	Transform new_xform = p2d->get_transform();
	*pre_xform=new_xform;
	set_ignore_transform_notification(true);
	set_global_transform(new_xform);
	set_ignore_transform_notification(false);

	setting=false;


}

void StaticBody::_update_xform() {

	if (!pre_xform || !pending)
		return;

	setting=true;


	Transform new_xform = get_global_transform(); //obtain the new one

	//set_block_transform_notify(true);
	set_ignore_transform_notification(true);
	PhysicsServer::get_singleton()->body_set_state(get_rid(),PhysicsServer::BODY_STATE_TRANSFORM,*pre_xform); //then simulate motion!
	set_global_transform(*pre_xform); //but restore state to previous one in both visual and physics
	set_ignore_transform_notification(false);

	PhysicsServer::get_singleton()->body_static_simulate_motion(get_rid(),new_xform); //then simulate motion!

	setting=false;
	pending=false;

}

void StaticBody::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_SCENE: {

			if (pre_xform)
				*pre_xform = get_global_transform();
			pending=false;
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (simulating_motion && !pending && is_inside_scene() && !setting && !get_scene()->is_editor_hint()) {

				call_deferred(SceneStringNames::get_singleton()->_update_xform);
				pending=true;
			}

		} break;
	}


}

void StaticBody::set_simulate_motion(bool p_enable) {

	if (p_enable==simulating_motion)
		return;
	simulating_motion=p_enable;

	if (p_enable) {
		pre_xform = memnew( Transform );
		if (is_inside_scene())
			*pre_xform=get_transform();
//		query = PhysicsServer::get_singleton()->query_create(this,"_state_notify",Variant());
	//	PhysicsServer::get_singleton()->query_body_direct_state(query,get_rid());
		PhysicsServer::get_singleton()->body_set_force_integration_callback(get_rid(),this,"_state_notify");

	} else {
		memdelete( pre_xform );
		pre_xform=NULL;
		PhysicsServer::get_singleton()->body_set_force_integration_callback(get_rid(),NULL,StringName());
		pending=false;
	}
}

bool StaticBody::is_simulating_motion() const {

	return simulating_motion;
}

void StaticBody::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_simulate_motion","enabled"),&StaticBody::set_simulate_motion);
	ObjectTypeDB::bind_method(_MD("is_simulating_motion"),&StaticBody::is_simulating_motion);
	ObjectTypeDB::bind_method(_MD("_update_xform"),&StaticBody::_update_xform);
	ObjectTypeDB::bind_method(_MD("_state_notify"),&StaticBody::_state_notify);
	ObjectTypeDB::bind_method(_MD("set_constant_linear_velocity","vel"),&StaticBody::set_constant_linear_velocity);
	ObjectTypeDB::bind_method(_MD("set_constant_angular_velocity","vel"),&StaticBody::set_constant_angular_velocity);
	ObjectTypeDB::bind_method(_MD("get_constant_linear_velocity"),&StaticBody::get_constant_linear_velocity);
	ObjectTypeDB::bind_method(_MD("get_constant_angular_velocity"),&StaticBody::get_constant_angular_velocity);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"simulate_motion"),_SCS("set_simulate_motion"),_SCS("is_simulating_motion"));
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,"constant_linear_velocity"),_SCS("set_constant_linear_velocity"),_SCS("get_constant_linear_velocity"));
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,"constant_angular_velocity"),_SCS("set_constant_angular_velocity"),_SCS("get_constant_angular_velocity"));
}

StaticBody::StaticBody() : PhysicsBody(PhysicsServer::BODY_MODE_STATIC) {

	simulating_motion=false;
	pre_xform=NULL;
	setting=false;
	pending=false;
	//constant_angular_velocity=0;

}

StaticBody::~StaticBody() {

	if (pre_xform)
		memdelete(pre_xform);
	//if (query.is_valid())
	//	PhysicsServer::get_singleton()->free(query);
}




void RigidBody::_body_enter_scene(ObjectID p_id) {

	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = obj ? obj->cast_to<Node>() : NULL;
	ERR_FAIL_COND(!node);

	Map<ObjectID,BodyState>::Element *E=contact_monitor->body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(E->get().in_scene);

	E->get().in_scene=true;
	emit_signal(SceneStringNames::get_singleton()->body_enter,node);

	for(int i=0;i<E->get().shapes.size();i++) {

		emit_signal(SceneStringNames::get_singleton()->body_enter_shape,p_id,node,E->get().shapes[i].body_shape,E->get().shapes[i].local_shape);
	}

}

void RigidBody::_body_exit_scene(ObjectID p_id) {

	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = obj ? obj->cast_to<Node>() : NULL;
	ERR_FAIL_COND(!node);
	Map<ObjectID,BodyState>::Element *E=contact_monitor->body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(!E->get().in_scene);
	E->get().in_scene=false;
	emit_signal(SceneStringNames::get_singleton()->body_exit,node);
	for(int i=0;i<E->get().shapes.size();i++) {

		emit_signal(SceneStringNames::get_singleton()->body_exit_shape,p_id,node,E->get().shapes[i].body_shape,E->get().shapes[i].local_shape);
	}
}

void RigidBody::_body_inout(int p_status, ObjectID p_instance, int p_body_shape,int p_local_shape) {

	bool body_in = p_status==1;
	ObjectID objid=p_instance;

	Object *obj = ObjectDB::get_instance(objid);
	Node *node = obj ? obj->cast_to<Node>() : NULL;

	Map<ObjectID,BodyState>::Element *E=contact_monitor->body_map.find(objid);

	ERR_FAIL_COND(!body_in && !E);

	if (body_in) {
		if (!E) {

			E = contact_monitor->body_map.insert(objid,BodyState());
			E->get().rc=0;
			E->get().in_scene=node && node->is_inside_scene();
			if (node) {
				node->connect(SceneStringNames::get_singleton()->enter_scene,this,SceneStringNames::get_singleton()->_body_enter_scene,make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->exit_scene,this,SceneStringNames::get_singleton()->_body_exit_scene,make_binds(objid));
				if (E->get().in_scene) {
					emit_signal(SceneStringNames::get_singleton()->body_enter,node);
				}
			}

		}
		E->get().rc++;
		if (node)
			E->get().shapes.insert(ShapePair(p_body_shape,p_local_shape));


		if (E->get().in_scene) {
			emit_signal(SceneStringNames::get_singleton()->body_enter_shape,objid,node,p_body_shape,p_local_shape);
		}

	} else {

		E->get().rc--;

		if (node)
			E->get().shapes.erase(ShapePair(p_body_shape,p_local_shape));

		if (E->get().rc==0) {

			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->enter_scene,this,SceneStringNames::get_singleton()->_body_enter_scene);
				node->disconnect(SceneStringNames::get_singleton()->exit_scene,this,SceneStringNames::get_singleton()->_body_exit_scene);
				if (E->get().in_scene)
					emit_signal(SceneStringNames::get_singleton()->body_exit,obj);

			}

			contact_monitor->body_map.erase(E);
		}
		if (node && E->get().in_scene) {
			emit_signal(SceneStringNames::get_singleton()->body_exit_shape,objid,obj,p_body_shape,p_local_shape);
		}

	}

}


struct _RigidBodyInOut {

	ObjectID id;
	int shape;
	int local_shape;
};

void RigidBody::_direct_state_changed(Object *p_state) {

	//eh.. fuck
#ifdef DEBUG_ENABLED

	state=p_state->cast_to<PhysicsDirectBodyState>();
#else
	state=(PhysicsDirectBodyState*)p_state; //trust it
#endif

	if (contact_monitor) {

		//untag all
		int rc=0;
		for( Map<ObjectID,BodyState>::Element *E=contact_monitor->body_map.front();E;E=E->next()) {

			for(int i=0;i<E->get().shapes.size();i++) {

				E->get().shapes[i].tagged=false;
				rc++;
			}
		}

		_RigidBodyInOut *toadd=(_RigidBodyInOut*)alloca(state->get_contact_count()*sizeof(_RigidBodyInOut));
		int toadd_count=0;//state->get_contact_count();
		RigidBody_RemoveAction *toremove=(RigidBody_RemoveAction*)alloca(rc*sizeof(RigidBody_RemoveAction));
		int toremove_count=0;

		//put the ones to add

		for(int i=0;i<state->get_contact_count();i++) {

			ObjectID obj = state->get_contact_collider_id(i);
			int local_shape = state->get_contact_local_shape(i);
			int shape = state->get_contact_collider_shape(i);
			toadd[i].local_shape=local_shape;
			toadd[i].id=obj;
			toadd[i].shape=shape;

			bool found=false;

			Map<ObjectID,BodyState>::Element *E=contact_monitor->body_map.find(obj);
			if (!E) {
				toadd_count++;
				continue;
			}

			ShapePair sp( shape,local_shape );
			int idx = E->get().shapes.find(sp);
			if (idx==-1) {

				toadd_count++;
				continue;
			}

			E->get().shapes[idx].tagged=true;
		}

		//put the ones to remove

		for( Map<ObjectID,BodyState>::Element *E=contact_monitor->body_map.front();E;E=E->next()) {

			for(int i=0;i<E->get().shapes.size();i++) {

				if (!E->get().shapes[i].tagged) {

					toremove[toremove_count].body_id=E->key();
					toremove[toremove_count].pair=E->get().shapes[i];
					toremove_count++;
				}
			}
		}


		//process remotions

		for(int i=0;i<toremove_count;i++) {

			_body_inout(0,toremove[i].body_id,toremove[i].pair.body_shape,toremove[i].pair.local_shape);
		}

		//process aditions

		for(int i=0;i<toadd_count;i++) {

			_body_inout(1,toadd[i].id,toadd[i].shape,toadd[i].local_shape);
		}

	}

	set_ignore_transform_notification(true);
	set_global_transform(state->get_transform());
	linear_velocity=state->get_linear_velocity();
	angular_velocity=state->get_angular_velocity();
	active=!state->is_sleeping();
	if (get_script_instance())
		get_script_instance()->call("_integrate_forces",state);
	set_ignore_transform_notification(false);

	state=NULL;
}

void RigidBody::_notification(int p_what) {


}

void RigidBody::set_mode(Mode p_mode) {

	mode=p_mode;
	switch(p_mode) {

		case MODE_RIGID: {

			PhysicsServer::get_singleton()->body_set_mode(get_rid(),PhysicsServer::BODY_MODE_RIGID);
		} break;
		case MODE_STATIC: {

			PhysicsServer::get_singleton()->body_set_mode(get_rid(),PhysicsServer::BODY_MODE_STATIC);

		} break;
		case MODE_CHARACTER: {
			PhysicsServer::get_singleton()->body_set_mode(get_rid(),PhysicsServer::BODY_MODE_CHARACTER);

		} break;
		case MODE_KINEMATIC: {

			PhysicsServer::get_singleton()->body_set_mode(get_rid(),PhysicsServer::BODY_MODE_KINEMATIC);
		} break;

	}
}

RigidBody::Mode RigidBody::get_mode() const{

	return mode;
}

void RigidBody::set_mass(real_t p_mass){

	ERR_FAIL_COND(p_mass<=0);
	mass=p_mass;
	_change_notify("mass");
	_change_notify("weight");
	PhysicsServer::get_singleton()->body_set_param(get_rid(),PhysicsServer::BODY_PARAM_MASS,mass);

}
real_t RigidBody::get_mass() const{

	return mass;
}

void RigidBody::set_weight(real_t p_weight){

	set_mass(p_weight/9.8);
}
real_t RigidBody::get_weight() const{

	return mass*9.8;
}


void RigidBody::set_friction(real_t p_friction){

	ERR_FAIL_COND(p_friction<0 || p_friction>1);

	friction=p_friction;
	PhysicsServer::get_singleton()->body_set_param(get_rid(),PhysicsServer::BODY_PARAM_FRICTION,friction);

}
real_t RigidBody::get_friction() const{

	return friction;
}

void RigidBody::set_bounce(real_t p_bounce){

	ERR_FAIL_COND(p_bounce<0 || p_bounce>1);

	bounce=p_bounce;
	PhysicsServer::get_singleton()->body_set_param(get_rid(),PhysicsServer::BODY_PARAM_BOUNCE,bounce);

}
real_t RigidBody::get_bounce() const{

	return bounce;
}

void RigidBody::set_axis_velocity(const Vector3& p_axis) {

	Vector3 v = state? state->get_linear_velocity() : linear_velocity;
	Vector3 axis = p_axis.normalized();
	v-=axis*axis.dot(v);
	v+=p_axis;
	if (state) {
		set_linear_velocity(v);
	} else {
		PhysicsServer::get_singleton()->body_set_axis_velocity(get_rid(),p_axis);
		linear_velocity=v;
	}
}

void RigidBody::set_linear_velocity(const Vector3& p_velocity){

	linear_velocity=p_velocity;
	if (state)
		state->set_linear_velocity(linear_velocity);
	else
		PhysicsServer::get_singleton()->body_set_state(get_rid(),PhysicsServer::BODY_STATE_LINEAR_VELOCITY,linear_velocity);

}

Vector3 RigidBody::get_linear_velocity() const{

	return linear_velocity;
}

void RigidBody::set_angular_velocity(const Vector3& p_velocity){

	angular_velocity=p_velocity;
	if (state)
		state->set_angular_velocity(angular_velocity);
	else
		PhysicsServer::get_singleton()->body_set_state(get_rid(),PhysicsServer::BODY_STATE_ANGULAR_VELOCITY,angular_velocity);
}
Vector3 RigidBody::get_angular_velocity() const{

	return angular_velocity;
}

void RigidBody::set_use_custom_integrator(bool p_enable){

	if (custom_integrator==p_enable)
		return;

	custom_integrator=p_enable;
	PhysicsServer::get_singleton()->body_set_omit_force_integration(get_rid(),p_enable);


}
bool RigidBody::is_using_custom_integrator(){

	return custom_integrator;
}

void RigidBody::set_active(bool p_active) {

	active=p_active;
	PhysicsServer::get_singleton()->body_set_state(get_rid(),PhysicsServer::BODY_STATE_SLEEPING,!active);

}

void RigidBody::set_can_sleep(bool p_active) {

	can_sleep=p_active;
	PhysicsServer::get_singleton()->body_set_state(get_rid(),PhysicsServer::BODY_STATE_CAN_SLEEP,p_active);
}

bool RigidBody::is_able_to_sleep() const {

	return can_sleep;
}

bool RigidBody::is_active() const {

	return active;
}

void RigidBody::set_max_contacts_reported(int p_amount) {

	max_contacts_reported=p_amount;
	PhysicsServer::get_singleton()->body_set_max_contacts_reported(get_rid(),p_amount);
}

int RigidBody::get_max_contacts_reported() const{

	return max_contacts_reported;
}

void RigidBody::apply_impulse(const Vector3& p_pos, const Vector3& p_impulse) {

	PhysicsServer::get_singleton()->body_apply_impulse(get_rid(),p_pos,p_impulse);
}

void RigidBody::set_use_continuous_collision_detection(bool p_enable) {

	ccd=p_enable;
	PhysicsServer::get_singleton()->body_set_enable_continuous_collision_detection(get_rid(),p_enable);
}

bool RigidBody::is_using_continuous_collision_detection() const {


	return ccd;
}


void RigidBody::set_contact_monitor(bool p_enabled) {

	if (p_enabled==is_contact_monitor_enabled())
		return;

	if (!p_enabled) {

		for(Map<ObjectID,BodyState>::Element *E=contact_monitor->body_map.front();E;E=E->next()) {

			//clean up mess
		}

		memdelete( contact_monitor );
		contact_monitor=NULL;
	} else {

		contact_monitor = memnew( ContactMonitor );
	}

}

bool RigidBody::is_contact_monitor_enabled() const {

	return contact_monitor!=NULL;
}

void RigidBody::set_axis_lock(AxisLock p_lock) {

	axis_lock=p_lock;
	PhysicsServer::get_singleton()->body_set_axis_lock(get_rid(),PhysicsServer::BodyAxisLock(axis_lock));
}

RigidBody::AxisLock RigidBody::get_axis_lock() const {

	return axis_lock;
}


void RigidBody::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_mode","mode"),&RigidBody::set_mode);
	ObjectTypeDB::bind_method(_MD("get_mode"),&RigidBody::get_mode);

	ObjectTypeDB::bind_method(_MD("set_mass","mass"),&RigidBody::set_mass);
	ObjectTypeDB::bind_method(_MD("get_mass"),&RigidBody::get_mass);

	ObjectTypeDB::bind_method(_MD("set_weight","weight"),&RigidBody::set_weight);
	ObjectTypeDB::bind_method(_MD("get_weight"),&RigidBody::get_weight);

	ObjectTypeDB::bind_method(_MD("set_friction","friction"),&RigidBody::set_friction);
	ObjectTypeDB::bind_method(_MD("get_friction"),&RigidBody::get_friction);

	ObjectTypeDB::bind_method(_MD("set_bounce","bounce"),&RigidBody::set_bounce);
	ObjectTypeDB::bind_method(_MD("get_bounce"),&RigidBody::get_bounce);

	ObjectTypeDB::bind_method(_MD("set_linear_velocity","linear_velocity"),&RigidBody::set_linear_velocity);
	ObjectTypeDB::bind_method(_MD("get_linear_velocity"),&RigidBody::get_linear_velocity);

	ObjectTypeDB::bind_method(_MD("set_angular_velocity","angular_velocity"),&RigidBody::set_angular_velocity);
	ObjectTypeDB::bind_method(_MD("get_angular_velocity"),&RigidBody::get_angular_velocity);

	ObjectTypeDB::bind_method(_MD("set_max_contacts_reported","amount"),&RigidBody::set_max_contacts_reported);
	ObjectTypeDB::bind_method(_MD("get_max_contacts_reported"),&RigidBody::get_max_contacts_reported);

	ObjectTypeDB::bind_method(_MD("set_use_custom_integrator","enable"),&RigidBody::set_use_custom_integrator);
	ObjectTypeDB::bind_method(_MD("is_using_custom_integrator"),&RigidBody::is_using_custom_integrator);

	ObjectTypeDB::bind_method(_MD("set_contact_monitor","enabled"),&RigidBody::set_contact_monitor);
	ObjectTypeDB::bind_method(_MD("is_contact_monitor_enabled"),&RigidBody::is_contact_monitor_enabled);

	ObjectTypeDB::bind_method(_MD("set_use_continuous_collision_detection","enable"),&RigidBody::set_use_continuous_collision_detection);
	ObjectTypeDB::bind_method(_MD("is_using_continuous_collision_detection"),&RigidBody::is_using_continuous_collision_detection);

	ObjectTypeDB::bind_method(_MD("set_axis_velocity","axis_velocity"),&RigidBody::set_axis_velocity);
	ObjectTypeDB::bind_method(_MD("apply_impulse","pos","impulse"),&RigidBody::apply_impulse);

	ObjectTypeDB::bind_method(_MD("set_active","active"),&RigidBody::set_active);
	ObjectTypeDB::bind_method(_MD("is_active"),&RigidBody::is_active);

	ObjectTypeDB::bind_method(_MD("set_can_sleep","able_to_sleep"),&RigidBody::set_can_sleep);
	ObjectTypeDB::bind_method(_MD("is_able_to_sleep"),&RigidBody::is_able_to_sleep);

	ObjectTypeDB::bind_method(_MD("_direct_state_changed"),&RigidBody::_direct_state_changed);
	ObjectTypeDB::bind_method(_MD("_body_enter_scene"),&RigidBody::_body_enter_scene);
	ObjectTypeDB::bind_method(_MD("_body_exit_scene"),&RigidBody::_body_exit_scene);

	ObjectTypeDB::bind_method(_MD("set_axis_lock","axis_lock"),&RigidBody::set_axis_lock);
	ObjectTypeDB::bind_method(_MD("get_axis_lock"),&RigidBody::get_axis_lock);

	BIND_VMETHOD(MethodInfo("_integrate_forces",PropertyInfo(Variant::OBJECT,"state:PhysicsDirectBodyState")));

	ADD_PROPERTY( PropertyInfo(Variant::INT,"mode",PROPERTY_HINT_ENUM,"Rigid,Static,Character,Kinematic"),_SCS("set_mode"),_SCS("get_mode"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"mass",PROPERTY_HINT_EXP_RANGE,"0.01,65535,0.01"),_SCS("set_mass"),_SCS("get_mass"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"weight",PROPERTY_HINT_EXP_RANGE,"0.01,65535,0.01",PROPERTY_USAGE_EDITOR),_SCS("set_weight"),_SCS("get_weight"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"friction",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_friction"),_SCS("get_friction"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"bounce",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_bounce"),_SCS("get_bounce"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"custom_integrator"),_SCS("set_use_custom_integrator"),_SCS("is_using_custom_integrator"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"continuous_cd"),_SCS("set_use_continuous_collision_detection"),_SCS("is_using_continuous_collision_detection"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"contacts_reported"),_SCS("set_max_contacts_reported"),_SCS("get_max_contacts_reported"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"contact_monitor"),_SCS("set_contact_monitor"),_SCS("is_contact_monitor_enabled"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"active"),_SCS("set_active"),_SCS("is_active"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"can_sleep"),_SCS("set_can_sleep"),_SCS("is_able_to_sleep"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"axis_lock",PROPERTY_HINT_ENUM,"Disabled,Lock X,Lock Y,Lock Z"),_SCS("set_axis_lock"),_SCS("get_axis_lock"));
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR3,"velocity/linear"),_SCS("set_linear_velocity"),_SCS("get_linear_velocity"));
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR3,"velocity/angular"),_SCS("set_angular_velocity"),_SCS("get_angular_velocity"));

	ADD_SIGNAL( MethodInfo("body_enter_shape",PropertyInfo(Variant::INT,"body_id"),PropertyInfo(Variant::OBJECT,"body"),PropertyInfo(Variant::INT,"body_shape"),PropertyInfo(Variant::INT,"local_shape")));
	ADD_SIGNAL( MethodInfo("body_exit_shape",PropertyInfo(Variant::INT,"body_id"),PropertyInfo(Variant::OBJECT,"body"),PropertyInfo(Variant::INT,"body_shape"),PropertyInfo(Variant::INT,"local_shape")));
	ADD_SIGNAL( MethodInfo("body_enter",PropertyInfo(Variant::OBJECT,"body")));
	ADD_SIGNAL( MethodInfo("body_exit",PropertyInfo(Variant::OBJECT,"body")));

	BIND_CONSTANT( MODE_STATIC );
	BIND_CONSTANT( MODE_KINEMATIC );
	BIND_CONSTANT( MODE_RIGID );
	BIND_CONSTANT( MODE_CHARACTER );
}

RigidBody::RigidBody() : PhysicsBody(PhysicsServer::BODY_MODE_RIGID) {

	mode=MODE_RIGID;

	bounce=0;
	mass=1;
	friction=1;
	max_contacts_reported=0;
	state=NULL;

	//angular_velocity=0;
	active=true;
	ccd=false;

	custom_integrator=false;
	contact_monitor=NULL;
	can_sleep=true;

	axis_lock = AXIS_LOCK_DISABLED;

	PhysicsServer::get_singleton()->body_set_force_integration_callback(get_rid(),this,"_direct_state_changed");
}

RigidBody::~RigidBody() {

	if (contact_monitor)
		memdelete( contact_monitor );



}

