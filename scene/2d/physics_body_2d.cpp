/*************************************************************************/
/*  physics_body_2d.cpp                                                  */
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
#include "physics_body_2d.h"
#include "scene/scene_string_names.h"

void PhysicsBody2D::_notification(int p_what) {

/*
	switch(p_what) {

		case NOTIFICATION_TRANSFORM_CHANGED: {

			Physics2DServer::get_singleton()->body_set_state(get_rid(),Physics2DServer::BODY_STATE_TRANSFORM,get_global_transform());

		} break;
	}
	*/
}

void PhysicsBody2D::set_one_way_collision_direction(const Vector2& p_dir) {

	one_way_collision_direction=p_dir;
	Physics2DServer::get_singleton()->body_set_one_way_collision_direction(get_rid(),p_dir);
}

Vector2 PhysicsBody2D::get_one_way_collision_direction() const{

	return one_way_collision_direction;
}


void PhysicsBody2D::set_one_way_collision_max_depth(float p_depth) {

	one_way_collision_max_depth=p_depth;
	Physics2DServer::get_singleton()->body_set_one_way_collision_max_depth(get_rid(),p_depth);

}

float PhysicsBody2D::get_one_way_collision_max_depth() const{

	return one_way_collision_max_depth;
}


void PhysicsBody2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_layer_mask","mask"),&PhysicsBody2D::set_layer_mask);
	ObjectTypeDB::bind_method(_MD("get_layer_mask"),&PhysicsBody2D::get_layer_mask);
	ObjectTypeDB::bind_method(_MD("set_one_way_collision_direction","dir"),&PhysicsBody2D::set_one_way_collision_direction);
	ObjectTypeDB::bind_method(_MD("get_one_way_collision_direction"),&PhysicsBody2D::get_one_way_collision_direction);
	ObjectTypeDB::bind_method(_MD("set_one_way_collision_max_depth","depth"),&PhysicsBody2D::set_one_way_collision_max_depth);
	ObjectTypeDB::bind_method(_MD("get_one_way_collision_max_depth"),&PhysicsBody2D::get_one_way_collision_max_depth);
	ObjectTypeDB::bind_method(_MD("add_collision_exception_with","body:PhysicsBody2D"),&PhysicsBody2D::add_collision_exception_with);
	ObjectTypeDB::bind_method(_MD("remove_collision_exception_with","body:PhysicsBody2D"),&PhysicsBody2D::remove_collision_exception_with);
	ADD_PROPERTY(PropertyInfo(Variant::INT,"layers",PROPERTY_HINT_ALL_FLAGS),_SCS("set_layer_mask"),_SCS("get_layer_mask"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::VECTOR2,"one_way_collision/direction"),_SCS("set_one_way_collision_direction"),_SCS("get_one_way_collision_direction"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::REAL,"one_way_collision/max_depth"),_SCS("set_one_way_collision_max_depth"),_SCS("get_one_way_collision_max_depth"));
}

void PhysicsBody2D::set_layer_mask(uint32_t p_mask) {

	mask=p_mask;
	Physics2DServer::get_singleton()->body_set_layer_mask(get_rid(),p_mask);
}

uint32_t PhysicsBody2D::get_layer_mask() const {

	return mask;
}

PhysicsBody2D::PhysicsBody2D(Physics2DServer::BodyMode p_mode) : CollisionObject2D( Physics2DServer::get_singleton()->body_create(p_mode), false) {

	mask=1;
	set_one_way_collision_max_depth(0);

}

void PhysicsBody2D::add_collision_exception_with(Node* p_node) {

	ERR_FAIL_NULL(p_node);
	PhysicsBody2D *physics_body = p_node->cast_to<PhysicsBody2D>();
	if (!physics_body) {
		ERR_EXPLAIN("Collision exception only works between two objects of PhysicsBody type");
	}
	ERR_FAIL_COND(!physics_body);
	Physics2DServer::get_singleton()->body_add_collision_exception(get_rid(),physics_body->get_rid());

}

void PhysicsBody2D::remove_collision_exception_with(Node* p_node) {

	ERR_FAIL_NULL(p_node);
	PhysicsBody2D *physics_body = p_node->cast_to<PhysicsBody2D>();
	if (!physics_body) {
		ERR_EXPLAIN("Collision exception only works between two objects of PhysicsBody type");
	}
	ERR_FAIL_COND(!physics_body);
	Physics2DServer::get_singleton()->body_remove_collision_exception(get_rid(),physics_body->get_rid());
}

void StaticBody2D::set_constant_linear_velocity(const Vector2& p_vel) {

	constant_linear_velocity=p_vel;
	Physics2DServer::get_singleton()->body_set_state(get_rid(),Physics2DServer::BODY_STATE_LINEAR_VELOCITY,constant_linear_velocity);

}

void StaticBody2D::set_constant_angular_velocity(real_t p_vel) {

	constant_angular_velocity=p_vel;
	Physics2DServer::get_singleton()->body_set_state(get_rid(),Physics2DServer::BODY_STATE_ANGULAR_VELOCITY,constant_angular_velocity);
}

Vector2 StaticBody2D::get_constant_linear_velocity() const {

	return constant_linear_velocity;
}
real_t StaticBody2D::get_constant_angular_velocity() const {

	return constant_angular_velocity;
}
#if 0
void StaticBody2D::_update_xform() {

	if (!pre_xform || !pending)
		return;

	setting=true;


	Matrix32 new_xform = get_global_transform(); //obtain the new one

	set_block_transform_notify(true);
	Physics2DServer::get_singleton()->body_set_state(get_rid(),Physics2DServer::BODY_STATE_TRANSFORM,*pre_xform); //then simulate motion!
	set_global_transform(*pre_xform); //but restore state to previous one in both visual and physics
	set_block_transform_notify(false);

	Physics2DServer::get_singleton()->body_static_simulate_motion(get_rid(),new_xform); //then simulate motion!

	setting=false;
	pending=false;

}
#endif


void StaticBody2D::set_friction(real_t p_friction){

	ERR_FAIL_COND(p_friction<0 || p_friction>1);

	friction=p_friction;
	Physics2DServer::get_singleton()->body_set_param(get_rid(),Physics2DServer::BODY_PARAM_FRICTION,friction);

}
real_t StaticBody2D::get_friction() const{

	return friction;
}

void StaticBody2D::set_bounce(real_t p_bounce){

	ERR_FAIL_COND(p_bounce<0 || p_bounce>1);

	bounce=p_bounce;
	Physics2DServer::get_singleton()->body_set_param(get_rid(),Physics2DServer::BODY_PARAM_BOUNCE,bounce);

}
real_t StaticBody2D::get_bounce() const{

	return bounce;
}

void StaticBody2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_constant_linear_velocity","vel"),&StaticBody2D::set_constant_linear_velocity);
	ObjectTypeDB::bind_method(_MD("set_constant_angular_velocity","vel"),&StaticBody2D::set_constant_angular_velocity);
	ObjectTypeDB::bind_method(_MD("get_constant_linear_velocity"),&StaticBody2D::get_constant_linear_velocity);
	ObjectTypeDB::bind_method(_MD("get_constant_angular_velocity"),&StaticBody2D::get_constant_angular_velocity);
	ObjectTypeDB::bind_method(_MD("set_friction","friction"),&StaticBody2D::set_friction);
	ObjectTypeDB::bind_method(_MD("get_friction"),&StaticBody2D::get_friction);

	ObjectTypeDB::bind_method(_MD("set_bounce","bounce"),&StaticBody2D::set_bounce);
	ObjectTypeDB::bind_method(_MD("get_bounce"),&StaticBody2D::get_bounce);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2,"constant_linear_velocity"),_SCS("set_constant_linear_velocity"),_SCS("get_constant_linear_velocity"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"constant_angular_velocity"),_SCS("set_constant_angular_velocity"),_SCS("get_constant_angular_velocity"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"friction",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_friction"),_SCS("get_friction"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"bounce",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_bounce"),_SCS("get_bounce"));
}

StaticBody2D::StaticBody2D() : PhysicsBody2D(Physics2DServer::BODY_MODE_STATIC) {

	constant_angular_velocity=0;
	bounce=0;
	friction=1;


}

StaticBody2D::~StaticBody2D() {

}




void RigidBody2D::_body_enter_tree(ObjectID p_id) {

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

void RigidBody2D::_body_exit_tree(ObjectID p_id) {

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

void RigidBody2D::_body_inout(int p_status, ObjectID p_instance, int p_body_shape,int p_local_shape) {

	bool body_in = p_status==1;
	ObjectID objid=p_instance;

	Object *obj = ObjectDB::get_instance(objid);
	Node *node = obj ? obj->cast_to<Node>() : NULL;

	Map<ObjectID,BodyState>::Element *E=contact_monitor->body_map.find(objid);

	ERR_FAIL_COND(!body_in && !E);

	if (body_in) {
		if (!E) {

			E = contact_monitor->body_map.insert(objid,BodyState());
//			E->get().rc=0;
			E->get().in_scene=node && node->is_inside_tree();
			if (node) {
				node->connect(SceneStringNames::get_singleton()->enter_tree,this,SceneStringNames::get_singleton()->_body_enter_tree,make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->exit_tree,this,SceneStringNames::get_singleton()->_body_exit_tree,make_binds(objid));
				if (E->get().in_scene) {
					emit_signal(SceneStringNames::get_singleton()->body_enter,node);
				}
			}

			//E->get().rc++;
		}

		if (node)
			E->get().shapes.insert(ShapePair(p_body_shape,p_local_shape));


		if (E->get().in_scene) {
			emit_signal(SceneStringNames::get_singleton()->body_enter_shape,objid,node,p_body_shape,p_local_shape);
		}

	} else {

		//E->get().rc--;

		if (node)
			E->get().shapes.erase(ShapePair(p_body_shape,p_local_shape));

		bool in_scene = E->get().in_scene;

		if (E->get().shapes.empty()) {

			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->enter_tree,this,SceneStringNames::get_singleton()->_body_enter_tree);
				node->disconnect(SceneStringNames::get_singleton()->exit_tree,this,SceneStringNames::get_singleton()->_body_exit_tree);
				if (in_scene)
					emit_signal(SceneStringNames::get_singleton()->body_exit,obj);

			}

			contact_monitor->body_map.erase(E);
		}
		if (node && in_scene) {
			emit_signal(SceneStringNames::get_singleton()->body_exit_shape,objid,obj,p_body_shape,p_local_shape);
		}

	}

}


struct _RigidBody2DInOut {

	ObjectID id;
	int shape;
	int local_shape;
};

void RigidBody2D::_direct_state_changed(Object *p_state) {

	//eh.. fuck
#ifdef DEBUG_ENABLED

	state=p_state->cast_to<Physics2DDirectBodyState>();
#else
	state=(Physics2DDirectBodyState*)p_state; //trust it
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

		_RigidBody2DInOut *toadd=(_RigidBody2DInOut*)alloca(state->get_contact_count()*sizeof(_RigidBody2DInOut));
		int toadd_count=0;//state->get_contact_count();
		RigidBody2D_RemoveAction *toremove=(RigidBody2D_RemoveAction*)alloca(rc*sizeof(RigidBody2D_RemoveAction));
		int toremove_count=0;

		//put the ones to add

		for(int i=0;i<state->get_contact_count();i++) {

			ObjectID obj = state->get_contact_collider_id(i);
			int local_shape = state->get_contact_local_shape(i);
			int shape = state->get_contact_collider_shape(i);
			toadd[i].local_shape=local_shape;
			toadd[i].id=obj;
			toadd[i].shape=shape;

//			bool found=false;

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

	set_block_transform_notify(true); // don't want notify (would feedback loop)
	if (mode!=MODE_KINEMATIC)
		set_global_transform(state->get_transform());
	linear_velocity=state->get_linear_velocity();
	angular_velocity=state->get_angular_velocity();
	sleeping=state->is_sleeping();
	if (get_script_instance())
		get_script_instance()->call("_integrate_forces",state);
	set_block_transform_notify(false); // want it back

	state=NULL;
}


void RigidBody2D::set_mode(Mode p_mode) {

	mode=p_mode;
	switch(p_mode) {

		case MODE_RIGID: {

			Physics2DServer::get_singleton()->body_set_mode(get_rid(),Physics2DServer::BODY_MODE_RIGID);
		} break;
		case MODE_STATIC: {

			Physics2DServer::get_singleton()->body_set_mode(get_rid(),Physics2DServer::BODY_MODE_STATIC);

		} break;
		case MODE_KINEMATIC: {

			Physics2DServer::get_singleton()->body_set_mode(get_rid(),Physics2DServer::BODY_MODE_KINEMATIC);

		} break;
		case MODE_CHARACTER: {
			Physics2DServer::get_singleton()->body_set_mode(get_rid(),Physics2DServer::BODY_MODE_CHARACTER);

		} break;

	}
}

RigidBody2D::Mode RigidBody2D::get_mode() const{

	return mode;
}

void RigidBody2D::set_mass(real_t p_mass){

	ERR_FAIL_COND(p_mass<=0);
	mass=p_mass;
	_change_notify("mass");
	_change_notify("weight");
	Physics2DServer::get_singleton()->body_set_param(get_rid(),Physics2DServer::BODY_PARAM_MASS,mass);

}
real_t RigidBody2D::get_mass() const{

	return mass;
}

void RigidBody2D::set_weight(real_t p_weight){

	set_mass(p_weight/9.8);
}
real_t RigidBody2D::get_weight() const{

	return mass*9.8;
}


void RigidBody2D::set_friction(real_t p_friction){

	ERR_FAIL_COND(p_friction<0 || p_friction>1);

	friction=p_friction;
	Physics2DServer::get_singleton()->body_set_param(get_rid(),Physics2DServer::BODY_PARAM_FRICTION,friction);

}
real_t RigidBody2D::get_friction() const{

	return friction;
}

void RigidBody2D::set_bounce(real_t p_bounce){

	ERR_FAIL_COND(p_bounce<0 || p_bounce>1);

	bounce=p_bounce;
	Physics2DServer::get_singleton()->body_set_param(get_rid(),Physics2DServer::BODY_PARAM_BOUNCE,bounce);

}
real_t RigidBody2D::get_bounce() const{

	return bounce;
}


void RigidBody2D::set_gravity_scale(real_t p_gravity_scale){

	gravity_scale=p_gravity_scale;
	Physics2DServer::get_singleton()->body_set_param(get_rid(),Physics2DServer::BODY_PARAM_GRAVITY_SCALE,gravity_scale);

}
real_t RigidBody2D::get_gravity_scale() const{

	return gravity_scale;
}

void RigidBody2D::set_linear_damp(real_t p_linear_damp){

	ERR_FAIL_COND(p_linear_damp<-1);
	linear_damp=p_linear_damp;
	Physics2DServer::get_singleton()->body_set_param(get_rid(),Physics2DServer::BODY_PARAM_LINEAR_DAMP,linear_damp);

}
real_t RigidBody2D::get_linear_damp() const{

	return linear_damp;
}

void RigidBody2D::set_angular_damp(real_t p_angular_damp){

	ERR_FAIL_COND(p_angular_damp<-1);
	angular_damp=p_angular_damp;
	Physics2DServer::get_singleton()->body_set_param(get_rid(),Physics2DServer::BODY_PARAM_ANGULAR_DAMP,angular_damp);

}
real_t RigidBody2D::get_angular_damp() const{

	return angular_damp;
}

void RigidBody2D::set_axis_velocity(const Vector2& p_axis) {

	Vector2 v = state? state->get_linear_velocity() : linear_velocity;
	Vector2 axis = p_axis.normalized();
	v-=axis*axis.dot(v);
	v+=p_axis;
	if (state) {
		set_linear_velocity(v);
	} else {
		Physics2DServer::get_singleton()->body_set_axis_velocity(get_rid(),p_axis);
		linear_velocity=v;
	}
}

void RigidBody2D::set_linear_velocity(const Vector2& p_velocity){

	linear_velocity=p_velocity;
	if (state)
		state->set_linear_velocity(linear_velocity);
	else {

		Physics2DServer::get_singleton()->body_set_state(get_rid(),Physics2DServer::BODY_STATE_LINEAR_VELOCITY,linear_velocity);
	}

}

Vector2 RigidBody2D::get_linear_velocity() const{

	return linear_velocity;
}

void RigidBody2D::set_angular_velocity(real_t p_velocity){

	angular_velocity=p_velocity;
	if (state)
		state->set_angular_velocity(angular_velocity);
	else
		Physics2DServer::get_singleton()->body_set_state(get_rid(),Physics2DServer::BODY_STATE_ANGULAR_VELOCITY,angular_velocity);
}
real_t RigidBody2D::get_angular_velocity() const{

	return angular_velocity;
}

void RigidBody2D::set_use_custom_integrator(bool p_enable){

	if (custom_integrator==p_enable)
		return;

	custom_integrator=p_enable;
	Physics2DServer::get_singleton()->body_set_omit_force_integration(get_rid(),p_enable);


}
bool RigidBody2D::is_using_custom_integrator(){

	return custom_integrator;
}

void RigidBody2D::set_sleeping(bool p_sleeping) {

	sleeping=p_sleeping;
	Physics2DServer::get_singleton()->body_set_state(get_rid(),Physics2DServer::BODY_STATE_SLEEPING,sleeping);

}

void RigidBody2D::set_can_sleep(bool p_active) {

	can_sleep=p_active;
	Physics2DServer::get_singleton()->body_set_state(get_rid(),Physics2DServer::BODY_STATE_CAN_SLEEP,p_active);
}

bool RigidBody2D::is_able_to_sleep() const {

	return can_sleep;
}

bool RigidBody2D::is_sleeping() const {

	return sleeping;
}

void RigidBody2D::set_max_contacts_reported(int p_amount) {

	max_contacts_reported=p_amount;
	Physics2DServer::get_singleton()->body_set_max_contacts_reported(get_rid(),p_amount);
}

int RigidBody2D::get_max_contacts_reported() const{

	return max_contacts_reported;
}

void RigidBody2D::apply_impulse(const Vector2& p_pos, const Vector2& p_impulse) {

	Physics2DServer::get_singleton()->body_apply_impulse(get_rid(),p_pos,p_impulse);
}

void RigidBody2D::set_applied_force(const Vector2& p_force) {

	Physics2DServer::get_singleton()->body_set_applied_force(get_rid(), p_force);
};

Vector2 RigidBody2D::get_applied_force() const {

	return Physics2DServer::get_singleton()->body_get_applied_force(get_rid());
};


void RigidBody2D::set_continuous_collision_detection_mode(CCDMode p_mode) {

	ccd_mode=p_mode;
	Physics2DServer::get_singleton()->body_set_continuous_collision_detection_mode(get_rid(),Physics2DServer::CCDMode(p_mode));

}

RigidBody2D::CCDMode RigidBody2D::get_continuous_collision_detection_mode() const {

	return ccd_mode;
}


Array RigidBody2D::get_colliding_bodies() const {

	ERR_FAIL_COND_V(!contact_monitor,Array());

	Array ret;
	ret.resize(contact_monitor->body_map.size());
	int idx=0;
	for (const Map<ObjectID,BodyState>::Element *E=contact_monitor->body_map.front();E;E=E->next()) {
		Object *obj = ObjectDB::get_instance(E->key());
		if (!obj) {
			ret.resize( ret.size() -1 ); //ops
		} else {
			ret[idx++]=obj;
		}

	}

	return ret;
}

void RigidBody2D::set_contact_monitor(bool p_enabled) {

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

bool RigidBody2D::is_contact_monitor_enabled() const {

	return contact_monitor!=NULL;
}



void RigidBody2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_mode","mode"),&RigidBody2D::set_mode);
	ObjectTypeDB::bind_method(_MD("get_mode"),&RigidBody2D::get_mode);

	ObjectTypeDB::bind_method(_MD("set_mass","mass"),&RigidBody2D::set_mass);
	ObjectTypeDB::bind_method(_MD("get_mass"),&RigidBody2D::get_mass);

	ObjectTypeDB::bind_method(_MD("set_weight","weight"),&RigidBody2D::set_weight);
	ObjectTypeDB::bind_method(_MD("get_weight"),&RigidBody2D::get_weight);

	ObjectTypeDB::bind_method(_MD("set_friction","friction"),&RigidBody2D::set_friction);
	ObjectTypeDB::bind_method(_MD("get_friction"),&RigidBody2D::get_friction);

	ObjectTypeDB::bind_method(_MD("set_bounce","bounce"),&RigidBody2D::set_bounce);
	ObjectTypeDB::bind_method(_MD("get_bounce"),&RigidBody2D::get_bounce);

	ObjectTypeDB::bind_method(_MD("set_gravity_scale","gravity_scale"),&RigidBody2D::set_gravity_scale);
	ObjectTypeDB::bind_method(_MD("get_gravity_scale"),&RigidBody2D::get_gravity_scale);

	ObjectTypeDB::bind_method(_MD("set_linear_damp","linear_damp"),&RigidBody2D::set_linear_damp);
	ObjectTypeDB::bind_method(_MD("get_linear_damp"),&RigidBody2D::get_linear_damp);

	ObjectTypeDB::bind_method(_MD("set_angular_damp","angular_damp"),&RigidBody2D::set_angular_damp);
	ObjectTypeDB::bind_method(_MD("get_angular_damp"),&RigidBody2D::get_angular_damp);

	ObjectTypeDB::bind_method(_MD("set_linear_velocity","linear_velocity"),&RigidBody2D::set_linear_velocity);
	ObjectTypeDB::bind_method(_MD("get_linear_velocity"),&RigidBody2D::get_linear_velocity);

	ObjectTypeDB::bind_method(_MD("set_angular_velocity","angular_velocity"),&RigidBody2D::set_angular_velocity);
	ObjectTypeDB::bind_method(_MD("get_angular_velocity"),&RigidBody2D::get_angular_velocity);

	ObjectTypeDB::bind_method(_MD("set_max_contacts_reported","amount"),&RigidBody2D::set_max_contacts_reported);
	ObjectTypeDB::bind_method(_MD("get_max_contacts_reported"),&RigidBody2D::get_max_contacts_reported);

	ObjectTypeDB::bind_method(_MD("set_use_custom_integrator","enable"),&RigidBody2D::set_use_custom_integrator);
	ObjectTypeDB::bind_method(_MD("is_using_custom_integrator"),&RigidBody2D::is_using_custom_integrator);

	ObjectTypeDB::bind_method(_MD("set_contact_monitor","enabled"),&RigidBody2D::set_contact_monitor);
	ObjectTypeDB::bind_method(_MD("is_contact_monitor_enabled"),&RigidBody2D::is_contact_monitor_enabled);

	ObjectTypeDB::bind_method(_MD("set_continuous_collision_detection_mode","mode"),&RigidBody2D::set_continuous_collision_detection_mode);
	ObjectTypeDB::bind_method(_MD("get_continuous_collision_detection_mode"),&RigidBody2D::get_continuous_collision_detection_mode);

	ObjectTypeDB::bind_method(_MD("set_axis_velocity","axis_velocity"),&RigidBody2D::set_axis_velocity);
	ObjectTypeDB::bind_method(_MD("apply_impulse","pos","impulse"),&RigidBody2D::apply_impulse);

	ObjectTypeDB::bind_method(_MD("set_applied_force","force"),&RigidBody2D::set_applied_force);
	ObjectTypeDB::bind_method(_MD("get_applied_force"),&RigidBody2D::get_applied_force);

	ObjectTypeDB::bind_method(_MD("set_sleeping","sleeping"),&RigidBody2D::set_sleeping);
	ObjectTypeDB::bind_method(_MD("is_sleeping"),&RigidBody2D::is_sleeping);

	ObjectTypeDB::bind_method(_MD("set_can_sleep","able_to_sleep"),&RigidBody2D::set_can_sleep);
	ObjectTypeDB::bind_method(_MD("is_able_to_sleep"),&RigidBody2D::is_able_to_sleep);

	ObjectTypeDB::bind_method(_MD("_direct_state_changed"),&RigidBody2D::_direct_state_changed);
	ObjectTypeDB::bind_method(_MD("_body_enter_tree"),&RigidBody2D::_body_enter_tree);
	ObjectTypeDB::bind_method(_MD("_body_exit_tree"),&RigidBody2D::_body_exit_tree);

	ObjectTypeDB::bind_method(_MD("get_colliding_bodies"),&RigidBody2D::get_colliding_bodies);

	BIND_VMETHOD(MethodInfo("_integrate_forces",PropertyInfo(Variant::OBJECT,"state:Physics2DDirectBodyState")));

	ADD_PROPERTY( PropertyInfo(Variant::INT,"mode",PROPERTY_HINT_ENUM,"Rigid,Static,Character,Kinematic"),_SCS("set_mode"),_SCS("get_mode"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"mass",PROPERTY_HINT_EXP_RANGE,"0.01,65535,0.01"),_SCS("set_mass"),_SCS("get_mass"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"weight",PROPERTY_HINT_EXP_RANGE,"0.01,65535,0.01",PROPERTY_USAGE_EDITOR),_SCS("set_weight"),_SCS("get_weight"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"friction",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_friction"),_SCS("get_friction"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"bounce",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_bounce"),_SCS("get_bounce"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"gravity_scale",PROPERTY_HINT_RANGE,"-128,128,0.01"),_SCS("set_gravity_scale"),_SCS("get_gravity_scale"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"custom_integrator"),_SCS("set_use_custom_integrator"),_SCS("is_using_custom_integrator"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"continuous_cd",PROPERTY_HINT_ENUM,"Disabled,Cast Ray,Cast Shape"),_SCS("set_continuous_collision_detection_mode"),_SCS("get_continuous_collision_detection_mode"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"contacts_reported"),_SCS("set_max_contacts_reported"),_SCS("get_max_contacts_reported"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"contact_monitor"),_SCS("set_contact_monitor"),_SCS("is_contact_monitor_enabled"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"sleeping"),_SCS("set_sleeping"),_SCS("is_sleeping"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"can_sleep"),_SCS("set_can_sleep"),_SCS("is_able_to_sleep"));
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2,"velocity/linear"),_SCS("set_linear_velocity"),_SCS("get_linear_velocity"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"velocity/angular"),_SCS("set_angular_velocity"),_SCS("get_angular_velocity"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"damp_override/linear",PROPERTY_HINT_RANGE,"-1,128,0.01"),_SCS("set_linear_damp"),_SCS("get_linear_damp"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"damp_override/angular",PROPERTY_HINT_RANGE,"-1,128,0.01"),_SCS("set_angular_damp"),_SCS("get_angular_damp"));

	ADD_SIGNAL( MethodInfo("body_enter_shape",PropertyInfo(Variant::INT,"body_id"),PropertyInfo(Variant::OBJECT,"body"),PropertyInfo(Variant::INT,"body_shape"),PropertyInfo(Variant::INT,"local_shape")));
	ADD_SIGNAL( MethodInfo("body_exit_shape",PropertyInfo(Variant::INT,"body_id"),PropertyInfo(Variant::OBJECT,"body"),PropertyInfo(Variant::INT,"body_shape"),PropertyInfo(Variant::INT,"local_shape")));
	ADD_SIGNAL( MethodInfo("body_enter",PropertyInfo(Variant::OBJECT,"body")));
	ADD_SIGNAL( MethodInfo("body_exit",PropertyInfo(Variant::OBJECT,"body")));

	BIND_CONSTANT( MODE_STATIC );
	BIND_CONSTANT( MODE_KINEMATIC );
	BIND_CONSTANT( MODE_RIGID );
	BIND_CONSTANT( MODE_CHARACTER );

	BIND_CONSTANT( CCD_MODE_DISABLED );
	BIND_CONSTANT( CCD_MODE_CAST_RAY );
	BIND_CONSTANT( CCD_MODE_CAST_SHAPE );

}

RigidBody2D::RigidBody2D() : PhysicsBody2D(Physics2DServer::BODY_MODE_RIGID) {

	mode=MODE_RIGID;

	bounce=0;
	mass=1;
	friction=1;

	gravity_scale=1;
	linear_damp=-1;
	angular_damp=-1;

	max_contacts_reported=0;
	state=NULL;

	angular_velocity=0;
	sleeping=false;
	ccd_mode=CCD_MODE_DISABLED;

	custom_integrator=false;
	contact_monitor=NULL;
	can_sleep=true;

	Physics2DServer::get_singleton()->body_set_force_integration_callback(get_rid(),this,"_direct_state_changed");
}

RigidBody2D::~RigidBody2D() {

	if (contact_monitor)
		memdelete( contact_monitor );



}

//////////////////////////


Variant KinematicBody2D::_get_collider() const {

	ObjectID oid=get_collider();
	if (oid==0)
		return Variant();
	Object *obj = ObjectDB::get_instance(oid);
	if (!obj)
		return Variant();

	Reference *ref = obj->cast_to<Reference>();
	if (ref) {
		return Ref<Reference>(ref);
	}

	return obj;
}


bool KinematicBody2D::_ignores_mode(Physics2DServer::BodyMode p_mode) const {

	switch(p_mode) {
		case Physics2DServer::BODY_MODE_STATIC: return !collide_static;
		case Physics2DServer::BODY_MODE_KINEMATIC: return !collide_kinematic;
		case Physics2DServer::BODY_MODE_RIGID: return !collide_rigid;
		case Physics2DServer::BODY_MODE_CHARACTER: return !collide_character;
	}

	return true;
}

Vector2 KinematicBody2D::move(const Vector2& p_motion) {

	//give me back regular physics engine logic
	//this is madness
	//and most people using this function will think
	//what it does is simpler than using physics
	//this took about a week to get right..
	//but is it right? who knows at this point..


	colliding=false;
	ERR_FAIL_COND_V(!is_inside_tree(),Vector2());
	Physics2DDirectSpaceState *dss = Physics2DServer::get_singleton()->space_get_direct_state(get_world_2d()->get_space());
	ERR_FAIL_COND_V(!dss,Vector2());
	const int max_shapes=32;
	Vector2 sr[max_shapes*2];
	int res_shapes;

	Set<RID> exclude;
	exclude.insert(get_rid());


	//recover first
	int recover_attempts=4;

	bool collided=false;
	uint32_t mask=0;
	if (collide_static)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_STATIC_BODY;
	if (collide_kinematic)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_KINEMATIC_BODY;
	if (collide_rigid)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_RIGID_BODY;
	if (collide_character)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_CHARACTER_BODY;

//	print_line("motion: "+p_motion+" margin: "+rtos(margin));

	//print_line("margin: "+rtos(margin));
	do {

		//motion recover
		for(int i=0;i<get_shape_count();i++) {

			if (is_shape_set_as_trigger(i))
				continue;
			if (dss->collide_shape(get_shape(i)->get_rid(), get_global_transform() * get_shape_transform(i),Vector2(),margin,sr,max_shapes,res_shapes,exclude,get_layer_mask(),mask))
				collided=true;

		}

		if (!collided)
			break;

		Vector2 recover_motion;

		for(int i=0;i<res_shapes;i++) {

			Vector2 a = sr[i*2+0];
			Vector2 b = sr[i*2+1];

			float d = a.distance_to(b);

			//if (d<margin)
			///	continue;
			recover_motion+=(b-a)*0.4;
		}

		if (recover_motion==Vector2()) {
			collided=false;
			break;
		}

		Matrix32 gt = get_global_transform();
		gt.elements[2]+=recover_motion;
		set_global_transform(gt);

		recover_attempts--;

	} while (recover_attempts);


	//move second
	float safe = 1.0;
	float unsafe = 1.0;
	int best_shape=-1;

	for(int i=0;i<get_shape_count();i++) {

		if (is_shape_set_as_trigger(i))
			continue;

		float lsafe,lunsafe;
		bool valid = dss->cast_motion(get_shape(i)->get_rid(), get_global_transform() * get_shape_transform(i), p_motion, 0,lsafe,lunsafe,exclude,get_layer_mask(),mask);
		//print_line("shape: "+itos(i)+" travel:"+rtos(ltravel));
		if (!valid) {

			safe=0;
			unsafe=0;
			best_shape=i; //sadly it's the best
			break;
		}
		if (lsafe==1.0) {
			continue;
		}
		if (lsafe < safe) {

			safe=lsafe;
			unsafe=lunsafe;
			best_shape=i;
		}
	}


	//print_line("best shape: "+itos(best_shape)+" motion "+p_motion);

	if (safe>=1) {
		//not collided
		colliding=false;
	} else {

		//it collided, let's get the rest info in unsafe advance
		Matrix32 ugt = get_global_transform();
		ugt.elements[2]+=p_motion*unsafe;
		Physics2DDirectSpaceState::ShapeRestInfo rest_info;
		bool c2 = dss->rest_info(get_shape(best_shape)->get_rid(), ugt*get_shape_transform(best_shape), Vector2(), margin,&rest_info,exclude,get_layer_mask(),mask);
		if (!c2) {
			//should not happen, but floating point precision is so weird..

			colliding=false;
		} else {


			//print_line("Travel: "+rtos(travel));
			colliding=true;
			collision=rest_info.point;
			normal=rest_info.normal;
			collider=rest_info.collider_id;
			collider_vel=rest_info.linear_velocity;
			collider_shape=rest_info.shape;
			collider_metadata=rest_info.metadata;
		}

	}

	Vector2 motion=p_motion*safe;
	Matrix32 gt = get_global_transform();
	gt.elements[2]+=motion;
	set_global_transform(gt);

	return p_motion-motion;

}

Vector2 KinematicBody2D::move_to(const Vector2& p_position) {

	return move(p_position-get_global_pos());
}

bool KinematicBody2D::can_move_to(const Vector2& p_position, bool p_discrete) {

	ERR_FAIL_COND_V(!is_inside_tree(),false);
	Physics2DDirectSpaceState *dss = Physics2DServer::get_singleton()->space_get_direct_state(get_world_2d()->get_space());
	ERR_FAIL_COND_V(!dss,false);

	uint32_t mask=0;
	if (collide_static)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_STATIC_BODY;
	if (collide_kinematic)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_KINEMATIC_BODY;
	if (collide_rigid)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_RIGID_BODY;
	if (collide_character)
		mask|=Physics2DDirectSpaceState::TYPE_MASK_CHARACTER_BODY;

	Vector2 motion = p_position-get_global_pos();
	Matrix32 xform=get_global_transform();

	if (p_discrete) {

		xform.elements[2]+=motion;
		motion=Vector2();
	}

	Set<RID> exclude;
	exclude.insert(get_rid());

	//fill exclude list..
	for(int i=0;i<get_shape_count();i++) {


		bool col = dss->intersect_shape(get_shape(i)->get_rid(), xform * get_shape_transform(i),motion,0,NULL,0,exclude,get_layer_mask(),mask);
		if (col)
			return false;
	}

	return true;
}

bool KinematicBody2D::is_colliding() const {

	ERR_FAIL_COND_V(!is_inside_tree(),false);

	return colliding;
}
Vector2 KinematicBody2D::get_collision_pos() const {

	ERR_FAIL_COND_V(!colliding,Vector2());
	return collision;

}
Vector2 KinematicBody2D::get_collision_normal() const {

	ERR_FAIL_COND_V(!colliding,Vector2());
	return normal;

}

Vector2 KinematicBody2D::get_collider_velocity() const {

	return collider_vel;
}

ObjectID KinematicBody2D::get_collider() const {

	ERR_FAIL_COND_V(!colliding,0);
	return collider;
}


int KinematicBody2D::get_collider_shape() const {

	ERR_FAIL_COND_V(!colliding,0);
	return collider_shape;
}

Variant KinematicBody2D::get_collider_metadata() const {

	ERR_FAIL_COND_V(!colliding,0);
	return collider_metadata;

}

void KinematicBody2D::set_collide_with_static_bodies(bool p_enable) {

	collide_static=p_enable;
}
bool KinematicBody2D::can_collide_with_static_bodies() const {

	return collide_static;
}

void KinematicBody2D::set_collide_with_rigid_bodies(bool p_enable) {

	collide_rigid=p_enable;

}
bool KinematicBody2D::can_collide_with_rigid_bodies() const {


	return collide_rigid;
}

void KinematicBody2D::set_collide_with_kinematic_bodies(bool p_enable) {

	collide_kinematic=p_enable;

}
bool KinematicBody2D::can_collide_with_kinematic_bodies() const {

	return collide_kinematic;
}

void KinematicBody2D::set_collide_with_character_bodies(bool p_enable) {

	collide_character=p_enable;
}
bool KinematicBody2D::can_collide_with_character_bodies() const {

	return collide_character;
}

void KinematicBody2D::set_collision_margin(float p_margin) {

	margin=p_margin;
}

float KinematicBody2D::get_collision_margin() const{

	return margin;
}

void KinematicBody2D::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("move","rel_vec"),&KinematicBody2D::move);
	ObjectTypeDB::bind_method(_MD("move_to","position"),&KinematicBody2D::move_to);

	ObjectTypeDB::bind_method(_MD("can_move_to","position"),&KinematicBody2D::can_move_to);

	ObjectTypeDB::bind_method(_MD("is_colliding"),&KinematicBody2D::is_colliding);

	ObjectTypeDB::bind_method(_MD("get_collision_pos"),&KinematicBody2D::get_collision_pos);
	ObjectTypeDB::bind_method(_MD("get_collision_normal"),&KinematicBody2D::get_collision_normal);
	ObjectTypeDB::bind_method(_MD("get_collider_velocity"),&KinematicBody2D::get_collider_velocity);
	ObjectTypeDB::bind_method(_MD("get_collider:Object"),&KinematicBody2D::_get_collider);
	ObjectTypeDB::bind_method(_MD("get_collider_shape"),&KinematicBody2D::get_collider_shape);
	ObjectTypeDB::bind_method(_MD("get_collider_metadata"),&KinematicBody2D::get_collider_metadata);


	ObjectTypeDB::bind_method(_MD("set_collide_with_static_bodies","enable"),&KinematicBody2D::set_collide_with_static_bodies);
	ObjectTypeDB::bind_method(_MD("can_collide_with_static_bodies"),&KinematicBody2D::can_collide_with_static_bodies);

	ObjectTypeDB::bind_method(_MD("set_collide_with_kinematic_bodies","enable"),&KinematicBody2D::set_collide_with_kinematic_bodies);
	ObjectTypeDB::bind_method(_MD("can_collide_with_kinematic_bodies"),&KinematicBody2D::can_collide_with_kinematic_bodies);

	ObjectTypeDB::bind_method(_MD("set_collide_with_rigid_bodies","enable"),&KinematicBody2D::set_collide_with_rigid_bodies);
	ObjectTypeDB::bind_method(_MD("can_collide_with_rigid_bodies"),&KinematicBody2D::can_collide_with_rigid_bodies);

	ObjectTypeDB::bind_method(_MD("set_collide_with_character_bodies","enable"),&KinematicBody2D::set_collide_with_character_bodies);
	ObjectTypeDB::bind_method(_MD("can_collide_with_character_bodies"),&KinematicBody2D::can_collide_with_character_bodies);

	ObjectTypeDB::bind_method(_MD("set_collision_margin","pixels"),&KinematicBody2D::set_collision_margin);
	ObjectTypeDB::bind_method(_MD("get_collision_margin","pixels"),&KinematicBody2D::get_collision_margin);

	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"collide_with/static"),_SCS("set_collide_with_static_bodies"),_SCS("can_collide_with_static_bodies"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"collide_with/kinematic"),_SCS("set_collide_with_kinematic_bodies"),_SCS("can_collide_with_kinematic_bodies"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"collide_with/rigid"),_SCS("set_collide_with_rigid_bodies"),_SCS("can_collide_with_rigid_bodies"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"collide_with/character"),_SCS("set_collide_with_character_bodies"),_SCS("can_collide_with_character_bodies"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"collision/margin",PROPERTY_HINT_RANGE,"0.001,256,0.001"),_SCS("set_collision_margin"),_SCS("get_collision_margin"));


}

KinematicBody2D::KinematicBody2D() : PhysicsBody2D(Physics2DServer::BODY_MODE_KINEMATIC){

	collide_static=true;
	collide_rigid=true;
	collide_kinematic=true;
	collide_character=true;

	colliding=false;
	collider=0;

	collider_shape=0;

	margin=0.08;
}
KinematicBody2D::~KinematicBody2D()  {


}

