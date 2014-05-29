/*************************************************************************/
/*  physics_server_sw.cpp                                                */
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
#include "physics_server_sw.h"
#include "broad_phase_basic.h"
#include "broad_phase_octree.h"

RID PhysicsServerSW::shape_create(ShapeType p_shape) {

	ShapeSW *shape=NULL;
	switch(p_shape) {

		case SHAPE_PLANE: {

			shape=memnew( PlaneShapeSW );
		} break;
		case SHAPE_RAY: {

			shape=memnew( RayShapeSW );
		} break;
		case SHAPE_SPHERE: {

			shape=memnew( SphereShapeSW);
		} break;
		case SHAPE_BOX: {

			shape=memnew( BoxShapeSW);
		} break;
		case SHAPE_CAPSULE: {

			shape=memnew( CapsuleShapeSW );
		} break;
		case SHAPE_CONVEX_POLYGON: {

			shape=memnew( ConvexPolygonShapeSW );
		} break;
		case SHAPE_CONCAVE_POLYGON: {

			shape=memnew( ConcavePolygonShapeSW );
		} break;
		case SHAPE_HEIGHTMAP: {

			shape=memnew( HeightMapShapeSW );
		} break;
		case SHAPE_CUSTOM: {

			ERR_FAIL_V(RID());

		} break;

	}

	RID id = shape_owner.make_rid(shape);
	shape->set_self(id);

	return id;
};

void PhysicsServerSW::shape_set_data(RID p_shape, const Variant& p_data) {

	ShapeSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);
	shape->set_data(p_data);


};


void PhysicsServerSW::shape_set_custom_solver_bias(RID p_shape, real_t p_bias) {

	ShapeSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);
	shape->set_custom_bias(p_bias);

}


PhysicsServer::ShapeType PhysicsServerSW::shape_get_type(RID p_shape) const {

	const ShapeSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape,SHAPE_CUSTOM);
	return shape->get_type();

};

Variant PhysicsServerSW::shape_get_data(RID p_shape) const {

	const ShapeSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape,Variant());
	ERR_FAIL_COND_V(!shape->is_configured(),Variant());
	return shape->get_data();

};

real_t PhysicsServerSW::shape_get_custom_solver_bias(RID p_shape) const {

	const ShapeSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape,0);
	return shape->get_custom_bias();

}


RID PhysicsServerSW::space_create() {

	SpaceSW *space = memnew( SpaceSW );
	RID id = space_owner.make_rid(space);
	space->set_self(id);
	RID area_id = area_create();
	AreaSW *area = area_owner.get(area_id);
	ERR_FAIL_COND_V(!area,RID());
	space->set_default_area(area);
	area->set_space(space);
	area->set_priority(-1);

	return id;
};

void PhysicsServerSW::space_set_active(RID p_space,bool p_active) {

	SpaceSW *space = space_owner.get(p_space);
	ERR_FAIL_COND(!space);
	if (p_active)
		active_spaces.insert(space);
	else
		active_spaces.erase(space);
}

bool PhysicsServerSW::space_is_active(RID p_space) const {

	const SpaceSW *space = space_owner.get(p_space);
	ERR_FAIL_COND_V(!space,false);

	return active_spaces.has(space);

}

void PhysicsServerSW::space_set_param(RID p_space,SpaceParameter p_param, real_t p_value) {

	SpaceSW *space = space_owner.get(p_space);
	ERR_FAIL_COND(!space);

	space->set_param(p_param,p_value);

}

real_t PhysicsServerSW::space_get_param(RID p_space,SpaceParameter p_param) const {

	const SpaceSW *space = space_owner.get(p_space);
	ERR_FAIL_COND_V(!space,0);
	return space->get_param(p_param);
}

PhysicsDirectSpaceState* PhysicsServerSW::space_get_direct_state(RID p_space) {

	SpaceSW *space = space_owner.get(p_space);
	ERR_FAIL_COND_V(!space,NULL);
	if (!doing_sync || space->is_locked()) {

		ERR_EXPLAIN("Space state is inaccesible right now, wait for iteration or fixed process notification.");
		ERR_FAIL_V(NULL);


	}

	return space->get_direct_state();
}

RID PhysicsServerSW::area_create() {

	AreaSW *area = memnew( AreaSW );
	RID rid = area_owner.make_rid(area);
	area->set_self(rid);
	return rid;
};

void PhysicsServerSW::area_set_space(RID p_area, RID p_space) {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);
	SpaceSW *space=NULL;
	if (p_space.is_valid()) {
		space = space_owner.get(p_space);
		ERR_FAIL_COND(!space);
	}

	area->set_space(space);

};

RID PhysicsServerSW::area_get_space(RID p_area) const {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area,RID());

	SpaceSW *space = area->get_space();
	if (!space)
		return RID();
	return space->get_self();
};

void PhysicsServerSW::area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode) {


	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	area->set_space_override_mode(p_mode);
}

PhysicsServer::AreaSpaceOverrideMode PhysicsServerSW::area_get_space_override_mode(RID p_area) const {

	const AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area,AREA_SPACE_OVERRIDE_DISABLED);

	return area->get_space_override_mode();
}


void PhysicsServerSW::area_add_shape(RID p_area, RID p_shape, const Transform& p_transform) {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	ShapeSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);

	area->add_shape(shape,p_transform);

}

void PhysicsServerSW::area_set_shape(RID p_area, int p_shape_idx,RID p_shape) {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	ShapeSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);
	ERR_FAIL_COND(!shape->is_configured());

	area->set_shape(p_shape_idx,shape);

}
void PhysicsServerSW::area_set_shape_transform(RID p_area, int p_shape_idx, const Transform& p_transform) {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	area->set_shape_transform(p_shape_idx,p_transform);
}

int PhysicsServerSW::area_get_shape_count(RID p_area) const {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area,-1);

	return area->get_shape_count();

}
RID PhysicsServerSW::area_get_shape(RID p_area, int p_shape_idx) const {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area,RID());

	ShapeSW *shape = area->get_shape(p_shape_idx);
	ERR_FAIL_COND_V(!shape,RID());

	return shape->get_self();
}
Transform PhysicsServerSW::area_get_shape_transform(RID p_area, int p_shape_idx) const {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area,Transform());

	return area->get_shape_transform(p_shape_idx);
}

void PhysicsServerSW::area_remove_shape(RID p_area, int p_shape_idx) {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	area->remove_shape(p_shape_idx);
}

void PhysicsServerSW::area_clear_shapes(RID p_area) {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	while(area->get_shape_count())
		area->remove_shape(0);

}

void PhysicsServerSW::area_attach_object_instance_ID(RID p_area,ObjectID p_ID) {

	if (space_owner.owns(p_area)) {
		SpaceSW *space=space_owner.get(p_area);
		p_area=space->get_default_area()->get_self();
	}
	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);
	area->set_instance_id(p_ID);

}
ObjectID PhysicsServerSW::area_get_object_instance_ID(RID p_area) const {

	if (space_owner.owns(p_area)) {
		SpaceSW *space=space_owner.get(p_area);
		p_area=space->get_default_area()->get_self();
	}
	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area,0);
	return area->get_instance_id();


}


void PhysicsServerSW::area_set_param(RID p_area,AreaParameter p_param,const Variant& p_value) {

	if (space_owner.owns(p_area)) {
		SpaceSW *space=space_owner.get(p_area);
		p_area=space->get_default_area()->get_self();
	}
	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);
	area->set_param(p_param,p_value);

};


void PhysicsServerSW::area_set_transform(RID p_area, const Transform& p_transform) {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);
	area->set_transform(p_transform);

};

Variant PhysicsServerSW::area_get_param(RID p_area,AreaParameter p_param) const {

	if (space_owner.owns(p_area)) {
		SpaceSW *space=space_owner.get(p_area);
		p_area=space->get_default_area()->get_self();
	}
	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area,Variant());

	return area->get_param(p_param);
};

Transform PhysicsServerSW::area_get_transform(RID p_area) const {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area,Transform());

	return area->get_transform();
};

void PhysicsServerSW::area_set_monitor_callback(RID p_area,Object *p_receiver,const StringName& p_method) {

	AreaSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	area->set_monitor_callback(p_receiver?p_receiver->get_instance_ID():0,p_method);


}


/* BODY API */

RID PhysicsServerSW::body_create(BodyMode p_mode,bool p_init_sleeping) {

	BodySW *body = memnew( BodySW );
	if (p_mode!=BODY_MODE_RIGID)
		body->set_mode(p_mode);
	if (p_init_sleeping)
		body->set_state(BODY_STATE_SLEEPING,p_init_sleeping);
	RID rid = body_owner.make_rid(body);
	body->set_self(rid);
	return rid;
};


void PhysicsServerSW::body_set_space(RID p_body, RID p_space) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	SpaceSW *space=NULL;

	if (p_space.is_valid()) {
		space = space_owner.get(p_space);
		ERR_FAIL_COND(!space);
	}

	if (body->get_space()==space)
		return; //pointles

	body->set_space(space);

};

RID PhysicsServerSW::body_get_space(RID p_body) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,RID());

	SpaceSW *space = body->get_space();
	if (!space)
		return RID();
	return space->get_self();
};


void PhysicsServerSW::body_set_mode(RID p_body, BodyMode p_mode) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_mode(p_mode);
};

PhysicsServer::BodyMode PhysicsServerSW::body_get_mode(RID p_body, BodyMode p_mode) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,BODY_MODE_STATIC);

	return body->get_mode();
};

void PhysicsServerSW::body_add_shape(RID p_body, RID p_shape, const Transform& p_transform) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	ShapeSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);

	body->add_shape(shape,p_transform);

}

void PhysicsServerSW::body_set_shape(RID p_body, int p_shape_idx,RID p_shape) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	ShapeSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);
	ERR_FAIL_COND(!shape->is_configured());

	body->set_shape(p_shape_idx,shape);

}
void PhysicsServerSW::body_set_shape_transform(RID p_body, int p_shape_idx, const Transform& p_transform) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_shape_transform(p_shape_idx,p_transform);
}

int PhysicsServerSW::body_get_shape_count(RID p_body) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,-1);

	return body->get_shape_count();

}
RID PhysicsServerSW::body_get_shape(RID p_body, int p_shape_idx) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,RID());

	ShapeSW *shape = body->get_shape(p_shape_idx);
	ERR_FAIL_COND_V(!shape,RID());

	return shape->get_self();
}

void PhysicsServerSW::body_set_shape_as_trigger(RID p_body, int p_shape_idx,bool p_enable) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

}

bool PhysicsServerSW::body_is_shape_set_as_trigger(RID p_body, int p_shape_idx) const{

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,false);

	// todo ?

	return false;
}


Transform PhysicsServerSW::body_get_shape_transform(RID p_body, int p_shape_idx) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,Transform());

	return body->get_shape_transform(p_shape_idx);
}

void PhysicsServerSW::body_remove_shape(RID p_body, int p_shape_idx) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->remove_shape(p_shape_idx);
}

void PhysicsServerSW::body_clear_shapes(RID p_body) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	while(body->get_shape_count())
		body->remove_shape(0);

}

void PhysicsServerSW::body_set_enable_continuous_collision_detection(RID p_body,bool p_enable) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_continuous_collision_detection(p_enable);

}

bool PhysicsServerSW::body_is_continuous_collision_detection_enabled(RID p_body) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,false);

	return body->is_continuous_collision_detection_enabled();
}

void PhysicsServerSW::body_attach_object_instance_ID(RID p_body,uint32_t p_ID) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_instance_id(p_ID);

};

uint32_t PhysicsServerSW::body_get_object_instance_ID(RID p_body) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,0);

	return body->get_instance_id();
};


void PhysicsServerSW::body_set_user_flags(RID p_body, uint32_t p_flags) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

};

uint32_t PhysicsServerSW::body_get_user_flags(RID p_body, uint32_t p_flags) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,0);

	return 0;
};

void PhysicsServerSW::body_set_param(RID p_body, BodyParameter p_param, float p_value) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_param(p_param,p_value);
};

float PhysicsServerSW::body_get_param(RID p_body, BodyParameter p_param) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,0);

	return body->get_param(p_param);
};


void PhysicsServerSW::body_static_simulate_motion(RID p_body,const Transform& p_new_transform) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	body->simulate_motion(p_new_transform,last_step);

};

void PhysicsServerSW::body_set_state(RID p_body, BodyState p_state, const Variant& p_variant) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_state(p_state,p_variant);
};

Variant PhysicsServerSW::body_get_state(RID p_body, BodyState p_state) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,Variant());

	return body->get_state(p_state);
};


void PhysicsServerSW::body_set_applied_force(RID p_body, const Vector3& p_force) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_applied_force(p_force);
};

Vector3 PhysicsServerSW::body_get_applied_force(RID p_body) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,Vector3());
	return body->get_applied_force();
};

void PhysicsServerSW::body_set_applied_torque(RID p_body, const Vector3& p_torque) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_applied_torque(p_torque);
};

Vector3 PhysicsServerSW::body_get_applied_torque(RID p_body) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,Vector3());

	return body->get_applied_torque();
};

void PhysicsServerSW::body_apply_impulse(RID p_body, const Vector3& p_pos, const Vector3& p_impulse) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->apply_impulse(p_pos,p_impulse);
};

void PhysicsServerSW::body_set_axis_velocity(RID p_body, const Vector3& p_axis_velocity) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	Vector3 v = body->get_linear_velocity();
	Vector3 axis = p_axis_velocity.normalized();
	v-=axis*axis.dot(v);
	v+=p_axis_velocity;
	body->set_linear_velocity(v);

};


void PhysicsServerSW::body_set_axis_lock(RID p_body,BodyAxisLock p_lock) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	body->set_axis_lock(p_lock);

}

PhysicsServerSW::BodyAxisLock PhysicsServerSW::body_get_axis_lock(RID p_body) const{

	const BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,BODY_AXIS_LOCK_DISABLED);
	return body->get_axis_lock();

}



void PhysicsServerSW::body_add_collision_exception(RID p_body, RID p_body_b) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->add_exception(p_body_b);

};

void PhysicsServerSW::body_remove_collision_exception(RID p_body, RID p_body_b) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->remove_exception(p_body);

};

void PhysicsServerSW::body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	for(int i=0;i<body->get_exceptions().size();i++) {
		p_exceptions->push_back(body->get_exceptions()[i]);
	}

};

void PhysicsServerSW::body_set_contacts_reported_depth_treshold(RID p_body, float p_treshold) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

};

float PhysicsServerSW::body_get_contacts_reported_depth_treshold(RID p_body) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,0);
	return 0;
};

void PhysicsServerSW::body_set_omit_force_integration(RID p_body,bool p_omit) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_omit_force_integration(p_omit);
};

bool PhysicsServerSW::body_is_omitting_force_integration(RID p_body) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,false);
	return body->get_omit_force_integration();
};

void PhysicsServerSW::body_set_max_contacts_reported(RID p_body, int p_contacts) {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	body->set_max_contacts_reported(p_contacts);
}

int PhysicsServerSW::body_get_max_contacts_reported(RID p_body) const {

	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body,-1);
	return body->get_max_contacts_reported();
}

void PhysicsServerSW::body_set_force_integration_callback(RID p_body,Object *p_receiver,const StringName& p_method,const Variant& p_udata) {


	BodySW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	body->set_force_integration_callback(p_receiver?p_receiver->get_instance_ID():ObjectID(0),p_method,p_udata);

}


/* JOINT API */

#if 0
void PhysicsServerSW::joint_set_param(RID p_joint, JointParam p_param, real_t p_value) {

	JointSW *joint = joint_owner.get(p_joint);
	ERR_FAIL_COND(!joint);

	switch(p_param) {
		case JOINT_PARAM_BIAS: joint->set_bias(p_value); break;
		case JOINT_PARAM_MAX_BIAS: joint->set_max_bias(p_value); break;
		case JOINT_PARAM_MAX_FORCE: joint->set_max_force(p_value); break;
	}


}

real_t PhysicsServerSW::joint_get_param(RID p_joint,JointParam p_param) const {

	const JointSW *joint = joint_owner.get(p_joint);
	ERR_FAIL_COND_V(!joint,-1);

	switch(p_param) {
		case JOINT_PARAM_BIAS: return joint->get_bias(); break;
		case JOINT_PARAM_MAX_BIAS: return joint->get_max_bias(); break;
		case JOINT_PARAM_MAX_FORCE: return joint->get_max_force(); break;
	}

	return 0;
}


RID PhysicsServerSW::pin_joint_create(const Vector3& p_pos,RID p_body_a,RID p_body_b) {

	BodySW *A=body_owner.get(p_body_a);
	ERR_FAIL_COND_V(!A,RID());
	BodySW *B=NULL;
	if (body_owner.owns(p_body_b)) {
		B=body_owner.get(p_body_b);
		ERR_FAIL_COND_V(!B,RID());
	}

	JointSW *joint = memnew( PinJointSW(p_pos,A,B) );
	RID self = joint_owner.make_rid(joint);
	joint->set_self(self);

	return self;
}

RID PhysicsServerSW::groove_joint_create(const Vector3& p_a_groove1,const Vector3& p_a_groove2, const Vector3& p_b_anchor, RID p_body_a,RID p_body_b) {


	BodySW *A=body_owner.get(p_body_a);
	ERR_FAIL_COND_V(!A,RID());

	BodySW *B=body_owner.get(p_body_b);
	ERR_FAIL_COND_V(!B,RID());

	JointSW *joint = memnew( GrooveJointSW(p_a_groove1,p_a_groove2,p_b_anchor,A,B) );
	RID self = joint_owner.make_rid(joint);
	joint->set_self(self);
	return self;


}

RID PhysicsServerSW::damped_spring_joint_create(const Vector3& p_anchor_a,const Vector3& p_anchor_b,RID p_body_a,RID p_body_b) {

	BodySW *A=body_owner.get(p_body_a);
	ERR_FAIL_COND_V(!A,RID());

	BodySW *B=body_owner.get(p_body_b);
	ERR_FAIL_COND_V(!B,RID());

	JointSW *joint = memnew( DampedSpringJointSW(p_anchor_a,p_anchor_b,A,B) );
	RID self = joint_owner.make_rid(joint);
	joint->set_self(self);
	return self;

}

void PhysicsServerSW::damped_string_joint_set_param(RID p_joint, DampedStringParam p_param, real_t p_value) {


	JointSW *j = joint_owner.get(p_joint);
	ERR_FAIL_COND(!j);
	ERR_FAIL_COND(j->get_type()!=JOINT_DAMPED_SPRING);

	DampedSpringJointSW *dsj = static_cast<DampedSpringJointSW*>(j);
	dsj->set_param(p_param,p_value);
}

real_t PhysicsServerSW::damped_string_joint_get_param(RID p_joint, DampedStringParam p_param) const {

	JointSW *j = joint_owner.get(p_joint);
	ERR_FAIL_COND_V(!j,0);
	ERR_FAIL_COND_V(j->get_type()!=JOINT_DAMPED_SPRING,0);

	DampedSpringJointSW *dsj = static_cast<DampedSpringJointSW*>(j);
	return dsj->get_param(p_param);
}

PhysicsServer::JointType PhysicsServerSW::joint_get_type(RID p_joint) const {


	JointSW *joint = joint_owner.get(p_joint);
	ERR_FAIL_COND_V(!joint,JOINT_PIN);

	return joint->get_type();
}

#endif

void PhysicsServerSW::free(RID p_rid) {

	if (shape_owner.owns(p_rid)) {

		ShapeSW *shape = shape_owner.get(p_rid);

		while(shape->get_owners().size()) {
			ShapeOwnerSW *so=shape->get_owners().front()->key();
			so->remove_shape(shape);
		}

		shape_owner.free(p_rid);
		memdelete(shape);
	} else if (body_owner.owns(p_rid)) {

		BodySW *body = body_owner.get(p_rid);

//		if (body->get_state_query())
//			_clear_query(body->get_state_query());

//		if (body->get_direct_state_query())
//			_clear_query(body->get_direct_state_query());

		body->set_space(NULL);


		while( body->get_shape_count() ) {

			body->remove_shape(0);
		}

		while (body->get_constraint_map().size()) {
			RID self = body->get_constraint_map().front()->key()->get_self();
			ERR_FAIL_COND(!self.is_valid());
			free(self);
		}

		body_owner.free(p_rid);
		memdelete(body);

	} else if (area_owner.owns(p_rid)) {

		AreaSW *area = area_owner.get(p_rid);

//		if (area->get_monitor_query())
//			_clear_query(area->get_monitor_query());

		area->set_space(NULL);

		while( area->get_shape_count() ) {

			area->remove_shape(0);
		}

		area_owner.free(p_rid);
		memdelete(area);
	} else if (space_owner.owns(p_rid)) {

		SpaceSW *space = space_owner.get(p_rid);

		while(space->get_objects().size()) {
			CollisionObjectSW *co = (CollisionObjectSW *)space->get_objects().front()->get();
			co->set_space(NULL);
		}

		active_spaces.erase(space);
		free(space->get_default_area()->get_self());
		space_owner.free(p_rid);
		memdelete(space);
	} else if (joint_owner.owns(p_rid)) {

		JointSW *joint = joint_owner.get(p_rid);

		joint_owner.free(p_rid);
		memdelete(joint);

	} else {

		ERR_EXPLAIN("Invalid ID");
		ERR_FAIL();
	}


};

void PhysicsServerSW::set_active(bool p_active) {

	active=p_active;
};

void PhysicsServerSW::init() {

	doing_sync=true;
	last_step=0.001;
	iterations=8;// 8?
	stepper = memnew( StepSW );
	direct_state = memnew( PhysicsDirectBodyStateSW );
};


void PhysicsServerSW::step(float p_step) {


	if (!active)
		return;


	doing_sync=false;

	last_step=p_step;
	PhysicsDirectBodyStateSW::singleton->step=p_step;

	for( Set<const SpaceSW*>::Element *E=active_spaces.front();E;E=E->next()) {

		stepper->step((SpaceSW*)E->get(),p_step,iterations);
	}
};

void PhysicsServerSW::sync() {

};

void PhysicsServerSW::flush_queries() {

	if (!active)
		return;

	doing_sync=true;
	for( Set<const SpaceSW*>::Element *E=active_spaces.front();E;E=E->next()) {

		SpaceSW *space=(SpaceSW *)E->get();
		space->call_queries();
	}

};



void PhysicsServerSW::finish() {

	memdelete(stepper);
	memdelete(direct_state);
};


PhysicsServerSW::PhysicsServerSW() {

	BroadPhaseSW::create_func=BroadPhaseOctree::_create;

	active=true;

};

PhysicsServerSW::~PhysicsServerSW() {

};


