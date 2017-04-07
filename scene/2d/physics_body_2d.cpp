/*************************************************************************/
/*  physics_body_2d.cpp                                                  */
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

void PhysicsBody2D::set_one_way_collision_direction(const Vector2 &p_dir) {

	one_way_collision_direction = p_dir;
	Physics2DServer::get_singleton()->body_set_one_way_collision_direction(get_rid(), p_dir);
}

Vector2 PhysicsBody2D::get_one_way_collision_direction() const {

	return one_way_collision_direction;
}

void PhysicsBody2D::set_one_way_collision_max_depth(float p_depth) {

	one_way_collision_max_depth = p_depth;
	Physics2DServer::get_singleton()->body_set_one_way_collision_max_depth(get_rid(), p_depth);
}

float PhysicsBody2D::get_one_way_collision_max_depth() const {

	return one_way_collision_max_depth;
}

void PhysicsBody2D::_set_layers(uint32_t p_mask) {

	set_collision_layer(p_mask);
	set_collision_mask(p_mask);
}

uint32_t PhysicsBody2D::_get_layers() const {

	return get_collision_layer();
}

void PhysicsBody2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_collision_layer", "mask"), &PhysicsBody2D::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &PhysicsBody2D::get_collision_layer);
	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &PhysicsBody2D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &PhysicsBody2D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_bit", "bit", "value"), &PhysicsBody2D::set_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("get_collision_mask_bit", "bit"), &PhysicsBody2D::get_collision_mask_bit);

	ClassDB::bind_method(D_METHOD("set_collision_layer_bit", "bit", "value"), &PhysicsBody2D::set_collision_layer_bit);
	ClassDB::bind_method(D_METHOD("get_collision_layer_bit", "bit"), &PhysicsBody2D::get_collision_layer_bit);

	ClassDB::bind_method(D_METHOD("_set_layers", "mask"), &PhysicsBody2D::_set_layers);
	ClassDB::bind_method(D_METHOD("_get_layers"), &PhysicsBody2D::_get_layers);
	ClassDB::bind_method(D_METHOD("set_one_way_collision_direction", "dir"), &PhysicsBody2D::set_one_way_collision_direction);
	ClassDB::bind_method(D_METHOD("get_one_way_collision_direction"), &PhysicsBody2D::get_one_way_collision_direction);
	ClassDB::bind_method(D_METHOD("set_one_way_collision_max_depth", "depth"), &PhysicsBody2D::set_one_way_collision_max_depth);
	ClassDB::bind_method(D_METHOD("get_one_way_collision_max_depth"), &PhysicsBody2D::get_one_way_collision_max_depth);
	ClassDB::bind_method(D_METHOD("add_collision_exception_with", "body:PhysicsBody2D"), &PhysicsBody2D::add_collision_exception_with);
	ClassDB::bind_method(D_METHOD("remove_collision_exception_with", "body:PhysicsBody2D"), &PhysicsBody2D::remove_collision_exception_with);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "layers", PROPERTY_HINT_LAYERS_2D_PHYSICS, "", 0), "_set_layers", "_get_layers"); //for backwards compat

	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_GROUP("", "");
	ADD_PROPERTYNZ(PropertyInfo(Variant::VECTOR2, "one_way_collision/direction"), "set_one_way_collision_direction", "get_one_way_collision_direction");
	ADD_PROPERTYNZ(PropertyInfo(Variant::REAL, "one_way_collision/max_depth"), "set_one_way_collision_max_depth", "get_one_way_collision_max_depth");
}

void PhysicsBody2D::set_collision_layer(uint32_t p_mask) {

	mask = p_mask;
	Physics2DServer::get_singleton()->body_set_layer_mask(get_rid(), p_mask);
}

uint32_t PhysicsBody2D::get_collision_layer() const {

	return mask;
}

void PhysicsBody2D::set_collision_mask(uint32_t p_mask) {

	collision_mask = p_mask;
	Physics2DServer::get_singleton()->body_set_collision_mask(get_rid(), p_mask);
}

uint32_t PhysicsBody2D::get_collision_mask() const {

	return collision_mask;
}

void PhysicsBody2D::set_collision_mask_bit(int p_bit, bool p_value) {

	uint32_t mask = get_collision_mask();
	if (p_value)
		mask |= 1 << p_bit;
	else
		mask &= ~(1 << p_bit);
	set_collision_mask(mask);
}
bool PhysicsBody2D::get_collision_mask_bit(int p_bit) const {

	return get_collision_mask() & (1 << p_bit);
}

void PhysicsBody2D::set_collision_layer_bit(int p_bit, bool p_value) {

	uint32_t mask = get_collision_layer();
	if (p_value)
		mask |= 1 << p_bit;
	else
		mask &= ~(1 << p_bit);
	set_collision_layer(mask);
}

bool PhysicsBody2D::get_collision_layer_bit(int p_bit) const {

	return get_collision_layer() & (1 << p_bit);
}

PhysicsBody2D::PhysicsBody2D(Physics2DServer::BodyMode p_mode)
	: CollisionObject2D(Physics2DServer::get_singleton()->body_create(p_mode), false) {

	mask = 1;
	collision_mask = 1;
	set_one_way_collision_max_depth(0);
	set_pickable(false);
}

void PhysicsBody2D::add_collision_exception_with(Node *p_node) {

	ERR_FAIL_NULL(p_node);
	PhysicsBody2D *physics_body = p_node->cast_to<PhysicsBody2D>();
	if (!physics_body) {
		ERR_EXPLAIN("Collision exception only works between two objects of PhysicsBody type");
	}
	ERR_FAIL_COND(!physics_body);
	Physics2DServer::get_singleton()->body_add_collision_exception(get_rid(), physics_body->get_rid());
}

void PhysicsBody2D::remove_collision_exception_with(Node *p_node) {

	ERR_FAIL_NULL(p_node);
	PhysicsBody2D *physics_body = p_node->cast_to<PhysicsBody2D>();
	if (!physics_body) {
		ERR_EXPLAIN("Collision exception only works between two objects of PhysicsBody type");
	}
	ERR_FAIL_COND(!physics_body);
	Physics2DServer::get_singleton()->body_remove_collision_exception(get_rid(), physics_body->get_rid());
}

void StaticBody2D::set_constant_linear_velocity(const Vector2 &p_vel) {

	constant_linear_velocity = p_vel;
	Physics2DServer::get_singleton()->body_set_state(get_rid(), Physics2DServer::BODY_STATE_LINEAR_VELOCITY, constant_linear_velocity);
}

void StaticBody2D::set_constant_angular_velocity(real_t p_vel) {

	constant_angular_velocity = p_vel;
	Physics2DServer::get_singleton()->body_set_state(get_rid(), Physics2DServer::BODY_STATE_ANGULAR_VELOCITY, constant_angular_velocity);
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


	Transform2D new_xform = get_global_transform(); //obtain the new one

	set_block_transform_notify(true);
	Physics2DServer::get_singleton()->body_set_state(get_rid(),Physics2DServer::BODY_STATE_TRANSFORM,*pre_xform); //then simulate motion!
	set_global_transform(*pre_xform); //but restore state to previous one in both visual and physics
	set_block_transform_notify(false);

	Physics2DServer::get_singleton()->body_static_simulate_motion(get_rid(),new_xform); //then simulate motion!

	setting=false;
	pending=false;

}
#endif

void StaticBody2D::set_friction(real_t p_friction) {

	ERR_FAIL_COND(p_friction < 0 || p_friction > 1);

	friction = p_friction;
	Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_FRICTION, friction);
}
real_t StaticBody2D::get_friction() const {

	return friction;
}

void StaticBody2D::set_bounce(real_t p_bounce) {

	ERR_FAIL_COND(p_bounce < 0 || p_bounce > 1);

	bounce = p_bounce;
	Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_BOUNCE, bounce);
}
real_t StaticBody2D::get_bounce() const {

	return bounce;
}

void StaticBody2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_constant_linear_velocity", "vel"), &StaticBody2D::set_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_constant_angular_velocity", "vel"), &StaticBody2D::set_constant_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_linear_velocity"), &StaticBody2D::get_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_angular_velocity"), &StaticBody2D::get_constant_angular_velocity);
	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &StaticBody2D::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &StaticBody2D::get_friction);

	ClassDB::bind_method(D_METHOD("set_bounce", "bounce"), &StaticBody2D::set_bounce);
	ClassDB::bind_method(D_METHOD("get_bounce"), &StaticBody2D::get_bounce);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "constant_linear_velocity"), "set_constant_linear_velocity", "get_constant_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "constant_angular_velocity"), "set_constant_angular_velocity", "get_constant_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "friction", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_bounce", "get_bounce");
}

StaticBody2D::StaticBody2D()
	: PhysicsBody2D(Physics2DServer::BODY_MODE_STATIC) {

	constant_angular_velocity = 0;
	bounce = 0;
	friction = 1;
}

StaticBody2D::~StaticBody2D() {
}

void RigidBody2D::_body_enter_tree(ObjectID p_id) {

	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = obj ? obj->cast_to<Node>() : NULL;
	ERR_FAIL_COND(!node);

	Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(E->get().in_scene);

	contact_monitor->locked = true;

	E->get().in_scene = true;
	emit_signal(SceneStringNames::get_singleton()->body_entered, node);

	for (int i = 0; i < E->get().shapes.size(); i++) {

		emit_signal(SceneStringNames::get_singleton()->body_shape_entered, p_id, node, E->get().shapes[i].body_shape, E->get().shapes[i].local_shape);
	}

	contact_monitor->locked = false;
}

void RigidBody2D::_body_exit_tree(ObjectID p_id) {

	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = obj ? obj->cast_to<Node>() : NULL;
	ERR_FAIL_COND(!node);
	Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(!E->get().in_scene);
	E->get().in_scene = false;

	contact_monitor->locked = true;

	emit_signal(SceneStringNames::get_singleton()->body_exited, node);

	for (int i = 0; i < E->get().shapes.size(); i++) {

		emit_signal(SceneStringNames::get_singleton()->body_shape_exited, p_id, node, E->get().shapes[i].body_shape, E->get().shapes[i].local_shape);
	}

	contact_monitor->locked = false;
}

void RigidBody2D::_body_inout(int p_status, ObjectID p_instance, int p_body_shape, int p_local_shape) {

	bool body_in = p_status == 1;
	ObjectID objid = p_instance;

	Object *obj = ObjectDB::get_instance(objid);
	Node *node = obj ? obj->cast_to<Node>() : NULL;

	Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(objid);

	/*if (obj) {
		if (body_in)
			print_line("in: "+String(obj->call("get_name")));
		else
			print_line("out: "+String(obj->call("get_name")));
	}*/

	ERR_FAIL_COND(!body_in && !E);

	if (body_in) {
		if (!E) {

			E = contact_monitor->body_map.insert(objid, BodyState());
			//E->get().rc=0;
			E->get().in_scene = node && node->is_inside_tree();
			if (node) {
				node->connect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree, make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->tree_exited, this, SceneStringNames::get_singleton()->_body_exit_tree, make_binds(objid));
				if (E->get().in_scene) {
					emit_signal(SceneStringNames::get_singleton()->body_entered, node);
				}
			}

			//E->get().rc++;
		}

		if (node)
			E->get().shapes.insert(ShapePair(p_body_shape, p_local_shape));

		if (E->get().in_scene) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_entered, objid, node, p_body_shape, p_local_shape);
		}

	} else {

		//E->get().rc--;

		if (node)
			E->get().shapes.erase(ShapePair(p_body_shape, p_local_shape));

		bool in_scene = E->get().in_scene;

		if (E->get().shapes.empty()) {

			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree);
				node->disconnect(SceneStringNames::get_singleton()->tree_exited, this, SceneStringNames::get_singleton()->_body_exit_tree);
				if (in_scene)
					emit_signal(SceneStringNames::get_singleton()->body_exited, obj);
			}

			contact_monitor->body_map.erase(E);
		}
		if (node && in_scene) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_exited, objid, obj, p_body_shape, p_local_shape);
		}
	}
}

struct _RigidBody2DInOut {

	ObjectID id;
	int shape;
	int local_shape;
};

bool RigidBody2D::_test_motion(const Vector2 &p_motion, float p_margin, const Ref<Physics2DTestMotionResult> &p_result) {

	Physics2DServer::MotionResult *r = NULL;
	if (p_result.is_valid())
		r = p_result->get_result_ptr();
	return Physics2DServer::get_singleton()->body_test_motion(get_rid(), get_global_transform(), p_motion, p_margin, r);
}

void RigidBody2D::_direct_state_changed(Object *p_state) {

//eh.. fuck
#ifdef DEBUG_ENABLED

	state = p_state->cast_to<Physics2DDirectBodyState>();
#else
	state = (Physics2DDirectBodyState *)p_state; //trust it
#endif

	set_block_transform_notify(true); // don't want notify (would feedback loop)
	if (mode != MODE_KINEMATIC)
		set_global_transform(state->get_transform());
	linear_velocity = state->get_linear_velocity();
	angular_velocity = state->get_angular_velocity();
	if (sleeping != state->is_sleeping()) {
		sleeping = state->is_sleeping();
		emit_signal(SceneStringNames::get_singleton()->sleeping_state_changed);
	}
	if (get_script_instance())
		get_script_instance()->call("_integrate_forces", state);
	set_block_transform_notify(false); // want it back

	if (contact_monitor) {

		contact_monitor->locked = true;

		//untag all
		int rc = 0;
		for (Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.front(); E; E = E->next()) {

			for (int i = 0; i < E->get().shapes.size(); i++) {

				E->get().shapes[i].tagged = false;
				rc++;
			}
		}

		_RigidBody2DInOut *toadd = (_RigidBody2DInOut *)alloca(state->get_contact_count() * sizeof(_RigidBody2DInOut));
		int toadd_count = 0; //state->get_contact_count();
		RigidBody2D_RemoveAction *toremove = (RigidBody2D_RemoveAction *)alloca(rc * sizeof(RigidBody2D_RemoveAction));
		int toremove_count = 0;

		//put the ones to add

		for (int i = 0; i < state->get_contact_count(); i++) {

			ObjectID obj = state->get_contact_collider_id(i);
			int local_shape = state->get_contact_local_shape(i);
			int shape = state->get_contact_collider_shape(i);

			//bool found=false;

			Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(obj);
			if (!E) {
				toadd[toadd_count].local_shape = local_shape;
				toadd[toadd_count].id = obj;
				toadd[toadd_count].shape = shape;
				toadd_count++;
				continue;
			}

			ShapePair sp(shape, local_shape);
			int idx = E->get().shapes.find(sp);
			if (idx == -1) {

				toadd[toadd_count].local_shape = local_shape;
				toadd[toadd_count].id = obj;
				toadd[toadd_count].shape = shape;
				toadd_count++;
				continue;
			}

			E->get().shapes[idx].tagged = true;
		}

		//put the ones to remove

		for (Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.front(); E; E = E->next()) {

			for (int i = 0; i < E->get().shapes.size(); i++) {

				if (!E->get().shapes[i].tagged) {

					toremove[toremove_count].body_id = E->key();
					toremove[toremove_count].pair = E->get().shapes[i];
					toremove_count++;
				}
			}
		}

		//process remotions

		for (int i = 0; i < toremove_count; i++) {

			_body_inout(0, toremove[i].body_id, toremove[i].pair.body_shape, toremove[i].pair.local_shape);
		}

		//process aditions

		for (int i = 0; i < toadd_count; i++) {

			_body_inout(1, toadd[i].id, toadd[i].shape, toadd[i].local_shape);
		}

		contact_monitor->locked = false;
	}

	state = NULL;
}

void RigidBody2D::set_mode(Mode p_mode) {

	mode = p_mode;
	switch (p_mode) {

		case MODE_RIGID: {

			Physics2DServer::get_singleton()->body_set_mode(get_rid(), Physics2DServer::BODY_MODE_RIGID);
		} break;
		case MODE_STATIC: {

			Physics2DServer::get_singleton()->body_set_mode(get_rid(), Physics2DServer::BODY_MODE_STATIC);

		} break;
		case MODE_KINEMATIC: {

			Physics2DServer::get_singleton()->body_set_mode(get_rid(), Physics2DServer::BODY_MODE_KINEMATIC);

		} break;
		case MODE_CHARACTER: {
			Physics2DServer::get_singleton()->body_set_mode(get_rid(), Physics2DServer::BODY_MODE_CHARACTER);

		} break;
	}
}

RigidBody2D::Mode RigidBody2D::get_mode() const {

	return mode;
}

void RigidBody2D::set_mass(real_t p_mass) {

	ERR_FAIL_COND(p_mass <= 0);
	mass = p_mass;
	_change_notify("mass");
	_change_notify("weight");
	Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_MASS, mass);
}
real_t RigidBody2D::get_mass() const {

	return mass;
}

void RigidBody2D::set_inertia(real_t p_inertia) {

	ERR_FAIL_COND(p_inertia <= 0);
	Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_INERTIA, p_inertia);
}

real_t RigidBody2D::get_inertia() const {

	return Physics2DServer::get_singleton()->body_get_param(get_rid(), Physics2DServer::BODY_PARAM_INERTIA);
}

void RigidBody2D::set_weight(real_t p_weight) {

	set_mass(p_weight / 9.8);
}
real_t RigidBody2D::get_weight() const {

	return mass * 9.8;
}

void RigidBody2D::set_friction(real_t p_friction) {

	ERR_FAIL_COND(p_friction < 0 || p_friction > 1);

	friction = p_friction;
	Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_FRICTION, friction);
}
real_t RigidBody2D::get_friction() const {

	return friction;
}

void RigidBody2D::set_bounce(real_t p_bounce) {

	ERR_FAIL_COND(p_bounce < 0 || p_bounce > 1);

	bounce = p_bounce;
	Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_BOUNCE, bounce);
}
real_t RigidBody2D::get_bounce() const {

	return bounce;
}

void RigidBody2D::set_gravity_scale(real_t p_gravity_scale) {

	gravity_scale = p_gravity_scale;
	Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_GRAVITY_SCALE, gravity_scale);
}
real_t RigidBody2D::get_gravity_scale() const {

	return gravity_scale;
}

void RigidBody2D::set_linear_damp(real_t p_linear_damp) {

	ERR_FAIL_COND(p_linear_damp < -1);
	linear_damp = p_linear_damp;
	Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_LINEAR_DAMP, linear_damp);
}
real_t RigidBody2D::get_linear_damp() const {

	return linear_damp;
}

void RigidBody2D::set_angular_damp(real_t p_angular_damp) {

	ERR_FAIL_COND(p_angular_damp < -1);
	angular_damp = p_angular_damp;
	Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_ANGULAR_DAMP, angular_damp);
}
real_t RigidBody2D::get_angular_damp() const {

	return angular_damp;
}

void RigidBody2D::set_axis_velocity(const Vector2 &p_axis) {

	Vector2 v = state ? state->get_linear_velocity() : linear_velocity;
	Vector2 axis = p_axis.normalized();
	v -= axis * axis.dot(v);
	v += p_axis;
	if (state) {
		set_linear_velocity(v);
	} else {
		Physics2DServer::get_singleton()->body_set_axis_velocity(get_rid(), p_axis);
		linear_velocity = v;
	}
}

void RigidBody2D::set_linear_velocity(const Vector2 &p_velocity) {

	linear_velocity = p_velocity;
	if (state)
		state->set_linear_velocity(linear_velocity);
	else {

		Physics2DServer::get_singleton()->body_set_state(get_rid(), Physics2DServer::BODY_STATE_LINEAR_VELOCITY, linear_velocity);
	}
}

Vector2 RigidBody2D::get_linear_velocity() const {

	return linear_velocity;
}

void RigidBody2D::set_angular_velocity(real_t p_velocity) {

	angular_velocity = p_velocity;
	if (state)
		state->set_angular_velocity(angular_velocity);
	else
		Physics2DServer::get_singleton()->body_set_state(get_rid(), Physics2DServer::BODY_STATE_ANGULAR_VELOCITY, angular_velocity);
}
real_t RigidBody2D::get_angular_velocity() const {

	return angular_velocity;
}

void RigidBody2D::set_use_custom_integrator(bool p_enable) {

	if (custom_integrator == p_enable)
		return;

	custom_integrator = p_enable;
	Physics2DServer::get_singleton()->body_set_omit_force_integration(get_rid(), p_enable);
}
bool RigidBody2D::is_using_custom_integrator() {

	return custom_integrator;
}

void RigidBody2D::set_sleeping(bool p_sleeping) {

	sleeping = p_sleeping;
	Physics2DServer::get_singleton()->body_set_state(get_rid(), Physics2DServer::BODY_STATE_SLEEPING, sleeping);
}

void RigidBody2D::set_can_sleep(bool p_active) {

	can_sleep = p_active;
	Physics2DServer::get_singleton()->body_set_state(get_rid(), Physics2DServer::BODY_STATE_CAN_SLEEP, p_active);
}

bool RigidBody2D::is_able_to_sleep() const {

	return can_sleep;
}

bool RigidBody2D::is_sleeping() const {

	return sleeping;
}

void RigidBody2D::set_max_contacts_reported(int p_amount) {

	max_contacts_reported = p_amount;
	Physics2DServer::get_singleton()->body_set_max_contacts_reported(get_rid(), p_amount);
}

int RigidBody2D::get_max_contacts_reported() const {

	return max_contacts_reported;
}

void RigidBody2D::apply_impulse(const Vector2 &p_offset, const Vector2 &p_impulse) {

	Physics2DServer::get_singleton()->body_apply_impulse(get_rid(), p_offset, p_impulse);
}

void RigidBody2D::set_applied_force(const Vector2 &p_force) {

	Physics2DServer::get_singleton()->body_set_applied_force(get_rid(), p_force);
};

Vector2 RigidBody2D::get_applied_force() const {

	return Physics2DServer::get_singleton()->body_get_applied_force(get_rid());
};

void RigidBody2D::set_applied_torque(const float p_torque) {

	Physics2DServer::get_singleton()->body_set_applied_torque(get_rid(), p_torque);
};

float RigidBody2D::get_applied_torque() const {

	return Physics2DServer::get_singleton()->body_get_applied_torque(get_rid());
};

void RigidBody2D::add_force(const Vector2 &p_offset, const Vector2 &p_force) {

	Physics2DServer::get_singleton()->body_add_force(get_rid(), p_offset, p_force);
}

void RigidBody2D::set_continuous_collision_detection_mode(CCDMode p_mode) {

	ccd_mode = p_mode;
	Physics2DServer::get_singleton()->body_set_continuous_collision_detection_mode(get_rid(), Physics2DServer::CCDMode(p_mode));
}

RigidBody2D::CCDMode RigidBody2D::get_continuous_collision_detection_mode() const {

	return ccd_mode;
}

Array RigidBody2D::get_colliding_bodies() const {

	ERR_FAIL_COND_V(!contact_monitor, Array());

	Array ret;
	ret.resize(contact_monitor->body_map.size());
	int idx = 0;
	for (const Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.front(); E; E = E->next()) {
		Object *obj = ObjectDB::get_instance(E->key());
		if (!obj) {
			ret.resize(ret.size() - 1); //ops
		} else {
			ret[idx++] = obj;
		}
	}

	return ret;
}

void RigidBody2D::set_contact_monitor(bool p_enabled) {

	if (p_enabled == is_contact_monitor_enabled())
		return;

	if (!p_enabled) {

		if (contact_monitor->locked) {
			ERR_EXPLAIN("Can't disable contact monitoring during in/out callback. Use call_deferred(\"set_contact_monitor\",false) instead");
		}
		ERR_FAIL_COND(contact_monitor->locked);

		for (Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.front(); E; E = E->next()) {

			//clean up mess
		}

		memdelete(contact_monitor);
		contact_monitor = NULL;
	} else {

		contact_monitor = memnew(ContactMonitor);
		contact_monitor->locked = false;
	}
}

bool RigidBody2D::is_contact_monitor_enabled() const {

	return contact_monitor != NULL;
}

void RigidBody2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &RigidBody2D::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &RigidBody2D::get_mode);

	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &RigidBody2D::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &RigidBody2D::get_mass);

	ClassDB::bind_method(D_METHOD("get_inertia"), &RigidBody2D::get_inertia);
	ClassDB::bind_method(D_METHOD("set_inertia", "inertia"), &RigidBody2D::set_inertia);

	ClassDB::bind_method(D_METHOD("set_weight", "weight"), &RigidBody2D::set_weight);
	ClassDB::bind_method(D_METHOD("get_weight"), &RigidBody2D::get_weight);

	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &RigidBody2D::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &RigidBody2D::get_friction);

	ClassDB::bind_method(D_METHOD("set_bounce", "bounce"), &RigidBody2D::set_bounce);
	ClassDB::bind_method(D_METHOD("get_bounce"), &RigidBody2D::get_bounce);

	ClassDB::bind_method(D_METHOD("set_gravity_scale", "gravity_scale"), &RigidBody2D::set_gravity_scale);
	ClassDB::bind_method(D_METHOD("get_gravity_scale"), &RigidBody2D::get_gravity_scale);

	ClassDB::bind_method(D_METHOD("set_linear_damp", "linear_damp"), &RigidBody2D::set_linear_damp);
	ClassDB::bind_method(D_METHOD("get_linear_damp"), &RigidBody2D::get_linear_damp);

	ClassDB::bind_method(D_METHOD("set_angular_damp", "angular_damp"), &RigidBody2D::set_angular_damp);
	ClassDB::bind_method(D_METHOD("get_angular_damp"), &RigidBody2D::get_angular_damp);

	ClassDB::bind_method(D_METHOD("set_linear_velocity", "linear_velocity"), &RigidBody2D::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &RigidBody2D::get_linear_velocity);

	ClassDB::bind_method(D_METHOD("set_angular_velocity", "angular_velocity"), &RigidBody2D::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &RigidBody2D::get_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_max_contacts_reported", "amount"), &RigidBody2D::set_max_contacts_reported);
	ClassDB::bind_method(D_METHOD("get_max_contacts_reported"), &RigidBody2D::get_max_contacts_reported);

	ClassDB::bind_method(D_METHOD("set_use_custom_integrator", "enable"), &RigidBody2D::set_use_custom_integrator);
	ClassDB::bind_method(D_METHOD("is_using_custom_integrator"), &RigidBody2D::is_using_custom_integrator);

	ClassDB::bind_method(D_METHOD("set_contact_monitor", "enabled"), &RigidBody2D::set_contact_monitor);
	ClassDB::bind_method(D_METHOD("is_contact_monitor_enabled"), &RigidBody2D::is_contact_monitor_enabled);

	ClassDB::bind_method(D_METHOD("set_continuous_collision_detection_mode", "mode"), &RigidBody2D::set_continuous_collision_detection_mode);
	ClassDB::bind_method(D_METHOD("get_continuous_collision_detection_mode"), &RigidBody2D::get_continuous_collision_detection_mode);

	ClassDB::bind_method(D_METHOD("set_axis_velocity", "axis_velocity"), &RigidBody2D::set_axis_velocity);
	ClassDB::bind_method(D_METHOD("apply_impulse", "offset", "impulse"), &RigidBody2D::apply_impulse);

	ClassDB::bind_method(D_METHOD("set_applied_force", "force"), &RigidBody2D::set_applied_force);
	ClassDB::bind_method(D_METHOD("get_applied_force"), &RigidBody2D::get_applied_force);

	ClassDB::bind_method(D_METHOD("set_applied_torque", "torque"), &RigidBody2D::set_applied_torque);
	ClassDB::bind_method(D_METHOD("get_applied_torque"), &RigidBody2D::get_applied_torque);

	ClassDB::bind_method(D_METHOD("add_force", "offset", "force"), &RigidBody2D::add_force);

	ClassDB::bind_method(D_METHOD("set_sleeping", "sleeping"), &RigidBody2D::set_sleeping);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &RigidBody2D::is_sleeping);

	ClassDB::bind_method(D_METHOD("set_can_sleep", "able_to_sleep"), &RigidBody2D::set_can_sleep);
	ClassDB::bind_method(D_METHOD("is_able_to_sleep"), &RigidBody2D::is_able_to_sleep);

	ClassDB::bind_method(D_METHOD("test_motion", "motion", "margin", "result:Physics2DTestMotionResult"), &RigidBody2D::_test_motion, DEFVAL(0.08), DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("_direct_state_changed"), &RigidBody2D::_direct_state_changed);
	ClassDB::bind_method(D_METHOD("_body_enter_tree"), &RigidBody2D::_body_enter_tree);
	ClassDB::bind_method(D_METHOD("_body_exit_tree"), &RigidBody2D::_body_exit_tree);

	ClassDB::bind_method(D_METHOD("get_colliding_bodies"), &RigidBody2D::get_colliding_bodies);

	BIND_VMETHOD(MethodInfo("_integrate_forces", PropertyInfo(Variant::OBJECT, "state:Physics2DDirectBodyState")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Rigid,Static,Character,Kinematic"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "mass", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "weight", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01", PROPERTY_USAGE_EDITOR), "set_weight", "get_weight");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "friction", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_bounce", "get_bounce");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "gravity_scale", PROPERTY_HINT_RANGE, "-128,128,0.01"), "set_gravity_scale", "get_gravity_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "custom_integrator"), "set_use_custom_integrator", "is_using_custom_integrator");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "continuous_cd", PROPERTY_HINT_ENUM, "Disabled,Cast Ray,Cast Shape"), "set_continuous_collision_detection_mode", "get_continuous_collision_detection_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "contacts_reported"), "set_max_contacts_reported", "get_max_contacts_reported");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "contact_monitor"), "set_contact_monitor", "is_contact_monitor_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleeping", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "can_sleep"), "set_can_sleep", "is_able_to_sleep");
	ADD_GROUP("Linear", "linear_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "linear_damp", PROPERTY_HINT_RANGE, "-1,128,0.01"), "set_linear_damp", "get_linear_damp");
	ADD_GROUP("Angular", "angular_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_damp", PROPERTY_HINT_RANGE, "-1,128,0.01"), "set_angular_damp", "get_angular_damp");

	ADD_SIGNAL(MethodInfo("body_shape_entered", PropertyInfo(Variant::INT, "body_id"), PropertyInfo(Variant::OBJECT, "body"), PropertyInfo(Variant::INT, "body_shape"), PropertyInfo(Variant::INT, "local_shape")));
	ADD_SIGNAL(MethodInfo("body_shape_exited", PropertyInfo(Variant::INT, "body_id"), PropertyInfo(Variant::OBJECT, "body"), PropertyInfo(Variant::INT, "body_shape"), PropertyInfo(Variant::INT, "local_shape")));
	ADD_SIGNAL(MethodInfo("body_entered", PropertyInfo(Variant::OBJECT, "body")));
	ADD_SIGNAL(MethodInfo("body_exited", PropertyInfo(Variant::OBJECT, "body")));
	ADD_SIGNAL(MethodInfo("sleeping_state_changed"));

	BIND_CONSTANT(MODE_STATIC);
	BIND_CONSTANT(MODE_KINEMATIC);
	BIND_CONSTANT(MODE_RIGID);
	BIND_CONSTANT(MODE_CHARACTER);

	BIND_CONSTANT(CCD_MODE_DISABLED);
	BIND_CONSTANT(CCD_MODE_CAST_RAY);
	BIND_CONSTANT(CCD_MODE_CAST_SHAPE);
}

RigidBody2D::RigidBody2D()
	: PhysicsBody2D(Physics2DServer::BODY_MODE_RIGID) {

	mode = MODE_RIGID;

	bounce = 0;
	mass = 1;
	friction = 1;

	gravity_scale = 1;
	linear_damp = -1;
	angular_damp = -1;

	max_contacts_reported = 0;
	state = NULL;

	angular_velocity = 0;
	sleeping = false;
	ccd_mode = CCD_MODE_DISABLED;

	custom_integrator = false;
	contact_monitor = NULL;
	can_sleep = true;

	Physics2DServer::get_singleton()->body_set_force_integration_callback(get_rid(), this, "_direct_state_changed");
}

RigidBody2D::~RigidBody2D() {

	if (contact_monitor)
		memdelete(contact_monitor);
}

//////////////////////////

Variant KinematicBody2D::_get_collider() const {

	ObjectID oid = get_collider();
	if (oid == 0)
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

void KinematicBody2D::revert_motion() {

	Transform2D gt = get_global_transform();
	gt.elements[2] -= travel;
	travel = Vector2();
	set_global_transform(gt);
}

Vector2 KinematicBody2D::get_travel() const {

	return travel;
}

Vector2 KinematicBody2D::move(const Vector2 &p_motion) {

#if 1

	Transform2D gt = get_global_transform();
	Physics2DServer::MotionResult result;
	colliding = Physics2DServer::get_singleton()->body_test_motion(get_rid(), gt, p_motion, margin, &result);

	collider_metadata = result.collider_metadata;
	collider_shape = result.collider_shape;
	collider_vel = result.collider_velocity;
	collision = result.collision_point;
	normal = result.collision_normal;
	collider = result.collider_id;

	gt.elements[2] += result.motion;
	set_global_transform(gt);
	travel = result.motion;

	return result.remainder;

#else
	//give me back regular physics engine logic
	//this is madness
	//and most people using this function will think
	//what it does is simpler than using physics
	//this took about a week to get right..
	//but is it right? who knows at this point..

	colliding = false;
	ERR_FAIL_COND_V(!is_inside_tree(), Vector2());
	Physics2DDirectSpaceState *dss = Physics2DServer::get_singleton()->space_get_direct_state(get_world_2d()->get_space());
	ERR_FAIL_COND_V(!dss, Vector2());
	const int max_shapes = 32;
	Vector2 sr[max_shapes * 2];
	int res_shapes;

	Set<RID> exclude;
	exclude.insert(get_rid());

	//recover first
	int recover_attempts = 4;

	bool collided = false;
	uint32_t mask = 0;
	if (true)
		mask |= Physics2DDirectSpaceState::TYPE_MASK_STATIC_BODY;
	if (true)
		mask |= Physics2DDirectSpaceState::TYPE_MASK_KINEMATIC_BODY;
	if (true)
		mask |= Physics2DDirectSpaceState::TYPE_MASK_RIGID_BODY;
	if (true)
		mask |= Physics2DDirectSpaceState::TYPE_MASK_CHARACTER_BODY;

	//print_line("margin: "+rtos(margin));
	do {

		//motion recover
		for (int i = 0; i < get_shape_count(); i++) {

			if (is_shape_set_as_trigger(i))
				continue;
			if (dss->collide_shape(get_shape(i)->get_rid(), get_global_transform() * get_shape_transform(i), Vector2(), margin, sr, max_shapes, res_shapes, exclude, get_layer_mask(), mask))
				collided = true;
		}

		if (!collided)
			break;

		Vector2 recover_motion;

		for (int i = 0; i < res_shapes; i++) {

			Vector2 a = sr[i * 2 + 0];
			Vector2 b = sr[i * 2 + 1];

			float d = a.distance_to(b);

			/*
			if (d<margin)
				continue;
			*/
			recover_motion += (b - a) * 0.4;
		}

		if (recover_motion == Vector2()) {
			collided = false;
			break;
		}

		Transform2D gt = get_global_transform();
		gt.elements[2] += recover_motion;
		set_global_transform(gt);

		recover_attempts--;

	} while (recover_attempts);

	//move second
	float safe = 1.0;
	float unsafe = 1.0;
	int best_shape = -1;

	for (int i = 0; i < get_shape_count(); i++) {

		if (is_shape_set_as_trigger(i))
			continue;

		float lsafe, lunsafe;
		bool valid = dss->cast_motion(get_shape(i)->get_rid(), get_global_transform() * get_shape_transform(i), p_motion, 0, lsafe, lunsafe, exclude, get_layer_mask(), mask);
		//print_line("shape: "+itos(i)+" travel:"+rtos(ltravel));
		if (!valid) {

			safe = 0;
			unsafe = 0;
			best_shape = i; //sadly it's the best
			break;
		}
		if (lsafe == 1.0) {
			continue;
		}
		if (lsafe < safe) {

			safe = lsafe;
			unsafe = lunsafe;
			best_shape = i;
		}
	}

	//print_line("best shape: "+itos(best_shape)+" motion "+p_motion);

	if (safe >= 1) {
		//not collided
		colliding = false;

	} else {

		//it collided, let's get the rest info in unsafe advance
		Transform2D ugt = get_global_transform();
		ugt.elements[2] += p_motion * unsafe;
		Physics2DDirectSpaceState::ShapeRestInfo rest_info;
		bool c2 = dss->rest_info(get_shape(best_shape)->get_rid(), ugt * get_shape_transform(best_shape), Vector2(), margin, &rest_info, exclude, get_layer_mask(), mask);
		if (!c2) {
			//should not happen, but floating point precision is so weird..

			colliding = false;
		} else {

			//print_line("Travel: "+rtos(travel));
			colliding = true;
			collision = rest_info.point;
			normal = rest_info.normal;
			collider = rest_info.collider_id;
			collider_vel = rest_info.linear_velocity;
			collider_shape = rest_info.shape;
			collider_metadata = rest_info.metadata;
		}
	}

	Vector2 motion = p_motion * safe;
	Transform2D gt = get_global_transform();
	gt.elements[2] += motion;
	set_global_transform(gt);

	return p_motion - motion;
#endif
}

Vector2 KinematicBody2D::move_and_slide(const Vector2 &p_linear_velocity, const Vector2 &p_floor_direction, float p_slope_stop_min_velocity, int p_max_bounces, float p_floor_max_angle) {

	Vector2 motion = (move_and_slide_floor_velocity + p_linear_velocity) * get_fixed_process_delta_time();
	Vector2 lv = p_linear_velocity;

	move_and_slide_on_floor = false;
	move_and_slide_on_ceiling = false;
	move_and_slide_on_wall = false;
	move_and_slide_colliders.clear();
	move_and_slide_floor_velocity = Vector2();

	while (p_max_bounces) {

		motion = move(motion);

		if (is_colliding()) {

			if (p_floor_direction == Vector2()) {
				//all is a wall
				move_and_slide_on_wall = true;
			} else {
				if (get_collision_normal().dot(p_floor_direction) >= Math::cos(p_floor_max_angle)) { //floor

					move_and_slide_on_floor = true;
					move_and_slide_floor_velocity = get_collider_velocity();

					if (get_travel().length() < 1 && ABS((lv.x - move_and_slide_floor_velocity.x)) < p_slope_stop_min_velocity) {
						revert_motion();
						return Vector2();
					}
				} else if (get_collision_normal().dot(-p_floor_direction) <= Math::cos(p_floor_max_angle)) { //ceiling
					move_and_slide_on_ceiling = true;
				} else {
					move_and_slide_on_wall = true;
				}
			}

			Vector2 n = get_collision_normal();
			motion = motion.slide(n);
			lv = lv.slide(n);
			Variant collider = _get_collider();
			if (collider.get_type() != Variant::NIL) {
				move_and_slide_colliders.push_back(collider);
			}

		} else {
			break;
		}

		p_max_bounces--;
		if (motion == Vector2())
			break;
	}

	return lv;
}

bool KinematicBody2D::is_move_and_slide_on_floor() const {

	return move_and_slide_on_floor;
}
bool KinematicBody2D::is_move_and_slide_on_wall() const {

	return move_and_slide_on_wall;
}
bool KinematicBody2D::is_move_and_slide_on_ceiling() const {

	return move_and_slide_on_ceiling;
}
Array KinematicBody2D::get_move_and_slide_colliders() const {

	return move_and_slide_colliders;
}

Vector2 KinematicBody2D::move_to(const Vector2 &p_position) {

	return move(p_position - get_global_position());
}

bool KinematicBody2D::test_move(const Transform2D &p_from, const Vector2 &p_motion) {

	ERR_FAIL_COND_V(!is_inside_tree(), false);

	return Physics2DServer::get_singleton()->body_test_motion(get_rid(), p_from, p_motion, margin);
}

Vector2 KinematicBody2D::get_collision_pos() const {

	ERR_FAIL_COND_V(!colliding, Vector2());
	return collision;
}

Vector2 KinematicBody2D::get_collision_normal() const {

	ERR_FAIL_COND_V(!colliding, Vector2());
	return normal;
}

Vector2 KinematicBody2D::get_collider_velocity() const {

	return collider_vel;
}

ObjectID KinematicBody2D::get_collider() const {

	ERR_FAIL_COND_V(!colliding, 0);
	return collider;
}

int KinematicBody2D::get_collider_shape() const {

	ERR_FAIL_COND_V(!colliding, 0);
	return collider_shape;
}

Variant KinematicBody2D::get_collider_metadata() const {

	ERR_FAIL_COND_V(!colliding, 0);
	return collider_metadata;
}

bool KinematicBody2D::is_colliding() const {

	return colliding;
}

void KinematicBody2D::set_collision_margin(float p_margin) {

	margin = p_margin;
}

float KinematicBody2D::get_collision_margin() const {

	return margin;
}

void KinematicBody2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("move", "rel_vec"), &KinematicBody2D::move);
	ClassDB::bind_method(D_METHOD("move_to", "position"), &KinematicBody2D::move_to);
	ClassDB::bind_method(D_METHOD("move_and_slide", "linear_velocity", "floor_normal", "slope_stop_min_velocity", "max_bounces", "floor_max_angle"), &KinematicBody2D::move_and_slide, DEFVAL(Vector2(0, 0)), DEFVAL(5), DEFVAL(4), DEFVAL(Math::deg2rad((float)45)));

	ClassDB::bind_method(D_METHOD("test_move", "from", "rel_vec"), &KinematicBody2D::test_move);
	ClassDB::bind_method(D_METHOD("get_travel"), &KinematicBody2D::get_travel);
	ClassDB::bind_method(D_METHOD("revert_motion"), &KinematicBody2D::revert_motion);

	ClassDB::bind_method(D_METHOD("is_colliding"), &KinematicBody2D::is_colliding);

	ClassDB::bind_method(D_METHOD("get_collision_pos"), &KinematicBody2D::get_collision_pos);
	ClassDB::bind_method(D_METHOD("get_collision_normal"), &KinematicBody2D::get_collision_normal);
	ClassDB::bind_method(D_METHOD("get_collider_velocity"), &KinematicBody2D::get_collider_velocity);
	ClassDB::bind_method(D_METHOD("get_collider:Variant"), &KinematicBody2D::_get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_shape"), &KinematicBody2D::get_collider_shape);
	ClassDB::bind_method(D_METHOD("get_collider_metadata:Variant"), &KinematicBody2D::get_collider_metadata);
	ClassDB::bind_method(D_METHOD("get_move_and_slide_colliders"), &KinematicBody2D::get_move_and_slide_colliders);
	ClassDB::bind_method(D_METHOD("is_move_and_slide_on_floor"), &KinematicBody2D::is_move_and_slide_on_floor);
	ClassDB::bind_method(D_METHOD("is_move_and_slide_on_ceiling"), &KinematicBody2D::is_move_and_slide_on_ceiling);
	ClassDB::bind_method(D_METHOD("is_move_and_slide_on_wall"), &KinematicBody2D::is_move_and_slide_on_wall);

	ClassDB::bind_method(D_METHOD("set_collision_margin", "pixels"), &KinematicBody2D::set_collision_margin);
	ClassDB::bind_method(D_METHOD("get_collision_margin", "pixels"), &KinematicBody2D::get_collision_margin);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "collision/margin", PROPERTY_HINT_RANGE, "0.001,256,0.001"), "set_collision_margin", "get_collision_margin");
}

KinematicBody2D::KinematicBody2D()
	: PhysicsBody2D(Physics2DServer::BODY_MODE_KINEMATIC) {

	colliding = false;
	collider = 0;

	collider_shape = 0;

	margin = 0.08;

	move_and_slide_on_floor = false;
	move_and_slide_on_ceiling = false;
	move_and_slide_on_wall = false;
}
KinematicBody2D::~KinematicBody2D() {
}
