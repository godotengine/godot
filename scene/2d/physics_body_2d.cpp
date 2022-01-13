/*************************************************************************/
/*  physics_body_2d.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/core_string_names.h"
#include "core/engine.h"
#include "core/list.h"
#include "core/math/math_funcs.h"
#include "core/method_bind_ext.gen.inc"
#include "core/object.h"
#include "core/rid.h"
#include "scene/scene_string_names.h"

void PhysicsBody2D::_notification(int p_what) {
}

void PhysicsBody2D::_set_layers(uint32_t p_mask) {
	set_collision_layer(p_mask);
	set_collision_mask(p_mask);
}

uint32_t PhysicsBody2D::_get_layers() const {
	return get_collision_layer();
}

void PhysicsBody2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_set_layers", "mask"), &PhysicsBody2D::_set_layers);
	ClassDB::bind_method(D_METHOD("_get_layers"), &PhysicsBody2D::_get_layers);
	ClassDB::bind_method(D_METHOD("get_collision_exceptions"), &PhysicsBody2D::get_collision_exceptions);
	ClassDB::bind_method(D_METHOD("add_collision_exception_with", "body"), &PhysicsBody2D::add_collision_exception_with);
	ClassDB::bind_method(D_METHOD("remove_collision_exception_with", "body"), &PhysicsBody2D::remove_collision_exception_with);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "layers", PROPERTY_HINT_LAYERS_2D_PHYSICS, "", 0), "_set_layers", "_get_layers"); //for backwards compat
}

PhysicsBody2D::PhysicsBody2D(Physics2DServer::BodyMode p_mode) :
		CollisionObject2D(RID_PRIME(Physics2DServer::get_singleton()->body_create()), false) {
	Physics2DServer::get_singleton()->body_set_mode(get_rid(), p_mode);
	set_pickable(false);
}

Array PhysicsBody2D::get_collision_exceptions() {
	List<RID> exceptions;
	Physics2DServer::get_singleton()->body_get_collision_exceptions(get_rid(), &exceptions);
	Array ret;
	for (List<RID>::Element *E = exceptions.front(); E; E = E->next()) {
		RID body = E->get();
		ObjectID instance_id = Physics2DServer::get_singleton()->body_get_object_instance_id(body);
		Object *obj = ObjectDB::get_instance(instance_id);
		PhysicsBody2D *physics_body = Object::cast_to<PhysicsBody2D>(obj);
		ret.append(physics_body);
	}
	return ret;
}

void PhysicsBody2D::add_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	PhysicsBody2D *physics_body = Object::cast_to<PhysicsBody2D>(p_node);
	ERR_FAIL_COND_MSG(!physics_body, "Collision exception only works between two objects of PhysicsBody type.");
	Physics2DServer::get_singleton()->body_add_collision_exception(get_rid(), physics_body->get_rid());
}

void PhysicsBody2D::remove_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	PhysicsBody2D *physics_body = Object::cast_to<PhysicsBody2D>(p_node);
	ERR_FAIL_COND_MSG(!physics_body, "Collision exception only works between two objects of PhysicsBody type.");
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

#ifndef DISABLE_DEPRECATED
void StaticBody2D::set_friction(real_t p_friction) {
	if (p_friction == 1.0 && physics_material_override.is_null()) { // default value, don't create an override for that
		return;
	}

	WARN_DEPRECATED_MSG("The method set_friction has been deprecated and will be removed in the future, use physics material instead.");

	ERR_FAIL_COND_MSG(p_friction < 0 || p_friction > 1, "Friction must be between 0 and 1.");

	if (physics_material_override.is_null()) {
		physics_material_override.instance();
		set_physics_material_override(physics_material_override);
	}
	physics_material_override->set_friction(p_friction);
}

real_t StaticBody2D::get_friction() const {
	WARN_DEPRECATED_MSG("The method get_friction has been deprecated and will be removed in the future, use physics material instead.");

	if (physics_material_override.is_null()) {
		return 1;
	}

	return physics_material_override->get_friction();
}

void StaticBody2D::set_bounce(real_t p_bounce) {
	if (p_bounce == 0.0 && physics_material_override.is_null()) { // default value, don't create an override for that
		return;
	}

	WARN_DEPRECATED_MSG("The method set_bounce has been deprecated and will be removed in the future, use physics material instead.");

	ERR_FAIL_COND_MSG(p_bounce < 0 || p_bounce > 1, "Bounce must be between 0 and 1.");

	if (physics_material_override.is_null()) {
		physics_material_override.instance();
		set_physics_material_override(physics_material_override);
	}
	physics_material_override->set_bounce(p_bounce);
}

real_t StaticBody2D::get_bounce() const {
	WARN_DEPRECATED_MSG("The method get_bounce has been deprecated and will be removed in the future, use physics material instead.");

	if (physics_material_override.is_null()) {
		return 0;
	}

	return physics_material_override->get_bounce();
}
#endif // DISABLE_DEPRECATED

void StaticBody2D::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
	if (physics_material_override.is_valid()) {
		if (physics_material_override->is_connected(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics")) {
			physics_material_override->disconnect(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics");
		}
	}

	physics_material_override = p_physics_material_override;

	if (physics_material_override.is_valid()) {
		physics_material_override->connect(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics");
	}
	_reload_physics_characteristics();
}

Ref<PhysicsMaterial> StaticBody2D::get_physics_material_override() const {
	return physics_material_override;
}

void StaticBody2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant_linear_velocity", "vel"), &StaticBody2D::set_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_constant_angular_velocity", "vel"), &StaticBody2D::set_constant_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_linear_velocity"), &StaticBody2D::get_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_angular_velocity"), &StaticBody2D::get_constant_angular_velocity);

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &StaticBody2D::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &StaticBody2D::get_friction);

	ClassDB::bind_method(D_METHOD("set_bounce", "bounce"), &StaticBody2D::set_bounce);
	ClassDB::bind_method(D_METHOD("get_bounce"), &StaticBody2D::get_bounce);
#endif // DISABLE_DEPRECATED

	ClassDB::bind_method(D_METHOD("set_physics_material_override", "physics_material_override"), &StaticBody2D::set_physics_material_override);
	ClassDB::bind_method(D_METHOD("get_physics_material_override"), &StaticBody2D::get_physics_material_override);

	ClassDB::bind_method(D_METHOD("_reload_physics_characteristics"), &StaticBody2D::_reload_physics_characteristics);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "constant_linear_velocity"), "set_constant_linear_velocity", "get_constant_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "constant_angular_velocity"), "set_constant_angular_velocity", "get_constant_angular_velocity");
#ifndef DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "friction", PROPERTY_HINT_RANGE, "0,1,0.01", 0), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce", PROPERTY_HINT_RANGE, "0,1,0.01", 0), "set_bounce", "get_bounce");
#endif // DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material_override", "get_physics_material_override");
}

StaticBody2D::StaticBody2D() :
		PhysicsBody2D(Physics2DServer::BODY_MODE_STATIC) {
	constant_angular_velocity = 0;
}

StaticBody2D::~StaticBody2D() {
}

void StaticBody2D::_reload_physics_characteristics() {
	if (physics_material_override.is_null()) {
		Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_BOUNCE, 0);
		Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_FRICTION, 1);
	} else {
		Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_BOUNCE, physics_material_override->computed_bounce());
		Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_FRICTION, physics_material_override->computed_friction());
	}
}

void RigidBody2D::_body_enter_tree(ObjectID p_id) {
	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = Object::cast_to<Node>(obj);
	ERR_FAIL_COND(!node);

	ERR_FAIL_COND(!contact_monitor);
	Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(E->get().in_scene);

	contact_monitor->locked = true;

	E->get().in_scene = true;
	emit_signal(SceneStringNames::get_singleton()->body_entered, node);

	for (int i = 0; i < E->get().shapes.size(); i++) {
		emit_signal(SceneStringNames::get_singleton()->body_shape_entered, E->get().rid, node, E->get().shapes[i].body_shape, E->get().shapes[i].local_shape);
	}

	contact_monitor->locked = false;
}

void RigidBody2D::_body_exit_tree(ObjectID p_id) {
	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = Object::cast_to<Node>(obj);
	ERR_FAIL_COND(!node);
	ERR_FAIL_COND(!contact_monitor);
	Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(!E->get().in_scene);
	E->get().in_scene = false;

	contact_monitor->locked = true;

	emit_signal(SceneStringNames::get_singleton()->body_exited, node);

	for (int i = 0; i < E->get().shapes.size(); i++) {
		emit_signal(SceneStringNames::get_singleton()->body_shape_exited, E->get().rid, node, E->get().shapes[i].body_shape, E->get().shapes[i].local_shape);
	}

	contact_monitor->locked = false;
}

void RigidBody2D::_body_inout(int p_status, const RID &p_body, ObjectID p_instance, int p_body_shape, int p_local_shape) {
	bool body_in = p_status == 1;
	ObjectID objid = p_instance;

	Object *obj = ObjectDB::get_instance(objid);
	Node *node = Object::cast_to<Node>(obj);

	ERR_FAIL_COND(!contact_monitor);
	Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(objid);

	ERR_FAIL_COND(!body_in && !E);

	if (body_in) {
		if (!E) {
			E = contact_monitor->body_map.insert(objid, BodyState());
			E->get().rid = p_body;
			//E->get().rc=0;
			E->get().in_scene = node && node->is_inside_tree();
			if (node) {
				node->connect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree, make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_body_exit_tree, make_binds(objid));
				if (E->get().in_scene) {
					emit_signal(SceneStringNames::get_singleton()->body_entered, node);
				}
			}

			//E->get().rc++;
		}

		if (node) {
			E->get().shapes.insert(ShapePair(p_body_shape, p_local_shape));
		}

		if (E->get().in_scene) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_entered, p_body, node, p_body_shape, p_local_shape);
		}

	} else {
		//E->get().rc--;

		if (node) {
			E->get().shapes.erase(ShapePair(p_body_shape, p_local_shape));
		}

		bool in_scene = E->get().in_scene;

		if (E->get().shapes.empty()) {
			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree);
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_body_exit_tree);
				if (in_scene) {
					emit_signal(SceneStringNames::get_singleton()->body_exited, node);
				}
			}

			contact_monitor->body_map.erase(E);
		}
		if (node && in_scene) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_exited, p_body, node, p_body_shape, p_local_shape);
		}
	}
}

struct _RigidBody2DInOut {
	RID rid;
	ObjectID id;
	int shape;
	int local_shape;
};

bool RigidBody2D::_test_motion(const Vector2 &p_motion, bool p_infinite_inertia, float p_margin, const Ref<Physics2DTestMotionResult> &p_result) {
	Physics2DServer::MotionResult *r = nullptr;
	Physics2DServer::MotionResult temp_result;
	if (p_result.is_valid()) {
		r = p_result->get_result_ptr();
	} else {
		r = &temp_result;
	}

	bool colliding = Physics2DServer::get_singleton()->body_test_motion(get_rid(), get_global_transform(), p_motion, p_infinite_inertia, p_margin, r);

	if (colliding) {
		// Don't report collision when the whole motion is done.
		return (r->collision_safe_fraction < 1.0);
	} else {
		return false;
	}
}

void RigidBody2D::_direct_state_changed(Object *p_state) {
	state = Object::cast_to<Physics2DDirectBodyState>(p_state);
	ERR_FAIL_COND_MSG(!state, "Method '_direct_state_changed' must receive a valid Physics2DDirectBodyState object as argument");

	set_block_transform_notify(true); // don't want notify (would feedback loop)
	if (mode != MODE_KINEMATIC) {
		set_global_transform(state->get_transform());
	}
	linear_velocity = state->get_linear_velocity();
	angular_velocity = state->get_angular_velocity();
	if (sleeping != state->is_sleeping()) {
		sleeping = state->is_sleeping();
		emit_signal(SceneStringNames::get_singleton()->sleeping_state_changed);
	}
	if (get_script_instance()) {
		get_script_instance()->call("_integrate_forces", state);
	}
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
			RID rid = state->get_contact_collider(i);
			ObjectID obj = state->get_contact_collider_id(i);
			int local_shape = state->get_contact_local_shape(i);
			int shape = state->get_contact_collider_shape(i);

			//bool found=false;

			Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(obj);
			if (!E) {
				toadd[toadd_count].rid = rid;
				toadd[toadd_count].local_shape = local_shape;
				toadd[toadd_count].id = obj;
				toadd[toadd_count].shape = shape;
				toadd_count++;
				continue;
			}

			ShapePair sp(shape, local_shape);
			int idx = E->get().shapes.find(sp);
			if (idx == -1) {
				toadd[toadd_count].rid = rid;
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
					toremove[toremove_count].rid = E->get().rid;
					toremove[toremove_count].body_id = E->key();
					toremove[toremove_count].pair = E->get().shapes[i];
					toremove_count++;
				}
			}
		}

		//process remotions

		for (int i = 0; i < toremove_count; i++) {
			_body_inout(0, toremove[i].rid, toremove[i].body_id, toremove[i].pair.body_shape, toremove[i].pair.local_shape);
		}

		//process aditions

		for (int i = 0; i < toadd_count; i++) {
			_body_inout(1, toadd[i].rid, toadd[i].id, toadd[i].shape, toadd[i].local_shape);
		}

		contact_monitor->locked = false;
	}

	state = nullptr;
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
	ERR_FAIL_COND(p_inertia < 0);
	Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_INERTIA, p_inertia);
}

real_t RigidBody2D::get_inertia() const {
	return Physics2DServer::get_singleton()->body_get_param(get_rid(), Physics2DServer::BODY_PARAM_INERTIA);
}

void RigidBody2D::set_weight(real_t p_weight) {
	set_mass(p_weight / (real_t(GLOBAL_DEF("physics/2d/default_gravity", 98)) / 10));
}

real_t RigidBody2D::get_weight() const {
	return mass * (real_t(GLOBAL_DEF("physics/2d/default_gravity", 98)) / 10);
}

#ifndef DISABLE_DEPRECATED
void RigidBody2D::set_friction(real_t p_friction) {
	if (p_friction == 1.0 && physics_material_override.is_null()) { // default value, don't create an override for that
		return;
	}

	WARN_DEPRECATED_MSG("The method set_friction has been deprecated and will be removed in the future, use physics material instead.");

	ERR_FAIL_COND_MSG(p_friction < 0 || p_friction > 1, "Friction must be between 0 and 1.");

	if (physics_material_override.is_null()) {
		physics_material_override.instance();
		set_physics_material_override(physics_material_override);
	}
	physics_material_override->set_friction(p_friction);
}
real_t RigidBody2D::get_friction() const {
	WARN_DEPRECATED_MSG("The method get_friction has been deprecated and will be removed in the future, use physics material instead.");

	if (physics_material_override.is_null()) {
		return 1;
	}

	return physics_material_override->get_friction();
}

void RigidBody2D::set_bounce(real_t p_bounce) {
	if (p_bounce == 0.0 && physics_material_override.is_null()) { // default value, don't create an override for that
		return;
	}

	WARN_DEPRECATED_MSG("The method set_bounce has been deprecated and will be removed in the future, use physics material instead.");

	ERR_FAIL_COND(p_bounce < 0 || p_bounce > 1);

	if (physics_material_override.is_null()) {
		physics_material_override.instance();
		set_physics_material_override(physics_material_override);
	}
	physics_material_override->set_bounce(p_bounce);
}
real_t RigidBody2D::get_bounce() const {
	WARN_DEPRECATED_MSG("The method get_bounce has been deprecated and will be removed in the future, use physics material instead.");

	if (physics_material_override.is_null()) {
		return 0;
	}

	return physics_material_override->get_bounce();
}
#endif // DISABLE_DEPRECATED

void RigidBody2D::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
	if (physics_material_override.is_valid()) {
		if (physics_material_override->is_connected(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics")) {
			physics_material_override->disconnect(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics");
		}
	}

	physics_material_override = p_physics_material_override;

	if (physics_material_override.is_valid()) {
		physics_material_override->connect(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics");
	}
	_reload_physics_characteristics();
}

Ref<PhysicsMaterial> RigidBody2D::get_physics_material_override() const {
	return physics_material_override;
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
	if (state) {
		state->set_linear_velocity(linear_velocity);
	} else {
		Physics2DServer::get_singleton()->body_set_state(get_rid(), Physics2DServer::BODY_STATE_LINEAR_VELOCITY, linear_velocity);
	}
}

Vector2 RigidBody2D::get_linear_velocity() const {
	return linear_velocity;
}

void RigidBody2D::set_angular_velocity(real_t p_velocity) {
	angular_velocity = p_velocity;
	if (state) {
		state->set_angular_velocity(angular_velocity);
	} else {
		Physics2DServer::get_singleton()->body_set_state(get_rid(), Physics2DServer::BODY_STATE_ANGULAR_VELOCITY, angular_velocity);
	}
}
real_t RigidBody2D::get_angular_velocity() const {
	return angular_velocity;
}

void RigidBody2D::set_use_custom_integrator(bool p_enable) {
	if (custom_integrator == p_enable) {
		return;
	}

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

void RigidBody2D::apply_central_impulse(const Vector2 &p_impulse) {
	Physics2DServer::get_singleton()->body_apply_central_impulse(get_rid(), p_impulse);
}

void RigidBody2D::apply_impulse(const Vector2 &p_offset, const Vector2 &p_impulse) {
	Physics2DServer::get_singleton()->body_apply_impulse(get_rid(), p_offset, p_impulse);
}

void RigidBody2D::apply_torque_impulse(float p_torque) {
	Physics2DServer::get_singleton()->body_apply_torque_impulse(get_rid(), p_torque);
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

void RigidBody2D::add_central_force(const Vector2 &p_force) {
	Physics2DServer::get_singleton()->body_add_central_force(get_rid(), p_force);
}

void RigidBody2D::add_force(const Vector2 &p_offset, const Vector2 &p_force) {
	Physics2DServer::get_singleton()->body_add_force(get_rid(), p_offset, p_force);
}

void RigidBody2D::add_torque(const float p_torque) {
	Physics2DServer::get_singleton()->body_add_torque(get_rid(), p_torque);
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
	if (p_enabled == is_contact_monitor_enabled()) {
		return;
	}

	if (!p_enabled) {
		ERR_FAIL_COND_MSG(contact_monitor->locked, "Can't disable contact monitoring during in/out callback. Use call_deferred(\"set_contact_monitor\", false) instead.");

		for (Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.front(); E; E = E->next()) {
			//clean up mess
			Object *obj = ObjectDB::get_instance(E->key());
			Node *node = Object::cast_to<Node>(obj);

			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree);
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_body_exit_tree);
			}
		}

		memdelete(contact_monitor);
		contact_monitor = nullptr;
	} else {
		contact_monitor = memnew(ContactMonitor);
		contact_monitor->locked = false;
	}
}

bool RigidBody2D::is_contact_monitor_enabled() const {
	return contact_monitor != nullptr;
}

void RigidBody2D::_notification(int p_what) {
#ifdef TOOLS_ENABLED
	if (p_what == NOTIFICATION_ENTER_TREE) {
		if (Engine::get_singleton()->is_editor_hint()) {
			set_notify_local_transform(true); //used for warnings and only in editor
		}
	}

	if (p_what == NOTIFICATION_LOCAL_TRANSFORM_CHANGED) {
		if (Engine::get_singleton()->is_editor_hint()) {
			update_configuration_warning();
		}
	}

#endif
}

String RigidBody2D::get_configuration_warning() const {
	Transform2D t = get_transform();

	String warning = CollisionObject2D::get_configuration_warning();

	if ((get_mode() == MODE_RIGID || get_mode() == MODE_CHARACTER) && (ABS(t.elements[0].length() - 1.0) > 0.05 || ABS(t.elements[1].length() - 1.0) > 0.05)) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("Size changes to RigidBody2D (in character or rigid modes) will be overridden by the physics engine when running.\nChange the size in children collision shapes instead.");
	}

	return warning;
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

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &RigidBody2D::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &RigidBody2D::get_friction);

	ClassDB::bind_method(D_METHOD("set_bounce", "bounce"), &RigidBody2D::set_bounce);
	ClassDB::bind_method(D_METHOD("get_bounce"), &RigidBody2D::get_bounce);
#endif // DISABLE_DEPRECATED

	ClassDB::bind_method(D_METHOD("set_physics_material_override", "physics_material_override"), &RigidBody2D::set_physics_material_override);
	ClassDB::bind_method(D_METHOD("get_physics_material_override"), &RigidBody2D::get_physics_material_override);

	ClassDB::bind_method(D_METHOD("_reload_physics_characteristics"), &RigidBody2D::_reload_physics_characteristics);

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
	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &RigidBody2D::apply_central_impulse);
	ClassDB::bind_method(D_METHOD("apply_impulse", "offset", "impulse"), &RigidBody2D::apply_impulse);
	ClassDB::bind_method(D_METHOD("apply_torque_impulse", "torque"), &RigidBody2D::apply_torque_impulse);

	ClassDB::bind_method(D_METHOD("set_applied_force", "force"), &RigidBody2D::set_applied_force);
	ClassDB::bind_method(D_METHOD("get_applied_force"), &RigidBody2D::get_applied_force);

	ClassDB::bind_method(D_METHOD("set_applied_torque", "torque"), &RigidBody2D::set_applied_torque);
	ClassDB::bind_method(D_METHOD("get_applied_torque"), &RigidBody2D::get_applied_torque);

	ClassDB::bind_method(D_METHOD("add_central_force", "force"), &RigidBody2D::add_central_force);
	ClassDB::bind_method(D_METHOD("add_force", "offset", "force"), &RigidBody2D::add_force);
	ClassDB::bind_method(D_METHOD("add_torque", "torque"), &RigidBody2D::add_torque);

	ClassDB::bind_method(D_METHOD("set_sleeping", "sleeping"), &RigidBody2D::set_sleeping);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &RigidBody2D::is_sleeping);

	ClassDB::bind_method(D_METHOD("set_can_sleep", "able_to_sleep"), &RigidBody2D::set_can_sleep);
	ClassDB::bind_method(D_METHOD("is_able_to_sleep"), &RigidBody2D::is_able_to_sleep);

	ClassDB::bind_method(D_METHOD("test_motion", "motion", "infinite_inertia", "margin", "result"), &RigidBody2D::_test_motion, DEFVAL(true), DEFVAL(0.08), DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("_direct_state_changed"), &RigidBody2D::_direct_state_changed);
	ClassDB::bind_method(D_METHOD("_body_enter_tree"), &RigidBody2D::_body_enter_tree);
	ClassDB::bind_method(D_METHOD("_body_exit_tree"), &RigidBody2D::_body_exit_tree);

	ClassDB::bind_method(D_METHOD("get_colliding_bodies"), &RigidBody2D::get_colliding_bodies);

	BIND_VMETHOD(MethodInfo("_integrate_forces", PropertyInfo(Variant::OBJECT, "state", PROPERTY_HINT_RESOURCE_TYPE, "Physics2DDirectBodyState")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Rigid,Static,Character,Kinematic"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "mass", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "inertia", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01", 0), "set_inertia", "get_inertia");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "weight", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01", PROPERTY_USAGE_EDITOR), "set_weight", "get_weight");
#ifndef DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "friction", PROPERTY_HINT_RANGE, "0,1,0.01", 0), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce", PROPERTY_HINT_RANGE, "0,1,0.01", 0), "set_bounce", "get_bounce");
#endif // DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material_override", "get_physics_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "gravity_scale", PROPERTY_HINT_RANGE, "-128,128,0.01"), "set_gravity_scale", "get_gravity_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "custom_integrator"), "set_use_custom_integrator", "is_using_custom_integrator");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "continuous_cd", PROPERTY_HINT_ENUM, "Disabled,Cast Ray,Cast Shape"), "set_continuous_collision_detection_mode", "get_continuous_collision_detection_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "contacts_reported", PROPERTY_HINT_RANGE, "0,64,1,or_greater"), "set_max_contacts_reported", "get_max_contacts_reported");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "contact_monitor"), "set_contact_monitor", "is_contact_monitor_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleeping", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "can_sleep"), "set_can_sleep", "is_able_to_sleep");
	ADD_GROUP("Linear", "linear_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "linear_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"), "set_linear_damp", "get_linear_damp");
	ADD_GROUP("Angular", "angular_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"), "set_angular_damp", "get_angular_damp");
	ADD_GROUP("Applied Forces", "applied_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "applied_force"), "set_applied_force", "get_applied_force");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "applied_torque"), "set_applied_torque", "get_applied_torque");

	ADD_SIGNAL(MethodInfo("body_shape_entered", PropertyInfo(Variant::_RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("body_shape_exited", PropertyInfo(Variant::_RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("body_entered", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("body_exited", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("sleeping_state_changed"));

	BIND_ENUM_CONSTANT(MODE_RIGID);
	BIND_ENUM_CONSTANT(MODE_STATIC);
	BIND_ENUM_CONSTANT(MODE_CHARACTER);
	BIND_ENUM_CONSTANT(MODE_KINEMATIC);

	BIND_ENUM_CONSTANT(CCD_MODE_DISABLED);
	BIND_ENUM_CONSTANT(CCD_MODE_CAST_RAY);
	BIND_ENUM_CONSTANT(CCD_MODE_CAST_SHAPE);
}

RigidBody2D::RigidBody2D() :
		PhysicsBody2D(Physics2DServer::BODY_MODE_RIGID) {
	mode = MODE_RIGID;

	mass = 1;

	gravity_scale = 1;
	linear_damp = -1;
	angular_damp = -1;

	max_contacts_reported = 0;
	state = nullptr;

	angular_velocity = 0;
	sleeping = false;
	ccd_mode = CCD_MODE_DISABLED;

	custom_integrator = false;
	contact_monitor = nullptr;
	can_sleep = true;

	Physics2DServer::get_singleton()->body_set_force_integration_callback(get_rid(), this, "_direct_state_changed");
}

RigidBody2D::~RigidBody2D() {
	if (contact_monitor) {
		memdelete(contact_monitor);
	}
}

void RigidBody2D::_reload_physics_characteristics() {
	if (physics_material_override.is_null()) {
		Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_BOUNCE, 0);
		Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_FRICTION, 1);
	} else {
		Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_BOUNCE, physics_material_override->computed_bounce());
		Physics2DServer::get_singleton()->body_set_param(get_rid(), Physics2DServer::BODY_PARAM_FRICTION, physics_material_override->computed_friction());
	}
}

//////////////////////////

Ref<KinematicCollision2D> KinematicBody2D::_move(const Vector2 &p_motion, bool p_infinite_inertia, bool p_exclude_raycast_shapes, bool p_test_only) {
	Collision col;

	if (move_and_collide(p_motion, p_infinite_inertia, col, p_exclude_raycast_shapes, p_test_only)) {
		// Create a new instance when the cached reference is invalid or still in use in script.
		if (motion_cache.is_null() || motion_cache->reference_get_count() > 1) {
			motion_cache.instance();
			motion_cache->owner = this;
		}

		motion_cache->collision = col;

		return motion_cache;
	}

	return Ref<KinematicCollision2D>();
}

bool KinematicBody2D::separate_raycast_shapes(bool p_infinite_inertia, Collision &r_collision) {
	Physics2DServer::SeparationResult sep_res[8]; //max 8 rays

	Transform2D gt = get_global_transform();

	Vector2 recover;
	int hits = Physics2DServer::get_singleton()->body_test_ray_separation(get_rid(), gt, p_infinite_inertia, recover, sep_res, 8, margin);
	int deepest = -1;
	float deepest_depth;
	for (int i = 0; i < hits; i++) {
		if (deepest == -1 || sep_res[i].collision_depth > deepest_depth) {
			deepest = i;
			deepest_depth = sep_res[i].collision_depth;
		}
	}

	gt.elements[2] += recover;
	set_global_transform(gt);

	if (deepest != -1) {
		r_collision.collider = sep_res[deepest].collider_id;
		r_collision.collider_rid = sep_res[deepest].collider;
		r_collision.collider_metadata = sep_res[deepest].collider_metadata;
		r_collision.collider_shape = sep_res[deepest].collider_shape;
		r_collision.collider_vel = sep_res[deepest].collider_velocity;
		r_collision.collision = sep_res[deepest].collision_point;
		r_collision.normal = sep_res[deepest].collision_normal;
		r_collision.local_shape = sep_res[deepest].collision_local_shape;
		r_collision.travel = recover;
		r_collision.remainder = Vector2();

		return true;
	} else {
		return false;
	}
}

bool KinematicBody2D::move_and_collide(const Vector2 &p_motion, bool p_infinite_inertia, Collision &r_collision, bool p_exclude_raycast_shapes, bool p_test_only, bool p_cancel_sliding, const Set<RID> &p_exclude) {
	if (sync_to_physics) {
		ERR_PRINT("Functions move_and_slide and move_and_collide do not work together with 'sync to physics' option. Please read the documentation.");
	}
	Transform2D gt = get_global_transform();
	Physics2DServer::MotionResult result;

	bool colliding = Physics2DServer::get_singleton()->body_test_motion(get_rid(), gt, p_motion, p_infinite_inertia, margin, &result, p_exclude_raycast_shapes, p_exclude);

	// Restore direction of motion to be along original motion,
	// in order to avoid sliding due to recovery,
	// but only if collision depth is low enough to avoid tunneling.
	if (p_cancel_sliding) {
		real_t motion_length = p_motion.length();
		real_t precision = 0.001;

		if (colliding) {
			// Can't just use margin as a threshold because collision depth is calculated on unsafe motion,
			// so even in normal resting cases the depth can be a bit more than the margin.
			precision += motion_length * (result.collision_unsafe_fraction - result.collision_safe_fraction);

			if (result.collision_depth > (real_t)margin + precision) {
				p_cancel_sliding = false;
			}
		}

		if (p_cancel_sliding) {
			// When motion is null, recovery is the resulting motion.
			Vector2 motion_normal;
			if (motion_length > CMP_EPSILON) {
				motion_normal = p_motion / motion_length;
			}

			// Check depth of recovery.
			real_t projected_length = result.motion.dot(motion_normal);
			Vector2 recovery = result.motion - motion_normal * projected_length;
			real_t recovery_length = recovery.length();
			// Fixes cases where canceling slide causes the motion to go too deep into the ground,
			// because we're only taking rest information into account and not general recovery.
			if (recovery_length < (real_t)margin + precision) {
				// Apply adjustment to motion.
				result.motion = motion_normal * projected_length;
				result.remainder = p_motion - result.motion;
			}
		}
	}

	if (colliding) {
		r_collision.collider_metadata = result.collider_metadata;
		r_collision.collider_shape = result.collider_shape;
		r_collision.collider_vel = result.collider_velocity;
		r_collision.collision = result.collision_point;
		r_collision.normal = result.collision_normal;
		r_collision.collider = result.collider_id;
		r_collision.collider_rid = result.collider;
		r_collision.travel = result.motion;
		r_collision.remainder = result.remainder;
		r_collision.local_shape = result.collision_local_shape;
	}

	if (!p_test_only) {
		gt.elements[2] += result.motion;
		set_global_transform(gt);
	}

	return colliding;
}

//so, if you pass 45 as limit, avoid numerical precision errors when angle is 45.
#define FLOOR_ANGLE_THRESHOLD 0.01

Vector2 KinematicBody2D::_move_and_slide_internal(const Vector2 &p_linear_velocity, const Vector2 &p_snap, const Vector2 &p_up_direction, bool p_stop_on_slope, int p_max_slides, float p_floor_max_angle, bool p_infinite_inertia) {
	Vector2 body_velocity = p_linear_velocity;
	Vector2 body_velocity_normal = body_velocity.normalized();
	Vector2 up_direction = p_up_direction.normalized();
	bool was_on_floor = on_floor;

	// Hack in order to work with calling from _process as well as from _physics_process; calling from thread is risky
	float delta = Engine::get_singleton()->is_in_physics_frame() ? get_physics_process_delta_time() : get_process_delta_time();

	Vector2 current_floor_velocity = floor_velocity;
	if (on_floor && on_floor_body.is_valid()) {
		//this approach makes sure there is less delay between the actual body velocity and the one we saved
		Physics2DDirectBodyState *bs = Physics2DServer::get_singleton()->body_get_direct_state(on_floor_body);
		if (bs) {
			Transform2D gt = get_global_transform();
			Vector2 local_position = gt.elements[2] - bs->get_transform().elements[2];
			current_floor_velocity = bs->get_velocity_at_local_position(local_position);
		} else {
			// Body is removed or destroyed, invalidate floor.
			current_floor_velocity = Vector2();
			on_floor_body = RID();
		}
	}

	colliders.clear();
	on_floor = false;
	on_ceiling = false;
	on_wall = false;
	floor_normal = Vector2();
	floor_velocity = Vector2();

	if (current_floor_velocity != Vector2() && on_floor_body.is_valid()) {
		Collision floor_collision;
		Set<RID> exclude;
		exclude.insert(on_floor_body);
		if (move_and_collide(current_floor_velocity * delta, p_infinite_inertia, floor_collision, true, false, false, exclude)) {
			colliders.push_back(floor_collision);
			_set_collision_direction(floor_collision, up_direction, p_floor_max_angle);
		}
	}

	on_floor_body = RID();
	Vector2 motion = body_velocity * delta;

	// No sliding on first attempt to keep floor motion stable when possible,
	// when stop on slope is enabled.
	bool sliding_enabled = !p_stop_on_slope;
	for (int iteration = 0; iteration < p_max_slides; ++iteration) {
		Collision collision;
		bool found_collision = false;

		for (int i = 0; i < 2; ++i) {
			bool collided;
			if (i == 0) { //collide
				collided = move_and_collide(motion, p_infinite_inertia, collision, true, false, !sliding_enabled);
				if (!collided) {
					motion = Vector2(); //clear because no collision happened and motion completed
				}
			} else { //separate raycasts (if any)
				collided = separate_raycast_shapes(p_infinite_inertia, collision);
				if (collided) {
					collision.remainder = motion; //keep
					collision.travel = Vector2();
				}
			}

			if (collided) {
				found_collision = true;

				colliders.push_back(collision);

				_set_collision_direction(collision, up_direction, p_floor_max_angle);

				if (on_floor && p_stop_on_slope) {
					if ((body_velocity_normal + up_direction).length() < 0.01) {
						Transform2D gt = get_global_transform();
						if (collision.travel.length() > margin) {
							gt.elements[2] -= collision.travel.slide(up_direction);
						} else {
							gt.elements[2] -= collision.travel;
						}
						set_global_transform(gt);
						return Vector2();
					}
				}

				if (sliding_enabled || !on_floor) {
					motion = collision.remainder.slide(collision.normal);
					body_velocity = body_velocity.slide(collision.normal);
				} else {
					motion = collision.remainder;
				}
			}

			sliding_enabled = true;
		}

		if (!found_collision || motion == Vector2()) {
			break;
		}
	}

	if (was_on_floor && p_snap != Vector2() && !on_floor) {
		// Apply snap.
		Collision col;
		Transform2D gt = get_global_transform();

		if (move_and_collide(p_snap, p_infinite_inertia, col, false, true, false)) {
			bool apply = true;
			if (up_direction != Vector2()) {
				if (Math::acos(col.normal.dot(up_direction)) <= p_floor_max_angle + FLOOR_ANGLE_THRESHOLD) {
					on_floor = true;
					floor_normal = col.normal;
					on_floor_body = col.collider_rid;
					floor_velocity = col.collider_vel;
					if (p_stop_on_slope) {
						// move and collide may stray the object a bit because of pre un-stucking,
						// so only ensure that motion happens on floor direction in this case.
						if (col.travel.length() > margin) {
							col.travel = up_direction * up_direction.dot(col.travel);
						} else {
							col.travel = Vector2();
						}
					}
				} else {
					apply = false;
				}
			}

			if (apply) {
				gt.elements[2] += col.travel;
				set_global_transform(gt);
			}
		}
	}

	if (moving_platform_apply_velocity_on_leave != PLATFORM_VEL_ON_LEAVE_NEVER) {
		// Add last platform velocity when just left a moving platform.
		if (!on_floor) {
			if (moving_platform_apply_velocity_on_leave == PLATFORM_VEL_ON_LEAVE_UPWARD_ONLY && current_floor_velocity.dot(up_direction) < 0) {
				current_floor_velocity = current_floor_velocity.slide(up_direction);
			}
			return body_velocity + current_floor_velocity;
		}
	}

	return body_velocity;
}

Vector2 KinematicBody2D::move_and_slide(const Vector2 &p_linear_velocity, const Vector2 &p_up_direction, bool p_stop_on_slope, int p_max_slides, float p_floor_max_angle, bool p_infinite_inertia) {
	return _move_and_slide_internal(p_linear_velocity, Vector2(), p_up_direction, p_stop_on_slope, p_max_slides, p_floor_max_angle, p_infinite_inertia);
}

Vector2 KinematicBody2D::move_and_slide_with_snap(const Vector2 &p_linear_velocity, const Vector2 &p_snap, const Vector2 &p_up_direction, bool p_stop_on_slope, int p_max_slides, float p_floor_max_angle, bool p_infinite_inertia) {
	return _move_and_slide_internal(p_linear_velocity, p_snap, p_up_direction, p_stop_on_slope, p_max_slides, p_floor_max_angle, p_infinite_inertia);
}

void KinematicBody2D::_set_collision_direction(const Collision &p_collision, const Vector2 &p_up_direction, float p_floor_max_angle) {
	if (p_up_direction == Vector2()) {
		//all is a wall
		on_wall = true;
	} else {
		if (Math::acos(p_collision.normal.dot(p_up_direction)) <= p_floor_max_angle + FLOOR_ANGLE_THRESHOLD) { //floor
			on_floor = true;
			floor_normal = p_collision.normal;
			on_floor_body = p_collision.collider_rid;
			floor_velocity = p_collision.collider_vel;
		} else if (Math::acos(p_collision.normal.dot(-p_up_direction)) <= p_floor_max_angle + FLOOR_ANGLE_THRESHOLD) { //ceiling
			on_ceiling = true;
		} else {
			on_wall = true;
		}
	}
}

bool KinematicBody2D::is_on_floor() const {
	return on_floor;
}
bool KinematicBody2D::is_on_wall() const {
	return on_wall;
}
bool KinematicBody2D::is_on_ceiling() const {
	return on_ceiling;
}

Vector2 KinematicBody2D::get_floor_normal() const {
	return floor_normal;
}

real_t KinematicBody2D::get_floor_angle(const Vector2 &p_up_direction) const {
	ERR_FAIL_COND_V(p_up_direction == Vector2(), 0);
	return Math::acos(floor_normal.dot(p_up_direction));
}

Vector2 KinematicBody2D::get_floor_velocity() const {
	return floor_velocity;
}

void KinematicBody2D::set_moving_platform_apply_velocity_on_leave(MovingPlatformApplyVelocityOnLeave p_on_leave_apply_velocity) {
	moving_platform_apply_velocity_on_leave = p_on_leave_apply_velocity;
}

KinematicBody2D::MovingPlatformApplyVelocityOnLeave KinematicBody2D::get_moving_platform_apply_velocity_on_leave() const {
	return moving_platform_apply_velocity_on_leave;
}

bool KinematicBody2D::test_move(const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia) {
	ERR_FAIL_COND_V(!is_inside_tree(), false);

	Physics2DServer::MotionResult result;
	bool colliding = Physics2DServer::get_singleton()->body_test_motion(get_rid(), p_from, p_motion, p_infinite_inertia, margin, &result);

	if (colliding) {
		// Don't report collision when the whole motion is done.
		return (result.collision_safe_fraction < 1.0);
	} else {
		return false;
	}
}

void KinematicBody2D::set_safe_margin(float p_margin) {
	margin = p_margin;
}

float KinematicBody2D::get_safe_margin() const {
	return margin;
}

int KinematicBody2D::get_slide_count() const {
	return colliders.size();
}

KinematicBody2D::Collision KinematicBody2D::get_slide_collision(int p_bounce) const {
	ERR_FAIL_INDEX_V(p_bounce, colliders.size(), Collision());
	return colliders[p_bounce];
}

Ref<KinematicCollision2D> KinematicBody2D::_get_slide_collision(int p_bounce) {
	ERR_FAIL_INDEX_V(p_bounce, colliders.size(), Ref<KinematicCollision2D>());
	if (p_bounce >= slide_colliders.size()) {
		slide_colliders.resize(p_bounce + 1);
	}

	// Create a new instance when the cached reference is invalid or still in use in script.
	if (slide_colliders[p_bounce].is_null() || slide_colliders[p_bounce]->reference_get_count() > 1) {
		slide_colliders.write[p_bounce].instance();
		slide_colliders.write[p_bounce]->owner = this;
	}

	slide_colliders.write[p_bounce]->collision = colliders[p_bounce];
	return slide_colliders[p_bounce];
}

Ref<KinematicCollision2D> KinematicBody2D::_get_last_slide_collision() {
	if (colliders.size() == 0) {
		return Ref<KinematicCollision2D>();
	}
	return _get_slide_collision(colliders.size() - 1);
}

void KinematicBody2D::set_sync_to_physics(bool p_enable) {
	if (sync_to_physics == p_enable) {
		return;
	}
	sync_to_physics = p_enable;

	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	if (p_enable) {
		Physics2DServer::get_singleton()->body_set_force_integration_callback(get_rid(), this, "_direct_state_changed");
		set_only_update_transform_changes(true);
		set_notify_local_transform(true);
	} else {
		Physics2DServer::get_singleton()->body_set_force_integration_callback(get_rid(), nullptr, "");
		set_only_update_transform_changes(false);
		set_notify_local_transform(false);
	}
}

bool KinematicBody2D::is_sync_to_physics_enabled() const {
	return sync_to_physics;
}

void KinematicBody2D::_direct_state_changed(Object *p_state) {
	if (!sync_to_physics) {
		return;
	}

	Physics2DDirectBodyState *state = Object::cast_to<Physics2DDirectBodyState>(p_state);
	ERR_FAIL_COND_MSG(!state, "Method '_direct_state_changed' must receive a valid Physics2DDirectBodyState object as argument");

	last_valid_transform = state->get_transform();
	set_notify_local_transform(false);
	set_global_transform(last_valid_transform);
	set_notify_local_transform(true);
}

void KinematicBody2D::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		last_valid_transform = get_global_transform();

		// Reset move_and_slide() data.
		on_floor = false;
		on_floor_body = RID();
		on_ceiling = false;
		on_wall = false;
		colliders.clear();
		floor_velocity = Vector2();
	}

	if (p_what == NOTIFICATION_LOCAL_TRANSFORM_CHANGED) {
		//used by sync to physics, send the new transform to the physics
		Transform2D new_transform = get_global_transform();
		Physics2DServer::get_singleton()->body_set_state(get_rid(), Physics2DServer::BODY_STATE_TRANSFORM, new_transform);
		//but then revert changes
		set_notify_local_transform(false);
		set_global_transform(last_valid_transform);
		set_notify_local_transform(true);
	}
}
void KinematicBody2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("move_and_collide", "rel_vec", "infinite_inertia", "exclude_raycast_shapes", "test_only"), &KinematicBody2D::_move, DEFVAL(true), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("move_and_slide", "linear_velocity", "up_direction", "stop_on_slope", "max_slides", "floor_max_angle", "infinite_inertia"), &KinematicBody2D::move_and_slide, DEFVAL(Vector2(0, 0)), DEFVAL(false), DEFVAL(4), DEFVAL(Math::deg2rad((float)45)), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("move_and_slide_with_snap", "linear_velocity", "snap", "up_direction", "stop_on_slope", "max_slides", "floor_max_angle", "infinite_inertia"), &KinematicBody2D::move_and_slide_with_snap, DEFVAL(Vector2(0, 0)), DEFVAL(false), DEFVAL(4), DEFVAL(Math::deg2rad((float)45)), DEFVAL(true));

	ClassDB::bind_method(D_METHOD("test_move", "from", "rel_vec", "infinite_inertia"), &KinematicBody2D::test_move, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("is_on_floor"), &KinematicBody2D::is_on_floor);
	ClassDB::bind_method(D_METHOD("is_on_ceiling"), &KinematicBody2D::is_on_ceiling);
	ClassDB::bind_method(D_METHOD("is_on_wall"), &KinematicBody2D::is_on_wall);
	ClassDB::bind_method(D_METHOD("get_floor_normal"), &KinematicBody2D::get_floor_normal);
	ClassDB::bind_method(D_METHOD("get_floor_angle", "up_direction"), &KinematicBody2D::get_floor_angle, DEFVAL(Vector2(0.0, -1.0)));
	ClassDB::bind_method(D_METHOD("get_floor_velocity"), &KinematicBody2D::get_floor_velocity);

	ClassDB::bind_method(D_METHOD("set_safe_margin", "pixels"), &KinematicBody2D::set_safe_margin);
	ClassDB::bind_method(D_METHOD("get_safe_margin"), &KinematicBody2D::get_safe_margin);

	ClassDB::bind_method(D_METHOD("set_moving_platform_apply_velocity_on_leave", "on_leave_apply_velocity"), &KinematicBody2D::set_moving_platform_apply_velocity_on_leave);
	ClassDB::bind_method(D_METHOD("get_moving_platform_apply_velocity_on_leave"), &KinematicBody2D::get_moving_platform_apply_velocity_on_leave);

	ClassDB::bind_method(D_METHOD("get_slide_count"), &KinematicBody2D::get_slide_count);
	ClassDB::bind_method(D_METHOD("get_slide_collision", "slide_idx"), &KinematicBody2D::_get_slide_collision);
	ClassDB::bind_method(D_METHOD("get_last_slide_collision"), &KinematicBody2D::_get_last_slide_collision);

	ClassDB::bind_method(D_METHOD("set_sync_to_physics", "enable"), &KinematicBody2D::set_sync_to_physics);
	ClassDB::bind_method(D_METHOD("is_sync_to_physics_enabled"), &KinematicBody2D::is_sync_to_physics_enabled);

	ClassDB::bind_method(D_METHOD("_direct_state_changed"), &KinematicBody2D::_direct_state_changed);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "collision/safe_margin", PROPERTY_HINT_RANGE, "0.001,256,0.001"), "set_safe_margin", "get_safe_margin");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motion/sync_to_physics"), "set_sync_to_physics", "is_sync_to_physics_enabled");
	ADD_GROUP("Moving platform", "moving_platform");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "moving_platform_apply_velocity_on_leave", PROPERTY_HINT_ENUM, "Always,Upward Only,Never", PROPERTY_USAGE_DEFAULT), "set_moving_platform_apply_velocity_on_leave", "get_moving_platform_apply_velocity_on_leave");

	BIND_ENUM_CONSTANT(PLATFORM_VEL_ON_LEAVE_ALWAYS);
	BIND_ENUM_CONSTANT(PLATFORM_VEL_ON_LEAVE_UPWARD_ONLY);
	BIND_ENUM_CONSTANT(PLATFORM_VEL_ON_LEAVE_NEVER);
}

KinematicBody2D::KinematicBody2D() :
		PhysicsBody2D(Physics2DServer::BODY_MODE_KINEMATIC) {
	margin = 0.08;

	on_floor = false;
	on_ceiling = false;
	on_wall = false;
	sync_to_physics = false;
}
KinematicBody2D::~KinematicBody2D() {
	if (motion_cache.is_valid()) {
		motion_cache->owner = nullptr;
	}

	for (int i = 0; i < slide_colliders.size(); i++) {
		if (slide_colliders[i].is_valid()) {
			slide_colliders.write[i]->owner = nullptr;
		}
	}
}

////////////////////////

Vector2 KinematicCollision2D::get_position() const {
	return collision.collision;
}
Vector2 KinematicCollision2D::get_normal() const {
	return collision.normal;
}
Vector2 KinematicCollision2D::get_travel() const {
	return collision.travel;
}
Vector2 KinematicCollision2D::get_remainder() const {
	return collision.remainder;
}

real_t KinematicCollision2D::get_angle(const Vector2 &p_up_direction) const {
	ERR_FAIL_COND_V(p_up_direction == Vector2(), 0);
	return collision.get_angle(p_up_direction);
}

Object *KinematicCollision2D::get_local_shape() const {
	if (!owner) {
		return nullptr;
	}
	uint32_t ownerid = owner->shape_find_owner(collision.local_shape);
	return owner->shape_owner_get_owner(ownerid);
}

Object *KinematicCollision2D::get_collider() const {
	if (collision.collider) {
		return ObjectDB::get_instance(collision.collider);
	}

	return nullptr;
}
ObjectID KinematicCollision2D::get_collider_id() const {
	return collision.collider;
}
RID KinematicCollision2D::get_collider_rid() const {
	return collision.collider_rid;
}
Object *KinematicCollision2D::get_collider_shape() const {
	Object *collider = get_collider();
	if (collider) {
		CollisionObject2D *obj2d = Object::cast_to<CollisionObject2D>(collider);
		if (obj2d) {
			uint32_t ownerid = obj2d->shape_find_owner(collision.collider_shape);
			return obj2d->shape_owner_get_owner(ownerid);
		}
	}

	return nullptr;
}
int KinematicCollision2D::get_collider_shape_index() const {
	return collision.collider_shape;
}
Vector2 KinematicCollision2D::get_collider_velocity() const {
	return collision.collider_vel;
}
Variant KinematicCollision2D::get_collider_metadata() const {
	return Variant();
}

void KinematicCollision2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_position"), &KinematicCollision2D::get_position);
	ClassDB::bind_method(D_METHOD("get_normal"), &KinematicCollision2D::get_normal);
	ClassDB::bind_method(D_METHOD("get_travel"), &KinematicCollision2D::get_travel);
	ClassDB::bind_method(D_METHOD("get_remainder"), &KinematicCollision2D::get_remainder);
	ClassDB::bind_method(D_METHOD("get_angle", "up_direction"), &KinematicCollision2D::get_angle, DEFVAL(Vector2(0.0, -1.0)));
	ClassDB::bind_method(D_METHOD("get_local_shape"), &KinematicCollision2D::get_local_shape);
	ClassDB::bind_method(D_METHOD("get_collider"), &KinematicCollision2D::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_id"), &KinematicCollision2D::get_collider_id);
	ClassDB::bind_method(D_METHOD("get_collider_rid"), &KinematicCollision2D::get_collider_rid);
	ClassDB::bind_method(D_METHOD("get_collider_shape"), &KinematicCollision2D::get_collider_shape);
	ClassDB::bind_method(D_METHOD("get_collider_shape_index"), &KinematicCollision2D::get_collider_shape_index);
	ClassDB::bind_method(D_METHOD("get_collider_velocity"), &KinematicCollision2D::get_collider_velocity);
	ClassDB::bind_method(D_METHOD("get_collider_metadata"), &KinematicCollision2D::get_collider_metadata);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position"), "", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "normal"), "", "get_normal");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "travel"), "", "get_travel");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "remainder"), "", "get_remainder");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "local_shape"), "", "get_local_shape");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collider"), "", "get_collider");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_id"), "", "get_collider_id");
	ADD_PROPERTY(PropertyInfo(Variant::_RID, "collider_rid"), "", "get_collider_rid");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collider_shape"), "", "get_collider_shape");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_shape_index"), "", "get_collider_shape_index");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "collider_velocity"), "", "get_collider_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::NIL, "collider_metadata", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT), "", "get_collider_metadata");
}

KinematicCollision2D::KinematicCollision2D() {
	collision.collider = 0;
	collision.collider_shape = 0;
	collision.local_shape = 0;
	owner = nullptr;
}
