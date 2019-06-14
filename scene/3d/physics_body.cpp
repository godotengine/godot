/*************************************************************************/
/*  physics_body.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/core_string_names.h"
#include "core/engine.h"
#include "core/list.h"
#include "core/method_bind_ext.gen.inc"
#include "core/object.h"
#include "core/rid.h"
#include "scene/scene_string_names.h"

#ifdef TOOLS_ENABLED
#include "editor/plugins/spatial_editor_plugin.h"
#endif

void PhysicsBody::_notification(int p_what) {
}

Vector3 PhysicsBody::get_linear_velocity() const {

	return Vector3();
}
Vector3 PhysicsBody::get_angular_velocity() const {

	return Vector3();
}

float PhysicsBody::get_inverse_mass() const {

	return 0;
}

void PhysicsBody::_bind_methods() {
}

PhysicsBody::PhysicsBody(PhysicsServer::BodyMode p_mode) :
		CollisionObject(PhysicsServer::get_singleton()->body_create(p_mode), COLLISION_OBJECT_TYPE_BODY) {
}

#ifndef DISABLE_DEPRECATED
void StaticBody::set_friction(real_t p_friction) {

	if (p_friction == 1.0 && physics_material_override.is_null()) { // default value, don't create an override for that
		return;
	}

	ERR_EXPLAIN("The method set_friction has been deprecated and will be removed in the future, use physics material instead.");
	WARN_DEPRECATED;

	ERR_FAIL_COND(p_friction < 0 || p_friction > 1);

	if (physics_material_override.is_null()) {
		physics_material_override.instance();
		set_physics_material_override(physics_material_override);
	}
	physics_material_override->set_friction(p_friction);
}

real_t StaticBody::get_friction() const {

	ERR_EXPLAIN("The method get_friction has been deprecated and will be removed in the future, use physics material instead.");
	WARN_DEPRECATED;

	if (physics_material_override.is_null()) {
		return 1;
	}

	return physics_material_override->get_friction();
}

void StaticBody::set_bounce(real_t p_bounce) {

	if (p_bounce == 0.0 && physics_material_override.is_null()) { // default value, don't create an override for that
		return;
	}

	ERR_EXPLAIN("The method set_bounce has been deprecated and will be removed in the future, use physics material instead.");
	WARN_DEPRECATED;

	ERR_FAIL_COND(p_bounce < 0 || p_bounce > 1);

	if (physics_material_override.is_null()) {
		physics_material_override.instance();
		set_physics_material_override(physics_material_override);
	}
	physics_material_override->set_bounce(p_bounce);
}

real_t StaticBody::get_bounce() const {

	ERR_EXPLAIN("The method get_bounce has been deprecated and will be removed in the future, use physics material instead.");
	WARN_DEPRECATED;

	if (physics_material_override.is_null()) {
		return 0;
	}

	return physics_material_override->get_bounce();
}
#endif

void StaticBody::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
	if (physics_material_override.is_valid()) {
		if (physics_material_override->is_connected(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics"))
			physics_material_override->disconnect(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics");
	}

	physics_material_override = p_physics_material_override;

	if (physics_material_override.is_valid()) {
		physics_material_override->connect(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics");
	}
	_reload_physics_characteristics();
}

Ref<PhysicsMaterial> StaticBody::get_physics_material_override() const {
	return physics_material_override;
}

void StaticBody::set_constant_linear_velocity(const Vector3 &p_vel) {

	constant_linear_velocity = p_vel;
	PhysicsServer::get_singleton()->body_set_state(get_rid(), PhysicsServer::BODY_STATE_LINEAR_VELOCITY, constant_linear_velocity);
}

void StaticBody::set_constant_angular_velocity(const Vector3 &p_vel) {

	constant_angular_velocity = p_vel;
	PhysicsServer::get_singleton()->body_set_state(get_rid(), PhysicsServer::BODY_STATE_ANGULAR_VELOCITY, constant_angular_velocity);
}

Vector3 StaticBody::get_constant_linear_velocity() const {

	return constant_linear_velocity;
}
Vector3 StaticBody::get_constant_angular_velocity() const {

	return constant_angular_velocity;
}

void StaticBody::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_constant_linear_velocity", "vel"), &StaticBody::set_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_constant_angular_velocity", "vel"), &StaticBody::set_constant_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_linear_velocity"), &StaticBody::get_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_angular_velocity"), &StaticBody::get_constant_angular_velocity);

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &StaticBody::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &StaticBody::get_friction);

	ClassDB::bind_method(D_METHOD("set_bounce", "bounce"), &StaticBody::set_bounce);
	ClassDB::bind_method(D_METHOD("get_bounce"), &StaticBody::get_bounce);
#endif // DISABLE_DEPRECATED

	ClassDB::bind_method(D_METHOD("set_physics_material_override", "physics_material_override"), &StaticBody::set_physics_material_override);
	ClassDB::bind_method(D_METHOD("get_physics_material_override"), &StaticBody::get_physics_material_override);

	ClassDB::bind_method(D_METHOD("_reload_physics_characteristics"), &StaticBody::_reload_physics_characteristics);

	ClassDB::bind_method(D_METHOD("get_collision_exceptions"), &PhysicsBody::get_collision_exceptions);
	ClassDB::bind_method(D_METHOD("add_collision_exception_with", "body"), &PhysicsBody::add_collision_exception_with);
	ClassDB::bind_method(D_METHOD("remove_collision_exception_with", "body"), &PhysicsBody::remove_collision_exception_with);

#ifndef DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "friction", PROPERTY_HINT_RANGE, "0,1,0.01", 0), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce", PROPERTY_HINT_RANGE, "0,1,0.01", 0), "set_bounce", "get_bounce");
#endif // DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material_override", "get_physics_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "constant_linear_velocity"), "set_constant_linear_velocity", "get_constant_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "constant_angular_velocity"), "set_constant_angular_velocity", "get_constant_angular_velocity");
}

StaticBody::StaticBody() :
		PhysicsBody(PhysicsServer::BODY_MODE_STATIC) {
}

StaticBody::~StaticBody() {}

void StaticBody::_reload_physics_characteristics() {
	if (physics_material_override.is_null()) {
		PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_BOUNCE, 0);
		PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_FRICTION, 1);
	} else {
		PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_BOUNCE, physics_material_override->computed_bounce());
		PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_FRICTION, physics_material_override->computed_friction());
	}
}

void RigidBody::_body_enter_tree(ObjectID p_id) {

	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = Object::cast_to<Node>(obj);
	ERR_FAIL_COND(!node);

	ERR_FAIL_COND(!contact_monitor);
	Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(E->get().in_tree);

	E->get().in_tree = true;

	contact_monitor->locked = true;

	emit_signal(SceneStringNames::get_singleton()->body_entered, node);

	for (int i = 0; i < E->get().shapes.size(); i++) {

		emit_signal(SceneStringNames::get_singleton()->body_shape_entered, p_id, node, E->get().shapes[i].body_shape, E->get().shapes[i].local_shape);
	}

	contact_monitor->locked = false;
}

void RigidBody::_body_exit_tree(ObjectID p_id) {

	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = Object::cast_to<Node>(obj);
	ERR_FAIL_COND(!node);
	ERR_FAIL_COND(!contact_monitor);
	Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(!E->get().in_tree);
	E->get().in_tree = false;

	contact_monitor->locked = true;

	emit_signal(SceneStringNames::get_singleton()->body_exited, node);

	for (int i = 0; i < E->get().shapes.size(); i++) {

		emit_signal(SceneStringNames::get_singleton()->body_shape_exited, p_id, node, E->get().shapes[i].body_shape, E->get().shapes[i].local_shape);
	}

	contact_monitor->locked = false;
}

void RigidBody::_body_inout(int p_status, ObjectID p_instance, int p_body_shape, int p_local_shape) {

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
			//E->get().rc=0;
			E->get().in_tree = node && node->is_inside_tree();
			if (node) {
				node->connect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree, make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_body_exit_tree, make_binds(objid));
				if (E->get().in_tree) {
					emit_signal(SceneStringNames::get_singleton()->body_entered, node);
				}
			}
		}
		//E->get().rc++;
		if (node)
			E->get().shapes.insert(ShapePair(p_body_shape, p_local_shape));

		if (E->get().in_tree) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_entered, objid, node, p_body_shape, p_local_shape);
		}

	} else {

		//E->get().rc--;

		if (node)
			E->get().shapes.erase(ShapePair(p_body_shape, p_local_shape));

		bool in_tree = E->get().in_tree;

		if (E->get().shapes.empty()) {

			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree);
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_body_exit_tree);
				if (in_tree)
					emit_signal(SceneStringNames::get_singleton()->body_exited, node);
			}

			contact_monitor->body_map.erase(E);
		}
		if (node && in_tree) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_exited, objid, obj, p_body_shape, p_local_shape);
		}
	}
}

struct _RigidBodyInOut {

	ObjectID id;
	int shape;
	int local_shape;
};

void RigidBody::_direct_state_changed(Object *p_state) {

#ifdef DEBUG_ENABLED
	state = Object::cast_to<PhysicsDirectBodyState>(p_state);
#else
	state = (PhysicsDirectBodyState *)p_state; //trust it
#endif

	set_ignore_transform_notification(true);
	set_global_transform(state->get_transform());
	linear_velocity = state->get_linear_velocity();
	angular_velocity = state->get_angular_velocity();
	if (sleeping != state->is_sleeping()) {
		sleeping = state->is_sleeping();
		emit_signal(SceneStringNames::get_singleton()->sleeping_state_changed);
	}
	if (get_script_instance())
		get_script_instance()->call("_integrate_forces", state);
	set_ignore_transform_notification(false);

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

		_RigidBodyInOut *toadd = (_RigidBodyInOut *)alloca(state->get_contact_count() * sizeof(_RigidBodyInOut));
		int toadd_count = 0; //state->get_contact_count();
		RigidBody_RemoveAction *toremove = (RigidBody_RemoveAction *)alloca(rc * sizeof(RigidBody_RemoveAction));
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

void RigidBody::_notification(int p_what) {

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

void RigidBody::set_mode(Mode p_mode) {

	mode = p_mode;
	switch (p_mode) {

		case MODE_RIGID: {

			PhysicsServer::get_singleton()->body_set_mode(get_rid(), PhysicsServer::BODY_MODE_RIGID);
		} break;
		case MODE_STATIC: {

			PhysicsServer::get_singleton()->body_set_mode(get_rid(), PhysicsServer::BODY_MODE_STATIC);

		} break;
		case MODE_CHARACTER: {
			PhysicsServer::get_singleton()->body_set_mode(get_rid(), PhysicsServer::BODY_MODE_CHARACTER);

		} break;
		case MODE_KINEMATIC: {

			PhysicsServer::get_singleton()->body_set_mode(get_rid(), PhysicsServer::BODY_MODE_KINEMATIC);
		} break;
	}
}

RigidBody::Mode RigidBody::get_mode() const {

	return mode;
}

void RigidBody::set_mass(real_t p_mass) {

	ERR_FAIL_COND(p_mass <= 0);
	mass = p_mass;
	_change_notify("mass");
	_change_notify("weight");
	PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_MASS, mass);
}
real_t RigidBody::get_mass() const {

	return mass;
}

void RigidBody::set_weight(real_t p_weight) {

	set_mass(p_weight / real_t(GLOBAL_DEF("physics/3d/default_gravity", 9.8)));
}
real_t RigidBody::get_weight() const {

	return mass * real_t(GLOBAL_DEF("physics/3d/default_gravity", 9.8));
}

#ifndef DISABLE_DEPRECATED
void RigidBody::set_friction(real_t p_friction) {

	if (p_friction == 1.0 && physics_material_override.is_null()) { // default value, don't create an override for that
		return;
	}

	ERR_EXPLAIN("The method set_friction has been deprecated and will be removed in the future, use physics material instead.");
	WARN_DEPRECATED;

	ERR_FAIL_COND(p_friction < 0 || p_friction > 1);

	if (physics_material_override.is_null()) {
		physics_material_override.instance();
		set_physics_material_override(physics_material_override);
	}
	physics_material_override->set_friction(p_friction);
}
real_t RigidBody::get_friction() const {

	ERR_EXPLAIN("The method get_friction has been deprecated and will be removed in the future, use physics material instead.");
	WARN_DEPRECATED;

	if (physics_material_override.is_null()) {
		return 1;
	}

	return physics_material_override->get_friction();
}

void RigidBody::set_bounce(real_t p_bounce) {

	if (p_bounce == 0.0 && physics_material_override.is_null()) { // default value, don't create an override for that
		return;
	}

	ERR_EXPLAIN("The method set_bounce has been deprecated and will be removed in the future, use physics material instead.");
	WARN_DEPRECATED;

	ERR_FAIL_COND(p_bounce < 0 || p_bounce > 1);

	if (physics_material_override.is_null()) {
		physics_material_override.instance();
		set_physics_material_override(physics_material_override);
	}
	physics_material_override->set_bounce(p_bounce);
}
real_t RigidBody::get_bounce() const {
	ERR_EXPLAIN("The method get_bounce has been deprecated and will be removed in the future, use physics material instead.");
	WARN_DEPRECATED;
	if (physics_material_override.is_null()) {
		return 0;
	}

	return physics_material_override->get_bounce();
}
#endif // DISABLE_DEPRECATED

void RigidBody::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
	if (physics_material_override.is_valid()) {
		if (physics_material_override->is_connected(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics"))
			physics_material_override->disconnect(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics");
	}

	physics_material_override = p_physics_material_override;

	if (physics_material_override.is_valid()) {
		physics_material_override->connect(CoreStringNames::get_singleton()->changed, this, "_reload_physics_characteristics");
	}
	_reload_physics_characteristics();
}

Ref<PhysicsMaterial> RigidBody::get_physics_material_override() const {
	return physics_material_override;
}

void RigidBody::set_gravity_scale(real_t p_gravity_scale) {

	gravity_scale = p_gravity_scale;
	PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_GRAVITY_SCALE, gravity_scale);
}
real_t RigidBody::get_gravity_scale() const {

	return gravity_scale;
}

void RigidBody::set_linear_damp(real_t p_linear_damp) {

	ERR_FAIL_COND(p_linear_damp < -1);
	linear_damp = p_linear_damp;
	PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_LINEAR_DAMP, linear_damp);
}
real_t RigidBody::get_linear_damp() const {

	return linear_damp;
}

void RigidBody::set_angular_damp(real_t p_angular_damp) {

	ERR_FAIL_COND(p_angular_damp < -1);
	angular_damp = p_angular_damp;
	PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_ANGULAR_DAMP, angular_damp);
}
real_t RigidBody::get_angular_damp() const {

	return angular_damp;
}

void RigidBody::set_axis_velocity(const Vector3 &p_axis) {

	Vector3 v = state ? state->get_linear_velocity() : linear_velocity;
	Vector3 axis = p_axis.normalized();
	v -= axis * axis.dot(v);
	v += p_axis;
	if (state) {
		set_linear_velocity(v);
	} else {
		PhysicsServer::get_singleton()->body_set_axis_velocity(get_rid(), p_axis);
		linear_velocity = v;
	}
}

void RigidBody::set_linear_velocity(const Vector3 &p_velocity) {

	linear_velocity = p_velocity;
	if (state)
		state->set_linear_velocity(linear_velocity);
	else
		PhysicsServer::get_singleton()->body_set_state(get_rid(), PhysicsServer::BODY_STATE_LINEAR_VELOCITY, linear_velocity);
}

Vector3 RigidBody::get_linear_velocity() const {

	return linear_velocity;
}

void RigidBody::set_angular_velocity(const Vector3 &p_velocity) {

	angular_velocity = p_velocity;
	if (state)
		state->set_angular_velocity(angular_velocity);
	else
		PhysicsServer::get_singleton()->body_set_state(get_rid(), PhysicsServer::BODY_STATE_ANGULAR_VELOCITY, angular_velocity);
}
Vector3 RigidBody::get_angular_velocity() const {

	return angular_velocity;
}

void RigidBody::set_use_custom_integrator(bool p_enable) {

	if (custom_integrator == p_enable)
		return;

	custom_integrator = p_enable;
	PhysicsServer::get_singleton()->body_set_omit_force_integration(get_rid(), p_enable);
}
bool RigidBody::is_using_custom_integrator() {

	return custom_integrator;
}

void RigidBody::set_sleeping(bool p_sleeping) {

	sleeping = p_sleeping;
	PhysicsServer::get_singleton()->body_set_state(get_rid(), PhysicsServer::BODY_STATE_SLEEPING, sleeping);
}

void RigidBody::set_can_sleep(bool p_active) {

	can_sleep = p_active;
	PhysicsServer::get_singleton()->body_set_state(get_rid(), PhysicsServer::BODY_STATE_CAN_SLEEP, p_active);
}

bool RigidBody::is_able_to_sleep() const {

	return can_sleep;
}

bool RigidBody::is_sleeping() const {

	return sleeping;
}

void RigidBody::set_max_contacts_reported(int p_amount) {

	max_contacts_reported = p_amount;
	PhysicsServer::get_singleton()->body_set_max_contacts_reported(get_rid(), p_amount);
}

int RigidBody::get_max_contacts_reported() const {

	return max_contacts_reported;
}

void RigidBody::add_central_force(const Vector3 &p_force) {
	PhysicsServer::get_singleton()->body_add_central_force(get_rid(), p_force);
}

void RigidBody::add_force(const Vector3 &p_force, const Vector3 &p_pos) {
	PhysicsServer::get_singleton()->body_add_force(get_rid(), p_force, p_pos);
}

void RigidBody::add_torque(const Vector3 &p_torque) {
	PhysicsServer::get_singleton()->body_add_torque(get_rid(), p_torque);
}

void RigidBody::apply_central_impulse(const Vector3 &p_impulse) {
	PhysicsServer::get_singleton()->body_apply_central_impulse(get_rid(), p_impulse);
}

void RigidBody::apply_impulse(const Vector3 &p_pos, const Vector3 &p_impulse) {

	PhysicsServer::get_singleton()->body_apply_impulse(get_rid(), p_pos, p_impulse);
}

void RigidBody::apply_torque_impulse(const Vector3 &p_impulse) {
	PhysicsServer::get_singleton()->body_apply_torque_impulse(get_rid(), p_impulse);
}

void RigidBody::set_use_continuous_collision_detection(bool p_enable) {

	ccd = p_enable;
	PhysicsServer::get_singleton()->body_set_enable_continuous_collision_detection(get_rid(), p_enable);
}

bool RigidBody::is_using_continuous_collision_detection() const {

	return ccd;
}

void RigidBody::set_contact_monitor(bool p_enabled) {

	if (p_enabled == is_contact_monitor_enabled())
		return;

	if (!p_enabled) {

		if (contact_monitor->locked) {
			ERR_EXPLAIN("Can't disable contact monitoring during in/out callback. Use call_deferred(\"set_contact_monitor\",false) instead");
		}
		ERR_FAIL_COND(contact_monitor->locked);

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
		contact_monitor = NULL;
	} else {

		contact_monitor = memnew(ContactMonitor);
		contact_monitor->locked = false;
	}
}

bool RigidBody::is_contact_monitor_enabled() const {

	return contact_monitor != NULL;
}

void RigidBody::set_axis_lock(PhysicsServer::BodyAxis p_axis, bool p_lock) {
	PhysicsServer::get_singleton()->body_set_axis_lock(get_rid(), p_axis, p_lock);
}

bool RigidBody::get_axis_lock(PhysicsServer::BodyAxis p_axis) const {
	return PhysicsServer::get_singleton()->body_is_axis_locked(get_rid(), p_axis);
}

Array RigidBody::get_colliding_bodies() const {

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

String RigidBody::get_configuration_warning() const {

	Transform t = get_transform();

	String warning = CollisionObject::get_configuration_warning();

	if ((get_mode() == MODE_RIGID || get_mode() == MODE_CHARACTER) && (ABS(t.basis.get_axis(0).length() - 1.0) > 0.05 || ABS(t.basis.get_axis(1).length() - 1.0) > 0.05 || ABS(t.basis.get_axis(2).length() - 1.0) > 0.05)) {
		if (warning != String()) {
			warning += "\n";
		}
		warning += TTR("Size changes to RigidBody (in character or rigid modes) will be overridden by the physics engine when running.\nChange the size in children collision shapes instead.");
	}

	return warning;
}

void RigidBody::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &RigidBody::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &RigidBody::get_mode);

	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &RigidBody::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &RigidBody::get_mass);

	ClassDB::bind_method(D_METHOD("set_weight", "weight"), &RigidBody::set_weight);
	ClassDB::bind_method(D_METHOD("get_weight"), &RigidBody::get_weight);

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &RigidBody::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &RigidBody::get_friction);

	ClassDB::bind_method(D_METHOD("set_bounce", "bounce"), &RigidBody::set_bounce);
	ClassDB::bind_method(D_METHOD("get_bounce"), &RigidBody::get_bounce);
#endif // DISABLE_DEPRECATED

	ClassDB::bind_method(D_METHOD("set_physics_material_override", "physics_material_override"), &RigidBody::set_physics_material_override);
	ClassDB::bind_method(D_METHOD("get_physics_material_override"), &RigidBody::get_physics_material_override);

	ClassDB::bind_method(D_METHOD("_reload_physics_characteristics"), &RigidBody::_reload_physics_characteristics);

	ClassDB::bind_method(D_METHOD("set_linear_velocity", "linear_velocity"), &RigidBody::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &RigidBody::get_linear_velocity);

	ClassDB::bind_method(D_METHOD("set_angular_velocity", "angular_velocity"), &RigidBody::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &RigidBody::get_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_gravity_scale", "gravity_scale"), &RigidBody::set_gravity_scale);
	ClassDB::bind_method(D_METHOD("get_gravity_scale"), &RigidBody::get_gravity_scale);

	ClassDB::bind_method(D_METHOD("set_linear_damp", "linear_damp"), &RigidBody::set_linear_damp);
	ClassDB::bind_method(D_METHOD("get_linear_damp"), &RigidBody::get_linear_damp);

	ClassDB::bind_method(D_METHOD("set_angular_damp", "angular_damp"), &RigidBody::set_angular_damp);
	ClassDB::bind_method(D_METHOD("get_angular_damp"), &RigidBody::get_angular_damp);

	ClassDB::bind_method(D_METHOD("set_max_contacts_reported", "amount"), &RigidBody::set_max_contacts_reported);
	ClassDB::bind_method(D_METHOD("get_max_contacts_reported"), &RigidBody::get_max_contacts_reported);

	ClassDB::bind_method(D_METHOD("set_use_custom_integrator", "enable"), &RigidBody::set_use_custom_integrator);
	ClassDB::bind_method(D_METHOD("is_using_custom_integrator"), &RigidBody::is_using_custom_integrator);

	ClassDB::bind_method(D_METHOD("set_contact_monitor", "enabled"), &RigidBody::set_contact_monitor);
	ClassDB::bind_method(D_METHOD("is_contact_monitor_enabled"), &RigidBody::is_contact_monitor_enabled);

	ClassDB::bind_method(D_METHOD("set_use_continuous_collision_detection", "enable"), &RigidBody::set_use_continuous_collision_detection);
	ClassDB::bind_method(D_METHOD("is_using_continuous_collision_detection"), &RigidBody::is_using_continuous_collision_detection);

	ClassDB::bind_method(D_METHOD("set_axis_velocity", "axis_velocity"), &RigidBody::set_axis_velocity);

	ClassDB::bind_method(D_METHOD("add_central_force", "force"), &RigidBody::add_central_force);
	ClassDB::bind_method(D_METHOD("add_force", "force", "position"), &RigidBody::add_force);
	ClassDB::bind_method(D_METHOD("add_torque", "torque"), &RigidBody::add_torque);

	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &RigidBody::apply_central_impulse);
	ClassDB::bind_method(D_METHOD("apply_impulse", "position", "impulse"), &RigidBody::apply_impulse);
	ClassDB::bind_method(D_METHOD("apply_torque_impulse", "impulse"), &RigidBody::apply_torque_impulse);

	ClassDB::bind_method(D_METHOD("set_sleeping", "sleeping"), &RigidBody::set_sleeping);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &RigidBody::is_sleeping);

	ClassDB::bind_method(D_METHOD("set_can_sleep", "able_to_sleep"), &RigidBody::set_can_sleep);
	ClassDB::bind_method(D_METHOD("is_able_to_sleep"), &RigidBody::is_able_to_sleep);

	ClassDB::bind_method(D_METHOD("_direct_state_changed"), &RigidBody::_direct_state_changed);
	ClassDB::bind_method(D_METHOD("_body_enter_tree"), &RigidBody::_body_enter_tree);
	ClassDB::bind_method(D_METHOD("_body_exit_tree"), &RigidBody::_body_exit_tree);

	ClassDB::bind_method(D_METHOD("set_axis_lock", "axis", "lock"), &RigidBody::set_axis_lock);
	ClassDB::bind_method(D_METHOD("get_axis_lock", "axis"), &RigidBody::get_axis_lock);

	ClassDB::bind_method(D_METHOD("get_colliding_bodies"), &RigidBody::get_colliding_bodies);

	BIND_VMETHOD(MethodInfo("_integrate_forces", PropertyInfo(Variant::OBJECT, "state", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsDirectBodyState")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Rigid,Static,Character,Kinematic"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "mass", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "weight", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01", PROPERTY_USAGE_EDITOR), "set_weight", "get_weight");
#ifndef DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "friction", PROPERTY_HINT_RANGE, "0,1,0.01", 0), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce", PROPERTY_HINT_RANGE, "0,1,0.01", 0), "set_bounce", "get_bounce");
#endif // DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material_override", "get_physics_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "gravity_scale", PROPERTY_HINT_RANGE, "-128,128,0.01"), "set_gravity_scale", "get_gravity_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "custom_integrator"), "set_use_custom_integrator", "is_using_custom_integrator");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "continuous_cd"), "set_use_continuous_collision_detection", "is_using_continuous_collision_detection");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "contacts_reported"), "set_max_contacts_reported", "get_max_contacts_reported");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "contact_monitor"), "set_contact_monitor", "is_contact_monitor_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleeping", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "can_sleep"), "set_can_sleep", "is_able_to_sleep");
	ADD_GROUP("Axis Lock", "axis_lock_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_linear_x"), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_X);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_linear_y"), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_Y);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_linear_z"), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_Z);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_angular_x"), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_ANGULAR_X);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_angular_y"), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_ANGULAR_Y);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_angular_z"), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_ANGULAR_Z);
	ADD_GROUP("Linear", "linear_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "linear_damp", PROPERTY_HINT_RANGE, "-1,128,0.01"), "set_linear_damp", "get_linear_damp");
	ADD_GROUP("Angular", "angular_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_damp", PROPERTY_HINT_RANGE, "-1,128,0.01"), "set_angular_damp", "get_angular_damp");

	ADD_SIGNAL(MethodInfo("body_shape_entered", PropertyInfo(Variant::INT, "body_id"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape"), PropertyInfo(Variant::INT, "local_shape")));
	ADD_SIGNAL(MethodInfo("body_shape_exited", PropertyInfo(Variant::INT, "body_id"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape"), PropertyInfo(Variant::INT, "local_shape")));
	ADD_SIGNAL(MethodInfo("body_entered", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("body_exited", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("sleeping_state_changed"));

	BIND_ENUM_CONSTANT(MODE_RIGID);
	BIND_ENUM_CONSTANT(MODE_STATIC);
	BIND_ENUM_CONSTANT(MODE_CHARACTER);
	BIND_ENUM_CONSTANT(MODE_KINEMATIC);
}

RigidBody::RigidBody() :
		PhysicsBody(PhysicsServer::BODY_MODE_RIGID) {

	mode = MODE_RIGID;

	mass = 1;
	max_contacts_reported = 0;
	state = NULL;

	gravity_scale = 1;
	linear_damp = -1;
	angular_damp = -1;

	//angular_velocity=0;
	sleeping = false;
	ccd = false;

	custom_integrator = false;
	contact_monitor = NULL;
	can_sleep = true;

	PhysicsServer::get_singleton()->body_set_force_integration_callback(get_rid(), this, "_direct_state_changed");
}

RigidBody::~RigidBody() {

	if (contact_monitor)
		memdelete(contact_monitor);
}

void RigidBody::_reload_physics_characteristics() {
	if (physics_material_override.is_null()) {
		PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_BOUNCE, 0);
		PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_FRICTION, 1);
	} else {
		PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_BOUNCE, physics_material_override->computed_bounce());
		PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_FRICTION, physics_material_override->computed_friction());
	}
}

//////////////////////////////////////////////////////
//////////////////////////

Ref<KinematicCollision> KinematicBody::_move(const Vector3 &p_motion, bool p_infinite_inertia, bool p_exclude_raycast_shapes, bool p_test_only) {

	Collision col;
	if (move_and_collide(p_motion, p_infinite_inertia, col, p_exclude_raycast_shapes, p_test_only)) {
		if (motion_cache.is_null()) {
			motion_cache.instance();
			motion_cache->owner = this;
		}

		motion_cache->collision = col;

		return motion_cache;
	}

	return Ref<KinematicCollision>();
}

bool KinematicBody::move_and_collide(const Vector3 &p_motion, bool p_infinite_inertia, Collision &r_collision, bool p_exclude_raycast_shapes, bool p_test_only) {

	Transform gt = get_global_transform();
	PhysicsServer::MotionResult result;
	bool colliding = PhysicsServer::get_singleton()->body_test_motion(get_rid(), gt, p_motion, p_infinite_inertia, &result, p_exclude_raycast_shapes);

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

	for (int i = 0; i < 3; i++) {
		if (locked_axis & (1 << i)) {
			result.motion[i] = 0;
		}
	}

	if (!p_test_only) {
		gt.origin += result.motion;
		set_global_transform(gt);
	}

	return colliding;
}

//so, if you pass 45 as limit, avoid numerical precision erros when angle is 45.
#define FLOOR_ANGLE_THRESHOLD 0.01

Vector3 KinematicBody::move_and_slide(const Vector3 &p_linear_velocity, const Vector3 &p_floor_direction, bool p_stop_on_slope, int p_max_slides, float p_floor_max_angle, bool p_infinite_inertia) {

	Vector3 lv = p_linear_velocity;

	for (int i = 0; i < 3; i++) {
		if (locked_axis & (1 << i)) {
			lv[i] = 0;
		}
	}

	// Hack in order to work with calling from _process as well as from _physics_process; calling from thread is risky
	Vector3 motion = (floor_velocity + lv) * (Engine::get_singleton()->is_in_physics_frame() ? get_physics_process_delta_time() : get_process_delta_time());

	on_floor = false;
	on_ceiling = false;
	on_wall = false;
	colliders.clear();
	floor_velocity = Vector3();

	Vector3 lv_n = p_linear_velocity.normalized();

	while (p_max_slides) {

		Collision collision;
		bool found_collision = false;

		for (int i = 0; i < 2; ++i) {
			bool collided;
			if (i == 0) { //collide
				collided = move_and_collide(motion, p_infinite_inertia, collision);
				if (!collided) {
					motion = Vector3(); //clear because no collision happened and motion completed
				}
			} else { //separate raycasts (if any)
				collided = separate_raycast_shapes(p_infinite_inertia, collision);
				if (collided) {
					collision.remainder = motion; //keep
					collision.travel = Vector3();
				}
			}

			if (collided) {
				found_collision = true;

				colliders.push_back(collision);
				motion = collision.remainder;

				bool is_on_slope = false;
				if (p_floor_direction == Vector3()) {
					//all is a wall
					on_wall = true;
				} else {
					if (collision.normal.dot(p_floor_direction) >= Math::cos(p_floor_max_angle + FLOOR_ANGLE_THRESHOLD)) { //floor

						on_floor = true;
						on_floor_body = collision.collider_rid;
						floor_velocity = collision.collider_vel;

						if (p_stop_on_slope) {
							if ((lv_n + p_floor_direction).length() < 0.01 && collision.travel.length() < 1) {
								Transform gt = get_global_transform();
								gt.origin -= collision.travel;
								set_global_transform(gt);
								return Vector3();
							}
						}

						is_on_slope = true;

					} else if (collision.normal.dot(-p_floor_direction) >= Math::cos(p_floor_max_angle + FLOOR_ANGLE_THRESHOLD)) { //ceiling
						on_ceiling = true;
					} else {
						on_wall = true;
					}
				}

				if (p_stop_on_slope && is_on_slope) {
					motion = motion.slide(p_floor_direction);
					lv = lv.slide(p_floor_direction);
				} else {
					Vector3 n = collision.normal;
					motion = motion.slide(n);
					lv = lv.slide(n);
				}

				for (int j = 0; j < 3; j++) {
					if (locked_axis & (1 << j)) {
						lv[j] = 0;
					}
				}
			}
		}

		if (!found_collision || motion == Vector3())
			break;

		--p_max_slides;
	}

	return lv;
}

Vector3 KinematicBody::move_and_slide_with_snap(const Vector3 &p_linear_velocity, const Vector3 &p_snap, const Vector3 &p_floor_direction, bool p_stop_on_slope, int p_max_slides, float p_floor_max_angle, bool p_infinite_inertia) {

	bool was_on_floor = on_floor;

	Vector3 ret = move_and_slide(p_linear_velocity, p_floor_direction, p_stop_on_slope, p_max_slides, p_floor_max_angle, p_infinite_inertia);
	if (!was_on_floor || p_snap == Vector3()) {
		return ret;
	}

	Collision col;
	Transform gt = get_global_transform();

	if (move_and_collide(p_snap, p_infinite_inertia, col, false, true)) {

		bool apply = true;
		if (p_floor_direction != Vector3()) {
			if (Math::acos(p_floor_direction.normalized().dot(col.normal)) < p_floor_max_angle) {
				on_floor = true;
				on_floor_body = col.collider_rid;
				floor_velocity = col.collider_vel;
				if (p_stop_on_slope) {
					// move and collide may stray the object a bit because of pre un-stucking,
					// so only ensure that motion happens on floor direction in this case.
					col.travel = p_floor_direction * p_floor_direction.dot(col.travel);
				}
			} else {
				apply = false; //snapped with floor direction, but did not snap to a floor, do not snap.
			}
		}
		if (apply) {
			gt.origin += col.travel;
			set_global_transform(gt);
		}
	}

	return ret;
}

bool KinematicBody::is_on_floor() const {

	return on_floor;
}

bool KinematicBody::is_on_wall() const {

	return on_wall;
}
bool KinematicBody::is_on_ceiling() const {

	return on_ceiling;
}

Vector3 KinematicBody::get_floor_velocity() const {

	return floor_velocity;
}

bool KinematicBody::test_move(const Transform &p_from, const Vector3 &p_motion, bool p_infinite_inertia) {

	ERR_FAIL_COND_V(!is_inside_tree(), false);

	return PhysicsServer::get_singleton()->body_test_motion(get_rid(), p_from, p_motion, p_infinite_inertia);
}

bool KinematicBody::separate_raycast_shapes(bool p_infinite_inertia, Collision &r_collision) {

	PhysicsServer::SeparationResult sep_res[8]; //max 8 rays

	Transform gt = get_global_transform();

	Vector3 recover;
	int hits = PhysicsServer::get_singleton()->body_test_ray_separation(get_rid(), gt, p_infinite_inertia, recover, sep_res, 8, margin);
	int deepest = -1;
	float deepest_depth;
	for (int i = 0; i < hits; i++) {
		if (deepest == -1 || sep_res[i].collision_depth > deepest_depth) {
			deepest = i;
			deepest_depth = sep_res[i].collision_depth;
		}
	}

	gt.origin += recover;
	set_global_transform(gt);

	if (deepest != -1) {
		r_collision.collider = sep_res[deepest].collider_id;
		r_collision.collider_metadata = sep_res[deepest].collider_metadata;
		r_collision.collider_shape = sep_res[deepest].collider_shape;
		r_collision.collider_vel = sep_res[deepest].collider_velocity;
		r_collision.collision = sep_res[deepest].collision_point;
		r_collision.normal = sep_res[deepest].collision_normal;
		r_collision.local_shape = sep_res[deepest].collision_local_shape;
		r_collision.travel = recover;
		r_collision.remainder = Vector3();

		return true;
	} else {
		return false;
	}
}

void KinematicBody::set_axis_lock(PhysicsServer::BodyAxis p_axis, bool p_lock) {
	PhysicsServer::get_singleton()->body_set_axis_lock(get_rid(), p_axis, p_lock);
}

bool KinematicBody::get_axis_lock(PhysicsServer::BodyAxis p_axis) const {
	return PhysicsServer::get_singleton()->body_is_axis_locked(get_rid(), p_axis);
}

void KinematicBody::set_safe_margin(float p_margin) {

	margin = p_margin;
	PhysicsServer::get_singleton()->body_set_kinematic_safe_margin(get_rid(), margin);
}

float KinematicBody::get_safe_margin() const {

	return margin;
}
int KinematicBody::get_slide_count() const {

	return colliders.size();
}

KinematicBody::Collision KinematicBody::get_slide_collision(int p_bounce) const {
	ERR_FAIL_INDEX_V(p_bounce, colliders.size(), Collision());
	return colliders[p_bounce];
}

Ref<KinematicCollision> KinematicBody::_get_slide_collision(int p_bounce) {

	ERR_FAIL_INDEX_V(p_bounce, colliders.size(), Ref<KinematicCollision>());
	if (p_bounce >= slide_colliders.size()) {
		slide_colliders.resize(p_bounce + 1);
	}

	if (slide_colliders[p_bounce].is_null()) {
		slide_colliders.write[p_bounce].instance();
		slide_colliders.write[p_bounce]->owner = this;
	}

	slide_colliders.write[p_bounce]->collision = colliders[p_bounce];
	return slide_colliders[p_bounce];
}

void KinematicBody::_bind_methods() {

	ClassDB::bind_method(D_METHOD("move_and_collide", "rel_vec", "infinite_inertia", "exclude_raycast_shapes", "test_only"), &KinematicBody::_move, DEFVAL(true), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("move_and_slide", "linear_velocity", "floor_normal", "stop_on_slope", "max_slides", "floor_max_angle", "infinite_inertia"), &KinematicBody::move_and_slide, DEFVAL(Vector3(0, 0, 0)), DEFVAL(false), DEFVAL(4), DEFVAL(Math::deg2rad((float)45)), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("move_and_slide_with_snap", "linear_velocity", "snap", "floor_normal", "stop_on_slope", "max_slides", "floor_max_angle", "infinite_inertia"), &KinematicBody::move_and_slide_with_snap, DEFVAL(Vector3(0, 0, 0)), DEFVAL(false), DEFVAL(4), DEFVAL(Math::deg2rad((float)45)), DEFVAL(true));

	ClassDB::bind_method(D_METHOD("test_move", "from", "rel_vec", "infinite_inertia"), &KinematicBody::test_move, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("is_on_floor"), &KinematicBody::is_on_floor);
	ClassDB::bind_method(D_METHOD("is_on_ceiling"), &KinematicBody::is_on_ceiling);
	ClassDB::bind_method(D_METHOD("is_on_wall"), &KinematicBody::is_on_wall);
	ClassDB::bind_method(D_METHOD("get_floor_velocity"), &KinematicBody::get_floor_velocity);

	ClassDB::bind_method(D_METHOD("set_axis_lock", "axis", "lock"), &KinematicBody::set_axis_lock);
	ClassDB::bind_method(D_METHOD("get_axis_lock", "axis"), &KinematicBody::get_axis_lock);

	ClassDB::bind_method(D_METHOD("set_safe_margin", "pixels"), &KinematicBody::set_safe_margin);
	ClassDB::bind_method(D_METHOD("get_safe_margin"), &KinematicBody::get_safe_margin);

	ClassDB::bind_method(D_METHOD("get_slide_count"), &KinematicBody::get_slide_count);
	ClassDB::bind_method(D_METHOD("get_slide_collision", "slide_idx"), &KinematicBody::_get_slide_collision);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "move_lock_x", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_X);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "move_lock_y", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_Y);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "move_lock_z", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_Z);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "collision/safe_margin", PROPERTY_HINT_RANGE, "0.001,256,0.001"), "set_safe_margin", "get_safe_margin");
}

KinematicBody::KinematicBody() :
		PhysicsBody(PhysicsServer::BODY_MODE_KINEMATIC) {

	margin = 0.001;
	locked_axis = 0;
	on_floor = false;
	on_ceiling = false;
	on_wall = false;
}
KinematicBody::~KinematicBody() {

	if (motion_cache.is_valid()) {
		motion_cache->owner = NULL;
	}

	for (int i = 0; i < slide_colliders.size(); i++) {
		if (slide_colliders[i].is_valid()) {
			slide_colliders.write[i]->owner = NULL;
		}
	}
}
///////////////////////////////////////

Vector3 KinematicCollision::get_position() const {

	return collision.collision;
}
Vector3 KinematicCollision::get_normal() const {
	return collision.normal;
}
Vector3 KinematicCollision::get_travel() const {
	return collision.travel;
}
Vector3 KinematicCollision::get_remainder() const {
	return collision.remainder;
}
Object *KinematicCollision::get_local_shape() const {
	if (!owner) return NULL;
	uint32_t ownerid = owner->shape_find_owner(collision.local_shape);
	return owner->shape_owner_get_owner(ownerid);
}

Object *KinematicCollision::get_collider() const {

	if (collision.collider) {
		return ObjectDB::get_instance(collision.collider);
	}

	return NULL;
}
ObjectID KinematicCollision::get_collider_id() const {

	return collision.collider;
}
Object *KinematicCollision::get_collider_shape() const {

	Object *collider = get_collider();
	if (collider) {
		CollisionObject *obj2d = Object::cast_to<CollisionObject>(collider);
		if (obj2d) {
			uint32_t ownerid = obj2d->shape_find_owner(collision.collider_shape);
			return obj2d->shape_owner_get_owner(ownerid);
		}
	}

	return NULL;
}
int KinematicCollision::get_collider_shape_index() const {

	return collision.collider_shape;
}
Vector3 KinematicCollision::get_collider_velocity() const {

	return collision.collider_vel;
}
Variant KinematicCollision::get_collider_metadata() const {

	return Variant();
}

void KinematicCollision::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_position"), &KinematicCollision::get_position);
	ClassDB::bind_method(D_METHOD("get_normal"), &KinematicCollision::get_normal);
	ClassDB::bind_method(D_METHOD("get_travel"), &KinematicCollision::get_travel);
	ClassDB::bind_method(D_METHOD("get_remainder"), &KinematicCollision::get_remainder);
	ClassDB::bind_method(D_METHOD("get_local_shape"), &KinematicCollision::get_local_shape);
	ClassDB::bind_method(D_METHOD("get_collider"), &KinematicCollision::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_id"), &KinematicCollision::get_collider_id);
	ClassDB::bind_method(D_METHOD("get_collider_shape"), &KinematicCollision::get_collider_shape);
	ClassDB::bind_method(D_METHOD("get_collider_shape_index"), &KinematicCollision::get_collider_shape_index);
	ClassDB::bind_method(D_METHOD("get_collider_velocity"), &KinematicCollision::get_collider_velocity);
	ClassDB::bind_method(D_METHOD("get_collider_metadata"), &KinematicCollision::get_collider_metadata);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "position"), "", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "normal"), "", "get_normal");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "travel"), "", "get_travel");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "remainder"), "", "get_remainder");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "local_shape"), "", "get_local_shape");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collider"), "", "get_collider");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_id"), "", "get_collider_id");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collider_shape"), "", "get_collider_shape");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_shape_index"), "", "get_collider_shape_index");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "collider_velocity"), "", "get_collider_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::NIL, "collider_metadata", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT), "", "get_collider_metadata");
}

KinematicCollision::KinematicCollision() {
	collision.collider = 0;
	collision.collider_shape = 0;
	collision.local_shape = 0;
	owner = NULL;
}

///////////////////////////////////////

bool PhysicalBone::JointData::_set(const StringName &p_name, const Variant &p_value) {

	return false;
}

bool PhysicalBone::JointData::_get(const StringName &p_name, Variant &r_ret) const {

	return false;
}

void PhysicalBone::JointData::_get_property_list(List<PropertyInfo> *p_list) const {
}

bool PhysicalBone::FixedJointData::_set(const StringName &p_name, const Variant &p_value) {
	if (JointData::_set(p_name, p_value)) {
		return true;
	}

	return false;
}

bool PhysicalBone::FixedJointData::_get(const StringName &p_name, Variant &r_ret) const {
	if (JointData::_get(p_name, r_ret)) {
		return true;
	}

	return false;
}

void PhysicalBone::FixedJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);
}

bool PhysicalBone::SliderJointData::_set(const StringName &p_name, const Variant &p_value) {
	if (JointData::_set(p_name, p_value)) {
		return true;
	}

	if ("slider_joint/limit_active" == p_name) {
		limit_active = p_value;
	} else if ("slider_joint/lower_limit" == p_name) {
		lower_limit = p_value;
	} else if ("slider_joint/upper_limit" == p_name) {
		upper_limit = p_value;
	} else if ("slider_joint/motor_is_enabled" == p_name) {
		motor_is_enabled = p_value;
	} else if ("slider_joint/motor_velocity_target" == p_name) {
		motor_velocity_target = p_value;
	} else if ("slider_joint/motor_position_target" == p_name) {
		motor_position_target = p_value;
	} else if ("slider_joint/motor_max_impulse" == p_name) {
		motor_max_impulse = p_value;
	} else if ("slider_joint/motor_error_reduction_parameter" == p_name) {
		motor_error_reduction_parameter = p_value;
	} else if ("slider_joint/motor_spring_constant" == p_name) {
		motor_spring_constant = p_value;
	} else if ("slider_joint/motor_damping_constant" == p_name) {
		motor_damping_constant = p_value;
	} else if ("slider_joint/motor_maximum_error" == p_name) {
		motor_maximum_error = p_value;
	} else {
		return false;
	}

	return true;
}

bool PhysicalBone::SliderJointData::_get(const StringName &p_name, Variant &r_ret) const {
	if (JointData::_get(p_name, r_ret)) {
		return true;
	}

	if ("slider_joint/limit_active" == p_name) {
		r_ret = limit_active;
	} else if ("slider_joint/lower_limit" == p_name) {
		r_ret = lower_limit;
	} else if ("slider_joint/upper_limit" == p_name) {
		r_ret = upper_limit;
	} else if ("slider_joint/motor_is_enabled" == p_name) {
		r_ret = motor_is_enabled;
	} else if ("slider_joint/motor_velocity_target" == p_name) {
		r_ret = motor_velocity_target;
	} else if ("slider_joint/motor_position_target" == p_name) {
		r_ret = motor_position_target;
	} else if ("slider_joint/motor_max_impulse" == p_name) {
		r_ret = motor_max_impulse;
	} else if ("slider_joint/motor_error_reduction_parameter" == p_name) {
		r_ret = motor_error_reduction_parameter;
	} else if ("slider_joint/motor_spring_constant" == p_name) {
		r_ret = motor_spring_constant;
	} else if ("slider_joint/motor_damping_constant" == p_name) {
		r_ret = motor_damping_constant;
	} else if ("slider_joint/motor_maximum_error" == p_name) {
		r_ret = motor_maximum_error;
	} else
		return false;

	return true;
}

void PhysicalBone::SliderJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);

	p_list->push_back(PropertyInfo(Variant::BOOL, "slider_joint/limit_active"));
	p_list->push_back(PropertyInfo(Variant::REAL, "slider_joint/lower_limit"));
	p_list->push_back(PropertyInfo(Variant::REAL, "slider_joint/upper_limit"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "slider_joint/motor_is_enabled"));
	p_list->push_back(PropertyInfo(Variant::REAL, "slider_joint/motor_velocity_target"));
	p_list->push_back(PropertyInfo(Variant::REAL, "slider_joint/motor_position_target"));
	p_list->push_back(PropertyInfo(Variant::REAL, "slider_joint/motor_max_impulse"));
	p_list->push_back(PropertyInfo(Variant::REAL, "slider_joint/motor_error_reduction_parameter"));
	p_list->push_back(PropertyInfo(Variant::REAL, "slider_joint/motor_spring_constant"));
	p_list->push_back(PropertyInfo(Variant::REAL, "slider_joint/motor_damping_constant"));
	p_list->push_back(PropertyInfo(Variant::REAL, "slider_joint/motor_maximum_error"));
}

bool PhysicalBone::HingeJointData::_set(const StringName &p_name, const Variant &p_value) {
	if (JointData::_set(p_name, p_value)) {
		return true;
	}

	if ("hinge_joint/limit_active" == p_name) {
		limit_active = p_value;
	} else if ("hinge_joint/lower_limit" == p_name) {
		lower_limit = p_value;
	} else if ("hinge_joint/upper_limit" == p_name) {
		upper_limit = p_value;
	} else if ("hinge_joint/motor_is_enabled" == p_name) {
		motor_is_enabled = p_value;
	} else if ("hinge_joint/motor_velocity_target" == p_name) {
		motor_velocity_target = p_value;
	} else if ("hinge_joint/motor_position_target" == p_name) {
		motor_position_target = p_value;
	} else if ("hinge_joint/motor_max_impulse" == p_name) {
		motor_max_impulse = p_value;
	} else if ("hinge_joint/motor_error_reduction_parameter" == p_name) {
		motor_error_reduction_parameter = p_value;
	} else if ("hinge_joint/motor_spring_constant" == p_name) {
		motor_spring_constant = p_value;
	} else if ("hinge_joint/motor_damping_constant" == p_name) {
		motor_damping_constant = p_value;
	} else if ("hinge_joint/motor_maximum_error" == p_name) {
		motor_maximum_error = p_value;
	} else
		return false;

	return true;
}

bool PhysicalBone::HingeJointData::_get(const StringName &p_name, Variant &r_ret) const {
	if (JointData::_get(p_name, r_ret)) {
		return true;
	}

	if ("hinge_joint/limit_active" == p_name) {
		r_ret = limit_active;
	} else if ("hinge_joint/lower_limit" == p_name) {
		r_ret = lower_limit;
	} else if ("hinge_joint/upper_limit" == p_name) {
		r_ret = upper_limit;
	} else if ("hinge_joint/motor_is_enabled" == p_name) {
		r_ret = motor_is_enabled;
	} else if ("hinge_joint/motor_velocity_target" == p_name) {
		r_ret = motor_velocity_target;
	} else if ("hinge_joint/motor_position_target" == p_name) {
		r_ret = motor_position_target;
	} else if ("hinge_joint/motor_max_impulse" == p_name) {
		r_ret = motor_max_impulse;
	} else if ("hinge_joint/motor_error_reduction_parameter" == p_name) {
		r_ret = motor_error_reduction_parameter;
	} else if ("hinge_joint/motor_spring_constant" == p_name) {
		r_ret = motor_spring_constant;
	} else if ("hinge_joint/motor_damping_constant" == p_name) {
		r_ret = motor_damping_constant;
	} else if ("hinge_joint/motor_maximum_error" == p_name) {
		r_ret = motor_maximum_error;
	} else
		return false;

	return true;
}

void PhysicalBone::HingeJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);

	p_list->push_back(PropertyInfo(Variant::BOOL, "hinge_joint/limit_active"));
	p_list->push_back(PropertyInfo(Variant::REAL, "hinge_joint/lower_limit"));
	p_list->push_back(PropertyInfo(Variant::REAL, "hinge_joint/upper_limit"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "hinge_joint/motor_is_enabled"));
	p_list->push_back(PropertyInfo(Variant::REAL, "hinge_joint/motor_velocity_target"));
	p_list->push_back(PropertyInfo(Variant::REAL, "hinge_joint/motor_position_target"));
	p_list->push_back(PropertyInfo(Variant::REAL, "hinge_joint/motor_max_impulse"));
	p_list->push_back(PropertyInfo(Variant::REAL, "hinge_joint/motor_error_reduction_parameter"));
	p_list->push_back(PropertyInfo(Variant::REAL, "hinge_joint/motor_spring_constant"));
	p_list->push_back(PropertyInfo(Variant::REAL, "hinge_joint/motor_damping_constant"));
	p_list->push_back(PropertyInfo(Variant::REAL, "hinge_joint/motor_maximum_error"));
}

bool PhysicalBone::SphericalJointData::_set(const StringName &p_name, const Variant &p_value) {
	if (JointData::_set(p_name, p_value)) {
		return true;
	}

	if ("spherical_joint/motor_is_enabled" == p_name) {
		motor_is_enabled = p_value;
	} else if ("spherical_joint/motor_velocity_target" == p_name) {
		motor_velocity_target = p_value;
	} else if ("spherical_joint/motor_rotation_target" == p_name) {
		motor_rotation_target = p_value;
	} else if ("spherical_joint/motor_max_impulse" == p_name) {
		motor_max_impulse = p_value;
	} else if ("spherical_joint/motor_error_reduction_parameter" == p_name) {
		motor_error_reduction_parameter = p_value;
	} else if ("spherical_joint/motor_spring_constant" == p_name) {
		motor_spring_constant = p_value;
	} else if ("spherical_joint/motor_damping_constant" == p_name) {
		motor_damping_constant = p_value;
	} else if ("spherical_joint/motor_maximum_error" == p_name) {
		motor_maximum_error = p_value;
	} else {
		return false;
	}

	return true;
}

bool PhysicalBone::SphericalJointData::_get(const StringName &p_name, Variant &r_ret) const {
	if (JointData::_get(p_name, r_ret)) {
		return true;
	}

	if ("spherical_joint/motor_is_enabled" == p_name) {
		r_ret = motor_is_enabled;
	} else if ("spherical_joint/motor_velocity_target" == p_name) {
		r_ret = motor_velocity_target;
	} else if ("spherical_joint/motor_rotation_target" == p_name) {
		r_ret = motor_rotation_target;
	} else if ("spherical_joint/motor_max_impulse" == p_name) {
		r_ret = motor_max_impulse;
	} else if ("spherical_joint/motor_error_reduction_parameter" == p_name) {
		r_ret = motor_error_reduction_parameter;
	} else if ("spherical_joint/motor_spring_constant" == p_name) {
		r_ret = motor_spring_constant;
	} else if ("spherical_joint/motor_damping_constant" == p_name) {
		r_ret = motor_damping_constant;
	} else if ("spherical_joint/motor_maximum_error" == p_name) {
		r_ret = motor_maximum_error;
	} else
		return false;

	return true;
}

void PhysicalBone::SphericalJointData::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back(PropertyInfo(Variant::BOOL, "spherical_joint/motor_is_enabled"));
	p_list->push_back(PropertyInfo(Variant::VECTOR3, "spherical_joint/motor_velocity_target"));
	p_list->push_back(PropertyInfo(Variant::VECTOR3, "spherical_joint/motor_rotation_target"));
	p_list->push_back(PropertyInfo(Variant::REAL, "spherical_joint/motor_max_impulse"));
	p_list->push_back(PropertyInfo(Variant::REAL, "spherical_joint/motor_error_reduction_parameter"));
	p_list->push_back(PropertyInfo(Variant::REAL, "spherical_joint/motor_spring_constant"));
	p_list->push_back(PropertyInfo(Variant::REAL, "spherical_joint/motor_damping_constant"));
	p_list->push_back(PropertyInfo(Variant::REAL, "spherical_joint/motor_maximum_error"));
}

bool PhysicalBone::PlanarJointData::_set(const StringName &p_name, const Variant &p_value) {
	if (JointData::_set(p_name, p_value)) {
		return true;
	}

	return false;
}

bool PhysicalBone::PlanarJointData::_get(const StringName &p_name, Variant &r_ret) const {
	if (JointData::_get(p_name, r_ret)) {
		return true;
	}

	return false;
}

void PhysicalBone::PlanarJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);
}

bool PhysicalBone::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "bone_name") {
		set_bone_name(p_value);
		return true;
	}

	if (joint_data) {
		if (joint_data->_set(p_name, p_value)) {
#ifdef TOOLS_ENABLED
			if (get_gizmo().is_valid())
				get_gizmo()->redraw();
#endif
			return true;
		}
	}

	return false;
}

bool PhysicalBone::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "bone_name") {
		r_ret = get_bone_name();
		return true;
	}

	if (joint_data) {
		return joint_data->_get(p_name, r_ret);
	}

	return false;
}

void PhysicalBone::_get_property_list(List<PropertyInfo> *p_list) const {

	Skeleton *parent = find_skeleton_parent(get_parent());

	if (parent) {

		String names("--,");
		for (int i = 0; i < parent->get_bone_count(); i++) {
			if (i > 0)
				names += ",";
			names += parent->get_bone_name(i);
		}

		p_list->push_back(PropertyInfo(Variant::STRING, "bone_name", PROPERTY_HINT_ENUM, names));
	} else {

		p_list->push_back(PropertyInfo(Variant::STRING, "bone_name"));
	}

	if (joint_data) {
		joint_data->_get_property_list(p_list);
	}
}

void PhysicalBone::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:

			parent_skeleton = find_skeleton_parent(get_parent());

			update_bone_id();
			reset_to_rest_position();
			reset_physics_simulation_state();

			break;
		case NOTIFICATION_EXIT_TREE:

			if (parent_skeleton) {
				parent_skeleton->unbind_physical_bone_from_bone(bone_id);
				parent_skeleton = NULL;
				update_bone_id();
			}

			break;
		case NOTIFICATION_TRANSFORM_CHANGED:

			if (Engine::get_singleton()->is_editor_hint()) {
				update_offset();
			}

			break;
	}
}

void PhysicalBone::_direct_state_changed(Object *p_state) {

	if (!_internal_simulate_physics) {
		return;
	}

	/// Update bone transform

	Transform global_transform(PhysicsServer::get_singleton()->bone_get_transform(get_rid()));

	set_ignore_transform_notification(true);
	set_global_transform(global_transform);
	set_ignore_transform_notification(false);

	// Update skeleton
	if (parent_skeleton) {
		if (-1 != bone_id) {

			parent_skeleton->set_bone_global_pose(
					bone_id,
					parent_skeleton->get_global_transform().affine_inverse() * (global_transform * body_offset_inverse));
		}
	}

#ifdef DEBUG_ENABLED
	PhysicsDirectBodyState *state = Object::cast_to<PhysicsDirectBodyState>(p_state);
#else
	PhysicsDirectBodyState *state = (PhysicsDirectBodyState *)p_state; //trust it
#endif

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

		_RigidBodyInOut *toadd = (_RigidBodyInOut *)alloca(state->get_contact_count() * sizeof(_RigidBodyInOut));
		int toadd_count = 0; //state->get_contact_count();
		RigidBody_RemoveAction *toremove = (RigidBody_RemoveAction *)alloca(rc * sizeof(RigidBody_RemoveAction));
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
}

void PhysicalBone::_body_enter_tree(ObjectID p_id) {

	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = Object::cast_to<Node>(obj);
	ERR_FAIL_COND(!node);

	ERR_FAIL_COND(!contact_monitor);
	Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(E->get().in_tree);

	E->get().in_tree = true;

	contact_monitor->locked = true;

	emit_signal(SceneStringNames::get_singleton()->body_entered, node);

	for (int i = 0; i < E->get().shapes.size(); i++) {

		emit_signal(SceneStringNames::get_singleton()->body_shape_entered, p_id, node, E->get().shapes[i].body_shape, E->get().shapes[i].local_shape);
	}

	contact_monitor->locked = false;
}

void PhysicalBone::_body_exit_tree(ObjectID p_id) {

	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = Object::cast_to<Node>(obj);
	ERR_FAIL_COND(!node);
	ERR_FAIL_COND(!contact_monitor);
	Map<ObjectID, BodyState>::Element *E = contact_monitor->body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(!E->get().in_tree);
	E->get().in_tree = false;

	contact_monitor->locked = true;

	emit_signal(SceneStringNames::get_singleton()->body_exited, node);

	for (int i = 0; i < E->get().shapes.size(); i++) {

		emit_signal(SceneStringNames::get_singleton()->body_shape_exited, p_id, node, E->get().shapes[i].body_shape, E->get().shapes[i].local_shape);
	}

	contact_monitor->locked = false;
}

void PhysicalBone::_body_inout(int p_status, ObjectID p_instance, int p_body_shape, int p_local_shape) {

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
			//E->get().rc=0;
			E->get().in_tree = node && node->is_inside_tree();
			if (node) {
				node->connect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree, make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_body_exit_tree, make_binds(objid));
				if (E->get().in_tree) {
					emit_signal(SceneStringNames::get_singleton()->body_entered, node);
				}
			}
		}
		//E->get().rc++;
		if (node)
			E->get().shapes.insert(ShapePair(p_body_shape, p_local_shape));

		if (E->get().in_tree) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_entered, objid, node, p_body_shape, p_local_shape);
		}

	} else {

		//E->get().rc--;

		if (node)
			E->get().shapes.erase(ShapePair(p_body_shape, p_local_shape));

		bool in_tree = E->get().in_tree;

		if (E->get().shapes.empty()) {

			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree);
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_body_exit_tree);
				if (in_tree)
					emit_signal(SceneStringNames::get_singleton()->body_exited, node);
			}

			contact_monitor->body_map.erase(E);
		}
		if (node && in_tree) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_exited, objid, obj, p_body_shape, p_local_shape);
		}
	}
}

void PhysicalBone::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_direct_state_changed"), &PhysicalBone::_direct_state_changed);

	ClassDB::bind_method(D_METHOD("_body_enter_tree"), &PhysicalBone::_body_enter_tree);
	ClassDB::bind_method(D_METHOD("_body_exit_tree"), &PhysicalBone::_body_exit_tree);

	ClassDB::bind_method(D_METHOD("set_disable_parent_collision", "disable"), &PhysicalBone::set_disable_parent_collision);
	ClassDB::bind_method(D_METHOD("get_disable_parent_collision"), &PhysicalBone::get_disable_parent_collision);

	ClassDB::bind_method(D_METHOD("set_joint_type", "joint_type"), &PhysicalBone::set_joint_type);
	ClassDB::bind_method(D_METHOD("get_joint_type"), &PhysicalBone::get_joint_type);

	ClassDB::bind_method(D_METHOD("set_joint_offset", "offset"), &PhysicalBone::set_joint_offset);
	ClassDB::bind_method(D_METHOD("get_joint_offset"), &PhysicalBone::get_joint_offset);

	ClassDB::bind_method(D_METHOD("set_body_offset", "offset"), &PhysicalBone::set_body_offset);
	ClassDB::bind_method(D_METHOD("get_body_offset"), &PhysicalBone::get_body_offset);

	ClassDB::bind_method(D_METHOD("is_simulating_physics"), &PhysicalBone::is_simulating_physics);

	ClassDB::bind_method(D_METHOD("set_bone_name", "bone_id"), &PhysicalBone::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name"), &PhysicalBone::get_bone_name);

	ClassDB::bind_method(D_METHOD("set_bone_id", "bone_id"), &PhysicalBone::set_bone_id);
	ClassDB::bind_method(D_METHOD("get_bone_id"), &PhysicalBone::get_bone_id);

	ClassDB::bind_method(D_METHOD("set_link_mass", "mass"), &PhysicalBone::set_link_mass);
	ClassDB::bind_method(D_METHOD("get_link_mass"), &PhysicalBone::get_link_mass);

	ClassDB::bind_method(D_METHOD("set_weight", "weight"), &PhysicalBone::set_weight);
	ClassDB::bind_method(D_METHOD("get_weight"), &PhysicalBone::get_weight);

	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &PhysicalBone::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &PhysicalBone::get_friction);

	ClassDB::bind_method(D_METHOD("set_bounce", "bounce"), &PhysicalBone::set_bounce);
	ClassDB::bind_method(D_METHOD("get_bounce"), &PhysicalBone::get_bounce);

	ClassDB::bind_method(D_METHOD("set_contact_monitor", "enabled"), &PhysicalBone::set_contact_monitor);
	ClassDB::bind_method(D_METHOD("is_contact_monitor_enabled"), &PhysicalBone::is_contact_monitor_enabled);

	ClassDB::bind_method(D_METHOD("set_max_contacts_reported", "amount"), &PhysicalBone::set_max_contacts_reported);
	ClassDB::bind_method(D_METHOD("get_max_contacts_reported"), &PhysicalBone::get_max_contacts_reported);

	ClassDB::bind_method(D_METHOD("motor_set_active", "active"), &PhysicalBone::motor_set_active);
	ClassDB::bind_method(D_METHOD("motor_set_position_target", "position"), &PhysicalBone::motor_set_position_target);
	ClassDB::bind_method(D_METHOD("motor_set_rotation_target", "rotation"), &PhysicalBone::motor_set_rotation_target);
	ClassDB::bind_method(D_METHOD("motor_set_rotation_target_basis", "rotation"), &PhysicalBone::motor_set_rotation_target_basis);
	ClassDB::bind_method(D_METHOD("motor_set_velocity", "velocity"), &PhysicalBone::motor_set_velocity);
	ClassDB::bind_method(D_METHOD("motor_set_max_impulse", "max_impulse"), &PhysicalBone::motor_set_max_impulse);
	ClassDB::bind_method(D_METHOD("motor_set_error_reduction_parameter", "error_reduction_parameter"), &PhysicalBone::motor_set_error_reduction_parameter);
	ClassDB::bind_method(D_METHOD("motor_set_spring_constant", "spring_constant"), &PhysicalBone::motor_set_spring_constant);
	ClassDB::bind_method(D_METHOD("motor_set_damping_constant", "damping_constant"), &PhysicalBone::motor_set_damping_constant);
	ClassDB::bind_method(D_METHOD("motor_set_maximum_error", "maximum_error"), &PhysicalBone::motor_set_maximum_error);

	ClassDB::bind_method(D_METHOD("get_joint_force"), &PhysicalBone::get_joint_force);
	ClassDB::bind_method(D_METHOD("get_joint_torque"), &PhysicalBone::get_joint_torque);

	ClassDB::bind_method(D_METHOD("get_colliding_bodies"), &PhysicalBone::get_colliding_bodies);

	ADD_GROUP("Joint", "joint_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "joint_type", PROPERTY_HINT_ENUM, "None,Fixed,Slider,Hinge,Spherical,Planar"), "set_joint_type", "get_joint_type");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "joint_offset"), "set_joint_offset", "get_joint_offset");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disable_parent_collision"), "set_disable_parent_collision", "get_disable_parent_collision");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "body_offset"), "set_body_offset", "get_body_offset");

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "link_mass", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01"), "set_link_mass", "get_link_mass");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "weight", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01"), "set_weight", "get_weight");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "friction", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_bounce", "get_bounce");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "contact_monitor"), "set_contact_monitor", "is_contact_monitor_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "contacts_reported"), "set_max_contacts_reported", "get_max_contacts_reported");

	ADD_SIGNAL(MethodInfo("body_shape_entered", PropertyInfo(Variant::INT, "body_id"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape"), PropertyInfo(Variant::INT, "local_shape")));
	ADD_SIGNAL(MethodInfo("body_shape_exited", PropertyInfo(Variant::INT, "body_id"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape"), PropertyInfo(Variant::INT, "local_shape")));
	ADD_SIGNAL(MethodInfo("body_entered", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("body_exited", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));

	BIND_ENUM_CONSTANT(JOINT_TYPE_NONE);
	BIND_ENUM_CONSTANT(JOINT_TYPE_FIXED);
	BIND_ENUM_CONSTANT(JOINT_TYPE_SLIDER);
	BIND_ENUM_CONSTANT(JOINT_TYPE_HINGE);
	BIND_ENUM_CONSTANT(JOINT_TYPE_SPHERICAL);
	BIND_ENUM_CONSTANT(JOINT_TYPE_PLANAR);
}

Skeleton *PhysicalBone::find_skeleton_parent(Node *p_parent) {
	if (!p_parent) {
		return NULL;
	}
	Skeleton *s = Object::cast_to<Skeleton>(p_parent);
	return s ? s : find_skeleton_parent(p_parent->get_parent());
}

void PhysicalBone::_reset_joint_offset_origin() {
	// Clamp joint origin to bone origin
	if (parent_skeleton) {
		joint_offset.origin = body_offset_inverse.origin;
	}
}

void PhysicalBone::_setup_physical_bone() {

	if (!parent_skeleton)
		return;

	if (parent_skeleton->get_skip_physics_tree_rebuild())
		return;

	/// Step 1 Reset bone position

	PhysicsServer::get_singleton()->bone_set_id(
			get_rid(),
			physical_bone_id);

	if (parent_physical_bone_id >= 0)
		PhysicsServer::get_singleton()->bone_set_parent_id(
				get_rid(),
				parent_physical_bone_id);

	reset_to_rest_position();

	PhysicsServer::get_singleton()->bone_set_joint_transform(
			get_rid(),
			get_joint_offset());
}

void PhysicalBone::_reload_joint() {

	if (!parent_skeleton)
		return;

	if (parent_skeleton->get_skip_physics_tree_rebuild())
		return;

	/// Step 2 Set joint

	switch (get_joint_type()) {
		case JOINT_TYPE_NONE:
		case JOINT_TYPE_FIXED: {
			PhysicsServer::get_singleton()->bone_joint_fixed_setup(
					get_rid());

		} break;
		case JOINT_TYPE_SLIDER: {
			PhysicsServer::get_singleton()->bone_joint_slider_setup(
					get_rid());
		} break;
		case JOINT_TYPE_HINGE: {
			PhysicsServer::get_singleton()->bone_joint_hinge_setup(
					get_rid());
		} break;
		case JOINT_TYPE_SPHERICAL: {
			PhysicsServer::get_singleton()->bone_joint_spherical_setup(
					get_rid());
		} break;
		case JOINT_TYPE_PLANAR: {
			PhysicsServer::get_singleton()->bone_joint_planar_setup(
					get_rid());
		} break;
	}

	/// Step 3 Set joint limits and motors

	switch (get_joint_type()) {
		case JOINT_TYPE_NONE:
			return; // Stop here
		case JOINT_TYPE_FIXED: {
		} break;
		case JOINT_TYPE_SLIDER: {

			const PhysicalBone::HingeJointData *j =
					static_cast<const PhysicalBone::HingeJointData *>(
							get_joint_data());

			PhysicsServer::get_singleton()->bone_set_joint_limit_active(
					get_rid(),
					j->limit_active);

			if (j->limit_active) {
				PhysicsServer::get_singleton()->bone_set_joint_lower_limit(
						get_rid(),
						j->lower_limit);

				PhysicsServer::get_singleton()->bone_set_joint_upper_limit(
						get_rid(),
						j->upper_limit);
			}

			PhysicsServer::get_singleton()->bone_set_motor_enabled(get_rid(), j->motor_is_enabled);

			if (j->motor_is_enabled) {
				PhysicsServer::get_singleton()->bone_set_velocity_target(get_rid(), Vector3(j->motor_velocity_target, 0, 0));
				PhysicsServer::get_singleton()->bone_set_position_target(get_rid(), j->motor_position_target);
				PhysicsServer::get_singleton()->bone_set_max_motor_impulse(get_rid(), j->motor_max_impulse);
				PhysicsServer::get_singleton()->bone_set_error_reduction_parameter(get_rid(), j->motor_error_reduction_parameter);
				PhysicsServer::get_singleton()->bone_set_spring_constant(get_rid(), j->motor_spring_constant);
				PhysicsServer::get_singleton()->bone_set_damping_constant(get_rid(), j->motor_damping_constant);
				PhysicsServer::get_singleton()->bone_set_maximum_error(get_rid(), j->motor_maximum_error);
			}

		} break;
		case JOINT_TYPE_HINGE: {

			const PhysicalBone::SliderJointData *j =
					static_cast<const PhysicalBone::SliderJointData *>(
							get_joint_data());

			PhysicsServer::get_singleton()->bone_set_joint_limit_active(
					get_rid(),
					j->limit_active);

			if (j->limit_active) {
				PhysicsServer::get_singleton()->bone_set_joint_lower_limit(
						get_rid(),
						j->lower_limit);

				PhysicsServer::get_singleton()->bone_set_joint_upper_limit(
						get_rid(),
						j->upper_limit);
			}

			PhysicsServer::get_singleton()->bone_set_motor_enabled(get_rid(), j->motor_is_enabled);

			if (j->motor_is_enabled) {
				PhysicsServer::get_singleton()->bone_set_velocity_target(get_rid(), Vector3(j->motor_velocity_target, 0, 0));
				PhysicsServer::get_singleton()->bone_set_position_target(get_rid(), j->motor_position_target);
				PhysicsServer::get_singleton()->bone_set_max_motor_impulse(get_rid(), j->motor_max_impulse);
				PhysicsServer::get_singleton()->bone_set_error_reduction_parameter(get_rid(), j->motor_error_reduction_parameter);
				PhysicsServer::get_singleton()->bone_set_spring_constant(get_rid(), j->motor_spring_constant);
				PhysicsServer::get_singleton()->bone_set_damping_constant(get_rid(), j->motor_damping_constant);
				PhysicsServer::get_singleton()->bone_set_maximum_error(get_rid(), j->motor_maximum_error);
			}

		} break;
		case JOINT_TYPE_SPHERICAL: {

			const PhysicalBone::SphericalJointData *j =
					static_cast<const PhysicalBone::SphericalJointData *>(
							get_joint_data());

			PhysicsServer::get_singleton()->bone_set_motor_enabled(get_rid(), j->motor_is_enabled);

			if (j->motor_is_enabled) {
				PhysicsServer::get_singleton()->bone_set_velocity_target(get_rid(), j->motor_velocity_target);
				PhysicsServer::get_singleton()->bone_set_rotation_target(get_rid(), Basis(j->motor_rotation_target * Math_PI / 180.));
				PhysicsServer::get_singleton()->bone_set_max_motor_impulse(get_rid(), j->motor_max_impulse);
				PhysicsServer::get_singleton()->bone_set_error_reduction_parameter(get_rid(), j->motor_error_reduction_parameter);
				PhysicsServer::get_singleton()->bone_set_spring_constant(get_rid(), j->motor_spring_constant);
				PhysicsServer::get_singleton()->bone_set_damping_constant(get_rid(), j->motor_damping_constant);
				PhysicsServer::get_singleton()->bone_set_maximum_error(get_rid(), j->motor_maximum_error);
			}

		} break;
		case JOINT_TYPE_PLANAR: {
		} break;
	}
}

void PhysicalBone::_set_gizmo_move_joint(bool p_move_joint) {
#ifdef TOOLS_ENABLED
	gizmo_move_joint = p_move_joint;
	SpatialEditor::get_singleton()->update_transform_gizmo();
#endif
}

#ifdef TOOLS_ENABLED
Transform PhysicalBone::get_global_gizmo_transform() const {
	return gizmo_move_joint ? get_global_transform() * joint_offset : get_global_transform();
}

Transform PhysicalBone::get_local_gizmo_transform() const {
	return gizmo_move_joint ? get_transform() * joint_offset : get_transform();
}
#endif

const PhysicalBone::JointData *PhysicalBone::get_joint_data() const {
	return joint_data;
}

Skeleton *PhysicalBone::find_skeleton_parent() {
	return find_skeleton_parent(this);
}

void PhysicalBone::set_disable_parent_collision(bool p_col) {
	disable_parent_collision = p_col;

	PhysicsServer::get_singleton()->bone_set_disable_parent_collision(get_rid(), disable_parent_collision);
}

bool PhysicalBone::get_disable_parent_collision() const {
	return disable_parent_collision;
}

void PhysicalBone::set_joint_type(JointType p_joint_type) {

	if (p_joint_type == get_joint_type())
		return;

	if (joint_data)
		memdelete(joint_data);
	joint_data = NULL;

	switch (p_joint_type) {
		case JOINT_TYPE_FIXED:
			joint_data = memnew(FixedJointData);
			break;
		case JOINT_TYPE_SLIDER:
			joint_data = memnew(SliderJointData);
			break;
		case JOINT_TYPE_HINGE:
			joint_data = memnew(HingeJointData);
			break;
		case JOINT_TYPE_SPHERICAL:
			joint_data = memnew(SphericalJointData);
			break;
		case JOINT_TYPE_PLANAR:
			joint_data = memnew(PlanarJointData);
			break;
		case JOINT_TYPE_NONE:
			break;
	}

	_setup_physical_bone();
	_reload_joint();

#ifdef TOOLS_ENABLED
	_change_notify();
	if (get_gizmo().is_valid())
		get_gizmo()->redraw();
#endif
}

PhysicalBone::JointType PhysicalBone::get_joint_type() const {
	return joint_data ? joint_data->get_joint_type() : JOINT_TYPE_NONE;
}

void PhysicalBone::set_joint_offset(const Transform &p_offset) {
	joint_offset = p_offset;

	_reset_joint_offset_origin();

	reset_to_rest_position();

#ifdef TOOLS_ENABLED
	if (get_gizmo().is_valid())
		get_gizmo()->redraw();
#endif
}

const Transform &PhysicalBone::get_body_offset() const {
	return body_offset;
}

void PhysicalBone::set_body_offset(const Transform &p_offset) {
	body_offset = p_offset;

	if (-1 != bone_id)
		body_offset.basis = parent_skeleton->get_bone_global_pose(bone_id).basis;

	body_offset_inverse = body_offset.affine_inverse();

	_reset_joint_offset_origin();
	reset_to_rest_position();

#ifdef TOOLS_ENABLED
	if (get_gizmo().is_valid())
		get_gizmo()->redraw();
#endif
}

const Transform &PhysicalBone::get_joint_offset() const {
	return joint_offset;
}

void PhysicalBone::reset_physics_simulation_state() {
	if (parent_skeleton && parent_skeleton->get_active_physics()) {
		_start_physics_simulation();
	} else {
		_stop_physics_simulation();
	}
}

bool PhysicalBone::is_simulating_physics() {
	return _internal_simulate_physics;
}

void PhysicalBone::set_bone_id(int p_bone_id) {
	set_bone_name(parent_skeleton->get_bone_name(p_bone_id));
}

void PhysicalBone::set_bone_name(const String &p_name) {

	bone_name = p_name;
	bone_id = -1;

	update_bone_id();
	reset_to_rest_position();
}

const String &PhysicalBone::get_bone_name() const {

	return bone_name;
}

void PhysicalBone::set_link_mass(real_t p_mass) {

	ERR_FAIL_COND(p_mass <= 0);
	link_mass = p_mass;

	_update_link_mass();
}

real_t PhysicalBone::get_link_mass() const {

	return link_mass;
}

void PhysicalBone::set_weight(real_t p_weight) {

	set_link_mass(p_weight / real_t(GLOBAL_DEF("physics/3d/default_gravity", 9.8)));
}

real_t PhysicalBone::get_weight() const {

	return link_mass * real_t(GLOBAL_DEF("physics/3d/default_gravity", 9.8));
}

void PhysicalBone::set_friction(real_t p_friction) {

	ERR_FAIL_COND(p_friction < 0 || p_friction > 1);

	friction = p_friction;
	PhysicsServer::get_singleton()->bone_set_param(get_rid(), PhysicsServer::BODY_PARAM_FRICTION, friction);
}

real_t PhysicalBone::get_friction() const {

	return friction;
}

void PhysicalBone::set_bounce(real_t p_bounce) {

	ERR_FAIL_COND(p_bounce < 0 || p_bounce > 1);

	bounce = p_bounce;
	PhysicsServer::get_singleton()->bone_set_param(get_rid(), PhysicsServer::BODY_PARAM_BOUNCE, bounce);
}

real_t PhysicalBone::get_bounce() const {

	return bounce;
}

void PhysicalBone::set_contact_monitor(bool p_enabled) {

	if (p_enabled == is_contact_monitor_enabled())
		return;

	if (!p_enabled) {

		if (contact_monitor->locked) {
			ERR_EXPLAIN("Can't disable contact monitoring during in/out callback. Use call_deferred(\"set_contact_monitor\",false) instead");
		}
		ERR_FAIL_COND(contact_monitor->locked);

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
		contact_monitor = NULL;
	} else {

		contact_monitor = memnew(ContactMonitor);
		contact_monitor->locked = false;
	}
}

bool PhysicalBone::is_contact_monitor_enabled() const {

	return contact_monitor != NULL;
}

void PhysicalBone::set_max_contacts_reported(int p_amount) {

	max_contacts_reported = p_amount;
	PhysicsServer::get_singleton()->bone_set_max_contacts_reported(get_rid(), p_amount);
}

int PhysicalBone::get_max_contacts_reported() const {

	return max_contacts_reported;
}

Array PhysicalBone::get_colliding_bodies() const {

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

void PhysicalBone::motor_set_active(bool p_active) {

	PhysicsServer::get_singleton()->bone_set_motor_enabled(get_rid(), p_active);
}

void PhysicalBone::motor_set_position_target(real_t p_position) {
	PhysicsServer::get_singleton()->bone_set_position_target(get_rid(), p_position);
}

void PhysicalBone::motor_set_rotation_target(Vector3 p_rotation) {
	PhysicsServer::get_singleton()->bone_set_rotation_target(get_rid(), Basis(p_rotation * Math_PI / 180.));
}

void PhysicalBone::motor_set_rotation_target_basis(Basis p_rotation) {
	PhysicsServer::get_singleton()->bone_set_rotation_target(get_rid(), p_rotation);
}

void PhysicalBone::motor_set_velocity(Vector3 p_velocity) {
	PhysicsServer::get_singleton()->bone_set_velocity_target(get_rid(), p_velocity);
}

void PhysicalBone::motor_set_max_impulse(real_t p_impulse) {
	PhysicsServer::get_singleton()->bone_set_max_motor_impulse(get_rid(), p_impulse);
}

void PhysicalBone::motor_set_error_reduction_parameter(real_t p_erp) {
	PhysicsServer::get_singleton()->bone_set_error_reduction_parameter(get_rid(), p_erp);
}

void PhysicalBone::motor_set_spring_constant(real_t p_sk) {
	PhysicsServer::get_singleton()->bone_set_spring_constant(get_rid(), p_sk);
}

void PhysicalBone::motor_set_damping_constant(real_t p_dk) {
	PhysicsServer::get_singleton()->bone_set_damping_constant(get_rid(), p_dk);
}

void PhysicalBone::motor_set_maximum_error(real_t p_me) {
	PhysicsServer::get_singleton()->bone_set_maximum_error(get_rid(), p_me);
}

Vector3 PhysicalBone::get_joint_force() {

	return PhysicsServer::get_singleton()->bone_joint_get_force(get_rid());
}

Vector3 PhysicalBone::get_joint_torque() {

	return PhysicsServer::get_singleton()->bone_joint_get_torque(get_rid());
}

PhysicalBone::PhysicalBone() :
		CollisionObject(
				PhysicsServer::get_singleton()->bone_create(),
				COLLISION_OBJECT_TYPE_BONE),
#ifdef TOOLS_ENABLED
		gizmo_move_joint(false),
#endif
		contact_monitor(NULL),
		max_contacts_reported(0),
		joint_data(NULL),
		disable_parent_collision(true),
		parent_skeleton(NULL),
		_internal_simulate_physics(false),
		bone_id(-1),
		physical_bone_id(-1),
		parent_physical_bone_id(-1),
		bone_name(""),
		bounce(0),
		link_mass(1),
		friction(1),
		gravity_scale(1) {

	_update_link_mass();
}

PhysicalBone::~PhysicalBone() {
	if (joint_data) {
		memdelete(joint_data);
		joint_data = NULL;
	}

	if (contact_monitor)
		memdelete(contact_monitor);
}

void PhysicalBone::update_bone_id() {
	if (!parent_skeleton) {

		bone_id = -1;
		physical_bone_id = -1;
		parent_physical_bone_id = -1;

		return;
	}

	const int new_bone_id = parent_skeleton->find_bone(bone_name);

	if (new_bone_id != bone_id) {
		if (-1 != bone_id) {
			// Assert the unbind from old node
			parent_skeleton->unbind_physical_bone_from_bone(bone_id);
			parent_skeleton->unbind_child_node_from_bone(bone_id, this);
		}

		bone_id = new_bone_id;

		// These are managed by the skeleton in the "_rebuild_physical_tree"
		physical_bone_id = -1;
		parent_physical_bone_id = -1;

		_reset_joint_offset_origin();

		parent_skeleton->bind_physical_bone_to_bone(bone_id, this);
	}
}

void PhysicalBone::update_offset() {
#ifdef TOOLS_ENABLED

	if (parent_skeleton) {

		Transform bone_transform(parent_skeleton->get_global_transform());
		if (-1 != bone_id)
			bone_transform *= parent_skeleton->get_bone_global_pose(bone_id);

		if (gizmo_move_joint) {
			bone_transform *= body_offset;
			set_joint_offset(bone_transform.affine_inverse() * get_global_transform());
		} else {
			set_body_offset(bone_transform.affine_inverse() * get_global_transform());
		}
	}
#endif
}

void PhysicalBone::reset_to_rest_position() {
	if (parent_skeleton) {
		set_ignore_transform_notification(true);
		if (-1 == bone_id) {
			set_global_transform(parent_skeleton->get_global_transform() * body_offset);
		} else {
			set_global_transform(
					parent_skeleton->get_global_transform() *
					parent_skeleton->get_bone_global_pose(bone_id) *
					body_offset);
		}
		set_ignore_transform_notification(false);

		PhysicsServer::get_singleton()->bone_set_transform(
				get_rid(),
				get_global_transform());
	}
}

void PhysicalBone::_update_link_mass() {

	PhysicsServer::get_singleton()->bone_set_link_mass(
			get_rid(),
			link_mass);
}

void PhysicalBone::_start_physics_simulation() {
	if (_internal_simulate_physics || !parent_skeleton) {
		return;
	}
	reset_to_rest_position();
	PhysicsServer::get_singleton()->bone_set_force_integration_callback(get_rid(), this, "_direct_state_changed");
	parent_skeleton->set_bone_ignore_animation(bone_id, true);
	_internal_simulate_physics = true;
	_update_link_mass();
}

void PhysicalBone::_stop_physics_simulation() {
	if (!_internal_simulate_physics || !parent_skeleton) {
		return;
	}
	PhysicsServer::get_singleton()->bone_set_force_integration_callback(get_rid(), NULL, "");
	parent_skeleton->set_bone_ignore_animation(bone_id, false);
	_internal_simulate_physics = false;
}
