/*************************************************************************/
/*  physics_body.cpp                                                     */
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

Array PhysicsBody::get_collision_exceptions() {
	List<RID> exceptions;
	PhysicsServer::get_singleton()->body_get_collision_exceptions(get_rid(), &exceptions);
	Array ret;
	for (List<RID>::Element *E = exceptions.front(); E; E = E->next()) {
		RID body = E->get();
		ObjectID instance_id = PhysicsServer::get_singleton()->body_get_object_instance_id(body);
		Object *obj = ObjectDB::get_instance(instance_id);
		PhysicsBody *physics_body = Object::cast_to<PhysicsBody>(obj);
		ret.append(physics_body);
	}
	return ret;
}

void PhysicsBody::add_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	CollisionObject *collision_object = Object::cast_to<CollisionObject>(p_node);
	ERR_FAIL_COND_MSG(!collision_object, "Collision exception only works between two CollisionObject.");
	PhysicsServer::get_singleton()->body_add_collision_exception(get_rid(), collision_object->get_rid());
}

void PhysicsBody::remove_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	CollisionObject *collision_object = Object::cast_to<CollisionObject>(p_node);
	ERR_FAIL_COND_MSG(!collision_object, "Collision exception only works between two CollisionObject.");
	PhysicsServer::get_singleton()->body_remove_collision_exception(get_rid(), collision_object->get_rid());
}

void PhysicsBody::_set_layers(uint32_t p_mask) {
	set_collision_layer(p_mask);
	set_collision_mask(p_mask);
}

uint32_t PhysicsBody::_get_layers() const {
	return get_collision_layer();
}

void PhysicsBody::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_set_layers", "mask"), &PhysicsBody::_set_layers);
	ClassDB::bind_method(D_METHOD("_get_layers"), &PhysicsBody::_get_layers);
}

PhysicsBody::PhysicsBody(PhysicsServer::BodyMode p_mode) :
		CollisionObject(RID_PRIME(PhysicsServer::get_singleton()->body_create(p_mode)), false) {
}

#ifndef DISABLE_DEPRECATED
void StaticBody::set_friction(real_t p_friction) {
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

real_t StaticBody::get_friction() const {
	WARN_DEPRECATED_MSG("The method get_friction has been deprecated and will be removed in the future, use physics material instead.");

	if (physics_material_override.is_null()) {
		return 1;
	}

	return physics_material_override->get_friction();
}

void StaticBody::set_bounce(real_t p_bounce) {
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

real_t StaticBody::get_bounce() const {
	WARN_DEPRECATED_MSG("The method get_bounce has been deprecated and will be removed in the future, use physics material instead.");

	if (physics_material_override.is_null()) {
		return 0;
	}

	return physics_material_override->get_bounce();
}
#endif

void StaticBody::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
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
		emit_signal(SceneStringNames::get_singleton()->body_shape_entered, E->get().rid, node, E->get().shapes[i].body_shape, E->get().shapes[i].local_shape);
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
		emit_signal(SceneStringNames::get_singleton()->body_shape_exited, E->get().rid, node, E->get().shapes[i].body_shape, E->get().shapes[i].local_shape);
	}

	contact_monitor->locked = false;
}

void RigidBody::_body_inout(int p_status, const RID &p_body, ObjectID p_instance, int p_body_shape, int p_local_shape) {
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
		if (node) {
			E->get().shapes.insert(ShapePair(p_body_shape, p_local_shape));
		}

		if (E->get().in_tree) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_entered, p_body, node, p_body_shape, p_local_shape);
		}

	} else {
		//E->get().rc--;

		if (node) {
			E->get().shapes.erase(ShapePair(p_body_shape, p_local_shape));
		}

		bool in_tree = E->get().in_tree;

		if (E->get().shapes.empty()) {
			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree);
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_body_exit_tree);
				if (in_tree) {
					emit_signal(SceneStringNames::get_singleton()->body_exited, node);
				}
			}

			contact_monitor->body_map.erase(E);
		}
		if (node && in_tree) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_exited, p_body, obj, p_body_shape, p_local_shape);
		}
	}
}

struct _RigidBodyInOut {
	RID rid;
	ObjectID id;
	int shape;
	int local_shape;
};

void RigidBody::_direct_state_changed(Object *p_state) {
	state = Object::cast_to<PhysicsDirectBodyState>(p_state);
	ERR_FAIL_COND_MSG(!state, "Method '_direct_state_changed' must receive a valid PhysicsDirectBodyState object as argument");

	set_ignore_transform_notification(true);
	set_global_transform(state->get_transform());
	linear_velocity = state->get_linear_velocity();
	angular_velocity = state->get_angular_velocity();
	inverse_inertia_tensor = state->get_inverse_inertia_tensor();
	if (sleeping != state->is_sleeping()) {
		sleeping = state->is_sleeping();
		emit_signal(SceneStringNames::get_singleton()->sleeping_state_changed);
	}
	if (get_script_instance()) {
		get_script_instance()->call("_integrate_forces", state);
	}
	set_ignore_transform_notification(false);
	_on_transform_changed();

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
	update_configuration_warning();
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

	WARN_DEPRECATED_MSG("The method set_friction has been deprecated and will be removed in the future, use physics material instead.");

	ERR_FAIL_COND(p_friction < 0 || p_friction > 1);

	if (physics_material_override.is_null()) {
		physics_material_override.instance();
		set_physics_material_override(physics_material_override);
	}
	physics_material_override->set_friction(p_friction);
}
real_t RigidBody::get_friction() const {
	WARN_DEPRECATED_MSG("The method get_friction has been deprecated and will be removed in the future, use physics material instead.");

	if (physics_material_override.is_null()) {
		return 1;
	}

	return physics_material_override->get_friction();
}

void RigidBody::set_bounce(real_t p_bounce) {
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
real_t RigidBody::get_bounce() const {
	WARN_DEPRECATED_MSG("The method get_bounce has been deprecated and will be removed in the future, use physics material instead.");
	if (physics_material_override.is_null()) {
		return 0;
	}

	return physics_material_override->get_bounce();
}
#endif // DISABLE_DEPRECATED

void RigidBody::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
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
	if (state) {
		state->set_linear_velocity(linear_velocity);
	} else {
		PhysicsServer::get_singleton()->body_set_state(get_rid(), PhysicsServer::BODY_STATE_LINEAR_VELOCITY, linear_velocity);
	}
}

Vector3 RigidBody::get_linear_velocity() const {
	return linear_velocity;
}

void RigidBody::set_angular_velocity(const Vector3 &p_velocity) {
	angular_velocity = p_velocity;
	if (state) {
		state->set_angular_velocity(angular_velocity);
	} else {
		PhysicsServer::get_singleton()->body_set_state(get_rid(), PhysicsServer::BODY_STATE_ANGULAR_VELOCITY, angular_velocity);
	}
}
Vector3 RigidBody::get_angular_velocity() const {
	return angular_velocity;
}

Basis RigidBody::get_inverse_inertia_tensor() {
	return inverse_inertia_tensor;
}

void RigidBody::set_use_custom_integrator(bool p_enable) {
	if (custom_integrator == p_enable) {
		return;
	}

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

bool RigidBody::is_contact_monitor_enabled() const {
	return contact_monitor != nullptr;
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
			warning += "\n\n";
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

	ClassDB::bind_method(D_METHOD("get_inverse_inertia_tensor"), &RigidBody::get_inverse_inertia_tensor);

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
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "mass", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01,or_greater"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "weight", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01,or_greater", PROPERTY_USAGE_EDITOR), "set_weight", "get_weight");
#ifndef DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "friction", PROPERTY_HINT_RANGE, "0,1,0.01", 0), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce", PROPERTY_HINT_RANGE, "0,1,0.01", 0), "set_bounce", "get_bounce");
#endif // DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material_override", "get_physics_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "gravity_scale", PROPERTY_HINT_RANGE, "-128,128,0.01"), "set_gravity_scale", "get_gravity_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "custom_integrator"), "set_use_custom_integrator", "is_using_custom_integrator");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "continuous_cd"), "set_use_continuous_collision_detection", "is_using_continuous_collision_detection");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "contacts_reported", PROPERTY_HINT_RANGE, "0,64,1,or_greater"), "set_max_contacts_reported", "get_max_contacts_reported");
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
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "linear_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"), "set_linear_damp", "get_linear_damp");
	ADD_GROUP("Angular", "angular_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"), "set_angular_damp", "get_angular_damp");

	ADD_SIGNAL(MethodInfo("body_shape_entered", PropertyInfo(Variant::_RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("body_shape_exited", PropertyInfo(Variant::_RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
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
	state = nullptr;

	gravity_scale = 1;
	linear_damp = -1;
	angular_damp = -1;

	//angular_velocity=0;
	sleeping = false;
	ccd = false;

	custom_integrator = false;
	contact_monitor = nullptr;
	can_sleep = true;

	PhysicsServer::get_singleton()->body_set_force_integration_callback(get_rid(), this, "_direct_state_changed");
}

RigidBody::~RigidBody() {
	if (contact_monitor) {
		memdelete(contact_monitor);
	}
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

	bool collided = move_and_collide(p_motion, p_infinite_inertia, col, p_exclude_raycast_shapes, p_test_only);

	// Ugly hack as a hot fix, 65b3200 fix an issue but cause a problem with Bullet that broke games using Bullet.
	// The bug is something internal to Bullet, seems to be related to the Bullet’s margin. As not proper fix was found yet,
	// this temporary solution solves the issue for games using Bullet.
	bool is_bullet = PhysicsServerManager::current_server_id != 0;

	// Don't report collision when the whole motion is done.
	if (collided && (col.collision_safe_fraction < 1 || is_bullet)) {
		// Create a new instance when the cached reference is invalid or still in use in script.
		if (motion_cache.is_null() || motion_cache->reference_get_count() > 1) {
			motion_cache.instance();
			motion_cache->owner = this;
		}

		motion_cache->collision = col;

		return motion_cache;
	}

	return Ref<KinematicCollision>();
}

bool KinematicBody::move_and_collide(const Vector3 &p_motion, bool p_infinite_inertia, Collision &r_collision, bool p_exclude_raycast_shapes, bool p_test_only, bool p_cancel_sliding, const Set<RID> &p_exclude) {
	if (sync_to_physics) {
		ERR_PRINT("Functions move_and_slide and move_and_collide do not work together with 'sync to physics' option. Please read the documentation.");
	}

	Transform gt = get_global_transform();
	PhysicsServer::MotionResult result;
	bool colliding = PhysicsServer::get_singleton()->body_test_motion(get_rid(), gt, p_motion, p_infinite_inertia, &result, p_exclude_raycast_shapes, p_exclude);

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
			Vector3 motion_normal;
			if (motion_length > CMP_EPSILON) {
				motion_normal = p_motion / motion_length;
			}

			// Check depth of recovery.
			real_t projected_length = result.motion.dot(motion_normal);
			Vector3 recovery = result.motion - motion_normal * projected_length;
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
		r_collision.collision_safe_fraction = result.collision_safe_fraction;
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

//so, if you pass 45 as limit, avoid numerical precision errors when angle is 45.
#define FLOOR_ANGLE_THRESHOLD 0.01

Vector3 KinematicBody::_move_and_slide_internal(const Vector3 &p_linear_velocity, const Vector3 &p_snap, const Vector3 &p_up_direction, bool p_stop_on_slope, int p_max_slides, float p_floor_max_angle, bool p_infinite_inertia) {
	Vector3 body_velocity = p_linear_velocity;
	Vector3 body_velocity_normal = body_velocity.normalized();
	Vector3 up_direction = p_up_direction.normalized();
	bool was_on_floor = on_floor;

	for (int i = 0; i < 3; i++) {
		if (locked_axis & (1 << i)) {
			body_velocity[i] = 0;
		}
	}

	// Hack in order to work with calling from _process as well as from _physics_process; calling from thread is risky
	float delta = Engine::get_singleton()->is_in_physics_frame() ? get_physics_process_delta_time() : get_process_delta_time();

	Vector3 current_floor_velocity = floor_velocity;
	if (on_floor && on_floor_body.is_valid()) {
		// This approach makes sure there is less delay between the actual body velocity and the one we saved.
		PhysicsDirectBodyState *bs = PhysicsServer::get_singleton()->body_get_direct_state(on_floor_body);
		if (bs) {
			Transform gt = get_global_transform();
			Vector3 local_position = gt.origin - bs->get_transform().origin;
			current_floor_velocity = bs->get_velocity_at_local_position(local_position);
		} else {
			// Body is removed or destroyed, invalidate floor.
			current_floor_velocity = Vector3();
			on_floor_body = RID();
		}
	}

	colliders.clear();
	on_floor = false;
	on_ceiling = false;
	on_wall = false;
	floor_normal = Vector3();
	floor_velocity = Vector3();

	if (current_floor_velocity != Vector3() && on_floor_body.is_valid()) {
		Collision floor_collision;
		Set<RID> exclude;
		exclude.insert(on_floor_body);
		if (move_and_collide(current_floor_velocity * delta, p_infinite_inertia, floor_collision, true, false, false, exclude)) {
			colliders.push_back(floor_collision);
			_set_collision_direction(floor_collision, up_direction, p_floor_max_angle);
		}
	}

	on_floor_body = RID();
	Vector3 motion = body_velocity * delta;

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

				_set_collision_direction(collision, up_direction, p_floor_max_angle);

				if (on_floor && p_stop_on_slope) {
					if ((body_velocity_normal + up_direction).length() < 0.01) {
						Transform gt = get_global_transform();
						if (collision.travel.length() > margin) {
							gt.origin -= collision.travel.slide(up_direction);
						} else {
							gt.origin -= collision.travel;
						}
						set_global_transform(gt);
						return Vector3();
					}
				}

				if (sliding_enabled || !on_floor) {
					motion = collision.remainder.slide(collision.normal);
					body_velocity = body_velocity.slide(collision.normal);

					for (int j = 0; j < 3; j++) {
						if (locked_axis & (1 << j)) {
							body_velocity[j] = 0;
						}
					}
				} else {
					motion = collision.remainder;
				}
			}

			sliding_enabled = true;
		}

		if (!found_collision || motion == Vector3()) {
			break;
		}
	}

	if (was_on_floor && p_snap != Vector3() && !on_floor) {
		// Apply snap.
		Collision col;
		Transform gt = get_global_transform();

		if (move_and_collide(p_snap, p_infinite_inertia, col, false, true, false)) {
			bool apply = true;
			if (up_direction != Vector3()) {
				if (Math::acos(col.normal.dot(up_direction)) <= p_floor_max_angle + FLOOR_ANGLE_THRESHOLD) {
					on_floor = true;
					floor_normal = col.normal;
					on_floor_body = col.collider_rid;
					floor_velocity = col.collider_vel;
					if (p_stop_on_slope) {
						// move and collide may stray the object a bit because of pre un-stucking,
						// so only ensure that motion happens on floor direction in this case.
						if (col.travel.length() > margin) {
							col.travel = col.travel.project(up_direction);
						} else {
							col.travel = Vector3();
						}
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

Vector3 KinematicBody::move_and_slide(const Vector3 &p_linear_velocity, const Vector3 &p_up_direction, bool p_stop_on_slope, int p_max_slides, float p_floor_max_angle, bool p_infinite_inertia) {
	return _move_and_slide_internal(p_linear_velocity, Vector3(), p_up_direction, p_stop_on_slope, p_max_slides, p_floor_max_angle, p_infinite_inertia);
}

Vector3 KinematicBody::move_and_slide_with_snap(const Vector3 &p_linear_velocity, const Vector3 &p_snap, const Vector3 &p_up_direction, bool p_stop_on_slope, int p_max_slides, float p_floor_max_angle, bool p_infinite_inertia) {
	return _move_and_slide_internal(p_linear_velocity, p_snap, p_up_direction, p_stop_on_slope, p_max_slides, p_floor_max_angle, p_infinite_inertia);
}

void KinematicBody::_set_collision_direction(const Collision &p_collision, const Vector3 &p_up_direction, float p_floor_max_angle) {
	if (p_up_direction == Vector3()) {
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

bool KinematicBody::is_on_floor() const {
	return on_floor;
}

bool KinematicBody::is_on_wall() const {
	return on_wall;
}
bool KinematicBody::is_on_ceiling() const {
	return on_ceiling;
}

Vector3 KinematicBody::get_floor_normal() const {
	return floor_normal;
}

real_t KinematicBody::get_floor_angle(const Vector3 &p_up_direction) const {
	ERR_FAIL_COND_V(p_up_direction == Vector3(), 0);
	return Math::acos(floor_normal.dot(p_up_direction));
}

Vector3 KinematicBody::get_floor_velocity() const {
	return floor_velocity;
}

void KinematicBody::set_moving_platform_apply_velocity_on_leave(MovingPlatformApplyVelocityOnLeave p_on_leave_apply_velocity) {
	moving_platform_apply_velocity_on_leave = p_on_leave_apply_velocity;
}

KinematicBody::MovingPlatformApplyVelocityOnLeave KinematicBody::get_moving_platform_apply_velocity_on_leave() const {
	return moving_platform_apply_velocity_on_leave;
}

bool KinematicBody::test_move(const Transform &p_from, const Vector3 &p_motion, bool p_infinite_inertia) {
	ERR_FAIL_COND_V(!is_inside_tree(), false);

	PhysicsServer::MotionResult result;
	bool colliding = PhysicsServer::get_singleton()->body_test_motion(get_rid(), p_from, p_motion, p_infinite_inertia, &result);

	if (colliding) {
		// Don't report collision when the whole motion is done.
		return (result.collision_safe_fraction < 1.0);
	} else {
		return false;
	}
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
		r_collision.collider_rid = sep_res[deepest].collider;
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
	if (p_lock) {
		locked_axis |= p_axis;
	} else {
		locked_axis &= (~p_axis);
	}
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

	// Create a new instance when the cached reference is invalid or still in use in script.
	if (slide_colliders[p_bounce].is_null() || slide_colliders[p_bounce]->reference_get_count() > 1) {
		slide_colliders.write[p_bounce].instance();
		slide_colliders.write[p_bounce]->owner = this;
	}

	slide_colliders.write[p_bounce]->collision = colliders[p_bounce];
	return slide_colliders[p_bounce];
}

Ref<KinematicCollision> KinematicBody::_get_last_slide_collision() {
	if (colliders.size() == 0) {
		return Ref<KinematicCollision>();
	}
	return _get_slide_collision(colliders.size() - 1);
}

void KinematicBody::set_sync_to_physics(bool p_enable) {
	if (sync_to_physics == p_enable) {
		return;
	}
	sync_to_physics = p_enable;

	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	if (p_enable) {
		PhysicsServer::get_singleton()->body_set_force_integration_callback(get_rid(), this, "_direct_state_changed");
		set_only_update_transform_changes(true);
		set_notify_local_transform(true);
	} else {
		PhysicsServer::get_singleton()->body_set_force_integration_callback(get_rid(), nullptr, "");
		set_only_update_transform_changes(false);
		set_notify_local_transform(false);
	}
}

bool KinematicBody::is_sync_to_physics_enabled() const {
	return sync_to_physics;
}

void KinematicBody::_direct_state_changed(Object *p_state) {
	if (!sync_to_physics) {
		return;
	}

	PhysicsDirectBodyState *state = Object::cast_to<PhysicsDirectBodyState>(p_state);
	ERR_FAIL_COND_MSG(!state, "Method '_direct_state_changed' must receive a valid PhysicsDirectBodyState object as argument");

	last_valid_transform = state->get_transform();
	set_notify_local_transform(false);
	set_global_transform(last_valid_transform);
	set_notify_local_transform(true);
	_on_transform_changed();
}

void KinematicBody::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		last_valid_transform = get_global_transform();

		// Reset move_and_slide() data.
		on_floor = false;
		on_floor_body = RID();
		on_ceiling = false;
		on_wall = false;
		colliders.clear();
		floor_velocity = Vector3();
	}

	if (p_what == NOTIFICATION_LOCAL_TRANSFORM_CHANGED) {
		//used by sync to physics, send the new transform to the physics
		Transform new_transform = get_global_transform();
		PhysicsServer::get_singleton()->body_set_state(get_rid(), PhysicsServer::BODY_STATE_TRANSFORM, new_transform);
		//but then revert changes
		set_notify_local_transform(false);
		set_global_transform(last_valid_transform);
		set_notify_local_transform(true);
		_on_transform_changed();
	}
}

void KinematicBody::_bind_methods() {
	ClassDB::bind_method(D_METHOD("move_and_collide", "rel_vec", "infinite_inertia", "exclude_raycast_shapes", "test_only"), &KinematicBody::_move, DEFVAL(true), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("move_and_slide", "linear_velocity", "up_direction", "stop_on_slope", "max_slides", "floor_max_angle", "infinite_inertia"), &KinematicBody::move_and_slide, DEFVAL(Vector3(0, 0, 0)), DEFVAL(false), DEFVAL(4), DEFVAL(Math::deg2rad((float)45)), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("move_and_slide_with_snap", "linear_velocity", "snap", "up_direction", "stop_on_slope", "max_slides", "floor_max_angle", "infinite_inertia"), &KinematicBody::move_and_slide_with_snap, DEFVAL(Vector3(0, 0, 0)), DEFVAL(false), DEFVAL(4), DEFVAL(Math::deg2rad((float)45)), DEFVAL(true));

	ClassDB::bind_method(D_METHOD("test_move", "from", "rel_vec", "infinite_inertia"), &KinematicBody::test_move, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("is_on_floor"), &KinematicBody::is_on_floor);
	ClassDB::bind_method(D_METHOD("is_on_ceiling"), &KinematicBody::is_on_ceiling);
	ClassDB::bind_method(D_METHOD("is_on_wall"), &KinematicBody::is_on_wall);
	ClassDB::bind_method(D_METHOD("get_floor_normal"), &KinematicBody::get_floor_normal);
	ClassDB::bind_method(D_METHOD("get_floor_angle", "up_direction"), &KinematicBody::get_floor_angle, DEFVAL(Vector3(0.0, 1.0, 0.0)));
	ClassDB::bind_method(D_METHOD("get_floor_velocity"), &KinematicBody::get_floor_velocity);

	ClassDB::bind_method(D_METHOD("set_axis_lock", "axis", "lock"), &KinematicBody::set_axis_lock);
	ClassDB::bind_method(D_METHOD("get_axis_lock", "axis"), &KinematicBody::get_axis_lock);

	ClassDB::bind_method(D_METHOD("set_safe_margin", "pixels"), &KinematicBody::set_safe_margin);
	ClassDB::bind_method(D_METHOD("get_safe_margin"), &KinematicBody::get_safe_margin);

	ClassDB::bind_method(D_METHOD("set_moving_platform_apply_velocity_on_leave", "on_leave_apply_velocity"), &KinematicBody::set_moving_platform_apply_velocity_on_leave);
	ClassDB::bind_method(D_METHOD("get_moving_platform_apply_velocity_on_leave"), &KinematicBody::get_moving_platform_apply_velocity_on_leave);

	ClassDB::bind_method(D_METHOD("get_slide_count"), &KinematicBody::get_slide_count);
	ClassDB::bind_method(D_METHOD("get_slide_collision", "slide_idx"), &KinematicBody::_get_slide_collision);
	ClassDB::bind_method(D_METHOD("get_last_slide_collision"), &KinematicBody::_get_last_slide_collision);

	ClassDB::bind_method(D_METHOD("set_sync_to_physics", "enable"), &KinematicBody::set_sync_to_physics);
	ClassDB::bind_method(D_METHOD("is_sync_to_physics_enabled"), &KinematicBody::is_sync_to_physics_enabled);

	ClassDB::bind_method(D_METHOD("_direct_state_changed"), &KinematicBody::_direct_state_changed);

	ADD_GROUP("Axis Lock", "axis_lock_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_motion_x"), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_X);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_motion_y"), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_Y);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_motion_z"), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_Z);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "move_lock_x", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_X);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "move_lock_y", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_Y);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "move_lock_z", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_axis_lock", "get_axis_lock", PhysicsServer::BODY_AXIS_LINEAR_Z);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "collision/safe_margin", PROPERTY_HINT_RANGE, "0.001,256,0.001"), "set_safe_margin", "get_safe_margin");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motion/sync_to_physics"), "set_sync_to_physics", "is_sync_to_physics_enabled");
	ADD_GROUP("Moving Platform", "moving_platform");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "moving_platform_apply_velocity_on_leave", PROPERTY_HINT_ENUM, "Always,Upward Only,Never", PROPERTY_USAGE_DEFAULT), "set_moving_platform_apply_velocity_on_leave", "get_moving_platform_apply_velocity_on_leave");

	BIND_ENUM_CONSTANT(PLATFORM_VEL_ON_LEAVE_ALWAYS);
	BIND_ENUM_CONSTANT(PLATFORM_VEL_ON_LEAVE_UPWARD_ONLY);
	BIND_ENUM_CONSTANT(PLATFORM_VEL_ON_LEAVE_NEVER);
}

KinematicBody::KinematicBody() :
		PhysicsBody(PhysicsServer::BODY_MODE_KINEMATIC) {
	locked_axis = 0;
	on_floor = false;
	on_ceiling = false;
	on_wall = false;

	set_safe_margin(0.001);
}

KinematicBody::~KinematicBody() {
	if (motion_cache.is_valid()) {
		motion_cache->owner = nullptr;
	}

	for (int i = 0; i < slide_colliders.size(); i++) {
		if (slide_colliders[i].is_valid()) {
			slide_colliders.write[i]->owner = nullptr;
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

real_t KinematicCollision::get_angle(const Vector3 &p_up_direction) const {
	ERR_FAIL_COND_V(p_up_direction == Vector3(), 0);
	return collision.get_angle(p_up_direction);
}

Object *KinematicCollision::get_local_shape() const {
	if (!owner) {
		return nullptr;
	}
	uint32_t ownerid = owner->shape_find_owner(collision.local_shape);
	return owner->shape_owner_get_owner(ownerid);
}

Object *KinematicCollision::get_collider() const {
	if (collision.collider) {
		return ObjectDB::get_instance(collision.collider);
	}

	return nullptr;
}
ObjectID KinematicCollision::get_collider_id() const {
	return collision.collider;
}
RID KinematicCollision::get_collider_rid() const {
	return collision.collider_rid;
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

	return nullptr;
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
	ClassDB::bind_method(D_METHOD("get_angle", "up_direction"), &KinematicCollision::get_angle, DEFVAL(Vector3(0.0, 1.0, 0.0)));
	ClassDB::bind_method(D_METHOD("get_local_shape"), &KinematicCollision::get_local_shape);
	ClassDB::bind_method(D_METHOD("get_collider"), &KinematicCollision::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_id"), &KinematicCollision::get_collider_id);
	ClassDB::bind_method(D_METHOD("get_collider_rid"), &KinematicCollision::get_collider_rid);
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
	ADD_PROPERTY(PropertyInfo(Variant::_RID, "collider_rid"), "", "get_collider_rid");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collider_shape"), "", "get_collider_shape");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_shape_index"), "", "get_collider_shape_index");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "collider_velocity"), "", "get_collider_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::NIL, "collider_metadata", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT), "", "get_collider_metadata");
}

KinematicCollision::KinematicCollision() {
	collision.collider = 0;
	collision.collider_shape = 0;
	collision.local_shape = 0;
	owner = nullptr;
}

///////////////////////////////////////

bool PhysicalBone::JointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
	return false;
}

bool PhysicalBone::JointData::_get(const StringName &p_name, Variant &r_ret) const {
	return false;
}

void PhysicalBone::JointData::_get_property_list(List<PropertyInfo> *p_list) const {
}

void PhysicalBone::apply_central_impulse(const Vector3 &p_impulse) {
	PhysicsServer::get_singleton()->body_apply_central_impulse(get_rid(), p_impulse);
}

void PhysicalBone::apply_impulse(const Vector3 &p_pos, const Vector3 &p_impulse) {
	PhysicsServer::get_singleton()->body_apply_impulse(get_rid(), p_pos, p_impulse);
}

bool PhysicalBone::PinJointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
	if (JointData::_set(p_name, p_value, j)) {
		return true;
	}

	if ("joint_constraints/bias" == p_name) {
		bias = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->pin_joint_set_param(j, PhysicsServer::PIN_JOINT_BIAS, bias);
		}

	} else if ("joint_constraints/damping" == p_name) {
		damping = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->pin_joint_set_param(j, PhysicsServer::PIN_JOINT_DAMPING, damping);
		}

	} else if ("joint_constraints/impulse_clamp" == p_name) {
		impulse_clamp = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->pin_joint_set_param(j, PhysicsServer::PIN_JOINT_IMPULSE_CLAMP, impulse_clamp);
		}

	} else {
		return false;
	}

	return true;
}

bool PhysicalBone::PinJointData::_get(const StringName &p_name, Variant &r_ret) const {
	if (JointData::_get(p_name, r_ret)) {
		return true;
	}

	if ("joint_constraints/bias" == p_name) {
		r_ret = bias;
	} else if ("joint_constraints/damping" == p_name) {
		r_ret = damping;
	} else if ("joint_constraints/impulse_clamp" == p_name) {
		r_ret = impulse_clamp;
	} else {
		return false;
	}

	return true;
}

void PhysicalBone::PinJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);

	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/bias"), PROPERTY_HINT_RANGE, "0.01,0.99,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/damping"), PROPERTY_HINT_RANGE, "0.01,8.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/impulse_clamp"), PROPERTY_HINT_RANGE, "0.0,64.0,0.01"));
}

bool PhysicalBone::ConeJointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
	if (JointData::_set(p_name, p_value, j)) {
		return true;
	}

	if ("joint_constraints/swing_span" == p_name) {
		swing_span = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->cone_twist_joint_set_param(j, PhysicsServer::CONE_TWIST_JOINT_SWING_SPAN, swing_span);
		}

	} else if ("joint_constraints/twist_span" == p_name) {
		twist_span = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->cone_twist_joint_set_param(j, PhysicsServer::CONE_TWIST_JOINT_TWIST_SPAN, twist_span);
		}

	} else if ("joint_constraints/bias" == p_name) {
		bias = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->cone_twist_joint_set_param(j, PhysicsServer::CONE_TWIST_JOINT_BIAS, bias);
		}

	} else if ("joint_constraints/softness" == p_name) {
		softness = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->cone_twist_joint_set_param(j, PhysicsServer::CONE_TWIST_JOINT_SOFTNESS, softness);
		}

	} else if ("joint_constraints/relaxation" == p_name) {
		relaxation = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->cone_twist_joint_set_param(j, PhysicsServer::CONE_TWIST_JOINT_RELAXATION, relaxation);
		}

	} else {
		return false;
	}

	return true;
}

bool PhysicalBone::ConeJointData::_get(const StringName &p_name, Variant &r_ret) const {
	if (JointData::_get(p_name, r_ret)) {
		return true;
	}

	if ("joint_constraints/swing_span" == p_name) {
		r_ret = Math::rad2deg(swing_span);
	} else if ("joint_constraints/twist_span" == p_name) {
		r_ret = Math::rad2deg(twist_span);
	} else if ("joint_constraints/bias" == p_name) {
		r_ret = bias;
	} else if ("joint_constraints/softness" == p_name) {
		r_ret = softness;
	} else if ("joint_constraints/relaxation" == p_name) {
		r_ret = relaxation;
	} else {
		return false;
	}

	return true;
}

void PhysicalBone::ConeJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);

	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/swing_span"), PROPERTY_HINT_RANGE, "-180,180,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/twist_span"), PROPERTY_HINT_RANGE, "-40000,40000,0.1,or_lesser,or_greater"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/bias"), PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/softness"), PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/relaxation"), PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
}

bool PhysicalBone::HingeJointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
	if (JointData::_set(p_name, p_value, j)) {
		return true;
	}

	if ("joint_constraints/angular_limit_enabled" == p_name) {
		angular_limit_enabled = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->hinge_joint_set_flag(j, PhysicsServer::HINGE_JOINT_FLAG_USE_LIMIT, angular_limit_enabled);
		}

	} else if ("joint_constraints/angular_limit_upper" == p_name) {
		angular_limit_upper = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->hinge_joint_set_param(j, PhysicsServer::HINGE_JOINT_LIMIT_UPPER, angular_limit_upper);
		}

	} else if ("joint_constraints/angular_limit_lower" == p_name) {
		angular_limit_lower = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->hinge_joint_set_param(j, PhysicsServer::HINGE_JOINT_LIMIT_LOWER, angular_limit_lower);
		}

	} else if ("joint_constraints/angular_limit_bias" == p_name) {
		angular_limit_bias = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->hinge_joint_set_param(j, PhysicsServer::HINGE_JOINT_LIMIT_BIAS, angular_limit_bias);
		}

	} else if ("joint_constraints/angular_limit_softness" == p_name) {
		angular_limit_softness = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->hinge_joint_set_param(j, PhysicsServer::HINGE_JOINT_LIMIT_SOFTNESS, angular_limit_softness);
		}

	} else if ("joint_constraints/angular_limit_relaxation" == p_name) {
		angular_limit_relaxation = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->hinge_joint_set_param(j, PhysicsServer::HINGE_JOINT_LIMIT_RELAXATION, angular_limit_relaxation);
		}

	} else {
		return false;
	}

	return true;
}

bool PhysicalBone::HingeJointData::_get(const StringName &p_name, Variant &r_ret) const {
	if (JointData::_get(p_name, r_ret)) {
		return true;
	}

	if ("joint_constraints/angular_limit_enabled" == p_name) {
		r_ret = angular_limit_enabled;
	} else if ("joint_constraints/angular_limit_upper" == p_name) {
		r_ret = Math::rad2deg(angular_limit_upper);
	} else if ("joint_constraints/angular_limit_lower" == p_name) {
		r_ret = Math::rad2deg(angular_limit_lower);
	} else if ("joint_constraints/angular_limit_bias" == p_name) {
		r_ret = angular_limit_bias;
	} else if ("joint_constraints/angular_limit_softness" == p_name) {
		r_ret = angular_limit_softness;
	} else if ("joint_constraints/angular_limit_relaxation" == p_name) {
		r_ret = angular_limit_relaxation;
	} else {
		return false;
	}

	return true;
}

void PhysicalBone::HingeJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);

	p_list->push_back(PropertyInfo(Variant::BOOL, PNAME("joint_constraints/angular_limit_enabled")));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/angular_limit_upper"), PROPERTY_HINT_RANGE, "-180,180,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/angular_limit_lower"), PROPERTY_HINT_RANGE, "-180,180,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/angular_limit_bias"), PROPERTY_HINT_RANGE, "0.01,0.99,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/angular_limit_softness"), PROPERTY_HINT_RANGE, "0.01,16,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/angular_limit_relaxation"), PROPERTY_HINT_RANGE, "0.01,16,0.01"));
}

bool PhysicalBone::SliderJointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
	if (JointData::_set(p_name, p_value, j)) {
		return true;
	}

	if ("joint_constraints/linear_limit_upper" == p_name) {
		linear_limit_upper = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->slider_joint_set_param(j, PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_UPPER, linear_limit_upper);
		}

	} else if ("joint_constraints/linear_limit_lower" == p_name) {
		linear_limit_lower = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->slider_joint_set_param(j, PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_LOWER, linear_limit_lower);
		}

	} else if ("joint_constraints/linear_limit_softness" == p_name) {
		linear_limit_softness = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->slider_joint_set_param(j, PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS, linear_limit_softness);
		}

	} else if ("joint_constraints/linear_limit_restitution" == p_name) {
		linear_limit_restitution = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->slider_joint_set_param(j, PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION, linear_limit_restitution);
		}

	} else if ("joint_constraints/linear_limit_damping" == p_name) {
		linear_limit_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->slider_joint_set_param(j, PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_DAMPING, linear_limit_restitution);
		}

	} else if ("joint_constraints/angular_limit_upper" == p_name) {
		angular_limit_upper = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->slider_joint_set_param(j, PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_UPPER, angular_limit_upper);
		}

	} else if ("joint_constraints/angular_limit_lower" == p_name) {
		angular_limit_lower = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->slider_joint_set_param(j, PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_LOWER, angular_limit_lower);
		}

	} else if ("joint_constraints/angular_limit_softness" == p_name) {
		angular_limit_softness = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->slider_joint_set_param(j, PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS, angular_limit_softness);
		}

	} else if ("joint_constraints/angular_limit_restitution" == p_name) {
		angular_limit_restitution = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->slider_joint_set_param(j, PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS, angular_limit_softness);
		}

	} else if ("joint_constraints/angular_limit_damping" == p_name) {
		angular_limit_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->slider_joint_set_param(j, PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_DAMPING, angular_limit_damping);
		}

	} else {
		return false;
	}

	return true;
}

bool PhysicalBone::SliderJointData::_get(const StringName &p_name, Variant &r_ret) const {
	if (JointData::_get(p_name, r_ret)) {
		return true;
	}

	if ("joint_constraints/linear_limit_upper" == p_name) {
		r_ret = linear_limit_upper;
	} else if ("joint_constraints/linear_limit_lower" == p_name) {
		r_ret = linear_limit_lower;
	} else if ("joint_constraints/linear_limit_softness" == p_name) {
		r_ret = linear_limit_softness;
	} else if ("joint_constraints/linear_limit_restitution" == p_name) {
		r_ret = linear_limit_restitution;
	} else if ("joint_constraints/linear_limit_damping" == p_name) {
		r_ret = linear_limit_damping;
	} else if ("joint_constraints/angular_limit_upper" == p_name) {
		r_ret = Math::rad2deg(angular_limit_upper);
	} else if ("joint_constraints/angular_limit_lower" == p_name) {
		r_ret = Math::rad2deg(angular_limit_lower);
	} else if ("joint_constraints/angular_limit_softness" == p_name) {
		r_ret = angular_limit_softness;
	} else if ("joint_constraints/angular_limit_restitution" == p_name) {
		r_ret = angular_limit_restitution;
	} else if ("joint_constraints/angular_limit_damping" == p_name) {
		r_ret = angular_limit_damping;
	} else {
		return false;
	}

	return true;
}

void PhysicalBone::SliderJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);

	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/linear_limit_upper")));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/linear_limit_lower")));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/linear_limit_softness"), PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/linear_limit_restitution"), PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/linear_limit_damping"), PROPERTY_HINT_RANGE, "0,16.0,0.01"));

	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/angular_limit_upper"), PROPERTY_HINT_RANGE, "-180,180,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/angular_limit_lower"), PROPERTY_HINT_RANGE, "-180,180,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/angular_limit_softness"), PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/angular_limit_restitution"), PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, PNAME("joint_constraints/angular_limit_damping"), PROPERTY_HINT_RANGE, "0,16.0,0.01"));
}

bool PhysicalBone::SixDOFJointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
	if (JointData::_set(p_name, p_value, j)) {
		return true;
	}

	String path = p_name;

	if (!path.begins_with("joint_constraints/")) {
		return false;
	}

	Vector3::Axis axis;
	{
		const String axis_s = path.get_slicec('/', 1);
		if ("x" == axis_s) {
			axis = Vector3::AXIS_X;
		} else if ("y" == axis_s) {
			axis = Vector3::AXIS_Y;
		} else if ("z" == axis_s) {
			axis = Vector3::AXIS_Z;
		} else {
			return false;
		}
	}

	String var_name = path.get_slicec('/', 2);

	if ("linear_limit_enabled" == var_name) {
		axis_data[axis].linear_limit_enabled = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(j, axis, PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT, axis_data[axis].linear_limit_enabled);
		}

	} else if ("linear_limit_upper" == var_name) {
		axis_data[axis].linear_limit_upper = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_LINEAR_UPPER_LIMIT, axis_data[axis].linear_limit_upper);
		}

	} else if ("linear_limit_lower" == var_name) {
		axis_data[axis].linear_limit_lower = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_LINEAR_LOWER_LIMIT, axis_data[axis].linear_limit_lower);
		}

	} else if ("linear_limit_softness" == var_name) {
		axis_data[axis].linear_limit_softness = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS, axis_data[axis].linear_limit_softness);
		}

	} else if ("linear_spring_enabled" == var_name) {
		axis_data[axis].linear_spring_enabled = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(j, axis, PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING, axis_data[axis].linear_spring_enabled);
		}

	} else if ("linear_spring_stiffness" == var_name) {
		axis_data[axis].linear_spring_stiffness = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_LINEAR_SPRING_STIFFNESS, axis_data[axis].linear_spring_stiffness);
		}

	} else if ("linear_spring_damping" == var_name) {
		axis_data[axis].linear_spring_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_LINEAR_SPRING_DAMPING, axis_data[axis].linear_spring_damping);
		}

	} else if ("linear_equilibrium_point" == var_name) {
		axis_data[axis].linear_equilibrium_point = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_LINEAR_SPRING_EQUILIBRIUM_POINT, axis_data[axis].linear_equilibrium_point);
		}

	} else if ("linear_restitution" == var_name) {
		axis_data[axis].linear_restitution = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_LINEAR_RESTITUTION, axis_data[axis].linear_restitution);
		}

	} else if ("linear_damping" == var_name) {
		axis_data[axis].linear_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_LINEAR_DAMPING, axis_data[axis].linear_damping);
		}

	} else if ("angular_limit_enabled" == var_name) {
		axis_data[axis].angular_limit_enabled = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(j, axis, PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT, axis_data[axis].angular_limit_enabled);
		}

	} else if ("angular_limit_upper" == var_name) {
		axis_data[axis].angular_limit_upper = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_ANGULAR_UPPER_LIMIT, axis_data[axis].angular_limit_upper);
		}

	} else if ("angular_limit_lower" == var_name) {
		axis_data[axis].angular_limit_lower = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_ANGULAR_LOWER_LIMIT, axis_data[axis].angular_limit_lower);
		}

	} else if ("angular_limit_softness" == var_name) {
		axis_data[axis].angular_limit_softness = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS, axis_data[axis].angular_limit_softness);
		}

	} else if ("angular_restitution" == var_name) {
		axis_data[axis].angular_restitution = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_ANGULAR_RESTITUTION, axis_data[axis].angular_restitution);
		}

	} else if ("angular_damping" == var_name) {
		axis_data[axis].angular_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_ANGULAR_DAMPING, axis_data[axis].angular_damping);
		}

	} else if ("erp" == var_name) {
		axis_data[axis].erp = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_ANGULAR_ERP, axis_data[axis].erp);
		}

	} else if ("angular_spring_enabled" == var_name) {
		axis_data[axis].angular_spring_enabled = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(j, axis, PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING, axis_data[axis].angular_spring_enabled);
		}

	} else if ("angular_spring_stiffness" == var_name) {
		axis_data[axis].angular_spring_stiffness = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_ANGULAR_SPRING_STIFFNESS, axis_data[axis].angular_spring_stiffness);
		}

	} else if ("angular_spring_damping" == var_name) {
		axis_data[axis].angular_spring_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_ANGULAR_SPRING_DAMPING, axis_data[axis].angular_spring_damping);
		}

	} else if ("angular_equilibrium_point" == var_name) {
		axis_data[axis].angular_equilibrium_point = p_value;
		if (j.is_valid()) {
			PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer::G6DOF_JOINT_ANGULAR_SPRING_EQUILIBRIUM_POINT, axis_data[axis].angular_equilibrium_point);
		}

	} else {
		return false;
	}

	return true;
}

bool PhysicalBone::SixDOFJointData::_get(const StringName &p_name, Variant &r_ret) const {
	if (JointData::_get(p_name, r_ret)) {
		return true;
	}

	String path = p_name;

	if (!path.begins_with("joint_constraints/")) {
		return false;
	}

	int axis;
	{
		const String axis_s = path.get_slicec('/', 1);
		if ("x" == axis_s) {
			axis = 0;
		} else if ("y" == axis_s) {
			axis = 1;
		} else if ("z" == axis_s) {
			axis = 2;
		} else {
			return false;
		}
	}

	String var_name = path.get_slicec('/', 2);

	if ("linear_limit_enabled" == var_name) {
		r_ret = axis_data[axis].linear_limit_enabled;
	} else if ("linear_limit_upper" == var_name) {
		r_ret = axis_data[axis].linear_limit_upper;
	} else if ("linear_limit_lower" == var_name) {
		r_ret = axis_data[axis].linear_limit_lower;
	} else if ("linear_limit_softness" == var_name) {
		r_ret = axis_data[axis].linear_limit_softness;
	} else if ("linear_spring_enabled" == var_name) {
		r_ret = axis_data[axis].linear_spring_enabled;
	} else if ("linear_spring_stiffness" == var_name) {
		r_ret = axis_data[axis].linear_spring_stiffness;
	} else if ("linear_spring_damping" == var_name) {
		r_ret = axis_data[axis].linear_spring_damping;
	} else if ("linear_equilibrium_point" == var_name) {
		r_ret = axis_data[axis].linear_equilibrium_point;
	} else if ("linear_restitution" == var_name) {
		r_ret = axis_data[axis].linear_restitution;
	} else if ("linear_damping" == var_name) {
		r_ret = axis_data[axis].linear_damping;
	} else if ("angular_limit_enabled" == var_name) {
		r_ret = axis_data[axis].angular_limit_enabled;
	} else if ("angular_limit_upper" == var_name) {
		r_ret = Math::rad2deg(axis_data[axis].angular_limit_upper);
	} else if ("angular_limit_lower" == var_name) {
		r_ret = Math::rad2deg(axis_data[axis].angular_limit_lower);
	} else if ("angular_limit_softness" == var_name) {
		r_ret = axis_data[axis].angular_limit_softness;
	} else if ("angular_restitution" == var_name) {
		r_ret = axis_data[axis].angular_restitution;
	} else if ("angular_damping" == var_name) {
		r_ret = axis_data[axis].angular_damping;
	} else if ("erp" == var_name) {
		r_ret = axis_data[axis].erp;
	} else if ("angular_spring_enabled" == var_name) {
		r_ret = axis_data[axis].angular_spring_enabled;
	} else if ("angular_spring_stiffness" == var_name) {
		r_ret = axis_data[axis].angular_spring_stiffness;
	} else if ("angular_spring_damping" == var_name) {
		r_ret = axis_data[axis].angular_spring_damping;
	} else if ("angular_equilibrium_point" == var_name) {
		r_ret = axis_data[axis].angular_equilibrium_point;
	} else {
		return false;
	}

	return true;
}

void PhysicalBone::SixDOFJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	const StringName axis_names[] = { PNAME("x"), PNAME("y"), PNAME("z") };
	for (int i = 0; i < 3; ++i) {
		const String prefix = vformat("%s/%s/", PNAME("joint_constraints"), axis_names[i]);
		p_list->push_back(PropertyInfo(Variant::BOOL, prefix + PNAME("linear_limit_enabled")));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("linear_limit_upper")));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("linear_limit_lower")));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("linear_limit_softness"), PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::BOOL, prefix + PNAME("linear_spring_enabled")));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("linear_spring_stiffness")));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("linear_spring_damping")));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("linear_equilibrium_point")));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("linear_restitution"), PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("linear_damping"), PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::BOOL, prefix + PNAME("angular_limit_enabled")));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("angular_limit_upper"), PROPERTY_HINT_RANGE, "-180,180,0.01"));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("angular_limit_lower"), PROPERTY_HINT_RANGE, "-180,180,0.01"));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("angular_limit_softness"), PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("angular_restitution"), PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("angular_damping"), PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("erp")));
		p_list->push_back(PropertyInfo(Variant::BOOL, prefix + PNAME("angular_spring_enabled")));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("angular_spring_stiffness")));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("angular_spring_damping")));
		p_list->push_back(PropertyInfo(Variant::REAL, prefix + PNAME("angular_equilibrium_point")));
	}
}

bool PhysicalBone::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "bone_name") {
		set_bone_name(p_value);
		return true;
	}

	if (joint_data) {
		if (joint_data->_set(p_name, p_value, joint)) {
#ifdef TOOLS_ENABLED
			if (get_gizmo().is_valid()) {
				get_gizmo()->redraw();
			}
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
		String names;
		for (int i = 0; i < parent->get_bone_count(); i++) {
			if (i > 0) {
				names += ",";
			}
			names += parent->get_bone_name(i);
		}

		p_list->push_back(PropertyInfo(Variant::STRING, PNAME("bone_name"), PROPERTY_HINT_ENUM, names));
	} else {
		p_list->push_back(PropertyInfo(Variant::STRING, PNAME("bone_name")));
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
			_reset_physics_simulation_state();
			if (!joint.is_valid() && joint_data) {
				_reload_joint();
			}
			break;
		case NOTIFICATION_EXIT_TREE:
			if (parent_skeleton) {
				if (-1 != bone_id) {
					parent_skeleton->unbind_physical_bone_from_bone(bone_id);
					parent_skeleton->unbind_child_node_from_bone(bone_id, this);
					bone_id = -1;
				}
			}
			parent_skeleton = nullptr;
			if (joint.is_valid()) {
				PhysicsServer::get_singleton()->free(joint);
				joint = RID();
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
	if (!simulate_physics || !_internal_simulate_physics) {
		return;
	}

	/// Update bone transform

	PhysicsDirectBodyState *state = Object::cast_to<PhysicsDirectBodyState>(p_state);
	ERR_FAIL_COND_MSG(!state, "Method '_direct_state_changed' must receive a valid PhysicsDirectBodyState object as argument");

	Transform global_transform(state->get_transform());

	set_ignore_transform_notification(true);
	set_global_transform(global_transform);
	set_ignore_transform_notification(false);
	_on_transform_changed();

	// Update skeleton
	if (parent_skeleton) {
		if (-1 != bone_id) {
			parent_skeleton->set_bone_global_pose_override(bone_id, parent_skeleton->get_global_transform().affine_inverse() * (global_transform * body_offset_inverse), 1.0, true);
		}
	}
}

void PhysicalBone::_bind_methods() {
	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &PhysicalBone::apply_central_impulse);
	ClassDB::bind_method(D_METHOD("apply_impulse", "position", "impulse"), &PhysicalBone::apply_impulse);

	ClassDB::bind_method(D_METHOD("_direct_state_changed"), &PhysicalBone::_direct_state_changed);

	ClassDB::bind_method(D_METHOD("set_joint_type", "joint_type"), &PhysicalBone::set_joint_type);
	ClassDB::bind_method(D_METHOD("get_joint_type"), &PhysicalBone::get_joint_type);

	ClassDB::bind_method(D_METHOD("set_joint_offset", "offset"), &PhysicalBone::set_joint_offset);
	ClassDB::bind_method(D_METHOD("get_joint_offset"), &PhysicalBone::get_joint_offset);

	ClassDB::bind_method(D_METHOD("set_body_offset", "offset"), &PhysicalBone::set_body_offset);
	ClassDB::bind_method(D_METHOD("get_body_offset"), &PhysicalBone::get_body_offset);

	ClassDB::bind_method(D_METHOD("is_static_body"), &PhysicalBone::is_static_body);

	ClassDB::bind_method(D_METHOD("get_simulate_physics"), &PhysicalBone::get_simulate_physics);

	ClassDB::bind_method(D_METHOD("is_simulating_physics"), &PhysicalBone::is_simulating_physics);

	ClassDB::bind_method(D_METHOD("get_bone_id"), &PhysicalBone::get_bone_id);

	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &PhysicalBone::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &PhysicalBone::get_mass);

	ClassDB::bind_method(D_METHOD("set_weight", "weight"), &PhysicalBone::set_weight);
	ClassDB::bind_method(D_METHOD("get_weight"), &PhysicalBone::get_weight);

	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &PhysicalBone::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &PhysicalBone::get_friction);

	ClassDB::bind_method(D_METHOD("set_bounce", "bounce"), &PhysicalBone::set_bounce);
	ClassDB::bind_method(D_METHOD("get_bounce"), &PhysicalBone::get_bounce);

	ClassDB::bind_method(D_METHOD("set_gravity_scale", "gravity_scale"), &PhysicalBone::set_gravity_scale);
	ClassDB::bind_method(D_METHOD("get_gravity_scale"), &PhysicalBone::get_gravity_scale);

	ADD_GROUP("Joint", "joint_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "joint_type", PROPERTY_HINT_ENUM, "None,PinJoint,ConeJoint,HingeJoint,SliderJoint,6DOFJoint"), "set_joint_type", "get_joint_type");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "joint_offset"), "set_joint_offset", "get_joint_offset");

	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "body_offset"), "set_body_offset", "get_body_offset");

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "mass", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "weight", PROPERTY_HINT_EXP_RANGE, "0.01,65535,0.01"), "set_weight", "get_weight");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "friction", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_bounce", "get_bounce");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "gravity_scale", PROPERTY_HINT_RANGE, "-10,10,0.01"), "set_gravity_scale", "get_gravity_scale");

	BIND_ENUM_CONSTANT(JOINT_TYPE_NONE);
	BIND_ENUM_CONSTANT(JOINT_TYPE_PIN);
	BIND_ENUM_CONSTANT(JOINT_TYPE_CONE);
	BIND_ENUM_CONSTANT(JOINT_TYPE_HINGE);
	BIND_ENUM_CONSTANT(JOINT_TYPE_SLIDER);
	BIND_ENUM_CONSTANT(JOINT_TYPE_6DOF);
}

Skeleton *PhysicalBone::find_skeleton_parent(Node *p_parent) {
	if (!p_parent) {
		return nullptr;
	}
	Skeleton *s = Object::cast_to<Skeleton>(p_parent);
	return s ? s : find_skeleton_parent(p_parent->get_parent());
}

void PhysicalBone::_fix_joint_offset() {
	// Clamp joint origin to bone origin
	if (parent_skeleton) {
		joint_offset.origin = body_offset.affine_inverse().origin;
	}
}

void PhysicalBone::_reload_joint() {
	if (joint.is_valid()) {
		PhysicsServer::get_singleton()->free(joint);
		joint = RID();
	}

	if (!parent_skeleton) {
		return;
	}

	PhysicalBone *body_a = parent_skeleton->get_physical_bone_parent(bone_id);
	if (!body_a) {
		return;
	}

	Transform joint_transf = get_global_transform() * joint_offset;
	Transform local_a = body_a->get_global_transform().affine_inverse() * joint_transf;
	local_a.orthonormalize();

	switch (get_joint_type()) {
		case JOINT_TYPE_PIN: {
			joint = RID_PRIME(PhysicsServer::get_singleton()->joint_create_pin(body_a->get_rid(), local_a.origin, get_rid(), joint_offset.origin));
			const PinJointData *pjd(static_cast<const PinJointData *>(joint_data));
			PhysicsServer::get_singleton()->pin_joint_set_param(joint, PhysicsServer::PIN_JOINT_BIAS, pjd->bias);
			PhysicsServer::get_singleton()->pin_joint_set_param(joint, PhysicsServer::PIN_JOINT_DAMPING, pjd->damping);
			PhysicsServer::get_singleton()->pin_joint_set_param(joint, PhysicsServer::PIN_JOINT_IMPULSE_CLAMP, pjd->impulse_clamp);

		} break;
		case JOINT_TYPE_CONE: {
			joint = RID_PRIME(PhysicsServer::get_singleton()->joint_create_cone_twist(body_a->get_rid(), local_a, get_rid(), joint_offset));
			const ConeJointData *cjd(static_cast<const ConeJointData *>(joint_data));
			PhysicsServer::get_singleton()->cone_twist_joint_set_param(joint, PhysicsServer::CONE_TWIST_JOINT_SWING_SPAN, cjd->swing_span);
			PhysicsServer::get_singleton()->cone_twist_joint_set_param(joint, PhysicsServer::CONE_TWIST_JOINT_TWIST_SPAN, cjd->twist_span);
			PhysicsServer::get_singleton()->cone_twist_joint_set_param(joint, PhysicsServer::CONE_TWIST_JOINT_BIAS, cjd->bias);
			PhysicsServer::get_singleton()->cone_twist_joint_set_param(joint, PhysicsServer::CONE_TWIST_JOINT_SOFTNESS, cjd->softness);
			PhysicsServer::get_singleton()->cone_twist_joint_set_param(joint, PhysicsServer::CONE_TWIST_JOINT_RELAXATION, cjd->relaxation);

		} break;
		case JOINT_TYPE_HINGE: {
			joint = RID_PRIME(PhysicsServer::get_singleton()->joint_create_hinge(body_a->get_rid(), local_a, get_rid(), joint_offset));
			const HingeJointData *hjd(static_cast<const HingeJointData *>(joint_data));
			PhysicsServer::get_singleton()->hinge_joint_set_flag(joint, PhysicsServer::HINGE_JOINT_FLAG_USE_LIMIT, hjd->angular_limit_enabled);
			PhysicsServer::get_singleton()->hinge_joint_set_param(joint, PhysicsServer::HINGE_JOINT_LIMIT_UPPER, hjd->angular_limit_upper);
			PhysicsServer::get_singleton()->hinge_joint_set_param(joint, PhysicsServer::HINGE_JOINT_LIMIT_LOWER, hjd->angular_limit_lower);
			PhysicsServer::get_singleton()->hinge_joint_set_param(joint, PhysicsServer::HINGE_JOINT_LIMIT_BIAS, hjd->angular_limit_bias);
			PhysicsServer::get_singleton()->hinge_joint_set_param(joint, PhysicsServer::HINGE_JOINT_LIMIT_SOFTNESS, hjd->angular_limit_softness);
			PhysicsServer::get_singleton()->hinge_joint_set_param(joint, PhysicsServer::HINGE_JOINT_LIMIT_RELAXATION, hjd->angular_limit_relaxation);

		} break;
		case JOINT_TYPE_SLIDER: {
			joint = RID_PRIME(PhysicsServer::get_singleton()->joint_create_slider(body_a->get_rid(), local_a, get_rid(), joint_offset));
			const SliderJointData *sjd(static_cast<const SliderJointData *>(joint_data));
			PhysicsServer::get_singleton()->slider_joint_set_param(joint, PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_UPPER, sjd->linear_limit_upper);
			PhysicsServer::get_singleton()->slider_joint_set_param(joint, PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_LOWER, sjd->linear_limit_lower);
			PhysicsServer::get_singleton()->slider_joint_set_param(joint, PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS, sjd->linear_limit_softness);
			PhysicsServer::get_singleton()->slider_joint_set_param(joint, PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION, sjd->linear_limit_restitution);
			PhysicsServer::get_singleton()->slider_joint_set_param(joint, PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_DAMPING, sjd->linear_limit_restitution);
			PhysicsServer::get_singleton()->slider_joint_set_param(joint, PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_UPPER, sjd->angular_limit_upper);
			PhysicsServer::get_singleton()->slider_joint_set_param(joint, PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_LOWER, sjd->angular_limit_lower);
			PhysicsServer::get_singleton()->slider_joint_set_param(joint, PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS, sjd->angular_limit_softness);
			PhysicsServer::get_singleton()->slider_joint_set_param(joint, PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS, sjd->angular_limit_softness);
			PhysicsServer::get_singleton()->slider_joint_set_param(joint, PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_DAMPING, sjd->angular_limit_damping);

		} break;
		case JOINT_TYPE_6DOF: {
			joint = RID_PRIME(PhysicsServer::get_singleton()->joint_create_generic_6dof(body_a->get_rid(), local_a, get_rid(), joint_offset));
			const SixDOFJointData *g6dofjd(static_cast<const SixDOFJointData *>(joint_data));
			for (int axis = 0; axis < 3; ++axis) {
				PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT, g6dofjd->axis_data[axis].linear_limit_enabled);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_LINEAR_UPPER_LIMIT, g6dofjd->axis_data[axis].linear_limit_upper);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_LINEAR_LOWER_LIMIT, g6dofjd->axis_data[axis].linear_limit_lower);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS, g6dofjd->axis_data[axis].linear_limit_softness);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING, g6dofjd->axis_data[axis].linear_spring_enabled);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_LINEAR_SPRING_STIFFNESS, g6dofjd->axis_data[axis].linear_spring_stiffness);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_LINEAR_SPRING_DAMPING, g6dofjd->axis_data[axis].linear_spring_damping);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_LINEAR_SPRING_EQUILIBRIUM_POINT, g6dofjd->axis_data[axis].linear_equilibrium_point);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_LINEAR_RESTITUTION, g6dofjd->axis_data[axis].linear_restitution);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_LINEAR_DAMPING, g6dofjd->axis_data[axis].linear_damping);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT, g6dofjd->axis_data[axis].angular_limit_enabled);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_ANGULAR_UPPER_LIMIT, g6dofjd->axis_data[axis].angular_limit_upper);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_ANGULAR_LOWER_LIMIT, g6dofjd->axis_data[axis].angular_limit_lower);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS, g6dofjd->axis_data[axis].angular_limit_softness);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_ANGULAR_RESTITUTION, g6dofjd->axis_data[axis].angular_restitution);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_ANGULAR_DAMPING, g6dofjd->axis_data[axis].angular_damping);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_ANGULAR_ERP, g6dofjd->axis_data[axis].erp);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING, g6dofjd->axis_data[axis].angular_spring_enabled);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_ANGULAR_SPRING_STIFFNESS, g6dofjd->axis_data[axis].angular_spring_stiffness);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_ANGULAR_SPRING_DAMPING, g6dofjd->axis_data[axis].angular_spring_damping);
				PhysicsServer::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer::G6DOF_JOINT_ANGULAR_SPRING_EQUILIBRIUM_POINT, g6dofjd->axis_data[axis].angular_equilibrium_point);
			}

		} break;
		case JOINT_TYPE_NONE: {
		} break;
	}
}

void PhysicalBone::_on_bone_parent_changed() {
	_reload_joint();
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

void PhysicalBone::set_joint_type(JointType p_joint_type) {
	if (p_joint_type == get_joint_type()) {
		return;
	}

	if (joint_data) {
		memdelete(joint_data);
	}
	joint_data = nullptr;
	switch (p_joint_type) {
		case JOINT_TYPE_PIN:
			joint_data = memnew(PinJointData);
			break;
		case JOINT_TYPE_CONE:
			joint_data = memnew(ConeJointData);
			break;
		case JOINT_TYPE_HINGE:
			joint_data = memnew(HingeJointData);
			break;
		case JOINT_TYPE_SLIDER:
			joint_data = memnew(SliderJointData);
			break;
		case JOINT_TYPE_6DOF:
			joint_data = memnew(SixDOFJointData);
			break;
		case JOINT_TYPE_NONE:
			break;
	}

	_reload_joint();

#ifdef TOOLS_ENABLED
	_change_notify();
	if (get_gizmo().is_valid()) {
		get_gizmo()->redraw();
	}
#endif
}

PhysicalBone::JointType PhysicalBone::get_joint_type() const {
	return joint_data ? joint_data->get_joint_type() : JOINT_TYPE_NONE;
}

void PhysicalBone::set_joint_offset(const Transform &p_offset) {
	joint_offset = p_offset;

	_fix_joint_offset();

	set_ignore_transform_notification(true);
	reset_to_rest_position();
	set_ignore_transform_notification(false);

#ifdef TOOLS_ENABLED
	if (get_gizmo().is_valid()) {
		get_gizmo()->redraw();
	}
#endif
}

const Transform &PhysicalBone::get_body_offset() const {
	return body_offset;
}

void PhysicalBone::set_body_offset(const Transform &p_offset) {
	body_offset = p_offset;
	body_offset_inverse = body_offset.affine_inverse();

	_fix_joint_offset();

	set_ignore_transform_notification(true);
	reset_to_rest_position();
	set_ignore_transform_notification(false);

#ifdef TOOLS_ENABLED
	if (get_gizmo().is_valid()) {
		get_gizmo()->redraw();
	}
#endif
}

const Transform &PhysicalBone::get_joint_offset() const {
	return joint_offset;
}

void PhysicalBone::set_static_body(bool p_static) {
	static_body = p_static;

	set_as_toplevel(!static_body);

	_reset_physics_simulation_state();
}

bool PhysicalBone::is_static_body() {
	return static_body;
}

void PhysicalBone::set_simulate_physics(bool p_simulate) {
	if (simulate_physics == p_simulate) {
		return;
	}

	simulate_physics = p_simulate;
	_reset_physics_simulation_state();
}

bool PhysicalBone::get_simulate_physics() {
	return simulate_physics;
}

bool PhysicalBone::is_simulating_physics() {
	return _internal_simulate_physics && !_internal_static_body;
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

void PhysicalBone::set_mass(real_t p_mass) {
	ERR_FAIL_COND(p_mass <= 0);
	mass = p_mass;
	PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_MASS, mass);
}

real_t PhysicalBone::get_mass() const {
	return mass;
}

void PhysicalBone::set_weight(real_t p_weight) {
	set_mass(p_weight / real_t(GLOBAL_DEF("physics/3d/default_gravity", 9.8)));
}

real_t PhysicalBone::get_weight() const {
	return mass * real_t(GLOBAL_DEF("physics/3d/default_gravity", 9.8));
}

void PhysicalBone::set_friction(real_t p_friction) {
	ERR_FAIL_COND(p_friction < 0 || p_friction > 1);

	friction = p_friction;
	PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_FRICTION, friction);
}

real_t PhysicalBone::get_friction() const {
	return friction;
}

void PhysicalBone::set_bounce(real_t p_bounce) {
	ERR_FAIL_COND(p_bounce < 0 || p_bounce > 1);

	bounce = p_bounce;
	PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_BOUNCE, bounce);
}

real_t PhysicalBone::get_bounce() const {
	return bounce;
}

void PhysicalBone::set_gravity_scale(real_t p_gravity_scale) {
	gravity_scale = p_gravity_scale;
	PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_GRAVITY_SCALE, gravity_scale);
}

real_t PhysicalBone::get_gravity_scale() const {
	return gravity_scale;
}

PhysicalBone::PhysicalBone() :
		PhysicsBody(PhysicsServer::BODY_MODE_STATIC),
#ifdef TOOLS_ENABLED
		gizmo_move_joint(false),
#endif
		joint_data(nullptr),
		parent_skeleton(nullptr),
		static_body(false),
		_internal_static_body(false),
		simulate_physics(false),
		_internal_simulate_physics(false),
		bone_id(-1),
		bone_name(""),
		bounce(0),
		mass(1),
		friction(1),
		gravity_scale(1) {

	set_static_body(static_body);
	_reset_physics_simulation_state();
}

PhysicalBone::~PhysicalBone() {
	if (joint_data) {
		memdelete(joint_data);
	}
}

void PhysicalBone::update_bone_id() {
	if (!parent_skeleton) {
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

		parent_skeleton->bind_physical_bone_to_bone(bone_id, this);

		_fix_joint_offset();
		_internal_static_body = !static_body; // Force staticness reset
		_reset_staticness_state();
	}
}

void PhysicalBone::update_offset() {
#ifdef TOOLS_ENABLED
	if (parent_skeleton) {
		Transform bone_transform(parent_skeleton->get_global_transform());
		if (-1 != bone_id) {
			bone_transform *= parent_skeleton->get_bone_global_pose(bone_id);
		}

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
		if (-1 == bone_id) {
			set_global_transform(parent_skeleton->get_global_transform() * body_offset);
		} else {
			set_global_transform(parent_skeleton->get_global_transform() * parent_skeleton->get_bone_global_pose(bone_id) * body_offset);
		}
	}
}

void PhysicalBone::_reset_physics_simulation_state() {
	if (simulate_physics && !static_body) {
		_start_physics_simulation();
	} else {
		_stop_physics_simulation();
	}

	_reset_staticness_state();
}

void PhysicalBone::_reset_staticness_state() {
	if (parent_skeleton && -1 != bone_id) {
		if (static_body && simulate_physics) { // With this check I'm sure the position of this body is updated only when it's necessary

			if (_internal_static_body) {
				return;
			}

			parent_skeleton->bind_child_node_to_bone(bone_id, this);
			_internal_static_body = true;
		} else {
			if (!_internal_static_body) {
				return;
			}

			parent_skeleton->unbind_child_node_from_bone(bone_id, this);
			_internal_static_body = false;
		}
	}
}

void PhysicalBone::_start_physics_simulation() {
	if (_internal_simulate_physics || !parent_skeleton) {
		return;
	}
	reset_to_rest_position();
	PhysicsServer::get_singleton()->body_set_mode(get_rid(), PhysicsServer::BODY_MODE_RIGID);
	PhysicsServer::get_singleton()->body_set_collision_layer(get_rid(), get_collision_layer());
	PhysicsServer::get_singleton()->body_set_collision_mask(get_rid(), get_collision_mask());
	PhysicsServer::get_singleton()->body_set_force_integration_callback(get_rid(), this, "_direct_state_changed");
	_internal_simulate_physics = true;
}

void PhysicalBone::_stop_physics_simulation() {
	if (!_internal_simulate_physics || !parent_skeleton) {
		return;
	}
	PhysicsServer::get_singleton()->body_set_mode(get_rid(), PhysicsServer::BODY_MODE_STATIC);
	PhysicsServer::get_singleton()->body_set_collision_layer(get_rid(), 0);
	PhysicsServer::get_singleton()->body_set_collision_mask(get_rid(), 0);
	PhysicsServer::get_singleton()->body_set_force_integration_callback(get_rid(), nullptr, "");
	parent_skeleton->set_bone_global_pose_override(bone_id, Transform(), 0.0, false);
	_internal_simulate_physics = false;
}
