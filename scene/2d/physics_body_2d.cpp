/*************************************************************************/
/*  physics_body_2d.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "scene/scene_string_names.h"

void PhysicsBody2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("move_and_collide", "linear_velocity", "test_only", "safe_margin"), &PhysicsBody2D::_move, DEFVAL(false), DEFVAL(0.08));
	ClassDB::bind_method(D_METHOD("test_move", "from", "linear_velocity", "collision", "safe_margin"), &PhysicsBody2D::test_move, DEFVAL(Variant()), DEFVAL(0.08));

	ClassDB::bind_method(D_METHOD("get_collision_exceptions"), &PhysicsBody2D::get_collision_exceptions);
	ClassDB::bind_method(D_METHOD("add_collision_exception_with", "body"), &PhysicsBody2D::add_collision_exception_with);
	ClassDB::bind_method(D_METHOD("remove_collision_exception_with", "body"), &PhysicsBody2D::remove_collision_exception_with);
}

PhysicsBody2D::PhysicsBody2D(PhysicsServer2D::BodyMode p_mode) :
		CollisionObject2D(PhysicsServer2D::get_singleton()->body_create(), false) {
	set_body_mode(p_mode);
	set_pickable(false);
}

PhysicsBody2D::~PhysicsBody2D() {
	if (motion_cache.is_valid()) {
		motion_cache->owner = nullptr;
	}
}

Ref<KinematicCollision2D> PhysicsBody2D::_move(const Vector2 &p_linear_velocity, bool p_test_only, real_t p_margin) {
	// Hack in order to work with calling from _process as well as from _physics_process; calling from thread is risky.
	double delta = Engine::get_singleton()->is_in_physics_frame() ? get_physics_process_delta_time() : get_process_delta_time();

	PhysicsServer2D::MotionParameters parameters(get_global_transform(), p_linear_velocity * delta, p_margin);

	PhysicsServer2D::MotionResult result;
	if (move_and_collide(parameters, result, p_test_only)) {
		// Create a new instance when the cached reference is invalid or still in use in script.
		if (motion_cache.is_null() || motion_cache->reference_get_count() > 1) {
			motion_cache.instantiate();
			motion_cache->owner = this;
		}

		motion_cache->result = result;
		return motion_cache;
	}

	return Ref<KinematicCollision2D>();
}

bool PhysicsBody2D::move_and_collide(const PhysicsServer2D::MotionParameters &p_parameters, PhysicsServer2D::MotionResult &r_result, bool p_test_only, bool p_cancel_sliding) {
	if (is_only_update_transform_changes_enabled()) {
		ERR_PRINT("Move functions do not work together with 'sync to physics' option. Please read the documentation.");
	}

	bool colliding = PhysicsServer2D::get_singleton()->body_test_motion(get_rid(), p_parameters, &r_result);

	// Restore direction of motion to be along original motion,
	// in order to avoid sliding due to recovery,
	// but only if collision depth is low enough to avoid tunneling.
	if (p_cancel_sliding) {
		real_t motion_length = p_parameters.motion.length();
		real_t precision = 0.001;

		if (colliding) {
			// Can't just use margin as a threshold because collision depth is calculated on unsafe motion,
			// so even in normal resting cases the depth can be a bit more than the margin.
			precision += motion_length * (r_result.collision_unsafe_fraction - r_result.collision_safe_fraction);

			if (r_result.collision_depth > p_parameters.margin + precision) {
				p_cancel_sliding = false;
			}
		}

		if (p_cancel_sliding) {
			// When motion is null, recovery is the resulting motion.
			Vector2 motion_normal;
			if (motion_length > CMP_EPSILON) {
				motion_normal = p_parameters.motion / motion_length;
			}

			// Check depth of recovery.
			real_t projected_length = r_result.travel.dot(motion_normal);
			Vector2 recovery = r_result.travel - motion_normal * projected_length;
			real_t recovery_length = recovery.length();
			// Fixes cases where canceling slide causes the motion to go too deep into the ground,
			// because we're only taking rest information into account and not general recovery.
			if (recovery_length < p_parameters.margin + precision) {
				// Apply adjustment to motion.
				r_result.travel = motion_normal * projected_length;
				r_result.remainder = p_parameters.motion - r_result.travel;
			}
		}
	}

	if (!p_test_only) {
		Transform2D gt = p_parameters.from;
		gt.elements[2] += r_result.travel;
		set_global_transform(gt);
	}

	return colliding;
}

bool PhysicsBody2D::test_move(const Transform2D &p_from, const Vector2 &p_linear_velocity, const Ref<KinematicCollision2D> &r_collision, real_t p_margin) {
	ERR_FAIL_COND_V(!is_inside_tree(), false);

	PhysicsServer2D::MotionResult *r = nullptr;
	if (r_collision.is_valid()) {
		// Needs const_cast because method bindings don't support non-const Ref.
		r = const_cast<PhysicsServer2D::MotionResult *>(&r_collision->result);
	}

	// Hack in order to work with calling from _process as well as from _physics_process; calling from thread is risky.
	double delta = Engine::get_singleton()->is_in_physics_frame() ? get_physics_process_delta_time() : get_process_delta_time();

	PhysicsServer2D::MotionParameters parameters(p_from, p_linear_velocity * delta, p_margin);

	return PhysicsServer2D::get_singleton()->body_test_motion(get_rid(), parameters, r);
}

TypedArray<PhysicsBody2D> PhysicsBody2D::get_collision_exceptions() {
	List<RID> exceptions;
	PhysicsServer2D::get_singleton()->body_get_collision_exceptions(get_rid(), &exceptions);
	Array ret;
	for (const RID &body : exceptions) {
		ObjectID instance_id = PhysicsServer2D::get_singleton()->body_get_object_instance_id(body);
		Object *obj = ObjectDB::get_instance(instance_id);
		PhysicsBody2D *physics_body = Object::cast_to<PhysicsBody2D>(obj);
		ret.append(physics_body);
	}
	return ret;
}

void PhysicsBody2D::add_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	PhysicsBody2D *physics_body = Object::cast_to<PhysicsBody2D>(p_node);
	ERR_FAIL_COND_MSG(!physics_body, "Collision exception only works between two objects of PhysicsBody2D type.");
	PhysicsServer2D::get_singleton()->body_add_collision_exception(get_rid(), physics_body->get_rid());
}

void PhysicsBody2D::remove_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	PhysicsBody2D *physics_body = Object::cast_to<PhysicsBody2D>(p_node);
	ERR_FAIL_COND_MSG(!physics_body, "Collision exception only works between two objects of PhysicsBody2D type.");
	PhysicsServer2D::get_singleton()->body_remove_collision_exception(get_rid(), physics_body->get_rid());
}

void StaticBody2D::set_constant_linear_velocity(const Vector2 &p_vel) {
	constant_linear_velocity = p_vel;

	PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_LINEAR_VELOCITY, constant_linear_velocity);
}

void StaticBody2D::set_constant_angular_velocity(real_t p_vel) {
	constant_angular_velocity = p_vel;

	PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_ANGULAR_VELOCITY, constant_angular_velocity);
}

Vector2 StaticBody2D::get_constant_linear_velocity() const {
	return constant_linear_velocity;
}

real_t StaticBody2D::get_constant_angular_velocity() const {
	return constant_angular_velocity;
}

void StaticBody2D::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
	if (physics_material_override.is_valid()) {
		if (physics_material_override->is_connected(CoreStringNames::get_singleton()->changed, callable_mp(this, &StaticBody2D::_reload_physics_characteristics))) {
			physics_material_override->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &StaticBody2D::_reload_physics_characteristics));
		}
	}

	physics_material_override = p_physics_material_override;

	if (physics_material_override.is_valid()) {
		physics_material_override->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &StaticBody2D::_reload_physics_characteristics));
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

	ClassDB::bind_method(D_METHOD("set_physics_material_override", "physics_material_override"), &StaticBody2D::set_physics_material_override);
	ClassDB::bind_method(D_METHOD("get_physics_material_override"), &StaticBody2D::get_physics_material_override);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material_override", "get_physics_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "constant_linear_velocity"), "set_constant_linear_velocity", "get_constant_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "constant_angular_velocity"), "set_constant_angular_velocity", "get_constant_angular_velocity");
}

StaticBody2D::StaticBody2D(PhysicsServer2D::BodyMode p_mode) :
		PhysicsBody2D(p_mode) {
}

void StaticBody2D::_reload_physics_characteristics() {
	if (physics_material_override.is_null()) {
		PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_BOUNCE, 0);
		PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_FRICTION, 1);
	} else {
		PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_BOUNCE, physics_material_override->computed_bounce());
		PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_FRICTION, physics_material_override->computed_friction());
	}
}

void AnimatableBody2D::set_sync_to_physics(bool p_enable) {
	if (sync_to_physics == p_enable) {
		return;
	}

	sync_to_physics = p_enable;

	_update_kinematic_motion();
}

bool AnimatableBody2D::is_sync_to_physics_enabled() const {
	return sync_to_physics;
}

void AnimatableBody2D::_update_kinematic_motion() {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
#endif

	if (sync_to_physics) {
		PhysicsServer2D::get_singleton()->body_set_state_sync_callback(get_rid(), this, _body_state_changed_callback);
		set_only_update_transform_changes(true);
		set_notify_local_transform(true);
	} else {
		PhysicsServer2D::get_singleton()->body_set_state_sync_callback(get_rid(), nullptr, nullptr);
		set_only_update_transform_changes(false);
		set_notify_local_transform(false);
	}
}

void AnimatableBody2D::_body_state_changed_callback(void *p_instance, PhysicsDirectBodyState2D *p_state) {
	AnimatableBody2D *body = (AnimatableBody2D *)p_instance;
	body->_body_state_changed(p_state);
}

void AnimatableBody2D::_body_state_changed(PhysicsDirectBodyState2D *p_state) {
	if (!sync_to_physics) {
		return;
	}

	last_valid_transform = p_state->get_transform();
	set_notify_local_transform(false);
	set_global_transform(last_valid_transform);
	set_notify_local_transform(true);
}

void AnimatableBody2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			last_valid_transform = get_global_transform();
			_update_kinematic_motion();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			set_only_update_transform_changes(false);
			set_notify_local_transform(false);
		} break;

		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			// Used by sync to physics, send the new transform to the physics...
			Transform2D new_transform = get_global_transform();

			PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_TRANSFORM, new_transform);

			// ... but then revert changes.
			set_notify_local_transform(false);
			set_global_transform(last_valid_transform);
			set_notify_local_transform(true);
		} break;
	}
}

void AnimatableBody2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sync_to_physics", "enable"), &AnimatableBody2D::set_sync_to_physics);
	ClassDB::bind_method(D_METHOD("is_sync_to_physics_enabled"), &AnimatableBody2D::is_sync_to_physics_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sync_to_physics"), "set_sync_to_physics", "is_sync_to_physics_enabled");
}

AnimatableBody2D::AnimatableBody2D() :
		StaticBody2D(PhysicsServer2D::BODY_MODE_KINEMATIC) {
}

void RigidDynamicBody2D::_body_enter_tree(ObjectID p_id) {
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

void RigidDynamicBody2D::_body_exit_tree(ObjectID p_id) {
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

void RigidDynamicBody2D::_body_inout(int p_status, const RID &p_body, ObjectID p_instance, int p_body_shape, int p_local_shape) {
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
				node->connect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &RigidDynamicBody2D::_body_enter_tree), make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &RigidDynamicBody2D::_body_exit_tree), make_binds(objid));
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

		if (E->get().shapes.is_empty()) {
			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &RigidDynamicBody2D::_body_enter_tree));
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &RigidDynamicBody2D::_body_exit_tree));
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

struct _RigidDynamicBody2DInOut {
	RID rid;
	ObjectID id;
	int shape = 0;
	int local_shape = 0;
};

void RigidDynamicBody2D::_body_state_changed_callback(void *p_instance, PhysicsDirectBodyState2D *p_state) {
	RigidDynamicBody2D *body = (RigidDynamicBody2D *)p_instance;
	body->_body_state_changed(p_state);
}

void RigidDynamicBody2D::_body_state_changed(PhysicsDirectBodyState2D *p_state) {
	set_block_transform_notify(true); // don't want notify (would feedback loop)
	if (!freeze || freeze_mode != FREEZE_MODE_KINEMATIC) {
		set_global_transform(p_state->get_transform());
	}

	linear_velocity = p_state->get_linear_velocity();
	angular_velocity = p_state->get_angular_velocity();

	if (sleeping != p_state->is_sleeping()) {
		sleeping = p_state->is_sleeping();
		emit_signal(SceneStringNames::get_singleton()->sleeping_state_changed);
	}

	GDVIRTUAL_CALL(_integrate_forces, p_state);

	set_block_transform_notify(false); // want it back

	if (contact_monitor) {
		contact_monitor->locked = true;

		//untag all
		int rc = 0;
		for (KeyValue<ObjectID, BodyState> &E : contact_monitor->body_map) {
			for (int i = 0; i < E.value.shapes.size(); i++) {
				E.value.shapes[i].tagged = false;
				rc++;
			}
		}

		_RigidDynamicBody2DInOut *toadd = (_RigidDynamicBody2DInOut *)alloca(p_state->get_contact_count() * sizeof(_RigidDynamicBody2DInOut));
		int toadd_count = 0; //state->get_contact_count();
		RigidDynamicBody2D_RemoveAction *toremove = (RigidDynamicBody2D_RemoveAction *)alloca(rc * sizeof(RigidDynamicBody2D_RemoveAction));
		int toremove_count = 0;

		//put the ones to add

		for (int i = 0; i < p_state->get_contact_count(); i++) {
			RID rid = p_state->get_contact_collider(i);
			ObjectID obj = p_state->get_contact_collider_id(i);
			int local_shape = p_state->get_contact_local_shape(i);
			int shape = p_state->get_contact_collider_shape(i);

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

		for (const KeyValue<ObjectID, BodyState> &E : contact_monitor->body_map) {
			for (int i = 0; i < E.value.shapes.size(); i++) {
				if (!E.value.shapes[i].tagged) {
					toremove[toremove_count].rid = E.value.rid;
					toremove[toremove_count].body_id = E.key;
					toremove[toremove_count].pair = E.value.shapes[i];
					toremove_count++;
				}
			}
		}

		//process removals

		for (int i = 0; i < toremove_count; i++) {
			_body_inout(0, toremove[i].rid, toremove[i].body_id, toremove[i].pair.body_shape, toremove[i].pair.local_shape);
		}

		//process additions

		for (int i = 0; i < toadd_count; i++) {
			_body_inout(1, toadd[i].rid, toadd[i].id, toadd[i].shape, toadd[i].local_shape);
		}

		contact_monitor->locked = false;
	}
}

void RigidDynamicBody2D::_apply_body_mode() {
	if (freeze) {
		switch (freeze_mode) {
			case FREEZE_MODE_STATIC: {
				set_body_mode(PhysicsServer2D::BODY_MODE_STATIC);
			} break;
			case FREEZE_MODE_KINEMATIC: {
				set_body_mode(PhysicsServer2D::BODY_MODE_KINEMATIC);
			} break;
		}
	} else if (lock_rotation) {
		set_body_mode(PhysicsServer2D::BODY_MODE_DYNAMIC_LINEAR);
	} else {
		set_body_mode(PhysicsServer2D::BODY_MODE_DYNAMIC);
	}
}

void RigidDynamicBody2D::set_lock_rotation_enabled(bool p_lock_rotation) {
	if (p_lock_rotation == lock_rotation) {
		return;
	}

	lock_rotation = p_lock_rotation;
	_apply_body_mode();
}

bool RigidDynamicBody2D::is_lock_rotation_enabled() const {
	return lock_rotation;
}

void RigidDynamicBody2D::set_freeze_enabled(bool p_freeze) {
	if (p_freeze == freeze) {
		return;
	}

	freeze = p_freeze;
	_apply_body_mode();
}

bool RigidDynamicBody2D::is_freeze_enabled() const {
	return freeze;
}

void RigidDynamicBody2D::set_freeze_mode(FreezeMode p_freeze_mode) {
	if (p_freeze_mode == freeze_mode) {
		return;
	}

	freeze_mode = p_freeze_mode;
	_apply_body_mode();
}

RigidDynamicBody2D::FreezeMode RigidDynamicBody2D::get_freeze_mode() const {
	return freeze_mode;
}

void RigidDynamicBody2D::set_mass(real_t p_mass) {
	ERR_FAIL_COND(p_mass <= 0);
	mass = p_mass;
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_MASS, mass);
}

real_t RigidDynamicBody2D::get_mass() const {
	return mass;
}

void RigidDynamicBody2D::set_inertia(real_t p_inertia) {
	ERR_FAIL_COND(p_inertia < 0);
	inertia = p_inertia;
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_INERTIA, inertia);
}

real_t RigidDynamicBody2D::get_inertia() const {
	return inertia;
}

void RigidDynamicBody2D::set_center_of_mass_mode(CenterOfMassMode p_mode) {
	if (center_of_mass_mode == p_mode) {
		return;
	}

	center_of_mass_mode = p_mode;

	switch (center_of_mass_mode) {
		case CENTER_OF_MASS_MODE_AUTO: {
			center_of_mass = Vector2();
			PhysicsServer2D::get_singleton()->body_reset_mass_properties(get_rid());
			if (inertia != 0.0) {
				PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_INERTIA, inertia);
			}
		} break;

		case CENTER_OF_MASS_MODE_CUSTOM: {
			PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_CENTER_OF_MASS, center_of_mass);
		} break;
	}
}

RigidDynamicBody2D::CenterOfMassMode RigidDynamicBody2D::get_center_of_mass_mode() const {
	return center_of_mass_mode;
}

void RigidDynamicBody2D::set_center_of_mass(const Vector2 &p_center_of_mass) {
	if (center_of_mass == p_center_of_mass) {
		return;
	}

	ERR_FAIL_COND(center_of_mass_mode != CENTER_OF_MASS_MODE_CUSTOM);
	center_of_mass = p_center_of_mass;

	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_CENTER_OF_MASS, center_of_mass);
}

const Vector2 &RigidDynamicBody2D::get_center_of_mass() const {
	return center_of_mass;
}

void RigidDynamicBody2D::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
	if (physics_material_override.is_valid()) {
		if (physics_material_override->is_connected(CoreStringNames::get_singleton()->changed, callable_mp(this, &RigidDynamicBody2D::_reload_physics_characteristics))) {
			physics_material_override->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &RigidDynamicBody2D::_reload_physics_characteristics));
		}
	}

	physics_material_override = p_physics_material_override;

	if (physics_material_override.is_valid()) {
		physics_material_override->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &RigidDynamicBody2D::_reload_physics_characteristics));
	}
	_reload_physics_characteristics();
}

Ref<PhysicsMaterial> RigidDynamicBody2D::get_physics_material_override() const {
	return physics_material_override;
}

void RigidDynamicBody2D::set_gravity_scale(real_t p_gravity_scale) {
	gravity_scale = p_gravity_scale;
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_GRAVITY_SCALE, gravity_scale);
}

real_t RigidDynamicBody2D::get_gravity_scale() const {
	return gravity_scale;
}

void RigidDynamicBody2D::set_linear_damp_mode(DampMode p_mode) {
	linear_damp_mode = p_mode;
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_LINEAR_DAMP_MODE, linear_damp_mode);
}

RigidDynamicBody2D::DampMode RigidDynamicBody2D::get_linear_damp_mode() const {
	return linear_damp_mode;
}

void RigidDynamicBody2D::set_angular_damp_mode(DampMode p_mode) {
	angular_damp_mode = p_mode;
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_ANGULAR_DAMP_MODE, angular_damp_mode);
}

RigidDynamicBody2D::DampMode RigidDynamicBody2D::get_angular_damp_mode() const {
	return angular_damp_mode;
}

void RigidDynamicBody2D::set_linear_damp(real_t p_linear_damp) {
	ERR_FAIL_COND(p_linear_damp < -1);
	linear_damp = p_linear_damp;
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_LINEAR_DAMP, linear_damp);
}

real_t RigidDynamicBody2D::get_linear_damp() const {
	return linear_damp;
}

void RigidDynamicBody2D::set_angular_damp(real_t p_angular_damp) {
	ERR_FAIL_COND(p_angular_damp < -1);
	angular_damp = p_angular_damp;
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_ANGULAR_DAMP, angular_damp);
}

real_t RigidDynamicBody2D::get_angular_damp() const {
	return angular_damp;
}

void RigidDynamicBody2D::set_axis_velocity(const Vector2 &p_axis) {
	Vector2 axis = p_axis.normalized();
	linear_velocity -= axis * axis.dot(linear_velocity);
	linear_velocity += p_axis;
	PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_LINEAR_VELOCITY, linear_velocity);
}

void RigidDynamicBody2D::set_linear_velocity(const Vector2 &p_velocity) {
	linear_velocity = p_velocity;
	PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_LINEAR_VELOCITY, linear_velocity);
}

Vector2 RigidDynamicBody2D::get_linear_velocity() const {
	return linear_velocity;
}

void RigidDynamicBody2D::set_angular_velocity(real_t p_velocity) {
	angular_velocity = p_velocity;
	PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_ANGULAR_VELOCITY, angular_velocity);
}

real_t RigidDynamicBody2D::get_angular_velocity() const {
	return angular_velocity;
}

void RigidDynamicBody2D::set_use_custom_integrator(bool p_enable) {
	if (custom_integrator == p_enable) {
		return;
	}

	custom_integrator = p_enable;
	PhysicsServer2D::get_singleton()->body_set_omit_force_integration(get_rid(), p_enable);
}

bool RigidDynamicBody2D::is_using_custom_integrator() {
	return custom_integrator;
}

void RigidDynamicBody2D::set_sleeping(bool p_sleeping) {
	sleeping = p_sleeping;
	PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_SLEEPING, sleeping);
}

void RigidDynamicBody2D::set_can_sleep(bool p_active) {
	can_sleep = p_active;
	PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_CAN_SLEEP, p_active);
}

bool RigidDynamicBody2D::is_able_to_sleep() const {
	return can_sleep;
}

bool RigidDynamicBody2D::is_sleeping() const {
	return sleeping;
}

void RigidDynamicBody2D::set_max_contacts_reported(int p_amount) {
	max_contacts_reported = p_amount;
	PhysicsServer2D::get_singleton()->body_set_max_contacts_reported(get_rid(), p_amount);
}

int RigidDynamicBody2D::get_max_contacts_reported() const {
	return max_contacts_reported;
}

void RigidDynamicBody2D::apply_central_impulse(const Vector2 &p_impulse) {
	PhysicsServer2D::get_singleton()->body_apply_central_impulse(get_rid(), p_impulse);
}

void RigidDynamicBody2D::apply_impulse(const Vector2 &p_impulse, const Vector2 &p_position) {
	PhysicsServer2D::get_singleton()->body_apply_impulse(get_rid(), p_impulse, p_position);
}

void RigidDynamicBody2D::apply_torque_impulse(real_t p_torque) {
	PhysicsServer2D::get_singleton()->body_apply_torque_impulse(get_rid(), p_torque);
}

void RigidDynamicBody2D::set_applied_force(const Vector2 &p_force) {
	PhysicsServer2D::get_singleton()->body_set_applied_force(get_rid(), p_force);
};

Vector2 RigidDynamicBody2D::get_applied_force() const {
	return PhysicsServer2D::get_singleton()->body_get_applied_force(get_rid());
};

void RigidDynamicBody2D::set_applied_torque(const real_t p_torque) {
	PhysicsServer2D::get_singleton()->body_set_applied_torque(get_rid(), p_torque);
};

real_t RigidDynamicBody2D::get_applied_torque() const {
	return PhysicsServer2D::get_singleton()->body_get_applied_torque(get_rid());
};

void RigidDynamicBody2D::add_central_force(const Vector2 &p_force) {
	PhysicsServer2D::get_singleton()->body_add_central_force(get_rid(), p_force);
}

void RigidDynamicBody2D::add_force(const Vector2 &p_force, const Vector2 &p_position) {
	PhysicsServer2D::get_singleton()->body_add_force(get_rid(), p_force, p_position);
}

void RigidDynamicBody2D::add_torque(const real_t p_torque) {
	PhysicsServer2D::get_singleton()->body_add_torque(get_rid(), p_torque);
}

void RigidDynamicBody2D::set_continuous_collision_detection_mode(CCDMode p_mode) {
	ccd_mode = p_mode;
	PhysicsServer2D::get_singleton()->body_set_continuous_collision_detection_mode(get_rid(), PhysicsServer2D::CCDMode(p_mode));
}

RigidDynamicBody2D::CCDMode RigidDynamicBody2D::get_continuous_collision_detection_mode() const {
	return ccd_mode;
}

TypedArray<Node2D> RigidDynamicBody2D::get_colliding_bodies() const {
	ERR_FAIL_COND_V(!contact_monitor, Array());

	TypedArray<Node2D> ret;
	ret.resize(contact_monitor->body_map.size());
	int idx = 0;
	for (const KeyValue<ObjectID, BodyState> &E : contact_monitor->body_map) {
		Object *obj = ObjectDB::get_instance(E.key);
		if (!obj) {
			ret.resize(ret.size() - 1); //ops
		} else {
			ret[idx++] = obj;
		}
	}

	return ret;
}

void RigidDynamicBody2D::set_contact_monitor(bool p_enabled) {
	if (p_enabled == is_contact_monitor_enabled()) {
		return;
	}

	if (!p_enabled) {
		ERR_FAIL_COND_MSG(contact_monitor->locked, "Can't disable contact monitoring during in/out callback. Use call_deferred(\"set_contact_monitor\", false) instead.");

		for (const KeyValue<ObjectID, BodyState> &E : contact_monitor->body_map) {
			//clean up mess
			Object *obj = ObjectDB::get_instance(E.key);
			Node *node = Object::cast_to<Node>(obj);

			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &RigidDynamicBody2D::_body_enter_tree));
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &RigidDynamicBody2D::_body_exit_tree));
			}
		}

		memdelete(contact_monitor);
		contact_monitor = nullptr;
	} else {
		contact_monitor = memnew(ContactMonitor);
		contact_monitor->locked = false;
	}
}

bool RigidDynamicBody2D::is_contact_monitor_enabled() const {
	return contact_monitor != nullptr;
}

void RigidDynamicBody2D::_notification(int p_what) {
#ifdef TOOLS_ENABLED
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (Engine::get_singleton()->is_editor_hint()) {
				set_notify_local_transform(true); //used for warnings and only in editor
			}
		} break;

		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			if (Engine::get_singleton()->is_editor_hint()) {
				update_configuration_warnings();
			}
		} break;
	}
#endif
}

TypedArray<String> RigidDynamicBody2D::get_configuration_warnings() const {
	Transform2D t = get_transform();

	TypedArray<String> warnings = CollisionObject2D::get_configuration_warnings();

	if (ABS(t.elements[0].length() - 1.0) > 0.05 || ABS(t.elements[1].length() - 1.0) > 0.05) {
		warnings.push_back(TTR("Size changes to RigidDynamicBody2D will be overridden by the physics engine when running.\nChange the size in children collision shapes instead."));
	}

	return warnings;
}

void RigidDynamicBody2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &RigidDynamicBody2D::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &RigidDynamicBody2D::get_mass);

	ClassDB::bind_method(D_METHOD("get_inertia"), &RigidDynamicBody2D::get_inertia);
	ClassDB::bind_method(D_METHOD("set_inertia", "inertia"), &RigidDynamicBody2D::set_inertia);

	ClassDB::bind_method(D_METHOD("set_center_of_mass_mode", "mode"), &RigidDynamicBody2D::set_center_of_mass_mode);
	ClassDB::bind_method(D_METHOD("get_center_of_mass_mode"), &RigidDynamicBody2D::get_center_of_mass_mode);

	ClassDB::bind_method(D_METHOD("set_center_of_mass", "center_of_mass"), &RigidDynamicBody2D::set_center_of_mass);
	ClassDB::bind_method(D_METHOD("get_center_of_mass"), &RigidDynamicBody2D::get_center_of_mass);

	ClassDB::bind_method(D_METHOD("set_physics_material_override", "physics_material_override"), &RigidDynamicBody2D::set_physics_material_override);
	ClassDB::bind_method(D_METHOD("get_physics_material_override"), &RigidDynamicBody2D::get_physics_material_override);

	ClassDB::bind_method(D_METHOD("set_gravity_scale", "gravity_scale"), &RigidDynamicBody2D::set_gravity_scale);
	ClassDB::bind_method(D_METHOD("get_gravity_scale"), &RigidDynamicBody2D::get_gravity_scale);

	ClassDB::bind_method(D_METHOD("set_linear_damp_mode", "linear_damp_mode"), &RigidDynamicBody2D::set_linear_damp_mode);
	ClassDB::bind_method(D_METHOD("get_linear_damp_mode"), &RigidDynamicBody2D::get_linear_damp_mode);

	ClassDB::bind_method(D_METHOD("set_angular_damp_mode", "angular_damp_mode"), &RigidDynamicBody2D::set_angular_damp_mode);
	ClassDB::bind_method(D_METHOD("get_angular_damp_mode"), &RigidDynamicBody2D::get_angular_damp_mode);

	ClassDB::bind_method(D_METHOD("set_linear_damp", "linear_damp"), &RigidDynamicBody2D::set_linear_damp);
	ClassDB::bind_method(D_METHOD("get_linear_damp"), &RigidDynamicBody2D::get_linear_damp);

	ClassDB::bind_method(D_METHOD("set_angular_damp", "angular_damp"), &RigidDynamicBody2D::set_angular_damp);
	ClassDB::bind_method(D_METHOD("get_angular_damp"), &RigidDynamicBody2D::get_angular_damp);

	ClassDB::bind_method(D_METHOD("set_linear_velocity", "linear_velocity"), &RigidDynamicBody2D::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &RigidDynamicBody2D::get_linear_velocity);

	ClassDB::bind_method(D_METHOD("set_angular_velocity", "angular_velocity"), &RigidDynamicBody2D::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &RigidDynamicBody2D::get_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_max_contacts_reported", "amount"), &RigidDynamicBody2D::set_max_contacts_reported);
	ClassDB::bind_method(D_METHOD("get_max_contacts_reported"), &RigidDynamicBody2D::get_max_contacts_reported);

	ClassDB::bind_method(D_METHOD("set_use_custom_integrator", "enable"), &RigidDynamicBody2D::set_use_custom_integrator);
	ClassDB::bind_method(D_METHOD("is_using_custom_integrator"), &RigidDynamicBody2D::is_using_custom_integrator);

	ClassDB::bind_method(D_METHOD("set_contact_monitor", "enabled"), &RigidDynamicBody2D::set_contact_monitor);
	ClassDB::bind_method(D_METHOD("is_contact_monitor_enabled"), &RigidDynamicBody2D::is_contact_monitor_enabled);

	ClassDB::bind_method(D_METHOD("set_continuous_collision_detection_mode", "mode"), &RigidDynamicBody2D::set_continuous_collision_detection_mode);
	ClassDB::bind_method(D_METHOD("get_continuous_collision_detection_mode"), &RigidDynamicBody2D::get_continuous_collision_detection_mode);

	ClassDB::bind_method(D_METHOD("set_axis_velocity", "axis_velocity"), &RigidDynamicBody2D::set_axis_velocity);
	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &RigidDynamicBody2D::apply_central_impulse, Vector2());
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "position"), &RigidDynamicBody2D::apply_impulse, Vector2());
	ClassDB::bind_method(D_METHOD("apply_torque_impulse", "torque"), &RigidDynamicBody2D::apply_torque_impulse);

	ClassDB::bind_method(D_METHOD("set_applied_force", "force"), &RigidDynamicBody2D::set_applied_force);
	ClassDB::bind_method(D_METHOD("get_applied_force"), &RigidDynamicBody2D::get_applied_force);

	ClassDB::bind_method(D_METHOD("set_applied_torque", "torque"), &RigidDynamicBody2D::set_applied_torque);
	ClassDB::bind_method(D_METHOD("get_applied_torque"), &RigidDynamicBody2D::get_applied_torque);

	ClassDB::bind_method(D_METHOD("add_central_force", "force"), &RigidDynamicBody2D::add_central_force);
	ClassDB::bind_method(D_METHOD("add_force", "force", "position"), &RigidDynamicBody2D::add_force, Vector2());
	ClassDB::bind_method(D_METHOD("add_torque", "torque"), &RigidDynamicBody2D::add_torque);

	ClassDB::bind_method(D_METHOD("set_sleeping", "sleeping"), &RigidDynamicBody2D::set_sleeping);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &RigidDynamicBody2D::is_sleeping);

	ClassDB::bind_method(D_METHOD("set_can_sleep", "able_to_sleep"), &RigidDynamicBody2D::set_can_sleep);
	ClassDB::bind_method(D_METHOD("is_able_to_sleep"), &RigidDynamicBody2D::is_able_to_sleep);

	ClassDB::bind_method(D_METHOD("set_lock_rotation_enabled", "lock_rotation"), &RigidDynamicBody2D::set_lock_rotation_enabled);
	ClassDB::bind_method(D_METHOD("is_lock_rotation_enabled"), &RigidDynamicBody2D::is_lock_rotation_enabled);

	ClassDB::bind_method(D_METHOD("set_freeze_enabled", "freeze_mode"), &RigidDynamicBody2D::set_freeze_enabled);
	ClassDB::bind_method(D_METHOD("is_freeze_enabled"), &RigidDynamicBody2D::is_freeze_enabled);

	ClassDB::bind_method(D_METHOD("set_freeze_mode", "freeze_mode"), &RigidDynamicBody2D::set_freeze_mode);
	ClassDB::bind_method(D_METHOD("get_freeze_mode"), &RigidDynamicBody2D::get_freeze_mode);

	ClassDB::bind_method(D_METHOD("get_colliding_bodies"), &RigidDynamicBody2D::get_colliding_bodies);

	GDVIRTUAL_BIND(_integrate_forces, "state");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass", PROPERTY_HINT_RANGE, "0.01,1000,0.01,or_greater,exp"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inertia", PROPERTY_HINT_RANGE, "0,1000,0.01,or_greater,exp"), "set_inertia", "get_inertia");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "center_of_mass_mode", PROPERTY_HINT_ENUM, "Auto,Custom", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_center_of_mass_mode", "get_center_of_mass_mode");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "center_of_mass", PROPERTY_HINT_RANGE, "-10,10,0.01,or_lesser,or_greater"), "set_center_of_mass", "get_center_of_mass");
	ADD_LINKED_PROPERTY("center_of_mass_mode", "center_of_mass");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material_override", "get_physics_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gravity_scale", PROPERTY_HINT_RANGE, "-128,128,0.01"), "set_gravity_scale", "get_gravity_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "custom_integrator"), "set_use_custom_integrator", "is_using_custom_integrator");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "continuous_cd", PROPERTY_HINT_ENUM, "Disabled,Cast Ray,Cast Shape"), "set_continuous_collision_detection_mode", "get_continuous_collision_detection_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "contacts_reported", PROPERTY_HINT_RANGE, "0,64,1,or_greater"), "set_max_contacts_reported", "get_max_contacts_reported");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "contact_monitor"), "set_contact_monitor", "is_contact_monitor_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleeping", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "can_sleep"), "set_can_sleep", "is_able_to_sleep");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lock_rotation"), "set_lock_rotation_enabled", "is_lock_rotation_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "freeze"), "set_freeze_enabled", "is_freeze_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "freeze_mode", PROPERTY_HINT_ENUM, "Static,Kinematic"), "set_freeze_mode", "get_freeze_mode");
	ADD_GROUP("Linear", "linear_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "linear_damp_mode", PROPERTY_HINT_ENUM, "Combine,Replace"), "set_linear_damp_mode", "get_linear_damp_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "linear_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"), "set_linear_damp", "get_linear_damp");
	ADD_GROUP("Angular", "angular_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "angular_damp_mode", PROPERTY_HINT_ENUM, "Combine,Replace"), "set_angular_damp_mode", "get_angular_damp_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"), "set_angular_damp", "get_angular_damp");
	ADD_GROUP("Applied Forces", "applied_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "applied_force"), "set_applied_force", "get_applied_force");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "applied_torque"), "set_applied_torque", "get_applied_torque");

	ADD_SIGNAL(MethodInfo("body_shape_entered", PropertyInfo(Variant::RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("body_shape_exited", PropertyInfo(Variant::RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("body_entered", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("body_exited", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("sleeping_state_changed"));

	BIND_ENUM_CONSTANT(FREEZE_MODE_STATIC);
	BIND_ENUM_CONSTANT(FREEZE_MODE_KINEMATIC);

	BIND_ENUM_CONSTANT(CENTER_OF_MASS_MODE_AUTO);
	BIND_ENUM_CONSTANT(CENTER_OF_MASS_MODE_CUSTOM);

	BIND_ENUM_CONSTANT(DAMP_MODE_COMBINE);
	BIND_ENUM_CONSTANT(DAMP_MODE_REPLACE);

	BIND_ENUM_CONSTANT(CCD_MODE_DISABLED);
	BIND_ENUM_CONSTANT(CCD_MODE_CAST_RAY);
	BIND_ENUM_CONSTANT(CCD_MODE_CAST_SHAPE);
}

void RigidDynamicBody2D::_validate_property(PropertyInfo &property) const {
	if (center_of_mass_mode != CENTER_OF_MASS_MODE_CUSTOM) {
		if (property.name == "center_of_mass") {
			property.usage = PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}
}

RigidDynamicBody2D::RigidDynamicBody2D() :
		PhysicsBody2D(PhysicsServer2D::BODY_MODE_DYNAMIC) {
	PhysicsServer2D::get_singleton()->body_set_state_sync_callback(get_rid(), this, _body_state_changed_callback);
}

RigidDynamicBody2D::~RigidDynamicBody2D() {
	if (contact_monitor) {
		memdelete(contact_monitor);
	}
}

void RigidDynamicBody2D::_reload_physics_characteristics() {
	if (physics_material_override.is_null()) {
		PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_BOUNCE, 0);
		PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_FRICTION, 1);
	} else {
		PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_BOUNCE, physics_material_override->computed_bounce());
		PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_FRICTION, physics_material_override->computed_friction());
	}
}

//////////////////////////

// So, if you pass 45 as limit, avoid numerical precision errors when angle is 45.
#define FLOOR_ANGLE_THRESHOLD 0.01

bool CharacterBody2D::move_and_slide() {
	// Hack in order to work with calling from _process as well as from _physics_process; calling from thread is risky.
	double delta = Engine::get_singleton()->is_in_physics_frame() ? get_physics_process_delta_time() : get_process_delta_time();

	Vector2 current_platform_velocity = platform_velocity;
	Transform2D gt = get_global_transform();
	previous_position = gt.elements[2];

	if ((on_floor || on_wall) && platform_rid.is_valid()) {
		bool excluded = false;
		if (on_floor) {
			excluded = (moving_platform_floor_layers & platform_layer) == 0;
		} else if (on_wall) {
			excluded = (moving_platform_wall_layers & platform_layer) == 0;
		}
		if (!excluded) {
			//this approach makes sure there is less delay between the actual body velocity and the one we saved
			PhysicsDirectBodyState2D *bs = PhysicsServer2D::get_singleton()->body_get_direct_state(platform_rid);
			if (bs) {
				Vector2 local_position = gt.elements[2] - bs->get_transform().elements[2];
				current_platform_velocity = bs->get_velocity_at_local_position(local_position);
			}
		} else {
			current_platform_velocity = Vector2();
		}
	}

	motion_results.clear();
	last_motion = Vector2();

	bool was_on_floor = on_floor;
	on_floor = false;
	on_ceiling = false;
	on_wall = false;

	if (!current_platform_velocity.is_equal_approx(Vector2())) {
		PhysicsServer2D::MotionParameters parameters(get_global_transform(), current_platform_velocity * delta, margin);
		parameters.exclude_bodies.insert(platform_rid);
		if (platform_object_id.is_valid()) {
			parameters.exclude_objects.insert(platform_object_id);
		}

		PhysicsServer2D::MotionResult floor_result;
		if (move_and_collide(parameters, floor_result, false, false)) {
			motion_results.push_back(floor_result);
			_set_collision_direction(floor_result);
		}
	}

	if (motion_mode == MOTION_MODE_GROUNDED) {
		_move_and_slide_grounded(delta, was_on_floor);
	} else {
		_move_and_slide_free(delta);
	}

	// Compute real velocity.
	real_velocity = get_position_delta() / delta;

	if (moving_platform_apply_velocity_on_leave != PLATFORM_VEL_ON_LEAVE_NEVER) {
		// Add last platform velocity when just left a moving platform.
		if (!on_floor && !on_wall) {
			if (moving_platform_apply_velocity_on_leave == PLATFORM_VEL_ON_LEAVE_UPWARD_ONLY && current_platform_velocity.dot(up_direction) < 0) {
				current_platform_velocity = current_platform_velocity.slide(up_direction);
			}
			motion_velocity += current_platform_velocity;
		}
	}

	return motion_results.size() > 0;
}

void CharacterBody2D::_move_and_slide_grounded(double p_delta, bool p_was_on_floor) {
	Vector2 motion = motion_velocity * p_delta;
	Vector2 motion_slide_up = motion.slide(up_direction);

	Vector2 prev_floor_normal = floor_normal;

	platform_rid = RID();
	platform_object_id = ObjectID();
	floor_normal = Vector2();
	platform_velocity = Vector2();

	// No sliding on first attempt to keep floor motion stable when possible,
	// When stop on slope is enabled or when there is no up direction.
	bool sliding_enabled = !floor_stop_on_slope;
	// Constant speed can be applied only the first time sliding is enabled.
	bool can_apply_constant_speed = sliding_enabled;
	// If the platform's ceiling push down the body.
	bool apply_ceiling_velocity = false;
	bool first_slide = true;
	bool vel_dir_facing_up = motion_velocity.dot(up_direction) > 0;
	Vector2 last_travel;

	for (int iteration = 0; iteration < max_slides; ++iteration) {
		PhysicsServer2D::MotionParameters parameters(get_global_transform(), motion, margin);

		Vector2 prev_position = parameters.from.elements[2];

		PhysicsServer2D::MotionResult result;
		bool collided = move_and_collide(parameters, result, false, !sliding_enabled);

		last_motion = result.travel;

		if (collided) {
			motion_results.push_back(result);
			_set_collision_direction(result);

			// If we hit a ceiling platform, we set the vertical motion_velocity to at least the platform one.
			if (on_ceiling && result.collider_velocity != Vector2() && result.collider_velocity.dot(up_direction) < 0) {
				// If ceiling sliding is on, only apply when the ceiling is flat or when the motion is upward.
				if (!slide_on_ceiling || motion.dot(up_direction) < 0 || (result.collision_normal + up_direction).length() < 0.01) {
					apply_ceiling_velocity = true;
					Vector2 ceiling_vertical_velocity = up_direction * up_direction.dot(result.collider_velocity);
					Vector2 motion_vertical_velocity = up_direction * up_direction.dot(motion_velocity);
					if (motion_vertical_velocity.dot(up_direction) > 0 || ceiling_vertical_velocity.length_squared() > motion_vertical_velocity.length_squared()) {
						motion_velocity = ceiling_vertical_velocity + motion_velocity.slide(up_direction);
					}
				}
			}

			if (on_floor && floor_stop_on_slope && (motion_velocity.normalized() + up_direction).length() < 0.01) {
				Transform2D gt = get_global_transform();
				if (result.travel.length() <= margin + CMP_EPSILON) {
					gt.elements[2] -= result.travel;
				}
				set_global_transform(gt);
				motion_velocity = Vector2();
				last_motion = Vector2();
				motion = Vector2();
				break;
			}

			if (result.remainder.is_equal_approx(Vector2())) {
				motion = Vector2();
				break;
			}

			// Move on floor only checks.
			if (floor_block_on_wall && on_wall && motion_slide_up.dot(result.collision_normal) <= 0) {
				// Avoid to move forward on a wall if floor_block_on_wall is true.
				if (p_was_on_floor && !on_floor && !vel_dir_facing_up) {
					// If the movement is large the body can be prevented from reaching the walls.
					if (result.travel.length() <= margin + CMP_EPSILON) {
						// Cancels the motion.
						Transform2D gt = get_global_transform();
						gt.elements[2] -= result.travel;
						set_global_transform(gt);
					}
					// Determines if you are on the ground.
					_snap_on_floor(true, false);
					motion_velocity = Vector2();
					last_motion = Vector2();
					motion = Vector2();
					break;
				}
				// Prevents the body from being able to climb a slope when it moves forward against the wall.
				else if (!on_floor) {
					motion = up_direction * up_direction.dot(result.remainder);
					motion = motion.slide(result.collision_normal);
				} else {
					motion = result.remainder;
				}
			}
			// Constant Speed when the slope is upward.
			else if (floor_constant_speed && is_on_floor_only() && can_apply_constant_speed && p_was_on_floor && motion.dot(result.collision_normal) < 0) {
				can_apply_constant_speed = false;
				Vector2 motion_slide_norm = result.remainder.slide(result.collision_normal).normalized();
				motion = motion_slide_norm * (motion_slide_up.length() - result.travel.slide(up_direction).length() - last_travel.slide(up_direction).length());
			}
			// Regular sliding, the last part of the test handle the case when you don't want to slide on the ceiling.
			else if ((sliding_enabled || !on_floor) && (!on_ceiling || slide_on_ceiling || !vel_dir_facing_up) && !apply_ceiling_velocity) {
				Vector2 slide_motion = result.remainder.slide(result.collision_normal);
				if (slide_motion.dot(motion_velocity) > 0.0) {
					motion = slide_motion;
				} else {
					motion = Vector2();
				}
				if (slide_on_ceiling && on_ceiling) {
					// Apply slide only in the direction of the input motion, otherwise just stop to avoid jittering when moving against a wall.
					if (vel_dir_facing_up) {
						motion_velocity = motion_velocity.slide(result.collision_normal);
					} else {
						// Avoid acceleration in slope when falling.
						motion_velocity = up_direction * up_direction.dot(motion_velocity);
					}
				}
			}
			// No sliding on first attempt to keep floor motion stable when possible.
			else {
				motion = result.remainder;
				if (on_ceiling && !slide_on_ceiling && vel_dir_facing_up) {
					motion_velocity = motion_velocity.slide(up_direction);
					motion = motion.slide(up_direction);
				}
			}

			last_travel = result.travel;
		}
		// When you move forward in a downward slope you don’t collide because you will be in the air.
		// This test ensures that constant speed is applied, only if the player is still on the ground after the snap is applied.
		else if (floor_constant_speed && first_slide && _on_floor_if_snapped(p_was_on_floor, vel_dir_facing_up)) {
			can_apply_constant_speed = false;
			sliding_enabled = true;
			Transform2D gt = get_global_transform();
			gt.elements[2] = prev_position;
			set_global_transform(gt);

			Vector2 motion_slide_norm = motion.slide(prev_floor_normal).normalized();
			motion = motion_slide_norm * (motion_slide_up.length());
			collided = true;
		}

		can_apply_constant_speed = !can_apply_constant_speed && !sliding_enabled;
		sliding_enabled = true;
		first_slide = false;

		if (!collided || motion.is_equal_approx(Vector2())) {
			break;
		}
	}

	_snap_on_floor(p_was_on_floor, vel_dir_facing_up);

	// Reset the gravity accumulation when touching the ground.
	if (on_floor && !vel_dir_facing_up) {
		motion_velocity = motion_velocity.slide(up_direction);
	}
}

void CharacterBody2D::_move_and_slide_free(double p_delta) {
	Vector2 motion = motion_velocity * p_delta;

	platform_rid = RID();
	platform_object_id = ObjectID();
	floor_normal = Vector2();
	platform_velocity = Vector2();

	bool first_slide = true;
	for (int iteration = 0; iteration < max_slides; ++iteration) {
		PhysicsServer2D::MotionParameters parameters(get_global_transform(), motion, margin);

		PhysicsServer2D::MotionResult result;
		bool collided = move_and_collide(parameters, result, false, false);

		last_motion = result.travel;

		if (collided) {
			motion_results.push_back(result);
			_set_collision_direction(result);

			if (result.remainder.is_equal_approx(Vector2())) {
				motion = Vector2();
				break;
			}

			if (free_mode_min_slide_angle != 0 && result.get_angle(-motion_velocity.normalized()) < free_mode_min_slide_angle + FLOOR_ANGLE_THRESHOLD) {
				motion = Vector2();
			} else if (first_slide) {
				Vector2 motion_slide_norm = result.remainder.slide(result.collision_normal).normalized();
				motion = motion_slide_norm * (motion.length() - result.travel.length());
			} else {
				motion = result.remainder.slide(result.collision_normal);
			}

			if (motion.dot(motion_velocity) <= 0.0) {
				motion = Vector2();
			}
		}

		if (!collided || motion.is_equal_approx(Vector2())) {
			break;
		}

		first_slide = false;
	}
}

void CharacterBody2D::_snap_on_floor(bool was_on_floor, bool vel_dir_facing_up) {
	if (on_floor || !was_on_floor || vel_dir_facing_up) {
		return;
	}

	// Snap by at least collision margin to keep floor state consistent.
	real_t length = MAX(floor_snap_length, margin);

	PhysicsServer2D::MotionParameters parameters(get_global_transform(), -up_direction * length, margin);
	parameters.collide_separation_ray = true;

	PhysicsServer2D::MotionResult result;
	if (move_and_collide(parameters, result, true, false)) {
		if (result.get_angle(up_direction) <= floor_max_angle + FLOOR_ANGLE_THRESHOLD) {
			on_floor = true;
			floor_normal = result.collision_normal;
			_set_platform_data(result);

			if (floor_stop_on_slope) {
				// move and collide may stray the object a bit because of pre un-stucking,
				// so only ensure that motion happens on floor direction in this case.
				if (result.travel.length() > margin) {
					result.travel = up_direction * up_direction.dot(result.travel);
				} else {
					result.travel = Vector2();
				}
			}

			parameters.from.elements[2] += result.travel;
			set_global_transform(parameters.from);
		}
	}
}

bool CharacterBody2D::_on_floor_if_snapped(bool was_on_floor, bool vel_dir_facing_up) {
	if (up_direction == Vector2() || on_floor || !was_on_floor || vel_dir_facing_up) {
		return false;
	}

	// Snap by at least collision margin to keep floor state consistent.
	real_t length = MAX(floor_snap_length, margin);

	PhysicsServer2D::MotionParameters parameters(get_global_transform(), -up_direction * length, margin);
	parameters.collide_separation_ray = true;

	PhysicsServer2D::MotionResult result;
	if (move_and_collide(parameters, result, true, false)) {
		if (result.get_angle(up_direction) <= floor_max_angle + FLOOR_ANGLE_THRESHOLD) {
			return true;
		}
	}

	return false;
}

void CharacterBody2D::_set_collision_direction(const PhysicsServer2D::MotionResult &p_result) {
	if (motion_mode == MOTION_MODE_GROUNDED && p_result.get_angle(up_direction) <= floor_max_angle + FLOOR_ANGLE_THRESHOLD) { //floor
		on_floor = true;
		floor_normal = p_result.collision_normal;
		_set_platform_data(p_result);
	} else if (motion_mode == MOTION_MODE_GROUNDED && p_result.get_angle(-up_direction) <= floor_max_angle + FLOOR_ANGLE_THRESHOLD) { //ceiling
		on_ceiling = true;
	} else {
		on_wall = true;
		wall_normal = p_result.collision_normal;
		// Don't apply wall velocity when the collider is a CharacterBody2D.
		if (Object::cast_to<CharacterBody2D>(ObjectDB::get_instance(p_result.collider_id)) == nullptr) {
			_set_platform_data(p_result);
		}
	}
}

void CharacterBody2D::_set_platform_data(const PhysicsServer2D::MotionResult &p_result) {
	platform_rid = p_result.collider;
	platform_object_id = p_result.collider_id;
	platform_velocity = p_result.collider_velocity;
	platform_layer = PhysicsServer2D::get_singleton()->body_get_collision_layer(platform_rid);
}

const Vector2 &CharacterBody2D::get_motion_velocity() const {
	return motion_velocity;
}

void CharacterBody2D::set_motion_velocity(const Vector2 &p_velocity) {
	motion_velocity = p_velocity;
}

bool CharacterBody2D::is_on_floor() const {
	return on_floor;
}

bool CharacterBody2D::is_on_floor_only() const {
	return on_floor && !on_wall && !on_ceiling;
}

bool CharacterBody2D::is_on_wall() const {
	return on_wall;
}

bool CharacterBody2D::is_on_wall_only() const {
	return on_wall && !on_floor && !on_ceiling;
}

bool CharacterBody2D::is_on_ceiling() const {
	return on_ceiling;
}

bool CharacterBody2D::is_on_ceiling_only() const {
	return on_ceiling && !on_floor && !on_wall;
}

const Vector2 &CharacterBody2D::get_floor_normal() const {
	return floor_normal;
}

const Vector2 &CharacterBody2D::get_wall_normal() const {
	return wall_normal;
}

const Vector2 &CharacterBody2D::get_last_motion() const {
	return last_motion;
}

Vector2 CharacterBody2D::get_position_delta() const {
	return get_global_transform().elements[2] - previous_position;
}

const Vector2 &CharacterBody2D::get_real_velocity() const {
	return real_velocity;
}

real_t CharacterBody2D::get_floor_angle(const Vector2 &p_up_direction) const {
	ERR_FAIL_COND_V(p_up_direction == Vector2(), 0);
	return Math::acos(floor_normal.dot(p_up_direction));
}

const Vector2 &CharacterBody2D::get_platform_velocity() const {
	return platform_velocity;
}

int CharacterBody2D::get_slide_collision_count() const {
	return motion_results.size();
}

PhysicsServer2D::MotionResult CharacterBody2D::get_slide_collision(int p_bounce) const {
	ERR_FAIL_INDEX_V(p_bounce, motion_results.size(), PhysicsServer2D::MotionResult());
	return motion_results[p_bounce];
}

Ref<KinematicCollision2D> CharacterBody2D::_get_slide_collision(int p_bounce) {
	ERR_FAIL_INDEX_V(p_bounce, motion_results.size(), Ref<KinematicCollision2D>());
	if (p_bounce >= slide_colliders.size()) {
		slide_colliders.resize(p_bounce + 1);
	}

	// Create a new instance when the cached reference is invalid or still in use in script.
	if (slide_colliders[p_bounce].is_null() || slide_colliders[p_bounce]->reference_get_count() > 1) {
		slide_colliders.write[p_bounce].instantiate();
		slide_colliders.write[p_bounce]->owner = this;
	}

	slide_colliders.write[p_bounce]->result = motion_results[p_bounce];
	return slide_colliders[p_bounce];
}

Ref<KinematicCollision2D> CharacterBody2D::_get_last_slide_collision() {
	if (motion_results.size() == 0) {
		return Ref<KinematicCollision2D>();
	}
	return _get_slide_collision(motion_results.size() - 1);
}

void CharacterBody2D::set_safe_margin(real_t p_margin) {
	margin = p_margin;
}

real_t CharacterBody2D::get_safe_margin() const {
	return margin;
}

bool CharacterBody2D::is_floor_stop_on_slope_enabled() const {
	return floor_stop_on_slope;
}

void CharacterBody2D::set_floor_stop_on_slope_enabled(bool p_enabled) {
	floor_stop_on_slope = p_enabled;
}

bool CharacterBody2D::is_floor_constant_speed_enabled() const {
	return floor_constant_speed;
}

void CharacterBody2D::set_floor_constant_speed_enabled(bool p_enabled) {
	floor_constant_speed = p_enabled;
}

bool CharacterBody2D::is_floor_block_on_wall_enabled() const {
	return floor_block_on_wall;
}

void CharacterBody2D::set_floor_block_on_wall_enabled(bool p_enabled) {
	floor_block_on_wall = p_enabled;
}

bool CharacterBody2D::is_slide_on_ceiling_enabled() const {
	return slide_on_ceiling;
}

void CharacterBody2D::set_slide_on_ceiling_enabled(bool p_enabled) {
	slide_on_ceiling = p_enabled;
}

uint32_t CharacterBody2D::get_moving_platform_floor_layers() const {
	return moving_platform_floor_layers;
}

void CharacterBody2D::set_moving_platform_floor_layers(uint32_t p_exclude_layers) {
	moving_platform_floor_layers = p_exclude_layers;
}

uint32_t CharacterBody2D::get_moving_platform_wall_layers() const {
	return moving_platform_wall_layers;
}

void CharacterBody2D::set_moving_platform_wall_layers(uint32_t p_exclude_layers) {
	moving_platform_wall_layers = p_exclude_layers;
}

void CharacterBody2D::set_motion_mode(MotionMode p_mode) {
	motion_mode = p_mode;
}

CharacterBody2D::MotionMode CharacterBody2D::get_motion_mode() const {
	return motion_mode;
}

void CharacterBody2D::set_moving_platform_apply_velocity_on_leave(MovingPlatformApplyVelocityOnLeave p_on_leave_apply_velocity) {
	moving_platform_apply_velocity_on_leave = p_on_leave_apply_velocity;
}

CharacterBody2D::MovingPlatformApplyVelocityOnLeave CharacterBody2D::get_moving_platform_apply_velocity_on_leave() const {
	return moving_platform_apply_velocity_on_leave;
}

int CharacterBody2D::get_max_slides() const {
	return max_slides;
}

void CharacterBody2D::set_max_slides(int p_max_slides) {
	ERR_FAIL_COND(p_max_slides < 1);
	max_slides = p_max_slides;
}

real_t CharacterBody2D::get_floor_max_angle() const {
	return floor_max_angle;
}

void CharacterBody2D::set_floor_max_angle(real_t p_radians) {
	floor_max_angle = p_radians;
}

real_t CharacterBody2D::get_floor_snap_length() {
	return floor_snap_length;
}

void CharacterBody2D::set_floor_snap_length(real_t p_floor_snap_length) {
	ERR_FAIL_COND(p_floor_snap_length < 0);
	floor_snap_length = p_floor_snap_length;
}

real_t CharacterBody2D::get_free_mode_min_slide_angle() const {
	return free_mode_min_slide_angle;
}

void CharacterBody2D::set_free_mode_min_slide_angle(real_t p_radians) {
	free_mode_min_slide_angle = p_radians;
}

const Vector2 &CharacterBody2D::get_up_direction() const {
	return up_direction;
}

void CharacterBody2D::set_up_direction(const Vector2 &p_up_direction) {
	ERR_FAIL_COND_MSG(p_up_direction == Vector2(), "up_direction can't be equal to Vector2.ZERO, consider using Free motion mode instead.");
	up_direction = p_up_direction.normalized();
}

void CharacterBody2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// Reset move_and_slide() data.
			on_floor = false;
			platform_rid = RID();
			platform_object_id = ObjectID();
			on_ceiling = false;
			on_wall = false;
			motion_results.clear();
			platform_velocity = Vector2();
		} break;
	}
}

void CharacterBody2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("move_and_slide"), &CharacterBody2D::move_and_slide);

	ClassDB::bind_method(D_METHOD("set_motion_velocity", "motion_velocity"), &CharacterBody2D::set_motion_velocity);
	ClassDB::bind_method(D_METHOD("get_motion_velocity"), &CharacterBody2D::get_motion_velocity);

	ClassDB::bind_method(D_METHOD("set_safe_margin", "pixels"), &CharacterBody2D::set_safe_margin);
	ClassDB::bind_method(D_METHOD("get_safe_margin"), &CharacterBody2D::get_safe_margin);
	ClassDB::bind_method(D_METHOD("is_floor_stop_on_slope_enabled"), &CharacterBody2D::is_floor_stop_on_slope_enabled);
	ClassDB::bind_method(D_METHOD("set_floor_stop_on_slope_enabled", "enabled"), &CharacterBody2D::set_floor_stop_on_slope_enabled);
	ClassDB::bind_method(D_METHOD("set_floor_constant_speed_enabled", "enabled"), &CharacterBody2D::set_floor_constant_speed_enabled);
	ClassDB::bind_method(D_METHOD("is_floor_constant_speed_enabled"), &CharacterBody2D::is_floor_constant_speed_enabled);
	ClassDB::bind_method(D_METHOD("set_floor_block_on_wall_enabled", "enabled"), &CharacterBody2D::set_floor_block_on_wall_enabled);
	ClassDB::bind_method(D_METHOD("is_floor_block_on_wall_enabled"), &CharacterBody2D::is_floor_block_on_wall_enabled);
	ClassDB::bind_method(D_METHOD("set_slide_on_ceiling_enabled", "enabled"), &CharacterBody2D::set_slide_on_ceiling_enabled);
	ClassDB::bind_method(D_METHOD("is_slide_on_ceiling_enabled"), &CharacterBody2D::is_slide_on_ceiling_enabled);

	ClassDB::bind_method(D_METHOD("set_moving_platform_floor_layers", "exclude_layer"), &CharacterBody2D::set_moving_platform_floor_layers);
	ClassDB::bind_method(D_METHOD("get_moving_platform_floor_layers"), &CharacterBody2D::get_moving_platform_floor_layers);
	ClassDB::bind_method(D_METHOD("set_moving_platform_wall_layers", "exclude_layer"), &CharacterBody2D::set_moving_platform_wall_layers);
	ClassDB::bind_method(D_METHOD("get_moving_platform_wall_layers"), &CharacterBody2D::get_moving_platform_wall_layers);

	ClassDB::bind_method(D_METHOD("get_max_slides"), &CharacterBody2D::get_max_slides);
	ClassDB::bind_method(D_METHOD("set_max_slides", "max_slides"), &CharacterBody2D::set_max_slides);
	ClassDB::bind_method(D_METHOD("get_floor_max_angle"), &CharacterBody2D::get_floor_max_angle);
	ClassDB::bind_method(D_METHOD("set_floor_max_angle", "radians"), &CharacterBody2D::set_floor_max_angle);
	ClassDB::bind_method(D_METHOD("get_floor_snap_length"), &CharacterBody2D::get_floor_snap_length);
	ClassDB::bind_method(D_METHOD("set_floor_snap_length", "floor_snap_length"), &CharacterBody2D::set_floor_snap_length);
	ClassDB::bind_method(D_METHOD("get_free_mode_min_slide_angle"), &CharacterBody2D::get_free_mode_min_slide_angle);
	ClassDB::bind_method(D_METHOD("set_free_mode_min_slide_angle", "radians"), &CharacterBody2D::set_free_mode_min_slide_angle);
	ClassDB::bind_method(D_METHOD("get_up_direction"), &CharacterBody2D::get_up_direction);
	ClassDB::bind_method(D_METHOD("set_up_direction", "up_direction"), &CharacterBody2D::set_up_direction);
	ClassDB::bind_method(D_METHOD("set_motion_mode", "mode"), &CharacterBody2D::set_motion_mode);
	ClassDB::bind_method(D_METHOD("get_motion_mode"), &CharacterBody2D::get_motion_mode);
	ClassDB::bind_method(D_METHOD("set_moving_platform_apply_velocity_on_leave", "on_leave_apply_velocity"), &CharacterBody2D::set_moving_platform_apply_velocity_on_leave);
	ClassDB::bind_method(D_METHOD("get_moving_platform_apply_velocity_on_leave"), &CharacterBody2D::get_moving_platform_apply_velocity_on_leave);

	ClassDB::bind_method(D_METHOD("is_on_floor"), &CharacterBody2D::is_on_floor);
	ClassDB::bind_method(D_METHOD("is_on_floor_only"), &CharacterBody2D::is_on_floor_only);
	ClassDB::bind_method(D_METHOD("is_on_ceiling"), &CharacterBody2D::is_on_ceiling);
	ClassDB::bind_method(D_METHOD("is_on_ceiling_only"), &CharacterBody2D::is_on_ceiling_only);
	ClassDB::bind_method(D_METHOD("is_on_wall"), &CharacterBody2D::is_on_wall);
	ClassDB::bind_method(D_METHOD("is_on_wall_only"), &CharacterBody2D::is_on_wall_only);
	ClassDB::bind_method(D_METHOD("get_floor_normal"), &CharacterBody2D::get_floor_normal);
	ClassDB::bind_method(D_METHOD("get_wall_normal"), &CharacterBody2D::get_wall_normal);
	ClassDB::bind_method(D_METHOD("get_last_motion"), &CharacterBody2D::get_last_motion);
	ClassDB::bind_method(D_METHOD("get_position_delta"), &CharacterBody2D::get_position_delta);
	ClassDB::bind_method(D_METHOD("get_real_velocity"), &CharacterBody2D::get_real_velocity);
	ClassDB::bind_method(D_METHOD("get_floor_angle", "up_direction"), &CharacterBody2D::get_floor_angle, DEFVAL(Vector2(0.0, -1.0)));
	ClassDB::bind_method(D_METHOD("get_platform_velocity"), &CharacterBody2D::get_platform_velocity);
	ClassDB::bind_method(D_METHOD("get_slide_collision_count"), &CharacterBody2D::get_slide_collision_count);
	ClassDB::bind_method(D_METHOD("get_slide_collision", "slide_idx"), &CharacterBody2D::_get_slide_collision);
	ClassDB::bind_method(D_METHOD("get_last_slide_collision"), &CharacterBody2D::_get_last_slide_collision);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "motion_mode", PROPERTY_HINT_ENUM, "Grounded,Free", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_motion_mode", "get_motion_mode");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "up_direction"), "set_up_direction", "get_up_direction");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "motion_velocity", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_motion_velocity", "get_motion_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "slide_on_ceiling"), "set_slide_on_ceiling_enabled", "is_slide_on_ceiling_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_slides", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_max_slides", "get_max_slides");

	ADD_GROUP("Free Mode", "free_mode_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "free_mode_min_slide_angle", PROPERTY_HINT_RANGE, "0,180,0.1,radians", PROPERTY_USAGE_DEFAULT), "set_free_mode_min_slide_angle", "get_free_mode_min_slide_angle");
	ADD_GROUP("Floor", "floor_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "floor_stop_on_slope"), "set_floor_stop_on_slope_enabled", "is_floor_stop_on_slope_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "floor_constant_speed"), "set_floor_constant_speed_enabled", "is_floor_constant_speed_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "floor_block_on_wall"), "set_floor_block_on_wall_enabled", "is_floor_block_on_wall_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "floor_max_angle", PROPERTY_HINT_RANGE, "0,180,0.1,radians"), "set_floor_max_angle", "get_floor_max_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "floor_snap_length", PROPERTY_HINT_RANGE, "0,32,0.1,or_greater"), "set_floor_snap_length", "get_floor_snap_length");
	ADD_GROUP("Moving platform", "moving_platform");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "moving_platform_apply_velocity_on_leave", PROPERTY_HINT_ENUM, "Always,Upward Only,Never", PROPERTY_USAGE_DEFAULT), "set_moving_platform_apply_velocity_on_leave", "get_moving_platform_apply_velocity_on_leave");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "moving_platform_floor_layers", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_moving_platform_floor_layers", "get_moving_platform_floor_layers");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "moving_platform_wall_layers", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_moving_platform_wall_layers", "get_moving_platform_wall_layers");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision/safe_margin", PROPERTY_HINT_RANGE, "0.001,256,0.001"), "set_safe_margin", "get_safe_margin");

	BIND_ENUM_CONSTANT(MOTION_MODE_GROUNDED);
	BIND_ENUM_CONSTANT(MOTION_MODE_FREE);

	BIND_ENUM_CONSTANT(PLATFORM_VEL_ON_LEAVE_ALWAYS);
	BIND_ENUM_CONSTANT(PLATFORM_VEL_ON_LEAVE_UPWARD_ONLY);
	BIND_ENUM_CONSTANT(PLATFORM_VEL_ON_LEAVE_NEVER);
}

void CharacterBody2D::_validate_property(PropertyInfo &property) const {
	if (motion_mode == MOTION_MODE_FREE) {
		if (property.name.begins_with("floor_") || property.name == "up_direction" || property.name == "slide_on_ceiling") {
			property.usage = PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL;
		}
	} else {
		if (property.name == "free_mode_min_slide_angle") {
			property.usage = PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}
}

CharacterBody2D::CharacterBody2D() :
		PhysicsBody2D(PhysicsServer2D::BODY_MODE_KINEMATIC) {
}

CharacterBody2D::~CharacterBody2D() {
	for (int i = 0; i < slide_colliders.size(); i++) {
		if (slide_colliders[i].is_valid()) {
			slide_colliders.write[i]->owner = nullptr;
		}
	}
}

////////////////////////

Vector2 KinematicCollision2D::get_position() const {
	return result.collision_point;
}

Vector2 KinematicCollision2D::get_normal() const {
	return result.collision_normal;
}

Vector2 KinematicCollision2D::get_travel() const {
	return result.travel;
}

Vector2 KinematicCollision2D::get_remainder() const {
	return result.remainder;
}

real_t KinematicCollision2D::get_angle(const Vector2 &p_up_direction) const {
	ERR_FAIL_COND_V(p_up_direction == Vector2(), 0);
	return result.get_angle(p_up_direction);
}

Object *KinematicCollision2D::get_local_shape() const {
	if (!owner) {
		return nullptr;
	}
	uint32_t ownerid = owner->shape_find_owner(result.collision_local_shape);
	return owner->shape_owner_get_owner(ownerid);
}

Object *KinematicCollision2D::get_collider() const {
	if (result.collider_id.is_valid()) {
		return ObjectDB::get_instance(result.collider_id);
	}

	return nullptr;
}

ObjectID KinematicCollision2D::get_collider_id() const {
	return result.collider_id;
}

RID KinematicCollision2D::get_collider_rid() const {
	return result.collider;
}

Object *KinematicCollision2D::get_collider_shape() const {
	Object *collider = get_collider();
	if (collider) {
		CollisionObject2D *obj2d = Object::cast_to<CollisionObject2D>(collider);
		if (obj2d) {
			uint32_t ownerid = obj2d->shape_find_owner(result.collider_shape);
			return obj2d->shape_owner_get_owner(ownerid);
		}
	}

	return nullptr;
}

int KinematicCollision2D::get_collider_shape_index() const {
	return result.collider_shape;
}

Vector2 KinematicCollision2D::get_collider_velocity() const {
	return result.collider_velocity;
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
}
