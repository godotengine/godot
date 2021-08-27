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
	ClassDB::bind_method(D_METHOD("move_and_collide", "rel_vec", "test_only", "safe_margin"), &PhysicsBody2D::_move, DEFVAL(false), DEFVAL(0.08));
	ClassDB::bind_method(D_METHOD("test_move", "from", "rel_vec", "collision", "safe_margin"), &PhysicsBody2D::test_move, DEFVAL(Variant()), DEFVAL(0.08));

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

Ref<KinematicCollision2D> PhysicsBody2D::_move(const Vector2 &p_motion, bool p_test_only, real_t p_margin) {
	PhysicsServer2D::MotionResult result;

	if (move_and_collide(p_motion, result, p_margin, p_test_only)) {
		if (motion_cache.is_null()) {
			motion_cache.instantiate();
			motion_cache->owner = this;
		}

		motion_cache->result = result;
		return motion_cache;
	}

	return Ref<KinematicCollision2D>();
}

bool PhysicsBody2D::move_and_collide(const Vector2 &p_motion, PhysicsServer2D::MotionResult &r_result, real_t p_margin, bool p_test_only, bool p_cancel_sliding, bool p_collide_separation_ray, const Set<RID> &p_exclude) {
	if (is_only_update_transform_changes_enabled()) {
		ERR_PRINT("Move functions do not work together with 'sync to physics' option. Please read the documentation.");
	}
	Transform2D gt = get_global_transform();
	bool colliding = PhysicsServer2D::get_singleton()->body_test_motion(get_rid(), gt, p_motion, p_margin, &r_result, p_collide_separation_ray, p_exclude);

	// Restore direction of motion to be along original motion,
	// in order to avoid sliding due to recovery,
	// but only if collision depth is low enough to avoid tunneling.
	if (p_cancel_sliding) {
		real_t motion_length = p_motion.length();
		real_t precision = 0.001;

		if (colliding) {
			// Can't just use margin as a threshold because collision depth is calculated on unsafe motion,
			// so even in normal resting cases the depth can be a bit more than the margin.
			precision += motion_length * (r_result.collision_unsafe_fraction - r_result.collision_safe_fraction);

			if (r_result.collision_depth > (real_t)p_margin + precision) {
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
			real_t projected_length = r_result.travel.dot(motion_normal);
			Vector2 recovery = r_result.travel - motion_normal * projected_length;
			real_t recovery_length = recovery.length();
			// Fixes cases where canceling slide causes the motion to go too deep into the ground,
			// because we're only taking rest information into account and not general recovery.
			if (recovery_length < (real_t)p_margin + precision) {
				// Apply adjustment to motion.
				r_result.travel = motion_normal * projected_length;
				r_result.remainder = p_motion - r_result.travel;
			}
		}
	}

	if (!p_test_only) {
		gt.elements[2] += r_result.travel;
		set_global_transform(gt);
	}

	return colliding;
}

bool PhysicsBody2D::test_move(const Transform2D &p_from, const Vector2 &p_motion, const Ref<KinematicCollision2D> &r_collision, real_t p_margin) {
	ERR_FAIL_COND_V(!is_inside_tree(), false);

	PhysicsServer2D::MotionResult *r = nullptr;
	if (r_collision.is_valid()) {
		// Needs const_cast because method bindings don't support non-const Ref.
		r = const_cast<PhysicsServer2D::MotionResult *>(&r_collision->result);
	}

	return PhysicsServer2D::get_singleton()->body_test_motion(get_rid(), p_from, p_motion, p_margin, r);
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

	if (kinematic_motion) {
		_update_kinematic_motion();
	} else {
		PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_LINEAR_VELOCITY, constant_linear_velocity);
	}
}

void StaticBody2D::set_constant_angular_velocity(real_t p_vel) {
	constant_angular_velocity = p_vel;

	if (kinematic_motion) {
		_update_kinematic_motion();
	} else {
		PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_ANGULAR_VELOCITY, constant_angular_velocity);
	}
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

void StaticBody2D::set_kinematic_motion_enabled(bool p_enabled) {
	if (p_enabled == kinematic_motion) {
		return;
	}

	kinematic_motion = p_enabled;

	if (kinematic_motion) {
		set_body_mode(PhysicsServer2D::BODY_MODE_KINEMATIC);
	} else {
		set_body_mode(PhysicsServer2D::BODY_MODE_STATIC);
	}

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		update_configuration_warnings();
		return;
	}
#endif

	_update_kinematic_motion();
}

bool StaticBody2D::is_kinematic_motion_enabled() const {
	return kinematic_motion;
}

void StaticBody2D::set_sync_to_physics(bool p_enable) {
	if (sync_to_physics == p_enable) {
		return;
	}

	sync_to_physics = p_enable;

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		update_configuration_warnings();
		return;
	}
#endif

	if (kinematic_motion) {
		_update_kinematic_motion();
	}
}

bool StaticBody2D::is_sync_to_physics_enabled() const {
	return sync_to_physics;
}

void StaticBody2D::_direct_state_changed(Object *p_state) {
	if (!sync_to_physics) {
		return;
	}

	PhysicsDirectBodyState2D *state = Object::cast_to<PhysicsDirectBodyState2D>(p_state);
	ERR_FAIL_NULL_MSG(state, "Method '_direct_state_changed' must receive a valid PhysicsDirectBodyState2D object as argument");

	last_valid_transform = state->get_transform();
	set_notify_local_transform(false);
	set_global_transform(last_valid_transform);
	set_notify_local_transform(true);
}

TypedArray<String> StaticBody2D::get_configuration_warnings() const {
	TypedArray<String> warnings = PhysicsBody2D::get_configuration_warnings();

	if (sync_to_physics && !kinematic_motion) {
		warnings.push_back(TTR("Sync to physics works only when kinematic motion is enabled."));
	}

	return warnings;
}

void StaticBody2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			last_valid_transform = get_global_transform();
		} break;

		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			// Used by sync to physics, send the new transform to the physics...
			Transform2D new_transform = get_global_transform();

			real_t delta_time = get_physics_process_delta_time();
			new_transform.translate(constant_linear_velocity * delta_time);
			new_transform.set_rotation(new_transform.get_rotation() + constant_angular_velocity * delta_time);

			PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_TRANSFORM, new_transform);

			// ... but then revert changes.
			set_notify_local_transform(false);
			set_global_transform(last_valid_transform);
			set_notify_local_transform(true);
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				return;
			}
#endif

			ERR_FAIL_COND(!kinematic_motion);

			Transform2D new_transform = get_global_transform();

			real_t delta_time = get_physics_process_delta_time();
			new_transform.translate(constant_linear_velocity * delta_time);
			new_transform.set_rotation(new_transform.get_rotation() + constant_angular_velocity * delta_time);

			if (sync_to_physics) {
				// Propagate transform change to node.
				set_global_transform(new_transform);
			} else {
				PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_TRANSFORM, new_transform);

				// Propagate transform change to node.
				set_block_transform_notify(true);
				set_global_transform(new_transform);
				set_block_transform_notify(false);
			}
		} break;
	}
}

void StaticBody2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant_linear_velocity", "vel"), &StaticBody2D::set_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_constant_angular_velocity", "vel"), &StaticBody2D::set_constant_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_linear_velocity"), &StaticBody2D::get_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_angular_velocity"), &StaticBody2D::get_constant_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_kinematic_motion_enabled", "enabled"), &StaticBody2D::set_kinematic_motion_enabled);
	ClassDB::bind_method(D_METHOD("is_kinematic_motion_enabled"), &StaticBody2D::is_kinematic_motion_enabled);

	ClassDB::bind_method(D_METHOD("set_physics_material_override", "physics_material_override"), &StaticBody2D::set_physics_material_override);
	ClassDB::bind_method(D_METHOD("get_physics_material_override"), &StaticBody2D::get_physics_material_override);

	ClassDB::bind_method(D_METHOD("set_sync_to_physics", "enable"), &StaticBody2D::set_sync_to_physics);
	ClassDB::bind_method(D_METHOD("is_sync_to_physics_enabled"), &StaticBody2D::is_sync_to_physics_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material_override", "get_physics_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "constant_linear_velocity"), "set_constant_linear_velocity", "get_constant_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "constant_angular_velocity"), "set_constant_angular_velocity", "get_constant_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "kinematic_motion"), "set_kinematic_motion_enabled", "is_kinematic_motion_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sync_to_physics"), "set_sync_to_physics", "is_sync_to_physics_enabled");
}

StaticBody2D::StaticBody2D() :
		PhysicsBody2D(PhysicsServer2D::BODY_MODE_STATIC) {
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

void StaticBody2D::_update_kinematic_motion() {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
#endif

	if (kinematic_motion && sync_to_physics) {
		PhysicsServer2D::get_singleton()->body_set_force_integration_callback(get_rid(), callable_mp(this, &StaticBody2D::_direct_state_changed));
		set_only_update_transform_changes(true);
		set_notify_local_transform(true);
	} else {
		PhysicsServer2D::get_singleton()->body_set_force_integration_callback(get_rid(), Callable());
		set_only_update_transform_changes(false);
		set_notify_local_transform(false);
	}

	bool needs_physics_process = false;
	if (kinematic_motion) {
		if (!Math::is_zero_approx(constant_angular_velocity) || !constant_linear_velocity.is_equal_approx(Vector2())) {
			needs_physics_process = true;
		}
	}

	set_physics_process_internal(needs_physics_process);
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
				node->connect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &RigidBody2D::_body_enter_tree), make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &RigidBody2D::_body_exit_tree), make_binds(objid));
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
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &RigidBody2D::_body_enter_tree));
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &RigidBody2D::_body_exit_tree));
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
	int shape = 0;
	int local_shape = 0;
};

void RigidBody2D::_direct_state_changed(Object *p_state) {
#ifdef DEBUG_ENABLED
	state = Object::cast_to<PhysicsDirectBodyState2D>(p_state);
	ERR_FAIL_NULL_MSG(state, "Method '_direct_state_changed' must receive a valid PhysicsDirectBodyState2D object as argument");
#else
	state = (PhysicsDirectBodyState2D *)p_state; //trust it
#endif

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

	GDVIRTUAL_CALL(_integrate_forces, state);

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

	state = nullptr;
}

void RigidBody2D::set_mode(Mode p_mode) {
	mode = p_mode;
	switch (p_mode) {
		case MODE_DYNAMIC: {
			set_body_mode(PhysicsServer2D::BODY_MODE_DYNAMIC);
		} break;
		case MODE_STATIC: {
			set_body_mode(PhysicsServer2D::BODY_MODE_STATIC);

		} break;
		case MODE_KINEMATIC: {
			set_body_mode(PhysicsServer2D::BODY_MODE_KINEMATIC);

		} break;
		case MODE_DYNAMIC_LOCKED: {
			set_body_mode(PhysicsServer2D::BODY_MODE_DYNAMIC_LOCKED);

		} break;
	}
}

RigidBody2D::Mode RigidBody2D::get_mode() const {
	return mode;
}

void RigidBody2D::set_mass(real_t p_mass) {
	ERR_FAIL_COND(p_mass <= 0);
	mass = p_mass;
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_MASS, mass);
}

real_t RigidBody2D::get_mass() const {
	return mass;
}

void RigidBody2D::set_inertia(real_t p_inertia) {
	ERR_FAIL_COND(p_inertia < 0);
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_INERTIA, p_inertia);
}

real_t RigidBody2D::get_inertia() const {
	return PhysicsServer2D::get_singleton()->body_get_param(get_rid(), PhysicsServer2D::BODY_PARAM_INERTIA);
}

void RigidBody2D::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
	if (physics_material_override.is_valid()) {
		if (physics_material_override->is_connected(CoreStringNames::get_singleton()->changed, callable_mp(this, &RigidBody2D::_reload_physics_characteristics))) {
			physics_material_override->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &RigidBody2D::_reload_physics_characteristics));
		}
	}

	physics_material_override = p_physics_material_override;

	if (physics_material_override.is_valid()) {
		physics_material_override->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &RigidBody2D::_reload_physics_characteristics));
	}
	_reload_physics_characteristics();
}

Ref<PhysicsMaterial> RigidBody2D::get_physics_material_override() const {
	return physics_material_override;
}

void RigidBody2D::set_gravity_scale(real_t p_gravity_scale) {
	gravity_scale = p_gravity_scale;
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_GRAVITY_SCALE, gravity_scale);
}

real_t RigidBody2D::get_gravity_scale() const {
	return gravity_scale;
}

void RigidBody2D::set_linear_damp(real_t p_linear_damp) {
	ERR_FAIL_COND(p_linear_damp < -1);
	linear_damp = p_linear_damp;
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_LINEAR_DAMP, linear_damp);
}

real_t RigidBody2D::get_linear_damp() const {
	return linear_damp;
}

void RigidBody2D::set_angular_damp(real_t p_angular_damp) {
	ERR_FAIL_COND(p_angular_damp < -1);
	angular_damp = p_angular_damp;
	PhysicsServer2D::get_singleton()->body_set_param(get_rid(), PhysicsServer2D::BODY_PARAM_ANGULAR_DAMP, angular_damp);
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
		PhysicsServer2D::get_singleton()->body_set_axis_velocity(get_rid(), p_axis);
		linear_velocity = v;
	}
}

void RigidBody2D::set_linear_velocity(const Vector2 &p_velocity) {
	linear_velocity = p_velocity;
	if (state) {
		state->set_linear_velocity(linear_velocity);
	} else {
		PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_LINEAR_VELOCITY, linear_velocity);
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
		PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_ANGULAR_VELOCITY, angular_velocity);
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
	PhysicsServer2D::get_singleton()->body_set_omit_force_integration(get_rid(), p_enable);
}

bool RigidBody2D::is_using_custom_integrator() {
	return custom_integrator;
}

void RigidBody2D::set_sleeping(bool p_sleeping) {
	sleeping = p_sleeping;
	PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_SLEEPING, sleeping);
}

void RigidBody2D::set_can_sleep(bool p_active) {
	can_sleep = p_active;
	PhysicsServer2D::get_singleton()->body_set_state(get_rid(), PhysicsServer2D::BODY_STATE_CAN_SLEEP, p_active);
}

bool RigidBody2D::is_able_to_sleep() const {
	return can_sleep;
}

bool RigidBody2D::is_sleeping() const {
	return sleeping;
}

void RigidBody2D::set_max_contacts_reported(int p_amount) {
	max_contacts_reported = p_amount;
	PhysicsServer2D::get_singleton()->body_set_max_contacts_reported(get_rid(), p_amount);
}

int RigidBody2D::get_max_contacts_reported() const {
	return max_contacts_reported;
}

void RigidBody2D::apply_central_impulse(const Vector2 &p_impulse) {
	PhysicsServer2D::get_singleton()->body_apply_central_impulse(get_rid(), p_impulse);
}

void RigidBody2D::apply_impulse(const Vector2 &p_impulse, const Vector2 &p_position) {
	PhysicsServer2D::get_singleton()->body_apply_impulse(get_rid(), p_impulse, p_position);
}

void RigidBody2D::apply_torque_impulse(real_t p_torque) {
	PhysicsServer2D::get_singleton()->body_apply_torque_impulse(get_rid(), p_torque);
}

void RigidBody2D::set_applied_force(const Vector2 &p_force) {
	PhysicsServer2D::get_singleton()->body_set_applied_force(get_rid(), p_force);
};

Vector2 RigidBody2D::get_applied_force() const {
	return PhysicsServer2D::get_singleton()->body_get_applied_force(get_rid());
};

void RigidBody2D::set_applied_torque(const real_t p_torque) {
	PhysicsServer2D::get_singleton()->body_set_applied_torque(get_rid(), p_torque);
};

real_t RigidBody2D::get_applied_torque() const {
	return PhysicsServer2D::get_singleton()->body_get_applied_torque(get_rid());
};

void RigidBody2D::add_central_force(const Vector2 &p_force) {
	PhysicsServer2D::get_singleton()->body_add_central_force(get_rid(), p_force);
}

void RigidBody2D::add_force(const Vector2 &p_force, const Vector2 &p_position) {
	PhysicsServer2D::get_singleton()->body_add_force(get_rid(), p_force, p_position);
}

void RigidBody2D::add_torque(const real_t p_torque) {
	PhysicsServer2D::get_singleton()->body_add_torque(get_rid(), p_torque);
}

void RigidBody2D::set_continuous_collision_detection_mode(CCDMode p_mode) {
	ccd_mode = p_mode;
	PhysicsServer2D::get_singleton()->body_set_continuous_collision_detection_mode(get_rid(), PhysicsServer2D::CCDMode(p_mode));
}

RigidBody2D::CCDMode RigidBody2D::get_continuous_collision_detection_mode() const {
	return ccd_mode;
}

TypedArray<Node2D> RigidBody2D::get_colliding_bodies() const {
	ERR_FAIL_COND_V(!contact_monitor, Array());

	TypedArray<Node2D> ret;
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
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &RigidBody2D::_body_enter_tree));
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &RigidBody2D::_body_exit_tree));
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

TypedArray<String> RigidBody2D::get_configuration_warnings() const {
	Transform2D t = get_transform();

	TypedArray<String> warnings = CollisionObject2D::get_configuration_warnings();

	if ((get_mode() == MODE_DYNAMIC || get_mode() == MODE_DYNAMIC_LOCKED) && (ABS(t.elements[0].length() - 1.0) > 0.05 || ABS(t.elements[1].length() - 1.0) > 0.05)) {
		warnings.push_back(TTR("Size changes to RigidBody2D (in dynamic modes) will be overridden by the physics engine when running.\nChange the size in children collision shapes instead."));
	}

	return warnings;
}

void RigidBody2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &RigidBody2D::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &RigidBody2D::get_mode);

	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &RigidBody2D::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &RigidBody2D::get_mass);

	ClassDB::bind_method(D_METHOD("get_inertia"), &RigidBody2D::get_inertia);
	ClassDB::bind_method(D_METHOD("set_inertia", "inertia"), &RigidBody2D::set_inertia);

	ClassDB::bind_method(D_METHOD("set_physics_material_override", "physics_material_override"), &RigidBody2D::set_physics_material_override);
	ClassDB::bind_method(D_METHOD("get_physics_material_override"), &RigidBody2D::get_physics_material_override);

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
	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &RigidBody2D::apply_central_impulse, Vector2());
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "position"), &RigidBody2D::apply_impulse, Vector2());
	ClassDB::bind_method(D_METHOD("apply_torque_impulse", "torque"), &RigidBody2D::apply_torque_impulse);

	ClassDB::bind_method(D_METHOD("set_applied_force", "force"), &RigidBody2D::set_applied_force);
	ClassDB::bind_method(D_METHOD("get_applied_force"), &RigidBody2D::get_applied_force);

	ClassDB::bind_method(D_METHOD("set_applied_torque", "torque"), &RigidBody2D::set_applied_torque);
	ClassDB::bind_method(D_METHOD("get_applied_torque"), &RigidBody2D::get_applied_torque);

	ClassDB::bind_method(D_METHOD("add_central_force", "force"), &RigidBody2D::add_central_force);
	ClassDB::bind_method(D_METHOD("add_force", "force", "position"), &RigidBody2D::add_force, Vector2());
	ClassDB::bind_method(D_METHOD("add_torque", "torque"), &RigidBody2D::add_torque);

	ClassDB::bind_method(D_METHOD("set_sleeping", "sleeping"), &RigidBody2D::set_sleeping);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &RigidBody2D::is_sleeping);

	ClassDB::bind_method(D_METHOD("set_can_sleep", "able_to_sleep"), &RigidBody2D::set_can_sleep);
	ClassDB::bind_method(D_METHOD("is_able_to_sleep"), &RigidBody2D::is_able_to_sleep);

	ClassDB::bind_method(D_METHOD("get_colliding_bodies"), &RigidBody2D::get_colliding_bodies);

	GDVIRTUAL_BIND(_integrate_forces, "state");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Dynamic,Static,DynamicLocked,Kinematic"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass", PROPERTY_HINT_RANGE, "0.01,65535,0.01,exp"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inertia", PROPERTY_HINT_RANGE, "0.01,65535,0.01,exp", PROPERTY_USAGE_NONE), "set_inertia", "get_inertia");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material_override", "get_physics_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gravity_scale", PROPERTY_HINT_RANGE, "-128,128,0.01"), "set_gravity_scale", "get_gravity_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "custom_integrator"), "set_use_custom_integrator", "is_using_custom_integrator");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "continuous_cd", PROPERTY_HINT_ENUM, "Disabled,Cast Ray,Cast Shape"), "set_continuous_collision_detection_mode", "get_continuous_collision_detection_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "contacts_reported", PROPERTY_HINT_RANGE, "0,64,1,or_greater"), "set_max_contacts_reported", "get_max_contacts_reported");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "contact_monitor"), "set_contact_monitor", "is_contact_monitor_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleeping", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "can_sleep"), "set_can_sleep", "is_able_to_sleep");
	ADD_GROUP("Linear", "linear_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "linear_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"), "set_linear_damp", "get_linear_damp");
	ADD_GROUP("Angular", "angular_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"), "set_angular_damp", "get_angular_damp");
	ADD_GROUP("Applied Forces", "applied_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "applied_force"), "set_applied_force", "get_applied_force");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "applied_torque"), "set_applied_torque", "get_applied_torque");

	ADD_SIGNAL(MethodInfo("body_shape_entered", PropertyInfo(Variant::RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape"), PropertyInfo(Variant::INT, "local_shape")));
	ADD_SIGNAL(MethodInfo("body_shape_exited", PropertyInfo(Variant::RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape"), PropertyInfo(Variant::INT, "local_shape")));
	ADD_SIGNAL(MethodInfo("body_entered", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("body_exited", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("sleeping_state_changed"));

	BIND_ENUM_CONSTANT(MODE_DYNAMIC);
	BIND_ENUM_CONSTANT(MODE_STATIC);
	BIND_ENUM_CONSTANT(MODE_DYNAMIC_LOCKED);
	BIND_ENUM_CONSTANT(MODE_KINEMATIC);

	BIND_ENUM_CONSTANT(CCD_MODE_DISABLED);
	BIND_ENUM_CONSTANT(CCD_MODE_CAST_RAY);
	BIND_ENUM_CONSTANT(CCD_MODE_CAST_SHAPE);
}

RigidBody2D::RigidBody2D() :
		PhysicsBody2D(PhysicsServer2D::BODY_MODE_DYNAMIC) {
	PhysicsServer2D::get_singleton()->body_set_force_integration_callback(get_rid(), callable_mp(this, &RigidBody2D::_direct_state_changed));
}

RigidBody2D::~RigidBody2D() {
	if (contact_monitor) {
		memdelete(contact_monitor);
	}
}

void RigidBody2D::_reload_physics_characteristics() {
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
	float delta = Engine::get_singleton()->is_in_physics_frame() ? get_physics_process_delta_time() : get_process_delta_time();

	Vector2 current_platform_velocity = platform_velocity;

	if ((on_floor || on_wall) && platform_rid.is_valid()) {
		bool excluded = (moving_platform_ignore_layers & platform_layer) != 0;
		if (!excluded) {
			// This approach makes sure there is less delay between the actual body velocity and the one we saved.
			PhysicsDirectBodyState2D *bs = PhysicsServer2D::get_singleton()->body_get_direct_state(platform_rid);
			if (bs) {
				Transform2D gt = get_global_transform();
				Vector2 local_position = gt.elements[2] - bs->get_transform().elements[2];
				current_platform_velocity = bs->get_velocity_at_local_position(local_position);
			}
		} else {
			current_platform_velocity = Vector2();
		}
	}

	motion_results.clear();

	bool was_on_floor = on_floor;
	on_floor = false;
	on_ceiling = false;
	on_wall = false;

	if (!current_platform_velocity.is_equal_approx(Vector2())) {
		PhysicsServer2D::MotionResult floor_result;
		Set<RID> exclude;
		exclude.insert(platform_rid);
		if (move_and_collide(current_platform_velocity * delta, floor_result, margin, false, false, false, exclude)) {
			motion_results.push_back(floor_result);
			_set_collision_direction(floor_result);
		}
	}

	if (motion_mode == MOTION_MODE_GROUNDED) {
		_move_and_slide_grounded(delta, was_on_floor, current_platform_velocity);
	} else {
		_move_and_slide_free(delta);
	}

	if (!on_floor && !on_wall) {
		// Add last platform velocity when just left a moving platform.
		linear_velocity += current_platform_velocity;
	}

	return motion_results.size() > 0;
}

void CharacterBody2D::_move_and_slide_grounded(real_t p_delta, bool p_was_on_floor, const Vector2 &p_prev_platform_velocity) {
	Vector2 motion = linear_velocity * p_delta;
	Vector2 motion_slide_up = motion.slide(up_direction);

	Vector2 prev_floor_normal = floor_normal;
	RID prev_platform_rid = platform_rid;
	int prev_platform_layer = platform_layer;

	platform_rid = RID();
	floor_normal = Vector2();
	platform_velocity = Vector2();

	// No sliding on first attempt to keep floor motion stable when possible,
	// When stop on slope is enabled or when there is no up direction.
	bool sliding_enabled = !floor_stop_on_slope;
	// Constant speed can be applied only the first time sliding is enabled.
	bool can_apply_constant_speed = sliding_enabled;
	bool first_slide = true;
	bool vel_dir_facing_up = linear_velocity.dot(up_direction) > 0;
	Vector2 last_travel;

	for (int iteration = 0; iteration < max_slides; ++iteration) {
		PhysicsServer2D::MotionResult result;

		Vector2 prev_position = get_global_transform().elements[2];

		bool collided = move_and_collide(motion, result, margin, false, !sliding_enabled);

		if (collided) {
			motion_results.push_back(result);
			_set_collision_direction(result);

			if (on_floor && floor_stop_on_slope && (linear_velocity.normalized() + up_direction).length() < 0.01) {
				Transform2D gt = get_global_transform();
				if (result.travel.length() > margin) {
					gt.elements[2] -= result.travel.slide(up_direction);
				} else {
					gt.elements[2] -= result.travel;
				}
				set_global_transform(gt);
				linear_velocity = Vector2();
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
					if (result.travel.length() <= margin) {
						// Cancels the motion.
						Transform2D gt = get_global_transform();
						gt.elements[2] -= result.travel;
						set_global_transform(gt);
					}
					on_floor = true;
					platform_rid = prev_platform_rid;
					platform_layer = prev_platform_layer;
					platform_velocity = p_prev_platform_velocity;
					floor_normal = prev_floor_normal;
					linear_velocity = Vector2();
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
			else if ((sliding_enabled || !on_floor) && (!on_ceiling || slide_on_ceiling || !vel_dir_facing_up)) {
				Vector2 slide_motion = result.remainder.slide(result.collision_normal);
				if (slide_motion.dot(linear_velocity) > 0.0) {
					motion = slide_motion;
				} else {
					motion = Vector2();
				}
				if (slide_on_ceiling && on_ceiling) {
					// Apply slide only in the direction of the input motion, otherwise just stop to avoid jittering when moving against a wall.
					if (vel_dir_facing_up) {
						linear_velocity = linear_velocity.slide(result.collision_normal);
					} else {
						// Avoid acceleration in slope when falling.
						linear_velocity = up_direction * up_direction.dot(linear_velocity);
					}
				}
			}
			// No sliding on first attempt to keep floor motion stable when possible.
			else {
				motion = result.remainder;
				if (on_ceiling && !slide_on_ceiling && vel_dir_facing_up) {
					linear_velocity = linear_velocity.slide(up_direction);
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
		linear_velocity = linear_velocity.slide(up_direction);
	}
}

void CharacterBody2D::_move_and_slide_free(real_t p_delta) {
	Vector2 motion = linear_velocity * p_delta;

	platform_rid = RID();
	floor_normal = Vector2();
	platform_velocity = Vector2();

	bool first_slide = true;
	for (int iteration = 0; iteration < max_slides; ++iteration) {
		PhysicsServer2D::MotionResult result;

		bool collided = move_and_collide(motion, result, margin, false, false);

		if (collided) {
			motion_results.push_back(result);
			_set_collision_direction(result);

			if (free_mode_min_slide_angle != 0 && result.get_angle(-linear_velocity.normalized()) < free_mode_min_slide_angle + FLOOR_ANGLE_THRESHOLD) {
				motion = Vector2();
			} else if (first_slide) {
				Vector2 motion_slide_norm = result.remainder.slide(result.collision_normal).normalized();
				motion = motion_slide_norm * (motion.length() - result.travel.length());
			} else {
				motion = result.remainder.slide(result.collision_normal);
			}

			if (motion.dot(linear_velocity) <= 0.0) {
				motion = Vector2();
			}
		}

		first_slide = false;

		if (!collided || motion.is_equal_approx(Vector2())) {
			break;
		}
	}
}

void CharacterBody2D::_snap_on_floor(bool was_on_floor, bool vel_dir_facing_up) {
	if (Math::is_equal_approx(floor_snap_length, 0) || on_floor || !was_on_floor || vel_dir_facing_up) {
		return;
	}

	Transform2D gt = get_global_transform();
	PhysicsServer2D::MotionResult result;
	if (move_and_collide(up_direction * -floor_snap_length, result, margin, true, false, true)) {
		bool apply = true;
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
		} else {
			apply = false;
		}

		if (apply) {
			gt.elements[2] += result.travel;
			set_global_transform(gt);
		}
	}
}

bool CharacterBody2D::_on_floor_if_snapped(bool was_on_floor, bool vel_dir_facing_up) {
	if (Math::is_equal_approx(floor_snap_length, 0) || up_direction == Vector2() || on_floor || !was_on_floor || vel_dir_facing_up) {
		return false;
	}

	PhysicsServer2D::MotionResult result;
	if (move_and_collide(up_direction * -floor_snap_length, result, margin, true, false, true)) {
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
		// Don't apply wall velocity when the collider is a CharacterBody2D.
		if (Object::cast_to<CharacterBody2D>(ObjectDB::get_instance(p_result.collider_id)) == nullptr) {
			_set_platform_data(p_result);
		}
	}
}

void CharacterBody2D::_set_platform_data(const PhysicsServer2D::MotionResult &p_result) {
	platform_rid = p_result.collider;
	platform_velocity = p_result.collider_velocity;
	platform_layer = 0;
	CollisionObject2D *collision_object = Object::cast_to<CollisionObject2D>(ObjectDB::get_instance(p_result.collider_id));
	if (collision_object) {
		platform_layer = collision_object->get_collision_layer();
	}
}

const Vector2 &CharacterBody2D::get_linear_velocity() const {
	return linear_velocity;
}

void CharacterBody2D::set_linear_velocity(const Vector2 &p_velocity) {
	linear_velocity = p_velocity;
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

Vector2 CharacterBody2D::get_floor_normal() const {
	return floor_normal;
}

real_t CharacterBody2D::get_floor_angle(const Vector2 &p_up_direction) const {
	ERR_FAIL_COND_V(p_up_direction == Vector2(), 0);
	return Math::acos(floor_normal.dot(p_up_direction));
}

Vector2 CharacterBody2D::get_platform_velocity() const {
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

	if (slide_colliders[p_bounce].is_null()) {
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

uint32_t CharacterBody2D::get_moving_platform_ignore_layers() const {
	return moving_platform_ignore_layers;
}

void CharacterBody2D::set_moving_platform_ignore_layers(uint32_t p_exclude_layers) {
	moving_platform_ignore_layers = p_exclude_layers;
}

void CharacterBody2D::set_motion_mode(MotionMode p_mode) {
	motion_mode = p_mode;
}

CharacterBody2D::MotionMode CharacterBody2D::get_motion_mode() const {
	return motion_mode;
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
			on_ceiling = false;
			on_wall = false;
			motion_results.clear();
			platform_velocity = Vector2();
		} break;
	}
}

void CharacterBody2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("move_and_slide"), &CharacterBody2D::move_and_slide);

	ClassDB::bind_method(D_METHOD("set_linear_velocity", "linear_velocity"), &CharacterBody2D::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &CharacterBody2D::get_linear_velocity);

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

	ClassDB::bind_method(D_METHOD("set_moving_platform_ignore_layers", "exclude_layer"), &CharacterBody2D::set_moving_platform_ignore_layers);
	ClassDB::bind_method(D_METHOD("get_moving_platform_ignore_layers"), &CharacterBody2D::get_moving_platform_ignore_layers);

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

	ClassDB::bind_method(D_METHOD("is_on_floor"), &CharacterBody2D::is_on_floor);
	ClassDB::bind_method(D_METHOD("is_on_floor_only"), &CharacterBody2D::is_on_floor_only);
	ClassDB::bind_method(D_METHOD("is_on_ceiling"), &CharacterBody2D::is_on_ceiling);
	ClassDB::bind_method(D_METHOD("is_on_ceiling_only"), &CharacterBody2D::is_on_ceiling_only);
	ClassDB::bind_method(D_METHOD("is_on_wall"), &CharacterBody2D::is_on_wall);
	ClassDB::bind_method(D_METHOD("is_on_wall_only"), &CharacterBody2D::is_on_wall_only);
	ClassDB::bind_method(D_METHOD("get_floor_normal"), &CharacterBody2D::get_floor_normal);
	ClassDB::bind_method(D_METHOD("get_floor_angle", "up_direction"), &CharacterBody2D::get_floor_angle, DEFVAL(Vector2(0.0, -1.0)));
	ClassDB::bind_method(D_METHOD("get_platform_velocity"), &CharacterBody2D::get_platform_velocity);
	ClassDB::bind_method(D_METHOD("get_slide_collision_count"), &CharacterBody2D::get_slide_collision_count);
	ClassDB::bind_method(D_METHOD("get_slide_collision", "slide_idx"), &CharacterBody2D::_get_slide_collision);
	ClassDB::bind_method(D_METHOD("get_last_slide_collision"), &CharacterBody2D::_get_last_slide_collision);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "motion_mode", PROPERTY_HINT_ENUM, "Grounded,Free", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_motion_mode", "get_motion_mode");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "slide_on_ceiling"), "set_slide_on_ceiling_enabled", "is_slide_on_ceiling_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_slides", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_max_slides", "get_max_slides");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "up_direction"), "set_up_direction", "get_up_direction");
	ADD_GROUP("Free Mode", "free_mode_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "free_mode_min_slide_angle", PROPERTY_HINT_RANGE, "0,180,0.1,radians", PROPERTY_USAGE_DEFAULT), "set_free_mode_min_slide_angle", "get_free_mode_min_slide_angle");
	ADD_GROUP("Floor", "floor_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "floor_stop_on_slope"), "set_floor_stop_on_slope_enabled", "is_floor_stop_on_slope_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "floor_constant_speed"), "set_floor_constant_speed_enabled", "is_floor_constant_speed_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "floor_block_on_wall"), "set_floor_block_on_wall_enabled", "is_floor_block_on_wall_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "floor_max_angle", PROPERTY_HINT_RANGE, "0,180,0.1,radians"), "set_floor_max_angle", "get_floor_max_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "floor_snap_length", PROPERTY_HINT_RANGE, "0,1000,0.1"), "set_floor_snap_length", "get_floor_snap_length");
	ADD_GROUP("Moving platform", "moving_platform");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "moving_platform_ignore_layers", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_moving_platform_ignore_layers", "get_moving_platform_ignore_layers");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision/safe_margin", PROPERTY_HINT_RANGE, "0.001,256,0.001"), "set_safe_margin", "get_safe_margin");

	BIND_ENUM_CONSTANT(MOTION_MODE_GROUNDED);
	BIND_ENUM_CONSTANT(MOTION_MODE_FREE);
}

void CharacterBody2D::_validate_property(PropertyInfo &property) const {
	if (motion_mode == MOTION_MODE_FREE) {
		if (property.name.begins_with("floor_") || property.name == "up_direction" || property.name == "slide_on_ceiling") {
			property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
		}
	} else {
		if (property.name == "free_mode_min_slide_angle") {
			property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
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
	ADD_PROPERTY(PropertyInfo(Variant::RID, "collider_rid"), "", "get_collider_rid");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collider_shape"), "", "get_collider_shape");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_shape_index"), "", "get_collider_shape_index");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "collider_velocity"), "", "get_collider_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::NIL, "collider_metadata", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT), "", "get_collider_metadata");
}
