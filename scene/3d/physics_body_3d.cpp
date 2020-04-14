/*************************************************************************/
/*  physics_body_3d.cpp                                                  */
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

#include "physics_body_3d.h"

#include "core/core_string_names.h"
#include "scene/scene_string_names.h"

void PhysicsBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("move_and_collide", "linear_velocity", "test_only", "safe_margin", "max_collisions"), &PhysicsBody3D::_move, DEFVAL(false), DEFVAL(0.001), DEFVAL(1));
	ClassDB::bind_method(D_METHOD("test_move", "from", "linear_velocity", "collision", "safe_margin", "max_collisions"), &PhysicsBody3D::test_move, DEFVAL(Variant()), DEFVAL(0.001), DEFVAL(1));

	ClassDB::bind_method(D_METHOD("set_axis_lock", "axis", "lock"), &PhysicsBody3D::set_axis_lock);
	ClassDB::bind_method(D_METHOD("get_axis_lock", "axis"), &PhysicsBody3D::get_axis_lock);

	ClassDB::bind_method(D_METHOD("get_collision_exceptions"), &PhysicsBody3D::get_collision_exceptions);
	ClassDB::bind_method(D_METHOD("add_collision_exception_with", "body"), &PhysicsBody3D::add_collision_exception_with);
	ClassDB::bind_method(D_METHOD("remove_collision_exception_with", "body"), &PhysicsBody3D::remove_collision_exception_with);

	ADD_GROUP("Axis Lock", "axis_lock_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_linear_x"), "set_axis_lock", "get_axis_lock", PhysicsServer3D::BODY_AXIS_LINEAR_X);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_linear_y"), "set_axis_lock", "get_axis_lock", PhysicsServer3D::BODY_AXIS_LINEAR_Y);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_linear_z"), "set_axis_lock", "get_axis_lock", PhysicsServer3D::BODY_AXIS_LINEAR_Z);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_angular_x"), "set_axis_lock", "get_axis_lock", PhysicsServer3D::BODY_AXIS_ANGULAR_X);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_angular_y"), "set_axis_lock", "get_axis_lock", PhysicsServer3D::BODY_AXIS_ANGULAR_Y);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "axis_lock_angular_z"), "set_axis_lock", "get_axis_lock", PhysicsServer3D::BODY_AXIS_ANGULAR_Z);
}

PhysicsBody3D::PhysicsBody3D(PhysicsServer3D::BodyMode p_mode) :
		CollisionObject3D(PhysicsServer3D::get_singleton()->body_create(), false) {
	set_body_mode(p_mode);
}

PhysicsBody3D::~PhysicsBody3D() {
	if (motion_cache.is_valid()) {
		motion_cache->owner = nullptr;
	}
}

TypedArray<PhysicsBody3D> PhysicsBody3D::get_collision_exceptions() {
	List<RID> exceptions;
	PhysicsServer3D::get_singleton()->body_get_collision_exceptions(get_rid(), &exceptions);
	Array ret;
	for (const RID &body : exceptions) {
		ObjectID instance_id = PhysicsServer3D::get_singleton()->body_get_object_instance_id(body);
		Object *obj = ObjectDB::get_instance(instance_id);
		PhysicsBody3D *physics_body = Object::cast_to<PhysicsBody3D>(obj);
		ret.append(physics_body);
	}
	return ret;
}

void PhysicsBody3D::add_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	CollisionObject3D *collision_object = Object::cast_to<CollisionObject3D>(p_node);
	ERR_FAIL_COND_MSG(!collision_object, "Collision exception only works between two CollisionObject3Ds.");
	PhysicsServer3D::get_singleton()->body_add_collision_exception(get_rid(), collision_object->get_rid());
}

void PhysicsBody3D::remove_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	CollisionObject3D *collision_object = Object::cast_to<CollisionObject3D>(p_node);
	ERR_FAIL_COND_MSG(!collision_object, "Collision exception only works between two CollisionObject3Ds.");
	PhysicsServer3D::get_singleton()->body_remove_collision_exception(get_rid(), collision_object->get_rid());
}

Ref<KinematicCollision3D> PhysicsBody3D::_move(const Vector3 &p_linear_velocity, bool p_test_only, real_t p_margin, int p_max_collisions) {
	// Hack in order to work with calling from _process as well as from _physics_process; calling from thread is risky
	double delta = Engine::get_singleton()->is_in_physics_frame() ? get_physics_process_delta_time() : get_process_delta_time();

	PhysicsServer3D::MotionParameters parameters(get_global_transform(), p_linear_velocity * delta, p_margin);
	parameters.max_collisions = p_max_collisions;

	PhysicsServer3D::MotionResult result;
	if (move_and_collide(parameters, result, p_test_only)) {
		// Create a new instance when the cached reference is invalid or still in use in script.
		if (motion_cache.is_null() || motion_cache->reference_get_count() > 1) {
			motion_cache.instantiate();
			motion_cache->owner = this;
		}

		motion_cache->result = result;

		return motion_cache;
	}

	return Ref<KinematicCollision3D>();
}

bool PhysicsBody3D::move_and_collide(const PhysicsServer3D::MotionParameters &p_parameters, PhysicsServer3D::MotionResult &r_result, bool p_test_only, bool p_cancel_sliding) {
	bool colliding = PhysicsServer3D::get_singleton()->body_test_motion(get_rid(), p_parameters, &r_result);

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

			if (r_result.collisions[0].depth > p_parameters.margin + precision) {
				p_cancel_sliding = false;
			}
		}

		if (p_cancel_sliding) {
			// When motion is null, recovery is the resulting motion.
			Vector3 motion_normal;
			if (motion_length > CMP_EPSILON) {
				motion_normal = p_parameters.motion / motion_length;
			}

			// Check depth of recovery.
			real_t projected_length = r_result.travel.dot(motion_normal);
			Vector3 recovery = r_result.travel - motion_normal * projected_length;
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

	for (int i = 0; i < 3; i++) {
		if (locked_axis & (1 << i)) {
			r_result.travel[i] = 0;
		}
	}

	if (!p_test_only) {
		Transform3D gt = p_parameters.from;
		gt.origin += r_result.travel;
		set_global_transform(gt);
	}

	return colliding;
}

bool PhysicsBody3D::test_move(const Transform3D &p_from, const Vector3 &p_linear_velocity, const Ref<KinematicCollision3D> &r_collision, real_t p_margin, int p_max_collisions) {
	ERR_FAIL_COND_V(!is_inside_tree(), false);

	PhysicsServer3D::MotionResult *r = nullptr;
	if (r_collision.is_valid()) {
		// Needs const_cast because method bindings don't support non-const Ref.
		r = const_cast<PhysicsServer3D::MotionResult *>(&r_collision->result);
	}

	// Hack in order to work with calling from _process as well as from _physics_process; calling from thread is risky
	double delta = Engine::get_singleton()->is_in_physics_frame() ? get_physics_process_delta_time() : get_process_delta_time();

	PhysicsServer3D::MotionParameters parameters(p_from, p_linear_velocity * delta, p_margin);

	return PhysicsServer3D::get_singleton()->body_test_motion(get_rid(), parameters, r);
}

void PhysicsBody3D::set_axis_lock(PhysicsServer3D::BodyAxis p_axis, bool p_lock) {
	if (p_lock) {
		locked_axis |= p_axis;
	} else {
		locked_axis &= (~p_axis);
	}
	PhysicsServer3D::get_singleton()->body_set_axis_lock(get_rid(), p_axis, p_lock);
}

bool PhysicsBody3D::get_axis_lock(PhysicsServer3D::BodyAxis p_axis) const {
	return (locked_axis & p_axis);
}

Vector3 PhysicsBody3D::get_linear_velocity() const {
	return Vector3();
}

Vector3 PhysicsBody3D::get_angular_velocity() const {
	return Vector3();
}

real_t PhysicsBody3D::get_inverse_mass() const {
	return 0;
}

void StaticBody3D::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
	if (physics_material_override.is_valid()) {
		if (physics_material_override->is_connected(CoreStringNames::get_singleton()->changed, callable_mp(this, &StaticBody3D::_reload_physics_characteristics))) {
			physics_material_override->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &StaticBody3D::_reload_physics_characteristics));
		}
	}

	physics_material_override = p_physics_material_override;

	if (physics_material_override.is_valid()) {
		physics_material_override->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &StaticBody3D::_reload_physics_characteristics));
	}
	_reload_physics_characteristics();
}

Ref<PhysicsMaterial> StaticBody3D::get_physics_material_override() const {
	return physics_material_override;
}

void StaticBody3D::set_constant_linear_velocity(const Vector3 &p_vel) {
	constant_linear_velocity = p_vel;

	PhysicsServer3D::get_singleton()->body_set_state(get_rid(), PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY, constant_linear_velocity);
}

void StaticBody3D::set_constant_angular_velocity(const Vector3 &p_vel) {
	constant_angular_velocity = p_vel;

	PhysicsServer3D::get_singleton()->body_set_state(get_rid(), PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY, constant_angular_velocity);
}

Vector3 StaticBody3D::get_constant_linear_velocity() const {
	return constant_linear_velocity;
}

Vector3 StaticBody3D::get_constant_angular_velocity() const {
	return constant_angular_velocity;
}

void StaticBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant_linear_velocity", "vel"), &StaticBody3D::set_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_constant_angular_velocity", "vel"), &StaticBody3D::set_constant_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_linear_velocity"), &StaticBody3D::get_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_angular_velocity"), &StaticBody3D::get_constant_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_physics_material_override", "physics_material_override"), &StaticBody3D::set_physics_material_override);
	ClassDB::bind_method(D_METHOD("get_physics_material_override"), &StaticBody3D::get_physics_material_override);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material_override", "get_physics_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "constant_linear_velocity"), "set_constant_linear_velocity", "get_constant_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "constant_angular_velocity"), "set_constant_angular_velocity", "get_constant_angular_velocity");
}

StaticBody3D::StaticBody3D(PhysicsServer3D::BodyMode p_mode) :
		PhysicsBody3D(p_mode) {
}

void StaticBody3D::_reload_physics_characteristics() {
	if (physics_material_override.is_null()) {
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_BOUNCE, 0);
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_FRICTION, 1);
	} else {
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_BOUNCE, physics_material_override->computed_bounce());
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_FRICTION, physics_material_override->computed_friction());
	}
}

Vector3 AnimatableBody3D::get_linear_velocity() const {
	return linear_velocity;
}

Vector3 AnimatableBody3D::get_angular_velocity() const {
	return angular_velocity;
}

void AnimatableBody3D::set_sync_to_physics(bool p_enable) {
	if (sync_to_physics == p_enable) {
		return;
	}

	sync_to_physics = p_enable;

	_update_kinematic_motion();
}

bool AnimatableBody3D::is_sync_to_physics_enabled() const {
	return sync_to_physics;
}

void AnimatableBody3D::_update_kinematic_motion() {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
#endif

	if (sync_to_physics) {
		set_only_update_transform_changes(true);
		set_notify_local_transform(true);
	} else {
		set_only_update_transform_changes(false);
		set_notify_local_transform(false);
	}
}

void AnimatableBody3D::_body_state_changed_callback(void *p_instance, PhysicsDirectBodyState3D *p_state) {
	AnimatableBody3D *body = (AnimatableBody3D *)p_instance;
	body->_body_state_changed(p_state);
}

void AnimatableBody3D::_body_state_changed(PhysicsDirectBodyState3D *p_state) {
	linear_velocity = p_state->get_linear_velocity();
	angular_velocity = p_state->get_angular_velocity();

	if (!sync_to_physics) {
		return;
	}

	last_valid_transform = p_state->get_transform();
	set_notify_local_transform(false);
	set_global_transform(last_valid_transform);
	set_notify_local_transform(true);
	_on_transform_changed();
}

void AnimatableBody3D::_notification(int p_what) {
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
			Transform3D new_transform = get_global_transform();

			PhysicsServer3D::get_singleton()->body_set_state(get_rid(), PhysicsServer3D::BODY_STATE_TRANSFORM, new_transform);

			// ... but then revert changes.
			set_notify_local_transform(false);
			set_global_transform(last_valid_transform);
			set_notify_local_transform(true);
			_on_transform_changed();
		} break;
	}
}

void AnimatableBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sync_to_physics", "enable"), &AnimatableBody3D::set_sync_to_physics);
	ClassDB::bind_method(D_METHOD("is_sync_to_physics_enabled"), &AnimatableBody3D::is_sync_to_physics_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sync_to_physics"), "set_sync_to_physics", "is_sync_to_physics_enabled");
}

AnimatableBody3D::AnimatableBody3D() :
		StaticBody3D(PhysicsServer3D::BODY_MODE_KINEMATIC) {
	PhysicsServer3D::get_singleton()->body_set_state_sync_callback(get_rid(), this, _body_state_changed_callback);
}

void RigidDynamicBody3D::_body_enter_tree(ObjectID p_id) {
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

void RigidDynamicBody3D::_body_exit_tree(ObjectID p_id) {
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

void RigidDynamicBody3D::_body_inout(int p_status, const RID &p_body, ObjectID p_instance, int p_body_shape, int p_local_shape) {
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
				node->connect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &RigidDynamicBody3D::_body_enter_tree), make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &RigidDynamicBody3D::_body_exit_tree), make_binds(objid));
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

		if (E->get().shapes.is_empty()) {
			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &RigidDynamicBody3D::_body_enter_tree));
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &RigidDynamicBody3D::_body_exit_tree));
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

struct _RigidDynamicBodyInOut {
	RID rid;
	ObjectID id;
	int shape = 0;
	int local_shape = 0;
};

void RigidDynamicBody3D::_body_state_changed_callback(void *p_instance, PhysicsDirectBodyState3D *p_state) {
	RigidDynamicBody3D *body = (RigidDynamicBody3D *)p_instance;
	body->_body_state_changed(p_state);
}

void RigidDynamicBody3D::_body_state_changed(PhysicsDirectBodyState3D *p_state) {
	set_ignore_transform_notification(true);
	set_global_transform(p_state->get_transform());

	linear_velocity = p_state->get_linear_velocity();
	angular_velocity = p_state->get_angular_velocity();

	inverse_inertia_tensor = p_state->get_inverse_inertia_tensor();

	if (sleeping != p_state->is_sleeping()) {
		sleeping = p_state->is_sleeping();
		emit_signal(SceneStringNames::get_singleton()->sleeping_state_changed);
	}

	GDVIRTUAL_CALL(_integrate_forces, p_state);

	set_ignore_transform_notification(false);
	_on_transform_changed();

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

		_RigidDynamicBodyInOut *toadd = (_RigidDynamicBodyInOut *)alloca(p_state->get_contact_count() * sizeof(_RigidDynamicBodyInOut));
		int toadd_count = 0; //state->get_contact_count();
		RigidDynamicBody3D_RemoveAction *toremove = (RigidDynamicBody3D_RemoveAction *)alloca(rc * sizeof(RigidDynamicBody3D_RemoveAction));
		int toremove_count = 0;

		//put the ones to add

		for (int i = 0; i < p_state->get_contact_count(); i++) {
			RID rid = p_state->get_contact_collider(i);
			ObjectID obj = p_state->get_contact_collider_id(i);
			int local_shape = p_state->get_contact_local_shape(i);
			int shape = p_state->get_contact_collider_shape(i);

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
			_body_inout(1, toremove[i].rid, toadd[i].id, toadd[i].shape, toadd[i].local_shape);
		}

		contact_monitor->locked = false;
	}
}

void RigidDynamicBody3D::_notification(int p_what) {
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

void RigidDynamicBody3D::_apply_body_mode() {
	if (freeze) {
		switch (freeze_mode) {
			case FREEZE_MODE_STATIC: {
				set_body_mode(PhysicsServer3D::BODY_MODE_STATIC);
			} break;
			case FREEZE_MODE_KINEMATIC: {
				set_body_mode(PhysicsServer3D::BODY_MODE_KINEMATIC);
			} break;
		}
	} else if (lock_rotation) {
		set_body_mode(PhysicsServer3D::BODY_MODE_DYNAMIC_LINEAR);
	} else {
		set_body_mode(PhysicsServer3D::BODY_MODE_DYNAMIC);
	}
}

void RigidDynamicBody3D::set_lock_rotation_enabled(bool p_lock_rotation) {
	if (p_lock_rotation == lock_rotation) {
		return;
	}

	lock_rotation = p_lock_rotation;
	_apply_body_mode();
}

bool RigidDynamicBody3D::is_lock_rotation_enabled() const {
	return lock_rotation;
}

void RigidDynamicBody3D::set_freeze_enabled(bool p_freeze) {
	if (p_freeze == freeze) {
		return;
	}

	freeze = p_freeze;
	_apply_body_mode();
}

bool RigidDynamicBody3D::is_freeze_enabled() const {
	return freeze;
}

void RigidDynamicBody3D::set_freeze_mode(FreezeMode p_freeze_mode) {
	if (p_freeze_mode == freeze_mode) {
		return;
	}

	freeze_mode = p_freeze_mode;
	_apply_body_mode();
}

RigidDynamicBody3D::FreezeMode RigidDynamicBody3D::get_freeze_mode() const {
	return freeze_mode;
}

void RigidDynamicBody3D::set_mass(real_t p_mass) {
	ERR_FAIL_COND(p_mass <= 0);
	mass = p_mass;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_MASS, mass);
}

real_t RigidDynamicBody3D::get_mass() const {
	return mass;
}

void RigidDynamicBody3D::set_inertia(const Vector3 &p_inertia) {
	ERR_FAIL_COND(p_inertia.x < 0);
	ERR_FAIL_COND(p_inertia.y < 0);
	ERR_FAIL_COND(p_inertia.z < 0);

	inertia = p_inertia;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_INERTIA, inertia);
}

const Vector3 &RigidDynamicBody3D::get_inertia() const {
	return inertia;
}

void RigidDynamicBody3D::set_center_of_mass_mode(CenterOfMassMode p_mode) {
	if (center_of_mass_mode == p_mode) {
		return;
	}

	center_of_mass_mode = p_mode;

	switch (center_of_mass_mode) {
		case CENTER_OF_MASS_MODE_AUTO: {
			center_of_mass = Vector3();
			PhysicsServer3D::get_singleton()->body_reset_mass_properties(get_rid());
			if (inertia != Vector3()) {
				PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_INERTIA, inertia);
			}
		} break;

		case CENTER_OF_MASS_MODE_CUSTOM: {
			PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_CENTER_OF_MASS, center_of_mass);
		} break;
	}
}

RigidDynamicBody3D::CenterOfMassMode RigidDynamicBody3D::get_center_of_mass_mode() const {
	return center_of_mass_mode;
}

void RigidDynamicBody3D::set_center_of_mass(const Vector3 &p_center_of_mass) {
	if (center_of_mass == p_center_of_mass) {
		return;
	}

	ERR_FAIL_COND(center_of_mass_mode != CENTER_OF_MASS_MODE_CUSTOM);
	center_of_mass = p_center_of_mass;

	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_CENTER_OF_MASS, center_of_mass);
}

const Vector3 &RigidDynamicBody3D::get_center_of_mass() const {
	return center_of_mass;
}

void RigidDynamicBody3D::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
	if (physics_material_override.is_valid()) {
		if (physics_material_override->is_connected(CoreStringNames::get_singleton()->changed, callable_mp(this, &RigidDynamicBody3D::_reload_physics_characteristics))) {
			physics_material_override->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &RigidDynamicBody3D::_reload_physics_characteristics));
		}
	}

	physics_material_override = p_physics_material_override;

	if (physics_material_override.is_valid()) {
		physics_material_override->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &RigidDynamicBody3D::_reload_physics_characteristics));
	}
	_reload_physics_characteristics();
}

Ref<PhysicsMaterial> RigidDynamicBody3D::get_physics_material_override() const {
	return physics_material_override;
}

void RigidDynamicBody3D::set_gravity_scale(real_t p_gravity_scale) {
	gravity_scale = p_gravity_scale;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_GRAVITY_SCALE, gravity_scale);
}

real_t RigidDynamicBody3D::get_gravity_scale() const {
	return gravity_scale;
}

void RigidDynamicBody3D::set_linear_damp_mode(DampMode p_mode) {
	linear_damp_mode = p_mode;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_LINEAR_DAMP_MODE, linear_damp_mode);
}

RigidDynamicBody3D::DampMode RigidDynamicBody3D::get_linear_damp_mode() const {
	return linear_damp_mode;
}

void RigidDynamicBody3D::set_angular_damp_mode(DampMode p_mode) {
	angular_damp_mode = p_mode;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_ANGULAR_DAMP_MODE, angular_damp_mode);
}

RigidDynamicBody3D::DampMode RigidDynamicBody3D::get_angular_damp_mode() const {
	return angular_damp_mode;
}

void RigidDynamicBody3D::set_linear_damp(real_t p_linear_damp) {
	ERR_FAIL_COND(p_linear_damp < 0.0);
	linear_damp = p_linear_damp;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_LINEAR_DAMP, linear_damp);
}

real_t RigidDynamicBody3D::get_linear_damp() const {
	return linear_damp;
}

void RigidDynamicBody3D::set_angular_damp(real_t p_angular_damp) {
	ERR_FAIL_COND(p_angular_damp < 0.0);
	angular_damp = p_angular_damp;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_ANGULAR_DAMP, angular_damp);
}

real_t RigidDynamicBody3D::get_angular_damp() const {
	return angular_damp;
}

void RigidDynamicBody3D::set_axis_velocity(const Vector3 &p_axis) {
	Vector3 axis = p_axis.normalized();
	linear_velocity -= axis * axis.dot(linear_velocity);
	linear_velocity += p_axis;
	PhysicsServer3D::get_singleton()->body_set_state(get_rid(), PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY, linear_velocity);
}

void RigidDynamicBody3D::set_linear_velocity(const Vector3 &p_velocity) {
	linear_velocity = p_velocity;
	PhysicsServer3D::get_singleton()->body_set_state(get_rid(), PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY, linear_velocity);
}

Vector3 RigidDynamicBody3D::get_linear_velocity() const {
	return linear_velocity;
}

void RigidDynamicBody3D::set_angular_velocity(const Vector3 &p_velocity) {
	angular_velocity = p_velocity;
	PhysicsServer3D::get_singleton()->body_set_state(get_rid(), PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY, angular_velocity);
}

Vector3 RigidDynamicBody3D::get_angular_velocity() const {
	return angular_velocity;
}

Basis RigidDynamicBody3D::get_inverse_inertia_tensor() const {
	return inverse_inertia_tensor;
}

void RigidDynamicBody3D::set_use_custom_integrator(bool p_enable) {
	if (custom_integrator == p_enable) {
		return;
	}

	custom_integrator = p_enable;
	PhysicsServer3D::get_singleton()->body_set_omit_force_integration(get_rid(), p_enable);
}

bool RigidDynamicBody3D::is_using_custom_integrator() {
	return custom_integrator;
}

void RigidDynamicBody3D::set_sleeping(bool p_sleeping) {
	sleeping = p_sleeping;
	PhysicsServer3D::get_singleton()->body_set_state(get_rid(), PhysicsServer3D::BODY_STATE_SLEEPING, sleeping);
}

void RigidDynamicBody3D::set_can_sleep(bool p_active) {
	can_sleep = p_active;
	PhysicsServer3D::get_singleton()->body_set_state(get_rid(), PhysicsServer3D::BODY_STATE_CAN_SLEEP, p_active);
}

bool RigidDynamicBody3D::is_able_to_sleep() const {
	return can_sleep;
}

bool RigidDynamicBody3D::is_sleeping() const {
	return sleeping;
}

void RigidDynamicBody3D::set_max_contacts_reported(int p_amount) {
	max_contacts_reported = p_amount;
	PhysicsServer3D::get_singleton()->body_set_max_contacts_reported(get_rid(), p_amount);
}

int RigidDynamicBody3D::get_max_contacts_reported() const {
	return max_contacts_reported;
}

void RigidDynamicBody3D::add_central_force(const Vector3 &p_force) {
	PhysicsServer3D::get_singleton()->body_add_central_force(get_rid(), p_force);
}

void RigidDynamicBody3D::add_force(const Vector3 &p_force, const Vector3 &p_position) {
	PhysicsServer3D *singleton = PhysicsServer3D::get_singleton();
	singleton->body_add_force(get_rid(), p_force, p_position);
}

void RigidDynamicBody3D::add_torque(const Vector3 &p_torque) {
	PhysicsServer3D::get_singleton()->body_add_torque(get_rid(), p_torque);
}

void RigidDynamicBody3D::apply_central_impulse(const Vector3 &p_impulse) {
	PhysicsServer3D::get_singleton()->body_apply_central_impulse(get_rid(), p_impulse);
}

void RigidDynamicBody3D::apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position) {
	PhysicsServer3D *singleton = PhysicsServer3D::get_singleton();
	singleton->body_apply_impulse(get_rid(), p_impulse, p_position);
}

void RigidDynamicBody3D::apply_torque_impulse(const Vector3 &p_impulse) {
	PhysicsServer3D::get_singleton()->body_apply_torque_impulse(get_rid(), p_impulse);
}

void RigidDynamicBody3D::set_use_continuous_collision_detection(bool p_enable) {
	ccd = p_enable;
	PhysicsServer3D::get_singleton()->body_set_enable_continuous_collision_detection(get_rid(), p_enable);
}

bool RigidDynamicBody3D::is_using_continuous_collision_detection() const {
	return ccd;
}

void RigidDynamicBody3D::set_contact_monitor(bool p_enabled) {
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
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &RigidDynamicBody3D::_body_enter_tree));
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &RigidDynamicBody3D::_body_exit_tree));
			}
		}

		memdelete(contact_monitor);
		contact_monitor = nullptr;
	} else {
		contact_monitor = memnew(ContactMonitor);
		contact_monitor->locked = false;
	}
}

bool RigidDynamicBody3D::is_contact_monitor_enabled() const {
	return contact_monitor != nullptr;
}

Array RigidDynamicBody3D::get_colliding_bodies() const {
	ERR_FAIL_COND_V(!contact_monitor, Array());

	Array ret;
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

TypedArray<String> RigidDynamicBody3D::get_configuration_warnings() const {
	Transform3D t = get_transform();

	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (ABS(t.basis.get_axis(0).length() - 1.0) > 0.05 || ABS(t.basis.get_axis(1).length() - 1.0) > 0.05 || ABS(t.basis.get_axis(2).length() - 1.0) > 0.05) {
		warnings.push_back(TTR("Size changes to RigidDynamicBody will be overridden by the physics engine when running.\nChange the size in children collision shapes instead."));
	}

	return warnings;
}

void RigidDynamicBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &RigidDynamicBody3D::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &RigidDynamicBody3D::get_mass);

	ClassDB::bind_method(D_METHOD("set_inertia", "inertia"), &RigidDynamicBody3D::set_inertia);
	ClassDB::bind_method(D_METHOD("get_inertia"), &RigidDynamicBody3D::get_inertia);

	ClassDB::bind_method(D_METHOD("set_center_of_mass_mode", "mode"), &RigidDynamicBody3D::set_center_of_mass_mode);
	ClassDB::bind_method(D_METHOD("get_center_of_mass_mode"), &RigidDynamicBody3D::get_center_of_mass_mode);

	ClassDB::bind_method(D_METHOD("set_center_of_mass", "center_of_mass"), &RigidDynamicBody3D::set_center_of_mass);
	ClassDB::bind_method(D_METHOD("get_center_of_mass"), &RigidDynamicBody3D::get_center_of_mass);

	ClassDB::bind_method(D_METHOD("set_physics_material_override", "physics_material_override"), &RigidDynamicBody3D::set_physics_material_override);
	ClassDB::bind_method(D_METHOD("get_physics_material_override"), &RigidDynamicBody3D::get_physics_material_override);

	ClassDB::bind_method(D_METHOD("set_linear_velocity", "linear_velocity"), &RigidDynamicBody3D::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &RigidDynamicBody3D::get_linear_velocity);

	ClassDB::bind_method(D_METHOD("set_angular_velocity", "angular_velocity"), &RigidDynamicBody3D::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &RigidDynamicBody3D::get_angular_velocity);

	ClassDB::bind_method(D_METHOD("get_inverse_inertia_tensor"), &RigidDynamicBody3D::get_inverse_inertia_tensor);

	ClassDB::bind_method(D_METHOD("set_gravity_scale", "gravity_scale"), &RigidDynamicBody3D::set_gravity_scale);
	ClassDB::bind_method(D_METHOD("get_gravity_scale"), &RigidDynamicBody3D::get_gravity_scale);

	ClassDB::bind_method(D_METHOD("set_linear_damp_mode", "linear_damp_mode"), &RigidDynamicBody3D::set_linear_damp_mode);
	ClassDB::bind_method(D_METHOD("get_linear_damp_mode"), &RigidDynamicBody3D::get_linear_damp_mode);

	ClassDB::bind_method(D_METHOD("set_angular_damp_mode", "angular_damp_mode"), &RigidDynamicBody3D::set_angular_damp_mode);
	ClassDB::bind_method(D_METHOD("get_angular_damp_mode"), &RigidDynamicBody3D::get_angular_damp_mode);

	ClassDB::bind_method(D_METHOD("set_linear_damp", "linear_damp"), &RigidDynamicBody3D::set_linear_damp);
	ClassDB::bind_method(D_METHOD("get_linear_damp"), &RigidDynamicBody3D::get_linear_damp);

	ClassDB::bind_method(D_METHOD("set_angular_damp", "angular_damp"), &RigidDynamicBody3D::set_angular_damp);
	ClassDB::bind_method(D_METHOD("get_angular_damp"), &RigidDynamicBody3D::get_angular_damp);

	ClassDB::bind_method(D_METHOD("set_max_contacts_reported", "amount"), &RigidDynamicBody3D::set_max_contacts_reported);
	ClassDB::bind_method(D_METHOD("get_max_contacts_reported"), &RigidDynamicBody3D::get_max_contacts_reported);

	ClassDB::bind_method(D_METHOD("set_use_custom_integrator", "enable"), &RigidDynamicBody3D::set_use_custom_integrator);
	ClassDB::bind_method(D_METHOD("is_using_custom_integrator"), &RigidDynamicBody3D::is_using_custom_integrator);

	ClassDB::bind_method(D_METHOD("set_contact_monitor", "enabled"), &RigidDynamicBody3D::set_contact_monitor);
	ClassDB::bind_method(D_METHOD("is_contact_monitor_enabled"), &RigidDynamicBody3D::is_contact_monitor_enabled);

	ClassDB::bind_method(D_METHOD("set_use_continuous_collision_detection", "enable"), &RigidDynamicBody3D::set_use_continuous_collision_detection);
	ClassDB::bind_method(D_METHOD("is_using_continuous_collision_detection"), &RigidDynamicBody3D::is_using_continuous_collision_detection);

	ClassDB::bind_method(D_METHOD("set_axis_velocity", "axis_velocity"), &RigidDynamicBody3D::set_axis_velocity);

	ClassDB::bind_method(D_METHOD("add_central_force", "force"), &RigidDynamicBody3D::add_central_force);
	ClassDB::bind_method(D_METHOD("add_force", "force", "position"), &RigidDynamicBody3D::add_force, Vector3());
	ClassDB::bind_method(D_METHOD("add_torque", "torque"), &RigidDynamicBody3D::add_torque);

	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &RigidDynamicBody3D::apply_central_impulse);
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "position"), &RigidDynamicBody3D::apply_impulse, Vector3());
	ClassDB::bind_method(D_METHOD("apply_torque_impulse", "impulse"), &RigidDynamicBody3D::apply_torque_impulse);

	ClassDB::bind_method(D_METHOD("set_sleeping", "sleeping"), &RigidDynamicBody3D::set_sleeping);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &RigidDynamicBody3D::is_sleeping);

	ClassDB::bind_method(D_METHOD("set_can_sleep", "able_to_sleep"), &RigidDynamicBody3D::set_can_sleep);
	ClassDB::bind_method(D_METHOD("is_able_to_sleep"), &RigidDynamicBody3D::is_able_to_sleep);

	ClassDB::bind_method(D_METHOD("set_lock_rotation_enabled", "lock_rotation"), &RigidDynamicBody3D::set_lock_rotation_enabled);
	ClassDB::bind_method(D_METHOD("is_lock_rotation_enabled"), &RigidDynamicBody3D::is_lock_rotation_enabled);

	ClassDB::bind_method(D_METHOD("set_freeze_enabled", "freeze_mode"), &RigidDynamicBody3D::set_freeze_enabled);
	ClassDB::bind_method(D_METHOD("is_freeze_enabled"), &RigidDynamicBody3D::is_freeze_enabled);

	ClassDB::bind_method(D_METHOD("set_freeze_mode", "freeze_mode"), &RigidDynamicBody3D::set_freeze_mode);
	ClassDB::bind_method(D_METHOD("get_freeze_mode"), &RigidDynamicBody3D::get_freeze_mode);

	ClassDB::bind_method(D_METHOD("get_colliding_bodies"), &RigidDynamicBody3D::get_colliding_bodies);

	GDVIRTUAL_BIND(_integrate_forces, "state");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass", PROPERTY_HINT_RANGE, "0.01,1000,0.01,or_greater,exp"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "inertia", PROPERTY_HINT_RANGE, "0,1000,0.01,or_greater,exp"), "set_inertia", "get_inertia");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "center_of_mass_mode", PROPERTY_HINT_ENUM, "Auto,Custom", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_center_of_mass_mode", "get_center_of_mass_mode");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "center_of_mass", PROPERTY_HINT_RANGE, "-10,10,0.01,or_lesser,or_greater"), "set_center_of_mass", "get_center_of_mass");
	ADD_LINKED_PROPERTY("center_of_mass_mode", "center_of_mass");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material_override", "get_physics_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gravity_scale", PROPERTY_HINT_RANGE, "-128,128,0.01"), "set_gravity_scale", "get_gravity_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "custom_integrator"), "set_use_custom_integrator", "is_using_custom_integrator");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "continuous_cd"), "set_use_continuous_collision_detection", "is_using_continuous_collision_detection");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "contacts_reported", PROPERTY_HINT_RANGE, "0,64,1,or_greater"), "set_max_contacts_reported", "get_max_contacts_reported");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "contact_monitor"), "set_contact_monitor", "is_contact_monitor_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleeping", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "can_sleep"), "set_can_sleep", "is_able_to_sleep");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lock_rotation"), "set_lock_rotation_enabled", "is_lock_rotation_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "freeze"), "set_freeze_enabled", "is_freeze_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "freeze_mode", PROPERTY_HINT_ENUM, "Static,Kinematic"), "set_freeze_mode", "get_freeze_mode");
	ADD_GROUP("Linear", "linear_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "linear_damp_mode", PROPERTY_HINT_ENUM, "Combine,Replace"), "set_linear_damp_mode", "get_linear_damp_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "linear_damp", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"), "set_linear_damp", "get_linear_damp");
	ADD_GROUP("Angular", "angular_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "angular_damp_mode", PROPERTY_HINT_ENUM, "Combine,Replace"), "set_angular_damp_mode", "get_angular_damp_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_damp", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"), "set_angular_damp", "get_angular_damp");

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
}

void RigidDynamicBody3D::_validate_property(PropertyInfo &property) const {
	if (center_of_mass_mode != CENTER_OF_MASS_MODE_CUSTOM) {
		if (property.name == "center_of_mass") {
			property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}
}

RigidDynamicBody3D::RigidDynamicBody3D() :
		PhysicsBody3D(PhysicsServer3D::BODY_MODE_DYNAMIC) {
	PhysicsServer3D::get_singleton()->body_set_state_sync_callback(get_rid(), this, _body_state_changed_callback);
}

RigidDynamicBody3D::~RigidDynamicBody3D() {
	if (contact_monitor) {
		memdelete(contact_monitor);
	}
}

void RigidDynamicBody3D::_reload_physics_characteristics() {
	if (physics_material_override.is_null()) {
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_BOUNCE, 0);
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_FRICTION, 1);
	} else {
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_BOUNCE, physics_material_override->computed_bounce());
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_FRICTION, physics_material_override->computed_friction());
	}
}

///////////////////////////////////////

//so, if you pass 45 as limit, avoid numerical precision errors when angle is 45.
#define FLOOR_ANGLE_THRESHOLD 0.01

bool CharacterBody3D::move_and_slide() {
	// Hack in order to work with calling from _process as well as from _physics_process; calling from thread is risky
	double delta = Engine::get_singleton()->is_in_physics_frame() ? get_physics_process_delta_time() : get_process_delta_time();

	for (int i = 0; i < 3; i++) {
		if (locked_axis & (1 << i)) {
			motion_velocity[i] = 0.0;
		}
	}

	Transform3D gt = get_global_transform();
	previous_position = gt.origin;

	Vector3 current_platform_velocity = platform_velocity;

	if ((collision_state.floor || collision_state.wall) && platform_rid.is_valid()) {
		bool excluded = false;
		if (collision_state.floor) {
			excluded = (moving_platform_floor_layers & platform_layer) == 0;
		} else if (collision_state.wall) {
			excluded = (moving_platform_wall_layers & platform_layer) == 0;
		}
		if (!excluded) {
			//this approach makes sure there is less delay between the actual body velocity and the one we saved
			PhysicsDirectBodyState3D *bs = PhysicsServer3D::get_singleton()->body_get_direct_state(platform_rid);
			if (bs) {
				Vector3 local_position = gt.origin - bs->get_transform().origin;
				current_platform_velocity = bs->get_velocity_at_local_position(local_position);
			}
		} else {
			current_platform_velocity = Vector3();
		}
	}

	motion_results.clear();

	bool was_on_floor = collision_state.floor;
	collision_state.state = 0;

	last_motion = Vector3();

	if (!current_platform_velocity.is_equal_approx(Vector3())) {
		PhysicsServer3D::MotionParameters parameters(get_global_transform(), current_platform_velocity * delta, margin);
		parameters.exclude_bodies.insert(platform_rid);
		if (platform_object_id.is_valid()) {
			parameters.exclude_objects.insert(platform_object_id);
		}

		PhysicsServer3D::MotionResult floor_result;
		if (move_and_collide(parameters, floor_result, false, false)) {
			motion_results.push_back(floor_result);

			CollisionState result_state;
			_set_collision_direction(floor_result, result_state);
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
		if (!collision_state.floor && !collision_state.wall) {
			if (moving_platform_apply_velocity_on_leave == PLATFORM_VEL_ON_LEAVE_UPWARD_ONLY && current_platform_velocity.dot(up_direction) < 0) {
				current_platform_velocity = current_platform_velocity.slide(up_direction);
			}
			motion_velocity += current_platform_velocity;
		}
	}

	return motion_results.size() > 0;
}

void CharacterBody3D::_move_and_slide_grounded(double p_delta, bool p_was_on_floor) {
	Vector3 motion = motion_velocity * p_delta;
	Vector3 motion_slide_up = motion.slide(up_direction);
	Vector3 prev_floor_normal = floor_normal;

	platform_rid = RID();
	platform_object_id = ObjectID();
	platform_velocity = Vector3();
	platform_ceiling_velocity = Vector3();
	floor_normal = Vector3();
	wall_normal = Vector3();
	ceiling_normal = Vector3();

	// No sliding on first attempt to keep floor motion stable when possible,
	// When stop on slope is enabled or when there is no up direction.
	bool sliding_enabled = !floor_stop_on_slope;
	// Constant speed can be applied only the first time sliding is enabled.
	bool can_apply_constant_speed = sliding_enabled;
	// If the platform's ceiling push down the body.
	bool apply_ceiling_velocity = false;
	bool first_slide = true;
	bool vel_dir_facing_up = motion_velocity.dot(up_direction) > 0;
	Vector3 total_travel;

	for (int iteration = 0; iteration < max_slides; ++iteration) {
		PhysicsServer3D::MotionParameters parameters(get_global_transform(), motion, margin);
		parameters.max_collisions = 4;

		PhysicsServer3D::MotionResult result;
		bool collided = move_and_collide(parameters, result, false, !sliding_enabled);

		last_motion = result.travel;

		if (collided) {
			motion_results.push_back(result);

			CollisionState previous_state = collision_state;

			CollisionState result_state;
			_set_collision_direction(result, result_state);

			// If we hit a ceiling platform, we set the vertical motion_velocity to at least the platform one.
			if (collision_state.ceiling && platform_ceiling_velocity != Vector3() && platform_ceiling_velocity.dot(up_direction) < 0) {
				// If ceiling sliding is on, only apply when the ceiling is flat or when the motion is upward.
				if (!slide_on_ceiling || motion.dot(up_direction) < 0 || (ceiling_normal + up_direction).length() < 0.01) {
					apply_ceiling_velocity = true;
					Vector3 ceiling_vertical_velocity = up_direction * up_direction.dot(platform_ceiling_velocity);
					Vector3 motion_vertical_velocity = up_direction * up_direction.dot(motion_velocity);
					if (motion_vertical_velocity.dot(up_direction) > 0 || ceiling_vertical_velocity.length_squared() > motion_vertical_velocity.length_squared()) {
						motion_velocity = ceiling_vertical_velocity + motion_velocity.slide(up_direction);
					}
				}
			}

			if (collision_state.floor && floor_stop_on_slope && (motion_velocity.normalized() + up_direction).length() < 0.01) {
				Transform3D gt = get_global_transform();
				if (result.travel.length() <= margin + CMP_EPSILON) {
					gt.origin -= result.travel;
				}
				set_global_transform(gt);
				motion_velocity = Vector3();
				motion = Vector3();
				last_motion = Vector3();
				break;
			}

			if (result.remainder.is_equal_approx(Vector3())) {
				motion = Vector3();
				break;
			}

			// Apply regular sliding by default.
			bool apply_default_sliding = true;

			// Wall collision checks.
			if (result_state.wall && (motion_slide_up.dot(wall_normal) <= 0)) {
				// Move on floor only checks.
				if (floor_block_on_wall) {
					// Needs horizontal motion from current motion instead of motion_slide_up
					// to properly test the angle and avoid standing on slopes
					Vector3 horizontal_motion = motion.slide(up_direction);
					Vector3 horizontal_normal = wall_normal.slide(up_direction).normalized();
					real_t motion_angle = Math::abs(Math::acos(-horizontal_normal.dot(horizontal_motion.normalized())));

					// Avoid to move forward on a wall if floor_block_on_wall is true.
					// Applies only when the motion angle is under 90 degrees,
					// in order to avoid blocking lateral motion along a wall.
					if (motion_angle < .5 * Math_PI) {
						apply_default_sliding = false;

						if (p_was_on_floor && !vel_dir_facing_up) {
							// Cancel the motion.
							Transform3D gt = get_global_transform();
							real_t travel_total = result.travel.length();
							real_t cancel_dist_max = MIN(0.1, margin * 20);
							if (travel_total <= margin + CMP_EPSILON) {
								gt.origin -= result.travel;
							} else if (travel_total < cancel_dist_max) { // If the movement is large the body can be prevented from reaching the walls.
								gt.origin -= result.travel.slide(up_direction);
								// Keep remaining motion in sync with amount canceled.
								motion = motion.slide(up_direction);
							}
							set_global_transform(gt);
							result.travel = Vector3(); // Cancel for constant speed computation.

							// Determines if you are on the ground, and limits the possibility of climbing on the walls because of the approximations.
							_snap_on_floor(true, false);
						} else {
							// If the movement is not cancelled we only keep the remaining.
							motion = result.remainder;
						}

						// Apply slide on forward in order to allow only lateral motion on next step.
						Vector3 forward = wall_normal.slide(up_direction).normalized();
						motion = motion.slide(forward);
						// Avoid accelerating when you jump on the wall and smooth falling.
						motion_velocity = motion_velocity.slide(forward);

						// Allow only lateral motion along previous floor when already on floor.
						// Fixes slowing down when moving in diagonal against an inclined wall.
						if (p_was_on_floor && !vel_dir_facing_up && (motion.dot(up_direction) > 0.0)) {
							// Slide along the corner between the wall and previous floor.
							Vector3 floor_side = prev_floor_normal.cross(wall_normal);
							if (floor_side != Vector3()) {
								motion = floor_side * motion.dot(floor_side);
							}
						}

						// Stop all motion when a second wall is hit (unless sliding down or jumping),
						// in order to avoid jittering in corner cases.
						bool stop_all_motion = previous_state.wall && !vel_dir_facing_up;

						// Allow sliding when the body falls.
						if (!collision_state.floor && motion.dot(up_direction) < 0) {
							Vector3 slide_motion = motion.slide(wall_normal);
							// Test again to allow sliding only if the result goes downwards.
							// Fixes jittering issues at the bottom of inclined walls.
							if (slide_motion.dot(up_direction) < 0) {
								stop_all_motion = false;
								motion = slide_motion;
							}
						}

						if (stop_all_motion) {
							motion = Vector3();
							motion_velocity = Vector3();
						}
					}
				}

				// Stop horizontal motion when under wall slide threshold.
				if (p_was_on_floor && (wall_min_slide_angle > 0.0) && result_state.wall) {
					Vector3 horizontal_normal = wall_normal.slide(up_direction).normalized();
					real_t motion_angle = Math::abs(Math::acos(-horizontal_normal.dot(motion_slide_up.normalized())));
					if (motion_angle < wall_min_slide_angle) {
						motion = up_direction * motion.dot(up_direction);
						motion_velocity = up_direction * motion_velocity.dot(up_direction);

						apply_default_sliding = false;
					}
				}
			}

			if (apply_default_sliding) {
				// Regular sliding, the last part of the test handle the case when you don't want to slide on the ceiling.
				if ((sliding_enabled || !collision_state.floor) && (!collision_state.ceiling || slide_on_ceiling || !vel_dir_facing_up) && !apply_ceiling_velocity) {
					const PhysicsServer3D::MotionCollision &collision = result.collisions[0];

					Vector3 slide_motion = result.remainder.slide(collision.normal);
					if (collision_state.floor && !collision_state.wall) {
						// Slide using the intersection between the motion plane and the floor plane,
						// in order to keep the direction intact.
						real_t motion_length = slide_motion.length();
						slide_motion = up_direction.cross(result.remainder).cross(floor_normal);

						// Keep the length from default slide to change speed in slopes by default,
						// when constant speed is not enabled.
						slide_motion.normalize();
						slide_motion *= motion_length;
					}

					if (slide_motion.dot(motion_velocity) > 0.0) {
						motion = slide_motion;
					} else {
						motion = Vector3();
					}

					if (slide_on_ceiling && result_state.ceiling) {
						// Apply slide only in the direction of the input motion, otherwise just stop to avoid jittering when moving against a wall.
						if (vel_dir_facing_up) {
							motion_velocity = motion_velocity.slide(collision.normal);
						} else {
							// Avoid acceleration in slope when falling.
							motion_velocity = up_direction * up_direction.dot(motion_velocity);
						}
					}
				}
				// No sliding on first attempt to keep floor motion stable when possible.
				else {
					motion = result.remainder;
					if (result_state.ceiling && !slide_on_ceiling && vel_dir_facing_up) {
						motion_velocity = motion_velocity.slide(up_direction);
						motion = motion.slide(up_direction);
					}
				}
			}

			total_travel += result.travel;

			// Apply Constant Speed.
			if (p_was_on_floor && floor_constant_speed && can_apply_constant_speed && collision_state.floor && !motion.is_equal_approx(Vector3())) {
				Vector3 travel_slide_up = total_travel.slide(up_direction);
				motion = motion.normalized() * MAX(0, (motion_slide_up.length() - travel_slide_up.length()));
			}
		}
		// When you move forward in a downward slope you don’t collide because you will be in the air.
		// This test ensures that constant speed is applied, only if the player is still on the ground after the snap is applied.
		else if (floor_constant_speed && first_slide && _on_floor_if_snapped(p_was_on_floor, vel_dir_facing_up)) {
			can_apply_constant_speed = false;
			sliding_enabled = true;
			Transform3D gt = get_global_transform();
			gt.origin = gt.origin - result.travel;
			set_global_transform(gt);

			// Slide using the intersection between the motion plane and the floor plane,
			// in order to keep the direction intact.
			Vector3 motion_slide_norm = up_direction.cross(motion).cross(prev_floor_normal);
			motion_slide_norm.normalize();

			motion = motion_slide_norm * (motion_slide_up.length());
			collided = true;
		}

		if (!collided || motion.is_equal_approx(Vector3())) {
			break;
		}

		can_apply_constant_speed = !can_apply_constant_speed && !sliding_enabled;
		sliding_enabled = true;
		first_slide = false;
	}

	_snap_on_floor(p_was_on_floor, vel_dir_facing_up);

	// Reset the gravity accumulation when touching the ground.
	if (collision_state.floor && !vel_dir_facing_up) {
		motion_velocity = motion_velocity.slide(up_direction);
	}
}

void CharacterBody3D::_move_and_slide_free(double p_delta) {
	Vector3 motion = motion_velocity * p_delta;

	platform_rid = RID();
	platform_object_id = ObjectID();
	floor_normal = Vector3();
	platform_velocity = Vector3();

	bool first_slide = true;
	for (int iteration = 0; iteration < max_slides; ++iteration) {
		PhysicsServer3D::MotionParameters parameters(get_global_transform(), motion, margin);

		PhysicsServer3D::MotionResult result;
		bool collided = move_and_collide(parameters, result, false, false);

		last_motion = result.travel;

		if (collided) {
			motion_results.push_back(result);

			CollisionState result_state;
			_set_collision_direction(result, result_state);

			if (result.remainder.is_equal_approx(Vector3())) {
				motion = Vector3();
				break;
			}

			if (wall_min_slide_angle != 0 && Math::acos(wall_normal.dot(-motion_velocity.normalized())) < wall_min_slide_angle + FLOOR_ANGLE_THRESHOLD) {
				motion = Vector3();
				if (result.travel.length() < margin + CMP_EPSILON) {
					Transform3D gt = get_global_transform();
					gt.origin -= result.travel;
					set_global_transform(gt);
				}
			} else if (first_slide) {
				Vector3 motion_slide_norm = result.remainder.slide(wall_normal).normalized();
				motion = motion_slide_norm * (motion.length() - result.travel.length());
			} else {
				motion = result.remainder.slide(wall_normal);
			}

			if (motion.dot(motion_velocity) <= 0.0) {
				motion = Vector3();
			}
		}

		if (!collided || motion.is_equal_approx(Vector3())) {
			break;
		}

		first_slide = false;
	}
}

void CharacterBody3D::_snap_on_floor(bool was_on_floor, bool vel_dir_facing_up) {
	if (collision_state.floor || !was_on_floor || vel_dir_facing_up) {
		return;
	}

	// Snap by at least collision margin to keep floor state consistent.
	real_t length = MAX(floor_snap_length, margin);

	PhysicsServer3D::MotionParameters parameters(get_global_transform(), -up_direction * length, margin);
	parameters.max_collisions = 4;
	parameters.collide_separation_ray = true;

	PhysicsServer3D::MotionResult result;
	if (move_and_collide(parameters, result, true, false)) {
		CollisionState result_state;
		// Apply direction for floor only.
		_set_collision_direction(result, result_state, CollisionState(true, false, false));

		if (result_state.floor) {
			if (floor_stop_on_slope) {
				// move and collide may stray the object a bit because of pre un-stucking,
				// so only ensure that motion happens on floor direction in this case.
				if (result.travel.length() > margin) {
					result.travel = up_direction * up_direction.dot(result.travel);
				} else {
					result.travel = Vector3();
				}
			}

			parameters.from.origin += result.travel;
			set_global_transform(parameters.from);
		}
	}
}

bool CharacterBody3D::_on_floor_if_snapped(bool was_on_floor, bool vel_dir_facing_up) {
	if (up_direction == Vector3() || collision_state.floor || !was_on_floor || vel_dir_facing_up) {
		return false;
	}

	// Snap by at least collision margin to keep floor state consistent.
	real_t length = MAX(floor_snap_length, margin);

	PhysicsServer3D::MotionParameters parameters(get_global_transform(), -up_direction * length, margin);
	parameters.max_collisions = 4;
	parameters.collide_separation_ray = true;

	PhysicsServer3D::MotionResult result;
	if (move_and_collide(parameters, result, true, false)) {
		CollisionState result_state;
		// Don't apply direction for any type.
		_set_collision_direction(result, result_state, CollisionState());

		return result_state.floor;
	}

	return false;
}

void CharacterBody3D::_set_collision_direction(const PhysicsServer3D::MotionResult &p_result, CollisionState &r_state, CollisionState p_apply_state) {
	r_state.state = 0;

	real_t wall_depth = -1.0;
	real_t floor_depth = -1.0;

	bool was_on_wall = collision_state.wall;
	Vector3 prev_wall_normal = wall_normal;
	int wall_collision_count = 0;
	Vector3 combined_wall_normal;

	for (int i = p_result.collision_count - 1; i >= 0; i--) {
		const PhysicsServer3D::MotionCollision &collision = p_result.collisions[i];

		if (motion_mode == MOTION_MODE_GROUNDED) {
			// Check if any collision is floor.
			real_t floor_angle = collision.get_angle(up_direction);
			if (floor_angle <= floor_max_angle + FLOOR_ANGLE_THRESHOLD) {
				r_state.floor = true;
				if (p_apply_state.floor && collision.depth > floor_depth) {
					collision_state.floor = true;
					floor_normal = collision.normal;
					floor_depth = collision.depth;
					_set_platform_data(collision);
				}
				continue;
			}

			// Check if any collision is ceiling.
			real_t ceiling_angle = collision.get_angle(-up_direction);
			if (ceiling_angle <= floor_max_angle + FLOOR_ANGLE_THRESHOLD) {
				r_state.ceiling = true;
				if (p_apply_state.ceiling) {
					platform_ceiling_velocity = collision.collider_velocity;
					ceiling_normal = collision.normal;
					collision_state.ceiling = true;
				}
				continue;
			}
		}

		// Collision is wall by default.
		r_state.wall = true;

		if (p_apply_state.wall && collision.depth > wall_depth) {
			collision_state.wall = true;
			wall_depth = collision.depth;
			wall_normal = collision.normal;

			// Don't apply wall velocity when the collider is a CharacterBody3D.
			if (Object::cast_to<CharacterBody3D>(ObjectDB::get_instance(collision.collider_id)) == nullptr) {
				_set_platform_data(collision);
			}
		}

		// Collect normal for calculating average.
		combined_wall_normal += collision.normal;
		wall_collision_count++;
	}

	if (r_state.wall) {
		if (wall_collision_count > 1 && !r_state.floor) {
			// Check if wall normals cancel out to floor support.
			if (!r_state.floor && motion_mode == MOTION_MODE_GROUNDED) {
				combined_wall_normal.normalize();
				real_t floor_angle = Math::acos(combined_wall_normal.dot(up_direction));
				if (floor_angle <= floor_max_angle + FLOOR_ANGLE_THRESHOLD) {
					r_state.floor = true;
					r_state.wall = false;
					if (p_apply_state.floor) {
						collision_state.floor = true;
						floor_normal = combined_wall_normal;
					}
					if (p_apply_state.wall) {
						collision_state.wall = was_on_wall;
						wall_normal = prev_wall_normal;
					}
					return;
				}
			}
		}
	}
}

void CharacterBody3D::_set_platform_data(const PhysicsServer3D::MotionCollision &p_collision) {
	platform_rid = p_collision.collider;
	platform_object_id = p_collision.collider_id;
	platform_velocity = p_collision.collider_velocity;
	platform_layer = PhysicsServer3D::get_singleton()->body_get_collision_layer(platform_rid);
}

void CharacterBody3D::set_safe_margin(real_t p_margin) {
	margin = p_margin;
}

real_t CharacterBody3D::get_safe_margin() const {
	return margin;
}

const Vector3 &CharacterBody3D::get_motion_velocity() const {
	return motion_velocity;
}

void CharacterBody3D::set_motion_velocity(const Vector3 &p_velocity) {
	motion_velocity = p_velocity;
}

bool CharacterBody3D::is_on_floor() const {
	return collision_state.floor;
}

bool CharacterBody3D::is_on_floor_only() const {
	return collision_state.floor && !collision_state.wall && !collision_state.ceiling;
}

bool CharacterBody3D::is_on_wall() const {
	return collision_state.wall;
}

bool CharacterBody3D::is_on_wall_only() const {
	return collision_state.wall && !collision_state.floor && !collision_state.ceiling;
}

bool CharacterBody3D::is_on_ceiling() const {
	return collision_state.ceiling;
}

bool CharacterBody3D::is_on_ceiling_only() const {
	return collision_state.ceiling && !collision_state.floor && !collision_state.wall;
}

const Vector3 &CharacterBody3D::get_floor_normal() const {
	return floor_normal;
}

const Vector3 &CharacterBody3D::get_wall_normal() const {
	return wall_normal;
}

const Vector3 &CharacterBody3D::get_last_motion() const {
	return last_motion;
}

Vector3 CharacterBody3D::get_position_delta() const {
	return get_transform().origin - previous_position;
}

const Vector3 &CharacterBody3D::get_real_velocity() const {
	return real_velocity;
}

real_t CharacterBody3D::get_floor_angle(const Vector3 &p_up_direction) const {
	ERR_FAIL_COND_V(p_up_direction == Vector3(), 0);
	return Math::acos(floor_normal.dot(p_up_direction));
}

const Vector3 &CharacterBody3D::get_platform_velocity() const {
	return platform_velocity;
}

Vector3 CharacterBody3D::get_linear_velocity() const {
	return get_real_velocity();
}

int CharacterBody3D::get_slide_collision_count() const {
	return motion_results.size();
}

PhysicsServer3D::MotionResult CharacterBody3D::get_slide_collision(int p_bounce) const {
	ERR_FAIL_INDEX_V(p_bounce, motion_results.size(), PhysicsServer3D::MotionResult());
	return motion_results[p_bounce];
}

Ref<KinematicCollision3D> CharacterBody3D::_get_slide_collision(int p_bounce) {
	ERR_FAIL_INDEX_V(p_bounce, motion_results.size(), Ref<KinematicCollision3D>());
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

Ref<KinematicCollision3D> CharacterBody3D::_get_last_slide_collision() {
	if (motion_results.size() == 0) {
		return Ref<KinematicCollision3D>();
	}
	return _get_slide_collision(motion_results.size() - 1);
}

bool CharacterBody3D::is_floor_stop_on_slope_enabled() const {
	return floor_stop_on_slope;
}

void CharacterBody3D::set_floor_stop_on_slope_enabled(bool p_enabled) {
	floor_stop_on_slope = p_enabled;
}

bool CharacterBody3D::is_floor_constant_speed_enabled() const {
	return floor_constant_speed;
}

void CharacterBody3D::set_floor_constant_speed_enabled(bool p_enabled) {
	floor_constant_speed = p_enabled;
}

bool CharacterBody3D::is_floor_block_on_wall_enabled() const {
	return floor_block_on_wall;
}

void CharacterBody3D::set_floor_block_on_wall_enabled(bool p_enabled) {
	floor_block_on_wall = p_enabled;
}

bool CharacterBody3D::is_slide_on_ceiling_enabled() const {
	return slide_on_ceiling;
}

void CharacterBody3D::set_slide_on_ceiling_enabled(bool p_enabled) {
	slide_on_ceiling = p_enabled;
}

uint32_t CharacterBody3D::get_moving_platform_floor_layers() const {
	return moving_platform_floor_layers;
}

void CharacterBody3D::set_moving_platform_floor_layers(uint32_t p_exclude_layers) {
	moving_platform_floor_layers = p_exclude_layers;
}

uint32_t CharacterBody3D::get_moving_platform_wall_layers() const {
	return moving_platform_wall_layers;
}

void CharacterBody3D::set_moving_platform_wall_layers(uint32_t p_exclude_layers) {
	moving_platform_wall_layers = p_exclude_layers;
}

void CharacterBody3D::set_motion_mode(MotionMode p_mode) {
	motion_mode = p_mode;
}

CharacterBody3D::MotionMode CharacterBody3D::get_motion_mode() const {
	return motion_mode;
}

void CharacterBody3D::set_moving_platform_apply_velocity_on_leave(MovingPlatformApplyVelocityOnLeave p_on_leave_apply_velocity) {
	moving_platform_apply_velocity_on_leave = p_on_leave_apply_velocity;
}

CharacterBody3D::MovingPlatformApplyVelocityOnLeave CharacterBody3D::get_moving_platform_apply_velocity_on_leave() const {
	return moving_platform_apply_velocity_on_leave;
}

int CharacterBody3D::get_max_slides() const {
	return max_slides;
}

void CharacterBody3D::set_max_slides(int p_max_slides) {
	ERR_FAIL_COND(p_max_slides < 1);
	max_slides = p_max_slides;
}

real_t CharacterBody3D::get_floor_max_angle() const {
	return floor_max_angle;
}

void CharacterBody3D::set_floor_max_angle(real_t p_radians) {
	floor_max_angle = p_radians;
}

real_t CharacterBody3D::get_floor_snap_length() {
	return floor_snap_length;
}

void CharacterBody3D::set_floor_snap_length(real_t p_floor_snap_length) {
	ERR_FAIL_COND(p_floor_snap_length < 0);
	floor_snap_length = p_floor_snap_length;
}

real_t CharacterBody3D::get_wall_min_slide_angle() const {
	return wall_min_slide_angle;
}

void CharacterBody3D::set_wall_min_slide_angle(real_t p_radians) {
	wall_min_slide_angle = p_radians;
}

const Vector3 &CharacterBody3D::get_up_direction() const {
	return up_direction;
}

void CharacterBody3D::set_up_direction(const Vector3 &p_up_direction) {
	ERR_FAIL_COND_MSG(p_up_direction == Vector3(), "up_direction can't be equal to Vector3.ZERO, consider using Free motion mode instead.");
	up_direction = p_up_direction.normalized();
}

void CharacterBody3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// Reset move_and_slide() data.
			collision_state.state = 0;
			platform_rid = RID();
			platform_object_id = ObjectID();
			motion_results.clear();
			platform_velocity = Vector3();
		} break;
	}
}

void CharacterBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("move_and_slide"), &CharacterBody3D::move_and_slide);

	ClassDB::bind_method(D_METHOD("set_motion_velocity", "motion_velocity"), &CharacterBody3D::set_motion_velocity);
	ClassDB::bind_method(D_METHOD("get_motion_velocity"), &CharacterBody3D::get_motion_velocity);

	ClassDB::bind_method(D_METHOD("set_safe_margin", "pixels"), &CharacterBody3D::set_safe_margin);
	ClassDB::bind_method(D_METHOD("get_safe_margin"), &CharacterBody3D::get_safe_margin);
	ClassDB::bind_method(D_METHOD("is_floor_stop_on_slope_enabled"), &CharacterBody3D::is_floor_stop_on_slope_enabled);
	ClassDB::bind_method(D_METHOD("set_floor_stop_on_slope_enabled", "enabled"), &CharacterBody3D::set_floor_stop_on_slope_enabled);
	ClassDB::bind_method(D_METHOD("set_floor_constant_speed_enabled", "enabled"), &CharacterBody3D::set_floor_constant_speed_enabled);
	ClassDB::bind_method(D_METHOD("is_floor_constant_speed_enabled"), &CharacterBody3D::is_floor_constant_speed_enabled);
	ClassDB::bind_method(D_METHOD("set_floor_block_on_wall_enabled", "enabled"), &CharacterBody3D::set_floor_block_on_wall_enabled);
	ClassDB::bind_method(D_METHOD("is_floor_block_on_wall_enabled"), &CharacterBody3D::is_floor_block_on_wall_enabled);
	ClassDB::bind_method(D_METHOD("set_slide_on_ceiling_enabled", "enabled"), &CharacterBody3D::set_slide_on_ceiling_enabled);
	ClassDB::bind_method(D_METHOD("is_slide_on_ceiling_enabled"), &CharacterBody3D::is_slide_on_ceiling_enabled);

	ClassDB::bind_method(D_METHOD("set_moving_platform_floor_layers", "exclude_layer"), &CharacterBody3D::set_moving_platform_floor_layers);
	ClassDB::bind_method(D_METHOD("get_moving_platform_floor_layers"), &CharacterBody3D::get_moving_platform_floor_layers);
	ClassDB::bind_method(D_METHOD("set_moving_platform_wall_layers", "exclude_layer"), &CharacterBody3D::set_moving_platform_wall_layers);
	ClassDB::bind_method(D_METHOD("get_moving_platform_wall_layers"), &CharacterBody3D::get_moving_platform_wall_layers);

	ClassDB::bind_method(D_METHOD("get_max_slides"), &CharacterBody3D::get_max_slides);
	ClassDB::bind_method(D_METHOD("set_max_slides", "max_slides"), &CharacterBody3D::set_max_slides);
	ClassDB::bind_method(D_METHOD("get_floor_max_angle"), &CharacterBody3D::get_floor_max_angle);
	ClassDB::bind_method(D_METHOD("set_floor_max_angle", "radians"), &CharacterBody3D::set_floor_max_angle);
	ClassDB::bind_method(D_METHOD("get_floor_snap_length"), &CharacterBody3D::get_floor_snap_length);
	ClassDB::bind_method(D_METHOD("set_floor_snap_length", "floor_snap_length"), &CharacterBody3D::set_floor_snap_length);
	ClassDB::bind_method(D_METHOD("get_wall_min_slide_angle"), &CharacterBody3D::get_wall_min_slide_angle);
	ClassDB::bind_method(D_METHOD("set_wall_min_slide_angle", "radians"), &CharacterBody3D::set_wall_min_slide_angle);
	ClassDB::bind_method(D_METHOD("get_up_direction"), &CharacterBody3D::get_up_direction);
	ClassDB::bind_method(D_METHOD("set_up_direction", "up_direction"), &CharacterBody3D::set_up_direction);
	ClassDB::bind_method(D_METHOD("set_motion_mode", "mode"), &CharacterBody3D::set_motion_mode);
	ClassDB::bind_method(D_METHOD("get_motion_mode"), &CharacterBody3D::get_motion_mode);
	ClassDB::bind_method(D_METHOD("set_moving_platform_apply_velocity_on_leave", "on_leave_apply_velocity"), &CharacterBody3D::set_moving_platform_apply_velocity_on_leave);
	ClassDB::bind_method(D_METHOD("get_moving_platform_apply_velocity_on_leave"), &CharacterBody3D::get_moving_platform_apply_velocity_on_leave);

	ClassDB::bind_method(D_METHOD("is_on_floor"), &CharacterBody3D::is_on_floor);
	ClassDB::bind_method(D_METHOD("is_on_floor_only"), &CharacterBody3D::is_on_floor_only);
	ClassDB::bind_method(D_METHOD("is_on_ceiling"), &CharacterBody3D::is_on_ceiling);
	ClassDB::bind_method(D_METHOD("is_on_ceiling_only"), &CharacterBody3D::is_on_ceiling_only);
	ClassDB::bind_method(D_METHOD("is_on_wall"), &CharacterBody3D::is_on_wall);
	ClassDB::bind_method(D_METHOD("is_on_wall_only"), &CharacterBody3D::is_on_wall_only);
	ClassDB::bind_method(D_METHOD("get_floor_normal"), &CharacterBody3D::get_floor_normal);
	ClassDB::bind_method(D_METHOD("get_wall_normal"), &CharacterBody3D::get_wall_normal);
	ClassDB::bind_method(D_METHOD("get_last_motion"), &CharacterBody3D::get_last_motion);
	ClassDB::bind_method(D_METHOD("get_position_delta"), &CharacterBody3D::get_position_delta);
	ClassDB::bind_method(D_METHOD("get_real_velocity"), &CharacterBody3D::get_real_velocity);
	ClassDB::bind_method(D_METHOD("get_floor_angle", "up_direction"), &CharacterBody3D::get_floor_angle, DEFVAL(Vector3(0.0, 1.0, 0.0)));
	ClassDB::bind_method(D_METHOD("get_platform_velocity"), &CharacterBody3D::get_platform_velocity);
	ClassDB::bind_method(D_METHOD("get_slide_collision_count"), &CharacterBody3D::get_slide_collision_count);
	ClassDB::bind_method(D_METHOD("get_slide_collision", "slide_idx"), &CharacterBody3D::_get_slide_collision);
	ClassDB::bind_method(D_METHOD("get_last_slide_collision"), &CharacterBody3D::_get_last_slide_collision);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "motion_mode", PROPERTY_HINT_ENUM, "Grounded,Free", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_motion_mode", "get_motion_mode");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "up_direction"), "set_up_direction", "get_up_direction");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "slide_on_ceiling"), "set_slide_on_ceiling_enabled", "is_slide_on_ceiling_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "motion_velocity", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_motion_velocity", "get_motion_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_slides", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_max_slides", "get_max_slides");
	ADD_GROUP("Free Mode", "free_mode_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "wall_min_slide_angle", PROPERTY_HINT_RANGE, "0,180,0.1,radians", PROPERTY_USAGE_DEFAULT), "set_wall_min_slide_angle", "get_wall_min_slide_angle");
	ADD_GROUP("Floor", "floor_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "floor_stop_on_slope"), "set_floor_stop_on_slope_enabled", "is_floor_stop_on_slope_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "floor_constant_speed"), "set_floor_constant_speed_enabled", "is_floor_constant_speed_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "floor_block_on_wall"), "set_floor_block_on_wall_enabled", "is_floor_block_on_wall_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "floor_max_angle", PROPERTY_HINT_RANGE, "0,180,0.1,radians"), "set_floor_max_angle", "get_floor_max_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "floor_snap_length", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater"), "set_floor_snap_length", "get_floor_snap_length");
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

void CharacterBody3D::_validate_property(PropertyInfo &property) const {
	if (motion_mode == MOTION_MODE_FREE) {
		if (property.name.begins_with("floor_") || property.name == "up_direction" || property.name == "slide_on_ceiling") {
			property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}
}

CharacterBody3D::CharacterBody3D() :
		PhysicsBody3D(PhysicsServer3D::BODY_MODE_KINEMATIC) {
}

CharacterBody3D::~CharacterBody3D() {
	for (int i = 0; i < slide_colliders.size(); i++) {
		if (slide_colliders[i].is_valid()) {
			slide_colliders.write[i]->owner = nullptr;
		}
	}
}

///////////////////////////////////////

Vector3 KinematicCollision3D::get_travel() const {
	return result.travel;
}

Vector3 KinematicCollision3D::get_remainder() const {
	return result.remainder;
}

int KinematicCollision3D::get_collision_count() const {
	return result.collision_count;
}

Vector3 KinematicCollision3D::get_position(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, Vector3());
	return result.collisions[p_collision_index].position;
}

Vector3 KinematicCollision3D::get_normal(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, Vector3());
	return result.collisions[p_collision_index].normal;
}

real_t KinematicCollision3D::get_angle(int p_collision_index, const Vector3 &p_up_direction) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, 0.0);
	ERR_FAIL_COND_V(p_up_direction == Vector3(), 0);
	return result.collisions[p_collision_index].get_angle(p_up_direction);
}

Object *KinematicCollision3D::get_local_shape(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, nullptr);
	if (!owner) {
		return nullptr;
	}
	uint32_t ownerid = owner->shape_find_owner(result.collisions[p_collision_index].local_shape);
	return owner->shape_owner_get_owner(ownerid);
}

Object *KinematicCollision3D::get_collider(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, nullptr);
	if (result.collisions[p_collision_index].collider_id.is_valid()) {
		return ObjectDB::get_instance(result.collisions[p_collision_index].collider_id);
	}

	return nullptr;
}

ObjectID KinematicCollision3D::get_collider_id(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, ObjectID());
	return result.collisions[p_collision_index].collider_id;
}

RID KinematicCollision3D::get_collider_rid(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, RID());
	return result.collisions[p_collision_index].collider;
}

Object *KinematicCollision3D::get_collider_shape(int p_collision_index) const {
	Object *collider = get_collider(p_collision_index);
	if (collider) {
		CollisionObject3D *obj2d = Object::cast_to<CollisionObject3D>(collider);
		if (obj2d) {
			uint32_t ownerid = obj2d->shape_find_owner(result.collisions[p_collision_index].collider_shape);
			return obj2d->shape_owner_get_owner(ownerid);
		}
	}

	return nullptr;
}

int KinematicCollision3D::get_collider_shape_index(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, 0);
	return result.collisions[p_collision_index].collider_shape;
}

Vector3 KinematicCollision3D::get_collider_velocity(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, Vector3());
	return result.collisions[p_collision_index].collider_velocity;
}

void KinematicCollision3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_travel"), &KinematicCollision3D::get_travel);
	ClassDB::bind_method(D_METHOD("get_remainder"), &KinematicCollision3D::get_remainder);
	ClassDB::bind_method(D_METHOD("get_collision_count"), &KinematicCollision3D::get_collision_count);
	ClassDB::bind_method(D_METHOD("get_position", "collision_index"), &KinematicCollision3D::get_position, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_normal", "collision_index"), &KinematicCollision3D::get_normal, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_angle", "collision_index", "up_direction"), &KinematicCollision3D::get_angle, DEFVAL(0), DEFVAL(Vector3(0.0, 1.0, 0.0)));
	ClassDB::bind_method(D_METHOD("get_local_shape", "collision_index"), &KinematicCollision3D::get_local_shape, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider", "collision_index"), &KinematicCollision3D::get_collider, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider_id", "collision_index"), &KinematicCollision3D::get_collider_id, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider_rid", "collision_index"), &KinematicCollision3D::get_collider_rid, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider_shape", "collision_index"), &KinematicCollision3D::get_collider_shape, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider_shape_index", "collision_index"), &KinematicCollision3D::get_collider_shape_index, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider_velocity", "collision_index"), &KinematicCollision3D::get_collider_velocity, DEFVAL(0));
}

///////////////////////////////////////

bool PhysicalBone3D::JointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
	return false;
}

bool PhysicalBone3D::JointData::_get(const StringName &p_name, Variant &r_ret) const {
	return false;
}

void PhysicalBone3D::JointData::_get_property_list(List<PropertyInfo> *p_list) const {
}

void PhysicalBone3D::apply_central_impulse(const Vector3 &p_impulse) {
	PhysicsServer3D::get_singleton()->body_apply_central_impulse(get_rid(), p_impulse);
}

void PhysicalBone3D::apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position) {
	PhysicsServer3D::get_singleton()->body_apply_impulse(get_rid(), p_impulse, p_position);
}

void PhysicalBone3D::reset_physics_simulation_state() {
	if (simulate_physics) {
		_start_physics_simulation();
	} else {
		_stop_physics_simulation();
	}
}

void PhysicalBone3D::reset_to_rest_position() {
	if (parent_skeleton) {
		if (-1 == bone_id) {
			set_global_transform(parent_skeleton->get_global_transform() * body_offset);
		} else {
			set_global_transform(parent_skeleton->get_global_transform() * parent_skeleton->get_bone_global_pose(bone_id) * body_offset);
		}
	}
}

bool PhysicalBone3D::PinJointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
	if (JointData::_set(p_name, p_value, j)) {
		return true;
	}

	if ("joint_constraints/bias" == p_name) {
		bias = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->pin_joint_set_param(j, PhysicsServer3D::PIN_JOINT_BIAS, bias);
		}

	} else if ("joint_constraints/damping" == p_name) {
		damping = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->pin_joint_set_param(j, PhysicsServer3D::PIN_JOINT_DAMPING, damping);
		}

	} else if ("joint_constraints/impulse_clamp" == p_name) {
		impulse_clamp = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->pin_joint_set_param(j, PhysicsServer3D::PIN_JOINT_IMPULSE_CLAMP, impulse_clamp);
		}

	} else {
		return false;
	}

	return true;
}

bool PhysicalBone3D::PinJointData::_get(const StringName &p_name, Variant &r_ret) const {
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

void PhysicalBone3D::PinJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);

	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/bias", PROPERTY_HINT_RANGE, "0.01,0.99,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/damping", PROPERTY_HINT_RANGE, "0.01,8.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/impulse_clamp", PROPERTY_HINT_RANGE, "0.0,64.0,0.01"));
}

bool PhysicalBone3D::ConeJointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
	if (JointData::_set(p_name, p_value, j)) {
		return true;
	}

	if ("joint_constraints/swing_span" == p_name) {
		swing_span = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(j, PhysicsServer3D::CONE_TWIST_JOINT_SWING_SPAN, swing_span);
		}

	} else if ("joint_constraints/twist_span" == p_name) {
		twist_span = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(j, PhysicsServer3D::CONE_TWIST_JOINT_TWIST_SPAN, twist_span);
		}

	} else if ("joint_constraints/bias" == p_name) {
		bias = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(j, PhysicsServer3D::CONE_TWIST_JOINT_BIAS, bias);
		}

	} else if ("joint_constraints/softness" == p_name) {
		softness = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(j, PhysicsServer3D::CONE_TWIST_JOINT_SOFTNESS, softness);
		}

	} else if ("joint_constraints/relaxation" == p_name) {
		relaxation = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(j, PhysicsServer3D::CONE_TWIST_JOINT_RELAXATION, relaxation);
		}

	} else {
		return false;
	}

	return true;
}

bool PhysicalBone3D::ConeJointData::_get(const StringName &p_name, Variant &r_ret) const {
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

void PhysicalBone3D::ConeJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);

	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/swing_span", PROPERTY_HINT_RANGE, "-180,180,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/twist_span", PROPERTY_HINT_RANGE, "-40000,40000,0.1,or_lesser,or_greater"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/bias", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/relaxation", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
}

bool PhysicalBone3D::HingeJointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
	if (JointData::_set(p_name, p_value, j)) {
		return true;
	}

	if ("joint_constraints/angular_limit_enabled" == p_name) {
		angular_limit_enabled = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->hinge_joint_set_flag(j, PhysicsServer3D::HINGE_JOINT_FLAG_USE_LIMIT, angular_limit_enabled);
		}

	} else if ("joint_constraints/angular_limit_upper" == p_name) {
		angular_limit_upper = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->hinge_joint_set_param(j, PhysicsServer3D::HINGE_JOINT_LIMIT_UPPER, angular_limit_upper);
		}

	} else if ("joint_constraints/angular_limit_lower" == p_name) {
		angular_limit_lower = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->hinge_joint_set_param(j, PhysicsServer3D::HINGE_JOINT_LIMIT_LOWER, angular_limit_lower);
		}

	} else if ("joint_constraints/angular_limit_bias" == p_name) {
		angular_limit_bias = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->hinge_joint_set_param(j, PhysicsServer3D::HINGE_JOINT_LIMIT_BIAS, angular_limit_bias);
		}

	} else if ("joint_constraints/angular_limit_softness" == p_name) {
		angular_limit_softness = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->hinge_joint_set_param(j, PhysicsServer3D::HINGE_JOINT_LIMIT_SOFTNESS, angular_limit_softness);
		}

	} else if ("joint_constraints/angular_limit_relaxation" == p_name) {
		angular_limit_relaxation = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->hinge_joint_set_param(j, PhysicsServer3D::HINGE_JOINT_LIMIT_RELAXATION, angular_limit_relaxation);
		}

	} else {
		return false;
	}

	return true;
}

bool PhysicalBone3D::HingeJointData::_get(const StringName &p_name, Variant &r_ret) const {
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

void PhysicalBone3D::HingeJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);

	p_list->push_back(PropertyInfo(Variant::BOOL, "joint_constraints/angular_limit_enabled"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/angular_limit_upper", PROPERTY_HINT_RANGE, "-180,180,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/angular_limit_lower", PROPERTY_HINT_RANGE, "-180,180,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/angular_limit_bias", PROPERTY_HINT_RANGE, "0.01,0.99,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/angular_limit_softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/angular_limit_relaxation", PROPERTY_HINT_RANGE, "0.01,16,0.01"));
}

bool PhysicalBone3D::SliderJointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
	if (JointData::_set(p_name, p_value, j)) {
		return true;
	}

	if ("joint_constraints/linear_limit_upper" == p_name) {
		linear_limit_upper = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->slider_joint_set_param(j, PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_UPPER, linear_limit_upper);
		}

	} else if ("joint_constraints/linear_limit_lower" == p_name) {
		linear_limit_lower = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->slider_joint_set_param(j, PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_LOWER, linear_limit_lower);
		}

	} else if ("joint_constraints/linear_limit_softness" == p_name) {
		linear_limit_softness = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->slider_joint_set_param(j, PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS, linear_limit_softness);
		}

	} else if ("joint_constraints/linear_limit_restitution" == p_name) {
		linear_limit_restitution = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->slider_joint_set_param(j, PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION, linear_limit_restitution);
		}

	} else if ("joint_constraints/linear_limit_damping" == p_name) {
		linear_limit_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->slider_joint_set_param(j, PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_DAMPING, linear_limit_restitution);
		}

	} else if ("joint_constraints/angular_limit_upper" == p_name) {
		angular_limit_upper = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->slider_joint_set_param(j, PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_UPPER, angular_limit_upper);
		}

	} else if ("joint_constraints/angular_limit_lower" == p_name) {
		angular_limit_lower = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->slider_joint_set_param(j, PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_LOWER, angular_limit_lower);
		}

	} else if ("joint_constraints/angular_limit_softness" == p_name) {
		angular_limit_softness = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->slider_joint_set_param(j, PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS, angular_limit_softness);
		}

	} else if ("joint_constraints/angular_limit_restitution" == p_name) {
		angular_limit_restitution = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->slider_joint_set_param(j, PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS, angular_limit_softness);
		}

	} else if ("joint_constraints/angular_limit_damping" == p_name) {
		angular_limit_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->slider_joint_set_param(j, PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_DAMPING, angular_limit_damping);
		}

	} else {
		return false;
	}

	return true;
}

bool PhysicalBone3D::SliderJointData::_get(const StringName &p_name, Variant &r_ret) const {
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

void PhysicalBone3D::SliderJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	JointData::_get_property_list(p_list);

	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/linear_limit_upper"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/linear_limit_lower"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/linear_limit_softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/linear_limit_restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/linear_limit_damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"));

	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/angular_limit_upper", PROPERTY_HINT_RANGE, "-180,180,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/angular_limit_lower", PROPERTY_HINT_RANGE, "-180,180,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/angular_limit_softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/angular_limit_restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/angular_limit_damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"));
}

bool PhysicalBone3D::SixDOFJointData::_set(const StringName &p_name, const Variant &p_value, RID j) {
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
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(j, axis, PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT, axis_data[axis].linear_limit_enabled);
		}

	} else if ("linear_limit_upper" == var_name) {
		axis_data[axis].linear_limit_upper = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_LINEAR_UPPER_LIMIT, axis_data[axis].linear_limit_upper);
		}

	} else if ("linear_limit_lower" == var_name) {
		axis_data[axis].linear_limit_lower = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_LINEAR_LOWER_LIMIT, axis_data[axis].linear_limit_lower);
		}

	} else if ("linear_limit_softness" == var_name) {
		axis_data[axis].linear_limit_softness = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS, axis_data[axis].linear_limit_softness);
		}

	} else if ("linear_spring_enabled" == var_name) {
		axis_data[axis].linear_spring_enabled = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(j, axis, PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING, axis_data[axis].linear_spring_enabled);
		}

	} else if ("linear_spring_stiffness" == var_name) {
		axis_data[axis].linear_spring_stiffness = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_STIFFNESS, axis_data[axis].linear_spring_stiffness);
		}

	} else if ("linear_spring_damping" == var_name) {
		axis_data[axis].linear_spring_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_DAMPING, axis_data[axis].linear_spring_damping);
		}

	} else if ("linear_equilibrium_point" == var_name) {
		axis_data[axis].linear_equilibrium_point = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_EQUILIBRIUM_POINT, axis_data[axis].linear_equilibrium_point);
		}

	} else if ("linear_restitution" == var_name) {
		axis_data[axis].linear_restitution = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_LINEAR_RESTITUTION, axis_data[axis].linear_restitution);
		}

	} else if ("linear_damping" == var_name) {
		axis_data[axis].linear_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_LINEAR_DAMPING, axis_data[axis].linear_damping);
		}

	} else if ("angular_limit_enabled" == var_name) {
		axis_data[axis].angular_limit_enabled = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(j, axis, PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT, axis_data[axis].angular_limit_enabled);
		}

	} else if ("angular_limit_upper" == var_name) {
		axis_data[axis].angular_limit_upper = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_ANGULAR_UPPER_LIMIT, axis_data[axis].angular_limit_upper);
		}

	} else if ("angular_limit_lower" == var_name) {
		axis_data[axis].angular_limit_lower = Math::deg2rad(real_t(p_value));
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_ANGULAR_LOWER_LIMIT, axis_data[axis].angular_limit_lower);
		}

	} else if ("angular_limit_softness" == var_name) {
		axis_data[axis].angular_limit_softness = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS, axis_data[axis].angular_limit_softness);
		}

	} else if ("angular_restitution" == var_name) {
		axis_data[axis].angular_restitution = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_ANGULAR_RESTITUTION, axis_data[axis].angular_restitution);
		}

	} else if ("angular_damping" == var_name) {
		axis_data[axis].angular_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_ANGULAR_DAMPING, axis_data[axis].angular_damping);
		}

	} else if ("erp" == var_name) {
		axis_data[axis].erp = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_ANGULAR_ERP, axis_data[axis].erp);
		}

	} else if ("angular_spring_enabled" == var_name) {
		axis_data[axis].angular_spring_enabled = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(j, axis, PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING, axis_data[axis].angular_spring_enabled);
		}

	} else if ("angular_spring_stiffness" == var_name) {
		axis_data[axis].angular_spring_stiffness = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_STIFFNESS, axis_data[axis].angular_spring_stiffness);
		}

	} else if ("angular_spring_damping" == var_name) {
		axis_data[axis].angular_spring_damping = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_DAMPING, axis_data[axis].angular_spring_damping);
		}

	} else if ("angular_equilibrium_point" == var_name) {
		axis_data[axis].angular_equilibrium_point = p_value;
		if (j.is_valid()) {
			PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(j, axis, PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_EQUILIBRIUM_POINT, axis_data[axis].angular_equilibrium_point);
		}

	} else {
		return false;
	}

	return true;
}

bool PhysicalBone3D::SixDOFJointData::_get(const StringName &p_name, Variant &r_ret) const {
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

void PhysicalBone3D::SixDOFJointData::_get_property_list(List<PropertyInfo> *p_list) const {
	const StringName axis_names[] = { "x", "y", "z" };
	for (int i = 0; i < 3; ++i) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "joint_constraints/" + axis_names[i] + "/linear_limit_enabled"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/linear_limit_upper"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/linear_limit_lower"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/linear_limit_softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::BOOL, "joint_constraints/" + axis_names[i] + "/linear_spring_enabled"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/linear_spring_stiffness"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/linear_spring_damping"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/linear_equilibrium_point"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/linear_restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/linear_damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::BOOL, "joint_constraints/" + axis_names[i] + "/angular_limit_enabled"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/angular_limit_upper", PROPERTY_HINT_RANGE, "-180,180,0.01"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/angular_limit_lower", PROPERTY_HINT_RANGE, "-180,180,0.01"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/angular_limit_softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/angular_restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/angular_damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/erp"));
		p_list->push_back(PropertyInfo(Variant::BOOL, "joint_constraints/" + axis_names[i] + "/angular_spring_enabled"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/angular_spring_stiffness"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/angular_spring_damping"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_constraints/" + axis_names[i] + "/angular_equilibrium_point"));
	}
}

bool PhysicalBone3D::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "bone_name") {
		set_bone_name(p_value);
		return true;
	}

	if (joint_data) {
		if (joint_data->_set(p_name, p_value, joint)) {
#ifdef TOOLS_ENABLED
			update_gizmos();
#endif
			return true;
		}
	}

	return false;
}

bool PhysicalBone3D::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "bone_name") {
		r_ret = get_bone_name();
		return true;
	}

	if (joint_data) {
		return joint_data->_get(p_name, r_ret);
	}

	return false;
}

void PhysicalBone3D::_get_property_list(List<PropertyInfo> *p_list) const {
	Skeleton3D *parent = find_skeleton_parent(get_parent());

	if (parent) {
		String names;
		for (int i = 0; i < parent->get_bone_count(); i++) {
			if (i > 0) {
				names += ",";
			}
			names += parent->get_bone_name(i);
		}

		p_list->push_back(PropertyInfo(Variant::STRING_NAME, "bone_name", PROPERTY_HINT_ENUM, names));
	} else {
		p_list->push_back(PropertyInfo(Variant::STRING_NAME, "bone_name"));
	}

	if (joint_data) {
		joint_data->_get_property_list(p_list);
	}
}

void PhysicalBone3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
			parent_skeleton = find_skeleton_parent(get_parent());
			update_bone_id();
			reset_to_rest_position();
			reset_physics_simulation_state();
			if (joint_data) {
				_reload_joint();
			}
			break;

		case NOTIFICATION_EXIT_TREE: {
			if (parent_skeleton) {
				if (-1 != bone_id) {
					parent_skeleton->unbind_physical_bone_from_bone(bone_id);
					bone_id = -1;
				}
			}
			parent_skeleton = nullptr;
			PhysicsServer3D::get_singleton()->joint_clear(joint);
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (Engine::get_singleton()->is_editor_hint()) {
				update_offset();
			}
		} break;
	}
}

void PhysicalBone3D::_body_state_changed_callback(void *p_instance, PhysicsDirectBodyState3D *p_state) {
	PhysicalBone3D *bone = (PhysicalBone3D *)p_instance;
	bone->_body_state_changed(p_state);
}

void PhysicalBone3D::_body_state_changed(PhysicsDirectBodyState3D *p_state) {
	if (!simulate_physics || !_internal_simulate_physics) {
		return;
	}

	/// Update bone transform.

	Transform3D global_transform(p_state->get_transform());

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

void PhysicalBone3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &PhysicalBone3D::apply_central_impulse);
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "position"), &PhysicalBone3D::apply_impulse, Vector3());

	ClassDB::bind_method(D_METHOD("set_joint_type", "joint_type"), &PhysicalBone3D::set_joint_type);
	ClassDB::bind_method(D_METHOD("get_joint_type"), &PhysicalBone3D::get_joint_type);

	ClassDB::bind_method(D_METHOD("set_joint_offset", "offset"), &PhysicalBone3D::set_joint_offset);
	ClassDB::bind_method(D_METHOD("get_joint_offset"), &PhysicalBone3D::get_joint_offset);
	ClassDB::bind_method(D_METHOD("set_joint_rotation", "euler"), &PhysicalBone3D::set_joint_rotation);
	ClassDB::bind_method(D_METHOD("get_joint_rotation"), &PhysicalBone3D::get_joint_rotation);

	ClassDB::bind_method(D_METHOD("set_body_offset", "offset"), &PhysicalBone3D::set_body_offset);
	ClassDB::bind_method(D_METHOD("get_body_offset"), &PhysicalBone3D::get_body_offset);

	ClassDB::bind_method(D_METHOD("get_simulate_physics"), &PhysicalBone3D::get_simulate_physics);

	ClassDB::bind_method(D_METHOD("is_simulating_physics"), &PhysicalBone3D::is_simulating_physics);

	ClassDB::bind_method(D_METHOD("get_bone_id"), &PhysicalBone3D::get_bone_id);

	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &PhysicalBone3D::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &PhysicalBone3D::get_mass);

	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &PhysicalBone3D::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &PhysicalBone3D::get_friction);

	ClassDB::bind_method(D_METHOD("set_bounce", "bounce"), &PhysicalBone3D::set_bounce);
	ClassDB::bind_method(D_METHOD("get_bounce"), &PhysicalBone3D::get_bounce);

	ClassDB::bind_method(D_METHOD("set_gravity_scale", "gravity_scale"), &PhysicalBone3D::set_gravity_scale);
	ClassDB::bind_method(D_METHOD("get_gravity_scale"), &PhysicalBone3D::get_gravity_scale);

	ClassDB::bind_method(D_METHOD("set_linear_damp_mode", "linear_damp_mode"), &PhysicalBone3D::set_linear_damp_mode);
	ClassDB::bind_method(D_METHOD("get_linear_damp_mode"), &PhysicalBone3D::get_linear_damp_mode);

	ClassDB::bind_method(D_METHOD("set_angular_damp_mode", "angular_damp_mode"), &PhysicalBone3D::set_angular_damp_mode);
	ClassDB::bind_method(D_METHOD("get_angular_damp_mode"), &PhysicalBone3D::get_angular_damp_mode);

	ClassDB::bind_method(D_METHOD("set_linear_damp", "linear_damp"), &PhysicalBone3D::set_linear_damp);
	ClassDB::bind_method(D_METHOD("get_linear_damp"), &PhysicalBone3D::get_linear_damp);

	ClassDB::bind_method(D_METHOD("set_angular_damp", "angular_damp"), &PhysicalBone3D::set_angular_damp);
	ClassDB::bind_method(D_METHOD("get_angular_damp"), &PhysicalBone3D::get_angular_damp);

	ClassDB::bind_method(D_METHOD("set_can_sleep", "able_to_sleep"), &PhysicalBone3D::set_can_sleep);
	ClassDB::bind_method(D_METHOD("is_able_to_sleep"), &PhysicalBone3D::is_able_to_sleep);

	ADD_GROUP("Joint", "joint_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "joint_type", PROPERTY_HINT_ENUM, "None,PinJoint,ConeJoint,HingeJoint,SliderJoint,6DOFJoint"), "set_joint_type", "get_joint_type");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "joint_offset"), "set_joint_offset", "get_joint_offset");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "joint_rotation", PROPERTY_HINT_RANGE, "-360,360,0.01,or_lesser,or_greater,radians"), "set_joint_rotation", "get_joint_rotation");

	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "body_offset"), "set_body_offset", "get_body_offset");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass", PROPERTY_HINT_RANGE, "0.01,1000,0.01,or_greater,exp"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bounce", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_bounce", "get_bounce");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gravity_scale", PROPERTY_HINT_RANGE, "-10,10,0.01"), "set_gravity_scale", "get_gravity_scale");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "linear_damp_mode", PROPERTY_HINT_ENUM, "Combine,Replace"), "set_linear_damp_mode", "get_linear_damp_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "linear_damp", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"), "set_linear_damp", "get_linear_damp");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "angular_damp_mode", PROPERTY_HINT_ENUM, "Combine,Replace"), "set_angular_damp_mode", "get_angular_damp_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_damp", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"), "set_angular_damp", "get_angular_damp");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "can_sleep"), "set_can_sleep", "is_able_to_sleep");

	BIND_ENUM_CONSTANT(DAMP_MODE_COMBINE);
	BIND_ENUM_CONSTANT(DAMP_MODE_REPLACE);

	BIND_ENUM_CONSTANT(JOINT_TYPE_NONE);
	BIND_ENUM_CONSTANT(JOINT_TYPE_PIN);
	BIND_ENUM_CONSTANT(JOINT_TYPE_CONE);
	BIND_ENUM_CONSTANT(JOINT_TYPE_HINGE);
	BIND_ENUM_CONSTANT(JOINT_TYPE_SLIDER);
	BIND_ENUM_CONSTANT(JOINT_TYPE_6DOF);
}

Skeleton3D *PhysicalBone3D::find_skeleton_parent(Node *p_parent) {
	if (!p_parent) {
		return nullptr;
	}
	Skeleton3D *s = Object::cast_to<Skeleton3D>(p_parent);
	return s ? s : find_skeleton_parent(p_parent->get_parent());
}

void PhysicalBone3D::_update_joint_offset() {
	_fix_joint_offset();

	set_ignore_transform_notification(true);
	reset_to_rest_position();
	set_ignore_transform_notification(false);

#ifdef TOOLS_ENABLED
	update_gizmos();
#endif
}

void PhysicalBone3D::_fix_joint_offset() {
	// Clamp joint origin to bone origin
	if (parent_skeleton) {
		joint_offset.origin = body_offset.affine_inverse().origin;
	}
}

void PhysicalBone3D::_reload_joint() {
	if (!parent_skeleton) {
		PhysicsServer3D::get_singleton()->joint_clear(joint);
		return;
	}

	PhysicalBone3D *body_a = parent_skeleton->get_physical_bone_parent(bone_id);
	if (!body_a) {
		PhysicsServer3D::get_singleton()->joint_clear(joint);
		return;
	}

	Transform3D joint_transf = get_global_transform() * joint_offset;
	Transform3D local_a = body_a->get_global_transform().affine_inverse() * joint_transf;
	local_a.orthonormalize();

	switch (get_joint_type()) {
		case JOINT_TYPE_PIN: {
			PhysicsServer3D::get_singleton()->joint_make_pin(joint, body_a->get_rid(), local_a.origin, get_rid(), joint_offset.origin);
			const PinJointData *pjd(static_cast<const PinJointData *>(joint_data));
			PhysicsServer3D::get_singleton()->pin_joint_set_param(joint, PhysicsServer3D::PIN_JOINT_BIAS, pjd->bias);
			PhysicsServer3D::get_singleton()->pin_joint_set_param(joint, PhysicsServer3D::PIN_JOINT_DAMPING, pjd->damping);
			PhysicsServer3D::get_singleton()->pin_joint_set_param(joint, PhysicsServer3D::PIN_JOINT_IMPULSE_CLAMP, pjd->impulse_clamp);

		} break;
		case JOINT_TYPE_CONE: {
			PhysicsServer3D::get_singleton()->joint_make_cone_twist(joint, body_a->get_rid(), local_a, get_rid(), joint_offset);
			const ConeJointData *cjd(static_cast<const ConeJointData *>(joint_data));
			PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(joint, PhysicsServer3D::CONE_TWIST_JOINT_SWING_SPAN, cjd->swing_span);
			PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(joint, PhysicsServer3D::CONE_TWIST_JOINT_TWIST_SPAN, cjd->twist_span);
			PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(joint, PhysicsServer3D::CONE_TWIST_JOINT_BIAS, cjd->bias);
			PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(joint, PhysicsServer3D::CONE_TWIST_JOINT_SOFTNESS, cjd->softness);
			PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(joint, PhysicsServer3D::CONE_TWIST_JOINT_RELAXATION, cjd->relaxation);

		} break;
		case JOINT_TYPE_HINGE: {
			PhysicsServer3D::get_singleton()->joint_make_hinge(joint, body_a->get_rid(), local_a, get_rid(), joint_offset);
			const HingeJointData *hjd(static_cast<const HingeJointData *>(joint_data));
			PhysicsServer3D::get_singleton()->hinge_joint_set_flag(joint, PhysicsServer3D::HINGE_JOINT_FLAG_USE_LIMIT, hjd->angular_limit_enabled);
			PhysicsServer3D::get_singleton()->hinge_joint_set_param(joint, PhysicsServer3D::HINGE_JOINT_LIMIT_UPPER, hjd->angular_limit_upper);
			PhysicsServer3D::get_singleton()->hinge_joint_set_param(joint, PhysicsServer3D::HINGE_JOINT_LIMIT_LOWER, hjd->angular_limit_lower);
			PhysicsServer3D::get_singleton()->hinge_joint_set_param(joint, PhysicsServer3D::HINGE_JOINT_LIMIT_BIAS, hjd->angular_limit_bias);
			PhysicsServer3D::get_singleton()->hinge_joint_set_param(joint, PhysicsServer3D::HINGE_JOINT_LIMIT_SOFTNESS, hjd->angular_limit_softness);
			PhysicsServer3D::get_singleton()->hinge_joint_set_param(joint, PhysicsServer3D::HINGE_JOINT_LIMIT_RELAXATION, hjd->angular_limit_relaxation);

		} break;
		case JOINT_TYPE_SLIDER: {
			PhysicsServer3D::get_singleton()->joint_make_slider(joint, body_a->get_rid(), local_a, get_rid(), joint_offset);
			const SliderJointData *sjd(static_cast<const SliderJointData *>(joint_data));
			PhysicsServer3D::get_singleton()->slider_joint_set_param(joint, PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_UPPER, sjd->linear_limit_upper);
			PhysicsServer3D::get_singleton()->slider_joint_set_param(joint, PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_LOWER, sjd->linear_limit_lower);
			PhysicsServer3D::get_singleton()->slider_joint_set_param(joint, PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS, sjd->linear_limit_softness);
			PhysicsServer3D::get_singleton()->slider_joint_set_param(joint, PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION, sjd->linear_limit_restitution);
			PhysicsServer3D::get_singleton()->slider_joint_set_param(joint, PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_DAMPING, sjd->linear_limit_restitution);
			PhysicsServer3D::get_singleton()->slider_joint_set_param(joint, PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_UPPER, sjd->angular_limit_upper);
			PhysicsServer3D::get_singleton()->slider_joint_set_param(joint, PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_LOWER, sjd->angular_limit_lower);
			PhysicsServer3D::get_singleton()->slider_joint_set_param(joint, PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS, sjd->angular_limit_softness);
			PhysicsServer3D::get_singleton()->slider_joint_set_param(joint, PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS, sjd->angular_limit_softness);
			PhysicsServer3D::get_singleton()->slider_joint_set_param(joint, PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_DAMPING, sjd->angular_limit_damping);

		} break;
		case JOINT_TYPE_6DOF: {
			PhysicsServer3D::get_singleton()->joint_make_generic_6dof(joint, body_a->get_rid(), local_a, get_rid(), joint_offset);
			const SixDOFJointData *g6dofjd(static_cast<const SixDOFJointData *>(joint_data));
			for (int axis = 0; axis < 3; ++axis) {
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT, g6dofjd->axis_data[axis].linear_limit_enabled);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_LINEAR_UPPER_LIMIT, g6dofjd->axis_data[axis].linear_limit_upper);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_LINEAR_LOWER_LIMIT, g6dofjd->axis_data[axis].linear_limit_lower);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS, g6dofjd->axis_data[axis].linear_limit_softness);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING, g6dofjd->axis_data[axis].linear_spring_enabled);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_STIFFNESS, g6dofjd->axis_data[axis].linear_spring_stiffness);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_DAMPING, g6dofjd->axis_data[axis].linear_spring_damping);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_EQUILIBRIUM_POINT, g6dofjd->axis_data[axis].linear_equilibrium_point);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_LINEAR_RESTITUTION, g6dofjd->axis_data[axis].linear_restitution);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_LINEAR_DAMPING, g6dofjd->axis_data[axis].linear_damping);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT, g6dofjd->axis_data[axis].angular_limit_enabled);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_ANGULAR_UPPER_LIMIT, g6dofjd->axis_data[axis].angular_limit_upper);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_ANGULAR_LOWER_LIMIT, g6dofjd->axis_data[axis].angular_limit_lower);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS, g6dofjd->axis_data[axis].angular_limit_softness);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_ANGULAR_RESTITUTION, g6dofjd->axis_data[axis].angular_restitution);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_ANGULAR_DAMPING, g6dofjd->axis_data[axis].angular_damping);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_ANGULAR_ERP, g6dofjd->axis_data[axis].erp);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING, g6dofjd->axis_data[axis].angular_spring_enabled);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_STIFFNESS, g6dofjd->axis_data[axis].angular_spring_stiffness);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_DAMPING, g6dofjd->axis_data[axis].angular_spring_damping);
				PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(joint, static_cast<Vector3::Axis>(axis), PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_EQUILIBRIUM_POINT, g6dofjd->axis_data[axis].angular_equilibrium_point);
			}

		} break;
		case JOINT_TYPE_NONE: {
		} break;
	}
}

void PhysicalBone3D::_on_bone_parent_changed() {
	_reload_joint();
}

#ifdef TOOLS_ENABLED
void PhysicalBone3D::_set_gizmo_move_joint(bool p_move_joint) {
	gizmo_move_joint = p_move_joint;
}

Transform3D PhysicalBone3D::get_global_gizmo_transform() const {
	return gizmo_move_joint ? get_global_transform() * joint_offset : get_global_transform();
}

Transform3D PhysicalBone3D::get_local_gizmo_transform() const {
	return gizmo_move_joint ? get_transform() * joint_offset : get_transform();
}
#endif

const PhysicalBone3D::JointData *PhysicalBone3D::get_joint_data() const {
	return joint_data;
}

Skeleton3D *PhysicalBone3D::find_skeleton_parent() {
	return find_skeleton_parent(this);
}

void PhysicalBone3D::set_joint_type(JointType p_joint_type) {
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
	notify_property_list_changed();
	update_gizmos();
#endif
}

PhysicalBone3D::JointType PhysicalBone3D::get_joint_type() const {
	return joint_data ? joint_data->get_joint_type() : JOINT_TYPE_NONE;
}

void PhysicalBone3D::set_joint_offset(const Transform3D &p_offset) {
	joint_offset = p_offset;

	_update_joint_offset();
}

const Transform3D &PhysicalBone3D::get_joint_offset() const {
	return joint_offset;
}

void PhysicalBone3D::set_joint_rotation(const Vector3 &p_euler_rad) {
	joint_offset.basis.set_euler_scale(p_euler_rad, joint_offset.basis.get_scale());

	_update_joint_offset();
}

Vector3 PhysicalBone3D::get_joint_rotation() const {
	return joint_offset.basis.get_rotation();
}

const Transform3D &PhysicalBone3D::get_body_offset() const {
	return body_offset;
}

void PhysicalBone3D::set_body_offset(const Transform3D &p_offset) {
	body_offset = p_offset;
	body_offset_inverse = body_offset.affine_inverse();

	_update_joint_offset();
}

void PhysicalBone3D::set_simulate_physics(bool p_simulate) {
	if (simulate_physics == p_simulate) {
		return;
	}

	simulate_physics = p_simulate;
	reset_physics_simulation_state();
}

bool PhysicalBone3D::get_simulate_physics() {
	return simulate_physics;
}

bool PhysicalBone3D::is_simulating_physics() {
	return _internal_simulate_physics;
}

void PhysicalBone3D::set_bone_name(const String &p_name) {
	bone_name = p_name;
	bone_id = -1;

	update_bone_id();
	reset_to_rest_position();
}

const String &PhysicalBone3D::get_bone_name() const {
	return bone_name;
}

void PhysicalBone3D::set_mass(real_t p_mass) {
	ERR_FAIL_COND(p_mass <= 0);
	mass = p_mass;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_MASS, mass);
}

real_t PhysicalBone3D::get_mass() const {
	return mass;
}

void PhysicalBone3D::set_friction(real_t p_friction) {
	ERR_FAIL_COND(p_friction < 0 || p_friction > 1);

	friction = p_friction;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_FRICTION, friction);
}

real_t PhysicalBone3D::get_friction() const {
	return friction;
}

void PhysicalBone3D::set_bounce(real_t p_bounce) {
	ERR_FAIL_COND(p_bounce < 0 || p_bounce > 1);

	bounce = p_bounce;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_BOUNCE, bounce);
}

real_t PhysicalBone3D::get_bounce() const {
	return bounce;
}

void PhysicalBone3D::set_gravity_scale(real_t p_gravity_scale) {
	gravity_scale = p_gravity_scale;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_GRAVITY_SCALE, gravity_scale);
}

real_t PhysicalBone3D::get_gravity_scale() const {
	return gravity_scale;
}

void PhysicalBone3D::set_linear_damp_mode(DampMode p_mode) {
	linear_damp_mode = p_mode;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_LINEAR_DAMP_MODE, linear_damp_mode);
}

PhysicalBone3D::DampMode PhysicalBone3D::get_linear_damp_mode() const {
	return linear_damp_mode;
}

void PhysicalBone3D::set_angular_damp_mode(DampMode p_mode) {
	angular_damp_mode = p_mode;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_ANGULAR_DAMP_MODE, angular_damp_mode);
}

PhysicalBone3D::DampMode PhysicalBone3D::get_angular_damp_mode() const {
	return angular_damp_mode;
}

void PhysicalBone3D::set_linear_damp(real_t p_linear_damp) {
	ERR_FAIL_COND(p_linear_damp < 0);

	linear_damp = p_linear_damp;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_LINEAR_DAMP, linear_damp);
}

real_t PhysicalBone3D::get_linear_damp() const {
	return linear_damp;
}

void PhysicalBone3D::set_angular_damp(real_t p_angular_damp) {
	ERR_FAIL_COND(p_angular_damp < 0);

	angular_damp = p_angular_damp;
	PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_ANGULAR_DAMP, angular_damp);
}

real_t PhysicalBone3D::get_angular_damp() const {
	return angular_damp;
}

void PhysicalBone3D::set_can_sleep(bool p_active) {
	can_sleep = p_active;
	PhysicsServer3D::get_singleton()->body_set_state(get_rid(), PhysicsServer3D::BODY_STATE_CAN_SLEEP, p_active);
}

bool PhysicalBone3D::is_able_to_sleep() const {
	return can_sleep;
}

PhysicalBone3D::PhysicalBone3D() :
		PhysicsBody3D(PhysicsServer3D::BODY_MODE_STATIC) {
	joint = PhysicsServer3D::get_singleton()->joint_create();
	reset_physics_simulation_state();
}

PhysicalBone3D::~PhysicalBone3D() {
	if (joint_data) {
		memdelete(joint_data);
	}
	PhysicsServer3D::get_singleton()->free(joint);
}

void PhysicalBone3D::update_bone_id() {
	if (!parent_skeleton) {
		return;
	}

	const int new_bone_id = parent_skeleton->find_bone(bone_name);

	if (new_bone_id != bone_id) {
		if (-1 != bone_id) {
			// Assert the unbind from old node
			parent_skeleton->unbind_physical_bone_from_bone(bone_id);
		}

		bone_id = new_bone_id;

		parent_skeleton->bind_physical_bone_to_bone(bone_id, this);

		_fix_joint_offset();
		reset_physics_simulation_state();
	}
}

void PhysicalBone3D::update_offset() {
#ifdef TOOLS_ENABLED
	if (parent_skeleton) {
		Transform3D bone_transform(parent_skeleton->get_global_transform());
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

void PhysicalBone3D::_start_physics_simulation() {
	if (_internal_simulate_physics || !parent_skeleton) {
		return;
	}
	reset_to_rest_position();
	set_body_mode(PhysicsServer3D::BODY_MODE_DYNAMIC);
	PhysicsServer3D::get_singleton()->body_set_collision_layer(get_rid(), get_collision_layer());
	PhysicsServer3D::get_singleton()->body_set_collision_mask(get_rid(), get_collision_mask());
	PhysicsServer3D::get_singleton()->body_set_state_sync_callback(get_rid(), this, _body_state_changed_callback);
	set_as_top_level(true);
	_internal_simulate_physics = true;
}

void PhysicalBone3D::_stop_physics_simulation() {
	if (!parent_skeleton) {
		return;
	}
	if (parent_skeleton->get_animate_physical_bones()) {
		set_body_mode(PhysicsServer3D::BODY_MODE_KINEMATIC);
		PhysicsServer3D::get_singleton()->body_set_collision_layer(get_rid(), get_collision_layer());
		PhysicsServer3D::get_singleton()->body_set_collision_mask(get_rid(), get_collision_mask());
	} else {
		set_body_mode(PhysicsServer3D::BODY_MODE_STATIC);
		PhysicsServer3D::get_singleton()->body_set_collision_layer(get_rid(), 0);
		PhysicsServer3D::get_singleton()->body_set_collision_mask(get_rid(), 0);
	}
	if (_internal_simulate_physics) {
		PhysicsServer3D::get_singleton()->body_set_state_sync_callback(get_rid(), nullptr, nullptr);
		parent_skeleton->set_bone_global_pose_override(bone_id, Transform3D(), 0.0, false);
		set_as_top_level(false);
		_internal_simulate_physics = false;
	}
}
