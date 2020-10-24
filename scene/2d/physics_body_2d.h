/*************************************************************************/
/*  physics_body_2d.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef PHYSICS_BODY_2D_H
#define PHYSICS_BODY_2D_H

#include "core/vset.h"
#include "scene/2d/collision_object_2d.h"
#include "scene/resources/physics_material.h"
#include "servers/physics_server_2d.h"

class KinematicCollision2D;

class RigidBody2D : public CollisionObject2D {
	GDCLASS(RigidBody2D, CollisionObject2D);

public:
	enum Mode {
		MODE_RIGID,
		MODE_STATIC,
		MODE_CHARACTER,
		MODE_KINEMATIC,
	};

	enum CCDMode {
		CCD_MODE_DISABLED,
		CCD_MODE_CAST_RAY,
		CCD_MODE_CAST_SHAPE,
	};

	struct Collision {
		ObjectID collider;
		RID collider_rid;
		int collider_shape = 0;
		Vector2 collider_velocity;
		Variant collider_metadata;
		Vector2 point;
		Vector2 normal;
		Vector2 travel;
		Vector2 remainder;
		int local_shape = 0;
	};

private:
	struct ShapePair {
		int body_shape;
		int local_shape;
		bool tagged;

		bool operator<(const ShapePair &p_sp) const {
			if (body_shape == p_sp.body_shape) {
				return local_shape < p_sp.local_shape;
			}
			return body_shape < p_sp.body_shape;
		}

		ShapePair() {}
		ShapePair(int p_bs, int p_ls) {
			body_shape = p_bs;
			local_shape = p_ls;
		}
	};

	struct RigidBody2DInOut {
		ObjectID id;
		int shape;
		int local_shape;
	};

	struct RigidBody2D_RemoveAction {
		ObjectID body_id;
		ShapePair pair;
	};

	struct BodyState {
		bool in_scene;
		VSet<ShapePair> shapes;
	};

	struct ContactMonitor {
		bool locked;
		Map<ObjectID, BodyState> body_map;
	};

	PhysicsDirectBodyState2D *state = nullptr;

	Mode mode = MODE_RIGID;
	Ref<PhysicsMaterial> physics_material_override;
	real_t mass = 1;
	real_t gravity_scale = 1;

	Vector2 linear_velocity;
	real_t linear_damp = -1;

	real_t angular_velocity = 0;
	real_t angular_damp = -1;

	CCDMode ccd_mode = CCD_MODE_DISABLED;
	bool can_sleep = true;
	bool sleeping = false;
	bool custom_integrator = false;
	ContactMonitor *contact_monitor = nullptr;
	int max_contacts_reported = 0;

	float margin = 0.08;
	bool sync_to_physics = false;

	uint32_t collision_layer = 1;
	uint32_t collision_mask = 1;

	RID on_floor_body;
	Vector2 floor_normal;
	Vector2 floor_velocity;
	bool on_floor = false;
	bool on_ceiling = false;
	bool on_wall = false;

	Vector<Collision> colliders;
	Vector<Ref<KinematicCollision2D>> slide_colliders;
	Ref<KinematicCollision2D> motion_cache;
	Transform2D last_valid_transform;

	bool _test_motion(const Vector2 &p_motion, bool p_infinite_inertia = true, float p_margin = 0.08, const Ref<PhysicsTestMotionResult2D> &p_result = Ref<PhysicsTestMotionResult2D>());
	void _direct_state_changed(Object *p_state);

	void _body_enter_tree(ObjectID p_id);
	void _body_exit_tree(ObjectID p_id);
	void _body_inout(int p_status, ObjectID p_instance, int p_body_shape, int p_local_shape);
	void _reload_physics_characteristics();
	bool _separate_raycast_shapes(bool p_infinite_inertia, Collision &r_collision);
	Ref<KinematicCollision2D> _move(const Vector2 &p_motion, bool p_infinite_inertia = true, bool p_exclude_raycast_shapes = true, bool p_test_only = false);
	Ref<KinematicCollision2D> _get_slide_collision(int p_bounce);

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual String get_configuration_warning() const override;

public:
	void set_mode(Mode p_mode);
	Mode get_mode() const;

	void set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override);
	Ref<PhysicsMaterial> get_physics_material_override() const;

	void set_mass(real_t p_mass);
	real_t get_mass() const;

	void set_inertia(real_t p_inertia);
	real_t get_inertia() const;

	void set_weight(real_t p_weight);
	real_t get_weight() const;

	void set_gravity_scale(real_t p_gravity_scale);
	real_t get_gravity_scale() const;

	void set_linear_velocity(const Vector2 &p_velocity);
	void set_axis_velocity(const Vector2 &p_axis);
	Vector2 get_linear_velocity() const;

	void set_linear_damp(real_t p_linear_damp);
	real_t get_linear_damp() const;

	void set_angular_velocity(real_t p_velocity);
	real_t get_angular_velocity() const;

	void set_angular_damp(real_t p_angular_damp);
	real_t get_angular_damp() const;

	void set_continuous_collision_detection_mode(CCDMode p_mode);
	CCDMode get_continuous_collision_detection_mode() const;

	void set_can_sleep(bool p_active);
	bool is_able_to_sleep() const;

	void set_sleeping(bool p_sleeping);
	bool is_sleeping() const;

	void set_use_custom_integrator(bool p_enable);
	bool is_using_custom_integrator();

	void set_contact_monitor(bool p_enabled);
	bool is_contact_monitor_enabled() const;

	void set_max_contacts_reported(int p_amount);
	int get_max_contacts_reported() const;

	void set_applied_force(const Vector2 &p_force);
	Vector2 get_applied_force() const;

	void set_applied_torque(const float p_torque);
	float get_applied_torque() const;

	void add_force(const Vector2 &p_force, const Vector2 &p_position = Vector2());
	void add_central_force(const Vector2 &p_force);
	void add_torque(float p_torque);

	void apply_impulse(const Vector2 &p_impulse, const Vector2 &p_position = Vector2());
	void apply_central_impulse(const Vector2 &p_impulse);
	void apply_torque_impulse(float p_torque);

	void set_safe_margin(float p_margin);
	float get_safe_margin() const;

	void set_sync_to_physics(bool p_enable);
	bool is_sync_to_physics_enabled() const;

	void set_collision_layer(uint32_t p_layer);
	uint32_t get_collision_layer() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_collision_layer_bit(int p_bit, bool p_value);
	bool get_collision_layer_bit(int p_bit) const;

	void set_collision_mask_bit(int p_bit, bool p_value);
	bool get_collision_mask_bit(int p_bit) const;

	TypedArray<RigidBody2D> get_collision_exceptions();
	void add_collision_exception_with(Node *p_node); // Must be RigidBody
	void remove_collision_exception_with(Node *p_node);

	TypedArray<Node2D> get_colliding_bodies() const;

	bool is_on_floor() const;
	bool is_on_wall() const;
	bool is_on_ceiling() const;
	Vector2 get_floor_normal() const;
	Vector2 get_floor_velocity() const;

	bool test_move(const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia = true);
	bool move_and_collide(const Vector2 &p_motion, bool p_infinite_inertia, Collision &r_collision, bool p_exclude_raycast_shapes = true, bool p_test_only = false);
	Vector2 move_and_slide(const Vector2 &p_linear_velocity, const Vector2 &p_up_direction = Vector2(0, 0), bool p_stop_on_slope = false, int p_max_slides = 4, float p_floor_max_angle = Math::deg2rad((float)45), bool p_infinite_inertia = true);
	Vector2 move_and_slide_with_snap(const Vector2 &p_linear_velocity, const Vector2 &p_snap, const Vector2 &p_up_direction = Vector2(0, 0), bool p_stop_on_slope = false, int p_max_slides = 4, float p_floor_max_angle = Math::deg2rad((float)45), bool p_infinite_inertia = true);

	int get_slide_count() const;
	Collision get_slide_collision(int p_bounce) const;

	RigidBody2D();
	~RigidBody2D();
};

class KinematicCollision2D : public Reference {
	GDCLASS(KinematicCollision2D, Reference);

	friend class RigidBody2D;
	RigidBody2D *owner = nullptr;
	RigidBody2D::Collision collision;

protected:
	static void _bind_methods();

public:
	Object *get_collider() const;
	ObjectID get_collider_id() const;
	Object *get_collider_shape() const;
	int get_collider_shape_index() const;
	Vector2 get_collider_velocity() const;
	Variant get_collider_metadata() const;
	Vector2 get_position() const;
	Vector2 get_normal() const;
	Vector2 get_travel() const;
	Vector2 get_remainder() const;
	Object *get_local_shape() const;
};

VARIANT_ENUM_CAST(RigidBody2D::Mode);
VARIANT_ENUM_CAST(RigidBody2D::CCDMode);

#endif // PHYSICS_BODY_2D_H
