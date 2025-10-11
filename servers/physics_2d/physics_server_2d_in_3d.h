/**************************************************************************/
/*  physics_server_2d_in_3d.h                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "physics_server_2d.h"
#include "servers/physics_3d/physics_server_3d.h"

static constexpr float SCALE_2D_TO_3D = 0.1;
static constexpr float SCALE_3D_TO_2D = 1.0 / SCALE_2D_TO_3D;
static constexpr Vector3 PLANE_NORMAL_3D = Vector3(0, 0, 1);

#define VECTOR2TO3(m_vec2) (SCALE_2D_TO_3D * Vector3(-m_vec2.x, -m_vec2.y, 0))
#define VECTOR3TO2(m_vec3) (SCALE_3D_TO_2D * Vector2(-m_vec3.x, -m_vec3.y))
#define VECTOR2TO3_FRONT(m_vec2) (SCALE_2D_TO_3D * Vector3(-m_vec2.x, -m_vec2.y, 5))
#define VECTOR2TO3_BACK(m_vec2) (SCALE_2D_TO_3D * Vector3(-m_vec2.x, -m_vec2.y, -5))
#define VECTOR2TO3_UNSCALED(m_vec2) (Vector3(-m_vec2.x, -m_vec2.y, 0))
#define VECTOR3TO2_UNSCALED(m_vec3) (Vector2(-m_vec3.x, -m_vec3.y))
#define SIZE2TO3(m_size2) (SCALE_2D_TO_3D * Vector3(m_size2.x, m_size2.y, 5))
#define SIZE3TO2(m_size3) (SCALE_3D_TO_2D * Vector2(m_size.x, m_size.y))
#define ANGLE2TO3(m_angles2) (Vector3(0, 0, m_angles2))
#define ANGLE3TO2(m_angles3) (m_angles3.z)

/*
 * How should the 2D plane be embedded into 3D?
 * The one-way collision directions need to line up.
 * In 2D, the one-way collision arrow points down towards Y+, and in 3D it points down towards Y-.
 * So the embedding should be e.g. (x, y) -> (-x, -y, 0).
 * That's (x, y, 0) rotated by 180 degrees around the Z axis.
 */

class PhysicsDirectBodyState2Din3D : public PhysicsDirectBodyState2D {
	GDCLASS(PhysicsDirectBodyState2Din3D, PhysicsDirectBodyState2D);

	PhysicsDirectBodyState3D *state_3d;

public:
	virtual Vector2 get_total_gravity() const override {
		Vector3 gravity_3d = state_3d->get_total_gravity();
		return VECTOR3TO2(gravity_3d);
	}
	virtual real_t get_total_linear_damp() const override { return state_3d->get_total_linear_damp(); } // TODO check, should be ok
	virtual real_t get_total_angular_damp() const override { return state_3d->get_total_angular_damp(); } // TODO check, should be ok

	virtual Vector2 get_center_of_mass() const override {
		Vector3 com = state_3d->get_center_of_mass();
		return VECTOR3TO2(com);
	}
	virtual Vector2 get_center_of_mass_local() const override {
		Vector3 com = state_3d->get_center_of_mass_local();
		return VECTOR3TO2(com);
	}
	virtual real_t get_inverse_mass() const override { return state_3d->get_inverse_mass(); } // TODO check, wrong?
	virtual real_t get_inverse_inertia() const override { return state_3d->get_inverse_inertia().z; } // TODO check, wrong?

	virtual void set_linear_velocity(const Vector2 &p_velocity) override { state_3d->set_linear_velocity(VECTOR2TO3(p_velocity)); }
	virtual Vector2 get_linear_velocity() const override {
		Vector3 velocity_3d = state_3d->get_linear_velocity();
		return VECTOR3TO2(velocity_3d);
	}

	virtual void set_angular_velocity(real_t p_velocity) override { state_3d->set_angular_velocity(ANGLE2TO3(p_velocity)); }
	virtual real_t get_angular_velocity() const override { return ANGLE3TO2(state_3d->get_angular_velocity()); }

	virtual void set_transform(const Transform2D &p_transform) override {
		Vector2 origin = p_transform.get_origin();
		state_3d->set_transform(Transform3D(Basis().rotated(Vector3::FORWARD, p_transform.get_rotation()), VECTOR2TO3(origin)));
	}
	virtual Transform2D get_transform() const override {
		Transform3D xform_3d = state_3d->get_transform();
		Vector3 origin = xform_3d.get_origin();
		return Transform2D(ANGLE3TO2(xform_3d.basis.get_euler()), // TODO check correctness
				VECTOR3TO2(origin));
	}

	virtual Vector2 get_velocity_at_local_position(const Vector2 &p_position) const override {
		Vector3 velocity = state_3d->get_velocity_at_local_position(VECTOR2TO3(p_position));
		return VECTOR3TO2(velocity);
	}

	virtual void apply_central_impulse(const Vector2 &p_impulse) override { state_3d->apply_central_impulse(VECTOR2TO3(p_impulse)); }
	virtual void apply_torque_impulse(real_t p_torque) override { state_3d->apply_torque_impulse(ANGLE2TO3(p_torque)); }
	virtual void apply_impulse(const Vector2 &p_impulse, const Vector2 &p_position = Vector2()) override { state_3d->apply_impulse(VECTOR2TO3(p_impulse), VECTOR2TO3(p_position)); }

	virtual void apply_central_force(const Vector2 &p_force) override { state_3d->apply_central_force(VECTOR2TO3(p_force)); }
	virtual void apply_force(const Vector2 &p_force, const Vector2 &p_position = Vector2()) override { state_3d->apply_force(VECTOR2TO3(p_force), VECTOR2TO3(p_position)); }
	virtual void apply_torque(real_t p_torque) override { state_3d->apply_torque(ANGLE2TO3(p_torque)); }

	virtual void add_constant_central_force(const Vector2 &p_force) override { state_3d->add_constant_central_force(VECTOR2TO3(p_force)); }
	virtual void add_constant_force(const Vector2 &p_force, const Vector2 &p_position = Vector2()) override { state_3d->add_constant_force(VECTOR2TO3(p_force), VECTOR2TO3(p_position)); }
	virtual void add_constant_torque(real_t p_torque) override { state_3d->add_constant_torque(ANGLE2TO3(p_torque)); }

	virtual void set_constant_force(const Vector2 &p_force) override { state_3d->set_constant_force(VECTOR2TO3(p_force)); }
	virtual Vector2 get_constant_force() const override {
		Vector3 force = state_3d->get_constant_force();
		return VECTOR3TO2(force);
	}

	virtual void set_constant_torque(real_t p_torque) override { state_3d->set_constant_torque(ANGLE2TO3(p_torque)); }
	virtual real_t get_constant_torque() const override { return ANGLE3TO2(state_3d->get_constant_torque()); }

	virtual void set_sleep_state(bool p_enable) override { state_3d->set_sleep_state(p_enable); }
	virtual bool is_sleeping() const override { return state_3d->is_sleeping(); }

	virtual void set_collision_layer(uint32_t p_layer) override { state_3d->set_collision_layer(p_layer); }
	virtual uint32_t get_collision_layer() const override { return state_3d->get_collision_layer(); }

	virtual void set_collision_mask(uint32_t p_mask) override { state_3d->set_collision_mask(p_mask); }
	virtual uint32_t get_collision_mask() const override { return state_3d->get_collision_mask(); }

	virtual int get_contact_count() const override { return state_3d->get_contact_count(); } // TODO fix. filter orthogonal to the plane and duplicates?

	virtual Vector2 get_contact_local_position(int p_contact_idx) const override {
		Vector3 pos = state_3d->get_contact_local_position(p_contact_idx);
		return VECTOR3TO2(pos);
	}
	virtual Vector2 get_contact_local_normal(int p_contact_idx) const override {
		Vector3 normal = state_3d->get_contact_local_normal(p_contact_idx);
		return VECTOR3TO2(normal).normalized();
	} // TODO good enough?
	virtual int get_contact_local_shape(int p_contact_idx) const override { return state_3d->get_contact_local_shape(p_contact_idx); }
	virtual Vector2 get_contact_local_velocity_at_position(int p_contact_idx) const override {
		Vector3 velocity = state_3d->get_contact_local_velocity_at_position(p_contact_idx);
		return VECTOR3TO2(velocity);
	}

	virtual RID get_contact_collider(int p_contact_idx) const override { return state_3d->get_contact_collider(p_contact_idx); }
	virtual Vector2 get_contact_collider_position(int p_contact_idx) const override {
		Vector3 pos = state_3d->get_contact_collider_position(p_contact_idx);
		return VECTOR3TO2(pos);
	}
	virtual ObjectID get_contact_collider_id(int p_contact_idx) const override { return state_3d->get_contact_collider_id(p_contact_idx); }
	virtual Object *get_contact_collider_object(int p_contact_idx) const override { return state_3d->get_contact_collider_object(p_contact_idx); }
	virtual int get_contact_collider_shape(int p_contact_idx) const override { return state_3d->get_contact_collider_shape(p_contact_idx); }
	virtual Vector2 get_contact_collider_velocity_at_position(int p_contact_idx) const override {
		Vector3 velocity = state_3d->get_contact_collider_velocity_at_position(p_contact_idx);
		return VECTOR3TO2(velocity);
	}
	virtual Vector2 get_contact_impulse(int p_contact_idx) const override {
		Vector3 impulse = state_3d->get_contact_impulse(p_contact_idx);
		return VECTOR3TO2(impulse);
	}

	virtual real_t get_step() const override { return state_3d->get_step(); }
	virtual void integrate_forces() override { state_3d->integrate_forces(); }

	virtual PhysicsDirectSpaceState2D *get_space_state() override { return nullptr; }

	PhysicsDirectBodyState2Din3D(PhysicsDirectBodyState3D *p_state) :
			state_3d(p_state) {}
};

class PhysicsDirectSpaceState2Din3D : public PhysicsDirectSpaceState2D {
	GDCLASS(PhysicsDirectSpaceState2Din3D, PhysicsDirectSpaceState2D);

	PhysicsDirectSpaceState3D *state_3d;

public:
	virtual bool intersect_ray(const RayParameters &p_parameters, RayResult &r_result) override { return false; }

	virtual int intersect_point(const PointParameters &p_parameters, ShapeResult *r_results, int p_result_max) override { return 0; }

	virtual int intersect_shape(const ShapeParameters &p_parameters, ShapeResult *r_results, int p_result_max) override { return 0; }
	virtual bool cast_motion(const ShapeParameters &p_parameters, real_t &p_closest_safe, real_t &p_closest_unsafe) override { return false; }
	virtual bool collide_shape(const ShapeParameters &p_parameters, Vector2 *r_results, int p_result_max, int &r_result_count) override { return false; }
	virtual bool rest_info(const ShapeParameters &p_parameters, ShapeRestInfo *r_info) override { return false; }

	PhysicsDirectSpaceState2Din3D(PhysicsDirectSpaceState3D *p_state) :
			state_3d(p_state) {}
};

class PhysicsServer2Din3D : public PhysicsServer2D {
	GDCLASS(PhysicsServer2Din3D, PhysicsServer2D);

	HashMap<RID, PhysicsDirectSpaceState2Din3D *> space_state;
	HashMap<RID, PhysicsDirectBodyState2Din3D *> body_state;

public:
	virtual RID world_boundary_shape_create() override { return RID(); }
	virtual RID separation_ray_shape_create() override { return PhysicsServer3D::get_singleton()->separation_ray_shape_create(); }
	virtual RID segment_shape_create() override { return RID(); }
	virtual RID circle_shape_create() override { return PhysicsServer3D::get_singleton()->sphere_shape_create(); }
	virtual RID rectangle_shape_create() override { return PhysicsServer3D::get_singleton()->box_shape_create(); }
	virtual RID capsule_shape_create() override { return PhysicsServer3D::get_singleton()->capsule_shape_create(); }
	virtual RID convex_polygon_shape_create() override { return PhysicsServer3D::get_singleton()->convex_polygon_shape_create(); }
	virtual RID concave_polygon_shape_create() override { return RID(); }

	virtual void shape_set_data(RID p_shape, const Variant &p_data) override {
		ShapeType shape_type = PhysicsServer2D::get_singleton()->shape_get_type(p_shape);
		switch (shape_type) {
			case ShapeType::SHAPE_RECTANGLE: {
				Vector2 extents = p_data;
				PhysicsServer3D::get_singleton()->shape_set_data(p_shape, SIZE2TO3(extents)); // NOTE: Extents are preserved under 180 degree rotation.
			} break;
			case ShapeType::SHAPE_CAPSULE: {
				float height, radius;
				if (p_data.get_type() == Variant::Type::VECTOR2) {
					Vector2 params = p_data;
					radius = params.x;
					height = params.y;
				} else if (p_data.get_type() == Variant::Type::ARRAY) {
					Array params = p_data;
					ERR_FAIL_COND(params.size() != 2);
					height = params[0];
					radius = params[1];
				} else {
					ERR_FAIL();
				}
				Dictionary data;
				data["height"] = SCALE_2D_TO_3D * height;
				data["radius"] = SCALE_2D_TO_3D * radius;
				PhysicsServer3D::get_singleton()->shape_set_data(p_shape, data);
			} break;
			case ShapeType::SHAPE_CONVEX_POLYGON: {
				if (p_data.get_type() != Variant::Type::PACKED_VECTOR2_ARRAY) {
					ERR_FAIL();
				}
				// TODO: Also handle PackedFloat32Array
				PackedVector2Array points = p_data;
				PackedVector3Array points_3d_array;
				points_3d_array.resize(points.size() * 2);
				Vector3 *points_3d = points_3d_array.ptrw();
				for (int i = 0; i < points.size(); i++) {
					points_3d[2 * i] = VECTOR2TO3_FRONT(points[i]);
					points_3d[2 * i + 1] = VECTOR2TO3_BACK(points[i]);
				}
				PhysicsServer3D::get_singleton()->shape_set_data(p_shape, points_3d_array);
			} break;
			case ShapeType::SHAPE_CIRCLE: {
				float radius = p_data;
				PhysicsServer3D::get_singleton()->shape_set_data(p_shape, SCALE_2D_TO_3D * radius);
			} break;
			case ShapeType::SHAPE_SEPARATION_RAY: {
				Dictionary params = p_data;
				float length = params["length"];
				params["length"] = SCALE_2D_TO_3D * length; // TODO: Copy instead of modify?
				PhysicsServer3D::get_singleton()->shape_set_data(p_shape, params);
			} break;
			default: {
				// Don't know this shape.
			} break;
		}
	}
	virtual void shape_set_custom_solver_bias(RID p_shape, real_t p_bias) override {}

	virtual ShapeType shape_get_type(RID p_shape) const override {
		PhysicsServer3D::ShapeType shape_type = PhysicsServer3D::get_singleton()->shape_get_type(p_shape);
		switch (shape_type) {
			case PhysicsServer3D::ShapeType::SHAPE_BOX:
				return ShapeType::SHAPE_RECTANGLE;
			case PhysicsServer3D::ShapeType::SHAPE_CAPSULE:
				return ShapeType::SHAPE_CAPSULE;
			case PhysicsServer3D::ShapeType::SHAPE_CONVEX_POLYGON:
				return ShapeType::SHAPE_CONVEX_POLYGON;
			case PhysicsServer3D::ShapeType::SHAPE_SPHERE:
				return ShapeType::SHAPE_CIRCLE;
			case PhysicsServer3D::ShapeType::SHAPE_SEPARATION_RAY:
				return ShapeType::SHAPE_SEPARATION_RAY;
			// TODO cases
			default:
				return ShapeType::SHAPE_CIRCLE;
		}
	}
	virtual Variant shape_get_data(RID p_shape) const override { return Variant(); }
	virtual real_t shape_get_custom_solver_bias(RID p_shape) const override { return 0; }

	virtual bool shape_collide(RID p_shape_A, const Transform2D &p_xform_A, const Vector2 &p_motion_A, RID p_shape_B, const Transform2D &p_xform_B, const Vector2 &p_motion_B, Vector2 *r_results, int p_result_max, int &r_result_count) override { return false; }

	/* SPACE API */

	virtual RID space_create() override { return PhysicsServer3D::get_singleton()->space_create(); }
	virtual void space_set_active(RID p_space, bool p_active) override { PhysicsServer3D::get_singleton()->space_set_active(p_space, p_active); }
	virtual bool space_is_active(RID p_space) const override { return PhysicsServer3D::get_singleton()->space_is_active(p_space); }

	virtual void space_set_param(RID p_space, SpaceParameter p_param, real_t p_value) override {}
	virtual real_t space_get_param(RID p_space, SpaceParameter p_param) const override { return 0; }

	virtual PhysicsDirectSpaceState2D *space_get_direct_state(RID p_space) override {
		PhysicsDirectSpaceState3D *state_3d = PhysicsServer3D::get_singleton()->space_get_direct_state(p_space);
		if (!state_3d) {
			return nullptr;
		}
		PhysicsDirectSpaceState2Din3D *state_2d = nullptr;
		if (space_state.has(p_space)) {
			state_2d = space_state.get(p_space);
		} else {
			state_2d = memnew(PhysicsDirectSpaceState2Din3D(state_3d));
			space_state.insert(p_space, state_2d);
		}
		return state_2d;
	}

	virtual void space_set_debug_contacts(RID p_space, int p_max_contacts) override {}
	virtual Vector<Vector2> space_get_contacts(RID p_space) const override { return Vector<Vector2>(); }
	virtual int space_get_contact_count(RID p_space) const override { return 0; }

	/* AREA API */

	virtual RID area_create() override { return PhysicsServer3D::get_singleton()->area_create(); }

	virtual void area_set_space(RID p_area, RID p_space) override { PhysicsServer3D::get_singleton()->area_set_space(p_area, p_space); }
	virtual RID area_get_space(RID p_area) const override { return PhysicsServer3D::get_singleton()->area_get_space(p_area); }

	virtual void area_add_shape(RID p_area, RID p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false) override {
		Vector2 origin = p_transform.get_origin();
		Transform3D xform_3d = Transform3D(Basis().rotated(PLANE_NORMAL_3D, p_transform.get_rotation()), VECTOR2TO3(origin));
		PhysicsServer3D::get_singleton()->area_add_shape(p_area, p_shape, xform_3d, p_disabled);
	}
	virtual void area_set_shape(RID p_area, int p_shape_idx, RID p_shape) override { PhysicsServer3D::get_singleton()->area_set_shape(p_area, p_shape_idx, p_shape); }
	virtual void area_set_shape_transform(RID p_area, int p_shape_idx, const Transform2D &p_transform) override {
		Vector2 origin = p_transform.get_origin();
		Transform3D xform_3d = Transform3D(Basis().rotated(PLANE_NORMAL_3D, p_transform.get_rotation()), VECTOR2TO3(origin));
		PhysicsServer3D::get_singleton()->area_set_shape_transform(p_area, p_shape_idx, xform_3d);
	}

	virtual int area_get_shape_count(RID p_area) const override { return PhysicsServer3D::get_singleton()->area_get_shape_count(p_area); }
	virtual RID area_get_shape(RID p_area, int p_shape_idx) const override { return PhysicsServer3D::get_singleton()->area_get_shape(p_area, p_shape_idx); }
	virtual Transform2D area_get_shape_transform(RID p_area, int p_shape_idx) const override {
		Transform3D xform_3d = PhysicsServer3D::get_singleton()->area_get_shape_transform(p_area, p_shape_idx);
		Vector3 origin = xform_3d.get_origin();
		return Transform2D(ANGLE3TO2(xform_3d.basis.get_euler()), // TODO check
				VECTOR3TO2(origin));
	}

	virtual void area_remove_shape(RID p_area, int p_shape_idx) override { PhysicsServer3D::get_singleton()->area_remove_shape(p_area, p_shape_idx); }
	virtual void area_clear_shapes(RID p_area) override { PhysicsServer3D::get_singleton()->area_clear_shapes(p_area); }

	virtual void area_set_shape_disabled(RID p_area, int p_shape, bool p_disabled) override { PhysicsServer3D::get_singleton()->area_set_shape_disabled(p_area, p_shape, p_disabled); }

	virtual void area_attach_object_instance_id(RID p_area, ObjectID p_id) override { PhysicsServer3D::get_singleton()->area_attach_object_instance_id(p_area, p_id); }
	virtual ObjectID area_get_object_instance_id(RID p_area) const override { return PhysicsServer3D::get_singleton()->area_get_object_instance_id(p_area); }

	virtual void area_attach_canvas_instance_id(RID p_area, ObjectID p_id) override {} // TODO hmmm
	virtual ObjectID area_get_canvas_instance_id(RID p_area) const override { return ObjectID(); } // TODO hmmm

	virtual void area_set_param(RID p_area, AreaParameter p_param, const Variant &p_value) override {
		switch (p_param) {
			case AREA_PARAM_GRAVITY: {
				float gravity = p_value;
				PhysicsServer3D::get_singleton()->area_set_param(p_area, PhysicsServer3D::AreaParameter::AREA_PARAM_GRAVITY, SCALE_2D_TO_3D * gravity);
			} break;
			case AREA_PARAM_GRAVITY_VECTOR: {
				Vector2 gravity = p_value;
				PhysicsServer3D::get_singleton()->area_set_param(p_area, PhysicsServer3D::AreaParameter::AREA_PARAM_GRAVITY_VECTOR, VECTOR2TO3_UNSCALED(gravity)); // NOTE: No scale to avoid double scaling?
			} break;
			case AREA_PARAM_GRAVITY_POINT_UNIT_DISTANCE: {
				float unit_distance = p_value;
				PhysicsServer3D::get_singleton()->area_set_param(p_area, PhysicsServer3D::AreaParameter::AREA_PARAM_GRAVITY_POINT_UNIT_DISTANCE, SCALE_2D_TO_3D * unit_distance);
			} break;
			case AREA_PARAM_GRAVITY_OVERRIDE_MODE:
			case AREA_PARAM_GRAVITY_IS_POINT:
			/*
			case AREA_PARAM_LINEAR_DAMP_OVERRIDE_MODE:
			case AREA_PARAM_LINEAR_DAMP:
			case AREA_PARAM_ANGULAR_DAMP_OVERRIDE_MODE:
			case AREA_PARAM_ANGULAR_DAMP:
			*/
			case AREA_PARAM_PRIORITY: {
				// TODO: make sure this works
				PhysicsServer3D::get_singleton()->area_set_param(p_area, (PhysicsServer3D::AreaParameter)p_param, p_value);
			} break;
			default: {
				// Don't know what.
			} break;
		}
	}
	virtual void area_set_transform(RID p_area, const Transform2D &p_transform) override {
		Vector2 origin = p_transform.get_origin();
		Transform3D xform_3d = Transform3D(Basis().rotated(PLANE_NORMAL_3D, p_transform.get_rotation()), VECTOR2TO3(origin));
		PhysicsServer3D::get_singleton()->area_set_transform(p_area, xform_3d);
	}

	virtual Variant area_get_param(RID p_area, AreaParameter p_param) const override {
		switch (p_param) {
			case AREA_PARAM_GRAVITY: {
				float gravity = PhysicsServer3D::get_singleton()->area_get_param(p_area, PhysicsServer3D::AreaParameter::AREA_PARAM_GRAVITY);
				return SCALE_3D_TO_2D * gravity;
			} break;
			case AREA_PARAM_GRAVITY_VECTOR: {
				Vector3 gravity = PhysicsServer3D::get_singleton()->area_get_param(p_area, PhysicsServer3D::AreaParameter::AREA_PARAM_GRAVITY_VECTOR);
				return VECTOR3TO2_UNSCALED(gravity); // TODO: No scale to avoid double scaling?
			} break;
			case AREA_PARAM_GRAVITY_POINT_UNIT_DISTANCE: {
				float unit_distance = PhysicsServer3D::get_singleton()->area_get_param(p_area, PhysicsServer3D::AreaParameter::AREA_PARAM_GRAVITY_POINT_UNIT_DISTANCE);
				return SCALE_3D_TO_2D * unit_distance;
			} break;
			case AREA_PARAM_GRAVITY_OVERRIDE_MODE:
			case AREA_PARAM_GRAVITY_IS_POINT:
			/*
			case AREA_PARAM_LINEAR_DAMP_OVERRIDE_MODE:
			case AREA_PARAM_LINEAR_DAMP:
			case AREA_PARAM_ANGULAR_DAMP_OVERRIDE_MODE:
			case AREA_PARAM_ANGULAR_DAMP:
			*/
			case AREA_PARAM_PRIORITY: {
				// TODO: make sure this works
				return PhysicsServer3D::get_singleton()->area_get_param(p_area, (PhysicsServer3D::AreaParameter)p_param);
			} break;
			default: {
				// Don't know what.
			} break;
		}
		return PhysicsServer3D::get_singleton()->area_get_param(p_area, (PhysicsServer3D::AreaParameter)p_param);
	}
	virtual Transform2D area_get_transform(RID p_area) const override {
		Transform3D xform_3d = PhysicsServer3D::get_singleton()->area_get_transform(p_area);
		Vector3 origin = xform_3d.get_origin();
		return Transform2D(ANGLE3TO2(xform_3d.basis.get_euler()), // TODO check
				VECTOR3TO2(origin));
	}

	virtual void area_set_collision_layer(RID p_area, uint32_t p_layer) override { PhysicsServer3D::get_singleton()->area_set_collision_layer(p_area, p_layer); }
	virtual uint32_t area_get_collision_layer(RID p_area) const override { return PhysicsServer3D::get_singleton()->area_get_collision_layer(p_area); }

	virtual void area_set_collision_mask(RID p_area, uint32_t p_mask) override { PhysicsServer3D::get_singleton()->area_set_collision_mask(p_area, p_mask); }
	virtual uint32_t area_get_collision_mask(RID p_area) const override { return PhysicsServer3D::get_singleton()->area_get_collision_mask(p_area); }

	virtual void area_set_monitorable(RID p_area, bool p_monitorable) override { PhysicsServer3D::get_singleton()->area_set_monitorable(p_area, p_monitorable); }
	virtual void area_set_pickable(RID p_area, bool p_pickable) override { PhysicsServer3D::get_singleton()->area_set_ray_pickable(p_area, p_pickable); } // TODO: Ray pickable is equivalent of pickable?

	virtual void area_set_monitor_callback(RID p_area, const Callable &p_callback) override { PhysicsServer3D::get_singleton()->area_set_monitor_callback(p_area, p_callback); } // NOTE: Callback signature compatible with 3D.
	virtual void area_set_area_monitor_callback(RID p_area, const Callable &p_callback) override { PhysicsServer3D::get_singleton()->area_set_area_monitor_callback(p_area, p_callback); } // NOTE: Callback signature compatible with 3D.

	/* BODY API */

	virtual RID body_create() override {
		RID body = PhysicsServer3D::get_singleton()->body_create();
		PhysicsServer3D::get_singleton()->body_set_axis_lock(body, PhysicsServer3D::BodyAxis::BODY_AXIS_LINEAR_Z, true);
		PhysicsServer3D::get_singleton()->body_set_axis_lock(body, PhysicsServer3D::BodyAxis::BODY_AXIS_ANGULAR_X, true);
		PhysicsServer3D::get_singleton()->body_set_axis_lock(body, PhysicsServer3D::BodyAxis::BODY_AXIS_ANGULAR_Y, true);
		return body;
	}

	virtual void body_set_space(RID p_body, RID p_space) override { PhysicsServer3D::get_singleton()->body_set_space(p_body, p_space); }
	virtual RID body_get_space(RID p_body) const override { return PhysicsServer3D::get_singleton()->body_get_space(p_body); }

	virtual void body_set_mode(RID p_body, BodyMode p_mode) override { PhysicsServer3D::get_singleton()->body_set_mode(p_body, (PhysicsServer3D::BodyMode)p_mode); }
	virtual BodyMode body_get_mode(RID p_body) const override { return (BodyMode)PhysicsServer3D::get_singleton()->body_get_mode(p_body); }

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false) override {
		Vector2 origin = p_transform.get_origin();
		Transform3D xform_3d = Transform3D(Basis().rotated(PLANE_NORMAL_3D, p_transform.get_rotation()), VECTOR2TO3(origin));
		PhysicsServer3D::get_singleton()->body_add_shape(p_body, p_shape, xform_3d, p_disabled);
	}
	virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape) override { PhysicsServer3D::get_singleton()->body_set_shape(p_body, p_shape_idx, p_shape); }
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform2D &p_transform) override {
		Vector2 origin = p_transform.get_origin();
		Transform3D xform_3d = Transform3D(Basis().rotated(PLANE_NORMAL_3D, p_transform.get_rotation()), VECTOR2TO3(origin));
		PhysicsServer3D::get_singleton()->body_set_shape_transform(p_body, p_shape_idx, xform_3d);
	}

	virtual int body_get_shape_count(RID p_body) const override { return PhysicsServer3D::get_singleton()->body_get_shape_count(p_body); }
	virtual RID body_get_shape(RID p_body, int p_shape_idx) const override { return PhysicsServer3D::get_singleton()->body_get_shape(p_body, p_shape_idx); }
	virtual Transform2D body_get_shape_transform(RID p_body, int p_shape_idx) const override {
		Transform3D xform_3d = PhysicsServer3D::get_singleton()->body_get_shape_transform(p_body, p_shape_idx);
		Vector3 origin = xform_3d.get_origin();
		return Transform2D(ANGLE3TO2(xform_3d.basis.get_euler()), // TODO check
				VECTOR3TO2(origin));
	}

	virtual void body_set_shape_disabled(RID p_body, int p_shape, bool p_disabled) override { PhysicsServer3D::get_singleton()->body_set_shape_disabled(p_body, p_shape, p_disabled); }
	virtual void body_set_shape_as_one_way_collision(RID p_body, int p_shape, bool p_enabled, real_t p_margin = 0) override { PhysicsServer3D::get_singleton()->body_set_shape_as_one_way_collision(p_body, p_shape, p_enabled, p_margin); }

	virtual void body_remove_shape(RID p_body, int p_shape_idx) override { PhysicsServer3D::get_singleton()->body_remove_shape(p_body, p_shape_idx); }
	virtual void body_clear_shapes(RID p_body) override { PhysicsServer3D::get_singleton()->body_clear_shapes(p_body); }

	virtual void body_attach_object_instance_id(RID p_body, ObjectID p_id) override { PhysicsServer3D::get_singleton()->body_attach_object_instance_id(p_body, p_id); }
	virtual ObjectID body_get_object_instance_id(RID p_body) const override { return PhysicsServer3D::get_singleton()->body_get_object_instance_id(p_body); }

	virtual void body_attach_canvas_instance_id(RID p_body, ObjectID p_id) override {}
	virtual ObjectID body_get_canvas_instance_id(RID p_body) const override { return ObjectID(); }

	virtual void body_set_continuous_collision_detection_mode(RID p_body, CCDMode p_mode) override { PhysicsServer3D::get_singleton()->body_set_enable_continuous_collision_detection(p_body, p_mode != CCDMode::CCD_MODE_DISABLED); }
	virtual CCDMode body_get_continuous_collision_detection_mode(RID p_body) const override { return PhysicsServer3D::get_singleton()->body_is_continuous_collision_detection_enabled(p_body) ? CCDMode::CCD_MODE_CAST_RAY : CCDMode::CCD_MODE_DISABLED; } // TODO: what represents true better? cast ray or shape?

	virtual void body_set_collision_layer(RID p_body, uint32_t p_layer) override { PhysicsServer3D::get_singleton()->body_set_collision_layer(p_body, p_layer); }
	virtual uint32_t body_get_collision_layer(RID p_body) const override { return PhysicsServer3D::get_singleton()->body_get_collision_layer(p_body); }

	virtual void body_set_collision_mask(RID p_body, uint32_t p_mask) override { PhysicsServer3D::get_singleton()->body_set_collision_mask(p_body, p_mask); }
	virtual uint32_t body_get_collision_mask(RID p_body) const override { return PhysicsServer3D::get_singleton()->body_get_collision_mask(p_body); }

	virtual void body_set_collision_priority(RID p_body, real_t p_priority) override {}
	virtual real_t body_get_collision_priority(RID p_body) const override { return 0; }

	virtual void body_set_param(RID p_body, BodyParameter p_param, const Variant &p_value) override {}
	virtual Variant body_get_param(RID p_body, BodyParameter p_param) const override { return Variant(); }

	virtual void body_reset_mass_properties(RID p_body) override {}

	virtual void body_set_state(RID p_body, BodyState p_state, const Variant &p_variant) override {
		switch (p_state) {
			case BodyState::BODY_STATE_TRANSFORM: {
				Transform2D xform_2d = p_variant;
				Vector2 origin = xform_2d.get_origin();
				Transform3D xform = Transform3D(Basis().rotated(PLANE_NORMAL_3D, xform_2d.get_rotation()), VECTOR2TO3(origin));
				PhysicsServer3D::get_singleton()->body_set_state(p_body, PhysicsServer3D::BodyState::BODY_STATE_TRANSFORM, xform);
			} break;
			case BodyState::BODY_STATE_LINEAR_VELOCITY: {
				Vector2 velocity_2d = p_variant;
				Vector3 velocity = VECTOR2TO3(velocity_2d);
				PhysicsServer3D::get_singleton()->body_set_state(p_body, PhysicsServer3D::BodyState::BODY_STATE_LINEAR_VELOCITY, velocity);
			} break;
			case BodyState::BODY_STATE_ANGULAR_VELOCITY: {
				real_t angular_velocity_2d = p_variant;
				Vector3 angular_velocity = ANGLE2TO3(angular_velocity_2d);
				PhysicsServer3D::get_singleton()->body_set_state(p_body, PhysicsServer3D::BodyState::BODY_STATE_ANGULAR_VELOCITY, angular_velocity);
			} break;
			case BodyState::BODY_STATE_CAN_SLEEP: {
				bool can_sleep = p_variant;
				PhysicsServer3D::get_singleton()->body_set_state(p_body, PhysicsServer3D::BodyState::BODY_STATE_CAN_SLEEP, can_sleep);
			} break;
			case BodyState::BODY_STATE_SLEEPING: {
				bool sleeping = p_variant;
				PhysicsServer3D::get_singleton()->body_set_state(p_body, PhysicsServer3D::BodyState::BODY_STATE_SLEEPING, sleeping);
			} break;
		}
	}
	virtual Variant body_get_state(RID p_body, BodyState p_state) const override {
		switch (p_state) {
			case BodyState::BODY_STATE_TRANSFORM: {
				Transform3D xform_3d = PhysicsServer3D::get_singleton()->body_get_state(p_body, PhysicsServer3D::BodyState::BODY_STATE_TRANSFORM);
				Vector3 origin = xform_3d.get_origin();
				return Transform2D(ANGLE3TO2(xform_3d.basis.get_euler()), // TODO check
						VECTOR3TO2(origin));
			} break;
			case BodyState::BODY_STATE_LINEAR_VELOCITY: {
				Vector3 velocity_3d = PhysicsServer3D::get_singleton()->body_get_state(p_body, PhysicsServer3D::BodyState::BODY_STATE_LINEAR_VELOCITY);
				return VECTOR3TO2(velocity_3d);
			} break;
			case BodyState::BODY_STATE_ANGULAR_VELOCITY: {
				Vector3 velocity_3d = PhysicsServer3D::get_singleton()->body_get_state(p_body, PhysicsServer3D::BodyState::BODY_STATE_ANGULAR_VELOCITY);
				return ANGLE3TO2(velocity_3d);
			} break;
			case BodyState::BODY_STATE_CAN_SLEEP: {
				return PhysicsServer3D::get_singleton()->body_get_state(p_body, PhysicsServer3D::BodyState::BODY_STATE_CAN_SLEEP);
			} break;
			case BodyState::BODY_STATE_SLEEPING: {
				return PhysicsServer3D::get_singleton()->body_get_state(p_body, PhysicsServer3D::BodyState::BODY_STATE_SLEEPING);
			} break;
			default:
				return Variant();
		}
	}

	virtual void body_apply_central_impulse(RID p_body, const Vector2 &p_impulse) override { PhysicsServer3D::get_singleton()->body_apply_central_impulse(p_body, VECTOR2TO3(p_impulse)); }
	virtual void body_apply_torque_impulse(RID p_body, real_t p_torque) override { PhysicsServer3D::get_singleton()->body_apply_torque_impulse(p_body, ANGLE2TO3(p_torque)); }
	virtual void body_apply_impulse(RID p_body, const Vector2 &p_impulse, const Vector2 &p_position = Vector2()) override { PhysicsServer3D::get_singleton()->body_apply_impulse(p_body, VECTOR2TO3(p_impulse), VECTOR2TO3(p_position)); }

	virtual void body_apply_central_force(RID p_body, const Vector2 &p_force) override { PhysicsServer3D::get_singleton()->body_apply_central_force(p_body, VECTOR2TO3(p_force)); }
	virtual void body_apply_force(RID p_body, const Vector2 &p_force, const Vector2 &p_position = Vector2()) override { PhysicsServer3D::get_singleton()->body_apply_force(p_body, VECTOR2TO3(p_force), VECTOR2TO3(p_position)); }
	virtual void body_apply_torque(RID p_body, real_t p_torque) override { PhysicsServer3D::get_singleton()->body_apply_torque(p_body, ANGLE2TO3(p_torque)); }

	virtual void body_add_constant_central_force(RID p_body, const Vector2 &p_force) override { PhysicsServer3D::get_singleton()->body_add_constant_central_force(p_body, VECTOR2TO3(p_force)); }
	virtual void body_add_constant_force(RID p_body, const Vector2 &p_force, const Vector2 &p_position = Vector2()) override { PhysicsServer3D::get_singleton()->body_add_constant_force(p_body, VECTOR2TO3(p_force), VECTOR2TO3(p_position)); }
	virtual void body_add_constant_torque(RID p_body, real_t p_torque) override { PhysicsServer3D::get_singleton()->body_add_constant_torque(p_body, ANGLE2TO3(p_torque)); }

	virtual void body_set_constant_force(RID p_body, const Vector2 &p_force) override { PhysicsServer3D::get_singleton()->body_set_constant_force(p_body, VECTOR2TO3(p_force)); }
	virtual Vector2 body_get_constant_force(RID p_body) const override {
		Vector3 force = PhysicsServer3D::get_singleton()->body_get_constant_force(p_body);
		return VECTOR3TO2(force);
	}

	virtual void body_set_constant_torque(RID p_body, real_t p_torque) override { PhysicsServer3D::get_singleton()->body_set_constant_torque(p_body, ANGLE2TO3(p_torque)); }
	virtual real_t body_get_constant_torque(RID p_body) const override { return ANGLE3TO2(PhysicsServer3D::get_singleton()->body_get_constant_torque(p_body)); }

	virtual void body_set_axis_velocity(RID p_body, const Vector2 &p_axis_velocity) override { PhysicsServer3D::get_singleton()->body_set_axis_velocity(p_body, VECTOR2TO3(p_axis_velocity)); }

	virtual void body_add_collision_exception(RID p_body, RID p_body_b) override { PhysicsServer3D::get_singleton()->body_add_collision_exception(p_body, p_body_b); }
	virtual void body_remove_collision_exception(RID p_body, RID p_body_b) override { PhysicsServer3D::get_singleton()->body_remove_collision_exception(p_body, p_body_b); }
	virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) override { PhysicsServer3D::get_singleton()->body_get_collision_exceptions(p_body, p_exceptions); }

	virtual void body_set_max_contacts_reported(RID p_body, int p_contacts) override { PhysicsServer3D::get_singleton()->body_set_max_contacts_reported(p_body, p_contacts); }
	virtual int body_get_max_contacts_reported(RID p_body) const override { return PhysicsServer3D::get_singleton()->body_get_max_contacts_reported(p_body); }

	virtual void body_set_contacts_reported_depth_threshold(RID p_body, real_t p_threshold) override { PhysicsServer3D::get_singleton()->body_set_contacts_reported_depth_threshold(p_body, SCALE_2D_TO_3D * p_threshold); }
	virtual real_t body_get_contacts_reported_depth_threshold(RID p_body) const override { return SCALE_3D_TO_2D * PhysicsServer3D::get_singleton()->body_get_contacts_reported_depth_threshold(p_body); }

	virtual void body_set_omit_force_integration(RID p_body, bool p_omit) override { PhysicsServer3D::get_singleton()->body_set_omit_force_integration(p_body, p_omit); }
	virtual bool body_is_omitting_force_integration(RID p_body) const override { return PhysicsServer3D::get_singleton()->body_is_omitting_force_integration(p_body); }

	void body_state_sync_callback_3d(PhysicsDirectBodyState3D *p_state_3d, RID p_body, const Callable &p_callable_2d) {
		PhysicsDirectBodyState2Din3D *state_2d = nullptr;
		if (body_state.has(p_body)) {
			state_2d = body_state.get(p_body);
		} else {
			state_2d = memnew(PhysicsDirectBodyState2Din3D(p_state_3d));
			body_state.insert(p_body, state_2d);
		}
		p_callable_2d.call(state_2d);
	}
	virtual void body_set_state_sync_callback(RID p_body, const Callable &p_callable) override { PhysicsServer3D::get_singleton()->body_set_state_sync_callback(p_body, callable_mp(this, &PhysicsServer2Din3D::body_state_sync_callback_3d).bind(p_body, p_callable)); }
	virtual void body_set_force_integration_callback(RID p_body, const Callable &p_callable, const Variant &p_udata = Variant()) override {} // TODO

	virtual bool body_collide_shape(RID p_body, int p_body_shape, RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, Vector2 *r_results, int p_result_max, int &r_result_count) override { return false; } // TODO

	virtual void body_set_pickable(RID p_body, bool p_pickable) override { PhysicsServer3D::get_singleton()->body_set_ray_pickable(p_body, p_pickable); } // TODO: is ray pickable the analog of pickable?

	virtual PhysicsDirectBodyState2D *body_get_direct_state(RID p_body) override {
		PhysicsDirectBodyState3D *state_3d = PhysicsServer3D::get_singleton()->body_get_direct_state(p_body);
		if (!state_3d) {
			return nullptr;
		}
		PhysicsDirectBodyState2Din3D *state_2d = nullptr;
		if (body_state.has(p_body)) {
			state_2d = body_state.get(p_body);
		} else {
			state_2d = memnew(PhysicsDirectBodyState2Din3D(state_3d));
			body_state.insert(p_body, state_2d);
		}
		return state_2d;
	}

	virtual bool body_test_motion(RID p_body, const MotionParameters &p_parameters, MotionResult *r_result = nullptr) override {
		PhysicsServer3D::MotionResult motion_result;
		PhysicsServer3D::MotionParameters motion_params;
		motion_params.from = Transform3D(Basis().rotated(PLANE_NORMAL_3D, p_parameters.from.get_rotation()), VECTOR2TO3(p_parameters.from.get_origin()));
		motion_params.motion = VECTOR2TO3(p_parameters.motion);
		motion_params.margin = SCALE_2D_TO_3D * p_parameters.margin / 8.0f; // TODO: OK to match defaults like this? 0.08 in 2D -> 0.001 in 3D.
		motion_params.max_collisions = 1; // TODO: ?
		motion_params.collide_separation_ray = p_parameters.collide_separation_ray;
		motion_params.exclude_bodies = p_parameters.exclude_bodies;
		motion_params.exclude_objects = p_parameters.exclude_objects;
		motion_params.recovery_as_collision = p_parameters.recovery_as_collision;
		bool collided = PhysicsServer3D::get_singleton()->body_test_motion(p_body, motion_params, r_result ? &motion_result : nullptr);
		if (r_result) {
			r_result->travel = VECTOR3TO2(motion_result.travel);
			r_result->remainder = VECTOR3TO2(motion_result.remainder);
			r_result->collision_depth = SCALE_3D_TO_2D * motion_result.collision_depth;
			r_result->collision_safe_fraction = motion_result.collision_safe_fraction;
			r_result->collision_unsafe_fraction = motion_result.collision_unsafe_fraction;
			if (motion_result.collision_count > 0) {
				// TODO: Handle multiple?
				PhysicsServer3D::MotionCollision &collision = motion_result.collisions[0];
				r_result->collision_point = VECTOR3TO2(collision.position);
				r_result->collision_normal = VECTOR3TO2(collision.normal).normalized();
				r_result->collider_velocity = VECTOR3TO2(collision.collider_velocity);
				r_result->collision_local_shape = collision.local_shape;
				r_result->collider_id = collision.collider_id;
				r_result->collider = collision.collider;
				r_result->collider_shape = collision.collider_shape;
			}
		}
		return collided;
	}

	/* JOINT API */
	// TODO

	virtual RID joint_create() override { return RID(); }

	virtual void joint_clear(RID p_joint) override {}

	virtual void joint_set_param(RID p_joint, JointParam p_param, real_t p_value) override {}
	virtual real_t joint_get_param(RID p_joint, JointParam p_param) const override { return 0; }

	virtual void joint_disable_collisions_between_bodies(RID p_joint, const bool p_disable) override {}
	virtual bool joint_is_disabled_collisions_between_bodies(RID p_joint) const override { return false; }

	virtual void joint_make_pin(RID p_joint, const Vector2 &p_anchor, RID p_body_a, RID p_body_b = RID()) override {}
	virtual void joint_make_groove(RID p_joint, const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, RID p_body_a, RID p_body_b) override {}
	virtual void joint_make_damped_spring(RID p_joint, const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, RID p_body_a, RID p_body_b = RID()) override {}

	virtual void pin_joint_set_param(RID p_joint, PinJointParam p_param, real_t p_value) override {}
	virtual real_t pin_joint_get_param(RID p_joint, PinJointParam p_param) const override { return 0; }

	virtual void pin_joint_set_flag(RID p_joint, PinJointFlag p_flag, bool p_enabled) override {}
	virtual bool pin_joint_get_flag(RID p_joint, PinJointFlag p_flag) const override { return false; }

	virtual void damped_spring_joint_set_param(RID p_joint, DampedSpringParam p_param, real_t p_value) override {}
	virtual real_t damped_spring_joint_get_param(RID p_joint, DampedSpringParam p_param) const override { return 0; }

	virtual JointType joint_get_type(RID p_joint) const override { return JointType::JOINT_TYPE_PIN; }

	/* MISC */

	virtual void free_rid(RID p_rid) override {
		if (space_state.has(p_rid)) {
			memdelete(space_state.get(p_rid));
			space_state.erase(p_rid);
			PhysicsServer3D::get_singleton()->free_rid(p_rid);
		} else if (body_state.has(p_rid)) {
			memdelete(body_state.get(p_rid));
			body_state.erase(p_rid);
			PhysicsServer3D::get_singleton()->free_rid(p_rid);
		} else {
			// TODO: Don't return null RIDs in the server, so this check can be removed.
			if (p_rid.is_valid()) {
				PhysicsServer3D::get_singleton()->free_rid(p_rid);
			}
		}
	}

	virtual void set_active(bool p_active) override {}
	virtual void init() override {}
	virtual void step(real_t p_step) override {}
	virtual void sync() override {}
	virtual void flush_queries() override {}
	virtual void end_sync() override {}
	virtual void finish() override {}

	virtual bool is_flushing_queries() const override { return false; }

	virtual int get_process_info(ProcessInfo p_info) override { return 0; }
};
