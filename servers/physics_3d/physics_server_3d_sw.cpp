/*************************************************************************/
/*  physics_server_3d_sw.cpp                                             */
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

#include "physics_server_3d_sw.h"

#include "broad_phase_3d_bvh.h"
#include "core/debugger/engine_debugger.h"
#include "core/os/os.h"
#include "joints/cone_twist_joint_3d_sw.h"
#include "joints/generic_6dof_joint_3d_sw.h"
#include "joints/hinge_joint_3d_sw.h"
#include "joints/pin_joint_3d_sw.h"
#include "joints/slider_joint_3d_sw.h"

#define FLUSH_QUERY_CHECK(m_object) \
	ERR_FAIL_COND_MSG(m_object->get_space() && flushing_queries, "Can't change this state while flushing queries. Use call_deferred() or set_deferred() to change monitoring state instead.");

RID PhysicsServer3DSW::plane_shape_create() {
	Shape3DSW *shape = memnew(PlaneShape3DSW);
	RID rid = shape_owner.make_rid(shape);
	shape->set_self(rid);
	return rid;
}
RID PhysicsServer3DSW::ray_shape_create() {
	Shape3DSW *shape = memnew(RayShape3DSW);
	RID rid = shape_owner.make_rid(shape);
	shape->set_self(rid);
	return rid;
}
RID PhysicsServer3DSW::sphere_shape_create() {
	Shape3DSW *shape = memnew(SphereShape3DSW);
	RID rid = shape_owner.make_rid(shape);
	shape->set_self(rid);
	return rid;
}
RID PhysicsServer3DSW::box_shape_create() {
	Shape3DSW *shape = memnew(BoxShape3DSW);
	RID rid = shape_owner.make_rid(shape);
	shape->set_self(rid);
	return rid;
}
RID PhysicsServer3DSW::capsule_shape_create() {
	Shape3DSW *shape = memnew(CapsuleShape3DSW);
	RID rid = shape_owner.make_rid(shape);
	shape->set_self(rid);
	return rid;
}
RID PhysicsServer3DSW::cylinder_shape_create() {
	Shape3DSW *shape = memnew(CylinderShape3DSW);
	RID rid = shape_owner.make_rid(shape);
	shape->set_self(rid);
	return rid;
}
RID PhysicsServer3DSW::convex_polygon_shape_create() {
	Shape3DSW *shape = memnew(ConvexPolygonShape3DSW);
	RID rid = shape_owner.make_rid(shape);
	shape->set_self(rid);
	return rid;
}
RID PhysicsServer3DSW::concave_polygon_shape_create() {
	Shape3DSW *shape = memnew(ConcavePolygonShape3DSW);
	RID rid = shape_owner.make_rid(shape);
	shape->set_self(rid);
	return rid;
}
RID PhysicsServer3DSW::heightmap_shape_create() {
	Shape3DSW *shape = memnew(HeightMapShape3DSW);
	RID rid = shape_owner.make_rid(shape);
	shape->set_self(rid);
	return rid;
}
RID PhysicsServer3DSW::custom_shape_create() {
	ERR_FAIL_V(RID());
}

void PhysicsServer3DSW::shape_set_data(RID p_shape, const Variant &p_data) {
	Shape3DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);
	shape->set_data(p_data);
};

void PhysicsServer3DSW::shape_set_custom_solver_bias(RID p_shape, real_t p_bias) {
	Shape3DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);
	shape->set_custom_bias(p_bias);
}

PhysicsServer3D::ShapeType PhysicsServer3DSW::shape_get_type(RID p_shape) const {
	const Shape3DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND_V(!shape, SHAPE_CUSTOM);
	return shape->get_type();
};

Variant PhysicsServer3DSW::shape_get_data(RID p_shape) const {
	const Shape3DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND_V(!shape, Variant());
	ERR_FAIL_COND_V(!shape->is_configured(), Variant());
	return shape->get_data();
};

void PhysicsServer3DSW::shape_set_margin(RID p_shape, real_t p_margin) {
}

real_t PhysicsServer3DSW::shape_get_margin(RID p_shape) const {
	return 0.0;
}

real_t PhysicsServer3DSW::shape_get_custom_solver_bias(RID p_shape) const {
	const Shape3DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND_V(!shape, 0);
	return shape->get_custom_bias();
}

RID PhysicsServer3DSW::space_create() {
	Space3DSW *space = memnew(Space3DSW);
	RID id = space_owner.make_rid(space);
	space->set_self(id);
	RID area_id = area_create();
	Area3DSW *area = area_owner.getornull(area_id);
	ERR_FAIL_COND_V(!area, RID());
	space->set_default_area(area);
	area->set_space(space);
	area->set_priority(-1);
	RID sgb = body_create();
	body_set_space(sgb, id);
	body_set_mode(sgb, BODY_MODE_STATIC);
	space->set_static_global_body(sgb);

	return id;
};

void PhysicsServer3DSW::space_set_active(RID p_space, bool p_active) {
	Space3DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND(!space);
	if (p_active) {
		active_spaces.insert(space);
	} else {
		active_spaces.erase(space);
	}
}

bool PhysicsServer3DSW::space_is_active(RID p_space) const {
	const Space3DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND_V(!space, false);

	return active_spaces.has(space);
}

void PhysicsServer3DSW::space_set_param(RID p_space, SpaceParameter p_param, real_t p_value) {
	Space3DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND(!space);

	space->set_param(p_param, p_value);
}

real_t PhysicsServer3DSW::space_get_param(RID p_space, SpaceParameter p_param) const {
	const Space3DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND_V(!space, 0);
	return space->get_param(p_param);
}

PhysicsDirectSpaceState3D *PhysicsServer3DSW::space_get_direct_state(RID p_space) {
	Space3DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND_V(!space, nullptr);
	ERR_FAIL_COND_V_MSG((using_threads && !doing_sync) || space->is_locked(), nullptr, "Space state is inaccessible right now, wait for iteration or physics process notification.");

	return space->get_direct_state();
}

void PhysicsServer3DSW::space_set_debug_contacts(RID p_space, int p_max_contacts) {
	Space3DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND(!space);
	space->set_debug_contacts(p_max_contacts);
}

Vector<Vector3> PhysicsServer3DSW::space_get_contacts(RID p_space) const {
	Space3DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND_V(!space, Vector<Vector3>());
	return space->get_debug_contacts();
}

int PhysicsServer3DSW::space_get_contact_count(RID p_space) const {
	Space3DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND_V(!space, 0);
	return space->get_debug_contact_count();
}

RID PhysicsServer3DSW::area_create() {
	Area3DSW *area = memnew(Area3DSW);
	RID rid = area_owner.make_rid(area);
	area->set_self(rid);
	return rid;
};

void PhysicsServer3DSW::area_set_space(RID p_area, RID p_space) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	Space3DSW *space = nullptr;
	if (p_space.is_valid()) {
		space = space_owner.getornull(p_space);
		ERR_FAIL_COND(!space);
	}

	if (area->get_space() == space) {
		return; //pointless
	}

	area->clear_constraints();
	area->set_space(space);
};

RID PhysicsServer3DSW::area_get_space(RID p_area) const {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, RID());

	Space3DSW *space = area->get_space();
	if (!space) {
		return RID();
	}
	return space->get_self();
};

void PhysicsServer3DSW::area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_space_override_mode(p_mode);
}

PhysicsServer3D::AreaSpaceOverrideMode PhysicsServer3DSW::area_get_space_override_mode(RID p_area) const {
	const Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, AREA_SPACE_OVERRIDE_DISABLED);

	return area->get_space_override_mode();
}

void PhysicsServer3DSW::area_add_shape(RID p_area, RID p_shape, const Transform &p_transform, bool p_disabled) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	Shape3DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);

	area->add_shape(shape, p_transform, p_disabled);
}

void PhysicsServer3DSW::area_set_shape(RID p_area, int p_shape_idx, RID p_shape) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	Shape3DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);
	ERR_FAIL_COND(!shape->is_configured());

	area->set_shape(p_shape_idx, shape);
}

void PhysicsServer3DSW::area_set_shape_transform(RID p_area, int p_shape_idx, const Transform &p_transform) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_shape_transform(p_shape_idx, p_transform);
}

int PhysicsServer3DSW::area_get_shape_count(RID p_area) const {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, -1);

	return area->get_shape_count();
}

RID PhysicsServer3DSW::area_get_shape(RID p_area, int p_shape_idx) const {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, RID());

	Shape3DSW *shape = area->get_shape(p_shape_idx);
	ERR_FAIL_COND_V(!shape, RID());

	return shape->get_self();
}

Transform PhysicsServer3DSW::area_get_shape_transform(RID p_area, int p_shape_idx) const {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, Transform());

	return area->get_shape_transform(p_shape_idx);
}

void PhysicsServer3DSW::area_remove_shape(RID p_area, int p_shape_idx) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->remove_shape(p_shape_idx);
}

void PhysicsServer3DSW::area_clear_shapes(RID p_area) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	while (area->get_shape_count()) {
		area->remove_shape(0);
	}
}

void PhysicsServer3DSW::area_set_shape_disabled(RID p_area, int p_shape_idx, bool p_disabled) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	ERR_FAIL_INDEX(p_shape_idx, area->get_shape_count());
	FLUSH_QUERY_CHECK(area);
	area->set_shape_as_disabled(p_shape_idx, p_disabled);
}

void PhysicsServer3DSW::area_attach_object_instance_id(RID p_area, ObjectID p_id) {
	if (space_owner.owns(p_area)) {
		Space3DSW *space = space_owner.getornull(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	area->set_instance_id(p_id);
}

ObjectID PhysicsServer3DSW::area_get_object_instance_id(RID p_area) const {
	if (space_owner.owns(p_area)) {
		Space3DSW *space = space_owner.getornull(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, ObjectID());
	return area->get_instance_id();
}

void PhysicsServer3DSW::area_set_param(RID p_area, AreaParameter p_param, const Variant &p_value) {
	if (space_owner.owns(p_area)) {
		Space3DSW *space = space_owner.getornull(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	area->set_param(p_param, p_value);
};

void PhysicsServer3DSW::area_set_transform(RID p_area, const Transform &p_transform) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	area->set_transform(p_transform);
};

Variant PhysicsServer3DSW::area_get_param(RID p_area, AreaParameter p_param) const {
	if (space_owner.owns(p_area)) {
		Space3DSW *space = space_owner.getornull(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, Variant());

	return area->get_param(p_param);
};

Transform PhysicsServer3DSW::area_get_transform(RID p_area) const {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, Transform());

	return area->get_transform();
};

void PhysicsServer3DSW::area_set_collision_layer(RID p_area, uint32_t p_layer) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_collision_layer(p_layer);
}

void PhysicsServer3DSW::area_set_collision_mask(RID p_area, uint32_t p_mask) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_collision_mask(p_mask);
}

void PhysicsServer3DSW::area_set_monitorable(RID p_area, bool p_monitorable) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	FLUSH_QUERY_CHECK(area);

	area->set_monitorable(p_monitorable);
}

void PhysicsServer3DSW::area_set_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_monitor_callback(p_receiver ? p_receiver->get_instance_id() : ObjectID(), p_method);
}

void PhysicsServer3DSW::area_set_ray_pickable(RID p_area, bool p_enable) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_ray_pickable(p_enable);
}

void PhysicsServer3DSW::area_set_area_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) {
	Area3DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_area_monitor_callback(p_receiver ? p_receiver->get_instance_id() : ObjectID(), p_method);
}

/* BODY API */

RID PhysicsServer3DSW::body_create() {
	Body3DSW *body = memnew(Body3DSW);
	RID rid = body_owner.make_rid(body);
	body->set_self(rid);
	return rid;
};

void PhysicsServer3DSW::body_set_space(RID p_body, RID p_space) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	Space3DSW *space = nullptr;
	if (p_space.is_valid()) {
		space = space_owner.getornull(p_space);
		ERR_FAIL_COND(!space);
	}

	if (body->get_space() == space) {
		return; //pointless
	}

	body->clear_constraint_map();
	body->set_space(space);
};

RID PhysicsServer3DSW::body_get_space(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, RID());

	Space3DSW *space = body->get_space();
	if (!space) {
		return RID();
	}
	return space->get_self();
};

void PhysicsServer3DSW::body_set_mode(RID p_body, BodyMode p_mode) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_mode(p_mode);
};

PhysicsServer3D::BodyMode PhysicsServer3DSW::body_get_mode(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, BODY_MODE_STATIC);

	return body->get_mode();
};

void PhysicsServer3DSW::body_add_shape(RID p_body, RID p_shape, const Transform &p_transform, bool p_disabled) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	Shape3DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);

	body->add_shape(shape, p_transform, p_disabled);
}

void PhysicsServer3DSW::body_set_shape(RID p_body, int p_shape_idx, RID p_shape) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	Shape3DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);
	ERR_FAIL_COND(!shape->is_configured());

	body->set_shape(p_shape_idx, shape);
}

void PhysicsServer3DSW::body_set_shape_transform(RID p_body, int p_shape_idx, const Transform &p_transform) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_shape_transform(p_shape_idx, p_transform);
}

int PhysicsServer3DSW::body_get_shape_count(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, -1);

	return body->get_shape_count();
}

RID PhysicsServer3DSW::body_get_shape(RID p_body, int p_shape_idx) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, RID());

	Shape3DSW *shape = body->get_shape(p_shape_idx);
	ERR_FAIL_COND_V(!shape, RID());

	return shape->get_self();
}

void PhysicsServer3DSW::body_set_shape_disabled(RID p_body, int p_shape_idx, bool p_disabled) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	ERR_FAIL_INDEX(p_shape_idx, body->get_shape_count());
	FLUSH_QUERY_CHECK(body);

	body->set_shape_as_disabled(p_shape_idx, p_disabled);
}

Transform PhysicsServer3DSW::body_get_shape_transform(RID p_body, int p_shape_idx) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, Transform());

	return body->get_shape_transform(p_shape_idx);
}

void PhysicsServer3DSW::body_remove_shape(RID p_body, int p_shape_idx) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->remove_shape(p_shape_idx);
}

void PhysicsServer3DSW::body_clear_shapes(RID p_body) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	while (body->get_shape_count()) {
		body->remove_shape(0);
	}
}

void PhysicsServer3DSW::body_set_enable_continuous_collision_detection(RID p_body, bool p_enable) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_continuous_collision_detection(p_enable);
}

bool PhysicsServer3DSW::body_is_continuous_collision_detection_enabled(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, false);

	return body->is_continuous_collision_detection_enabled();
}

void PhysicsServer3DSW::body_set_collision_layer(RID p_body, uint32_t p_layer) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_collision_layer(p_layer);
	body->wakeup();
}

uint32_t PhysicsServer3DSW::body_get_collision_layer(RID p_body) const {
	const Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_collision_layer();
}

void PhysicsServer3DSW::body_set_collision_mask(RID p_body, uint32_t p_mask) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_collision_mask(p_mask);
	body->wakeup();
}

uint32_t PhysicsServer3DSW::body_get_collision_mask(RID p_body) const {
	const Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_collision_mask();
}

void PhysicsServer3DSW::body_attach_object_instance_id(RID p_body, ObjectID p_id) {
	Body3DSW *body = body_owner.getornull(p_body);
	if (body) {
		body->set_instance_id(p_id);
		return;
	}

	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	if (soft_body) {
		soft_body->set_instance_id(p_id);
		return;
	}

	ERR_FAIL_MSG("Invalid ID.");
};

ObjectID PhysicsServer3DSW::body_get_object_instance_id(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, ObjectID());

	return body->get_instance_id();
};

void PhysicsServer3DSW::body_set_user_flags(RID p_body, uint32_t p_flags) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
};

uint32_t PhysicsServer3DSW::body_get_user_flags(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return 0;
};

void PhysicsServer3DSW::body_set_param(RID p_body, BodyParameter p_param, real_t p_value) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_param(p_param, p_value);
};

real_t PhysicsServer3DSW::body_get_param(RID p_body, BodyParameter p_param) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_param(p_param);
};

void PhysicsServer3DSW::body_set_kinematic_safe_margin(RID p_body, real_t p_margin) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	body->set_kinematic_margin(p_margin);
}

real_t PhysicsServer3DSW::body_get_kinematic_safe_margin(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_kinematic_margin();
}

void PhysicsServer3DSW::body_set_state(RID p_body, BodyState p_state, const Variant &p_variant) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_state(p_state, p_variant);
};

Variant PhysicsServer3DSW::body_get_state(RID p_body, BodyState p_state) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, Variant());

	return body->get_state(p_state);
};

void PhysicsServer3DSW::body_set_applied_force(RID p_body, const Vector3 &p_force) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_applied_force(p_force);
	body->wakeup();
};

Vector3 PhysicsServer3DSW::body_get_applied_force(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, Vector3());
	return body->get_applied_force();
};

void PhysicsServer3DSW::body_set_applied_torque(RID p_body, const Vector3 &p_torque) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_applied_torque(p_torque);
	body->wakeup();
};

Vector3 PhysicsServer3DSW::body_get_applied_torque(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, Vector3());

	return body->get_applied_torque();
};

void PhysicsServer3DSW::body_add_central_force(RID p_body, const Vector3 &p_force) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->add_central_force(p_force);
	body->wakeup();
}

void PhysicsServer3DSW::body_add_force(RID p_body, const Vector3 &p_force, const Vector3 &p_position) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->add_force(p_force, p_position);
	body->wakeup();
};

void PhysicsServer3DSW::body_add_torque(RID p_body, const Vector3 &p_torque) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->add_torque(p_torque);
	body->wakeup();
};

void PhysicsServer3DSW::body_apply_central_impulse(RID p_body, const Vector3 &p_impulse) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	_update_shapes();

	body->apply_central_impulse(p_impulse);
	body->wakeup();
}

void PhysicsServer3DSW::body_apply_impulse(RID p_body, const Vector3 &p_impulse, const Vector3 &p_position) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	_update_shapes();

	body->apply_impulse(p_impulse, p_position);
	body->wakeup();
};

void PhysicsServer3DSW::body_apply_torque_impulse(RID p_body, const Vector3 &p_impulse) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	_update_shapes();

	body->apply_torque_impulse(p_impulse);
	body->wakeup();
};

void PhysicsServer3DSW::body_set_axis_velocity(RID p_body, const Vector3 &p_axis_velocity) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	_update_shapes();

	Vector3 v = body->get_linear_velocity();
	Vector3 axis = p_axis_velocity.normalized();
	v -= axis * axis.dot(v);
	v += p_axis_velocity;
	body->set_linear_velocity(v);
	body->wakeup();
};

void PhysicsServer3DSW::body_set_axis_lock(RID p_body, BodyAxis p_axis, bool p_lock) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_axis_lock(p_axis, p_lock);
	body->wakeup();
}

bool PhysicsServer3DSW::body_is_axis_locked(RID p_body, BodyAxis p_axis) const {
	const Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);
	return body->is_axis_locked(p_axis);
}

void PhysicsServer3DSW::body_add_collision_exception(RID p_body, RID p_body_b) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->add_exception(p_body_b);
	body->wakeup();
};

void PhysicsServer3DSW::body_remove_collision_exception(RID p_body, RID p_body_b) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->remove_exception(p_body_b);
	body->wakeup();
};

void PhysicsServer3DSW::body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	for (int i = 0; i < body->get_exceptions().size(); i++) {
		p_exceptions->push_back(body->get_exceptions()[i]);
	}
};

void PhysicsServer3DSW::body_set_contacts_reported_depth_threshold(RID p_body, real_t p_threshold) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
};

real_t PhysicsServer3DSW::body_get_contacts_reported_depth_threshold(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);
	return 0;
};

void PhysicsServer3DSW::body_set_omit_force_integration(RID p_body, bool p_omit) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_omit_force_integration(p_omit);
};

bool PhysicsServer3DSW::body_is_omitting_force_integration(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, false);
	return body->get_omit_force_integration();
};

void PhysicsServer3DSW::body_set_max_contacts_reported(RID p_body, int p_contacts) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	body->set_max_contacts_reported(p_contacts);
}

int PhysicsServer3DSW::body_get_max_contacts_reported(RID p_body) const {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, -1);
	return body->get_max_contacts_reported();
}

void PhysicsServer3DSW::body_set_force_integration_callback(RID p_body, const Callable &p_callable, const Variant &p_udata) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	body->set_force_integration_callback(p_callable, p_udata);
}

void PhysicsServer3DSW::body_set_ray_pickable(RID p_body, bool p_enable) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	body->set_ray_pickable(p_enable);
}

bool PhysicsServer3DSW::body_test_motion(RID p_body, const Transform &p_from, const Vector3 &p_motion, bool p_infinite_inertia, MotionResult *r_result, bool p_exclude_raycast_shapes) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, false);
	ERR_FAIL_COND_V(!body->get_space(), false);
	ERR_FAIL_COND_V(body->get_space()->is_locked(), false);

	_update_shapes();

	return body->get_space()->test_body_motion(body, p_from, p_motion, p_infinite_inertia, body->get_kinematic_margin(), r_result, p_exclude_raycast_shapes);
}

int PhysicsServer3DSW::body_test_ray_separation(RID p_body, const Transform &p_transform, bool p_infinite_inertia, Vector3 &r_recover_motion, SeparationResult *r_results, int p_result_max, real_t p_margin) {
	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, false);
	ERR_FAIL_COND_V(!body->get_space(), false);
	ERR_FAIL_COND_V(body->get_space()->is_locked(), false);

	_update_shapes();

	return body->get_space()->test_body_ray_separation(body, p_transform, p_infinite_inertia, r_recover_motion, r_results, p_result_max, p_margin);
}

PhysicsDirectBodyState3D *PhysicsServer3DSW::body_get_direct_state(RID p_body) {
	ERR_FAIL_COND_V_MSG((using_threads && !doing_sync), nullptr, "Body state is inaccessible right now, wait for iteration or physics process notification.");

	Body3DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, nullptr);
	ERR_FAIL_COND_V_MSG(body->get_space()->is_locked(), nullptr, "Body state is inaccessible right now, wait for iteration or physics process notification.");

	direct_state->body = body;
	return direct_state;
}

/* SOFT BODY */

RID PhysicsServer3DSW::soft_body_create() {
	SoftBody3DSW *soft_body = memnew(SoftBody3DSW);
	RID rid = soft_body_owner.make_rid(soft_body);
	soft_body->set_self(rid);
	return rid;
}

void PhysicsServer3DSW::soft_body_update_rendering_server(RID p_body, RenderingServerHandler *p_rendering_server_handler) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->update_rendering_server(p_rendering_server_handler);
}

void PhysicsServer3DSW::soft_body_set_space(RID p_body, RID p_space) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	Space3DSW *space = nullptr;
	if (p_space.is_valid()) {
		space = space_owner.getornull(p_space);
		ERR_FAIL_COND(!space);
	}

	if (soft_body->get_space() == space) {
		return;
	}

	soft_body->set_space(space);
}

RID PhysicsServer3DSW::soft_body_get_space(RID p_body) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, RID());

	Space3DSW *space = soft_body->get_space();
	if (!space) {
		return RID();
	}
	return space->get_self();
}

void PhysicsServer3DSW::soft_body_set_collision_layer(RID p_body, uint32_t p_layer) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_collision_layer(p_layer);
}

uint32_t PhysicsServer3DSW::soft_body_get_collision_layer(RID p_body) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, 0);

	return soft_body->get_collision_layer();
}

void PhysicsServer3DSW::soft_body_set_collision_mask(RID p_body, uint32_t p_mask) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_collision_mask(p_mask);
}

uint32_t PhysicsServer3DSW::soft_body_get_collision_mask(RID p_body) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, 0);

	return soft_body->get_collision_mask();
}

void PhysicsServer3DSW::soft_body_add_collision_exception(RID p_body, RID p_body_b) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->add_exception(p_body_b);
}

void PhysicsServer3DSW::soft_body_remove_collision_exception(RID p_body, RID p_body_b) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->remove_exception(p_body_b);
}

void PhysicsServer3DSW::soft_body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	for (int i = 0; i < soft_body->get_exceptions().size(); i++) {
		p_exceptions->push_back(soft_body->get_exceptions()[i]);
	}
}

void PhysicsServer3DSW::soft_body_set_state(RID p_body, BodyState p_state, const Variant &p_variant) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_state(p_state, p_variant);
}

Variant PhysicsServer3DSW::soft_body_get_state(RID p_body, BodyState p_state) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, Variant());

	return soft_body->get_state(p_state);
}

void PhysicsServer3DSW::soft_body_set_transform(RID p_body, const Transform &p_transform) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_state(BODY_STATE_TRANSFORM, p_transform);
}

void PhysicsServer3DSW::soft_body_set_ray_pickable(RID p_body, bool p_enable) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_ray_pickable(p_enable);
}

void PhysicsServer3DSW::soft_body_set_simulation_precision(RID p_body, int p_simulation_precision) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_iteration_count(p_simulation_precision);
}

int PhysicsServer3DSW::soft_body_get_simulation_precision(RID p_body) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, 0.f);

	return soft_body->get_iteration_count();
}

void PhysicsServer3DSW::soft_body_set_total_mass(RID p_body, real_t p_total_mass) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_total_mass(p_total_mass);
}

real_t PhysicsServer3DSW::soft_body_get_total_mass(RID p_body) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, 0.f);

	return soft_body->get_total_mass();
}

void PhysicsServer3DSW::soft_body_set_linear_stiffness(RID p_body, real_t p_stiffness) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_linear_stiffness(p_stiffness);
}

real_t PhysicsServer3DSW::soft_body_get_linear_stiffness(RID p_body) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, 0.f);

	return soft_body->get_linear_stiffness();
}

void PhysicsServer3DSW::soft_body_set_pressure_coefficient(RID p_body, real_t p_pressure_coefficient) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_pressure_coefficient(p_pressure_coefficient);
}

real_t PhysicsServer3DSW::soft_body_get_pressure_coefficient(RID p_body) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, 0.f);

	return soft_body->get_pressure_coefficient();
}

void PhysicsServer3DSW::soft_body_set_damping_coefficient(RID p_body, real_t p_damping_coefficient) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_damping_coefficient(p_damping_coefficient);
}

real_t PhysicsServer3DSW::soft_body_get_damping_coefficient(RID p_body) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, 0.f);

	return soft_body->get_damping_coefficient();
}

void PhysicsServer3DSW::soft_body_set_drag_coefficient(RID p_body, real_t p_drag_coefficient) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_drag_coefficient(p_drag_coefficient);
}

real_t PhysicsServer3DSW::soft_body_get_drag_coefficient(RID p_body) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, 0.f);

	return soft_body->get_drag_coefficient();
}

void PhysicsServer3DSW::soft_body_set_mesh(RID p_body, const REF &p_mesh) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_mesh(p_mesh);
}

AABB PhysicsServer3DSW::soft_body_get_bounds(RID p_body) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, AABB());

	return soft_body->get_bounds();
}

void PhysicsServer3DSW::soft_body_move_point(RID p_body, int p_point_index, const Vector3 &p_global_position) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->set_vertex_position(p_point_index, p_global_position);
}

Vector3 PhysicsServer3DSW::soft_body_get_point_global_position(RID p_body, int p_point_index) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, Vector3());

	return soft_body->get_vertex_position(p_point_index);
}

void PhysicsServer3DSW::soft_body_remove_all_pinned_points(RID p_body) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	soft_body->unpin_all_vertices();
}

void PhysicsServer3DSW::soft_body_pin_point(RID p_body, int p_point_index, bool p_pin) {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND(!soft_body);

	if (p_pin) {
		soft_body->pin_vertex(p_point_index);
	} else {
		soft_body->unpin_vertex(p_point_index);
	}
}

bool PhysicsServer3DSW::soft_body_is_point_pinned(RID p_body, int p_point_index) const {
	SoftBody3DSW *soft_body = soft_body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!soft_body, false);

	return soft_body->is_vertex_pinned(p_point_index);
}

/* JOINT API */

RID PhysicsServer3DSW::joint_create() {
	Joint3DSW *joint = memnew(Joint3DSW);
	RID rid = joint_owner.make_rid(joint);
	joint->set_self(rid);
	return rid;
}

void PhysicsServer3DSW::joint_clear(RID p_joint) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	if (joint->get_type() != JOINT_TYPE_MAX) {
		Joint3DSW *empty_joint = memnew(Joint3DSW);
		empty_joint->copy_settings_from(joint);

		joint_owner.replace(p_joint, empty_joint);
		memdelete(joint);
	}
}

void PhysicsServer3DSW::joint_make_pin(RID p_joint, RID p_body_A, const Vector3 &p_local_A, RID p_body_B, const Vector3 &p_local_B) {
	Body3DSW *body_A = body_owner.getornull(p_body_A);
	ERR_FAIL_COND(!body_A);

	if (!p_body_B.is_valid()) {
		ERR_FAIL_COND(!body_A->get_space());
		p_body_B = body_A->get_space()->get_static_global_body();
	}

	Body3DSW *body_B = body_owner.getornull(p_body_B);
	ERR_FAIL_COND(!body_B);

	ERR_FAIL_COND(body_A == body_B);

	Joint3DSW *prev_joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(prev_joint == nullptr);

	Joint3DSW *joint = memnew(PinJoint3DSW(body_A, p_local_A, body_B, p_local_B));

	joint->copy_settings_from(prev_joint);
	joint_owner.replace(p_joint, joint);
	memdelete(prev_joint);
}

void PhysicsServer3DSW::pin_joint_set_param(RID p_joint, PinJointParam p_param, real_t p_value) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);
	ERR_FAIL_COND(joint->get_type() != JOINT_TYPE_PIN);
	PinJoint3DSW *pin_joint = static_cast<PinJoint3DSW *>(joint);
	pin_joint->set_param(p_param, p_value);
}

real_t PhysicsServer3DSW::pin_joint_get_param(RID p_joint, PinJointParam p_param) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, 0);
	ERR_FAIL_COND_V(joint->get_type() != JOINT_TYPE_PIN, 0);
	PinJoint3DSW *pin_joint = static_cast<PinJoint3DSW *>(joint);
	return pin_joint->get_param(p_param);
}

void PhysicsServer3DSW::pin_joint_set_local_a(RID p_joint, const Vector3 &p_A) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);
	ERR_FAIL_COND(joint->get_type() != JOINT_TYPE_PIN);
	PinJoint3DSW *pin_joint = static_cast<PinJoint3DSW *>(joint);
	pin_joint->set_pos_a(p_A);
}

Vector3 PhysicsServer3DSW::pin_joint_get_local_a(RID p_joint) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, Vector3());
	ERR_FAIL_COND_V(joint->get_type() != JOINT_TYPE_PIN, Vector3());
	PinJoint3DSW *pin_joint = static_cast<PinJoint3DSW *>(joint);
	return pin_joint->get_position_a();
}

void PhysicsServer3DSW::pin_joint_set_local_b(RID p_joint, const Vector3 &p_B) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);
	ERR_FAIL_COND(joint->get_type() != JOINT_TYPE_PIN);
	PinJoint3DSW *pin_joint = static_cast<PinJoint3DSW *>(joint);
	pin_joint->set_pos_b(p_B);
}

Vector3 PhysicsServer3DSW::pin_joint_get_local_b(RID p_joint) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, Vector3());
	ERR_FAIL_COND_V(joint->get_type() != JOINT_TYPE_PIN, Vector3());
	PinJoint3DSW *pin_joint = static_cast<PinJoint3DSW *>(joint);
	return pin_joint->get_position_b();
}

void PhysicsServer3DSW::joint_make_hinge(RID p_joint, RID p_body_A, const Transform &p_frame_A, RID p_body_B, const Transform &p_frame_B) {
	Body3DSW *body_A = body_owner.getornull(p_body_A);
	ERR_FAIL_COND(!body_A);

	if (!p_body_B.is_valid()) {
		ERR_FAIL_COND(!body_A->get_space());
		p_body_B = body_A->get_space()->get_static_global_body();
	}

	Body3DSW *body_B = body_owner.getornull(p_body_B);
	ERR_FAIL_COND(!body_B);

	ERR_FAIL_COND(body_A == body_B);

	Joint3DSW *prev_joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(prev_joint == nullptr);

	Joint3DSW *joint = memnew(HingeJoint3DSW(body_A, body_B, p_frame_A, p_frame_B));

	joint->copy_settings_from(prev_joint);
	joint_owner.replace(p_joint, joint);
	memdelete(prev_joint);
}

void PhysicsServer3DSW::joint_make_hinge_simple(RID p_joint, RID p_body_A, const Vector3 &p_pivot_A, const Vector3 &p_axis_A, RID p_body_B, const Vector3 &p_pivot_B, const Vector3 &p_axis_B) {
	Body3DSW *body_A = body_owner.getornull(p_body_A);
	ERR_FAIL_COND(!body_A);

	if (!p_body_B.is_valid()) {
		ERR_FAIL_COND(!body_A->get_space());
		p_body_B = body_A->get_space()->get_static_global_body();
	}

	Body3DSW *body_B = body_owner.getornull(p_body_B);
	ERR_FAIL_COND(!body_B);

	ERR_FAIL_COND(body_A == body_B);

	Joint3DSW *prev_joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(prev_joint == nullptr);

	Joint3DSW *joint = memnew(HingeJoint3DSW(body_A, body_B, p_pivot_A, p_pivot_B, p_axis_A, p_axis_B));

	joint->copy_settings_from(prev_joint);
	joint_owner.replace(p_joint, joint);
	memdelete(prev_joint);
}

void PhysicsServer3DSW::hinge_joint_set_param(RID p_joint, HingeJointParam p_param, real_t p_value) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);
	ERR_FAIL_COND(joint->get_type() != JOINT_TYPE_HINGE);
	HingeJoint3DSW *hinge_joint = static_cast<HingeJoint3DSW *>(joint);
	hinge_joint->set_param(p_param, p_value);
}

real_t PhysicsServer3DSW::hinge_joint_get_param(RID p_joint, HingeJointParam p_param) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, 0);
	ERR_FAIL_COND_V(joint->get_type() != JOINT_TYPE_HINGE, 0);
	HingeJoint3DSW *hinge_joint = static_cast<HingeJoint3DSW *>(joint);
	return hinge_joint->get_param(p_param);
}

void PhysicsServer3DSW::hinge_joint_set_flag(RID p_joint, HingeJointFlag p_flag, bool p_value) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);
	ERR_FAIL_COND(joint->get_type() != JOINT_TYPE_HINGE);
	HingeJoint3DSW *hinge_joint = static_cast<HingeJoint3DSW *>(joint);
	hinge_joint->set_flag(p_flag, p_value);
}

bool PhysicsServer3DSW::hinge_joint_get_flag(RID p_joint, HingeJointFlag p_flag) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, false);
	ERR_FAIL_COND_V(joint->get_type() != JOINT_TYPE_HINGE, false);
	HingeJoint3DSW *hinge_joint = static_cast<HingeJoint3DSW *>(joint);
	return hinge_joint->get_flag(p_flag);
}

void PhysicsServer3DSW::joint_set_solver_priority(RID p_joint, int p_priority) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);
	joint->set_priority(p_priority);
}

int PhysicsServer3DSW::joint_get_solver_priority(RID p_joint) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, 0);
	return joint->get_priority();
}

void PhysicsServer3DSW::joint_disable_collisions_between_bodies(RID p_joint, const bool p_disable) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);

	joint->disable_collisions_between_bodies(p_disable);

	if (2 == joint->get_body_count()) {
		Body3DSW *body_a = *joint->get_body_ptr();
		Body3DSW *body_b = *(joint->get_body_ptr() + 1);

		if (p_disable) {
			body_add_collision_exception(body_a->get_self(), body_b->get_self());
			body_add_collision_exception(body_b->get_self(), body_a->get_self());
		} else {
			body_remove_collision_exception(body_a->get_self(), body_b->get_self());
			body_remove_collision_exception(body_b->get_self(), body_a->get_self());
		}
	}
}

bool PhysicsServer3DSW::joint_is_disabled_collisions_between_bodies(RID p_joint) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, true);

	return joint->is_disabled_collisions_between_bodies();
}

PhysicsServer3DSW::JointType PhysicsServer3DSW::joint_get_type(RID p_joint) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, JOINT_TYPE_PIN);
	return joint->get_type();
}

void PhysicsServer3DSW::joint_make_slider(RID p_joint, RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B) {
	Body3DSW *body_A = body_owner.getornull(p_body_A);
	ERR_FAIL_COND(!body_A);

	if (!p_body_B.is_valid()) {
		ERR_FAIL_COND(!body_A->get_space());
		p_body_B = body_A->get_space()->get_static_global_body();
	}

	Body3DSW *body_B = body_owner.getornull(p_body_B);
	ERR_FAIL_COND(!body_B);

	ERR_FAIL_COND(body_A == body_B);

	Joint3DSW *prev_joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(prev_joint == nullptr);

	Joint3DSW *joint = memnew(SliderJoint3DSW(body_A, body_B, p_local_frame_A, p_local_frame_B));

	joint->copy_settings_from(prev_joint);
	joint_owner.replace(p_joint, joint);
	memdelete(prev_joint);
}

void PhysicsServer3DSW::slider_joint_set_param(RID p_joint, SliderJointParam p_param, real_t p_value) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);
	ERR_FAIL_COND(joint->get_type() != JOINT_TYPE_SLIDER);
	SliderJoint3DSW *slider_joint = static_cast<SliderJoint3DSW *>(joint);
	slider_joint->set_param(p_param, p_value);
}

real_t PhysicsServer3DSW::slider_joint_get_param(RID p_joint, SliderJointParam p_param) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, 0);
	ERR_FAIL_COND_V(joint->get_type() != JOINT_TYPE_CONE_TWIST, 0);
	SliderJoint3DSW *slider_joint = static_cast<SliderJoint3DSW *>(joint);
	return slider_joint->get_param(p_param);
}

void PhysicsServer3DSW::joint_make_cone_twist(RID p_joint, RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B) {
	Body3DSW *body_A = body_owner.getornull(p_body_A);
	ERR_FAIL_COND(!body_A);

	if (!p_body_B.is_valid()) {
		ERR_FAIL_COND(!body_A->get_space());
		p_body_B = body_A->get_space()->get_static_global_body();
	}

	Body3DSW *body_B = body_owner.getornull(p_body_B);
	ERR_FAIL_COND(!body_B);

	ERR_FAIL_COND(body_A == body_B);

	Joint3DSW *prev_joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(prev_joint == nullptr);

	Joint3DSW *joint = memnew(ConeTwistJoint3DSW(body_A, body_B, p_local_frame_A, p_local_frame_B));

	joint->copy_settings_from(prev_joint);
	joint_owner.replace(p_joint, joint);
	memdelete(prev_joint);
}

void PhysicsServer3DSW::cone_twist_joint_set_param(RID p_joint, ConeTwistJointParam p_param, real_t p_value) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);
	ERR_FAIL_COND(joint->get_type() != JOINT_TYPE_CONE_TWIST);
	ConeTwistJoint3DSW *cone_twist_joint = static_cast<ConeTwistJoint3DSW *>(joint);
	cone_twist_joint->set_param(p_param, p_value);
}

real_t PhysicsServer3DSW::cone_twist_joint_get_param(RID p_joint, ConeTwistJointParam p_param) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, 0);
	ERR_FAIL_COND_V(joint->get_type() != JOINT_TYPE_CONE_TWIST, 0);
	ConeTwistJoint3DSW *cone_twist_joint = static_cast<ConeTwistJoint3DSW *>(joint);
	return cone_twist_joint->get_param(p_param);
}

void PhysicsServer3DSW::joint_make_generic_6dof(RID p_joint, RID p_body_A, const Transform &p_local_frame_A, RID p_body_B, const Transform &p_local_frame_B) {
	Body3DSW *body_A = body_owner.getornull(p_body_A);
	ERR_FAIL_COND(!body_A);

	if (!p_body_B.is_valid()) {
		ERR_FAIL_COND(!body_A->get_space());
		p_body_B = body_A->get_space()->get_static_global_body();
	}

	Body3DSW *body_B = body_owner.getornull(p_body_B);
	ERR_FAIL_COND(!body_B);

	ERR_FAIL_COND(body_A == body_B);

	Joint3DSW *prev_joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(prev_joint == nullptr);

	Joint3DSW *joint = memnew(Generic6DOFJoint3DSW(body_A, body_B, p_local_frame_A, p_local_frame_B, true));

	joint->copy_settings_from(prev_joint);
	joint_owner.replace(p_joint, joint);
	memdelete(prev_joint);
}

void PhysicsServer3DSW::generic_6dof_joint_set_param(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisParam p_param, real_t p_value) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);
	ERR_FAIL_COND(joint->get_type() != JOINT_TYPE_6DOF);
	Generic6DOFJoint3DSW *generic_6dof_joint = static_cast<Generic6DOFJoint3DSW *>(joint);
	generic_6dof_joint->set_param(p_axis, p_param, p_value);
}

real_t PhysicsServer3DSW::generic_6dof_joint_get_param(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisParam p_param) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, 0);
	ERR_FAIL_COND_V(joint->get_type() != JOINT_TYPE_6DOF, 0);
	Generic6DOFJoint3DSW *generic_6dof_joint = static_cast<Generic6DOFJoint3DSW *>(joint);
	return generic_6dof_joint->get_param(p_axis, p_param);
}

void PhysicsServer3DSW::generic_6dof_joint_set_flag(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisFlag p_flag, bool p_enable) {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);
	ERR_FAIL_COND(joint->get_type() != JOINT_TYPE_6DOF);
	Generic6DOFJoint3DSW *generic_6dof_joint = static_cast<Generic6DOFJoint3DSW *>(joint);
	generic_6dof_joint->set_flag(p_axis, p_flag, p_enable);
}

bool PhysicsServer3DSW::generic_6dof_joint_get_flag(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisFlag p_flag) const {
	Joint3DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, false);
	ERR_FAIL_COND_V(joint->get_type() != JOINT_TYPE_6DOF, false);
	Generic6DOFJoint3DSW *generic_6dof_joint = static_cast<Generic6DOFJoint3DSW *>(joint);
	return generic_6dof_joint->get_flag(p_axis, p_flag);
}

void PhysicsServer3DSW::free(RID p_rid) {
	_update_shapes(); //just in case

	if (shape_owner.owns(p_rid)) {
		Shape3DSW *shape = shape_owner.getornull(p_rid);

		while (shape->get_owners().size()) {
			ShapeOwner3DSW *so = shape->get_owners().front()->key();
			so->remove_shape(shape);
		}

		shape_owner.free(p_rid);
		memdelete(shape);
	} else if (body_owner.owns(p_rid)) {
		Body3DSW *body = body_owner.getornull(p_rid);

		/*
		if (body->get_state_query())
			_clear_query(body->get_state_query());

		if (body->get_direct_state_query())
			_clear_query(body->get_direct_state_query());
		*/

		body->set_space(nullptr);

		while (body->get_shape_count()) {
			body->remove_shape(0);
		}

		body_owner.free(p_rid);
		memdelete(body);
	} else if (soft_body_owner.owns(p_rid)) {
		SoftBody3DSW *soft_body = soft_body_owner.getornull(p_rid);

		soft_body->set_space(nullptr);

		soft_body_owner.free(p_rid);
		memdelete(soft_body);
	} else if (area_owner.owns(p_rid)) {
		Area3DSW *area = area_owner.getornull(p_rid);

		/*
		if (area->get_monitor_query())
			_clear_query(area->get_monitor_query());
		*/

		area->set_space(nullptr);

		while (area->get_shape_count()) {
			area->remove_shape(0);
		}

		area_owner.free(p_rid);
		memdelete(area);
	} else if (space_owner.owns(p_rid)) {
		Space3DSW *space = space_owner.getornull(p_rid);

		while (space->get_objects().size()) {
			CollisionObject3DSW *co = (CollisionObject3DSW *)space->get_objects().front()->get();
			co->set_space(nullptr);
		}

		active_spaces.erase(space);
		free(space->get_default_area()->get_self());
		free(space->get_static_global_body());

		space_owner.free(p_rid);
		memdelete(space);
	} else if (joint_owner.owns(p_rid)) {
		Joint3DSW *joint = joint_owner.getornull(p_rid);

		joint_owner.free(p_rid);
		memdelete(joint);

	} else {
		ERR_FAIL_MSG("Invalid ID.");
	}
};

void PhysicsServer3DSW::set_active(bool p_active) {
	active = p_active;
};

void PhysicsServer3DSW::init() {
	last_step = 0.001;
	iterations = 8; // 8?
	stepper = memnew(Step3DSW);
	direct_state = memnew(PhysicsDirectBodyState3DSW);
};

void PhysicsServer3DSW::step(real_t p_step) {
#ifndef _3D_DISABLED

	if (!active) {
		return;
	}

	_update_shapes();

	last_step = p_step;
	PhysicsDirectBodyState3DSW::singleton->step = p_step;

	island_count = 0;
	active_objects = 0;
	collision_pairs = 0;
	for (Set<const Space3DSW *>::Element *E = active_spaces.front(); E; E = E->next()) {
		stepper->step((Space3DSW *)E->get(), p_step, iterations);
		island_count += E->get()->get_island_count();
		active_objects += E->get()->get_active_objects();
		collision_pairs += E->get()->get_collision_pairs();
	}
#endif
}

void PhysicsServer3DSW::sync() {
	doing_sync = true;
};

void PhysicsServer3DSW::flush_queries() {
#ifndef _3D_DISABLED

	if (!active) {
		return;
	}

	flushing_queries = true;

	uint64_t time_beg = OS::get_singleton()->get_ticks_usec();

	for (Set<const Space3DSW *>::Element *E = active_spaces.front(); E; E = E->next()) {
		Space3DSW *space = (Space3DSW *)E->get();
		space->call_queries();
	}

	flushing_queries = false;

	if (EngineDebugger::is_profiling("servers")) {
		uint64_t total_time[Space3DSW::ELAPSED_TIME_MAX];
		static const char *time_name[Space3DSW::ELAPSED_TIME_MAX] = {
			"integrate_forces",
			"generate_islands",
			"setup_constraints",
			"solve_constraints",
			"integrate_velocities"
		};

		for (int i = 0; i < Space3DSW::ELAPSED_TIME_MAX; i++) {
			total_time[i] = 0;
		}

		for (Set<const Space3DSW *>::Element *E = active_spaces.front(); E; E = E->next()) {
			for (int i = 0; i < Space3DSW::ELAPSED_TIME_MAX; i++) {
				total_time[i] += E->get()->get_elapsed_time(Space3DSW::ElapsedTime(i));
			}
		}

		Array values;
		values.resize(Space3DSW::ELAPSED_TIME_MAX * 2);
		for (int i = 0; i < Space3DSW::ELAPSED_TIME_MAX; i++) {
			values[i * 2 + 0] = time_name[i];
			values[i * 2 + 1] = USEC_TO_SEC(total_time[i]);
		}
		values.push_back("flush_queries");
		values.push_back(USEC_TO_SEC(OS::get_singleton()->get_ticks_usec() - time_beg));

		values.push_front("physics");
		EngineDebugger::profiler_add_frame_data("servers", values);
	}
#endif
};

void PhysicsServer3DSW::end_sync() {
	doing_sync = false;
};

void PhysicsServer3DSW::finish() {
	memdelete(stepper);
	memdelete(direct_state);
};

int PhysicsServer3DSW::get_process_info(ProcessInfo p_info) {
	switch (p_info) {
		case INFO_ACTIVE_OBJECTS: {
			return active_objects;
		} break;
		case INFO_COLLISION_PAIRS: {
			return collision_pairs;
		} break;
		case INFO_ISLAND_COUNT: {
			return island_count;
		} break;
	}

	return 0;
}

void PhysicsServer3DSW::_update_shapes() {
	while (pending_shape_update_list.first()) {
		pending_shape_update_list.first()->self()->_shape_changed();
		pending_shape_update_list.remove(pending_shape_update_list.first());
	}
}

void PhysicsServer3DSW::_shape_col_cbk(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B, void *p_userdata) {
	CollCbkData *cbk = (CollCbkData *)p_userdata;

	if (cbk->max == 0) {
		return;
	}

	if (cbk->amount == cbk->max) {
		//find least deep
		real_t min_depth = 1e20;
		int min_depth_idx = 0;
		for (int i = 0; i < cbk->amount; i++) {
			real_t d = cbk->ptr[i * 2 + 0].distance_squared_to(cbk->ptr[i * 2 + 1]);
			if (d < min_depth) {
				min_depth = d;
				min_depth_idx = i;
			}
		}

		real_t d = p_point_A.distance_squared_to(p_point_B);
		if (d < min_depth) {
			return;
		}
		cbk->ptr[min_depth_idx * 2 + 0] = p_point_A;
		cbk->ptr[min_depth_idx * 2 + 1] = p_point_B;

	} else {
		cbk->ptr[cbk->amount * 2 + 0] = p_point_A;
		cbk->ptr[cbk->amount * 2 + 1] = p_point_B;
		cbk->amount++;
	}
}

PhysicsServer3DSW *PhysicsServer3DSW::singletonsw = nullptr;
PhysicsServer3DSW::PhysicsServer3DSW(bool p_using_threads) {
	singletonsw = this;
	BroadPhase3DSW::create_func = BroadPhase3DBVH::_create;

	island_count = 0;
	active_objects = 0;
	collision_pairs = 0;
	using_threads = p_using_threads;
	active = true;
	flushing_queries = false;
	doing_sync = false;
};
