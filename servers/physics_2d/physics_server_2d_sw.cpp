/*************************************************************************/
/*  physics_server_2d_sw.cpp                                             */
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

#include "physics_server_2d_sw.h"

#include "broad_phase_2d_basic.h"
#include "broad_phase_2d_hash_grid.h"
#include "collision_solver_2d_sw.h"
#include "core/debugger/engine_debugger.h"
#include "core/os/os.h"
#include "core/project_settings.h"

#define FLUSH_QUERY_CHECK(m_object) \
	ERR_FAIL_COND_MSG(m_object->get_space() && flushing_queries, "Can't change this state while flushing queries. Use call_deferred() or set_deferred() to change monitoring state instead.");

RID PhysicsServer2DSW::_shape_create(ShapeType p_shape) {
	Shape2DSW *shape = nullptr;
	switch (p_shape) {
		case SHAPE_LINE: {
			shape = memnew(LineShape2DSW);
		} break;
		case SHAPE_RAY: {
			shape = memnew(RayShape2DSW);
		} break;
		case SHAPE_SEGMENT: {
			shape = memnew(SegmentShape2DSW);
		} break;
		case SHAPE_CIRCLE: {
			shape = memnew(CircleShape2DSW);
		} break;
		case SHAPE_RECTANGLE: {
			shape = memnew(RectangleShape2DSW);
		} break;
		case SHAPE_CAPSULE: {
			shape = memnew(CapsuleShape2DSW);
		} break;
		case SHAPE_CONVEX_POLYGON: {
			shape = memnew(ConvexPolygonShape2DSW);
		} break;
		case SHAPE_CONCAVE_POLYGON: {
			shape = memnew(ConcavePolygonShape2DSW);
		} break;
		case SHAPE_CUSTOM: {
			ERR_FAIL_V(RID());

		} break;
	}

	RID id = shape_owner.make_rid(shape);
	shape->set_self(id);

	return id;
}

RID PhysicsServer2DSW::line_shape_create() {
	return _shape_create(SHAPE_LINE);
}

RID PhysicsServer2DSW::ray_shape_create() {
	return _shape_create(SHAPE_RAY);
}

RID PhysicsServer2DSW::segment_shape_create() {
	return _shape_create(SHAPE_SEGMENT);
}

RID PhysicsServer2DSW::circle_shape_create() {
	return _shape_create(SHAPE_CIRCLE);
}

RID PhysicsServer2DSW::rectangle_shape_create() {
	return _shape_create(SHAPE_RECTANGLE);
}

RID PhysicsServer2DSW::capsule_shape_create() {
	return _shape_create(SHAPE_CAPSULE);
}

RID PhysicsServer2DSW::convex_polygon_shape_create() {
	return _shape_create(SHAPE_CONVEX_POLYGON);
}

RID PhysicsServer2DSW::concave_polygon_shape_create() {
	return _shape_create(SHAPE_CONCAVE_POLYGON);
}

void PhysicsServer2DSW::shape_set_data(RID p_shape, const Variant &p_data) {
	Shape2DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);
	shape->set_data(p_data);
};

void PhysicsServer2DSW::shape_set_custom_solver_bias(RID p_shape, real_t p_bias) {
	Shape2DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);
	shape->set_custom_bias(p_bias);
}

PhysicsServer2D::ShapeType PhysicsServer2DSW::shape_get_type(RID p_shape) const {
	const Shape2DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND_V(!shape, SHAPE_CUSTOM);
	return shape->get_type();
};

Variant PhysicsServer2DSW::shape_get_data(RID p_shape) const {
	const Shape2DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND_V(!shape, Variant());
	ERR_FAIL_COND_V(!shape->is_configured(), Variant());
	return shape->get_data();
};

real_t PhysicsServer2DSW::shape_get_custom_solver_bias(RID p_shape) const {
	const Shape2DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND_V(!shape, 0);
	return shape->get_custom_bias();
}

void PhysicsServer2DSW::_shape_col_cbk(const Vector2 &p_point_A, const Vector2 &p_point_B, void *p_userdata) {
	CollCbkData *cbk = (CollCbkData *)p_userdata;

	if (cbk->max == 0) {
		return;
	}

	if (cbk->valid_dir != Vector2()) {
		if (p_point_A.distance_squared_to(p_point_B) > cbk->valid_depth * cbk->valid_depth) {
			cbk->invalid_by_dir++;
			return;
		}
		Vector2 rel_dir = (p_point_A - p_point_B).normalized();

		if (cbk->valid_dir.dot(rel_dir) < Math_SQRT12) { //sqrt(2)/2.0 - 45 degrees
			cbk->invalid_by_dir++;

			/*
			print_line("A: "+p_point_A);
			print_line("B: "+p_point_B);
			print_line("discard too angled "+rtos(cbk->valid_dir.dot((p_point_A-p_point_B))));
			print_line("resnorm: "+(p_point_A-p_point_B).normalized());
			print_line("distance: "+rtos(p_point_A.distance_to(p_point_B)));
			*/
			return;
		}
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
		cbk->passed++;

	} else {
		cbk->ptr[cbk->amount * 2 + 0] = p_point_A;
		cbk->ptr[cbk->amount * 2 + 1] = p_point_B;
		cbk->amount++;
		cbk->passed++;
	}
}

bool PhysicsServer2DSW::shape_collide(RID p_shape_A, const Transform2D &p_xform_A, const Vector2 &p_motion_A, RID p_shape_B, const Transform2D &p_xform_B, const Vector2 &p_motion_B, Vector2 *r_results, int p_result_max, int &r_result_count) {
	Shape2DSW *shape_A = shape_owner.getornull(p_shape_A);
	ERR_FAIL_COND_V(!shape_A, false);
	Shape2DSW *shape_B = shape_owner.getornull(p_shape_B);
	ERR_FAIL_COND_V(!shape_B, false);

	if (p_result_max == 0) {
		return CollisionSolver2DSW::solve(shape_A, p_xform_A, p_motion_A, shape_B, p_xform_B, p_motion_B, nullptr, nullptr);
	}

	CollCbkData cbk;
	cbk.max = p_result_max;
	cbk.amount = 0;
	cbk.passed = 0;
	cbk.ptr = r_results;

	bool res = CollisionSolver2DSW::solve(shape_A, p_xform_A, p_motion_A, shape_B, p_xform_B, p_motion_B, _shape_col_cbk, &cbk);
	r_result_count = cbk.amount;
	return res;
}

RID PhysicsServer2DSW::space_create() {
	Space2DSW *space = memnew(Space2DSW);
	RID id = space_owner.make_rid(space);
	space->set_self(id);
	RID area_id = area_create();
	Area2DSW *area = area_owner.getornull(area_id);
	ERR_FAIL_COND_V(!area, RID());
	space->set_default_area(area);
	area->set_space(space);
	area->set_priority(-1);

	return id;
};

void PhysicsServer2DSW::space_set_active(RID p_space, bool p_active) {
	Space2DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND(!space);
	if (p_active) {
		active_spaces.insert(space);
	} else {
		active_spaces.erase(space);
	}
}

bool PhysicsServer2DSW::space_is_active(RID p_space) const {
	const Space2DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND_V(!space, false);

	return active_spaces.has(space);
}

void PhysicsServer2DSW::space_set_param(RID p_space, SpaceParameter p_param, real_t p_value) {
	Space2DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND(!space);

	space->set_param(p_param, p_value);
}

real_t PhysicsServer2DSW::space_get_param(RID p_space, SpaceParameter p_param) const {
	const Space2DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND_V(!space, 0);
	return space->get_param(p_param);
}

void PhysicsServer2DSW::space_set_debug_contacts(RID p_space, int p_max_contacts) {
	Space2DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND(!space);
	space->set_debug_contacts(p_max_contacts);
}

Vector<Vector2> PhysicsServer2DSW::space_get_contacts(RID p_space) const {
	Space2DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND_V(!space, Vector<Vector2>());
	return space->get_debug_contacts();
}

int PhysicsServer2DSW::space_get_contact_count(RID p_space) const {
	Space2DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND_V(!space, 0);
	return space->get_debug_contact_count();
}

PhysicsDirectSpaceState2D *PhysicsServer2DSW::space_get_direct_state(RID p_space) {
	Space2DSW *space = space_owner.getornull(p_space);
	ERR_FAIL_COND_V(!space, nullptr);
	ERR_FAIL_COND_V_MSG((using_threads && !doing_sync) || space->is_locked(), nullptr, "Space state is inaccessible right now, wait for iteration or physics process notification.");

	return space->get_direct_state();
}

RID PhysicsServer2DSW::area_create() {
	Area2DSW *area = memnew(Area2DSW);
	RID rid = area_owner.make_rid(area);
	area->set_self(rid);
	return rid;
};

void PhysicsServer2DSW::area_set_space(RID p_area, RID p_space) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	Space2DSW *space = nullptr;
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

RID PhysicsServer2DSW::area_get_space(RID p_area) const {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, RID());

	Space2DSW *space = area->get_space();
	if (!space) {
		return RID();
	}
	return space->get_self();
};

void PhysicsServer2DSW::area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_space_override_mode(p_mode);
}

PhysicsServer2D::AreaSpaceOverrideMode PhysicsServer2DSW::area_get_space_override_mode(RID p_area) const {
	const Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, AREA_SPACE_OVERRIDE_DISABLED);

	return area->get_space_override_mode();
}

void PhysicsServer2DSW::area_add_shape(RID p_area, RID p_shape, const Transform2D &p_transform, bool p_disabled) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	Shape2DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);

	area->add_shape(shape, p_transform, p_disabled);
}

void PhysicsServer2DSW::area_set_shape(RID p_area, int p_shape_idx, RID p_shape) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	Shape2DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);
	ERR_FAIL_COND(!shape->is_configured());

	area->set_shape(p_shape_idx, shape);
}

void PhysicsServer2DSW::area_set_shape_transform(RID p_area, int p_shape_idx, const Transform2D &p_transform) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_shape_transform(p_shape_idx, p_transform);
}

void PhysicsServer2DSW::area_set_shape_disabled(RID p_area, int p_shape, bool p_disabled) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	ERR_FAIL_INDEX(p_shape, area->get_shape_count());
	FLUSH_QUERY_CHECK(area);

	area->set_shape_as_disabled(p_shape, p_disabled);
}

int PhysicsServer2DSW::area_get_shape_count(RID p_area) const {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, -1);

	return area->get_shape_count();
}

RID PhysicsServer2DSW::area_get_shape(RID p_area, int p_shape_idx) const {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, RID());

	Shape2DSW *shape = area->get_shape(p_shape_idx);
	ERR_FAIL_COND_V(!shape, RID());

	return shape->get_self();
}

Transform2D PhysicsServer2DSW::area_get_shape_transform(RID p_area, int p_shape_idx) const {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, Transform2D());

	return area->get_shape_transform(p_shape_idx);
}

void PhysicsServer2DSW::area_remove_shape(RID p_area, int p_shape_idx) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->remove_shape(p_shape_idx);
}

void PhysicsServer2DSW::area_clear_shapes(RID p_area) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	while (area->get_shape_count()) {
		area->remove_shape(0);
	}
}

void PhysicsServer2DSW::area_attach_object_instance_id(RID p_area, ObjectID p_id) {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.getornull(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	area->set_instance_id(p_id);
}

ObjectID PhysicsServer2DSW::area_get_object_instance_id(RID p_area) const {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.getornull(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, ObjectID());
	return area->get_instance_id();
}

void PhysicsServer2DSW::area_attach_canvas_instance_id(RID p_area, ObjectID p_id) {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.getornull(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	area->set_canvas_instance_id(p_id);
}

ObjectID PhysicsServer2DSW::area_get_canvas_instance_id(RID p_area) const {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.getornull(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, ObjectID());
	return area->get_canvas_instance_id();
}

void PhysicsServer2DSW::area_set_param(RID p_area, AreaParameter p_param, const Variant &p_value) {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.getornull(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	area->set_param(p_param, p_value);
};

void PhysicsServer2DSW::area_set_transform(RID p_area, const Transform2D &p_transform) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	area->set_transform(p_transform);
};

Variant PhysicsServer2DSW::area_get_param(RID p_area, AreaParameter p_param) const {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.getornull(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, Variant());

	return area->get_param(p_param);
};

Transform2D PhysicsServer2DSW::area_get_transform(RID p_area) const {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND_V(!area, Transform2D());

	return area->get_transform();
};

void PhysicsServer2DSW::area_set_pickable(RID p_area, bool p_pickable) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	area->set_pickable(p_pickable);
}

void PhysicsServer2DSW::area_set_monitorable(RID p_area, bool p_monitorable) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);
	FLUSH_QUERY_CHECK(area);

	area->set_monitorable(p_monitorable);
}

void PhysicsServer2DSW::area_set_collision_mask(RID p_area, uint32_t p_mask) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_collision_mask(p_mask);
}

void PhysicsServer2DSW::area_set_collision_layer(RID p_area, uint32_t p_layer) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_collision_layer(p_layer);
}

void PhysicsServer2DSW::area_set_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_monitor_callback(p_receiver ? p_receiver->get_instance_id() : ObjectID(), p_method);
}

void PhysicsServer2DSW::area_set_area_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) {
	Area2DSW *area = area_owner.getornull(p_area);
	ERR_FAIL_COND(!area);

	area->set_area_monitor_callback(p_receiver ? p_receiver->get_instance_id() : ObjectID(), p_method);
}

/* BODY API */

RID PhysicsServer2DSW::body_create() {
	Body2DSW *body = memnew(Body2DSW);
	RID rid = body_owner.make_rid(body);
	body->set_self(rid);
	return rid;
}

void PhysicsServer2DSW::body_set_space(RID p_body, RID p_space) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	Space2DSW *space = nullptr;
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

RID PhysicsServer2DSW::body_get_space(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, RID());

	Space2DSW *space = body->get_space();
	if (!space) {
		return RID();
	}
	return space->get_self();
};

void PhysicsServer2DSW::body_set_mode(RID p_body, BodyMode p_mode) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	FLUSH_QUERY_CHECK(body);

	body->set_mode(p_mode);
};

PhysicsServer2D::BodyMode PhysicsServer2DSW::body_get_mode(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, BODY_MODE_STATIC);

	return body->get_mode();
};

void PhysicsServer2DSW::body_add_shape(RID p_body, RID p_shape, const Transform2D &p_transform, bool p_disabled) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	Shape2DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);

	body->add_shape(shape, p_transform, p_disabled);
}

void PhysicsServer2DSW::body_set_shape(RID p_body, int p_shape_idx, RID p_shape) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	Shape2DSW *shape = shape_owner.getornull(p_shape);
	ERR_FAIL_COND(!shape);
	ERR_FAIL_COND(!shape->is_configured());

	body->set_shape(p_shape_idx, shape);
}

void PhysicsServer2DSW::body_set_shape_transform(RID p_body, int p_shape_idx, const Transform2D &p_transform) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_shape_transform(p_shape_idx, p_transform);
}

void PhysicsServer2DSW::body_set_shape_metadata(RID p_body, int p_shape_idx, const Variant &p_metadata) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	body->set_shape_metadata(p_shape_idx, p_metadata);
}

Variant PhysicsServer2DSW::body_get_shape_metadata(RID p_body, int p_shape_idx) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, Variant());
	return body->get_shape_metadata(p_shape_idx);
}

int PhysicsServer2DSW::body_get_shape_count(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, -1);

	return body->get_shape_count();
}

RID PhysicsServer2DSW::body_get_shape(RID p_body, int p_shape_idx) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, RID());

	Shape2DSW *shape = body->get_shape(p_shape_idx);
	ERR_FAIL_COND_V(!shape, RID());

	return shape->get_self();
}

Transform2D PhysicsServer2DSW::body_get_shape_transform(RID p_body, int p_shape_idx) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, Transform2D());

	return body->get_shape_transform(p_shape_idx);
}

void PhysicsServer2DSW::body_remove_shape(RID p_body, int p_shape_idx) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->remove_shape(p_shape_idx);
}

void PhysicsServer2DSW::body_clear_shapes(RID p_body) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	while (body->get_shape_count()) {
		body->remove_shape(0);
	}
}

void PhysicsServer2DSW::body_set_shape_disabled(RID p_body, int p_shape_idx, bool p_disabled) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	ERR_FAIL_INDEX(p_shape_idx, body->get_shape_count());
	FLUSH_QUERY_CHECK(body);

	body->set_shape_as_disabled(p_shape_idx, p_disabled);
}

void PhysicsServer2DSW::body_set_shape_as_one_way_collision(RID p_body, int p_shape_idx, bool p_enable, float p_margin) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	ERR_FAIL_INDEX(p_shape_idx, body->get_shape_count());
	FLUSH_QUERY_CHECK(body);

	body->set_shape_as_one_way_collision(p_shape_idx, p_enable, p_margin);
}

void PhysicsServer2DSW::body_set_continuous_collision_detection_mode(RID p_body, CCDMode p_mode) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	body->set_continuous_collision_detection_mode(p_mode);
}

PhysicsServer2DSW::CCDMode PhysicsServer2DSW::body_get_continuous_collision_detection_mode(RID p_body) const {
	const Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, CCD_MODE_DISABLED);

	return body->get_continuous_collision_detection_mode();
}

void PhysicsServer2DSW::body_attach_object_instance_id(RID p_body, ObjectID p_id) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_instance_id(p_id);
};

ObjectID PhysicsServer2DSW::body_get_object_instance_id(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, ObjectID());

	return body->get_instance_id();
};

void PhysicsServer2DSW::body_attach_canvas_instance_id(RID p_body, ObjectID p_id) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_canvas_instance_id(p_id);
};

ObjectID PhysicsServer2DSW::body_get_canvas_instance_id(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, ObjectID());

	return body->get_canvas_instance_id();
};

void PhysicsServer2DSW::body_set_collision_layer(RID p_body, uint32_t p_layer) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	body->set_collision_layer(p_layer);
};

uint32_t PhysicsServer2DSW::body_get_collision_layer(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_collision_layer();
};

void PhysicsServer2DSW::body_set_collision_mask(RID p_body, uint32_t p_mask) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	body->set_collision_mask(p_mask);
};

uint32_t PhysicsServer2DSW::body_get_collision_mask(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_collision_mask();
};

void PhysicsServer2DSW::body_set_param(RID p_body, BodyParameter p_param, real_t p_value) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_param(p_param, p_value);
};

real_t PhysicsServer2DSW::body_get_param(RID p_body, BodyParameter p_param) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_param(p_param);
};

void PhysicsServer2DSW::body_set_state(RID p_body, BodyState p_state, const Variant &p_variant) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_state(p_state, p_variant);
};

Variant PhysicsServer2DSW::body_get_state(RID p_body, BodyState p_state) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, Variant());

	return body->get_state(p_state);
};

void PhysicsServer2DSW::body_set_applied_force(RID p_body, const Vector2 &p_force) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_applied_force(p_force);
	body->wakeup();
};

Vector2 PhysicsServer2DSW::body_get_applied_force(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, Vector2());
	return body->get_applied_force();
};

void PhysicsServer2DSW::body_set_applied_torque(RID p_body, real_t p_torque) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_applied_torque(p_torque);
	body->wakeup();
};

real_t PhysicsServer2DSW::body_get_applied_torque(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_applied_torque();
};

void PhysicsServer2DSW::body_apply_central_impulse(RID p_body, const Vector2 &p_impulse) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->apply_central_impulse(p_impulse);
	body->wakeup();
}

void PhysicsServer2DSW::body_apply_torque_impulse(RID p_body, real_t p_torque) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	_update_shapes();

	body->apply_torque_impulse(p_torque);
	body->wakeup();
}

void PhysicsServer2DSW::body_apply_impulse(RID p_body, const Vector2 &p_impulse, const Vector2 &p_position) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	_update_shapes();

	body->apply_impulse(p_impulse, p_position);
	body->wakeup();
};

void PhysicsServer2DSW::body_add_central_force(RID p_body, const Vector2 &p_force) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->add_central_force(p_force);
	body->wakeup();
};

void PhysicsServer2DSW::body_add_force(RID p_body, const Vector2 &p_force, const Vector2 &p_position) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->add_force(p_force, p_position);
	body->wakeup();
};

void PhysicsServer2DSW::body_add_torque(RID p_body, real_t p_torque) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->add_torque(p_torque);
	body->wakeup();
};

void PhysicsServer2DSW::body_set_axis_velocity(RID p_body, const Vector2 &p_axis_velocity) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	_update_shapes();

	Vector2 v = body->get_linear_velocity();
	Vector2 axis = p_axis_velocity.normalized();
	v -= axis * axis.dot(v);
	v += p_axis_velocity;
	body->set_linear_velocity(v);
	body->wakeup();
};

void PhysicsServer2DSW::body_add_collision_exception(RID p_body, RID p_body_b) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->add_exception(p_body_b);
	body->wakeup();
};

void PhysicsServer2DSW::body_remove_collision_exception(RID p_body, RID p_body_b) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->remove_exception(p_body_b);
	body->wakeup();
};

void PhysicsServer2DSW::body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	for (int i = 0; i < body->get_exceptions().size(); i++) {
		p_exceptions->push_back(body->get_exceptions()[i]);
	}
};

void PhysicsServer2DSW::body_set_contacts_reported_depth_threshold(RID p_body, real_t p_threshold) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
};

real_t PhysicsServer2DSW::body_get_contacts_reported_depth_threshold(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, 0);
	return 0;
};

void PhysicsServer2DSW::body_set_omit_force_integration(RID p_body, bool p_omit) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);

	body->set_omit_force_integration(p_omit);
};

bool PhysicsServer2DSW::body_is_omitting_force_integration(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, false);
	return body->get_omit_force_integration();
};

void PhysicsServer2DSW::body_set_max_contacts_reported(RID p_body, int p_contacts) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	body->set_max_contacts_reported(p_contacts);
}

int PhysicsServer2DSW::body_get_max_contacts_reported(RID p_body) const {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, -1);
	return body->get_max_contacts_reported();
}

void PhysicsServer2DSW::body_set_force_integration_callback(RID p_body, Object *p_receiver, const StringName &p_method, const Variant &p_udata) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	body->set_force_integration_callback(p_receiver ? p_receiver->get_instance_id() : ObjectID(), p_method, p_udata);
}

bool PhysicsServer2DSW::body_collide_shape(RID p_body, int p_body_shape, RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, Vector2 *r_results, int p_result_max, int &r_result_count) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, false);
	ERR_FAIL_INDEX_V(p_body_shape, body->get_shape_count(), false);

	return shape_collide(body->get_shape(p_body_shape)->get_self(), body->get_transform() * body->get_shape_transform(p_body_shape), Vector2(), p_shape, p_shape_xform, p_motion, r_results, p_result_max, r_result_count);
}

void PhysicsServer2DSW::body_set_pickable(RID p_body, bool p_pickable) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND(!body);
	body->set_pickable(p_pickable);
}

bool PhysicsServer2DSW::body_test_motion(RID p_body, const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia, real_t p_margin, MotionResult *r_result, bool p_exclude_raycast_shapes) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, false);
	ERR_FAIL_COND_V(!body->get_space(), false);
	ERR_FAIL_COND_V(body->get_space()->is_locked(), false);

	_update_shapes();

	return body->get_space()->test_body_motion(body, p_from, p_motion, p_infinite_inertia, p_margin, r_result, p_exclude_raycast_shapes);
}

int PhysicsServer2DSW::body_test_ray_separation(RID p_body, const Transform2D &p_transform, bool p_infinite_inertia, Vector2 &r_recover_motion, SeparationResult *r_results, int p_result_max, float p_margin) {
	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, false);
	ERR_FAIL_COND_V(!body->get_space(), false);
	ERR_FAIL_COND_V(body->get_space()->is_locked(), false);

	return body->get_space()->test_body_ray_separation(body, p_transform, p_infinite_inertia, r_recover_motion, r_results, p_result_max, p_margin);
}

PhysicsDirectBodyState2D *PhysicsServer2DSW::body_get_direct_state(RID p_body) {
	ERR_FAIL_COND_V_MSG((using_threads && !doing_sync), nullptr, "Body state is inaccessible right now, wait for iteration or physics process notification.");

	if (!body_owner.owns(p_body)) {
		return nullptr;
	}

	Body2DSW *body = body_owner.getornull(p_body);
	ERR_FAIL_COND_V(!body, nullptr);
	ERR_FAIL_COND_V(!body->get_space(), nullptr);
	ERR_FAIL_COND_V_MSG(body->get_space()->is_locked(), nullptr, "Body state is inaccessible right now, wait for iteration or physics process notification.");

	direct_state->body = body;
	return direct_state;
}

/* JOINT API */

void PhysicsServer2DSW::joint_set_param(RID p_joint, JointParam p_param, real_t p_value) {
	Joint2DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);

	switch (p_param) {
		case JOINT_PARAM_BIAS:
			joint->set_bias(p_value);
			break;
		case JOINT_PARAM_MAX_BIAS:
			joint->set_max_bias(p_value);
			break;
		case JOINT_PARAM_MAX_FORCE:
			joint->set_max_force(p_value);
			break;
	}
}

real_t PhysicsServer2DSW::joint_get_param(RID p_joint, JointParam p_param) const {
	const Joint2DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, -1);

	switch (p_param) {
		case JOINT_PARAM_BIAS:
			return joint->get_bias();
			break;
		case JOINT_PARAM_MAX_BIAS:
			return joint->get_max_bias();
			break;
		case JOINT_PARAM_MAX_FORCE:
			return joint->get_max_force();
			break;
	}

	return 0;
}

void PhysicsServer2DSW::joint_disable_collisions_between_bodies(RID p_joint, const bool p_disable) {
	Joint2DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!joint);

	joint->disable_collisions_between_bodies(p_disable);

	if (2 == joint->get_body_count()) {
		Body2DSW *body_a = *joint->get_body_ptr();
		Body2DSW *body_b = *(joint->get_body_ptr() + 1);

		if (p_disable) {
			body_add_collision_exception(body_a->get_self(), body_b->get_self());
			body_add_collision_exception(body_b->get_self(), body_a->get_self());
		} else {
			body_remove_collision_exception(body_a->get_self(), body_b->get_self());
			body_remove_collision_exception(body_b->get_self(), body_a->get_self());
		}
	}
}

bool PhysicsServer2DSW::joint_is_disabled_collisions_between_bodies(RID p_joint) const {
	const Joint2DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, true);

	return joint->is_disabled_collisions_between_bodies();
}

RID PhysicsServer2DSW::pin_joint_create(const Vector2 &p_pos, RID p_body_a, RID p_body_b) {
	Body2DSW *A = body_owner.getornull(p_body_a);
	ERR_FAIL_COND_V(!A, RID());
	Body2DSW *B = nullptr;
	if (body_owner.owns(p_body_b)) {
		B = body_owner.getornull(p_body_b);
		ERR_FAIL_COND_V(!B, RID());
	}

	Joint2DSW *joint = memnew(PinJoint2DSW(p_pos, A, B));
	RID self = joint_owner.make_rid(joint);
	joint->set_self(self);

	return self;
}

RID PhysicsServer2DSW::groove_joint_create(const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, RID p_body_a, RID p_body_b) {
	Body2DSW *A = body_owner.getornull(p_body_a);
	ERR_FAIL_COND_V(!A, RID());

	Body2DSW *B = body_owner.getornull(p_body_b);
	ERR_FAIL_COND_V(!B, RID());

	Joint2DSW *joint = memnew(GrooveJoint2DSW(p_a_groove1, p_a_groove2, p_b_anchor, A, B));
	RID self = joint_owner.make_rid(joint);
	joint->set_self(self);
	return self;
}

RID PhysicsServer2DSW::damped_spring_joint_create(const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, RID p_body_a, RID p_body_b) {
	Body2DSW *A = body_owner.getornull(p_body_a);
	ERR_FAIL_COND_V(!A, RID());

	Body2DSW *B = body_owner.getornull(p_body_b);
	ERR_FAIL_COND_V(!B, RID());

	Joint2DSW *joint = memnew(DampedSpringJoint2DSW(p_anchor_a, p_anchor_b, A, B));
	RID self = joint_owner.make_rid(joint);
	joint->set_self(self);
	return self;
}

void PhysicsServer2DSW::pin_joint_set_param(RID p_joint, PinJointParam p_param, real_t p_value) {
	Joint2DSW *j = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!j);
	ERR_FAIL_COND(j->get_type() != JOINT_PIN);

	PinJoint2DSW *pin_joint = static_cast<PinJoint2DSW *>(j);
	pin_joint->set_param(p_param, p_value);
}

real_t PhysicsServer2DSW::pin_joint_get_param(RID p_joint, PinJointParam p_param) const {
	Joint2DSW *j = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!j, 0);
	ERR_FAIL_COND_V(j->get_type() != JOINT_PIN, 0);

	PinJoint2DSW *pin_joint = static_cast<PinJoint2DSW *>(j);
	return pin_joint->get_param(p_param);
}

void PhysicsServer2DSW::damped_spring_joint_set_param(RID p_joint, DampedSpringParam p_param, real_t p_value) {
	Joint2DSW *j = joint_owner.getornull(p_joint);
	ERR_FAIL_COND(!j);
	ERR_FAIL_COND(j->get_type() != JOINT_DAMPED_SPRING);

	DampedSpringJoint2DSW *dsj = static_cast<DampedSpringJoint2DSW *>(j);
	dsj->set_param(p_param, p_value);
}

real_t PhysicsServer2DSW::damped_spring_joint_get_param(RID p_joint, DampedSpringParam p_param) const {
	Joint2DSW *j = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!j, 0);
	ERR_FAIL_COND_V(j->get_type() != JOINT_DAMPED_SPRING, 0);

	DampedSpringJoint2DSW *dsj = static_cast<DampedSpringJoint2DSW *>(j);
	return dsj->get_param(p_param);
}

PhysicsServer2D::JointType PhysicsServer2DSW::joint_get_type(RID p_joint) const {
	Joint2DSW *joint = joint_owner.getornull(p_joint);
	ERR_FAIL_COND_V(!joint, JOINT_PIN);

	return joint->get_type();
}

void PhysicsServer2DSW::free(RID p_rid) {
	_update_shapes(); // just in case

	if (shape_owner.owns(p_rid)) {
		Shape2DSW *shape = shape_owner.getornull(p_rid);

		while (shape->get_owners().size()) {
			ShapeOwner2DSW *so = shape->get_owners().front()->key();
			so->remove_shape(shape);
		}

		shape_owner.free(p_rid);
		memdelete(shape);
	} else if (body_owner.owns(p_rid)) {
		Body2DSW *body = body_owner.getornull(p_rid);

		/*
		if (body->get_state_query())
			_clear_query(body->get_state_query());

		if (body->get_direct_state_query())
			_clear_query(body->get_direct_state_query());
		*/

		body_set_space(p_rid, RID());

		while (body->get_shape_count()) {
			body->remove_shape(0);
		}

		body_owner.free(p_rid);
		memdelete(body);

	} else if (area_owner.owns(p_rid)) {
		Area2DSW *area = area_owner.getornull(p_rid);

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
		Space2DSW *space = space_owner.getornull(p_rid);

		while (space->get_objects().size()) {
			CollisionObject2DSW *co = (CollisionObject2DSW *)space->get_objects().front()->get();
			co->set_space(nullptr);
		}

		active_spaces.erase(space);
		free(space->get_default_area()->get_self());
		space_owner.free(p_rid);
		memdelete(space);
	} else if (joint_owner.owns(p_rid)) {
		Joint2DSW *joint = joint_owner.getornull(p_rid);

		joint_owner.free(p_rid);
		memdelete(joint);

	} else {
		ERR_FAIL_MSG("Invalid ID.");
	}
};

void PhysicsServer2DSW::set_active(bool p_active) {
	active = p_active;
};

void PhysicsServer2DSW::init() {
	doing_sync = false;
	last_step = 0.001;
	iterations = 8; // 8?
	stepper = memnew(Step2DSW);
	direct_state = memnew(PhysicsDirectBodyState2DSW);
};

void PhysicsServer2DSW::step(real_t p_step) {
	if (!active) {
		return;
	}

	_update_shapes();

	doing_sync = false;

	last_step = p_step;
	PhysicsDirectBodyState2DSW::singleton->step = p_step;
	island_count = 0;
	active_objects = 0;
	collision_pairs = 0;
	for (Set<const Space2DSW *>::Element *E = active_spaces.front(); E; E = E->next()) {
		stepper->step((Space2DSW *)E->get(), p_step, iterations);
		island_count += E->get()->get_island_count();
		active_objects += E->get()->get_active_objects();
		collision_pairs += E->get()->get_collision_pairs();
	}
};

void PhysicsServer2DSW::sync() {
	doing_sync = true;
};

void PhysicsServer2DSW::flush_queries() {
	if (!active) {
		return;
	}

	flushing_queries = true;

	uint64_t time_beg = OS::get_singleton()->get_ticks_usec();

	for (Set<const Space2DSW *>::Element *E = active_spaces.front(); E; E = E->next()) {
		Space2DSW *space = (Space2DSW *)E->get();
		space->call_queries();
	}

	flushing_queries = false;

	if (EngineDebugger::is_profiling("servers")) {
		uint64_t total_time[Space2DSW::ELAPSED_TIME_MAX];
		static const char *time_name[Space2DSW::ELAPSED_TIME_MAX] = {
			"integrate_forces",
			"generate_islands",
			"setup_constraints",
			"solve_constraints",
			"integrate_velocities"
		};

		for (int i = 0; i < Space2DSW::ELAPSED_TIME_MAX; i++) {
			total_time[i] = 0;
		}

		for (Set<const Space2DSW *>::Element *E = active_spaces.front(); E; E = E->next()) {
			for (int i = 0; i < Space2DSW::ELAPSED_TIME_MAX; i++) {
				total_time[i] += E->get()->get_elapsed_time(Space2DSW::ElapsedTime(i));
			}
		}

		Array values;
		values.resize(Space2DSW::ELAPSED_TIME_MAX * 2);
		for (int i = 0; i < Space2DSW::ELAPSED_TIME_MAX; i++) {
			values[i * 2 + 0] = time_name[i];
			values[i * 2 + 1] = USEC_TO_SEC(total_time[i]);
		}
		values.push_back("flush_queries");
		values.push_back(USEC_TO_SEC(OS::get_singleton()->get_ticks_usec() - time_beg));

		values.push_front("physics_2d");
		EngineDebugger::profiler_add_frame_data("servers", values);
	}
}

void PhysicsServer2DSW::end_sync() {
	doing_sync = false;
}

void PhysicsServer2DSW::finish() {
	memdelete(stepper);
	memdelete(direct_state);
};

void PhysicsServer2DSW::_update_shapes() {
	while (pending_shape_update_list.first()) {
		pending_shape_update_list.first()->self()->_shape_changed();
		pending_shape_update_list.remove(pending_shape_update_list.first());
	}
}

int PhysicsServer2DSW::get_process_info(ProcessInfo p_info) {
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

PhysicsServer2DSW *PhysicsServer2DSW::singletonsw = nullptr;

PhysicsServer2DSW::PhysicsServer2DSW() {
	singletonsw = this;
	BroadPhase2DSW::create_func = BroadPhase2DHashGrid::_create;
	//BroadPhase2DSW::create_func=BroadPhase2DBasic::_create;

	active = true;
	island_count = 0;
	active_objects = 0;
	collision_pairs = 0;
#ifdef NO_THREADS
	using_threads = false;
#else
	using_threads = int(ProjectSettings::get_singleton()->get("physics/2d/thread_model")) == 2;
#endif
	flushing_queries = false;
};
