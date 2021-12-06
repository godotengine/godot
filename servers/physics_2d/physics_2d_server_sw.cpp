/*************************************************************************/
/*  physics_2d_server_sw.cpp                                             */
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

#include "physics_2d_server_sw.h"
#include "broad_phase_2d_basic.h"
#include "broad_phase_2d_bvh.h"
#include "broad_phase_2d_hash_grid.h"
#include "collision_solver_2d_sw.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "core/script_language.h"

#define FLUSH_QUERY_CHECK(m_object) \
	ERR_FAIL_COND_MSG(m_object->get_space() && flushing_queries, "Can't change this state while flushing queries. Use call_deferred() or set_deferred() to change monitoring state instead.");

RID Physics2DServerSW::_shape_create(ShapeType p_shape) {
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

RID Physics2DServerSW::line_shape_create() {
	return _shape_create(SHAPE_LINE);
}

RID Physics2DServerSW::ray_shape_create() {
	return _shape_create(SHAPE_RAY);
}
RID Physics2DServerSW::segment_shape_create() {
	return _shape_create(SHAPE_SEGMENT);
}
RID Physics2DServerSW::circle_shape_create() {
	return _shape_create(SHAPE_CIRCLE);
}
RID Physics2DServerSW::rectangle_shape_create() {
	return _shape_create(SHAPE_RECTANGLE);
}
RID Physics2DServerSW::capsule_shape_create() {
	return _shape_create(SHAPE_CAPSULE);
}

RID Physics2DServerSW::convex_polygon_shape_create() {
	return _shape_create(SHAPE_CONVEX_POLYGON);
}
RID Physics2DServerSW::concave_polygon_shape_create() {
	return _shape_create(SHAPE_CONCAVE_POLYGON);
}

void Physics2DServerSW::shape_set_data(RID p_shape, const Variant &p_data) {
	Shape2DSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);
	shape->set_data(p_data);
};

void Physics2DServerSW::shape_set_custom_solver_bias(RID p_shape, real_t p_bias) {
	Shape2DSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);
	shape->set_custom_bias(p_bias);
}

Physics2DServer::ShapeType Physics2DServerSW::shape_get_type(RID p_shape) const {
	const Shape2DSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, SHAPE_CUSTOM);
	return shape->get_type();
};

Variant Physics2DServerSW::shape_get_data(RID p_shape) const {
	const Shape2DSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, Variant());
	ERR_FAIL_COND_V(!shape->is_configured(), Variant());
	return shape->get_data();
};

real_t Physics2DServerSW::shape_get_custom_solver_bias(RID p_shape) const {
	const Shape2DSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND_V(!shape, 0);
	return shape->get_custom_bias();
}

void Physics2DServerSW::_shape_col_cbk(const Vector2 &p_point_A, const Vector2 &p_point_B, void *p_userdata) {
	CollCbkData *cbk = (CollCbkData *)p_userdata;

	if (cbk->max == 0) {
		return;
	}

	Vector2 rel_dir = (p_point_A - p_point_B);
	real_t rel_length2 = rel_dir.length_squared();
	if (cbk->valid_dir != Vector2()) {
		if (cbk->valid_depth < 10e20) {
			if (rel_length2 > cbk->valid_depth * cbk->valid_depth ||
					(rel_length2 > CMP_EPSILON && cbk->valid_dir.dot(rel_dir.normalized()) < CMP_EPSILON)) {
				cbk->invalid_by_dir++;
				return;
			}
		} else {
			if (rel_length2 > 0 && cbk->valid_dir.dot(rel_dir.normalized()) < CMP_EPSILON) {
				return;
			}
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

		if (rel_length2 < min_depth) {
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

bool Physics2DServerSW::shape_collide(RID p_shape_A, const Transform2D &p_xform_A, const Vector2 &p_motion_A, RID p_shape_B, const Transform2D &p_xform_B, const Vector2 &p_motion_B, Vector2 *r_results, int p_result_max, int &r_result_count) {
	Shape2DSW *shape_A = shape_owner.get(p_shape_A);
	ERR_FAIL_COND_V(!shape_A, false);
	Shape2DSW *shape_B = shape_owner.get(p_shape_B);
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

RID Physics2DServerSW::space_create() {
	Space2DSW *space = memnew(Space2DSW);
	RID id = space_owner.make_rid(space);
	space->set_self(id);
	RID area_id = RID_PRIME(area_create());
	Area2DSW *area = area_owner.get(area_id);
	ERR_FAIL_COND_V(!area, RID());
	space->set_default_area(area);
	area->set_space(space);
	area->set_priority(-1);

	return id;
};

void Physics2DServerSW::space_set_active(RID p_space, bool p_active) {
	Space2DSW *space = space_owner.get(p_space);
	ERR_FAIL_COND(!space);
	if (p_active) {
		active_spaces.insert(space);
	} else {
		active_spaces.erase(space);
	}
}

bool Physics2DServerSW::space_is_active(RID p_space) const {
	const Space2DSW *space = space_owner.get(p_space);
	ERR_FAIL_COND_V(!space, false);

	return active_spaces.has(space);
}

void Physics2DServerSW::space_set_param(RID p_space, SpaceParameter p_param, real_t p_value) {
	Space2DSW *space = space_owner.get(p_space);
	ERR_FAIL_COND(!space);

	space->set_param(p_param, p_value);
}

real_t Physics2DServerSW::space_get_param(RID p_space, SpaceParameter p_param) const {
	const Space2DSW *space = space_owner.get(p_space);
	ERR_FAIL_COND_V(!space, 0);
	return space->get_param(p_param);
}

void Physics2DServerSW::space_set_debug_contacts(RID p_space, int p_max_contacts) {
	Space2DSW *space = space_owner.get(p_space);
	ERR_FAIL_COND(!space);
	space->set_debug_contacts(p_max_contacts);
}

Vector<Vector2> Physics2DServerSW::space_get_contacts(RID p_space) const {
	Space2DSW *space = space_owner.get(p_space);
	ERR_FAIL_COND_V(!space, Vector<Vector2>());
	return space->get_debug_contacts();
}

int Physics2DServerSW::space_get_contact_count(RID p_space) const {
	Space2DSW *space = space_owner.get(p_space);
	ERR_FAIL_COND_V(!space, 0);
	return space->get_debug_contact_count();
}

Physics2DDirectSpaceState *Physics2DServerSW::space_get_direct_state(RID p_space) {
	Space2DSW *space = space_owner.get(p_space);
	ERR_FAIL_COND_V(!space, nullptr);
	ERR_FAIL_COND_V_MSG((using_threads && !doing_sync) || space->is_locked(), nullptr, "Space state is inaccessible right now, wait for iteration or physics process notification.");

	return space->get_direct_state();
}

RID Physics2DServerSW::area_create() {
	Area2DSW *area = memnew(Area2DSW);
	RID rid = area_owner.make_rid(area);
	area->set_self(rid);
	return rid;
};

void Physics2DServerSW::area_set_space(RID p_area, RID p_space) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	Space2DSW *space = nullptr;
	if (p_space.is_valid()) {
		space = space_owner.get(p_space);
		ERR_FAIL_COND(!space);
	}

	if (area->get_space() == space) {
		return; //pointless
	}

	area->clear_constraints();
	area->set_space(space);
};

RID Physics2DServerSW::area_get_space(RID p_area) const {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area, RID());

	Space2DSW *space = area->get_space();
	if (!space) {
		return RID();
	}
	return space->get_self();
};

void Physics2DServerSW::area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	area->set_space_override_mode(p_mode);
}

Physics2DServer::AreaSpaceOverrideMode Physics2DServerSW::area_get_space_override_mode(RID p_area) const {
	const Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area, AREA_SPACE_OVERRIDE_DISABLED);

	return area->get_space_override_mode();
}

void Physics2DServerSW::area_add_shape(RID p_area, RID p_shape, const Transform2D &p_transform, bool p_disabled) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	Shape2DSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);

	area->add_shape(shape, p_transform, p_disabled);
}

void Physics2DServerSW::area_set_shape(RID p_area, int p_shape_idx, RID p_shape) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	Shape2DSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);
	ERR_FAIL_COND(!shape->is_configured());

	area->set_shape(p_shape_idx, shape);
}
void Physics2DServerSW::area_set_shape_transform(RID p_area, int p_shape_idx, const Transform2D &p_transform) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	area->set_shape_transform(p_shape_idx, p_transform);
}

void Physics2DServerSW::area_set_shape_disabled(RID p_area, int p_shape, bool p_disabled) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);
	ERR_FAIL_INDEX(p_shape, area->get_shape_count());
	FLUSH_QUERY_CHECK(area);

	area->set_shape_disabled(p_shape, p_disabled);
}

int Physics2DServerSW::area_get_shape_count(RID p_area) const {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area, -1);

	return area->get_shape_count();
}
RID Physics2DServerSW::area_get_shape(RID p_area, int p_shape_idx) const {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area, RID());

	Shape2DSW *shape = area->get_shape(p_shape_idx);
	ERR_FAIL_COND_V(!shape, RID());

	return shape->get_self();
}
Transform2D Physics2DServerSW::area_get_shape_transform(RID p_area, int p_shape_idx) const {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area, Transform2D());

	return area->get_shape_transform(p_shape_idx);
}

void Physics2DServerSW::area_remove_shape(RID p_area, int p_shape_idx) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	area->remove_shape(p_shape_idx);
}

void Physics2DServerSW::area_clear_shapes(RID p_area) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	while (area->get_shape_count()) {
		area->remove_shape(0);
	}
}

void Physics2DServerSW::area_attach_object_instance_id(RID p_area, ObjectID p_id) {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.get(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);
	area->set_instance_id(p_id);
}
ObjectID Physics2DServerSW::area_get_object_instance_id(RID p_area) const {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.get(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area, 0);
	return area->get_instance_id();
}

void Physics2DServerSW::area_attach_canvas_instance_id(RID p_area, ObjectID p_id) {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.get(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);
	area->set_canvas_instance_id(p_id);
}
ObjectID Physics2DServerSW::area_get_canvas_instance_id(RID p_area) const {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.get(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area, 0);
	return area->get_canvas_instance_id();
}

void Physics2DServerSW::area_set_param(RID p_area, AreaParameter p_param, const Variant &p_value) {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.get(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);
	area->set_param(p_param, p_value);
};

void Physics2DServerSW::area_set_transform(RID p_area, const Transform2D &p_transform) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);
	area->set_transform(p_transform);
};

Variant Physics2DServerSW::area_get_param(RID p_area, AreaParameter p_param) const {
	if (space_owner.owns(p_area)) {
		Space2DSW *space = space_owner.get(p_area);
		p_area = space->get_default_area()->get_self();
	}
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area, Variant());

	return area->get_param(p_param);
};

Transform2D Physics2DServerSW::area_get_transform(RID p_area) const {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND_V(!area, Transform2D());

	return area->get_transform();
};

void Physics2DServerSW::area_set_pickable(RID p_area, bool p_pickable) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);
	area->set_pickable(p_pickable);
}

void Physics2DServerSW::area_set_monitorable(RID p_area, bool p_monitorable) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);
	FLUSH_QUERY_CHECK(area);

	area->set_monitorable(p_monitorable);
}

void Physics2DServerSW::area_set_collision_mask(RID p_area, uint32_t p_mask) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	area->set_collision_mask(p_mask);
}

void Physics2DServerSW::area_set_collision_layer(RID p_area, uint32_t p_layer) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	area->set_collision_layer(p_layer);
}

void Physics2DServerSW::area_set_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	area->set_monitor_callback(p_receiver ? p_receiver->get_instance_id() : 0, p_method);
}

void Physics2DServerSW::area_set_area_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) {
	Area2DSW *area = area_owner.get(p_area);
	ERR_FAIL_COND(!area);

	area->set_area_monitor_callback(p_receiver ? p_receiver->get_instance_id() : 0, p_method);
}

/* BODY API */

RID Physics2DServerSW::body_create() {
	Body2DSW *body = memnew(Body2DSW);
	RID rid = body_owner.make_rid(body);
	body->set_self(rid);
	return rid;
}

void Physics2DServerSW::body_set_space(RID p_body, RID p_space) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	Space2DSW *space = nullptr;
	if (p_space.is_valid()) {
		space = space_owner.get(p_space);
		ERR_FAIL_COND(!space);
	}

	if (body->get_space() == space) {
		return; //pointless
	}

	body->clear_constraint_map();
	body->set_space(space);
};

RID Physics2DServerSW::body_get_space(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, RID());

	Space2DSW *space = body->get_space();
	if (!space) {
		return RID();
	}
	return space->get_self();
};

void Physics2DServerSW::body_set_mode(RID p_body, BodyMode p_mode) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	FLUSH_QUERY_CHECK(body);

	body->set_mode(p_mode);
};

Physics2DServer::BodyMode Physics2DServerSW::body_get_mode(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, BODY_MODE_STATIC);

	return body->get_mode();
};

void Physics2DServerSW::body_add_shape(RID p_body, RID p_shape, const Transform2D &p_transform, bool p_disabled) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	Shape2DSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);

	body->add_shape(shape, p_transform, p_disabled);
}

void Physics2DServerSW::body_set_shape(RID p_body, int p_shape_idx, RID p_shape) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	Shape2DSW *shape = shape_owner.get(p_shape);
	ERR_FAIL_COND(!shape);
	ERR_FAIL_COND(!shape->is_configured());

	body->set_shape(p_shape_idx, shape);
}
void Physics2DServerSW::body_set_shape_transform(RID p_body, int p_shape_idx, const Transform2D &p_transform) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_shape_transform(p_shape_idx, p_transform);
}

void Physics2DServerSW::body_set_shape_metadata(RID p_body, int p_shape_idx, const Variant &p_metadata) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	body->set_shape_metadata(p_shape_idx, p_metadata);
}

Variant Physics2DServerSW::body_get_shape_metadata(RID p_body, int p_shape_idx) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, Variant());
	return body->get_shape_metadata(p_shape_idx);
}

int Physics2DServerSW::body_get_shape_count(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, -1);

	return body->get_shape_count();
}
RID Physics2DServerSW::body_get_shape(RID p_body, int p_shape_idx) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, RID());

	Shape2DSW *shape = body->get_shape(p_shape_idx);
	ERR_FAIL_COND_V(!shape, RID());

	return shape->get_self();
}
Transform2D Physics2DServerSW::body_get_shape_transform(RID p_body, int p_shape_idx) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, Transform2D());

	return body->get_shape_transform(p_shape_idx);
}

void Physics2DServerSW::body_remove_shape(RID p_body, int p_shape_idx) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->remove_shape(p_shape_idx);
}

void Physics2DServerSW::body_clear_shapes(RID p_body) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	while (body->get_shape_count()) {
		body->remove_shape(0);
	}
}

void Physics2DServerSW::body_set_shape_disabled(RID p_body, int p_shape_idx, bool p_disabled) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	ERR_FAIL_INDEX(p_shape_idx, body->get_shape_count());
	FLUSH_QUERY_CHECK(body);

	body->set_shape_disabled(p_shape_idx, p_disabled);
}
void Physics2DServerSW::body_set_shape_as_one_way_collision(RID p_body, int p_shape_idx, bool p_enable, float p_margin) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	ERR_FAIL_INDEX(p_shape_idx, body->get_shape_count());
	FLUSH_QUERY_CHECK(body);

	body->set_shape_as_one_way_collision(p_shape_idx, p_enable, p_margin);
}

void Physics2DServerSW::body_set_continuous_collision_detection_mode(RID p_body, CCDMode p_mode) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	body->set_continuous_collision_detection_mode(p_mode);
}

Physics2DServerSW::CCDMode Physics2DServerSW::body_get_continuous_collision_detection_mode(RID p_body) const {
	const Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, CCD_MODE_DISABLED);

	return body->get_continuous_collision_detection_mode();
}

void Physics2DServerSW::body_attach_object_instance_id(RID p_body, uint32_t p_id) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_instance_id(p_id);
};

uint32_t Physics2DServerSW::body_get_object_instance_id(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_instance_id();
};

void Physics2DServerSW::body_attach_canvas_instance_id(RID p_body, uint32_t p_id) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_canvas_instance_id(p_id);
};

uint32_t Physics2DServerSW::body_get_canvas_instance_id(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_canvas_instance_id();
};

void Physics2DServerSW::body_set_collision_layer(RID p_body, uint32_t p_layer) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	body->set_collision_layer(p_layer);
};

uint32_t Physics2DServerSW::body_get_collision_layer(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_collision_layer();
};

void Physics2DServerSW::body_set_collision_mask(RID p_body, uint32_t p_mask) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	body->set_collision_mask(p_mask);
};

uint32_t Physics2DServerSW::body_get_collision_mask(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_collision_mask();
};

void Physics2DServerSW::body_set_param(RID p_body, BodyParameter p_param, real_t p_value) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_param(p_param, p_value);
};

real_t Physics2DServerSW::body_get_param(RID p_body, BodyParameter p_param) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_param(p_param);
};

void Physics2DServerSW::body_set_state(RID p_body, BodyState p_state, const Variant &p_variant) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_state(p_state, p_variant);
};

Variant Physics2DServerSW::body_get_state(RID p_body, BodyState p_state) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, Variant());

	return body->get_state(p_state);
};

void Physics2DServerSW::body_set_applied_force(RID p_body, const Vector2 &p_force) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_applied_force(p_force);
	body->wakeup();
};

Vector2 Physics2DServerSW::body_get_applied_force(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, Vector2());
	return body->get_applied_force();
};

void Physics2DServerSW::body_set_applied_torque(RID p_body, real_t p_torque) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_applied_torque(p_torque);
	body->wakeup();
};

real_t Physics2DServerSW::body_get_applied_torque(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, 0);

	return body->get_applied_torque();
};

void Physics2DServerSW::body_apply_central_impulse(RID p_body, const Vector2 &p_impulse) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->apply_central_impulse(p_impulse);
	body->wakeup();
}

void Physics2DServerSW::body_apply_torque_impulse(RID p_body, real_t p_torque) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	_update_shapes();

	body->apply_torque_impulse(p_torque);
	body->wakeup();
}

void Physics2DServerSW::body_apply_impulse(RID p_body, const Vector2 &p_pos, const Vector2 &p_impulse) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	_update_shapes();

	body->apply_impulse(p_pos, p_impulse);
	body->wakeup();
};

void Physics2DServerSW::body_add_central_force(RID p_body, const Vector2 &p_force) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->add_central_force(p_force);
	body->wakeup();
};

void Physics2DServerSW::body_add_force(RID p_body, const Vector2 &p_offset, const Vector2 &p_force) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->add_force(p_offset, p_force);
	body->wakeup();
};

void Physics2DServerSW::body_add_torque(RID p_body, real_t p_torque) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->add_torque(p_torque);
	body->wakeup();
};

void Physics2DServerSW::body_set_axis_velocity(RID p_body, const Vector2 &p_axis_velocity) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	_update_shapes();

	Vector2 v = body->get_linear_velocity();
	Vector2 axis = p_axis_velocity.normalized();
	v -= axis * axis.dot(v);
	v += p_axis_velocity;
	body->set_linear_velocity(v);
	body->wakeup();
};

void Physics2DServerSW::body_add_collision_exception(RID p_body, RID p_body_b) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->add_exception(p_body_b);
	body->wakeup();
};

void Physics2DServerSW::body_remove_collision_exception(RID p_body, RID p_body_b) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->remove_exception(p_body_b);
	body->wakeup();
};

void Physics2DServerSW::body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	for (int i = 0; i < body->get_exceptions().size(); i++) {
		p_exceptions->push_back(body->get_exceptions()[i]);
	}
};

void Physics2DServerSW::body_set_contacts_reported_depth_threshold(RID p_body, real_t p_threshold) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
};

real_t Physics2DServerSW::body_get_contacts_reported_depth_threshold(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, 0);
	return 0;
};

void Physics2DServerSW::body_set_omit_force_integration(RID p_body, bool p_omit) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);

	body->set_omit_force_integration(p_omit);
};

bool Physics2DServerSW::body_is_omitting_force_integration(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, false);
	return body->get_omit_force_integration();
};

void Physics2DServerSW::body_set_max_contacts_reported(RID p_body, int p_contacts) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	body->set_max_contacts_reported(p_contacts);
}

int Physics2DServerSW::body_get_max_contacts_reported(RID p_body) const {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, -1);
	return body->get_max_contacts_reported();
}

void Physics2DServerSW::body_set_force_integration_callback(RID p_body, Object *p_receiver, const StringName &p_method, const Variant &p_udata) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	body->set_force_integration_callback(p_receiver ? p_receiver->get_instance_id() : ObjectID(0), p_method, p_udata);
}

bool Physics2DServerSW::body_collide_shape(RID p_body, int p_body_shape, RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, Vector2 *r_results, int p_result_max, int &r_result_count) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, false);
	ERR_FAIL_INDEX_V(p_body_shape, body->get_shape_count(), false);

	return shape_collide(body->get_shape(p_body_shape)->get_self(), body->get_transform() * body->get_shape_transform(p_body_shape), Vector2(), p_shape, p_shape_xform, p_motion, r_results, p_result_max, r_result_count);
}

void Physics2DServerSW::body_set_pickable(RID p_body, bool p_pickable) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND(!body);
	body->set_pickable(p_pickable);
}

bool Physics2DServerSW::body_test_motion(RID p_body, const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia, real_t p_margin, MotionResult *r_result, bool p_exclude_raycast_shapes, const Set<RID> &p_exclude) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, false);
	ERR_FAIL_COND_V(!body->get_space(), false);
	ERR_FAIL_COND_V(body->get_space()->is_locked(), false);

	_update_shapes();

	return body->get_space()->test_body_motion(body, p_from, p_motion, p_infinite_inertia, p_margin, r_result, p_exclude_raycast_shapes, p_exclude);
}

int Physics2DServerSW::body_test_ray_separation(RID p_body, const Transform2D &p_transform, bool p_infinite_inertia, Vector2 &r_recover_motion, SeparationResult *r_results, int p_result_max, float p_margin) {
	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V(!body, false);
	ERR_FAIL_COND_V(!body->get_space(), false);
	ERR_FAIL_COND_V(body->get_space()->is_locked(), false);

	return body->get_space()->test_body_ray_separation(body, p_transform, p_infinite_inertia, r_recover_motion, r_results, p_result_max, p_margin);
}

Physics2DDirectBodyState *Physics2DServerSW::body_get_direct_state(RID p_body) {
	if (!body_owner.owns(p_body)) {
		return nullptr;
	}

	Body2DSW *body = body_owner.get(p_body);
	ERR_FAIL_COND_V_MSG(!body, nullptr, "Body with RID " + itos(p_body.get_id()) + " not owned by this server.");

	if (!body->get_space()) {
		return nullptr;
	}

	ERR_FAIL_COND_V_MSG((using_threads && !doing_sync) || body->get_space()->is_locked(), nullptr, "Body state is inaccessible right now, wait for iteration or physics process notification.");

	return body->get_direct_state();
}

/* JOINT API */

void Physics2DServerSW::joint_set_param(RID p_joint, JointParam p_param, real_t p_value) {
	Joint2DSW *joint = joint_owner.get(p_joint);
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

real_t Physics2DServerSW::joint_get_param(RID p_joint, JointParam p_param) const {
	const Joint2DSW *joint = joint_owner.get(p_joint);
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

void Physics2DServerSW::joint_disable_collisions_between_bodies(RID p_joint, const bool p_disable) {
	Joint2DSW *joint = joint_owner.get(p_joint);
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

bool Physics2DServerSW::joint_is_disabled_collisions_between_bodies(RID p_joint) const {
	const Joint2DSW *joint = joint_owner.get(p_joint);
	ERR_FAIL_COND_V(!joint, true);

	return joint->is_disabled_collisions_between_bodies();
}

RID Physics2DServerSW::pin_joint_create(const Vector2 &p_pos, RID p_body_a, RID p_body_b) {
	Body2DSW *A = body_owner.get(p_body_a);
	ERR_FAIL_COND_V(!A, RID());
	Body2DSW *B = nullptr;
	if (body_owner.owns(p_body_b)) {
		B = body_owner.get(p_body_b);
		ERR_FAIL_COND_V(!B, RID());
	}

	Joint2DSW *joint = memnew(PinJoint2DSW(p_pos, A, B));
	RID self = joint_owner.make_rid(joint);
	joint->set_self(self);

	return self;
}

RID Physics2DServerSW::groove_joint_create(const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, RID p_body_a, RID p_body_b) {
	Body2DSW *A = body_owner.get(p_body_a);
	ERR_FAIL_COND_V(!A, RID());

	Body2DSW *B = body_owner.get(p_body_b);
	ERR_FAIL_COND_V(!B, RID());

	Joint2DSW *joint = memnew(GrooveJoint2DSW(p_a_groove1, p_a_groove2, p_b_anchor, A, B));
	RID self = joint_owner.make_rid(joint);
	joint->set_self(self);
	return self;
}

RID Physics2DServerSW::damped_spring_joint_create(const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, RID p_body_a, RID p_body_b) {
	Body2DSW *A = body_owner.get(p_body_a);
	ERR_FAIL_COND_V(!A, RID());

	Body2DSW *B = body_owner.get(p_body_b);
	ERR_FAIL_COND_V(!B, RID());

	Joint2DSW *joint = memnew(DampedSpringJoint2DSW(p_anchor_a, p_anchor_b, A, B));
	RID self = joint_owner.make_rid(joint);
	joint->set_self(self);
	return self;
}

void Physics2DServerSW::pin_joint_set_param(RID p_joint, PinJointParam p_param, real_t p_value) {
	Joint2DSW *j = joint_owner.get(p_joint);
	ERR_FAIL_COND(!j);
	ERR_FAIL_COND(j->get_type() != JOINT_PIN);

	PinJoint2DSW *pin_joint = static_cast<PinJoint2DSW *>(j);
	pin_joint->set_param(p_param, p_value);
}

real_t Physics2DServerSW::pin_joint_get_param(RID p_joint, PinJointParam p_param) const {
	Joint2DSW *j = joint_owner.get(p_joint);
	ERR_FAIL_COND_V(!j, 0);
	ERR_FAIL_COND_V(j->get_type() != JOINT_PIN, 0);

	PinJoint2DSW *pin_joint = static_cast<PinJoint2DSW *>(j);
	return pin_joint->get_param(p_param);
}

void Physics2DServerSW::damped_string_joint_set_param(RID p_joint, DampedStringParam p_param, real_t p_value) {
	Joint2DSW *j = joint_owner.get(p_joint);
	ERR_FAIL_COND(!j);
	ERR_FAIL_COND(j->get_type() != JOINT_DAMPED_SPRING);

	DampedSpringJoint2DSW *dsj = static_cast<DampedSpringJoint2DSW *>(j);
	dsj->set_param(p_param, p_value);
}

real_t Physics2DServerSW::damped_string_joint_get_param(RID p_joint, DampedStringParam p_param) const {
	Joint2DSW *j = joint_owner.get(p_joint);
	ERR_FAIL_COND_V(!j, 0);
	ERR_FAIL_COND_V(j->get_type() != JOINT_DAMPED_SPRING, 0);

	DampedSpringJoint2DSW *dsj = static_cast<DampedSpringJoint2DSW *>(j);
	return dsj->get_param(p_param);
}

Physics2DServer::JointType Physics2DServerSW::joint_get_type(RID p_joint) const {
	Joint2DSW *joint = joint_owner.get(p_joint);
	ERR_FAIL_COND_V(!joint, JOINT_PIN);

	return joint->get_type();
}

void Physics2DServerSW::free(RID p_rid) {
	_update_shapes(); // just in case

	if (shape_owner.owns(p_rid)) {
		Shape2DSW *shape = shape_owner.get(p_rid);

		while (shape->get_owners().size()) {
			ShapeOwner2DSW *so = shape->get_owners().front()->key();
			so->remove_shape(shape);
		}

		shape_owner.free(p_rid);
		memdelete(shape);
	} else if (body_owner.owns(p_rid)) {
		Body2DSW *body = body_owner.get(p_rid);

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
		Area2DSW *area = area_owner.get(p_rid);

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
		Space2DSW *space = space_owner.get(p_rid);

		while (space->get_objects().size()) {
			CollisionObject2DSW *co = (CollisionObject2DSW *)space->get_objects().front()->get();
			co->set_space(nullptr);
		}

		active_spaces.erase(space);
		free(space->get_default_area()->get_self());
		space_owner.free(p_rid);
		memdelete(space);
	} else if (joint_owner.owns(p_rid)) {
		Joint2DSW *joint = joint_owner.get(p_rid);

		joint_owner.free(p_rid);
		memdelete(joint);

	} else {
		ERR_FAIL_MSG("Invalid ID.");
	}
};

void Physics2DServerSW::set_active(bool p_active) {
	active = p_active;
};

void Physics2DServerSW::set_collision_iterations(int p_iterations) {
	iterations = p_iterations;
};

void Physics2DServerSW::init() {
	doing_sync = false;
	iterations = 8; // 8?
	stepper = memnew(Step2DSW);
};

void Physics2DServerSW::step(real_t p_step) {
	if (!active) {
		return;
	}

	_update_shapes();

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

void Physics2DServerSW::sync() {
	doing_sync = true;
};

void Physics2DServerSW::flush_queries() {
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

	if (ScriptDebugger::get_singleton() && ScriptDebugger::get_singleton()->is_profiling()) {
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

		ScriptDebugger::get_singleton()->add_profiling_frame_data("physics_2d", values);
	}
}

void Physics2DServerSW::end_sync() {
	doing_sync = false;
}

void Physics2DServerSW::finish() {
	memdelete(stepper);
};

void Physics2DServerSW::_update_shapes() {
	while (pending_shape_update_list.first()) {
		pending_shape_update_list.first()->self()->_shape_changed();
		pending_shape_update_list.remove(pending_shape_update_list.first());
	}
}

int Physics2DServerSW::get_process_info(ProcessInfo p_info) {
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

Physics2DServerSW *Physics2DServerSW::singletonsw = nullptr;

Physics2DServerSW::Physics2DServerSW() {
	singletonsw = this;

	GLOBAL_DEF("physics/2d/use_bvh", true);
	GLOBAL_DEF("physics/2d/bp_hash_table_size", 4096);
	GLOBAL_DEF("physics/2d/cell_size", 128);
	GLOBAL_DEF("physics/2d/large_object_surface_threshold_in_cells", 512);
	GLOBAL_DEF("physics/2d/bvh_collision_margin", 1.0);
	ProjectSettings::get_singleton()->set_custom_property_info("physics/2d/bvh_collision_margin", PropertyInfo(Variant::REAL, "physics/2d/bvh_collision_margin", PROPERTY_HINT_RANGE, "0.0,20.0,0.1"));

	bool use_bvh = GLOBAL_GET("physics/2d/use_bvh");

	if (use_bvh) {
		BroadPhase2DSW::create_func = BroadPhase2DBVH::_create;
	} else {
		BroadPhase2DSW::create_func = BroadPhase2DHashGrid::_create;
	}

	active = true;
	island_count = 0;
	active_objects = 0;
	collision_pairs = 0;
#ifdef NO_THREADS
	using_threads = false;
#else
	using_threads = int(GLOBAL_GET("physics/2d/thread_model")) == 2;
#endif
	flushing_queries = false;
};

Physics2DServerSW::~Physics2DServerSW(){

};
