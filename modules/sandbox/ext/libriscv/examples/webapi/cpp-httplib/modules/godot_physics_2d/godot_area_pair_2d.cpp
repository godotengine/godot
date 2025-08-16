/**************************************************************************/
/*  godot_area_pair_2d.cpp                                                */
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

#include "godot_area_pair_2d.h"
#include "godot_collision_solver_2d.h"

bool GodotAreaPair2D::setup(real_t p_step) {
	bool result = false;
	if (area->collides_with(body) && GodotCollisionSolver2D::solve(body->get_shape(body_shape), body->get_transform() * body->get_shape_transform(body_shape), Vector2(), area->get_shape(area_shape), area->get_transform() * area->get_shape_transform(area_shape), Vector2(), nullptr, this)) {
		result = true;
	}

	process_collision = false;
	has_space_override = false;
	if (result != colliding) {
		if ((int)area->get_param(PhysicsServer2D::AREA_PARAM_GRAVITY_OVERRIDE_MODE) != PhysicsServer2D::AREA_SPACE_OVERRIDE_DISABLED) {
			has_space_override = true;
		} else if ((int)area->get_param(PhysicsServer2D::AREA_PARAM_LINEAR_DAMP_OVERRIDE_MODE) != PhysicsServer2D::AREA_SPACE_OVERRIDE_DISABLED) {
			has_space_override = true;
		} else if ((int)area->get_param(PhysicsServer2D::AREA_PARAM_ANGULAR_DAMP_OVERRIDE_MODE) != PhysicsServer2D::AREA_SPACE_OVERRIDE_DISABLED) {
			has_space_override = true;
		}
		process_collision = has_space_override;

		if (area->has_monitor_callback()) {
			process_collision = true;
		}

		colliding = result;
	}

	return process_collision;
}

bool GodotAreaPair2D::pre_solve(real_t p_step) {
	if (!process_collision) {
		return false;
	}

	if (colliding) {
		if (has_space_override) {
			body_has_attached_area = true;
			body->add_area(area);
		}

		if (area->has_monitor_callback()) {
			area->add_body_to_query(body, body_shape, area_shape);
		}
	} else {
		if (has_space_override) {
			body_has_attached_area = false;
			body->remove_area(area);
		}

		if (area->has_monitor_callback()) {
			area->remove_body_from_query(body, body_shape, area_shape);
		}
	}

	return false; // Never do any post solving.
}

void GodotAreaPair2D::solve(real_t p_step) {
	// Nothing to do.
}

GodotAreaPair2D::GodotAreaPair2D(GodotBody2D *p_body, int p_body_shape, GodotArea2D *p_area, int p_area_shape) {
	body = p_body;
	area = p_area;
	body_shape = p_body_shape;
	area_shape = p_area_shape;
	body->add_constraint(this, 0);
	area->add_constraint(this);
	if (p_body->get_mode() == PhysicsServer2D::BODY_MODE_KINEMATIC) { //need to be active to process pair
		p_body->set_active(true);
	}
}

GodotAreaPair2D::~GodotAreaPair2D() {
	if (colliding) {
		if (body_has_attached_area) {
			body_has_attached_area = false;
			body->remove_area(area);
		}
		if (area->has_monitor_callback()) {
			area->remove_body_from_query(body, body_shape, area_shape);
		}
	}
	body->remove_constraint(this, 0);
	area->remove_constraint(this);
}

//////////////////////////////////

bool GodotArea2Pair2D::setup(real_t p_step) {
	bool result_a = area_a->collides_with(area_b);
	bool result_b = area_b->collides_with(area_a);
	if ((result_a || result_b) && !GodotCollisionSolver2D::solve(area_a->get_shape(shape_a), area_a->get_transform() * area_a->get_shape_transform(shape_a), Vector2(), area_b->get_shape(shape_b), area_b->get_transform() * area_b->get_shape_transform(shape_b), Vector2(), nullptr, this)) {
		result_a = false;
		result_b = false;
	}

	bool process_collision = false;

	process_collision_a = false;
	if (result_a != colliding_a) {
		if (area_a->has_area_monitor_callback() && area_b_monitorable) {
			process_collision_a = true;
			process_collision = true;
		}
		colliding_a = result_a;
	}

	process_collision_b = false;
	if (result_b != colliding_b) {
		if (area_b->has_area_monitor_callback() && area_a_monitorable) {
			process_collision_b = true;
			process_collision = true;
		}
		colliding_b = result_b;
	}

	return process_collision;
}

bool GodotArea2Pair2D::pre_solve(real_t p_step) {
	if (process_collision_a) {
		if (colliding_a) {
			area_a->add_area_to_query(area_b, shape_b, shape_a);
		} else {
			area_a->remove_area_from_query(area_b, shape_b, shape_a);
		}
	}

	if (process_collision_b) {
		if (colliding_b) {
			area_b->add_area_to_query(area_a, shape_a, shape_b);
		} else {
			area_b->remove_area_from_query(area_a, shape_a, shape_b);
		}
	}

	return false; // Never do any post solving.
}

void GodotArea2Pair2D::solve(real_t p_step) {
	// Nothing to do.
}

GodotArea2Pair2D::GodotArea2Pair2D(GodotArea2D *p_area_a, int p_shape_a, GodotArea2D *p_area_b, int p_shape_b) {
	area_a = p_area_a;
	area_b = p_area_b;
	shape_a = p_shape_a;
	shape_b = p_shape_b;
	area_a_monitorable = area_a->is_monitorable();
	area_b_monitorable = area_b->is_monitorable();
	area_a->add_constraint(this);
	area_b->add_constraint(this);
}

GodotArea2Pair2D::~GodotArea2Pair2D() {
	if (colliding_a) {
		if (area_a->has_area_monitor_callback() && area_b_monitorable) {
			area_a->remove_area_from_query(area_b, shape_b, shape_a);
		}
	}

	if (colliding_b) {
		if (area_b->has_area_monitor_callback() && area_a_monitorable) {
			area_b->remove_area_from_query(area_a, shape_a, shape_b);
		}
	}

	area_a->remove_constraint(this);
	area_b->remove_constraint(this);
}
