/*************************************************************************/
/*  area_pair_2d_sw.cpp                                                  */
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

#include "area_pair_2d_sw.h"

#include "collision_solver_2d_sw.h"

bool AreaPair2DSW::setup(real_t p_step) {
	bool overlap = false;
	if (body->mask_has_layer(area) && CollisionSolver2DSW::solve(body->get_shape(body_shape), body->get_transform() * body->get_shape_transform(body_shape), Vector2(), area->get_shape(area_shape), area->get_transform() * area->get_shape_transform(area_shape), Vector2(), nullptr, this)) {
		overlap = true;
	}

	process_collision = false;
	if (overlap != colliding) {
		colliding = overlap;
		if (area->get_space_override_mode() != PhysicsServer2D::AREA_SPACE_OVERRIDE_DISABLED) {
			process_collision = true;
		} else if (area->has_monitor_callback()) {
			process_collision = true;
		}
	}

	return process_collision;
}

bool AreaPair2DSW::pre_solve(real_t p_step) {
	if (!process_collision) {
		return false;
	}

	if (colliding) {
		if (area->get_space_override_mode() != PhysicsServer2D::AREA_SPACE_OVERRIDE_DISABLED) {
			body->add_area(area);
		}

		if (area->has_monitor_callback()) {
			area->add_body_to_query(body, body_shape, area_shape);
		}
	} else {
		if (area->get_space_override_mode() != PhysicsServer2D::AREA_SPACE_OVERRIDE_DISABLED) {
			body->remove_area(area);
		}

		if (area->has_monitor_callback()) {
			area->remove_body_from_query(body, body_shape, area_shape);
		}
	}

	return false; // Never do any post solving.
}

void AreaPair2DSW::solve(real_t p_step) {
	// Nothing to do.
}

AreaPair2DSW::AreaPair2DSW(Body2DSW *p_body, int p_body_shape, Area2DSW *p_area, int p_area_shape) {
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

AreaPair2DSW::~AreaPair2DSW() {
	if (colliding) {
		if (area->get_space_override_mode() != PhysicsServer2D::AREA_SPACE_OVERRIDE_DISABLED) {
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

bool Area2Pair2DSW::setup(real_t p_step) {
	bool overlap = false;
	if (CollisionSolver2DSW::solve(area_a->get_shape(shape_a), area_a->get_transform() * area_a->get_shape_transform(shape_a), Vector2(), area_b->get_shape(shape_b), area_b->get_transform() * area_b->get_shape_transform(shape_b), Vector2(), nullptr, this)) {
		overlap = true;
	}

	bool b_collides_with_a = overlap && area_b->mask_has_layer(area_a);
	bool a_collides_with_b = overlap && area_a->mask_has_layer(area_b);
	process_collision_with_a = false;
	process_collision_with_b = false;

	if (b_collides_with_a != b_colliding_with_a) {
		b_colliding_with_a = b_collides_with_a;
		if (area_a->has_area_monitor_callback() && area_b->is_monitorable()) {
			process_collision_with_a = true;
		}
	}

	if (a_collides_with_b != a_colliding_with_b) {
		a_colliding_with_b = a_collides_with_b;
		if (area_b->has_area_monitor_callback() && area_a->is_monitorable()) {
			process_collision_with_b = true;
		}
	}

	return process_collision_with_a || process_collision_with_b;
}

bool Area2Pair2DSW::pre_solve(real_t p_step) {
	if (process_collision_with_a) {
		if (b_colliding_with_a) {
			area_a->add_area_to_query(area_b, shape_b, shape_a);
		} else { // b no longer colliding with a
			area_a->remove_area_from_query(area_b, shape_b, shape_a);
		}
	}

	if (process_collision_with_b) {
		if (a_colliding_with_b) {
			area_b->add_area_to_query(area_a, shape_a, shape_b);
		} else { // a no longer colliding with b
			area_b->remove_area_from_query(area_a, shape_a, shape_b);
		}
	}

	return false; // Never do any post solving.
}

void Area2Pair2DSW::solve(real_t p_step) {
	// Nothing to do.
}

Area2Pair2DSW::Area2Pair2DSW(Area2DSW *p_area_a, int p_shape_a, Area2DSW *p_area_b, int p_shape_b) {
	area_a = p_area_a;
	area_b = p_area_b;
	shape_a = p_shape_a;
	shape_b = p_shape_b;
	area_a->add_constraint(this);
	area_b->add_constraint(this);
}

Area2Pair2DSW::~Area2Pair2DSW() {
	if (a_colliding_with_b && area_b->has_area_monitor_callback()) {
		area_b->remove_area_from_query(area_a, shape_a, shape_b);
	}

	if (b_colliding_with_a && area_a->has_area_monitor_callback()) {
		area_a->remove_area_from_query(area_b, shape_b, shape_a);
	}

	area_a->remove_constraint(this);
	area_b->remove_constraint(this);
}
