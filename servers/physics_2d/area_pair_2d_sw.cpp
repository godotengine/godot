/**************************************************************************/
/*  area_pair_2d_sw.cpp                                                   */
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

#include "area_pair_2d_sw.h"
#include "collision_solver_2d_sw.h"

bool AreaPair2DSW::setup(real_t p_step) {
	bool result = false;
	if (area->test_collision_mask(body) && CollisionSolver2DSW::solve(body->get_shape(body_shape), body->get_transform() * body->get_shape_transform(body_shape), Vector2(), area->get_shape(area_shape), area->get_transform() * area->get_shape_transform(area_shape), Vector2(), nullptr, this)) {
		result = true;
	}

	if (result != colliding) {
		if (result) {
			if (area->get_space_override_mode() != Physics2DServer::AREA_SPACE_OVERRIDE_DISABLED) {
				body->add_area(area);
			}
			if (area->has_monitor_callback()) {
				area->add_body_to_query(body, body_shape, area_shape);
			}

		} else {
			if (area->get_space_override_mode() != Physics2DServer::AREA_SPACE_OVERRIDE_DISABLED) {
				body->remove_area(area);
			}
			if (area->has_monitor_callback()) {
				area->remove_body_from_query(body, body_shape, area_shape);
			}
		}

		colliding = result;
	}

	return false; //never do any post solving
}

void AreaPair2DSW::solve(real_t p_step) {
}

AreaPair2DSW::AreaPair2DSW(Body2DSW *p_body, int p_body_shape, Area2DSW *p_area, int p_area_shape) {
	body = p_body;
	area = p_area;
	body_shape = p_body_shape;
	area_shape = p_area_shape;
	colliding = false;
	body->add_constraint(this, 0);
	area->add_constraint(this);
	if (p_body->get_mode() == Physics2DServer::BODY_MODE_KINEMATIC) { //need to be active to process pair
		p_body->set_active(true);
	}
}

AreaPair2DSW::~AreaPair2DSW() {
	if (colliding) {
		if (area->get_space_override_mode() != Physics2DServer::AREA_SPACE_OVERRIDE_DISABLED) {
			body->remove_area(area);
		}
		if (area->has_monitor_callback()) {
			area->remove_body_from_query(body, body_shape, area_shape);
		}
	}
	body->remove_constraint(this);
	area->remove_constraint(this);
}

//////////////////////////////////

bool Area2Pair2DSW::setup(real_t p_step) {
	bool result = false;
	if (area_a->test_collision_mask(area_b) && CollisionSolver2DSW::solve(area_a->get_shape(shape_a), area_a->get_transform() * area_a->get_shape_transform(shape_a), Vector2(), area_b->get_shape(shape_b), area_b->get_transform() * area_b->get_shape_transform(shape_b), Vector2(), nullptr, this)) {
		result = true;
	}

	if (result != colliding) {
		if (result) {
			if (area_b->has_area_monitor_callback() && area_a_monitorable) {
				area_b->add_area_to_query(area_a, shape_a, shape_b);
			}

			if (area_a->has_area_monitor_callback() && area_b_monitorable) {
				area_a->add_area_to_query(area_b, shape_b, shape_a);
			}

		} else {
			if (area_b->has_area_monitor_callback() && area_a_monitorable) {
				area_b->remove_area_from_query(area_a, shape_a, shape_b);
			}

			if (area_a->has_area_monitor_callback() && area_b_monitorable) {
				area_a->remove_area_from_query(area_b, shape_b, shape_a);
			}
		}

		colliding = result;
	}

	return false; //never do any post solving
}

void Area2Pair2DSW::solve(real_t p_step) {
}

Area2Pair2DSW::Area2Pair2DSW(Area2DSW *p_area_a, int p_shape_a, Area2DSW *p_area_b, int p_shape_b) {
	area_a = p_area_a;
	area_b = p_area_b;
	shape_a = p_shape_a;
	shape_b = p_shape_b;
	colliding = false;
	area_a_monitorable = area_a->is_monitorable();
	area_b_monitorable = area_b->is_monitorable();
	area_a->add_constraint(this);
	area_b->add_constraint(this);
}

Area2Pair2DSW::~Area2Pair2DSW() {
	if (colliding) {
		if (area_b->has_area_monitor_callback() && area_a_monitorable) {
			area_b->remove_area_from_query(area_a, shape_a, shape_b);
		}

		if (area_a->has_area_monitor_callback() && area_b_monitorable) {
			area_a->remove_area_from_query(area_b, shape_b, shape_a);
		}
	}

	area_a->remove_constraint(this);
	area_b->remove_constraint(this);
}
