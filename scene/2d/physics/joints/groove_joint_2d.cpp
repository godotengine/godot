/**************************************************************************/
/*  groove_joint_2d.cpp                                                   */
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

#include "groove_joint_2d.h"

#include "scene/2d/physics/physics_body_2d.h"
#include "servers/rendering/rendering_server.h"

#ifdef DEBUG_ENABLED
void GrooveJoint2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			if (!is_inside_tree()) {
				break;
			}

			if (!Engine::get_singleton()->is_editor_hint() && !get_tree()->is_debugging_collisions_hint()) {
				break;
			}

			_prepare_debug_canvas_item();
			RenderingServer *rs = RenderingServer::get_singleton();
			rs->canvas_item_add_line(_get_debug_canvas_item(), Point2(-10, 0), Point2(+10, 0), Color(0.7, 0.6, 0.0, 0.5), 3);
			rs->canvas_item_add_line(_get_debug_canvas_item(), Point2(-10, length), Point2(+10, length), Color(0.7, 0.6, 0.0, 0.5), 3);
			rs->canvas_item_add_line(_get_debug_canvas_item(), Point2(0, 0), Point2(0, length), Color(0.7, 0.6, 0.0, 0.5), 3);
			rs->canvas_item_add_line(_get_debug_canvas_item(), Point2(-10, initial_offset), Point2(+10, initial_offset), Color(0.8, 0.8, 0.9, 0.5), 5);
		} break;
	}
}
#endif // DEBUG_ENABLED

void GrooveJoint2D::_configure_joint(RID p_joint, PhysicsBody2D *body_a, PhysicsBody2D *body_b) {
	Transform2D gt = get_global_transform();
	Vector2 groove_A1 = gt.get_origin();
	Vector2 groove_A2 = gt.xform(Vector2(0, length));
	Vector2 anchor_B = gt.xform(Vector2(0, initial_offset));

	PhysicsServer2D::get_singleton()->joint_make_groove(p_joint, groove_A1, groove_A2, anchor_B, body_a->get_rid(), body_b->get_rid());
}

void GrooveJoint2D::set_length(real_t p_length) {
	length = p_length;
	queue_redraw();
}

real_t GrooveJoint2D::get_length() const {
	return length;
}

void GrooveJoint2D::set_initial_offset(real_t p_initial_offset) {
	initial_offset = p_initial_offset;
	queue_redraw();
}

real_t GrooveJoint2D::get_initial_offset() const {
	return initial_offset;
}

void GrooveJoint2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_length", "length"), &GrooveJoint2D::set_length);
	ClassDB::bind_method(D_METHOD("get_length"), &GrooveJoint2D::get_length);
	ClassDB::bind_method(D_METHOD("set_initial_offset", "offset"), &GrooveJoint2D::set_initial_offset);
	ClassDB::bind_method(D_METHOD("get_initial_offset"), &GrooveJoint2D::get_initial_offset);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length", PROPERTY_HINT_RANGE, "1,65535,1,exp,suffix:px"), "set_length", "get_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "initial_offset", PROPERTY_HINT_RANGE, "1,65535,1,exp,suffix:px"), "set_initial_offset", "get_initial_offset");
}

GrooveJoint2D::GrooveJoint2D() {
}
