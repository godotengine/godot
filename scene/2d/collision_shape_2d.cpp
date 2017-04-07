/*************************************************************************/
/*  collision_shape_2d.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "collision_shape_2d.h"
#include "collision_object_2d.h"
#include "scene/resources/capsule_shape_2d.h"
#include "scene/resources/circle_shape_2d.h"
#include "scene/resources/concave_polygon_shape_2d.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "scene/resources/rectangle_shape_2d.h"
#include "scene/resources/segment_shape_2d.h"
#include "scene/resources/shape_line_2d.h"

void CollisionShape2D::_add_to_collision_object(Object *p_obj) {

	if (unparenting)
		return;

	CollisionObject2D *co = p_obj->cast_to<CollisionObject2D>();
	ERR_FAIL_COND(!co);
	update_shape_index = co->get_shape_count();
	co->add_shape(shape, get_transform());
	if (trigger)
		co->set_shape_as_trigger(co->get_shape_count() - 1, true);
}

void CollisionShape2D::_shape_changed() {

	update();
	_update_parent();
}

void CollisionShape2D::_update_parent() {

	Node *parent = get_parent();
	if (!parent)
		return;
	CollisionObject2D *co = parent->cast_to<CollisionObject2D>();
	if (!co)
		return;
	co->_update_shapes_from_children();
}

void CollisionShape2D::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {
			unparenting = false;
			can_update_body = get_tree()->is_editor_hint();
			if (!get_tree()->is_editor_hint()) {
				//display above all else
				set_z_as_relative(false);
				set_z(VS::CANVAS_ITEM_Z_MAX - 1);
			}

		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {

			if (!is_inside_tree())
				break;
			if (can_update_body) {
				_update_parent();
			} else if (update_shape_index >= 0) {

				CollisionObject2D *co = get_parent()->cast_to<CollisionObject2D>();
				if (co) {
					co->set_shape_transform(update_shape_index, get_transform());
				}
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			can_update_body = false;

		} break;
		/*
		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (!is_inside_scene())
				break;
			_update_parent();

		} break;*/
		case NOTIFICATION_DRAW: {

			if (!get_tree()->is_editor_hint() && !get_tree()->is_debugging_collisions_hint()) {
				break;
			}

			if (!shape.is_valid()) {
				break;
			}

			rect = Rect2();

			Color draw_col = get_tree()->get_debug_collisions_color();
			shape->draw(get_canvas_item(), draw_col);

			rect = shape->get_rect();
			rect = rect.grow(3);

		} break;
		case NOTIFICATION_UNPARENTED: {
			unparenting = true;
			_update_parent();
		} break;
	}
}

void CollisionShape2D::set_shape(const Ref<Shape2D> &p_shape) {

	if (shape.is_valid())
		shape->disconnect("changed", this, "_shape_changed");
	shape = p_shape;
	update();
	if (is_inside_tree() && can_update_body)
		_update_parent();
	if (is_inside_tree() && !can_update_body && update_shape_index >= 0) {
		CollisionObject2D *co = get_parent()->cast_to<CollisionObject2D>();
		if (co) {
			co->set_shape(update_shape_index, p_shape);
		}
	}
	if (shape.is_valid())
		shape->connect("changed", this, "_shape_changed");

	update_configuration_warning();
}

Ref<Shape2D> CollisionShape2D::get_shape() const {

	return shape;
}

Rect2 CollisionShape2D::get_item_rect() const {

	return rect;
}

void CollisionShape2D::set_trigger(bool p_trigger) {

	trigger = p_trigger;
	if (can_update_body) {
		_update_parent();
	} else if (is_inside_tree() && update_shape_index >= 0) {
		CollisionObject2D *co = get_parent()->cast_to<CollisionObject2D>();
		if (co) {
			co->set_shape_as_trigger(update_shape_index, p_trigger);
		}
	}
}

bool CollisionShape2D::is_trigger() const {

	return trigger;
}

void CollisionShape2D::_set_update_shape_index(int p_index) {

	update_shape_index = p_index;
}

int CollisionShape2D::_get_update_shape_index() const {

	return update_shape_index;
}

String CollisionShape2D::get_configuration_warning() const {

	if (!get_parent()->cast_to<CollisionObject2D>()) {
		return TTR("CollisionShape2D only serves to provide a collision shape to a CollisionObject2D derived node. Please only use it as a child of Area2D, StaticBody2D, RigidBody2D, KinematicBody2D, etc. to give them a shape.");
	}

	if (!shape.is_valid()) {
		return TTR("A shape must be provided for CollisionShape2D to function. Please create a shape resource for it!");
	}

	return String();
}

void CollisionShape2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &CollisionShape2D::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &CollisionShape2D::get_shape);
	ClassDB::bind_method(D_METHOD("_shape_changed"), &CollisionShape2D::_shape_changed);
	ClassDB::bind_method(D_METHOD("_add_to_collision_object"), &CollisionShape2D::_add_to_collision_object);
	ClassDB::bind_method(D_METHOD("set_trigger", "enable"), &CollisionShape2D::set_trigger);
	ClassDB::bind_method(D_METHOD("is_trigger"), &CollisionShape2D::is_trigger);

	ClassDB::bind_method(D_METHOD("_set_update_shape_index", "index"), &CollisionShape2D::_set_update_shape_index);
	ClassDB::bind_method(D_METHOD("_get_update_shape_index"), &CollisionShape2D::_get_update_shape_index);

	ClassDB::bind_method(D_METHOD("get_collision_object_shape_index"), &CollisionShape2D::get_collision_object_shape_index);

	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape2D"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "trigger"), "set_trigger", "is_trigger");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "_update_shape_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_update_shape_index", "_get_update_shape_index");
}

CollisionShape2D::CollisionShape2D() {

	rect = Rect2(-Point2(10, 10), Point2(20, 20));
	set_notify_local_transform(true);
	trigger = false;
	unparenting = false;
	can_update_body = false;
	update_shape_index = -1;
}
