/*************************************************************************/
/*  solid_shape_2d.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "solid_shape_2d.h"

void SolidShape2D::_shape_changed() {

	update();
}

void SolidShape2D::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_DRAW:
			if (!shape.is_valid())
				return;
			shape->draw(get_canvas_item(), color);
			break;
	}
}

void SolidShape2D::set_shape(const Ref<Shape2D> &p_shape) {

	if (shape.is_valid())
		shape->disconnect("changed", this, "_shape_changed");
	shape = p_shape;
	update();
	if (shape.is_valid())
		shape->connect("changed", this, "_shape_changed");

	update_configuration_warning();
}

String SolidShape2D::get_configuration_warning() const {

	if (!shape.is_valid()) {
		return TTR("A shape must be provided for SolidShape2D to function. Please create a shape resource for it!");
	}

	return String();
}

Ref<Shape2D> SolidShape2D::get_shape() const {

	return shape;
}

void SolidShape2D::set_color(const Color &p_color) {

	color = p_color;
	update();
}

Color SolidShape2D::get_color() const {

	return color;
}

void SolidShape2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &SolidShape2D::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &SolidShape2D::get_shape);

	ClassDB::bind_method(D_METHOD("set_color", "color"), &SolidShape2D::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &SolidShape2D::get_color);

	ClassDB::bind_method(D_METHOD("_shape_changed"), &SolidShape2D::_shape_changed);

	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape2D"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
}

SolidShape2D::SolidShape2D() {

	color = Color(1, 1, 1);
}
