/*************************************************************************/
/*  reference_rect.cpp                                                   */
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

#include "reference_rect.h"

#include "core/engine.h"

void ReferenceRect::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		if (!is_inside_tree()) {
			return;
		}
		if (Engine::get_singleton()->is_editor_hint() || !editor_only) {
			draw_rect(Rect2(Point2(), get_size()), border_color, false, border_width);
		}
	}
}

void ReferenceRect::set_border_color(const Color &p_color) {
	border_color = p_color;
	update();
}

Color ReferenceRect::get_border_color() const {
	return border_color;
}

void ReferenceRect::set_border_width(float p_width) {
	border_width = MAX(0.0, p_width);
	update();
}

float ReferenceRect::get_border_width() const {
	return border_width;
}

void ReferenceRect::set_editor_only(const bool &p_enabled) {
	editor_only = p_enabled;
	update();
}

bool ReferenceRect::get_editor_only() const {
	return editor_only;
}

void ReferenceRect::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_border_color"), &ReferenceRect::get_border_color);
	ClassDB::bind_method(D_METHOD("set_border_color", "color"), &ReferenceRect::set_border_color);

	ClassDB::bind_method(D_METHOD("get_border_width"), &ReferenceRect::get_border_width);
	ClassDB::bind_method(D_METHOD("set_border_width", "width"), &ReferenceRect::set_border_width);

	ClassDB::bind_method(D_METHOD("get_editor_only"), &ReferenceRect::get_editor_only);
	ClassDB::bind_method(D_METHOD("set_editor_only", "enabled"), &ReferenceRect::set_editor_only);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_color"), "set_border_color", "get_border_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "border_width", PROPERTY_HINT_RANGE, "0.0,5.0,0.1,or_greater"), "set_border_width", "get_border_width");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_only"), "set_editor_only", "get_editor_only");
}
