/*************************************************************************/
/*  viewport_container.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "viewport_container.h"

#include "scene/main/viewport.h"

Size2 ViewportContainer::get_minimum_size() const {

	if (stretch)
		return Size2();
	Size2 ms;
	for (int i = 0; i < get_child_count(); i++) {

		Viewport *c = Object::cast_to<Viewport>(get_child(i));
		if (!c)
			continue;

		Size2 minsize = c->get_size();
		ms.width = MAX(ms.width, minsize.width);
		ms.height = MAX(ms.height, minsize.height);
	}

	return ms;
}

void ViewportContainer::set_stretch(bool p_enable) {

	stretch = p_enable;
	queue_sort();
	update();
}

bool ViewportContainer::is_stretch_enabled() const {

	return stretch;
}

void ViewportContainer::set_stretch_shrink(int p_shrink) {

	ERR_FAIL_COND(p_shrink < 1);
	if (shrink == p_shrink)
		return;

	shrink = p_shrink;

	if (!stretch)
		return;

	for (int i = 0; i < get_child_count(); i++) {

		Viewport *c = Object::cast_to<Viewport>(get_child(i));
		if (!c)
			continue;

		c->set_size(get_size() / shrink);
	}

	update();
}

int ViewportContainer::get_stretch_shrink() const {

	return shrink;
}

void ViewportContainer::_notification(int p_what) {

	if (p_what == NOTIFICATION_RESIZED) {

		if (!stretch)
			return;

		for (int i = 0; i < get_child_count(); i++) {

			Viewport *c = Object::cast_to<Viewport>(get_child(i));
			if (!c)
				continue;

			c->set_size(get_size() / shrink);
		}
	}

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_VISIBILITY_CHANGED) {

		for (int i = 0; i < get_child_count(); i++) {

			Viewport *c = Object::cast_to<Viewport>(get_child(i));
			if (!c)
				continue;

			if (is_visible_in_tree())
				c->set_update_mode(Viewport::UPDATE_ALWAYS);
			else
				c->set_update_mode(Viewport::UPDATE_DISABLED);
		}
	}

	if (p_what == NOTIFICATION_DRAW) {

		for (int i = 0; i < get_child_count(); i++) {

			Viewport *c = Object::cast_to<Viewport>(get_child(i));
			if (!c)
				continue;

			if (stretch)
				draw_texture_rect(c->get_texture(), Rect2(Vector2(), get_size() * Size2(1, -1)));
			else
				draw_texture_rect(c->get_texture(), Rect2(Vector2(), c->get_size() * Size2(1, -1)));
		}
	}
}

void ViewportContainer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_stretch", "enable"), &ViewportContainer::set_stretch);
	ClassDB::bind_method(D_METHOD("is_stretch_enabled"), &ViewportContainer::is_stretch_enabled);

	ClassDB::bind_method(D_METHOD("set_stretch_shrink", "amount"), &ViewportContainer::set_stretch_shrink);
	ClassDB::bind_method(D_METHOD("get_stretch_shrink"), &ViewportContainer::get_stretch_shrink);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stretch"), "set_stretch", "is_stretch_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "stretch_shrink"), "set_stretch_shrink", "get_stretch_shrink");
}

ViewportContainer::ViewportContainer() {

	stretch = false;
	shrink = 1;
}
