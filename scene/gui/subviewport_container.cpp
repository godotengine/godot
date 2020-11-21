/*************************************************************************/
/*  subviewport_container.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "subviewport_container.h"

#include "core/config/engine.h"
#include "scene/main/viewport.h"

Size2 SubViewportContainer::get_minimum_size() const {
	if (stretch) {
		return Size2();
	}
	Size2 ms;
	for (int i = 0; i < get_child_count(); i++) {
		SubViewport *c = Object::cast_to<SubViewport>(get_child(i));
		if (!c) {
			continue;
		}

		Size2 minsize = c->get_size();
		ms.width = MAX(ms.width, minsize.width);
		ms.height = MAX(ms.height, minsize.height);
	}

	return ms;
}

void SubViewportContainer::set_stretch(bool p_enable) {
	stretch = p_enable;
	update_minimum_size();
	queue_sort();
	update();
}

bool SubViewportContainer::is_stretch_enabled() const {
	return stretch;
}

void SubViewportContainer::set_stretch_shrink(int p_shrink) {
	ERR_FAIL_COND(p_shrink < 1);
	if (shrink == p_shrink) {
		return;
	}

	shrink = p_shrink;

	if (!stretch) {
		return;
	}

	for (int i = 0; i < get_child_count(); i++) {
		SubViewport *c = Object::cast_to<SubViewport>(get_child(i));
		if (!c) {
			continue;
		}

		c->set_size(get_size() / shrink);
	}

	update();
}

int SubViewportContainer::get_stretch_shrink() const {
	return shrink;
}

void SubViewportContainer::_notification(int p_what) {
	if (p_what == NOTIFICATION_RESIZED) {
		if (!stretch) {
			return;
		}

		for (int i = 0; i < get_child_count(); i++) {
			SubViewport *c = Object::cast_to<SubViewport>(get_child(i));
			if (!c) {
				continue;
			}

			c->set_size(get_size() / shrink);
		}
	}

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		for (int i = 0; i < get_child_count(); i++) {
			SubViewport *c = Object::cast_to<SubViewport>(get_child(i));
			if (!c) {
				continue;
			}

			if (is_visible_in_tree()) {
				c->set_update_mode(SubViewport::UPDATE_ALWAYS);
			} else {
				c->set_update_mode(SubViewport::UPDATE_DISABLED);
			}

			c->set_handle_input_locally(false); //do not handle input locally here
		}
	}

	if (p_what == NOTIFICATION_DRAW) {
		for (int i = 0; i < get_child_count(); i++) {
			SubViewport *c = Object::cast_to<SubViewport>(get_child(i));
			if (!c) {
				continue;
			}

			if (stretch) {
				draw_texture_rect(c->get_texture(), Rect2(Vector2(), get_size()));
			} else {
				draw_texture_rect(c->get_texture(), Rect2(Vector2(), c->get_size()));
			}
		}
	}
}

void SubViewportContainer::input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	Transform2D xform = get_global_transform();

	if (stretch) {
		Transform2D scale_xf;
		scale_xf.scale(Vector2(shrink, shrink));
		xform *= scale_xf;
	}

	Ref<InputEvent> ev = p_event->xformed_by(xform.affine_inverse());

	for (int i = 0; i < get_child_count(); i++) {
		SubViewport *c = Object::cast_to<SubViewport>(get_child(i));
		if (!c || c->is_input_disabled()) {
			continue;
		}

		c->push_input(ev);
	}
}

void SubViewportContainer::unhandled_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	Transform2D xform = get_global_transform();

	if (stretch) {
		Transform2D scale_xf;
		scale_xf.scale(Vector2(shrink, shrink));
		xform *= scale_xf;
	}

	Ref<InputEvent> ev = p_event->xformed_by(xform.affine_inverse());

	for (int i = 0; i < get_child_count(); i++) {
		SubViewport *c = Object::cast_to<SubViewport>(get_child(i));
		if (!c || c->is_input_disabled()) {
			continue;
		}

		c->push_unhandled_input(ev);
	}
}

void SubViewportContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_stretch", "enable"), &SubViewportContainer::set_stretch);
	ClassDB::bind_method(D_METHOD("is_stretch_enabled"), &SubViewportContainer::is_stretch_enabled);

	ClassDB::bind_method(D_METHOD("set_stretch_shrink", "amount"), &SubViewportContainer::set_stretch_shrink);
	ClassDB::bind_method(D_METHOD("get_stretch_shrink"), &SubViewportContainer::get_stretch_shrink);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stretch"), "set_stretch", "is_stretch_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "stretch_shrink"), "set_stretch_shrink", "get_stretch_shrink");
}

SubViewportContainer::SubViewportContainer() {
	set_process_input(true);
	set_process_unhandled_input(true);
}
