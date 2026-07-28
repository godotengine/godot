/**************************************************************************/
/*  subviewport_container.cpp                                             */
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

#include "subviewport_container.h"

#include "core/config/engine.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"

Size2 SubViewportContainer::get_minimum_size() const {
	if (stretch) {
		return Size2();
	}
	return Container::_get_minimum_size();
}

void SubViewportContainer::set_stretch(bool p_enable) {
	if (stretch == p_enable) {
		return;
	}

	stretch = p_enable;
	recalc_force_viewport_sizes();
	update_minimum_size();
	queue_sort();
	queue_redraw();
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

	recalc_force_viewport_sizes();
	queue_redraw();
}

void SubViewportContainer::recalc_force_viewport_sizes() {
	if (!stretch) {
		return;
	}

	// Make sure that all child SubViewports have the correct size.
	for (int i = 0; i < get_child_count(); i++) {
		SubViewport *c = Object::cast_to<SubViewport>(get_child(i));
		if (!c) {
			continue;
		}

		Size2 render_size = get_size();

		if (get_viewport() != nullptr) {
			Transform2D parent_sxform = get_viewport()->get_final_transform();
			Size2 parent_zoom = parent_sxform.get_scale();

			// Calculate visual size from parent scaled canvas transform
			int render_width = (int)(get_size().x * parent_zoom.x);
			int render_height = (int)(get_size().y * parent_zoom.y);
			double nominal_aspect = render_height / (double)render_width;

			// Limit render size to 1.5x window size to avoid error
			Size2i window_size = get_viewport()->get_window()->get_size();
			int max_width = (int)(window_size.x * 1.5);
			int max_height = (int)(window_size.y * 1.5);

			if (render_width > max_width) {
				render_width = max_width;
				render_height = (int)(nominal_aspect * render_width);
			} else {
				if (render_height > max_height) {
					render_height = max_height;
					render_width = (int)(nominal_aspect * render_width);
				}
			}

			render_size = Size2i(render_width, render_height);

			// Ensure subviewports work in the overridden resolution
			c->set_size_2d_override(get_size());

			if (!c->is_size_2d_override_stretch_enabled()) {
				c->set_size_2d_override_stretch(true);
			}
		}

		c->set_size_force(render_size / shrink);
	}
}

int SubViewportContainer::get_stretch_shrink() const {
	return shrink;
}

Vector<int> SubViewportContainer::get_allowed_size_flags_horizontal() const {
	Vector<int> flags;
	flags.append(SIZE_MAXIMIZE);
	return flags;
}

Vector<int> SubViewportContainer::get_allowed_size_flags_vertical() const {
	Vector<int> flags;
	flags.append(SIZE_MAXIMIZE);
	return flags;
}

void SubViewportContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_RESIZED: {
			recalc_force_viewport_sizes();
		} break;

		case NOTIFICATION_ENTER_TREE: {
			// Capture parent viewport resizing as target resolution may change while canvas resolution is unchanged
			if (!get_viewport()->is_connected("size_changed", callable_mp(this, &SubViewportContainer::recalc_force_viewport_sizes))) {
				get_viewport()->connect("size_changed", callable_mp(this, &SubViewportContainer::recalc_force_viewport_sizes));
			}
			[[fallthrough]];
		}
		case NOTIFICATION_VISIBILITY_CHANGED: {
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
		} break;

		case NOTIFICATION_DRAW: {
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
		} break;

		case NOTIFICATION_FOCUS_ENTER: {
			// If focused, send InputEvent to the SubViewport before the Gui-Input stage.
			set_process_input(true);
			set_process_unhandled_input(false);
		} break;

		case NOTIFICATION_FOCUS_EXIT: {
			// A different Control has focus and should receive Gui-Input before the InputEvent is sent to the SubViewport.
			set_process_input(false);
			set_process_unhandled_input(true);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			get_viewport()->disconnect("size_changed", callable_mp(this, &SubViewportContainer::recalc_force_viewport_sizes));
		} break;
	}
}

void SubViewportContainer::_notify_viewports(int p_notification) {
	for (int i = 0; i < get_child_count(); i++) {
		SubViewport *c = Object::cast_to<SubViewport>(get_child(i));
		if (!c) {
			continue;
		}
		c->notification(p_notification);
	}
}

void SubViewportContainer::input(const Ref<InputEvent> &p_event) {
	_propagate_nonpositional_event(p_event);
}

void SubViewportContainer::unhandled_input(const Ref<InputEvent> &p_event) {
	_propagate_nonpositional_event(p_event);
}

void SubViewportContainer::_propagate_nonpositional_event(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	if (_is_propagated_in_gui_input(p_event)) {
		return;
	}

	bool send;
	if (GDVIRTUAL_CALL(_propagate_input_event, p_event, send)) {
		if (!send) {
			return;
		}
	}

	_send_event_to_viewports(p_event);
}

void SubViewportContainer::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	if (!_is_propagated_in_gui_input(p_event)) {
		return;
	}

	bool send;
	if (GDVIRTUAL_CALL(_propagate_input_event, p_event, send)) {
		if (!send) {
			return;
		}
	}

	if (stretch && shrink > 1) {
		Transform2D xform;
		xform.scale(Vector2(1, 1) / shrink);
		_send_event_to_viewports(p_event->xformed_by(xform));
	} else {
		_send_event_to_viewports(p_event);
	}
}

void SubViewportContainer::_send_event_to_viewports(const Ref<InputEvent> &p_event) {
	for (int i = 0; i < get_child_count(); i++) {
		SubViewport *c = Object::cast_to<SubViewport>(get_child(i));
		if (!c || c->is_input_disabled()) {
			continue;
		}

		c->push_input(p_event, true);
	}
}

bool SubViewportContainer::_is_propagated_in_gui_input(const Ref<InputEvent> &p_event) {
	// Propagation of events with a position property happen in gui_input
	// Propagation of other events happen in input
	if (Object::cast_to<InputEventMouse>(*p_event) || Object::cast_to<InputEventScreenDrag>(*p_event) || Object::cast_to<InputEventScreenTouch>(*p_event) || Object::cast_to<InputEventGesture>(*p_event)) {
		return true;
	}
	return false;
}

void SubViewportContainer::set_mouse_target(bool p_enable) {
	mouse_target = p_enable;
}

bool SubViewportContainer::is_mouse_target_enabled() {
	return mouse_target;
}

void SubViewportContainer::add_child_notify(Node *p_child) {
	if (Object::cast_to<SubViewport>(p_child)) {
		queue_redraw();
	}
}

void SubViewportContainer::remove_child_notify(Node *p_child) {
	if (Object::cast_to<SubViewport>(p_child)) {
		queue_redraw();
	}
}

PackedStringArray SubViewportContainer::get_configuration_warnings() const {
	PackedStringArray warnings = Container::get_configuration_warnings();

	bool has_viewport = false;
	for (int i = 0; i < get_child_count(); i++) {
		if (Object::cast_to<SubViewport>(get_child(i))) {
			has_viewport = true;
			break;
		}
	}
	if (!has_viewport) {
		warnings.push_back(RTR("This node doesn't have a SubViewport as child, so it can't display its intended content.\nConsider adding a SubViewport as a child to provide something displayable."));
	}

	if (get_default_cursor_shape() != Control::CURSOR_ARROW) {
		warnings.push_back(RTR("The default mouse cursor shape of SubViewportContainer has no effect.\nConsider leaving it at its initial value `CURSOR_ARROW`."));
	}

	return warnings;
}

void SubViewportContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_stretch", "enable"), &SubViewportContainer::set_stretch);
	ClassDB::bind_method(D_METHOD("is_stretch_enabled"), &SubViewportContainer::is_stretch_enabled);

	ClassDB::bind_method(D_METHOD("set_stretch_shrink", "amount"), &SubViewportContainer::set_stretch_shrink);
	ClassDB::bind_method(D_METHOD("get_stretch_shrink"), &SubViewportContainer::get_stretch_shrink);

	ClassDB::bind_method(D_METHOD("set_mouse_target", "amount"), &SubViewportContainer::set_mouse_target);
	ClassDB::bind_method(D_METHOD("is_mouse_target_enabled"), &SubViewportContainer::is_mouse_target_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stretch"), "set_stretch", "is_stretch_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "stretch_shrink", PROPERTY_HINT_RANGE, "1,32,1,or_greater"), "set_stretch_shrink", "get_stretch_shrink");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mouse_target"), "set_mouse_target", "is_mouse_target_enabled");

	GDVIRTUAL_BIND(_propagate_input_event, "event");
}

SubViewportContainer::SubViewportContainer() {
	set_process_unhandled_input(true);
	set_focus_mode(FOCUS_CLICK);
}
