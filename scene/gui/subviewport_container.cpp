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
#include "core/object/class_db.h"
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
		ms = ms.max(minsize);
	}

	return ms;
}

void SubViewportContainer::set_mode(ResolutionMode p_mode) {
	if (mode == p_mode) {
		return;
	}

	mode = p_mode;
	notify_property_list_changed();
	recalc_force_viewport_sizes();
	update_minimum_size();
	queue_sort();
	queue_redraw();
}

SubViewportContainer::ResolutionMode SubViewportContainer::get_mode() const {
	return mode;
}
void SubViewportContainer::set_fixed_resolution(const Vector2i &p_fixed_resolution) {
	if (fixed_resolution == p_fixed_resolution) {
		return;
	}

	fixed_resolution = p_fixed_resolution.maxi(1);
	recalc_force_viewport_sizes();
	update_minimum_size();
	queue_sort();
	queue_redraw();
}

Vector2i SubViewportContainer::get_fixed_resolution() const {
	return fixed_resolution;
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

	// If stretch is enabled, make sure that all child SubViwewports have the correct size.
	for (int i = 0; i < get_child_count(); i++) {
		SubViewport *c = Object::cast_to<SubViewport>(get_child(i));
		if (!c) {
			continue;
		}

		switch (mode) {
			case RESOLUTION_MODE_SHRINK: {
				c->set_size_force(get_size() / shrink);
			} break;
			case RESOLUTION_MODE_FIXED: {
				c->set_size_force(fixed_resolution);
			} break;
		}
	}
}

int SubViewportContainer::get_stretch_shrink() const {
	return shrink;
}

Vector<int> SubViewportContainer::get_allowed_size_flags_horizontal() const {
	return Vector<int>();
}

Vector<int> SubViewportContainer::get_allowed_size_flags_vertical() const {
	return Vector<int>();
}

void SubViewportContainer::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "stretch_shrink") {
		p_property.usage = mode == RESOLUTION_MODE_SHRINK ? PROPERTY_USAGE_DEFAULT : PROPERTY_USAGE_NO_EDITOR;
	} else if (p_property.name == "fixed_resolution") {
		p_property.usage = mode == RESOLUTION_MODE_FIXED ? PROPERTY_USAGE_DEFAULT : PROPERTY_USAGE_NO_EDITOR;
	}
}

void SubViewportContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_RESIZED: {
			recalc_force_viewport_sizes();
		} break;

		case NOTIFICATION_ENTER_TREE:
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

	Vector2 s;
	switch (mode) {
		case RESOLUTION_MODE_SHRINK: {
			s = Vector2(1, 1) / shrink;
		} break;
		case RESOLUTION_MODE_FIXED: {
			s = Vector2(fixed_resolution) / get_size();
		} break;
	}

	if (stretch) {
		_send_event_to_viewports(p_event->xformed_by(Transform2D().scaled(s)));
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

		c->push_input(p_event);
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
	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &SubViewportContainer::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &SubViewportContainer::get_mode);

	ClassDB::bind_method(D_METHOD("set_fixed_resolution", "fixed_resolution"), &SubViewportContainer::set_fixed_resolution);
	ClassDB::bind_method(D_METHOD("get_fixed_resolution"), &SubViewportContainer::get_fixed_resolution);

	ClassDB::bind_method(D_METHOD("set_stretch", "enable"), &SubViewportContainer::set_stretch);
	ClassDB::bind_method(D_METHOD("is_stretch_enabled"), &SubViewportContainer::is_stretch_enabled);

	ClassDB::bind_method(D_METHOD("set_stretch_shrink", "amount"), &SubViewportContainer::set_stretch_shrink);
	ClassDB::bind_method(D_METHOD("get_stretch_shrink"), &SubViewportContainer::get_stretch_shrink);

	ClassDB::bind_method(D_METHOD("set_mouse_target", "amount"), &SubViewportContainer::set_mouse_target);
	ClassDB::bind_method(D_METHOD("is_mouse_target_enabled"), &SubViewportContainer::is_mouse_target_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Shrink,Fixed"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "fixed_resolution"), "set_fixed_resolution", "get_fixed_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stretch"), "set_stretch", "is_stretch_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "stretch_shrink", PROPERTY_HINT_RANGE, "1,32,1,or_greater"), "set_stretch_shrink", "get_stretch_shrink");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mouse_target"), "set_mouse_target", "is_mouse_target_enabled");

	BIND_ENUM_CONSTANT(RESOLUTION_MODE_SHRINK);
	BIND_ENUM_CONSTANT(RESOLUTION_MODE_FIXED);

	GDVIRTUAL_BIND(_propagate_input_event, "event");
}

SubViewportContainer::SubViewportContainer() {
	set_process_unhandled_input(true);
	set_focus_mode(FOCUS_CLICK);
}
