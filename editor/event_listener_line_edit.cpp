/*************************************************************************/
/*  event_listener_line_edit.cpp                                         */
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

#include "editor/event_listener_line_edit.h"

bool EventListenerLineEdit::_is_event_allowed(const Ref<InputEvent> &p_event) const {
	const Ref<InputEventMouseButton> mb = p_event;
	const Ref<InputEventKey> k = p_event;
	const Ref<InputEventJoypadButton> jb = p_event;
	const Ref<InputEventJoypadMotion> jm = p_event;

	return (mb.is_valid() && (allowed_input_types & INPUT_MOUSE_BUTTON)) ||
			(k.is_valid() && (allowed_input_types & INPUT_KEY)) ||
			(jb.is_valid() && (allowed_input_types & INPUT_JOY_BUTTON)) ||
			(jm.is_valid() && (allowed_input_types & INPUT_JOY_MOTION));
}

void EventListenerLineEdit::gui_input(const Ref<InputEvent> &p_event) {
	const Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		LineEdit::gui_input(p_event);
		return;
	}

	// Allow mouse button click on the clear button without being treated as an event.
	const Ref<InputEventMouseButton> b = p_event;
	if (b.is_valid() && _is_over_clear_button(b->get_position())) {
		LineEdit::gui_input(p_event);
		return;
	}

	// First event will be an event which is used to focus this control - i.e. a mouse click, or a tab press.
	// Ignore the first one so that clicking into the LineEdit does not override the current event.
	// Ignore is reset to true when the control is unfocused.
	// This class also specially handles grab_focus() calls.
	if (ignore_next_event) {
		ignore_next_event = false;
		return;
	}

	accept_event();
	if (!p_event->is_pressed() || p_event->is_echo() || p_event->is_match(event) || !_is_event_allowed(p_event)) {
		return;
	}

	event = p_event;
	set_text(event->as_text());
	emit_signal("event_changed", event);
}

void EventListenerLineEdit::_on_text_changed(const String &p_text) {
	if (p_text.is_empty()) {
		clear_event();
	}
}

void EventListenerLineEdit::_on_focus() {
	set_placeholder(TTR("Listening for input..."));
}

void EventListenerLineEdit::_on_unfocus() {
	ignore_next_event = true;
	set_placeholder(TTR("Filter by event..."));
}

Ref<InputEvent> EventListenerLineEdit::get_event() const {
	return event;
}

void EventListenerLineEdit::clear_event() {
	if (event.is_valid()) {
		event = Ref<InputEvent>();
		set_text("");
		emit_signal("event_changed", event);
	}
}

void EventListenerLineEdit::set_allowed_input_types(int input_types) {
	allowed_input_types = input_types;
}

int EventListenerLineEdit::get_allowed_input_types() const {
	return allowed_input_types;
}

void EventListenerLineEdit::grab_focus() {
	// If we grab focus through code, we don't need to ignore the first event!
	ignore_next_event = false;
	Control::grab_focus();
}

void EventListenerLineEdit::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			connect("text_changed", callable_mp(this, &EventListenerLineEdit::_on_text_changed));
			connect("focus_entered", callable_mp(this, &EventListenerLineEdit::_on_focus));
			connect("focus_exited", callable_mp(this, &EventListenerLineEdit::_on_unfocus));
			set_right_icon(get_theme_icon(SNAME("Keyboard"), SNAME("EditorIcons")));
			set_clear_button_enabled(true);
		} break;
	}
}

void EventListenerLineEdit::_bind_methods() {
	ADD_SIGNAL(MethodInfo("event_changed", PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
}

EventListenerLineEdit::EventListenerLineEdit() {
	set_caret_blink_enabled(false);
	set_placeholder(TTR("Filter by event..."));
}
