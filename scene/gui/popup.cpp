/*************************************************************************/
/*  popup.cpp                                                            */
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

#include "popup.h"

#include "core/engine.h"
#include "core/os/keyboard.h"
#include "scene/gui/panel.h"

void Popup::_input_from_window(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> key = p_event;
	if (key.is_valid() && key->is_pressed() && key->get_keycode() == KEY_ESCAPE) {
		_close_pressed();
	}
}

void Popup::_parent_focused() {
	_close_pressed();
}
void Popup::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {

				parent_visible = get_parent_visible_window();
				if (parent_visible) {
					parent_visible->connect("focus_entered", callable_mp(this, &Popup::_parent_focused));
				}
			} else {
				if (parent_visible) {
					parent_visible->disconnect("focus_entered", callable_mp(this, &Popup::_parent_focused));
					parent_visible = nullptr;
				}

				emit_signal("popup_hide");
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (parent_visible) {
				parent_visible->disconnect("focus_entered", callable_mp(this, &Popup::_parent_focused));
				parent_visible = nullptr;
			}
		} break;
		case NOTIFICATION_WM_CLOSE_REQUEST: {
			_close_pressed();
		} break;
	}
}

void Popup::_close_pressed() {

	Window *parent_window = parent_visible;
	if (parent_visible) {
		parent_visible->disconnect("focus_entered", callable_mp(this, &Popup::_parent_focused));
		parent_visible = nullptr;
	}

	call_deferred("hide");

	emit_signal("cancelled");

	if (parent_window) {
		//parent_window->grab_focus();
	}
}

void Popup::set_as_minsize() {
	set_size(get_contents_minimum_size());
}
void Popup::_bind_methods() {

	ADD_SIGNAL(MethodInfo("popup_hide"));
}

Popup::Popup() {

	parent_visible = nullptr;

	set_wrap_controls(true);
	set_visible(false);
	set_transient(true);
	set_flag(FLAG_BORDERLESS, true);
	set_flag(FLAG_RESIZE_DISABLED, true);

	connect("window_input", callable_mp(this, &Popup::_input_from_window));
}

Popup::~Popup() {
}

Size2 PopupPanel::_get_contents_minimum_size() const {

	Ref<StyleBox> p = get_theme_stylebox("panel", "PopupPanel");

	Size2 ms;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c || c == panel)
			continue;

		if (c->is_set_as_toplevel())
			continue;

		Size2 cms = c->get_combined_minimum_size();
		ms.x = MAX(cms.x, ms.x);
		ms.y = MAX(cms.y, ms.y);
	}

	return ms + p->get_minimum_size();
}

void PopupPanel::_update_child_rects() {

	Ref<StyleBox> p = get_theme_stylebox("panel", "PopupPanel");

	Vector2 cpos(p->get_offset());
	Vector2 csize(get_size() - p->get_minimum_size());

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c)
			continue;

		if (c->is_set_as_toplevel())
			continue;

		if (c == panel) {
			c->set_position(Vector2());
			c->set_size(get_size());
		} else {
			c->set_position(cpos);
			c->set_size(csize);
		}
	}
}

void PopupPanel::_notification(int p_what) {

	if (p_what == NOTIFICATION_READY || p_what == NOTIFICATION_ENTER_TREE) {

		panel->add_theme_style_override("panel", get_theme_stylebox("panel", "PopupPanel"));
		_update_child_rects();
	} else if (p_what == NOTIFICATION_WM_SIZE_CHANGED) {

		_update_child_rects();
	}
}

PopupPanel::PopupPanel() {

	panel = memnew(Panel);
	add_child(panel);
}
