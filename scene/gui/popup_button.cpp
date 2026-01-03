/**************************************************************************/
/*  popup_button.cpp                                                      */
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

#include "popup_button.h"

void PopupButton::_setup_popup() {
	popup->hide();
	popup->set_layout_direction((Window::LayoutDirection)get_layout_direction());
	popup->connect(SceneStringName(visibility_changed), callable_mp(this, &PopupButton::_popup_visibility_changed));
}

void PopupButton::_popup_visibility_changed() {
	set_pressed(popup->is_visible());

	if (!popup->is_visible()) {
		set_process_internal(false);
	} else if (switch_on_hover) {
		set_process_internal(true);
	}
}

void PopupButton::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			if (popup) {
				popup->set_layout_direction((Window::LayoutDirection)get_layout_direction());
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (popup && !is_visible_in_tree()) {
				popup->hide();
			}
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			PopupButton *other_button = Object::cast_to<PopupButton>(get_viewport()->gui_get_hovered_control());
			if (other_button && other_button != this && other_button->is_switch_on_hover() && !other_button->is_disabled() &&
					(get_parent()->is_ancestor_of(other_button) || other_button->get_parent()->is_ancestor_of(popup))) {
				popup->hide();
				other_button->pressed();
			}
		} break;
	}
}

void PopupButton::_bind_methods() {
	GDVIRTUAL_BIND(_create_popup);
	GDVIRTUAL_BIND(_about_to_popup);
	GDVIRTUAL_BIND(_setup_popup_position);
	GDVIRTUAL_BIND(_post_popup);

	ClassDB::bind_method(D_METHOD("set_switch_on_hover", "enable"), &PopupButton::set_switch_on_hover);
	ClassDB::bind_method(D_METHOD("is_switch_on_hover"), &PopupButton::is_switch_on_hover);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "switch_on_hover"), "set_switch_on_hover", "is_switch_on_hover");

	ClassDB::bind_method(D_METHOD("get_generic_popup"), &PopupButton::get_generic_popup);
	ClassDB::bind_method(D_METHOD("show_popup"), &PopupButton::show_popup);

	ADD_SIGNAL(MethodInfo("about_to_popup"));
}

void PopupButton::add_child_notify(Node *p_child) {
	if (!popup) {
		Popup *potential_popup = Object::cast_to<Popup>(p_child);
		if (potential_popup) {
			popup = potential_popup;
			_setup_popup();
		}
	}
}

Popup *PopupButton::create_popup() {
	Popup *new_popup = nullptr;
	GDVIRTUAL_CALL(_create_popup, new_popup);
	return new_popup;
}

void PopupButton::about_to_popup() {
	GDVIRTUAL_CALL(_about_to_popup);
}

void PopupButton::setup_popup_position() {
	if (GDVIRTUAL_CALL(_setup_popup_position)) {
		return;
	}

	Rect2 rect = get_screen_rect();
	rect.position.y += rect.size.height;
	if (get_viewport()->is_embedding_subwindows() && popup->get_force_native()) {
		Transform2D xform = get_viewport()->get_popup_base_transform_native();
		rect = xform.xform(rect);
	}
	Rect2i scr_usable = DisplayServer::get_singleton()->screen_get_usable_rect(get_window()->get_current_screen());
	Size2i max_size;
	if (scr_usable.has_area()) {
		real_t max_h = scr_usable.get_end().y - rect.position.y;
		if (max_h >= 4 * rect.size.height) {
			max_size = Size2(RS::get_singleton()->get_maximum_viewport_size().width, max_h);
		}
	}
	popup->set_max_size(max_size);
	if (is_layout_rtl()) {
		rect.position.x += rect.size.width - popup->get_size().width;
	}
	popup->set_position(rect.position);
}

void PopupButton::post_popup() {
	GDVIRTUAL_CALL(_post_popup);
}

void PopupButton::ensure_popup() {
	if (popup) {
		return;
	}
	popup = create_popup();
	ERR_FAIL_NULL_MSG(popup, "PopupButton has no Popup. Either add one as a child or return it from _create_popup().");

	if (!popup->get_parent()) {
		add_child(popup, false, INTERNAL_MODE_FRONT);
	}
	_setup_popup();
}

void PopupButton::pressed() {
	ensure_popup();
	if (!popup) {
		return;
	}

	if (popup->is_visible()) {
		popup->hide();
	} else {
		show_popup();
	}
}

Popup *PopupButton::get_generic_popup() {
	ensure_popup();
	return popup;
}

void PopupButton::show_popup() {
	if (!popup) {
		return;
	}
	emit_signal(SNAME("about_to_popup"));
	about_to_popup();

	setup_popup_position();
	popup->popup();

	post_popup();
}

PopupButton::PopupButton() {
	set_action_mode(ACTION_MODE_BUTTON_PRESS);
	set_toggle_mode(true);
}
