/*************************************************************************/
/*  menu_button.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "menu_button.h"

#include "core/os/keyboard.h"
#include "scene/main/window.h"

void MenuButton::unhandled_key_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (!_is_focus_owner_in_shorcut_context()) {
		return;
	}

	if (disable_shortcuts) {
		return;
	}

	if (p_event->is_pressed() && !p_event->is_echo() && (Object::cast_to<InputEventKey>(p_event.ptr()) || Object::cast_to<InputEventJoypadButton>(p_event.ptr()) || Object::cast_to<InputEventAction>(*p_event) || Object::cast_to<InputEventShortcut>(*p_event))) {
		if (!get_parent() || !is_visible_in_tree() || is_disabled()) {
			return;
		}

		if (popup->activate_item_by_event(p_event, false)) {
			accept_event();
		}
	}
}

void MenuButton::_popup_visibility_changed(bool p_visible) {
	set_pressed(p_visible);

	if (!p_visible) {
		set_process_internal(false);
		return;
	}

	if (switch_on_hover) {
		Window *window = Object::cast_to<Window>(get_viewport());
		if (window) {
			mouse_pos_adjusted = window->get_position();

			if (window->is_embedded()) {
				Window *window_parent = Object::cast_to<Window>(window->get_parent()->get_viewport());
				while (window_parent) {
					if (!window_parent->is_embedded()) {
						mouse_pos_adjusted += window_parent->get_position();
						break;
					}

					window_parent = Object::cast_to<Window>(window_parent->get_parent()->get_viewport());
				}
			}

			set_process_internal(true);
		}
	}
}

void MenuButton::pressed() {
	emit_signal(SNAME("about_to_popup"));
	Size2 size = get_size() * get_viewport()->get_canvas_transform().get_scale();

	popup->set_size(Size2(size.width, 0));
	Point2 gp = get_screen_position();
	gp.y += size.y;
	if (is_layout_rtl()) {
		gp.x += size.width - popup->get_size().width;
	}
	popup->set_position(gp);
	popup->set_parent_rect(Rect2(Point2(gp - popup->get_position()), size));

	popup->take_mouse_focus();
	popup->popup();
}

void MenuButton::gui_input(const Ref<InputEvent> &p_event) {
	BaseButton::gui_input(p_event);
}

PopupMenu *MenuButton::get_popup() const {
	return popup;
}

void MenuButton::_set_items(const Array &p_items) {
	popup->set("items", p_items);
}

Array MenuButton::_get_items() const {
	return popup->get("items");
}

void MenuButton::set_switch_on_hover(bool p_enabled) {
	switch_on_hover = p_enabled;
}

bool MenuButton::is_switch_on_hover() {
	return switch_on_hover;
}

void MenuButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible_in_tree()) {
				popup->hide();
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			Vector2i mouse_pos = DisplayServer::get_singleton()->mouse_get_position() - mouse_pos_adjusted;
			MenuButton *menu_btn_other = Object::cast_to<MenuButton>(get_viewport()->gui_find_control(mouse_pos));

			if (menu_btn_other && menu_btn_other != this && menu_btn_other->is_switch_on_hover() && !menu_btn_other->is_disabled() &&
					(get_parent()->is_ancestor_of(menu_btn_other) || menu_btn_other->get_parent()->is_ancestor_of(popup))) {
				popup->hide();
				menu_btn_other->pressed();
			}
		} break;
	}
}

void MenuButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_popup"), &MenuButton::get_popup);
	ClassDB::bind_method(D_METHOD("_set_items"), &MenuButton::_set_items);
	ClassDB::bind_method(D_METHOD("_get_items"), &MenuButton::_get_items);
	ClassDB::bind_method(D_METHOD("set_switch_on_hover", "enable"), &MenuButton::set_switch_on_hover);
	ClassDB::bind_method(D_METHOD("is_switch_on_hover"), &MenuButton::is_switch_on_hover);
	ClassDB::bind_method(D_METHOD("set_disable_shortcuts", "disabled"), &MenuButton::set_disable_shortcuts);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "items", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_items", "_get_items");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "switch_on_hover"), "set_switch_on_hover", "is_switch_on_hover");

	ADD_SIGNAL(MethodInfo("about_to_popup"));
}

void MenuButton::set_disable_shortcuts(bool p_disabled) {
	disable_shortcuts = p_disabled;
}

MenuButton::MenuButton() {
	set_flat(true);
	set_toggle_mode(true);
	set_disable_shortcuts(false);
	set_process_unhandled_key_input(true);
	set_focus_mode(FOCUS_NONE);
	set_action_mode(ACTION_MODE_BUTTON_PRESS);

	popup = memnew(PopupMenu);
	popup->hide();
	add_child(popup, false, INTERNAL_MODE_FRONT);
	popup->connect("about_to_popup", callable_mp(this, &MenuButton::_popup_visibility_changed), varray(true));
	popup->connect("popup_hide", callable_mp(this, &MenuButton::_popup_visibility_changed), varray(false));
}

MenuButton::~MenuButton() {
}
