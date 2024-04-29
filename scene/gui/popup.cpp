/**************************************************************************/
/*  popup.cpp                                                             */
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

#include "popup.h"

#include "core/config/engine.h"
#include "core/os/keyboard.h"
#include "scene/gui/panel.h"
#include "scene/theme/theme_db.h"

void Popup::_input_from_window(const Ref<InputEvent> &p_event) {
	if (get_flag(FLAG_POPUP) && p_event->is_action_pressed(SNAME("ui_cancel"), false, true)) {
		_close_pressed();
	}
}

void Popup::_initialize_visible_parents() {
	if (is_embedded()) {
		visible_parents.clear();

		Window *parent_window = this;
		while (parent_window) {
			parent_window = parent_window->get_parent_visible_window();
			if (parent_window) {
				visible_parents.push_back(parent_window);
				parent_window->connect("focus_entered", callable_mp(this, &Popup::_parent_focused));
				parent_window->connect("tree_exited", callable_mp(this, &Popup::_deinitialize_visible_parents));
			}
		}
	}
}

void Popup::_deinitialize_visible_parents() {
	if (is_embedded()) {
		for (Window *parent_window : visible_parents) {
			parent_window->disconnect("focus_entered", callable_mp(this, &Popup::_parent_focused));
			parent_window->disconnect("tree_exited", callable_mp(this, &Popup::_deinitialize_visible_parents));
		}

		visible_parents.clear();
	}
}

void Popup::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_in_edited_scene_root()) {
				if (is_visible()) {
					_initialize_visible_parents();
				} else {
					_deinitialize_visible_parents();
					emit_signal(SNAME("popup_hide"));
					popped_up = false;
				}
			}
		} break;

		case NOTIFICATION_WM_WINDOW_FOCUS_IN: {
			if (!is_in_edited_scene_root()) {
				if (has_focus()) {
					popped_up = true;
				}
			}
		} break;

		case NOTIFICATION_UNPARENTED:
		case NOTIFICATION_EXIT_TREE: {
			if (!is_in_edited_scene_root()) {
				_deinitialize_visible_parents();
			}
		} break;

		case NOTIFICATION_WM_CLOSE_REQUEST: {
			if (!is_in_edited_scene_root()) {
				_close_pressed();
			}
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			if (!is_in_edited_scene_root() && get_flag(FLAG_POPUP)) {
				_close_pressed();
			}
		} break;
	}
}

void Popup::_parent_focused() {
	if (popped_up && get_flag(FLAG_POPUP)) {
		_close_pressed();
	}
}

void Popup::_close_pressed() {
	popped_up = false;

	_deinitialize_visible_parents();

	call_deferred(SNAME("hide"));
}

void Popup::_post_popup() {
	Window::_post_popup();
	popped_up = true;
}

void Popup::_validate_property(PropertyInfo &p_property) const {
	if (
			p_property.name == "transient" ||
			p_property.name == "exclusive" ||
			p_property.name == "popup_window" ||
			p_property.name == "unfocusable") {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

Rect2i Popup::_popup_adjust_rect() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Rect2());
	Rect2i parent_rect = get_usable_parent_rect();

	if (parent_rect == Rect2i()) {
		return Rect2i();
	}

	Rect2i current(get_position(), get_size());

	if (current.position.x + current.size.x > parent_rect.position.x + parent_rect.size.x) {
		current.position.x = parent_rect.position.x + parent_rect.size.x - current.size.x;
	}

	if (current.position.x < parent_rect.position.x) {
		current.position.x = parent_rect.position.x;
	}

	if (current.position.y + current.size.y > parent_rect.position.y + parent_rect.size.y) {
		current.position.y = parent_rect.position.y + parent_rect.size.y - current.size.y;
	}

	if (current.position.y < parent_rect.position.y) {
		current.position.y = parent_rect.position.y;
	}

	if (current.size.y > parent_rect.size.y) {
		current.size.y = parent_rect.size.y;
	}

	if (current.size.x > parent_rect.size.x) {
		current.size.x = parent_rect.size.x;
	}

	// Early out if max size not set.
	Size2i popup_max_size = get_max_size();
	if (popup_max_size <= Size2()) {
		return current;
	}

	if (current.size.x > popup_max_size.x) {
		current.size.x = popup_max_size.x;
	}

	if (current.size.y > popup_max_size.y) {
		current.size.y = popup_max_size.y;
	}

	return current;
}

void Popup::_bind_methods() {
	ADD_SIGNAL(MethodInfo("popup_hide"));

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, Popup, panel_style, "panel");
}

Popup::Popup() {
	set_wrap_controls(true);
	set_visible(false);
	set_transient(true);
	set_flag(FLAG_BORDERLESS, true);
	set_flag(FLAG_RESIZE_DISABLED, true);
	set_flag(FLAG_POPUP, true);

	connect("window_input", callable_mp(this, &Popup::_input_from_window));
}

Popup::~Popup() {
}

Size2 PopupPanel::_get_contents_minimum_size() const {
	Size2 ms;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c || c == panel) {
			continue;
		}

		if (c->is_set_as_top_level()) {
			continue;
		}

		Size2 cms = c->get_combined_minimum_size();
		ms.x = MAX(cms.x, ms.x);
		ms.y = MAX(cms.y, ms.y);
	}

	return ms + theme_cache.panel_style->get_minimum_size();
}

void PopupPanel::_update_child_rects() {
	Vector2 cpos(theme_cache.panel_style->get_offset());
	Vector2 csize(get_size() - theme_cache.panel_style->get_minimum_size());

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}

		if (c->is_set_as_top_level()) {
			continue;
		}

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
	switch (p_what) {
		case NOTIFICATION_READY:
		case NOTIFICATION_THEME_CHANGED: {
			panel->add_theme_style_override("panel", theme_cache.panel_style);
			_update_child_rects();
		} break;

		case NOTIFICATION_WM_SIZE_CHANGED: {
			_update_child_rects();
		} break;
	}
}

void PopupPanel::_bind_methods() {
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, PopupPanel, panel_style, "panel");
}

PopupPanel::PopupPanel() {
	panel = memnew(Panel);
	add_child(panel, false, INTERNAL_MODE_FRONT);
}
