/*************************************************************************/
/*  popup.cpp                                                            */
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

#include "popup.h"

#include "core/config/engine.h"
#include "core/os/keyboard.h"
#include "scene/gui/panel.h"

void Popup::_input_from_window(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> key = p_event;
	if (key.is_valid() && key->is_pressed() && key->get_keycode() == Key::ESCAPE) {
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
		for (uint32_t i = 0; i < visible_parents.size(); ++i) {
			visible_parents[i]->disconnect("focus_entered", callable_mp(this, &Popup::_parent_focused));
			visible_parents[i]->disconnect("tree_exited", callable_mp(this, &Popup::_deinitialize_visible_parents));
		}

		visible_parents.clear();
	}
}

void Popup::_update_theme_item_cache() {
	Window::_update_theme_item_cache();

	theme_cache.panel_style = get_theme_stylebox(SNAME("panel"));
}

void Popup::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				_initialize_visible_parents();
			} else {
				_deinitialize_visible_parents();
				emit_signal(SNAME("popup_hide"));
				popped_up = false;
			}
		} break;

		case NOTIFICATION_WM_WINDOW_FOCUS_IN: {
			if (has_focus()) {
				popped_up = true;
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_deinitialize_visible_parents();
		} break;

		case NOTIFICATION_WM_CLOSE_REQUEST:
		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			_close_pressed();
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

void Popup::_bind_methods() {
	ADD_SIGNAL(MethodInfo("popup_hide"));
}

Rect2i Popup::_popup_adjust_rect() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Rect2());
	Rect2i parent = get_usable_parent_rect();

	if (parent == Rect2i()) {
		return Rect2i();
	}

	Rect2i current(get_position(), get_size());

	if (current.position.x + current.size.x > parent.position.x + parent.size.x) {
		current.position.x = parent.position.x + parent.size.x - current.size.x;
	}

	if (current.position.x < parent.position.x) {
		current.position.x = parent.position.x;
	}

	if (current.position.y + current.size.y > parent.position.y + parent.size.y) {
		current.position.y = parent.position.y + parent.size.y - current.size.y;
	}

	if (current.position.y < parent.position.y) {
		current.position.y = parent.position.y;
	}

	if (current.size.y > parent.size.y) {
		current.size.y = parent.size.y;
	}

	if (current.size.x > parent.size.x) {
		current.size.x = parent.size.x;
	}

	// Early out if max size not set.
	Size2i max_size = get_max_size();
	if (max_size <= Size2()) {
		return current;
	}

	if (current.size.x > max_size.x) {
		current.size.x = max_size.x;
	}

	if (current.size.y > max_size.y) {
		current.size.y = max_size.y;
	}

	return current;
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

void PopupPanel::_update_theme_item_cache() {
	Popup::_update_theme_item_cache();

	theme_cache.panel_style = get_theme_stylebox(SNAME("panel"));
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

PopupPanel::PopupPanel() {
	panel = memnew(Panel);
	add_child(panel, false, INTERNAL_MODE_FRONT);
}
