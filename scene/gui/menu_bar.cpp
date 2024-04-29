/**************************************************************************/
/*  menu_bar.cpp                                                          */
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

#include "menu_bar.h"

#include "core/os/keyboard.h"
#include "scene/main/window.h"
#include "scene/theme/theme_db.h"

void MenuBar::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());
	if (is_native_menu()) {
		// Handled by OS.
		return;
	}

	MutexLock lock(mutex);
	if (p_event->is_action("ui_left", true) && p_event->is_pressed()) {
		int new_sel = selected_menu;
		int old_sel = (selected_menu < 0) ? 0 : selected_menu;
		do {
			new_sel--;
			if (new_sel < 0) {
				new_sel = menu_cache.size() - 1;
			}
			if (old_sel == new_sel) {
				return;
			}
		} while (menu_cache[new_sel].hidden || menu_cache[new_sel].disabled);

		if (selected_menu != new_sel) {
			selected_menu = new_sel;
			focused_menu = selected_menu;
			if (active_menu >= 0) {
				get_menu_popup(active_menu)->hide();
			}
			_open_popup(selected_menu, true);
		}
		return;
	} else if (p_event->is_action("ui_right", true) && p_event->is_pressed()) {
		int new_sel = selected_menu;
		int old_sel = (selected_menu < 0) ? menu_cache.size() - 1 : selected_menu;
		do {
			new_sel++;
			if (new_sel >= menu_cache.size()) {
				new_sel = 0;
			}
			if (old_sel == new_sel) {
				return;
			}
		} while (menu_cache[new_sel].hidden || menu_cache[new_sel].disabled);

		if (selected_menu != new_sel) {
			selected_menu = new_sel;
			focused_menu = selected_menu;
			if (active_menu >= 0) {
				get_menu_popup(active_menu)->hide();
			}
			_open_popup(selected_menu, true);
		}
		return;
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		int old_sel = selected_menu;
		focused_menu = _get_index_at_point(mm->get_position());
		if (focused_menu >= 0) {
			selected_menu = focused_menu;
		}
		if (selected_menu != old_sel) {
			queue_redraw();
		}
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->is_pressed() && (mb->get_button_index() == MouseButton::LEFT || mb->get_button_index() == MouseButton::RIGHT)) {
			int index = _get_index_at_point(mb->get_position());
			if (index >= 0) {
				_open_popup(index);
			}
		}
	}
}

void MenuBar::_open_popup(int p_index, bool p_focus_item) {
	ERR_FAIL_INDEX(p_index, menu_cache.size());

	PopupMenu *pm = get_menu_popup(p_index);
	if (pm->is_visible()) {
		pm->hide();
		return;
	}

	Rect2 item_rect = _get_menu_item_rect(p_index);
	Point2 screen_pos = get_screen_position() + item_rect.position * get_viewport()->get_canvas_transform().get_scale();
	Size2 screen_size = item_rect.size * get_viewport()->get_canvas_transform().get_scale();

	active_menu = p_index;

	pm->set_size(Size2(screen_size.x, 0));
	screen_pos.y += screen_size.y;
	if (is_layout_rtl()) {
		screen_pos.x += screen_size.x - pm->get_size().width;
	}
	pm->set_position(screen_pos);
	pm->popup();

	if (p_focus_item) {
		for (int i = 0; i < pm->get_item_count(); i++) {
			if (!pm->is_item_disabled(i)) {
				pm->set_focused_item(i);
				break;
			}
		}
	}

	queue_redraw();
}

void MenuBar::shortcut_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (disable_shortcuts) {
		return;
	}

	if (p_event->is_pressed() && (Object::cast_to<InputEventKey>(p_event.ptr()) || Object::cast_to<InputEventJoypadButton>(p_event.ptr()) || Object::cast_to<InputEventAction>(*p_event) || Object::cast_to<InputEventShortcut>(*p_event))) {
		if (!get_parent() || !is_visible_in_tree()) {
			return;
		}

		Vector<PopupMenu *> popups = _get_popups();
		for (int i = 0; i < popups.size(); i++) {
			if (menu_cache[i].hidden || menu_cache[i].disabled) {
				continue;
			}
			if (popups[i]->activate_item_by_event(p_event, false)) {
				accept_event();
				return;
			}
		}
	}
}

void MenuBar::_popup_visibility_changed(bool p_visible) {
	if (!p_visible) {
		active_menu = -1;
		focused_menu = -1;
		set_process_internal(false);
		queue_redraw();
		return;
	}

	if (switch_on_hover) {
		Window *wnd = Object::cast_to<Window>(get_viewport());
		if (wnd) {
			mouse_pos_adjusted = wnd->get_position();

			if (wnd->is_embedded()) {
				Window *wnd_parent = Object::cast_to<Window>(wnd->get_parent()->get_viewport());
				while (wnd_parent) {
					if (!wnd_parent->is_embedded()) {
						mouse_pos_adjusted += wnd_parent->get_position();
						break;
					}

					wnd_parent = Object::cast_to<Window>(wnd_parent->get_parent()->get_viewport());
				}
			}

			set_process_internal(true);
		}
	}
}

void MenuBar::_update_submenu(const String &p_menu_name, PopupMenu *p_child) {
	int count = p_child->get_item_count();
	global_menus.insert(p_menu_name);
	for (int i = 0; i < count; i++) {
		if (p_child->is_item_separator(i)) {
			DisplayServer::get_singleton()->global_menu_add_separator(p_menu_name);
		} else if (!p_child->get_item_submenu(i).is_empty()) {
			Node *n = p_child->get_node_or_null(p_child->get_item_submenu(i));
			ERR_FAIL_NULL_MSG(n, "Item subnode does not exist: '" + p_child->get_item_submenu(i) + "'.");
			PopupMenu *pm = Object::cast_to<PopupMenu>(n);
			ERR_FAIL_NULL_MSG(pm, "Item subnode is not a PopupMenu: '" + p_child->get_item_submenu(i) + "'.");

			DisplayServer::get_singleton()->global_menu_add_submenu_item(p_menu_name, atr(p_child->get_item_text(i)), p_menu_name + "/" + itos(i));
			_update_submenu(p_menu_name + "/" + itos(i), pm);
		} else {
			int index = DisplayServer::get_singleton()->global_menu_add_item(p_menu_name, atr(p_child->get_item_text(i)), callable_mp(p_child, &PopupMenu::activate_item), Callable(), i);

			if (p_child->is_item_checkable(i)) {
				DisplayServer::get_singleton()->global_menu_set_item_checkable(p_menu_name, index, true);
			}
			if (p_child->is_item_radio_checkable(i)) {
				DisplayServer::get_singleton()->global_menu_set_item_radio_checkable(p_menu_name, index, true);
			}
			DisplayServer::get_singleton()->global_menu_set_item_checked(p_menu_name, index, p_child->is_item_checked(i));
			DisplayServer::get_singleton()->global_menu_set_item_disabled(p_menu_name, index, p_child->is_item_disabled(i));
			DisplayServer::get_singleton()->global_menu_set_item_max_states(p_menu_name, index, p_child->get_item_max_states(i));
			DisplayServer::get_singleton()->global_menu_set_item_icon(p_menu_name, index, p_child->get_item_icon(i));
			DisplayServer::get_singleton()->global_menu_set_item_state(p_menu_name, index, p_child->get_item_state(i));
			DisplayServer::get_singleton()->global_menu_set_item_indentation_level(p_menu_name, index, p_child->get_item_indent(i));
			DisplayServer::get_singleton()->global_menu_set_item_tooltip(p_menu_name, index, p_child->get_item_tooltip(i));
			if (!p_child->is_item_shortcut_disabled(i) && p_child->get_item_shortcut(i).is_valid() && p_child->get_item_shortcut(i)->has_valid_event()) {
				Array events = p_child->get_item_shortcut(i)->get_events();
				for (int j = 0; j < events.size(); j++) {
					Ref<InputEventKey> ie = events[j];
					if (ie.is_valid()) {
						DisplayServer::get_singleton()->global_menu_set_item_accelerator(p_menu_name, index, ie->get_keycode_with_modifiers());
						break;
					}
				}
			} else if (p_child->get_item_accelerator(i) != Key::NONE) {
				DisplayServer::get_singleton()->global_menu_set_item_accelerator(p_menu_name, i, p_child->get_item_accelerator(i));
			}
		}
	}
}

bool MenuBar::is_native_menu() const {
#ifdef TOOLS_ENABLED
	if (is_part_of_edited_scene()) {
		return false;
	}
#endif

	return (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_GLOBAL_MENU) && is_native);
}

void MenuBar::_clear_menu() {
	if (!DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_GLOBAL_MENU)) {
		return;
	}

	// Remove root menu items.
	int count = DisplayServer::get_singleton()->global_menu_get_item_count("_main");
	for (int i = count - 1; i >= 0; i--) {
		if (global_menus.has(DisplayServer::get_singleton()->global_menu_get_item_submenu("_main", i))) {
			DisplayServer::get_singleton()->global_menu_remove_item("_main", i);
		}
	}
	// Erase submenu contents.
	for (const String &E : global_menus) {
		DisplayServer::get_singleton()->global_menu_clear(E);
	}
	global_menus.clear();
}

void MenuBar::_update_menu() {
	_clear_menu();

	if (!is_visible_in_tree()) {
		return;
	}

	int index = start_index;
	if (is_native_menu()) {
		Vector<PopupMenu *> popups = _get_popups();
		String root_name = "MenuBar<" + String::num_int64((uint64_t)this, 16) + ">";
		for (int i = 0; i < popups.size(); i++) {
			if (menu_cache[i].hidden) {
				continue;
			}
			String menu_name = atr(String(popups[i]->get_meta("_menu_name", popups[i]->get_name())));

			index = DisplayServer::get_singleton()->global_menu_add_submenu_item("_main", menu_name, root_name + "/" + itos(i), index);
			if (menu_cache[i].disabled) {
				DisplayServer::get_singleton()->global_menu_set_item_disabled("_main", index, true);
			}
			_update_submenu(root_name + "/" + itos(i), popups[i]);
			index++;
		}
	}
	update_minimum_size();
	queue_redraw();
}

void MenuBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (get_menu_count() > 0) {
				_refresh_menu_names();
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			_clear_menu();
		} break;
		case NOTIFICATION_MOUSE_EXIT: {
			focused_menu = -1;
			selected_menu = -1;
			queue_redraw();
		} break;
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			for (int i = 0; i < menu_cache.size(); i++) {
				shape(menu_cache.write[i]);
			}
			_update_menu();
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_menu();
		} break;
		case NOTIFICATION_DRAW: {
			if (is_native_menu()) {
				return;
			}
			for (int i = 0; i < menu_cache.size(); i++) {
				_draw_menu_item(i);
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			MutexLock lock(mutex);

			if (is_native_menu()) {
				// Handled by OS.
				return;
			}

			Vector2 pos = DisplayServer::get_singleton()->mouse_get_position() - mouse_pos_adjusted - get_global_position();
			if (pos == old_mouse_pos) {
				return;
			}
			old_mouse_pos = pos;

			int index = _get_index_at_point(pos);
			if (index >= 0 && index != active_menu) {
				selected_menu = index;
				focused_menu = selected_menu;
				if (active_menu >= 0) {
					get_menu_popup(active_menu)->hide();
				}
				_open_popup(index);
			}
		} break;
	}
}

int MenuBar::_get_index_at_point(const Point2 &p_point) const {
	Ref<StyleBox> style = theme_cache.normal;
	int offset = 0;
	Point2 point = p_point;
	if (is_layout_rtl()) {
		point.x = get_size().x - point.x;
	}

	for (int i = 0; i < menu_cache.size(); i++) {
		if (menu_cache[i].hidden) {
			continue;
		}
		Size2 size = menu_cache[i].text_buf->get_size() + style->get_minimum_size();
		if (point.x > offset && point.x < offset + size.x) {
			if (point.y > 0 && point.y < size.y) {
				return i;
			}
		}
		offset += size.x + theme_cache.h_separation;
	}
	return -1;
}

Rect2 MenuBar::_get_menu_item_rect(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, menu_cache.size(), Rect2());

	Ref<StyleBox> style = theme_cache.normal;

	int offset = 0;
	for (int i = 0; i < p_index; i++) {
		if (menu_cache[i].hidden) {
			continue;
		}
		Size2 size = menu_cache[i].text_buf->get_size() + style->get_minimum_size();
		offset += size.x + theme_cache.h_separation;
	}

	Size2 size = menu_cache[p_index].text_buf->get_size() + style->get_minimum_size();
	if (is_layout_rtl()) {
		return Rect2(Point2(get_size().x - offset - size.x, 0), size);
	} else {
		return Rect2(Point2(offset, 0), size);
	}
}

void MenuBar::_draw_menu_item(int p_index) {
	ERR_FAIL_INDEX(p_index, menu_cache.size());

	RID ci = get_canvas_item();
	bool hovered = (focused_menu == p_index);
	bool pressed = (active_menu == p_index);
	bool rtl = is_layout_rtl();

	if (menu_cache[p_index].hidden) {
		return;
	}

	Color color;
	Ref<StyleBox> style;
	Rect2 item_rect = _get_menu_item_rect(p_index);

	if (menu_cache[p_index].disabled) {
		if (rtl && has_theme_stylebox(SNAME("disabled_mirrored"))) {
			style = theme_cache.disabled_mirrored;
		} else {
			style = theme_cache.disabled;
		}
		if (!flat) {
			style->draw(ci, item_rect);
		}
		color = theme_cache.font_disabled_color;
	} else if (hovered && pressed && has_theme_stylebox("hover_pressed")) {
		if (rtl && has_theme_stylebox(SNAME("hover_pressed_mirrored"))) {
			style = theme_cache.hover_pressed_mirrored;
		} else {
			style = theme_cache.hover_pressed;
		}
		if (!flat) {
			style->draw(ci, item_rect);
		}
		if (has_theme_color(SNAME("font_hover_pressed_color"))) {
			color = theme_cache.font_hover_pressed_color;
		}
	} else if (pressed) {
		if (rtl && has_theme_stylebox(SNAME("pressed_mirrored"))) {
			style = theme_cache.pressed_mirrored;
		} else {
			style = theme_cache.pressed;
		}
		if (!flat) {
			style->draw(ci, item_rect);
		}
		if (has_theme_color(SNAME("font_pressed_color"))) {
			color = theme_cache.font_pressed_color;
		} else {
			color = theme_cache.font_color;
		}
	} else if (hovered) {
		if (rtl && has_theme_stylebox(SNAME("hover_mirrored"))) {
			style = theme_cache.hover_mirrored;
		} else {
			style = theme_cache.hover;
		}
		if (!flat) {
			style->draw(ci, item_rect);
		}
		color = theme_cache.font_hover_color;
	} else {
		if (rtl && has_theme_stylebox(SNAME("normal_mirrored"))) {
			style = theme_cache.normal_mirrored;
		} else {
			style = theme_cache.normal;
		}
		if (!flat) {
			style->draw(ci, item_rect);
		}
		// Focus colors only take precedence over normal state.
		if (has_focus()) {
			color = theme_cache.font_focus_color;
		} else {
			color = theme_cache.font_color;
		}
	}

	Point2 text_ofs = item_rect.position + Point2(style->get_margin(SIDE_LEFT), style->get_margin(SIDE_TOP));

	Color font_outline_color = theme_cache.font_outline_color;
	int outline_size = theme_cache.outline_size;
	if (outline_size > 0 && font_outline_color.a > 0) {
		menu_cache[p_index].text_buf->draw_outline(ci, text_ofs, outline_size, font_outline_color);
	}
	menu_cache[p_index].text_buf->draw(ci, text_ofs, color);
}

void MenuBar::shape(Menu &p_menu) {
	p_menu.text_buf->clear();
	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		p_menu.text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		p_menu.text_buf->set_direction((TextServer::Direction)text_direction);
	}
	p_menu.text_buf->add_string(atr(p_menu.name), theme_cache.font, theme_cache.font_size, language);
}

void MenuBar::_refresh_menu_names() {
	Vector<PopupMenu *> popups = _get_popups();
	for (int i = 0; i < popups.size(); i++) {
		if (!popups[i]->has_meta("_menu_name") && String(popups[i]->get_name()) != get_menu_title(i)) {
			menu_cache.write[i].name = popups[i]->get_name();
			shape(menu_cache.write[i]);
		}
	}
	_update_menu();
}

Vector<PopupMenu *> MenuBar::_get_popups() const {
	Vector<PopupMenu *> popups;
	for (int i = 0; i < get_child_count(); i++) {
		PopupMenu *pm = Object::cast_to<PopupMenu>(get_child(i));
		if (!pm) {
			continue;
		}
		popups.push_back(pm);
	}
	return popups;
}

int MenuBar::get_menu_idx_from_control(PopupMenu *p_child) const {
	ERR_FAIL_NULL_V(p_child, -1);
	ERR_FAIL_COND_V(p_child->get_parent() != this, -1);

	Vector<PopupMenu *> popups = _get_popups();
	for (int i = 0; i < popups.size(); i++) {
		if (popups[i] == p_child) {
			return i;
		}
	}

	return -1;
}

void MenuBar::add_child_notify(Node *p_child) {
	Control::add_child_notify(p_child);

	PopupMenu *pm = Object::cast_to<PopupMenu>(p_child);
	if (!pm) {
		return;
	}
	Menu menu = Menu(p_child->get_name());
	shape(menu);

	menu_cache.push_back(menu);
	p_child->connect("renamed", callable_mp(this, &MenuBar::_refresh_menu_names));
	p_child->connect("menu_changed", callable_mp(this, &MenuBar::_update_menu));
	p_child->connect("about_to_popup", callable_mp(this, &MenuBar::_popup_visibility_changed).bind(true));
	p_child->connect("popup_hide", callable_mp(this, &MenuBar::_popup_visibility_changed).bind(false));

	_update_menu();
}

void MenuBar::move_child_notify(Node *p_child) {
	Control::move_child_notify(p_child);

	PopupMenu *pm = Object::cast_to<PopupMenu>(p_child);
	if (!pm) {
		return;
	}

	int old_idx = -1;
	String menu_name = String(pm->get_meta("_menu_name", pm->get_name()));
	// Find the previous menu index of the control.
	for (int i = 0; i < get_menu_count(); i++) {
		if (get_menu_title(i) == menu_name) {
			old_idx = i;
			break;
		}
	}
	Menu menu = menu_cache[old_idx];
	menu_cache.remove_at(old_idx);
	menu_cache.insert(get_menu_idx_from_control(pm), menu);

	_update_menu();
}

void MenuBar::remove_child_notify(Node *p_child) {
	Control::remove_child_notify(p_child);

	PopupMenu *pm = Object::cast_to<PopupMenu>(p_child);
	if (!pm) {
		return;
	}

	int idx = get_menu_idx_from_control(pm);

	menu_cache.remove_at(idx);

	p_child->remove_meta("_menu_name");
	p_child->remove_meta("_menu_tooltip");

	p_child->disconnect("renamed", callable_mp(this, &MenuBar::_refresh_menu_names));
	p_child->disconnect("menu_changed", callable_mp(this, &MenuBar::_update_menu));
	p_child->disconnect("about_to_popup", callable_mp(this, &MenuBar::_popup_visibility_changed));
	p_child->disconnect("popup_hide", callable_mp(this, &MenuBar::_popup_visibility_changed));

	_update_menu();
}

void MenuBar::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_switch_on_hover", "enable"), &MenuBar::set_switch_on_hover);
	ClassDB::bind_method(D_METHOD("is_switch_on_hover"), &MenuBar::is_switch_on_hover);
	ClassDB::bind_method(D_METHOD("set_disable_shortcuts", "disabled"), &MenuBar::set_disable_shortcuts);

	ClassDB::bind_method(D_METHOD("set_prefer_global_menu", "enabled"), &MenuBar::set_prefer_global_menu);
	ClassDB::bind_method(D_METHOD("is_prefer_global_menu"), &MenuBar::is_prefer_global_menu);
	ClassDB::bind_method(D_METHOD("is_native_menu"), &MenuBar::is_native_menu);

	ClassDB::bind_method(D_METHOD("get_menu_count"), &MenuBar::get_menu_count);

	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &MenuBar::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &MenuBar::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &MenuBar::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &MenuBar::get_language);
	ClassDB::bind_method(D_METHOD("set_flat", "enabled"), &MenuBar::set_flat);
	ClassDB::bind_method(D_METHOD("is_flat"), &MenuBar::is_flat);
	ClassDB::bind_method(D_METHOD("set_start_index", "enabled"), &MenuBar::set_start_index);
	ClassDB::bind_method(D_METHOD("get_start_index"), &MenuBar::get_start_index);

	ClassDB::bind_method(D_METHOD("set_menu_title", "menu", "title"), &MenuBar::set_menu_title);
	ClassDB::bind_method(D_METHOD("get_menu_title", "menu"), &MenuBar::get_menu_title);

	ClassDB::bind_method(D_METHOD("set_menu_tooltip", "menu", "tooltip"), &MenuBar::set_menu_tooltip);
	ClassDB::bind_method(D_METHOD("get_menu_tooltip", "menu"), &MenuBar::get_menu_tooltip);

	ClassDB::bind_method(D_METHOD("set_menu_disabled", "menu", "disabled"), &MenuBar::set_menu_disabled);
	ClassDB::bind_method(D_METHOD("is_menu_disabled", "menu"), &MenuBar::is_menu_disabled);

	ClassDB::bind_method(D_METHOD("set_menu_hidden", "menu", "hidden"), &MenuBar::set_menu_hidden);
	ClassDB::bind_method(D_METHOD("is_menu_hidden", "menu"), &MenuBar::is_menu_hidden);

	ClassDB::bind_method(D_METHOD("get_menu_popup", "menu"), &MenuBar::get_menu_popup);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flat"), "set_flat", "is_flat");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "start_index"), "set_start_index", "get_start_index");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "switch_on_hover"), "set_switch_on_hover", "is_switch_on_hover");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "prefer_global_menu"), "set_prefer_global_menu", "is_prefer_global_menu");

	ADD_GROUP("BiDi", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID, ""), "set_language", "get_language");

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, MenuBar, normal);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, MenuBar, normal_mirrored);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, MenuBar, disabled);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, MenuBar, disabled_mirrored);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, MenuBar, pressed);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, MenuBar, pressed_mirrored);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, MenuBar, hover);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, MenuBar, hover_mirrored);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, MenuBar, hover_pressed);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, MenuBar, hover_pressed_mirrored);

	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT, MenuBar, font);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT_SIZE, MenuBar, font_size);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, MenuBar, outline_size);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, MenuBar, font_outline_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, MenuBar, font_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, MenuBar, font_disabled_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, MenuBar, font_pressed_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, MenuBar, font_hover_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, MenuBar, font_hover_pressed_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, MenuBar, font_focus_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, MenuBar, h_separation);
}

void MenuBar::set_switch_on_hover(bool p_enabled) {
	switch_on_hover = p_enabled;
}

bool MenuBar::is_switch_on_hover() {
	return switch_on_hover;
}

void MenuBar::set_disable_shortcuts(bool p_disabled) {
	disable_shortcuts = p_disabled;
}

void MenuBar::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		_update_menu();
	}
}

Control::TextDirection MenuBar::get_text_direction() const {
	return text_direction;
}

void MenuBar::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		_update_menu();
	}
}

String MenuBar::get_language() const {
	return language;
}

void MenuBar::set_flat(bool p_enabled) {
	if (flat != p_enabled) {
		flat = p_enabled;
		queue_redraw();
	}
}

bool MenuBar::is_flat() const {
	return flat;
}

void MenuBar::set_start_index(int p_index) {
	if (start_index != p_index) {
		start_index = p_index;
		_update_menu();
	}
}

int MenuBar::get_start_index() const {
	return start_index;
}

void MenuBar::set_prefer_global_menu(bool p_enabled) {
	if (is_native != p_enabled) {
		if (is_native) {
			_clear_menu();
		}
		is_native = p_enabled;
		_update_menu();
	}
}

bool MenuBar::is_prefer_global_menu() const {
	return is_native;
}

Size2 MenuBar::get_minimum_size() const {
	if (is_native_menu()) {
		return Size2();
	}

	Ref<StyleBox> style = theme_cache.normal;

	Vector2 size;
	for (int i = 0; i < menu_cache.size(); i++) {
		if (menu_cache[i].hidden) {
			continue;
		}
		Size2 sz = menu_cache[i].text_buf->get_size() + style->get_minimum_size();
		size.y = MAX(size.y, sz.y);
		size.x += sz.x;
	}
	if (menu_cache.size() > 1) {
		size.x += theme_cache.h_separation * (menu_cache.size() - 1);
	}
	return size;
}

int MenuBar::get_menu_count() const {
	return menu_cache.size();
}

void MenuBar::set_menu_title(int p_menu, const String &p_title) {
	ERR_FAIL_INDEX(p_menu, menu_cache.size());
	PopupMenu *pm = get_menu_popup(p_menu);
	if (p_title == pm->get_name()) {
		pm->remove_meta("_menu_name");
	} else {
		pm->set_meta("_menu_name", p_title);
	}
	menu_cache.write[p_menu].name = p_title;
	shape(menu_cache.write[p_menu]);
	_update_menu();
}

String MenuBar::get_menu_title(int p_menu) const {
	ERR_FAIL_INDEX_V(p_menu, menu_cache.size(), String());
	return menu_cache[p_menu].name;
}

void MenuBar::set_menu_tooltip(int p_menu, const String &p_tooltip) {
	ERR_FAIL_INDEX(p_menu, menu_cache.size());
	PopupMenu *pm = get_menu_popup(p_menu);
	pm->set_meta("_menu_tooltip", p_tooltip);
	menu_cache.write[p_menu].name = p_tooltip;
}

String MenuBar::get_menu_tooltip(int p_menu) const {
	ERR_FAIL_INDEX_V(p_menu, menu_cache.size(), String());
	return menu_cache[p_menu].tooltip;
}

void MenuBar::set_menu_disabled(int p_menu, bool p_disabled) {
	ERR_FAIL_INDEX(p_menu, menu_cache.size());
	menu_cache.write[p_menu].disabled = p_disabled;
	_update_menu();
}

bool MenuBar::is_menu_disabled(int p_menu) const {
	ERR_FAIL_INDEX_V(p_menu, menu_cache.size(), false);
	return menu_cache[p_menu].disabled;
}

void MenuBar::set_menu_hidden(int p_menu, bool p_hidden) {
	ERR_FAIL_INDEX(p_menu, menu_cache.size());
	menu_cache.write[p_menu].hidden = p_hidden;
	_update_menu();
}

bool MenuBar::is_menu_hidden(int p_menu) const {
	ERR_FAIL_INDEX_V(p_menu, menu_cache.size(), false);
	return menu_cache[p_menu].hidden;
}

PopupMenu *MenuBar::get_menu_popup(int p_idx) const {
	Vector<PopupMenu *> controls = _get_popups();
	if (p_idx >= 0 && p_idx < controls.size()) {
		return controls[p_idx];
	} else {
		return nullptr;
	}
}

String MenuBar::get_tooltip(const Point2 &p_pos) const {
	int index = _get_index_at_point(p_pos);
	if (index >= 0 && index < menu_cache.size()) {
		return menu_cache[index].tooltip;
	} else {
		return String();
	}
}

MenuBar::MenuBar() {
	set_process_shortcut_input(true);
}

MenuBar::~MenuBar() {
}
