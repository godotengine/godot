/**************************************************************************/
/*  editor_nav_tabs.cpp                                                   */
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

#include "editor_nav_tabs.h"

#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/themes/editor_scale.h"

Rect2i EditorNavTabs::prev_rect = Rect2i();
bool EditorNavTabs::was_showed = false;

void EditorNavTabs::popup_dialog(Vector<EditorTab *> p_tabs, bool p_next_tab) {
	if (was_showed) {
		popup(prev_rect);
	} else {
		popup_centered_clamped(Size2(600, 440) * EDSCALE, 0.8f);
	}

	Vector<EditorTab *> sort_list;
	sort_list.append_array(p_tabs);
	sort_list.sort_custom<EditorTab::EditorTabComparator>();
	_tabs = sort_list;

	_update_tab_list();

	if (tab_list->get_item_count() == 0) {
		tab_list->select(0);
	} else if (tab_list->get_item_count() > 1) {
		if (p_next_tab) {
			// Select the second to pass directly to the previous tab.
			tab_list->select(1);
		} else {
			tab_list->select(tab_list->get_item_count() - 1);
		}
	}

	// Always recroll at the beginning to see the first tabs on top.
	if (p_next_tab) {
		tab_list->get_v_scroll_bar()->scroll_to(0);
	} else {
		tab_list->get_v_scroll_bar()->scroll_to(tab_list->get_v_scroll_bar()->get_max());
	}
	tab_list->grab_focus();
}

void EditorNavTabs::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				prev_rect = Rect2i(get_position(), get_size());
				was_showed = true;
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			const int icon_size = get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor));
			tab_list->set_fixed_icon_size(Size2(icon_size, icon_size));
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			_close();
		} break;
	}
}

void EditorNavTabs::_update_tab_list() {
	tab_list->clear();

	int index = 0;
	for (const EditorTab *tab : _tabs) {
		tab_list->add_item(vformat("%s (%s)", tab->get_name(), tab->get_last_used()), tab->get_icon());
		tab_list->set_item_tooltip(index, tab->get_resource_path());
		tab_list->set_item_selectable(index, true);
		index++;
	}
}

void EditorNavTabs::_tab_list_clicked(int p_item, Vector2 p_local_mouse_pos, MouseButton p_mouse_button_index) {
	_select_tab(p_item);
}

void EditorNavTabs::_select_tab(int p_item) {
	EditorTab *tab = nullptr;
	if (p_item >= 0 && p_item < _tabs.size()) {
		tab = _tabs[p_item];
	}

	_close();

	if (tab) {
		this->emit_signal("selected", tab);
	}
}

void EditorNavTabs::_close() {
	tab_list->clear();
	_tabs.clear();
	hide();
}

void EditorNavTabs::shortcut_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> event_key = p_event;
	if (event_key.is_valid()) {
		if ((event_key->is_pressed() && !event_key->is_echo()) || Object::cast_to<InputEventShortcut>(*p_event)) {
			if (ED_IS_SHORTCUT("editor/next_tab", p_event)) {
				_select_next();
				get_tree()->get_root()->set_input_as_handled();
			}
			if (ED_IS_SHORTCUT("editor/prev_tab", p_event)) {
				_select_prev();
				get_tree()->get_root()->set_input_as_handled();
			}
		}

		if (p_event->is_released() && !event_key->is_command_or_control_pressed()) {
			_select_tab(tab_list->get_current());
		}
	}
}

void EditorNavTabs::_select_next() {
	if (tab_list->get_item_count() > 0) {
		int next_tab = tab_list->get_current() + 1;
		next_tab %= _tabs.size();
		tab_list->select(next_tab, true);
	}
}

void EditorNavTabs::_select_prev() {
	if (tab_list->get_item_count() > 0) {
		int next_tab = tab_list->get_current() - 1;
		next_tab = next_tab >= 0 ? next_tab : _tabs.size() - 1;
		tab_list->select(next_tab, true);
	}
}

void EditorNavTabs::_list_gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed()) {
		// Mouse wheel is grabbed by the EditorSceneTabs or by the ViewPort 2D/3D and not the ItemList
		// for un unknown reason. Need to manage the mouse wheel here.
		if (mb->get_button_index() == MouseButton::WHEEL_UP && mb->is_pressed()) {
			tab_list->get_v_scroll_bar()->set_value(tab_list->get_v_scroll_bar()->get_value() - tab_list->get_v_scroll_bar()->get_page() * mb->get_factor() / 8);
			set_input_as_handled();
		}
		if (mb->get_button_index() == MouseButton::WHEEL_DOWN && mb->is_pressed()) {
			tab_list->get_v_scroll_bar()->set_value(tab_list->get_v_scroll_bar()->get_value() + tab_list->get_v_scroll_bar()->get_page() * mb->get_factor() / 8);
			set_input_as_handled();
		}
	}
}

void EditorNavTabs::_bind_methods() {
	ADD_SIGNAL(MethodInfo("selected", PropertyInfo(Variant::OBJECT, "tab", PROPERTY_HINT_RESOURCE_TYPE, "EditorTab")));
}

EditorNavTabs::EditorNavTabs() {
	set_process_shortcut_input(true);

	this->get_ok_button()->hide();
	this->get_cancel_button()->hide();
	this->set_flag(FLAG_BORDERLESS, true);

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(vbc);

	tab_list = memnew(ItemList);
	tab_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tab_list->set_custom_minimum_size(Size2(150, 60) * EDSCALE); //need to give a bit of limit to avoid it from disappearing
	tab_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tab_list->connect("item_clicked", callable_mp(this, &EditorNavTabs::_tab_list_clicked), CONNECT_DEFERRED);
	tab_list->connect(SceneStringName(gui_input), callable_mp(this, &EditorNavTabs::_list_gui_input));
	vbc->add_child(tab_list);
}
