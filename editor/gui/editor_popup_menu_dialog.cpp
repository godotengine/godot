/**************************************************************************/
/*  editor_popup_menu_dialog.cpp                                          */
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

#include "editor_popup_menu_dialog.h"

#include "core/string/fuzzy_search.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/item_list.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"

EditorPopupMenuDialog::EditorPopupMenuDialog() {
	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->add_theme_constant_override("separation", 0);
	add_child(vbc);

	{
		// Search bar
		MarginContainer *mc = memnew(MarginContainer);
		mc->add_theme_constant_override("margin_top", 6);
		mc->add_theme_constant_override("margin_bottom", 6);
		mc->add_theme_constant_override("margin_left", 1);
		mc->add_theme_constant_override("margin_right", 1);
		vbc->add_child(mc);

		search_box = memnew(LineEdit);
		search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		search_box->set_clear_button_enabled(true);
		mc->add_child(search_box);
	}

	{
		item_list = memnew(ItemList);
		item_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		item_list->connect("item_activated", callable_mp(this, &EditorPopupMenuDialog::handle_item_activated));
		vbc->add_child(item_list);
	}

	search_box->connect(SceneStringName(text_changed), callable_mp(this, &EditorPopupMenuDialog::_search_box_text_changed));
	search_box->connect(SceneStringName(gui_input), callable_mp(this, &EditorPopupMenuDialog::handle_search_box_input));
	register_text_enter(search_box);
	get_ok_button()->hide();
}

void EditorPopupMenuDialog::popup_dialog(PopupMenu *p_popup_menu, const String &p_title, const String &p_search_placeholder) {
	ERR_FAIL_COND(p_popup_menu->get_item_count() == 0);

	popup_menu = p_popup_menu;

	int item_count = p_popup_menu->get_item_count();
	for (int i = 0; i < item_count; i++) {
		String text = p_popup_menu->get_item_text(i);

		if (p_popup_menu->is_item_separator(i)) {
			continue;
		}

		PopupMenuItem item;
		item.text = text;
		item.icon = p_popup_menu->get_item_icon(i);
		item.popup_menu_index = i;

		items.push_back(item);
	}

	item_visible_status.resize(items.size());

	_search_box_text_changed("");
	get_ok_button()->set_disabled(true);

	set_title(p_title);
	search_box->set_placeholder(p_search_placeholder);

	popup_centered_clamped(Size2(300, 650) * EDSCALE, 0.8f);
	search_box->grab_focus();
	item_list->get_v_scroll_bar()->set_value(0);
}

void EditorPopupMenuDialog::ok_pressed() {
	PackedInt32Array selected_items = item_list->get_selected_items();
	ERR_FAIL_COND(selected_items.size() != 1);

	int absolute_index = item_list->get_item_metadata(selected_items[0]);
	if (absolute_index >= 0 && absolute_index < items.size()) {
		callable_mp(popup_menu, &PopupMenu::activate_item).call_deferred(items[absolute_index].popup_menu_index);
	}

	cleanup();
	hide();
}

void EditorPopupMenuDialog::cancel_pressed() {
	cleanup();
}

void EditorPopupMenuDialog::cleanup() {
	items.clear();
	item_visible_status.clear();

	item_list->clear();
	search_box->clear();
}

void EditorPopupMenuDialog::handle_search_box_input(const Ref<InputEvent> &p_ie) {
	if (num_visible_results < 0) {
		return;
	}

	Ref<InputEventKey> key_event = p_ie;
	if (key_event.is_valid() && key_event->is_pressed()) {
		bool move_selection = false;

		switch (key_event->get_keycode()) {
			case Key::UP:
			case Key::DOWN:
			case Key::PAGEUP:
			case Key::PAGEDOWN: {
				move_selection = true;
			} break;
			default:
				break; // Let the event through so it will reach the search box.
		}

		if (move_selection) {
			_move_selection_index(key_event->get_keycode());
			search_box->accept_event();
		}
	}
}

void EditorPopupMenuDialog::_move_selection_index(Key p_key) {
	// Don't move selection if there are no results.
	if (num_visible_results <= 0) {
		return;
	}
	const int max_index = num_visible_results - 1;

	PackedInt32Array selected_items = item_list->get_selected_items();
	if (selected_items.size() != 1) {
		return;
	}

	int idx = selected_items[0];
	if (p_key == Key::UP) {
		idx = (idx == 0) ? max_index : (idx - 1);
	} else if (p_key == Key::DOWN) {
		idx = (idx == max_index) ? 0 : (idx + 1);
	} else if (p_key == Key::PAGEUP) {
		idx = (idx == 0) ? idx : MAX(idx - 10, 0);
	} else if (p_key == Key::PAGEDOWN) {
		idx = (idx == max_index) ? idx : MIN(idx + 10, max_index);
	}

	item_list->deselect_all();
	item_list->select(idx);
	item_list->ensure_current_is_visible();
}

void EditorPopupMenuDialog::handle_item_activated(int p_index) {
	ok_pressed();
}

void EditorPopupMenuDialog::_search_box_text_changed(const String &p_query) {
	int total_item_count = items.size();

	if (p_query.is_empty()) {
		num_visible_results = total_item_count;
		item_visible_status.fill(true);
	} else {
		_update_fuzzy_search_results(p_query);
	}

	// The selected id is the item's index in EditorPopupMenuDialog::items, NOT the filtered ItemList.
	int selected_id = -1;
	Vector<int> selections = item_list->get_selected_items();
	if (selections.size() == 1) {
		selected_id = item_list->get_item_metadata(selections[0]);
	}

	item_list->set_item_count(num_visible_results);
	item_list->deselect_all();

	bool has_selected_item = false;
	int item_index = 0;
	for (int i = 0; i < total_item_count; i++) {
		if (item_visible_status[i]) {
			const PopupMenuItem &item = items.get(i);
			item_list->set_item_text(item_index, item.text);
			item_list->set_item_icon(item_index, item.icon);
			item_list->set_item_metadata(item_index, i); // Use EditorPopupMenuDialog::items index as id.
			if (selected_id == i) {
				item_list->select(item_index);
				has_selected_item = true;
			}
			item_index++;
		}
	}

	// Select first item if none were selected.
	if (num_visible_results > 0 && !has_selected_item) {
		item_list->select(0);
	}

	get_ok_button()->set_disabled(num_visible_results == 0);
}

void EditorPopupMenuDialog::_update_fuzzy_search_results(const String &p_query) {
	FuzzySearch fuzzy_search;
	fuzzy_search.set_query(p_query);
	fuzzy_search.max_results = items.size();
	bool fuzzy_matching = EDITOR_GET("filesystem/quick_open_dialog/enable_fuzzy_matching");
	int max_misses = EDITOR_GET("filesystem/quick_open_dialog/max_fuzzy_misses");
	fuzzy_search.allow_subsequences = fuzzy_matching;
	fuzzy_search.max_misses = fuzzy_matching ? max_misses : 0;

	num_visible_results = 0;
	item_visible_status.fill(false);

	int count = items.size();
	for (int i = 0; i < count; i++) {
		FuzzySearchResult result;
		if (fuzzy_search.search(items[i].text, result)) {
			num_visible_results++;
			item_visible_status.set(i, true);
		}
	}
}
