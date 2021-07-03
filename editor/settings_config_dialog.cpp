/*************************************************************************/
/*  settings_config_dialog.cpp                                           */
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

#include "settings_config_dialog.h"

#include "core/config/project_settings.h"
#include "core/input/input_map.h"
#include "core/os/keyboard.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor_file_system.h"
#include "editor_log.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "scene/gui/margin_container.h"

void EditorSettingsDialog::ok_pressed() {
	if (!EditorSettings::get_singleton()) {
		return;
	}

	_settings_save();
	timer->stop();
}

void EditorSettingsDialog::_settings_changed() {
	timer->start();
}

void EditorSettingsDialog::_settings_property_edited(const String &p_name) {
	String full_name = inspector->get_full_item_path(p_name);

	if (full_name == "interface/theme/accent_color" || full_name == "interface/theme/base_color" || full_name == "interface/theme/contrast") {
		EditorSettings::get_singleton()->set_manually("interface/theme/preset", "Custom"); // set preset to Custom
	} else if (full_name.begins_with("text_editor/theme/highlighting")) {
		EditorSettings::get_singleton()->set_manually("text_editor/theme/color_theme", "Custom");
	}
}

void EditorSettingsDialog::_settings_save() {
	EditorSettings::get_singleton()->notify_changes();
	EditorSettings::get_singleton()->save();
}

void EditorSettingsDialog::cancel_pressed() {
	if (!EditorSettings::get_singleton()) {
		return;
	}

	EditorSettings::get_singleton()->notify_changes();
}

void EditorSettingsDialog::popup_edit_settings() {
	if (!EditorSettings::get_singleton()) {
		return;
	}

	EditorSettings::get_singleton()->list_text_editor_themes(); // make sure we have an up to date list of themes

	inspector->edit(EditorSettings::get_singleton());
	inspector->get_inspector()->update_tree();

	search_box->select_all();
	search_box->grab_focus();

	_update_shortcuts();
	set_process_unhandled_input(true);

	// Restore valid window bounds or pop up at default size.
	Rect2 saved_size = EditorSettings::get_singleton()->get_project_metadata("dialog_bounds", "editor_settings", Rect2());
	if (saved_size != Rect2()) {
		popup(saved_size);
	} else {
		popup_centered_clamped(Size2(900, 700) * EDSCALE, 0.8);
	}

	_focus_current_search_box();
}

void EditorSettingsDialog::_filter_shortcuts(const String &p_filter) {
	shortcut_filter = p_filter;
	_update_shortcuts();
}

void EditorSettingsDialog::_undo_redo_callback(void *p_self, const String &p_name) {
	EditorNode::get_log()->add_message(p_name, EditorLog::MSG_TYPE_EDITOR);
}

void EditorSettingsDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				EditorSettings::get_singleton()->set_project_metadata("dialog_bounds", "editor_settings", Rect2(get_position(), get_size()));
				set_process_unhandled_input(false);
			}
		} break;
		case NOTIFICATION_READY: {
			undo_redo->set_method_notify_callback(EditorDebuggerNode::_method_changeds, nullptr);
			undo_redo->set_property_notify_callback(EditorDebuggerNode::_property_changeds, nullptr);
			undo_redo->set_commit_notify_callback(_undo_redo_callback, this);
		} break;
		case NOTIFICATION_ENTER_TREE: {
			_update_icons();
		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			_update_icons();
			// Update theme colors.
			inspector->update_category_list();
			_update_shortcuts();
		} break;
	}
}

void EditorSettingsDialog::unhandled_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	const Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->is_pressed()) {
		bool handled = false;

		if (ED_IS_SHORTCUT("ui_undo", p_event)) {
			String action = undo_redo->get_current_action_name();
			if (action != "") {
				EditorNode::get_log()->add_message("Undo: " + action, EditorLog::MSG_TYPE_EDITOR);
			}
			undo_redo->undo();
			handled = true;
		}

		if (ED_IS_SHORTCUT("ui_redo", p_event)) {
			undo_redo->redo();
			String action = undo_redo->get_current_action_name();
			if (action != "") {
				EditorNode::get_log()->add_message("Redo: " + action, EditorLog::MSG_TYPE_EDITOR);
			}
			handled = true;
		}

		if (k->get_keycode_with_modifiers() == (KeyModifierMask::CMD | Key::F)) {
			_focus_current_search_box();
			handled = true;
		}

		if (handled) {
			set_input_as_handled();
		}
	}
}

void EditorSettingsDialog::_update_icons() {
	search_box->set_right_icon(shortcuts->get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
	search_box->set_clear_button_enabled(true);
	shortcut_search_box->set_right_icon(shortcuts->get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
	shortcut_search_box->set_clear_button_enabled(true);

	restart_close_button->set_icon(shortcuts->get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
	restart_container->add_theme_style_override("panel", shortcuts->get_theme_stylebox(SNAME("bg"), SNAME("Tree")));
	restart_icon->set_texture(shortcuts->get_theme_icon(SNAME("StatusWarning"), SNAME("EditorIcons")));
	restart_label->add_theme_color_override("font_color", shortcuts->get_theme_color(SNAME("warning_color"), SNAME("Editor")));
}

void EditorSettingsDialog::_event_config_confirmed() {
	Ref<InputEventKey> k = shortcut_editor->get_event();
	if (k.is_null()) {
		return;
	}

	if (current_event_index == -1) {
		// Add new event
		current_events.push_back(k);
	} else {
		// Edit existing event
		current_events[current_event_index] = k;
	}

	if (is_editing_action) {
		_update_builtin_action(current_edited_identifier, current_events);
	} else {
		_update_shortcut_events(current_edited_identifier, current_events);
	}
}

void EditorSettingsDialog::_update_builtin_action(const String &p_name, const Array &p_events) {
	Array old_input_array = EditorSettings::get_singleton()->get_builtin_action_overrides(p_name);

	undo_redo->create_action(TTR("Edit Built-in Action") + " '" + p_name + "'");
	undo_redo->add_do_method(EditorSettings::get_singleton(), "set_builtin_action_override", p_name, p_events);
	undo_redo->add_undo_method(EditorSettings::get_singleton(), "set_builtin_action_override", p_name, old_input_array);
	undo_redo->add_do_method(this, "_settings_changed");
	undo_redo->add_undo_method(this, "_settings_changed");
	undo_redo->commit_action();

	_update_shortcuts();
}

void EditorSettingsDialog::_update_shortcut_events(const String &p_path, const Array &p_events) {
	Ref<Shortcut> current_sc = EditorSettings::get_singleton()->get_shortcut(p_path);

	undo_redo->create_action(TTR("Edit Shortcut") + " '" + p_path + "'");
	undo_redo->add_do_method(current_sc.ptr(), "set_events", p_events);
	undo_redo->add_undo_method(current_sc.ptr(), "set_events", current_sc->get_events());
	undo_redo->add_do_method(this, "_update_shortcuts");
	undo_redo->add_undo_method(this, "_update_shortcuts");
	undo_redo->add_do_method(this, "_settings_changed");
	undo_redo->add_undo_method(this, "_settings_changed");
	undo_redo->commit_action();
}

Array EditorSettingsDialog::_event_list_to_array_helper(List<Ref<InputEvent>> &p_events) {
	Array events;

	// Convert the list to an array, and only keep key events as this is for the editor.
	for (List<Ref<InputEvent>>::Element *E = p_events.front(); E; E = E->next()) {
		Ref<InputEventKey> k = E->get();
		if (k.is_valid()) {
			events.append(E->get());
		}
	}

	return events;
}

void EditorSettingsDialog::_create_shortcut_treeitem(TreeItem *p_parent, const String &p_shortcut_identifier, const String &p_display, Array &p_events, bool p_allow_revert, bool p_is_action, bool p_is_collapsed) {
	TreeItem *shortcut_item = shortcuts->create_item(p_parent);
	shortcut_item->set_collapsed(p_is_collapsed);
	shortcut_item->set_text(0, p_display);

	Ref<InputEvent> primary = p_events.size() > 0 ? Ref<InputEvent>(p_events[0]) : Ref<InputEvent>();
	Ref<InputEvent> secondary = p_events.size() > 1 ? Ref<InputEvent>(p_events[1]) : Ref<InputEvent>();

	String sc_text = "None";
	if (primary.is_valid()) {
		sc_text = primary->as_text();

		if (secondary.is_valid()) {
			sc_text += ", " + secondary->as_text();

			if (p_events.size() > 2) {
				sc_text += " (+" + itos(p_events.size() - 2) + ")";
			}
		}
	}

	shortcut_item->set_text(1, sc_text);
	if (sc_text == "None") {
		// Fade out unassigned shortcut labels for easier visual grepping.
		shortcut_item->set_custom_color(1, shortcuts->get_theme_color("font_color", "Label") * Color(1, 1, 1, 0.5));
	}

	if (p_allow_revert) {
		shortcut_item->add_button(1, shortcuts->get_theme_icon("Reload", "EditorIcons"), SHORTCUT_REVERT);
	}

	shortcut_item->add_button(1, shortcuts->get_theme_icon("Add", "EditorIcons"), SHORTCUT_ADD);
	shortcut_item->add_button(1, shortcuts->get_theme_icon("Close", "EditorIcons"), SHORTCUT_ERASE);

	shortcut_item->set_meta("is_action", p_is_action);
	shortcut_item->set_meta("type", "shortcut");
	shortcut_item->set_meta("shortcut_identifier", p_shortcut_identifier);
	shortcut_item->set_meta("events", p_events);

	// Shortcut Input Events
	for (int i = 0; i < p_events.size(); i++) {
		Ref<InputEvent> ie = p_events[i];
		if (ie.is_null()) {
			continue;
		}

		TreeItem *event_item = shortcuts->create_item(shortcut_item);

		event_item->set_text(0, shortcut_item->get_child_count() == 1 ? "Primary" : "");
		event_item->set_text(1, ie->as_text());

		event_item->add_button(1, shortcuts->get_theme_icon("Edit", "EditorIcons"), SHORTCUT_EDIT);
		event_item->add_button(1, shortcuts->get_theme_icon("Close", "EditorIcons"), SHORTCUT_ERASE);

		event_item->set_custom_bg_color(0, shortcuts->get_theme_color("dark_color_3", "Editor"));
		event_item->set_custom_bg_color(1, shortcuts->get_theme_color("dark_color_3", "Editor"));

		event_item->set_meta("is_action", p_is_action);
		event_item->set_meta("type", "event");
		event_item->set_meta("event_index", i);
	}
}

void EditorSettingsDialog::_update_shortcuts() {
	// Before clearing the tree, take note of which categories are collapsed so that this state can be maintained when the tree is repopulated.
	Map<String, bool> collapsed;

	if (shortcuts->get_root() && shortcuts->get_root()->get_first_child()) {
		TreeItem *ti = shortcuts->get_root()->get_first_child();
		while (ti) {
			// Not all items have valid or unique text in the first column - so if it has an identifier, use that, as it should be unique.
			if (ti->get_first_child() && ti->has_meta("shortcut_identifier")) {
				collapsed[ti->get_meta("shortcut_identifier")] = ti->is_collapsed();
			} else {
				collapsed[ti->get_text(0)] = ti->is_collapsed();
			}

			// Try go down tree
			TreeItem *ti_next = ti->get_first_child();
			// Try go across tree
			if (!ti_next) {
				ti_next = ti->get_next();
			}
			// Try go up tree, to next node
			if (!ti_next) {
				ti_next = ti->get_parent()->get_next();
			}

			ti = ti_next;
		}
	}

	shortcuts->clear();

	TreeItem *root = shortcuts->create_item();
	Map<String, TreeItem *> sections;

	// Set up section for Common/Built-in actions
	TreeItem *common_section = shortcuts->create_item(root);
	sections["Common"] = common_section;
	common_section->set_text(0, TTR("Common"));
	common_section->set_selectable(0, false);
	common_section->set_selectable(1, false);
	if (collapsed.has("Common")) {
		common_section->set_collapsed(collapsed["Common"]);
	}
	common_section->set_custom_bg_color(0, shortcuts->get_theme_color(SNAME("prop_subsection"), SNAME("Editor")));
	common_section->set_custom_bg_color(1, shortcuts->get_theme_color(SNAME("prop_subsection"), SNAME("Editor")));

	// Get the action map for the editor, and add each item to the "Common" section.
	OrderedHashMap<StringName, InputMap::Action> action_map = InputMap::get_singleton()->get_action_map();
	for (OrderedHashMap<StringName, InputMap::Action>::Element E = action_map.front(); E; E = E.next()) {
		String action_name = E.key();
		InputMap::Action action = E.get();

		Array events; // Need to get the list of events into an array so it can be set as metadata on the item.
		Vector<String> event_strings;

		List<Ref<InputEvent>> all_default_events = InputMap::get_singleton()->get_builtins_with_feature_overrides_applied().find(action_name).value();
		List<Ref<InputEventKey>> key_default_events;
		// Remove all non-key events from the defaults. Only check keys, since we are in the editor.
		for (List<Ref<InputEvent>>::Element *I = all_default_events.front(); I; I = I->next()) {
			Ref<InputEventKey> k = I->get();
			if (k.is_valid()) {
				key_default_events.push_back(k);
			}
		}

		// Join the text of the events with a delimiter so they can all be displayed in one cell.
		String events_display_string = event_strings.is_empty() ? "None" : String("; ").join(event_strings);

		if (!shortcut_filter.is_subsequence_ofi(action_name) && (events_display_string == "None" || !shortcut_filter.is_subsequence_ofi(events_display_string))) {
			continue;
		}

		Array action_events = _event_list_to_array_helper(action.inputs);
		Array default_events = _event_list_to_array_helper(all_default_events);
		bool same_as_defaults = Shortcut::is_event_array_equal(default_events, action_events);
		bool collapse = !collapsed.has(action_name) || (collapsed.has(action_name) && collapsed[action_name]);

		_create_shortcut_treeitem(common_section, action_name, action_name, action_events, !same_as_defaults, true, collapse);
	}

	// Editor Shortcuts

	List<String> slist;
	EditorSettings::get_singleton()->get_shortcut_list(&slist);

	for (const String &E : slist) {
		Ref<Shortcut> sc = EditorSettings::get_singleton()->get_shortcut(E);
		if (!sc->has_meta("original")) {
			continue;
		}

		// Shortcut Section

		TreeItem *section;
		String section_name = E.get_slice("/", 0);

		if (sections.has(section_name)) {
			section = sections[section_name];
		} else {
			section = shortcuts->create_item(root);

			String item_name = section_name.capitalize();
			section->set_text(0, item_name);
			section->set_selectable(0, false);
			section->set_selectable(1, false);
			section->set_custom_bg_color(0, shortcuts->get_theme_color(SNAME("prop_subsection"), SNAME("Editor")));
			section->set_custom_bg_color(1, shortcuts->get_theme_color(SNAME("prop_subsection"), SNAME("Editor")));

			if (collapsed.has(item_name)) {
				section->set_collapsed(collapsed[item_name]);
			}

			sections[section_name] = section;
		}

		// Shortcut Item

		if (!shortcut_filter.is_subsequence_ofi(sc->get_name())) {
			continue;
		}

		Array original = sc->get_meta("original");
		Array shortcuts_array = sc->get_events().duplicate(true);
		bool same_as_defaults = Shortcut::is_event_array_equal(original, shortcuts_array);
		bool collapse = !collapsed.has(E) || (collapsed.has(E) && collapsed[E]);

		_create_shortcut_treeitem(section, E, sc->get_name(), shortcuts_array, !same_as_defaults, false, collapse);
	}

	// remove sections with no shortcuts
	for (KeyValue<String, TreeItem *> &E : sections) {
		TreeItem *section = E.value;
		if (section->get_first_child() == nullptr) {
			root->remove_child(section);
		}
	}
}

void EditorSettingsDialog::_shortcut_button_pressed(Object *p_item, int p_column, int p_idx) {
	TreeItem *ti = Object::cast_to<TreeItem>(p_item);
	ERR_FAIL_COND_MSG(!ti, "Object passed is not a TreeItem");

	ShortcutButton button_idx = (ShortcutButton)p_idx;

	is_editing_action = ti->get_meta("is_action");

	String type = ti->get_meta("type");

	if (type == "event") {
		current_edited_identifier = ti->get_parent()->get_meta("shortcut_identifier");
		current_events = ti->get_parent()->get_meta("events");
		current_event_index = ti->get_meta("event_index");
	} else { // Type is "shortcut"
		current_edited_identifier = ti->get_meta("shortcut_identifier");
		current_events = ti->get_meta("events");
		current_event_index = -1;
	}

	switch (button_idx) {
		case EditorSettingsDialog::SHORTCUT_ADD: {
			// Only for "shortcut" types
			shortcut_editor->popup_and_configure();
		} break;
		case EditorSettingsDialog::SHORTCUT_EDIT: {
			// Only for "event" types
			shortcut_editor->popup_and_configure(current_events[current_event_index]);
		} break;
		case EditorSettingsDialog::SHORTCUT_ERASE: {
			if (type == "shortcut") {
				if (is_editing_action) {
					_update_builtin_action(current_edited_identifier, Array());
				} else {
					_update_shortcut_events(current_edited_identifier, Array());
				}
			} else if (type == "event") {
				current_events.remove_at(current_event_index);

				if (is_editing_action) {
					_update_builtin_action(current_edited_identifier, current_events);
				} else {
					_update_shortcut_events(current_edited_identifier, current_events);
				}
			}
		} break;
		case EditorSettingsDialog::SHORTCUT_REVERT: {
			// Only for "shortcut" types
			if (is_editing_action) {
				List<Ref<InputEvent>> defaults = InputMap::get_singleton()->get_builtins_with_feature_overrides_applied()[current_edited_identifier];
				Array events = _event_list_to_array_helper(defaults);

				_update_builtin_action(current_edited_identifier, events);
			} else {
				Ref<Shortcut> sc = EditorSettings::get_singleton()->get_shortcut(current_edited_identifier);
				Array original = sc->get_meta("original");
				_update_shortcut_events(current_edited_identifier, original);
			}
		} break;
		default:
			break;
	}
}

Variant EditorSettingsDialog::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	TreeItem *selected = shortcuts->get_selected();

	// Only allow drag for events
	if (!selected || !selected->has_meta("type") || selected->get_meta("type") != "event") {
		return Variant();
	}

	String label_text = "Event " + itos(selected->get_meta("event_index"));
	Label *label = memnew(Label(label_text));
	label->set_modulate(Color(1, 1, 1, 1.0f));
	shortcuts->set_drag_preview(label);

	shortcuts->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN);

	return Dictionary(); // No data required
}

bool EditorSettingsDialog::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	TreeItem *selected = shortcuts->get_selected();
	TreeItem *item = shortcuts->get_item_at_position(p_point);
	if (!selected || !item || item == selected || !item->has_meta("type") || item->get_meta("type") != "event") {
		return false;
	}

	// Don't allow moving an events in-between shortcuts.
	if (selected->get_parent()->get_meta("shortcut_identifier") != item->get_parent()->get_meta("shortcut_identifier")) {
		return false;
	}

	return true;
}

void EditorSettingsDialog::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from)) {
		return;
	}

	TreeItem *selected = shortcuts->get_selected();
	TreeItem *target = shortcuts->get_item_at_position(p_point);

	if (!target) {
		return;
	}

	int target_event_index = target->get_meta("event_index");
	int index_moving_from = selected->get_meta("event_index");

	Array events = selected->get_parent()->get_meta("events");

	Variant event_moved = events[index_moving_from];
	events.remove_at(index_moving_from);
	events.insert(target_event_index, event_moved);

	String ident = selected->get_parent()->get_meta("shortcut_identifier");
	if (selected->get_meta("is_action")) {
		_update_builtin_action(ident, events);
	} else {
		_update_shortcut_events(ident, events);
	}
}

void EditorSettingsDialog::_tabs_tab_changed(int p_tab) {
	_focus_current_search_box();
}

void EditorSettingsDialog::_focus_current_search_box() {
	Control *tab = tabs->get_current_tab_control();
	LineEdit *current_search_box = nullptr;
	if (tab == tab_general) {
		current_search_box = search_box;
	} else if (tab == tab_shortcuts) {
		current_search_box = shortcut_search_box;
	}

	if (current_search_box) {
		current_search_box->grab_focus();
		current_search_box->select_all();
	}
}

void EditorSettingsDialog::_editor_restart() {
	EditorNode::get_singleton()->save_all_scenes();
	EditorNode::get_singleton()->restart_editor();
}

void EditorSettingsDialog::_editor_restart_request() {
	restart_container->show();
}

void EditorSettingsDialog::_editor_restart_close() {
	restart_container->hide();
}

void EditorSettingsDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_shortcuts"), &EditorSettingsDialog::_update_shortcuts);
	ClassDB::bind_method(D_METHOD("_settings_changed"), &EditorSettingsDialog::_settings_changed);

	ClassDB::bind_method(D_METHOD("_get_drag_data_fw"), &EditorSettingsDialog::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("_can_drop_data_fw"), &EditorSettingsDialog::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("_drop_data_fw"), &EditorSettingsDialog::drop_data_fw);
}

EditorSettingsDialog::EditorSettingsDialog() {
	set_title(TTR("Editor Settings"));

	undo_redo = memnew(UndoRedo);

	tabs = memnew(TabContainer);
	tabs->set_tab_align(TabContainer::ALIGN_LEFT);
	tabs->connect("tab_changed", callable_mp(this, &EditorSettingsDialog::_tabs_tab_changed));
	add_child(tabs);

	// General Tab

	tab_general = memnew(VBoxContainer);
	tabs->add_child(tab_general);
	tab_general->set_name(TTR("General"));

	HBoxContainer *hbc = memnew(HBoxContainer);
	hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tab_general->add_child(hbc);

	search_box = memnew(LineEdit);
	search_box->set_placeholder(TTR("Search"));
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->add_child(search_box);

	inspector = memnew(SectionedInspector);
	inspector->get_inspector()->set_use_filter(true);
	inspector->register_search_box(search_box);
	inspector->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	inspector->get_inspector()->set_undo_redo(undo_redo);
	tab_general->add_child(inspector);
	inspector->get_inspector()->connect("property_edited", callable_mp(this, &EditorSettingsDialog::_settings_property_edited));
	inspector->get_inspector()->connect("restart_requested", callable_mp(this, &EditorSettingsDialog::_editor_restart_request));

	restart_container = memnew(PanelContainer);
	tab_general->add_child(restart_container);
	HBoxContainer *restart_hb = memnew(HBoxContainer);
	restart_container->add_child(restart_hb);
	restart_icon = memnew(TextureRect);
	restart_icon->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	restart_hb->add_child(restart_icon);
	restart_label = memnew(Label);
	restart_label->set_text(TTR("The editor must be restarted for changes to take effect."));
	restart_hb->add_child(restart_label);
	restart_hb->add_spacer();
	Button *restart_button = memnew(Button);
	restart_button->connect("pressed", callable_mp(this, &EditorSettingsDialog::_editor_restart));
	restart_hb->add_child(restart_button);
	restart_button->set_text(TTR("Save & Restart"));
	restart_close_button = memnew(Button);
	restart_close_button->set_flat(true);
	restart_close_button->connect("pressed", callable_mp(this, &EditorSettingsDialog::_editor_restart_close));
	restart_hb->add_child(restart_close_button);
	restart_container->hide();

	// Shortcuts Tab

	tab_shortcuts = memnew(VBoxContainer);

	tabs->add_child(tab_shortcuts);
	tab_shortcuts->set_name(TTR("Shortcuts"));

	shortcut_search_box = memnew(LineEdit);
	shortcut_search_box->set_placeholder(TTR("Search"));
	shortcut_search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tab_shortcuts->add_child(shortcut_search_box);
	shortcut_search_box->connect("text_changed", callable_mp(this, &EditorSettingsDialog::_filter_shortcuts));

	shortcuts = memnew(Tree);
	shortcuts->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	shortcuts->set_columns(2);
	shortcuts->set_hide_root(true);
	shortcuts->set_column_titles_visible(true);
	shortcuts->set_column_title(0, TTR("Name"));
	shortcuts->set_column_title(1, TTR("Binding"));
	shortcuts->connect("button_pressed", callable_mp(this, &EditorSettingsDialog::_shortcut_button_pressed));
	tab_shortcuts->add_child(shortcuts);

	shortcuts->set_drag_forwarding(this);

	// Adding event dialog
	shortcut_editor = memnew(InputEventConfigurationDialog);
	shortcut_editor->connect("confirmed", callable_mp(this, &EditorSettingsDialog::_event_config_confirmed));
	shortcut_editor->set_allowed_input_types(InputEventConfigurationDialog::InputType::INPUT_KEY);
	add_child(shortcut_editor);

	set_hide_on_ok(true);

	timer = memnew(Timer);
	timer->set_wait_time(1.5);
	timer->connect("timeout", callable_mp(this, &EditorSettingsDialog::_settings_save));
	timer->set_one_shot(true);
	add_child(timer);
	EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &EditorSettingsDialog::_settings_changed));
	get_ok_button()->set_text(TTR("Close"));

	updating = false;
}

EditorSettingsDialog::~EditorSettingsDialog() {
	memdelete(undo_redo);
}
