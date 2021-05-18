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
	} else if (full_name.begins_with("text_editor/highlighting")) {
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

void EditorSettingsDialog::_unhandled_input(const Ref<InputEvent> &p_event) {
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

		if (k->get_keycode_with_modifiers() == (KEY_MASK_CMD | KEY_F)) {
			_focus_current_search_box();
			handled = true;
		}

		if (handled) {
			set_input_as_handled();
		}
	}
}

void EditorSettingsDialog::_update_icons() {
	search_box->set_right_icon(shortcuts->get_theme_icon("Search", "EditorIcons"));
	search_box->set_clear_button_enabled(true);
	shortcut_search_box->set_right_icon(shortcuts->get_theme_icon("Search", "EditorIcons"));
	shortcut_search_box->set_clear_button_enabled(true);

	restart_close_button->set_icon(shortcuts->get_theme_icon("Close", "EditorIcons"));
	restart_container->add_theme_style_override("panel", shortcuts->get_theme_stylebox("bg", "Tree"));
	restart_icon->set_texture(shortcuts->get_theme_icon("StatusWarning", "EditorIcons"));
	restart_label->add_theme_color_override("font_color", shortcuts->get_theme_color("warning_color", "Editor"));
}

void EditorSettingsDialog::_event_config_confirmed() {
	Ref<InputEventKey> k = shortcut_editor->get_event();
	if (k.is_null()) {
		return;
	}

	if (editing_action) {
		if (current_action_event_index == -1) {
			// Add new event
			current_action_events.push_back(k);
		} else {
			// Edit existing event
			current_action_events[current_action_event_index] = k;
		}

		_update_builtin_action(current_action, current_action_events);
	} else {
		k = k->duplicate();
		Ref<Shortcut> current_sc = EditorSettings::get_singleton()->get_shortcut(shortcut_being_edited);

		undo_redo->create_action(TTR("Change Shortcut") + " '" + shortcut_being_edited + "'");
		undo_redo->add_do_method(current_sc.ptr(), "set_shortcut", k);
		undo_redo->add_undo_method(current_sc.ptr(), "set_shortcut", current_sc->get_shortcut());
		undo_redo->add_do_method(this, "_update_shortcuts");
		undo_redo->add_undo_method(this, "_update_shortcuts");
		undo_redo->add_do_method(this, "_settings_changed");
		undo_redo->add_undo_method(this, "_settings_changed");
		undo_redo->commit_action();
	}
}

void EditorSettingsDialog::_update_builtin_action(const String &p_name, const Array &p_events) {
	Array old_input_array = EditorSettings::get_singleton()->get_builtin_action_overrides(current_action);

	undo_redo->create_action(TTR("Edit Built-in Action"));
	undo_redo->add_do_method(EditorSettings::get_singleton(), "set_builtin_action_override", p_name, p_events);
	undo_redo->add_undo_method(EditorSettings::get_singleton(), "set_builtin_action_override", p_name, old_input_array);
	undo_redo->add_do_method(this, "_settings_changed");
	undo_redo->add_undo_method(this, "_settings_changed");
	undo_redo->commit_action();

	_update_shortcuts();
}

void EditorSettingsDialog::_update_shortcuts() {
	// Before clearing the tree, take note of which categories are collapsed so that this state can be maintained when the tree is repopulated.
	Map<String, bool> collapsed;

	if (shortcuts->get_root() && shortcuts->get_root()->get_first_child()) {
		for (TreeItem *item = shortcuts->get_root()->get_first_child(); item; item = item->get_next()) {
			collapsed[item->get_text(0)] = item->is_collapsed();
		}
	}
	shortcuts->clear();

	TreeItem *root = shortcuts->create_item();
	Map<String, TreeItem *> sections;

	// Set up section for Common/Built-in actions
	TreeItem *common_section = shortcuts->create_item(root);

	sections["Common"] = common_section;
	common_section->set_text(0, TTR("Common"));
	if (collapsed.has("Common")) {
		common_section->set_collapsed(collapsed["Common"]);
	}
	common_section->set_custom_bg_color(0, shortcuts->get_theme_color("prop_subsection", "Editor"));
	common_section->set_custom_bg_color(1, shortcuts->get_theme_color("prop_subsection", "Editor"));

	// Get the action map for the editor, and add each item to the "Common" section.
	OrderedHashMap<StringName, InputMap::Action> action_map = InputMap::get_singleton()->get_action_map();
	for (OrderedHashMap<StringName, InputMap::Action>::Element E = action_map.front(); E; E = E.next()) {
		String action_name = E.key();

		if (!shortcut_filter.is_subsequence_ofi(action_name)) {
			continue;
		}

		InputMap::Action action = E.get();

		Array events; // Need to get the list of events into an array so it can be set as metadata on the item.
		Vector<String> event_strings;

		List<Ref<InputEvent>> all_default_events = InputMap::get_singleton()->get_builtins().find(action_name).value();
		List<Ref<InputEventKey>> key_default_events;
		// Remove all non-key events from the defaults. Only check keys, since we are in the editor.
		for (List<Ref<InputEvent>>::Element *I = all_default_events.front(); I; I = I->next()) {
			Ref<InputEventKey> k = I->get();
			if (k.is_valid()) {
				key_default_events.push_back(k);
			}
		}

		bool same_as_defaults = key_default_events.size() == action.inputs.size(); // Initially this is set to just whether the arrays are equal. Later we check the events if needed.

		int count = 0;
		for (List<Ref<InputEvent>>::Element *I = action.inputs.front(); I; I = I->next()) {
			// Add event and event text to respective arrays.
			events.push_back(I->get());
			event_strings.push_back(I->get()->as_text());

			// Only check if the events have been the same so far - once one fails, we don't need to check any more.
			if (same_as_defaults && !key_default_events[count]->shortcut_match(I->get())) {
				same_as_defaults = false;
			}
			count++;
		}

		// Join the text of the events with a delimiter so they can all be displayed in one cell.
		String events_display_string = event_strings.is_empty() ? "None" : String("; ").join(event_strings);

		TreeItem *item = shortcuts->create_item(common_section);
		item->set_text(0, action_name);
		item->set_text(1, events_display_string);

		if (!same_as_defaults) {
			item->add_button(1, shortcuts->get_theme_icon("Reload", "EditorIcons"), 2);
		}

		if (events_display_string == "None") {
			// Fade out unassigned shortcut labels for easier visual grepping.
			item->set_custom_color(1, shortcuts->get_theme_color("font_color", "Label") * Color(1, 1, 1, 0.5));
		}

		item->add_button(1, shortcuts->get_theme_icon("Edit", "EditorIcons"), 0);
		item->add_button(1, shortcuts->get_theme_icon("Close", "EditorIcons"), 1);
		item->set_tooltip(0, action_name);
		item->set_tooltip(1, events_display_string);
		item->set_metadata(0, "Common");
		item->set_metadata(1, events);
	}

	// Editor Shortcuts

	List<String> slist;
	EditorSettings::get_singleton()->get_shortcut_list(&slist);

	for (List<String>::Element *E = slist.front(); E; E = E->next()) {
		Ref<Shortcut> sc = EditorSettings::get_singleton()->get_shortcut(E->get());
		if (!sc->has_meta("original")) {
			continue;
		}

		Ref<InputEvent> original = sc->get_meta("original");

		String section_name = E->get().get_slice("/", 0);

		TreeItem *section;

		if (sections.has(section_name)) {
			section = sections[section_name];
		} else {
			section = shortcuts->create_item(root);

			String item_name = section_name.capitalize();
			section->set_text(0, item_name);

			if (collapsed.has(item_name)) {
				section->set_collapsed(collapsed[item_name]);
			}

			sections[section_name] = section;
			section->set_custom_bg_color(0, shortcuts->get_theme_color("prop_subsection", "Editor"));
			section->set_custom_bg_color(1, shortcuts->get_theme_color("prop_subsection", "Editor"));
		}

		// Don't match unassigned shortcuts when searching for assigned keys in search results.
		// This prevents all unassigned shortcuts from appearing when searching a string like "no".
		if (shortcut_filter.is_subsequence_ofi(sc->get_name()) || (sc->get_as_text() != "None" && shortcut_filter.is_subsequence_ofi(sc->get_as_text()))) {
			TreeItem *item = shortcuts->create_item(section);

			item->set_text(0, sc->get_name());
			item->set_text(1, sc->get_as_text());

			if (!sc->is_shortcut(original) && !(sc->get_shortcut().is_null() && original.is_null())) {
				item->add_button(1, shortcuts->get_theme_icon("Reload", "EditorIcons"), 2);
			}

			if (sc->get_as_text() == "None") {
				// Fade out unassigned shortcut labels for easier visual grepping.
				item->set_custom_color(1, shortcuts->get_theme_color("font_color", "Label") * Color(1, 1, 1, 0.5));
			}

			item->add_button(1, shortcuts->get_theme_icon("Edit", "EditorIcons"), 0);
			item->add_button(1, shortcuts->get_theme_icon("Close", "EditorIcons"), 1);
			item->set_tooltip(0, E->get());
			item->set_metadata(0, E->get());
		}
	}

	// remove sections with no shortcuts
	for (Map<String, TreeItem *>::Element *E = sections.front(); E; E = E->next()) {
		TreeItem *section = E->get();
		if (section->get_first_child() == nullptr) {
			root->remove_child(section);
		}
	}
}

void EditorSettingsDialog::_shortcut_button_pressed(Object *p_item, int p_column, int p_idx) {
	TreeItem *ti = Object::cast_to<TreeItem>(p_item);
	ERR_FAIL_COND(!ti);

	button_idx = p_idx;

	if (ti->get_metadata(0) == "Common") {
		// Editing a Built-in action, which can have multiple bindings.
		editing_action = true;
		current_action = ti->get_text(0);

		switch (button_idx) {
			case SHORTCUT_REVERT: {
				Array events;
				List<Ref<InputEvent>> defaults = InputMap::get_singleton()->get_builtins()[current_action];

				// Convert the list to an array, and only keep key events as this is for the editor.
				for (List<Ref<InputEvent>>::Element *E = defaults.front(); E; E = E->next()) {
					Ref<InputEventKey> k = E->get();
					if (k.is_valid()) {
						events.append(E->get());
					}
				}

				_update_builtin_action(current_action, events);
			} break;
			case SHORTCUT_EDIT:
			case SHORTCUT_ERASE: {
				// For Edit end Delete, we will show a popup which displays each event so the user can select which one to edit/delete.
				current_action_events = ti->get_metadata(1);
				action_popup->clear();

				for (int i = 0; i < current_action_events.size(); i++) {
					Ref<InputEvent> ie = current_action_events[i];
					action_popup->add_item(ie->as_text());
					action_popup->set_item_metadata(i, ie);
				}

				if (button_idx == SHORTCUT_EDIT) {
					// If editing, add a button which can be used to add an additional event.
					action_popup->add_icon_item(get_theme_icon("Add", "EditorIcons"), TTR("Add"));
				}

				action_popup->set_position(get_position() + get_mouse_position());
				action_popup->take_mouse_focus();
				action_popup->popup();
				action_popup->set_as_minsize();
			} break;
			default:
				break;
		}
	} else {
		// Editing an Editor Shortcut, which can only have 1 binding.
		String item = ti->get_metadata(0);
		Ref<Shortcut> sc = EditorSettings::get_singleton()->get_shortcut(item);
		editing_action = false;

		switch (button_idx) {
			case EditorSettingsDialog::SHORTCUT_EDIT:
				shortcut_editor->popup_and_configure(sc->get_shortcut());
				shortcut_being_edited = item;
				break;
			case EditorSettingsDialog::SHORTCUT_ERASE: {
				if (!sc.is_valid()) {
					return; //pointless, there is nothing
				}

				undo_redo->create_action(TTR("Erase Shortcut"));
				undo_redo->add_do_method(sc.ptr(), "set_shortcut", Ref<InputEvent>());
				undo_redo->add_undo_method(sc.ptr(), "set_shortcut", sc->get_shortcut());
				undo_redo->add_do_method(this, "_update_shortcuts");
				undo_redo->add_undo_method(this, "_update_shortcuts");
				undo_redo->add_do_method(this, "_settings_changed");
				undo_redo->add_undo_method(this, "_settings_changed");
				undo_redo->commit_action();
			} break;
			case EditorSettingsDialog::SHORTCUT_REVERT: {
				if (!sc.is_valid()) {
					return; //pointless, there is nothing
				}

				Ref<InputEvent> original = sc->get_meta("original");

				undo_redo->create_action(TTR("Restore Shortcut"));
				undo_redo->add_do_method(sc.ptr(), "set_shortcut", original);
				undo_redo->add_undo_method(sc.ptr(), "set_shortcut", sc->get_shortcut());
				undo_redo->add_do_method(this, "_update_shortcuts");
				undo_redo->add_undo_method(this, "_update_shortcuts");
				undo_redo->add_do_method(this, "_settings_changed");
				undo_redo->add_undo_method(this, "_settings_changed");
				undo_redo->commit_action();
			} break;
			default:
				break;
		}
	}
}

void EditorSettingsDialog::_builtin_action_popup_index_pressed(int p_index) {
	switch (button_idx) {
		case SHORTCUT_EDIT: {
			if (p_index == action_popup->get_item_count() - 1) {
				// Selected last item in list (Add button), therefore add new
				current_action_event_index = -1;
				shortcut_editor->popup_and_configure();
			} else {
				// Configure existing
				current_action_event_index = p_index;
				shortcut_editor->popup_and_configure(action_popup->get_item_metadata(p_index));
			}
		} break;
		case SHORTCUT_ERASE: {
			current_action_events.remove(p_index);
			_update_builtin_action(current_action, current_action_events);
		} break;
		default:
			break;
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
	ClassDB::bind_method(D_METHOD("_unhandled_input"), &EditorSettingsDialog::_unhandled_input);
	ClassDB::bind_method(D_METHOD("_update_shortcuts"), &EditorSettingsDialog::_update_shortcuts);
	ClassDB::bind_method(D_METHOD("_settings_changed"), &EditorSettingsDialog::_settings_changed);
}

EditorSettingsDialog::EditorSettingsDialog() {
	action_popup = memnew(PopupMenu);
	action_popup->connect("index_pressed", callable_mp(this, &EditorSettingsDialog::_builtin_action_popup_index_pressed));
	add_child(action_popup);

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
