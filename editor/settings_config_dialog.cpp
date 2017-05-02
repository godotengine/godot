/*************************************************************************/
/*  settings_config_dialog.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor_file_system.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "global_config.h"
#include "os/keyboard.h"
#include "scene/gui/margin_container.h"

void EditorSettingsDialog::ok_pressed() {

	if (!EditorSettings::get_singleton())
		return;

	_settings_save();
	timer->stop();
}

void EditorSettingsDialog::_settings_changed() {

	timer->start();
}

void EditorSettingsDialog::_settings_property_edited(const String &p_name) {

	String full_name = property_editor->get_full_item_path(p_name);

	// Small usability workaround to update the text color settings when the
	// color theme is changed
	if (full_name == "text_editor/theme/color_theme") {
		property_editor->get_property_editor()->update_tree();
	}
}

void EditorSettingsDialog::_settings_save() {

	EditorSettings::get_singleton()->notify_changes();
	EditorSettings::get_singleton()->save();
}

void EditorSettingsDialog::cancel_pressed() {

	if (!EditorSettings::get_singleton())
		return;

	EditorSettings::get_singleton()->notify_changes();
}

void EditorSettingsDialog::popup_edit_settings() {

	if (!EditorSettings::get_singleton())
		return;

	EditorSettings::get_singleton()->list_text_editor_themes(); // make sure we have an up to date list of themes

	property_editor->edit(EditorSettings::get_singleton());
	property_editor->get_property_editor()->update_tree();

	search_box->select_all();
	search_box->grab_focus();

	_update_shortcuts();

	// Restore valid window bounds or pop up at default size.
	if (EditorSettings::get_singleton()->has("interface/dialogs/editor_settings_bounds")) {
		popup(EditorSettings::get_singleton()->get("interface/dialogs/editor_settings_bounds"));
	} else {
		popup_centered_ratio(0.7);
	}
}

void EditorSettingsDialog::_clear_search_box() {

	if (search_box->get_text() == "")
		return;

	search_box->clear();
	property_editor->get_property_editor()->update_tree();
}

void EditorSettingsDialog::_clear_shortcut_search_box() {
	if (shortcut_search_box->get_text() == "")
		return;

	shortcut_search_box->clear();
}

void EditorSettingsDialog::_filter_shortcuts(const String &p_filter) {
	shortcut_filter = p_filter;
	_update_shortcuts();
}

void EditorSettingsDialog::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			clear_button->set_icon(get_icon("Close", "EditorIcons"));
			shortcut_clear_button->set_icon(get_icon("Close", "EditorIcons"));
		} break;
		case NOTIFICATION_POPUP_HIDE: {
			EditorSettings::get_singleton()->set("interface/dialogs/editor_settings_bounds", get_rect());
		} break;
	}
}

void EditorSettingsDialog::_update_shortcuts() {

	shortcuts->clear();

	List<String> slist;
	EditorSettings::get_singleton()->get_shortcut_list(&slist);
	TreeItem *root = shortcuts->create_item();

	Map<String, TreeItem *> sections;

	for (List<String>::Element *E = slist.front(); E; E = E->next()) {

		Ref<ShortCut> sc = EditorSettings::get_singleton()->get_shortcut(E->get());
		if (!sc->has_meta("original"))
			continue;

		InputEvent original = sc->get_meta("original");

		String section_name = E->get().get_slice("/", 0);

		TreeItem *section;

		if (sections.has(section_name)) {
			section = sections[section_name];
		} else {
			section = shortcuts->create_item(root);
			section->set_text(0, section_name.capitalize());

			sections[section_name] = section;
			section->set_custom_bg_color(0, get_color("prop_subsection", "Editor"));
			section->set_custom_bg_color(1, get_color("prop_subsection", "Editor"));
		}

		if (shortcut_filter.is_subsequence_ofi(sc->get_name())) {
			TreeItem *item = shortcuts->create_item(section);

			item->set_text(0, sc->get_name());
			item->set_text(1, sc->get_as_text());
			if (!sc->is_shortcut(original) && !(sc->get_shortcut().type == InputEvent::NONE && original.type == InputEvent::NONE)) {
				item->add_button(1, get_icon("Reload", "EditorIcons"), 2);
			}
			item->add_button(1, get_icon("Edit", "EditorIcons"), 0);
			item->add_button(1, get_icon("Close", "EditorIcons"), 1);
			item->set_tooltip(0, E->get());
			item->set_metadata(0, E->get());
		}
	}

	// remove sections with no shortcuts
	for (Map<String, TreeItem *>::Element *E = sections.front(); E; E = E->next()) {
		TreeItem *section = E->get();
		if (section->get_children() == NULL) {
			root->remove_child(section);
		}
	}
}

void EditorSettingsDialog::_shortcut_button_pressed(Object *p_item, int p_column, int p_idx) {

	TreeItem *ti = p_item->cast_to<TreeItem>();
	ERR_FAIL_COND(!ti);

	String item = ti->get_metadata(0);
	Ref<ShortCut> sc = EditorSettings::get_singleton()->get_shortcut(item);

	if (p_idx == 0) {
		press_a_key_label->set_text(TTR("Press a Key.."));
		last_wait_for_key = InputEvent();
		press_a_key->popup_centered(Size2(250, 80) * EDSCALE);
		press_a_key->grab_focus();
		press_a_key->get_ok()->set_focus_mode(FOCUS_NONE);
		press_a_key->get_cancel()->set_focus_mode(FOCUS_NONE);
		shortcut_configured = item;

	} else if (p_idx == 1) { //erase
		if (!sc.is_valid())
			return; //pointless, there is nothing

		UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
		ur->create_action("Erase Shortcut");
		ur->add_do_method(sc.ptr(), "set_shortcut", InputEvent());
		ur->add_undo_method(sc.ptr(), "set_shortcut", sc->get_shortcut());
		ur->add_do_method(this, "_update_shortcuts");
		ur->add_undo_method(this, "_update_shortcuts");
		ur->add_do_method(this, "_settings_changed");
		ur->add_undo_method(this, "_settings_changed");
		ur->commit_action();
	} else if (p_idx == 2) { //revert to original
		if (!sc.is_valid())
			return; //pointless, there is nothing

		InputEvent original = sc->get_meta("original");

		UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
		ur->create_action("Restore Shortcut");
		ur->add_do_method(sc.ptr(), "set_shortcut", original);
		ur->add_undo_method(sc.ptr(), "set_shortcut", sc->get_shortcut());
		ur->add_do_method(this, "_update_shortcuts");
		ur->add_undo_method(this, "_update_shortcuts");
		ur->add_do_method(this, "_settings_changed");
		ur->add_undo_method(this, "_settings_changed");
		ur->commit_action();
	}
}

void EditorSettingsDialog::_wait_for_key(const InputEvent &p_event) {

	if (p_event.type == InputEvent::KEY && p_event.key.pressed && p_event.key.scancode != 0) {

		last_wait_for_key = p_event;
		String str = keycode_get_string(p_event.key.scancode).capitalize();
		if (p_event.key.mod.meta)
			str = TTR("Meta+") + str;
		if (p_event.key.mod.shift)
			str = TTR("Shift+") + str;
		if (p_event.key.mod.alt)
			str = TTR("Alt+") + str;
		if (p_event.key.mod.control)
			str = TTR("Control+") + str;

		press_a_key_label->set_text(str);
		press_a_key->accept_event();
	}
}

void EditorSettingsDialog::_press_a_key_confirm() {

	if (last_wait_for_key.type != InputEvent::KEY)
		return;

	InputEvent ie;
	ie.type = InputEvent::KEY;
	ie.key.scancode = last_wait_for_key.key.scancode;
	ie.key.mod = last_wait_for_key.key.mod;

	Ref<ShortCut> sc = EditorSettings::get_singleton()->get_shortcut(shortcut_configured);

	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action("Change Shortcut '" + shortcut_configured + "'");
	ur->add_do_method(sc.ptr(), "set_shortcut", ie);
	ur->add_undo_method(sc.ptr(), "set_shortcut", sc->get_shortcut());
	ur->add_do_method(this, "_update_shortcuts");
	ur->add_undo_method(this, "_update_shortcuts");
	ur->add_do_method(this, "_settings_changed");
	ur->add_undo_method(this, "_settings_changed");
	ur->commit_action();
}

void EditorSettingsDialog::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_settings_save"), &EditorSettingsDialog::_settings_save);
	ClassDB::bind_method(D_METHOD("_settings_changed"), &EditorSettingsDialog::_settings_changed);
	ClassDB::bind_method(D_METHOD("_settings_property_edited"), &EditorSettingsDialog::_settings_property_edited);
	ClassDB::bind_method(D_METHOD("_clear_search_box"), &EditorSettingsDialog::_clear_search_box);
	ClassDB::bind_method(D_METHOD("_clear_shortcut_search_box"), &EditorSettingsDialog::_clear_shortcut_search_box);
	ClassDB::bind_method(D_METHOD("_shortcut_button_pressed"), &EditorSettingsDialog::_shortcut_button_pressed);
	ClassDB::bind_method(D_METHOD("_filter_shortcuts"), &EditorSettingsDialog::_filter_shortcuts);
	ClassDB::bind_method(D_METHOD("_update_shortcuts"), &EditorSettingsDialog::_update_shortcuts);
	ClassDB::bind_method(D_METHOD("_press_a_key_confirm"), &EditorSettingsDialog::_press_a_key_confirm);
	ClassDB::bind_method(D_METHOD("_wait_for_key"), &EditorSettingsDialog::_wait_for_key);
}

EditorSettingsDialog::EditorSettingsDialog() {

	set_title(TTR("Editor Settings"));
	set_resizable(true);

	tabs = memnew(TabContainer);
	add_child(tabs);
	//set_child_rect(tabs);

	VBoxContainer *vbc = memnew(VBoxContainer);
	tabs->add_child(vbc);
	vbc->set_name(TTR("General"));

	HBoxContainer *hbc = memnew(HBoxContainer);
	hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(hbc);

	Label *l = memnew(Label);
	l->set_text(TTR("Search:") + " ");
	hbc->add_child(l);

	search_box = memnew(LineEdit);
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->add_child(search_box);

	clear_button = memnew(ToolButton);
	hbc->add_child(clear_button);
	clear_button->connect("pressed", this, "_clear_search_box");

	property_editor = memnew(SectionedPropertyEditor);
	//property_editor->hide_top_label();
	property_editor->get_property_editor()->set_use_filter(true);
	property_editor->get_property_editor()->register_text_enter(search_box);
	property_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(property_editor);
	property_editor->get_property_editor()->connect("property_edited", this, "_settings_property_edited");

	vbc = memnew(VBoxContainer);
	tabs->add_child(vbc);
	vbc->set_name(TTR("Shortcuts"));

	hbc = memnew(HBoxContainer);
	hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(hbc);

	l = memnew(Label);
	l->set_text(TTR("Search:") + " ");
	hbc->add_child(l);

	shortcut_search_box = memnew(LineEdit);
	shortcut_search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->add_child(shortcut_search_box);
	shortcut_search_box->connect("text_changed", this, "_filter_shortcuts");

	shortcut_clear_button = memnew(ToolButton);
	hbc->add_child(shortcut_clear_button);
	shortcut_clear_button->connect("pressed", this, "_clear_shortcut_search_box");

	shortcuts = memnew(Tree);
	vbc->add_margin_child("Shortcut List:", shortcuts, true);
	shortcuts->set_columns(2);
	shortcuts->set_hide_root(true);
	//shortcuts->set_hide_folding(true);
	shortcuts->set_column_titles_visible(true);
	shortcuts->set_column_title(0, "Name");
	shortcuts->set_column_title(1, "Binding");
	shortcuts->connect("button_pressed", this, "_shortcut_button_pressed");

	press_a_key = memnew(ConfirmationDialog);
	press_a_key->set_focus_mode(FOCUS_ALL);
	add_child(press_a_key);

	l = memnew(Label);
	l->set_text(TTR("Press a Key.."));
	l->set_area_as_parent_rect();
	l->set_align(Label::ALIGN_CENTER);
	l->set_margin(MARGIN_TOP, 20);
	l->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_BEGIN, 30);
	press_a_key_label = l;
	press_a_key->add_child(l);
	press_a_key->connect("gui_input", this, "_wait_for_key");
	press_a_key->connect("confirmed", this, "_press_a_key_confirm");
	//Button *load = memnew( Button );

	//load->set_text("Load..");
	//hbc->add_child(load);

	//get_ok()->set_text("Apply");
	set_hide_on_ok(true);
	//get_cancel()->set_text("Close");

	timer = memnew(Timer);
	timer->set_wait_time(1.5);
	timer->connect("timeout", this, "_settings_save");
	timer->set_one_shot(true);
	add_child(timer);
	EditorSettings::get_singleton()->connect("settings_changed", this, "_settings_changed");
	get_ok()->set_text(TTR("Close"));

	updating = false;
}
