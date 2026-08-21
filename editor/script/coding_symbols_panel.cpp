/**************************************************************************/
/*  coding_symbols_panel.cpp                                              */
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

#include "coding_symbols_panel.h"

#include "core/object/callable_mp.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/spin_box.h"
#include "servers/display/display_server.h"

void CodingSymbolsPanel::_bind_methods() {
	ADD_SIGNAL(MethodInfo("symbol_selected", PropertyInfo(Variant::STRING, "symbol")));
}

void CodingSymbolsPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			DisplayServer::get_singleton()->set_hardware_keyboard_connection_change_callback(callable_mp(this, &CodingSymbolsPanel::_hardware_keyboard_connected));
			//_hardware_keyboard_connected(DisplayServer::get_singleton()->has_hardware_keyboard());
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			settings_button->set_button_icon(get_editor_theme_icon(SNAME("Edit")));
			add_btn->set_button_icon(get_editor_theme_icon(SNAME("Add")));
			default_btn->set_button_icon(get_editor_theme_icon(SNAME("Reload")));
		} break;
	}
}

void CodingSymbolsPanel::_hardware_keyboard_connected(bool p_connected) {
	set_visible(!p_connected);
}

void CodingSymbolsPanel::_on_visible_limit_changed(double p_value) {
	visible_limit = (int)p_value;
	_rebuild_toolbar();
}

void CodingSymbolsPanel::_load_defaults() {
	for (int i = 0; i < symbols_metadata.size(); i++) {
		if (symbols_metadata[i].panel_button) {
			symbols_metadata[i].panel_button->queue_free();
		}
	}
	symbols_metadata.clear();

	Vector<String> d = { "Tab", "(", ")", "{", "}", "[", "]", "=", "+", "-", "*", "/", ".", ",", ":", ";", "<", ">", "!", "&", "|", "\"", "'", "_", "?" };
	for (const String &s : d) {
		_add_custom_button(s, (s == "Tab" ? "\t" : s));
	}
	_rebuild_toolbar();
}

void CodingSymbolsPanel::_add_custom_button(const String &p_label, const String &p_code) {
	Button *btn = memnew(Button);
	btn->set_text(p_label);
	btn->set_custom_minimum_size(Size2(64, 64));
	btn->set_focus_mode(FOCUS_NONE);
	btn->add_theme_font_override(SNAME("font"), get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
	btn->connect("pressed", callable_mp(this, &CodingSymbolsPanel::_on_symbol_pressed).bind(p_code));

	SymbolData symbol_data;
	symbol_data.label = p_label;
	symbol_data.code = p_code;
	symbol_data.panel_button = btn;
	symbols_metadata.push_back(symbol_data);

	_rebuild_toolbar();
}

void CodingSymbolsPanel::_rebuild_toolbar() {
	for (int i = 0; i < symbols_metadata.size(); i++) {
		if (symbols_metadata[i].panel_button->get_parent() == flow_container) {
			flow_container->remove_child(symbols_metadata[i].panel_button);
		}
	}

	int visible_count = 0;
	for (int i = 0; i < symbols_metadata.size(); i++) {
		if ((visible_count < visible_limit) || expanded) {
			flow_container->add_child(symbols_metadata[i].panel_button);
		}
		visible_count++;
	}

	flow_container->move_child(settings_button, -1);
	flow_container->move_child(expand_button, -1);
	expand_button->set_visible(visible_count > visible_limit);
}

void CodingSymbolsPanel::_rebuild_settings_list() {
	while (settings_list_vbox->get_child_count() > 0) {
		Node *child = settings_list_vbox->get_child(0);
		settings_list_vbox->remove_child(child);
		child->queue_free();
	}

	Ref<Texture2D> up_icon = get_theme_icon(SNAME("MoveUp"), SNAME("EditorIcons"));
	Ref<Texture2D> down_icon = get_theme_icon(SNAME("MoveDown"), SNAME("EditorIcons"));
	Ref<Texture2D> del_icon = get_theme_icon(SNAME("Remove"), SNAME("EditorIcons"));

	for (int i = 0; i < symbols_metadata.size(); i++) {
		SymbolSettingsRow *row = memnew(SymbolSettingsRow);
		row->index = i;
		row->panel = this;

		Button *up_btn = memnew(Button);
		up_btn->set_button_icon(get_editor_theme_icon(SNAME("MoveUp")));
		up_btn->set_disabled(i == 0);
		up_btn->connect("pressed", callable_mp(this, &CodingSymbolsPanel::_move_symbol_relative).bind(i, -1));
		row->add_child(up_btn);

		Button *down_btn = memnew(Button);
		down_btn->set_button_icon(get_editor_theme_icon(SNAME("MoveDown")));
		down_btn->set_disabled(i == symbols_metadata.size() - 1);
		down_btn->connect("pressed", callable_mp(this, &CodingSymbolsPanel::_move_symbol_relative).bind(i, 1));
		row->add_child(down_btn);

		Label *lbl = memnew(Label);
		lbl->set_text(symbols_metadata[i].label);
		lbl->set_h_size_flags(SIZE_EXPAND_FILL);
		row->add_child(lbl);

		Button *del = memnew(Button);
		del->set_button_icon(get_editor_theme_icon(SNAME("Remove")));
		del->connect("pressed", callable_mp(this, &CodingSymbolsPanel::_delete_symbol).bind(i));
		row->add_child(del);

		settings_list_vbox->add_child(row);
		settings_list_vbox->add_child(memnew(HSeparator));
	}
}

void CodingSymbolsPanel::_move_symbol_relative(int p_idx, int p_delta) {
	int target = p_idx + p_delta;
	if (target < 0 || target >= symbols_metadata.size()) {
		return;
	}

	SymbolData temp = symbols_metadata[p_idx];
	symbols_metadata.write[p_idx] = symbols_metadata[target];
	symbols_metadata.write[target] = temp;

	_rebuild_settings_list();
	_rebuild_toolbar();
}

void CodingSymbolsPanel::_delete_symbol(int p_idx) {
	symbols_metadata[p_idx].panel_button->queue_free();
	symbols_metadata.remove_at(p_idx);
	_rebuild_settings_list();
	_rebuild_toolbar();
}

void CodingSymbolsPanel::_on_add_custom_entry() {
	if (custom_label_edit->get_text().is_empty()) {
		return;
	}
	_add_custom_button(custom_label_edit->get_text(), custom_code_edit->get_text());
	custom_label_edit->clear();
	custom_code_edit->clear();
	_rebuild_settings_list();
}

void CodingSymbolsPanel::_on_settings_pressed() {
	_rebuild_settings_list();
	int new_height = get_viewport_rect().size.y * 0.7;
	settings_dialog->popup_centered(Size2(400, new_height));
}

void CodingSymbolsPanel::_toggle_expand() {
	expanded = !expanded;
	_rebuild_toolbar();
	expand_button->set_text(expanded ? "Less" : "...");
}

void CodingSymbolsPanel::_on_symbol_pressed(const String &s) {
	emit_signal("symbol_selected", s);
}

CodingSymbolsPanel::CodingSymbolsPanel() {
	set_h_size_flags(SIZE_EXPAND_FILL);
	flow_container = memnew(HFlowContainer);
	flow_container->add_theme_constant_override("h_separation", 8);
	flow_container->add_theme_constant_override("v_separation", 8);
	add_child(flow_container);

	settings_dialog = memnew(ConfirmationDialog);
	settings_dialog->set_title("Customize Symbols");
	settings_dialog->set_min_size(Size2(250, 300));
	add_child(settings_dialog);

	VBoxContainer *main_vbox = memnew(VBoxContainer);
	settings_dialog->add_child(main_vbox);

	HBoxContainer *top_row = memnew(HBoxContainer);
	top_row->add_child(memnew(Label("Visible limit: ")));
	visible_count_spin = memnew(SpinBox);
	visible_count_spin->set_min(1);
	visible_count_spin->set_max(100);
	visible_count_spin->set_value(visible_limit);
	visible_count_spin->connect("value_changed", callable_mp(this, &CodingSymbolsPanel::_on_visible_limit_changed));
	top_row->add_child(visible_count_spin);
	main_vbox->add_child(top_row);
	main_vbox->add_child(memnew(HSeparator));

	default_btn = memnew(Button);
	default_btn->set_text("Restore Defaults");
	default_btn->connect("pressed", callable_mp(this, &CodingSymbolsPanel::_load_defaults));
	top_row->add_spacer();
	top_row->add_child(default_btn);

	ScrollContainer *scroll = memnew(ScrollContainer);
	scroll->set_v_size_flags(SIZE_EXPAND_FILL);
	main_vbox->add_child(scroll);

	settings_list_vbox = memnew(VBoxContainer);
	settings_list_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
	scroll->add_child(settings_list_vbox);

	main_vbox->add_child(memnew(HSeparator));

	HBoxContainer *add_row = memnew(HBoxContainer);
	custom_label_edit = memnew(LineEdit);
	custom_label_edit->set_placeholder("Label");
	custom_label_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	add_row->add_child(custom_label_edit);

	custom_code_edit = memnew(LineEdit);
	custom_code_edit->set_placeholder("Code");
	custom_code_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	add_row->add_child(custom_code_edit);

	add_btn = memnew(Button);
	add_btn->connect("pressed", callable_mp(this, &CodingSymbolsPanel::_on_add_custom_entry));
	add_row->add_child(add_btn);
	main_vbox->add_child(add_row);

	settings_button = memnew(Button);
	settings_button->set_custom_minimum_size(Size2(64, 64));
	settings_button->connect("pressed", callable_mp(this, &CodingSymbolsPanel::_on_settings_pressed));
	flow_container->add_child(settings_button);

	expand_button = memnew(Button);
	expand_button->set_text("...");
	expand_button->set_custom_minimum_size(Size2(64, 64));
	expand_button->connect("pressed", callable_mp(this, &CodingSymbolsPanel::_toggle_expand));
	flow_container->add_child(expand_button);

	_load_defaults();
}
