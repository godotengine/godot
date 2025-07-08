/**************************************************************************/
/*  editor_bottom_panel.cpp                                               */
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

#include "editor_bottom_panel.h"

#include "editor/debugger/editor_debugger_node.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_toaster.h"
#include "editor/gui/editor_version_button.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/split_container.h"

void EditorBottomPanel::_update_margins() {
	TabContainer::_update_margins();
	get_tab_bar()->set_offset(SIDE_RIGHT, get_tab_bar()->get_offset(SIDE_RIGHT) - bottom_hbox->get_size().x);
}

void EditorBottomPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			pin_button->set_button_icon(get_editor_theme_icon(SNAME("Pin")));
			expand_button->set_button_icon(get_editor_theme_icon(SNAME("ExpandBottomDock")));
		} break;
	}
}

void EditorBottomPanel::_on_tab_changed(int p_idx) {
	callable_mp(this, &EditorBottomPanel::_repaint).call_deferred();
}

void EditorBottomPanel::_repaint() {
	bool panel_collapsed = get_current_tab() == -1;
	SplitContainer *center_split = Object::cast_to<SplitContainer>(get_parent());
	ERR_FAIL_NULL(center_split);

	center_split->set_dragger_visibility(panel_collapsed ? SplitContainer::DRAGGER_HIDDEN : SplitContainer::DRAGGER_VISIBLE);
	center_split->set_collapsed(panel_collapsed);

	pin_button->set_visible(!panel_collapsed);
	expand_button->set_visible(!panel_collapsed);
	if (expand_button->is_pressed()) {
		EditorNode::get_top_split()->set_visible(panel_collapsed);
	}

	if (panel_collapsed) {
		// Hide panel when not showing anything.
		remove_theme_style_override(SceneStringName(panel));
	} else {
		add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles)));
	}
}

void EditorBottomPanel::save_layout_to_config(Ref<ConfigFile> p_config_file, const String &p_section) const {
	p_config_file->set_value(p_section, "selected_bottom_panel_item", get_current_tab() != -1 ? Variant(get_current_tab()) : Variant());
}

void EditorBottomPanel::load_layout_from_config(Ref<ConfigFile> p_config_file, const String &p_section) {
	if (p_config_file->has_section_key(p_section, "selected_bottom_panel_item")) {
		int stored_current_tab = p_config_file->get_value(p_section, "selected_bottom_panel_item");

		if (stored_current_tab >= 0 && stored_current_tab < get_tab_bar()->get_tab_count()) {
			// Make sure we don't try to open contextual editors which are not enabled in the current context.
			if (!get_tab_bar()->is_tab_hidden(stored_current_tab)) {
				set_current_tab(stored_current_tab);
				return;
			}
		}
	}

	// If there is no active tab we need to collapse the panel.
	set_current_tab(-1);
}

void EditorBottomPanel::make_item_visible(Control *p_item, bool p_visible, bool p_ignore_lock) {
	// Don't allow changing tabs involuntarily when tabs are locked.
	if (!p_ignore_lock && lock_panel_switching && pin_button->is_visible()) {
		return;
	}

	p_item->set_visible(p_visible);
}

void EditorBottomPanel::move_item_to_end(Control *p_item) {
	move_child(p_item, -1);
}

void EditorBottomPanel::hide_bottom_panel() {
	set_current_tab(-1);
}

void EditorBottomPanel::toggle_last_opened_bottom_panel() {
	set_current_tab(get_current_tab() == -1 ? get_previous_tab() : -1);
}

void EditorBottomPanel::shortcut_input(const Ref<InputEvent> &p_event) {
	if (p_event.is_null() || !p_event->is_pressed() || p_event->is_echo()) {
		return;
	}

	for (uint32_t i = 0; i < dock_shortcuts.size(); i++) {
		if (dock_shortcuts[i].is_valid() && dock_shortcuts[i]->matches_event(p_event)) {
			bottom_docks[i]->set_visible(!bottom_docks[i]->is_visible());
			break;
		}
	}
}

void EditorBottomPanel::_pin_button_toggled(bool p_pressed) {
	lock_panel_switching = p_pressed;
}

void EditorBottomPanel::set_expanded(bool p_expanded) {
	expand_button->set_pressed(p_expanded);
}

void EditorBottomPanel::_expand_button_toggled(bool p_pressed) {
	EditorNode::get_top_split()->set_visible(!p_pressed);
}

Button *EditorBottomPanel::add_item(String p_text, Control *p_item, const Ref<Shortcut> &p_shortcut, bool p_at_front) {
	p_item->set_name(p_text);
	add_child(p_item);
	if (p_at_front) {
		move_child(p_item, 0);
	}
	bottom_docks.push_back(p_item);
	dock_shortcuts.push_back(p_shortcut);

	set_process_shortcut_input(is_processing_shortcut_input() || p_shortcut.is_valid());

	// Still return a dummy button for compatibility reasons.
	Button *tb = memnew(Button);
	tb->set_toggle_mode(true);
	tb->connect(SceneStringName(visibility_changed), callable_mp(this, &EditorBottomPanel::_on_button_visibility_changed).bind(tb, p_item));
	legacy_buttons.push_back(tb);
	return tb;
}

void EditorBottomPanel::remove_item(Control *p_item) {
	int item_idx = bottom_docks.find(p_item);
	ERR_FAIL_COND_MSG(item_idx == -1, vformat("Cannot remove unknown dock \"%s\" from the bottom panel.", p_item->get_name()));

	bottom_docks.remove_at(item_idx);
	dock_shortcuts.remove_at(item_idx);

	legacy_buttons[item_idx]->queue_free();
	legacy_buttons.remove_at(item_idx);

	remove_child(p_item);
}

void EditorBottomPanel::_on_button_visibility_changed(Button *p_button, Control *p_control) {
	int tab_index = get_tab_idx_from_control(p_control);
	if (tab_index == -1) {
		return;
	}

	// Ignore the tab if the button is hidden.
	get_tab_bar()->set_tab_hidden(tab_index, !p_button->is_visible());
}

EditorBottomPanel::EditorBottomPanel() {
	get_tab_bar()->connect(SceneStringName(gui_input), callable_mp(EditorDockManager::get_singleton(), &EditorDockManager::_dock_container_gui_input).bind(this));
	get_tab_bar()->connect("tab_changed", callable_mp(this, &EditorBottomPanel::_on_tab_changed));
	set_custom_minimum_size(Size2(400 * EDSCALE, 0));
	set_tabs_position(TabPosition::POSITION_BOTTOM);
	set_deselect_enabled(true);

	bottom_hbox = memnew(HBoxContainer);
	bottom_hbox->set_anchors_and_offsets_preset(Control::PRESET_RIGHT_WIDE);
	bottom_hbox->set_h_grow_direction(Control::GROW_DIRECTION_END);
	get_tab_bar()->add_child(bottom_hbox);

	editor_toaster = memnew(EditorToaster);
	bottom_hbox->add_child(editor_toaster);

	EditorVersionButton *version_btn = memnew(EditorVersionButton(EditorVersionButton::FORMAT_BASIC));
	// Fade out the version label to be less prominent, but still readable.
	version_btn->set_self_modulate(Color(1, 1, 1, 0.65));
	version_btn->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	bottom_hbox->add_child(version_btn);

	// Add a dummy control node for horizontal spacing.
	Control *h_spacer = memnew(Control);
	bottom_hbox->add_child(h_spacer);

	pin_button = memnew(Button);
	bottom_hbox->add_child(pin_button);
	pin_button->hide();
	pin_button->set_theme_type_variation("BottomPanelButton");
	pin_button->set_toggle_mode(true);
	pin_button->set_tooltip_text(TTRC("Pin Bottom Panel Switching"));
	pin_button->connect(SceneStringName(toggled), callable_mp(this, &EditorBottomPanel::_pin_button_toggled));

	expand_button = memnew(Button);
	bottom_hbox->add_child(expand_button);
	expand_button->hide();
	expand_button->set_theme_type_variation("BottomPanelButton");
	expand_button->set_toggle_mode(true);
	expand_button->set_accessibility_name(TTRC("Expand Bottom Panel"));
	expand_button->set_shortcut(ED_SHORTCUT_AND_COMMAND("editor/bottom_panel_expand", TTRC("Expand Bottom Panel"), KeyModifierMask::SHIFT | Key::F12));
	expand_button->connect(SceneStringName(toggled), callable_mp(this, &EditorBottomPanel::_expand_button_toggled));

	callable_mp(this, &EditorBottomPanel::_repaint).call_deferred();
}

EditorBottomPanel::~EditorBottomPanel() {
	for (Button *b : legacy_buttons) {
		memdelete(b);
	}
}
