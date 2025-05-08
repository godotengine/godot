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
#include "editor/editor_command_palette.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_toaster.h"
#include "editor/gui/editor_version_button.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/split_container.h"

void EditorBottomPanel::_update_margins() {
	TabContainer::_update_margins();
	get_tab_bar()->set_offset(SIDE_RIGHT, get_tab_bar()->get_offset(SIDE_RIGHT) - bottom_hbox->get_size().x - right_margin);
}

void EditorBottomPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			pin_button->set_button_icon(get_editor_theme_icon(SNAME("Pin")));
			expand_button->set_button_icon(get_editor_theme_icon(SNAME("ExpandBottomDock")));
			right_margin = get_theme_stylebox("tabbar_background")->get_margin(SIDE_RIGHT);
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
		add_theme_style_override(SceneStringName(panel), get_theme_stylebox("BottomPanelHidden", EditorStringName(EditorStyles)));
	} else {
		if (EditorDebuggerNode::get_singleton() == get_current_tab_control()) {
			// This is the debug panel which uses tabs, so the top section should be smaller.
			add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("BottomPanelDebuggerOverride"), EditorStringName(EditorStyles)));
		} else {
			add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles)));
		}
	}
}

void EditorBottomPanel::_pin_button_toggled(bool p_pressed) {
	lock_panel_switching = p_pressed;
}

void EditorBottomPanel::_expand_button_toggled(bool p_pressed) {
	EditorNode::get_top_split()->set_visible(!p_pressed);
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
	// Don't allow changing tabs involuntarily when tabs are locked
	if (!p_ignore_lock && lock_panel_switching && pin_button->is_visible()) {
		return;
	}

	if (p_visible) {
		p_item->show();
	} else {
		set_current_tab(get_previous_tab());
	}
}

void EditorBottomPanel::move_item_to_end(Control *p_item) {
	move_child(p_item, -1);
}

void EditorBottomPanel::hide_bottom_panel() {
	set_current_tab(-1);
}

void EditorBottomPanel::toggle_last_opened_bottom_panel() {
	// Select by control instead of index, so that the last bottom panel is opened correctly
	// if it's been reordered since.
	if (get_previous_tab() != -1) {
		set_current_tab(get_previous_tab() != get_current_tab() ? get_previous_tab() : -1);
	} else {
		// Try to open an adjacent panel, otherwise hide the bottom panel.
		if (!select_previous_available() && !select_next_available()) {
			set_current_tab(-1);
		}
	}
}

void EditorBottomPanel::set_expanded(bool p_expanded) {
	expand_button->set_pressed(p_expanded);
}

Button *EditorBottomPanel::add_item(String p_text, Control *p_item, const Ref<Shortcut> &p_shortcut, bool p_at_front) {
	p_item->set_name(p_text);
	add_child(p_item);
	if (p_at_front) {
		move_child(p_item, 0);
	}

	// Still return a dummy button for compatibility reasons.
	Button *tb = memnew(Button);
	dummy_buttons.insert(tb);
	tb->connect(SceneStringName(visibility_changed), callable_mp(this, &EditorBottomPanel::_on_button_visibility_changed).bind(tb, p_item));
	tb->set_shortcut(p_shortcut);

	p_item->set_meta("_editor_dummy_button", tb);
	return tb;
}

void EditorBottomPanel::remove_item(Control *p_item) {
	remove_child(p_item);

	// Delete dummy button that might have been added.
	if (p_item->has_meta("_editor_dummy_button")) {
		Button *tb = static_cast<Button *>((Object *)p_item->get_meta("_editor_dummy_button"));
		dummy_buttons.erase(tb);
		memdelete(tb);
		p_item->remove_meta("_editor_dummy_button");
	}
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
	get_tab_bar()->connect("tab_changed", callable_mp(this, &EditorBottomPanel::_on_tab_changed));
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
	pin_button->set_tooltip_text(TTR("Pin Bottom Panel Switching"));
	pin_button->set_accessibility_name(TTRC("Pin Bottom Panel"));
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
	for (Button *b : dummy_buttons) {
		memdelete(b);
	}
}
