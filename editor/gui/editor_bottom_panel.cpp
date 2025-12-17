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
#include "editor/docks/editor_dock.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_toaster.h"
#include "editor/gui/editor_version_button.h"
#include "editor/scene/editor_scene_tabs.h"
#include "editor/settings/editor_command_palette.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"

void EditorBottomPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			layout_popup = get_popup();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			pin_button->set_button_icon(get_editor_theme_icon(SNAME("Pin")));
			expand_button->set_button_icon(get_editor_theme_icon(SNAME("ExpandBottomDock")));
		} break;
	}
}

void EditorBottomPanel::_on_tab_changed(int p_idx) {
	_update_center_split_offset();
	_repaint();
}

void EditorBottomPanel::_theme_changed() {
	int icon_width = get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor));
	int margin = bottom_hbox->get_minimum_size().width;
	if (get_popup()) {
		margin -= icon_width;
	}

	// Add margin to make space for the right side popup button.
	icon_spacer->set_custom_minimum_size(Vector2(icon_width, 0));

	// Need to get stylebox from EditorNode to update theme correctly.
	Ref<StyleBox> bottom_tabbar_style = EditorNode::get_singleton()->get_editor_theme()->get_stylebox(SNAME("tabbar_background"), SNAME("BottomPanel"))->duplicate();
	bottom_tabbar_style->set_content_margin(is_layout_rtl() ? SIDE_LEFT : SIDE_RIGHT, margin + bottom_tabbar_style->get_content_margin(is_layout_rtl() ? SIDE_RIGHT : SIDE_LEFT));
	add_theme_style_override("tabbar_background", bottom_tabbar_style);

	if (get_current_tab() == -1) {
		// Hide panel when not showing anything.
		remove_theme_style_override(SceneStringName(panel));
	} else {
		add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles)));
	}
}

void EditorBottomPanel::set_bottom_panel_offset(int p_offset) {
	EditorDock *current_tab = Object::cast_to<EditorDock>(get_current_tab_control());
	if (current_tab) {
		dock_offsets[current_tab->get_effective_layout_key()] = p_offset;
	}
}

int EditorBottomPanel::get_bottom_panel_offset() {
	EditorDock *current_tab = Object::cast_to<EditorDock>(get_current_tab_control());
	if (current_tab) {
		return dock_offsets[current_tab->get_effective_layout_key()];
	}
	return 0;
}

void EditorBottomPanel::_repaint() {
	bool panel_collapsed = get_current_tab() == -1;

	if (panel_collapsed && get_popup()) {
		set_popup(nullptr);
	} else if (!panel_collapsed && !get_popup()) {
		set_popup(layout_popup);
	}
	if (!panel_collapsed && (previous_tab != -1)) {
		return;
	}
	previous_tab = get_current_tab();

	DockSplitContainer *center_split = EditorNode::get_center_split();
	ERR_FAIL_NULL(center_split);

	center_split->set_dragger_visibility(panel_collapsed ? SplitContainer::DRAGGER_HIDDEN : SplitContainer::DRAGGER_VISIBLE);
	center_split->set_collapsed(panel_collapsed);

	pin_button->set_visible(!panel_collapsed);
	expand_button->set_visible(!panel_collapsed);
	if (expand_button->is_pressed()) {
		_expand_button_toggled(!panel_collapsed);
	} else {
		_theme_changed();
	}
}

void EditorBottomPanel::save_layout_to_config(Ref<ConfigFile> p_config_file, const String &p_section) const {
	Dictionary offsets;
	for (const KeyValue<String, int> &E : dock_offsets) {
		offsets[E.key] = E.value;
	}
	p_config_file->set_value(p_section, "bottom_panel_offsets", offsets);
}

void EditorBottomPanel::load_layout_from_config(Ref<ConfigFile> p_config_file, const String &p_section) {
	const Dictionary offsets = p_config_file->get_value(p_section, "bottom_panel_offsets", Dictionary());
	const LocalVector<Variant> offset_list = offsets.get_key_list();

	for (const Variant &v : offset_list) {
		dock_offsets[v] = offsets[v];
	}
	_update_center_split_offset();
}

void EditorBottomPanel::make_item_visible(Control *p_item, bool p_visible, bool p_ignore_lock) {
	// Don't allow changing tabs involuntarily when tabs are locked.
	if (!p_ignore_lock && lock_panel_switching && pin_button->is_visible()) {
		return;
	}

	EditorDock *dock = _get_dock_from_control(p_item);
	ERR_FAIL_NULL(dock);
	dock->set_visible(p_visible);
}

void EditorBottomPanel::hide_bottom_panel() {
	set_current_tab(-1);
}

void EditorBottomPanel::toggle_last_opened_bottom_panel() {
	set_current_tab(get_current_tab() == -1 ? get_previous_tab() : -1);
}

void EditorBottomPanel::_pin_button_toggled(bool p_pressed) {
	lock_panel_switching = p_pressed;
}

void EditorBottomPanel::set_expanded(bool p_expanded) {
	expand_button->set_pressed(p_expanded);
}

void EditorBottomPanel::_expand_button_toggled(bool p_pressed) {
	EditorNode::get_top_split()->set_visible(!p_pressed);

	Button *distraction_free = EditorNode::get_singleton()->get_distraction_free_button();
	distraction_free->set_meta("_scene_tabs_owned", !p_pressed);
	EditorNode::get_singleton()->update_distraction_free_button_theme();
	if (p_pressed) {
		distraction_free->reparent(bottom_hbox);
		bottom_hbox->move_child(distraction_free, -2);
	} else {
		distraction_free->get_parent()->remove_child(distraction_free);
		EditorSceneTabs::get_singleton()->add_extra_button(distraction_free);
	}
	_theme_changed();
}

void EditorBottomPanel::_update_center_split_offset() {
	DockSplitContainer *center_split = EditorNode::get_center_split();
	ERR_FAIL_NULL(center_split);

	center_split->set_split_offset(get_bottom_panel_offset());
}

EditorDock *EditorBottomPanel::_get_dock_from_control(Control *p_control) const {
	return Object::cast_to<EditorDock>(p_control->get_parent());
}

Button *EditorBottomPanel::add_item(String p_text, Control *p_item, const Ref<Shortcut> &p_shortcut, bool p_at_front) {
	EditorDock *dock = memnew(EditorDock);
	dock->add_child(p_item);
	dock->set_title(p_text);
	dock->set_dock_shortcut(p_shortcut);
	dock->set_global(false);
	dock->set_transient(true);
	dock->set_default_slot(DockConstants::DOCK_SLOT_BOTTOM);
	dock->set_available_layouts(EditorDock::DOCK_LAYOUT_HORIZONTAL);
	EditorDockManager::get_singleton()->add_dock(dock);
	bottom_docks.push_back(dock);

	p_item->show(); // Compatibility in case it was hidden.

	// Still return a dummy button for compatibility reasons.
	Button *tb = memnew(Button);
	tb->set_toggle_mode(true);
	tb->connect(SceneStringName(visibility_changed), callable_mp(this, &EditorBottomPanel::_on_button_visibility_changed).bind(tb, dock));
	legacy_buttons.push_back(tb);
	return tb;
}

void EditorBottomPanel::remove_item(Control *p_item) {
	EditorDock *dock = _get_dock_from_control(p_item);
	ERR_FAIL_NULL_MSG(dock, vformat("Cannot remove unknown dock \"%s\" from the bottom panel.", p_item->get_name()));

	int item_idx = bottom_docks.find(dock);
	ERR_FAIL_COND(item_idx == -1);

	bottom_docks.remove_at(item_idx);

	legacy_buttons[item_idx]->queue_free();
	legacy_buttons.remove_at(item_idx);

	EditorDockManager::get_singleton()->remove_dock(dock);
	dock->remove_child(p_item);
	dock->queue_free();
}

void EditorBottomPanel::_on_button_visibility_changed(Button *p_button, EditorDock *p_dock) {
	if (p_button->is_visible()) {
		p_dock->open();
	} else {
		p_dock->close();
	}
}

EditorBottomPanel::EditorBottomPanel() {
	get_tab_bar()->connect("tab_changed", callable_mp(this, &EditorBottomPanel::_on_tab_changed));
	set_tabs_position(TabPosition::POSITION_BOTTOM);
	set_deselect_enabled(true);

	bottom_hbox = memnew(HBoxContainer);
	bottom_hbox->set_mouse_filter(MOUSE_FILTER_IGNORE);
	bottom_hbox->set_anchors_and_offsets_preset(Control::PRESET_RIGHT_WIDE);
	get_tab_bar()->add_child(bottom_hbox);

	icon_spacer = memnew(Control);
	icon_spacer->set_mouse_filter(MOUSE_FILTER_IGNORE);
	bottom_hbox->add_child(icon_spacer);

	bottom_hbox->add_child(memnew(VSeparator));

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
