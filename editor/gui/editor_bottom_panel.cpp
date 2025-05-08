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
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_toaster.h"
#include "editor/gui/editor_version_button.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/split_container.h"

Size2 EditorBottomPanel::get_minimum_size() const {
	return back_panel->get_minimum_size() + Size2(0, (get_current_tab() == -1 ? 0 : get_current_tab_control()->get_minimum_size().y + tab_offset));
}

int EditorBottomPanel::_get_tab_height() const {
	return bottom_hbox->get_minimum_size().y + (get_current_tab() == -1 ? 0 : tab_offset);
}

void EditorBottomPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			pin_button->set_button_icon(get_editor_theme_icon(SNAME("Pin")));
			expand_button->set_button_icon(get_editor_theme_icon(SNAME("ExpandBottomDock")));
			left_button->set_button_icon(get_editor_theme_icon(SNAME("Back")));
			right_button->set_button_icon(get_editor_theme_icon(SNAME("Forward")));
		}
			[[fallthrough]];
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			tab_offset = back_panel->get_theme_stylebox(SceneStringName(panel))->get_margin(SIDE_BOTTOM) + (int)EditorSettings::get_singleton()->get_setting("interface/theme/additional_spacing");
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			if (is_layout_rtl()) {
				bottom_hbox->move_child(left_button, button_scroll->get_index() + 1);
				bottom_hbox->move_child(right_button, 0);
			} else {
				bottom_hbox->move_child(right_button, button_scroll->get_index() + 1);
				bottom_hbox->move_child(left_button, 0);
			}
		} break;
	}
}

void EditorBottomPanel::_scroll(bool p_right) {
	HScrollBar *h_scroll = button_scroll->get_h_scroll_bar();
	if (Input::get_singleton()->is_key_pressed(Key::CTRL)) {
		h_scroll->set_value(p_right ? h_scroll->get_max() : 0);
	} else if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		h_scroll->set_value(h_scroll->get_value() + h_scroll->get_page() * (p_right ? 1 : -1));
	} else {
		h_scroll->set_value(h_scroll->get_value() + (h_scroll->get_page() * 0.5) * (p_right ? 1 : -1));
	}
}

void EditorBottomPanel::_update_scroll_buttons() {
	bool show_arrows = button_hbox->get_size().width > button_scroll->get_size().width;
	left_button->set_visible(show_arrows);
	right_button->set_visible(show_arrows);

	if (show_arrows) {
		_update_disabled_buttons();
	}
}

void EditorBottomPanel::_update_disabled_buttons() {
	HScrollBar *h_scroll = button_scroll->get_h_scroll_bar();
	left_button->set_disabled(h_scroll->get_value() == 0);
	right_button->set_disabled(h_scroll->get_value() + h_scroll->get_page() == h_scroll->get_max());
}

void EditorBottomPanel::_on_tab_changed(int p_idx) {
	callable_mp(this, &EditorBottomPanel::_repaint).call_deferred();
}

void EditorBottomPanel::_on_button_visibility_changed(Button *p_button, Control *p_control) {
	int tab_index = get_tab_idx_from_control(p_control);
	if (tab_index == -1) {
		return;
	}

	// Ignore the tab if the button is hidden.
	get_tab_bar()->set_tab_hidden(tab_index, !p_button->is_visible());
}

void EditorBottomPanel::_repaint() {
	int current_idx = get_current_tab();
	Vector<Control *> tab_controls = _get_tab_controls();
	Button *current_button = nullptr;

	// Update button toggles.
	for (int i = 0; i < tab_controls.size(); i++) {
		Button *tb = static_cast<Button *>((Object *)tab_controls[i]->get_meta("_editor_bottom_button"));
		if (tb) {
			tb->set_pressed_no_signal(i == current_idx);
			if (i == current_idx) {
				current_button = tb;
			}
		}
	}

	SplitContainer *center_split = Object::cast_to<SplitContainer>(get_parent());
	ERR_FAIL_NULL(center_split);

	if (current_idx == -1) {
		add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles)));
		center_split->set_dragger_visibility(SplitContainer::DRAGGER_HIDDEN);
		center_split->set_collapsed(true);
		pin_button->hide();

		expand_button->hide();
		if (expand_button->is_pressed()) {
			EditorNode::get_top_split()->show();
		}
	} else {
		if (EditorDebuggerNode::get_singleton() == tab_controls[current_idx]) {
			// This is the debug panel which uses tabs, so the top section should be smaller.
			add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("BottomPanelDebuggerOverride"), EditorStringName(EditorStyles)));
		} else {
			add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles)));
		}

		center_split->set_dragger_visibility(SplitContainer::DRAGGER_VISIBLE);
		center_split->set_collapsed(false);
		pin_button->show();

		expand_button->show();
		if (expand_button->is_pressed()) {
			EditorNode::get_top_split()->hide();
		}
		callable_mp(button_scroll, &ScrollContainer::ensure_control_visible).call_deferred(current_button);
	}
}

void EditorBottomPanel::_pin_button_toggled(bool p_pressed) {
	lock_panel_switching = p_pressed;
}

void EditorBottomPanel::_expand_button_toggled(bool p_pressed) {
	EditorNode::get_top_split()->set_visible(!p_pressed);
}

bool EditorBottomPanel::_button_drag_hover(const Vector2 &, const Variant &, Button *p_button, Control *p_control) {
	if (!p_button->is_pressed()) {
		p_control->show();
	}
	return false;
}

void EditorBottomPanel::save_layout_to_config(Ref<ConfigFile> p_config_file, const String &p_section) const {
	if (get_current_tab() != -1) {
		p_config_file->set_value(p_section, "selected_bottom_panel_item", get_current_tab());
	} else {
		p_config_file->set_value(p_section, "selected_bottom_panel_item", Variant());
	}
}

void EditorBottomPanel::load_layout_from_config(Ref<ConfigFile> p_config_file, const String &p_section) {
	bool has_active_tab = false;
	if (p_config_file->has_section_key(p_section, "selected_bottom_panel_item")) {
		int stored_current_tab = p_config_file->get_value(p_section, "selected_bottom_panel_item");

		if (stored_current_tab >= 0 && stored_current_tab < get_tab_bar()->get_tab_count()) {
			// Make sure we don't try to open contextual editors which are not enabled in the current context.
			if (!get_tab_bar()->is_tab_hidden(stored_current_tab)) {
				set_current_tab(stored_current_tab);
				has_active_tab = true;
			}
		}
	}
	// If there is no active tab we need to collapse the panel.
	if (!has_active_tab) {
		set_current_tab(-1);
	}
}

Button *EditorBottomPanel::add_item(String p_text, Control *p_item, const Ref<Shortcut> &p_shortcut, bool p_at_front) {
	p_item->set_name(p_text);
	p_item->hide();
	add_child(p_item);
	if (p_at_front) {
		move_child(p_item, 0);
	}

	// add_child attaches the meta data.
	Button *tb = static_cast<Button *>((Object *)p_item->get_meta("_editor_bottom_button"));
	if (tb) {
		tb->set_shortcut(p_shortcut);
	}

	return tb;
}

void EditorBottomPanel::remove_item(Control *p_item) {
	remove_child(p_item);
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
		// Open the first panel in the list if no panel was opened this session.
		if (!select_next_available()) {
			select_previous_available();
		}
	}
}

void EditorBottomPanel::set_expanded(bool p_expanded) {
	expand_button->set_pressed(p_expanded);
}

void EditorBottomPanel::add_child_notify(Node *p_child) {
	// Don't add the bottom panel to the tabbar.
	if (p_child == back_panel || p_child == get_tab_bar()) {
		Container::add_child_notify(p_child);
		return;
	}
	TabContainer::add_child_notify(p_child);

	Control *c = as_sortable_control(p_child, SortableVisibilityMode::IGNORE);
	if (!c) {
		return;
	}
	c->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	c->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);

	Button *tb = memnew(Button);
	tb->set_theme_type_variation("BottomPanelButton");
	tb->connect(SceneStringName(toggled), callable_mp((CanvasItem *)c, &CanvasItem::set_visible));
	tb->connect(SceneStringName(visibility_changed), callable_mp(this, &EditorBottomPanel::_on_button_visibility_changed).bind(tb, c));
	tb->set_drag_forwarding(Callable(), callable_mp(this, &EditorBottomPanel::_button_drag_hover).bind(tb, c), Callable());
	tb->set_text(p_child->get_name());
	tb->set_toggle_mode(true);
	tb->set_focus_mode(Control::FOCUS_NONE);
	button_hbox->add_child(tb);

	p_child->set_meta("_editor_bottom_button", tb);

	// TabBar won't emit the "tab_changed" signal when not inside the tree.
	if (!is_inside_tree()) {
		callable_mp(this, &EditorBottomPanel::_repaint).call_deferred();
	}
}

void EditorBottomPanel::move_child_notify(Node *p_child) {
	// Don't add the bottom panel to the tabbar.
	if (p_child == back_panel || p_child == get_tab_bar()) {
		Container::move_child_notify(p_child);
		return;
	}
	TabContainer::move_child_notify(p_child);

	Vector<Control *> tab_controls = _get_tab_controls();

	// Sort custom tab buttons according to new order.
	for (int i = 0; i < tab_controls.size(); i++) {
		Node *tb = static_cast<Node *>((Object *)tab_controls[i]->get_meta("_editor_bottom_button"));
		if (tb) {
			button_hbox->move_child(tb, i);
		}
	}
}

void EditorBottomPanel::remove_child_notify(Node *p_child) {
	// Don't add the bottom panel to the tabbar.
	if (p_child == back_panel || p_child == get_tab_bar()) {
		Container::remove_child_notify(p_child);
		return;
	}

	Control *c = as_sortable_control(p_child, SortableVisibilityMode::IGNORE);
	if (!c) {
		return;
	}

	// Finally update the tabs.
	TabContainer::remove_child_notify(p_child);

	// Delete bottom button.
	Button *tb = static_cast<Button *>((Object *)p_child->get_meta("_editor_bottom_button"));
	p_child->remove_meta("_editor_bottom_button");
	button_hbox->remove_child(tb);
	memdelete(tb);

	// TabBar won't emit the "tab_changed" signal when not inside the tree.
	if (!is_inside_tree()) {
		callable_mp(this, &EditorBottomPanel::_repaint).call_deferred();
	}
}

EditorBottomPanel::EditorBottomPanel() {
	// Hacks to create a custom tab bar.
	get_tab_bar()->connect("tab_changed", callable_mp(this, &EditorBottomPanel::_on_tab_changed));
	get_tab_bar()->hide();
	set_tabs_position(TabPosition::POSITION_BOTTOM);
	set_deselect_enabled(true);
	add_theme_style_override("tabbar_background", memnew(StyleBoxEmpty));

	back_panel = memnew(PanelContainer);
	back_panel->set_custom_minimum_size(Size2(0, 24 * EDSCALE)); // Adjust for the height of the "Expand Bottom Dock" icon.
	add_child(back_panel);
	children_removing.push_back(back_panel); // Hide this panel from the logic of the TabContainer by faking it as deleted.
	back_panel->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	back_panel->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);

	bottom_hbox = memnew(HBoxContainer);
	bottom_hbox->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_WIDE);
	bottom_hbox->set_v_size_flags(Control::SIZE_SHRINK_END);
	bottom_hbox->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	back_panel->add_child(bottom_hbox);

	left_button = memnew(Button);
	left_button->set_tooltip_text(TTR("Scroll Left\nHold Ctrl to scroll to the begin.\nHold Shift to scroll one page."));
	left_button->set_accessibility_name(TTRC("Scroll Left"));
	left_button->set_theme_type_variation("BottomPanelButton");
	left_button->set_focus_mode(Control::FOCUS_NONE);
	left_button->connect(SceneStringName(pressed), callable_mp(this, &EditorBottomPanel::_scroll).bind(false));
	bottom_hbox->add_child(left_button);
	left_button->hide();

	button_scroll = memnew(ScrollContainer);
	button_scroll->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	button_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_SHOW_NEVER);
	button_scroll->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	button_scroll->get_h_scroll_bar()->connect(CoreStringName(changed), callable_mp(this, &EditorBottomPanel::_update_scroll_buttons), CONNECT_DEFERRED);
	button_scroll->get_h_scroll_bar()->connect(SceneStringName(value_changed), callable_mp(this, &EditorBottomPanel::_update_disabled_buttons).unbind(1), CONNECT_DEFERRED);
	bottom_hbox->add_child(button_scroll);

	right_button = memnew(Button);
	right_button->set_tooltip_text(TTR("Scroll Right\nHold Ctrl to scroll to the end.\nHold Shift to scroll one page."));
	right_button->set_accessibility_name(TTRC("Scroll Right"));
	right_button->set_theme_type_variation("BottomPanelButton");
	right_button->set_focus_mode(Control::FOCUS_NONE);
	right_button->connect(SceneStringName(pressed), callable_mp(this, &EditorBottomPanel::_scroll).bind(true));
	bottom_hbox->add_child(right_button);
	right_button->hide();

	callable_mp(this, &EditorBottomPanel::_update_scroll_buttons).call_deferred();

	button_hbox = memnew(HBoxContainer);
	button_hbox->set_h_size_flags(Control::SIZE_EXPAND | Control::SIZE_SHRINK_BEGIN);
	button_scroll->add_child(button_hbox);

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
	pin_button->set_theme_type_variation("FlatMenuButton");
	pin_button->set_toggle_mode(true);
	pin_button->set_tooltip_text(TTR("Pin Bottom Panel Switching"));
	pin_button->set_accessibility_name(TTRC("Pin Bottom Panel"));
	pin_button->connect(SceneStringName(toggled), callable_mp(this, &EditorBottomPanel::_pin_button_toggled));

	expand_button = memnew(Button);
	bottom_hbox->add_child(expand_button);
	expand_button->hide();
	expand_button->set_theme_type_variation("FlatMenuButton");
	expand_button->set_toggle_mode(true);
	expand_button->set_accessibility_name(TTRC("Expand Bottom Panel"));
	expand_button->set_shortcut(ED_SHORTCUT_AND_COMMAND("editor/bottom_panel_expand", TTRC("Expand Bottom Panel"), KeyModifierMask::SHIFT | Key::F12));
	expand_button->connect(SceneStringName(toggled), callable_mp(this, &EditorBottomPanel::_expand_button_toggled));
}
