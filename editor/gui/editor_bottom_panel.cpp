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
#include "editor/themes/editor_scale.h"
#include "editor/themes/editor_theme_manager.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/split_container.h"
#include "scene/resources/style_box_flat.h"

void EditorBottomPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			pin_button->set_button_icon(get_editor_theme_icon(SNAME("Pin")));
			expand_button->set_button_icon(get_editor_theme_icon(SNAME("ExpandBottomDock")));
			button_menu->set_button_icon(get_editor_theme_icon(SNAME("GuiTabMenuHl")));

			if (get_theme_color(SNAME("base_color"), EditorStringName(Editor)) != Color(0, 0, 0)) {
				stylebox_shadow_left->set_border_color(get_theme_color(SNAME("dark_color_1")) * Color(1, 1, 1, EditorThemeManager::is_dark_theme() ? 0.4 : 0.3));
				stylebox_shadow_right->set_border_color(get_theme_color(SNAME("dark_color_1")) * Color(1, 1, 1, EditorThemeManager::is_dark_theme() ? 0.4 : 0.3));
			} else {
				stylebox_shadow_left->set_border_color(Color(0.3, 0.3, 0.3, 0.45));
				stylebox_shadow_right->set_border_color(Color(0.3, 0.3, 0.3, 0.45));
			}
		} break;
	}
}

void EditorBottomPanel::_switch_by_control(bool p_visible, Control *p_control, bool p_ignore_lock) {
	for (int i = 0; i < items.size(); i++) {
		if (items[i].control == p_control) {
			callable_mp(this, &EditorBottomPanel::_switch_to_item).call_deferred(p_visible, i, p_ignore_lock);
			return;
		}
	}
}

void EditorBottomPanel::_switch_to_item(bool p_visible, int p_idx, bool p_ignore_lock) {
	ERR_FAIL_INDEX(p_idx, items.size());

	if (get_tree()->is_connected("process_frame", callable_mp(this, &EditorBottomPanel::_focus_pressed_button))) {
		get_tree()->disconnect("process_frame", callable_mp(this, &EditorBottomPanel::_focus_pressed_button));
	}

	if (items[p_idx].control->is_visible() == p_visible) {
		get_tree()->connect("process_frame", callable_mp(this, &EditorBottomPanel::_focus_pressed_button).bind(items[p_idx].button->get_instance_id()), CONNECT_ONE_SHOT);
		return;
	}

	SplitContainer *center_split = Object::cast_to<SplitContainer>(get_parent());
	ERR_FAIL_NULL(center_split);

	if (p_visible) {
		if (!p_ignore_lock && lock_panel_switching && pin_button->is_visible()) {
			return;
		}

		for (int i = 0; i < items.size(); i++) {
			items[i].button->set_pressed_no_signal(i == p_idx);
			items[i].control->set_visible(i == p_idx);
		}
		if (EditorDebuggerNode::get_singleton() == items[p_idx].control) {
			// This is the debug panel which uses tabs, so the top section should be smaller.
			add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("BottomPanelDebuggerOverride"), EditorStringName(EditorStyles)));
		} else {
			add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles)));
		}
		get_tree()->connect("process_frame", callable_mp(this, &EditorBottomPanel::_focus_pressed_button).bind(items[p_idx].button->get_instance_id()), CONNECT_ONE_SHOT);
		center_split->set_dragger_visibility(SplitContainer::DRAGGER_VISIBLE);
		center_split->set_collapsed(false);
		pin_button->show();

		expand_button->show();
		if (expand_button->is_pressed()) {
			EditorNode::get_top_split()->hide();
		}
	} else {
		add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles)));
		items[p_idx].button->set_pressed_no_signal(false);
		items[p_idx].control->set_visible(false);
		center_split->set_dragger_visibility(SplitContainer::DRAGGER_HIDDEN);
		center_split->set_collapsed(true);
		pin_button->hide();

		expand_button->hide();
		if (expand_button->is_pressed()) {
			EditorNode::get_top_split()->show();
		}
	}

	last_opened_control = items[p_idx].control;
}

void EditorBottomPanel::_pin_button_toggled(bool p_pressed) {
	lock_panel_switching = p_pressed;
}

void EditorBottomPanel::_expand_button_toggled(bool p_pressed) {
	EditorNode::get_top_split()->set_visible(!p_pressed);
}

bool EditorBottomPanel::_button_drag_hover(const Vector2 &, const Variant &, Button *p_button, Control *p_control) {
	if (!p_button->is_pressed()) {
		_switch_by_control(true, p_control, true);
	}
	return false;
}

void EditorBottomPanel::_focus_pressed_button(ObjectID btn_id) {
	Object *btn_instance = ObjectDB::get_instance(btn_id);
	if (!btn_instance) {
		return;
	}
	button_sc->ensure_control_visible(Object::cast_to<Button>(btn_instance));
}

void EditorBottomPanel::_add_item_to_popup() {
	button_popup_menu->clear();
	for (int i = 0; i < button_hbox->get_child_count(); i++) {
		Button *btn = Object::cast_to<Button>(button_hbox->get_child(i));
		if (!btn->is_visible()) {
			continue;
		}
		button_popup_menu->add_item(btn->get_text());
		button_popup_menu->set_item_metadata(button_popup_menu->get_item_count() - 1, btn->get_index());
	}
}

void EditorBottomPanel::_popup_position() {
	if (button_popup_menu->is_visible()) {
		button_popup_menu->hide();
		return;
	}
	button_popup_menu->reset_size();
	Vector2 popup_pos = button_menu->get_screen_rect().position - Vector2(0, button_popup_menu->get_size().y);
	if (is_layout_rtl()) {
		popup_pos.x -= button_popup_menu->get_size().x - button_menu->get_size().x;
	}
	button_popup_menu->set_position(popup_pos);
	button_popup_menu->popup();
}

void EditorBottomPanel::_popup_item_pressed(int p_idx) {
	Button *btn = Object::cast_to<Button>(button_hbox->get_child(button_popup_menu->get_item_metadata(p_idx)));
	btn->emit_signal(SceneStringName(toggled), true);
}

void EditorBottomPanel::_set_button_sc_controls() {
	if (button_hbox->get_size().x < button_sc->get_size().x) {
		button_menu->set_visible(false);
		shadow_spacer->set_visible(false);
		shadow_panel_left->add_theme_style_override(SceneStringName(panel), stylebox_shadow_empty);
		shadow_panel_right->add_theme_style_override(SceneStringName(panel), stylebox_shadow_empty);
	} else {
		button_menu->set_visible(true);
		shadow_spacer->set_visible(true);
		if (is_layout_rtl()) {
			if (button_sc->get_h_scroll_bar()->get_value() + 3 >= button_sc->get_h_scroll_bar()->get_max() - button_sc->get_size().x) {
				shadow_panel_left->add_theme_style_override(SceneStringName(panel), stylebox_shadow_empty);
			} else {
				shadow_panel_left->add_theme_style_override(SceneStringName(panel), stylebox_shadow_left);
			}
			if (button_sc->get_h_scroll_bar()->get_value() == 0) {
				shadow_panel_right->add_theme_style_override(SceneStringName(panel), stylebox_shadow_empty);
			} else {
				shadow_panel_right->add_theme_style_override(SceneStringName(panel), stylebox_shadow_right);
			}
		} else {
			if (button_sc->get_h_scroll_bar()->get_value() == 0) {
				shadow_panel_left->add_theme_style_override(SceneStringName(panel), stylebox_shadow_empty);
			} else {
				shadow_panel_left->add_theme_style_override(SceneStringName(panel), stylebox_shadow_left);
			}
			if (button_sc->get_h_scroll_bar()->get_value() + 3 >= button_sc->get_h_scroll_bar()->get_max() - button_sc->get_size().x) {
				shadow_panel_right->add_theme_style_override(SceneStringName(panel), stylebox_shadow_empty);
			} else {
				shadow_panel_right->add_theme_style_override(SceneStringName(panel), stylebox_shadow_right);
			}
		}
	}
}

void EditorBottomPanel::_sc_input(const Ref<InputEvent> &p_gui_input) {
	Ref<InputEventMouseButton> mb = p_gui_input;

	if (mb.is_valid()) {
		if (mb->is_pressed()) {
			if ((mb->get_button_index() == MouseButton::WHEEL_UP && mb->is_alt_pressed()) || (mb->get_button_index() == MouseButton::WHEEL_DOWN && mb->is_alt_pressed())) {
				mb->set_factor(mb->get_factor() * 7);
			}
		}
	}
}

void EditorBottomPanel::save_layout_to_config(Ref<ConfigFile> p_config_file, const String &p_section) const {
	int selected_item_idx = -1;
	for (int i = 0; i < items.size(); i++) {
		if (items[i].button->is_pressed()) {
			selected_item_idx = i;
			break;
		}
	}
	if (selected_item_idx != -1) {
		p_config_file->set_value(p_section, "selected_bottom_panel_item", selected_item_idx);
	} else {
		p_config_file->set_value(p_section, "selected_bottom_panel_item", Variant());
	}
}

void EditorBottomPanel::load_layout_from_config(Ref<ConfigFile> p_config_file, const String &p_section) {
	bool has_active_tab = false;
	if (p_config_file->has_section_key(p_section, "selected_bottom_panel_item")) {
		int selected_item_idx = p_config_file->get_value(p_section, "selected_bottom_panel_item");
		if (selected_item_idx >= 0 && selected_item_idx < items.size()) {
			// Make sure we don't try to open contextual editors which are not enabled in the current context.
			if (items[selected_item_idx].button->is_visible()) {
				_switch_to_item(true, selected_item_idx);
				has_active_tab = true;
			}
		}
	}
	// If there is no active tab we need to collapse the panel.
	if (!has_active_tab) {
		items[0].control->show(); // _switch_to_item() can collapse only visible tabs.
		_switch_to_item(false, 0);
	}
}

Button *EditorBottomPanel::add_item(String p_text, Control *p_item, const Ref<Shortcut> &p_shortcut, bool p_at_front) {
	Button *tb = memnew(Button);
	tb->set_theme_type_variation("BottomPanelButton");
	tb->connect(SceneStringName(toggled), callable_mp(this, &EditorBottomPanel::_switch_by_control).bind(p_item, true), CONNECT_DEFERRED);
	tb->set_drag_forwarding(Callable(), callable_mp(this, &EditorBottomPanel::_button_drag_hover).bind(tb, p_item), Callable());
	tb->set_text(p_text);
	tb->set_shortcut(p_shortcut);
	tb->set_toggle_mode(true);
	tb->set_focus_mode(Control::FOCUS_NONE);
	item_vbox->add_child(p_item);

	bottom_hbox->move_to_front();
	button_hbox->add_child(tb);
	if (p_at_front) {
		button_hbox->move_child(tb, 0);
	}
	p_item->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	p_item->hide();

	BottomPanelItem bpi;
	bpi.button = tb;
	bpi.control = p_item;
	bpi.name = p_text;
	if (p_at_front) {
		items.insert(0, bpi);
	} else {
		items.push_back(bpi);
	}

	return tb;
}

void EditorBottomPanel::remove_item(Control *p_item) {
	bool was_visible = false;
	for (int i = 0; i < items.size(); i++) {
		if (items[i].control == p_item) {
			if (p_item->is_visible_in_tree()) {
				was_visible = true;
			}
			item_vbox->remove_child(items[i].control);
			button_hbox->remove_child(items[i].button);
			memdelete(items[i].button);
			items.remove_at(i);
			break;
		}
	}

	if (was_visible) {
		// Open the first panel to ensure that if the removed dock was visible, the bottom
		// panel will not collapse.
		_switch_to_item(true, 0);
	} else if (last_opened_control == p_item) {
		// When a dock is removed by plugins, it might not have been visible, and it
		// might have been the last_opened_control. We need to make sure to reset the last opened control.
		last_opened_control = items[0].control;
	}
}

void EditorBottomPanel::make_item_visible(Control *p_item, bool p_visible) {
	_switch_by_control(p_visible, p_item);
}

void EditorBottomPanel::move_item_to_end(Control *p_item) {
	for (int i = 0; i < items.size(); i++) {
		if (items[i].control == p_item) {
			items[i].button->move_to_front();
			SWAP(items.write[i], items.write[items.size() - 1]);
			break;
		}
	}
}

void EditorBottomPanel::hide_bottom_panel() {
	for (int i = 0; i < items.size(); i++) {
		if (items[i].control->is_visible()) {
			_switch_to_item(false, i);
			break;
		}
	}
}

void EditorBottomPanel::toggle_last_opened_bottom_panel() {
	// Select by control instead of index, so that the last bottom panel is opened correctly
	// if it's been reordered since.
	if (last_opened_control) {
		_switch_by_control(!last_opened_control->is_visible(), last_opened_control, true);
	} else {
		// Open the first panel in the list if no panel was opened this session.
		_switch_to_item(true, 0, true);
	}
}

EditorBottomPanel::EditorBottomPanel() {
	item_vbox = memnew(VBoxContainer);
	add_child(item_vbox);

	bottom_hbox = memnew(HBoxContainer);
	bottom_hbox->set_custom_minimum_size(Size2(0, 24 * EDSCALE)); // Adjust for the height of the "Expand Bottom Dock" icon.
	item_vbox->add_child(bottom_hbox);

	button_tab_bar_hbox = memnew(HBoxContainer);
	button_tab_bar_hbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	button_tab_bar_hbox->add_theme_constant_override("separation", 0);
	bottom_hbox->add_child(button_tab_bar_hbox);

	button_menu = memnew(Button);
	button_menu->set_theme_type_variation("FlatMenuButton");
	button_menu->set_toggle_mode(true);
	button_menu->set_action_mode(BaseButton::ACTION_MODE_BUTTON_PRESS);
	button_menu->set_tooltip_text(TTR("List all options."));
	button_menu->connect(SceneStringName(pressed), callable_mp(this, &EditorBottomPanel::_popup_position));
	button_tab_bar_hbox->add_child(button_menu);

	button_popup_menu = memnew(PopupMenu);
	add_child(button_popup_menu);
	button_popup_menu->connect(SceneStringName(id_pressed), callable_mp(this, &EditorBottomPanel::_popup_item_pressed));
	button_popup_menu->connect("popup_hide", callable_mp((BaseButton *)button_menu, &BaseButton::set_pressed).bind(false));
	button_popup_menu->connect("about_to_popup", callable_mp((BaseButton *)button_menu, &BaseButton::set_pressed).bind(true));

	shadow_spacer = memnew(Control);
	shadow_spacer->set_custom_minimum_size(Size2(2 * EDSCALE, 0));
	button_tab_bar_hbox->add_child(shadow_spacer);

	stylebox_shadow_empty.instantiate();
	stylebox_shadow_left.instantiate();
	stylebox_shadow_left->set_draw_center(false);
	stylebox_shadow_left->set_border_blend(true);
	stylebox_shadow_left->set_border_width_all(0);
	stylebox_shadow_right = stylebox_shadow_left->duplicate();
	stylebox_shadow_left->set_border_width(Side::SIDE_LEFT, 3 * EDSCALE);
	stylebox_shadow_left->set_expand_margin(Side::SIDE_LEFT, 1);
	stylebox_shadow_right->set_border_width(Side::SIDE_RIGHT, 3 * EDSCALE);
	stylebox_shadow_right->set_expand_margin(Side::SIDE_RIGHT, 1);

	shadow_panel_left = memnew(Panel);
	shadow_panel_left->set_custom_minimum_size(Size2(4 * EDSCALE, 0));
	shadow_panel_left->add_theme_style_override(SceneStringName(panel), stylebox_shadow_left);
	button_tab_bar_hbox->add_child(shadow_panel_left);

	button_sc = memnew(ScrollContainer);
	button_sc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	button_sc->set_follow_focus(true);
	button_sc->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_SHOW_NEVER);
	button_sc->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	button_sc->connect(SceneStringName(draw), callable_mp(this, &EditorBottomPanel::_set_button_sc_controls));
	button_sc->connect(SceneStringName(gui_input), callable_mp(this, &EditorBottomPanel::_sc_input));
	button_tab_bar_hbox->add_child(button_sc);

	button_hbox = memnew(HBoxContainer);
	button_sc->add_child(button_hbox);
	button_hbox->connect(SceneStringName(resized), callable_mp(this, &EditorBottomPanel::_add_item_to_popup));

	shadow_panel_right = memnew(Panel);
	shadow_panel_right->set_custom_minimum_size(Size2(4 * EDSCALE, 0));
	shadow_panel_right->add_theme_style_override(SceneStringName(panel), stylebox_shadow_right);
	button_tab_bar_hbox->add_child(shadow_panel_right);

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
	pin_button->connect(SceneStringName(toggled), callable_mp(this, &EditorBottomPanel::_pin_button_toggled));

	expand_button = memnew(Button);
	bottom_hbox->add_child(expand_button);
	expand_button->hide();
	expand_button->set_theme_type_variation("FlatMenuButton");
	expand_button->set_toggle_mode(true);
	expand_button->set_shortcut(ED_SHORTCUT_AND_COMMAND("editor/bottom_panel_expand", TTR("Expand Bottom Panel"), KeyModifierMask::SHIFT | Key::F12));
	expand_button->connect(SceneStringName(toggled), callable_mp(this, &EditorBottomPanel::_expand_button_toggled));
}
