/**************************************************************************/
/*  touch_actions_panel.cpp                                               */
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

#include "touch_actions_panel.h"

#include "core/input/input.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_settings.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/style_box_flat.h"

void TouchActionsPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			DisplayServer::get_singleton()->set_hardware_keyboard_connection_change_callback(callable_mp(this, &TouchActionsPanel::_hardware_keyboard_connected));
			_hardware_keyboard_connected(DisplayServer::get_singleton()->has_hardware_keyboard());
			if (!is_floating) {
				get_parent()->move_child(this, embedded_panel_index);
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			set_process_input(is_visible_in_tree());
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			if (is_floating) {
				drag_handle->set_texture(get_editor_theme_icon(SNAME("DragHandle")));
				layout_toggle_button->set_button_icon(get_editor_theme_icon(SNAME("Orientation")));
				lock_panel_button->set_button_icon(get_editor_theme_icon(SNAME("Lock")));
			} else {
				if (embedded_panel_index == 1) {
					panel_pos_button->set_button_icon(get_editor_theme_icon(SNAME("ControlAlignLeftWide")));
				} else {
					panel_pos_button->set_button_icon(get_editor_theme_icon(SNAME("ControlAlignRightWide")));
				}
			}
			save_button->set_button_icon(get_editor_theme_icon(SNAME("Save")));
			delete_button->set_button_icon(get_editor_theme_icon(SNAME("Remove")));
			undo_button->set_button_icon(get_editor_theme_icon(SNAME("UndoRedo")));
			redo_button->set_button_icon(get_editor_theme_icon(SNAME("Redo")));
			cut_button->set_button_icon(get_editor_theme_icon(SNAME("ActionCut")));
			copy_button->set_button_icon(get_editor_theme_icon(SNAME("ActionCopy")));
			paste_button->set_button_icon(get_editor_theme_icon(SNAME("ActionPaste")));
		} break;
	}
}

void TouchActionsPanel::input(const Ref<InputEvent> &event) {
	if (ctrl_btn_pressed) {
		event->call(SNAME("set_ctrl_pressed"), true);
	}

	if (shift_btn_pressed) {
		event->call(SNAME("set_shift_pressed"), true);
	}

	if (alt_btn_pressed) {
		event->call(SNAME("set_alt_pressed"), true);
	}
}

void TouchActionsPanel::_hardware_keyboard_connected(bool p_connected) {
	set_visible(!p_connected);
}

void TouchActionsPanel::_simulate_editor_shortcut(const String &p_shortcut_name) {
	Ref<Shortcut> shortcut = ED_GET_SHORTCUT(p_shortcut_name);

	if (shortcut.is_valid() && !shortcut->get_events().is_empty()) {
		Ref<InputEventKey> event = shortcut->get_events()[0];
		if (event.is_valid()) {
			event->set_pressed(true);
			Input::get_singleton()->parse_input_event(event);
		}
	}
}

void TouchActionsPanel::_simulate_key_press(Key p_keycode) {
	Ref<InputEventKey> event;
	event.instantiate();
	event->set_keycode(p_keycode);
	event->set_pressed(true);
	Input::get_singleton()->parse_input_event(event);
}

void TouchActionsPanel::_on_modifier_button_toggled(bool p_pressed, int p_modifier) {
	switch ((Modifier)p_modifier) {
		case MODIFIER_CTRL:
			ctrl_btn_pressed = p_pressed;
			break;
		case MODIFIER_SHIFT:
			shift_btn_pressed = p_pressed;
			break;
		case MODIFIER_ALT:
			alt_btn_pressed = p_pressed;
			break;
	}
}

Button *TouchActionsPanel::_add_new_action_button(const String &p_shortcut, const String &p_name, Key p_keycode) {
	Button *action_button = memnew(Button);
	action_button->set_theme_type_variation("FlatMenuButton");
	action_button->set_accessibility_name(p_name);
	action_button->set_focus_mode(FOCUS_ACCESSIBILITY);
	action_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	if (p_keycode == Key::NONE) {
		action_button->connect(SceneStringName(pressed), callable_mp(this, &TouchActionsPanel::_simulate_editor_shortcut).bind(p_shortcut));
	} else {
		action_button->connect(SceneStringName(pressed), callable_mp(this, &TouchActionsPanel::_simulate_key_press).bind(p_keycode));
	}
	box->add_child(action_button);
	return action_button;
}

void TouchActionsPanel::_add_new_modifier_button(Modifier p_modifier) {
	String text;
	switch (p_modifier) {
		case MODIFIER_CTRL:
			text = "Ctrl";
			break;
		case MODIFIER_SHIFT:
			text = "Shift";
			break;
		case MODIFIER_ALT:
			text = "Alt";
			break;
	}
	Button *toggle_button = memnew(Button);
	toggle_button->set_text(text);
	toggle_button->set_toggle_mode(true);
	toggle_button->set_theme_type_variation("FlatMenuButton");
	toggle_button->set_accessibility_name(text);
	toggle_button->set_focus_mode(FOCUS_ACCESSIBILITY);
	toggle_button->connect(SceneStringName(toggled), callable_mp(this, &TouchActionsPanel::_on_modifier_button_toggled).bind((int)p_modifier));
	box->add_child(toggle_button);
}

void TouchActionsPanel::_on_drag_handle_gui_input(const Ref<InputEvent> &p_event) {
	if (locked_panel) {
		return;
	}
	Ref<InputEventMouseButton> mouse_button_event = p_event;
	if (mouse_button_event.is_valid() && mouse_button_event->get_button_index() == MouseButton::LEFT) {
		if (mouse_button_event->is_pressed()) {
			dragging = true;
			drag_offset = mouse_button_event->get_position();
		} else {
			if (dragging) {
				dragging = false;
				EditorSettings::get_singleton()->set("_touch_actions_panel_position", get_position());
				EditorSettings::get_singleton()->save();
			}
		}
	}

	Ref<InputEventMouseMotion> mouse_motion_event = p_event;
	if (dragging && mouse_motion_event.is_valid()) {
		Vector2 new_position = get_position() + mouse_motion_event->get_relative();
		const float margin = 25.0;
		Vector2 parent_size = get_parent_area_size();
		Vector2 panel_size = get_size();
		new_position = new_position.clamp(Vector2(margin, margin), parent_size - panel_size - Vector2(margin, margin));
		set_position(new_position);
	}
}

void TouchActionsPanel::_switch_layout() {
	box->set_vertical(!box->is_vertical());
	reset_size();
	queue_redraw();
	EditorSettings::get_singleton()->set("_touch_actions_panel_vertical_layout", box->is_vertical());
	EditorSettings::get_singleton()->save();
}

void TouchActionsPanel::_lock_panel_toggled(bool p_pressed) {
	locked_panel = p_pressed;
	layout_toggle_button->set_visible(!p_pressed);
	drag_handle->set_visible(!p_pressed);
	reset_size();
	queue_redraw();
}

void TouchActionsPanel::_switch_embedded_panel_side() {
	if (embedded_panel_index == 0) {
		embedded_panel_index = 1;
		panel_pos_button->set_button_icon(get_editor_theme_icon(SNAME("ControlAlignLeftWide")));
	} else {
		embedded_panel_index = 0;
		panel_pos_button->set_button_icon(get_editor_theme_icon(SNAME("ControlAlignRightWide")));
	}
	get_parent()->move_child(this, embedded_panel_index); // Parent is a hbox with only two children -- TouchActionsPanel and main Editor UI.
	EditorSettings::get_singleton()->set("_touch_actions_panel_embed_index", embedded_panel_index);
	EditorSettings::get_singleton()->save();
}

TouchActionsPanel::TouchActionsPanel() {
	int panel_mode = EDITOR_GET("interface/touchscreen/touch_actions_panel");
	is_floating = panel_mode == 2;

	if (is_floating) {
		Ref<StyleBoxFlat> panel_style;
		panel_style.instantiate();
		panel_style->set_bg_color(Color(0.1, 0.1, 0.1, 1));
		panel_style->set_border_color(Color(0.3, 0.3, 0.3, 1));
		panel_style->set_border_width_all(3);
		panel_style->set_corner_radius_all(10);
		panel_style->set_content_margin_all(12);
		add_theme_style_override(SceneStringName(panel), panel_style);

		set_position(EDITOR_DEF("_touch_actions_panel_position", Point2(480, 480))); // Dropped it here for no good reason — users can move it anyway.
	}

	box = memnew(BoxContainer);
	box->add_theme_constant_override("separation", 20);
	if (is_floating) {
		box->set_vertical(EDITOR_DEF("_touch_actions_panel_vertical_layout", false));
	} else {
		box->set_vertical(true);
	}
	add_child(box);

	if (is_floating) {
		drag_handle = memnew(TextureRect);
		drag_handle->set_custom_minimum_size(Size2(40, 40));
		drag_handle->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
		drag_handle->connect(SceneStringName(gui_input), callable_mp(this, &TouchActionsPanel::_on_drag_handle_gui_input));
		box->add_child(drag_handle);

		layout_toggle_button = memnew(Button);
		layout_toggle_button->set_theme_type_variation("FlatMenuButton");
		layout_toggle_button->set_accessibility_name(TTRC("Switch Layout"));
		layout_toggle_button->set_focus_mode(FOCUS_ACCESSIBILITY);
		layout_toggle_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
		layout_toggle_button->connect(SceneStringName(pressed), callable_mp(this, &TouchActionsPanel::_switch_layout));
		box->add_child(layout_toggle_button);

		lock_panel_button = memnew(Button);
		lock_panel_button->set_toggle_mode(true);
		lock_panel_button->set_theme_type_variation("FlatMenuButton");
		lock_panel_button->set_accessibility_name(TTRC("Lock Panel"));
		lock_panel_button->set_focus_mode(FOCUS_ACCESSIBILITY);
		lock_panel_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
		lock_panel_button->connect(SceneStringName(toggled), callable_mp(this, &TouchActionsPanel::_lock_panel_toggled));
		box->add_child(lock_panel_button);
	} else {
		panel_pos_button = memnew(Button);
		panel_pos_button->set_theme_type_variation("FlatMenuButton");
		panel_pos_button->set_accessibility_name(TTRC("Switch Embedded Panel Position"));
		panel_pos_button->set_focus_mode(FOCUS_ACCESSIBILITY);
		panel_pos_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
		panel_pos_button->connect(SceneStringName(pressed), callable_mp(this, &TouchActionsPanel::_switch_embedded_panel_side));
		box->add_child(panel_pos_button);

		embedded_panel_index = EDITOR_DEF("_touch_actions_panel_embed_index", 0);
	}

	ColorRect *separator = memnew(ColorRect);
	separator->set_color(Color(0.5, 0.5, 0.5));
	separator->set_custom_minimum_size(Size2(2, 2));
	box->add_child(separator);

	// Add action buttons.
	save_button = _add_new_action_button("editor/save_scene", TTRC("Save"));
	delete_button = _add_new_action_button("", TTRC("Delete"), Key::KEY_DELETE);
	undo_button = _add_new_action_button("ui_undo", TTRC("Undo"));
	redo_button = _add_new_action_button("ui_redo", TTRC("Redo"));
	cut_button = _add_new_action_button("ui_cut", TTRC("Cut"));
	copy_button = _add_new_action_button("ui_copy", TTRC("Copy"));
	paste_button = _add_new_action_button("ui_paste", TTRC("Paste"));

	_add_new_modifier_button(MODIFIER_CTRL);
	_add_new_modifier_button(MODIFIER_SHIFT);
	_add_new_modifier_button(MODIFIER_ALT);
}
