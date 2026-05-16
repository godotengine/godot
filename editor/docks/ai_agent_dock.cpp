/**************************************************************************/
/*  ai_agent_dock.cpp                                                     */
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

#include "ai_agent_dock.h"

#include "editor/editor_string_names.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/center_container.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/style_box_flat.h"

Label *AIAgentDock::_create_label(const String &p_text, bool p_muted) {
	Label *label = memnew(Label);
	label->set_text(p_text);
	label->set_clip_text(true);
	label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	label->set_v_size_flags(Control::SIZE_SHRINK_CENTER);

	if (p_muted) {
		muted_labels.push_back(label);
	}

	return label;
}

void AIAgentDock::_add_task_row(VBoxContainer *p_parent, const String &p_task, const String &p_time) {
	HBoxContainer *row = memnew(HBoxContainer);
	row->add_theme_constant_override(SNAME("separation"), 6 * EDSCALE);
	p_parent->add_child(row);

	Label *task_label = _create_label(p_task);
	task_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	row->add_child(task_label);

	Label *time_label = _create_label(p_time, true);
	time_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_RIGHT);
	time_label->set_custom_minimum_size(Size2(36, 0) * EDSCALE);
	row->add_child(time_label);
}

void AIAgentDock::_update_theme() {
	refresh_button->set_button_icon(get_editor_theme_icon(SNAME("Reload")));
	settings_button->set_button_icon(get_editor_theme_icon(SNAME("Tools")));
	compose_button->set_button_icon(get_editor_theme_icon(SNAME("Edit")));
	add_button->set_button_icon(get_editor_theme_icon(SNAME("Add")));
	send_button->set_button_icon(get_editor_theme_icon(SNAME("GuiArrowUp")));
	empty_state_icon->set_texture(get_editor_theme_icon(SNAME("GodotMonochrome")));

	const Color base_color = get_theme_color(SNAME("base_color"), EditorStringName(Editor));
	const Color mono_color = get_theme_color(SNAME("mono_color"), EditorStringName(Editor));
	const Color muted_color = get_theme_color(SNAME("font_disabled_color"), EditorStringName(Editor));
	const Color warning_color = get_theme_color(SNAME("warning_color"), EditorStringName(Editor));

	Color prompt_bg = base_color.lerp(mono_color, 0.035);
	prompt_bg.a = 1.0;
	Color prompt_border = mono_color;
	prompt_border.a *= 0.12;

	prompt_panel_style->set_bg_color(prompt_bg);
	prompt_panel_style->set_border_color(prompt_border);

	prompt->add_theme_color_override(SNAME("font_placeholder_color"), muted_color);
	empty_state_icon->set_modulate(Color(muted_color, 0.45));

	for (Label *label : muted_labels) {
		label->add_theme_color_override(SceneStringName(font_color), muted_color);
	}
	for (Label *label : access_labels) {
		label->add_theme_color_override(SceneStringName(font_color), warning_color);
	}
}

void AIAgentDock::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			_update_theme();
		} break;
	}
}

AIAgentDock::AIAgentDock() {
	singleton = this;
	set_name(TTRC("AI Agent"));
	set_icon_name("EditorPlugin");
	set_dock_shortcut(ED_SHORTCUT_AND_COMMAND("docks/open_ai_agent", TTRC("Open AI Agent Dock")));
	set_default_slot(EditorDock::DOCK_SLOT_RIGHT_UL);

	prompt_panel_style.instantiate();
	prompt_panel_style->set_border_width_all(1 * EDSCALE);
	prompt_panel_style->set_corner_radius_all(8 * EDSCALE);
	prompt_panel_style->set_content_margin_individual(12 * EDSCALE, 10 * EDSCALE, 10 * EDSCALE, 10 * EDSCALE);

	MarginContainer *root_margin = memnew(MarginContainer);
	root_margin->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	root_margin->add_theme_constant_override("margin_left", 10 * EDSCALE);
	root_margin->add_theme_constant_override("margin_top", 10 * EDSCALE);
	root_margin->add_theme_constant_override("margin_right", 10 * EDSCALE);
	root_margin->add_theme_constant_override("margin_bottom", 8 * EDSCALE);
	add_child(root_margin);

	VBoxContainer *main_vb = memnew(VBoxContainer);
	main_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	main_vb->add_theme_constant_override(SNAME("separation"), 8 * EDSCALE);
	root_margin->add_child(main_vb);

	HBoxContainer *header_hb = memnew(HBoxContainer);
	header_hb->add_theme_constant_override(SNAME("separation"), 4 * EDSCALE);
	main_vb->add_child(header_hb);

	Label *tasks_label = _create_label(TTRC("Tasks"), true);
	tasks_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	header_hb->add_child(tasks_label);

	refresh_button = memnew(Button);
	refresh_button->set_theme_type_variation(SceneStringName(FlatButton));
	refresh_button->set_tooltip_text(TTRC("Refresh tasks."));
	refresh_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	header_hb->add_child(refresh_button);

	settings_button = memnew(Button);
	settings_button->set_theme_type_variation(SceneStringName(FlatButton));
	settings_button->set_tooltip_text(TTRC("Agent settings."));
	settings_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	header_hb->add_child(settings_button);

	compose_button = memnew(Button);
	compose_button->set_theme_type_variation(SceneStringName(FlatButton));
	compose_button->set_tooltip_text(TTRC("Start a new task."));
	compose_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	header_hb->add_child(compose_button);

	VBoxContainer *tasks_vb = memnew(VBoxContainer);
	tasks_vb->add_theme_constant_override(SNAME("separation"), 10 * EDSCALE);
	main_vb->add_child(tasks_vb);

	_add_task_row(tasks_vb, TTRC("Fix failing tests"), TTRC("38m"));
	_add_task_row(tasks_vb, TTRC("Explain CircleCI Chunk"), TTRC("37m"));
	_add_task_row(tasks_vb, TTRC("说明 Main::setup() 的OS初始化"), TTRC("2h"));

	Label *view_all_label = _create_label(TTRC("View all (50)"), true);
	main_vb->add_child(view_all_label);

	CenterContainer *empty_state = memnew(CenterContainer);
	empty_state->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	empty_state->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	main_vb->add_child(empty_state);

	empty_state_icon = memnew(TextureRect);
	empty_state_icon->set_custom_minimum_size(Size2(48, 48) * EDSCALE);
	empty_state_icon->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
	empty_state_icon->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	empty_state->add_child(empty_state_icon);

	prompt_panel = memnew(PanelContainer);
	prompt_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	prompt_panel->add_theme_style_override(SceneStringName(panel), prompt_panel_style);
	main_vb->add_child(prompt_panel);

	VBoxContainer *prompt_vb = memnew(VBoxContainer);
	prompt_vb->add_theme_constant_override(SNAME("separation"), 8 * EDSCALE);
	prompt_panel->add_child(prompt_vb);

	prompt = memnew(LineEdit);
	prompt->set_flat(true);
	prompt->set_placeholder(TTRC("Ask Codex anything. @ to use plugins or mention files"));
	prompt->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	prompt_vb->add_child(prompt);

	HBoxContainer *prompt_status_hb = memnew(HBoxContainer);
	prompt_status_hb->add_theme_constant_override(SNAME("separation"), 6 * EDSCALE);
	prompt_vb->add_child(prompt_status_hb);

	add_button = memnew(Button);
	add_button->set_theme_type_variation(SceneStringName(FlatButton));
	add_button->set_tooltip_text(TTRC("Add context."));
	add_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	prompt_status_hb->add_child(add_button);

	Label *access_label = _create_label(TTRC("Full access"));
	access_labels.push_back(access_label);
	prompt_status_hb->add_child(access_label);

	Control *prompt_spacer = memnew(Control);
	prompt_spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	prompt_status_hb->add_child(prompt_spacer);

	Label *model_label = _create_label(TTRC("5.5 Extra High"), true);
	prompt_status_hb->add_child(model_label);

	VSeparator *status_separator = memnew(VSeparator);
	prompt_status_hb->add_child(status_separator);

	Label *context_label = _create_label(TTRC("IDE context"), true);
	prompt_status_hb->add_child(context_label);

	send_button = memnew(Button);
	send_button->set_theme_type_variation(SceneStringName(FlatButton));
	send_button->set_tooltip_text(TTRC("Send message."));
	send_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	prompt_status_hb->add_child(send_button);

	HBoxContainer *work_hb = memnew(HBoxContainer);
	work_hb->add_theme_constant_override(SNAME("separation"), 4 * EDSCALE);
	main_vb->add_child(work_hb);

	Label *work_locally_label = _create_label(TTRC("Work locally"), true);
	work_locally_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	work_hb->add_child(work_locally_label);
}

AIAgentDock::~AIAgentDock() {
	singleton = nullptr;
}
