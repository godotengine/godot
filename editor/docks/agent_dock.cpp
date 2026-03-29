/**************************************************************************/
/*  agent_dock.cpp                                                        */
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

#include "agent_dock.h"

#include "core/os/keyboard.h"
#include "editor/settings/editor_command_palette.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/text_edit.h"

void AgentDock::_send_message() {
	String text = chat_input->get_text().strip_edges();
	if (text.is_empty() || is_processing) {
		return;
	}

	_append_message(TTR("You"), text);
	chat_input->set_text("");
	_set_processing(true);

	// Emit signal for the controller to handle.
	emit_signal(SNAME("message_sent"), text);
}

void AgentDock::_on_input_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_pressed() && !k->is_echo()) {
		if (k->get_keycode() == Key::ENTER || k->get_keycode() == Key::KP_ENTER) {
			if (!k->is_shift_pressed()) {
				_send_message();
				chat_input->accept_event();
			}
		}
	}
}

void AgentDock::_append_message(const String &p_sender, const String &p_text, const Color &p_color) {
	if (p_color == Color(1, 1, 1)) {
		chat_display->append_text("[b]" + p_sender + ":[/b] " + p_text.xml_escape() + "\n");
	} else {
		String color_hex = p_color.to_html(false);
		chat_display->append_text("[color=#" + color_hex + "][b]" + p_sender + ":[/b][/color] " + p_text.xml_escape() + "\n");
	}
}

void AgentDock::_append_system_message(const String &p_text) {
	chat_display->append_text("[color=#565f89][i]" + p_text.xml_escape() + "[/i][/color]\n");
}

void AgentDock::_clear_chat() {
	chat_display->clear();
}

void AgentDock::_update_context_display() {
	// Will be updated by editor signals in later phases.
	context_label->set_text(TTR("No context"));
}

void AgentDock::_on_settings_pressed() {
	// TODO: Open agent settings dialog in a later phase.
}

void AgentDock::_set_processing(bool p_processing) {
	is_processing = p_processing;
	send_button->set_disabled(p_processing);
	chat_input->set_editable(!p_processing);
	if (p_processing) {
		status_label->set_text(TTR("Thinking..."));
	} else {
		status_label->set_text(TTR("Idle"));
	}
}

void AgentDock::append_streamed_token(const String &p_token) {
	if (!is_streaming) {
		is_streaming = true;
		chat_display->append_text("[color=#7aa2f7][b]" + TTR("Agent") + ":[/b][/color] ");
	}
	chat_display->append_text(p_token.xml_escape());
}

void AgentDock::finish_streamed_response() {
	if (is_streaming) {
		chat_display->append_text("\n");
		is_streaming = false;
	}
	_set_processing(false);
}

void AgentDock::set_context_text(const String &p_text) {
	context_label->set_text(p_text);
}

void AgentDock::set_status(const String &p_status) {
	status_label->set_text(p_status);
}

void AgentDock::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_append_system_message(TTR("Agent ready. Type a message to begin."));
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			if (send_button) {
				send_button->set_button_icon(get_theme_icon(SNAME("ArrowRight"), SNAME("EditorIcons")));
			}
			if (settings_button) {
				settings_button->set_button_icon(get_theme_icon(SNAME("Tools"), SNAME("EditorIcons")));
			}
		} break;
	}
}

void AgentDock::_bind_methods() {
	ADD_SIGNAL(MethodInfo("message_sent", PropertyInfo(Variant::STRING, "message")));
}

void AgentDock::append_system_message(const String &p_text) {
	_append_system_message(p_text);
}

AgentDock::AgentDock() {
	singleton = this;
	set_name(TTRC("Agent"));
	set_icon_name("Agent");
	set_dock_shortcut(ED_SHORTCUT_AND_COMMAND("docks/open_agent", TTRC("Open Agent Dock")));
	set_default_slot(EditorDock::DOCK_SLOT_RIGHT_UR);

	main_vbox = memnew(VBoxContainer);
	main_vbox->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(main_vbox);

	// Context bar.
	context_bar = memnew(HBoxContainer);
	main_vbox->add_child(context_bar);

	context_label = memnew(Label);
	context_label->set_text(TTR("No context"));
	context_label->set_h_size_flags(SIZE_EXPAND_FILL);
	context_label->set_clip_text(true);
	context_label->add_theme_font_size_override("font_size", 11);
	context_bar->add_child(context_label);

	// Chat display.
	chat_display = memnew(RichTextLabel);
	chat_display->set_v_size_flags(SIZE_EXPAND_FILL);
	chat_display->set_h_size_flags(SIZE_EXPAND_FILL);
	chat_display->set_scroll_follow(true);
	chat_display->set_use_bbcode(true);
	chat_display->set_selection_enabled(true);
	chat_display->set_focus_mode(FOCUS_CLICK);
	main_vbox->add_child(chat_display);

	// Status label.
	status_label = memnew(Label);
	status_label->set_text(TTR("Idle"));
	status_label->add_theme_font_size_override("font_size", 11);
	status_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_RIGHT);
	main_vbox->add_child(status_label);

	// Input area.
	input_hbox = memnew(HBoxContainer);
	main_vbox->add_child(input_hbox);

	chat_input = memnew(TextEdit);
	chat_input->set_h_size_flags(SIZE_EXPAND_FILL);
	chat_input->set_custom_minimum_size(Size2(0, 60));
	chat_input->set_placeholder(TTRC("Type a message..."));
	chat_input->set_line_wrapping_mode(TextEdit::LINE_WRAPPING_BOUNDARY);
	chat_input->connect("gui_input", callable_mp(this, &AgentDock::_on_input_gui_input));
	input_hbox->add_child(chat_input);

	send_button = memnew(Button);
	send_button->set_text(TTR("Send"));
	send_button->set_custom_minimum_size(Size2(60, 0));
	send_button->connect(SceneStringName(pressed), callable_mp(this, &AgentDock::_send_message));
	input_hbox->add_child(send_button);

	// Settings bar.
	settings_bar = memnew(HBoxContainer);
	main_vbox->add_child(settings_bar);

	model_selector = memnew(OptionButton);
	model_selector->set_h_size_flags(SIZE_EXPAND_FILL);
	model_selector->set_clip_text(true);
	model_selector->add_item("Claude Sonnet 4");
	model_selector->add_item("Claude Opus 4");
	settings_bar->add_child(model_selector);

	settings_button = memnew(Button);
	settings_button->set_tooltip_text(TTR("Agent Settings"));
	settings_button->connect(SceneStringName(pressed), callable_mp(this, &AgentDock::_on_settings_pressed));
	settings_bar->add_child(settings_button);
}

AgentDock::~AgentDock() {
	singleton = nullptr;
}
