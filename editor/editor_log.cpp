/**************************************************************************/
/*  editor_log.cpp                                                        */
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

#include "editor_log.h"

#include "core/os/keyboard.h"
#include "core/version.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "scene/gui/center_container.h"
#include "scene/resources/dynamic_font.h"

#ifdef MODULE_REGEX_ENABLED
#include "modules/regex/regex.h"
#endif

void EditorLog::_error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, ErrorHandlerType p_type) {
	EditorLog *self = (EditorLog *)p_self;
	if (self->current != Thread::get_caller_id()) {
		return;
	}

	String err_str;
	if (p_errorexp && p_errorexp[0]) {
		err_str = String::utf8(p_errorexp);
	} else {
		err_str = String::utf8(p_file) + ":" + itos(p_line) + " - " + String::utf8(p_error);
	}

	if (p_type == ERR_HANDLER_WARNING) {
		self->add_message(err_str, MSG_TYPE_WARNING);
	} else {
		self->add_message(err_str, MSG_TYPE_ERROR);
	}
}

void EditorLog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			if (log != nullptr) {
				Ref<DynamicFont> df_output_code = get_font("output_source", "EditorFonts");
				if (df_output_code.is_valid()) {
					log->add_font_override("normal_font", get_font("output_source", "EditorFonts"));
					log->add_color_override("selection_color", get_color("accent_color", "Editor") * Color(1, 1, 1, 0.4));
				}
			}

			theme_cache.error_color = get_color("error_color", "Editor");
			theme_cache.error_icon = get_icon("Error", "EditorIcons");
			theme_cache.warning_color = get_color("warning_color", "Editor");
			theme_cache.warning_icon = get_icon("Warning", "EditorIcons");
			theme_cache.message_color = get_color("font_color", "Editor") * Color(1, 1, 1, 0.6);

			terminal_colors._default = EDITOR_GET("text_editor/highlighting/text_color");

			terminal_colors.bright_white = EDITOR_GET("text_editor/highlighting/caret_color");
			terminal_colors.bright_black = EDITOR_GET("text_editor/highlighting/text_selected_color");

			terminal_colors.red = EDITOR_GET("text_editor/highlighting/keyword_color");
			terminal_colors.green = EDITOR_GET("text_editor/highlighting/base_type_color");
			terminal_colors.blue = EDITOR_GET("text_editor/highlighting/bookmark_color");
			terminal_colors.yellow = EDITOR_GET("text_editor/highlighting/string_color");
			terminal_colors.magenta = EDITOR_GET("text_editor/highlighting/control_flow_keyword_color");
			terminal_colors.cyan = EDITOR_GET("text_editor/highlighting/symbol_color");

			float M = 1.3f;

			terminal_colors.white = EDITOR_GET("text_editor/highlighting/text_color");
			terminal_colors.black = EDITOR_GET("text_editor/highlighting/caret_background_color");

			terminal_colors.bright_red = terminal_colors.red * M;
			terminal_colors.bright_green = terminal_colors.green * M;
			terminal_colors.bright_blue = terminal_colors.blue * M;
			terminal_colors.bright_yellow = terminal_colors.yellow * M;
			terminal_colors.bright_magenta = terminal_colors.magenta * M;
			terminal_colors.bright_cyan = terminal_colors.cyan * M;
		} break;
	}
}

void EditorLog::_clear_request() {
	log->clear();
	tool_button->set_icon(Ref<Texture>());
}

void EditorLog::_copy_request() {
	String text = log->get_selected_text();

	if (text == "") {
		text = log->get_text();
	}

	if (text != "") {
		OS::get_singleton()->set_clipboard(text);
	}
}

void EditorLog::clear() {
	_clear_request();
}

void EditorLog::copy() {
	_copy_request();
}

bool EditorLog::_log_push_terminal_color(TerminalColor::Col p_color) {
	Color c;

	switch (p_color) {
		default: {
			return false;
		} break;
		case TerminalColor::DEFAULT: {
			c = terminal_colors._default;
		} break;
		case TerminalColor::WHITE: {
			c = terminal_colors.white;
		} break;
		case TerminalColor::BLACK: {
			c = terminal_colors.black;
		} break;
		case TerminalColor::RED: {
			c = terminal_colors.red;
		} break;
		case TerminalColor::GREEN: {
			c = terminal_colors.green;
		} break;
		case TerminalColor::BLUE: {
			c = terminal_colors.blue;
		} break;
		case TerminalColor::YELLOW: {
			c = terminal_colors.yellow;
		} break;
		case TerminalColor::MAGENTA: {
			c = terminal_colors.magenta;
		} break;
		case TerminalColor::CYAN: {
			c = terminal_colors.cyan;
		} break;
		case TerminalColor::BRIGHT_WHITE: {
			c = terminal_colors.bright_white;
		} break;
		case TerminalColor::BRIGHT_BLACK: {
			c = terminal_colors.bright_black;
		} break;
		case TerminalColor::BRIGHT_RED: {
			c = terminal_colors.bright_red;
		} break;
		case TerminalColor::BRIGHT_GREEN: {
			c = terminal_colors.bright_green;
		} break;
		case TerminalColor::BRIGHT_BLUE: {
			c = terminal_colors.bright_blue;
		} break;
		case TerminalColor::BRIGHT_YELLOW: {
			c = terminal_colors.bright_yellow;
		} break;
		case TerminalColor::BRIGHT_MAGENTA: {
			c = terminal_colors.bright_magenta;
		} break;
		case TerminalColor::BRIGHT_CYAN: {
			c = terminal_colors.bright_cyan;
		} break;
	}

	log->push_color(c);
	return true;
}

void EditorLog::_log_add_text(const String &p_msg) {
#ifdef MODULE_REGEX_ENABLED
	log->add_text(strip_ansi_regex->sub(p_msg, "", true));
#else
	log->add_text(p_msg);
#endif
}

void EditorLog::add_message(const String &p_msg, MessageType p_type) {
	bool restore = p_type != MSG_TYPE_STD;
	switch (p_type) {
		case MSG_TYPE_STD: {
		} break;
		case MSG_TYPE_ERROR: {
			log->push_color(theme_cache.error_color);
			Ref<Texture> icon = theme_cache.error_icon;
			log->add_image(icon);
			log->add_text(" ");
			tool_button->set_icon(icon);
		} break;
		case MSG_TYPE_WARNING: {
			log->push_color(theme_cache.warning_color);
			Ref<Texture> icon = theme_cache.warning_icon;
			log->add_image(icon);
			log->add_text(" ");
			tool_button->set_icon(icon);
		} break;
		case MSG_TYPE_EDITOR: {
			// Distinguish editor messages from messages printed by the project
			log->push_color(theme_cache.message_color);
		} break;
	}

	// Terminal colors.
	int start_text = 0;
	int start_color = 0;
	TerminalColor::Col prev_col = TerminalColor::DEFAULT;
	TerminalColor::Col curr_col = TerminalColor::DEFAULT;

	int found = 0;
	bool any_colors_found = false;

	while (found != -1) {
		found = TerminalColor::find(p_msg, start_text, curr_col, start_color);

		if (found != -1) {
			if (start_color != start_text) {
				String section = p_msg.substr(start_text, start_color - start_text);

				if (_log_push_terminal_color(prev_col)) {
					restore = true;
				}
				_log_add_text(section);
			}
			prev_col = curr_col;
			start_text = found;
			any_colors_found = true;
		}
	}

	if (!any_colors_found) {
		log->add_text(p_msg);
	} else {
		String section = p_msg.substr(start_text);
		if (section.length()) {
			if (_log_push_terminal_color(prev_col)) {
				restore = true;
			}
			_log_add_text(section);
		}
	}

	log->add_newline();

	if (restore) {
		log->pop();
	}
}

void EditorLog::set_tool_button(ToolButton *p_tool_button) {
	tool_button = p_tool_button;
}

void EditorLog::_undo_redo_cbk(void *p_self, const String &p_name) {
	EditorLog *self = (EditorLog *)p_self;
	self->add_message(p_name, EditorLog::MSG_TYPE_EDITOR);
}

void EditorLog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_clear_request"), &EditorLog::_clear_request);
	ClassDB::bind_method(D_METHOD("_copy_request"), &EditorLog::_copy_request);
	ADD_SIGNAL(MethodInfo("clear_request"));
	ADD_SIGNAL(MethodInfo("copy_request"));
}

EditorLog::EditorLog() {
#ifdef MODULE_REGEX_ENABLED
	strip_ansi_regex = memnew(RegEx("\u001b\\[((?:\\d|;)*)([a-zA-Z])"));
#endif
	VBoxContainer *vb = this;

	HBoxContainer *hb = memnew(HBoxContainer);
	vb->add_child(hb);
	title = memnew(Label);
	title->set_text(TTR("Output:"));
	title->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(title);

	copybutton = memnew(Button);
	hb->add_child(copybutton);
	copybutton->set_text(TTR("Copy"));
	copybutton->set_shortcut(ED_SHORTCUT("editor/copy_output", TTR("Copy Selection"), KEY_MASK_CMD | KEY_C));
	copybutton->connect("pressed", this, "_copy_request");

	clearbutton = memnew(Button);
	hb->add_child(clearbutton);
	clearbutton->set_text(TTR("Clear"));
	clearbutton->set_shortcut(ED_SHORTCUT("editor/clear_output", TTR("Clear Output"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_K));
	clearbutton->connect("pressed", this, "_clear_request");

	log = memnew(RichTextLabel);
	log->set_scroll_follow(true);
	log->set_selection_enabled(true);
	log->set_focus_mode(FOCUS_CLICK);
	log->set_custom_minimum_size(Size2(0, 180) * EDSCALE);
	log->set_v_size_flags(SIZE_EXPAND_FILL);
	log->set_h_size_flags(SIZE_EXPAND_FILL);
	log->set_deselect_on_focus_loss_enabled(false);
	vb->add_child(log);
	add_message(VERSION_FULL_NAME " (c) 2007-2022 Juan Linietsky, Ariel Manzur & Godot Contributors.");

	eh.errfunc = _error_handler;
	eh.userdata = this;
	add_error_handler(&eh);

	current = Thread::get_caller_id();

	add_constant_override("separation", get_constant("separation", "VBoxContainer"));

	EditorNode::get_undo_redo()->set_commit_notify_callback(_undo_redo_cbk, this);
}

void EditorLog::deinit() {
#ifdef MODULE_REGEX_ENABLED
	memdelete(strip_ansi_regex);
#endif

	remove_error_handler(&eh);
}

EditorLog::~EditorLog() {
}
