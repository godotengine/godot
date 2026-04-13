/**************************************************************************/
/*  output_panel.cpp                                                      */
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

#include "output_panel.h"

#include "core/config/engine.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/object/message_queue.h"
#include "core/version.h"
#include "scene/gui/rich_text_label.h"
#include "scene/resources/style_box.h"
#include "scene/theme/theme_db.h"

void OutputPanel::_update_theme() {
	if (!log) {
		return;
	}

	if (theme_cache.normal_font.is_valid()) {
		log->add_theme_font_override("normal_font", theme_cache.normal_font);
	} else {
		log->remove_theme_font_override("normal_font");
	}
	if (theme_cache.bold_font.is_valid()) {
		log->add_theme_font_override("bold_font", theme_cache.bold_font);
	} else {
		log->remove_theme_font_override("bold_font");
	}
	if (theme_cache.font_size > 0) {
		log->begin_bulk_theme_override();
		log->add_theme_font_size_override("normal_font_size", theme_cache.font_size);
		log->add_theme_font_size_override("bold_font_size", theme_cache.font_size);
		log->end_bulk_theme_override();
	}
	log->add_theme_constant_override("text_highlight_h_padding", 0);
	log->add_theme_constant_override("text_highlight_v_padding", 0);

	_update_log_rect();
}

void OutputPanel::_update_log_rect() {
	if (!log) {
		return;
	}
	log->set_anchor(SIDE_LEFT, 0.0f);
	log->set_anchor(SIDE_TOP, 0.0f);
	log->set_anchor(SIDE_RIGHT, 1.0f);
	log->set_anchor(SIDE_BOTTOM, 1.0f);
	log->Control::set_offset(SIDE_LEFT, PADDING);
	log->Control::set_offset(SIDE_TOP, PADDING);
	log->Control::set_offset(SIDE_RIGHT, -PADDING);
	log->Control::set_offset(SIDE_BOTTOM, -PADDING);
}

bool OutputPanel::_should_display(const LogMessage &p_message) const {
	switch (p_message.type) {
		case MSG_TYPE_STD:
		case MSG_TYPE_STD_RICH:
			return show_std;
		case MSG_TYPE_WARNING:
			return show_warnings;
		case MSG_TYPE_ERROR:
			return show_errors;
	}
	return true;
}

void OutputPanel::_rebuild_log() {
	if (!log || messages.is_empty()) {
		return;
	}
	log->clear();

	int line_count = 0;
	int start_index = 0;
	int initial_skip = 0;

	for (start_index = messages.size() - 1; start_index >= 0; start_index--) {
		const LogMessage &msg = messages[start_index];
		if (_should_display(msg)) {
			line_count += msg.count;
		}
		if (line_count >= max_lines) {
			initial_skip = line_count - max_lines;
			break;
		}
		if (start_index == 0) {
			break;
		}
	}

	for (int i = start_index; i < messages.size(); i++) {
		LogMessage msg = messages[i];
		if (!_should_display(msg)) {
			continue;
		}
		for (int j = initial_skip; j < msg.count; j++) {
			initial_skip = 0;
			_add_log_line(msg);
		}
	}
}

void OutputPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			theme_cache.panel_style->draw(ci, Rect2(Point2(), get_size()));
		} break;

		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			_update_theme();
			_rebuild_log();
			queue_redraw();
		} break;

		case NOTIFICATION_RESIZED: {
			_update_log_rect();
			queue_redraw();
		} break;

		case NOTIFICATION_READY: {
			if (Engine::get_singleton()->is_editor_hint()) {
				break;
			}

#ifndef DEBUG_ENABLED
			if (debug_only) {
				set_visible(false);
				break;
			}
#endif

			print_handler.printfunc = _print_handler;
			print_handler.userdata = this;
			error_handler.errfunc = _error_handler;
			error_handler.userdata = this;
			add_message(GODOT_VERSION_FULL_NAME " (c) 2007-present Juan Linietsky, Ariel Manzur & Godot Contributors.");

#ifdef DEBUG_ENABLED
			add_print_handler(&print_handler);
			add_error_handler(&error_handler);
#else
			if (!debug_only) {
				add_print_handler(&print_handler);
				add_error_handler(&error_handler);
			}
#endif
		} break;

		case NOTIFICATION_EXIT_TREE: {
			remove_print_handler(&print_handler);
			remove_error_handler(&error_handler);
		} break;
	}
}

void OutputPanel::_print_handler(void *p_self, const String &p_string, bool p_error, bool p_rich) {
	OutputPanel *self = static_cast<OutputPanel *>(p_self);
	if (p_error) {
		if (!self->show_errors) {
			return;
		}
		if (!Thread::is_main_thread()) {
			MessageQueue::get_main_singleton()->push_callable(callable_mp(self, &OutputPanel::add_message), p_string, MSG_TYPE_ERROR);
		} else {
			self->add_message(p_string, MSG_TYPE_ERROR);
		}
		return;
	}
	if (!self->show_std) {
		return;
	}
	MessageType type = p_rich ? MSG_TYPE_STD_RICH : MSG_TYPE_STD;
	if (!Thread::is_main_thread()) {
		MessageQueue::get_main_singleton()->push_callable(callable_mp(self, &OutputPanel::add_message), p_string, type);
	} else {
		self->add_message(p_string, type);
	}
}

void OutputPanel::_error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, bool p_editor_notify, ErrorHandlerType p_type) {
	OutputPanel *self = static_cast<OutputPanel *>(p_self);

	String err_str;
	if (p_errorexp && p_errorexp[0]) {
		err_str = String::utf8(p_errorexp).replace("[", "[lb]");
	} else {
		err_str = vformat("[url]%s:%d[/url] - %s", String::utf8(p_file).replace("[", "[lb]"), p_line, String::utf8(p_error).replace("[", "[lb]"));
	}

	MessageType msg_type = (p_type == ERR_HANDLER_WARNING) ? MSG_TYPE_WARNING : MSG_TYPE_ERROR;
	if (msg_type == MSG_TYPE_WARNING && !self->show_warnings) {
		return;
	}
	if (msg_type == MSG_TYPE_ERROR && !self->show_errors) {
		return;
	}
	if (!Thread::is_main_thread()) {
		MessageQueue::get_main_singleton()->push_callable(callable_mp(self, &OutputPanel::add_message), err_str, msg_type);
	} else {
		self->add_message(err_str, msg_type);
	}
}

void OutputPanel::add_message(const String &p_msg, MessageType p_type) {
	Vector<String> lines = p_msg.split("\n", true);
	int line_count = lines.size();
	for (int i = 0; i < line_count; i++) {
		_process_message(lines[i], p_type, i == line_count - 1);
	}
}

void OutputPanel::_process_message(const String &p_msg, MessageType p_type, bool p_clear) {
	if (messages.size() > 0 && messages[messages.size() - 1].text == p_msg && messages[messages.size() - 1].type == p_type) {
		LogMessage &previous = messages.write[messages.size() - 1];
		previous.count++;
		_add_log_line(previous);
	} else {
		LogMessage message(p_msg, p_type, p_clear);
		_add_log_line(message);
		messages.push_back(message);
	}
	while (messages.size() > (uint32_t)max_lines) {
		messages.remove_at(0);
	}
	if (log->get_paragraph_count() > max_lines + 1) {
		callable_mp(this, &OutputPanel::_rebuild_log).call_deferred();
	}
}

void OutputPanel::_add_log_line(LogMessage &p_message) {
	if (!is_inside_tree()) {
		return;
	}
	switch (p_message.type) {
		case MSG_TYPE_STD: {
			log->push_color(theme_cache.message_color);
		} break;
		case MSG_TYPE_STD_RICH: {
		} break;
		case MSG_TYPE_WARNING: {
			log->push_color(theme_cache.warning_color);
			log->push_bold();
			log->add_text("WARNING: ");
			log->pop();
		} break;
		case MSG_TYPE_ERROR: {
			log->push_color(theme_cache.error_color);
			log->push_bold();
			log->add_text("ERROR: ");
			log->pop();
		} break;
	}
	if (p_message.type == MSG_TYPE_STD_RICH || p_message.type == MSG_TYPE_ERROR || p_message.type == MSG_TYPE_WARNING) {
		log->append_text(p_message.text);
	} else {
		log->add_text(p_message.text);
	}
	if (p_message.clear || p_message.type != MSG_TYPE_STD_RICH) {
		log->pop_all();
	}
	log->add_newline();
}

void OutputPanel::set_max_lines(int p_max_lines) {
	max_lines = MAX(1, p_max_lines);
	callable_mp(this, &OutputPanel::_rebuild_log).call_deferred();
}
int OutputPanel::get_max_lines() const {
	return max_lines;
}

void OutputPanel::set_show_std(bool p_show) {
	show_std = p_show;
	_rebuild_log();
}
bool OutputPanel::get_show_std() const {
	return show_std;
}
void OutputPanel::set_show_warnings(bool p_show) {
	show_warnings = p_show;
	_rebuild_log();
}
bool OutputPanel::get_show_warnings() const {
	return show_warnings;
}
void OutputPanel::set_show_errors(bool p_show) {
	show_errors = p_show;
	_rebuild_log();
}
bool OutputPanel::get_show_errors() const {
	return show_errors;
}
void OutputPanel::set_debug_only(bool p_debug_only) {
	debug_only = p_debug_only;
}
bool OutputPanel::get_debug_only() const {
	return debug_only;
}

void OutputPanel::clear() {
	if (log) {
		log->clear();
	}
	messages.clear();
}

void OutputPanel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("clear"), &OutputPanel::clear);
	ClassDB::bind_method(D_METHOD("set_max_lines", "max_lines"), &OutputPanel::set_max_lines);
	ClassDB::bind_method(D_METHOD("get_max_lines"), &OutputPanel::get_max_lines);
	ClassDB::bind_method(D_METHOD("set_show_std", "enabled"), &OutputPanel::set_show_std);
	ClassDB::bind_method(D_METHOD("get_show_std"), &OutputPanel::get_show_std);
	ClassDB::bind_method(D_METHOD("set_show_warnings", "enabled"), &OutputPanel::set_show_warnings);
	ClassDB::bind_method(D_METHOD("get_show_warnings"), &OutputPanel::get_show_warnings);
	ClassDB::bind_method(D_METHOD("set_show_errors", "enabled"), &OutputPanel::set_show_errors);
	ClassDB::bind_method(D_METHOD("get_show_errors"), &OutputPanel::get_show_errors);
	ClassDB::bind_method(D_METHOD("set_debug_only", "enabled"), &OutputPanel::set_debug_only);
	ClassDB::bind_method(D_METHOD("get_debug_only"), &OutputPanel::get_debug_only);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_lines", PROPERTY_HINT_RANGE, "1,10000,1"), "set_max_lines", "get_max_lines");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_only"), "set_debug_only", "get_debug_only");
	ADD_GROUP("Filters", "show_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_std"), "set_show_std", "get_show_std");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_warnings"), "set_show_warnings", "get_show_warnings");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_errors"), "set_show_errors", "get_show_errors");

	BIND_ENUM_CONSTANT(MSG_TYPE_STD);
	BIND_ENUM_CONSTANT(MSG_TYPE_STD_RICH);
	BIND_ENUM_CONSTANT(MSG_TYPE_WARNING);
	BIND_ENUM_CONSTANT(MSG_TYPE_ERROR);

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, OutputPanel, panel_style, "background");
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT, OutputPanel, normal_font);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT, OutputPanel, bold_font);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT_SIZE, OutputPanel, font_size);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, OutputPanel, error_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, OutputPanel, warning_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, OutputPanel, message_color);
}

OutputPanel::OutputPanel() {
	log = memnew(RichTextLabel);
	log->set_name("Log");
	log->set_threaded(false);
	log->set_use_bbcode(true);
	log->set_scroll_follow(true);
	log->set_selection_enabled(false);
	log->set_context_menu_enabled(true);
	log->set_focus_mode(FOCUS_CLICK);
	log->set_v_size_flags(SIZE_EXPAND_FILL);
	log->set_h_size_flags(SIZE_EXPAND_FILL);
	log->set_deselect_on_focus_loss_enabled(false);
	log->set_anchor(SIDE_LEFT, 0.0f);
	log->set_anchor(SIDE_TOP, 0.0f);
	log->set_anchor(SIDE_RIGHT, 1.0f);
	log->set_anchor(SIDE_BOTTOM, 1.0f);
	log->Control::set_offset(SIDE_LEFT, PADDING);
	log->Control::set_offset(SIDE_TOP, PADDING);
	log->Control::set_offset(SIDE_RIGHT, -PADDING);
	log->Control::set_offset(SIDE_BOTTOM, -PADDING);
	add_child(log, false, INTERNAL_MODE_FRONT);

	print_handler.printfunc = nullptr;
	print_handler.userdata = nullptr;
	error_handler.errfunc = nullptr;
	error_handler.userdata = nullptr;
}

OutputPanel::~OutputPanel() {
	remove_print_handler(&print_handler);
	remove_error_handler(&error_handler);
}
