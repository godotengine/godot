/*************************************************************************/
/*  editor_log.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "editor_log.h"

#include "core/os/keyboard.h"
#include "core/version.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "scene/gui/center_container.h"
#include "scene/resources/font.h"

void EditorLog::_error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, bool p_editor_notify, ErrorHandlerType p_type) {
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

	if (p_editor_notify) {
		err_str += " (User)";
	}

	if (p_type == ERR_HANDLER_WARNING) {
		self->add_message(err_str, MSG_TYPE_WARNING);
	} else {
		self->add_message(err_str, MSG_TYPE_ERROR);
	}
}

void EditorLog::_update_theme() {
	Ref<Font> normal_font = get_theme_font(SNAME("output_source"), SNAME("EditorFonts"));
	if (normal_font.is_valid()) {
		log->add_theme_font_override("normal_font", normal_font);
	}

	log->add_theme_font_size_override("normal_font_size", get_theme_font_size(SNAME("output_source_size"), SNAME("EditorFonts")));
	log->add_theme_color_override("selection_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.4));

	Ref<Font> bold_font = get_theme_font(SNAME("bold"), SNAME("EditorFonts"));
	if (bold_font.is_valid()) {
		log->add_theme_font_override("bold_font", bold_font);
	}

	type_filter_map[MSG_TYPE_STD]->toggle_button->set_icon(get_theme_icon(SNAME("Popup"), SNAME("EditorIcons")));
	type_filter_map[MSG_TYPE_ERROR]->toggle_button->set_icon(get_theme_icon(SNAME("StatusError"), SNAME("EditorIcons")));
	type_filter_map[MSG_TYPE_WARNING]->toggle_button->set_icon(get_theme_icon(SNAME("StatusWarning"), SNAME("EditorIcons")));
	type_filter_map[MSG_TYPE_EDITOR]->toggle_button->set_icon(get_theme_icon(SNAME("Edit"), SNAME("EditorIcons")));

	clear_button->set_icon(get_theme_icon(SNAME("Clear"), SNAME("EditorIcons")));
	copy_button->set_icon(get_theme_icon(SNAME("ActionCopy"), SNAME("EditorIcons")));
	collapse_button->set_icon(get_theme_icon(SNAME("CombineLines"), SNAME("EditorIcons")));
	show_search_button->set_icon(get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
}

void EditorLog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_theme();
			_load_state();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_update_theme();
			_rebuild_log();
		} break;
		default:
			break;
	}
}

void EditorLog::_set_collapse(bool p_collapse) {
	collapse = p_collapse;
	_start_state_save_timer();
	_rebuild_log();
}

void EditorLog::_start_state_save_timer() {
	if (!is_loading_state) {
		save_state_timer->start();
	}
}

void EditorLog::_save_state() {
	Ref<ConfigFile> config;
	config.instantiate();
	// Load and amend existing config if it exists.
	config->load(EditorSettings::get_singleton()->get_project_settings_dir().plus_file("editor_layout.cfg"));

	const String section = "editor_log";
	for (const KeyValue<MessageType, LogFilter *> &E : type_filter_map) {
		config->set_value(section, "log_filter_" + itos(E.key), E.value->is_active());
	}

	config->set_value(section, "collapse", collapse);
	config->set_value(section, "show_search", search_box->is_visible());

	config->save(EditorSettings::get_singleton()->get_project_settings_dir().plus_file("editor_layout.cfg"));
}

void EditorLog::_load_state() {
	is_loading_state = true;

	Ref<ConfigFile> config;
	config.instantiate();
	config->load(EditorSettings::get_singleton()->get_project_settings_dir().plus_file("editor_layout.cfg"));

	// Run the below code even if config->load returns an error, since we want the defaults to be set even if the file does not exist yet.
	const String section = "editor_log";
	for (const KeyValue<MessageType, LogFilter *> &E : type_filter_map) {
		E.value->set_active(config->get_value(section, "log_filter_" + itos(E.key), true));
	}

	collapse = config->get_value(section, "collapse", false);
	collapse_button->set_pressed(collapse);
	bool show_search = config->get_value(section, "show_search", true);
	search_box->set_visible(show_search);
	show_search_button->set_pressed(show_search);

	is_loading_state = false;
}

void EditorLog::_clear_request() {
	log->clear();
	messages.clear();
	_reset_message_counts();
	tool_button->set_icon(Ref<Texture2D>());
}

void EditorLog::_copy_request() {
	String text = log->get_selected_text();

	if (text == "") {
		text = log->get_text();
	}

	if (text != "") {
		DisplayServer::get_singleton()->clipboard_set(text);
	}
}

void EditorLog::clear() {
	_clear_request();
}

void EditorLog::_process_message(const String &p_msg, MessageType p_type) {
	if (messages.size() > 0 && messages[messages.size() - 1].text == p_msg) {
		// If previous message is the same as the new one, increase previous count rather than adding another
		// instance to the messages list.
		LogMessage &previous = messages.write[messages.size() - 1];
		previous.count++;

		_add_log_line(previous, collapse);
	} else {
		// Different message to the previous one received.
		LogMessage message(p_msg, p_type);
		_add_log_line(message);
		messages.push_back(message);
	}

	type_filter_map[p_type]->set_message_count(type_filter_map[p_type]->get_message_count() + 1);
}

void EditorLog::add_message(const String &p_msg, MessageType p_type) {
	// Make text split by new lines their own message.
	// See #41321 for reasoning. At time of writing, multiple print()'s in running projects
	// get grouped together and sent to the editor log as one message. This can mess with the
	// search functionality (see the comments on the PR above for more details). This behaviour
	// also matches that of other IDE's.
	Vector<String> lines = p_msg.split("\n", true);

	for (int i = 0; i < lines.size(); i++) {
		_process_message(lines[i], p_type);
	}
}

void EditorLog::set_tool_button(Button *p_tool_button) {
	tool_button = p_tool_button;
}

void EditorLog::_undo_redo_cbk(void *p_self, const String &p_name) {
	EditorLog *self = (EditorLog *)p_self;
	self->add_message(p_name, EditorLog::MSG_TYPE_EDITOR);
}

void EditorLog::_rebuild_log() {
	log->clear();

	for (int msg_idx = 0; msg_idx < messages.size(); msg_idx++) {
		LogMessage msg = messages[msg_idx];

		if (collapse) {
			// If collapsing, only log one instance of the message.
			_add_log_line(msg);
		} else {
			// If not collapsing, log each instance on a line.
			for (int i = 0; i < msg.count; i++) {
				_add_log_line(msg);
			}
		}
	}
}

void EditorLog::_add_log_line(LogMessage &p_message, bool p_replace_previous) {
	// Only add the message to the log if it passes the filters.
	bool filter_active = type_filter_map[p_message.type]->is_active();
	String search_text = search_box->get_text();
	bool search_match = search_text == String() || p_message.text.findn(search_text) > -1;

	if (!filter_active || !search_match) {
		return;
	}

	if (p_replace_previous) {
		// Remove last line if replacing, as it will be replace by the next added line.
		// Why "- 2"? RichTextLabel is weird. When you add a line with add_newline(), it also adds an element to the list of lines which is null/blank,
		// but it still counts as a line. So if you remove the last line (count - 1) you are actually removing nothing...
		log->remove_line(log->get_paragraph_count() - 2);
	}

	switch (p_message.type) {
		case MSG_TYPE_STD: {
		} break;
		case MSG_TYPE_ERROR: {
			log->push_color(get_theme_color(SNAME("error_color"), SNAME("Editor")));
			Ref<Texture2D> icon = get_theme_icon(SNAME("Error"), SNAME("EditorIcons"));
			log->add_image(icon);
			log->add_text(" ");
			tool_button->set_icon(icon);
		} break;
		case MSG_TYPE_WARNING: {
			log->push_color(get_theme_color(SNAME("warning_color"), SNAME("Editor")));
			Ref<Texture2D> icon = get_theme_icon(SNAME("Warning"), SNAME("EditorIcons"));
			log->add_image(icon);
			log->add_text(" ");
			tool_button->set_icon(icon);
		} break;
		case MSG_TYPE_EDITOR: {
			// Distinguish editor messages from messages printed by the project
			log->push_color(get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.6));
		} break;
	}

	// If collapsing, add the count of this message in bold at the start of the line.
	if (collapse && p_message.count > 1) {
		log->push_bold();
		log->add_text(vformat("(%s) ", itos(p_message.count)));
		log->pop();
	}

	log->add_text(p_message.text);

	// Need to use pop() to exit out of the RichTextLabels current "push" stack.
	// We only "push" in the above switch when message type != STD, so only pop when that is the case.
	if (p_message.type != MSG_TYPE_STD) {
		log->pop();
	}

	log->add_newline();
}

void EditorLog::_set_filter_active(bool p_active, MessageType p_message_type) {
	type_filter_map[p_message_type]->set_active(p_active);
	_start_state_save_timer();
	_rebuild_log();
}

void EditorLog::_set_search_visible(bool p_visible) {
	search_box->set_visible(p_visible);
	if (p_visible) {
		search_box->grab_focus();
	}
	_start_state_save_timer();
}

void EditorLog::_search_changed(const String &p_text) {
	_rebuild_log();
}

void EditorLog::_reset_message_counts() {
	for (const KeyValue<MessageType, LogFilter *> &E : type_filter_map) {
		E.value->set_message_count(0);
	}
}

void EditorLog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("clear_request"));
	ADD_SIGNAL(MethodInfo("copy_request"));
}

EditorLog::EditorLog() {
	save_state_timer = memnew(Timer);
	save_state_timer->set_wait_time(2);
	save_state_timer->set_one_shot(true);
	save_state_timer->connect("timeout", callable_mp(this, &EditorLog::_save_state));
	add_child(save_state_timer);

	HBoxContainer *hb = this;

	VBoxContainer *vb_left = memnew(VBoxContainer);
	vb_left->set_custom_minimum_size(Size2(0, 180) * EDSCALE);
	vb_left->set_v_size_flags(SIZE_EXPAND_FILL);
	vb_left->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(vb_left);

	// Log - Rich Text Label.
	log = memnew(RichTextLabel);
	log->set_scroll_follow(true);
	log->set_selection_enabled(true);
	log->set_focus_mode(FOCUS_CLICK);
	log->set_v_size_flags(SIZE_EXPAND_FILL);
	log->set_h_size_flags(SIZE_EXPAND_FILL);
	vb_left->add_child(log);

	// Search box
	search_box = memnew(LineEdit);
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_box->set_placeholder(TTR("Filter messages"));
	search_box->set_right_icon(get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
	search_box->set_clear_button_enabled(true);
	search_box->set_visible(true);
	search_box->connect("text_changed", callable_mp(this, &EditorLog::_search_changed));
	vb_left->add_child(search_box);

	VBoxContainer *vb_right = memnew(VBoxContainer);
	hb->add_child(vb_right);

	// Tools grid
	HBoxContainer *hb_tools = memnew(HBoxContainer);
	hb_tools->set_h_size_flags(SIZE_SHRINK_CENTER);
	vb_right->add_child(hb_tools);

	// Clear.
	clear_button = memnew(Button);
	clear_button->set_flat(true);
	clear_button->set_focus_mode(FOCUS_NONE);
	clear_button->set_shortcut(ED_SHORTCUT("editor/clear_output", TTR("Clear Output"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::K));
	clear_button->set_shortcut_context(this);
	clear_button->connect("pressed", callable_mp(this, &EditorLog::_clear_request));
	hb_tools->add_child(clear_button);

	// Copy.
	copy_button = memnew(Button);
	copy_button->set_flat(true);
	copy_button->set_focus_mode(FOCUS_NONE);
	copy_button->set_shortcut(ED_SHORTCUT("editor/copy_output", TTR("Copy Selection"), KeyModifierMask::CMD | Key::C));
	copy_button->set_shortcut_context(this);
	copy_button->connect("pressed", callable_mp(this, &EditorLog::_copy_request));
	hb_tools->add_child(copy_button);

	// A second hbox to make a 2x2 grid of buttons.
	HBoxContainer *hb_tools2 = memnew(HBoxContainer);
	hb_tools2->set_h_size_flags(SIZE_SHRINK_CENTER);
	vb_right->add_child(hb_tools2);

	// Collapse.
	collapse_button = memnew(Button);
	collapse_button->set_flat(true);
	collapse_button->set_focus_mode(FOCUS_NONE);
	collapse_button->set_tooltip(TTR("Collapse duplicate messages into one log entry. Shows number of occurrences."));
	collapse_button->set_toggle_mode(true);
	collapse_button->set_pressed(false);
	collapse_button->connect("toggled", callable_mp(this, &EditorLog::_set_collapse));
	hb_tools2->add_child(collapse_button);

	// Show Search.
	show_search_button = memnew(Button);
	show_search_button->set_flat(true);
	show_search_button->set_focus_mode(FOCUS_NONE);
	show_search_button->set_toggle_mode(true);
	show_search_button->set_pressed(true);
	show_search_button->set_shortcut(ED_SHORTCUT("editor/open_search", TTR("Focus Search/Filter Bar"), KeyModifierMask::CMD | Key::F));
	show_search_button->set_shortcut_context(this);
	show_search_button->connect("toggled", callable_mp(this, &EditorLog::_set_search_visible));
	hb_tools2->add_child(show_search_button);

	// Message Type Filters.
	vb_right->add_child(memnew(HSeparator));

	LogFilter *std_filter = memnew(LogFilter(MSG_TYPE_STD));
	std_filter->initialize_button(TTR("Toggle visibility of standard output messages."), callable_mp(this, &EditorLog::_set_filter_active));
	vb_right->add_child(std_filter->toggle_button);
	type_filter_map.insert(MSG_TYPE_STD, std_filter);

	LogFilter *error_filter = memnew(LogFilter(MSG_TYPE_ERROR));
	error_filter->initialize_button(TTR("Toggle visibility of errors."), callable_mp(this, &EditorLog::_set_filter_active));
	vb_right->add_child(error_filter->toggle_button);
	type_filter_map.insert(MSG_TYPE_ERROR, error_filter);

	LogFilter *warning_filter = memnew(LogFilter(MSG_TYPE_WARNING));
	warning_filter->initialize_button(TTR("Toggle visibility of warnings."), callable_mp(this, &EditorLog::_set_filter_active));
	vb_right->add_child(warning_filter->toggle_button);
	type_filter_map.insert(MSG_TYPE_WARNING, warning_filter);

	LogFilter *editor_filter = memnew(LogFilter(MSG_TYPE_EDITOR));
	editor_filter->initialize_button(TTR("Toggle visibility of editor messages."), callable_mp(this, &EditorLog::_set_filter_active));
	vb_right->add_child(editor_filter->toggle_button);
	type_filter_map.insert(MSG_TYPE_EDITOR, editor_filter);

	add_message(VERSION_FULL_NAME " (c) 2007-2021 Juan Linietsky, Ariel Manzur & Godot Contributors.");

	eh.errfunc = _error_handler;
	eh.userdata = this;
	add_error_handler(&eh);

	current = Thread::get_caller_id();

	EditorNode::get_undo_redo()->set_commit_notify_callback(_undo_redo_cbk, this);
}

void EditorLog::deinit() {
	remove_error_handler(&eh);
}

EditorLog::~EditorLog() {
	for (const KeyValue<MessageType, LogFilter *> &E : type_filter_map) {
		memdelete(E.value);
	}
}
