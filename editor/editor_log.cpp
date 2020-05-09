/*************************************************************************/
/*  editor_log.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "scene/resources/dynamic_font.h"

void EditorLog::_error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, ErrorHandlerType p_type) {

	EditorLog *self = (EditorLog *)p_self;
	if (self->current != Thread::get_caller_id())
		return;

	String err_str;
	if (p_errorexp && p_errorexp[0]) {
		err_str = p_errorexp;
	} else {
		err_str = String(p_file) + ":" + itos(p_line) + " - " + String(p_error);
	}

	if (p_type == ERR_HANDLER_WARNING) {
		self->add_message(err_str, MSG_TYPE_WARNING);
	} else {
		self->add_message(err_str, MSG_TYPE_ERROR);
	}
}

void EditorLog::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		log->add_theme_font_override("normal_font", get_theme_font("output_source", "EditorFonts"));
	} else if (p_what == NOTIFICATION_THEME_CHANGED) {
		Ref<DynamicFont> df_output_code = get_theme_font("output_source", "EditorFonts");
		if (df_output_code.is_valid()) {
			if (log != nullptr) {
				log->add_theme_font_override("normal_font", get_theme_font("output_source", "EditorFonts"));
			}
		}
	}
}

void EditorLog::_clear_request() {

	log->clear();
	tool_button->set_icon(Ref<Texture2D>());
}

void EditorLog::_copy_request() {

	log->selection_copy();
}

void EditorLog::clear() {
	_clear_request();
}

void EditorLog::copy() {
	_copy_request();
}

void EditorLog::add_message(const String &p_msg, MessageType p_type) {

	log->add_newline();

	bool restore = p_type != MSG_TYPE_STD;
	switch (p_type) {
		case MSG_TYPE_STD: {
		} break;
		case MSG_TYPE_ERROR: {
			log->push_color(get_theme_color("error_color", "Editor"));
			Ref<Texture2D> icon = get_theme_icon("Error", "EditorIcons");
			log->add_image(icon);
			log->add_text(" ");
			tool_button->set_icon(icon);
		} break;
		case MSG_TYPE_WARNING: {
			log->push_color(get_theme_color("warning_color", "Editor"));
			Ref<Texture2D> icon = get_theme_icon("Warning", "EditorIcons");
			log->add_image(icon);
			log->add_text(" ");
			tool_button->set_icon(icon);
		} break;
		case MSG_TYPE_EDITOR: {
			// Distinguish editor messages from messages printed by the project
			log->push_color(get_theme_color("font_color", "Editor") * Color(1, 1, 1, 0.6));
		} break;
	}

	log->add_text(p_msg);

	if (restore)
		log->pop();
}

void EditorLog::set_tool_button(ToolButton *p_tool_button) {
	tool_button = p_tool_button;
}

void EditorLog::_undo_redo_cbk(void *p_self, const String &p_name) {

	EditorLog *self = (EditorLog *)p_self;
	self->add_message(p_name, EditorLog::MSG_TYPE_EDITOR);
}

void EditorLog::_gui_input(const Ref<InputEvent> &p_event) {

	const Ref<InputEventMouseButton> mb_ref = p_event;
	if (mb_ref.is_valid() && mb_ref->is_pressed() && mb_ref->get_button_index() == BUTTON_RIGHT) {
		const InputEventMouseButton &mb = **mb_ref;
		context_menu->set_position(get_global_transform().xform(mb.get_position()));
		context_menu->set_size(Size2(0, 0));
		context_menu->popup();
	}
}

void EditorLog::_on_context_menu_item_selected(int action_id) {

	switch (action_id) {
		case CONTEXT_COPY:
			_copy_request();
			break;

		case CONTEXT_CLEAR:
			_clear_request();
			break;
	}
}

void EditorLog::_bind_methods() {

	ADD_SIGNAL(MethodInfo("clear_request"));
	ADD_SIGNAL(MethodInfo("copy_request"));
}

EditorLog::EditorLog() {

	VBoxContainer *vb = this;

	log = memnew(RichTextLabel);
	log->set_scroll_follow(true);
	log->set_selection_enabled(true);
	log->set_focus_mode(FOCUS_CLICK);
	log->set_custom_minimum_size(Size2(0, 180) * EDSCALE);
	log->set_v_size_flags(SIZE_EXPAND_FILL);
	log->set_h_size_flags(SIZE_EXPAND_FILL);
	log->connect("gui_input", callable_mp(this, &EditorLog::_gui_input));
	vb->add_child(log);
	add_message(VERSION_FULL_NAME " (c) 2007-2020 Juan Linietsky, Ariel Manzur & Godot Contributors.");

	context_menu = memnew(PopupMenu);
	context_menu->connect("id_pressed", callable_mp(this, &EditorLog::_on_context_menu_item_selected));
	context_menu->add_shortcut(ED_SHORTCUT("editor/copy_output", TTR("Copy Selection"), KEY_MASK_CMD | KEY_C), CONTEXT_COPY);
	context_menu->add_shortcut(ED_SHORTCUT("editor/clear_output", TTR("Clear Output"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_K), CONTEXT_CLEAR);

	add_child(context_menu);

	eh.errfunc = _error_handler;
	eh.userdata = this;
	add_error_handler(&eh);

	current = Thread::get_caller_id();

	add_theme_constant_override("separation", get_theme_constant("separation", "VBoxContainer"));

	EditorNode::get_undo_redo()->set_commit_notify_callback(_undo_redo_cbk, this);
}

void EditorLog::deinit() {

	remove_error_handler(&eh);
}

EditorLog::~EditorLog() {
}
