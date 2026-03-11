/**************************************************************************/
/*  editor_log.h                                                          */
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

#ifndef EDITOR_LOG_H
#define EDITOR_LOG_H

#include "scene/gui/control.h"
#include "scene/gui/label.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/texture_button.h"
//#include "scene/gui/empty_control.h"
#include "core/os/thread.h"
#include "pane_drag.h"
#include "scene/gui/box_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tool_button.h"

class RegEx;

class EditorLog : public VBoxContainer {
	GDCLASS(EditorLog, VBoxContainer);

	struct {
		Color error_color;
		Ref<Texture> error_icon;

		Color warning_color;
		Ref<Texture> warning_icon;

		Color message_color;

	} theme_cache;

	struct
	{
		Color _default;
		Color white;
		Color black;
		Color red;
		Color green;
		Color blue;
		Color yellow;
		Color magenta;
		Color cyan;
		Color bright_white;
		Color bright_black;
		Color bright_red;
		Color bright_green;
		Color bright_blue;
		Color bright_yellow;
		Color bright_magenta;
		Color bright_cyan;
	} terminal_colors;

	Button *clearbutton;
	Button *copybutton;
	Label *title;
	RichTextLabel *log;
	HBoxContainer *title_hb;
	//PaneDrag *pd;
	ToolButton *tool_button;

	static void _error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, ErrorHandlerType p_type);

	ErrorHandlerList eh;

	Thread::ID current;

	RegEx *strip_ansi_regex = nullptr;

	//void _dragged(const Point2& p_ofs);
	void _clear_request();
	void _copy_request();
	static void _undo_redo_cbk(void *p_self, const String &p_name);

	void _log_add_text(const String &p_msg);
	bool _log_push_terminal_color(TerminalColor::Col p_color);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	enum MessageType {
		MSG_TYPE_STD,
		MSG_TYPE_ERROR,
		MSG_TYPE_WARNING,
		MSG_TYPE_EDITOR
	};

	void add_message(const String &p_msg, MessageType p_type = MSG_TYPE_STD);
	void set_tool_button(ToolButton *p_tool_button);
	void deinit();

	void clear();
	void copy();
	EditorLog();
	~EditorLog();
};

#endif // EDITOR_LOG_H
