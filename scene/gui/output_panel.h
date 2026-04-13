/**************************************************************************/
/*  output_panel.h                                                        */
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

#pragma once

#include "core/os/thread.h"
#include "scene/gui/control.h"

class RichTextLabel;
class StyleBox;

class OutputPanel : public Control {
	GDCLASS(OutputPanel, Control);

public:
	enum MessageType {
		MSG_TYPE_STD,
		MSG_TYPE_STD_RICH,
		MSG_TYPE_WARNING,
		MSG_TYPE_ERROR,
	};

private:
	struct LogMessage {
		String text;
		MessageType type;
		int count = 1;
		bool clear = true;

		LogMessage() {}
		LogMessage(const String &p_text, MessageType p_type, bool p_clear) : text(p_text), type(p_type), clear(p_clear) {}
	};

	struct ThemeCache {
		Ref<StyleBox> panel_style;
		Ref<Font> normal_font;
		Ref<Font> bold_font;
		int font_size = 0;
		Color error_color;
		Color warning_color;
		Color message_color;
	} theme_cache;

	static constexpr float PADDING = 8.0f;

	int max_lines = 500;
	bool show_std = true;
	bool show_warnings = true;
	bool show_errors = true;
	bool debug_only = true;

	Vector<LogMessage> messages;
	RichTextLabel *log = nullptr;

	PrintHandlerList print_handler;
	ErrorHandlerList error_handler;

	static void _print_handler(void *p_self, const String &p_string, bool p_error, bool p_rich);
	static void _error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, bool p_editor_notify, ErrorHandlerType p_type);

	void _process_message(const String &p_msg, MessageType p_type, bool p_clear);
	void _add_log_line(LogMessage &p_message);
	void _rebuild_log();
	void _update_log_rect();
	bool _should_display(const LogMessage &p_message) const;
	void _update_theme();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void add_message(const String &p_msg, MessageType p_type = MSG_TYPE_STD);

	void set_max_lines(int p_max_lines);
	int get_max_lines() const;

	void set_show_std(bool p_show);
	bool get_show_std() const;
	void set_show_warnings(bool p_show);
	bool get_show_warnings() const;
	void set_show_errors(bool p_show);
	bool get_show_errors() const;

	void set_debug_only(bool p_debug_only);
	bool get_debug_only() const;

	void clear();

	OutputPanel();
	~OutputPanel();
};

VARIANT_ENUM_CAST(OutputPanel::MessageType);
