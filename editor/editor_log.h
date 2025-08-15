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

#pragma once

#include "core/os/thread.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/rich_text_label.h"

class UndoRedo;
class CheckBox;

class EditorLog : public HBoxContainer {
	GDCLASS(EditorLog, HBoxContainer);

public:
	enum MessageType {
		MSG_TYPE_STD,
		MSG_TYPE_ERROR,
		MSG_TYPE_STD_RICH,
		MSG_TYPE_WARNING,
		MSG_TYPE_EDITOR,
	};

	bool is_whole_words() const { return _whole_words; }
	bool is_match_case() const { return _match_case; }

private:
	struct LogMessage {
		String text;
		MessageType type;
		int count = 1;
		bool clear = true;

		LogMessage() {}

		LogMessage(const String &p_text, MessageType p_type, bool p_clear) :
				text(p_text),
				type(p_type),
				clear(p_clear) {
		}
	};

	struct {
		Color error_color;
		Ref<Texture2D> error_icon;

		Color warning_color;
		Ref<Texture2D> warning_icon;

		Color message_color;
	} theme_cache;

	struct SearchMatch {
		int message_index;
		int visible_index;
		int text_position;
		int display_instance = 0; // Which instance of a repeated message (for non-collapsed mode)
	};

	// Encapsulates all data and functionality regarding filters.
	struct LogFilter {
	private:
		// Force usage of set method since it has functionality built-in.
		int message_count = 0;
		bool active = true;

	public:
		MessageType type;
		Button *toggle_button = nullptr;

		void initialize_button(const String &p_name, const String &p_tooltip, Callable p_toggled_callback) {
			toggle_button = memnew(Button);
			toggle_button->set_toggle_mode(true);
			toggle_button->set_pressed(true);
			toggle_button->set_text(itos(message_count));
			toggle_button->set_accessibility_name(TTRGET(p_name));
			toggle_button->set_tooltip_text(TTRGET(p_tooltip));
			toggle_button->set_focus_mode(FOCUS_ACCESSIBILITY);
			// When toggled call the callback and pass the MessageType this button is for.
			toggle_button->connect(SceneStringName(toggled), p_toggled_callback.bind(type));
		}

		int get_message_count() {
			return message_count;
		}

		void set_message_count(int p_count) {
			message_count = p_count;
			toggle_button->set_text(itos(message_count));
		}

		bool is_active() {
			return active;
		}

		void set_active(bool p_active) {
			toggle_button->set_pressed(p_active);
			active = p_active;
		}

		LogFilter(MessageType p_type) :
				type(p_type) {
		}
	};

	int line_limit = 10000;

	Vector<LogMessage> messages;
	// Maps MessageTypes to LogFilters for convenient access and storage (don't need 1 member per filter).
	HashMap<MessageType, LogFilter *> type_filter_map;

	RichTextLabel *log = nullptr;

	Button *clear_button = nullptr;
	Button *copy_button = nullptr;

	Button *collapse_button = nullptr;
	bool collapse = false;

	Button *show_search_button = nullptr;
	LineEdit *search_box = nullptr;

	HBoxContainer *search_nav_container = nullptr;
	Button *search_mode_button = nullptr;
	Button *search_prev_button = nullptr;
	Button *search_next_button = nullptr;
	Label *search_results_label = nullptr;
	CheckBox *_match_case_checkbox = nullptr;
	CheckBox *_whole_words_checkbox = nullptr;
	bool search_mode = false; // If true, search box is in "search mode" (searches for text), otherwise it is in "filter mode" (filters messages by text).
	bool _match_case = false;
	bool _whole_words = false;
	// Classic search state
	int current_search_index = -1;
	Vector<SearchMatch> search_matches; // Stores paragraph indices of matches

	// Reference to the "Output" button on the toolbar so we can update its icon when warnings or errors are encountered.
	Button *tool_button = nullptr;

	bool is_loading_state = false; // Used to disable saving requests while loading (some signals from buttons will try to trigger a save, which happens during loading).
	Timer *save_state_timer = nullptr;

	static void _error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, bool p_editor_notify, ErrorHandlerType p_type);

	ErrorHandlerList eh;

	//void _dragged(const Point2& p_ofs);
	void _meta_clicked(const String &p_meta);
	void _clear_request();
	void _copy_request();
	static void _undo_redo_cbk(void *p_self, const String &p_name);

	void _rebuild_log(const String &p_search_highlight = String());
	void _add_log_line(LogMessage &p_message, bool p_replace_previous = false,
			int p_message_index = -1, int p_display_instance = 0,
			const String &p_search_highlight = String());
	bool _check_display_message(LogMessage &p_message, bool p_ignore_search_filter = false);

	void _set_filter_active(bool p_active, MessageType p_message_type);
	void _set_search_visible(bool p_visible);
	void _search_changed(const String &p_text);
	void set_whole_words(bool p_whole_word);
	void set_match_case(bool p_match_case);
	void _perform_classic_search(const String &p_text);
	void _find_search_matches(const String &p_search_text);
	Vector<int> _find_matches_in_text(const String &p_text, const String &p_pattern);
	void _add_text_with_search_highlighting(const String &p_text, const String &p_search_text, int p_message_index = -1, int p_display_instance = 0);
	void _scroll_to_current_match();
	void _scroll_to_current_match_deferred(bool p_restore_scroll_follow);
	void _navigate_search_result(bool p_next);
	void _update_search_navigation();
	void _clear_search_highlights();
	void _on_search_mode_toggled(bool p_pressed);

	void _process_message(const String &p_msg, MessageType p_type, bool p_clear);
	void _reset_message_counts();

	void _set_collapse(bool p_collapse);

	void _start_state_save_timer();
	void _save_state();
	void _load_state();

	void _update_theme();
	void _editor_settings_changed();

protected:
	void _notification(int p_what);

public:
	void add_message(const String &p_msg, MessageType p_type = MSG_TYPE_STD);
	void set_tool_button(Button *p_tool_button);
	void register_undo_redo(UndoRedo *p_undo_redo);
	void deinit();

	void clear();

	EditorLog();
	~EditorLog();
};

VARIANT_ENUM_CAST(EditorLog::MessageType);
