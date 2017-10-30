/*************************************************************************/
/*  line_edit.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef LINE_EDIT_H
#define LINE_EDIT_H

#include "scene/gui/control.h"
#include "scene/gui/popup_menu.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class LineEdit : public Control {

	GDCLASS(LineEdit, Control);

public:
	enum Align {

		ALIGN_LEFT,
		ALIGN_CENTER,
		ALIGN_RIGHT,
		ALIGN_FILL
	};

	enum MenuItems {
		MENU_CUT,
		MENU_COPY,
		MENU_PASTE,
		MENU_CLEAR,
		MENU_SELECT_ALL,
		MENU_UNDO,
		MENU_REDO,
		MENU_MAX

	};

private:
	Align align;

	bool editable;
	bool pass;

	String undo_text;
	String text;
	String placeholder;
	float placeholder_alpha;
	String ime_text;
	Point2 ime_selection;

	PopupMenu *menu;

	int cursor_pos;
	int window_pos;
	int max_length; // 0 for no maximum

	int cached_width;

	struct Selection {

		int begin;
		int end;
		int cursor_start;
		bool enabled;
		bool creating;
		bool doubleclick;
		bool drag_attempt;
	} selection;

	struct TextOperation {
		int cursor_pos;
		String text;
	};
	List<TextOperation> undo_stack;
	List<TextOperation>::Element *undo_stack_pos;

	void _clear_undo_stack();
	void _clear_redo();
	void _create_undo_state();

	Timer *caret_blink_timer;

	static void _ime_text_callback(void *p_self, String p_text, Point2 p_selection);
	void _text_changed();
	void _emit_text_change();
	bool expand_to_text_length;

	bool caret_blink_enabled;
	bool draw_caret;
	bool window_has_focus;

	void shift_selection_check_pre(bool);
	void shift_selection_check_post(bool);

	void selection_clear();
	void selection_fill_at_cursor();
	void selection_delete();
	void set_window_pos(int p_pos);

	void set_cursor_at_pixel_pos(int p_x);

	void _reset_caret_blink_timer();
	void _toggle_draw_caret();

	void clear_internal();
	void changed_internal();

#ifdef TOOLS_ENABLED
	void _editor_settings_changed();
#endif

	void _gui_input(Ref<InputEvent> p_event);
	void _notification(int p_what);

protected:
	static void _bind_methods();

public:
	void set_align(Align p_align);
	Align get_align() const;

	virtual Variant get_drag_data(const Point2 &p_point);
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data);

	void menu_option(int p_option);
	PopupMenu *get_menu() const;

	void select_all();

	void delete_char();
	void delete_text(int p_from_column, int p_to_column);
	void set_text(String p_text);
	String get_text() const;
	void set_placeholder(String p_text);
	String get_placeholder() const;
	void set_placeholder_alpha(float p_alpha);
	float get_placeholder_alpha() const;
	void set_cursor_position(int p_pos);
	int get_cursor_position() const;
	void set_max_length(int p_max_length);
	int get_max_length() const;
	void append_at_cursor(String p_text);
	void clear();

	bool cursor_get_blink_enabled() const;
	void cursor_set_blink_enabled(const bool p_enabled);

	float cursor_get_blink_speed() const;
	void cursor_set_blink_speed(const float p_speed);

	void copy_text();
	void cut_text();
	void paste_text();
	void undo();
	void redo();

	void set_editable(bool p_editable);
	bool is_editable() const;

	void set_secret(bool p_secret);
	bool is_secret() const;

	void select(int p_from = 0, int p_to = -1);

	virtual Size2 get_minimum_size() const;

	void set_expand_to_text_length(bool p_enabled);
	bool get_expand_to_text_length() const;

	virtual bool is_text_field() const;
	LineEdit();
	~LineEdit();
};

VARIANT_ENUM_CAST(LineEdit::Align);
VARIANT_ENUM_CAST(LineEdit::MenuItems);

#endif
