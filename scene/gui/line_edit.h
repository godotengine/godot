/*************************************************************************/
/*  line_edit.h                                                          */
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

#ifndef LINE_EDIT_H
#define LINE_EDIT_H

#include "scene/gui/control.h"
#include "scene/gui/popup_menu.h"

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
	bool text_changed_dirty;

	String undo_text;
	String text;
	String placeholder;
	String placeholder_translated;
	String secret_character;
	float placeholder_alpha;
	String ime_text;
	Point2 ime_selection;

	bool selecting_enabled;

	bool context_menu_enabled;
	PopupMenu *menu;

	int cursor_pos;
	int window_pos;
	int max_length; // 0 for no maximum.

	int cached_width;
	int cached_placeholder_width;

	bool clear_button_enabled;

	bool shortcut_keys_enabled;

	bool virtual_keyboard_enabled = true;

	Ref<Texture2D> right_icon;

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
		int window_pos;
		int cached_width;
		String text;
	};
	List<TextOperation> undo_stack;
	List<TextOperation>::Element *undo_stack_pos;

	struct ClearButtonStatus {
		bool press_attempt;
		bool pressing_inside;
	} clear_button_status;

	bool _is_over_clear_button(const Point2 &p_pos) const;

	void _clear_undo_stack();
	void _clear_redo();
	void _create_undo_state();

	void _generate_context_menu();

	Timer *caret_blink_timer;

	void _text_changed();
	void _emit_text_change();
	bool expand_to_text_length;

	void update_cached_width();
	void update_placeholder_width();

	bool caret_blink_enabled;
	bool caret_force_displayed;
	bool draw_caret;
	bool window_has_focus;

	void shift_selection_check_pre(bool);
	void shift_selection_check_post(bool);

	void selection_fill_at_cursor();
	void set_window_pos(int p_pos);

	void set_cursor_at_pixel_pos(int p_x);
	int get_cursor_pixel_pos();

	void _reset_caret_blink_timer();
	void _toggle_draw_caret();

	void clear_internal();
	void changed_internal();

	void _editor_settings_changed();

	void _gui_input(Ref<InputEvent> p_event);
	void _notification(int p_what);

protected:
	static void _bind_methods();

public:
	void set_align(Align p_align);
	Align get_align() const;

	virtual Variant get_drag_data(const Point2 &p_point) override;
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const override;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data) override;

	virtual CursorShape get_cursor_shape(const Point2 &p_pos) const override;

	void menu_option(int p_option);
	void set_context_menu_enabled(bool p_enable);
	bool is_context_menu_enabled();
	PopupMenu *get_menu() const;

	void select(int p_from = 0, int p_to = -1);
	void select_all();
	void selection_delete();
	void deselect();

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

	bool cursor_get_force_displayed() const;
	void cursor_set_force_displayed(const bool p_enabled);

	void copy_text();
	void cut_text();
	void paste_text();
	void undo();
	void redo();

	void set_editable(bool p_editable);
	bool is_editable() const;

	void set_secret(bool p_secret);
	bool is_secret() const;

	void set_secret_character(const String &p_string);
	String get_secret_character() const;

	virtual Size2 get_minimum_size() const override;

	void set_expand_to_text_length(bool p_enabled);
	bool get_expand_to_text_length() const;

	void set_clear_button_enabled(bool p_enabled);
	bool is_clear_button_enabled() const;

	void set_shortcut_keys_enabled(bool p_enabled);
	bool is_shortcut_keys_enabled() const;

	void set_virtual_keyboard_enabled(bool p_enable);
	bool is_virtual_keyboard_enabled() const;

	void set_selecting_enabled(bool p_enabled);
	bool is_selecting_enabled() const;

	void set_right_icon(const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_right_icon();

	virtual bool is_text_field() const override;
	LineEdit();
	~LineEdit();
};

VARIANT_ENUM_CAST(LineEdit::Align);
VARIANT_ENUM_CAST(LineEdit::MenuItems);

#endif
