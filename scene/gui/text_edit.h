/*************************************************************************/
/*  text_edit.h                                                          */
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
#ifndef TEXT_EDIT_H
#define TEXT_EDIT_H

#include "scene/gui/control.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/scroll_bar.h"
#include "scene/main/timer.h"

class TextEdit : public Control {

	GDCLASS(TextEdit, Control);

	struct Cursor {
		int last_fit_x;
		int line, column; ///< cursor
		int x_ofs, line_ofs;
	} cursor;

	struct Selection {

		enum Mode {

			MODE_NONE,
			MODE_SHIFT,
			MODE_POINTER
		};

		Mode selecting_mode;
		int selecting_line, selecting_column;
		bool selecting_text;

		bool active;

		int from_line, from_column;
		int to_line, to_column;

		bool shiftclick_left;

	} selection;

	struct Cache {

		Ref<Texture> tab_icon;
		Ref<StyleBox> style_normal;
		Ref<StyleBox> style_focus;
		Ref<Font> font;
		Color completion_background_color;
		Color completion_selected_color;
		Color completion_existing_color;
		Color completion_font_color;
		Color caret_color;
		Color caret_background_color;
		Color line_number_color;
		Color font_color;
		Color font_selected_color;
		Color keyword_color;
		Color number_color;
		Color function_color;
		Color member_variable_color;
		Color selection_color;
		Color mark_color;
		Color breakpoint_color;
		Color current_line_color;
		Color line_length_guideline_color;
		Color brace_mismatch_color;
		Color word_highlighted_color;
		Color search_result_color;
		Color search_result_border_color;
		Color symbol_color;
		Color background_color;

		int row_height;
		int line_spacing;
		int line_number_w;
		int breakpoint_gutter_width;
		Size2 size;
	} cache;

	struct ColorRegion {

		Color color;
		String begin_key;
		String end_key;
		bool line_only;
		bool eq;
		ColorRegion(const String &p_begin_key = "", const String &p_end_key = "", const Color &p_color = Color(), bool p_line_only = false) {
			begin_key = p_begin_key;
			end_key = p_end_key;
			color = p_color;
			line_only = p_line_only || p_end_key == "";
			eq = begin_key == end_key;
		}
	};

	class Text {
	public:
		struct ColorRegionInfo {

			int region;
			bool end;
		};

		struct Line {
			int width_cache : 24;
			bool marked : 1;
			bool breakpoint : 1;
			Map<int, ColorRegionInfo> region_info;
			String data;
		};

	private:
		const Vector<ColorRegion> *color_regions;
		mutable Vector<Line> text;
		Ref<Font> font;
		int indent_size;

		void _update_line_cache(int p_line) const;

	public:
		void set_indent_size(int p_indent_size);
		void set_font(const Ref<Font> &p_font);
		void set_color_regions(const Vector<ColorRegion> *p_regions) { color_regions = p_regions; }
		int get_line_width(int p_line) const;
		int get_max_width() const;
		const Map<int, ColorRegionInfo> &get_color_region_info(int p_line);
		void set(int p_line, const String &p_text);
		void set_marked(int p_line, bool p_marked) { text[p_line].marked = p_marked; }
		bool is_marked(int p_line) const { return text[p_line].marked; }
		void set_breakpoint(int p_line, bool p_breakpoint) { text[p_line].breakpoint = p_breakpoint; }
		bool is_breakpoint(int p_line) const { return text[p_line].breakpoint; }
		void insert(int p_at, const String &p_text);
		void remove(int p_at);
		int size() const { return text.size(); }
		void clear();
		void clear_caches();
		_FORCE_INLINE_ const String &operator[](int p_line) const { return text[p_line].data; }
		Text() { indent_size = 4; }
	};

	struct TextOperation {

		enum Type {
			TYPE_NONE,
			TYPE_INSERT,
			TYPE_REMOVE
		};

		Type type;
		int from_line, from_column;
		int to_line, to_column;
		String text;
		uint32_t prev_version;
		uint32_t version;
		bool chain_forward;
		bool chain_backward;
	};

	String ime_text;
	Point2 ime_selection;

	TextOperation current_op;

	List<TextOperation> undo_stack;
	List<TextOperation>::Element *undo_stack_pos;

	void _clear_redo();
	void _do_text_op(const TextOperation &p_op, bool p_reverse);

	//syntax coloring
	HashMap<String, Color> keywords;

	Vector<ColorRegion> color_regions;

	Set<String> completion_prefixes;
	bool completion_enabled;
	Vector<String> completion_strings;
	Vector<String> completion_options;
	bool completion_active;
	bool completion_forced;
	String completion_current;
	String completion_base;
	int completion_index;
	Rect2i completion_rect;
	int completion_line_ofs;
	String completion_hint;
	int completion_hint_offset;

	bool setting_text;

	// data
	Text text;

	uint32_t version;
	uint32_t saved_version;

	int max_chars;
	bool readonly;
	bool syntax_coloring;
	bool indent_using_spaces;
	int indent_size;
	String space_indent;

	Timer *caret_blink_timer;
	bool caret_blink_enabled;
	bool draw_caret;
	bool window_has_focus;
	bool block_caret;

	bool setting_row;
	bool wrap;
	bool draw_tabs;
	bool cursor_changed_dirty;
	bool text_changed_dirty;
	bool undo_enabled;
	bool line_numbers;
	bool line_numbers_zero_padded;
	bool line_length_guideline;
	int line_length_guideline_col;
	bool draw_breakpoint_gutter;
	int breakpoint_gutter_width;

	bool highlight_all_occurrences;
	bool scroll_past_end_of_file_enabled;
	bool auto_brace_completion_enabled;
	bool brace_matching_enabled;
	bool auto_indent;
	bool cut_copy_line;
	bool insert_mode;
	bool select_identifiers_enabled;

	bool smooth_scroll_enabled;
	bool scrolling;
	float target_v_scroll;
	float v_scroll_speed;

	bool raised_from_completion;

	String highlighted_word;

	uint64_t last_dblclk;

	Timer *idle_detect;
	Timer *click_select_held;
	HScrollBar *h_scroll;
	VScrollBar *v_scroll;
	bool updating_scrolls;

	Object *tooltip_obj;
	StringName tooltip_func;
	Variant tooltip_ud;

	bool next_operation_is_complex;

	bool callhint_below;
	Vector2 callhint_offset;

	String search_text;
	uint32_t search_flags;
	int search_result_line;
	int search_result_col;

	bool context_menu_enabled;

	int get_visible_rows() const;

	int get_char_count();

	int get_char_pos_for(int p_px, String p_str) const;
	int get_column_x_offset(int p_char, String p_str);

	void adjust_viewport_to_cursor();
	void _scroll_moved(double);
	void _update_scrollbars();
	void _v_scroll_input();
	void _click_selection_held();

	void _pre_shift_selection();
	void _post_shift_selection();

	void _scroll_lines_up();
	void _scroll_lines_down();

	static void _ime_text_callback(void *p_self, String p_text, Point2 p_selection);

	//void mouse_motion(const Point& p_pos, const Point& p_rel, int p_button_mask);
	Size2 get_minimum_size() const;

	int get_row_height() const;

	void _reset_caret_blink_timer();
	void _toggle_draw_caret();

	void _update_caches();
	void _cursor_changed_emit();
	void _text_changed_emit();

	void _push_current_op();

	/* super internal api, undo/redo builds on it */

	void _base_insert_text(int p_line, int p_char, const String &p_text, int &r_end_line, int &r_end_column);
	String _base_get_text(int p_from_line, int p_from_column, int p_to_line, int p_to_column) const;
	void _base_remove_text(int p_from_line, int p_from_column, int p_to_line, int p_to_column);

	int _get_column_pos_of_word(const String &p_key, const String &p_search, uint32_t p_search_flags, int p_from_column);

	PoolVector<int> _search_bind(const String &p_key, uint32_t p_search_flags, int p_from_line, int p_from_column) const;

	PopupMenu *menu;

	void _clear();
	void _cancel_completion();
	void _cancel_code_hint();
	void _confirm_completion();
	void _update_completion_candidates();

protected:
	virtual String get_tooltip(const Point2 &p_pos) const;

	void _insert_text(int p_line, int p_char, const String &p_text, int *r_end_line = NULL, int *r_end_char = NULL);
	void _remove_text(int p_from_line, int p_from_column, int p_to_line, int p_to_column);
	void _insert_text_at_cursor(const String &p_text);
	void _gui_input(const Ref<InputEvent> &p_gui_input);
	void _notification(int p_what);

	void _consume_pair_symbol(CharType ch);
	void _consume_backspace_for_pair_symbol(int prev_line, int prev_column);

	static void _bind_methods();

public:
	enum MenuItems {
		MENU_CUT,
		MENU_COPY,
		MENU_PASTE,
		MENU_CLEAR,
		MENU_SELECT_ALL,
		MENU_UNDO,
		MENU_MAX

	};

	enum SearchFlags {

		SEARCH_MATCH_CASE = 1,
		SEARCH_WHOLE_WORDS = 2,
		SEARCH_BACKWARDS = 4
	};

	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const;

	void _get_mouse_pos(const Point2i &p_mouse, int &r_row, int &r_col) const;

	//void delete_char();
	//void delete_line();

	void begin_complex_operation();
	void end_complex_operation();

	bool is_insert_text_operation();

	void set_text(String p_text);
	void insert_text_at_cursor(const String &p_text);
	void insert_at(const String &p_text, int at);
	int get_line_count() const;
	void set_line_as_marked(int p_line, bool p_marked);
	void set_line_as_breakpoint(int p_line, bool p_breakpoint);
	bool is_line_set_as_breakpoint(int p_line) const;
	void get_breakpoints(List<int> *p_breakpoints) const;
	String get_text();
	String get_line(int line) const;
	void set_line(int line, String new_text);
	void backspace_at_cursor();

	void indent_selection_left();
	void indent_selection_right();

	inline void set_scroll_pass_end_of_file(bool p_enabled) {
		scroll_past_end_of_file_enabled = p_enabled;
		update();
	}
	inline void set_auto_brace_completion(bool p_enabled) {
		auto_brace_completion_enabled = p_enabled;
	}
	inline void set_brace_matching(bool p_enabled) {
		brace_matching_enabled = p_enabled;
		update();
	}
	inline void set_callhint_settings(bool below, Vector2 offset) {
		callhint_below = below;
		callhint_offset = offset;
	}
	void set_auto_indent(bool p_auto_indent);

	void center_viewport_to_cursor();

	void cursor_set_column(int p_col, bool p_adjust_viewport = true);
	void cursor_set_line(int p_row, bool p_adjust_viewport = true);

	int cursor_get_column() const;
	int cursor_get_line() const;

	bool cursor_get_blink_enabled() const;
	void cursor_set_blink_enabled(const bool p_enabled);

	float cursor_get_blink_speed() const;
	void cursor_set_blink_speed(const float p_speed);

	void cursor_set_block_mode(const bool p_enable);
	bool cursor_is_block_mode() const;

	void set_readonly(bool p_readonly);

	void set_max_chars(int p_max_chars);
	void set_wrap(bool p_wrap);

	void clear();

	void set_syntax_coloring(bool p_enabled);
	bool is_syntax_coloring_enabled() const;

	void cut();
	void copy();
	void paste();
	void select_all();
	void select(int p_from_line, int p_from_column, int p_to_line, int p_to_column);
	void deselect();

	void set_search_text(const String &p_search_text);
	void set_search_flags(uint32_t p_flags);
	void set_current_search_result(int line, int col);

	void set_highlight_all_occurrences(const bool p_enabled);
	bool is_highlight_all_occurrences_enabled() const;
	bool is_selection_active() const;
	int get_selection_from_line() const;
	int get_selection_from_column() const;
	int get_selection_to_line() const;
	int get_selection_to_column() const;
	String get_selection_text() const;

	String get_word_under_cursor() const;
	String get_word_at_pos(const Vector2 &p_pos) const;

	bool search(const String &p_key, uint32_t p_search_flags, int p_from_line, int p_from_column, int &r_line, int &r_column) const;

	void undo();
	void redo();
	void clear_undo_history();

	void set_indent_using_spaces(const bool p_use_spaces);
	bool is_indent_using_spaces() const;
	void set_indent_size(const int p_size);
	void set_draw_tabs(bool p_draw);
	bool is_drawing_tabs() const;

	void set_insert_mode(bool p_enabled);
	bool is_insert_mode() const;

	void add_keyword_color(const String &p_keyword, const Color &p_color);
	void add_color_region(const String &p_begin_key = String(), const String &p_end_key = String(), const Color &p_color = Color(), bool p_line_only = false);
	void clear_colors();

	int get_v_scroll() const;
	void set_v_scroll(int p_scroll);

	int get_h_scroll() const;
	void set_h_scroll(int p_scroll);

	void set_smooth_scroll_enabled(bool p_enable);
	bool is_smooth_scroll_enabled() const;

	void set_v_scroll_speed(float p_speed);
	float get_v_scroll_speed() const;

	uint32_t get_version() const;
	uint32_t get_saved_version() const;
	void tag_saved_version();

	void menu_option(int p_option);

	void set_show_line_numbers(bool p_show);
	bool is_show_line_numbers_enabled() const;

	void set_line_numbers_zero_padded(bool p_zero_padded);

	void set_show_line_length_guideline(bool p_show);
	void set_line_length_guideline_column(int p_column);

	void set_draw_breakpoint_gutter(bool p_draw);
	bool is_drawing_breakpoint_gutter() const;

	void set_breakpoint_gutter_width(int p_gutter_width);
	int get_breakpoint_gutter_width() const;

	void set_tooltip_request_func(Object *p_obj, const StringName &p_function, const Variant &p_udata);

	void set_completion(bool p_enabled, const Vector<String> &p_prefixes);
	void code_complete(const Vector<String> &p_strings, bool p_forced = false);
	void set_code_hint(const String &p_hint);
	void query_code_comple();

	void set_select_identifiers_on_hover(bool p_enable);
	bool is_selecting_identifiers_on_hover_enabled() const;

	void set_context_menu_enabled(bool p_enable);
	PopupMenu *get_menu() const;

	String get_text_for_completion();
	String get_text_for_lookup_completion();

	virtual bool is_text_field() const;
	TextEdit();
	~TextEdit();
};

VARIANT_ENUM_CAST(TextEdit::MenuItems);
VARIANT_ENUM_CAST(TextEdit::SearchFlags);

#endif // TEXT_EDIT_H
