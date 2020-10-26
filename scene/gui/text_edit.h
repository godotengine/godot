/*************************************************************************/
/*  text_edit.h                                                          */
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

#ifndef TEXT_EDIT_H
#define TEXT_EDIT_H

#include "scene/gui/control.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/scroll_bar.h"
#include "scene/main/timer.h"
#include "scene/resources/syntax_highlighter.h"

class TextEdit : public Control {
	GDCLASS(TextEdit, Control);

public:
	enum GutterType {
		GUTTER_TYPE_STRING,
		GUTTER_TPYE_ICON,
		GUTTER_TPYE_CUSTOM
	};

	enum SelectionMode {
		SELECTION_MODE_NONE,
		SELECTION_MODE_SHIFT,
		SELECTION_MODE_POINTER,
		SELECTION_MODE_WORD,
		SELECTION_MODE_LINE
	};

private:
	struct GutterInfo {
		GutterType type = GutterType::GUTTER_TYPE_STRING;
		String name = "";
		int width = 24;
		bool draw = true;
		bool clickable = false;
		bool overwritable = false;

		ObjectID custom_draw_obj = ObjectID();
		StringName custom_draw_callback;
	};
	Vector<GutterInfo> gutters;
	int gutters_width = 0;
	int gutter_padding = 0;

	void _update_gutter_width();

	class Text {
	public:
		struct Gutter {
			Variant metadata;
			bool clickable = false;

			Ref<Texture2D> icon = Ref<Texture2D>();
			String text = "";
			Color color = Color(1, 1, 1);
		};

		struct Line {
			Vector<Gutter> gutters;

			int32_t width_cache;
			bool marked;
			bool hidden;
			int32_t wrap_amount_cache;
			String data;
			Line() {
				width_cache = 0;
				marked = false;
				hidden = false;
				wrap_amount_cache = 0;
			}
		};

	private:
		mutable Vector<Line> text;
		Ref<Font> font;
		int indent_size = 4;
		int gutter_count = 0;

		void _update_line_cache(int p_line) const;

	public:
		void set_indent_size(int p_indent_size);
		void set_font(const Ref<Font> &p_font);
		int get_line_width(int p_line) const;
		int get_max_width(bool p_exclude_hidden = false) const;
		int get_char_width(char32_t c, char32_t next_c, int px) const;
		void set_line_wrap_amount(int p_line, int p_wrap_amount) const;
		int get_line_wrap_amount(int p_line) const;
		void set(int p_line, const String &p_text);
		void set_marked(int p_line, bool p_marked) { text.write[p_line].marked = p_marked; }
		bool is_marked(int p_line) const { return text[p_line].marked; }
		void set_hidden(int p_line, bool p_hidden) { text.write[p_line].hidden = p_hidden; }
		bool is_hidden(int p_line) const { return text[p_line].hidden; }
		void insert(int p_at, const String &p_text);
		void remove(int p_at);
		int size() const { return text.size(); }
		void clear();
		void clear_width_cache();
		void clear_wrap_cache();
		_FORCE_INLINE_ const String &operator[](int p_line) const { return text[p_line].data; }

		/* Gutters. */
		void add_gutter(int p_at);
		void remove_gutter(int p_gutter);
		void move_gutters(int p_from_line, int p_to_line);

		void set_line_gutter_metadata(int p_line, int p_gutter, const Variant &p_metadata) { text.write[p_line].gutters.write[p_gutter].metadata = p_metadata; }
		const Variant &get_line_gutter_metadata(int p_line, int p_gutter) const { return text[p_line].gutters[p_gutter].metadata; }

		void set_line_gutter_text(int p_line, int p_gutter, const String &p_text) { text.write[p_line].gutters.write[p_gutter].text = p_text; }
		const String &get_line_gutter_text(int p_line, int p_gutter) const { return text[p_line].gutters[p_gutter].text; }

		void set_line_gutter_icon(int p_line, int p_gutter, Ref<Texture2D> p_icon) { text.write[p_line].gutters.write[p_gutter].icon = p_icon; }
		const Ref<Texture2D> &get_line_gutter_icon(int p_line, int p_gutter) const { return text[p_line].gutters[p_gutter].icon; }

		void set_line_gutter_item_color(int p_line, int p_gutter, const Color &p_color) { text.write[p_line].gutters.write[p_gutter].color = p_color; }
		const Color &get_line_gutter_item_color(int p_line, int p_gutter) const { return text[p_line].gutters[p_gutter].color; }

		void set_line_gutter_clickable(int p_line, int p_gutter, bool p_clickable) { text.write[p_line].gutters.write[p_gutter].clickable = p_clickable; }
		bool is_line_gutter_clickable(int p_line, int p_gutter) const { return text[p_line].gutters[p_gutter].clickable; }
	};

	struct Cursor {
		int last_fit_x;
		int line, column; ///< cursor
		int x_ofs, line_ofs, wrap_ofs;
		Cursor() {
			last_fit_x = 0;
			line = 0;
			column = 0; ///< cursor
			x_ofs = 0;
			line_ofs = 0;
			wrap_ofs = 0;
		}
	} cursor;

	struct Selection {
		SelectionMode selecting_mode;
		int selecting_line, selecting_column;
		int selected_word_beg, selected_word_end, selected_word_origin;
		bool selecting_text;

		bool active;

		int from_line, from_column;
		int to_line, to_column;

		bool shiftclick_left;
		Selection() {
			selecting_mode = SelectionMode::SELECTION_MODE_NONE;
			selecting_line = 0;
			selecting_column = 0;
			selected_word_beg = 0;
			selected_word_end = 0;
			selected_word_origin = 0;
			selecting_text = false;
			active = false;
			from_line = 0;
			from_column = 0;
			to_line = 0;
			to_column = 0;
			shiftclick_left = false;
		}
	} selection;

	Map<int, Dictionary> syntax_highlighting_cache;

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
		TextOperation() {
			type = TYPE_NONE;
			from_line = 0;
			from_column = 0;
			to_line = 0;
			to_column = 0;
			prev_version = 0;
			version = 0;
			chain_forward = false;
			chain_backward = false;
		}
	};

	String ime_text;
	Point2 ime_selection;

	TextOperation current_op;

	List<TextOperation> undo_stack;
	List<TextOperation>::Element *undo_stack_pos;
	int undo_stack_max_size;

	void _clear_redo();
	void _do_text_op(const TextOperation &p_op, bool p_reverse);

	//syntax coloring
	Ref<SyntaxHighlighter> syntax_highlighter;
	Set<String> keywords;

	Dictionary _get_line_syntax_highlighting(int p_line);

	Set<String> completion_prefixes;
	bool completion_enabled;
	List<ScriptCodeCompletionOption> completion_sources;
	Vector<ScriptCodeCompletionOption> completion_options;
	bool completion_active;
	bool completion_forced;
	ScriptCodeCompletionOption completion_current;
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
	bool indent_using_spaces;
	int indent_size;
	String space_indent;

	Timer *caret_blink_timer;
	bool caret_blink_enabled;
	bool draw_caret;
	bool window_has_focus;
	bool block_caret;
	bool right_click_moves_caret;

	bool wrap_enabled;
	int wrap_at;
	int wrap_right_offset;

	bool first_draw;
	bool setting_row;
	bool draw_tabs;
	bool draw_spaces;
	bool override_selected_font_color;
	bool cursor_changed_dirty;
	bool text_changed_dirty;
	bool undo_enabled;
	bool line_length_guidelines;
	int line_length_guideline_soft_col;
	int line_length_guideline_hard_col;
	bool hiding_enabled;
	bool draw_minimap;
	int minimap_width;
	Point2 minimap_char_size;
	int minimap_line_spacing;

	bool highlight_all_occurrences;
	bool scroll_past_end_of_file_enabled;
	bool auto_brace_completion_enabled;
	bool brace_matching_enabled;
	bool highlight_current_line;
	bool auto_indent;
	String cut_copy_line;
	bool insert_mode;
	bool select_identifiers_enabled;

	bool smooth_scroll_enabled;
	bool scrolling;
	bool dragging_selection;
	bool dragging_minimap;
	bool can_drag_minimap;
	bool minimap_clicked;
	double minimap_scroll_ratio;
	double minimap_scroll_click_pos;
	float target_v_scroll;
	float v_scroll_speed;

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

	bool selecting_enabled;

	bool context_menu_enabled;
	bool shortcut_keys_enabled;
	bool virtual_keyboard_enabled = true;

	void _generate_context_menu();

	int get_visible_rows() const;
	int get_total_visible_rows() const;

	int _get_minimap_visible_rows() const;

	void update_cursor_wrap_offset();
	void _update_wrap_at();
	bool line_wraps(int line) const;
	int times_line_wraps(int line) const;
	Vector<String> get_wrap_rows_text(int p_line) const;
	int get_cursor_wrap_index() const;
	int get_line_wrap_index_at_col(int p_line, int p_column) const;
	int get_char_count();

	double get_scroll_pos_for_line(int p_line, int p_wrap_index = 0) const;
	void set_line_as_first_visible(int p_line, int p_wrap_index = 0);
	void set_line_as_center_visible(int p_line, int p_wrap_index = 0);
	void set_line_as_last_visible(int p_line, int p_wrap_index = 0);
	int get_first_visible_line() const;
	int get_last_visible_line() const;
	int get_last_visible_line_wrap_index() const;
	double get_visible_rows_offset() const;
	double get_v_scroll_offset() const;

	int get_char_pos_for_line(int p_px, int p_line, int p_wrap_index = 0) const;
	int get_column_x_offset_for_line(int p_char, int p_line) const;
	int get_char_pos_for(int p_px, String p_str) const;
	int get_column_x_offset(int p_char, String p_str) const;

	void adjust_viewport_to_cursor();
	double get_scroll_line_diff() const;
	void _scroll_moved(double);
	void _update_scrollbars();
	void _v_scroll_input();
	void _click_selection_held();

	void _update_selection_mode_pointer();
	void _update_selection_mode_word();
	void _update_selection_mode_line();

	void _update_minimap_click();
	void _update_minimap_drag();
	void _scroll_up(real_t p_delta);
	void _scroll_down(real_t p_delta);

	void _pre_shift_selection();
	void _post_shift_selection();

	void _scroll_lines_up();
	void _scroll_lines_down();

	//void mouse_motion(const Point& p_pos, const Point& p_rel, int p_button_mask);
	Size2 get_minimum_size() const override;
	int _get_control_height() const;

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

	Dictionary _search_bind(const String &p_key, uint32_t p_search_flags, int p_from_line, int p_from_column) const;

	PopupMenu *menu;

	void _clear();
	void _cancel_completion();
	void _cancel_code_hint();
	void _confirm_completion();
	void _update_completion_candidates();

	int _calculate_spaces_till_next_left_indent(int column);
	int _calculate_spaces_till_next_right_indent(int column);

protected:
	struct Cache {
		Ref<Texture2D> tab_icon;
		Ref<Texture2D> space_icon;
		Ref<Texture2D> folded_eol_icon;
		Ref<StyleBox> style_normal;
		Ref<StyleBox> style_focus;
		Ref<StyleBox> style_readonly;
		Ref<Font> font;
		Color completion_background_color;
		Color completion_selected_color;
		Color completion_existing_color;
		Color completion_font_color;
		Color caret_color;
		Color caret_background_color;
		Color font_color;
		Color font_color_selected;
		Color font_color_readonly;
		Color selection_color;
		Color mark_color;
		Color code_folding_color;
		Color current_line_color;
		Color line_length_guideline_color;
		Color brace_mismatch_color;
		Color word_highlighted_color;
		Color search_result_color;
		Color search_result_border_color;
		Color background_color;

		int row_height;
		int line_spacing;
		int minimap_width;
		Cache() {
			row_height = 0;
			line_spacing = 0;
			minimap_width = 0;
		}
	} cache;

	virtual String get_tooltip(const Point2 &p_pos) const override;

	void _insert_text(int p_line, int p_char, const String &p_text, int *r_end_line = nullptr, int *r_end_char = nullptr);
	void _remove_text(int p_from_line, int p_from_column, int p_to_line, int p_to_column);
	void _insert_text_at_cursor(const String &p_text);
	void _gui_input(const Ref<InputEvent> &p_gui_input);
	void _notification(int p_what);

	void _consume_pair_symbol(char32_t ch);
	void _consume_backspace_for_pair_symbol(int prev_line, int prev_column);

	static void _bind_methods();

public:
	/* Syntax Highlighting. */
	Ref<SyntaxHighlighter> get_syntax_highlighter();
	void set_syntax_highlighter(Ref<SyntaxHighlighter> p_syntax_highlighter);

	/* Gutters. */
	void add_gutter(int p_at = -1);
	void remove_gutter(int p_gutter);
	int get_gutter_count() const;

	void set_gutter_name(int p_gutter, const String &p_name);
	String get_gutter_name(int p_gutter) const;

	void set_gutter_type(int p_gutter, GutterType p_type);
	GutterType get_gutter_type(int p_gutter) const;

	void set_gutter_width(int p_gutter, int p_width);
	int get_gutter_width(int p_gutter) const;

	void set_gutter_draw(int p_gutter, bool p_draw);
	bool is_gutter_drawn(int p_gutter) const;

	void set_gutter_clickable(int p_gutter, bool p_clickable);
	bool is_gutter_clickable(int p_gutter) const;

	void set_gutter_overwritable(int p_gutter, bool p_overwritable);
	bool is_gutter_overwritable(int p_gutter) const;

	void set_gutter_custom_draw(int p_gutter, Object *p_object, const StringName &p_callback);

	// Line gutters.
	void set_line_gutter_metadata(int p_line, int p_gutter, const Variant &p_metadata);
	Variant get_line_gutter_metadata(int p_line, int p_gutter) const;

	void set_line_gutter_text(int p_line, int p_gutter, const String &p_text);
	String get_line_gutter_text(int p_line, int p_gutter) const;

	void set_line_gutter_icon(int p_line, int p_gutter, Ref<Texture2D> p_icon);
	Ref<Texture2D> get_line_gutter_icon(int p_line, int p_gutter) const;

	void set_line_gutter_item_color(int p_line, int p_gutter, const Color &p_color);
	Color get_line_gutter_item_color(int p_line, int p_gutter);

	void set_line_gutter_clickable(int p_line, int p_gutter, bool p_clickable);
	bool is_line_gutter_clickable(int p_line, int p_gutter) const;

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

	enum SearchFlags {
		SEARCH_MATCH_CASE = 1,
		SEARCH_WHOLE_WORDS = 2,
		SEARCH_BACKWARDS = 4
	};

	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;

	void _get_mouse_pos(const Point2i &p_mouse, int &r_row, int &r_col) const;
	void _get_minimap_mouse_row(const Point2i &p_mouse, int &r_row) const;

	//void delete_char();
	//void delete_line();

	void begin_complex_operation();
	void end_complex_operation();

	bool is_insert_text_operation();

	void set_highlighted_word(const String &new_word);
	void set_text(String p_text);
	void insert_text_at_cursor(const String &p_text);
	void insert_at(const String &p_text, int at);
	int get_line_count() const;
	void set_line_as_marked(int p_line, bool p_marked);

	void set_line_as_hidden(int p_line, bool p_hidden);
	bool is_line_hidden(int p_line) const;
	void fold_all_lines();
	void unhide_all_lines();
	int num_lines_from(int p_line_from, int visible_amount) const;
	int num_lines_from_rows(int p_line_from, int p_wrap_index_from, int visible_amount, int &wrap_index) const;
	int get_last_unhidden_line() const;

	bool can_fold(int p_line) const;
	bool is_folded(int p_line) const;
	Vector<int> get_folded_lines() const;
	void fold_line(int p_line);
	void unfold_line(int p_line);
	void toggle_fold_line(int p_line);

	String get_text();
	String get_line(int line) const;
	void set_line(int line, String new_text);
	int get_row_height() const;
	void backspace_at_cursor();

	void indent_left();
	void indent_right();
	int get_indent_level(int p_line) const;
	bool is_line_comment(int p_line) const;

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
	void cursor_set_line(int p_row, bool p_adjust_viewport = true, bool p_can_be_hidden = true, int p_wrap_index = 0);

	int cursor_get_column() const;
	int cursor_get_line() const;
	Vector2i _get_cursor_pixel_pos();

	bool cursor_get_blink_enabled() const;
	void cursor_set_blink_enabled(const bool p_enabled);

	float cursor_get_blink_speed() const;
	void cursor_set_blink_speed(const float p_speed);

	void cursor_set_block_mode(const bool p_enable);
	bool cursor_is_block_mode() const;

	void set_right_click_moves_caret(bool p_enable);
	bool is_right_click_moving_caret() const;

	SelectionMode get_selection_mode() const;
	void set_selection_mode(SelectionMode p_mode, int p_line = -1, int p_column = -1);
	int get_selection_line() const;
	int get_selection_column() const;

	void set_readonly(bool p_readonly);
	bool is_readonly() const;

	void set_max_chars(int p_max_chars);
	int get_max_chars() const;

	void set_wrap_enabled(bool p_wrap_enabled);
	bool is_wrap_enabled() const;

	void clear();

	void cut();
	void copy();
	void paste();
	void select_all();
	void select(int p_from_line, int p_from_column, int p_to_line, int p_to_column);
	void deselect();
	void swap_lines(int line1, int line2);

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
	int get_indent_size();
	void set_draw_tabs(bool p_draw);
	bool is_drawing_tabs() const;
	void set_draw_spaces(bool p_draw);
	bool is_drawing_spaces() const;
	void set_override_selected_font_color(bool p_override_selected_font_color);
	bool is_overriding_selected_font_color() const;

	void set_insert_mode(bool p_enabled);
	bool is_insert_mode() const;

	void add_keyword(const String &p_keyword);
	void clear_keywords();

	double get_v_scroll() const;
	void set_v_scroll(double p_scroll);

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

	void set_highlight_current_line(bool p_enabled);
	bool is_highlight_current_line_enabled() const;

	void set_show_line_length_guidelines(bool p_show);
	void set_line_length_guideline_soft_column(int p_column);
	void set_line_length_guideline_hard_column(int p_column);

	void set_draw_minimap(bool p_draw);
	bool is_drawing_minimap() const;

	void set_minimap_width(int p_minimap_width);
	int get_minimap_width() const;

	void set_hiding_enabled(bool p_enabled);
	bool is_hiding_enabled() const;

	void set_tooltip_request_func(Object *p_obj, const StringName &p_function, const Variant &p_udata);

	void set_completion(bool p_enabled, const Vector<String> &p_prefixes);
	void code_complete(const List<ScriptCodeCompletionOption> &p_strings, bool p_forced = false);
	void set_code_hint(const String &p_hint);
	void query_code_comple();

	void set_select_identifiers_on_hover(bool p_enable);
	bool is_selecting_identifiers_on_hover_enabled() const;

	void set_context_menu_enabled(bool p_enable);
	bool is_context_menu_enabled();

	void set_selecting_enabled(bool p_enabled);
	bool is_selecting_enabled() const;

	void set_shortcut_keys_enabled(bool p_enabled);
	bool is_shortcut_keys_enabled() const;

	void set_virtual_keyboard_enabled(bool p_enable);
	bool is_virtual_keyboard_enabled() const;

	PopupMenu *get_menu() const;

	String get_text_for_completion();
	String get_text_for_lookup_completion();

	virtual bool is_text_field() const override;
	TextEdit();
	~TextEdit();
};

VARIANT_ENUM_CAST(TextEdit::GutterType);
VARIANT_ENUM_CAST(TextEdit::SelectionMode);
VARIANT_ENUM_CAST(TextEdit::MenuItems);
VARIANT_ENUM_CAST(TextEdit::SearchFlags);

#endif // TEXT_EDIT_H
