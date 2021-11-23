/*************************************************************************/
/*  text_edit.h                                                          */
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

#ifndef TEXT_EDIT_H
#define TEXT_EDIT_H

#include "scene/gui/control.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/scroll_bar.h"
#include "scene/main/timer.h"
#include "scene/resources/syntax_highlighter.h"
#include "scene/resources/text_paragraph.h"

class TextEdit : public Control {
	GDCLASS(TextEdit, Control);

public:
	/* Caret. */
	enum CaretType {
		CARET_TYPE_LINE,
		CARET_TYPE_BLOCK
	};

	/* Selection */
	enum SelectionMode {
		SELECTION_MODE_NONE,
		SELECTION_MODE_SHIFT,
		SELECTION_MODE_POINTER,
		SELECTION_MODE_WORD,
		SELECTION_MODE_LINE
	};

	/* Line Wrapping.*/
	enum LineWrappingMode {
		LINE_WRAPPING_NONE,
		LINE_WRAPPING_BOUNDARY
	};

	/* Gutters. */
	enum GutterType {
		GUTTER_TYPE_STRING,
		GUTTER_TYPE_ICON,
		GUTTER_TYPE_CUSTOM
	};

	/* Contex Menu. */
	enum MenuItems {
		MENU_CUT,
		MENU_COPY,
		MENU_PASTE,
		MENU_CLEAR,
		MENU_SELECT_ALL,
		MENU_UNDO,
		MENU_REDO,
		MENU_DIR_INHERITED,
		MENU_DIR_AUTO,
		MENU_DIR_LTR,
		MENU_DIR_RTL,
		MENU_DISPLAY_UCC,
		MENU_INSERT_LRM,
		MENU_INSERT_RLM,
		MENU_INSERT_LRE,
		MENU_INSERT_RLE,
		MENU_INSERT_LRO,
		MENU_INSERT_RLO,
		MENU_INSERT_PDF,
		MENU_INSERT_ALM,
		MENU_INSERT_LRI,
		MENU_INSERT_RLI,
		MENU_INSERT_FSI,
		MENU_INSERT_PDI,
		MENU_INSERT_ZWJ,
		MENU_INSERT_ZWNJ,
		MENU_INSERT_WJ,
		MENU_INSERT_SHY,
		MENU_MAX

	};

	/* Search. */
	enum SearchFlags {
		SEARCH_MATCH_CASE = 1,
		SEARCH_WHOLE_WORDS = 2,
		SEARCH_BACKWARDS = 4
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

			String data;
			Array bidi_override;
			Ref<TextParagraph> data_buf;

			Color background_color = Color(0, 0, 0, 0);
			bool hidden = false;
			int height = 0;
			int width = 0;

			Line() {
				data_buf.instantiate();
			}
		};

	private:
		bool is_dirty = false;
		bool tab_size_dirty = false;

		mutable Vector<Line> text;
		Ref<Font> font;
		int font_size = -1;

		Dictionary opentype_features;
		String language;
		TextServer::Direction direction = TextServer::DIRECTION_AUTO;
		bool draw_control_chars = false;

		int line_height = -1;
		int max_width = -1;
		int width = -1;

		int tab_size = 4;
		int gutter_count = 0;

		void _calculate_line_height();
		void _calculate_max_line_width();

	public:
		void set_tab_size(int p_tab_size);
		int get_tab_size() const;
		void set_font(const Ref<Font> &p_font);
		void set_font_size(int p_font_size);
		void set_font_features(const Dictionary &p_features);
		void set_direction_and_language(TextServer::Direction p_direction, const String &p_language);
		void set_draw_control_chars(bool p_enabled);

		int get_line_height() const;
		int get_line_width(int p_line, int p_wrap_index = -1) const;
		int get_max_width() const;

		void set_width(float p_width);
		int get_line_wrap_amount(int p_line) const;

		Vector<Vector2i> get_line_wrap_ranges(int p_line) const;
		const Ref<TextParagraph> get_line_data(int p_line) const;

		void set(int p_line, const String &p_text, const Array &p_bidi_override);
		void set_hidden(int p_line, bool p_hidden) {
			text.write[p_line].hidden = p_hidden;
			if (!p_hidden && text[p_line].width > max_width) {
				max_width = text[p_line].width;
			} else if (p_hidden && text[p_line].width == max_width) {
				_calculate_max_line_width();
			}
		}
		bool is_hidden(int p_line) const { return text[p_line].hidden; }
		void insert(int p_at, const String &p_text, const Array &p_bidi_override);
		void remove(int p_at);
		int size() const { return text.size(); }
		void clear();

		void invalidate_cache(int p_line, int p_column = -1, const String &p_ime_text = String(), const Array &p_bidi_override = Array());
		void invalidate_all();
		void invalidate_all_lines();

		_FORCE_INLINE_ const String &operator[](int p_line) const;

		/* Gutters. */
		void add_gutter(int p_at);
		void remove_gutter(int p_gutter);
		void move_gutters(int p_from_line, int p_to_line);

		void set_line_gutter_metadata(int p_line, int p_gutter, const Variant &p_metadata) { text.write[p_line].gutters.write[p_gutter].metadata = p_metadata; }
		const Variant &get_line_gutter_metadata(int p_line, int p_gutter) const { return text[p_line].gutters[p_gutter].metadata; }

		void set_line_gutter_text(int p_line, int p_gutter, const String &p_text) { text.write[p_line].gutters.write[p_gutter].text = p_text; }
		const String &get_line_gutter_text(int p_line, int p_gutter) const { return text[p_line].gutters[p_gutter].text; }

		void set_line_gutter_icon(int p_line, int p_gutter, const Ref<Texture2D> &p_icon) { text.write[p_line].gutters.write[p_gutter].icon = p_icon; }
		const Ref<Texture2D> &get_line_gutter_icon(int p_line, int p_gutter) const { return text[p_line].gutters[p_gutter].icon; }

		void set_line_gutter_item_color(int p_line, int p_gutter, const Color &p_color) { text.write[p_line].gutters.write[p_gutter].color = p_color; }
		const Color &get_line_gutter_item_color(int p_line, int p_gutter) const { return text[p_line].gutters[p_gutter].color; }

		void set_line_gutter_clickable(int p_line, int p_gutter, bool p_clickable) { text.write[p_line].gutters.write[p_gutter].clickable = p_clickable; }
		bool is_line_gutter_clickable(int p_line, int p_gutter) const { return text[p_line].gutters[p_gutter].clickable; }

		/* Line style. */
		void set_line_background_color(int p_line, const Color &p_color) { text.write[p_line].background_color = p_color; }
		const Color get_line_background_color(int p_line) const { return text[p_line].background_color; }
	};

	/* Text */
	Text text;

	bool setting_text = false;

	// Text properties.
	String ime_text = "";
	Point2 ime_selection;

	/* Initialise to opposite first, so we get past the early-out in set_editable. */
	bool editable = false;

	TextDirection text_direction = TEXT_DIRECTION_AUTO;
	TextDirection input_direction = TEXT_DIRECTION_LTR;

	Dictionary opentype_features;
	String language = "";

	Control::StructuredTextParser st_parser = STRUCTURED_TEXT_DEFAULT;
	Array st_args;

	void _clear();
	void _update_caches();

	// User control.
	bool overtype_mode = false;
	bool context_menu_enabled = true;
	bool shortcut_keys_enabled = true;
	bool virtual_keyboard_enabled = true;
	bool middle_mouse_paste_enabled = true;

	// Overridable actions.
	String cut_copy_line = "";

	// Context menu.
	PopupMenu *menu = nullptr;
	PopupMenu *menu_dir = nullptr;
	PopupMenu *menu_ctl = nullptr;

	void _generate_context_menu();
	Key _get_menu_action_accelerator(const String &p_action);

	/* Versioning */
	struct TextOperation {
		enum Type {
			TYPE_NONE,
			TYPE_INSERT,
			TYPE_REMOVE
		};

		Type type = TYPE_NONE;
		int from_line = 0;
		int from_column = 0;
		int to_line = 0;
		int to_column = 0;
		String text;
		uint32_t prev_version = 0;
		uint32_t version = 0;
		bool chain_forward = false;
		bool chain_backward = false;
	};

	bool undo_enabled = true;
	int undo_stack_max_size = 50;

	int complex_operation_count = 0;
	bool next_operation_is_complex = false;

	TextOperation current_op;
	List<TextOperation> undo_stack;
	List<TextOperation>::Element *undo_stack_pos = nullptr;

	Timer *idle_detect;

	uint32_t version = 0;
	uint32_t saved_version = 0;

	void _push_current_op();
	void _do_text_op(const TextOperation &p_op, bool p_reverse);
	void _clear_redo();

	/* Search */
	Color search_result_color = Color(1, 1, 1);
	Color search_result_border_color = Color(1, 1, 1);

	String search_text = "";
	uint32_t search_flags = 0;

	int _get_column_pos_of_word(const String &p_key, const String &p_search, uint32_t p_search_flags, int p_from_column) const;

	/* Tooltip. */
	ObjectID tooltip_obj_id;
	StringName tooltip_func;
	Variant tooltip_ud;

	/* Mouse */
	struct LineDrawingCache {
		int y_offset = 0;
		Vector<int> first_visible_chars;
		Vector<int> last_visible_chars;
	};

	Map<int, LineDrawingCache> line_drawing_cache;

	int _get_char_pos_for_line(int p_px, int p_line, int p_wrap_index = 0) const;

	/* Caret. */
	struct Caret {
		Point2 draw_pos;
		bool visible = false;
		int last_fit_x = 0;
		int line = 0;
		int column = 0;
		int x_ofs = 0;
		int line_ofs = 0;
		int wrap_ofs = 0;
	} caret;

	bool setting_caret_line = false;
	bool caret_pos_dirty = false;

	Color caret_color = Color(1, 1, 1);
	Color caret_background_color = Color(0, 0, 0);

	CaretType caret_type = CaretType::CARET_TYPE_LINE;

	bool draw_caret = true;

	bool caret_blink_enabled = false;
	Timer *caret_blink_timer;

	bool move_caret_on_right_click = true;

	bool caret_mid_grapheme_enabled = false;

	void _emit_caret_changed();

	void _reset_caret_blink_timer();
	void _toggle_draw_caret();

	int _get_column_x_offset_for_line(int p_char, int p_line) const;

	/* Selection. */
	struct Selection {
		SelectionMode selecting_mode = SelectionMode::SELECTION_MODE_NONE;
		int selecting_line = 0;
		int selecting_column = 0;
		int selected_word_beg = 0;
		int selected_word_end = 0;
		int selected_word_origin = 0;
		bool selecting_text = false;

		bool active = false;

		int from_line = 0;
		int from_column = 0;
		int to_line = 0;
		int to_column = 0;

		bool shiftclick_left = false;
	} selection;

	bool selecting_enabled = true;
	bool deselect_on_focus_loss_enabled = true;

	Color font_selected_color = Color(1, 1, 1);
	Color selection_color = Color(1, 1, 1);
	bool override_selected_font_color = false;

	bool dragging_selection = false;

	Timer *click_select_held;
	uint64_t last_dblclk = 0;
	Vector2 last_dblclk_pos;
	void _click_selection_held();

	void _update_selection_mode_pointer();
	void _update_selection_mode_word();
	void _update_selection_mode_line();

	void _pre_shift_selection();
	void _post_shift_selection();

	/* Line wrapping. */
	LineWrappingMode line_wrapping_mode = LineWrappingMode::LINE_WRAPPING_NONE;

	int wrap_at_column = 0;
	int wrap_right_offset = 10;

	void _update_wrap_at_column(bool p_force = false);

	void _update_caret_wrap_offset();

	/* Viewport. */
	HScrollBar *h_scroll;
	VScrollBar *v_scroll;

	bool scroll_past_end_of_file_enabled = false;

	// Smooth scrolling.
	bool smooth_scroll_enabled = false;
	float target_v_scroll = 0.0;
	float v_scroll_speed = 80.0;

	// Scrolling.
	bool scrolling = false;
	bool updating_scrolls = false;

	void _update_scrollbars();
	int _get_control_height() const;

	void _v_scroll_input();
	void _scroll_moved(double p_to_val);

	double _get_visible_lines_offset() const;
	double _get_v_scroll_offset() const;

	void _scroll_up(real_t p_delta);
	void _scroll_down(real_t p_delta);

	void _scroll_lines_up();
	void _scroll_lines_down();

	// Minimap.
	bool draw_minimap = false;

	int minimap_width = 80;
	Point2 minimap_char_size = Point2(1, 2);
	int minimap_line_spacing = 1;

	// Minimap scroll.
	bool minimap_clicked = false;
	bool hovering_minimap = false;
	bool dragging_minimap = false;
	bool can_drag_minimap = false;

	double minimap_scroll_ratio = 0.0;
	double minimap_scroll_click_pos = 0.0;

	void _update_minimap_hover();
	void _update_minimap_click();
	void _update_minimap_drag();

	/* Gutters. */
	Vector<GutterInfo> gutters;
	int gutters_width = 0;
	int gutter_padding = 0;
	Vector2i hovered_gutter = Vector2i(-1, -1); // X = gutter index, Y = row.

	void _update_gutter_width();

	/* Syntax highlighting. */
	Ref<SyntaxHighlighter> syntax_highlighter;
	Map<int, Dictionary> syntax_highlighting_cache;

	Dictionary _get_line_syntax_highlighting(int p_line);

	/* Visual. */
	Ref<StyleBox> style_normal;
	Ref<StyleBox> style_focus;
	Ref<StyleBox> style_readonly;

	Ref<Texture2D> tab_icon;
	Ref<Texture2D> space_icon;

	Ref<Font> font;
	int font_size = 16;
	Color font_color = Color(1, 1, 1);
	Color font_readonly_color = Color(1, 1, 1);

	int outline_size = 0;
	Color outline_color = Color(1, 1, 1);

	int line_spacing = 1;

	Color background_color = Color(1, 1, 1);
	Color current_line_color = Color(1, 1, 1);
	Color word_highlighted_color = Color(1, 1, 1);

	bool window_has_focus = true;
	bool first_draw = true;

	bool highlight_current_line = false;
	bool highlight_all_occurrences = false;
	bool draw_control_chars = false;
	bool draw_tabs = false;
	bool draw_spaces = false;

	/*** Super internal Core API. Everything builds on it. ***/
	bool text_changed_dirty = false;
	void _text_changed_emit();

	void _insert_text(int p_line, int p_char, const String &p_text, int *r_end_line = nullptr, int *r_end_char = nullptr);
	void _remove_text(int p_from_line, int p_from_column, int p_to_line, int p_to_column);

	void _base_insert_text(int p_line, int p_char, const String &p_text, int &r_end_line, int &r_end_column);
	String _base_get_text(int p_from_line, int p_from_column, int p_to_line, int p_to_column) const;
	void _base_remove_text(int p_from_line, int p_from_column, int p_to_line, int p_to_column);

	/* Input actions. */
	void _swap_current_input_direction();
	void _new_line(bool p_split_current = true, bool p_above = false);
	void _move_caret_left(bool p_select, bool p_move_by_word = false);
	void _move_caret_right(bool p_select, bool p_move_by_word = false);
	void _move_caret_up(bool p_select);
	void _move_caret_down(bool p_select);
	void _move_caret_to_line_start(bool p_select);
	void _move_caret_to_line_end(bool p_select);
	void _move_caret_page_up(bool p_select);
	void _move_caret_page_down(bool p_select);
	void _do_backspace(bool p_word = false, bool p_all_to_left = false);
	void _delete(bool p_word = false, bool p_all_to_right = false);
	void _move_caret_document_start(bool p_select);
	void _move_caret_document_end(bool p_select);

protected:
	void _notification(int p_what);

	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	/* Internal API for CodeEdit, pending public API. */
	// brace matching
	bool highlight_matching_braces_enabled = false;
	Color brace_mismatch_color;

	// Line hiding.
	Color code_folding_color = Color(1, 1, 1);
	Ref<Texture2D> folded_eol_icon;

	bool hiding_enabled = false;

	void _set_hiding_enabled(bool p_enabled);
	bool _is_hiding_enabled() const;

	void _set_line_as_hidden(int p_line, bool p_hidden);
	bool _is_line_hidden(int p_line) const;

	void _unhide_all_lines();

	// Symbol lookup.
	String lookup_symbol_word;
	void _set_symbol_lookup_word(const String &p_symbol);

	/* Text manipulation */

	// Overridable actions
	virtual void _handle_unicode_input_internal(const uint32_t p_unicode);
	virtual void _backspace_internal();

	virtual void _cut_internal();
	virtual void _copy_internal();
	virtual void _paste_internal();
	virtual void _paste_primary_clipboard_internal();

	GDVIRTUAL1(_handle_unicode_input, int)
	GDVIRTUAL0(_backspace)
	GDVIRTUAL0(_cut)
	GDVIRTUAL0(_copy)
	GDVIRTUAL0(_paste)
	GDVIRTUAL0(_paste_primary_clipboard)

public:
	/* General overrides. */
	virtual void gui_input(const Ref<InputEvent> &p_gui_input) override;
	virtual Size2 get_minimum_size() const override;
	virtual bool is_text_field() const override;
	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;
	virtual String get_tooltip(const Point2 &p_pos) const override;
	void set_tooltip_request_func(Object *p_obj, const StringName &p_function, const Variant &p_udata);

	/* Text */
	// Text properties.
	bool has_ime_text() const;

	void set_editable(const bool p_editable);
	bool is_editable() const;

	void set_text_direction(TextDirection p_text_direction);
	TextDirection get_text_direction() const;

	void set_opentype_feature(const String &p_name, int p_value);
	int get_opentype_feature(const String &p_name) const;
	void clear_opentype_features();

	void set_language(const String &p_language);
	String get_language() const;

	void set_structured_text_bidi_override(Control::StructuredTextParser p_parser);
	Control::StructuredTextParser get_structured_text_bidi_override() const;
	void set_structured_text_bidi_override_options(Array p_args);
	Array get_structured_text_bidi_override_options() const;

	void set_tab_size(const int p_size);
	int get_tab_size() const;

	// User controls
	void set_overtype_mode_enabled(const bool p_enabled);
	bool is_overtype_mode_enabled() const;

	void set_context_menu_enabled(bool p_enabled);
	bool is_context_menu_enabled() const;

	void set_shortcut_keys_enabled(bool p_enabled);
	bool is_shortcut_keys_enabled() const;

	void set_virtual_keyboard_enabled(bool p_enabled);
	bool is_virtual_keyboard_enabled() const;

	void set_middle_mouse_paste_enabled(bool p_enabled);
	bool is_middle_mouse_paste_enabled() const;

	// Text manipulation
	void clear();

	void set_text(const String &p_text);
	String get_text() const;
	int get_line_count() const;

	void set_line(int p_line, const String &p_new_text);
	String get_line(int p_line) const;

	int get_line_width(int p_line, int p_wrap_index = -1) const;
	int get_line_height() const;

	int get_indent_level(int p_line) const;
	int get_first_non_whitespace_column(int p_line) const;

	void swap_lines(int p_from_line, int p_to_line);

	void insert_line_at(int p_at, const String &p_text);
	void insert_text_at_caret(const String &p_text);

	void remove_text(int p_from_line, int p_from_column, int p_to_line, int p_to_column);

	int get_last_unhidden_line() const;
	int get_next_visible_line_offset_from(int p_line_from, int p_visible_amount) const;
	Point2i get_next_visible_line_index_offset_from(int p_line_from, int p_wrap_index_from, int p_visible_amount) const;

	// Overridable actions
	void handle_unicode_input(const uint32_t p_unicode);
	void backspace();

	void cut();
	void copy();
	void paste();
	void paste_primary_clipboard();

	// Context menu.
	PopupMenu *get_menu() const;
	bool is_menu_visible() const;
	void menu_option(int p_option);

	/* Versioning */
	void begin_complex_operation();
	void end_complex_operation();

	bool has_undo() const;
	bool has_redo() const;
	void undo();
	void redo();
	void clear_undo_history();

	bool is_insert_text_operation() const;

	void tag_saved_version();

	uint32_t get_version() const;
	uint32_t get_saved_version() const;

	/* Search */
	void set_search_text(const String &p_search_text);
	void set_search_flags(uint32_t p_flags);

	Point2i search(const String &p_key, uint32_t p_search_flags, int p_from_line, int p_from_column) const;

	/* Mouse */
	Point2 get_local_mouse_pos() const;

	String get_word_at_pos(const Vector2 &p_pos) const;

	Point2i get_line_column_at_pos(const Point2i &p_pos, bool p_allow_out_of_bounds = true) const;
	Point2i get_pos_at_line_column(int p_line, int p_column) const;
	Rect2i get_rect_at_line_column(int p_line, int p_column) const;

	int get_minimap_line_at_pos(const Point2i &p_pos) const;

	bool is_dragging_cursor() const;

	/* Caret */
	void set_caret_type(CaretType p_type);
	CaretType get_caret_type() const;

	void set_caret_blink_enabled(const bool p_enabled);
	bool is_caret_blink_enabled() const;

	void set_caret_blink_speed(const float p_speed);
	float get_caret_blink_speed() const;

	void set_move_caret_on_right_click_enabled(const bool p_enabled);
	bool is_move_caret_on_right_click_enabled() const;

	void set_caret_mid_grapheme_enabled(const bool p_enabled);
	bool is_caret_mid_grapheme_enabled() const;

	bool is_caret_visible() const;
	Point2 get_caret_draw_pos() const;

	void set_caret_line(int p_line, bool p_adjust_viewport = true, bool p_can_be_hidden = true, int p_wrap_index = 0);
	int get_caret_line() const;

	void set_caret_column(int p_col, bool p_adjust_viewport = true);
	int get_caret_column() const;

	int get_caret_wrap_index() const;

	String get_word_under_caret() const;

	/* Selection. */
	void set_selecting_enabled(const bool p_enabled);
	bool is_selecting_enabled() const;

	void set_deselect_on_focus_loss_enabled(const bool p_enabled);
	bool is_deselect_on_focus_loss_enabled() const;

	void set_override_selected_font_color(bool p_override_selected_font_color);
	bool is_overriding_selected_font_color() const;

	void set_selection_mode(SelectionMode p_mode, int p_line = -1, int p_column = -1);
	SelectionMode get_selection_mode() const;

	void select_all();
	void select_word_under_caret();
	void select(int p_from_line, int p_from_column, int p_to_line, int p_to_column);

	bool has_selection() const;

	String get_selected_text() const;

	int get_selection_line() const;
	int get_selection_column() const;

	int get_selection_from_line() const;
	int get_selection_from_column() const;
	int get_selection_to_line() const;
	int get_selection_to_column() const;

	void deselect();
	void delete_selection();

	/* Line wrapping. */
	void set_line_wrapping_mode(LineWrappingMode p_wrapping_mode);
	LineWrappingMode get_line_wrapping_mode() const;

	bool is_line_wrapped(int p_line) const;
	int get_line_wrap_count(int p_line) const;
	int get_line_wrap_index_at_column(int p_line, int p_column) const;

	Vector<String> get_line_wrapped_text(int p_line) const;

	/* Viewport. */
	// Scrolling.
	void set_smooth_scroll_enabled(const bool p_enabled);
	bool is_smooth_scroll_enabled() const;

	void set_scroll_past_end_of_file_enabled(const bool p_enabled);
	bool is_scroll_past_end_of_file_enabled() const;

	void set_v_scroll(double p_scroll);
	double get_v_scroll() const;

	void set_h_scroll(int p_scroll);
	int get_h_scroll() const;

	void set_v_scroll_speed(float p_speed);
	float get_v_scroll_speed() const;

	double get_scroll_pos_for_line(int p_line, int p_wrap_index = 0) const;

	// Visible lines.
	void set_line_as_first_visible(int p_line, int p_wrap_index = 0);
	int get_first_visible_line() const;

	void set_line_as_center_visible(int p_line, int p_wrap_index = 0);

	void set_line_as_last_visible(int p_line, int p_wrap_index = 0);
	int get_last_full_visible_line() const;
	int get_last_full_visible_line_wrap_index() const;

	int get_visible_line_count() const;
	int get_visible_line_count_in_range(int p_from, int p_to) const;
	int get_total_visible_line_count() const;

	// Auto Adjust
	void adjust_viewport_to_caret();
	void center_viewport_to_caret();

	// Minimap
	void set_draw_minimap(bool p_enabled);
	bool is_drawing_minimap() const;

	void set_minimap_width(int p_minimap_width);
	int get_minimap_width() const;

	int get_minimap_visible_lines() const;

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
	int get_total_gutter_width() const;

	void set_gutter_draw(int p_gutter, bool p_draw);
	bool is_gutter_drawn(int p_gutter) const;

	void set_gutter_clickable(int p_gutter, bool p_clickable);
	bool is_gutter_clickable(int p_gutter) const;

	void set_gutter_overwritable(int p_gutter, bool p_overwritable);
	bool is_gutter_overwritable(int p_gutter) const;

	void merge_gutters(int p_from_line, int p_to_line);

	void set_gutter_custom_draw(int p_gutter, Object *p_object, const StringName &p_callback);

	// Line gutters.
	void set_line_gutter_metadata(int p_line, int p_gutter, const Variant &p_metadata);
	Variant get_line_gutter_metadata(int p_line, int p_gutter) const;

	void set_line_gutter_text(int p_line, int p_gutter, const String &p_text);
	String get_line_gutter_text(int p_line, int p_gutter) const;

	void set_line_gutter_icon(int p_line, int p_gutter, const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_line_gutter_icon(int p_line, int p_gutter) const;

	void set_line_gutter_item_color(int p_line, int p_gutter, const Color &p_color);
	Color get_line_gutter_item_color(int p_line, int p_gutter) const;

	void set_line_gutter_clickable(int p_line, int p_gutter, bool p_clickable);
	bool is_line_gutter_clickable(int p_line, int p_gutter) const;

	// Line style
	void set_line_background_color(int p_line, const Color &p_color);
	Color get_line_background_color(int p_line) const;

	/* Syntax Highlighting. */
	void set_syntax_highlighter(Ref<SyntaxHighlighter> p_syntax_highlighter);
	Ref<SyntaxHighlighter> get_syntax_highlighter() const;

	/* Visual. */
	void set_highlight_current_line(bool p_enabled);
	bool is_highlight_current_line_enabled() const;

	void set_highlight_all_occurrences(const bool p_enabled);
	bool is_highlight_all_occurrences_enabled() const;

	void set_draw_control_chars(bool p_enabled);
	bool get_draw_control_chars() const;

	void set_draw_tabs(bool p_enabled);
	bool is_drawing_tabs() const;

	void set_draw_spaces(bool p_enabled);
	bool is_drawing_spaces() const;

	TextEdit();
};

VARIANT_ENUM_CAST(TextEdit::CaretType);
VARIANT_ENUM_CAST(TextEdit::LineWrappingMode);
VARIANT_ENUM_CAST(TextEdit::SelectionMode);
VARIANT_ENUM_CAST(TextEdit::GutterType);
VARIANT_ENUM_CAST(TextEdit::MenuItems);
VARIANT_ENUM_CAST(TextEdit::SearchFlags);

#endif // TEXT_EDIT_H
