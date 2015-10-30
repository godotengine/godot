/*************************************************************************/
/*  text_edit.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#include "scene/gui/scroll_bar.h"
#include "scene/main/timer.h"


class TextEdit : public Control  {

	OBJ_TYPE( TextEdit, Control );

	struct Cursor {
		int last_fit_x;
		int line,column;	    ///< cursor
		int x_ofs,line_ofs;
	} cursor;

	struct Selection {

		enum Mode {

			MODE_NONE,
			MODE_SHIFT,
			MODE_POINTER
		};

		Mode selecting_mode;
		int selecting_line,selecting_column;
		bool selecting_test;


		bool active;

		int from_line,from_column;
		int to_line,to_column;

		bool shiftclick_left;

	} selection;

	struct Cache {

		Ref<Texture> tab_icon;
		Ref<StyleBox> style_normal;
		Ref<StyleBox> style_focus;
		Ref<Font> font;
		Color font_color;
		Color font_selected_color;
		Color keyword_color;
		Color selection_color;
		Color mark_color;
		Color breakpoint_color;
		Color current_line_color;
		Color brace_mismatch_color;

		int row_height;
		int line_spacing;
		int line_number_w;
		Size2 size;
	} cache;

	struct ColorRegion {

		Color color;
		String begin_key;
		String end_key;
		bool line_only;
		bool eq;
		ColorRegion(const String& p_begin_key="",const String& p_end_key="",const Color &p_color=Color(),bool p_line_only=false) { begin_key=p_begin_key; end_key=p_end_key; color=p_color; line_only=p_line_only || p_end_key==""; eq=begin_key==end_key; }
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
		       Map<int,ColorRegionInfo> region_info;
		       String data;
	       };
	private:
	       const Vector<ColorRegion> *color_regions;
	       mutable Vector<Line> text;
	       Ref<Font> font;
	       int tab_size;

	       void _update_line_cache(int p_line) const;

	public:


		void set_tab_size(int p_tab_size);
		void set_font(const Ref<Font>& p_font);
		void set_color_regions(const Vector<ColorRegion>*p_regions) { color_regions=p_regions; }
		int get_line_width(int p_line) const;
		int get_max_width() const;		
		const Map<int,ColorRegionInfo>& get_color_region_info(int p_line);
		void set(int p_line,const String& p_string);
		void set_marked(int p_line,bool p_marked) { text[p_line].marked=p_marked; }
		bool is_marked(int p_line) const { return text[p_line].marked; }
		void set_breakpoint(int p_line,bool p_breakpoint) { text[p_line].breakpoint=p_breakpoint; }
		bool is_breakpoint(int p_line) const { return text[p_line].breakpoint; }
		void insert(int p_at,const String& p_text);
		void remove(int p_at);
		int size() const { return text.size(); }
		void clear();
		void clear_caches();
        _FORCE_INLINE_ const String& operator[](int p_line) const { return text[p_line].data; }
		Text() { tab_size=4; }
       };

	struct TextOperation {

		enum Type {
			TYPE_NONE,
			TYPE_INSERT,
			TYPE_REMOVE
		};

		Type type;
		int from_line,from_column;
		int to_line, to_column;
		String text;
		uint32_t version;
		bool chain_forward;
		bool chain_backward;
	};

	TextOperation current_op;

	List<TextOperation> undo_stack;
    List<TextOperation>::Element *undo_stack_pos;

	void _clear_redo();
	void _do_text_op(const TextOperation& p_op, bool p_reverse);


	//syntax coloring
	Color symbol_color;
	HashMap<String,Color> keywords;
	Color custom_bg_color;

	Vector<ColorRegion> color_regions;

	Set<String> completion_prefixes;
	bool completion_enabled;
	Vector<String> completion_strings;
	Vector<String> completion_options;
	bool completion_active;
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
	int tab_size;

	bool setting_row;
	bool wrap;
	bool draw_tabs;
	bool cursor_changed_dirty;
	bool text_changed_dirty;
	bool undo_enabled;
	bool line_numbers;
	
	bool auto_brace_completion_enabled;
	bool brace_matching_enabled;
	bool cut_copy_line;

	uint64_t last_dblclk;

	Timer *idle_detect;
	HScrollBar *h_scroll;
	VScrollBar *v_scroll;
	bool updating_scrolls;


	Object *tooltip_obj;
	StringName tooltip_func;
	Variant tooltip_ud;
	
	bool next_operation_is_complex;

	int get_visible_rows() const;

	int get_char_count();

	int get_char_pos_for(int p_px,String p_pos) const;
	int get_column_x_offset(int p_column,String p_pos);

	void adjust_viewport_to_cursor();
	void _scroll_moved(double);
	void _update_scrollbars();

	void _pre_shift_selection();
	void _post_shift_selection();

//	void mouse_motion(const Point& p_pos, const Point& p_rel, int p_button_mask);
	Size2 get_minimum_size();

	int get_row_height() const;

	void _update_caches();
	void _cursor_changed_emit();
	void _text_changed_emit();
	
	void _begin_compex_operation();
	void _end_compex_operation();
	void _push_current_op();

	/* super internal api, undo/redo builds on it */
	
	void _base_insert_text(int p_line, int p_column,const String& p_text,int &r_end_line,int &r_end_column);
	String _base_get_text(int p_from_line, int p_from_column,int p_to_line,int p_to_column) const;
	void _base_remove_text(int p_from_line, int p_from_column,int p_to_line,int p_to_column);

	DVector<int> _search_bind(const String &p_key,uint32_t p_search_flags, int p_from_line,int p_from_column) const;

	void _clear();
	void _cancel_completion();
	void _cancel_code_hint();
	void _confirm_completion();
	void _update_completion_candidates();

	bool _get_mouse_pos(const Point2i& p_mouse, int &r_row, int &r_col) const;

protected:

	virtual String get_tooltip(const Point2& p_pos) const;
	
	void _insert_text(int p_line, int p_column,const String& p_text,int *r_end_line=NULL,int *r_end_char=NULL);
	void _remove_text(int p_from_line, int p_from_column,int p_to_line,int p_to_column);
	void _insert_text_at_cursor(const String& p_text);
	void _input_event(const InputEvent& p_input);
	void _notification(int p_what);
	
	void _consume_pair_symbol(CharType ch);
	void _consume_backspace_for_pair_symbol(int prev_line, int prev_column);
	
	static void _bind_methods();



public:

	enum SearchFlags {

		SEARCH_MATCH_CASE=1,
		SEARCH_WHOLE_WORDS=2,
		SEARCH_BACKWARDS=4
	};
	
	virtual CursorShape get_cursor_shape(const Point2& p_pos=Point2i()) const;

	//void delete_char();
	//void delete_line();

	void set_text(String p_text);
	void insert_text_at_cursor(const String& p_text);
    void insert_at(const String& p_text, int at);
	int get_line_count() const;
	void set_line_as_marked(int p_line,bool p_marked);
	void set_line_as_breakpoint(int p_line,bool p_breakpoint);
	bool is_line_set_as_breakpoint(int p_line) const;
	void get_breakpoints(List<int> *p_breakpoints) const;
	String get_text();
	String get_line(int line) const;
    void set_line(int line, String new_text);
	void backspace_at_cursor();
	
	inline void set_auto_brace_completion(bool p_enabled) {
		auto_brace_completion_enabled = p_enabled;
	}
	inline void set_brace_matching(bool p_enabled) {
		brace_matching_enabled=p_enabled;
		update();
	}

	void cursor_set_column(int p_col);
	void cursor_set_line(int p_row);

	int cursor_get_column() const;
	int cursor_get_line() const;

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
	void select(int p_from_line,int p_from_column,int p_to_line,int p_to_column);
	void deselect();

	bool is_selection_active() const;
	int get_selection_from_line() const;
    int get_selection_from_column() const;
	int get_selection_to_line() const;
	int get_selection_to_column() const;
	String get_selection_text() const;

	String get_word_under_cursor() const;

	bool search(const String &p_key,uint32_t p_search_flags, int p_from_line, int p_from_column,int &r_line,int &r_column) const;

	void undo();
	void redo();
	void clear_undo_history();


	void set_draw_tabs(bool p_draw);
	bool is_drawing_tabs() const;

	void add_keyword_color(const String& p_keyword,const Color& p_color);
	void add_color_region(const String& p_begin_key=String(),const String& p_end_key=String(),const Color &p_color=Color(),bool p_line_only=false);
	void set_symbol_color(const Color& p_color);
	void set_custom_bg_color(const Color& p_color);
	void clear_colors();

	int get_v_scroll() const;
	void set_v_scroll(int p_scroll);

	int get_h_scroll() const;
	void set_h_scroll(int p_scroll);

	uint32_t get_version() const;
	uint32_t get_saved_version() const;
	void tag_saved_version();

	void set_show_line_numbers(bool p_show);

	void set_tooltip_request_func(Object *p_obj, const StringName& p_function, const Variant& p_udata);

	void set_completion(bool p_enabled,const Vector<String>& p_prefixes);	
	void code_complete(const Vector<String> &p_strings);
	void set_code_hint(const String& p_hint);
	void query_code_comple();

	String get_text_for_completion();

    virtual bool is_text_field() const;
	TextEdit();
	~TextEdit();
};


#endif // TEXT_EDIT_H
