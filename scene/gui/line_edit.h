/*************************************************************************/
/*  line_edit.h                                                          */
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
#ifndef LINE_EDIT_H
#define LINE_EDIT_H

#include "scene/gui/control.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class LineEdit : public Control {
	
	OBJ_TYPE( LineEdit, Control );
	
	bool editable;
	bool pass;
	
	String undo_text;
	String text;
	
	int cursor_pos;
	int window_pos;
	int max_length; // 0 for no maximum
	
	struct Selection {
		
		int begin;
		int end;
		int cursor_start;
		bool enabled;
		bool creating;
		bool old_shift;
		bool doubleclick;
		bool drag_attempt;
	} selection;
	
	void shift_selection_check_pre(bool);
	void shift_selection_check_post(bool);
	
	void selection_clear();
	void selection_fill_at_cursor();
	void selection_delete();
	void set_window_pos(int p_pos);
	
	void set_cursor_at_pixel_pos(int p_x);
	
	void clear_internal();
	void changed_internal();
	
	void copy_text();
	void cut_text();
	void paste_text();
			

	void _input_event(InputEvent p_event);
	void _notification(int p_what);
	
protected:	
	static void _bind_methods();	
public:
	
		
	virtual Variant get_drag_data(const Point2& p_point);
	virtual bool can_drop_data(const Point2& p_point,const Variant& p_data) const;
	virtual void drop_data(const Point2& p_point,const Variant& p_data);

	
	void select_all();
	
	void delete_char();
	void set_text(String p_text);
	String get_text() const;
	void set_cursor_pos(int p_pos);
	int get_cursor_pos() const;
	void set_max_length(int p_max_length);
	int get_max_length() const;
	void append_at_cursor(String p_text);
	void clear();
	
	
	void set_editable(bool p_editable);
	bool is_editable() const;
	
	void set_secret(bool p_secret);
	bool is_secret() const;

	void select(int p_from=0, int p_to=-1);

	virtual Size2 get_minimum_size() const;

    virtual bool is_text_field() const;
	LineEdit();
	~LineEdit();
	
};

#endif
