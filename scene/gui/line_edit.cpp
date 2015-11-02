/*************************************************************************/
/*  line_edit.cpp                                                        */
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
#include "line_edit.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "print_string.h"
#include "label.h"

void LineEdit::_input_event(InputEvent p_event) {


	switch(p_event.type) {
		
		case InputEvent::MOUSE_BUTTON: {
			
			const InputEventMouseButton &b = p_event.mouse_button;

			if (b.button_index!=1)
				break;
			
			if (b.pressed) {
				
				set_cursor_at_pixel_pos(b.x);
				
				if (b.doubleclick) {
					
					selection.enabled=true;
					selection.begin=0;
					selection.end=text.length();
					selection.doubleclick=true;
				}

				selection.drag_attempt=false;
				
				if ((cursor_pos<selection.begin) || (cursor_pos>selection.end) || !selection.enabled)  {
					
					selection_clear();
					selection.cursor_start=cursor_pos;
					selection.creating=true;
				} else if (selection.enabled) {

					selection.drag_attempt=true;
				}
				
				//			if (!editable)
				//	non_editable_clicked_signal.call();
				update();
				
			} else {
				
				if ( (!selection.creating) && (!selection.doubleclick)) {
					selection_clear();
				}
				selection.creating=false;
				selection.doubleclick=false;

                // notify to show soft keyboard
                notification(NOTIFICATION_FOCUS_ENTER);
			}
			
			update();				
		} break;
		case InputEvent::MOUSE_MOTION: {
	
			const InputEventMouseMotion& m=p_event.mouse_motion;
			
			if (m.button_mask&1) {
				
				if (selection.creating) {
					set_cursor_at_pixel_pos(m.x);
					selection_fill_at_cursor();
				}
			}
			
		} break;
		case InputEvent::KEY: {
			
			const InputEventKey &k =p_event.key;
			
			if (!k.pressed)
				return;
			unsigned int code  = k.scancode; 


			if (k.mod.command) {

				bool handled=true;

				switch (code) {

					case (KEY_X): { // CUT

						if(k.mod.command && editable) {
							cut_text();
						}

					} break;

					case (KEY_C): { // COPY

						if(k.mod.command) {
							copy_text();
						}

					} break;

					case (KEY_V): { // PASTE

						if(k.mod.command && editable) {

							paste_text();
						}

					} break;

					case (KEY_Z): { // Simple One level undo

						if( k.mod.command && editable) {

							int old_cursor_pos = cursor_pos;
							text = undo_text;
							if(old_cursor_pos > text.length()) {
								set_cursor_pos(text.length());
							} else {
								set_cursor_pos(old_cursor_pos);
							}
						}

						emit_signal("text_changed",text);
						_change_notify("text");

					} break;

					case (KEY_U): { // Delete from start to cursor

						if( k.mod.command && editable) {

							selection_clear();
							undo_text = text;
							text = text.substr(cursor_pos,text.length()-cursor_pos);
							set_cursor_pos(0);
							emit_signal("text_changed",text);
							_change_notify("text");
						}


					} break;

					case (KEY_Y): { // PASTE (Yank for unix users)

						if(k.mod.command && editable) {

							paste_text();
						}

					} break;
					case (KEY_K): { // Delete from cursor_pos to end

						if(k.mod.command && editable) {

							selection_clear();
							undo_text = text;
							text = text.substr(0,cursor_pos);
							emit_signal("text_changed",text);
							_change_notify("text");
						}

					} break;
					default: { handled=false;}
				}

				if (handled) {
					accept_event();
					return;
				}
			}


			if (!k.mod.alt && !k.mod.meta && !k.mod.command) {

				bool handled=true;
				switch (code) {

					case KEY_ENTER:
					case KEY_RETURN: {

						emit_signal( "text_entered",text );
                        // notify to hide soft keyboard
						   notification(NOTIFICATION_FOCUS_EXIT);
						return;
					} break;

					case KEY_BACKSPACE: {

						if (editable) {
							undo_text = text;
							if (selection.enabled)
								selection_delete();
							else
								delete_char();
						}
					} break;
					case KEY_LEFT: {
						shift_selection_check_pre(k.mod.shift);
						set_cursor_pos(get_cursor_pos()-1);
						shift_selection_check_post(k.mod.shift);

					} break;
					case KEY_RIGHT: {

						shift_selection_check_pre(k.mod.shift);
						set_cursor_pos(get_cursor_pos()+1);
						shift_selection_check_post(k.mod.shift);
					} break;
					case KEY_DELETE: {

						if (editable) {
							undo_text = text;
							if (selection.enabled)
								selection_delete();
							else if (cursor_pos<text.length()) {

								set_cursor_pos(get_cursor_pos()+1);
								delete_char();
							}
						}

					} break;
					case KEY_HOME: {

						shift_selection_check_pre(k.mod.shift);
						set_cursor_pos(0);
						shift_selection_check_post(k.mod.shift);
					} break;
					case KEY_END: {

						shift_selection_check_pre(k.mod.shift);
						set_cursor_pos(text.length());
						shift_selection_check_post(k.mod.shift);
					} break;


					default: {

						if (k.unicode>=32 && k.scancode!=KEY_DELETE) {

							if (editable) {
								selection_delete();
								CharType ucodestr[2]={(CharType)k.unicode,0};
								append_at_cursor(ucodestr);
								emit_signal("text_changed",text);
								_change_notify("text");
							}

						} else {
							handled=false;
						}
					} break;
				}

				if (handled)
					accept_event();
				else
					return;


				selection.old_shift=k.mod.shift;
				update();

			}

			
			return;
			
		} break;
		
	}
}

Variant LineEdit::get_drag_data(const Point2& p_point) {

	if (selection.drag_attempt && selection.enabled) {
		String t = text.substr(selection.begin, selection.end - selection.begin);
		Label *l = memnew( Label );
		l->set_text(t);
		set_drag_preview(l);
		return 	t;
	}

	return Variant();

}
bool LineEdit::can_drop_data(const Point2& p_point,const Variant& p_data) const{

	return p_data.get_type()==Variant::STRING;
}
void LineEdit::drop_data(const Point2& p_point,const Variant& p_data){

	if (p_data.get_type()==Variant::STRING) {
		set_cursor_at_pixel_pos(p_point.x);
		int selected = selection.end - selection.begin;
		text.erase(selection.begin, selected);
		append_at_cursor(p_data);
		selection.begin = cursor_pos-selected;
		selection.end = cursor_pos;
	}
}


void LineEdit::_notification(int p_what) {
	
	switch(p_what) {
		
		case NOTIFICATION_RESIZED: {
			
			set_cursor_pos( get_cursor_pos() );
			
		} break;
		case NOTIFICATION_DRAW: {
		
			int width,height;
			
			Size2 size=get_size();
			width=size.width;
			height=size.height;
			
			RID ci = get_canvas_item();
			
			Ref<StyleBox> style = get_stylebox("normal");
			if (!is_editable())
				style=get_stylebox("read_only");

			Ref<Font> font=get_font("font");
			
			style->draw( ci, Rect2( Point2(), size ) );

			if (has_focus()) {

				get_stylebox("focus")->draw( ci, Rect2( Point2(), size ) );
			}

						
			int ofs=style->get_offset().x;
			int ofs_max=width-style->get_minimum_size().width;
			int char_ofs=window_pos;
			
			int y_area=height-style->get_minimum_size().height;
			int y_ofs=style->get_offset().y;
			
			int font_ascent=font->get_ascent();
			
			Color selection_color=get_color("selection_color");
			Color font_color=get_color("font_color");
			Color font_color_selected=get_color("font_color_selected");
			Color cursor_color=get_color("cursor_color");
			
			while(true) {
				
		//end of string, break!
				if (char_ofs>=text.length())
					break;

				CharType cchar=pass?'*':text[char_ofs];
				CharType next=pass?'*':text[char_ofs+1];
				int char_width=font->get_char_size( cchar,next ).width;
								
		// end of widget, break!
				if ( (ofs+char_width) > ofs_max )
					break;
				
				
				bool selected=selection.enabled && char_ofs>=selection.begin && char_ofs<selection.end;
				
				if (selected)
					VisualServer::get_singleton()->canvas_item_add_rect(ci,Rect2( Point2( ofs , y_ofs ),Size2( char_width, y_area )),selection_color);
				

				font->draw_char(ci,Point2( ofs , y_ofs+font_ascent ), cchar, next,selected?font_color_selected:font_color );
				
				if (char_ofs==cursor_pos && has_focus())
					VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(
						Point2( ofs , y_ofs ), Size2( 1, y_area ) ), cursor_color );
				
				ofs+=char_width;
				char_ofs++;
			}

			if (char_ofs==cursor_pos && has_focus()) //may be at the end
				VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(
					Point2( ofs , y_ofs ), Size2( 1, y_area ) ), cursor_color );		
			
		} break;
		case NOTIFICATION_FOCUS_ENTER: {

			if (OS::get_singleton()->has_virtual_keyboard())
				OS::get_singleton()->show_virtual_keyboard(get_text(),get_global_rect());

		} break;
		case NOTIFICATION_FOCUS_EXIT: {

			if (OS::get_singleton()->has_virtual_keyboard())
				OS::get_singleton()->hide_virtual_keyboard();

		} break;

	}
}

void LineEdit::copy_text() {
	
	if(selection.enabled) {
		
		OS::get_singleton()->set_clipboard(text.substr(selection.begin, selection.end - selection.begin));
	}
}

void LineEdit::cut_text() {
	
	if(selection.enabled) {
		undo_text = text;
		OS::get_singleton()->set_clipboard(text.substr(selection.begin, selection.end - selection.begin));
		selection_delete();
	}
}

void LineEdit::paste_text() {
	
	String paste_buffer = OS::get_singleton()->get_clipboard();
	
	if(paste_buffer != "") {

		if(selection.enabled) selection_delete();
		append_at_cursor(paste_buffer);

		emit_signal("text_changed",text);
		_change_notify("text");
	}

	

}

void LineEdit::shift_selection_check_pre(bool p_shift) {
	
	if (!selection.old_shift && p_shift)  {
		selection.cursor_start=cursor_pos;
	}
	if (!p_shift)
		selection_clear();
	
}

void LineEdit::shift_selection_check_post(bool p_shift) {
	
	if (p_shift)
		selection_fill_at_cursor();
}

void LineEdit::set_cursor_at_pixel_pos(int p_x) {
	
	int ofs=window_pos;
	int pixel_ofs=get_stylebox("normal")->get_offset().x;
	Ref<Font> font=get_font("font");

	while (ofs<text.length()) {
		
		int char_w=font->get_char_size( text[ofs] ).width;
		pixel_ofs+=char_w;
		
		if (pixel_ofs > p_x) { //found what we look for
			
			
			if ( (pixel_ofs-p_x) < (char_w >> 1 ) ) {
				
				ofs+=1;
			}
			
			break;
		}
		
		
		ofs++;
	}
	
	set_cursor_pos( ofs );
	
	/*
	int new_cursor_pos=p_x;
	int charwidth=draw_area->get_font_char_width(' ',0);
	new_cursor_pos=( ( (new_cursor_pos-2)+ (charwidth/2) ) /charwidth );
	if (new_cursor_pos>(int)text.length()) new_cursor_pos=text.length();
	set_cursor_pos(window_pos+new_cursor_pos); */
}


void LineEdit::delete_char() {
	
	if ((text.length()<=0) || (cursor_pos==0)) return;
	
	
	text.erase( cursor_pos-1, 1 );
	
	set_cursor_pos(get_cursor_pos()-1);
	
	if (cursor_pos==window_pos) {
		
	//	set_window_pos(cursor_pos-get_window_length());
	}
	
	emit_signal("text_changed",text);
	_change_notify("text");
}

void LineEdit::set_text(String p_text) {
	
	clear_internal();
	append_at_cursor(p_text);
	update();
	cursor_pos=0;
	window_pos=0;
}

void LineEdit::clear() {
	
	clear_internal();
}

String LineEdit::get_text() const {
	
	return text;
}

void LineEdit::set_cursor_pos(int p_pos) {
	
	if (p_pos>(int)text.length())
		p_pos=text.length();
	
	if(p_pos<0)
		p_pos=0;
	
	
	
	cursor_pos=p_pos;
	
//	if (cursor_pos>(window_pos+get_window_length())) {
//	set_window_pos(cursor_pos-get_window_lengt//h());
//	}
	
	if (!is_inside_tree()) {
		
		window_pos=cursor_pos;
		return;
	}
	
	Ref<StyleBox> style = get_stylebox("normal");
	Ref<Font> font=get_font("font");
	
	if (cursor_pos<window_pos) {
		/* Adjust window if cursor goes too much to the left */
		set_window_pos(cursor_pos);
	} else if (cursor_pos>window_pos) {
		/* Adjust window if cursor goes too much to the right */
		int window_width=get_size().width-style->get_minimum_size().width;
		
		if (window_width<0)
			return;
		int width_to_cursor=0;
		int wp=window_pos;
		
		for (int i=window_pos;i<cursor_pos;i++)
			width_to_cursor+=font->get_char_size( text[i] ).width;
		
		while(width_to_cursor>=window_width && wp<text.length()) {			
			
			width_to_cursor-=font->get_char_size( text[ wp ] ).width;
			wp++;
		}
		
		if (wp!=window_pos)
			set_window_pos( wp );
		
	}
	update();
}

int LineEdit::get_cursor_pos()  const {
	
	return cursor_pos;
}

void LineEdit::set_window_pos(int p_pos) {
	
	window_pos=p_pos;
	if (window_pos<0) window_pos=0;
}

void LineEdit::append_at_cursor(String p_text) {
	
	
	if ( ( max_length <= 0 ) || (text.length()+p_text.length() <= max_length)) {
		
		undo_text = text;
		String pre = text.substr( 0, cursor_pos );
		String post = text.substr( cursor_pos, text.length()-cursor_pos );
		text=pre+p_text+post;
		set_cursor_pos(cursor_pos+p_text.length());
		
	}
	
}

void LineEdit::clear_internal() {
	
	cursor_pos=0;
	window_pos=0;
	undo_text="";
	text="";
	update();
}

Size2 LineEdit::get_minimum_size() const {
	
	Ref<StyleBox> style = get_stylebox("normal");
	Ref<Font> font=get_font("font");
	
	Size2 min=style->get_minimum_size();
	min.height+=font->get_height();
	min.width+=get_constant("minimum_spaces")*font->get_char_size(' ').x;
	
	return min;
}

/* selection */

void LineEdit::selection_clear() {
	
	selection.begin=0;
	selection.end=0;
	selection.cursor_start=0;
	selection.enabled=false;
	selection.creating=false;
	selection.old_shift=false;
	selection.doubleclick=false;
	update();
}


void LineEdit::selection_delete() {
	
	if (selection.enabled) {
		
		undo_text = text;
		text.erase(selection.begin,selection.end-selection.begin);
		cursor_pos-=CLAMP( cursor_pos-selection.begin, 0, selection.end-selection.begin);
		
		if (cursor_pos>=text.length()) {
			
			cursor_pos=text.length();
		}
		if (window_pos>cursor_pos) {
			
			window_pos=cursor_pos;
		}
		
		emit_signal("text_changed",text);
		_change_notify("text");
	};
	
	selection_clear();
}

void LineEdit::set_max_length(int p_max_length) {
	
	ERR_FAIL_COND(p_max_length<0);
	max_length = p_max_length;
	set_text(text);
}

int LineEdit::get_max_length() const {
	
	return max_length;
}

void LineEdit::selection_fill_at_cursor() {
	
	int aux;
	
	selection.begin=cursor_pos;
	selection.end=selection.cursor_start;
	
	if (selection.end<selection.begin) {
		aux=selection.end;
		selection.end=selection.begin;
		selection.begin=aux;
	}
	
	selection.enabled=(selection.begin!=selection.end);
}

void LineEdit::select_all() {
	
	if (!text.length())
		return;
	
	selection.begin=0;
	selection.end=text.length();
	selection.enabled=true;
	update();
	
}
void LineEdit::set_editable(bool p_editable) {
	
	editable=p_editable;
	update();
}

bool LineEdit::is_editable() const {
	
	return editable;
}

void LineEdit::set_secret(bool p_secret) {
	
	pass=p_secret;
	update();
}
bool LineEdit::is_secret() const {
	
	return pass;
}

void LineEdit::select(int p_from, int p_to) {

	if (p_from==0 && p_to==0) {
		selection_clear();
		return;
	}

	int len = text.length();
	if (p_from<0)
		p_from=0;
	if (p_from>len)
		p_from=len;
	if (p_to<0 || p_to>len)
		p_to=len;

	if (p_from>=p_to)
		return;

	selection.enabled=true;
	selection.begin=p_from;
	selection.end=p_to;
	selection.creating=false;
	selection.old_shift=false;
	selection.doubleclick=false;
	update();
}

bool LineEdit::is_text_field() const {

    return true;
}

void LineEdit::_bind_methods() {
	

	ObjectTypeDB::bind_method(_MD("_input_event"),&LineEdit::_input_event);
	ObjectTypeDB::bind_method(_MD("clear"),&LineEdit::clear);	
	ObjectTypeDB::bind_method(_MD("select_all"),&LineEdit::select_all);	
	ObjectTypeDB::bind_method(_MD("set_text","text"),&LineEdit::set_text);
	ObjectTypeDB::bind_method(_MD("get_text"),&LineEdit::get_text);	
	ObjectTypeDB::bind_method(_MD("set_cursor_pos","pos"),&LineEdit::set_cursor_pos);
	ObjectTypeDB::bind_method(_MD("get_cursor_pos"),&LineEdit::get_cursor_pos);	
	ObjectTypeDB::bind_method(_MD("set_max_length","chars"),&LineEdit::set_max_length);
	ObjectTypeDB::bind_method(_MD("get_max_length"),&LineEdit::get_max_length);	
	ObjectTypeDB::bind_method(_MD("append_at_cursor","text"),&LineEdit::append_at_cursor);
	ObjectTypeDB::bind_method(_MD("set_editable","enabled"),&LineEdit::set_editable);
	ObjectTypeDB::bind_method(_MD("is_editable"),&LineEdit::is_editable);	
	ObjectTypeDB::bind_method(_MD("set_secret","enabled"),&LineEdit::set_secret);
	ObjectTypeDB::bind_method(_MD("is_secret"),&LineEdit::is_secret);	
	ObjectTypeDB::bind_method(_MD("select","from","to"),&LineEdit::select,DEFVAL(0),DEFVAL(-1));

	ADD_SIGNAL( MethodInfo("text_changed", PropertyInfo( Variant::STRING, "text" )) );
	ADD_SIGNAL( MethodInfo("text_entered", PropertyInfo( Variant::STRING, "text" )) );

	ADD_PROPERTY( PropertyInfo( Variant::STRING, "text" ), _SCS("set_text"),_SCS("get_text") );
	ADD_PROPERTY( PropertyInfo( Variant::INT, "max_length" ), _SCS("set_max_length"),_SCS("get_max_length") );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "editable" ), _SCS("set_editable"),_SCS("is_editable") );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "secret" ), _SCS("set_secret"),_SCS("is_secret") );

}

LineEdit::LineEdit() {
	
	cursor_pos=0;
	window_pos=0;
	max_length = 0;
	pass=false;
	
	selection_clear();
	set_focus_mode( FOCUS_ALL );
	editable=true;
	set_default_cursor_shape(CURSOR_IBEAM);
	set_stop_mouse(true);
		
	
}

LineEdit::~LineEdit() {
	
	
}

