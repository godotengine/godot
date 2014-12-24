/*************************************************************************/
/*  dialogs.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "dialogs.h"
#include "print_string.h"
#include "line_edit.h"
#include "translation.h"

void WindowDialog::_post_popup() {

	dragging=false; //just in case
}

bool WindowDialog::has_point(const Point2& p_point) const {


	int extra = get_constant("titlebar_height","WindowDialog");
	Rect2 r( Point2(), get_size() );
	r.pos.y-=extra;
	r.size.y+=extra;
	return r.has_point(p_point);

}

void WindowDialog::_input_event(const InputEvent& p_event) {

	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.button_index==BUTTON_LEFT) {

		if (p_event.mouse_button.pressed && p_event.mouse_button.y < 0)
			dragging=true;
		else if (dragging && !p_event.mouse_button.pressed)
			dragging=false;
	}


	if (p_event.type == InputEvent::MOUSE_MOTION && dragging) {

		Point2 rel( p_event.mouse_motion.relative_x, p_event.mouse_motion.relative_y );
		Point2 pos = get_pos();
		Size2 size = get_size();

		pos+=rel;

		if (pos.y<0)
			pos.y=0;

		set_pos(pos);
	}
}

void WindowDialog::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_DRAW: {

			RID ci = get_canvas_item();
			Size2 s = get_size();
			Ref<StyleBox> st = get_stylebox("panel","WindowDialog");
			st->draw(ci,Rect2(Point2(),s));
			int th = get_constant("title_height","WindowDialog");
			Color tc = get_color("title_color","WindowDialog");
			Ref<Font> font = get_font("title_font","WindowDialog");
			int ofs = (s.width-font->get_string_size(title).width)/2;
			//int ofs = st->get_margin(MARGIN_LEFT);
			draw_string(font,Point2(ofs,-th+font->get_ascent()),title,tc,s.width - st->get_minimum_size().width);


		} break;
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {

			close_button->set_normal_texture( get_icon("close","WindowDialog"));
			close_button->set_pressed_texture( get_icon("close","WindowDialog"));
			close_button->set_hover_texture( get_icon("close_hilite","WindowDialog"));
			close_button->set_anchor(MARGIN_LEFT,ANCHOR_END);
			close_button->set_begin( Point2( get_constant("close_h_ofs","WindowDialog"), -get_constant("close_v_ofs","WindowDialog") ));

		} break;
	}

}

void WindowDialog::_closed() {

	_close_pressed();
	hide();
}

void WindowDialog::set_title(const String& p_title) {

	title=XL_MESSAGE(p_title);
	update();
}

String WindowDialog::get_title() const {

	return title;
}


TextureButton *WindowDialog::get_close_button() {


	return close_button;
}

void WindowDialog::_bind_methods() {

	ObjectTypeDB::bind_method( _MD("_input_event"),&WindowDialog::_input_event);
	ObjectTypeDB::bind_method( _MD("set_title","title"),&WindowDialog::set_title);
	ObjectTypeDB::bind_method( _MD("get_title"),&WindowDialog::get_title);
	ObjectTypeDB::bind_method( _MD("_closed"),&WindowDialog::_closed);
	ObjectTypeDB::bind_method( _MD("get_close_button:TextureButton"),&WindowDialog::get_close_button);

	ADD_PROPERTY( PropertyInfo(Variant::STRING,"window/title",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_DEFAULT_INTL),_SCS("set_title"),_SCS("get_title"));
}

WindowDialog::WindowDialog() {

	//title="Hello!";
	dragging=false;
	close_button = memnew( TextureButton );
	add_child(close_button);
	close_button->connect("pressed",this,"_closed");

}

WindowDialog::~WindowDialog(){


}


void PopupDialog::_notification(int p_what) {

	if (p_what==NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();
		get_stylebox("panel","PopupMenu")->draw(ci,Rect2(Point2(),get_size()));
	}
}

PopupDialog::PopupDialog() {


}

PopupDialog::~PopupDialog() {


}


//


void AcceptDialog::_post_popup() {

	WindowDialog::_post_popup();
	get_ok()->grab_focus();

}

void AcceptDialog::_notification(int p_what) {
	
	if (p_what==NOTIFICATION_MODAL_CLOSE) {

		cancel_pressed();
	} if (p_what==NOTIFICATION_DRAW) {




	}	
}

void AcceptDialog::_builtin_text_entered(const String& p_text) {

	_ok_pressed();
}

void AcceptDialog::_ok_pressed() {
	
	if (hide_on_ok)
		hide();
	ok_pressed();
	emit_signal("confirmed");

}
void AcceptDialog::_close_pressed() {
	
	cancel_pressed();
}

String AcceptDialog::get_text() const {

	return label->get_text();
}
void AcceptDialog::set_text(String p_text) {
	
	label->set_text(p_text);
}

void AcceptDialog::set_hide_on_ok(bool p_hide) {

	hide_on_ok=p_hide;
}

bool AcceptDialog::get_hide_on_ok() const {

	return hide_on_ok;
}


void AcceptDialog::register_text_enter(Node *p_line_edit) {

	ERR_FAIL_NULL(p_line_edit);
	p_line_edit->connect("text_entered", this,"_builtin_text_entered");
}

void AcceptDialog::set_child_rect(Control *p_child) {

	ERR_FAIL_COND(p_child->get_parent()!=this);

	p_child->set_area_as_parent_rect(get_constant("margin","Dialogs"));
	p_child->set_margin(MARGIN_BOTTOM, get_constant("button_margin","Dialogs")+10);
}

void AcceptDialog::_custom_action(const String& p_action) {

	emit_signal("custom_action",p_action);
	custom_action(p_action);
}

Button* AcceptDialog::add_button(const String& p_text,bool p_right,const String& p_action) {


	Button *button = memnew( Button );
	button->set_text(p_text);
	if (p_right) {
		hbc->add_child(button);
		hbc->add_spacer();
	} else {

		hbc->add_child(button);
		hbc->move_child(button,0);
		hbc->add_spacer(true);
	}

	if (p_action!="") {
		button->connect("pressed",this,"_custom_action",make_binds(p_action));
	}

	return button;
}

Button* AcceptDialog::add_cancel(const String &p_cancel) {

	String c = p_cancel;
	if (p_cancel=="")
		c="Cancel";
	Button *b = swap_ok_cancel ? add_button("Cancel",true) : add_button("Cancel");
	b->connect("pressed",this,"_closed");
	return b;
}

void AcceptDialog::_bind_methods() {
	
	ObjectTypeDB::bind_method(_MD("_ok"),&AcceptDialog::_ok_pressed);
	ObjectTypeDB::bind_method(_MD("get_ok"),&AcceptDialog::get_ok);
	ObjectTypeDB::bind_method(_MD("get_label"),&AcceptDialog::get_label);
	ObjectTypeDB::bind_method(_MD("set_hide_on_ok","enabled"),&AcceptDialog::set_hide_on_ok);
	ObjectTypeDB::bind_method(_MD("get_hide_on_ok"),&AcceptDialog::get_hide_on_ok);
	ObjectTypeDB::bind_method(_MD("add_button:Button","text","right","action"),&AcceptDialog::add_button,DEFVAL(false),DEFVAL(""));
	ObjectTypeDB::bind_method(_MD("add_cancel:Button","name"),&AcceptDialog::add_cancel);
	ObjectTypeDB::bind_method(_MD("_builtin_text_entered"),&AcceptDialog::_builtin_text_entered);
	ObjectTypeDB::bind_method(_MD("register_text_enter:LineEdit","line_edit"),&AcceptDialog::register_text_enter);
	ObjectTypeDB::bind_method(_MD("_custom_action"),&AcceptDialog::_custom_action);
	ObjectTypeDB::bind_method(_MD("set_text","text"),&AcceptDialog::set_text);
	ObjectTypeDB::bind_method(_MD("get_text"),&AcceptDialog::get_text);

	ADD_SIGNAL( MethodInfo("confirmed") );
	ADD_SIGNAL( MethodInfo("custom_action",PropertyInfo(Variant::STRING,"action")) );

	
}


bool AcceptDialog::swap_ok_cancel=false;
void AcceptDialog::set_swap_ok_cancel(bool p_swap) {

	swap_ok_cancel=p_swap;
}

AcceptDialog::AcceptDialog() {
	
	int margin = get_constant("margin","Dialogs");
	int button_margin = get_constant("button_margin","Dialogs");
	
	
	label = memnew( Label );
	label->set_anchor(MARGIN_RIGHT,ANCHOR_END);
	label->set_anchor(MARGIN_BOTTOM,ANCHOR_END);
	label->set_begin( Point2( margin, margin) );
	label->set_end( Point2( margin, button_margin) );
	label->set_autowrap(true);
	add_child(label);

	hbc = memnew( HBoxContainer );
	hbc->set_area_as_parent_rect(margin);
	hbc->set_anchor_and_margin(MARGIN_TOP,ANCHOR_END,button_margin);
	add_child(hbc);

	hbc->add_spacer();
	ok = memnew( Button );
	ok->set_text("OK");
	hbc->add_child(ok);
	hbc->add_spacer();
	//add_child(ok);
	
	
	ok->connect("pressed", this,"_ok");
	set_as_toplevel(true);

	hide_on_ok=true;
	set_title("Alert!");
}


AcceptDialog::~AcceptDialog()
{
}


void ConfirmationDialog::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_cancel:Button"),&ConfirmationDialog::get_cancel);
}

Button *ConfirmationDialog::get_cancel() {

	return cancel;
}

ConfirmationDialog::ConfirmationDialog() {

	set_title("Please Confirm...");
	cancel = add_cancel();
}
