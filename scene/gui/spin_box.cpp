/*************************************************************************/
/*  spin_box.cpp                                                         */
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
#include "spin_box.h"


Size2 SpinBox::get_minimum_size() const {

	Size2 ms = line_edit->get_combined_minimum_size();
	ms.width+=last_w;
	return ms;
}


void SpinBox::_value_changed(double) {

	String value = String::num(get_val(),Math::decimals(get_step()));
	if (prefix!="")
		value=prefix+" "+value;
	if (suffix!="")
		value+=" "+suffix;
	line_edit->set_text(value);
}

void SpinBox::_text_entered(const String& p_string) {

	//if (!p_string.is_numeric())
	//	return;
	set_val( p_string.to_double() );
	_value_changed(0);
}


LineEdit *SpinBox::get_line_edit() {

	return line_edit;
}


void SpinBox::_input_event(const InputEvent& p_event) {

	if (p_event.type==InputEvent::MOUSE_BUTTON && p_event.mouse_button.pressed) {
		const InputEventMouseButton &mb=p_event.mouse_button;

		if (mb.doubleclick)
			return; //ignore doubleclick

		bool up = mb.y < (get_size().height/2);

		switch(mb.button_index) {

			case BUTTON_LEFT: {

				set_val( get_val() + (up?get_step():-get_step()));

			} break;
			case BUTTON_RIGHT: {

				set_val(  (up?get_max():get_min()) );

			} break;
			case BUTTON_WHEEL_UP: {

				set_val( get_val() + get_step() );
			} break;
			case BUTTON_WHEEL_DOWN: {

				set_val( get_val() - get_step() );
			} break;
		}
	}
}


void SpinBox::_line_edit_focus_exit() {

	_text_entered(line_edit->get_text());
}

void SpinBox::_notification(int p_what) {

	if (p_what==NOTIFICATION_DRAW) {

		Ref<Texture> updown = get_icon("updown");

		int w = updown->get_width();
		if (w!=last_w) {
			line_edit->set_margin(MARGIN_RIGHT,w);
			last_w=w;
		}

		RID ci = get_canvas_item();
		Size2i size = get_size();

		updown->draw(ci,Point2i(size.width-updown->get_width(),(size.height-updown->get_height())/2));
	} else if (p_what==NOTIFICATION_FOCUS_EXIT) {


		//_value_changed(0);
	} else if (p_what==NOTIFICATION_ENTER_TREE) {

		_value_changed(0);
	}

}


void SpinBox::set_suffix(const String& p_suffix) {

	suffix=p_suffix;
	_value_changed(0);

}

String SpinBox::get_suffix() const{

	return suffix;
}


void SpinBox::set_prefix(const String& p_prefix) {

	prefix=p_prefix;
	_value_changed(0);

}

String SpinBox::get_prefix() const{

	return prefix;
}

void SpinBox::set_editable(bool p_editable) {
	line_edit->set_editable(p_editable);
}

bool SpinBox::is_editable() const {

	return line_edit->is_editable();
}

void SpinBox::_bind_methods() {

	//ObjectTypeDB::bind_method(_MD("_value_changed"),&SpinBox::_value_changed);
	ObjectTypeDB::bind_method(_MD("_input_event"),&SpinBox::_input_event);
	ObjectTypeDB::bind_method(_MD("_text_entered"),&SpinBox::_text_entered);
	ObjectTypeDB::bind_method(_MD("set_suffix","suffix"),&SpinBox::set_suffix);
	ObjectTypeDB::bind_method(_MD("get_suffix"),&SpinBox::get_suffix);
	ObjectTypeDB::bind_method(_MD("set_prefix","prefix"),&SpinBox::set_prefix);
	ObjectTypeDB::bind_method(_MD("get_prefix"),&SpinBox::get_prefix);
	ObjectTypeDB::bind_method(_MD("set_editable","editable"),&SpinBox::set_editable);
	ObjectTypeDB::bind_method(_MD("is_editable"),&SpinBox::is_editable);
	ObjectTypeDB::bind_method(_MD("_line_edit_focus_exit"),&SpinBox::_line_edit_focus_exit);
	ObjectTypeDB::bind_method(_MD("get_line_edit"),&SpinBox::get_line_edit);


	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"editable"),_SCS("set_editable"),_SCS("is_editable"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"prefix"),_SCS("set_prefix"),_SCS("get_prefix"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"suffix"),_SCS("set_suffix"),_SCS("get_suffix"));


}

SpinBox::SpinBox() {

	last_w = 0;
	line_edit = memnew( LineEdit );
	add_child(line_edit);

	line_edit->set_area_as_parent_rect();
	//connect("value_changed",this,"_value_changed");
	line_edit->connect("text_entered",this,"_text_entered",Vector<Variant>(),CONNECT_DEFERRED);
	line_edit->connect("focus_exit",this,"_line_edit_focus_exit",Vector<Variant>(),CONNECT_DEFERRED);
}
