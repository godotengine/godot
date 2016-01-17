/*************************************************************************/
/*  color_picker.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "color_picker.h"




void ColorPicker::_notification(int p_what) {


	switch(p_what) {
		case NOTIFICATION_THEME_CHANGED: {

			_update_controls();
		} break;

/*		case NOTIFICATION_DRAW: {

			int w = get_constant("color_width");
			int h = ms.height;
			VisualServer::get_singleton()->canvas_item_add_rect(get_canvas_item(),Rect2(0,0,w,h),color);

		} break;*/
	}
}

void ColorPicker::_update_controls() {

	if (edit_alpha) {
		values[3]->show();
		scroll[3]->show();
		labels[3]->show();
	} else {
		values[3]->hide();
		scroll[3]->hide();
		labels[3]->hide();
	}

}


void ColorPicker::set_color(const Color& p_color) {

	color=p_color;
	_update_color();

}

void ColorPicker::set_edit_alpha(bool p_show) {

	edit_alpha=p_show;
	_update_controls();
	_update_color();
	color_box->update();
}

bool ColorPicker::is_editing_alpha() const {

	return edit_alpha;
}

void ColorPicker::_value_changed(double) {

	if (updating)
		return;

	switch(mode) {

		case MODE_RGB: {

			for(int i=0;i<4;i++) {
				color.components[i] = scroll[i]->get_val() / 255.0;
			}

		} break;
		case MODE_HSV: {

			color.set_hsv( CLAMP(scroll[0]->get_val()/359,0,0.9972), scroll[1]->get_val()/100, scroll[2]->get_val()/100 );
			color.a=scroll[3]->get_val()/100.0;

		} break;
		case MODE_RAW: {

			for(int i=0;i<4;i++) {
				color.components[i] = scroll[i]->get_val();
			}

		} break;

	}


	html->set_text(color.to_html(edit_alpha && color.a<1));

	color_box->update();

	emit_signal("color_changed",color);

}

void ColorPicker::_html_entered(const String& p_html) {

	if (updating)
		return;

	color = Color::html(p_html);
	_update_color();
	emit_signal("color_changed",color);
}

void ColorPicker::_update_color() {

	updating=true;

	switch(mode) {

		case MODE_RAW: {

			static const char*_lt[4]={"R","G","B","A"};

			for(int i=0;i<4;i++) {
				scroll[i]->set_max(255);
				scroll[i]->set_step(0.01);
				scroll[i]->set_val(color.components[i]);
				labels[i]->set_text(_lt[i]);
			}
		} break;
		case MODE_RGB: {

			static const char*_lt[4]={"R","G","B","A"};

			for(int i=0;i<4;i++) {
				scroll[i]->set_max(255);
				scroll[i]->set_step(1);
				scroll[i]->set_val(color.components[i]*255);
				labels[i]->set_text(_lt[i]);
			}

		} break;
		case MODE_HSV: {

			static const char*_lt[4]={"H","S","V","A"};

			for(int i=0;i<4;i++) {
				labels[i]->set_text(_lt[i]);
			}

			scroll[0]->set_max(359);
			scroll[0]->set_step(0.01);
			scroll[0]->set_val( color.get_h()*359 );

			scroll[1]->set_max(100);
			scroll[1]->set_step(0.01);
			scroll[1]->set_val( color.get_s()*100 );

			scroll[2]->set_max(100);
			scroll[2]->set_step(0.01);
			scroll[2]->set_val( color.get_v()*100 );

			scroll[3]->set_max(100);
			scroll[3]->set_step(0.01);
			scroll[3]->set_val( color.a*100);

		} break;
	}


	html->set_text(color.to_html(edit_alpha && color.a<1));

	color_box->update();
	updating=false;
}

Color ColorPicker::get_color() const {

	return color;
}


void ColorPicker::set_mode(Mode p_mode) {

	ERR_FAIL_INDEX(p_mode,3);
	mode=p_mode;
	if (mode_box->get_selected()!=p_mode)
		mode_box->select(p_mode);

	_update_controls();
	_update_color();
}

ColorPicker::Mode ColorPicker::get_mode() const {

	return mode;
}

void ColorPicker::_color_box_draw() {

	color_box->draw_rect( Rect2( Point2(), color_box->get_size()), color);
}

void ColorPicker::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_color","color"),&ColorPicker::set_color);
	ObjectTypeDB::bind_method(_MD("get_color"),&ColorPicker::get_color);
	ObjectTypeDB::bind_method(_MD("set_mode","mode"),&ColorPicker::set_mode);
	ObjectTypeDB::bind_method(_MD("get_mode"),&ColorPicker::get_mode);
	ObjectTypeDB::bind_method(_MD("set_edit_alpha","show"),&ColorPicker::set_edit_alpha);
	ObjectTypeDB::bind_method(_MD("is_editing_alpha"),&ColorPicker::is_editing_alpha);
	ObjectTypeDB::bind_method(_MD("_value_changed"),&ColorPicker::_value_changed);
	ObjectTypeDB::bind_method(_MD("_html_entered"),&ColorPicker::_html_entered);
	ObjectTypeDB::bind_method(_MD("_color_box_draw"),&ColorPicker::_color_box_draw);

	ADD_SIGNAL( MethodInfo("color_changed",PropertyInfo(Variant::COLOR,"color")));
}




ColorPicker::ColorPicker() {


	//edit_alpha=false;
	updating=true;
	edit_alpha=true;

	VBoxContainer *vbl = memnew( VBoxContainer );
	add_child(vbl);

	mode_box = memnew( OptionButton );
	mode_box->add_item("RGB");
	mode_box->add_item("HSV");
	mode_box->add_item("RAW");
	mode_box->connect("item_selected",this,"set_mode");

	color_box=memnew( Control );
	color_box->set_v_size_flags(SIZE_EXPAND_FILL);
	vbl->add_child(color_box);
	color_box->connect("draw",this,"_color_box_draw");

	vbl->add_child(mode_box);


	VBoxContainer *vbr = memnew( VBoxContainer );
	add_child(vbr);
	vbr->set_h_size_flags(SIZE_EXPAND_FILL);


	for(int i=0;i<4;i++) {

		HBoxContainer *hbc = memnew( HBoxContainer );

		labels[i]=memnew( Label );
		hbc->add_child(labels[i]);

		scroll[i]=memnew( HSlider );
		hbc->add_child(scroll[i]);

		values[i]=memnew( SpinBox );
		scroll[i]->share(values[i]);
		hbc->add_child(values[i]);


		scroll[i]->set_min(0);
		scroll[i]->set_page(0);
		scroll[i]->set_h_size_flags(SIZE_EXPAND_FILL);

		scroll[i]->connect("value_changed",this,"_value_changed");

		vbr->add_child(hbc);


	}

	HBoxContainer *hhb = memnew( HBoxContainer );
	vbr->add_child(hhb);
	html_num = memnew( Label );
	hhb->add_child(html_num);

	html = memnew( LineEdit );
	hhb->add_child(html);
	html->connect("text_entered",this,"_html_entered");
	html_num->set_text("#");
	html->set_h_size_flags(SIZE_EXPAND_FILL);


	mode=MODE_RGB;
	_update_controls();
	_update_color();
	updating=false;

}




/////////////////


void ColorPickerButton::_color_changed(const Color& p_color) {

	update();
	emit_signal("color_changed",p_color);
}


void ColorPickerButton::pressed() {

	Size2 ms = Size2(350, picker->get_combined_minimum_size().height+10);
	popup->set_pos(get_global_pos()-Size2(0,ms.height));
	popup->set_size(ms);
	popup->popup();
}

void ColorPickerButton::_notification(int p_what) {


	if (p_what==NOTIFICATION_DRAW) {

		Ref<StyleBox> normal = get_stylebox("normal" );
		draw_rect(Rect2(normal->get_offset(),get_size()-normal->get_minimum_size()),picker->get_color());
	}
}


void ColorPickerButton::set_color(const Color& p_color){


	picker->set_color(p_color);
	update();
}
Color ColorPickerButton::get_color() const{

	return picker->get_color();
}

void ColorPickerButton::set_edit_alpha(bool p_show) {

	picker->set_edit_alpha(p_show);
}

bool ColorPickerButton::is_editing_alpha() const{

	return picker->is_editing_alpha();

}

void ColorPickerButton::_bind_methods(){

	ObjectTypeDB::bind_method(_MD("set_color","color"),&ColorPickerButton::set_color);
	ObjectTypeDB::bind_method(_MD("get_color"),&ColorPickerButton::get_color);
	ObjectTypeDB::bind_method(_MD("set_edit_alpha","show"),&ColorPickerButton::set_edit_alpha);
	ObjectTypeDB::bind_method(_MD("is_editing_alpha"),&ColorPickerButton::is_editing_alpha);
	ObjectTypeDB::bind_method(_MD("_color_changed"),&ColorPickerButton::_color_changed);

	ADD_SIGNAL( MethodInfo("color_changed",PropertyInfo(Variant::COLOR,"color")));
	ADD_PROPERTY( PropertyInfo(Variant::COLOR,"color"),_SCS("set_color"),_SCS("get_color") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"edit_alpha"),_SCS("set_edit_alpha"),_SCS("is_editing_alpha") );

}

ColorPickerButton::ColorPickerButton() {

	popup = memnew( PopupPanel );
	picker = memnew( ColorPicker );
	popup->add_child(picker);
	popup->set_child_rect(picker);
	picker->connect("color_changed",this,"_color_changed");
	add_child(popup);
}

