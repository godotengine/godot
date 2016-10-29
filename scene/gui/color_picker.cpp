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

#include "scene/gui/separator.h"
#include "scene/main/viewport.h"
#include "os/os.h"
#include "os/input.h"
#include "os/keyboard.h"

void update_material(Ref<CanvasItemMaterial>mat,const Color& p_color,float h,float s,float v) {
	if (!mat.is_valid())
		return;
	Ref<Shader> sdr = mat->get_shader();
	if (!sdr.is_valid())
		return;

	mat->set_shader_param("R",p_color.r);
	mat->set_shader_param("G",p_color.g);
	mat->set_shader_param("B",p_color.b);
	mat->set_shader_param("H",h);
	mat->set_shader_param("S",s);
	mat->set_shader_param("V",v);
	mat->set_shader_param("A",p_color.a);
}

void ColorPicker::_notification(int p_what) {


	switch(p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			uv_material->set_shader(get_shader("uv_editor"));
			w_material->set_shader(get_shader("w_editor"));
			update_material(uv_material,color,h,s,v);
			update_material(w_material,color,h,s,v);
			_update_controls();
		} break;

		case NOTIFICATION_ENTER_TREE: {
			btn_pick->set_icon(get_icon("screen_picker", "ColorPicker"));
			update_material(uv_material, color,h,s,v);
			update_material(w_material, color,h,s,v);

			uv_edit->get_child(0)->cast_to<Control>()->update();
			w_edit->get_child(0)->cast_to<Control>()->update();
			_update_color();
		}


	}
}

void ColorPicker::set_focus_on_line_edit() {

	c_text->grab_focus();
	c_text->select();
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
	if (color != last_hsv) {
		h=color.get_h();
		s=color.get_s();
		v=color.get_v();
		last_hsv = color;
	}

	if (!is_inside_tree())
		return;

	update_material(uv_material, color,h,s,v);
	update_material(w_material, color,h,s,v);

	uv_edit->get_child(0)->cast_to<Control>()->update();
	w_edit->get_child(0)->cast_to<Control>()->update();
	_update_color();

}

void ColorPicker::set_edit_alpha(bool p_show) {

	edit_alpha=p_show;
	_update_controls();

	if (!is_inside_tree())
		return;

	_update_color();
	sample->update();
}

bool ColorPicker::is_editing_alpha() const {

	return edit_alpha;
}

void ColorPicker::_value_changed(double) {

	if (updating)
		return;

	for(int i=0;i<4;i++) {
		color.components[i] = scroll[i]->get_val()/(raw_mode_enabled?1.0:255.0);
	}

	set_color(color);

	_update_text_value();

	emit_signal("color_changed",color);

}

void ColorPicker::_html_entered(const String& p_html) {

	if (updating)
		return;

	color = Color::html(p_html);

	if (!is_inside_tree())
		return;

	_update_color();
	emit_signal("color_changed",color);
}

void ColorPicker::_update_color() {

	updating=true;

	for(int i=0;i<4;i++) {
		scroll[i]->set_max(255);
		scroll[i]->set_step(0.01);
		if (raw_mode_enabled) {
			if (i == 3)
				scroll[i]->set_max(1);
			scroll[i]->set_val(color.components[i]);
		} else {
			scroll[i]->set_val(color.components[i] * 255);
		}
	}

	_update_text_value();

	sample->update();
	updating=false;
}

void ColorPicker::_update_presets()
{
	Size2 size=bt_add_preset->get_size();
	preset->set_custom_minimum_size(Size2(size.width*presets.size(),size.height));
	Image i(size.x*presets.size(),size.y, false, Image::FORMAT_RGB);
	for (int y=0;y<size.y;y++)
		for (int x=0;x<size.x*presets.size();x++)
			i.put_pixel(x,y,presets[(int)x/size.x]);
	Ref<ImageTexture> t;
	t.instance();
	t->create_from_image(i);
	preset->set_texture(t);
}

void ColorPicker::_text_type_toggled()
{
	if (!get_tree()->is_editor_hint())
		return;
	text_is_constructor = !text_is_constructor;
	if (text_is_constructor) {
		text_type->set_text("");
		text_type->set_icon(get_icon("Script", "EditorIcons"));
	} else {
		text_type->set_text("#");
		text_type->set_icon(NULL);
	}
	_update_color();
}

Color ColorPicker::get_color() const {

	return color;
}

void ColorPicker::add_preset(const Color &p_color)
{
	if (presets.find(p_color)) {
		presets.move_to_back(presets.find(p_color));
	} else {
		presets.push_back(p_color);
	}
	_update_presets();
	if (presets.size()==10)
		bt_add_preset->hide();
}


void ColorPicker::set_raw_mode(bool p_enabled) {

	if (raw_mode_enabled==p_enabled)
		return;
	raw_mode_enabled=p_enabled;
	if (btn_mode->is_pressed()!=p_enabled)
		btn_mode->set_pressed(p_enabled);

	if (!is_inside_tree())
		return;

	_update_controls();
	_update_color();
}

bool ColorPicker::is_raw_mode() const {

	return raw_mode_enabled;
}


void ColorPicker::_update_text_value() {
	if (text_is_constructor) {
		String t = "Color("+String::num(color.r)+","+String::num(color.g)+","+String::num(color.b);
		if (edit_alpha && color.a<1)
			t+=(","+String::num(color.a)+")") ;
		else
			t+=")";
		c_text->set_text(t);
	} else {
		c_text->set_text(color.to_html(edit_alpha && color.a<1));
	}
}

void ColorPicker::_sample_draw() {
	sample->draw_rect(Rect2(Point2(),Size2(256,20)),color);
}

void ColorPicker::_hsv_draw(int p_wich,Control* c)
{
	if (!c)
		return;
	if (p_wich==0) {
		int x = CLAMP(c->get_size().x * s, 0, c->get_size().x);
		int y = CLAMP(c->get_size().y-c->get_size().y * v, 0, c->get_size().y);
		Color col = color;
		col.a=1;
		c->draw_line(Point2(x,0),Point2(x,c->get_size().y),col.inverted());
		c->draw_line(Point2(0, y),Point2(c->get_size().x, y),col.inverted());
		c->draw_line(Point2(x,y),Point2(x,y),Color(1,1,1),2);
	} else if (p_wich==1) {
		int y=c->get_size().y-c->get_size().y*h;
		Color col=Color();
		col.set_hsv(h,1,1);
		c->draw_line(Point2(0,y),Point2(c->get_size().x,y),col.inverted());
	}
}

void ColorPicker::_uv_input(const InputEvent &ev) {
	if (ev.type == InputEvent::MOUSE_BUTTON) {
		const InputEventMouseButton &bev = ev.mouse_button;
		if (bev.pressed) {
			changing_color = true;
			float x = CLAMP((float)bev.x,0,256);
			float y = CLAMP((float)bev.y,0,256);
			s=x/256;
			v=1.0-y/256.0;
			color.set_hsv(h,s,v,color.a);
			last_hsv = color;
			set_color(color);
			_update_color();
			emit_signal("color_changed", color);
		} else {
			changing_color = false;
		}
	} else if (ev.type == InputEvent::MOUSE_MOTION) {
		const InputEventMouse &bev = ev.mouse_motion;
		if (!changing_color)
			return;
		float x = CLAMP((float)bev.x,0,256);
		float y = CLAMP((float)bev.y,0,256);
		s=x/256;
		v=1.0-y/256.0;
		color.set_hsv(h,s,v,color.a);
		last_hsv = color;
		set_color(color);
		_update_color();
		emit_signal("color_changed", color);
	}
}

void ColorPicker::_w_input(const InputEvent &ev) {
	if (ev.type == InputEvent::MOUSE_BUTTON) {
		const InputEventMouseButton &bev = ev.mouse_button;
		if (bev.pressed) {
			changing_color = true;
			h=1-((float)bev.y)/256.0;

		} else {
			changing_color = false;
		}
		color.set_hsv(h,s,v,color.a);
		last_hsv = color;
		set_color(color);
		_update_color();
		emit_signal("color_changed", color);
	} else if (ev.type == InputEvent::MOUSE_MOTION) {
		const InputEventMouse &bev = ev.mouse_motion;
		if (!changing_color)
			return;
		float y = CLAMP((float)bev.y,0,256);
		h=1.0-y/256.0;
		color.set_hsv(h,s,v,color.a);
		last_hsv = color;
		set_color(color);
		_update_color();
		emit_signal("color_changed", color);
	}
}

void ColorPicker::_preset_input(const InputEvent &ev) {
	if (ev.type == InputEvent::MOUSE_BUTTON) {
		const InputEventMouseButton &bev = ev.mouse_button;
		if (bev.pressed && bev.button_index==BUTTON_LEFT) {
			int index = bev.x/(preset->get_size().x/presets.size());
			set_color(presets[index]);
		} else if (bev.pressed && bev.button_index==BUTTON_RIGHT) {
			int index = bev.x/(preset->get_size().x/presets.size());
			presets.erase(presets[index]);
			_update_presets();
			bt_add_preset->show();
		}
		_update_color();
		emit_signal("color_changed", color);
	} else if (ev.type == InputEvent::MOUSE_MOTION) {
		const InputEventMouse &mev = ev.mouse_motion;
		int index = mev.x * presets.size();
		if( preset->get_size().x != 0 ) {
			index /= preset->get_size().x;
		}
		if (index<0 || index >= presets.size())
			return;
		preset->set_tooltip("Color: #"+presets[index].to_html(presets[index].a<1)+"\n"
							"LMB: Set color\n"
							"RMB: Remove preset");
	}
}

void ColorPicker::_screen_input(const InputEvent &ev)
{
	if (ev.type==InputEvent::MOUSE_BUTTON) {
		const InputEventMouseButton &bev = ev.mouse_button;
		if (bev.button_index==BUTTON_LEFT&&!bev.pressed) {
			emit_signal("color_changed", color);
			screen->hide();
		}
	} else if (ev.type==InputEvent::MOUSE_MOTION) {
		const InputEventMouse &mev = ev.mouse_motion;
		Viewport *r=get_tree()->get_root();
		if (!r->get_rect().has_point(Point2(mev.global_x,mev.global_y)))
			return;
		Image img =r->get_screen_capture();
		if (!img.empty()) {
			last_capture=img;
			r->queue_screen_capture();
		}
		if (!last_capture.empty())
			set_color(last_capture.get_pixel(mev.global_x,mev.global_y));
	}
}

void ColorPicker::_add_preset_pressed() {
	add_preset(color);
}

void ColorPicker::_screen_pick_pressed()
{
	Viewport *r=get_tree()->get_root();
	if (!screen) {
		screen=memnew( Control );
		r->add_child(screen);
		screen->set_as_toplevel(true);
		screen->set_area_as_parent_rect();
		screen->connect("input_event",this,"_screen_input");
	}
	screen->raise();
	screen->show_modal();
	r->queue_screen_capture();
}

void ColorPicker::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_color","color"),&ColorPicker::set_color);
	ObjectTypeDB::bind_method(_MD("get_color"),&ColorPicker::get_color);
	ObjectTypeDB::bind_method(_MD("set_raw_mode","mode"),&ColorPicker::set_raw_mode);
	ObjectTypeDB::bind_method(_MD("is_raw_mode"),&ColorPicker::is_raw_mode);
	ObjectTypeDB::bind_method(_MD("set_edit_alpha","show"),&ColorPicker::set_edit_alpha);
	ObjectTypeDB::bind_method(_MD("is_editing_alpha"),&ColorPicker::is_editing_alpha);
	ObjectTypeDB::bind_method(_MD("add_preset"), &ColorPicker::add_preset);
	ObjectTypeDB::bind_method(_MD("_value_changed"),&ColorPicker::_value_changed);
	ObjectTypeDB::bind_method(_MD("_html_entered"),&ColorPicker::_html_entered);
	ObjectTypeDB::bind_method(_MD("_text_type_toggled"),&ColorPicker::_text_type_toggled);
	ObjectTypeDB::bind_method(_MD("_add_preset_pressed"), &ColorPicker::_add_preset_pressed);
	ObjectTypeDB::bind_method(_MD("_screen_pick_pressed"), &ColorPicker::_screen_pick_pressed);
	ObjectTypeDB::bind_method(_MD("_sample_draw"),&ColorPicker::_sample_draw);
	ObjectTypeDB::bind_method(_MD("_hsv_draw"),&ColorPicker::_hsv_draw);
	ObjectTypeDB::bind_method(_MD("_uv_input"),&ColorPicker::_uv_input);
	ObjectTypeDB::bind_method(_MD("_w_input"),&ColorPicker::_w_input);
	ObjectTypeDB::bind_method(_MD("_preset_input"),&ColorPicker::_preset_input);
	ObjectTypeDB::bind_method(_MD("_screen_input"),&ColorPicker::_screen_input);

	ADD_SIGNAL( MethodInfo("color_changed",PropertyInfo(Variant::COLOR,"color")));
}

ColorPicker::ColorPicker() :
	BoxContainer(true) {

	updating=true;
	edit_alpha=true;
	text_is_constructor = false;
	raw_mode_enabled=false;
	changing_color=false;
	screen=NULL;

	HBoxContainer *hb_smpl = memnew( HBoxContainer );
	btn_pick = memnew( ToolButton );
	btn_pick->connect("pressed",this,"_screen_pick_pressed");

	sample = memnew( TextureFrame );
	sample->set_h_size_flags(SIZE_EXPAND_FILL);
	sample->connect("draw",this,"_sample_draw");

	hb_smpl->add_child(sample);
	hb_smpl->add_child(btn_pick);
	add_child(hb_smpl);

	HBoxContainer *hb_edit = memnew( HBoxContainer );

	uv_edit= memnew ( TextureFrame );
	Image i(256, 256, false, Image::FORMAT_RGB);
	for (int y=0;y<256;y++)
		for (int x=0;x<256;x++)
			i.put_pixel(x,y,Color());
	Ref<ImageTexture> t;
	t.instance();
	t->create_from_image(i);
	uv_edit->set_texture(t);
	uv_edit->set_ignore_mouse(false);
	uv_edit->set_custom_minimum_size(Size2(256,256));
	uv_edit->connect("input_event", this, "_uv_input");
	Control *c= memnew( Control );
	uv_edit->add_child(c);
	c->set_area_as_parent_rect();
	c->set_stop_mouse(false);
	c->set_material(memnew ( CanvasItemMaterial ));
	Vector<Variant> args=Vector<Variant>();
	args.push_back(0);
	args.push_back(c);
	c->connect("draw",this,"_hsv_draw",args);

	add_child(hb_edit);
	w_edit= memnew( TextureFrame );
	i = Image(15, 256, false, Image::FORMAT_RGB);
	for (int y=0;y<256;y++)
		for (int x=0;x<15;x++)
			i.put_pixel(x,y,Color());
	Ref<ImageTexture> tw;
	tw.instance();
	tw->create_from_image(i);
	w_edit->set_texture(tw);
	w_edit->set_ignore_mouse(false);
	w_edit->set_custom_minimum_size(Size2(15,256));
	w_edit->connect("input_event", this, "_w_input");
	c= memnew( Control );
	w_edit->add_child(c);
	c->set_area_as_parent_rect();
	c->set_stop_mouse(false);
	c->set_material(memnew ( CanvasItemMaterial ));
	args.clear();
	args.push_back(1);
	args.push_back(c);
	c->connect("draw",this,"_hsv_draw",args);

	hb_edit->add_child(uv_edit);
	hb_edit->add_child(memnew( VSeparator ));
	hb_edit->add_child(w_edit);

	VBoxContainer *vbl = memnew( VBoxContainer );
	add_child(vbl);

	add_child(memnew( HSeparator ));

	VBoxContainer *vbr = memnew( VBoxContainer );
	add_child(vbr);
	vbr->set_h_size_flags(SIZE_EXPAND_FILL);
	const char* lt[4] = {"R","G","B","A"};

	for(int i=0;i<4;i++) {

		HBoxContainer *hbc = memnew( HBoxContainer );

		labels[i]=memnew( Label(lt[i]) );
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

	btn_mode = memnew( CheckButton );
	btn_mode->set_text("RAW Mode");
	btn_mode->connect("toggled", this, "set_raw_mode");
	hhb->add_child(btn_mode);
	vbr->add_child(hhb);
	text_type = memnew( Button );
	text_type->set_flat(true);
	text_type->connect("pressed", this, "_text_type_toggled");
	hhb->add_child(text_type);

	c_text = memnew( LineEdit );
	hhb->add_child(c_text);
	c_text->connect("text_entered",this,"_html_entered");
	text_type->set_text("#");
	c_text->set_h_size_flags(SIZE_EXPAND_FILL);


	_update_controls();
	//_update_color();
	updating=false;

	uv_material.instance();
	Ref<Shader> s_uv = get_shader("uv_editor");
	uv_material->set_shader(s_uv);

	w_material.instance();

	Ref<Shader> s_w = get_shader("w_editor");
	w_material->set_shader(s_w);

	uv_edit->set_material(uv_material);
	w_edit->set_material(w_material);

	set_color(Color(1,1,1));

	i.create(256,20,false,Image::FORMAT_RGB);
	for (int y=0;y<20;y++)
		for(int x=0;x<256;x++)
			if ((x/4+y/4)%2)
				i.put_pixel(x,y,Color(1,1,1));
			else
				i.put_pixel(x,y,Color(0.6,0.6,0.6));
	Ref<ImageTexture> t_smpl;
	t_smpl.instance();
	t_smpl->create_from_image(i);
	sample->set_texture(t_smpl);

	HBoxContainer *bbc = memnew( HBoxContainer );
	add_child(bbc);

	preset = memnew( TextureFrame );
	bbc->add_child(preset);
	preset->set_ignore_mouse(false);
	preset->connect("input_event", this, "_preset_input");

	bt_add_preset = memnew ( Button );
	bt_add_preset->set_icon(get_icon("add_preset"));
	bt_add_preset->set_tooltip("Add current color as a preset");
	bt_add_preset->connect("pressed", this, "_add_preset_pressed");
	bbc->add_child(bt_add_preset);
}




/////////////////


void ColorPickerButton::_color_changed(const Color& p_color) {

	update();
	emit_signal("color_changed",p_color);
}


void ColorPickerButton::pressed() {

	Size2 ms = Size2(300, picker->get_combined_minimum_size().height+10);
	popup->set_pos(get_global_pos()-Size2(0,ms.height));
	popup->set_size(ms);
	popup->popup();
	picker->set_focus_on_line_edit();


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

ColorPicker *ColorPickerButton::get_picker() {
	return picker;
}

void ColorPickerButton::_bind_methods(){

	ObjectTypeDB::bind_method(_MD("set_color","color"),&ColorPickerButton::set_color);
	ObjectTypeDB::bind_method(_MD("get_color"),&ColorPickerButton::get_color);
	ObjectTypeDB::bind_method(_MD("get_picker:ColorPicker"),&ColorPickerButton::get_picker);
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
