/*************************************************************************/
/*  separator.cpp                                                        */
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
#include "separator.h"


Size2 Separator::get_minimum_size() const {

	Size2 ms(3,3);
	ms[orientation]=get_constant("separation");
	return ms;


}

void Separator::_notification(int p_what) {


	switch(p_what) {

		case NOTIFICATION_DRAW: {

			Size2i size = get_size();
			Ref<StyleBox> style = get_stylebox("separator");
			Size2i ssize=style->get_minimum_size()+style->get_center_size();

			if (orientation==VERTICAL) {

				style->draw(get_canvas_item(),Rect2( (size.x-ssize.x)/2,0,ssize.x,size.y ));
			} else {

				style->draw(get_canvas_item(),Rect2( 0,(size.y-ssize.y)/2,size.x,ssize.y ));
			}

		} break;
	}
}

Separator::Separator()
{
}


Separator::~Separator()
{
}

HSeparator::HSeparator() {

	orientation=HORIZONTAL;
}

VSeparator::VSeparator() {

	orientation=VERTICAL;
}

Size2 CollapsibleVSeparator::get_minimum_size() const {

	return Size2(5,get_constant("separation"));
}

void CollapsibleVSeparator::add_control(Control *p_control) {
	controls.push_back(p_control);
}

void CollapsibleVSeparator::remove_control(Control *p_control) {
	controls.erase(p_control);
}

void CollapsibleVSeparator::set_collapsed(bool p_collapsed) {

	toggle_button->set_pressed(p_collapsed);

	collapsed = p_collapsed;
	for (List<Control *>::Element *E = controls.front(); E; E=E->next()) {
		if (collapsed)
			E->get()->hide();
		else
			E->get()->show();
	}
	update();
}

void CollapsibleVSeparator::_on_hovered(bool p_hovered) {

	hovering = p_hovered;
	update();
}

void CollapsibleVSeparator::_notification(int p_what) {

	if (p_what==NOTIFICATION_DRAW) {

		Size2i size = get_size();
		if (size.width==0 || size.height==0)
			return;

		RID ci = get_canvas_item();

		Color color(1,1,1);

		if (collapsed) {
			color.a = 0.7f;
		} else {
			color.a = 0.4f;
		}

		if (hovering)
			color = Color(.9,.9,0,0.7f);

		Size2 handle(MIN(3,size.width), size.height*0.33f);
		Point2 ofs((size.width-handle.width)/2, (size.height-handle.height)/2);

		draw_rect(Rect2(ofs,handle),color);
	}
}

void CollapsibleVSeparator::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("add_control","control"), &CollapsibleVSeparator::add_control);
	ObjectTypeDB::bind_method(_MD("remove_control","control"), &CollapsibleVSeparator::remove_control);
	ObjectTypeDB::bind_method(_MD("set_collapsed"), &CollapsibleVSeparator::set_collapsed);
	ObjectTypeDB::bind_method(_MD("_on_hovered"), &CollapsibleVSeparator::_on_hovered);
}

CollapsibleVSeparator::CollapsibleVSeparator() {

	collapsed = false;
	hovering=false;

	toggle_button = memnew(BaseButton);
	toggle_button->set_toggle_mode(true);
	toggle_button->set_area_as_parent_rect();
	toggle_button->connect("toggled", this, "set_collapsed");
	toggle_button->connect("hovered", this, "_on_hovered");
	add_child(toggle_button);

}
