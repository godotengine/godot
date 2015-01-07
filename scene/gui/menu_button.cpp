/*************************************************************************/
/*  menu_button.cpp                                                      */
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
#include "menu_button.h"
#include "os/keyboard.h"


void MenuButton::_unhandled_key_input(InputEvent p_event) {

	//check accelerators

	if (p_event.type==InputEvent::KEY && p_event.key.pressed) {

		if (!get_parent() || !is_visible() || is_disabled())
			return;

		uint32_t code=p_event.key.scancode;
		if (code==0)
			code=p_event.key.unicode;

		if (p_event.key.mod.control)
			code|=KEY_MASK_CTRL;
		if (p_event.key.mod.alt)
			code|=KEY_MASK_ALT;
		if (p_event.key.mod.meta)
			code|=KEY_MASK_META;
		if (p_event.key.mod.shift)
			code|=KEY_MASK_SHIFT;


		int item = popup->find_item_by_accelerator(code);
		if (item>=0 && ! popup->is_item_disabled(item))
			popup->activate_item(item);
		/*
		for(int i=0;i<items.size();i++) {


			if (items[i].accel==0)
				continue;

			if (items[i].accel==code) {

				emit_signal("item_pressed",items[i].ID);
			}
		}*/
	}

}


void MenuButton::pressed() {
	
	emit_signal("about_to_show");
	Size2 size=get_size();

	Point2 gp = get_global_pos();
	popup->set_global_pos( gp + Size2( 0, size.height ) );
	popup->set_size( Size2( size.width, 0) );	
	popup->set_parent_rect( Rect2(Point2(gp-popup->get_global_pos()),get_size()));
	popup->popup();
	popup->call_deferred("grab_click_focus");
	popup->set_invalidate_click_until_motion();
	
}

void MenuButton::_input_event(InputEvent p_event) {

	/*if (p_event.type==InputEvent::MOUSE_BUTTON && p_event.mouse_button.button_index==BUTTON_LEFT) {
		clicked=p_event.mouse_button.pressed;
	}
	if (clicked && p_event.type==InputEvent::MOUSE_MOTION && popup->is_visible()) {

		Point2 gt = Point2(p_event.mouse_motion.x,p_event.mouse_motion.y);
		gt = get_global_transform().xform(gt);
		Point2 lt = popup->get_transform().affine_inverse().xform(gt);
		if (popup->has_point(lt)) {
			//print_line("HAS POINT!!!");
			popup->call_deferred("grab_click_focus");
		}

	}*/

	BaseButton::_input_event(p_event);
}

PopupMenu *MenuButton::get_popup() {
	
	return popup;
}

Array MenuButton::_get_items() const {

	return popup->get("items");
}
void MenuButton::_set_items(const Array& p_items) {

	popup->set("items",p_items);
}

void MenuButton::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_popup"),&MenuButton::get_popup);		
	ObjectTypeDB::bind_method(_MD("_unhandled_key_input"),&MenuButton::_unhandled_key_input);
	ObjectTypeDB::bind_method(_MD("_set_items"),&MenuButton::_set_items);
	ObjectTypeDB::bind_method(_MD("_get_items"),&MenuButton::_get_items);

	ADD_PROPERTY( PropertyInfo(Variant::ARRAY,"items",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR), _SCS("_set_items"),_SCS("_get_items") );

	ADD_SIGNAL( MethodInfo("about_to_show") );
}
MenuButton::MenuButton() {
	

	set_flat(true);
	set_focus_mode(FOCUS_NONE);
	popup = memnew( PopupMenu );
	popup->hide();
	add_child(popup);
	popup->set_as_toplevel(true);
	set_process_unhandled_key_input(true);
	set_click_on_press(true);
}


MenuButton::~MenuButton() {

}


