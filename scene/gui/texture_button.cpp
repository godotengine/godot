/*************************************************************************/
/*  texture_button.cpp                                                   */
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
#include "texture_button.h"


Size2 TextureButton::get_minimum_size() const {

	if (normal.is_null()) {
		if (pressed.is_null()) {
			if (hover.is_null())
				if (click_mask.is_null())
					return Size2();
				else
					return click_mask->get_size();
			else
				return hover->get_size();
		} else
			return pressed->get_size();

	} else
		return normal->get_size();
}


bool TextureButton::has_point(const Point2& p_point) const {

	if (click_mask.is_valid()) {

		Point2i p =p_point;
		if (p.x<0 || p.x>=click_mask->get_size().width || p.y<0 || p.y>=click_mask->get_size().height)
			return false;

		return click_mask->get_bit(p);
	}

	return Control::has_point(p_point);
}

void TextureButton::_notification(int p_what) {

	switch( p_what ) {

		case NOTIFICATION_DRAW: {
			RID canvas_item = get_canvas_item();
			DrawMode draw_mode = get_draw_mode();
//			if (normal.is_null())
//				break;
			switch (draw_mode) {
				case DRAW_NORMAL: {

					if (normal.is_valid())
						normal->draw(canvas_item,Point2());
				} break;
				case DRAW_PRESSED: {

					if (pressed.is_null()) {
						if (hover.is_null()) {
							if (normal.is_valid())
								normal->draw(canvas_item,Point2());
						} else
							hover->draw(canvas_item,Point2());

					} else
						pressed->draw(canvas_item,Point2());
				} break;
				case DRAW_HOVER: {

					if (hover.is_null()) {
						if (pressed.is_valid() && is_pressed())
							pressed->draw(canvas_item,Point2());
						else if (normal.is_valid())
							normal->draw(canvas_item,Point2());
					} else
						hover->draw(canvas_item,Point2());
				} break;
				case DRAW_DISABLED: {

					if (disabled.is_null()) {
						if (normal.is_valid())
							normal->draw(canvas_item,Point2());
					} else
						disabled->draw(canvas_item,Point2());
				} break;
			}
			if (has_focus() && focused.is_valid()) {

				focused->draw(canvas_item, Point2());
			};

		} break;
	}
}

void TextureButton::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_normal_texture","texture:Texture"),&TextureButton::set_normal_texture);
	ObjectTypeDB::bind_method(_MD("set_pressed_texture","texture:Texture"),&TextureButton::set_pressed_texture);
	ObjectTypeDB::bind_method(_MD("set_hover_texture","texture:Texture"),&TextureButton::set_hover_texture);
	ObjectTypeDB::bind_method(_MD("set_disabled_texture","texture:Texture"),&TextureButton::set_disabled_texture);
	ObjectTypeDB::bind_method(_MD("set_focused_texture","texture:Texture"),&TextureButton::set_focused_texture);
	ObjectTypeDB::bind_method(_MD("set_click_mask","mask:BitMap"),&TextureButton::set_click_mask);

	ObjectTypeDB::bind_method(_MD("get_normal_texture:Texture"),&TextureButton::get_normal_texture);
	ObjectTypeDB::bind_method(_MD("get_pressed_texture:Texture"),&TextureButton::get_pressed_texture);
	ObjectTypeDB::bind_method(_MD("get_hover_texture:Texture"),&TextureButton::get_hover_texture);
	ObjectTypeDB::bind_method(_MD("get_disabled_texture:Texture"),&TextureButton::get_disabled_texture);
	ObjectTypeDB::bind_method(_MD("get_focused_texture:Texture"),&TextureButton::get_focused_texture);
	ObjectTypeDB::bind_method(_MD("get_click_mask:BitMap"),&TextureButton::get_click_mask);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"textures/normal",PROPERTY_HINT_RESOURCE_TYPE,"Texture"), _SCS("set_normal_texture"), _SCS("get_normal_texture"));
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"textures/pressed",PROPERTY_HINT_RESOURCE_TYPE,"Texture"), _SCS("set_pressed_texture"), _SCS("get_pressed_texture"));
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"textures/hover",PROPERTY_HINT_RESOURCE_TYPE,"Texture"), _SCS("set_hover_texture"), _SCS("get_hover_texture"));
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"textures/disabled",PROPERTY_HINT_RESOURCE_TYPE,"Texture"), _SCS("set_disabled_texture"), _SCS("get_disabled_texture"));
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"textures/focused",PROPERTY_HINT_RESOURCE_TYPE,"Texture"), _SCS("set_focused_texture"), _SCS("get_focused_texture"));
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"textures/click_mask",PROPERTY_HINT_RESOURCE_TYPE,"BitMap"), _SCS("set_click_mask"), _SCS("get_click_mask")) ;

}


void TextureButton::set_normal_texture(const Ref<Texture>& p_normal) {

	normal=p_normal;
	update();
	minimum_size_changed();

}

void TextureButton::set_pressed_texture(const Ref<Texture>& p_pressed) {

	pressed=p_pressed;
	update();

}
void TextureButton::set_hover_texture(const Ref<Texture>& p_hover) {

	hover=p_hover;
	update();

}
void TextureButton::set_disabled_texture(const Ref<Texture>& p_disabled) {

	disabled=p_disabled;
	update();

}
void TextureButton::set_click_mask(const Ref<BitMap>& p_click_mask) {

	click_mask=p_click_mask;
	update();
}

Ref<Texture> TextureButton::get_normal_texture() const {

	return normal;
}
Ref<Texture> TextureButton::get_pressed_texture() const {

	return pressed;
}
Ref<Texture> TextureButton::get_hover_texture() const {

	return hover;
}
Ref<Texture> TextureButton::get_disabled_texture() const {

	return disabled;
}
Ref<BitMap> TextureButton::get_click_mask() const {

	return click_mask;
}

Ref<Texture> TextureButton::get_focused_texture() const {

	return focused;
};

void TextureButton::set_focused_texture(const Ref<Texture>& p_focused) {

	focused = p_focused;
};


TextureButton::TextureButton() {
}
