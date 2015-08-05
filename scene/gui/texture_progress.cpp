/*************************************************************************/
/*  texture_progress.cpp                                                 */
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
#include "texture_progress.h"


void TextureProgress::set_under_texture(const Ref<Texture>& p_texture) {

	under=p_texture;
	update();
	minimum_size_changed();
}

Ref<Texture> TextureProgress::get_under_texture() const{

	return under;

}

void TextureProgress::set_over_texture(const Ref<Texture>& p_texture) {

	over=p_texture;
	update();
	minimum_size_changed();
}

Ref<Texture> TextureProgress::get_over_texture() const{

	return over;

}

Size2 TextureProgress::get_minimum_size() const {

	if (under.is_valid())
		return under->get_size();
	else if (over.is_valid())
		return over->get_size();
	else if (progress.is_valid())
		return progress->get_size();

	return Size2(1,1);
}

void TextureProgress::set_progress_texture(const Ref<Texture>& p_texture) {

	progress=p_texture;
	update();
	minimum_size_changed();
}

Ref<Texture> TextureProgress::get_progress_texture() const{

	return progress;

}


void TextureProgress::_notification(int p_what){

	switch(p_what) {

		case NOTIFICATION_DRAW: {


			if (under.is_valid())
				draw_texture(under,Point2());
			if (progress.is_valid()) {
				Size2 s = progress->get_size();
				draw_texture_rect_region(progress,Rect2(Point2(),Size2(s.x*get_unit_value(),s.y)),Rect2(Point2(),Size2(s.x*get_unit_value(),s.y)));
			}
			if (over.is_valid())
				draw_texture(over,Point2());

		} break;
	}
}

void TextureProgress::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_under_texture","tex"),&TextureProgress::set_under_texture);
	ObjectTypeDB::bind_method(_MD("get_under_texture"),&TextureProgress::get_under_texture);

	ObjectTypeDB::bind_method(_MD("set_progress_texture","tex"),&TextureProgress::set_progress_texture);
	ObjectTypeDB::bind_method(_MD("get_progress_texture"),&TextureProgress::get_progress_texture);

	ObjectTypeDB::bind_method(_MD("set_over_texture","tex"),&TextureProgress::set_over_texture);
	ObjectTypeDB::bind_method(_MD("get_over_texture"),&TextureProgress::get_over_texture);

	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"texture/under",PROPERTY_HINT_RESOURCE_TYPE,"Texture"),_SCS("set_under_texture"),_SCS("get_under_texture"));
	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"texture/over",PROPERTY_HINT_RESOURCE_TYPE,"Texture"),_SCS("set_over_texture"),_SCS("get_over_texture"));
	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"texture/progress",PROPERTY_HINT_RESOURCE_TYPE,"Texture"),_SCS("set_progress_texture"),_SCS("get_progress_texture"));

}


TextureProgress::TextureProgress()
{
}
