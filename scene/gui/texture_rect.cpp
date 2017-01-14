/*************************************************************************/
/*  texture_rect.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "texture_rect.h"
#include "servers/visual_server.h"

void TextureRect::_notification(int p_what) {

	if (p_what==NOTIFICATION_DRAW) {

		if (texture.is_null())
			return;


		switch(stretch_mode) {
			case STRETCH_SCALE_ON_EXPAND: {
				Size2 s=expand?get_size():texture->get_size();
				draw_texture_rect(texture,Rect2(Point2(),s),false);
			} break;
			case STRETCH_SCALE: {
				draw_texture_rect(texture,Rect2(Point2(),get_size()),false);
			} break;
			case STRETCH_TILE: {
				draw_texture_rect(texture,Rect2(Point2(),get_size()),true);
			} break;
			case STRETCH_KEEP: {
				draw_texture_rect(texture,Rect2(Point2(),texture->get_size()),false);

			} break;
			case STRETCH_KEEP_CENTERED: {

				Vector2 ofs = (get_size() - texture->get_size())/2;
				draw_texture_rect(texture,Rect2(ofs,texture->get_size()),false);
			} break;
			case STRETCH_KEEP_ASPECT_CENTERED:
			case STRETCH_KEEP_ASPECT: {

				Size2 size=get_size();
				int tex_width = texture->get_width() * size.height / texture ->get_height();
				int tex_height = size.height;

				if (tex_width>size.width) {
					tex_width=size.width;
					tex_height=texture->get_height() * tex_width / texture->get_width();
				}

				int ofs_x = 0;
				int ofs_y = 0;

				if (stretch_mode==STRETCH_KEEP_ASPECT_CENTERED) {
					ofs_x+=(size.width - tex_width)/2;
					ofs_y+=(size.height - tex_height)/2;
				}

				draw_texture_rect(texture,Rect2(ofs_x,ofs_y,tex_width,tex_height));
			} break;

		}

	}
}

Size2 TextureRect::get_minimum_size() const {

	if (!expand && !texture.is_null())
		return texture->get_size();
	else
		return Size2();
}
void TextureRect::_bind_methods() {


	ClassDB::bind_method(_MD("set_texture","texture"), & TextureRect::set_texture );
	ClassDB::bind_method(_MD("get_texture"), & TextureRect::get_texture );
	ClassDB::bind_method(_MD("set_expand","enable"), & TextureRect::set_expand );
	ClassDB::bind_method(_MD("has_expand"), & TextureRect::has_expand );
	ClassDB::bind_method(_MD("set_stretch_mode","stretch_mode"), & TextureRect::set_stretch_mode );
	ClassDB::bind_method(_MD("get_stretch_mode"), & TextureRect::get_stretch_mode );

	ADD_PROPERTYNZ( PropertyInfo( Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), _SCS("set_texture"),_SCS("get_texture") );
	ADD_PROPERTYNZ( PropertyInfo( Variant::BOOL, "expand" ), _SCS("set_expand"),_SCS("has_expand") );
	ADD_PROPERTYNO( PropertyInfo( Variant::INT, "stretch_mode",PROPERTY_HINT_ENUM,"Scale On Expand (Compat),Scale,Tile,Keep,Keep Centered,Keep Aspect,Keep Aspect Centered"), _SCS("set_stretch_mode"),_SCS("get_stretch_mode") );

	BIND_CONSTANT( STRETCH_SCALE_ON_EXPAND );
	BIND_CONSTANT( STRETCH_SCALE );
	BIND_CONSTANT( STRETCH_TILE );
	BIND_CONSTANT( STRETCH_KEEP );
	BIND_CONSTANT( STRETCH_KEEP_CENTERED );
	BIND_CONSTANT( STRETCH_KEEP_ASPECT );
	BIND_CONSTANT( STRETCH_KEEP_ASPECT_CENTERED );

}


void TextureRect::set_texture(const Ref<Texture>& p_tex) {

	texture=p_tex;
	update();
	/*
	if (texture.is_valid())
		texture->set_flags(texture->get_flags()&(~Texture::FLAG_REPEAT)); //remove repeat from texture, it looks bad in sprites
	*/
	minimum_size_changed();
}

Ref<Texture> TextureRect::get_texture() const {

	return texture;
}


void TextureRect::set_expand(bool p_expand) {

	expand=p_expand;
	update();
	minimum_size_changed();
}
bool TextureRect::has_expand() const {

	return expand;
}

void TextureRect::set_stretch_mode(StretchMode p_mode) {

	stretch_mode=p_mode;
	update();
}

TextureRect::StretchMode TextureRect::get_stretch_mode() const {

	return stretch_mode;
}

TextureRect::TextureRect() {


	expand=false;
	set_mouse_filter(MOUSE_FILTER_IGNORE);
	stretch_mode=STRETCH_SCALE_ON_EXPAND;
}


TextureRect::~TextureRect()
{
}


