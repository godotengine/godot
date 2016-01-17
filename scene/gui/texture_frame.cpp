/*************************************************************************/
/*  texture_frame.cpp                                                    */
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
#include "texture_frame.h"
#include "servers/visual_server.h"

void TextureFrame::_notification(int p_what) {

	if (p_what==NOTIFICATION_DRAW) {

		if (texture.is_null())
			return;


		Size2 s=expand?get_size():texture->get_size();
		RID ci = get_canvas_item();
		draw_texture_rect(texture,Rect2(Point2(),s),false,modulate);

/*
		Vector<Point2> points;
		points.resize(4);
		points[0]=Point2(0,0);
		points[1]=Point2(s.x,0);
		points[2]=Point2(s.x,s.y);
		points[3]=Point2(0,s.y);
		Vector<Point2> uvs;
		uvs.resize(4);
		uvs[0]=Point2(0,0);
		uvs[1]=Point2(1,0);
		uvs[2]=Point2(1,1);
		uvs[3]=Point2(0,1);

		VisualServer::get_singleton()->canvas_item_add_primitive(ci,points,Vector<Color>(),uvs,texture->get_rid());
*/
	}
}

Size2 TextureFrame::get_minimum_size() const {

	if (!expand && !texture.is_null())
		return texture->get_size();
	else
		return Size2();
}
void TextureFrame::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("set_texture","texture"), & TextureFrame::set_texture );
	ObjectTypeDB::bind_method(_MD("get_texture"), & TextureFrame::get_texture );
	ObjectTypeDB::bind_method(_MD("set_modulate","modulate"), & TextureFrame::set_modulate );
	ObjectTypeDB::bind_method(_MD("get_modulate"), & TextureFrame::get_modulate );
	ObjectTypeDB::bind_method(_MD("set_expand","enable"), & TextureFrame::set_expand );
	ObjectTypeDB::bind_method(_MD("has_expand"), & TextureFrame::has_expand );

	ADD_PROPERTYNZ( PropertyInfo( Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), _SCS("set_texture"),_SCS("get_texture") );
	ADD_PROPERTYNO( PropertyInfo( Variant::COLOR, "modulate"), _SCS("set_modulate"),_SCS("get_modulate") );
	ADD_PROPERTYNZ( PropertyInfo( Variant::BOOL, "expand" ), _SCS("set_expand"),_SCS("has_expand") );

}


void TextureFrame::set_texture(const Ref<Texture>& p_tex) {

	texture=p_tex;
	update();
	//if (texture.is_valid())
	//	texture->set_flags(texture->get_flags()&(~Texture::FLAG_REPEAT)); //remove repeat from texture, it looks bad in sprites
	minimum_size_changed();
}

Ref<Texture> TextureFrame::get_texture() const {

	return texture;
}

void TextureFrame::set_modulate(const Color& p_tex) {

	modulate=p_tex;
	update();
}

Color TextureFrame::get_modulate() const{

	return modulate;
}


void TextureFrame::set_expand(bool p_expand) {

	expand=p_expand;
	update();
	minimum_size_changed();
}
bool TextureFrame::has_expand() const {

	return expand;
}

TextureFrame::TextureFrame() {


	expand=false;
	modulate=Color(1,1,1,1);
	set_ignore_mouse(true);
}


TextureFrame::~TextureFrame()
{
}


