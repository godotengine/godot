/*************************************************************************/
/*  sprite.cpp                                                           */
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
#include "sprite.h"
#include "core/core_string_names.h"
#include "scene/scene_string_names.h"
#include "scene/main/viewport.h"

void Sprite::edit_set_pivot(const Point2& p_pivot) {

	set_offset(p_pivot);

}

Point2 Sprite::edit_get_pivot() const {

	return get_offset();
}
bool Sprite::edit_has_pivot() const {

	return true;
}

void Sprite::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_DRAW: {

			if (texture.is_null())
				return;




			RID ci = get_canvas_item();

			/*
			texture->draw(ci,Point2());
			break;
			*/

			Size2 s;
			Rect2 src_rect;

			if (region) {

				s=region_rect.size;
				src_rect=region_rect;
			} else {
				s = Size2(texture->get_size());
				s=s/Size2(hframes,vframes);

				src_rect.size=s;
				src_rect.pos.x+=float(frame%hframes)*s.x;
				src_rect.pos.y+=float(frame/hframes)*s.y;

			}

			Point2i ofs=offset;
			if (centered)
				ofs-=s/2;

			Rect2 dst_rect(ofs,s);

			if (hflip)
				dst_rect.size.x=-dst_rect.size.x;
			if (vflip)
				dst_rect.size.y=-dst_rect.size.y;

			texture->draw_rect_region(ci,dst_rect,src_rect,modulate);

		} break;
	}
}

void Sprite::set_texture(const Ref<Texture>& p_texture) {

	if (p_texture==texture)
		return;
	if (texture.is_valid()) {
		texture->disconnect(CoreStringNames::get_singleton()->changed,this,SceneStringNames::get_singleton()->update);
	}
	texture=p_texture;
	if (texture.is_valid()) {
		texture->set_flags(texture->get_flags()); //remove repeat from texture, it looks bad in sprites
		texture->connect(CoreStringNames::get_singleton()->changed,this,SceneStringNames::get_singleton()->update);
	}
	update();
	item_rect_changed();
}

Ref<Texture> Sprite::get_texture() const {

	return texture;
}

void Sprite::set_centered(bool p_center) {

	centered=p_center;
	update();
	item_rect_changed();
}

bool Sprite::is_centered() const {

	return centered;
}

void Sprite::set_offset(const Point2& p_offset) {

	offset=p_offset;
	update();
	item_rect_changed();
	_change_notify("offset");
}
Point2 Sprite::get_offset() const {

	return offset;
}

void Sprite::set_flip_h(bool p_flip) {

	hflip=p_flip;
	update();
}
bool Sprite::is_flipped_h() const {

	return hflip;
}

void Sprite::set_flip_v(bool p_flip) {

	vflip=p_flip;
	update();
}
bool Sprite::is_flipped_v() const {

	return vflip;
}

void Sprite::set_region(bool p_region) {

	if (p_region==region)
		return;

	region=p_region;
	update();
}

bool Sprite::is_region() const{

	return region;
}

void Sprite::set_region_rect(const Rect2& p_region_rect) {

	bool changed=region_rect!=p_region_rect;
	region_rect=p_region_rect;
	if (region && changed) {
		update();
		item_rect_changed();
	}
}

Rect2 Sprite::get_region_rect() const {

	return region_rect;
}

void Sprite::set_frame(int p_frame) {

	ERR_FAIL_INDEX(p_frame,vframes*hframes);

	if (frame != p_frame)
		item_rect_changed();

	frame=p_frame;

	emit_signal(SceneStringNames::get_singleton()->frame_changed);
}

int Sprite::get_frame() const {

	return frame;
}

void Sprite::set_vframes(int p_amount) {

	ERR_FAIL_COND(p_amount<1);
	vframes=p_amount;
	update();
	item_rect_changed();
	_change_notify("frame");
}
int Sprite::get_vframes() const {

	return vframes;
}

void Sprite::set_hframes(int p_amount) {

	ERR_FAIL_COND(p_amount<1);
	hframes=p_amount;
	update();
	item_rect_changed();
	_change_notify("frame");
}
int Sprite::get_hframes() const {

	return hframes;
}

void Sprite::set_modulate(const Color& p_color) {

	modulate=p_color;
	update();
}

Color Sprite::get_modulate() const{

	return modulate;
}


Rect2 Sprite::get_item_rect() const {

	if (texture.is_null())
		return Rect2(0,0,1,1);
	//if (texture.is_null())
	//	return CanvasItem::get_item_rect();

	Size2i s;

	if (region) {

		s=region_rect.size;
	} else {
		s = texture->get_size();
		s=s/Point2(hframes,vframes);
	}

	Point2i ofs=offset;
	if (centered)
		ofs-=s/2;

	if (s==Size2(0,0))
		s=Size2(1,1);

	return Rect2(ofs,s);
}


void Sprite::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_texture","texture:Texture"),&Sprite::set_texture);
	ObjectTypeDB::bind_method(_MD("get_texture:Texture"),&Sprite::get_texture);

	ObjectTypeDB::bind_method(_MD("set_centered","centered"),&Sprite::set_centered);
	ObjectTypeDB::bind_method(_MD("is_centered"),&Sprite::is_centered);

	ObjectTypeDB::bind_method(_MD("set_offset","offset"),&Sprite::set_offset);
	ObjectTypeDB::bind_method(_MD("get_offset"),&Sprite::get_offset);

	ObjectTypeDB::bind_method(_MD("set_flip_h","flip_h"),&Sprite::set_flip_h);
	ObjectTypeDB::bind_method(_MD("is_flipped_h"),&Sprite::is_flipped_h);

	ObjectTypeDB::bind_method(_MD("set_flip_v","flip_v"),&Sprite::set_flip_v);
	ObjectTypeDB::bind_method(_MD("is_flipped_v"),&Sprite::is_flipped_v);

	ObjectTypeDB::bind_method(_MD("set_region","enabled"),&Sprite::set_region);
	ObjectTypeDB::bind_method(_MD("is_region"),&Sprite::is_region);

	ObjectTypeDB::bind_method(_MD("set_region_rect","rect"),&Sprite::set_region_rect);
	ObjectTypeDB::bind_method(_MD("get_region_rect"),&Sprite::get_region_rect);

	ObjectTypeDB::bind_method(_MD("set_frame","frame"),&Sprite::set_frame);
	ObjectTypeDB::bind_method(_MD("get_frame"),&Sprite::get_frame);

	ObjectTypeDB::bind_method(_MD("set_vframes","vframes"),&Sprite::set_vframes);
	ObjectTypeDB::bind_method(_MD("get_vframes"),&Sprite::get_vframes);

	ObjectTypeDB::bind_method(_MD("set_hframes","hframes"),&Sprite::set_hframes);
	ObjectTypeDB::bind_method(_MD("get_hframes"),&Sprite::get_hframes);

	ObjectTypeDB::bind_method(_MD("set_modulate","modulate"),&Sprite::set_modulate);
	ObjectTypeDB::bind_method(_MD("get_modulate"),&Sprite::get_modulate);

	ADD_SIGNAL(MethodInfo("frame_changed"));

	ADD_PROPERTY( PropertyInfo( Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE,"Texture"), _SCS("set_texture"),_SCS("get_texture"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "centered"), _SCS("set_centered"),_SCS("is_centered"));
	ADD_PROPERTY( PropertyInfo( Variant::VECTOR2, "offset"), _SCS("set_offset"),_SCS("get_offset"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "flip_h"), _SCS("set_flip_h"),_SCS("is_flipped_h"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "flip_v"), _SCS("set_flip_v"),_SCS("is_flipped_v"));
	ADD_PROPERTY( PropertyInfo( Variant::INT, "vframes"), _SCS("set_vframes"),_SCS("get_vframes"));
	ADD_PROPERTY( PropertyInfo( Variant::INT, "hframes"), _SCS("set_hframes"),_SCS("get_hframes"));
	ADD_PROPERTY( PropertyInfo( Variant::INT, "frame"), _SCS("set_frame"),_SCS("get_frame"));
	ADD_PROPERTY( PropertyInfo( Variant::COLOR, "modulate"), _SCS("set_modulate"),_SCS("get_modulate"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "region"), _SCS("set_region"),_SCS("is_region"));
	ADD_PROPERTY( PropertyInfo( Variant::RECT2, "region_rect"), _SCS("set_region_rect"),_SCS("get_region_rect"));

}

Sprite::Sprite() {

	centered=true;
	hflip=false;
	vflip=false;
	region=false;

	frame=0;

	vframes=1;
	hframes=1;

	modulate=Color(1,1,1,1);


}




//////////////////////////// VPSPRITE
///
///
///


void ViewportSprite::edit_set_pivot(const Point2& p_pivot) {

	set_offset(p_pivot);
}

Point2 ViewportSprite::edit_get_pivot() const {

	return get_offset();
}
bool ViewportSprite::edit_has_pivot() const {

	return true;
}

void ViewportSprite::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			if (!viewport_path.is_empty()) {

				Node *n = get_node(viewport_path);
				ERR_FAIL_COND(!n);
				Viewport *vp=n->cast_to<Viewport>();
				ERR_FAIL_COND(!vp);

				Ref<RenderTargetTexture> rtt = vp->get_render_target_texture();
				texture=rtt;
				texture->connect("changed",this,"update");
				item_rect_changed();
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {

			if (texture.is_valid()) {

				texture->disconnect("changed",this,"update");
				texture=Ref<Texture>();
			}
		} break;
		case NOTIFICATION_DRAW: {

			if (texture.is_null())
				return;

			RID ci = get_canvas_item();

			/*
			texture->draw(ci,Point2());
			break;
			*/

			Size2i s;
			Rect2i src_rect;

			s = texture->get_size();

			src_rect.size=s;

			Point2i ofs=offset;
			if (centered)
				ofs-=s/2;

			Rect2i dst_rect(ofs,s);
			texture->draw_rect_region(ci,dst_rect,src_rect,modulate);

		} break;
	}
}

void ViewportSprite::set_viewport_path(const NodePath& p_viewport) {

	viewport_path=p_viewport;
	update();
	if (!is_inside_tree())
		return;

	if (texture.is_valid()) {
		texture->disconnect("changed",this,"update");
		texture=Ref<Texture>();
	}

	if (viewport_path.is_empty())
		return;


	Node *n = get_node(viewport_path);
	ERR_FAIL_COND(!n);
	Viewport *vp=n->cast_to<Viewport>();
	ERR_FAIL_COND(!vp);

	Ref<RenderTargetTexture> rtt = vp->get_render_target_texture();
	texture=rtt;

	if (texture.is_valid()) {
		texture->connect("changed",this,"update");
	}

	item_rect_changed();

}

NodePath ViewportSprite::get_viewport_path() const {

	return viewport_path;
}

void ViewportSprite::set_centered(bool p_center) {

	centered=p_center;
	update();
	item_rect_changed();
}

bool ViewportSprite::is_centered() const {

	return centered;
}

void ViewportSprite::set_offset(const Point2& p_offset) {

	offset=p_offset;
	update();
	item_rect_changed();
}
Point2 ViewportSprite::get_offset() const {

	return offset;
}
void ViewportSprite::set_modulate(const Color& p_color) {

	modulate=p_color;
	update();
}

Color ViewportSprite::get_modulate() const{

	return modulate;
}


Rect2 ViewportSprite::get_item_rect() const {

	if (texture.is_null())
		return Rect2(0,0,1,1);
	//if (texture.is_null())
	//	return CanvasItem::get_item_rect();

	Size2i s;

	s = texture->get_size();
	Point2i ofs=offset;
	if (centered)
		ofs-=s/2;

	if (s==Size2(0,0))
		s=Size2(1,1);

	return Rect2(ofs,s);
}


void ViewportSprite::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_viewport_path","path"),&ViewportSprite::set_viewport_path);
	ObjectTypeDB::bind_method(_MD("get_viewport_path"),&ViewportSprite::get_viewport_path);

	ObjectTypeDB::bind_method(_MD("set_centered","centered"),&ViewportSprite::set_centered);
	ObjectTypeDB::bind_method(_MD("is_centered"),&ViewportSprite::is_centered);

	ObjectTypeDB::bind_method(_MD("set_offset","offset"),&ViewportSprite::set_offset);
	ObjectTypeDB::bind_method(_MD("get_offset"),&ViewportSprite::get_offset);

	ObjectTypeDB::bind_method(_MD("set_modulate","modulate"),&ViewportSprite::set_modulate);
	ObjectTypeDB::bind_method(_MD("get_modulate"),&ViewportSprite::get_modulate);

	ADD_PROPERTY( PropertyInfo( Variant::NODE_PATH, "viewport"), _SCS("set_viewport_path"),_SCS("get_viewport_path"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "centered"), _SCS("set_centered"),_SCS("is_centered"));
	ADD_PROPERTY( PropertyInfo( Variant::VECTOR2, "offset"), _SCS("set_offset"),_SCS("get_offset"));
	ADD_PROPERTY( PropertyInfo( Variant::COLOR, "modulate"), _SCS("set_modulate"),_SCS("get_modulate"));

}

ViewportSprite::ViewportSprite() {

	centered=true;
	modulate=Color(1,1,1,1);
}
