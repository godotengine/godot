/*************************************************************************/
/*  animated_sprite.cpp                                                  */
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
#include "animated_sprite.h"
#include "scene/scene_string_names.h"
void AnimatedSprite::edit_set_pivot(const Point2& p_pivot) {

	set_offset(p_pivot);
}

Point2 AnimatedSprite::edit_get_pivot() const {

	return get_offset();
}
bool AnimatedSprite::edit_has_pivot() const {

	return true;
}


void SpriteFrames::add_frame(const Ref<Texture>& p_frame,int p_at_pos) {

	if (p_at_pos>=0 && p_at_pos<frames.size())
		frames.insert(p_at_pos,p_frame);
	else
		frames.push_back(p_frame);

	emit_changed();
}

int SpriteFrames::get_frame_count() const {

	return frames.size();
}

void SpriteFrames::remove_frame(int p_idx) {

	frames.remove(p_idx);
	emit_changed();
}
void SpriteFrames::clear() {

	frames.clear();
	emit_changed();
}


Array SpriteFrames::_get_frames() const {

	Array arr;
	arr.resize(frames.size());
	for(int i=0;i<frames.size();i++)
		arr[i]=frames[i];

	return arr;
}

void SpriteFrames::_set_frames(const Array& p_frames) {

	frames.resize(p_frames.size());
	for(int i=0;i<frames.size();i++)
		frames[i]=p_frames[i];

}


void SpriteFrames::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("add_frame","frame","atpos"),&SpriteFrames::add_frame,DEFVAL(-1));
	ObjectTypeDB::bind_method(_MD("get_frame_count"),&SpriteFrames::get_frame_count);
	ObjectTypeDB::bind_method(_MD("get_frame","idx"),&SpriteFrames::get_frame);
	ObjectTypeDB::bind_method(_MD("set_frame","idx","txt"),&SpriteFrames::set_frame);
	ObjectTypeDB::bind_method(_MD("remove_frame","idx"),&SpriteFrames::remove_frame);
	ObjectTypeDB::bind_method(_MD("clear"),&SpriteFrames::clear);

	ObjectTypeDB::bind_method(_MD("_set_frames"),&SpriteFrames::_set_frames);
	ObjectTypeDB::bind_method(_MD("_get_frames"),&SpriteFrames::_get_frames);

	ADD_PROPERTYNZ( PropertyInfo(Variant::ARRAY,"frames",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR),_SCS("_set_frames"),_SCS("_get_frames"));

}




SpriteFrames::SpriteFrames() {


}







//////////////////////////


void AnimatedSprite::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_DRAW: {

			if (frames.is_null())
				return;

			if (frame<0 || frame>=frames->get_frame_count())
				return;

			Ref<Texture> texture = frames->get_frame(frame);
			if (texture.is_null())
				return;

			//print_line("DECIDED TO DRAW");

			RID ci = get_canvas_item();

			/*
			texture->draw(ci,Point2());
			break;
			*/

			Size2i s;
			s = texture->get_size();
			Point2i ofs=offset;
			if (centered)
				ofs-=s/2;

			Rect2i dst_rect(ofs,s);

			if (hflip)
				dst_rect.size.x=-dst_rect.size.x;
			if (vflip)
				dst_rect.size.y=-dst_rect.size.y;

			texture->draw_rect(ci,dst_rect,false,modulate);
//			VisualServer::get_singleton()->canvas_item_add_texture_rect_region(ci,dst_rect,texture->get_rid(),src_rect,modulate);

		} break;
	}

}

void AnimatedSprite::set_sprite_frames(const Ref<SpriteFrames> &p_frames) {

	if (frames.is_valid())
		frames->disconnect("changed",this,"_res_changed");
	frames=p_frames;
	if (frames.is_valid())
		frames->connect("changed",this,"_res_changed");

	if (!frames.is_valid()) {
		frame=0;

	} else {
		set_frame(frame);
	}
	update();

}

Ref<SpriteFrames> AnimatedSprite::get_sprite_frames() const {

	return frames;
}

void AnimatedSprite::set_frame(int p_frame) {

	if (!frames.is_valid()) {
		return;
	}
	if (p_frame>=frames->get_frame_count())
		p_frame=frames->get_frame_count()-1;
	if (p_frame<0)
		p_frame=0;

	if (frame==p_frame)
		return;

	frame=p_frame;
	update();
	_change_notify("frame");
	emit_signal(SceneStringNames::get_singleton()->frame_changed);

}
int AnimatedSprite::get_frame() const {

	return frame;
}


void AnimatedSprite::set_centered(bool p_center) {

	centered=p_center;
	update();
	item_rect_changed();
}

bool AnimatedSprite::is_centered() const {

	return centered;
}

void AnimatedSprite::set_offset(const Point2& p_offset) {

	offset=p_offset;
	update();
	item_rect_changed();
	_change_notify("offset");
}
Point2 AnimatedSprite::get_offset() const {

	return offset;
}

void AnimatedSprite::set_flip_h(bool p_flip) {

	hflip=p_flip;
	update();
}
bool AnimatedSprite::is_flipped_h() const {

	return hflip;
}

void AnimatedSprite::set_flip_v(bool p_flip) {

	vflip=p_flip;
	update();
}
bool AnimatedSprite::is_flipped_v() const {

	return vflip;
}


void AnimatedSprite::set_modulate(const Color& p_color) {

	modulate=p_color;
	update();
}

Color AnimatedSprite::get_modulate() const{

	return modulate;
}


Rect2 AnimatedSprite::get_item_rect() const {

	if (!frames.is_valid() || !frames->get_frame_count() || frame<0 || frame>=frames->get_frame_count()) {
		return Node2D::get_item_rect();
	}

	Ref<Texture> t = frames->get_frame(frame);
	if (t.is_null())
		return Node2D::get_item_rect();
	Size2i s = t->get_size();

	Point2i ofs=offset;
	if (centered)
		ofs-=s/2;

	if (s==Size2(0,0))
		s=Size2(1,1);

	return Rect2(ofs,s);
}

void AnimatedSprite::_res_changed() {

	set_frame(frame);
	update();
}

void AnimatedSprite::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_sprite_frames","sprite_frames:SpriteFrames"),&AnimatedSprite::set_sprite_frames);
	ObjectTypeDB::bind_method(_MD("get_sprite_frames:SpriteFrames"),&AnimatedSprite::get_sprite_frames);

	ObjectTypeDB::bind_method(_MD("set_centered","centered"),&AnimatedSprite::set_centered);
	ObjectTypeDB::bind_method(_MD("is_centered"),&AnimatedSprite::is_centered);

	ObjectTypeDB::bind_method(_MD("set_offset","offset"),&AnimatedSprite::set_offset);
	ObjectTypeDB::bind_method(_MD("get_offset"),&AnimatedSprite::get_offset);

	ObjectTypeDB::bind_method(_MD("set_flip_h","flip_h"),&AnimatedSprite::set_flip_h);
	ObjectTypeDB::bind_method(_MD("is_flipped_h"),&AnimatedSprite::is_flipped_h);

	ObjectTypeDB::bind_method(_MD("set_flip_v","flip_v"),&AnimatedSprite::set_flip_v);
	ObjectTypeDB::bind_method(_MD("is_flipped_v"),&AnimatedSprite::is_flipped_v);

	ObjectTypeDB::bind_method(_MD("set_frame","frame"),&AnimatedSprite::set_frame);
	ObjectTypeDB::bind_method(_MD("get_frame"),&AnimatedSprite::get_frame);

	ObjectTypeDB::bind_method(_MD("set_modulate","modulate"),&AnimatedSprite::set_modulate);
	ObjectTypeDB::bind_method(_MD("get_modulate"),&AnimatedSprite::get_modulate);


	ObjectTypeDB::bind_method(_MD("_res_changed"),&AnimatedSprite::_res_changed);

	ADD_SIGNAL(MethodInfo("frame_changed"));

	ADD_PROPERTYNZ( PropertyInfo( Variant::OBJECT, "frames",PROPERTY_HINT_RESOURCE_TYPE,"SpriteFrames"), _SCS("set_sprite_frames"),_SCS("get_sprite_frames"));
	ADD_PROPERTYNZ( PropertyInfo( Variant::INT, "frame"), _SCS("set_frame"),_SCS("get_frame"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "centered"), _SCS("set_centered"),_SCS("is_centered"));
	ADD_PROPERTYNZ( PropertyInfo( Variant::VECTOR2, "offset"), _SCS("set_offset"),_SCS("get_offset"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "flip_h"), _SCS("set_flip_h"),_SCS("is_flipped_h"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "flip_v"), _SCS("set_flip_v"),_SCS("is_flipped_v"));
	ADD_PROPERTY( PropertyInfo( Variant::COLOR, "modulate"), _SCS("set_modulate"),_SCS("get_modulate"));

}

AnimatedSprite::AnimatedSprite() {

	centered=true;
	hflip=false;
	vflip=false;

	frame=0;


	modulate=Color(1,1,1,1);


}


