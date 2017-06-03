/*************************************************************************/
/*  sprite.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "os/os.h"
#include "scene/main/viewport.h"
#include "scene/scene_string_names.h"

void Sprite::edit_set_pivot(const Point2 &p_pivot) {

	set_offset(p_pivot);
}

Point2 Sprite::edit_get_pivot() const {

	return get_offset();
}
bool Sprite::edit_has_pivot() const {

	return true;
}

void Sprite::_notification(int p_what) {

	switch (p_what) {

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

				s = region_rect.size;
				src_rect = region_rect;
			} else {
				s = Size2(texture->get_size());
				s = s / Size2(hframes, vframes);

				src_rect.size = s;
				src_rect.pos.x += float(frame % hframes) * s.x;
				src_rect.pos.y += float(frame / hframes) * s.y;
			}

			Point2 ofs = offset;
			if (centered)
				ofs -= s / 2;
			if (Engine::get_singleton()->get_use_pixel_snap()) {
				ofs = ofs.floor();
			}

			Rect2 dst_rect(ofs, s);

			if (hflip)
				dst_rect.size.x = -dst_rect.size.x;
			if (vflip)
				dst_rect.size.y = -dst_rect.size.y;

			texture->draw_rect_region(ci, dst_rect, src_rect);

		} break;
	}
}

void Sprite::set_texture(const Ref<Texture> &p_texture) {

	if (p_texture == texture)
		return;
#ifdef DEBUG_ENABLED
	if (texture.is_valid()) {
		texture->disconnect(CoreStringNames::get_singleton()->changed, this, SceneStringNames::get_singleton()->update);
	}
#endif
	texture = p_texture;
#ifdef DEBUG_ENABLED
	if (texture.is_valid()) {
		texture->set_flags(texture->get_flags()); //remove repeat from texture, it looks bad in sprites
		texture->connect(CoreStringNames::get_singleton()->changed, this, SceneStringNames::get_singleton()->update);
	}
#endif
	update();
	emit_signal("texture_changed");
	item_rect_changed();
}

Ref<Texture> Sprite::get_texture() const {

	return texture;
}

void Sprite::set_centered(bool p_center) {

	centered = p_center;
	update();
	item_rect_changed();
}

bool Sprite::is_centered() const {

	return centered;
}

void Sprite::set_offset(const Point2 &p_offset) {

	offset = p_offset;
	update();
	item_rect_changed();
	_change_notify("offset");
}
Point2 Sprite::get_offset() const {

	return offset;
}

void Sprite::set_flip_h(bool p_flip) {

	hflip = p_flip;
	update();
}
bool Sprite::is_flipped_h() const {

	return hflip;
}

void Sprite::set_flip_v(bool p_flip) {

	vflip = p_flip;
	update();
}
bool Sprite::is_flipped_v() const {

	return vflip;
}

void Sprite::set_region(bool p_region) {

	if (p_region == region)
		return;

	region = p_region;
	update();
}

bool Sprite::is_region() const {

	return region;
}

void Sprite::set_region_rect(const Rect2 &p_region_rect) {

	if (region_rect == p_region_rect)
		return;

	region_rect = p_region_rect;

	if (region)
		item_rect_changed();

	_change_notify("region_rect");
}

Rect2 Sprite::get_region_rect() const {

	return region_rect;
}

void Sprite::set_frame(int p_frame) {

	ERR_FAIL_INDEX(p_frame, vframes * hframes);

	if (frame != p_frame)
		item_rect_changed();

	frame = p_frame;

	_change_notify("frame");
	emit_signal(SceneStringNames::get_singleton()->frame_changed);
}

int Sprite::get_frame() const {

	return frame;
}

void Sprite::set_vframes(int p_amount) {

	ERR_FAIL_COND(p_amount < 1);
	vframes = p_amount;
	update();
	item_rect_changed();
	_change_notify("frame");
}
int Sprite::get_vframes() const {

	return vframes;
}

void Sprite::set_hframes(int p_amount) {

	ERR_FAIL_COND(p_amount < 1);
	hframes = p_amount;
	update();
	item_rect_changed();
	_change_notify("frame");
}
int Sprite::get_hframes() const {

	return hframes;
}

Rect2 Sprite::get_item_rect() const {

	if (texture.is_null())
		return Rect2(0, 0, 1, 1);
	/*
	if (texture.is_null())
		return CanvasItem::get_item_rect();
	*/

	Size2i s;

	if (region) {

		s = region_rect.size;
	} else {
		s = texture->get_size();
		s = s / Point2(hframes, vframes);
	}

	Point2 ofs = offset;
	if (centered)
		ofs -= s / 2;

	if (s == Size2(0, 0))
		s = Size2(1, 1);

	return Rect2(ofs, s);
}

void Sprite::_validate_property(PropertyInfo &property) const {

	if (property.name == "frame") {

		property.hint = PROPERTY_HINT_SPRITE_FRAME;

		property.hint_string = "0," + itos(vframes * hframes - 1) + ",1";
	}
}

void Sprite::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_texture", "texture:Texture"), &Sprite::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture:Texture"), &Sprite::get_texture);

	ClassDB::bind_method(D_METHOD("set_centered", "centered"), &Sprite::set_centered);
	ClassDB::bind_method(D_METHOD("is_centered"), &Sprite::is_centered);

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &Sprite::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &Sprite::get_offset);

	ClassDB::bind_method(D_METHOD("set_flip_h", "flip_h"), &Sprite::set_flip_h);
	ClassDB::bind_method(D_METHOD("is_flipped_h"), &Sprite::is_flipped_h);

	ClassDB::bind_method(D_METHOD("set_flip_v", "flip_v"), &Sprite::set_flip_v);
	ClassDB::bind_method(D_METHOD("is_flipped_v"), &Sprite::is_flipped_v);

	ClassDB::bind_method(D_METHOD("set_region", "enabled"), &Sprite::set_region);
	ClassDB::bind_method(D_METHOD("is_region"), &Sprite::is_region);

	ClassDB::bind_method(D_METHOD("set_region_rect", "rect"), &Sprite::set_region_rect);
	ClassDB::bind_method(D_METHOD("get_region_rect"), &Sprite::get_region_rect);

	ClassDB::bind_method(D_METHOD("set_frame", "frame"), &Sprite::set_frame);
	ClassDB::bind_method(D_METHOD("get_frame"), &Sprite::get_frame);

	ClassDB::bind_method(D_METHOD("set_vframes", "vframes"), &Sprite::set_vframes);
	ClassDB::bind_method(D_METHOD("get_vframes"), &Sprite::get_vframes);

	ClassDB::bind_method(D_METHOD("set_hframes", "hframes"), &Sprite::set_hframes);
	ClassDB::bind_method(D_METHOD("get_hframes"), &Sprite::get_hframes);

	ADD_SIGNAL(MethodInfo("frame_changed"));
	ADD_SIGNAL(MethodInfo("texture_changed"));

	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_PROPERTYNO(PropertyInfo(Variant::BOOL, "centered"), "set_centered", "is_centered");
	ADD_PROPERTYNZ(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "flip_h"), "set_flip_h", "is_flipped_h");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "flip_v"), "set_flip_v", "is_flipped_v");
	ADD_PROPERTYNO(PropertyInfo(Variant::INT, "vframes", PROPERTY_HINT_RANGE, "1,16384,1"), "set_vframes", "get_vframes");
	ADD_PROPERTYNO(PropertyInfo(Variant::INT, "hframes", PROPERTY_HINT_RANGE, "1,16384,1"), "set_hframes", "get_hframes");
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "frame", PROPERTY_HINT_SPRITE_FRAME), "set_frame", "get_frame");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "region"), "set_region", "is_region");
	ADD_PROPERTYNZ(PropertyInfo(Variant::RECT2, "region_rect"), "set_region_rect", "get_region_rect");
}

Sprite::Sprite() {

	centered = true;
	hflip = false;
	vflip = false;
	region = false;

	frame = 0;

	vframes = 1;
	hframes = 1;
}

//////////////////////////// VPSPRITE
///
///
///

#if 0
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

			Point2 ofs=offset;
			if (centered)
				ofs-=s/2;

			if (OS::get_singleton()->get_use_pixel_snap()) {
				ofs=ofs.floor();
			}
			Rect2 dst_rect(ofs,s);
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
	/*
	if (texture.is_null())
		return CanvasItem::get_item_rect();
	*/

	Size2i s;

	s = texture->get_size();
	Point2 ofs=offset;
	if (centered)
		ofs-=s/2;

	if (s==Size2(0,0))
		s=Size2(1,1);

	return Rect2(ofs,s);
}

String ViewportSprite::get_configuration_warning() const {

	if (!has_node(viewport_path) || !get_node(viewport_path) || !get_node(viewport_path)->cast_to<Viewport>()) {
		return TTR("Path property must point to a valid Viewport node to work. Such Viewport must be set to 'render target' mode.");
	} else {

		Node *n = get_node(viewport_path);
		if (n) {
			Viewport *vp = n->cast_to<Viewport>();
			if (!vp->is_set_as_render_target()) {

				return TTR("The Viewport set in the path property must be set as 'render target' in order for this sprite to work.");
			}
		}
	}

	return String();

}

void ViewportSprite::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_viewport_path","path"),&ViewportSprite::set_viewport_path);
	ClassDB::bind_method(D_METHOD("get_viewport_path"),&ViewportSprite::get_viewport_path);

	ClassDB::bind_method(D_METHOD("set_centered","centered"),&ViewportSprite::set_centered);
	ClassDB::bind_method(D_METHOD("is_centered"),&ViewportSprite::is_centered);

	ClassDB::bind_method(D_METHOD("set_offset","offset"),&ViewportSprite::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"),&ViewportSprite::get_offset);

	ClassDB::bind_method(D_METHOD("set_modulate","modulate"),&ViewportSprite::set_modulate);
	ClassDB::bind_method(D_METHOD("get_modulate"),&ViewportSprite::get_modulate);

	ADD_PROPERTYNZ( PropertyInfo( Variant::NODE_PATH, "viewport"), "set_viewport_path","get_viewport_path");
	ADD_PROPERTYNO( PropertyInfo( Variant::BOOL, "centered"), "set_centered","is_centered");
	ADD_PROPERTYNZ( PropertyInfo( Variant::VECTOR2, "offset"), "set_offset","get_offset");
	ADD_PROPERTYNO( PropertyInfo( Variant::COLOR, "modulate"), "set_modulate","get_modulate");

}

ViewportSprite::ViewportSprite() {

	centered=true;
	modulate=Color(1,1,1,1);
}
#endif
