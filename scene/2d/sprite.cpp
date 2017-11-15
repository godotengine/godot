/*************************************************************************/
/*  sprite.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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

void Sprite::_edit_set_pivot(const Point2 &p_pivot) {

	set_offset(p_pivot);
}

Point2 Sprite::_edit_get_pivot() const {

	return get_offset();
}
bool Sprite::_edit_use_pivot() const {

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
			bool filter_clip = false;

			if (region) {

				s = region_rect.size;
				src_rect = region_rect;
				filter_clip = region_filter_clip;
			} else {
				s = Size2(texture->get_size());
				s = s / Size2(hframes, vframes);

				src_rect.size = s;
				src_rect.position.x += float(frame % hframes) * s.x;
				src_rect.position.y += float(frame / hframes) * s.y;
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

			texture->draw_rect_region(ci, dst_rect, src_rect, Color(1, 1, 1), false, normal_map, filter_clip);

		} break;
	}
}

void Sprite::set_texture(const Ref<Texture> &p_texture) {

	if (p_texture == texture)
		return;
	texture = p_texture;
	update();
	emit_signal("texture_changed");
	item_rect_changed();
	_change_notify("texture");
}

void Sprite::set_normal_map(const Ref<Texture> &p_texture) {

	normal_map = p_texture;
	update();
}

Ref<Texture> Sprite::get_normal_map() const {

	return normal_map;
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

void Sprite::set_region_filter_clip(bool p_enable) {
	region_filter_clip = p_enable;
	update();
}

bool Sprite::is_region_filter_clip_enabled() const {
	return region_filter_clip;
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
	_change_notify();
}
int Sprite::get_vframes() const {

	return vframes;
}

void Sprite::set_hframes(int p_amount) {

	ERR_FAIL_COND(p_amount < 1);
	hframes = p_amount;
	update();
	item_rect_changed();
	_change_notify();
}
int Sprite::get_hframes() const {

	return hframes;
}

Rect2 Sprite::_edit_get_rect() const {

	if (texture.is_null())
		return Rect2(0, 0, 1, 1);
	/*
	if (texture.is_null())
		return CanvasItem::_edit_get_rect();
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

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &Sprite::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &Sprite::get_texture);

	ClassDB::bind_method(D_METHOD("set_normal_map", "normal_map"), &Sprite::set_normal_map);
	ClassDB::bind_method(D_METHOD("get_normal_map"), &Sprite::get_normal_map);

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

	ClassDB::bind_method(D_METHOD("set_region_filter_clip", "enabled"), &Sprite::set_region_filter_clip);
	ClassDB::bind_method(D_METHOD("is_region_filter_clip_enabled"), &Sprite::is_region_filter_clip_enabled);

	ClassDB::bind_method(D_METHOD("set_frame", "frame"), &Sprite::set_frame);
	ClassDB::bind_method(D_METHOD("get_frame"), &Sprite::get_frame);

	ClassDB::bind_method(D_METHOD("set_vframes", "vframes"), &Sprite::set_vframes);
	ClassDB::bind_method(D_METHOD("get_vframes"), &Sprite::get_vframes);

	ClassDB::bind_method(D_METHOD("set_hframes", "hframes"), &Sprite::set_hframes);
	ClassDB::bind_method(D_METHOD("get_hframes"), &Sprite::get_hframes);

	ADD_SIGNAL(MethodInfo("frame_changed"));
	ADD_SIGNAL(MethodInfo("texture_changed"));

	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "normal_map", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_normal_map", "get_normal_map");
	ADD_GROUP("Offset", "");
	ADD_PROPERTYNO(PropertyInfo(Variant::BOOL, "centered"), "set_centered", "is_centered");
	ADD_PROPERTYNZ(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "flip_h"), "set_flip_h", "is_flipped_h");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "flip_v"), "set_flip_v", "is_flipped_v");
	ADD_GROUP("Animation", "");
	ADD_PROPERTYNO(PropertyInfo(Variant::INT, "vframes", PROPERTY_HINT_RANGE, "1,16384,1"), "set_vframes", "get_vframes");
	ADD_PROPERTYNO(PropertyInfo(Variant::INT, "hframes", PROPERTY_HINT_RANGE, "1,16384,1"), "set_hframes", "get_hframes");
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "frame", PROPERTY_HINT_SPRITE_FRAME), "set_frame", "get_frame");

	ADD_GROUP("Region", "region_");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "region_enabled"), "set_region", "is_region");
	ADD_PROPERTYNZ(PropertyInfo(Variant::RECT2, "region_rect"), "set_region_rect", "get_region_rect");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "region_filter_clip"), "set_region_filter_clip", "is_region_filter_clip_enabled");
}

Sprite::Sprite() {

	centered = true;
	hflip = false;
	vflip = false;
	region = false;
	region_filter_clip = false;

	frame = 0;

	vframes = 1;
	hframes = 1;
}
