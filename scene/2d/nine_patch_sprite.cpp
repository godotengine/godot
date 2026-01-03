/**************************************************************************/
/*  nine_patch_sprite.cpp                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "nine_patch_sprite.h"
#include "core/math/rect2.h"
#include "scene/main/viewport.h"

#ifdef TOOLS_ENABLED
Dictionary NinePatchSprite::_edit_get_state() const {
	Dictionary state = Node2D::_edit_get_state();
	state["offset"] = offset;
	state["size"] = size;
	return state;
}

void NinePatchSprite::_edit_set_state(const Dictionary &p_state) {
	Node2D::_edit_set_state(p_state);
	set_offset(p_state["offset"]);
	set_size(p_state["size"]);
}

void NinePatchSprite::_edit_set_pivot(const Point2 &p_pivot) {
	set_offset(get_offset() - p_pivot);
	set_position(get_transform().xform(p_pivot));
}

Point2 NinePatchSprite::_edit_get_pivot() const {
	return Vector2();
}

bool NinePatchSprite::_edit_use_pivot() const {
	return true;
}
void NinePatchSprite::_edit_set_rect(const Rect2 &p_edit_rect) {
	ERR_FAIL_COND_MSG(!Engine::get_singleton()->is_editor_hint(), "This function can only be used from editor plugins.");
	Vector2 center_offset = Vector2();
	if (get_centered()) {
		center_offset = p_edit_rect.size / 2.0;
	}
	set_position((get_position() - offset + get_transform().basis_xform(p_edit_rect.position + center_offset)));
	set_size(p_edit_rect.size);
}
Size2 NinePatchSprite::_edit_get_minimum_size() const {
	return Vector2(1, 1);
}
#endif // TOOLS_ENABLED

#ifdef DEBUG_ENABLED
bool NinePatchSprite::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	return get_rect().has_point(p_point);
	// return is_pixel_opaque(p_point);
}

Rect2 NinePatchSprite::_edit_get_rect() const {
	return get_rect();
}

bool NinePatchSprite::_edit_use_rect() const {
	return texture.is_valid();
}
#endif
void NinePatchSprite::_texture_changed() {
	queue_redraw();
}
// DEBUG_ENABLED
void NinePatchSprite::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			if (texture.is_null()) {
				return;
			}
			Rect2 rect = get_rect();
			Rect2 src_rect = region_rect;
			RID ci = get_canvas_item();
			RS::get_singleton()->canvas_item_add_nine_patch(ci, rect, src_rect, texture->get_rid(), Vector2(margin[SIDE_LEFT], margin[SIDE_TOP]), Vector2(margin[SIDE_RIGHT], margin[SIDE_BOTTOM]), RS::NinePatchAxisMode(axis_h), RS::NinePatchAxisMode(axis_v), draw_center);
		} break;
	}
}

void NinePatchSprite::_bind_methods() {
	ClassDB ::bind_method(D_METHOD("get_size"), &NinePatchSprite ::get_size);
	ClassDB ::bind_method(D_METHOD("set_size", "size"), &NinePatchSprite ::set_size);
	ClassDB ::bind_method(D_METHOD("get_offset"), &NinePatchSprite ::get_offset);
	ClassDB ::bind_method(D_METHOD("set_offset", "offset"), &NinePatchSprite ::set_offset);
	ClassDB ::bind_method(D_METHOD("get_centered"), &NinePatchSprite ::get_centered);
	ClassDB ::bind_method(D_METHOD("set_centered", "centered"), &NinePatchSprite ::set_centered);
	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &NinePatchSprite::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &NinePatchSprite::get_texture);
	ClassDB::bind_method(D_METHOD("set_patch_margin", "margin", "value"), &NinePatchSprite::set_patch_margin);
	ClassDB::bind_method(D_METHOD("get_patch_margin", "margin"), &NinePatchSprite::get_patch_margin);
	ClassDB::bind_method(D_METHOD("set_region_rect", "rect"), &NinePatchSprite::set_region_rect);
	ClassDB::bind_method(D_METHOD("get_region_rect"), &NinePatchSprite::get_region_rect);
	ClassDB::bind_method(D_METHOD("set_draw_center", "draw_center"), &NinePatchSprite::set_draw_center);
	ClassDB::bind_method(D_METHOD("is_draw_center_enabled"), &NinePatchSprite::is_draw_center_enabled);
	ClassDB::bind_method(D_METHOD("set_h_axis_stretch_mode", "mode"), &NinePatchSprite::set_h_axis_stretch_mode);
	ClassDB::bind_method(D_METHOD("get_h_axis_stretch_mode"), &NinePatchSprite::get_h_axis_stretch_mode);
	ClassDB::bind_method(D_METHOD("set_v_axis_stretch_mode", "mode"), &NinePatchSprite::set_v_axis_stretch_mode);
	ClassDB::bind_method(D_METHOD("get_v_axis_stretch_mode"), &NinePatchSprite::get_v_axis_stretch_mode);
	ClassDB::bind_method(D_METHOD("get_rect"), &NinePatchSprite::get_rect);
	// BIND_PROPERTY(texture, NineSprite, Variant::OBJECT, PROPERTY_HINT_RESOURCE_TYPE, "Texture2D");

	ADD_SIGNAL(MethodInfo("texture_changed"));
	ADD_SIGNAL(MethodInfo("resized", PropertyInfo(Variant::VECTOR2, "new_size")));

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size", PROPERTY_HINT_NONE), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset", PROPERTY_HINT_NONE), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "centered", PROPERTY_HINT_NONE), "set_centered", "get_centered");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_center"), "set_draw_center", "is_draw_center_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "region_rect", PROPERTY_HINT_NONE, "suffix:px"), "set_region_rect", "get_region_rect");

	ADD_GROUP("Patch Margin", "patch_margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "patch_margin_left", PROPERTY_HINT_RANGE, "0,16384,1,suffix:px"), "set_patch_margin", "get_patch_margin", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "patch_margin_top", PROPERTY_HINT_RANGE, "0,16384,1,suffix:px"), "set_patch_margin", "get_patch_margin", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "patch_margin_right", PROPERTY_HINT_RANGE, "0,16384,1,suffix:px"), "set_patch_margin", "get_patch_margin", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "patch_margin_bottom", PROPERTY_HINT_RANGE, "0,16384,1,suffix:px"), "set_patch_margin", "get_patch_margin", SIDE_BOTTOM);
	ADD_GROUP("Axis Stretch", "axis_stretch_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis_stretch_horizontal", PROPERTY_HINT_ENUM, "Stretch,Tile,Tile Fit"), "set_h_axis_stretch_mode", "get_h_axis_stretch_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis_stretch_vertical", PROPERTY_HINT_ENUM, "Stretch,Tile,Tile Fit"), "set_v_axis_stretch_mode", "get_v_axis_stretch_mode");
}

bool NinePatchSprite::get_centered() const {
	return centered;
}

void NinePatchSprite::set_centered(bool p_centered) {
	if (centered == p_centered) {
		return;
	}
	centered = p_centered;
	queue_redraw();
}

Vector2 NinePatchSprite::get_offset() const {
	return offset;
}

void NinePatchSprite::set_offset(Vector2 p_offset) {
	this->offset = p_offset;
	queue_redraw();
}

Vector2 NinePatchSprite::get_size() const {
	return size;
}

void NinePatchSprite::set_size(Vector2 p_size) {
	size = p_size;
	emit_signal(SceneStringName(resized), p_size);
	queue_redraw();
}

void NinePatchSprite::set_texture(const Ref<Texture2D> &p_tex) {
	if (texture == p_tex) {
		return;
	}

	if (texture.is_valid()) {
		texture->disconnect_changed(callable_mp(this, &NinePatchSprite::_texture_changed));
	}

	texture = p_tex;

	if (texture.is_valid()) {
		texture->connect_changed(callable_mp(this, &NinePatchSprite::_texture_changed));
	}

	queue_redraw();
	emit_signal(SceneStringName(texture_changed));
}

Ref<Texture2D> NinePatchSprite::get_texture() const {
	return texture;
}

void NinePatchSprite::set_patch_margin(Side p_side, int p_size) {
	ERR_FAIL_INDEX((int)p_side, 4);

	if (margin[p_side] == p_size) {
		return;
	}

	margin[p_side] = p_size;
	queue_redraw();
	// update_minimum_size();
}

int NinePatchSprite::get_patch_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);
	return margin[p_side];
}

void NinePatchSprite::set_region_rect(const Rect2 &p_region_rect) {
	if (region_rect == p_region_rect) {
		return;
	}

	region_rect = p_region_rect;
	item_rect_changed();
}

Rect2 NinePatchSprite::get_region_rect() const {
	return region_rect;
}

void NinePatchSprite::set_draw_center(bool p_enabled) {
	if (draw_center == p_enabled) {
		return;
	}

	draw_center = p_enabled;
	queue_redraw();
}

bool NinePatchSprite::is_draw_center_enabled() const {
	return draw_center;
}

void NinePatchSprite::set_h_axis_stretch_mode(AxisStretchMode p_mode) {
	if (axis_h == p_mode) {
		return;
	}

	axis_h = p_mode;
	queue_redraw();
}

AxisStretchMode NinePatchSprite::get_h_axis_stretch_mode() const {
	return axis_h;
}

void NinePatchSprite::set_v_axis_stretch_mode(AxisStretchMode p_mode) {
	if (axis_v == p_mode) {
		return;
	}

	axis_v = p_mode;
	queue_redraw();
}

AxisStretchMode NinePatchSprite::get_v_axis_stretch_mode() const {
	return axis_v;
}

NinePatchSprite::NinePatchSprite() {
	// set_mouse_filter(MOUSE_FILTER_IGNORE);
}

Rect2 NinePatchSprite::get_rect() const {
	if (texture.is_null()) {
		return Rect2(0, 0, 1, 1);
	}

	Size2i s = get_size();

	Point2 ofs = offset;
	if (centered) {
		ofs -= Size2(s) / 2;
	}

	if (get_viewport() && get_viewport()->is_snap_2d_transforms_to_pixel_enabled()) {
		ofs = (ofs + Point2(0.5, 0.5)).floor();
	}

	if (s == Size2(0, 0)) {
		s = Size2(1, 1);
	}

	return Rect2(ofs, s);
}

NinePatchSprite::~NinePatchSprite() {
}
