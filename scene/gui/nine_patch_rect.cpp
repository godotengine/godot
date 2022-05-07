/*************************************************************************/
/*  nine_patch_rect.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "nine_patch_rect.h"

#include "servers/visual_server.h"

void NinePatchRect::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		if (texture.is_null()) {
			return;
		}

		Rect2 rect = Rect2(Point2(), get_size());
		Rect2 src_rect = region_rect;

		texture->get_rect_region(rect, src_rect, rect, src_rect);

		RID ci = get_canvas_item();
		VS::get_singleton()->canvas_item_add_nine_patch(ci, rect, src_rect, texture->get_rid(), Vector2(margin[MARGIN_LEFT], margin[MARGIN_TOP]), Vector2(margin[MARGIN_RIGHT], margin[MARGIN_BOTTOM]), VS::NinePatchAxisMode(axis_h), VS::NinePatchAxisMode(axis_v), draw_center);
	}
}

Size2 NinePatchRect::get_minimum_size() const {
	return Size2(margin[MARGIN_LEFT] + margin[MARGIN_RIGHT], margin[MARGIN_TOP] + margin[MARGIN_BOTTOM]);
}
void NinePatchRect::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &NinePatchRect::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &NinePatchRect::get_texture);
	ClassDB::bind_method(D_METHOD("set_patch_margin", "margin", "value"), &NinePatchRect::set_patch_margin);
	ClassDB::bind_method(D_METHOD("get_patch_margin", "margin"), &NinePatchRect::get_patch_margin);
	ClassDB::bind_method(D_METHOD("set_region_rect", "rect"), &NinePatchRect::set_region_rect);
	ClassDB::bind_method(D_METHOD("get_region_rect"), &NinePatchRect::get_region_rect);
	ClassDB::bind_method(D_METHOD("set_draw_center", "draw_center"), &NinePatchRect::set_draw_center);
	ClassDB::bind_method(D_METHOD("is_draw_center_enabled"), &NinePatchRect::is_draw_center_enabled);
	ClassDB::bind_method(D_METHOD("set_h_axis_stretch_mode", "mode"), &NinePatchRect::set_h_axis_stretch_mode);
	ClassDB::bind_method(D_METHOD("get_h_axis_stretch_mode"), &NinePatchRect::get_h_axis_stretch_mode);
	ClassDB::bind_method(D_METHOD("set_v_axis_stretch_mode", "mode"), &NinePatchRect::set_v_axis_stretch_mode);
	ClassDB::bind_method(D_METHOD("get_v_axis_stretch_mode"), &NinePatchRect::get_v_axis_stretch_mode);

	ADD_SIGNAL(MethodInfo("texture_changed"));

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_center"), "set_draw_center", "is_draw_center_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "region_rect"), "set_region_rect", "get_region_rect");

	ADD_GROUP("Patch Margin", "patch_margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "patch_margin_left", PROPERTY_HINT_RANGE, "0,16384,1"), "set_patch_margin", "get_patch_margin", MARGIN_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "patch_margin_top", PROPERTY_HINT_RANGE, "0,16384,1"), "set_patch_margin", "get_patch_margin", MARGIN_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "patch_margin_right", PROPERTY_HINT_RANGE, "0,16384,1"), "set_patch_margin", "get_patch_margin", MARGIN_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "patch_margin_bottom", PROPERTY_HINT_RANGE, "0,16384,1"), "set_patch_margin", "get_patch_margin", MARGIN_BOTTOM);
	ADD_GROUP("Axis Stretch", "axis_stretch_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis_stretch_horizontal", PROPERTY_HINT_ENUM, "Stretch,Tile,Tile Fit"), "set_h_axis_stretch_mode", "get_h_axis_stretch_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis_stretch_vertical", PROPERTY_HINT_ENUM, "Stretch,Tile,Tile Fit"), "set_v_axis_stretch_mode", "get_v_axis_stretch_mode");

	BIND_ENUM_CONSTANT(AXIS_STRETCH_MODE_STRETCH);
	BIND_ENUM_CONSTANT(AXIS_STRETCH_MODE_TILE);
	BIND_ENUM_CONSTANT(AXIS_STRETCH_MODE_TILE_FIT);
}

void NinePatchRect::set_texture(const Ref<Texture> &p_tex) {
	if (texture == p_tex) {
		return;
	}
	texture = p_tex;
	update();
	/*
	if (texture.is_valid())
		texture->set_flags(texture->get_flags()&(~Texture::FLAG_REPEAT)); //remove repeat from texture, it looks bad in sprites
	*/
	minimum_size_changed();
	emit_signal("texture_changed");
	_change_notify("texture");
}

Ref<Texture> NinePatchRect::get_texture() const {
	return texture;
}

void NinePatchRect::set_patch_margin(Margin p_margin, int p_size) {
	ERR_FAIL_INDEX((int)p_margin, 4);
	margin[p_margin] = p_size;
	update();
	minimum_size_changed();
	switch (p_margin) {
		case MARGIN_LEFT:
			_change_notify("patch_margin_left");
			break;
		case MARGIN_TOP:
			_change_notify("patch_margin_top");
			break;
		case MARGIN_RIGHT:
			_change_notify("patch_margin_right");
			break;
		case MARGIN_BOTTOM:
			_change_notify("patch_margin_bottom");
			break;
	}
}

int NinePatchRect::get_patch_margin(Margin p_margin) const {
	ERR_FAIL_INDEX_V((int)p_margin, 4, 0);
	return margin[p_margin];
}

void NinePatchRect::set_region_rect(const Rect2 &p_region_rect) {
	if (region_rect == p_region_rect) {
		return;
	}

	region_rect = p_region_rect;

	item_rect_changed();
	_change_notify("region_rect");
}

Rect2 NinePatchRect::get_region_rect() const {
	return region_rect;
}

void NinePatchRect::set_draw_center(bool p_enabled) {
	draw_center = p_enabled;
	update();
}

bool NinePatchRect::is_draw_center_enabled() const {
	return draw_center;
}

void NinePatchRect::set_h_axis_stretch_mode(AxisStretchMode p_mode) {
	axis_h = p_mode;
	update_configuration_warning();
	update();
}

NinePatchRect::AxisStretchMode NinePatchRect::get_h_axis_stretch_mode() const {
	return axis_h;
}

void NinePatchRect::set_v_axis_stretch_mode(AxisStretchMode p_mode) {
	axis_v = p_mode;
	update_configuration_warning();
	update();
}

NinePatchRect::AxisStretchMode NinePatchRect::get_v_axis_stretch_mode() const {
	return axis_v;
}

String NinePatchRect::get_configuration_warning() const {
	String warning = Control::get_configuration_warning();

	if (String(GLOBAL_GET("rendering/quality/driver/driver_name")) == "GLES2") {
		if (axis_v > AXIS_STRETCH_MODE_STRETCH || axis_h > AXIS_STRETCH_MODE_STRETCH) {
			if (!warning.empty()) {
				warning += "\n\n";
			}
			warning += TTR("The Tile and Tile Fit options for Axis Stretch properties are only effective when using the GLES3 rendering backend.\nThe GLES2 backend is currently in use, so these modes will act like Stretch instead.");
		}
	}

	return warning;
}

NinePatchRect::NinePatchRect() {
	margin[MARGIN_LEFT] = 0;
	margin[MARGIN_RIGHT] = 0;
	margin[MARGIN_BOTTOM] = 0;
	margin[MARGIN_TOP] = 0;

	set_mouse_filter(MOUSE_FILTER_IGNORE);
	draw_center = true;

	axis_h = AXIS_STRETCH_MODE_STRETCH;
	axis_v = AXIS_STRETCH_MODE_STRETCH;
}

NinePatchRect::~NinePatchRect() {
}
