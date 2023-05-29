/**************************************************************************/
/*  style_box.cpp                                                         */
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

#include "style_box.h"

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/object/object.h"
#include "scene/main/canvas_item.h"

#include <limits.h>

Size2 StyleBox::get_minimum_size() const {
	Size2 min_size = Size2(get_margin(SIDE_LEFT) + get_margin(SIDE_RIGHT), get_margin(SIDE_TOP) + get_margin(SIDE_BOTTOM));
	Size2 custom_size;
	GDVIRTUAL_CALL(_get_minimum_size, custom_size);

	if (min_size.x < custom_size.x) {
		min_size.x = custom_size.x;
	}
	if (min_size.y < custom_size.y) {
		min_size.y = custom_size.y;
	}

	return min_size;
}

void StyleBox::set_content_margin(Side p_side, float p_value) {
	ERR_FAIL_INDEX((int)p_side, 4);

	content_margin[p_side] = p_value;
	emit_changed();
}

void StyleBox::set_content_margin_all(float p_value) {
	for (int i = 0; i < 4; i++) {
		content_margin[i] = p_value;
	}
	emit_changed();
}

void StyleBox::set_content_margin_individual(float p_left, float p_top, float p_right, float p_bottom) {
	content_margin[SIDE_LEFT] = p_left;
	content_margin[SIDE_TOP] = p_top;
	content_margin[SIDE_RIGHT] = p_right;
	content_margin[SIDE_BOTTOM] = p_bottom;
	emit_changed();
}

float StyleBox::get_content_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0.0);

	return content_margin[p_side];
}

float StyleBox::get_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0.0);

	if (content_margin[p_side] < 0) {
		return get_style_margin(p_side);
	} else {
		return content_margin[p_side];
	}
}

Point2 StyleBox::get_offset() const {
	return Point2(get_margin(SIDE_LEFT), get_margin(SIDE_TOP));
}

void StyleBox::draw(RID p_canvas_item, const Rect2 &p_rect) const {
	GDVIRTUAL_REQUIRED_CALL(_draw, p_canvas_item, p_rect);
}

Rect2 StyleBox::get_draw_rect(const Rect2 &p_rect) const {
	Rect2 ret;
	if (GDVIRTUAL_CALL(_get_draw_rect, p_rect, ret)) {
		return ret;
	}
	return p_rect;
}

CanvasItem *StyleBox::get_current_item_drawn() const {
	return CanvasItem::get_current_item_drawn();
}

bool StyleBox::test_mask(const Point2 &p_point, const Rect2 &p_rect) const {
	bool ret = true;
	GDVIRTUAL_CALL(_test_mask, p_point, p_rect, ret);
	return ret;
}

void StyleBox::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_minimum_size"), &StyleBox::get_minimum_size);

	ClassDB::bind_method(D_METHOD("set_content_margin", "margin", "offset"), &StyleBox::set_content_margin);
	ClassDB::bind_method(D_METHOD("set_content_margin_all", "offset"), &StyleBox::set_content_margin_all);
	ClassDB::bind_method(D_METHOD("get_content_margin", "margin"), &StyleBox::get_content_margin);

	ClassDB::bind_method(D_METHOD("get_margin", "margin"), &StyleBox::get_margin);
	ClassDB::bind_method(D_METHOD("get_offset"), &StyleBox::get_offset);

	ClassDB::bind_method(D_METHOD("draw", "canvas_item", "rect"), &StyleBox::draw);
	ClassDB::bind_method(D_METHOD("get_current_item_drawn"), &StyleBox::get_current_item_drawn);

	ClassDB::bind_method(D_METHOD("test_mask", "point", "rect"), &StyleBox::test_mask);

	ADD_GROUP("Content Margins", "content_margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "content_margin_left", PROPERTY_HINT_RANGE, "-1,2048,1,suffix:px"), "set_content_margin", "get_content_margin", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "content_margin_top", PROPERTY_HINT_RANGE, "-1,2048,1,suffix:px"), "set_content_margin", "get_content_margin", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "content_margin_right", PROPERTY_HINT_RANGE, "-1,2048,1,suffix:px"), "set_content_margin", "get_content_margin", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "content_margin_bottom", PROPERTY_HINT_RANGE, "-1,2048,1,suffix:px"), "set_content_margin", "get_content_margin", SIDE_BOTTOM);

	GDVIRTUAL_BIND(_draw, "to_canvas_item", "rect")
	GDVIRTUAL_BIND(_get_draw_rect, "rect")
	GDVIRTUAL_BIND(_get_minimum_size)
	GDVIRTUAL_BIND(_test_mask, "point", "rect")
}

StyleBox::StyleBox() {
	for (int i = 0; i < 4; i++) {
		content_margin[i] = -1;
	}
}

void StyleBoxTexture::set_texture(Ref<Texture2D> p_texture) {
	if (texture == p_texture) {
		return;
	}
	texture = p_texture;
	emit_changed();
}

Ref<Texture2D> StyleBoxTexture::get_texture() const {
	return texture;
}

void StyleBoxTexture::set_texture_margin(Side p_side, float p_size) {
	ERR_FAIL_INDEX((int)p_side, 4);

	texture_margin[p_side] = p_size;
	emit_changed();
}

void StyleBoxTexture::set_texture_margin_all(float p_size) {
	for (int i = 0; i < 4; i++) {
		texture_margin[i] = p_size;
	}
	emit_changed();
}

void StyleBoxTexture::set_texture_margin_individual(float p_left, float p_top, float p_right, float p_bottom) {
	texture_margin[SIDE_LEFT] = p_left;
	texture_margin[SIDE_TOP] = p_top;
	texture_margin[SIDE_RIGHT] = p_right;
	texture_margin[SIDE_BOTTOM] = p_bottom;
	emit_changed();
}

float StyleBoxTexture::get_texture_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0.0);

	return texture_margin[p_side];
}

float StyleBoxTexture::get_style_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0.0);

	return texture_margin[p_side];
}

Rect2 StyleBoxTexture::get_draw_rect(const Rect2 &p_rect) const {
	return p_rect.grow_individual(expand_margin[SIDE_LEFT], expand_margin[SIDE_TOP], expand_margin[SIDE_RIGHT], expand_margin[SIDE_BOTTOM]);
}

void StyleBoxTexture::draw(RID p_canvas_item, const Rect2 &p_rect) const {
	if (texture.is_null()) {
		return;
	}

	Rect2 rect = p_rect;
	Rect2 src_rect = region_rect;

	texture->get_rect_region(rect, src_rect, rect, src_rect);

	rect.position.x -= expand_margin[SIDE_LEFT];
	rect.position.y -= expand_margin[SIDE_TOP];
	rect.size.x += expand_margin[SIDE_LEFT] + expand_margin[SIDE_RIGHT];
	rect.size.y += expand_margin[SIDE_TOP] + expand_margin[SIDE_BOTTOM];

	Vector2 start_offset = Vector2(texture_margin[SIDE_LEFT], texture_margin[SIDE_TOP]);
	Vector2 end_offset = Vector2(texture_margin[SIDE_RIGHT], texture_margin[SIDE_BOTTOM]);

	RenderingServer::get_singleton()->canvas_item_add_nine_patch(p_canvas_item, rect, src_rect, texture->get_rid(), start_offset, end_offset, RS::NinePatchAxisMode(axis_h), RS::NinePatchAxisMode(axis_v), draw_center, modulate);
}

void StyleBoxTexture::set_draw_center(bool p_enabled) {
	draw_center = p_enabled;
	emit_changed();
}

bool StyleBoxTexture::is_draw_center_enabled() const {
	return draw_center;
}

void StyleBoxTexture::set_expand_margin(Side p_side, float p_size) {
	ERR_FAIL_INDEX((int)p_side, 4);
	expand_margin[p_side] = p_size;
	emit_changed();
}

void StyleBoxTexture::set_expand_margin_individual(float p_left, float p_top, float p_right, float p_bottom) {
	expand_margin[SIDE_LEFT] = p_left;
	expand_margin[SIDE_TOP] = p_top;
	expand_margin[SIDE_RIGHT] = p_right;
	expand_margin[SIDE_BOTTOM] = p_bottom;
	emit_changed();
}

void StyleBoxTexture::set_expand_margin_all(float p_expand_margin_size) {
	for (int i = 0; i < 4; i++) {
		expand_margin[i] = p_expand_margin_size;
	}
	emit_changed();
}

float StyleBoxTexture::get_expand_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);
	return expand_margin[p_side];
}

void StyleBoxTexture::set_region_rect(const Rect2 &p_region_rect) {
	if (region_rect == p_region_rect) {
		return;
	}

	region_rect = p_region_rect;
	emit_changed();
}

Rect2 StyleBoxTexture::get_region_rect() const {
	return region_rect;
}

void StyleBoxTexture::set_h_axis_stretch_mode(AxisStretchMode p_mode) {
	ERR_FAIL_INDEX((int)p_mode, 3);
	axis_h = p_mode;
	emit_changed();
}

StyleBoxTexture::AxisStretchMode StyleBoxTexture::get_h_axis_stretch_mode() const {
	return axis_h;
}

void StyleBoxTexture::set_v_axis_stretch_mode(AxisStretchMode p_mode) {
	ERR_FAIL_INDEX((int)p_mode, 3);
	axis_v = p_mode;
	emit_changed();
}

StyleBoxTexture::AxisStretchMode StyleBoxTexture::get_v_axis_stretch_mode() const {
	return axis_v;
}

void StyleBoxTexture::set_modulate(const Color &p_modulate) {
	if (modulate == p_modulate) {
		return;
	}
	modulate = p_modulate;
	emit_changed();
}

Color StyleBoxTexture::get_modulate() const {
	return modulate;
}

void StyleBoxTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &StyleBoxTexture::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &StyleBoxTexture::get_texture);

	ClassDB::bind_method(D_METHOD("set_texture_margin", "margin", "size"), &StyleBoxTexture::set_texture_margin);
	ClassDB::bind_method(D_METHOD("set_texture_margin_all", "size"), &StyleBoxTexture::set_texture_margin_all);
	ClassDB::bind_method(D_METHOD("get_texture_margin", "margin"), &StyleBoxTexture::get_texture_margin);

	ClassDB::bind_method(D_METHOD("set_expand_margin", "margin", "size"), &StyleBoxTexture::set_expand_margin);
	ClassDB::bind_method(D_METHOD("set_expand_margin_all", "size"), &StyleBoxTexture::set_expand_margin_all);
	ClassDB::bind_method(D_METHOD("get_expand_margin", "margin"), &StyleBoxTexture::get_expand_margin);

	ClassDB::bind_method(D_METHOD("set_region_rect", "region"), &StyleBoxTexture::set_region_rect);
	ClassDB::bind_method(D_METHOD("get_region_rect"), &StyleBoxTexture::get_region_rect);

	ClassDB::bind_method(D_METHOD("set_draw_center", "enable"), &StyleBoxTexture::set_draw_center);
	ClassDB::bind_method(D_METHOD("is_draw_center_enabled"), &StyleBoxTexture::is_draw_center_enabled);

	ClassDB::bind_method(D_METHOD("set_modulate", "color"), &StyleBoxTexture::set_modulate);
	ClassDB::bind_method(D_METHOD("get_modulate"), &StyleBoxTexture::get_modulate);

	ClassDB::bind_method(D_METHOD("set_h_axis_stretch_mode", "mode"), &StyleBoxTexture::set_h_axis_stretch_mode);
	ClassDB::bind_method(D_METHOD("get_h_axis_stretch_mode"), &StyleBoxTexture::get_h_axis_stretch_mode);

	ClassDB::bind_method(D_METHOD("set_v_axis_stretch_mode", "mode"), &StyleBoxTexture::set_v_axis_stretch_mode);
	ClassDB::bind_method(D_METHOD("get_v_axis_stretch_mode"), &StyleBoxTexture::get_v_axis_stretch_mode);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture");

	ADD_GROUP("Texture Margins", "texture_margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "texture_margin_left", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_texture_margin", "get_texture_margin", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "texture_margin_top", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_texture_margin", "get_texture_margin", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "texture_margin_right", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_texture_margin", "get_texture_margin", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "texture_margin_bottom", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_texture_margin", "get_texture_margin", SIDE_BOTTOM);

	ADD_GROUP("Expand Margins", "expand_margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "expand_margin_left", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_expand_margin", "get_expand_margin", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "expand_margin_top", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_expand_margin", "get_expand_margin", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "expand_margin_right", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_expand_margin", "get_expand_margin", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "expand_margin_bottom", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_expand_margin", "get_expand_margin", SIDE_BOTTOM);

	ADD_GROUP("Axis Stretch", "axis_stretch_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis_stretch_horizontal", PROPERTY_HINT_ENUM, "Stretch,Tile,Tile Fit"), "set_h_axis_stretch_mode", "get_h_axis_stretch_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis_stretch_vertical", PROPERTY_HINT_ENUM, "Stretch,Tile,Tile Fit"), "set_v_axis_stretch_mode", "get_v_axis_stretch_mode");

	ADD_GROUP("Sub-Region", "region_");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "region_rect", PROPERTY_HINT_NONE, "suffix:px"), "set_region_rect", "get_region_rect");

	ADD_GROUP("Modulate", "modulate_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "modulate_color"), "set_modulate", "get_modulate");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_center"), "set_draw_center", "is_draw_center_enabled");

	BIND_ENUM_CONSTANT(AXIS_STRETCH_MODE_STRETCH);
	BIND_ENUM_CONSTANT(AXIS_STRETCH_MODE_TILE);
	BIND_ENUM_CONSTANT(AXIS_STRETCH_MODE_TILE_FIT);
}

StyleBoxTexture::StyleBoxTexture() {}

StyleBoxTexture::~StyleBoxTexture() {}

////////////////

void StyleBoxFlat::set_bg_color(const Color &p_color) {
	bg_color = p_color;
	emit_changed();
}

Color StyleBoxFlat::get_bg_color() const {
	return bg_color;
}

void StyleBoxFlat::set_bg_secondary_color(const Color &p_color) {
	bg_secondary_color = p_color;
	emit_changed();
}

Color StyleBoxFlat::get_bg_secondary_color() const {
	return bg_secondary_color;
}

void StyleBoxFlat::set_bg_simple_gradient_orientation(SimpleGradientOrientation p_orientation) {
	if (center_gradient_orientation == p_orientation) {
		return;
	}
	center_gradient_orientation = p_orientation;
	emit_changed();
}

StyleBoxFlat::SimpleGradientOrientation StyleBoxFlat::get_bg_simple_gradient_orientation() const {
	return center_gradient_orientation;
}

void StyleBoxFlat::set_border_color(const Color &p_color) {
	border_color = p_color;
	emit_changed();
}

Color StyleBoxFlat::get_border_color() const {
	return border_color;
}

void StyleBoxFlat::set_border_secondary_color(const Color &p_color) {
	border_secondary_color = p_color;
	emit_changed();
}

Color StyleBoxFlat::get_border_secondary_color() const {
	return border_secondary_color;
}

void StyleBoxFlat::set_border_third_color(const Color &p_color) {
	border_third_color = p_color;
	emit_changed();
}

Color StyleBoxFlat::get_border_third_color() const {
	return border_third_color;
}

void StyleBoxFlat::set_border_fourth_color(const Color &p_color) {
	border_fourth_color = p_color;
	emit_changed();
}

Color StyleBoxFlat::get_border_fourth_color() const {
	return border_fourth_color;
}

void StyleBoxFlat::set_border_simple_gradient_orientation(SimpleGradientOrientation p_orientation) {
	if (border_gradient_orientation == p_orientation) {
		return;
	}
	border_gradient_orientation = p_orientation;
	emit_changed();
}

StyleBoxFlat::SimpleGradientOrientation StyleBoxFlat::get_border_simple_gradient_orientation() const {
	return border_gradient_orientation;
}

void StyleBoxFlat::set_border_simple_gradient_algorithm(SimpleGradientColoringAlgorithm p_algo) {
	if (border_gradient_algo == p_algo) {
		return;
	}
	border_gradient_algo = p_algo;
	emit_changed();
}

StyleBoxFlat::SimpleGradientColoringAlgorithm StyleBoxFlat::get_border_simple_gradient_algorithm() const {
	return border_gradient_algo;
}

void StyleBoxFlat::set_border_bevel_lighting_color(const Color &p_color) {
	border_bevel_light_color = p_color;
	emit_changed();
}

Color StyleBoxFlat::get_border_bevel_lighting_color() const {
	return border_bevel_light_color;
}

void StyleBoxFlat::set_border_bevel_darkening_color(const Color &p_color) {
	border_bevel_dark_color = p_color;
	emit_changed();
}

Color StyleBoxFlat::get_border_bevel_darkening_color() const {
	return border_bevel_dark_color;
}

void StyleBoxFlat::set_border_bevel_lighting_intensity(float p_intensity) {
	if (bevel_lighting_intensity == p_intensity) {
		return;
	}
	bevel_lighting_intensity = p_intensity;
	emit_changed();
}

float StyleBoxFlat::get_border_bevel_lighting_intensity() const {
	return bevel_lighting_intensity;
}

void StyleBoxFlat::set_border_bevel_darkening_intensity(float p_intensity) {
	if (bevel_darkening_intensity == p_intensity) {
		return;
	}
	bevel_darkening_intensity = p_intensity;
	emit_changed();
}

float StyleBoxFlat::get_border_bevel_darkening_intensity() const {
	return bevel_darkening_intensity;
}

void StyleBoxFlat::set_border_bevel_lighting_angle(float p_angle) {
	while (p_angle > 360) {
		p_angle -= 360;
	}
	while (p_angle < 0) {
		p_angle += 360;
	}
	if (bevel_lighting_angle == p_angle) {
		return;
	}

	bevel_lighting_angle = p_angle;
	emit_changed();
}

float StyleBoxFlat::get_border_bevel_lighting_angle() const {
	return bevel_lighting_angle;
}

void StyleBoxFlat::set_border_bevel_max_intensity_angle_ratio(float p_ratio) {
	if (bevel_max_intensity_angle_ratio == p_ratio) {
		return;
	}
	bevel_max_intensity_angle_ratio = p_ratio;
	emit_changed();
}

float StyleBoxFlat::get_border_bevel_max_intensity_angle_ratio() const {
	return bevel_max_intensity_angle_ratio;
}

void StyleBoxFlat::set_border_side_color(Side p_side, const Color &p_color) {
	ERR_FAIL_INDEX((int)p_side, 4);
	border_side_colors[p_side] = p_color;
	emit_changed();
}

void StyleBoxFlat::set_border_side_color_individual(const Color &p_left, const Color &p_top, const Color &p_right, const Color &p_bottom) {
	border_side_colors[SIDE_LEFT] = p_left;
	border_side_colors[SIDE_TOP] = p_top;
	border_side_colors[SIDE_RIGHT] = p_right;
	border_side_colors[SIDE_BOTTOM] = p_bottom;
	emit_changed();
}

void StyleBoxFlat::set_border_side_color_all(const Color &p_color) {
	for (int i = 0; i < 4; i++) {
		border_side_colors[i] = p_color;
	}
	emit_changed();
}

Color StyleBoxFlat::get_border_side_color(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, Color(0, 0, 0));
	return border_side_colors[p_side];
}

void StyleBoxFlat::set_border_side_secondary_color(Side p_side, const Color &p_color) {
	ERR_FAIL_INDEX((int)p_side, 4);
	border_side_secondary_colors[p_side] = p_color;
	emit_changed();
}

void StyleBoxFlat::set_border_side_secondary_color_individual(const Color &p_left, const Color &p_top, const Color &p_right, const Color &p_bottom) {
	border_side_secondary_colors[SIDE_LEFT] = p_left;
	border_side_secondary_colors[SIDE_TOP] = p_top;
	border_side_secondary_colors[SIDE_RIGHT] = p_right;
	border_side_secondary_colors[SIDE_BOTTOM] = p_bottom;
	emit_changed();
}

void StyleBoxFlat::set_border_side_secondary_color_all(const Color &p_color) {
	for (int i = 0; i < 4; i++) {
		border_side_secondary_colors[i] = p_color;
	}
	emit_changed();
}

Color StyleBoxFlat::get_border_side_secondary_color(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, Color(0, 0, 0));
	return border_side_secondary_colors[p_side];
}

void StyleBoxFlat::set_border_side_color_defined(Side p_side, bool p_defined) {
	ERR_FAIL_INDEX((int)p_side, 4);
	border_side_colors_defined[p_side] = p_defined;
	emit_changed();
	notify_property_list_changed();
}

bool StyleBoxFlat::get_border_side_color_defined(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);
	return border_side_colors_defined[p_side];
}

void StyleBoxFlat::set_border_side_coloring_style(Side p_side, BorderSideColoringStyle p_coloring_style) {
	ERR_FAIL_INDEX((int)p_side, 4);
	border_side_coloring_styles[p_side] = p_coloring_style;
	emit_changed();
	notify_property_list_changed();
}

StyleBoxFlat::BorderSideColoringStyle StyleBoxFlat::get_border_side_coloring_style(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, BORDER_SIDE_COLORING_STYLE_SOLID);
	return border_side_coloring_styles[p_side];
}

void StyleBoxFlat::set_corner_color_join_angle(Corner p_corner, float p_angle) {
	ERR_FAIL_INDEX((int)p_corner, 4);
	corner_color_join_angles[p_corner] = p_angle;
	emit_changed();
}

float StyleBoxFlat::get_corner_color_join_angle(Corner p_corner) const {
	ERR_FAIL_INDEX_V((int)p_corner, 4, 0);
	return corner_color_join_angles[p_corner];
}

void StyleBoxFlat::set_border_width_all(int p_size) {
	border_width[0] = p_size;
	border_width[1] = p_size;
	border_width[2] = p_size;
	border_width[3] = p_size;
	emit_changed();
}

int StyleBoxFlat::get_border_width_min() const {
	return MIN(MIN(border_width[0], border_width[1]), MIN(border_width[2], border_width[3]));
}

void StyleBoxFlat::set_border_width(Side p_side, int p_width) {
	ERR_FAIL_INDEX((int)p_side, 4);
	border_width[p_side] = p_width;
	emit_changed();
}

int StyleBoxFlat::get_border_width(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);
	return border_width[p_side];
}

void StyleBoxFlat::set_border_blend(bool p_blend) {
	blend_border = p_blend;
	emit_changed();
}

bool StyleBoxFlat::get_border_blend() const {
	return blend_border;
}

void StyleBoxFlat::set_border_coloring_style(BorderColoringStyle p_coloring_style) {
	border_coloring_style = p_coloring_style;
	emit_changed();
	notify_property_list_changed();
}

StyleBoxFlat::BorderColoringStyle StyleBoxFlat::get_border_coloring_style() const {
	return border_coloring_style;
}

void StyleBoxFlat::set_border_color_join_style(BorderColorJoinStyle p_join_style) {
	color_join_style = p_join_style;
	emit_changed();
}

StyleBoxFlat::BorderColorJoinStyle StyleBoxFlat::get_border_color_join_style() const {
	return color_join_style;
}

void StyleBoxFlat::set_corner_radius_all(int radius) {
	for (int i = 0; i < 4; i++) {
		corner_radius[i] = radius;
	}

	emit_changed();
}

void StyleBoxFlat::set_corner_radius_individual(const int radius_top_left, const int radius_top_right, const int radius_bottom_right, const int radius_bottom_left) {
	corner_radius[0] = radius_top_left;
	corner_radius[1] = radius_top_right;
	corner_radius[2] = radius_bottom_right;
	corner_radius[3] = radius_bottom_left;

	emit_changed();
}

void StyleBoxFlat::set_corner_radius(const Corner p_corner, const int radius) {
	ERR_FAIL_INDEX((int)p_corner, 4);
	corner_radius[p_corner] = radius;
	emit_changed();
}

int StyleBoxFlat::get_corner_radius(const Corner p_corner) const {
	ERR_FAIL_INDEX_V((int)p_corner, 4, 0);
	return corner_radius[p_corner];
}

void StyleBoxFlat::set_expand_margin(Side p_side, float p_size) {
	ERR_FAIL_INDEX((int)p_side, 4);
	expand_margin[p_side] = p_size;
	emit_changed();
}

void StyleBoxFlat::set_expand_margin_individual(float p_left, float p_top, float p_right, float p_bottom) {
	expand_margin[SIDE_LEFT] = p_left;
	expand_margin[SIDE_TOP] = p_top;
	expand_margin[SIDE_RIGHT] = p_right;
	expand_margin[SIDE_BOTTOM] = p_bottom;
	emit_changed();
}

void StyleBoxFlat::set_expand_margin_all(float p_expand_margin_size) {
	for (int i = 0; i < 4; i++) {
		expand_margin[i] = p_expand_margin_size;
	}
	emit_changed();
}

float StyleBoxFlat::get_expand_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0.0);
	return expand_margin[p_side];
}

void StyleBoxFlat::set_draw_center(bool p_enabled) {
	draw_center = p_enabled;
	emit_changed();
}

bool StyleBoxFlat::is_draw_center_enabled() const {
	return draw_center;
}

void StyleBoxFlat::set_skew(Vector2 p_skew) {
	skew = p_skew;
	emit_changed();
}

Vector2 StyleBoxFlat::get_skew() const {
	return skew;
}

void StyleBoxFlat::set_shadow_color(const Color &p_color) {
	shadow_color = p_color;
	emit_changed();
}

Color StyleBoxFlat::get_shadow_color() const {
	return shadow_color;
}

void StyleBoxFlat::set_shadow_size(const int &p_size) {
	shadow_size = p_size;
	emit_changed();
}

int StyleBoxFlat::get_shadow_size() const {
	return shadow_size;
}

void StyleBoxFlat::set_shadow_offset(const Point2 &p_offset) {
	shadow_offset = p_offset;
	emit_changed();
}

Point2 StyleBoxFlat::get_shadow_offset() const {
	return shadow_offset;
}

void StyleBoxFlat::set_anti_aliased(const bool &p_anti_aliased) {
	anti_aliased = p_anti_aliased;
	emit_changed();
	notify_property_list_changed();
}

bool StyleBoxFlat::is_anti_aliased() const {
	return anti_aliased;
}

void StyleBoxFlat::set_aa_size(const real_t p_aa_size) {
	aa_size = CLAMP(p_aa_size, 0.01, 10);
	emit_changed();
}

real_t StyleBoxFlat::get_aa_size() const {
	return aa_size;
}

void StyleBoxFlat::set_corner_detail(const int &p_corner_detail) {
	corner_detail = CLAMP(p_corner_detail, 1, 20);
	emit_changed();
}

int StyleBoxFlat::get_corner_detail() const {
	return corner_detail;
}

void StyleBoxFlat::set_center_fill_style(FillStyle p_fill_style) {
	center_fill_style = p_fill_style;
	emit_changed();
	notify_property_list_changed();
}

StyleBoxFlat::FillStyle StyleBoxFlat::get_center_fill_style() const {
	return center_fill_style;
}

inline void set_inner_corner_radius(const Rect2 style_rect, const Rect2 inner_rect, const real_t corner_radius[4], real_t *inner_corner_radius) {
	real_t border_left = inner_rect.position.x - style_rect.position.x;
	real_t border_top = inner_rect.position.y - style_rect.position.y;
	real_t border_right = style_rect.size.width - inner_rect.size.width - border_left;
	real_t border_bottom = style_rect.size.height - inner_rect.size.height - border_top;

	real_t rad;

	// Top left.
	rad = MIN(border_top, border_left);
	inner_corner_radius[0] = MAX(corner_radius[0] - rad, 0);

	// Top right;
	rad = MIN(border_top, border_right);
	inner_corner_radius[1] = MAX(corner_radius[1] - rad, 0);

	// Bottom right.
	rad = MIN(border_bottom, border_right);
	inner_corner_radius[2] = MAX(corner_radius[2] - rad, 0);

	// Bottom left.
	rad = MIN(border_bottom, border_left);
	inner_corner_radius[3] = MAX(corner_radius[3] - rad, 0);
}

enum RingColorPoints {
	LEFT_SIDE_BOTTOM,
	LEFT_SIDE_TOP,
	TOP_SIDE_LEFT,
	TOP_SIDE_RIGHT,
	RIGHT_SIDE_TOP,
	RIGHT_SIDE_BOTTOM,
	BOTTOM_SIDE_RIGHT,
	BOTTOM_SIDE_LEFT
};

inline void draw_ring(Vector<Vector2> &verts, Vector<int> &indices, Vector<Color> &colors, const Rect2 &style_rect, const real_t corner_radius[4],
		const real_t corner_color_join_angles[4],
		const Rect2 &ring_rect, const Rect2 &inner_rect, const Color inner_color_points[8], const Color outer_color_points[8], const int corner_detail, const Vector2 &skew,
		bool sharp_color_transitions = false, bool even_gradient_algo_inner = false, bool even_gradient_algo_outer = false) {
	int vert_offset = verts.size();
	if (!vert_offset) {
		vert_offset = 0;
	}

	bool is_perfect_square = corner_radius[0] == 0 && corner_radius[1] == 0 && corner_radius[2] == 0 && corner_radius[3] == 0;
	int adapted_corner_detail = is_perfect_square ? 1 : corner_detail;

	real_t ring_corner_radius[4];
	set_inner_corner_radius(style_rect, ring_rect, corner_radius, ring_corner_radius);

	// Corner radius center points.
	Vector<Point2> outer_points = {
		ring_rect.position + Vector2(ring_corner_radius[0], ring_corner_radius[0]), //tl
		Point2(ring_rect.position.x + ring_rect.size.x - ring_corner_radius[1], ring_rect.position.y + ring_corner_radius[1]), //tr
		ring_rect.position + ring_rect.size - Vector2(ring_corner_radius[2], ring_corner_radius[2]), //br
		Point2(ring_rect.position.x + ring_corner_radius[3], ring_rect.position.y + ring_rect.size.y - ring_corner_radius[3]) //bl
	};

	real_t inner_corner_radius[4];
	set_inner_corner_radius(style_rect, inner_rect, corner_radius, inner_corner_radius);

	Vector<Point2> inner_points = {
		inner_rect.position + Vector2(inner_corner_radius[0], inner_corner_radius[0]), //tl
		Point2(inner_rect.position.x + inner_rect.size.x - inner_corner_radius[1], inner_rect.position.y + inner_corner_radius[1]), //tr
		inner_rect.position + inner_rect.size - Vector2(inner_corner_radius[2], inner_corner_radius[2]), //br
		Point2(inner_rect.position.x + inner_corner_radius[3], inner_rect.position.y + inner_rect.size.y - inner_corner_radius[3]) //bl
	};

	for (int corner_index = 0; corner_index < 4; corner_index++) {
		// Compute edge color details depending on join type
		real_t corner_detail_offsets[48]; // arbitrary for the maximum detail level
		real_t corner_detail_color_blend[48];

		int visible_details = adapted_corner_detail;
		int actual_details = adapted_corner_detail;

		if (sharp_color_transitions) {
			// Sharp corner transition

			// Add one supplementary detail. Two will actually overlap to make a sharp transition.
			actual_details = adapted_corner_detail + 1;

			real_t join_angle_ratio = corner_color_join_angles[corner_index];

			real_t actual_join_ratio = (corner_index % 2 == 0 ? join_angle_ratio : 1.0 - join_angle_ratio);
			int cutoff_detail_index = int(visible_details * actual_join_ratio);

			int visible_detail_incr = 0;
			for (int detail = 0; detail <= actual_details; detail++) {
				corner_detail_offsets[detail] = visible_detail_incr / (double)visible_details;
				corner_detail_color_blend[detail] = detail > cutoff_detail_index ? (real_t)1.0 : (real_t)0.0;
				if (detail != cutoff_detail_index) {
					visible_detail_incr++;
				}
			}
		} else {
			// Smooth corner color transition
			for (int detail = 0; detail <= actual_details; detail++) {
				corner_detail_offsets[detail] = detail / (double)adapted_corner_detail;
				corner_detail_color_blend[detail] = corner_detail_offsets[detail];
			}
		}

		// Calculate the vertices.
		for (int detail = 0; detail <= actual_details; detail++) {
			// Loop over successive vertices of this corner

			int side_index = corner_index;
			for (int inner_outer = 0; inner_outer < 2; inner_outer++) {
				real_t radius;
				Color color_a;
				Color color_b;
				Point2 corner_point;

				bool use_alternate_color_algo = false;

				// Determine radius, corner_point, color
				if (inner_outer == 0) {
					radius = inner_corner_radius[corner_index];
					color_a = inner_color_points[side_index * 2 + 1];
					color_b = inner_color_points[(side_index * 2 + 2) % 8];
					corner_point = inner_points[corner_index];
					use_alternate_color_algo = even_gradient_algo_inner;
				} else {
					radius = ring_corner_radius[corner_index];
					color_a = outer_color_points[side_index * 2 + 1];
					color_b = outer_color_points[(side_index * 2 + 2) % 8];
					corner_point = outer_points[corner_index];
					use_alternate_color_algo = even_gradient_algo_outer;
				}

				// Trigonometric calc

				const real_t x = radius * (real_t)cos((corner_index + corner_detail_offsets[detail]) * (Math_TAU / 4.0) + Math_PI) + corner_point.x;
				const real_t y = radius * (real_t)sin((corner_index + corner_detail_offsets[detail]) * (Math_TAU / 4.0) + Math_PI) + corner_point.y;
				const float x_skew = -skew.x * (y - ring_rect.get_center().y);
				const float y_skew = -skew.y * (x - ring_rect.get_center().x);
				verts.push_back(Vector2(x + x_skew, y + y_skew));

				Color vertex_color;

				if (use_alternate_color_algo) {
					// Interpolate color from X Y position

					real_t rel_x;
					real_t rel_y;

					if (inner_outer == 0) {
						// Check if some cases would use the outer rect for mapping even on the inner rect
						rel_x = Math::inverse_lerp(inner_rect.position.x, inner_rect.position.x + inner_rect.size.x, x);
						rel_y = Math::inverse_lerp(inner_rect.position.y, inner_rect.position.y + inner_rect.size.y, y);
					} else {
						rel_x = Math::inverse_lerp(ring_rect.position.x, ring_rect.position.x + ring_rect.size.x, x);
						rel_y = Math::inverse_lerp(ring_rect.position.y, ring_rect.position.y + ring_rect.size.y, y);
					}

					// Corner assignments for this algorithm
					//
					//  0   1
					//
					//  3   2
					//

					Color vtx_color_top;
					Color vtx_color_btm;

					if (inner_outer == 0) {
						vtx_color_top = inner_color_points[0].lerp(inner_color_points[1], rel_x);
						vtx_color_btm = inner_color_points[3].lerp(inner_color_points[2], rel_x);
					} else {
						vtx_color_top = outer_color_points[0].lerp(outer_color_points[1], rel_x);
						vtx_color_btm = outer_color_points[3].lerp(outer_color_points[2], rel_x);
					}
					vertex_color = vtx_color_top.lerp(vtx_color_btm, rel_y);
				} else {
					real_t color_blend_factor = corner_detail_color_blend[detail];
					vertex_color = color_a.lerp(color_b, color_blend_factor);
				}

				colors.push_back(vertex_color);
			}
		}
	}

	int ring_vert_count = verts.size() - vert_offset;

	// Fill the indices and the colors for the border.
	for (int i = 0; i < ring_vert_count; i++) {
		indices.push_back(vert_offset + ((i + 0) % ring_vert_count));
		indices.push_back(vert_offset + ((i + 2) % ring_vert_count));
		indices.push_back(vert_offset + ((i + 1) % ring_vert_count));
	}
}

inline void draw_filled_rounded_rect(Vector<Vector2> &verts, Vector<int> &indices, Vector<Color> &colors, const Rect2 &style_rect, const real_t corner_radius[4],
		const Rect2 &ring_rect, const Color corner_colors[4], const int corner_detail, const Vector2 &skew) {
	int vert_offset = verts.size();
	if (!vert_offset) {
		vert_offset = 0;
	}

	bool is_perfect_square = corner_radius[0] == 0 && corner_radius[1] == 0 && corner_radius[2] == 0 && corner_radius[3] == 0;
	int adapted_corner_detail = is_perfect_square ? 1 : corner_detail;

	real_t ring_corner_radius[4];
	set_inner_corner_radius(style_rect, ring_rect, corner_radius, ring_corner_radius);

	// Corner radius center points.
	Vector<Point2> outer_points = {
		ring_rect.position + Vector2(ring_corner_radius[0], ring_corner_radius[0]), //tl
		Point2(ring_rect.position.x + ring_rect.size.x - ring_corner_radius[1], ring_rect.position.y + ring_corner_radius[1]), //tr
		ring_rect.position + ring_rect.size - Vector2(ring_corner_radius[2], ring_corner_radius[2]), //br
		Point2(ring_rect.position.x + ring_corner_radius[3], ring_rect.position.y + ring_rect.size.y - ring_corner_radius[3]) //bl
	};

	for (int corner_index = 0; corner_index < 4; corner_index++) {
		real_t corner_detail_offsets[48]; // arbitrary for the maximum detail level
		for (int detail = 0; detail <= adapted_corner_detail; detail++) {
			corner_detail_offsets[detail] = detail / (double)adapted_corner_detail;
		}

		// Calculate the vertices.
		for (int detail = 0; detail <= adapted_corner_detail; detail++) {
			// Loop over successive vertices of this corner
			real_t radius;
			Point2 corner_point;
			radius = ring_corner_radius[corner_index];
			corner_point = outer_points[corner_index];

			// Trigonometric calc
			const real_t x = radius * (real_t)cos((corner_index + corner_detail_offsets[detail]) * (Math_TAU / 4.0) + Math_PI) + corner_point.x;
			const real_t y = radius * (real_t)sin((corner_index + corner_detail_offsets[detail]) * (Math_TAU / 4.0) + Math_PI) + corner_point.y;
			const float x_skew = -skew.x * (y - ring_rect.get_center().y);
			const float y_skew = -skew.y * (x - ring_rect.get_center().x);
			verts.push_back(Vector2(x + x_skew, y + y_skew));

			// Interpolate color from X Y position
			const real_t rel_x = Math::inverse_lerp(ring_rect.position.x, ring_rect.position.x + ring_rect.size.x, x);
			const real_t rel_y = Math::inverse_lerp(ring_rect.position.y, ring_rect.position.y + ring_rect.size.y, y);

			// Corner assignments for now
			//
			//  0   1
			//
			//  3   2
			//
			Color vtx_color_top = corner_colors[0].lerp(corner_colors[1], rel_x);
			Color vtx_color_btm = corner_colors[3].lerp(corner_colors[2], rel_x);
			Color vertex_color = vtx_color_top.lerp(vtx_color_btm, rel_y);
			colors.push_back(vertex_color);
		}
	}

	int ring_vert_count = verts.size() - vert_offset;

	// Compute the triangles pattern to draw the rounded rectangle.
	// Consists of vertical stripes of two triangles each.

	int stripes_count = ring_vert_count / 2 - 1;
	int last_vert_id = ring_vert_count - 1;

	for (int i = 0; i < stripes_count; i++) {
		// Polygon 1.
		indices.push_back(vert_offset + i);
		indices.push_back(vert_offset + last_vert_id - i - 1);
		indices.push_back(vert_offset + i + 1);
		// Polygon 2.
		indices.push_back(vert_offset + i);
		indices.push_back(vert_offset + last_vert_id - 0 - i);
		indices.push_back(vert_offset + last_vert_id - 1 - i);
	}
}

inline void adapt_values(int p_index_a, int p_index_b, real_t *adapted_values, const real_t *p_values, const real_t p_width, const real_t p_max_a, const real_t p_max_b) {
	if (p_values[p_index_a] + p_values[p_index_b] > p_width) {
		real_t factor;
		real_t new_value;

		factor = (real_t)p_width / (real_t)(p_values[p_index_a] + p_values[p_index_b]);

		new_value = (p_values[p_index_a] * factor);
		if (new_value < adapted_values[p_index_a]) {
			adapted_values[p_index_a] = new_value;
		}
		new_value = (p_values[p_index_b] * factor);
		if (new_value < adapted_values[p_index_b]) {
			adapted_values[p_index_b] = new_value;
		}
	} else {
		adapted_values[p_index_a] = MIN(p_values[p_index_a], adapted_values[p_index_a]);
		adapted_values[p_index_b] = MIN(p_values[p_index_b], adapted_values[p_index_b]);
	}
	adapted_values[p_index_a] = MIN(p_max_a, adapted_values[p_index_a]);
	adapted_values[p_index_b] = MIN(p_max_b, adapted_values[p_index_b]);
}

Rect2 StyleBoxFlat::get_draw_rect(const Rect2 &p_rect) const {
	Rect2 draw_rect = p_rect.grow_individual(expand_margin[SIDE_LEFT], expand_margin[SIDE_TOP], expand_margin[SIDE_RIGHT], expand_margin[SIDE_BOTTOM]);

	if (shadow_size > 0) {
		Rect2 shadow_rect = draw_rect.grow(shadow_size);
		shadow_rect.position += shadow_offset;
		draw_rect = draw_rect.merge(shadow_rect);
	}

	return draw_rect;
}

void StyleBoxFlat::draw(RID p_canvas_item, const Rect2 &p_rect) const {
	bool draw_border = (border_width[0] > 0) || (border_width[1] > 0) || (border_width[2] > 0) || (border_width[3] > 0);
	bool draw_shadow = (shadow_size > 0);
	if (!draw_border && !draw_center && !draw_shadow) {
		return;
	}

	Rect2 style_rect = p_rect.grow_individual(expand_margin[SIDE_LEFT], expand_margin[SIDE_TOP], expand_margin[SIDE_RIGHT], expand_margin[SIDE_BOTTOM]);
	if (Math::is_zero_approx(style_rect.size.width) || Math::is_zero_approx(style_rect.size.height)) {
		return;
	}

	const bool rounded_corners = (corner_radius[0] > 0) || (corner_radius[1] > 0) || (corner_radius[2] > 0) || (corner_radius[3] > 0);
	// Only enable antialiasing if it is actually needed. This improve performances
	// and maximizes sharpness for non-skewed StyleBoxes with sharp corners.
	const bool aa_on = (rounded_corners || !skew.is_zero_approx()) && anti_aliased;
	const bool blend_on = blend_border && draw_border;

	Color bg_colors[8] = {};
	Color bg_colors_4p[4] = {};
	Color border_colors[8] = {};
	Color border_alpha_colors[8] = {};
	Color border_blend_colors[8] = {};
	Color border_inner_colors[8] = {};

	for (int i = 0; i < 8; i++) {
		border_colors[i] = border_color;
	}

	// Reduce border and corner sizes so that they fit within the outer dimensions

	// Adapt borders (prevent weird overlapping/glitchy drawings).
	real_t width = MAX(style_rect.size.width, 0);
	real_t height = MAX(style_rect.size.height, 0);
	real_t adapted_border[4] = { 1000000.0, 1000000.0, 1000000.0, 1000000.0 };
	adapt_values(SIDE_TOP, SIDE_BOTTOM, adapted_border, border_width, height, height, height);
	adapt_values(SIDE_LEFT, SIDE_RIGHT, adapted_border, border_width, width, width, width);

	// Adapt corners (prevent weird overlapping/glitchy drawings).
	real_t adapted_corner[4] = { 1000000.0, 1000000.0, 1000000.0, 1000000.0 };
	adapt_values(CORNER_TOP_RIGHT, CORNER_BOTTOM_RIGHT, adapted_corner, corner_radius, height, height - adapted_border[SIDE_BOTTOM], height - adapted_border[SIDE_TOP]);
	adapt_values(CORNER_TOP_LEFT, CORNER_BOTTOM_LEFT, adapted_corner, corner_radius, height, height - adapted_border[SIDE_BOTTOM], height - adapted_border[SIDE_TOP]);
	adapt_values(CORNER_TOP_LEFT, CORNER_TOP_RIGHT, adapted_corner, corner_radius, width, width - adapted_border[SIDE_RIGHT], width - adapted_border[SIDE_LEFT]);
	adapt_values(CORNER_BOTTOM_LEFT, CORNER_BOTTOM_RIGHT, adapted_corner, corner_radius, width, width - adapted_border[SIDE_RIGHT], width - adapted_border[SIDE_LEFT]);

	Rect2 infill_rect = style_rect.grow_individual(-adapted_border[SIDE_LEFT], -adapted_border[SIDE_TOP], -adapted_border[SIDE_RIGHT], -adapted_border[SIDE_BOTTOM]);

	// Compute global border colors depending on the coloring style
	switch (border_coloring_style) {
		case BorderColoringStyle::BORDER_COLORING_STYLE_SOLID: {
			for (int i = 0; i < 8; i++) {
				border_colors[i] = border_color;
			}
			break;
		}
		case BorderColoringStyle::BORDER_COLORING_STYLE_SIMPLE_GRADIENT:
			[[fallthrough]];
		case BorderColoringStyle::BORDER_COLORING_STYLE_SIMPLE_2D_GRADIENT: {
			Color border_corner_colors[4] = {};

			if (border_coloring_style == BorderColoringStyle::BORDER_COLORING_STYLE_SIMPLE_2D_GRADIENT) {
				border_corner_colors[border_gradient_orientation] = border_color;
				border_corner_colors[(border_gradient_orientation + 1) % 4] = border_secondary_color;
				border_corner_colors[(border_gradient_orientation + 2) % 4] = border_fourth_color;
				border_corner_colors[(border_gradient_orientation + 3) % 4] = border_third_color;
			} else {
				border_corner_colors[border_gradient_orientation] = border_color;
				border_corner_colors[(border_gradient_orientation + 1) % 4] = border_secondary_color;
				border_corner_colors[(border_gradient_orientation + 2) % 4] = border_secondary_color;
				border_corner_colors[(border_gradient_orientation + 3) % 4] = border_color;
			}

			if (border_gradient_algo == GRADIENT_ALGO_INTERPOLATE_CORNER_LIMITS) {
				real_t points_rel_x[8] = {};
				real_t points_rel_y[8] = {};

				int ref_left = style_rect.position.x;
				int ref_right = style_rect.position.x + style_rect.size.x;
				int ref_top = style_rect.position.y;
				int ref_bottom = style_rect.position.y + style_rect.size.y;

				points_rel_x[LEFT_SIDE_TOP] = 0;
				points_rel_x[LEFT_SIDE_BOTTOM] = 0;
				points_rel_x[RIGHT_SIDE_TOP] = 1;
				points_rel_x[RIGHT_SIDE_BOTTOM] = 1;

				points_rel_x[TOP_SIDE_LEFT] = Math::inverse_lerp(ref_left, ref_right, ref_left + adapted_corner[CORNER_TOP_LEFT]);
				points_rel_x[BOTTOM_SIDE_LEFT] = Math::inverse_lerp(ref_left, ref_right, ref_left + adapted_corner[CORNER_BOTTOM_LEFT]);
				points_rel_x[TOP_SIDE_RIGHT] = Math::inverse_lerp(ref_left, ref_right, ref_right - adapted_corner[CORNER_TOP_RIGHT]);
				points_rel_x[BOTTOM_SIDE_RIGHT] = Math::inverse_lerp(ref_left, ref_right, ref_right - adapted_corner[CORNER_BOTTOM_RIGHT]);

				points_rel_y[TOP_SIDE_LEFT] = 0;
				points_rel_y[TOP_SIDE_RIGHT] = 0;
				points_rel_y[BOTTOM_SIDE_LEFT] = 1;
				points_rel_y[BOTTOM_SIDE_RIGHT] = 1;

				points_rel_y[LEFT_SIDE_TOP] = Math::inverse_lerp(ref_top, ref_bottom, ref_top + adapted_corner[CORNER_TOP_LEFT]);
				points_rel_y[RIGHT_SIDE_TOP] = Math::inverse_lerp(ref_top, ref_bottom, ref_top + adapted_corner[CORNER_TOP_RIGHT]);
				points_rel_y[LEFT_SIDE_BOTTOM] = Math::inverse_lerp(ref_top, ref_bottom, ref_bottom - adapted_corner[CORNER_BOTTOM_LEFT]);
				points_rel_y[RIGHT_SIDE_BOTTOM] = Math::inverse_lerp(ref_top, ref_bottom, ref_bottom - adapted_corner[CORNER_BOTTOM_RIGHT]);

				for (int i = 0; i < 8; i++) {
					Color lerp_color_top = border_corner_colors[0].lerp(border_corner_colors[1], points_rel_x[i]);
					Color lerp_color_btm = border_corner_colors[3].lerp(border_corner_colors[2], points_rel_x[i]);
					Color lerp_2d_color = lerp_color_top.lerp(lerp_color_btm, points_rel_y[i]);
					border_colors[i] = lerp_2d_color;
				}

			} else {
				// Simpler algo
				// Set full colors at the corner limits
				for (int i = 0; i < 4; i++) {
					border_colors[(i * 2 + 1) % 8] = border_corner_colors[i];
					border_colors[(i * 2 + 2) % 8] = border_corner_colors[i];
				}
			}

			break;
		}
		case BorderColoringStyle::BORDER_COLORING_STYLE_OUTSET:
			[[fallthrough]];
		case BorderColoringStyle::BORDER_COLORING_STYLE_INSET: {
			// Bevel lighting algorithm
			// Mix base color with the lighting color according to the lighting angle, and do the same with the
			// darkening color on the opposite direction

			Color light_blend = border_color.lerp(border_bevel_light_color, bevel_lighting_intensity);
			Color dark_blend = border_color.lerp(border_bevel_dark_color, bevel_darkening_intensity);

			// Compute each side's exposure to light or darkness and the resulting color
			Color side_colors[4] = { border_color, border_color, border_color, border_color };
			for (int i = 0; i < 4; i++) {
				real_t corresponding_side_angle = ((4 + 2 - i) % 4) * 90;
				real_t ref_angle = corresponding_side_angle - bevel_lighting_angle;
				if (ref_angle < 0) {
					ref_angle += 360;
				}
				if (ref_angle > 360) {
					ref_angle -= 360;
				}
				real_t linear_exposure = ref_angle <= 180 ? -(ref_angle / 90) + 1.0 : (ref_angle / 90) - 3.0;
				// Apply max angle ratio to "open up the cone" that corresponds to the max exposure on either side (ligther or darker)
				real_t ratio_to_remap = 1.0 - bevel_max_intensity_angle_ratio;
				real_t ref_exposure = CLAMP(Math::remap(linear_exposure, -ratio_to_remap, ratio_to_remap, -1.0, 1.0), -1.0, 1.0);
				side_colors[i] = ref_exposure >= 0 ? border_color.lerp(light_blend, ref_exposure) : border_color.lerp(dark_blend, -ref_exposure);
			}

			// Apply computed colors to borders
			int start_i = border_coloring_style == BorderColoringStyle::BORDER_COLORING_STYLE_INSET ? 4 : 0;
			for (int i = 0; i < 4; i++) {
				border_colors[(start_i + i * 2) % 8] = side_colors[i];
				border_colors[(start_i + i * 2 + 1) % 8] = side_colors[i];
			}
			break;
		}
	}
	switch (center_fill_style) {
		case FillStyle::FILL_STYLE_SOLID: {
			for (int i = 0; i < 8; i++) {
				bg_colors[i] = bg_color;
			}
			for (int i = 0; i < 4; i++) {
				bg_colors_4p[i] = bg_color;
			}
			break;
		}
		case FillStyle::FILL_STYLE_SIMPLE_GRADIENT: {
			int first_color_start_i = 7 + center_gradient_orientation * 2;
			int second_color_start_i = first_color_start_i + 4;
			for (int i = 0; i < 4; i++) {
				bg_colors[(i + first_color_start_i) % 8] = bg_color;
				bg_colors[(i + second_color_start_i) % 8] = bg_secondary_color;
			}
			bg_colors_4p[CORNER_TOP_LEFT] = bg_colors[1];
			bg_colors_4p[CORNER_TOP_RIGHT] = bg_colors[4];
			bg_colors_4p[CORNER_BOTTOM_RIGHT] = bg_colors[5];
			bg_colors_4p[CORNER_BOTTOM_LEFT] = bg_colors[0];
			break;
		}
	}

	// Compute individual border point color overrides according to per-border settings
	for (int i = 0; i < 4; i++) {
		if (border_side_colors_defined[i]) {
			border_colors[i * 2] = border_side_colors[i];
			border_colors[i * 2 + 1] = border_side_colors[i];
			if (border_side_coloring_styles[i] == BORDER_SIDE_COLORING_STYLE_SIMPLE_GRADIENT) {
				int secondary_end_index = (i == SIDE_LEFT || i == SIDE_BOTTOM) ? i * 2 : i * 2 + 1;
				border_colors[secondary_end_index] = border_side_secondary_colors[i];
			}
		}
	}
	for (int i = 0; i < 8; i++) {
		border_alpha_colors[i] = Color(border_colors[i].r, border_colors[i].g, border_colors[i].b, 0);
		border_blend_colors[i] = draw_center ? bg_colors[i] : border_alpha_colors[i];
		border_inner_colors[i] = blend_on ? border_blend_colors[i] : border_colors[i];
	}
	bool use_alt_algo_for_inner_border = false;
	if (blend_on && draw_center) {
		for (int i = 0; i < 4; i++) {
			border_inner_colors[i] = bg_colors_4p[i];
		}
		use_alt_algo_for_inner_border = true;
	}

	const bool use_sharp_joins = (color_join_style == BORDER_COLOR_JOIN_STYLE_SHARP);
	real_t join_angles[4] = { corner_color_join_angles[CORNER_TOP_LEFT], corner_color_join_angles[CORNER_TOP_RIGHT], corner_color_join_angles[CORNER_BOTTOM_RIGHT], corner_color_join_angles[CORNER_BOTTOM_LEFT] };

	Rect2 border_style_rect = style_rect;
	if (aa_on) {
		for (int i = 0; i < 4; i++) {
			if (border_width[i] > 0) {
				border_style_rect = border_style_rect.grow_side((Side)i, -aa_size);
			}
		}
	}

	Vector<Point2> verts;
	Vector<int> indices;
	Vector<Color> colors;
	Vector<Point2> uvs;

	// Create shadow
	if (draw_shadow) {
		Rect2 shadow_inner_rect = style_rect;
		shadow_inner_rect.position += shadow_offset;

		Rect2 shadow_rect = style_rect.grow(shadow_size);
		shadow_rect.position += shadow_offset;

		Color shadow_color_transparent = Color(shadow_color.r, shadow_color.g, shadow_color.b, 0);

		Color shadow_colors[8] = {};
		Color shadow_colors_4p[4] = {};
		Color shadow_transparent_colors[8] = {};
		for (int i = 0; i < 8; i++) {
			shadow_colors[i] = shadow_color;
			shadow_transparent_colors[i] = shadow_color_transparent;
		}
		for (int i = 0; i < 4; i++) {
			shadow_colors_4p[i] = shadow_color;
		}

		draw_ring(verts, indices, colors, shadow_inner_rect, adapted_corner,
				join_angles,
				shadow_rect, shadow_inner_rect, shadow_colors,
				shadow_transparent_colors,
				corner_detail, skew, false, false, false);

		if (draw_center) {
			draw_filled_rounded_rect(verts, indices, colors, shadow_inner_rect, adapted_corner,
					shadow_inner_rect, shadow_colors_4p,
					corner_detail, skew);
		}
	}

	// Create border (no AA).
	if (draw_border && !aa_on) {
		draw_ring(verts, indices, colors, border_style_rect, adapted_corner,
				join_angles,
				border_style_rect, infill_rect,
				border_inner_colors, border_colors,
				corner_detail, skew, use_sharp_joins, use_alt_algo_for_inner_border, false);
	}

	// Create infill (no AA, or if border blending mode is active as there is no need to antialias the center in this case).
	if (draw_center && (!aa_on || blend_on)) {
		draw_filled_rounded_rect(verts, indices, colors, border_style_rect, adapted_corner,
				infill_rect, bg_colors_4p, corner_detail, skew);
	}

	if (aa_on) {
		real_t aa_border_width[4];
		real_t aa_border_width_half[4];
		real_t aa_fill_width[4];
		real_t aa_fill_width_half[4];
		if (draw_border) {
			for (int i = 0; i < 4; i++) {
				if (border_width[i] > 0) {
					aa_border_width[i] = aa_size;
					aa_border_width_half[i] = aa_size / 2;
					aa_fill_width[i] = 0;
					aa_fill_width_half[i] = 0;
				} else {
					aa_border_width[i] = 0;
					aa_border_width_half[i] = 0;
					aa_fill_width[i] = aa_size;
					aa_fill_width_half[i] = aa_size / 2;
				}
			}
		} else {
			for (int i = 0; i < 4; i++) {
				aa_border_width[i] = 0;
				aa_border_width_half[i] = 0;
				aa_fill_width[i] = aa_size;
				aa_fill_width_half[i] = aa_size / 2;
			}
		}

		if (draw_center) {
			// Infill rect, transparent side of antialiasing gradient (base infill rect enlarged by AA size)
			Rect2 infill_rect_aa_transparent = infill_rect.grow_individual(aa_fill_width_half[SIDE_LEFT], aa_fill_width_half[SIDE_TOP],
					aa_fill_width_half[SIDE_RIGHT], aa_fill_width_half[SIDE_BOTTOM]);
			// Infill rect, colored side of antialiasing gradient (base infill rect shrunk by AA size)
			Rect2 infill_rect_aa_colored = infill_rect_aa_transparent.grow_individual(-aa_fill_width[SIDE_LEFT], -aa_fill_width[SIDE_TOP],
					-aa_fill_width[SIDE_RIGHT], -aa_fill_width[SIDE_BOTTOM]);
			if (!blend_on) {
				// Create center fill, not antialiased yet
				draw_filled_rounded_rect(verts, indices, colors, border_style_rect, adapted_corner,
						infill_rect_aa_colored, bg_colors_4p,
						corner_detail, skew);
			}
			if (!blend_on || !draw_border) {
				Color alpha_bg_colors_4p[4] = {};
				for (int i = 0; i < 4; i++) {
					alpha_bg_colors_4p[i] = Color(bg_colors_4p[i].r, bg_colors_4p[i].g, bg_colors_4p[i].b, 0);
				}

				// Create infill fake AA gradient.
				draw_ring(verts, indices, colors, border_style_rect, adapted_corner,
						join_angles,
						infill_rect_aa_transparent, infill_rect_aa_colored, bg_colors_4p,
						alpha_bg_colors_4p, corner_detail, skew, false, true, true);
			}
		}

		if (draw_border) {
			// Inner border recct, fully colored side of antialiasing gradient (base inner rect enlarged by AA size)
			Rect2 inner_rect_aa_colored = infill_rect.grow_individual(aa_border_width_half[SIDE_LEFT], aa_border_width_half[SIDE_TOP],
					aa_border_width_half[SIDE_RIGHT], aa_border_width_half[SIDE_BOTTOM]);
			// Inner border rect, transparent side of antialiasing gradient (base inner rect shrunk by AA size)
			Rect2 inner_rect_aa_transparent = inner_rect_aa_colored.grow_individual(-aa_border_width[SIDE_LEFT], -aa_border_width[SIDE_TOP],
					-aa_border_width[SIDE_RIGHT], -aa_border_width[SIDE_BOTTOM]);
			// Outer border rect, transparent side of antialiasing gradient (base outer rect enlarged by AA size)
			Rect2 outer_rect_aa_transparent = style_rect.grow_individual(aa_border_width_half[SIDE_LEFT], aa_border_width_half[SIDE_TOP],
					aa_border_width_half[SIDE_RIGHT], aa_border_width_half[SIDE_BOTTOM]);
			// Outer border rect, colored side of antialiasing gradient (base outer rect shrunk by AA size)
			Rect2 outer_rect_aa_colored = border_style_rect.grow_individual(aa_border_width_half[SIDE_LEFT], aa_border_width_half[SIDE_TOP],
					aa_border_width_half[SIDE_RIGHT], aa_border_width_half[SIDE_BOTTOM]);

			// Create border ring, not antialiased yet
			draw_ring(verts, indices, colors, border_style_rect, adapted_corner,
					join_angles,
					outer_rect_aa_colored, ((blend_on) ? infill_rect : inner_rect_aa_colored),
					border_inner_colors,
					border_colors, corner_detail, skew, use_sharp_joins, use_alt_algo_for_inner_border, false);

			if (!blend_on) {
				// Add antialiasing on the ring inner border
				draw_ring(verts, indices, colors, border_style_rect, adapted_corner,
						join_angles,
						inner_rect_aa_colored, inner_rect_aa_transparent, border_blend_colors,
						border_colors, corner_detail, skew, use_sharp_joins, false, false);
			}
			// Add antialiasing on the ring outer border
			draw_ring(verts, indices, colors, border_style_rect, adapted_corner,
					join_angles,
					outer_rect_aa_transparent, outer_rect_aa_colored, border_colors,
					border_alpha_colors, corner_detail, skew, use_sharp_joins, false, false);
		}
	}

	// Compute UV coordinates.
	Rect2 uv_rect = style_rect.grow(aa_on ? aa_size : 0);
	uvs.resize(verts.size());
	for (int i = 0; i < verts.size(); i++) {
		uvs.write[i].x = (verts[i].x - uv_rect.position.x) / uv_rect.size.width;
		uvs.write[i].y = (verts[i].y - uv_rect.position.y) / uv_rect.size.height;
	}

	// Draw stylebox.
	RenderingServer *vs = RenderingServer::get_singleton();
	vs->canvas_item_add_triangle_array(p_canvas_item, indices, verts, colors, uvs);
}

float StyleBoxFlat::get_style_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0.0);
	return border_width[p_side];
}

void StyleBoxFlat::_validate_property(PropertyInfo &p_property) const {
	if (!anti_aliased && p_property.name == "anti_aliasing_size") {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
	// Choices that make sense on specific coloring styles only
	if ((p_property.name == "bg_secondary_color" || p_property.name == "bg_gradient_orientation") && center_fill_style != FILL_STYLE_SIMPLE_GRADIENT) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
	if ((p_property.name == "border_secondary_color" || p_property.name == "border_gradient_orientation" ||
				p_property.name == "border_gradient_algorithm") &&
			border_coloring_style != BORDER_COLORING_STYLE_SIMPLE_GRADIENT &&
			border_coloring_style != BORDER_COLORING_STYLE_SIMPLE_2D_GRADIENT) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
	if ((p_property.name == "border_third_color" || p_property.name == "border_fourth_color") && border_coloring_style != BORDER_COLORING_STYLE_SIMPLE_2D_GRADIENT) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
	if (border_coloring_style != BORDER_COLORING_STYLE_OUTSET && border_coloring_style != BORDER_COLORING_STYLE_INSET &&
			p_property.name.begins_with("border_bevel_")) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
	// Properties that are relevant if the corresponding border customization is enabled only
	if ((!border_side_colors_defined[0] && p_property.name == "border_left_color") ||
			(!border_side_colors_defined[1] && p_property.name == "border_top_color") ||
			(!border_side_colors_defined[2] && p_property.name == "border_right_color") ||
			(!border_side_colors_defined[3] && p_property.name == "border_bottom_color")) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
	if ((!border_side_colors_defined[0] && p_property.name == "border_left_coloring_style") ||
			(!border_side_colors_defined[1] && p_property.name == "border_top_coloring_style") ||
			(!border_side_colors_defined[2] && p_property.name == "border_right_coloring_style") ||
			(!border_side_colors_defined[3] && p_property.name == "border_bottom_coloring_style")) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
	// Secondary colors are shown if the border coloring style requires it
	if (((!border_side_colors_defined[0] || border_side_coloring_styles[0] == BORDER_SIDE_COLORING_STYLE_SOLID) && p_property.name == "border_left_secondary_color") ||
			((!border_side_colors_defined[1] || border_side_coloring_styles[1] == BORDER_SIDE_COLORING_STYLE_SOLID) && p_property.name == "border_top_secondary_color") ||
			((!border_side_colors_defined[2] || border_side_coloring_styles[2] == BORDER_SIDE_COLORING_STYLE_SOLID) && p_property.name == "border_right_secondary_color") ||
			((!border_side_colors_defined[3] || border_side_coloring_styles[3] == BORDER_SIDE_COLORING_STYLE_SOLID) && p_property.name == "border_bottom_secondary_color")) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

void StyleBoxFlat::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bg_color", "color"), &StyleBoxFlat::set_bg_color);
	ClassDB::bind_method(D_METHOD("get_bg_color"), &StyleBoxFlat::get_bg_color);

	ClassDB::bind_method(D_METHOD("set_bg_secondary_color", "color"), &StyleBoxFlat::set_bg_secondary_color);
	ClassDB::bind_method(D_METHOD("get_bg_secondary_color"), &StyleBoxFlat::get_bg_secondary_color);

	ClassDB::bind_method(D_METHOD("set_bg_simple_gradient_orientation", "orientation"), &StyleBoxFlat::set_bg_simple_gradient_orientation);
	ClassDB::bind_method(D_METHOD("get_bg_simple_gradient_orientation"), &StyleBoxFlat::get_bg_simple_gradient_orientation);

	ClassDB::bind_method(D_METHOD("set_center_fill_style", "fill_style"), &StyleBoxFlat::set_center_fill_style);
	ClassDB::bind_method(D_METHOD("get_center_fill_style"), &StyleBoxFlat::get_center_fill_style);

	ClassDB::bind_method(D_METHOD("set_border_color", "color"), &StyleBoxFlat::set_border_color);
	ClassDB::bind_method(D_METHOD("get_border_color"), &StyleBoxFlat::get_border_color);

	ClassDB::bind_method(D_METHOD("set_border_secondary_color", "color"), &StyleBoxFlat::set_border_secondary_color);
	ClassDB::bind_method(D_METHOD("get_border_secondary_color"), &StyleBoxFlat::get_border_secondary_color);

	ClassDB::bind_method(D_METHOD("set_border_third_color", "color"), &StyleBoxFlat::set_border_third_color);
	ClassDB::bind_method(D_METHOD("get_border_third_color"), &StyleBoxFlat::get_border_third_color);

	ClassDB::bind_method(D_METHOD("set_border_fourth_color", "color"), &StyleBoxFlat::set_border_fourth_color);
	ClassDB::bind_method(D_METHOD("get_border_fourth_color"), &StyleBoxFlat::get_border_fourth_color);

	ClassDB::bind_method(D_METHOD("set_border_simple_gradient_orientation", "orientation"), &StyleBoxFlat::set_border_simple_gradient_orientation);
	ClassDB::bind_method(D_METHOD("get_border_simple_gradient_orientation"), &StyleBoxFlat::get_border_simple_gradient_orientation);

	ClassDB::bind_method(D_METHOD("set_border_simple_gradient_algorithm", "algorithm"), &StyleBoxFlat::set_border_simple_gradient_algorithm);
	ClassDB::bind_method(D_METHOD("get_border_simple_gradient_algorithm"), &StyleBoxFlat::get_border_simple_gradient_algorithm);

	ClassDB::bind_method(D_METHOD("set_border_bevel_lighting_color", "color"), &StyleBoxFlat::set_border_bevel_lighting_color);
	ClassDB::bind_method(D_METHOD("get_border_bevel_lighting_color"), &StyleBoxFlat::get_border_bevel_lighting_color);

	ClassDB::bind_method(D_METHOD("set_border_bevel_darkening_color", "color"), &StyleBoxFlat::set_border_bevel_darkening_color);
	ClassDB::bind_method(D_METHOD("get_border_bevel_darkening_color"), &StyleBoxFlat::get_border_bevel_darkening_color);

	ClassDB::bind_method(D_METHOD("set_border_bevel_lighting_intensity", "intensity"), &StyleBoxFlat::set_border_bevel_lighting_intensity);
	ClassDB::bind_method(D_METHOD("get_border_bevel_lighting_intensity"), &StyleBoxFlat::get_border_bevel_lighting_intensity);

	ClassDB::bind_method(D_METHOD("set_border_bevel_darkening_intensity", "intensity"), &StyleBoxFlat::set_border_bevel_darkening_intensity);
	ClassDB::bind_method(D_METHOD("get_border_bevel_darkening_intensity"), &StyleBoxFlat::get_border_bevel_darkening_intensity);

	ClassDB::bind_method(D_METHOD("set_border_bevel_lighting_angle", "angle"), &StyleBoxFlat::set_border_bevel_lighting_angle);
	ClassDB::bind_method(D_METHOD("get_border_bevel_lighting_angle"), &StyleBoxFlat::get_border_bevel_lighting_angle);

	ClassDB::bind_method(D_METHOD("set_border_bevel_max_intensity_angle_ratio", "angle"), &StyleBoxFlat::set_border_bevel_max_intensity_angle_ratio);
	ClassDB::bind_method(D_METHOD("get_border_bevel_max_intensity_angle_ratio"), &StyleBoxFlat::get_border_bevel_max_intensity_angle_ratio);

	ClassDB::bind_method(D_METHOD("set_border_side_color", "side", "color"), &StyleBoxFlat::set_border_side_color);
	ClassDB::bind_method(D_METHOD("set_border_side_color_all", "color"), &StyleBoxFlat::set_border_side_color_all);
	ClassDB::bind_method(D_METHOD("get_border_side_color", "side"), &StyleBoxFlat::get_border_side_color);

	ClassDB::bind_method(D_METHOD("set_border_side_secondary_color", "side", "color"), &StyleBoxFlat::set_border_side_secondary_color);
	ClassDB::bind_method(D_METHOD("set_border_side_secondary_color_all", "color"), &StyleBoxFlat::set_border_side_secondary_color_all);
	ClassDB::bind_method(D_METHOD("get_border_side_secondary_color", "side"), &StyleBoxFlat::get_border_side_secondary_color);

	ClassDB::bind_method(D_METHOD("set_border_side_color_defined", "side", "defined"), &StyleBoxFlat::set_border_side_color_defined);
	ClassDB::bind_method(D_METHOD("get_border_side_color_defined", "side"), &StyleBoxFlat::get_border_side_color_defined);

	ClassDB::bind_method(D_METHOD("set_border_side_coloring_style", "side", "coloring_style"), &StyleBoxFlat::set_border_side_coloring_style);
	ClassDB::bind_method(D_METHOD("get_border_side_coloring_style", "side"), &StyleBoxFlat::get_border_side_coloring_style);

	ClassDB::bind_method(D_METHOD("set_corner_color_join_angle", "corner", "angle"), &StyleBoxFlat::set_corner_color_join_angle);
	ClassDB::bind_method(D_METHOD("get_corner_color_join_angle", "corner"), &StyleBoxFlat::get_corner_color_join_angle);

	ClassDB::bind_method(D_METHOD("set_border_width_all", "width"), &StyleBoxFlat::set_border_width_all);
	ClassDB::bind_method(D_METHOD("get_border_width_min"), &StyleBoxFlat::get_border_width_min);

	ClassDB::bind_method(D_METHOD("set_border_width", "margin", "width"), &StyleBoxFlat::set_border_width);
	ClassDB::bind_method(D_METHOD("get_border_width", "margin"), &StyleBoxFlat::get_border_width);

	ClassDB::bind_method(D_METHOD("set_border_blend", "blend"), &StyleBoxFlat::set_border_blend);
	ClassDB::bind_method(D_METHOD("get_border_blend"), &StyleBoxFlat::get_border_blend);

	ClassDB::bind_method(D_METHOD("set_border_coloring_style", "coloring_style"), &StyleBoxFlat::set_border_coloring_style);
	ClassDB::bind_method(D_METHOD("get_border_coloring_style"), &StyleBoxFlat::get_border_coloring_style);

	ClassDB::bind_method(D_METHOD("set_border_color_join_style", "join_style"), &StyleBoxFlat::set_border_color_join_style);
	ClassDB::bind_method(D_METHOD("get_border_color_join_style"), &StyleBoxFlat::get_border_color_join_style);

	ClassDB::bind_method(D_METHOD("set_corner_radius_all", "radius"), &StyleBoxFlat::set_corner_radius_all);

	ClassDB::bind_method(D_METHOD("set_corner_radius", "corner", "radius"), &StyleBoxFlat::set_corner_radius);
	ClassDB::bind_method(D_METHOD("get_corner_radius", "corner"), &StyleBoxFlat::get_corner_radius);

	ClassDB::bind_method(D_METHOD("set_expand_margin", "margin", "size"), &StyleBoxFlat::set_expand_margin);
	ClassDB::bind_method(D_METHOD("set_expand_margin_all", "size"), &StyleBoxFlat::set_expand_margin_all);
	ClassDB::bind_method(D_METHOD("get_expand_margin", "margin"), &StyleBoxFlat::get_expand_margin);

	ClassDB::bind_method(D_METHOD("set_draw_center", "draw_center"), &StyleBoxFlat::set_draw_center);
	ClassDB::bind_method(D_METHOD("is_draw_center_enabled"), &StyleBoxFlat::is_draw_center_enabled);

	ClassDB::bind_method(D_METHOD("set_skew", "skew"), &StyleBoxFlat::set_skew);
	ClassDB::bind_method(D_METHOD("get_skew"), &StyleBoxFlat::get_skew);

	ClassDB::bind_method(D_METHOD("set_shadow_color", "color"), &StyleBoxFlat::set_shadow_color);
	ClassDB::bind_method(D_METHOD("get_shadow_color"), &StyleBoxFlat::get_shadow_color);

	ClassDB::bind_method(D_METHOD("set_shadow_size", "size"), &StyleBoxFlat::set_shadow_size);
	ClassDB::bind_method(D_METHOD("get_shadow_size"), &StyleBoxFlat::get_shadow_size);

	ClassDB::bind_method(D_METHOD("set_shadow_offset", "offset"), &StyleBoxFlat::set_shadow_offset);
	ClassDB::bind_method(D_METHOD("get_shadow_offset"), &StyleBoxFlat::get_shadow_offset);

	ClassDB::bind_method(D_METHOD("set_anti_aliased", "anti_aliased"), &StyleBoxFlat::set_anti_aliased);
	ClassDB::bind_method(D_METHOD("is_anti_aliased"), &StyleBoxFlat::is_anti_aliased);

	ClassDB::bind_method(D_METHOD("set_aa_size", "size"), &StyleBoxFlat::set_aa_size);
	ClassDB::bind_method(D_METHOD("get_aa_size"), &StyleBoxFlat::get_aa_size);

	ClassDB::bind_method(D_METHOD("set_corner_detail", "detail"), &StyleBoxFlat::set_corner_detail);
	ClassDB::bind_method(D_METHOD("get_corner_detail"), &StyleBoxFlat::get_corner_detail);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_center"), "set_draw_center", "is_draw_center_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "skew"), "set_skew", "get_skew");

	ADD_GROUP("Background", "bg_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bg_fill_style", PROPERTY_HINT_ENUM, "Solid,Gradient"), "set_center_fill_style", "get_center_fill_style");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "bg_color"), "set_bg_color", "get_bg_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "bg_secondary_color"), "set_bg_secondary_color", "get_bg_secondary_color");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bg_gradient_orientation", PROPERTY_HINT_ENUM, "Right,Down,Left,Up"), "set_bg_simple_gradient_orientation", "get_bg_simple_gradient_orientation");

	ADD_GROUP("Border", "border_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "border_coloring_style", PROPERTY_HINT_ENUM, "Solid,Gradient,2D Gradient,Outset,Inset"), "set_border_coloring_style", "get_border_coloring_style");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_color"), "set_border_color", "get_border_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_secondary_color"), "set_border_secondary_color", "get_border_secondary_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_third_color"), "set_border_third_color", "get_border_third_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_fourth_color"), "set_border_fourth_color", "get_border_fourth_color");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "border_gradient_orientation", PROPERTY_HINT_ENUM, "Right,Down,Left,Up"), "set_border_simple_gradient_orientation", "get_border_simple_gradient_orientation");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "border_gradient_algorithm", PROPERTY_HINT_ENUM, "Interpolate Corner Limits,Full Corner Limits"), "set_border_simple_gradient_algorithm", "get_border_simple_gradient_algorithm");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_bevel_lighting_color"), "set_border_bevel_lighting_color", "get_border_bevel_lighting_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_bevel_darkening_color"), "set_border_bevel_darkening_color", "get_border_bevel_darkening_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "border_bevel_lighting_angle", PROPERTY_HINT_RANGE, "0.0,360.0,0.1,slider,degrees"), "set_border_bevel_lighting_angle", "get_border_bevel_lighting_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "border_bevel_lighting_intensity", PROPERTY_HINT_RANGE, "0,4,0.01"), "set_border_bevel_lighting_intensity", "get_border_bevel_lighting_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "border_bevel_darkening_intensity", PROPERTY_HINT_RANGE, "0,4,0.01"), "set_border_bevel_darkening_intensity", "get_border_bevel_darkening_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "border_bevel_max_intensity_angle_ratio", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_border_bevel_max_intensity_angle_ratio", "get_border_bevel_max_intensity_angle_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "border_blend"), "set_border_blend", "get_border_blend");

	ADD_GROUP("Border Width", "border_width_");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_width_left", PROPERTY_HINT_RANGE, "0,1024,1,suffix:px"), "set_border_width", "get_border_width", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_width_top", PROPERTY_HINT_RANGE, "0,1024,1,suffix:px"), "set_border_width", "get_border_width", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_width_right", PROPERTY_HINT_RANGE, "0,1024,1,suffix:px"), "set_border_width", "get_border_width", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_width_bottom", PROPERTY_HINT_RANGE, "0,1024,1,suffix:px"), "set_border_width", "get_border_width", SIDE_BOTTOM);

	ADD_GROUP("Border Left", "border_left_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "border_left_specify_color"), "set_border_side_color_defined", "get_border_side_color_defined", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_left_coloring_style", PROPERTY_HINT_ENUM, "Solid,Gradient"), "set_border_side_coloring_style", "get_border_side_coloring_style", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::COLOR, "border_left_color"), "set_border_side_color", "get_border_side_color", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::COLOR, "border_left_secondary_color"), "set_border_side_secondary_color", "get_border_side_secondary_color", SIDE_LEFT);

	ADD_GROUP("Border Top", "border_top_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "border_top_specify_color"), "set_border_side_color_defined", "get_border_side_color_defined", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_top_coloring_style", PROPERTY_HINT_ENUM, "Solid,Gradient"), "set_border_side_coloring_style", "get_border_side_coloring_style", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::COLOR, "border_top_color"), "set_border_side_color", "get_border_side_color", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::COLOR, "border_top_secondary_color"), "set_border_side_secondary_color", "get_border_side_secondary_color", SIDE_TOP);

	ADD_GROUP("Border Right", "border_right_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "border_right_specify_color"), "set_border_side_color_defined", "get_border_side_color_defined", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_right_coloring_style", PROPERTY_HINT_ENUM, "Solid,Gradient"), "set_border_side_coloring_style", "get_border_side_coloring_style", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::COLOR, "border_right_color"), "set_border_side_color", "get_border_side_color", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::COLOR, "border_right_secondary_color"), "set_border_side_secondary_color", "get_border_side_secondary_color", SIDE_RIGHT);

	ADD_GROUP("Border Bottom", "border_bottom_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "border_bottom_specify_color"), "set_border_side_color_defined", "get_border_side_color_defined", SIDE_BOTTOM);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_bottom_coloring_style", PROPERTY_HINT_ENUM, "Solid,Gradient"), "set_border_side_coloring_style", "get_border_side_coloring_style", SIDE_BOTTOM);
	ADD_PROPERTYI(PropertyInfo(Variant::COLOR, "border_bottom_color"), "set_border_side_color", "get_border_side_color", SIDE_BOTTOM);
	ADD_PROPERTYI(PropertyInfo(Variant::COLOR, "border_bottom_secondary_color"), "set_border_side_secondary_color", "get_border_side_secondary_color", SIDE_BOTTOM);

	ADD_GROUP("Corners", "corner_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "corner_detail", PROPERTY_HINT_RANGE, "1,20,1"), "set_corner_detail", "get_corner_detail");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "corner_color_join_style", PROPERTY_HINT_ENUM, "Smooth,Sharp"), "set_border_color_join_style", "get_border_color_join_style");

	ADD_GROUP("Corner Radius", "corner_radius_");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "corner_radius_top_left", PROPERTY_HINT_RANGE, "0,1024,1,suffix:px"), "set_corner_radius", "get_corner_radius", CORNER_TOP_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "corner_radius_top_right", PROPERTY_HINT_RANGE, "0,1024,1,suffix:px"), "set_corner_radius", "get_corner_radius", CORNER_TOP_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "corner_radius_bottom_right", PROPERTY_HINT_RANGE, "0,1024,1,suffix:px"), "set_corner_radius", "get_corner_radius", CORNER_BOTTOM_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "corner_radius_bottom_left", PROPERTY_HINT_RANGE, "0,1024,1,suffix:px"), "set_corner_radius", "get_corner_radius", CORNER_BOTTOM_LEFT);

	ADD_GROUP("Corner Color Join Angles", "corner_color_join_angle_");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "corner_color_join_angle_top_left", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_corner_color_join_angle", "get_corner_color_join_angle", CORNER_TOP_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "corner_color_join_angle_top_right", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_corner_color_join_angle", "get_corner_color_join_angle", CORNER_TOP_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "corner_color_join_angle_bottom_right", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_corner_color_join_angle", "get_corner_color_join_angle", CORNER_BOTTOM_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "corner_color_join_angle_bottom_left", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_corner_color_join_angle", "get_corner_color_join_angle", CORNER_BOTTOM_LEFT);

	ADD_GROUP("Expand Margins", "expand_margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "expand_margin_left", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_expand_margin", "get_expand_margin", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "expand_margin_top", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_expand_margin", "get_expand_margin", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "expand_margin_right", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_expand_margin", "get_expand_margin", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "expand_margin_bottom", PROPERTY_HINT_RANGE, "0,2048,1,suffix:px"), "set_expand_margin", "get_expand_margin", SIDE_BOTTOM);

	ADD_GROUP("Shadow", "shadow_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "shadow_color"), "set_shadow_color", "get_shadow_color");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_size", PROPERTY_HINT_RANGE, "0,100,1,or_greater,suffix:px"), "set_shadow_size", "get_shadow_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "shadow_offset", PROPERTY_HINT_NONE, "suffix:px"), "set_shadow_offset", "get_shadow_offset");

	ADD_GROUP("Anti Aliasing", "anti_aliasing_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "anti_aliasing"), "set_anti_aliased", "is_anti_aliased");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "anti_aliasing_size", PROPERTY_HINT_RANGE, "0.01,10,0.001,suffix:px"), "set_aa_size", "get_aa_size");

	BIND_ENUM_CONSTANT(BORDER_COLOR_JOIN_STYLE_GRADIENT);
	BIND_ENUM_CONSTANT(BORDER_COLOR_JOIN_STYLE_SHARP);

	BIND_ENUM_CONSTANT(BORDER_SIDE_COLORING_STYLE_SOLID);
	BIND_ENUM_CONSTANT(BORDER_SIDE_COLORING_STYLE_SIMPLE_GRADIENT);

	BIND_ENUM_CONSTANT(BORDER_COLORING_STYLE_SOLID);
	BIND_ENUM_CONSTANT(BORDER_COLORING_STYLE_SIMPLE_GRADIENT);
	BIND_ENUM_CONSTANT(BORDER_COLORING_STYLE_SIMPLE_2D_GRADIENT);
	BIND_ENUM_CONSTANT(BORDER_COLORING_STYLE_OUTSET);
	BIND_ENUM_CONSTANT(BORDER_COLORING_STYLE_INSET);

	BIND_ENUM_CONSTANT(FILL_STYLE_SOLID);
	BIND_ENUM_CONSTANT(FILL_STYLE_SIMPLE_GRADIENT);

	BIND_ENUM_CONSTANT(SIMPLE_GRADIENT_ORIENTATION_LEFT_TO_RIGHT);
	BIND_ENUM_CONSTANT(SIMPLE_GRADIENT_ORIENTATION_TOP_TO_BOTTOM);
	BIND_ENUM_CONSTANT(SIMPLE_GRADIENT_ORIENTATION_RIGHT_TO_LEFT);
	BIND_ENUM_CONSTANT(SIMPLE_GRADIENT_ORIENTATION_BOTTOM_TO_TOP);

	BIND_ENUM_CONSTANT(GRADIENT_ALGO_INTERPOLATE_CORNER_LIMITS);
	BIND_ENUM_CONSTANT(GRADIENT_ALGO_FULL_CORNER_LIMITS);
}

StyleBoxFlat::StyleBoxFlat() {}

StyleBoxFlat::~StyleBoxFlat() {}

void StyleBoxLine::set_color(const Color &p_color) {
	color = p_color;
	emit_changed();
}

Color StyleBoxLine::get_color() const {
	return color;
}

void StyleBoxLine::set_thickness(int p_thickness) {
	thickness = p_thickness;
	emit_changed();
}

int StyleBoxLine::get_thickness() const {
	return thickness;
}

void StyleBoxLine::set_vertical(bool p_vertical) {
	vertical = p_vertical;
	emit_changed();
}

bool StyleBoxLine::is_vertical() const {
	return vertical;
}

void StyleBoxLine::set_grow_end(float p_grow_end) {
	grow_end = p_grow_end;
	emit_changed();
}

float StyleBoxLine::get_grow_end() const {
	return grow_end;
}

void StyleBoxLine::set_grow_begin(float p_grow_begin) {
	grow_begin = p_grow_begin;
	emit_changed();
}

float StyleBoxLine::get_grow_begin() const {
	return grow_begin;
}

void StyleBoxLine::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_color", "color"), &StyleBoxLine::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &StyleBoxLine::get_color);
	ClassDB::bind_method(D_METHOD("set_thickness", "thickness"), &StyleBoxLine::set_thickness);
	ClassDB::bind_method(D_METHOD("get_thickness"), &StyleBoxLine::get_thickness);
	ClassDB::bind_method(D_METHOD("set_grow_begin", "offset"), &StyleBoxLine::set_grow_begin);
	ClassDB::bind_method(D_METHOD("get_grow_begin"), &StyleBoxLine::get_grow_begin);
	ClassDB::bind_method(D_METHOD("set_grow_end", "offset"), &StyleBoxLine::set_grow_end);
	ClassDB::bind_method(D_METHOD("get_grow_end"), &StyleBoxLine::get_grow_end);
	ClassDB::bind_method(D_METHOD("set_vertical", "vertical"), &StyleBoxLine::set_vertical);
	ClassDB::bind_method(D_METHOD("is_vertical"), &StyleBoxLine::is_vertical);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "grow_begin", PROPERTY_HINT_RANGE, "-300,300,1,suffix:px"), "set_grow_begin", "get_grow_begin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "grow_end", PROPERTY_HINT_RANGE, "-300,300,1,suffix:px"), "set_grow_end", "get_grow_end");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "thickness", PROPERTY_HINT_RANGE, "0,100,suffix:px"), "set_thickness", "get_thickness");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vertical"), "set_vertical", "is_vertical");
}

float StyleBoxLine::get_style_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);

	if (vertical) {
		if (p_side == SIDE_LEFT || p_side == SIDE_RIGHT) {
			return thickness / 2.0;
		}
	} else if (p_side == SIDE_TOP || p_side == SIDE_BOTTOM) {
		return thickness / 2.0;
	}

	return 0;
}

void StyleBoxLine::draw(RID p_canvas_item, const Rect2 &p_rect) const {
	RenderingServer *vs = RenderingServer::get_singleton();
	Rect2i r = p_rect;

	if (vertical) {
		r.position.y -= grow_begin;
		r.size.y += (grow_begin + grow_end);
		r.size.x = thickness;
	} else {
		r.position.x -= grow_begin;
		r.size.x += (grow_begin + grow_end);
		r.size.y = thickness;
	}

	vs->canvas_item_add_rect(p_canvas_item, r, color);
}

StyleBoxLine::StyleBoxLine() {}

StyleBoxLine::~StyleBoxLine() {}
