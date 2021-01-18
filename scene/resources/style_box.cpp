/*************************************************************************/
/*  style_box.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "style_box.h"
#include "scene/2d/canvas_item.h"

#include <limits.h>

bool StyleBox::test_mask(const Point2 &p_point, const Rect2 &p_rect) const {

	return true;
}

void StyleBox::set_default_margin(Margin p_margin, float p_value) {

	ERR_FAIL_INDEX((int)p_margin, 4);

	margin[p_margin] = p_value;
	emit_changed();
}
float StyleBox::get_default_margin(Margin p_margin) const {

	ERR_FAIL_INDEX_V((int)p_margin, 4, 0.0);

	return margin[p_margin];
}

float StyleBox::get_margin(Margin p_margin) const {

	ERR_FAIL_INDEX_V((int)p_margin, 4, 0.0);

	if (margin[p_margin] < 0)
		return get_style_margin(p_margin);
	else
		return margin[p_margin];
}

CanvasItem *StyleBox::get_current_item_drawn() const {
	return CanvasItem::get_current_item_drawn();
}

Size2 StyleBox::get_minimum_size() const {

	return Size2(get_margin(MARGIN_LEFT) + get_margin(MARGIN_RIGHT), get_margin(MARGIN_TOP) + get_margin(MARGIN_BOTTOM));
}

Point2 StyleBox::get_offset() const {

	return Point2(get_margin(MARGIN_LEFT), get_margin(MARGIN_TOP));
}

Size2 StyleBox::get_center_size() const {

	return Size2();
}

Rect2 StyleBox::get_draw_rect(const Rect2 &p_rect) const {
	return p_rect;
}

void StyleBox::_bind_methods() {

	ClassDB::bind_method(D_METHOD("test_mask", "point", "rect"), &StyleBox::test_mask);

	ClassDB::bind_method(D_METHOD("set_default_margin", "margin", "offset"), &StyleBox::set_default_margin);
	ClassDB::bind_method(D_METHOD("get_default_margin", "margin"), &StyleBox::get_default_margin);

	//ClassDB::bind_method(D_METHOD("set_default_margin"),&StyleBox::set_default_margin);
	//ClassDB::bind_method(D_METHOD("get_default_margin"),&StyleBox::get_default_margin);

	ClassDB::bind_method(D_METHOD("get_margin", "margin"), &StyleBox::get_margin);
	ClassDB::bind_method(D_METHOD("get_minimum_size"), &StyleBox::get_minimum_size);
	ClassDB::bind_method(D_METHOD("get_center_size"), &StyleBox::get_center_size);
	ClassDB::bind_method(D_METHOD("get_offset"), &StyleBox::get_offset);
	ClassDB::bind_method(D_METHOD("get_current_item_drawn"), &StyleBox::get_current_item_drawn);

	ClassDB::bind_method(D_METHOD("draw", "canvas_item", "rect"), &StyleBox::draw);

	ADD_GROUP("Content Margin", "content_margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "content_margin_left", PROPERTY_HINT_RANGE, "-1,2048,1"), "set_default_margin", "get_default_margin", MARGIN_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "content_margin_right", PROPERTY_HINT_RANGE, "-1,2048,1"), "set_default_margin", "get_default_margin", MARGIN_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "content_margin_top", PROPERTY_HINT_RANGE, "-1,2048,1"), "set_default_margin", "get_default_margin", MARGIN_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "content_margin_bottom", PROPERTY_HINT_RANGE, "-1,2048,1"), "set_default_margin", "get_default_margin", MARGIN_BOTTOM);
}

StyleBox::StyleBox() {

	for (int i = 0; i < 4; i++) {

		margin[i] = -1;
	}
}

void StyleBoxTexture::set_texture(Ref<Texture> p_texture) {

	if (texture == p_texture)
		return;
	texture = p_texture;
	if (p_texture.is_null()) {
		region_rect = Rect2(0, 0, 0, 0);
	} else {
		region_rect = Rect2(Point2(), texture->get_size());
	}
	emit_signal("texture_changed");
	emit_changed();
	_change_notify("texture");
}

Ref<Texture> StyleBoxTexture::get_texture() const {

	return texture;
}

void StyleBoxTexture::set_normal_map(Ref<Texture> p_normal_map) {

	if (normal_map == p_normal_map)
		return;
	normal_map = p_normal_map;
	emit_changed();
}

Ref<Texture> StyleBoxTexture::get_normal_map() const {

	return normal_map;
}

void StyleBoxTexture::set_margin_size(Margin p_margin, float p_size) {

	ERR_FAIL_INDEX((int)p_margin, 4);

	margin[p_margin] = p_size;
	emit_changed();
	static const char *margin_prop[4] = {
		"content_margin_left",
		"content_margin_top",
		"content_margin_right",
		"content_margin_bottom",
	};
	_change_notify(margin_prop[p_margin]);
}
float StyleBoxTexture::get_margin_size(Margin p_margin) const {

	ERR_FAIL_INDEX_V((int)p_margin, 4, 0.0);

	return margin[p_margin];
}

float StyleBoxTexture::get_style_margin(Margin p_margin) const {

	ERR_FAIL_INDEX_V((int)p_margin, 4, 0.0);

	return margin[p_margin];
}

Rect2 StyleBoxTexture::get_draw_rect(const Rect2 &p_rect) const {
	return p_rect.grow_individual(expand_margin[MARGIN_LEFT], expand_margin[MARGIN_TOP], expand_margin[MARGIN_RIGHT], expand_margin[MARGIN_BOTTOM]);
}

void StyleBoxTexture::draw(RID p_canvas_item, const Rect2 &p_rect) const {
	if (texture.is_null())
		return;

	Rect2 rect = p_rect;
	Rect2 src_rect = region_rect;

	texture->get_rect_region(rect, src_rect, rect, src_rect);

	rect.position.x -= expand_margin[MARGIN_LEFT];
	rect.position.y -= expand_margin[MARGIN_TOP];
	rect.size.x += expand_margin[MARGIN_LEFT] + expand_margin[MARGIN_RIGHT];
	rect.size.y += expand_margin[MARGIN_TOP] + expand_margin[MARGIN_BOTTOM];

	RID normal_rid;
	if (normal_map.is_valid())
		normal_rid = normal_map->get_rid();

	VisualServer::get_singleton()->canvas_item_add_nine_patch(p_canvas_item, rect, src_rect, texture->get_rid(), Vector2(margin[MARGIN_LEFT], margin[MARGIN_TOP]), Vector2(margin[MARGIN_RIGHT], margin[MARGIN_BOTTOM]), VS::NinePatchAxisMode(axis_h), VS::NinePatchAxisMode(axis_v), draw_center, modulate, normal_rid);
}

void StyleBoxTexture::set_draw_center(bool p_enabled) {

	draw_center = p_enabled;
	emit_changed();
}

bool StyleBoxTexture::is_draw_center_enabled() const {

	return draw_center;
}

Size2 StyleBoxTexture::get_center_size() const {

	if (texture.is_null())
		return Size2();

	return region_rect.size - get_minimum_size();
}

void StyleBoxTexture::set_expand_margin_size(Margin p_expand_margin, float p_size) {

	ERR_FAIL_INDEX((int)p_expand_margin, 4);
	expand_margin[p_expand_margin] = p_size;
	emit_changed();
}

void StyleBoxTexture::set_expand_margin_size_individual(float p_left, float p_top, float p_right, float p_bottom) {
	expand_margin[MARGIN_LEFT] = p_left;
	expand_margin[MARGIN_TOP] = p_top;
	expand_margin[MARGIN_RIGHT] = p_right;
	expand_margin[MARGIN_BOTTOM] = p_bottom;
	emit_changed();
}

void StyleBoxTexture::set_expand_margin_size_all(float p_expand_margin_size) {
	for (int i = 0; i < 4; i++) {

		expand_margin[i] = p_expand_margin_size;
	}
	emit_changed();
}

float StyleBoxTexture::get_expand_margin_size(Margin p_expand_margin) const {

	ERR_FAIL_INDEX_V((int)p_expand_margin, 4, 0);
	return expand_margin[p_expand_margin];
}

void StyleBoxTexture::set_region_rect(const Rect2 &p_region_rect) {

	if (region_rect == p_region_rect)
		return;

	region_rect = p_region_rect;
	emit_changed();
	_change_notify("region");
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
	if (modulate == p_modulate)
		return;
	modulate = p_modulate;
	emit_changed();
}

Color StyleBoxTexture::get_modulate() const {

	return modulate;
}

void StyleBoxTexture::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &StyleBoxTexture::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &StyleBoxTexture::get_texture);

	ClassDB::bind_method(D_METHOD("set_normal_map", "normal_map"), &StyleBoxTexture::set_normal_map);
	ClassDB::bind_method(D_METHOD("get_normal_map"), &StyleBoxTexture::get_normal_map);

	ClassDB::bind_method(D_METHOD("set_margin_size", "margin", "size"), &StyleBoxTexture::set_margin_size);
	ClassDB::bind_method(D_METHOD("get_margin_size", "margin"), &StyleBoxTexture::get_margin_size);

	ClassDB::bind_method(D_METHOD("set_expand_margin_size", "margin", "size"), &StyleBoxTexture::set_expand_margin_size);
	ClassDB::bind_method(D_METHOD("set_expand_margin_all", "size"), &StyleBoxTexture::set_expand_margin_size_all);
	ClassDB::bind_method(D_METHOD("set_expand_margin_individual", "size_left", "size_top", "size_right", "size_bottom"), &StyleBoxTexture::set_expand_margin_size_individual);
	ClassDB::bind_method(D_METHOD("get_expand_margin_size", "margin"), &StyleBoxTexture::get_expand_margin_size);

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

	ADD_SIGNAL(MethodInfo("texture_changed"));

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "normal_map", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_normal_map", "get_normal_map");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "region_rect"), "set_region_rect", "get_region_rect");
	ADD_GROUP("Margin", "margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "margin_left", PROPERTY_HINT_RANGE, "0,2048,1"), "set_margin_size", "get_margin_size", MARGIN_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "margin_right", PROPERTY_HINT_RANGE, "0,2048,1"), "set_margin_size", "get_margin_size", MARGIN_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "margin_top", PROPERTY_HINT_RANGE, "0,2048,1"), "set_margin_size", "get_margin_size", MARGIN_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "margin_bottom", PROPERTY_HINT_RANGE, "0,2048,1"), "set_margin_size", "get_margin_size", MARGIN_BOTTOM);
	ADD_GROUP("Expand Margin", "expand_margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "expand_margin_left", PROPERTY_HINT_RANGE, "0,2048,1"), "set_expand_margin_size", "get_expand_margin_size", MARGIN_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "expand_margin_right", PROPERTY_HINT_RANGE, "0,2048,1"), "set_expand_margin_size", "get_expand_margin_size", MARGIN_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "expand_margin_top", PROPERTY_HINT_RANGE, "0,2048,1"), "set_expand_margin_size", "get_expand_margin_size", MARGIN_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "expand_margin_bottom", PROPERTY_HINT_RANGE, "0,2048,1"), "set_expand_margin_size", "get_expand_margin_size", MARGIN_BOTTOM);
	ADD_GROUP("Axis Stretch", "axis_stretch_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis_stretch_horizontal", PROPERTY_HINT_ENUM, "Stretch,Tile,Tile Fit"), "set_h_axis_stretch_mode", "get_h_axis_stretch_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis_stretch_vertical", PROPERTY_HINT_ENUM, "Stretch,Tile,Tile Fit"), "set_v_axis_stretch_mode", "get_v_axis_stretch_mode");
	ADD_GROUP("Modulate", "modulate_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "modulate_color"), "set_modulate", "get_modulate");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_center"), "set_draw_center", "is_draw_center_enabled");

	BIND_ENUM_CONSTANT(AXIS_STRETCH_MODE_STRETCH);
	BIND_ENUM_CONSTANT(AXIS_STRETCH_MODE_TILE);
	BIND_ENUM_CONSTANT(AXIS_STRETCH_MODE_TILE_FIT);
}

StyleBoxTexture::StyleBoxTexture() {

	for (int i = 0; i < 4; i++) {
		margin[i] = 0;
		expand_margin[i] = 0;
	}
	draw_center = true;
	modulate = Color(1, 1, 1, 1);

	axis_h = AXIS_STRETCH_MODE_STRETCH;
	axis_v = AXIS_STRETCH_MODE_STRETCH;
}
StyleBoxTexture::~StyleBoxTexture() {
}

////////////////

void StyleBoxFlat::set_bg_color(const Color &p_color) {

	bg_color = p_color;
	emit_changed();
}

Color StyleBoxFlat::get_bg_color() const {

	return bg_color;
}

void StyleBoxFlat::set_border_color(const Color &p_color) {

	border_color = p_color;
	emit_changed();
}
Color StyleBoxFlat::get_border_color() const {

	return border_color;
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

void StyleBoxFlat::set_border_width(Margin p_margin, int p_width) {
	ERR_FAIL_INDEX((int)p_margin, 4);
	border_width[p_margin] = p_width;
	emit_changed();
}

int StyleBoxFlat::get_border_width(Margin p_margin) const {
	ERR_FAIL_INDEX_V((int)p_margin, 4, 0);
	return border_width[p_margin];
}

void StyleBoxFlat::set_border_blend(bool p_blend) {

	blend_border = p_blend;
	emit_changed();
}
bool StyleBoxFlat::get_border_blend() const {

	return blend_border;
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
int StyleBoxFlat::get_corner_radius_min() const {
	int smallest = corner_radius[0];
	for (int i = 1; i < 4; i++) {
		if (smallest > corner_radius[i]) {
			smallest = corner_radius[i];
		}
	}
	return smallest;
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

void StyleBoxFlat::set_expand_margin_size(Margin p_expand_margin, float p_size) {

	ERR_FAIL_INDEX((int)p_expand_margin, 4);
	expand_margin[p_expand_margin] = p_size;
	emit_changed();
}

void StyleBoxFlat::set_expand_margin_size_individual(float p_left, float p_top, float p_right, float p_bottom) {
	expand_margin[MARGIN_LEFT] = p_left;
	expand_margin[MARGIN_TOP] = p_top;
	expand_margin[MARGIN_RIGHT] = p_right;
	expand_margin[MARGIN_BOTTOM] = p_bottom;
	emit_changed();
}

void StyleBoxFlat::set_expand_margin_size_all(float p_expand_margin_size) {
	for (int i = 0; i < 4; i++) {

		expand_margin[i] = p_expand_margin_size;
	}
	emit_changed();
}

float StyleBoxFlat::get_expand_margin_size(Margin p_expand_margin) const {

	ERR_FAIL_INDEX_V((int)p_expand_margin, 4, 0.0);
	return expand_margin[p_expand_margin];
}
void StyleBoxFlat::set_draw_center(bool p_enabled) {

	draw_center = p_enabled;
	emit_changed();
}
bool StyleBoxFlat::is_draw_center_enabled() const {

	return draw_center;
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
}
bool StyleBoxFlat::is_anti_aliased() const {
	return anti_aliased;
}

void StyleBoxFlat::set_aa_size(const int &p_aa_size) {
	aa_size = CLAMP(p_aa_size, 1, 5);
	emit_changed();
}
int StyleBoxFlat::get_aa_size() const {
	return aa_size;
}

void StyleBoxFlat::set_corner_detail(const int &p_corner_detail) {
	corner_detail = CLAMP(p_corner_detail, 1, 20);
	emit_changed();
}
int StyleBoxFlat::get_corner_detail() const {
	return corner_detail;
}

Size2 StyleBoxFlat::get_center_size() const {

	return Size2();
}

inline void set_inner_corner_radius(const Rect2 style_rect, const Rect2 inner_rect, const int corner_radius[4], int *inner_corner_radius) {
	int border_left = inner_rect.position.x - style_rect.position.x;
	int border_top = inner_rect.position.y - style_rect.position.y;
	int border_right = style_rect.size.width - inner_rect.size.width - border_left;
	int border_bottom = style_rect.size.height - inner_rect.size.height - border_top;

	int rad;
	//tl
	rad = MIN(border_top, border_left);
	inner_corner_radius[0] = MAX(corner_radius[0] - rad, 0);

	//tr
	rad = MIN(border_top, border_right);
	inner_corner_radius[1] = MAX(corner_radius[1] - rad, 0);

	//br
	rad = MIN(border_bottom, border_right);
	inner_corner_radius[2] = MAX(corner_radius[2] - rad, 0);

	//bl
	rad = MIN(border_bottom, border_left);
	inner_corner_radius[3] = MAX(corner_radius[3] - rad, 0);
}

inline void draw_ring(Vector<Vector2> &verts, Vector<int> &indices, Vector<Color> &colors, const Rect2 &style_rect, const int corner_radius[4],
		const Rect2 &ring_rect, const Rect2 &inner_rect, const Color &inner_color, const Color &outer_color, const int corner_detail, const bool fill_center = false) {

	int vert_offset = verts.size();
	if (!vert_offset) {
		vert_offset = 0;
	}

	int adapted_corner_detail = (corner_radius[0] == 0 && corner_radius[1] == 0 && corner_radius[2] == 0 && corner_radius[3] == 0) ? 1 : corner_detail;

	int ring_corner_radius[4];
	set_inner_corner_radius(style_rect, ring_rect, corner_radius, ring_corner_radius);

	//corner radius center points
	Vector<Point2> outer_points;
	outer_points.push_back(ring_rect.position + Vector2(ring_corner_radius[0], ring_corner_radius[0])); //tl
	outer_points.push_back(Point2(ring_rect.position.x + ring_rect.size.x - ring_corner_radius[1], ring_rect.position.y + ring_corner_radius[1])); //tr
	outer_points.push_back(ring_rect.position + ring_rect.size - Vector2(ring_corner_radius[2], ring_corner_radius[2])); //br
	outer_points.push_back(Point2(ring_rect.position.x + ring_corner_radius[3], ring_rect.position.y + ring_rect.size.y - ring_corner_radius[3])); //bl

	int inner_corner_radius[4];
	set_inner_corner_radius(style_rect, inner_rect, corner_radius, inner_corner_radius);

	Vector<Point2> inner_points;
	inner_points.push_back(inner_rect.position + Vector2(inner_corner_radius[0], inner_corner_radius[0])); //tl
	inner_points.push_back(Point2(inner_rect.position.x + inner_rect.size.x - inner_corner_radius[1], inner_rect.position.y + inner_corner_radius[1])); //tr
	inner_points.push_back(inner_rect.position + inner_rect.size - Vector2(inner_corner_radius[2], inner_corner_radius[2])); //br
	inner_points.push_back(Point2(inner_rect.position.x + inner_corner_radius[3], inner_rect.position.y + inner_rect.size.y - inner_corner_radius[3])); //bl

	//calculate the vert array
	for (int corner_index = 0; corner_index < 4; corner_index++) {
		for (int detail = 0; detail <= adapted_corner_detail; detail++) {
			for (int inner_outer = 0; inner_outer < 2; inner_outer++) {
				float radius;
				Color color;
				Point2 corner_point;
				if (inner_outer == 0) {
					radius = inner_corner_radius[corner_index];
					color = inner_color;
					corner_point = inner_points[corner_index];
				} else {
					radius = ring_corner_radius[corner_index];
					color = outer_color;
					corner_point = outer_points[corner_index];
				}
				float x = radius * (float)cos((double)corner_index * Math_PI / 2.0 + (double)detail / (double)adapted_corner_detail * Math_PI / 2.0 + Math_PI) + corner_point.x;
				float y = radius * (float)sin((double)corner_index * Math_PI / 2.0 + (double)detail / (double)adapted_corner_detail * Math_PI / 2.0 + Math_PI) + corner_point.y;
				verts.push_back(Vector2(x, y));
				colors.push_back(color);
			}
		}
	}

	int ring_vert_count = verts.size() - vert_offset;

	//fill the indices and the colors for the border
	for (int i = 0; i < ring_vert_count; i++) {
		indices.push_back(vert_offset + ((i + 0) % ring_vert_count));
		indices.push_back(vert_offset + ((i + 2) % ring_vert_count));
		indices.push_back(vert_offset + ((i + 1) % ring_vert_count));
	}

	if (fill_center) {
		//fill the indices and the colors for the center
		for (int index = 0; index < ring_vert_count / 2; index += 2) {
			int i = index;
			//poly 1
			indices.push_back(vert_offset + i);
			indices.push_back(vert_offset + ring_vert_count - 4 - i);
			indices.push_back(vert_offset + i + 2);
			//poly 2
			indices.push_back(vert_offset + i);
			indices.push_back(vert_offset + ring_vert_count - 2 - i);
			indices.push_back(vert_offset + ring_vert_count - 4 - i);
		}
	}
}

inline void adapt_values(int p_index_a, int p_index_b, int *adapted_values, const int *p_values, const real_t p_width, const int p_max_a, const int p_max_b) {
	if (p_values[p_index_a] + p_values[p_index_b] > p_width) {
		float factor;
		int newValue;

		factor = (float)p_width / (float)(p_values[p_index_a] + p_values[p_index_b]);

		newValue = (int)(p_values[p_index_a] * factor);
		if (newValue < adapted_values[p_index_a]) {
			adapted_values[p_index_a] = newValue;
		}
		newValue = (int)(p_values[p_index_b] * factor);
		if (newValue < adapted_values[p_index_b]) {
			adapted_values[p_index_b] = newValue;
		}
	} else {
		adapted_values[p_index_a] = MIN(p_values[p_index_a], adapted_values[p_index_a]);
		adapted_values[p_index_b] = MIN(p_values[p_index_b], adapted_values[p_index_b]);
	}
	adapted_values[p_index_a] = MIN(p_max_a, adapted_values[p_index_a]);
	adapted_values[p_index_b] = MIN(p_max_b, adapted_values[p_index_b]);
}

Rect2 StyleBoxFlat::get_draw_rect(const Rect2 &p_rect) const {
	Rect2 draw_rect = p_rect.grow_individual(expand_margin[MARGIN_LEFT], expand_margin[MARGIN_TOP], expand_margin[MARGIN_RIGHT], expand_margin[MARGIN_BOTTOM]);

	if (shadow_size > 0) {
		Rect2 shadow_rect = draw_rect.grow(shadow_size);
		shadow_rect.position += shadow_offset;
		draw_rect = draw_rect.merge(shadow_rect);
	}

	return draw_rect;
}

void StyleBoxFlat::draw(RID p_canvas_item, const Rect2 &p_rect) const {

	//PREPARATIONS
	bool draw_border = (border_width[0] > 0) || (border_width[1] > 0) || (border_width[2] > 0) || (border_width[3] > 0);
	bool draw_shadow = (shadow_size > 0);
	if (!draw_border && !draw_center && !draw_shadow) {
		return;
	}

	Rect2 style_rect = p_rect.grow_individual(expand_margin[MARGIN_LEFT], expand_margin[MARGIN_TOP], expand_margin[MARGIN_RIGHT], expand_margin[MARGIN_BOTTOM]);
	if (Math::is_zero_approx(style_rect.size.width) || Math::is_zero_approx(style_rect.size.height)) {
		return;
	}

	bool rounded_corners = (corner_radius[0] > 0) || (corner_radius[1] > 0) || (corner_radius[2] > 0) || (corner_radius[3] > 0);
	bool aa_on = rounded_corners && anti_aliased;
	float aa_size_grow = 0.5 * ((float)aa_size + 1.0);

	bool blend_on = blend_border && draw_border;

	Color border_color_alpha = Color(border_color.r, border_color.g, border_color.b, 0);
	Color border_color_blend = (draw_center ? bg_color : border_color_alpha);
	Color border_color_inner = blend_on ? border_color_blend : border_color;

	//adapt borders (prevent weird overlapping/glitchy drawings)
	int width = MAX(style_rect.size.width, 0);
	int height = MAX(style_rect.size.height, 0);
	int adapted_border[4] = { INT_MAX, INT_MAX, INT_MAX, INT_MAX };
	adapt_values(MARGIN_TOP, MARGIN_BOTTOM, adapted_border, border_width, height, height, height);
	adapt_values(MARGIN_LEFT, MARGIN_RIGHT, adapted_border, border_width, width, width, width);

	//adapt corners (prevent weird overlapping/glitchy drawings)
	int adapted_corner[4] = { INT_MAX, INT_MAX, INT_MAX, INT_MAX };
	adapt_values(CORNER_TOP_RIGHT, CORNER_BOTTOM_RIGHT, adapted_corner, corner_radius, height, height - adapted_border[MARGIN_BOTTOM], height - adapted_border[MARGIN_TOP]);
	adapt_values(CORNER_TOP_LEFT, CORNER_BOTTOM_LEFT, adapted_corner, corner_radius, height, height - adapted_border[MARGIN_BOTTOM], height - adapted_border[MARGIN_TOP]);
	adapt_values(CORNER_TOP_LEFT, CORNER_TOP_RIGHT, adapted_corner, corner_radius, width, width - adapted_border[MARGIN_RIGHT], width - adapted_border[MARGIN_LEFT]);
	adapt_values(CORNER_BOTTOM_LEFT, CORNER_BOTTOM_RIGHT, adapted_corner, corner_radius, width, width - adapted_border[MARGIN_RIGHT], width - adapted_border[MARGIN_LEFT]);

	Rect2 infill_rect = style_rect.grow_individual(-adapted_border[MARGIN_LEFT], -adapted_border[MARGIN_TOP], -adapted_border[MARGIN_RIGHT], -adapted_border[MARGIN_BOTTOM]);

	Rect2 border_style_rect = style_rect;
	if (aa_on) {
		for (int i = 0; i < 4; i++) {
			if (border_width[i] > 0) {
				border_style_rect = border_style_rect.grow_margin((Margin)i, -aa_size_grow);
			}
		}
	}

	Vector<Point2> verts;
	Vector<int> indices;
	Vector<Color> colors;
	Vector<Point2> uvs;

	//DRAW SHADOW
	if (draw_shadow) {
		Rect2 shadow_inner_rect = style_rect;
		shadow_inner_rect.position += shadow_offset;

		Rect2 shadow_rect = style_rect.grow(shadow_size);
		shadow_rect.position += shadow_offset;

		Color shadow_color_transparent = Color(shadow_color.r, shadow_color.g, shadow_color.b, 0);

		draw_ring(verts, indices, colors, shadow_inner_rect, adapted_corner,
				shadow_rect, shadow_inner_rect, shadow_color, shadow_color_transparent, corner_detail);

		if (draw_center) {
			draw_ring(verts, indices, colors, shadow_inner_rect, adapted_corner,
					shadow_inner_rect, shadow_inner_rect, shadow_color, shadow_color, corner_detail, true);
		}
	}

	//DRAW border
	if (draw_border) {
		draw_ring(verts, indices, colors, border_style_rect, adapted_corner,
				border_style_rect, infill_rect, border_color_inner, border_color, corner_detail);
	}

	//DRAW INFILL
	if (draw_center && (!aa_on || blend_on || !draw_border)) {
		draw_ring(verts, indices, colors, border_style_rect, adapted_corner,
				infill_rect, infill_rect, bg_color, bg_color, corner_detail, true);
	}

	if (aa_on) {
		int aa_border_width[4];
		int aa_fill_width[4];
		if (draw_border) {
			for (int i = 0; i < 4; i++) {
				if (border_width[i] > 0) {
					aa_border_width[i] = aa_size_grow;
					aa_fill_width[i] = 0;
				} else {
					aa_border_width[i] = 0;
					aa_fill_width[i] = aa_size_grow;
				}
			}
		} else {
			for (int i = 0; i < 4; i++) {
				aa_border_width[i] = 0;
				aa_fill_width[i] = aa_size_grow;
			}
		}

		Rect2 infill_inner_rect = infill_rect.grow_individual(-aa_border_width[MARGIN_LEFT], -aa_border_width[MARGIN_TOP],
				-aa_border_width[MARGIN_RIGHT], -aa_border_width[MARGIN_BOTTOM]);

		if (draw_center) {
			if (!blend_on && draw_border) {
				//DRAW INFILL WITHIN BORDER AA
				draw_ring(verts, indices, colors, border_style_rect, adapted_corner,
						infill_inner_rect, infill_inner_rect, bg_color, bg_color, corner_detail, true);
			}

			if (!blend_on || !draw_border) {
				Rect2 infill_aa_rect = infill_rect.grow_individual(aa_fill_width[MARGIN_LEFT], aa_fill_width[MARGIN_TOP],
						aa_fill_width[MARGIN_RIGHT], aa_fill_width[MARGIN_BOTTOM]);

				Color alpha_bg = Color(bg_color.r, bg_color.g, bg_color.b, 0);

				//INFILL AA
				draw_ring(verts, indices, colors, style_rect, adapted_corner,
						infill_aa_rect, infill_rect, bg_color, alpha_bg, corner_detail);
			}
		}

		if (draw_border) {
			if (!blend_on) {
				//DRAW INNER BORDER AA
				draw_ring(verts, indices, colors, border_style_rect, adapted_corner,
						infill_rect, infill_inner_rect, border_color_blend, border_color, corner_detail);
			}

			//DRAW OUTER BORDER AA
			draw_ring(verts, indices, colors, border_style_rect, adapted_corner,
					style_rect, border_style_rect, border_color, border_color_alpha, corner_detail);
		}
	}

	//COMPUTE UV COORDINATES
	Rect2 uv_rect = style_rect.grow(aa_on ? aa_size_grow : 0);
	uvs.resize(verts.size());
	for (int i = 0; i < verts.size(); i++) {
		uvs.write[i].x = (verts[i].x - uv_rect.position.x) / uv_rect.size.width;
		uvs.write[i].y = (verts[i].y - uv_rect.position.y) / uv_rect.size.height;
	}

	//DRAWING
	VisualServer *vs = VisualServer::get_singleton();
	vs->canvas_item_add_triangle_array(p_canvas_item, indices, verts, colors, uvs);
}

float StyleBoxFlat::get_style_margin(Margin p_margin) const {
	ERR_FAIL_INDEX_V((int)p_margin, 4, 0.0);
	return border_width[p_margin];
}
void StyleBoxFlat::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_bg_color", "color"), &StyleBoxFlat::set_bg_color);
	ClassDB::bind_method(D_METHOD("get_bg_color"), &StyleBoxFlat::get_bg_color);

	ClassDB::bind_method(D_METHOD("set_border_color", "color"), &StyleBoxFlat::set_border_color);
	ClassDB::bind_method(D_METHOD("get_border_color"), &StyleBoxFlat::get_border_color);

	ClassDB::bind_method(D_METHOD("set_border_width_all", "width"), &StyleBoxFlat::set_border_width_all);
	ClassDB::bind_method(D_METHOD("get_border_width_min"), &StyleBoxFlat::get_border_width_min);

	ClassDB::bind_method(D_METHOD("set_border_width", "margin", "width"), &StyleBoxFlat::set_border_width);
	ClassDB::bind_method(D_METHOD("get_border_width", "margin"), &StyleBoxFlat::get_border_width);

	ClassDB::bind_method(D_METHOD("set_border_blend", "blend"), &StyleBoxFlat::set_border_blend);
	ClassDB::bind_method(D_METHOD("get_border_blend"), &StyleBoxFlat::get_border_blend);

	ClassDB::bind_method(D_METHOD("set_corner_radius_individual", "radius_top_left", "radius_top_right", "radius_bottom_right", "radius_bottom_left"), &StyleBoxFlat::set_corner_radius_individual);
	ClassDB::bind_method(D_METHOD("set_corner_radius_all", "radius"), &StyleBoxFlat::set_corner_radius_all);

	ClassDB::bind_method(D_METHOD("set_corner_radius", "corner", "radius"), &StyleBoxFlat::set_corner_radius);
	ClassDB::bind_method(D_METHOD("get_corner_radius", "corner"), &StyleBoxFlat::get_corner_radius);

	ClassDB::bind_method(D_METHOD("set_expand_margin", "margin", "size"), &StyleBoxFlat::set_expand_margin_size);
	ClassDB::bind_method(D_METHOD("set_expand_margin_all", "size"), &StyleBoxFlat::set_expand_margin_size_all);
	ClassDB::bind_method(D_METHOD("set_expand_margin_individual", "size_left", "size_top", "size_right", "size_bottom"), &StyleBoxFlat::set_expand_margin_size_individual);
	ClassDB::bind_method(D_METHOD("get_expand_margin", "margin"), &StyleBoxFlat::get_expand_margin_size);

	ClassDB::bind_method(D_METHOD("set_draw_center", "draw_center"), &StyleBoxFlat::set_draw_center);
	ClassDB::bind_method(D_METHOD("is_draw_center_enabled"), &StyleBoxFlat::is_draw_center_enabled);

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

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "bg_color"), "set_bg_color", "get_bg_color");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_center"), "set_draw_center", "is_draw_center_enabled");

	ADD_GROUP("Border Width", "border_width_");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_width_left", PROPERTY_HINT_RANGE, "0,1024,1"), "set_border_width", "get_border_width", MARGIN_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_width_top", PROPERTY_HINT_RANGE, "0,1024,1"), "set_border_width", "get_border_width", MARGIN_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_width_right", PROPERTY_HINT_RANGE, "0,1024,1"), "set_border_width", "get_border_width", MARGIN_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "border_width_bottom", PROPERTY_HINT_RANGE, "0,1024,1"), "set_border_width", "get_border_width", MARGIN_BOTTOM);

	ADD_GROUP("Border", "border_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_color"), "set_border_color", "get_border_color");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "border_blend"), "set_border_blend", "get_border_blend");

	ADD_GROUP("Corner Radius", "corner_radius_");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "corner_radius_top_left", PROPERTY_HINT_RANGE, "0,1024,1"), "set_corner_radius", "get_corner_radius", CORNER_TOP_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "corner_radius_top_right", PROPERTY_HINT_RANGE, "0,1024,1"), "set_corner_radius", "get_corner_radius", CORNER_TOP_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "corner_radius_bottom_right", PROPERTY_HINT_RANGE, "0,1024,1"), "set_corner_radius", "get_corner_radius", CORNER_BOTTOM_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "corner_radius_bottom_left", PROPERTY_HINT_RANGE, "0,1024,1"), "set_corner_radius", "get_corner_radius", CORNER_BOTTOM_LEFT);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "corner_detail", PROPERTY_HINT_RANGE, "1,20,1"), "set_corner_detail", "get_corner_detail");

	ADD_GROUP("Expand Margin", "expand_margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "expand_margin_left", PROPERTY_HINT_RANGE, "0,2048,1"), "set_expand_margin", "get_expand_margin", MARGIN_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "expand_margin_right", PROPERTY_HINT_RANGE, "0,2048,1"), "set_expand_margin", "get_expand_margin", MARGIN_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "expand_margin_top", PROPERTY_HINT_RANGE, "0,2048,1"), "set_expand_margin", "get_expand_margin", MARGIN_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "expand_margin_bottom", PROPERTY_HINT_RANGE, "0,2048,1"), "set_expand_margin", "get_expand_margin", MARGIN_BOTTOM);

	ADD_GROUP("Shadow", "shadow_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "shadow_color"), "set_shadow_color", "get_shadow_color");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_size", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_shadow_size", "get_shadow_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "shadow_offset"), "set_shadow_offset", "get_shadow_offset");

	ADD_GROUP("Anti Aliasing", "anti_aliasing_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "anti_aliasing"), "set_anti_aliased", "is_anti_aliased");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "anti_aliasing_size", PROPERTY_HINT_RANGE, "1,5,1"), "set_aa_size", "get_aa_size");
}

StyleBoxFlat::StyleBoxFlat() {

	bg_color = Color(0.6, 0.6, 0.6);
	shadow_color = Color(0, 0, 0, 0.6);
	border_color = Color(0.8, 0.8, 0.8);

	blend_border = false;
	draw_center = true;
	anti_aliased = true;

	shadow_size = 0;
	shadow_offset = Point2(0, 0);
	corner_detail = 8;
	aa_size = 1;

	border_width[0] = 0;
	border_width[1] = 0;
	border_width[2] = 0;
	border_width[3] = 0;

	expand_margin[0] = 0;
	expand_margin[1] = 0;
	expand_margin[2] = 0;
	expand_margin[3] = 0;

	corner_radius[0] = 0;
	corner_radius[1] = 0;
	corner_radius[2] = 0;
	corner_radius[3] = 0;
}
StyleBoxFlat::~StyleBoxFlat() {
}

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
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "grow_begin", PROPERTY_HINT_RANGE, "-300,300,1"), "set_grow_begin", "get_grow_begin");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "grow_end", PROPERTY_HINT_RANGE, "-300,300,1"), "set_grow_end", "get_grow_end");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "thickness", PROPERTY_HINT_RANGE, "0,10"), "set_thickness", "get_thickness");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vertical"), "set_vertical", "is_vertical");
}
float StyleBoxLine::get_style_margin(Margin p_margin) const {
	ERR_FAIL_INDEX_V((int)p_margin, 4, thickness);
	return thickness;
}
Size2 StyleBoxLine::get_center_size() const {
	return Size2();
}

void StyleBoxLine::draw(RID p_canvas_item, const Rect2 &p_rect) const {
	VisualServer *vs = VisualServer::get_singleton();
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

StyleBoxLine::StyleBoxLine() {
	grow_begin = 1.0;
	grow_end = 1.0;
	thickness = 1;
	color = Color(0.0, 0.0, 0.0);
	vertical = false;
}
StyleBoxLine::~StyleBoxLine() {}
