/*************************************************************************/
/*  polygon_2d.cpp                                                       */
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
#include "polygon_2d.h"

#include "core_string_names.h"
#include "engine.h"

void Polygon2D::edit_set_pivot(const Point2 &p_pivot) {

	set_offset(p_pivot);
}

Point2 Polygon2D::edit_get_pivot() const {

	return get_offset();
}
bool Polygon2D::edit_has_pivot() const {

	return true;
}

void Polygon2D::draw(RID p_canvas_item) {

	if (vertices.size() < 3)
		return;

	Vector<Vector2> points;
	Vector<Vector2> uvs;

	points.resize(vertices.size());

	int len = points.size();
	{

		PoolVector<Vector2>::Read polyr = vertices.read();
		for (int i = 0; i < len; i++) {
			points[i] = polyr[i] + offset;
		}
	}

	if (invert) {

		Rect2 bounds;
		int highest_idx = -1;
		float highest_y = -1e20;
		float sum = 0;

		for (int i = 0; i < len; i++) {
			if (i == 0)
				bounds.position = points[i];
			else
				bounds.expand_to(points[i]);
			if (points[i].y > highest_y) {
				highest_idx = i;
				highest_y = points[i].y;
			}
			int ni = (i + 1) % len;
			sum += (points[ni].x - points[i].x) * (points[ni].y + points[i].y);
		}

		bounds = bounds.grow(invert_border);

		Vector2 ep[7] = {
			Vector2(points[highest_idx].x, points[highest_idx].y + invert_border),
			Vector2(bounds.position + bounds.size),
			Vector2(bounds.position + Vector2(bounds.size.x, 0)),
			Vector2(bounds.position),
			Vector2(bounds.position + Vector2(0, bounds.size.y)),
			Vector2(points[highest_idx].x - CMP_EPSILON, points[highest_idx].y + invert_border),
			Vector2(points[highest_idx].x - CMP_EPSILON, points[highest_idx].y),
		};

		if (sum > 0) {
			SWAP(ep[1], ep[4]);
			SWAP(ep[2], ep[3]);
			SWAP(ep[5], ep[0]);
			SWAP(ep[6], points[highest_idx]);
		}

		points.resize(points.size() + 7);
		for (int i = points.size() - 1; i >= highest_idx + 7; i--) {

			points[i] = points[i - 7];
		}

		for (int i = 0; i < 7; i++) {

			points[highest_idx + i + 1] = ep[i];
		}

		len = points.size();
	}

	if (texture.is_valid()) {

		Transform2D texmat(tex_rot, tex_ofs);
		texmat.scale(tex_scale);
		Size2 tex_size = texture->get_size();
		uvs.resize(points.size());

		if (points.size() == uv.size()) {

			PoolVector<Vector2>::Read uvr = uv.read();

			for (int i = 0; i < len; i++) {
				uvs[i] = texmat.xform(uvr[i]) / tex_size;
			}

		} else {
			for (int i = 0; i < len; i++) {
				uvs[i] = texmat.xform(points[i]) / tex_size;
			}
		}
	}

	Vector<Color> colors;
	int color_len = vertex_colors.size();
	colors.resize(len);
	{
		PoolVector<Color>::Read color_r = vertex_colors.read();
		for (int i = 0; i < color_len && i < len; i++) {
			colors[i] = color_r[i];
		}
		for (int i = color_len; i < len; i++) {
			colors[i] = color;
		}
	}

	//			Vector<int> indices = Geometry::triangulate_polygon(points);
	//			VS::get_singleton()->canvas_item_add_triangle_array(get_canvas_item(), indices, points, colors, uvs, texture.is_valid() ? texture->get_rid() : RID());

	VS::get_singleton()->canvas_item_add_polygon(p_canvas_item, points, colors, uvs, texture.is_valid() ? texture->get_rid() : RID(), RID(), antialiased);
}

void Polygon2D::set_uv(const PoolVector<Vector2> &p_uv) {

	uv = p_uv;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

PoolVector<Vector2> Polygon2D::get_uv() const {

	return uv;
}

void Polygon2D::set_color(const Color &p_color) {

	color = p_color;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Color Polygon2D::get_color() const {

	return color;
}

void Polygon2D::set_vertex_colors(const PoolVector<Color> &p_colors) {

	vertex_colors = p_colors;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
PoolVector<Color> Polygon2D::get_vertex_colors() const {

	return vertex_colors;
}

void Polygon2D::set_texture(const Ref<Texture> &p_texture) {

	texture = p_texture;

	/*if (texture.is_valid()) {
		uint32_t flags=texture->get_flags();
		flags&=~Texture::FLAG_REPEAT;
		if (tex_tile)
			flags|=Texture::FLAG_REPEAT;

		texture->set_flags(flags);
	}*/
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Ref<Texture> Polygon2D::get_texture() const {

	return texture;
}

void Polygon2D::set_texture_offset(const Vector2 &p_offset) {

	tex_ofs = p_offset;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Vector2 Polygon2D::get_texture_offset() const {

	return tex_ofs;
}

void Polygon2D::set_texture_rotation(float p_rot) {

	tex_rot = p_rot;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
float Polygon2D::get_texture_rotation() const {

	return tex_rot;
}

void Polygon2D::_set_texture_rotationd(float p_rot) {

	set_texture_rotation(Math::deg2rad(p_rot));
}
float Polygon2D::_get_texture_rotationd() const {

	return Math::rad2deg(get_texture_rotation());
}

void Polygon2D::set_texture_scale(const Size2 &p_scale) {

	tex_scale = p_scale;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Size2 Polygon2D::get_texture_scale() const {

	return tex_scale;
}

void Polygon2D::set_invert(bool p_invert) {

	invert = p_invert;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
bool Polygon2D::get_invert() const {

	return invert;
}

void Polygon2D::set_antialiased(bool p_antialiased) {

	antialiased = p_antialiased;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
bool Polygon2D::get_antialiased() const {

	return antialiased;
}

void Polygon2D::set_invert_border(float p_invert_border) {

	invert_border = p_invert_border;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
float Polygon2D::get_invert_border() const {

	return invert_border;
}

void Polygon2D::set_offset(const Vector2 &p_offset) {

	offset = p_offset;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Vector2 Polygon2D::get_offset() const {

	return offset;
}

void Polygon2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_uv", "uv"), &Polygon2D::set_uv);
	ClassDB::bind_method(D_METHOD("get_uv"), &Polygon2D::get_uv);

	ClassDB::bind_method(D_METHOD("set_color", "color"), &Polygon2D::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &Polygon2D::get_color);

	ClassDB::bind_method(D_METHOD("set_vertex_colors", "vertex_colors"), &Polygon2D::set_vertex_colors);
	ClassDB::bind_method(D_METHOD("get_vertex_colors"), &Polygon2D::get_vertex_colors);

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &Polygon2D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &Polygon2D::get_texture);

	ClassDB::bind_method(D_METHOD("set_texture_offset", "texture_offset"), &Polygon2D::set_texture_offset);
	ClassDB::bind_method(D_METHOD("get_texture_offset"), &Polygon2D::get_texture_offset);

	ClassDB::bind_method(D_METHOD("set_texture_rotation", "texture_rotation"), &Polygon2D::set_texture_rotation);
	ClassDB::bind_method(D_METHOD("get_texture_rotation"), &Polygon2D::get_texture_rotation);

	ClassDB::bind_method(D_METHOD("_set_texture_rotationd", "texture_rotation"), &Polygon2D::_set_texture_rotationd);
	ClassDB::bind_method(D_METHOD("_get_texture_rotationd"), &Polygon2D::_get_texture_rotationd);

	ClassDB::bind_method(D_METHOD("set_texture_scale", "texture_scale"), &Polygon2D::set_texture_scale);
	ClassDB::bind_method(D_METHOD("get_texture_scale"), &Polygon2D::get_texture_scale);

	ClassDB::bind_method(D_METHOD("set_invert", "invert"), &Polygon2D::set_invert);
	ClassDB::bind_method(D_METHOD("get_invert"), &Polygon2D::get_invert);

	ClassDB::bind_method(D_METHOD("set_antialiased", "antialiased"), &Polygon2D::set_antialiased);
	ClassDB::bind_method(D_METHOD("get_antialiased"), &Polygon2D::get_antialiased);

	ClassDB::bind_method(D_METHOD("set_invert_border", "invert_border"), &Polygon2D::set_invert_border);
	ClassDB::bind_method(D_METHOD("get_invert_border"), &Polygon2D::get_invert_border);

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &Polygon2D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &Polygon2D::get_offset);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "uv"), "set_uv", "get_uv");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_COLOR_ARRAY, "vertex_colors"), "set_vertex_colors", "get_vertex_colors");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "antialiased"), "set_antialiased", "get_antialiased");
	ADD_GROUP("Texture", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_GROUP("Texture", "texture_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "texture_offset"), "set_texture_offset", "get_texture_offset");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "texture_scale"), "set_texture_scale", "get_texture_scale");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "texture_rotation", PROPERTY_HINT_RANGE, "-1440,1440,0.1"), "_set_texture_rotationd", "_get_texture_rotationd");

	ADD_GROUP("Invert", "invert_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert_enable"), "set_invert", "get_invert");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "invert_border", PROPERTY_HINT_RANGE, "0.1,16384,0.1"), "set_invert_border", "get_invert_border");
}

PoolVector<Vector2> Polygon2D::edit_get_uv() const {

	return get_uv();
}

void Polygon2D::edit_set_uv(const PoolVector<Vector2> &p_uv) {

	set_uv(p_uv);
}

Ref<Texture> Polygon2D::edit_get_texture() const {

	return get_texture();
}

Polygon2D::Polygon2D() {

	invert = 0;
	invert_border = 100;
	antialiased = false;
	tex_rot = 0;
	tex_tile = true;
	tex_scale = Vector2(1, 1);
	color = Color(1, 1, 1);
}

void Polygon2DInstance::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_DRAW: {

			if (polygon.is_valid()) {

				polygon->draw(get_canvas_item());
			}

		} break;
	}
}

void Polygon2DInstance::set_polygon(const Ref<Polygon2D> &p_polygon) {

	if (p_polygon == polygon)
		return;

	if (polygon.is_valid()) {
		polygon->disconnect(CoreStringNames::get_singleton()->changed, this, "_polygon_changed");
	}
	polygon = p_polygon;

	if (polygon.is_valid()) {
		polygon->connect(CoreStringNames::get_singleton()->changed, this, "_polygon_changed");
	}

	_polygon_changed();
	_change_notify("polygon");
	update_configuration_warning();
}

Ref<Polygon2D> Polygon2DInstance::get_polygon() const {

	return polygon;
}

void Polygon2DInstance::_polygon_changed() {

	if (is_inside_tree())
		update();
}

String Polygon2DInstance::get_configuration_warning() const {

	if (!is_visible_in_tree() || !is_inside_tree())
		return String();

	if (!polygon.is_valid()) {
		return TTR("A Polygon2D resource must be set or created for this node to work. Please set a property or draw a polygon.");
	}

	return String();
}

Rect2 Polygon2DInstance::get_item_rect() const {

	if (polygon.is_valid())
		return polygon->get_item_rect();
	else
		return Rect2();
}

void Polygon2DInstance::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_polygon", "polygon"), &Polygon2DInstance::set_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon"), &Polygon2DInstance::get_polygon);

	ClassDB::bind_method(D_METHOD("_polygon_changed"), &Polygon2DInstance::_polygon_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "polygon", PROPERTY_HINT_RESOURCE_TYPE, "Polygon2D"), "set_polygon", "get_polygon");
}

bool Polygon2DInstance::_has_resource() const {

	return polygon.is_valid();
}

void Polygon2DInstance::_create_resource(UndoRedo *undo_redo) {

	undo_redo->create_action(TTR("Create Polygon2D"));
	undo_redo->add_do_method(this, "set_polygon", Ref<Polygon2D>(memnew(Polygon2D)));
	undo_redo->add_undo_method(this, "set_polygon", Variant(REF()));
	undo_redo->commit_action();
}

int Polygon2DInstance::get_polygon_count() const {

	return polygon.is_valid() ? 1 : 0;
}

Ref<AbstractPolygon2D> Polygon2DInstance::get_nth_polygon(int p_idx) const {

	return polygon;
}

void Polygon2DInstance::append_polygon(const Vector<Point2> &p_vertices) {

	Ref<Polygon2D> polygon = Ref<Polygon2D>(memnew(Polygon2D));
	polygon->set_vertices(p_vertices);
	set_polygon(polygon);
}

void Polygon2DInstance::add_polygon_at_index(int p_idx, Ref<AbstractPolygon2D> p_polygon) {

	set_polygon(p_polygon);
}

void Polygon2DInstance::set_vertices(int p_idx, const Vector<Point2> &p_vertices) {

	if (polygon.is_valid())
		polygon->set_vertices(p_vertices);
}

void Polygon2DInstance::remove_polygon(int p_idx) {

	set_polygon(Ref<Polygon2D>());
}

Polygon2DInstance::Polygon2DInstance() {
}
