/*************************************************************************/
/*  editable_polygon_2d.h                                                */
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
#include "editable_polygon_2d.h"
#include "core_string_names.h"

void AbstractPolygon2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_vertices", "vertices"), &AbstractPolygon2D::set_vertices);
	ClassDB::bind_method(D_METHOD("get_vertices"), &AbstractPolygon2D::get_vertices);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "vertices"), "set_vertices", "get_vertices");
}

void AbstractPolygon2D::set_vertices(const Vector<Point2> &p_polygon) {

	vertices = Variant(p_polygon);
	rect_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Vector<Point2> AbstractPolygon2D::get_vertices() const {

	return Variant(vertices);
}

bool AbstractPolygon2D::is_empty() const {

	return vertices.size() == 0;
}

Rect2 AbstractPolygon2D::get_item_rect() const {

	if (rect_cache_dirty) {

		int l = vertices.size();
		PoolVector<Vector2>::Read r = vertices.read();
		item_rect = Rect2();
		for (int i = 0; i < l; i++) {
			Vector2 pos = r[i];
			if (i == 0)
				item_rect.position = pos;
			else
				item_rect.expand_to(pos);
		}

		// FIXME collision polygon aabb
		/*if (aabb == Rect2()) {

			aabb = Rect2(-10, -10, 20, 20);
		} else {
			aabb.position -= aabb.size * 0.3;
			aabb.size += aabb.size * 0.6(
		}*/

		item_rect = item_rect.grow(20);
		rect_cache_dirty = false;
	}

	return Rect2(item_rect.position + get_offset(), item_rect.size);
}

Vector2 AbstractPolygon2D::get_offset() const {

	return Vector2(0, 0);
}

Color AbstractPolygon2D::get_outline_color() const {

	return Color(0.5, 0.5, 0.5);
}

AbstractPolygon2D::AbstractPolygon2D() {

	rect_cache_dirty = true;
}

void EditablePolygonNode2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("append_polygon", "vertices"), &EditablePolygonNode2D::append_polygon);
	ClassDB::bind_method(D_METHOD("add_polygon_at_index", "index", "polygon"), &EditablePolygonNode2D::add_polygon_at_index);
	ClassDB::bind_method(D_METHOD("set_vertices", "index", "vertices"), &EditablePolygonNode2D::set_vertices);
	ClassDB::bind_method(D_METHOD("remove_polygon", "index"), &EditablePolygonNode2D::remove_polygon);
}
