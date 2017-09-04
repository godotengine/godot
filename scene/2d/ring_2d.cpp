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
#include "ring_2d.h"
#include "core_string_names.h"

void Ring2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_vertices", "vertices"), &Ring2D::set_vertices);
	ClassDB::bind_method(D_METHOD("get_vertices"), &Ring2D::get_vertices);

	ClassDB::bind_method(D_METHOD("set_offset", "vertices"), &Ring2D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &Ring2D::get_offset);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "vertices"), "set_vertices", "get_vertices");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
}

void Ring2D::set_vertices(const Vector<Point2> &p_polygon) {

	vertices = Variant(p_polygon);
	rect_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Vector<Point2> Ring2D::get_vertices() const {

	return Variant(vertices);
}

bool Ring2D::is_empty() const {

	return vertices.size() == 0;
}

Rect2 Ring2D::get_item_rect() const {

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

void Ring2D::set_offset(const Vector2 &p_offset) {

	offset = p_offset;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Vector2 Ring2D::get_offset() const {

	return offset;
}

Ring2D::Ring2D() {

	rect_cache_dirty = true;
	offset = Vector2(0, 0);
}
