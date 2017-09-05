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
#include "polygon_node_2d.h"
#include "core_string_names.h"

void PolygonNode2D::add_polygon(const Vector<Point2> &p_vertices) {

	Ref<Ring2D> ring = memnew(Ring2D);
	ring->set_vertices(p_vertices);
	Ref<Resource> polygon = new_polygon(ring);
	add_polygon_at_index(polygon, get_polygon_count());
}

void PolygonNode2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("new_polygon"), &PolygonNode2D::new_polygon);
	ClassDB::bind_method(D_METHOD("add_polygon", "vertices"), &PolygonNode2D::add_polygon);
	ClassDB::bind_method(D_METHOD("add_polygon_at_index", "index", "polygon"), &PolygonNode2D::add_polygon_at_index);
	ClassDB::bind_method(D_METHOD("remove_polygon", "index"), &PolygonNode2D::remove_polygon);
}
