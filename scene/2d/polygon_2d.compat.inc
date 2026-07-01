/**************************************************************************/
/*  polygon_2d.compat.inc                                                 */
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

#ifndef DISABLE_DEPRECATED

#include "polygon_2d.h"

#include "core/object/class_db.h"

void Polygon2D::_set_vertex_colors_118023(const Vector<Color> &p_colors) {
	TypedDictionary<int, Color> new_colors;
	for (int i = 0; i < p_colors.size(); i++) {
		new_colors.set(i, p_colors[i]);
	}
	set_vertex_colors(new_colors);
}

Vector<Color> Polygon2D::_get_vertex_colors_118023() const {
	Vector<Color> ret;
	TypedDictionary<int, Color> _vertex_colors = get_vertex_colors();
	ret.resize(get_polygon().size());
	for (int i = 0; i < ret.size(); i++) {
		ret.write[i] = _vertex_colors.get(i, get_color());
	}
	return ret;
}

void Polygon2D::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("get_vertex_colors"), &Polygon2D::_get_vertex_colors_118023);
	ClassDB::bind_compatibility_method(D_METHOD("set_vertex_colors", "colors"), &Polygon2D::_set_vertex_colors_118023);
}

#endif // DISABLE_DEPRECATED
