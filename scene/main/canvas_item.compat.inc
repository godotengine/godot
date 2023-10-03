/**************************************************************************/
/*  canvas_item.compat.inc                                                */
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

#include "canvas_item.h"

void CanvasItem::draw_dashed_line_compat_79666(const Point2 &p_from, const Point2 &p_to, const Color &p_color, real_t p_width, real_t p_dash, bool p_aligned) {
	draw_dashed_line(p_from, p_to, p_color, p_width, p_dash, p_aligned);
}

void CanvasItem::draw_line_compat_79666(const Point2 &p_from, const Point2 &p_to, const Color &p_color, real_t p_width, bool p_antialiased) {
	draw_line(p_from, p_to, p_color, p_width, p_antialiased);
}

void CanvasItem::draw_polyline_compat_79666(const Vector<Point2> &p_points, const Color &p_color, real_t p_width, bool p_antialiased) {
	draw_polyline(p_points, p_color, p_width, p_antialiased);
}

void CanvasItem::draw_arc_compat_79666(const Vector2 &p_center, real_t p_radius, real_t p_start_angle, real_t p_end_angle, int p_point_count, const Color &p_color, real_t p_width, bool p_antialiased) {
	draw_arc(p_center, p_radius, p_start_angle, p_end_angle, p_point_count, p_color, p_width, p_antialiased);
}

void CanvasItem::draw_multiline_compat_79666(const Vector<Point2> &p_points, const Color &p_color, real_t p_width) {
	draw_multiline(p_points, p_color, p_width);
}

void CanvasItem::draw_rect_compat_79666(const Rect2 &p_rect, const Color &p_color, bool p_filled, real_t p_width) {
	draw_rect(p_rect, p_color, p_filled, p_width);
}

void CanvasItem::draw_circle_compat_79666(const Point2 &p_pos, real_t p_radius, const Color &p_color) {
	draw_circle(p_pos, p_radius, p_color);
}

void CanvasItem::draw_colored_polygon_compat_79666(const Vector<Point2> &p_points, const Color &p_color, const Vector<Point2> &p_uvs, Ref<Texture2D> p_texture) {
	draw_colored_polygon(p_points, p_color, p_uvs, p_texture);
}

void CanvasItem::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("draw_dashed_line", "from", "to", "color", "width", "dash", "aligned"), &CanvasItem::draw_dashed_line_compat_79666, DEFVAL(-1.0), DEFVAL(2.0), DEFVAL(true));
	ClassDB::bind_compatibility_method(D_METHOD("draw_line", "from", "to", "color", "width", "antialiased"), &CanvasItem::draw_line_compat_79666, DEFVAL(-1.0), DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("draw_polyline", "points", "color", "width", "antialiased"), &CanvasItem::draw_polyline_compat_79666, DEFVAL(-1.0), DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("draw_arc", "center", "radius", "start_angle", "end_angle", "point_count", "color", "width", "antialiased"), &CanvasItem::draw_arc_compat_79666, DEFVAL(-1.0), DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("draw_multiline", "points", "color", "width"), &CanvasItem::draw_multiline_compat_79666, DEFVAL(-1.0));
	ClassDB::bind_compatibility_method(D_METHOD("draw_rect", "rect", "color", "filled", "width"), &CanvasItem::draw_rect_compat_79666, DEFVAL(true), DEFVAL(-1.0));
	ClassDB::bind_compatibility_method(D_METHOD("draw_circle", "pos", "radius", "color"), &CanvasItem::draw_circle_compat_79666);
	ClassDB::bind_compatibility_method(D_METHOD("draw_colored_polygon", "points", "color", "uvs", "texture"), &CanvasItem::draw_colored_polygon_compat_79666, DEFVAL(PackedVector2Array()), DEFVAL(Ref<Texture2D>()));
}

#endif
