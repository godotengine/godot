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

void CanvasItem::_draw_circle_bind_compat_84472(const Point2 &p_pos, real_t p_radius, const Color &p_color) {
	draw_circle(p_pos, p_radius, p_color, true, -1.0, false);
}

void CanvasItem::_draw_rect_bind_compat_84523(const Rect2 &p_rect, const Color &p_color, bool p_filled, real_t p_width) {
	draw_rect(p_rect, p_color, p_filled, p_width, false);
}

void CanvasItem::_draw_dashed_line_bind_compat_84523(const Point2 &p_from, const Point2 &p_to, const Color &p_color, real_t p_width, real_t p_dash, bool p_aligned) {
	draw_dashed_line(p_from, p_to, p_color, p_width, p_dash, p_aligned, false);
}

void CanvasItem::_draw_multiline_bind_compat_84523(const Vector<Point2> &p_points, const Color &p_color, real_t p_width) {
	draw_multiline(p_points, p_color, p_width, false);
}

void CanvasItem::_draw_multiline_colors_bind_compat_84523(const Vector<Point2> &p_points, const Vector<Color> &p_colors, real_t p_width) {
	draw_multiline_colors(p_points, p_colors, p_width, false);
}

void CanvasItem::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("draw_circle", "position", "radius", "color"), &CanvasItem::_draw_circle_bind_compat_84472);
	ClassDB::bind_compatibility_method(D_METHOD("draw_rect", "rect", "color", "filled", "width"), &CanvasItem::_draw_rect_bind_compat_84523, DEFVAL(true), DEFVAL(-1.0));
	ClassDB::bind_compatibility_method(D_METHOD("draw_dashed_line", "from", "to", "color", "width", "dash", "aligned"), &CanvasItem::_draw_dashed_line_bind_compat_84523, DEFVAL(-1.0), DEFVAL(2.0), DEFVAL(true));
	ClassDB::bind_compatibility_method(D_METHOD("draw_multiline", "points", "color", "width"), &CanvasItem::_draw_multiline_bind_compat_84523, DEFVAL(-1.0));
	ClassDB::bind_compatibility_method(D_METHOD("draw_multiline_colors", "points", "colors", "width"), &CanvasItem::_draw_multiline_colors_bind_compat_84523, DEFVAL(-1.0));
}

#endif
