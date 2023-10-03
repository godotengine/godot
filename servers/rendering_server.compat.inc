/**************************************************************************/
/*  rendering_server.compat.inc                                           */
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

#include "rendering_server.h"

void RenderingServer::canvas_item_add_line_compat_79666(RID p_item, const Point2 &p_from, const Point2 &p_to, const Color &p_color, float p_width, bool p_antialiased) {
	canvas_item_add_line(p_item, p_from, p_to, p_color, p_width, p_antialiased);
}

void RenderingServer::canvas_item_add_rect_compat_79666(RID p_item, const Rect2 &p_rect, const Color &p_color) {
	canvas_item_add_rect(p_item, p_rect, p_color);
}

void RenderingServer::canvas_item_add_circle_compat_79666(RID p_item, const Point2 &p_pos, float p_radius, const Color &p_color) {
	canvas_item_add_circle(p_item, p_pos, p_radius, p_color);
}

void RenderingServer::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("canvas_item_add_line", "item", "from", "to", "color", "width", "antialiased"), &RenderingServer::canvas_item_add_line_compat_79666, DEFVAL(-1.0), DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("canvas_item_add_rect", "item", "rect", "color"), &RenderingServer::canvas_item_add_rect_compat_79666);
	ClassDB::bind_compatibility_method(D_METHOD("canvas_item_add_circle", "item", "pos", "radius", "color"), &RenderingServer::canvas_item_add_circle_compat_79666);
}

#endif
