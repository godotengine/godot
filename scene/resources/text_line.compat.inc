/**************************************************************************/
/*  text_line.compat.inc                                                  */
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

bool TextLine::_add_string_compat_87243(const String &p_text, const Ref<Font> &p_font, int p_font_size, const String &p_language, const Variant &p_meta) {
	return add_string(p_text, p_font, (float)p_font_size, p_language, p_meta);
}

void TextLine::_draw_outline_compat_87243(RID p_canvas, const Vector2 &p_pos, int p_outline_size, const Color &p_color) const {
	draw_outline(p_canvas, p_pos, (float)p_outline_size, p_color);
}

void TextLine::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("add_string", "text", "font", "font_size", "language", "meta"), &TextLine::_add_string_compat_87243, DEFVAL(""), DEFVAL(Variant()));
	ClassDB::bind_compatibility_method(D_METHOD("draw_outline", "canvas", "pos", "outline_size", "color"), &TextLine::_draw_outline_compat_87243, DEFVAL(1), DEFVAL(Color(1, 1, 1)));
}

#endif
