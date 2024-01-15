/**************************************************************************/
/*  text_server.compat.inc                                                */
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

bool TextServer::_shaped_text_add_string_compat_87243(const RID &p_shaped, const String &p_text, const TypedArray<RID> &p_fonts, int64_t p_size, const Dictionary &p_opentype_features, const String &p_language, const Variant &p_meta) {
	return shaped_text_add_string(p_shaped, p_text, p_fonts, (double)p_size, p_opentype_features, p_language, p_meta);
}

void TextServer::_shaped_set_span_update_font_compat_87243(const RID &p_shaped, int64_t p_index, const TypedArray<RID> &p_fonts, int64_t p_size, const Dictionary &p_opentype_features) {
	shaped_set_span_update_font(p_shaped, p_index, p_fonts, (double)p_size, p_opentype_features);
}

void TextServer::_shaped_text_draw_outline_compat_87243(const RID &p_shaped, const RID &p_canvas, const Vector2 &p_pos, double p_clip_l, double p_clip_r, int64_t p_outline_size, const Color &p_color) const {
	shaped_text_draw_outline(p_shaped, p_canvas, p_pos, p_clip_l, p_clip_r, (double)p_outline_size, p_color);
}

void TextServer::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("shaped_text_add_string", "shaped", "text", "fonts", "size", "opentype_features", "language", "meta"), &TextServer::_shaped_text_add_string_compat_87243, DEFVAL(Dictionary()), DEFVAL(""), DEFVAL(Variant()));
	ClassDB::bind_compatibility_method(D_METHOD("shaped_text_draw_outline", "shaped", "canvas", "pos", "clip_l", "clip_r", "outline_size", "color"), &TextServer::_shaped_text_draw_outline_compat_87243, DEFVAL(-1), DEFVAL(-1), DEFVAL(1), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_compatibility_method(D_METHOD("shaped_set_span_update_font", "shaped", "index", "fonts", "size", "opentype_features"), &TextServer::_shaped_set_span_update_font_compat_87243, DEFVAL(Dictionary()));
}

#endif
