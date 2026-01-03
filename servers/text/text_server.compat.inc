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

bool TextServer::_save_support_data_bind_compat_XXXX(const String &p_filename) const {
	return save_support_data(p_filename, "full");
}

PackedByteArray TextServer::_get_support_data_bind_compat_XXXX() const {
	return get_support_data("full");
}

void TextServer::_font_draw_glyph_bind_compat_104872(const RID &p_font, const RID &p_canvas, int64_t p_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color) const {
	font_draw_glyph(p_font, p_canvas, p_size, p_pos, p_index, p_color, 0.0);
}

void TextServer::_font_draw_glyph_outline_bind_compat_104872(const RID &p_font, const RID &p_canvas, int64_t p_size, int64_t p_outline_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color) const {
	font_draw_glyph_outline(p_font, p_canvas, p_size, p_outline_size, p_pos, p_index, p_color, 0.0);
}

void TextServer::_shaped_text_draw_bind_compat_104872(const RID &p_shaped, const RID &p_canvas, const Vector2 &p_pos, double p_clip_l, double p_clip_r, const Color &p_color) const {
	shaped_text_draw(p_shaped, p_canvas, p_pos, p_clip_l, p_clip_r, p_color, 0.0);
}

void TextServer::_shaped_text_draw_outline_bind_compat_104872(const RID &p_shaped, const RID &p_canvas, const Vector2 &p_pos, double p_clip_l, double p_clip_r, int64_t p_outline_size, const Color &p_color) const {
	shaped_text_draw_outline(p_shaped, p_canvas, p_pos, p_clip_l, p_clip_r, p_outline_size, p_color, 0.0);
}

PackedInt32Array TextServer::_shaped_text_get_word_breaks_bind_compat_90732(const RID &p_shaped, BitField<TextServer::GraphemeFlag> p_grapheme_flags) const {
	return shaped_text_get_word_breaks(p_shaped, p_grapheme_flags, 0);
}

void TextServer::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("save_support_data", "filename"), &TextServer::_save_support_data_bind_compat_XXXX);
	ClassDB::bind_compatibility_method(D_METHOD("get_support_data"), &TextServer::_get_support_data_bind_compat_XXXX);
	ClassDB::bind_compatibility_method(D_METHOD("font_draw_glyph", "font_rid", "canvas", "size", "pos", "index", "color"), &TextServer::_font_draw_glyph_bind_compat_104872, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_compatibility_method(D_METHOD("font_draw_glyph_outline", "font_rid", "canvas", "size", "outline_size", "pos", "index", "color"), &TextServer::_font_draw_glyph_outline_bind_compat_104872, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_compatibility_method(D_METHOD("shaped_text_draw", "shaped", "canvas", "pos", "clip_l", "clip_r", "color"), &TextServer::_shaped_text_draw_bind_compat_104872, DEFVAL(-1), DEFVAL(-1), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_compatibility_method(D_METHOD("shaped_text_draw_outline", "shaped", "canvas", "pos", "clip_l", "clip_r", "outline_size", "color"), &TextServer::_shaped_text_draw_outline_bind_compat_104872, DEFVAL(-1), DEFVAL(-1), DEFVAL(1), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_compatibility_method(D_METHOD("shaped_text_get_word_breaks", "shaped", "grapheme_flags"), &TextServer::_shaped_text_get_word_breaks_bind_compat_90732, DEFVAL(GRAPHEME_IS_SPACE | GRAPHEME_IS_PUNCTUATION));
}

#endif
