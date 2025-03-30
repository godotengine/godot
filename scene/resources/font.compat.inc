/**************************************************************************/
/*  font.compat.inc                                                       */
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

void Font::_draw_string_bind_compat_104872(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int p_font_size, const Color &p_modulate, BitField<TextServer::JustificationFlag> p_jst_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	draw_string(p_canvas_item, p_pos, p_text, p_alignment, p_width, p_font_size, p_modulate, p_jst_flags, p_direction, p_orientation, 0.0);
}

void Font::_draw_multiline_string_bind_compat_104872(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int p_font_size, int p_max_lines, const Color &p_modulate, BitField<TextServer::LineBreakFlag> p_brk_flags, BitField<TextServer::JustificationFlag> p_jst_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	draw_multiline_string(p_canvas_item, p_pos, p_text, p_alignment, p_width, p_font_size, p_max_lines, p_modulate, p_brk_flags, p_jst_flags, p_direction, p_orientation, 0.0);
}

void Font::_draw_string_outline_bind_compat_104872(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int p_font_size, int p_size, const Color &p_modulate, BitField<TextServer::JustificationFlag> p_jst_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	draw_string_outline(p_canvas_item, p_pos, p_text, p_alignment, p_width, p_font_size, p_size, p_modulate, p_jst_flags, p_direction, p_orientation, 0.0);
}

void Font::_draw_multiline_string_outline_bind_compat_104872(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int p_font_size, int p_max_lines, int p_size, const Color &p_modulate, BitField<TextServer::LineBreakFlag> p_brk_flags, BitField<TextServer::JustificationFlag> p_jst_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	draw_multiline_string_outline(p_canvas_item, p_pos, p_text, p_alignment, p_width, p_font_size, p_max_lines, p_size, p_modulate, p_brk_flags, p_jst_flags, p_direction, p_orientation, 0.0);
}

real_t Font::_draw_char_bind_compat_104872(RID p_canvas_item, const Point2 &p_pos, char32_t p_char, int p_font_size, const Color &p_modulate) const {
	return draw_char(p_canvas_item, p_pos, p_char, p_font_size, p_modulate, 0.0);
}

real_t Font::_draw_char_outline_bind_compat_104872(RID p_canvas_item, const Point2 &p_pos, char32_t p_char, int p_font_size, int p_size, const Color &p_modulate) const {
	return draw_char_outline(p_canvas_item, p_pos, p_char, p_font_size, p_size, p_modulate, 0.0);
}

RID Font::_find_variation_bind_compat_80954(const Dictionary &p_variation_coordinates, int p_face_index, float p_strength, Transform2D p_transform) const {
	return find_variation(p_variation_coordinates, p_face_index, p_strength, p_transform, 0, 0, 0, 0, 0.0);
}

RID Font::_find_variation_bind_compat_87668(const Dictionary &p_variation_coordinates, int p_face_index, float p_strength, Transform2D p_transform, int p_spacing_top, int p_spacing_bottom, int p_spacing_space, int p_spacing_glyph) const {
	return find_variation(p_variation_coordinates, p_face_index, p_strength, p_transform, p_spacing_top, p_spacing_bottom, p_spacing_space, p_spacing_glyph, 0.0);
}

void Font::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("draw_string", "canvas_item", "pos", "text", "alignment", "width", "font_size", "modulate", "justification_flags", "direction", "orientation"), &Font::_draw_string_bind_compat_104872, DEFVAL(HORIZONTAL_ALIGNMENT_LEFT), DEFVAL(-1), DEFVAL(DEFAULT_FONT_SIZE), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND), DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(TextServer::ORIENTATION_HORIZONTAL));
	ClassDB::bind_compatibility_method(D_METHOD("draw_multiline_string", "canvas_item", "pos", "text", "alignment", "width", "font_size", "max_lines", "modulate", "brk_flags", "justification_flags", "direction", "orientation"), &Font::_draw_multiline_string_bind_compat_104872, DEFVAL(HORIZONTAL_ALIGNMENT_LEFT), DEFVAL(-1), DEFVAL(DEFAULT_FONT_SIZE), DEFVAL(-1), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND), DEFVAL(TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND), DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(TextServer::ORIENTATION_HORIZONTAL));

	ClassDB::bind_compatibility_method(D_METHOD("draw_string_outline", "canvas_item", "pos", "text", "alignment", "width", "font_size", "size", "modulate", "justification_flags", "direction", "orientation"), &Font::_draw_string_outline_bind_compat_104872, DEFVAL(HORIZONTAL_ALIGNMENT_LEFT), DEFVAL(-1), DEFVAL(DEFAULT_FONT_SIZE), DEFVAL(1), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND), DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(TextServer::ORIENTATION_HORIZONTAL));
	ClassDB::bind_compatibility_method(D_METHOD("draw_multiline_string_outline", "canvas_item", "pos", "text", "alignment", "width", "font_size", "max_lines", "size", "modulate", "brk_flags", "justification_flags", "direction", "orientation"), &Font::_draw_multiline_string_outline_bind_compat_104872, DEFVAL(HORIZONTAL_ALIGNMENT_LEFT), DEFVAL(-1), DEFVAL(DEFAULT_FONT_SIZE), DEFVAL(-1), DEFVAL(1), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND), DEFVAL(TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND), DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(TextServer::ORIENTATION_HORIZONTAL));

	ClassDB::bind_compatibility_method(D_METHOD("draw_char", "canvas_item", "pos", "char", "font_size", "modulate"), &Font::_draw_char_bind_compat_104872, DEFVAL(Color(1.0, 1.0, 1.0)));
	ClassDB::bind_compatibility_method(D_METHOD("draw_char_outline", "canvas_item", "pos", "char", "font_size", "size", "modulate"), &Font::_draw_char_outline_bind_compat_104872, DEFVAL(-1), DEFVAL(Color(1.0, 1.0, 1.0)));

	ClassDB::bind_compatibility_method(D_METHOD("find_variation", "variation_coordinates", "face_index", "strength", "transform"), &Font::_find_variation_bind_compat_80954, DEFVAL(0), DEFVAL(0.0), DEFVAL(Transform2D()));
	ClassDB::bind_compatibility_method(D_METHOD("find_variation", "variation_coordinates", "face_index", "strength", "transform", "spacing_top", "spacing_bottom", "spacing_space", "spacing_glyph"), &Font::_find_variation_bind_compat_87668, DEFVAL(0), DEFVAL(0.0), DEFVAL(Transform2D()), DEFVAL(0), DEFVAL(0), DEFVAL(0), DEFVAL(0));
}

#endif
