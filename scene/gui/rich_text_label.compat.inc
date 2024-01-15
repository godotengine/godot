/**************************************************************************/
/*  rich_text_label.compat.inc                                            */
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

void RichTextLabel::_push_meta_bind_compat_89024(const Variant &p_meta) {
	push_meta(p_meta, RichTextLabel::MetaUnderline::META_UNDERLINE_ALWAYS);
}

void RichTextLabel::_add_image_bind_compat_80410(const Ref<Texture2D> &p_image, const int p_width, const int p_height, const Color &p_color, InlineAlignment p_alignment, const Rect2 &p_region) {
	add_image(p_image, p_width, p_height, p_color, p_alignment, p_region, Variant(), false, String(), false);
}

void RichTextLabel::_push_dropcap_compat_87243(const String &p_string, const Ref<Font> &p_font, int p_size, const Rect2 &p_dropcap_margins, const Color &p_color, int p_ol_size, const Color &p_ol_color) {
	push_dropcap(p_string, p_font, (float)p_size, p_dropcap_margins, p_color, (float)p_ol_size, p_ol_color);
}

void RichTextLabel::_push_font_compat_87243(const Ref<Font> &p_font, int p_size) {
	push_font(p_font, (float)p_size);
}

void RichTextLabel::_push_font_size_compat_87243(int p_font_size) {
	push_font_size((float)p_font_size);
}

void RichTextLabel::_push_outline_size_compat_87243(int p_font_size) {
	push_outline_size((float)p_font_size);
}

void RichTextLabel::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("push_meta", "data"), &RichTextLabel::_push_meta_bind_compat_89024);

	ClassDB::bind_compatibility_method(D_METHOD("set_table_column_expand", "column", "expand", "ratio"), &RichTextLabel::set_table_column_expand);

	ClassDB::bind_compatibility_method(D_METHOD("add_image", "image", "width", "height", "color", "inline_align", "region"), &RichTextLabel::_add_image_bind_compat_80410, DEFVAL(0), DEFVAL(0), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(INLINE_ALIGNMENT_CENTER), DEFVAL(Rect2()));

	ClassDB::bind_compatibility_method(D_METHOD("push_dropcap", "string", "font", "size", "dropcap_margins", "color", "outline_size", "outline_color"), &RichTextLabel::_push_dropcap_compat_87243, DEFVAL(Rect2()), DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Color(0, 0, 0, 0)));
	ClassDB::bind_compatibility_method(D_METHOD("push_font", "font", "font_size"), &RichTextLabel::_push_font_compat_87243, DEFVAL(0));
	ClassDB::bind_compatibility_method(D_METHOD("push_font", "font", "font_size"), &RichTextLabel::_push_font_compat_87243);
	ClassDB::bind_compatibility_method(D_METHOD("push_font_size", "font_size"), &RichTextLabel::_push_font_size_compat_87243);
	ClassDB::bind_compatibility_method(D_METHOD("push_outline_size", "outline_size"), &RichTextLabel::_push_outline_size_compat_87243);
}

#endif // DISABLE_DEPRECATED
