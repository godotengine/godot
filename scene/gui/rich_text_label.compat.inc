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

void RichTextLabel::_push_font_bind_compat_79053(const Ref<Font> &p_font, int p_size) {
	push_font(p_font, p_size);
}

void RichTextLabel::_set_table_column_expand_bind_compat_79053(int p_column, bool p_expand, int p_ratio) {
	set_table_column_expand(p_column, p_expand, p_ratio, true);
}

void RichTextLabel::_set_table_column_expand_bind_compat_101482(int p_column, bool p_expand, int p_ratio) {
	set_table_column_expand(p_column, p_expand, p_ratio, true);
}

void RichTextLabel::_push_meta_bind_compat_99481(const Variant &p_meta, MetaUnderline p_underline_mode) {
	push_meta(p_meta, p_underline_mode, String());
}

void RichTextLabel::_push_meta_bind_compat_89024(const Variant &p_meta) {
	push_meta(p_meta, RichTextLabel::MetaUnderline::META_UNDERLINE_ALWAYS, String());
}

void RichTextLabel::_add_image_bind_compat_80410(const Ref<Texture2D> &p_image, const int p_width, const int p_height, const Color &p_color, InlineAlignment p_alignment, const Rect2 &p_region) {
	add_image(p_image, p_width, p_height, p_color, p_alignment, p_region, Variant(), false, String(), false, false, String());
}

void RichTextLabel::_add_image_bind_compat_76829(const Ref<Texture2D> &p_image, const int p_width, const int p_height, const Color &p_color, InlineAlignment p_alignment, const Rect2 &p_region, const Variant &p_key, bool p_pad, const String &p_tooltip, bool p_size_in_percent) {
	add_image(p_image, p_width, p_height, p_color, p_alignment, p_region, p_key, p_pad, p_tooltip, p_size_in_percent, p_size_in_percent, String());
}

void RichTextLabel::_push_table_bind_compat_76829(int p_columns, InlineAlignment p_alignment, int p_align_to_row) {
	push_table(p_columns, p_alignment, p_align_to_row, String());
}

bool RichTextLabel::_remove_paragraph_bind_compat_91098(int p_paragraph) {
	return remove_paragraph(p_paragraph, false);
}

void RichTextLabel::_push_underline_bind_compat_106300() {
	push_underline(Color(0, 0, 0, 0));
}

void RichTextLabel::_push_strikethrough_bind_compat_106300() {
	push_strikethrough(Color(0, 0, 0, 0));
}

void RichTextLabel::_add_image_bind_compat_107347(const Ref<Texture2D> &p_image, int p_width, int p_height, const Color &p_color, InlineAlignment p_alignment, const Rect2 &p_region, const Variant &p_key, bool p_pad, const String &p_tooltip, bool p_size_in_percent, const String &p_alt_text) {
	add_image(p_image, p_width, p_height, p_color, p_alignment, p_region, p_key, p_pad, p_tooltip, p_size_in_percent, p_size_in_percent, p_alt_text);
}

void RichTextLabel::_update_image_bind_compat_107347(const Variant &p_key, BitField<ImageUpdateMask> p_mask, const Ref<Texture2D> &p_image, int p_width, int p_height, const Color &p_color, InlineAlignment p_alignment, const Rect2 &p_region, bool p_pad, const String &p_tooltip, bool p_size_in_percent) {
	update_image(p_key, p_mask, p_image, p_width, p_height, p_color, p_alignment, p_region, p_pad, p_tooltip, p_size_in_percent, p_size_in_percent);
}

void RichTextLabel::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("push_font", "font", "font_size"), &RichTextLabel::_push_font_bind_compat_79053);
	ClassDB::bind_compatibility_method(D_METHOD("set_table_column_expand", "column", "expand", "ratio"), &RichTextLabel::_set_table_column_expand_bind_compat_79053);
	ClassDB::bind_compatibility_method(D_METHOD("set_table_column_expand", "column", "expand", "ratio"), &RichTextLabel::_set_table_column_expand_bind_compat_101482, DEFVAL(1));
	ClassDB::bind_compatibility_method(D_METHOD("push_meta", "data", "underline_mode"), &RichTextLabel::_push_meta_bind_compat_99481, DEFVAL(META_UNDERLINE_ALWAYS));
	ClassDB::bind_compatibility_method(D_METHOD("push_meta", "data"), &RichTextLabel::_push_meta_bind_compat_89024);
	ClassDB::bind_compatibility_method(D_METHOD("add_image", "image", "width", "height", "color", "inline_align", "region"), &RichTextLabel::_add_image_bind_compat_80410, DEFVAL(0), DEFVAL(0), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(INLINE_ALIGNMENT_CENTER), DEFVAL(Rect2()));
	ClassDB::bind_compatibility_method(D_METHOD("add_image", "image", "width", "height", "color", "inline_align", "region", "key", "pad", "tooltip", "size_in_percent"), &RichTextLabel::_add_image_bind_compat_76829, DEFVAL(0), DEFVAL(0), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(INLINE_ALIGNMENT_CENTER), DEFVAL(Rect2()), DEFVAL(Variant()), DEFVAL(false), DEFVAL(String()), DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("push_table", "columns", "inline_align", "align_to_row"), &RichTextLabel::_push_table_bind_compat_76829, DEFVAL(INLINE_ALIGNMENT_TOP), DEFVAL(-1));
	ClassDB::bind_compatibility_method(D_METHOD("remove_paragraph", "paragraph"), &RichTextLabel::_remove_paragraph_bind_compat_91098);
	ClassDB::bind_compatibility_method(D_METHOD("push_underline"), &RichTextLabel::_push_underline_bind_compat_106300);
	ClassDB::bind_compatibility_method(D_METHOD("push_strikethrough"), &RichTextLabel::_push_strikethrough_bind_compat_106300);
	ClassDB::bind_compatibility_method(D_METHOD("add_image", "image", "width", "height", "color", "inline_align", "region", "key", "pad", "tooltip", "size_in_percent", "alt_text"), &RichTextLabel::_add_image_bind_compat_107347, DEFVAL(0), DEFVAL(0), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(INLINE_ALIGNMENT_CENTER), DEFVAL(Rect2()), DEFVAL(Variant()), DEFVAL(false), DEFVAL(String()), DEFVAL(false), DEFVAL(String()));
	ClassDB::bind_compatibility_method(D_METHOD("update_image", "key", "mask", "image", "width", "height", "color", "inline_align", "region", "pad", "tooltip", "size_in_percent"), &RichTextLabel::_update_image_bind_compat_107347, DEFVAL(0), DEFVAL(0), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(INLINE_ALIGNMENT_CENTER), DEFVAL(Rect2()), DEFVAL(false), DEFVAL(String()), DEFVAL(false));
}

#endif // DISABLE_DEPRECATED
