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

PackedInt32Array TextServer::_shaped_text_get_word_breaks_bind_compat_90732(const RID &p_shaped, BitField<TextServer::GraphemeFlag> p_grapheme_flags) const {
	return shaped_text_get_word_breaks(p_shaped, p_grapheme_flags, 0);
}

int64_t TextServer::_shaped_text_get_trim_pos_bind_compat_99236(const RID &p_shaped) const {
	bool is_rtl = shaped_text_get_inferred_direction(p_shaped) == DIRECTION_RTL;

	return shaped_text_get_trim_pos(p_shaped, is_rtl);
}

int64_t TextServer::_shaped_text_get_ellipsis_pos_bind_compat_99236(const RID &p_shaped) const {
	bool is_rtl = shaped_text_get_inferred_direction(p_shaped) == DIRECTION_RTL;

	return shaped_text_get_ellipsis_pos(p_shaped, is_rtl);
}

void TextServer::_shaped_text_overrun_trim_to_width_bind_compat_99236(const RID &p_shaped, double p_width, BitField<TextServer::TextOverrunFlag> p_trim_flags) {
	shaped_text_overrun_trim_to_width(p_shaped, p_width, p_trim_flags, OVERRUN_TRIM_END);
}

void TextServer::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("shaped_text_get_word_breaks", "shaped", "grapheme_flags"), &TextServer::_shaped_text_get_word_breaks_bind_compat_90732, DEFVAL(GRAPHEME_IS_SPACE | GRAPHEME_IS_PUNCTUATION));
	ClassDB::bind_compatibility_method(D_METHOD("shaped_text_get_trim_pos", "shaped"), &TextServer::_shaped_text_get_trim_pos_bind_compat_99236);
	ClassDB::bind_compatibility_method(D_METHOD("shaped_text_get_ellipsis_pos", "shaped"), &TextServer::_shaped_text_get_ellipsis_pos_bind_compat_99236);
	ClassDB::bind_compatibility_method(D_METHOD("shaped_text_overrun_trim_to_width", "shaped", "width", "overrun_trim_flags"), &TextServer::_shaped_text_overrun_trim_to_width_bind_compat_99236, DEFVAL(0), DEFVAL(OVERRUN_NO_TRIM));
}

#endif
