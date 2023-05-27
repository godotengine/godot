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

#include "rich_text_label.h"

void RichTextLabel::push_list_compat_75017(int p_level, ListType p_list, bool p_capitalize) {
	push_list(p_level, p_list, p_capitalize);
}

void RichTextLabel::push_paragraph_compat_75250(HorizontalAlignment p_alignment, Control::TextDirection p_direction, const String &p_language, TextServer::StructuredTextParser p_st_parser) {
	push_paragraph(p_alignment, p_direction, p_language, p_st_parser);
}
void RichTextLabel::push_paragraph_compat_76401(HorizontalAlignment p_alignment, Control::TextDirection p_direction, const String &p_language, TextServer::StructuredTextParser p_st_parser, BitField<TextServer::JustificationFlag> p_jst_flags) {
	push_paragraph(p_alignment, p_direction, p_language, p_st_parser, p_jst_flags);
}

void RichTextLabel::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("push_list", "level", "type", "capitalize"), &RichTextLabel::push_list_compat_75017);
	ClassDB::bind_compatibility_method(D_METHOD("push_paragraph", "alignment", "base_direction", "language", "st_parser"), &RichTextLabel::push_paragraph_compat_75250, DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(""), DEFVAL(TextServer::STRUCTURED_TEXT_DEFAULT));
	ClassDB::bind_compatibility_method(D_METHOD("push_paragraph", "alignment", "base_direction", "language", "st_parser", "justification_flags"), &RichTextLabel::push_paragraph_compat_76401, DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(""), DEFVAL(TextServer::STRUCTURED_TEXT_DEFAULT), DEFVAL(TextServer::JUSTIFICATION_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_SKIP_LAST_LINE | TextServer::JUSTIFICATION_DO_NOT_SKIP_SINGLE_LINE));
}

#endif
