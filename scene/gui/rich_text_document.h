/**************************************************************************/
/*  rich_text_document.h                                                  */
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

#pragma once

#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"
#include "scene/resources/texture.h"

class RichTextDocument {
public:
	static constexpr char32_t OBJECT_REPLACEMENT_CHAR = 0xfffc;

	struct Style {
		bool bold = false;
		bool italic = false;
		bool has_underline = false;
		bool underline = false;
		bool strikethrough = false;
		bool code = false;
		bool has_color = false;
		bool has_bg_color = false;
		bool has_fg_color = false;
		bool has_outline_color = false;
		bool has_outline_size = false;
		bool has_url = false;
		bool url_visited = false;
		Color color;
		Color bg_color;
		Color fg_color;
		Color outline_color;
		int outline_size = 0;
		int font_size = 0;
		int alignment = -1;
		int indent_level = 0;
		int list_type = 0;
		int list_start = -1;
		bool list_capitalize = false;
		String font;
		String url;
		String url_tooltip;
		String language;
		String block_tag;
		Vector<String> preserved_tags;

		bool operator==(const Style &p_other) const;
		bool operator!=(const Style &p_other) const { return !(*this == p_other); }
		bool is_default() const;
	};

	struct StyleSpan {
		int from = 0;
		int to = 0;
		Style style;
	};

	struct InlineImage {
		int offset = 0;
		String source;
		HashMap<String, String> options;
		Ref<Texture2D> texture;

		bool operator==(const InlineImage &p_other) const {
			if (offset != p_other.offset || source != p_other.source || options.size() != p_other.options.size()) {
				return false;
			}
			for (const KeyValue<String, String> &E : options) {
				const HashMap<String, String>::ConstIterator other_option = p_other.options.find(E.key);
				if (!other_option || other_option->value != E.value) {
					return false;
				}
			}
			return true;
		}
		bool operator!=(const InlineImage &p_other) const { return !(*this == p_other); }
	};

	struct RawInline {
		int offset = 0;
		String bbcode;

		bool operator==(const RawInline &p_other) const { return offset == p_other.offset && bbcode == p_other.bbcode; }
		bool operator!=(const RawInline &p_other) const { return !(*this == p_other); }
	};

	String text;
	Vector<StyleSpan> spans;
	Vector<InlineImage> images;
	Vector<RawInline> raw_inlines;

	static Ref<Texture2D> load_image_texture(const String &p_source);
	static RichTextDocument parse_bbcode(const String &p_bbcode);
	String to_bbcode() const;

private:
	struct ParsedTag {
		String raw;
		String name;
		String value;
		HashMap<String, String> options;
		bool closing = false;
	};

	static int _find_unquoted(const String &p_src, char32_t p_chr, int p_from);
	static Vector<String> _split_unquoted(const String &p_src, char32_t p_splitter);
	static ParsedTag _parse_tag(const String &p_tag);
	static bool _parse_color(const String &p_text, Color &r_color);
	static bool _is_preserved_pair_tag(const String &p_name);
	static String _escape_text(const String &p_text);
	static String _close_tag_for(const String &p_tag);
};
