/**************************************************************************/
/*  rich_text_document.cpp                                                */
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

#include "rich_text_document.h"

#include "core/config/project_settings.h"
#include "core/io/image.h"
#include "core/io/resource_loader.h"
#include "scene/resources/image_texture.h"

static String quote_bbcode_option_value(const String &p_value) {
	if (p_value.is_empty()) {
		return "\"\"";
	}
	if (p_value.find_char(' ') >= 0 || p_value.find_char('\t') >= 0 || p_value.find_char('[') >= 0 || p_value.find_char(']') >= 0 || p_value.find_char('"') >= 0) {
		return "\"" + p_value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
	}
	return p_value;
}

struct BBCodeTag {
	String open;
	String close;

	bool operator==(const BBCodeTag &p_other) const {
		return open == p_other.open && close == p_other.close;
	}
};

struct BBCodeStyleSpanCompare {
	_FORCE_INLINE_ bool operator()(const RichTextDocument::StyleSpan &p_a, const RichTextDocument::StyleSpan &p_b) const {
		if (p_a.from == p_b.from) {
			return p_a.to < p_b.to;
		}
		return p_a.from < p_b.from;
	}
};

bool RichTextDocument::Style::operator==(const Style &p_other) const {
	return bold == p_other.bold &&
			italic == p_other.italic &&
			has_underline == p_other.has_underline &&
			underline == p_other.underline &&
			strikethrough == p_other.strikethrough &&
			code == p_other.code &&
			has_color == p_other.has_color &&
			(!has_color || color == p_other.color) &&
			has_bg_color == p_other.has_bg_color &&
			(!has_bg_color || bg_color == p_other.bg_color) &&
			has_fg_color == p_other.has_fg_color &&
			(!has_fg_color || fg_color == p_other.fg_color) &&
			has_outline_color == p_other.has_outline_color &&
			(!has_outline_color || outline_color == p_other.outline_color) &&
			has_outline_size == p_other.has_outline_size &&
			(!has_outline_size || outline_size == p_other.outline_size) &&
			has_url == p_other.has_url &&
			url_visited == p_other.url_visited &&
			font_size == p_other.font_size &&
			alignment == p_other.alignment &&
			indent_level == p_other.indent_level &&
			list_type == p_other.list_type &&
			list_start == p_other.list_start &&
			list_capitalize == p_other.list_capitalize &&
			font == p_other.font &&
			url == p_other.url &&
			url_tooltip == p_other.url_tooltip &&
			language == p_other.language &&
			block_tag == p_other.block_tag &&
			preserved_tags == p_other.preserved_tags;
}

bool RichTextDocument::Style::is_default() const {
	return !bold && !italic && !has_underline && !underline && !strikethrough && !code &&
			!has_color && !has_bg_color && !has_fg_color && !has_outline_color && !has_outline_size &&
			font_size <= 0 && alignment < 0 && indent_level <= 0 && list_type == 0 && list_start < 0 &&
			font.is_empty() && !has_url && !url_visited && url.is_empty() && url_tooltip.is_empty() && language.is_empty() &&
			block_tag.is_empty() && preserved_tags.is_empty();
}

Ref<Texture2D> RichTextDocument::load_image_texture(const String &p_source) {
	const String source = p_source.strip_edges();
	if (source.is_empty()) {
		return Ref<Texture2D>();
	}

	String image_path = source;
	bool filesystem_path = source.is_absolute_path() || source.begins_with("user://");
	if (image_path.begins_with("file://")) {
		image_path = image_path.trim_prefix("file://").uri_decode();
		if (image_path.length() >= 3 && image_path[0] == '/' && image_path[2] == ':') {
			image_path = image_path.substr(1);
		}
		filesystem_path = true;
	}

	if (!filesystem_path || source.begins_with("res://")) {
		Ref<Texture2D> texture = ResourceLoader::load(source, "Texture2D");
		if (texture.is_valid()) {
			return texture;
		}
		texture = ResourceLoader::load(source);
		if (texture.is_valid()) {
			return texture;
		}
	}

	if (source.begins_with("res://") || source.begins_with("user://")) {
		image_path = ProjectSettings::get_singleton()->globalize_path(source);
	}
	const Ref<Image> image = Image::load_from_file(image_path);
	if (image.is_null() || image->is_empty()) {
		return Ref<Texture2D>();
	}
	return ImageTexture::create_from_image(image);
}

int RichTextDocument::_find_unquoted(const String &p_src, char32_t p_chr, int p_from) {
	bool in_single_quote = false;
	bool in_double_quote = false;
	for (int i = p_from; i < p_src.length(); i++) {
		const char32_t c = p_src[i];
		if (in_double_quote) {
			if (c == '"') {
				in_double_quote = false;
			}
		} else if (in_single_quote) {
			if (c == '\'') {
				in_single_quote = false;
			}
		} else if (c == '"') {
			in_double_quote = true;
		} else if (c == '\'') {
			in_single_quote = true;
		} else if (c == p_chr) {
			return i;
		}
	}
	return -1;
}

Vector<String> RichTextDocument::_split_unquoted(const String &p_src, char32_t p_splitter) {
	Vector<String> ret;
	int from = 0;
	while (from < p_src.length()) {
		int end = _find_unquoted(p_src, p_splitter, from);
		if (end < 0) {
			end = p_src.length();
		}
		if (end > from) {
			ret.push_back(p_src.substr(from, end - from));
		}
		from = end + 1;
	}
	return ret;
}

RichTextDocument::ParsedTag RichTextDocument::_parse_tag(const String &p_tag) {
	ParsedTag tag;
	tag.raw = p_tag;

	String raw = p_tag.strip_edges();
	if (raw.begins_with("/")) {
		tag.closing = true;
		raw = raw.substr(1).strip_edges();
	}

	Vector<String> parts = _split_unquoted(raw, ' ');
	String name = parts.is_empty() ? raw : parts[0];
	const int value_pos = name.find_char('=');
	if (value_pos >= 0) {
		tag.value = name.substr(value_pos + 1).unquote();
		name = name.substr(0, value_pos);
	}
	tag.name = name.to_lower();

	bool first_part = true;
	for (const String &part : parts) {
		if (first_part) {
			first_part = false;
			continue;
		}
		const int option_pos = part.find_char('=');
		if (option_pos >= 0) {
			tag.options[part.substr(0, option_pos).to_lower()] = part.substr(option_pos + 1).unquote();
		}
	}
	return tag;
}

bool RichTextDocument::_parse_color(const String &p_text, Color &r_color) {
	if (p_text.is_empty()) {
		return false;
	}
	const Color invalid_color = Color(0, 0, 0, -1);
	r_color = Color::from_string(p_text, invalid_color);
	return r_color.a >= 0;
}

bool RichTextDocument::_is_preserved_pair_tag(const String &p_name) {
	return p_name == "table" || p_name == "cell" || p_name == "dropcap" ||
			p_name == "wave" || p_name == "shake" || p_name == "tornado" ||
			p_name == "rainbow" || p_name == "pulse" || p_name == "fade";
}

String RichTextDocument::_escape_text(const String &p_text) {
	return p_text.replace("[", "[lb]").replace("]", "[rb]");
}

String RichTextDocument::_close_tag_for(const String &p_tag) {
	ParsedTag tag = _parse_tag(p_tag);
	return "[/" + tag.name + "]";
}

RichTextDocument RichTextDocument::parse_bbcode(const String &p_bbcode) {
	RichTextDocument doc;
	Vector<Style> style_stack;
	Vector<int> style_start_stack;
	style_stack.push_back(Style());
	style_start_stack.push_back(0);

	auto append_text = [&](const String &p_text) {
		if (p_text.is_empty()) {
			return;
		}
		const int from = doc.text.length();
		doc.text += p_text;
		const int to = doc.text.length();
		const Style &style = style_stack[style_stack.size() - 1];
		if (!style.is_default()) {
			StyleSpan span;
			span.from = from;
			span.to = to;
			span.style = style;
			doc.spans.push_back(span);
		}
	};

	String bbcode = p_bbcode.replace("\r\n", "\n");
	int pos = 0;
	while (pos < bbcode.length()) {
		const int brk_pos = bbcode.find_char('[', pos);
		if (brk_pos < 0) {
			append_text(bbcode.substr(pos));
			break;
		}
		if (brk_pos > pos) {
			append_text(bbcode.substr(pos, brk_pos - pos));
		}

		const int brk_end = _find_unquoted(bbcode, ']', brk_pos + 1);
		if (brk_end < 0) {
			append_text(bbcode.substr(brk_pos));
			break;
		}

		const String raw_tag = bbcode.substr(brk_pos + 1, brk_end - brk_pos - 1);
		const ParsedTag tag = _parse_tag(raw_tag);

		if (tag.closing) {
			if (tag.name == "p" && style_stack.size() > 1) {
				if (!doc.text.is_empty() && doc.text[doc.text.length() - 1] != '\n') {
					append_text("\n");
				}
				style_stack.resize(style_stack.size() - 1);
				style_start_stack.resize(style_start_stack.size() - 1);
				pos = brk_end + 1;
				continue;
			}
			if (style_stack.size() > 1) {
				if (tag.name == "url" && style_stack[style_stack.size() - 1].url.is_empty()) {
					const int url_from = style_start_stack[style_start_stack.size() - 1];
					const int url_to = doc.text.length();
					const String url = doc.text.substr(url_from, url_to - url_from);
					if (!url.is_empty()) {
						for (StyleSpan &span : doc.spans) {
							if (span.from >= url_from && span.to <= url_to && span.style.url.is_empty()) {
								span.style.url = url;
							}
						}
					}
				}
				style_stack.resize(style_stack.size() - 1);
				style_start_stack.resize(style_start_stack.size() - 1);
				pos = brk_end + 1;
				continue;
			}
			append_text("[" + raw_tag + "]");
			pos = brk_end + 1;
			continue;
		}

		if (tag.name == "lb") {
			append_text("[");
			pos = brk_end + 1;
			continue;
		} else if (tag.name == "rb") {
			append_text("]");
			pos = brk_end + 1;
			continue;
		} else if (tag.name == "br") {
			append_text("\n");
			pos = brk_end + 1;
			continue;
		} else if (tag.name == "lrm") {
			append_text(String::chr(0x200e));
			pos = brk_end + 1;
			continue;
		} else if (tag.name == "rlm") {
			append_text(String::chr(0x200f));
			pos = brk_end + 1;
			continue;
		} else if (tag.name == "zwj") {
			append_text(String::chr(0x200d));
			pos = brk_end + 1;
			continue;
		} else if (tag.name == "zwnj") {
			append_text(String::chr(0x200c));
			pos = brk_end + 1;
			continue;
		} else if (tag.name == "wj") {
			append_text(String::chr(0x2060));
			pos = brk_end + 1;
			continue;
		} else if (tag.name == "shy") {
			append_text(String::chr(0x00ad));
			pos = brk_end + 1;
			continue;
		} else if (tag.name == "img") {
			int end = bbcode.find("[/img]", brk_end + 1);
			if (end < 0) {
				end = bbcode.find_char('[', brk_end + 1);
			}
			if (end < 0) {
				end = bbcode.length();
			}
			InlineImage image;
			image.offset = doc.text.length();
			image.source = bbcode.substr(brk_end + 1, end - brk_end - 1).strip_edges();
			image.options = tag.options;
			if (!tag.value.is_empty()) {
				if (image.source.is_empty()) {
					image.source = tag.value;
				} else {
					image.options["size"] = tag.value;
				}
			}
			image.texture = load_image_texture(image.source);
			doc.images.push_back(image);
			append_text(String::chr(OBJECT_REPLACEMENT_CHAR));
			pos = (end < bbcode.length() && bbcode.substr(end, 6).to_lower() == "[/img]") ? end + 6 : end;
			continue;
		} else if (tag.name == "hr") {
			RawInline raw_inline;
			raw_inline.offset = doc.text.length();
			raw_inline.bbcode = "[" + raw_tag + "]";
			doc.raw_inlines.push_back(raw_inline);
			append_text(String::chr(OBJECT_REPLACEMENT_CHAR));
			pos = brk_end + 1;
			continue;
		}

		Style next_style = style_stack[style_stack.size() - 1];
		bool pushed = true;
		if (tag.name == "b") {
			next_style.bold = true;
		} else if (tag.name == "i") {
			next_style.italic = true;
		} else if (tag.name == "u") {
			next_style.has_underline = true;
			next_style.underline = true;
		} else if (tag.name == "s") {
			next_style.strikethrough = true;
		} else if (tag.name == "code") {
			next_style.code = true;
		} else if (tag.name == "url") {
			next_style.has_url = true;
			next_style.url = tag.value;
			if (next_style.url.is_empty() && tag.options.has("href")) {
				next_style.url = tag.options["href"];
			}
			if (tag.options.has("tooltip")) {
				next_style.url_tooltip = tag.options["tooltip"];
			}
			if (tag.options.has("visited")) {
				const String visited = tag.options["visited"].to_lower();
				next_style.url_visited = visited != "false" && visited != "0" && visited != "no" && visited != "off";
			}
			if (tag.options.has("underline")) {
				const String underline = tag.options["underline"].to_lower();
				if (underline == "never" || underline == "false" || underline == "0") {
					next_style.has_underline = true;
					next_style.underline = false;
				} else if (underline == "always" || underline == "true" || underline == "1") {
					next_style.has_underline = true;
					next_style.underline = true;
				}
			}
		} else if (tag.name == "color") {
			Color color;
			String color_text = tag.value;
			if (color_text.is_empty() && tag.options.has("color")) {
				color_text = tag.options["color"];
			}
			if (_parse_color(color_text, color)) {
				next_style.has_color = true;
				next_style.color = color;
			}
		} else if (tag.name == "fgcolor") {
			Color color;
			if (_parse_color(tag.value, color)) {
				next_style.has_fg_color = true;
				next_style.fg_color = color;
			}
		} else if (tag.name == "bgcolor") {
			Color color;
			if (_parse_color(tag.value, color)) {
				next_style.has_bg_color = true;
				next_style.bg_color = color;
			}
		} else if (tag.name == "outline_color") {
			Color color;
			if (_parse_color(tag.value, color)) {
				next_style.has_outline_color = true;
				next_style.outline_color = color;
			}
		} else if (tag.name == "outline_size") {
			next_style.has_outline_size = true;
			next_style.outline_size = MAX(0, tag.value.to_int());
		} else if (tag.name == "font_size" || tag.name == "size") {
			next_style.font_size = MAX(1, tag.value.to_int());
		} else if (tag.name == "font") {
			next_style.font = tag.value;
			if (next_style.font.is_empty() && tag.options.has("name")) {
				next_style.font = tag.options["name"];
			}
			if (tag.options.has("size")) {
				next_style.font_size = MAX(1, tag.options["size"].to_int());
			}
		} else if (tag.name == "lang") {
			next_style.language = tag.value;
		} else if (tag.name == "left") {
			next_style.alignment = 0;
			next_style.block_tag = raw_tag;
		} else if (tag.name == "center") {
			next_style.alignment = 1;
			next_style.block_tag = raw_tag;
		} else if (tag.name == "right") {
			next_style.alignment = 2;
			next_style.block_tag = raw_tag;
		} else if (tag.name == "fill") {
			next_style.alignment = 3;
			next_style.block_tag = raw_tag;
		} else if (tag.name == "quote") {
			if (!doc.text.is_empty() && doc.text[doc.text.length() - 1] != '\n') {
				append_text("\n");
			}
			next_style.block_tag = raw_tag;
		} else if (tag.name == "indent") {
			next_style.indent_level = MAX(1, next_style.indent_level + 1);
			next_style.block_tag = raw_tag;
		} else if (tag.name == "ul") {
			next_style.indent_level = MAX(1, next_style.indent_level + 1);
			next_style.list_type = 4;
			next_style.block_tag = raw_tag;
		} else if (tag.name == "ol") {
			next_style.indent_level = MAX(1, next_style.indent_level + 1);
			next_style.list_type = 1;
			next_style.list_start = -1;
			next_style.list_capitalize = false;
			if (tag.options.has("start")) {
				next_style.list_start = MAX(1, tag.options["start"].to_int());
			}
			if (tag.options.has("type")) {
				const String type = tag.options["type"];
				if (type == "a") {
					next_style.list_type = 2;
				} else if (type == "A") {
					next_style.list_type = 2;
					next_style.list_capitalize = true;
				} else if (type == "i") {
					next_style.list_type = 3;
				} else if (type == "I") {
					next_style.list_type = 3;
					next_style.list_capitalize = true;
				}
			}
			next_style.block_tag = raw_tag;
		} else if (_is_preserved_pair_tag(tag.name)) {
			next_style.preserved_tags.push_back(raw_tag);
		} else {
			pushed = false;
		}

		if (pushed) {
			style_stack.push_back(next_style);
			style_start_stack.push_back(doc.text.length());
			pos = brk_end + 1;
		} else {
			append_text("[" + raw_tag + "]");
			pos = brk_end + 1;
		}
	}

	return doc;
}

String RichTextDocument::to_bbcode() const {
	String bbcode;

	auto append_range = [&](int p_from, int p_to) {
		int range_cursor = p_from;
		while (range_cursor < p_to) {
			bool appended_inline = false;
			for (const InlineImage &image : images) {
				if (image.offset == range_cursor) {
					String tag = "img";
					for (const KeyValue<String, String> &E : image.options) {
						if (E.key == "size" || E.key.begins_with("_")) {
							continue;
						}
						tag += " " + E.key + "=" + quote_bbcode_option_value(E.value);
					}
					if (image.options.has("size")) {
						tag += "=" + quote_bbcode_option_value(image.options["size"]);
					}
					bbcode += "[" + tag + "]" + image.source + "[/img]";
					range_cursor++;
					appended_inline = true;
					break;
				}
			}
			if (appended_inline) {
				continue;
			}
			for (const RawInline &raw_inline : raw_inlines) {
				if (raw_inline.offset == range_cursor) {
					bbcode += raw_inline.bbcode;
					range_cursor++;
					appended_inline = true;
					break;
				}
			}
			if (appended_inline) {
				continue;
			}
			int next = p_to;
			for (const InlineImage &image : images) {
				if (image.offset > range_cursor) {
					next = MIN(next, image.offset);
				}
			}
			for (const RawInline &raw_inline : raw_inlines) {
				if (raw_inline.offset > range_cursor) {
					next = MIN(next, raw_inline.offset);
				}
			}
			bbcode += _escape_text(text.substr(range_cursor, next - range_cursor));
			range_cursor = next;
		}
	};

	auto add_tag = [](Vector<BBCodeTag> &r_tags, const String &p_open, const String &p_close) {
		BBCodeTag tag;
		tag.open = p_open;
		tag.close = p_close;
		r_tags.push_back(tag);
	};

	auto style_to_tags = [&](const Style &p_style) {
		Vector<BBCodeTag> tags;

		for (const String &tag : p_style.preserved_tags) {
			add_tag(tags, "[" + tag + "]", _close_tag_for(tag));
		}

		const bool block_tag_has_indent = p_style.block_tag == "indent" || p_style.block_tag == "ul" || p_style.block_tag == "ol";
		const int extra_indent_tags = MAX(0, p_style.indent_level - (block_tag_has_indent ? 1 : 0));
		for (int i = 0; i < extra_indent_tags; i++) {
			add_tag(tags, "[indent]", "[/indent]");
		}
		if (!p_style.block_tag.is_empty()) {
			String block_tag = p_style.block_tag;
			if (p_style.list_start > 0 && block_tag == "ol") {
				block_tag += " start=" + itos(p_style.list_start);
			}
			add_tag(tags, "[" + block_tag + "]", _close_tag_for(block_tag));
		}
		if (!p_style.language.is_empty()) {
			add_tag(tags, "[lang=" + p_style.language + "]", "[/lang]");
		}

		if (p_style.has_bg_color) {
			add_tag(tags, "[bgcolor=" + p_style.bg_color.to_html(p_style.bg_color.a < 1.0) + "]", "[/bgcolor]");
		}
		if (p_style.has_color) {
			add_tag(tags, "[color=" + p_style.color.to_html(p_style.color.a < 1.0) + "]", "[/color]");
		}
		if (p_style.has_fg_color) {
			add_tag(tags, "[fgcolor=" + p_style.fg_color.to_html(p_style.fg_color.a < 1.0) + "]", "[/fgcolor]");
		}
		if (!p_style.font.is_empty()) {
			add_tag(tags, "[font=" + p_style.font + "]", "[/font]");
		}
		if (p_style.font_size > 0) {
			add_tag(tags, "[font_size=" + itos(p_style.font_size) + "]", "[/font_size]");
		}
		if (p_style.has_outline_color) {
			add_tag(tags, "[outline_color=" + p_style.outline_color.to_html(p_style.outline_color.a < 1.0) + "]", "[/outline_color]");
		}
		if (p_style.has_outline_size) {
			add_tag(tags, "[outline_size=" + itos(p_style.outline_size) + "]", "[/outline_size]");
		}
		if (!p_style.url.is_empty()) {
			String url_tag = "url=" + quote_bbcode_option_value(p_style.url);
			if (p_style.has_underline && !p_style.underline) {
				url_tag += " underline=never";
			}
			if (!p_style.url_tooltip.is_empty()) {
				url_tag += " tooltip=" + quote_bbcode_option_value(p_style.url_tooltip);
			}
			if (p_style.url_visited) {
				url_tag += " visited=true";
			}
			add_tag(tags, "[" + url_tag + "]", "[/url]");
		}

		if (p_style.code) {
			add_tag(tags, "[code]", "[/code]");
		}
		if (p_style.bold) {
			add_tag(tags, "[b]", "[/b]");
		}
		if (p_style.italic) {
			add_tag(tags, "[i]", "[/i]");
		}
		if (p_style.strikethrough) {
			add_tag(tags, "[s]", "[/s]");
		}
		if (p_style.has_underline && p_style.underline) {
			add_tag(tags, "[u]", "[/u]");
		}

		return tags;
	};

	Vector<StyleSpan> sorted_spans = spans;
	sorted_spans.sort_custom<BBCodeStyleSpanCompare>();

	Vector<int> boundaries;
	boundaries.push_back(0);
	boundaries.push_back(text.length());
	for (const StyleSpan &span : sorted_spans) {
		const int from = CLAMP(span.from, 0, text.length());
		const int to = CLAMP(span.to, 0, text.length());
		if (from < to) {
			boundaries.push_back(from);
			boundaries.push_back(to);
		}
	}
	for (const InlineImage &image : images) {
		if (image.offset >= 0 && image.offset < text.length()) {
			boundaries.push_back(image.offset);
			boundaries.push_back(image.offset + 1);
		}
	}
	for (const RawInline &raw_inline : raw_inlines) {
		if (raw_inline.offset >= 0 && raw_inline.offset < text.length()) {
			boundaries.push_back(raw_inline.offset);
			boundaries.push_back(raw_inline.offset + 1);
		}
	}
	boundaries.sort();

	Vector<int> unique_boundaries;
	for (int boundary : boundaries) {
		if (unique_boundaries.is_empty() || unique_boundaries[unique_boundaries.size() - 1] != boundary) {
			unique_boundaries.push_back(boundary);
		}
	}

	Vector<BBCodeTag> open_tags;
	int span_index = 0;
	for (int i = 0; i < unique_boundaries.size() - 1; i++) {
		const int from = unique_boundaries[i];
		const int to = unique_boundaries[i + 1];
		if (from >= to) {
			continue;
		}

		while (span_index < sorted_spans.size() && sorted_spans[span_index].to <= from) {
			span_index++;
		}

		Style style;
		if (span_index < sorted_spans.size() && sorted_spans[span_index].from <= from && sorted_spans[span_index].to > from) {
			style = sorted_spans[span_index].style;
		}
		const Vector<BBCodeTag> desired_tags = style_to_tags(style);
		int common_prefix = 0;
		while (common_prefix < open_tags.size() && common_prefix < desired_tags.size() && open_tags[common_prefix] == desired_tags[common_prefix]) {
			common_prefix++;
		}
		for (int j = open_tags.size() - 1; j >= common_prefix; j--) {
			bbcode += open_tags[j].close;
		}
		open_tags.resize(common_prefix);
		for (int j = common_prefix; j < desired_tags.size(); j++) {
			bbcode += desired_tags[j].open;
			open_tags.push_back(desired_tags[j]);
		}

		append_range(from, to);
	}

	for (int i = open_tags.size() - 1; i >= 0; i--) {
		bbcode += open_tags[i].close;
	}
	return bbcode;
}
