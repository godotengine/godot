/**************************************************************************/
/*  rich_text_edit.cpp                                                    */
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

#include "rich_text_edit.h"

#include "core/io/resource_loader.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup.h"
#include "scene/gui/spin_box.h"
#include "scene/resources/style_box.h"
#include "scene/resources/style_box_flat.h"
#include "scene/theme/theme_db.h"
#include "servers/rendering/rendering_server.h"

struct StyleSpanCompare {
	_FORCE_INLINE_ bool operator()(const RichTextEdit::StyleSpan &p_a, const RichTextEdit::StyleSpan &p_b) const {
		if (p_a.from == p_b.from) {
			return p_a.to < p_b.to;
		}
		return p_a.from < p_b.from;
	}
};

bool RichTextEdit::_is_url_auto_open_allowed(const String &p_url) {
	const String url = p_url.strip_edges().to_lower();
	return url.begins_with("http://") || url.begins_with("https://");
}

String RichTextEdit::_serialize_bbcode() const {
	RichTextDocument document;
	document.text = TextEdit::get_text();
	for (const StyleSpan &span : style_spans) {
		RichTextDocument::StyleSpan doc_span;
		doc_span.from = span.from;
		doc_span.to = span.to;
		doc_span.style = span.style;
		document.spans.push_back(doc_span);
	}
	for (const RichTextDocument::InlineImage &image : images) {
		if (image.offset >= 0 && image.offset < document.text.length() && document.text[image.offset] == RichTextDocument::OBJECT_REPLACEMENT_CHAR) {
			document.images.push_back(image);
		}
	}
	for (const RichTextDocument::RawInline &raw_inline : raw_inlines) {
		if (raw_inline.offset >= 0 && raw_inline.offset < document.text.length() && document.text[raw_inline.offset] == RichTextDocument::OBJECT_REPLACEMENT_CHAR) {
			document.raw_inlines.push_back(raw_inline);
		}
	}
	return document.to_bbcode();
}

void RichTextEdit::_mark_bbcode_dirty() {
	bbcode_dirty = true;
}

void RichTextEdit::_apply_document(const RichTextDocument &p_document) {
	style_spans.clear();
	deleted_style_ranges_for_undo.clear();
	deleted_images_for_undo.clear();
	deleted_raw_inlines_for_undo.clear();
	for (const RichTextDocument::StyleSpan &doc_span : p_document.spans) {
		StyleSpan span;
		span.from = doc_span.from;
		span.to = doc_span.to;
		span.style = doc_span.style;
		style_spans.push_back(span);
	}
	images = p_document.images;
	for (RichTextDocument::InlineImage &image : images) {
		if (image.texture.is_null()) {
			image.texture = RichTextDocument::load_image_texture(image.source);
		}
	}
	raw_inlines = p_document.raw_inlines;
}

void RichTextEdit::_sync_style_spans_for_text_change(const String &p_old_text, const String &p_new_text) {
	int prefix = 0;
	const int old_len = p_old_text.length();
	const int new_len = p_new_text.length();
	while (prefix < old_len && prefix < new_len && p_old_text[prefix] == p_new_text[prefix]) {
		prefix++;
	}

	int suffix = 0;
	while (suffix < old_len - prefix && suffix < new_len - prefix && p_old_text[old_len - suffix - 1] == p_new_text[new_len - suffix - 1]) {
		suffix++;
	}

	const int old_from = prefix;
	const int old_to = old_len - suffix;
	const int new_to = new_len - suffix;
	const int delta = (new_to - old_from) - (old_to - old_from);
	TextStyle insertion_style;
	if (new_to > old_from) {
		if (typing_style_override) {
			insertion_style = typing_style;
		} else if ((old_from == 0 || p_old_text[old_from - 1] == '\n') && old_from < old_len) {
			insertion_style = _get_style_at_offset(old_from);
		} else if (old_from > 0) {
			insertion_style = _get_style_at_offset(old_from - 1);
		} else {
			insertion_style = _get_style_at_offset(old_from);
		}
	}
	const bool inserted_unstyled_paragraph_break = old_from == old_to && new_to == old_from + 1 && old_from < p_new_text.length() && p_new_text[old_from] == '\n' && insertion_style.block_tag.is_empty();

	if (old_to > old_from) {
		DeletedStyleRange deleted_range;
		deleted_range.offset = old_from;
		deleted_range.text = p_old_text.substr(old_from, old_to - old_from);
		for (const StyleSpan &span : style_spans) {
			if (span.to <= old_from || span.from >= old_to) {
				continue;
			}

			StyleSpan deleted_span;
			deleted_span.from = MAX(span.from, old_from) - old_from;
			deleted_span.to = MIN(span.to, old_to) - old_from;
			deleted_span.style = span.style;
			if (deleted_span.from < deleted_span.to) {
				deleted_range.spans.push_back(deleted_span);
			}
		}
		deleted_style_ranges_for_undo.push_back(deleted_range);
	}

	Vector<StyleSpan> updated;
	for (StyleSpan span : style_spans) {
		if (span.to <= old_from) {
			updated.push_back(span);
		} else if (span.from >= old_to) {
			span.from += delta;
			span.to += delta;
			updated.push_back(span);
		} else {
			if (span.from < old_from) {
				StyleSpan left = span;
				left.to = old_from;
				updated.push_back(left);
			}
			if (span.to > old_to) {
				StyleSpan right = span;
				right.from = new_to;
				if (inserted_unstyled_paragraph_break && !span.style.block_tag.is_empty() && old_from < p_old_text.length() && p_old_text[old_from] == '\n') {
					right.from++;
				}
				right.to = span.to + delta;
				if (right.from < right.to) {
					updated.push_back(right);
				}
			}
		}
	}
	style_spans = updated;
	_sync_inline_objects_for_text_change(p_old_text, p_new_text, old_from, old_to, new_to);

	if (new_to > old_from) {
		const String inserted_text = p_new_text.substr(old_from, new_to - old_from);
		bool restored_deleted_styles = false;
		for (int i = deleted_style_ranges_for_undo.size() - 1; i >= 0; i--) {
			const DeletedStyleRange &deleted_range = deleted_style_ranges_for_undo[i];
			if (deleted_range.offset != old_from || deleted_range.text != inserted_text) {
				continue;
			}

			for (StyleSpan span : deleted_range.spans) {
				span.from += old_from;
				span.to += old_from;
				if (span.from < span.to) {
					style_spans.push_back(span);
				}
			}
			deleted_style_ranges_for_undo.remove_at(i);
			restored_deleted_styles = true;
			break;
		}

		if (restored_deleted_styles) {
			_merge_adjacent_spans();
		} else if (!insertion_style.is_default()) {
			_replace_style_range(old_from, new_to, insertion_style);
		} else {
			_merge_adjacent_spans();
		}
	} else {
		_merge_adjacent_spans();
	}
	typing_style_override = false;
}

void RichTextEdit::_sync_inline_objects_for_text_change(const String &p_old_text, const String &p_new_text, int p_old_from, int p_old_to, int p_new_to) {
	const int delta = (p_new_to - p_old_from) - (p_old_to - p_old_from);
	Vector<int> old_inline_offsets;
	for (int i = p_old_from; i < p_old_to && i < p_old_text.length(); i++) {
		if (p_old_text[i] == RichTextDocument::OBJECT_REPLACEMENT_CHAR) {
			old_inline_offsets.push_back(i);
		}
	}
	Vector<int> new_inline_offsets;
	for (int i = p_old_from; i < p_new_to && i < p_new_text.length(); i++) {
		if (p_new_text[i] == RichTextDocument::OBJECT_REPLACEMENT_CHAR) {
			new_inline_offsets.push_back(i);
		}
	}

	Vector<RichTextDocument::InlineImage> updated_images;
	for (const RichTextDocument::InlineImage &image : images) {
		if (image.offset < p_old_from) {
			updated_images.push_back(image);
		} else if (image.offset >= p_old_to) {
			RichTextDocument::InlineImage updated_image = image;
			updated_image.offset += delta;
			updated_images.push_back(updated_image);
		} else {
			const int inline_index = old_inline_offsets.find(image.offset);
			if (inline_index >= 0 && inline_index < new_inline_offsets.size()) {
				RichTextDocument::InlineImage updated_image = image;
				updated_image.offset = new_inline_offsets[inline_index];
				updated_images.push_back(updated_image);
			} else {
				deleted_images_for_undo.push_back(image);
			}
		}
	}
	for (int offset : new_inline_offsets) {
		bool has_image = false;
		for (const RichTextDocument::InlineImage &image : updated_images) {
			if (image.offset == offset) {
				has_image = true;
				break;
			}
		}
		if (has_image) {
			continue;
		}
		for (int i = deleted_images_for_undo.size() - 1; i >= 0; i--) {
			if (deleted_images_for_undo[i].offset == offset) {
				updated_images.push_back(deleted_images_for_undo[i]);
				deleted_images_for_undo.remove_at(i);
				break;
			}
		}
	}
	images = updated_images;

	Vector<RichTextDocument::RawInline> updated_raw_inlines;
	for (const RichTextDocument::RawInline &raw_inline : raw_inlines) {
		if (raw_inline.offset < p_old_from) {
			updated_raw_inlines.push_back(raw_inline);
		} else if (raw_inline.offset >= p_old_to) {
			RichTextDocument::RawInline updated_raw_inline = raw_inline;
			updated_raw_inline.offset += delta;
			updated_raw_inlines.push_back(updated_raw_inline);
		} else {
			const int inline_index = old_inline_offsets.find(raw_inline.offset);
			if (inline_index >= 0 && inline_index < new_inline_offsets.size()) {
				RichTextDocument::RawInline updated_raw_inline = raw_inline;
				updated_raw_inline.offset = new_inline_offsets[inline_index];
				updated_raw_inlines.push_back(updated_raw_inline);
			} else {
				deleted_raw_inlines_for_undo.push_back(raw_inline);
			}
		}
	}
	for (int offset : new_inline_offsets) {
		bool has_raw_inline = false;
		for (const RichTextDocument::RawInline &raw_inline : updated_raw_inlines) {
			if (raw_inline.offset == offset) {
				has_raw_inline = true;
				break;
			}
		}
		if (has_raw_inline) {
			continue;
		}
		for (int i = deleted_raw_inlines_for_undo.size() - 1; i >= 0; i--) {
			if (deleted_raw_inlines_for_undo[i].offset == offset) {
				updated_raw_inlines.push_back(deleted_raw_inlines_for_undo[i]);
				deleted_raw_inlines_for_undo.remove_at(i);
				break;
			}
		}
	}
	raw_inlines = updated_raw_inlines;
}

void RichTextEdit::_update_bbcode_from_text() {
	if (setting_bbcode_text) {
		return;
	}

	const bool had_style_data = !style_spans.is_empty() || !images.is_empty() || !raw_inlines.is_empty() || typing_style_override;
	syncing_text_change = true;
	String new_text = TextEdit::get_text();
	const auto count_object_replacement_chars = [](const String &p_text) {
		int count = 0;
		for (int i = 0; i < p_text.length(); i++) {
			if (p_text[i] == RichTextDocument::OBJECT_REPLACEMENT_CHAR) {
				count++;
			}
		}
		return count;
	};
	const auto inline_metadata_matches_text = [&](const String &p_text) {
		HashSet<int> offsets;
		for (const RichTextDocument::InlineImage &image : images) {
			if (image.offset < 0 || image.offset >= p_text.length() || p_text[image.offset] != RichTextDocument::OBJECT_REPLACEMENT_CHAR || offsets.has(image.offset)) {
				return false;
			}
			offsets.insert(image.offset);
		}
		for (const RichTextDocument::RawInline &raw_inline : raw_inlines) {
			if (raw_inline.offset < 0 || raw_inline.offset >= p_text.length() || p_text[raw_inline.offset] != RichTextDocument::OBJECT_REPLACEMENT_CHAR || offsets.has(raw_inline.offset)) {
				return false;
			}
			offsets.insert(raw_inline.offset);
		}
		return (int)offsets.size() == count_object_replacement_chars(p_text);
	};
	const bool skip_metadata_resync = pending_inline_metadata_restore_sync &&
			count_object_replacement_chars(tracked_text) != count_object_replacement_chars(new_text) &&
			inline_metadata_matches_text(new_text);
	pending_inline_metadata_restore_sync = false;
	if (!skip_metadata_resync) {
		_sync_style_spans_for_text_change(tracked_text, new_text);
	}
	tracked_text = new_text;
	source_text = new_text;
	_mark_bbcode_dirty();
	syncing_text_change = false;
	if (selected_image_offset >= 0 && (selected_image_offset >= new_text.length() || new_text[selected_image_offset] != RichTextDocument::OBJECT_REPLACEMENT_CHAR)) {
		_clear_selected_image();
	}

	_refresh_style_rendering();
	if (had_style_data || !style_spans.is_empty() || !images.is_empty() || !raw_inlines.is_empty() || typing_style_override) {
		_style_changed();
	}
}

void RichTextEdit::_replace_style_range(int p_from, int p_to, const TextStyle &p_style) {
	if (p_from > p_to) {
		SWAP(p_from, p_to);
	}
	if (p_from == p_to) {
		typing_style = p_style;
		return;
	}

	const int text_len = TextEdit::get_text().length();
	p_from = CLAMP(p_from, 0, text_len);
	p_to = CLAMP(p_to, 0, text_len);

	Vector<StyleSpan> updated;
	for (const StyleSpan &span : style_spans) {
		if (span.to <= p_from || span.from >= p_to) {
			updated.push_back(span);
			continue;
		}
		if (span.from < p_from) {
			StyleSpan left = span;
			left.to = p_from;
			updated.push_back(left);
		}
		if (span.to > p_to) {
			StyleSpan right = span;
			right.from = p_to;
			right.to = span.to;
			updated.push_back(right);
		}
	}

	if (!p_style.is_default()) {
		StyleSpan replacement;
		replacement.from = p_from;
		replacement.to = p_to;
		replacement.style = p_style;
		updated.push_back(replacement);
	}

	style_spans = updated;
	_merge_adjacent_spans();
}

void RichTextEdit::_apply_style_property(TextStyle &r_style, const TextStyle &p_toggle_reference, StyleProperty p_property, const Variant &p_value) const {
	switch (p_property) {
		case STYLE_PROPERTY_BOLD:
			r_style.bold = p_value.get_type() == Variant::BOOL ? bool(p_value) : !p_toggle_reference.bold;
			break;
		case STYLE_PROPERTY_ITALIC:
			r_style.italic = p_value.get_type() == Variant::BOOL ? bool(p_value) : !p_toggle_reference.italic;
			break;
		case STYLE_PROPERTY_UNDERLINE:
			r_style.has_underline = true;
			r_style.underline = p_value.get_type() == Variant::BOOL ? bool(p_value) : !(p_toggle_reference.has_underline ? p_toggle_reference.underline : !p_toggle_reference.url.is_empty());
			break;
		case STYLE_PROPERTY_STRIKETHROUGH:
			r_style.strikethrough = p_value.get_type() == Variant::BOOL ? bool(p_value) : !p_toggle_reference.strikethrough;
			break;
		case STYLE_PROPERTY_COLOR:
			r_style.has_color = true;
			r_style.color = p_value;
			break;
		case STYLE_PROPERTY_CLEAR_COLOR:
			r_style.has_color = false;
			break;
		case STYLE_PROPERTY_BG_COLOR:
			r_style.has_bg_color = true;
			r_style.bg_color = p_value;
			break;
		case STYLE_PROPERTY_CLEAR_BG_COLOR:
			r_style.has_bg_color = false;
			break;
		case STYLE_PROPERTY_OUTLINE_COLOR:
			r_style.has_outline_color = true;
			r_style.outline_color = p_value;
			break;
		case STYLE_PROPERTY_CLEAR_OUTLINE_COLOR:
			r_style.has_outline_color = false;
			break;
		case STYLE_PROPERTY_OUTLINE_SIZE:
			r_style.has_outline_size = true;
			r_style.outline_size = MAX(0, int(p_value));
			break;
		case STYLE_PROPERTY_CLEAR_OUTLINE_SIZE:
			r_style.has_outline_size = false;
			r_style.outline_size = 0;
			break;
		case STYLE_PROPERTY_FONT_SIZE:
			r_style.font_size = MAX(1, int(p_value));
			break;
		case STYLE_PROPERTY_CLEAR_FONT_SIZE:
			r_style.font_size = 0;
			break;
		case STYLE_PROPERTY_FONT:
			r_style.font = p_value;
			break;
		case STYLE_PROPERTY_CLEAR_FONT:
			r_style.font = "";
			break;
		case STYLE_PROPERTY_URL:
			r_style.has_url = !String(p_value).is_empty();
			r_style.url = p_value;
			if (r_style.url.is_empty()) {
				r_style.url_tooltip = "";
				r_style.url_visited = false;
			}
			break;
		case STYLE_PROPERTY_CLEAR_URL:
			r_style.has_url = false;
			r_style.url = "";
			r_style.url_tooltip = "";
			r_style.url_visited = false;
			break;
		case STYLE_PROPERTY_URL_TOOLTIP:
			r_style.url_tooltip = p_value;
			break;
		case STYLE_PROPERTY_CLEAR_URL_TOOLTIP:
			r_style.url_tooltip = "";
			break;
		case STYLE_PROPERTY_URL_VISITED:
			r_style.url_visited = true;
			break;
		case STYLE_PROPERTY_CLEAR_URL_VISITED:
			r_style.url_visited = false;
			break;
		case STYLE_PROPERTY_CODE:
			r_style.code = !p_toggle_reference.code;
			break;
		case STYLE_PROPERTY_BLOCK_TAG:
			if (p_value == "indent") {
				r_style.indent_level = MAX(1, r_style.indent_level + 1);
				if (r_style.block_tag.is_empty() || r_style.block_tag == "indent") {
					r_style.block_tag = p_value;
				}
				break;
			}

			r_style.block_tag = p_value;
			r_style.alignment = -1;
			r_style.list_type = 0;
			r_style.list_start = -1;
			if (r_style.block_tag == "left") {
				r_style.alignment = HORIZONTAL_ALIGNMENT_LEFT;
			} else if (r_style.block_tag == "center") {
				r_style.alignment = HORIZONTAL_ALIGNMENT_CENTER;
			} else if (r_style.block_tag == "right") {
				r_style.alignment = HORIZONTAL_ALIGNMENT_RIGHT;
			} else if (r_style.block_tag == "fill") {
				r_style.alignment = HORIZONTAL_ALIGNMENT_FILL;
			} else if (r_style.block_tag == "ul") {
				r_style.indent_level = MAX(1, r_style.indent_level);
				r_style.list_type = 4;
			} else if (r_style.block_tag == "ol") {
				r_style.indent_level = MAX(1, r_style.indent_level);
				r_style.list_type = 1;
			}
			break;
	}
}

void RichTextEdit::_apply_style_property_to_selection(StyleProperty p_property, const Variant &p_value, bool p_record_undo) {
	int from = 0;
	int to = 0;
	const bool has_range = _get_selection_offsets(from, to);
	if (!has_range) {
		TextStyle style = typing_style;
		_apply_style_property(style, style, p_property, p_value);
		if ((style == typing_style && typing_style_override) || (!typing_style_override && style.is_default())) {
			return;
		}

		typing_style = style;
		typing_style_override = true;
		if (p_record_undo) {
			source_text = TextEdit::get_text();
			_mark_bbcode_dirty();
		}
		_refresh_style_rendering();
		if (p_record_undo) {
			_style_changed();
		}
		return;
	}

	const TextStyle first_selected_style = _get_selection_common_style();
	Vector<int> boundaries;
	boundaries.push_back(from);
	boundaries.push_back(to);
	for (const StyleSpan &span : style_spans) {
		if (span.from > from && span.from < to) {
			boundaries.push_back(span.from);
		}
		if (span.to > from && span.to < to) {
			boundaries.push_back(span.to);
		}
	}
	boundaries.sort();

	Vector<StyleSpan> before_spans = style_spans;
	Vector<RichTextDocument::InlineImage> before_images = images;
	Vector<RichTextDocument::RawInline> before_raw_inlines = raw_inlines;
	TextStyle before_typing_style = typing_style;
	const bool before_typing_style_override = typing_style_override;
	for (int i = 0; i < boundaries.size() - 1; i++) {
		const int segment_from = boundaries[i];
		const int segment_to = boundaries[i + 1];
		if (segment_from == segment_to) {
			continue;
		}

		TextStyle style = _get_style_at_offset(segment_from);
		_apply_style_property(style, first_selected_style, p_property, p_value);
		_replace_style_range(segment_from, segment_to, style);
	}

	if (before_spans == style_spans && before_typing_style == typing_style && before_typing_style_override == typing_style_override) {
		return;
	}
	source_text = TextEdit::get_text();
	_mark_bbcode_dirty();
	if (p_record_undo) {
		_push_style_undo_snapshot(before_spans, before_images, before_raw_inlines, before_typing_style, before_typing_style_override);
	}
	_refresh_style_rendering();
	if (p_record_undo) {
		_style_changed();
	}
}

void RichTextEdit::_apply_style_property_to_url_ranges(StyleProperty p_property, const Variant &p_value) {
	int from = 0;
	int to = 0;
	const bool has_range = _get_selection_offsets(from, to);
	if (!has_range) {
		const int caret_offset = _get_caret_offset();
		from = caret_offset;
		to = caret_offset;
	}

	Vector<int> touched_indices;
	for (int i = 0; i < style_spans.size(); i++) {
		const StyleSpan &span = style_spans[i];
		if (span.style.url.is_empty()) {
			continue;
		}
		const bool touches_range = has_range ? (span.from < to && span.to > from) : (span.from <= from && span.to >= from);
		const bool touches_previous_character = !has_range && from > 0 && span.from <= from - 1 && span.to > from - 1;
		if (touches_range || touches_previous_character) {
			touched_indices.push_back(i);
		}
	}

	if (touched_indices.is_empty()) {
		return;
	}

	const Vector<StyleSpan> before_spans = style_spans;
	const Vector<RichTextDocument::InlineImage> before_images = images;
	const Vector<RichTextDocument::RawInline> before_raw_inlines = raw_inlines;
	const TextStyle before_typing_style = typing_style;
	const bool before_typing_style_override = typing_style_override;

	for (int index : touched_indices) {
		TextStyle style = style_spans[index].style;
		_apply_style_property(style, style, p_property, p_value);
		style_spans.write[index].style = style;
	}

	_merge_adjacent_spans();
	if (before_spans == style_spans) {
		return;
	}
	_push_style_undo_snapshot(before_spans, before_images, before_raw_inlines, before_typing_style, before_typing_style_override);
	source_text = TextEdit::get_text();
	_mark_bbcode_dirty();
	_refresh_style_rendering();
	_style_changed();
}

Vector<int> RichTextEdit::_get_style_boundaries_for_range(int p_from, int p_to) const {
	Vector<int> boundaries;
	boundaries.push_back(p_from);
	boundaries.push_back(p_to);
	for (const StyleSpan &span : style_spans) {
		if (span.from > p_from && span.from < p_to) {
			boundaries.push_back(span.from);
		}
		if (span.to > p_from && span.to < p_to) {
			boundaries.push_back(span.to);
		}
	}
	boundaries.sort();
	return boundaries;
}

void RichTextEdit::_apply_block_tag_to_selected_lines(const String &p_tag) {
	const int from_line = has_selection() ? get_selection_from_line() : get_caret_line();
	const int to_line = has_selection() ? get_selection_to_line() : get_caret_line();
	const Vector<StyleSpan> before_spans = style_spans;
	const Vector<RichTextDocument::InlineImage> before_images = images;
	const Vector<RichTextDocument::RawInline> before_raw_inlines = raw_inlines;
	const TextStyle before_typing_style = typing_style;
	const bool before_typing_style_override = typing_style_override;

	for (int line = from_line; line <= to_line; line++) {
		const int line_start = _get_line_start_offset(line);
		const int line_end = line_start + get_line(line).length();
		if (line_start == line_end) {
			continue;
		}
		const Vector<int> boundaries = _get_style_boundaries_for_range(line_start, line_end);
		for (int i = 0; i < boundaries.size() - 1; i++) {
			const int segment_from = boundaries[i];
			const int segment_to = boundaries[i + 1];
			if (segment_from == segment_to) {
				continue;
			}
			TextStyle style = _get_style_at_offset(segment_from);
			_apply_style_property(style, style, STYLE_PROPERTY_BLOCK_TAG, p_tag);
			_replace_style_range(segment_from, segment_to, style);
		}
	}

	if (before_spans == style_spans) {
		return;
	}
	_push_style_undo_snapshot(before_spans, before_images, before_raw_inlines, before_typing_style, before_typing_style_override);
	source_text = TextEdit::get_text();
	_mark_bbcode_dirty();
	_refresh_style_rendering();
	_style_changed();
}

void RichTextEdit::_clear_block_tag_from_selected_lines(const String &p_tag) {
	const int from_line = has_selection() ? get_selection_from_line() : get_caret_line();
	const int to_line = has_selection() ? get_selection_to_line() : get_caret_line();
	const Vector<StyleSpan> before_spans = style_spans;
	const Vector<RichTextDocument::InlineImage> before_images = images;
	const Vector<RichTextDocument::RawInline> before_raw_inlines = raw_inlines;
	const TextStyle before_typing_style = typing_style;
	const bool before_typing_style_override = typing_style_override;

	for (int line = from_line; line <= to_line; line++) {
		const int line_start = _get_line_start_offset(line);
		const int line_end = line_start + get_line(line).length();
		if (line_start == line_end) {
			continue;
		}

		const Vector<int> boundaries = _get_style_boundaries_for_range(line_start, line_end);
		for (int i = 0; i < boundaries.size() - 1; i++) {
			const int segment_from = boundaries[i];
			const int segment_to = boundaries[i + 1];
			if (segment_from == segment_to) {
				continue;
			}
			TextStyle style = _get_style_at_offset(segment_from);
			if (style.block_tag != p_tag) {
				continue;
			}
			style.block_tag = "";
			if (p_tag == "ul" || p_tag == "ol") {
				style.indent_level = MAX(0, style.indent_level - 1);
				style.list_type = 0;
				style.list_start = -1;
				style.list_capitalize = false;
				if (style.indent_level > 0) {
					style.block_tag = "indent";
				}
			}
			_replace_style_range(segment_from, segment_to, style);
		}
	}

	if (before_spans == style_spans) {
		return;
	}
	_push_style_undo_snapshot(before_spans, before_images, before_raw_inlines, before_typing_style, before_typing_style_override);
	source_text = TextEdit::get_text();
	_mark_bbcode_dirty();
	_refresh_style_rendering();
	_style_changed();
}

void RichTextEdit::_toggle_block_tag_on_selected_lines(const String &p_tag) {
	const int from_line = has_selection() ? get_selection_from_line() : get_caret_line();
	const int to_line = has_selection() ? get_selection_to_line() : get_caret_line();
	bool all_lines_have_tag = true;
	bool has_text_line = false;

	for (int line = from_line; line <= to_line; line++) {
		const int line_start = _get_line_start_offset(line);
		const int line_end = line_start + get_line(line).length();
		if (line_start == line_end) {
			continue;
		}
		has_text_line = true;
		const TextStyle style = _get_style_at_offset(line_start);
		if (style.block_tag != p_tag) {
			all_lines_have_tag = false;
			break;
		}
	}

	if (!has_text_line || !all_lines_have_tag) {
		_apply_block_tag_to_selected_lines(p_tag);
		return;
	}

	_clear_block_tag_from_selected_lines(p_tag);
}

RichTextEdit::TextStyle RichTextEdit::_get_insertion_style_at_offset(int p_offset) const {
	const int caret_line = get_caret_line();
	const int line_start = _get_line_start_offset(caret_line);
	const int line_length = get_line(caret_line).length();
	if (p_offset == line_start && line_length > 0) {
		return _get_style_at_offset(line_start);
	}
	if (p_offset > line_start) {
		return _get_style_at_offset(p_offset - 1);
	}
	return typing_style;
}

RichTextEdit::TextStyle RichTextEdit::_get_style_at_offset(int p_offset) const {
	for (const StyleSpan &span : style_spans) {
		if (p_offset >= span.from && p_offset < span.to) {
			return span.style;
		}
	}
	return TextStyle();
}

RichTextEdit::TextStyle RichTextEdit::_get_selection_common_style() const {
	int from = 0;
	int to = 0;
	if (!_get_selection_offsets(from, to)) {
		return typing_style;
	}
	return _get_style_at_offset(from);
}

int RichTextEdit::_get_default_font_size() const {
	const int text_edit_font_size = get_theme_font_size(SceneStringName(font_size));
	if (text_edit_font_size > 0) {
		return text_edit_font_size;
	}

	const int rich_text_label_font_size = get_theme_font_size(SNAME("normal_font_size"), SNAME("RichTextLabel"));
	if (rich_text_label_font_size > 0) {
		return rich_text_label_font_size;
	}

	return 16;
}

void RichTextEdit::_merge_adjacent_spans() {
	style_spans.sort_custom<StyleSpanCompare>();
	Vector<StyleSpan> merged;
	for (const StyleSpan &span : style_spans) {
		if (span.from >= span.to || span.style.is_default()) {
			continue;
		}
		if (!merged.is_empty() && merged.write[merged.size() - 1].to == span.from && merged[merged.size() - 1].style == span.style) {
			merged.write[merged.size() - 1].to = span.to;
		} else {
			merged.push_back(span);
		}
	}
	style_spans = merged;
}

int RichTextEdit::_get_line_start_offset(int p_line) const {
	int offset = 0;
	for (int i = 0; i < p_line; i++) {
		offset += get_line(i).length() + 1;
	}
	return offset;
}

int RichTextEdit::_get_caret_offset() const {
	return _get_line_start_offset(get_caret_line()) + get_caret_column();
}

bool RichTextEdit::_get_selection_offsets(int &r_from, int &r_to) const {
	if (!has_selection()) {
		return false;
	}
	r_from = _get_line_start_offset(get_selection_from_line()) + get_selection_from_column();
	r_to = _get_line_start_offset(get_selection_to_line()) + get_selection_to_column();
	if (r_from > r_to) {
		SWAP(r_from, r_to);
	}
	return r_from != r_to;
}

void RichTextEdit::_apply_style_to_selection(const TextStyle &p_style) {
	Vector<StyleSpan> before_spans = style_spans;
	Vector<RichTextDocument::InlineImage> before_images = images;
	Vector<RichTextDocument::RawInline> before_raw_inlines = raw_inlines;
	TextStyle before_typing_style = typing_style;
	const bool before_typing_style_override = typing_style_override;
	int from = 0;
	int to = 0;
	const bool has_range = _get_selection_offsets(from, to);
	if (has_range) {
		_replace_style_range(from, to, p_style);
	} else {
		typing_style = p_style;
		typing_style_override = true;
	}

	if (has_range) {
		_push_style_undo_snapshot(before_spans, before_images, before_raw_inlines, before_typing_style, before_typing_style_override);
	}
	source_text = TextEdit::get_text();
	_mark_bbcode_dirty();
	_refresh_style_rendering();
	_style_changed();
}

void RichTextEdit::_caret_style_context_changed() {
	if (!syncing_text_change) {
		typing_style_override = false;
	}
}

Variant RichTextEdit::_make_style_state_variant(const Vector<StyleSpan> &p_spans, const Vector<RichTextDocument::InlineImage> &p_images, const Vector<RichTextDocument::RawInline> &p_raw_inlines, const TextStyle &p_typing_style, bool p_typing_style_override) const {
	Dictionary state;
	Array spans;
	for (const StyleSpan &span : p_spans) {
		Dictionary style;
		style["bold"] = span.style.bold;
		style["italic"] = span.style.italic;
		style["has_underline"] = span.style.has_underline;
		style["underline"] = span.style.underline;
		style["strikethrough"] = span.style.strikethrough;
		style["code"] = span.style.code;
		style["has_color"] = span.style.has_color;
		style["color"] = span.style.color;
		style["has_bg_color"] = span.style.has_bg_color;
		style["bg_color"] = span.style.bg_color;
		style["has_fg_color"] = span.style.has_fg_color;
		style["fg_color"] = span.style.fg_color;
		style["has_outline_color"] = span.style.has_outline_color;
		style["outline_color"] = span.style.outline_color;
		style["has_outline_size"] = span.style.has_outline_size;
		style["outline_size"] = span.style.outline_size;
		style["has_url"] = span.style.has_url;
		style["url_visited"] = span.style.url_visited;
		style["font_size"] = span.style.font_size;
		style["alignment"] = span.style.alignment;
		style["indent_level"] = span.style.indent_level;
		style["list_type"] = span.style.list_type;
		style["list_start"] = span.style.list_start;
		style["list_capitalize"] = span.style.list_capitalize;
		style["font"] = span.style.font;
		style["url"] = span.style.url;
		style["url_tooltip"] = span.style.url_tooltip;
		style["language"] = span.style.language;
		style["block_tag"] = span.style.block_tag;
		Array preserved_tags;
		for (const String &tag : span.style.preserved_tags) {
			preserved_tags.push_back(tag);
		}
		style["preserved_tags"] = preserved_tags;

		Dictionary span_data;
		span_data["from"] = span.from;
		span_data["to"] = span.to;
		span_data["style"] = style;
		spans.push_back(span_data);
	}

	Array image_data_array;
	for (const RichTextDocument::InlineImage &image : p_images) {
		Dictionary image_data;
		image_data["offset"] = image.offset;
		image_data["source"] = image.source;
		Dictionary options;
		for (const KeyValue<String, String> &E : image.options) {
			options[E.key] = E.value;
		}
		image_data["options"] = options;
		image_data_array.push_back(image_data);
	}
	state["images"] = image_data_array;

	Array raw_inline_data_array;
	for (const RichTextDocument::RawInline &raw_inline : p_raw_inlines) {
		Dictionary raw_inline_data;
		raw_inline_data["offset"] = raw_inline.offset;
		raw_inline_data["bbcode"] = raw_inline.bbcode;
		raw_inline_data_array.push_back(raw_inline_data);
	}
	state["raw_inlines"] = raw_inline_data_array;

	Dictionary typing_style_data;
	typing_style_data["bold"] = p_typing_style.bold;
	typing_style_data["italic"] = p_typing_style.italic;
	typing_style_data["has_underline"] = p_typing_style.has_underline;
	typing_style_data["underline"] = p_typing_style.underline;
	typing_style_data["strikethrough"] = p_typing_style.strikethrough;
	typing_style_data["code"] = p_typing_style.code;
	typing_style_data["has_color"] = p_typing_style.has_color;
	typing_style_data["color"] = p_typing_style.color;
	typing_style_data["has_bg_color"] = p_typing_style.has_bg_color;
	typing_style_data["bg_color"] = p_typing_style.bg_color;
	typing_style_data["has_fg_color"] = p_typing_style.has_fg_color;
	typing_style_data["fg_color"] = p_typing_style.fg_color;
	typing_style_data["has_outline_color"] = p_typing_style.has_outline_color;
	typing_style_data["outline_color"] = p_typing_style.outline_color;
	typing_style_data["has_outline_size"] = p_typing_style.has_outline_size;
	typing_style_data["outline_size"] = p_typing_style.outline_size;
	typing_style_data["has_url"] = p_typing_style.has_url;
	typing_style_data["url_visited"] = p_typing_style.url_visited;
	typing_style_data["font_size"] = p_typing_style.font_size;
	typing_style_data["alignment"] = p_typing_style.alignment;
	typing_style_data["indent_level"] = p_typing_style.indent_level;
	typing_style_data["list_type"] = p_typing_style.list_type;
	typing_style_data["list_start"] = p_typing_style.list_start;
	typing_style_data["list_capitalize"] = p_typing_style.list_capitalize;
	typing_style_data["font"] = p_typing_style.font;
	typing_style_data["url"] = p_typing_style.url;
	typing_style_data["url_tooltip"] = p_typing_style.url_tooltip;
	typing_style_data["language"] = p_typing_style.language;
	typing_style_data["block_tag"] = p_typing_style.block_tag;

	state["spans"] = spans;
	state["typing_style"] = typing_style_data;
	state["typing_style_override"] = p_typing_style_override;
	return state;
}

void RichTextEdit::_restore_style_state_variant(const Variant &p_state) {
	Dictionary state = p_state;
	Array spans = state.get("spans", Array());

	Vector<StyleSpan> restored_spans;
	for (Dictionary span_data : spans) {
		Dictionary style_data = span_data.get("style", Dictionary());

		StyleSpan span;
		span.from = span_data.get("from", 0);
		span.to = span_data.get("to", 0);
		span.style.bold = style_data.get("bold", false);
		span.style.italic = style_data.get("italic", false);
		span.style.has_underline = style_data.get("has_underline", false);
		span.style.underline = style_data.get("underline", false);
		span.style.strikethrough = style_data.get("strikethrough", false);
		span.style.code = style_data.get("code", false);
		span.style.has_color = style_data.get("has_color", false);
		span.style.color = style_data.get("color", Color());
		span.style.has_bg_color = style_data.get("has_bg_color", false);
		span.style.bg_color = style_data.get("bg_color", Color());
		span.style.has_fg_color = style_data.get("has_fg_color", false);
		span.style.fg_color = style_data.get("fg_color", Color());
		span.style.has_outline_color = style_data.get("has_outline_color", false);
		span.style.outline_color = style_data.get("outline_color", Color());
		span.style.has_outline_size = style_data.get("has_outline_size", false);
		span.style.outline_size = style_data.get("outline_size", 0);
		span.style.has_url = style_data.get("has_url", false);
		span.style.url_visited = style_data.get("url_visited", false);
		span.style.font_size = style_data.get("font_size", 0);
		span.style.alignment = style_data.get("alignment", -1);
		span.style.indent_level = style_data.get("indent_level", 0);
		span.style.list_type = style_data.get("list_type", 0);
		span.style.list_start = style_data.get("list_start", -1);
		span.style.list_capitalize = style_data.get("list_capitalize", false);
		span.style.font = style_data.get("font", String());
		span.style.url = style_data.get("url", String());
		span.style.url_tooltip = style_data.get("url_tooltip", String());
		span.style.language = style_data.get("language", String());
		span.style.block_tag = style_data.get("block_tag", String());
		Array preserved_tags = style_data.get("preserved_tags", Array());
		for (const Variant &preserved_tag : preserved_tags) {
			span.style.preserved_tags.push_back(preserved_tag);
		}
		restored_spans.push_back(span);
	}

	Vector<RichTextDocument::InlineImage> restored_images;
	Array image_data_array = state.get("images", Array());
	for (Dictionary image_data : image_data_array) {
		RichTextDocument::InlineImage image;
		image.offset = image_data.get("offset", 0);
		image.source = image_data.get("source", String());
		Dictionary options = image_data.get("options", Dictionary());
		for (const KeyValue<Variant, Variant> &kv : options) {
			image.options[kv.key] = kv.value;
		}
		image.texture = RichTextDocument::load_image_texture(image.source);
		restored_images.push_back(image);
	}

	Vector<RichTextDocument::RawInline> restored_raw_inlines;
	Array raw_inline_data_array = state.get("raw_inlines", Array());
	for (Dictionary raw_inline_data : raw_inline_data_array) {
		RichTextDocument::RawInline raw_inline;
		raw_inline.offset = raw_inline_data.get("offset", 0);
		raw_inline.bbcode = raw_inline_data.get("bbcode", String());
		restored_raw_inlines.push_back(raw_inline);
	}

	Dictionary typing_style_data = state.get("typing_style", Dictionary());
	TextStyle restored_typing_style;
	restored_typing_style.bold = typing_style_data.get("bold", false);
	restored_typing_style.italic = typing_style_data.get("italic", false);
	restored_typing_style.has_underline = typing_style_data.get("has_underline", false);
	restored_typing_style.underline = typing_style_data.get("underline", false);
	restored_typing_style.strikethrough = typing_style_data.get("strikethrough", false);
	restored_typing_style.code = typing_style_data.get("code", false);
	restored_typing_style.has_color = typing_style_data.get("has_color", false);
	restored_typing_style.color = typing_style_data.get("color", Color());
	restored_typing_style.has_bg_color = typing_style_data.get("has_bg_color", false);
	restored_typing_style.bg_color = typing_style_data.get("bg_color", Color());
	restored_typing_style.has_fg_color = typing_style_data.get("has_fg_color", false);
	restored_typing_style.fg_color = typing_style_data.get("fg_color", Color());
	restored_typing_style.has_outline_color = typing_style_data.get("has_outline_color", false);
	restored_typing_style.outline_color = typing_style_data.get("outline_color", Color());
	restored_typing_style.has_outline_size = typing_style_data.get("has_outline_size", false);
	restored_typing_style.outline_size = typing_style_data.get("outline_size", 0);
	restored_typing_style.has_url = typing_style_data.get("has_url", false);
	restored_typing_style.url_visited = typing_style_data.get("url_visited", false);
	restored_typing_style.font_size = typing_style_data.get("font_size", 0);
	restored_typing_style.alignment = typing_style_data.get("alignment", -1);
	restored_typing_style.indent_level = typing_style_data.get("indent_level", 0);
	restored_typing_style.list_type = typing_style_data.get("list_type", 0);
	restored_typing_style.list_start = typing_style_data.get("list_start", -1);
	restored_typing_style.list_capitalize = typing_style_data.get("list_capitalize", false);
	restored_typing_style.font = typing_style_data.get("font", String());
	restored_typing_style.url = typing_style_data.get("url", String());
	restored_typing_style.url_tooltip = typing_style_data.get("url_tooltip", String());
	restored_typing_style.language = typing_style_data.get("language", String());
	restored_typing_style.block_tag = typing_style_data.get("block_tag", String());

	const bool restored_inline_metadata = images != restored_images || raw_inlines != restored_raw_inlines;
	style_spans = restored_spans;
	images = restored_images;
	raw_inlines = restored_raw_inlines;
	typing_style = restored_typing_style;
	typing_style_override = state.get("typing_style_override", false);
	pending_inline_metadata_restore_sync = restored_inline_metadata;
	_merge_adjacent_spans();
	_refresh_style_rendering();
	_style_changed();
}

void RichTextEdit::_push_style_undo_snapshot(const Vector<StyleSpan> &p_before_spans, const Vector<RichTextDocument::InlineImage> &p_before_images, const Vector<RichTextDocument::RawInline> &p_before_raw_inlines, const TextStyle &p_before_typing_style, bool p_before_typing_style_override) {
	if (p_before_spans == style_spans && p_before_images == images && p_before_raw_inlines == raw_inlines && p_before_typing_style == typing_style && p_before_typing_style_override == typing_style_override) {
		return;
	}

	push_custom_undo_operation(SNAME("rich_text_edit_style"), _make_style_state_variant(p_before_spans, p_before_images, p_before_raw_inlines, p_before_typing_style, p_before_typing_style_override), _make_style_state_variant(style_spans, images, raw_inlines, typing_style, typing_style_override));
}

void RichTextEdit::_refresh_style_rendering() {
	set_style_parser(callable_mp(this, &RichTextEdit::_get_line_style_spans));
	if (images.is_empty() && raw_inlines.is_empty()) {
		set_inline_object_handlers(Callable(), Callable(), Callable());
	} else {
		set_inline_object_handlers(callable_mp(this, &RichTextEdit::_get_line_inline_objects), callable_mp(this, &RichTextEdit::_draw_inline_object), callable_mp(this, &RichTextEdit::_inline_object_clicked));
	}
	queue_redraw();
}

void RichTextEdit::_style_changed() {
	_mark_bbcode_dirty();
	if (!is_inside_tree() || text_style_changed_dirty) {
		return;
	}

	text_style_changed_dirty = true;
	callable_mp(this, &RichTextEdit::_emit_text_style_changed).call_deferred();
}

void RichTextEdit::_emit_text_style_changed() {
	text_style_changed_dirty = false;
	emit_signal(SNAME("text_style_changed"));
}

Array RichTextEdit::_get_line_style_spans(int p_line) const {
	Array result;
	const int line_start = _get_line_start_offset(p_line);
	const int line_length = get_line(p_line).length();
	const int line_style_length = MAX(line_length, 1);
	const int line_style_end = line_start + line_style_length;
	for (const StyleSpan &span : style_spans) {
		const bool covers_empty_line_at_span_end = line_length == 0 && span.to == line_start && span.from < span.to;
		if (!covers_empty_line_at_span_end && (span.to <= line_start || span.from >= line_style_end)) {
			continue;
		}

		Ref<Font> font;
		int font_size = span.style.font_size;
		if (!span.style.font.is_empty()) {
			font = ResourceLoader::load(span.style.font, "Font");
		} else if (span.style.code) {
			font = get_theme_font(SNAME("mono_font"), SNAME("RichTextLabel"));
		} else if (span.style.bold && span.style.italic) {
			font = get_theme_font(SNAME("bold_italics_font"), SNAME("RichTextLabel"));
		} else if (span.style.bold) {
			font = get_theme_font(SNAME("bold_font"), SNAME("RichTextLabel"));
		} else if (span.style.italic) {
			font = get_theme_font(SNAME("italics_font"), SNAME("RichTextLabel"));
		}

		Dictionary info;
		info["from"] = MAX(span.from, line_start) - line_start;
		info["to"] = MIN(span.to, line_style_end) - line_start;
		if (span.style.has_color) {
			info["color"] = span.style.color;
		} else if (!span.style.url.is_empty()) {
			if (active_meta == span.style.url) {
				info["color"] = get_theme_color(SNAME("link_active_color"), SNAME("RichTextEdit"));
			} else if (meta_hovering && current_meta == span.style.url) {
				info["color"] = get_theme_color(SNAME("link_hover_color"), SNAME("RichTextEdit"));
			} else if (span.style.url_visited || visited_links.has(span.style.url)) {
				info["color"] = get_theme_color(SNAME("link_visited_color"), SNAME("RichTextEdit"));
			} else {
				info["color"] = get_theme_color(SNAME("link_color"), SNAME("RichTextEdit"));
			}
		}
		if (span.style.has_bg_color) {
			info["bg_color"] = span.style.bg_color;
		}
		if (span.style.has_fg_color) {
			info["fg_color"] = span.style.fg_color;
		}
		if (span.style.has_outline_color) {
			info["outline_color"] = span.style.outline_color;
		}
		if (span.style.has_outline_size) {
			info["outline_size"] = span.style.outline_size;
		}
		const bool underline = span.style.has_underline ? span.style.underline : !span.style.url.is_empty();
		if (underline) {
			info["underline"] = true;
		}
		if (!span.style.url.is_empty()) {
			info["url"] = span.style.url;
		}
		if (span.style.strikethrough) {
			info["strikethrough"] = true;
		}
		if (font.is_valid()) {
			info["font"] = font;
		}
		if (font_size > 0) {
			info["font_size"] = font_size;
		}
		if (span.style.alignment >= 0) {
			info["alignment"] = span.style.alignment;
		}
		if (span.style.indent_level > 0) {
			info["indent_level"] = span.style.indent_level;
		}
		if (span.style.list_type > 0) {
			info["list_type"] = span.style.list_type;
			if (span.style.list_start > 0) {
				info["list_start"] = span.style.list_start;
			}
			info["list_capitalize"] = span.style.list_capitalize;
		}
		if (span.style.block_tag == "quote" || span.style.block_tag.begins_with("quote ")) {
			info["quote"] = true;
			info["quote_margin_top"] = get_theme_constant(SNAME("quote_margin_top"), SNAME("RichTextEdit"));
			info["quote_margin_bottom"] = get_theme_constant(SNAME("quote_margin_bottom"), SNAME("RichTextEdit"));
			info["quote_border_color"] = get_theme_color(SNAME("quote_border_color"), SNAME("RichTextEdit"));
			info["quote_border_width"] = get_theme_constant(SNAME("quote_border_width"), SNAME("RichTextEdit"));
			info["quote_padding"] = get_theme_constant(SNAME("quote_padding"), SNAME("RichTextEdit"));
			if (!span.style.has_color) {
				info["color"] = get_theme_color(SNAME("quote_color"), SNAME("RichTextEdit"));
			}
		}
		if (info.size() > 2) {
			result.push_back(info);
		}
	}
	return result;
}

bool RichTextEdit::_get_url_at_offset(int p_offset, String &r_url) const {
	TextStyle style;
	if (!_get_url_style_at_offset(p_offset, style)) {
		return false;
	}
	r_url = style.url;
	return true;
}

bool RichTextEdit::_get_url_style_at_offset(int p_offset, TextStyle &r_style) const {
	if (p_offset < 0) {
		return false;
	}
	for (const StyleSpan &span : style_spans) {
		if (span.from <= p_offset && span.to > p_offset && !span.style.url.is_empty()) {
			r_style = span.style;
			return true;
		}
	}
	return false;
}

bool RichTextEdit::_get_url_at_position(const Point2 &p_position, String &r_url) const {
	TextStyle style;
	if (!_get_url_style_at_position(p_position, style)) {
		return false;
	}
	r_url = style.url;
	return true;
}

bool RichTextEdit::_get_url_style_at_position(const Point2 &p_position, TextStyle &r_style) const {
	const Point2i pos = get_line_column_at_pos(p_position, false, false);
	const int line = pos.y;
	if (line < 0 || line >= get_line_count()) {
		return false;
	}

	const int line_length = get_line(line).length();
	const int column = CLAMP(pos.x, 0, line_length);
	const int offset = _get_line_start_offset(line) + column;
	if (_get_url_style_at_offset(offset, r_style)) {
		return true;
	}
	if (column > 0) {
		return _get_url_style_at_offset(offset - 1, r_style);
	}
	return false;
}

void RichTextEdit::_update_meta_hover(const Point2 &p_position) {
	String meta;
	if (_get_url_at_position(p_position, meta)) {
		if (!meta_hovering || current_meta != meta) {
			if (meta_hovering) {
				emit_signal(SNAME("meta_hover_ended"), current_meta);
			}
			meta_hovering = true;
			current_meta = meta;
			emit_signal(SNAME("meta_hover_started"), meta);
			queue_redraw();
		}
	} else if (meta_hovering) {
		meta_hovering = false;
		emit_signal(SNAME("meta_hover_ended"), current_meta);
		current_meta = String();
		queue_redraw();
	}
}

bool RichTextEdit::_activate_url_at_position(const Point2 &p_position) {
	String meta;
	if (!_get_url_at_position(p_position, meta)) {
		return false;
	}

	if (has_connections(SNAME("meta_clicked"))) {
		active_meta = meta;
		visited_links.insert(meta);
		queue_redraw();
		emit_signal(SNAME("meta_clicked"), meta);
		return true;
	}

	if (_is_url_auto_open_allowed(meta)) {
		active_meta = meta;
		visited_links.insert(meta);
		queue_redraw();
		OS::get_singleton()->shell_open(meta.strip_edges());
		return true;
	}
	return false;
}

bool RichTextEdit::_should_activate_url(const Ref<InputEventMouseButton> &p_mouse_button) const {
	switch (link_activation_mode) {
		case LINK_ACTIVATION_AUTO:
			return !is_editable() || p_mouse_button->is_command_or_control_pressed();
		case LINK_ACTIVATION_CTRL_CLICK:
			return p_mouse_button->is_command_or_control_pressed();
		case LINK_ACTIVATION_CLICK:
			return true;
		case LINK_ACTIVATION_DISABLED:
			return false;
	}
	return false;
}

void RichTextEdit::_insert_newline(bool p_shift_pressed) {
	if (!is_editable()) {
		return;
	}
	if (has_selection()) {
		insert_text_at_caret("\n");
		accept_event();
		return;
	}

	const int line = get_caret_line();
	const int column = get_caret_column();
	const int caret_offset = _get_caret_offset();
	const String old_text = TextEdit::get_text();
	if (line < 0 || line >= get_line_count() || caret_offset < 0) {
		return;
	}

	const int line_length = get_line(line).length();
	const bool at_line_beginning = column == 0 || line_length == 0;
	TextStyle style_at_caret;
	if (line_length == 0) {
		style_at_caret = typing_style_override ? typing_style : _get_style_at_offset(caret_offset);
	} else if (at_line_beginning) {
		style_at_caret = _get_style_at_offset(caret_offset);
	} else {
		style_at_caret = _get_style_at_offset(caret_offset - 1);
	}

	if (p_shift_pressed && caret_offset < old_text.length() && old_text[caret_offset] == '\n' && !style_at_caret.is_default()) {
		const Vector<StyleSpan> before_spans = style_spans;

		typing_style = TextStyle();
		typing_style_override = true;
		insert_text_at_caret("\n");
		const String new_text = TextEdit::get_text();
		_sync_inline_objects_for_text_change(old_text, new_text, caret_offset, caret_offset, caret_offset + 1);

		Vector<StyleSpan> updated_spans;
		for (StyleSpan span : before_spans) {
			if (span.to <= caret_offset) {
				updated_spans.push_back(span);
			} else if (span.from >= caret_offset) {
				span.from++;
				span.to++;
				updated_spans.push_back(span);
			} else {
				StyleSpan left = span;
				left.to = caret_offset;
				if (left.from < left.to) {
					updated_spans.push_back(left);
				}

				StyleSpan right = span;
				right.from = caret_offset + 2;
				right.to = span.to + 1;
				if (right.from < right.to) {
					updated_spans.push_back(right);
				}
			}
		}

		style_spans = updated_spans;
		_merge_adjacent_spans();
		typing_style = TextStyle();
		typing_style_override = true;
		tracked_text = new_text;
		source_text = new_text;
		_mark_bbcode_dirty();
		_refresh_style_rendering();
		_style_changed();
		accept_event();
		return;
	}

	typing_style = p_shift_pressed ? TextStyle() : style_at_caret;
	typing_style_override = true;
	insert_text_at_caret("\n");
	if (tracked_text != TextEdit::get_text()) {
		_update_bbcode_from_text();
	}

	if (p_shift_pressed && at_line_beginning) {
		typing_style = style_at_caret;
		typing_style_override = true;
	} else {
		typing_style = p_shift_pressed ? TextStyle() : style_at_caret;
		typing_style_override = true;
	}

	source_text = TextEdit::get_text();
	_mark_bbcode_dirty();
	_refresh_style_rendering();
	_style_changed();
	accept_event();
}

Dictionary RichTextEdit::_make_image_inline_info(const RichTextDocument::InlineImage &p_image, int p_column) const {
	Dictionary info;
	info["column"] = p_column;
	info["offset"] = p_image.offset;
	info["cursor_shape"] = CURSOR_ARROW;

	Size2 texture_size;
	if (p_image.texture.is_valid()) {
		texture_size = p_image.texture->get_size();
	}

	float width = 0.0;
	float height = 0.0;
	const String size_text = p_image.options.has("size") ? p_image.options["size"] : String();
	if (!size_text.is_empty()) {
		Vector<String> parts = size_text.to_lower().split("x");
		if (parts.size() == 2) {
			width = parts[0].strip_edges().to_float();
			height = parts[1].strip_edges().to_float();
		} else {
			width = size_text.to_float();
		}
	}
	if (p_image.options.has("width")) {
		width = p_image.options["width"].to_float();
	}
	if (p_image.options.has("height")) {
		height = p_image.options["height"].to_float();
	}

	if (width > 0.0 || height > 0.0) {
		if (width <= 0.0 && texture_size.y > 0.0) {
			width = height * texture_size.x / texture_size.y;
		} else if (width <= 0.0) {
			width = height;
		}
		if (height <= 0.0 && texture_size.x > 0.0) {
			height = width * texture_size.y / texture_size.x;
		} else if (height <= 0.0) {
			height = width;
		}
		info["width"] = MAX(1.0f, width);
		info["height"] = MAX(1.0f, height);
		return info;
	}

	if (texture_size.x > 0.0 && texture_size.y > 0.0) {
		info["width"] = texture_size.x;
		info["height"] = texture_size.y;
		return info;
	}

	info["width_ratio"] = 1.0;
	info["height_ratio"] = 1.0;
	return info;
}

RichTextDocument::InlineImage *RichTextEdit::_get_image_at_offset(int p_offset) {
	for (RichTextDocument::InlineImage &image : images) {
		if (image.offset == p_offset) {
			return &image;
		}
	}
	return nullptr;
}

Size2 RichTextEdit::_get_image_option_size(const RichTextDocument::InlineImage &p_image, const Size2 &p_fallback_size) const {
	Size2 size = p_fallback_size;
	const String size_text = p_image.options.has("size") ? p_image.options["size"] : String();
	if (!size_text.is_empty()) {
		Vector<String> parts = size_text.to_lower().split("x");
		if (parts.size() == 2) {
			size.x = parts[0].strip_edges().to_float();
			size.y = parts[1].strip_edges().to_float();
		} else {
			size.x = size_text.to_float();
			size.y = size.x;
		}
	}
	if (p_image.options.has("width")) {
		size.x = p_image.options["width"].to_float();
	}
	if (p_image.options.has("height")) {
		size.y = p_image.options["height"].to_float();
	}
	if ((size.x <= 0.0f || size.y <= 0.0f) && p_image.texture.is_valid()) {
		const Size2 texture_size = p_image.texture->get_size();
		if (size.x <= 0.0f) {
			size.x = texture_size.x;
		}
		if (size.y <= 0.0f) {
			size.y = texture_size.y;
		}
	}
	return Size2(MAX(1.0f, size.x), MAX(1.0f, size.y));
}

float RichTextEdit::_get_fill_image_width() const {
	const Ref<StyleBox> style = get_theme_stylebox(SNAME("normal"));
	const float style_width = style.is_valid() ? style->get_minimum_size().width : 0.0f;
	const float fill_width = get_size().width - style_width - get_total_gutter_width() - get_line_start_margin();
	return MAX(1.0f, fill_width);
}

void RichTextEdit::_insert_image_with_options(const String &p_source, const HashMap<String, String> &p_options) {
	Vector<StyleSpan> before_spans = style_spans;
	Vector<RichTextDocument::InlineImage> before_images = images;
	Vector<RichTextDocument::RawInline> before_raw_inlines = raw_inlines;
	TextStyle before_typing_style = typing_style;
	const bool before_typing_style_override = typing_style_override;
	const int insert_offset = _get_caret_offset();

	begin_complex_operation();
	insert_text_at_caret(String::chr(RichTextDocument::OBJECT_REPLACEMENT_CHAR));
	const String new_text = TextEdit::get_text();
	_sync_style_spans_for_text_change(tracked_text, new_text);

	images.clear();
	for (RichTextDocument::InlineImage image : before_images) {
		if (image.offset >= insert_offset) {
			image.offset++;
		}
		images.push_back(image);
	}
	raw_inlines.clear();
	for (RichTextDocument::RawInline raw_inline : before_raw_inlines) {
		if (raw_inline.offset >= insert_offset) {
			raw_inline.offset++;
		}
		raw_inlines.push_back(raw_inline);
	}

	RichTextDocument::InlineImage image;
	image.offset = insert_offset;
	image.source = p_source;
	image.options = p_options;
	image.texture = RichTextDocument::load_image_texture(image.source);
	images.push_back(image);
	tracked_text = new_text;
	source_text = TextEdit::get_text();
	_mark_bbcode_dirty();
	_push_style_undo_snapshot(before_spans, before_images, before_raw_inlines, before_typing_style, before_typing_style_override);
	end_complex_operation();

	_refresh_style_rendering();
	_style_changed();
}

Vector<int> RichTextEdit::_get_selected_image_offsets() const {
	Vector<int> offsets;
	int from = 0;
	int to = 0;
	if (_get_selection_offsets(from, to)) {
		for (const RichTextDocument::InlineImage &image : images) {
			if (image.offset >= from && image.offset < to) {
				offsets.push_back(image.offset);
			}
		}
		return offsets;
	}

	if (selected_image_offset >= 0) {
		offsets.push_back(selected_image_offset);
		return offsets;
	}

	const int caret_offset = _get_caret_offset();
	for (const RichTextDocument::InlineImage &image : images) {
		if (image.offset == caret_offset || (caret_offset > 0 && image.offset == caret_offset - 1)) {
			offsets.push_back(image.offset);
			break;
		}
	}
	return offsets;
}

void RichTextEdit::_ensure_image_edit_controls() {
	if (image_edit_bar != nullptr) {
		return;
	}

	image_edit_bar = memnew(PanelContainer);
	add_child(image_edit_bar, false, INTERNAL_MODE_FRONT);
	image_edit_bar->hide();

	Ref<StyleBoxFlat> bar_panel;
	bar_panel.instantiate();
	bar_panel->set_bg_color(Color(0.12, 0.12, 0.12, 1.0));
	bar_panel->set_border_color(Color(0.35, 0.35, 0.35, 1.0));
	bar_panel->set_border_width_all(1);
	bar_panel->set_corner_radius_all(4);
	bar_panel->set_content_margin_individual(6, 4, 6, 4);
	image_edit_bar->add_theme_style_override(SceneStringName(panel), bar_panel);

	HBoxContainer *bar = memnew(HBoxContainer);
	image_edit_bar->add_child(bar);

	image_ratio_button = memnew(Button(RTR("Ratio")));
	image_ratio_button->set_toggle_mode(true);
	image_ratio_button->set_tooltip_text(RTR("Keep image width and height ratio"));
	bar->add_child(image_ratio_button);
	image_ratio_button->connect(SceneStringName(toggled), callable_mp(this, &RichTextEdit::_selected_image_ratio_toggled));

	image_fill_width_button = memnew(Button(RTR("Fill")));
	image_fill_width_button->set_toggle_mode(true);
	image_fill_width_button->set_tooltip_text(RTR("Fill editor width while keeping image ratio"));
	bar->add_child(image_fill_width_button);
	image_fill_width_button->connect(SceneStringName(toggled), callable_mp(this, &RichTextEdit::_selected_image_fill_width_toggled));

	Label *width_label = memnew(Label(RTR("W")));
	bar->add_child(width_label);

	image_width_spin = memnew(SpinBox);
	image_width_spin->set_min(1);
	image_width_spin->set_max(16384);
	image_width_spin->set_step(1);
	image_width_spin->set_custom_minimum_size(Size2(84, 0));
	bar->add_child(image_width_spin);
	image_width_spin->connect(SceneStringName(value_changed), callable_mp(this, &RichTextEdit::_selected_image_width_changed));

	Label *height_label = memnew(Label(RTR("H")));
	bar->add_child(height_label);

	image_height_spin = memnew(SpinBox);
	image_height_spin->set_min(1);
	image_height_spin->set_max(16384);
	image_height_spin->set_step(1);
	image_height_spin->set_custom_minimum_size(Size2(84, 0));
	bar->add_child(image_height_spin);
	image_height_spin->connect(SceneStringName(value_changed), callable_mp(this, &RichTextEdit::_selected_image_height_changed));

	Button *details_button = memnew(Button(RTR("Edit")));
	details_button->set_tooltip_text(RTR("Edit image alt text and link"));
	bar->add_child(details_button);
	details_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEdit::_pressed_selected_image_details));

	image_details_popup = memnew(PopupPanel);
	add_child(image_details_popup, false, INTERNAL_MODE_FRONT);

	VBoxContainer *details = memnew(VBoxContainer);
	image_details_popup->add_child(details);

	details->add_child(memnew(Label(RTR("Alt Text"))));
	image_alt_edit = memnew(LineEdit);
	image_alt_edit->set_custom_minimum_size(Size2(260, 0));
	details->add_child(image_alt_edit);

	details->add_child(memnew(Label(RTR("Link"))));
	image_link_edit = memnew(LineEdit);
	image_link_edit->set_custom_minimum_size(Size2(260, 0));
	details->add_child(image_link_edit);

	Button *apply_button = memnew(Button(RTR("Apply")));
	details->add_child(apply_button);
	apply_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEdit::_apply_selected_image_details));
}

void RichTextEdit::_select_image(int p_offset, const Rect2 &p_screen_rect) {
	selected_image_offset = p_offset;
	_ensure_image_edit_controls();

	RichTextDocument::InlineImage *image = _get_image_at_offset(p_offset);
	if (image == nullptr) {
		_clear_selected_image();
		return;
	}

	updating_image_edit_controls = true;
	const Size2 image_size = _get_image_option_size(*image, p_screen_rect.size);
	image_width_spin->set_value(image_size.x);
	image_height_spin->set_value(image_size.y);
	image_alt_edit->set_text(image->options.has("alt") ? image->options["alt"] : String());
	image_link_edit->set_text(image->options.has("link") ? image->options["link"] : String());
	image_ratio_button->set_pressed_no_signal(image->options.has("_keep_ratio") && image->options["_keep_ratio"] == "true");
	image_fill_width_button->set_pressed_no_signal(image->options.has("_fill_width") && image->options["_fill_width"] == "true");
	updating_image_edit_controls = false;

	image_edit_bar->reset_size();
	Size2 bar_size = image_edit_bar->get_size();
	if (bar_size.x <= 0.0f || bar_size.y <= 0.0f) {
		bar_size = image_edit_bar->get_combined_minimum_size();
		image_edit_bar->set_size(bar_size);
	}

	const Rect2 local_image_rect(p_screen_rect.position - get_screen_position(), p_screen_rect.size);
	Point2 bar_position(local_image_rect.position.x + (local_image_rect.size.x - bar_size.x) * 0.5f, local_image_rect.position.y - bar_size.y);
	bar_position.x = CLAMP(bar_position.x, 0.0f, MAX(0.0f, get_size().x - bar_size.x));
	image_edit_bar->set_position(bar_position);
	image_edit_bar->show();
	queue_redraw();
}

void RichTextEdit::_clear_selected_image() {
	selected_image_offset = -1;
	if (image_edit_bar != nullptr) {
		image_edit_bar->hide();
	}
	if (image_details_popup != nullptr) {
		image_details_popup->hide();
	}
	queue_redraw();
}

void RichTextEdit::_commit_selected_image_options(const String &p_width, const String &p_height, const String &p_alt, const String &p_link, bool p_record_undo) {
	if (selected_image_offset < 0) {
		return;
	}
	RichTextDocument::InlineImage *image = _get_image_at_offset(selected_image_offset);
	if (image == nullptr) {
		_clear_selected_image();
		return;
	}

	const Vector<StyleSpan> before_spans = style_spans;
	const Vector<RichTextDocument::InlineImage> before_images = images;
	const Vector<RichTextDocument::RawInline> before_raw_inlines = raw_inlines;
	const TextStyle before_typing_style = typing_style;
	const bool before_typing_style_override = typing_style_override;

	image->options.erase("size");
	if (p_width.is_empty()) {
		image->options.erase("width");
	} else {
		image->options["width"] = p_width;
	}
	if (p_height.is_empty()) {
		image->options.erase("height");
	} else {
		image->options["height"] = p_height;
	}
	if (p_alt.is_empty()) {
		image->options.erase("alt");
	} else {
		image->options["alt"] = p_alt;
	}
	if (p_link.is_empty()) {
		image->options.erase("link");
	} else {
		image->options["link"] = p_link;
	}

	source_text = TextEdit::get_text();
	_mark_bbcode_dirty();
	if (p_record_undo) {
		_push_style_undo_snapshot(before_spans, before_images, before_raw_inlines, before_typing_style, before_typing_style_override);
	}
	_refresh_style_rendering();
	_style_changed();
}

void RichTextEdit::_selected_image_width_changed(double p_value) {
	if (updating_image_edit_controls || selected_image_offset < 0) {
		return;
	}
	if (image_ratio_button != nullptr && image_ratio_button->is_pressed()) {
		RichTextDocument::InlineImage *image = _get_image_at_offset(selected_image_offset);
		if (image != nullptr) {
			const Size2 current_size = _get_image_option_size(*image, Size2(image_width_spin->get_value(), image_height_spin->get_value()));
			if (current_size.x > 0.0f) {
				updating_image_edit_controls = true;
				image_height_spin->set_value(MAX(1.0, p_value * current_size.y / current_size.x));
				updating_image_edit_controls = false;
			}
		}
	}
	_commit_selected_image_options(String::num_real(p_value, false), String::num_real(image_height_spin->get_value(), false), image_alt_edit->get_text(), image_link_edit->get_text());
}

void RichTextEdit::_selected_image_height_changed(double p_value) {
	if (updating_image_edit_controls || selected_image_offset < 0) {
		return;
	}
	if (image_ratio_button != nullptr && image_ratio_button->is_pressed()) {
		RichTextDocument::InlineImage *image = _get_image_at_offset(selected_image_offset);
		if (image != nullptr) {
			const Size2 current_size = _get_image_option_size(*image, Size2(image_width_spin->get_value(), image_height_spin->get_value()));
			if (current_size.y > 0.0f) {
				updating_image_edit_controls = true;
				image_width_spin->set_value(MAX(1.0, p_value * current_size.x / current_size.y));
				updating_image_edit_controls = false;
			}
		}
	}
	_commit_selected_image_options(String::num_real(image_width_spin->get_value(), false), String::num_real(p_value, false), image_alt_edit->get_text(), image_link_edit->get_text());
}

void RichTextEdit::_selected_image_ratio_toggled(bool p_pressed) {
	if (updating_image_edit_controls || selected_image_offset < 0) {
		return;
	}
	RichTextDocument::InlineImage *image = _get_image_at_offset(selected_image_offset);
	if (image == nullptr) {
		_clear_selected_image();
		return;
	}

	const Vector<StyleSpan> before_spans = style_spans;
	const Vector<RichTextDocument::InlineImage> before_images = images;
	const Vector<RichTextDocument::RawInline> before_raw_inlines = raw_inlines;
	const TextStyle before_typing_style = typing_style;
	const bool before_typing_style_override = typing_style_override;

	if (p_pressed) {
		image->options["_keep_ratio"] = "true";
	} else {
		image->options.erase("_keep_ratio");
	}
	_commit_selected_image_options(String::num_real(image_width_spin->get_value(), false), String::num_real(image_height_spin->get_value(), false), image_alt_edit->get_text(), image_link_edit->get_text(), false);
	_push_style_undo_snapshot(before_spans, before_images, before_raw_inlines, before_typing_style, before_typing_style_override);
}

void RichTextEdit::_selected_image_fill_width_toggled(bool p_pressed) {
	if (updating_image_edit_controls || selected_image_offset < 0) {
		return;
	}
	RichTextDocument::InlineImage *image = _get_image_at_offset(selected_image_offset);
	if (image == nullptr) {
		_clear_selected_image();
		return;
	}

	const Vector<StyleSpan> before_spans = style_spans;
	const Vector<RichTextDocument::InlineImage> before_images = images;
	const Vector<RichTextDocument::RawInline> before_raw_inlines = raw_inlines;
	const TextStyle before_typing_style = typing_style;
	const bool before_typing_style_override = typing_style_override;

	Size2 target_size;
	if (p_pressed) {
		const Size2 current_size = _get_image_option_size(*image, Size2(image_width_spin->get_value(), image_height_spin->get_value()));
		image->options["_fill_width"] = "true";
		image->options["_previous_width"] = String::num_real(current_size.x, false);
		image->options["_previous_height"] = String::num_real(current_size.y, false);
		const float fill_width = _get_fill_image_width();
		target_size = Size2(fill_width, MAX(1.0f, fill_width * current_size.y / current_size.x));
	} else {
		image->options.erase("_fill_width");
		if (image->options.has("_previous_width") && image->options.has("_previous_height")) {
			target_size = Size2(image->options["_previous_width"].to_float(), image->options["_previous_height"].to_float());
		} else {
			target_size = _get_image_option_size(*image, Size2(image_width_spin->get_value(), image_height_spin->get_value()));
		}
		image->options.erase("_previous_width");
		image->options.erase("_previous_height");
	}

	updating_image_edit_controls = true;
	image_width_spin->set_value(MAX(1.0f, target_size.x));
	image_height_spin->set_value(MAX(1.0f, target_size.y));
	updating_image_edit_controls = false;
	_commit_selected_image_options(String::num_real(image_width_spin->get_value(), false), String::num_real(image_height_spin->get_value(), false), image_alt_edit->get_text(), image_link_edit->get_text(), false);
	_push_style_undo_snapshot(before_spans, before_images, before_raw_inlines, before_typing_style, before_typing_style_override);
}

void RichTextEdit::_pressed_selected_image_details() {
	if (selected_image_offset < 0 || image_details_popup == nullptr) {
		return;
	}
	image_details_popup->set_position(get_screen_position() + image_edit_bar->get_position() + Vector2(0, image_edit_bar->get_size().y + 4));
	image_details_popup->reset_size();
	image_details_popup->popup();
}

void RichTextEdit::_apply_selected_image_details() {
	if (selected_image_offset < 0) {
		return;
	}
	_commit_selected_image_options(String::num_real(image_width_spin->get_value(), false), String::num_real(image_height_spin->get_value(), false), image_alt_edit->get_text(), image_link_edit->get_text());
	if (image_details_popup != nullptr) {
		image_details_popup->hide();
	}
}

Array RichTextEdit::_get_line_inline_objects(int p_line, const String &p_line_text) const {
	Array result;
	const int line_start = _get_line_start_offset(p_line);
	for (int i = 0; i < p_line_text.length(); i++) {
		if (p_line_text[i] != RichTextDocument::OBJECT_REPLACEMENT_CHAR) {
			continue;
		}
		const int offset = line_start + i;
		for (const RichTextDocument::InlineImage &image : images) {
			if (image.offset == offset) {
				result.push_back(_make_image_inline_info(image, i));
				break;
			}
		}
	}
	return result;
}

void RichTextEdit::_draw_inline_object(const Dictionary &p_info, const Rect2 &p_rect) {
	RID text_canvas_item = get_text_canvas_item();
	Rect2 draw_rect = p_rect;
	Rect2 src_rect;
	if (p_info.has("clip_rect")) {
		draw_rect = p_info["clip_rect"];
		if (draw_rect.size.x <= 0.0 || draw_rect.size.y <= 0.0 || p_rect.size.x <= 0.0 || p_rect.size.y <= 0.0) {
			return;
		}
		const Point2 src_pos = (draw_rect.position - p_rect.position) / p_rect.size;
		const Size2 src_size = draw_rect.size / p_rect.size;
		src_rect = Rect2(src_pos, src_size);
	}
	const auto draw_placeholder = [&]() {
		RS::get_singleton()->canvas_item_add_rect(text_canvas_item, draw_rect, Color(0.25, 0.25, 0.25, 0.35));
		const Color outline_color = Color(0.65, 0.65, 0.65, 0.8);
		RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(draw_rect.position, Size2(draw_rect.size.x, 1)), outline_color);
		RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(Point2(draw_rect.position.x, draw_rect.position.y + draw_rect.size.y - 1), Size2(draw_rect.size.x, 1)), outline_color);
		RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(draw_rect.position, Size2(1, draw_rect.size.y)), outline_color);
		RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(Point2(draw_rect.position.x + draw_rect.size.x - 1, draw_rect.position.y), Size2(1, draw_rect.size.y)), outline_color);
	};

	int offset = p_info.get("offset", -1);
	if (offset < 0) {
		int line = p_info.get("line", -1);
		int column = p_info.get("column", -1);
		if (line < 0 || column < 0) {
			return;
		}
		offset = _get_line_start_offset(line) + column;
	}
	if (offset < 0) {
		return;
	}
	for (const RichTextDocument::InlineImage &image : images) {
		if (image.offset != offset) {
			continue;
		}
		if (image.texture.is_valid()) {
			if (src_rect.has_area()) {
				const Size2 texture_size = image.texture->get_size();
				image.texture->draw_rect_region(text_canvas_item, draw_rect, Rect2(src_rect.position * texture_size, src_rect.size * texture_size), Color(1, 1, 1));
			} else {
				image.texture->draw_rect(text_canvas_item, draw_rect, false, Color(1, 1, 1));
			}
		} else {
			draw_placeholder();
		}
		if (selected_image_offset == offset) {
			const Color selection_color = Color(0.25, 0.55, 1.0, 1.0);
			RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(draw_rect.position, Size2(draw_rect.size.x, 2)), selection_color);
			RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(Point2(draw_rect.position.x, draw_rect.position.y + draw_rect.size.y - 2), Size2(draw_rect.size.x, 2)), selection_color);
			RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(draw_rect.position, Size2(2, draw_rect.size.y)), selection_color);
			RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(Point2(draw_rect.position.x + draw_rect.size.x - 2, draw_rect.position.y), Size2(2, draw_rect.size.y)), selection_color);
		}
		return;
	}
	draw_placeholder();
}

void RichTextEdit::_inline_object_clicked(const Dictionary &p_info, const Rect2 &p_rect) {
	int offset = p_info.get("offset", -1);
	if (offset < 0) {
		int line = p_info.get("line", -1);
		int column = p_info.get("column", -1);
		if (line >= 0 && column >= 0) {
			offset = _get_line_start_offset(line) + column;
		}
	}
	if (offset >= 0) {
		_select_image(offset, p_rect);
	}
}

void RichTextEdit::_notification(int p_what) {
	if (p_what == NOTIFICATION_MOUSE_EXIT && meta_hovering) {
		meta_hovering = false;
		emit_signal(SNAME("meta_hover_ended"), current_meta);
		current_meta = String();
	}
	if (p_what == NOTIFICATION_MOUSE_EXIT && !active_meta.is_empty()) {
		active_meta = String();
		queue_redraw();
	}
}

void RichTextEdit::set_text(const String &p_text) {
	source_text = p_text;
	setting_bbcode_text = true;
	if (use_bbcode) {
		RichTextDocument document = RichTextDocument::parse_bbcode(p_text);
		_apply_document(document);
		TextEdit::set_text(document.text);
		tracked_text = document.text;
		bbcode_text = p_text;
		bbcode_dirty = false;
	} else {
		TextEdit::set_text(p_text);
		tracked_text = p_text;
		style_spans.clear();
		deleted_style_ranges_for_undo.clear();
		images.clear();
		raw_inlines.clear();
		_mark_bbcode_dirty();
	}
	setting_bbcode_text = false;

	typing_style_override = false;
	clear_undo_history();
	_refresh_style_rendering();
	_style_changed();
	if (use_bbcode) {
		bbcode_text = p_text;
		bbcode_dirty = false;
	}
}

String RichTextEdit::get_text() const {
	return use_bbcode ? get_bbcode_text() : source_text;
}

void RichTextEdit::gui_input(const Ref<InputEvent> &p_gui_input) {
	ERR_FAIL_COND(p_gui_input.is_null());

	Ref<InputEventKey> key = p_gui_input;
	if (key.is_valid() && key->is_pressed() && !key->is_echo() && !key->is_command_or_control_pressed() && !key->is_alt_pressed() && key->is_action("ui_text_newline", true)) {
		_insert_newline(key->is_shift_pressed());
		return;
	}

	Ref<InputEventMouseButton> mb = p_gui_input;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT && _should_activate_url(mb)) {
		Vector2i mpos = mb->get_position();
		if (is_layout_rtl()) {
			mpos.x = get_size().x - mpos.x;
		}
		if (_activate_url_at_position(mpos)) {
			accept_event();
			return;
		}
	} else if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT && selected_image_offset >= 0) {
		_clear_selected_image();
	} else if (mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT && !active_meta.is_empty()) {
		active_meta = String();
		queue_redraw();
	}

	Ref<InputEventMouseMotion> mm = p_gui_input;
	if (mm.is_valid()) {
		Vector2i mpos = mm->get_position();
		if (is_layout_rtl()) {
			mpos.x = get_size().x - mpos.x;
		}
		_update_meta_hover(mpos);
	}

	TextEdit::gui_input(p_gui_input);
}

Control::CursorShape RichTextEdit::get_cursor_shape(const Point2 &p_pos) const {
	String meta;
	if (link_activation_mode != LINK_ACTIVATION_DISABLED && _get_url_at_position(p_pos, meta)) {
		return CURSOR_POINTING_HAND;
	}
	return TextEdit::get_cursor_shape(p_pos);
}

String RichTextEdit::get_tooltip(const Point2 &p_pos) const {
	TextStyle style;
	if (_get_url_style_at_position(p_pos, style) && !style.url_tooltip.is_empty()) {
		return style.url_tooltip;
	}
	return TextEdit::get_tooltip(p_pos);
}

Control *RichTextEdit::make_custom_tooltip(const String &p_text) const {
	if (p_text.is_empty()) {
		return nullptr;
	}

	PanelContainer *panel = memnew(PanelContainer);
	Ref<StyleBox> panel_style = get_theme_stylebox(SNAME("tooltip_panel"), SNAME("RichTextEdit"));
	if (panel_style.is_null()) {
		panel_style = get_theme_stylebox(SceneStringName(panel), SNAME("TooltipPanel"));
	}
	if (panel_style.is_valid()) {
		panel->add_theme_style_override(SceneStringName(panel), panel_style);
	}

	Label *label = memnew(Label);
	label->set_text(p_text);

	const Color font_color = get_theme_color(SNAME("tooltip_font_color"), SNAME("RichTextEdit"));
	label->add_theme_color_override(SceneStringName(font_color), font_color);

	Ref<Font> font = get_theme_font(SNAME("tooltip_font"), SNAME("RichTextEdit"));
	if (font.is_null()) {
		font = get_theme_font(SceneStringName(font), SNAME("TooltipLabel"));
	}
	if (font.is_valid()) {
		label->add_theme_font_override(SceneStringName(font), font);
	}

	const int font_size = get_theme_font_size(SNAME("tooltip_font_size"), SNAME("RichTextEdit"));
	if (font_size > 0) {
		label->add_theme_font_size_override(SceneStringName(font_size), font_size);
	}

	panel->add_child(label);
	return panel;
}

void RichTextEdit::set_use_bbcode(bool p_enable) {
	if (use_bbcode == p_enable) {
		return;
	}
	const String current_text = get_text();
	use_bbcode = p_enable;
	set_text(current_text);
	notify_property_list_changed();
}

bool RichTextEdit::is_using_bbcode() const {
	return use_bbcode;
}

void RichTextEdit::set_link_activation_mode(LinkActivationMode p_mode) {
	ERR_FAIL_INDEX((int)p_mode, LINK_ACTIVATION_DISABLED + 1);

	link_activation_mode = p_mode;
}

RichTextEdit::LinkActivationMode RichTextEdit::get_link_activation_mode() const {
	return link_activation_mode;
}

void RichTextEdit::set_bbcode_text(const String &p_bbcode) {
	use_bbcode = true;
	source_text = p_bbcode;
	RichTextDocument document = RichTextDocument::parse_bbcode(p_bbcode);

	setting_bbcode_text = true;
	bbcode_text = p_bbcode;
	bbcode_dirty = false;
	_apply_document(document);
	TextEdit::set_text(document.text);
	tracked_text = document.text;
	setting_bbcode_text = false;

	_merge_adjacent_spans();
	typing_style_override = false;
	deleted_style_ranges_for_undo.clear();
	clear_undo_history();
	_refresh_style_rendering();
	_style_changed();
	bbcode_text = p_bbcode;
	bbcode_dirty = false;
}

String RichTextEdit::get_bbcode_text() const {
	if (bbcode_dirty) {
		bbcode_text = _serialize_bbcode();
		bbcode_dirty = false;
	}
	return bbcode_text;
}

const Vector<RichTextEdit::StyleSpan> &RichTextEdit::get_style_spans() const {
	return style_spans;
}

void RichTextEdit::set_bold() {
	_apply_style_property_to_selection(STYLE_PROPERTY_BOLD, true);
}

void RichTextEdit::clear_bold() {
	_apply_style_property_to_selection(STYLE_PROPERTY_BOLD, false);
}

void RichTextEdit::toggle_bold() {
	_apply_style_property_to_selection(STYLE_PROPERTY_BOLD);
}

void RichTextEdit::set_italic() {
	_apply_style_property_to_selection(STYLE_PROPERTY_ITALIC, true);
}

void RichTextEdit::clear_italic() {
	_apply_style_property_to_selection(STYLE_PROPERTY_ITALIC, false);
}

void RichTextEdit::toggle_italic() {
	_apply_style_property_to_selection(STYLE_PROPERTY_ITALIC);
}

void RichTextEdit::set_underline() {
	_apply_style_property_to_selection(STYLE_PROPERTY_UNDERLINE, true);
}

void RichTextEdit::clear_underline() {
	_apply_style_property_to_selection(STYLE_PROPERTY_UNDERLINE, false);
}

void RichTextEdit::toggle_underline() {
	_apply_style_property_to_selection(STYLE_PROPERTY_UNDERLINE);
}

void RichTextEdit::set_strikethrough() {
	_apply_style_property_to_selection(STYLE_PROPERTY_STRIKETHROUGH, true);
}

void RichTextEdit::clear_strikethrough() {
	_apply_style_property_to_selection(STYLE_PROPERTY_STRIKETHROUGH, false);
}

void RichTextEdit::toggle_strikethrough() {
	_apply_style_property_to_selection(STYLE_PROPERTY_STRIKETHROUGH);
}

void RichTextEdit::set_selection_color(const Color &p_color) {
	_apply_style_property_to_selection(STYLE_PROPERTY_COLOR, p_color);
}

void RichTextEdit::clear_selection_color() {
	_apply_style_property_to_selection(STYLE_PROPERTY_CLEAR_COLOR);
}

void RichTextEdit::set_selection_bg_color(const Color &p_color) {
	_apply_style_property_to_selection(STYLE_PROPERTY_BG_COLOR, p_color);
}

void RichTextEdit::clear_selection_bg_color() {
	_apply_style_property_to_selection(STYLE_PROPERTY_CLEAR_BG_COLOR);
}

void RichTextEdit::set_selection_outline_color(const Color &p_color) {
	_apply_style_property_to_selection(STYLE_PROPERTY_OUTLINE_COLOR, p_color);
}

void RichTextEdit::clear_selection_outline_color() {
	_apply_style_property_to_selection(STYLE_PROPERTY_CLEAR_OUTLINE_COLOR);
}

void RichTextEdit::set_selection_outline_size(int p_size) {
	_apply_style_property_to_selection(STYLE_PROPERTY_OUTLINE_SIZE, p_size);
}

void RichTextEdit::clear_selection_outline_size() {
	_apply_style_property_to_selection(STYLE_PROPERTY_CLEAR_OUTLINE_SIZE);
}

void RichTextEdit::begin_selection_color_preview() {
	if (style_preview_active) {
		return;
	}

	style_preview_active = true;
	int from = 0;
	int to = 0;
	style_preview_had_selection = _get_selection_offsets(from, to);
	style_preview_before_spans = style_spans;
	style_preview_before_typing_style = typing_style;
	style_preview_before_typing_style_override = typing_style_override;
}

void RichTextEdit::preview_selection_color(const Color &p_color) {
	if (!style_preview_active) {
		begin_selection_color_preview();
	}
	_apply_style_property_to_selection(STYLE_PROPERTY_COLOR, p_color, false);
}

void RichTextEdit::end_selection_color_preview(bool p_commit) {
	if (!style_preview_active) {
		return;
	}

	const Vector<StyleSpan> before_spans = style_preview_before_spans;
	const TextStyle before_typing_style = style_preview_before_typing_style;
	const bool before_typing_style_override = style_preview_before_typing_style_override;
	const bool had_selection = style_preview_had_selection;
	style_preview_active = false;
	style_preview_had_selection = false;
	style_preview_before_spans.clear();

	if (p_commit) {
		if (had_selection) {
			_push_style_undo_snapshot(before_spans, images, raw_inlines, before_typing_style, before_typing_style_override);
		}
		source_text = TextEdit::get_text();
		_mark_bbcode_dirty();
		_style_changed();
	} else {
		style_spans = before_spans;
		typing_style = before_typing_style;
		typing_style_override = before_typing_style_override;
		_merge_adjacent_spans();
		source_text = TextEdit::get_text();
		_mark_bbcode_dirty();
		_refresh_style_rendering();
	}
}

void RichTextEdit::begin_selection_bg_color_preview() {
	if (style_preview_active) {
		return;
	}

	style_preview_active = true;
	int from = 0;
	int to = 0;
	style_preview_had_selection = _get_selection_offsets(from, to);
	style_preview_before_spans = style_spans;
	style_preview_before_typing_style = typing_style;
	style_preview_before_typing_style_override = typing_style_override;
}

void RichTextEdit::preview_selection_bg_color(const Color &p_color) {
	if (!style_preview_active) {
		begin_selection_bg_color_preview();
	}
	_apply_style_property_to_selection(STYLE_PROPERTY_BG_COLOR, p_color, false);
}

void RichTextEdit::end_selection_bg_color_preview(bool p_commit) {
	end_selection_color_preview(p_commit);
}

void RichTextEdit::set_selection_font_size(int p_size) {
	_apply_style_property_to_selection(STYLE_PROPERTY_FONT_SIZE, p_size);
}

void RichTextEdit::clear_selection_font_size() {
	_apply_style_property_to_selection(STYLE_PROPERTY_CLEAR_FONT_SIZE);
}

void RichTextEdit::set_selection_font(const String &p_font) {
	_apply_style_property_to_selection(STYLE_PROPERTY_FONT, p_font);
}

void RichTextEdit::clear_selection_font() {
	_apply_style_property_to_selection(STYLE_PROPERTY_CLEAR_FONT);
}

void RichTextEdit::set_selection_url(const String &p_url) {
	_apply_style_property_to_selection(STYLE_PROPERTY_URL, p_url);
}

void RichTextEdit::clear_selection_url() {
	_apply_style_property_to_selection(STYLE_PROPERTY_CLEAR_URL);
}

void RichTextEdit::set_selection_url_tooltip(const String &p_tooltip) {
	_apply_style_property_to_url_ranges(STYLE_PROPERTY_URL_TOOLTIP, p_tooltip);
}

void RichTextEdit::clear_selection_url_tooltip() {
	_apply_style_property_to_url_ranges(STYLE_PROPERTY_CLEAR_URL_TOOLTIP);
}

void RichTextEdit::set_selection_url_visited() {
	_apply_style_property_to_url_ranges(STYLE_PROPERTY_URL_VISITED);
}

void RichTextEdit::clear_selection_url_visited() {
	_apply_style_property_to_url_ranges(STYLE_PROPERTY_CLEAR_URL_VISITED);
}

void RichTextEdit::toggle_code() {
	_apply_style_property_to_selection(STYLE_PROPERTY_CODE);
}

void RichTextEdit::insert_image(const String &p_source, int p_width, int p_height, const String &p_alt) {
	HashMap<String, String> options;
	if (p_width >= 0) {
		options["width"] = itos(p_width);
	}
	if (p_height >= 0) {
		options["height"] = itos(p_height);
	}
	if (!p_alt.is_empty()) {
		options["alt"] = p_alt;
	}

	_insert_image_with_options(p_source, options);
}

void RichTextEdit::set_selection_image_size(int p_width, int p_height) {
	if (p_width < 0 && p_height < 0) {
		return;
	}

	const Vector<int> offsets = _get_selected_image_offsets();
	if (offsets.is_empty()) {
		return;
	}

	const Vector<StyleSpan> before_spans = style_spans;
	const Vector<RichTextDocument::InlineImage> before_images = images;
	const Vector<RichTextDocument::RawInline> before_raw_inlines = raw_inlines;
	const TextStyle before_typing_style = typing_style;
	const bool before_typing_style_override = typing_style_override;

	for (int offset : offsets) {
		RichTextDocument::InlineImage *image = _get_image_at_offset(offset);
		if (image == nullptr) {
			continue;
		}
		image->options.erase("size");
		if (p_width >= 0) {
			image->options["width"] = itos(p_width);
		} else {
			image->options.erase("width");
		}
		if (p_height >= 0) {
			image->options["height"] = itos(p_height);
		} else {
			image->options.erase("height");
		}
	}

	if (before_images == images) {
		return;
	}
	_push_style_undo_snapshot(before_spans, before_images, before_raw_inlines, before_typing_style, before_typing_style_override);
	source_text = TextEdit::get_text();
	_mark_bbcode_dirty();
	_refresh_style_rendering();
	_style_changed();
}

void RichTextEdit::set_alignment(HorizontalAlignment p_alignment) {
	switch (p_alignment) {
		case HORIZONTAL_ALIGNMENT_CENTER:
			_apply_block_tag_to_selected_lines("center");
			break;
		case HORIZONTAL_ALIGNMENT_RIGHT:
			_apply_block_tag_to_selected_lines("right");
			break;
		case HORIZONTAL_ALIGNMENT_FILL:
			_apply_block_tag_to_selected_lines("fill");
			break;
		case HORIZONTAL_ALIGNMENT_LEFT:
		default:
			_apply_block_tag_to_selected_lines("left");
			break;
	}
}

void RichTextEdit::toggle_quote() {
	_toggle_block_tag_on_selected_lines("quote");
}

void RichTextEdit::set_quote() {
	_apply_block_tag_to_selected_lines("quote");
}

void RichTextEdit::clear_quote() {
	_clear_block_tag_from_selected_lines("quote");
}

void RichTextEdit::toggle_unordered_list() {
	_toggle_block_tag_on_selected_lines("ul");
}

void RichTextEdit::set_unordered_list() {
	_apply_block_tag_to_selected_lines("ul");
}

void RichTextEdit::clear_unordered_list() {
	_clear_block_tag_from_selected_lines("ul");
}

void RichTextEdit::toggle_ordered_list() {
	_toggle_block_tag_on_selected_lines("ol");
}

void RichTextEdit::set_ordered_list() {
	_apply_block_tag_to_selected_lines("ol");
}

void RichTextEdit::clear_ordered_list() {
	_clear_block_tag_from_selected_lines("ol");
}

int RichTextEdit::get_current_font_size() const {
	int offset = 0;
	if (has_selection()) {
		int to = 0;
		_get_selection_offsets(offset, to);
	} else {
		const int caret_line = get_caret_line();
		const int caret_column = get_caret_column();
		const int line_start = _get_line_start_offset(caret_line);
		if (caret_column == 0) {
			offset = get_line(caret_line).is_empty() ? line_start : line_start;
		} else {
			offset = line_start + caret_column - 1;
		}
	}

	const TextStyle style = _get_style_at_offset(offset);
	return style.font_size > 0 ? style.font_size : _get_default_font_size();
}

void RichTextEdit::increase_indent() {
	_apply_block_tag_to_selected_lines("indent");
}

void RichTextEdit::decrease_indent() {
	int from_line = has_selection() ? get_selection_from_line() : get_caret_line();
	int to_line = has_selection() ? get_selection_to_line() : get_caret_line();
	const Vector<StyleSpan> before_spans = style_spans;
	const Vector<RichTextDocument::InlineImage> before_images = images;
	const Vector<RichTextDocument::RawInline> before_raw_inlines = raw_inlines;
	const TextStyle before_typing_style = typing_style;
	const bool before_typing_style_override = typing_style_override;

	for (int line = from_line; line <= to_line; line++) {
		const int line_start = _get_line_start_offset(line);
		const int line_end = line_start + get_line(line).length();
		if (line_start == line_end) {
			continue;
		}
		const Vector<int> boundaries = _get_style_boundaries_for_range(line_start, line_end);
		for (int i = 0; i < boundaries.size() - 1; i++) {
			const int segment_from = boundaries[i];
			const int segment_to = boundaries[i + 1];
			if (segment_from == segment_to) {
				continue;
			}
			TextStyle style = _get_style_at_offset(segment_from);
			if (style.indent_level <= 0) {
				continue;
			}
			style.indent_level = MAX(0, style.indent_level - 1);
			if (style.indent_level == 0 && (style.block_tag == "indent" || style.block_tag == "ul" || style.block_tag == "ol")) {
				style.block_tag = "";
				style.list_type = 0;
				style.list_start = -1;
				style.list_capitalize = false;
			} else if (style.block_tag == "indent" && style.indent_level > 0) {
				style.block_tag = "indent";
			}
			_replace_style_range(segment_from, segment_to, style);
		}
	}

	if (before_spans == style_spans) {
		return;
	}
	_push_style_undo_snapshot(before_spans, before_images, before_raw_inlines, before_typing_style, before_typing_style_override);
	source_text = TextEdit::get_text();
	_mark_bbcode_dirty();
	_refresh_style_rendering();
	_style_changed();
}

void RichTextEdit::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "syntax_highlighter") {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void RichTextEdit::_apply_custom_undo_operation(const StringName &p_type, const Variant &p_data) {
	if (p_type == SNAME("rich_text_edit_style")) {
		_restore_style_state_variant(p_data);
	}
}

void RichTextEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_use_bbcode", "enable"), &RichTextEdit::set_use_bbcode);
	ClassDB::bind_method(D_METHOD("is_using_bbcode"), &RichTextEdit::is_using_bbcode);
	ClassDB::bind_method(D_METHOD("set_link_activation_mode", "mode"), &RichTextEdit::set_link_activation_mode);
	ClassDB::bind_method(D_METHOD("get_link_activation_mode"), &RichTextEdit::get_link_activation_mode);
	ClassDB::bind_method(D_METHOD("set_bbcode_text", "text"), &RichTextEdit::set_bbcode_text);
	ClassDB::bind_method(D_METHOD("get_bbcode_text"), &RichTextEdit::get_bbcode_text);
	ClassDB::bind_method(D_METHOD("set_bold"), &RichTextEdit::set_bold);
	ClassDB::bind_method(D_METHOD("clear_bold"), &RichTextEdit::clear_bold);
	ClassDB::bind_method(D_METHOD("toggle_bold"), &RichTextEdit::toggle_bold);
	ClassDB::bind_method(D_METHOD("set_italic"), &RichTextEdit::set_italic);
	ClassDB::bind_method(D_METHOD("clear_italic"), &RichTextEdit::clear_italic);
	ClassDB::bind_method(D_METHOD("toggle_italic"), &RichTextEdit::toggle_italic);
	ClassDB::bind_method(D_METHOD("set_underline"), &RichTextEdit::set_underline);
	ClassDB::bind_method(D_METHOD("clear_underline"), &RichTextEdit::clear_underline);
	ClassDB::bind_method(D_METHOD("toggle_underline"), &RichTextEdit::toggle_underline);
	ClassDB::bind_method(D_METHOD("set_strikethrough"), &RichTextEdit::set_strikethrough);
	ClassDB::bind_method(D_METHOD("clear_strikethrough"), &RichTextEdit::clear_strikethrough);
	ClassDB::bind_method(D_METHOD("toggle_strikethrough"), &RichTextEdit::toggle_strikethrough);
	ClassDB::bind_method(D_METHOD("set_selection_color", "color"), &RichTextEdit::set_selection_color);
	ClassDB::bind_method(D_METHOD("clear_selection_color"), &RichTextEdit::clear_selection_color);
	ClassDB::bind_method(D_METHOD("set_selection_bg_color", "color"), &RichTextEdit::set_selection_bg_color);
	ClassDB::bind_method(D_METHOD("clear_selection_bg_color"), &RichTextEdit::clear_selection_bg_color);
	ClassDB::bind_method(D_METHOD("set_selection_outline_color", "color"), &RichTextEdit::set_selection_outline_color);
	ClassDB::bind_method(D_METHOD("clear_selection_outline_color"), &RichTextEdit::clear_selection_outline_color);
	ClassDB::bind_method(D_METHOD("set_selection_outline_size", "size"), &RichTextEdit::set_selection_outline_size);
	ClassDB::bind_method(D_METHOD("clear_selection_outline_size"), &RichTextEdit::clear_selection_outline_size);
	ClassDB::bind_method(D_METHOD("begin_selection_color_preview"), &RichTextEdit::begin_selection_color_preview);
	ClassDB::bind_method(D_METHOD("preview_selection_color", "color"), &RichTextEdit::preview_selection_color);
	ClassDB::bind_method(D_METHOD("end_selection_color_preview", "commit"), &RichTextEdit::end_selection_color_preview);
	ClassDB::bind_method(D_METHOD("set_selection_font_size", "font_size"), &RichTextEdit::set_selection_font_size);
	ClassDB::bind_method(D_METHOD("clear_selection_font_size"), &RichTextEdit::clear_selection_font_size);
	ClassDB::bind_method(D_METHOD("set_selection_font", "font"), &RichTextEdit::set_selection_font);
	ClassDB::bind_method(D_METHOD("clear_selection_font"), &RichTextEdit::clear_selection_font);
	ClassDB::bind_method(D_METHOD("set_selection_url", "url"), &RichTextEdit::set_selection_url);
	ClassDB::bind_method(D_METHOD("clear_selection_url"), &RichTextEdit::clear_selection_url);
	ClassDB::bind_method(D_METHOD("set_selection_url_tooltip", "tooltip"), &RichTextEdit::set_selection_url_tooltip);
	ClassDB::bind_method(D_METHOD("clear_selection_url_tooltip"), &RichTextEdit::clear_selection_url_tooltip);
	ClassDB::bind_method(D_METHOD("set_selection_url_visited"), &RichTextEdit::set_selection_url_visited);
	ClassDB::bind_method(D_METHOD("clear_selection_url_visited"), &RichTextEdit::clear_selection_url_visited);
	ClassDB::bind_method(D_METHOD("toggle_code"), &RichTextEdit::toggle_code);
	ClassDB::bind_method(D_METHOD("insert_image", "source", "width", "height", "alt"), &RichTextEdit::insert_image, DEFVAL(-1), DEFVAL(-1), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("set_selection_image_size", "width", "height"), &RichTextEdit::set_selection_image_size);
	ClassDB::bind_method(D_METHOD("set_alignment", "alignment"), &RichTextEdit::set_alignment);
	ClassDB::bind_method(D_METHOD("toggle_quote"), &RichTextEdit::toggle_quote);
	ClassDB::bind_method(D_METHOD("set_quote"), &RichTextEdit::set_quote);
	ClassDB::bind_method(D_METHOD("clear_quote"), &RichTextEdit::clear_quote);
	ClassDB::bind_method(D_METHOD("toggle_unordered_list"), &RichTextEdit::toggle_unordered_list);
	ClassDB::bind_method(D_METHOD("set_unordered_list"), &RichTextEdit::set_unordered_list);
	ClassDB::bind_method(D_METHOD("clear_unordered_list"), &RichTextEdit::clear_unordered_list);
	ClassDB::bind_method(D_METHOD("toggle_ordered_list"), &RichTextEdit::toggle_ordered_list);
	ClassDB::bind_method(D_METHOD("set_ordered_list"), &RichTextEdit::set_ordered_list);
	ClassDB::bind_method(D_METHOD("clear_ordered_list"), &RichTextEdit::clear_ordered_list);
	ClassDB::bind_method(D_METHOD("get_current_font_size"), &RichTextEdit::get_current_font_size);
	ClassDB::bind_method(D_METHOD("increase_indent"), &RichTextEdit::increase_indent);
	ClassDB::bind_method(D_METHOD("decrease_indent"), &RichTextEdit::decrease_indent);

	BIND_ENUM_CONSTANT(LINK_ACTIVATION_AUTO);
	BIND_ENUM_CONSTANT(LINK_ACTIVATION_CTRL_CLICK);
	BIND_ENUM_CONSTANT(LINK_ACTIVATION_CLICK);
	BIND_ENUM_CONSTANT(LINK_ACTIVATION_DISABLED);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bbcode_enabled"), "set_use_bbcode", "is_using_bbcode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "link_activation_mode", PROPERTY_HINT_ENUM, "Auto,Ctrl Click,Click,Disabled"), "set_link_activation_mode", "get_link_activation_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bbcode_text", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_bbcode_text", "get_bbcode_text");

	ADD_SIGNAL(MethodInfo("meta_clicked", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("meta_hover_started", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("meta_hover_ended", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("text_style_changed"));

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, RichTextEdit, tooltip_panel);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, RichTextEdit, tooltip_font_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT, RichTextEdit, tooltip_font);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT_SIZE, RichTextEdit, tooltip_font_size);

	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_COLOR, get_class_static(), "link_color", "link_color",
			[](Node *p_instance, const StringName &, const StringName &) {
				RichTextEdit *rich_text_edit = Object::cast_to<RichTextEdit>(p_instance);
				rich_text_edit->_refresh_style_rendering();
			});
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_COLOR, get_class_static(), "link_hover_color", "link_hover_color",
			[](Node *p_instance, const StringName &, const StringName &) {
				RichTextEdit *rich_text_edit = Object::cast_to<RichTextEdit>(p_instance);
				rich_text_edit->_refresh_style_rendering();
			});
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_COLOR, get_class_static(), "link_visited_color", "link_visited_color",
			[](Node *p_instance, const StringName &, const StringName &) {
				RichTextEdit *rich_text_edit = Object::cast_to<RichTextEdit>(p_instance);
				rich_text_edit->_refresh_style_rendering();
			});
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_COLOR, get_class_static(), "link_active_color", "link_active_color",
			[](Node *p_instance, const StringName &, const StringName &) {
				RichTextEdit *rich_text_edit = Object::cast_to<RichTextEdit>(p_instance);
				rich_text_edit->_refresh_style_rendering();
			});
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_COLOR, get_class_static(), "quote_color", "quote_color",
			[](Node *p_instance, const StringName &, const StringName &) {
				RichTextEdit *rich_text_edit = Object::cast_to<RichTextEdit>(p_instance);
				rich_text_edit->_refresh_style_rendering();
			});
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_COLOR, get_class_static(), "quote_border_color", "quote_border_color",
			[](Node *p_instance, const StringName &, const StringName &) {
				RichTextEdit *rich_text_edit = Object::cast_to<RichTextEdit>(p_instance);
				rich_text_edit->_refresh_style_rendering();
			});
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_CONSTANT, get_class_static(), "quote_margin_top", "quote_margin_top",
			[](Node *p_instance, const StringName &, const StringName &) {
				RichTextEdit *rich_text_edit = Object::cast_to<RichTextEdit>(p_instance);
				rich_text_edit->_refresh_style_rendering();
			});
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_CONSTANT, get_class_static(), "quote_margin_bottom", "quote_margin_bottom",
			[](Node *p_instance, const StringName &, const StringName &) {
				RichTextEdit *rich_text_edit = Object::cast_to<RichTextEdit>(p_instance);
				rich_text_edit->_refresh_style_rendering();
			});
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_CONSTANT, get_class_static(), "quote_border_width", "quote_border_width",
			[](Node *p_instance, const StringName &, const StringName &) {
				RichTextEdit *rich_text_edit = Object::cast_to<RichTextEdit>(p_instance);
				rich_text_edit->_refresh_style_rendering();
			});
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_CONSTANT, get_class_static(), "quote_padding", "quote_padding",
			[](Node *p_instance, const StringName &, const StringName &) {
				RichTextEdit *rich_text_edit = Object::cast_to<RichTextEdit>(p_instance);
				rich_text_edit->_refresh_style_rendering();
			});
}

RichTextEdit::RichTextEdit() {
	set_style_parser(callable_mp(this, &RichTextEdit::_get_line_style_spans));
	set_deselect_on_focus_loss_enabled(false);

	source_text = TextEdit::get_text();
	tracked_text = TextEdit::get_text();
	connect(SceneStringName(text_changed), callable_mp(this, &RichTextEdit::_update_bbcode_from_text));
	connect("caret_changed", callable_mp(this, &RichTextEdit::_caret_style_context_changed));
}
