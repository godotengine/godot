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
#include "core/string/string_builder.h"
#include "core/templates/hashfuncs.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup.h"
#include "scene/gui/spin_box.h"
#include "scene/resources/style_box.h"
#include "scene/resources/style_box_flat.h"
#include "scene/theme/theme_db.h"
#include "servers/display/display_server.h"
#include "servers/rendering/rendering_server.h"

// The process-wide rich clipboard: the copied range as a self-contained
// document (image textures are reloaded on paste, so none are retained here)
// and the plain text the system clipboard received with it.
static RichTextDocument rich_text_clipboard;
static bool rich_text_clipboard_valid = false;
static String rich_text_clipboard_plain_text;

// Undo keeps the formatting of this much recently removed text.
static constexpr int MAX_REMOVED_FRAGMENTS = 1024;
static constexpr int MAX_REMOVED_FRAGMENT_CHARS = 8 * 1024 * 1024;

// Theme items that feed the per-run render information.
static const char *STYLE_THEME_FONTS[] = { "bold_font", "italics_font", "bold_italics_font", "mono_font" };
static const char *STYLE_THEME_COLORS[] = { "link_color", "link_hover_color", "link_visited_color", "link_active_color", "quote_color", "quote_border_color", "divider_color" };
static const char *STYLE_THEME_CONSTANTS[] = { "quote_margin_top", "quote_margin_bottom", "quote_border_width", "quote_padding", "divider_margin_top", "divider_margin_bottom", "divider_width" };

bool RichTextEdit::_is_url_auto_open_allowed(const String &p_url) {
	const String url = p_url.strip_edges().to_lower();
	return url.begins_with("http://") || url.begins_with("https://");
}

/* Style table. */

uint32_t RichTextEdit::_hash_style(const TextStyle &p_style) {
	// Only fields that take part in Style::operator== contribute, so equal
	// styles always share a hash.
	uint32_t flags = 0;
	const bool bits[] = { p_style.bold, p_style.italic, p_style.has_underline, p_style.underline, p_style.strikethrough, p_style.overline, p_style.code,
		p_style.has_color, p_style.has_bg_color, p_style.has_fg_color, p_style.has_outline_color, p_style.has_outline_size, p_style.has_url,
		p_style.url_visited, p_style.list_capitalize };
	for (uint32_t i = 0; i < std_size(bits); i++) {
		flags |= (bits[i] ? 1u : 0u) << i;
	}
	uint32_t h = hash_murmur3_one_32(flags);
	if (p_style.has_color) {
		h = hash_murmur3_one_32(p_style.color.to_rgba32(), h);
	}
	if (p_style.has_bg_color) {
		h = hash_murmur3_one_32(p_style.bg_color.to_rgba32(), h);
	}
	if (p_style.has_fg_color) {
		h = hash_murmur3_one_32(p_style.fg_color.to_rgba32(), h);
	}
	if (p_style.has_outline_color) {
		h = hash_murmur3_one_32(p_style.outline_color.to_rgba32(), h);
	}
	if (p_style.has_outline_size) {
		h = hash_murmur3_one_32(p_style.outline_size, h);
	}
	h = hash_murmur3_one_32(p_style.font_size, h);
	h = hash_murmur3_one_32(p_style.alignment, h);
	h = hash_murmur3_one_32(p_style.indent_level, h);
	h = hash_murmur3_one_32(p_style.list_type, h);
	h = hash_murmur3_one_32(p_style.list_start, h);
	h = hash_murmur3_one_32(p_style.block_group, h);
	h = hash_murmur3_one_32(p_style.font.hash(), h);
	h = hash_murmur3_one_32(p_style.url.hash(), h);
	h = hash_murmur3_one_32(p_style.url_tooltip.hash(), h);
	h = hash_murmur3_one_32(p_style.language.hash(), h);
	h = hash_murmur3_one_32(p_style.block_tag.hash(), h);
	for (const String &tag : p_style.preserved_tags) {
		h = hash_murmur3_one_32(tag.hash(), h);
	}
	return hash_fmix32(h);
}

int RichTextEdit::_intern_style(const TextStyle &p_style) {
	if (p_style.is_default()) {
		return 0;
	}
	const uint32_t hash = _hash_style(p_style);
	LocalVector<int> *ids = style_ids_by_hash.getptr(hash);
	if (ids != nullptr) {
		for (int id : *ids) {
			if (styles[id] == p_style) {
				return id;
			}
		}
	}
	const int id = styles.size();
	styles.push_back(p_style);
	style_render_infos.push_back(Dictionary());
	style_render_info_valid.push_back(false);
	if (ids == nullptr) {
		ids = &style_ids_by_hash.insert(hash, LocalVector<int>())->value;
	}
	ids->push_back(id);
	return id;
}

void RichTextEdit::_clear_style_render_infos() {
	for (uint32_t i = 0; i < style_render_info_valid.size(); i++) {
		style_render_info_valid[i] = false;
		style_render_infos[i] = Dictionary();
	}
}

Dictionary RichTextEdit::_get_style_render_info(int p_style) const {
	if (style_render_info_valid[p_style]) {
		return style_render_infos[p_style];
	}
	const TextStyle &style = styles[p_style];
	Ref<Font> font;
	if (!style.font.is_empty()) {
		// Applications may map a family resource to its real or verified
		// synthesized face. Keep formatting flags in the native document/history.
		if (style_font_resolver.is_valid()) {
			font = style_font_resolver.call(style.font, style.bold, style.italic);
		}
		if (font.is_null()) {
			font = ResourceLoader::load(style.font, "Font");
		}
	} else if (style.code) {
		font = get_theme_font(SNAME("mono_font"));
	} else if (style.bold && style.italic) {
		font = get_theme_font(SNAME("bold_italics_font"));
	} else if (style.bold) {
		font = get_theme_font(SNAME("bold_font"));
	} else if (style.italic) {
		font = get_theme_font(SNAME("italics_font"));
	}

	Dictionary info;
	if (style.has_color) {
		info["color"] = style.color;
	} else if (!style.url.is_empty()) {
		if (active_meta == style.url) {
			info["color"] = get_theme_color(SNAME("link_active_color"), SNAME("RichTextEdit"));
		} else if (meta_hovering && current_meta == style.url) {
			info["color"] = get_theme_color(SNAME("link_hover_color"), SNAME("RichTextEdit"));
		} else if (style.url_visited || visited_links.has(style.url)) {
			info["color"] = get_theme_color(SNAME("link_visited_color"), SNAME("RichTextEdit"));
		} else {
			info["color"] = get_theme_color(SNAME("link_color"), SNAME("RichTextEdit"));
		}
	}
	if (style.has_bg_color) {
		info["bg_color"] = style.bg_color;
	}
	if (style.has_fg_color) {
		info["fg_color"] = style.fg_color;
	}
	if (style.has_outline_color) {
		info["outline_color"] = style.outline_color;
	}
	if (style.has_outline_size) {
		info["outline_size"] = style.outline_size;
	}
	const bool underline = style.has_underline ? style.underline : !style.url.is_empty();
	if (underline) {
		info["underline"] = true;
	}
	if (!style.url.is_empty()) {
		info["url"] = style.url;
	}
	if (style.strikethrough) {
		info["strikethrough"] = true;
	}
	if (style.overline) {
		info["overline"] = true;
	}
	for (const KeyValue<Variant, Variant> &paint : decoration_paint) {
		info[paint.key] = paint.value;
	}
	if (font.is_valid()) {
		info["font"] = font;
	}
	if (style.font_size > 0) {
		info["font_size"] = style.font_size;
	}
	if (style.alignment >= 0) {
		info["alignment"] = style.alignment;
	}
	if (style.indent_level > 0) {
		info["indent_level"] = style.indent_level;
	}
	if (style.list_type > 0) {
		info["list_type"] = style.list_type;
		if (style.list_start > 0) {
			info["list_start"] = style.list_start;
		}
		info["list_capitalize"] = style.list_capitalize;
	}
	if (style.block_tag == "quote" || style.block_tag.begins_with("quote ")) {
		info["quote"] = true;
		info["quote_group"] = style.block_group;
		info["quote_margin_top"] = get_theme_constant(SNAME("quote_margin_top"), SNAME("RichTextEdit"));
		info["quote_margin_bottom"] = get_theme_constant(SNAME("quote_margin_bottom"), SNAME("RichTextEdit"));
		info["quote_border_color"] = get_theme_color(SNAME("quote_border_color"), SNAME("RichTextEdit"));
		info["quote_border_width"] = get_theme_constant(SNAME("quote_border_width"), SNAME("RichTextEdit"));
		info["quote_padding"] = get_theme_constant(SNAME("quote_padding"), SNAME("RichTextEdit"));
		if (!style.has_color) {
			info["color"] = get_theme_color(SNAME("quote_color"), SNAME("RichTextEdit"));
		}
	}
	style_render_infos[p_style] = info;
	style_render_info_valid[p_style] = true;
	return info;
}

/* Model. */

void RichTextEdit::_set_paragraph_line_height(Paragraph &r_paragraph, const String &p_line_height) {
	r_paragraph.line_height = p_line_height;
	r_paragraph.line_height_kind = 0;
	r_paragraph.line_height_value = 0.0;
	const String value = p_line_height.strip_edges().to_lower();
	if (value.is_empty() || value == "normal" || !RichTextDocument::is_valid_line_height(value)) {
		return;
	}
	if (value.ends_with("px")) {
		r_paragraph.line_height_kind = 1;
		r_paragraph.line_height_value = value.left(value.length() - 2).strip_edges().to_float();
	} else if (value.ends_with("%")) {
		r_paragraph.line_height_kind = 2;
		r_paragraph.line_height_value = value.left(value.length() - 1).strip_edges().to_float();
	} else {
		r_paragraph.line_height_kind = 3;
		r_paragraph.line_height_value = value.to_float();
	}
}

void RichTextEdit::_reset_model() {
	paragraphs.clear();
	paragraphs.resize(MAX(get_line_count(), 1));
	styles.clear();
	styles.push_back(TextStyle());
	style_ids_by_hash.clear();
	style_render_infos.clear();
	style_render_infos.push_back(Dictionary());
	style_render_info_valid.clear();
	style_render_info_valid.push_back(true);
	removed_fragments.clear();
	removed_fragments_chars = 0;
	pending_fragment_active = false;
	pending_fragment.clear();
	list_numbers_valid = 0;
	model_revision++;
}

void RichTextEdit::_ensure_model_size() {
	// TextEdit::clear() and other base-class paths can replace the lines
	// without an edit notification; fall back to plain paragraphs then.
	const int line_count = get_line_count();
	if ((int)paragraphs.size() != line_count) {
		paragraphs.clear();
		paragraphs.resize(line_count);
		list_numbers_valid = 0;
		model_revision++;
	}
}

int RichTextEdit::_get_style_id_at(int p_line, int p_column) const {
	if (p_line < 0 || p_line >= (int)paragraphs.size() || p_column < 0) {
		return 0;
	}
	const Paragraph &paragraph = paragraphs[p_line];
	if (p_column >= _get_line_length(p_line)) {
		// The paragraph break; the last paragraph has none.
		return p_line < (int)paragraphs.size() - 1 ? paragraph.eol_style : 0;
	}
	int low = 0;
	int high = (int)paragraph.runs.size() - 1;
	while (low <= high) {
		const int mid = (low + high) / 2;
		const StyleRun &run = paragraph.runs[mid];
		if (p_column < run.from) {
			high = mid - 1;
		} else if (p_column >= run.to) {
			low = mid + 1;
		} else {
			return run.style;
		}
	}
	return 0;
}

int RichTextEdit::_get_insertion_style_id(int p_line, int p_column) const {
	// Typed text continues the character before it, except at the start of a
	// paragraph, where it takes the style of what it is inserted in front of.
	const int length = _get_line_length(p_line);
	if (p_column == 0 && (length > 0 || p_line < (int)paragraphs.size() - 1)) {
		return _get_style_id_at(p_line, 0);
	}
	if (p_column > 0) {
		return _get_style_id_at(p_line, p_column - 1);
	}
	if (p_line > 0) {
		return paragraphs[p_line - 1].eol_style;
	}
	return 0;
}

void RichTextEdit::_normalize_runs(LocalVector<StyleRun> &r_runs) {
	struct RunCompare {
		_FORCE_INLINE_ bool operator()(const StyleRun &p_a, const StyleRun &p_b) const {
			return p_a.from < p_b.from;
		}
	};
	r_runs.sort_custom<RunCompare>();
	LocalVector<StyleRun> merged;
	merged.reserve(r_runs.size());
	for (const StyleRun &run : r_runs) {
		if (run.from >= run.to || run.style == 0) {
			continue;
		}
		if (!merged.is_empty() && merged[merged.size() - 1].to == run.from && merged[merged.size() - 1].style == run.style) {
			merged[merged.size() - 1].to = run.to;
		} else {
			merged.push_back(run);
		}
	}
	r_runs = merged;
}

void RichTextEdit::_set_runs_range(LocalVector<StyleRun> &r_runs, int p_from, int p_to, int p_style) {
	if (p_from >= p_to) {
		return;
	}
	LocalVector<StyleRun> updated;
	updated.reserve(r_runs.size() + 2);
	for (const StyleRun &run : r_runs) {
		if (run.to <= p_from || run.from >= p_to) {
			updated.push_back(run);
			continue;
		}
		if (run.from < p_from) {
			updated.push_back({ run.from, p_from, run.style });
		}
		if (run.to > p_to) {
			updated.push_back({ p_to, run.to, run.style });
		}
	}
	if (p_style != 0) {
		updated.push_back({ p_from, p_to, p_style });
	}
	r_runs = updated;
	_normalize_runs(r_runs);
}

void RichTextEdit::_model_changed(int p_from_line, int p_to_line, bool p_invalidate) {
	model_revision++;
	list_numbers_valid = MIN(list_numbers_valid, MAX(p_from_line, 0));
	if (p_invalidate && get_line_count() > 0) {
		_invalidate_lines(p_from_line, p_to_line);
	}
	_style_changed();
}

int RichTextEdit::_get_line_start(int p_line) const {
	if (p_line >= get_line_count()) {
		return _get_total_character_count() + 1;
	}
	return _get_line_character_offset(MAX(p_line, 0));
}

void RichTextEdit::_offset_to_line_column(int p_offset, int &r_line, int &r_column) const {
	int low = 0;
	int high = get_line_count() - 1;
	while (low < high) {
		const int mid = (low + high + 1) / 2;
		if (_get_line_character_offset(mid) <= p_offset) {
			low = mid;
		} else {
			high = mid - 1;
		}
	}
	r_line = low;
	r_column = CLAMP(p_offset - _get_line_character_offset(low), 0, _get_line_length(low));
}

int RichTextEdit::_get_caret_offset() const {
	return _get_line_start(get_caret_line()) + get_caret_column();
}

bool RichTextEdit::_get_selection_range(int &r_from_line, int &r_from_column, int &r_to_line, int &r_to_column) const {
	if (!has_selection()) {
		return false;
	}
	r_from_line = get_selection_from_line();
	r_from_column = get_selection_from_column();
	r_to_line = get_selection_to_line();
	r_to_column = get_selection_to_column();
	return r_from_line != r_to_line || r_from_column != r_to_column;
}

bool RichTextEdit::_get_selection_offsets(int &r_from, int &r_to) const {
	int from_line = 0;
	int from_column = 0;
	int to_line = 0;
	int to_column = 0;
	if (!_get_selection_range(from_line, from_column, to_line, to_column)) {
		return false;
	}
	r_from = _get_line_start(from_line) + from_column;
	r_to = _get_line_start(to_line) + to_column;
	return true;
}

void RichTextEdit::_get_selected_line_range(int &r_from_line, int &r_to_line) const {
	r_from_line = has_selection() ? get_selection_from_line() : get_caret_line();
	r_to_line = has_selection() ? get_selection_to_line() : get_caret_line();
}

RichTextEdit::TextStyle RichTextEdit::_get_style_at_offset(int p_offset) const {
	int line = 0;
	int column = 0;
	_offset_to_line_column(p_offset, line, column);
	return styles[_get_style_id_at(line, column)];
}

/* Fragments. */

String RichTextEdit::_get_text_range(int p_from_line, int p_from_column, int p_to_line, int p_to_column) const {
	StringBuilder builder;
	for (int line = p_from_line; line <= p_to_line; line++) {
		const String line_text = get_line(line);
		const int from = line == p_from_line ? p_from_column : 0;
		const int to = line == p_to_line ? p_to_column : line_text.length();
		if (line > p_from_line) {
			builder.append("\n");
		}
		builder.append(line_text.substr(from, to - from));
	}
	return builder.as_string();
}

LocalVector<RichTextEdit::Paragraph> RichTextEdit::_extract_slices(int p_from_line, int p_from_column, int p_to_line, int p_to_column) const {
	LocalVector<Paragraph> slices;
	slices.resize(p_to_line - p_from_line + 1);
	for (int line = p_from_line; line <= p_to_line; line++) {
		const Paragraph &paragraph = paragraphs[line];
		Paragraph &slice = slices[line - p_from_line];
		const int from = line == p_from_line ? p_from_column : 0;
		const int to = line == p_to_line ? p_to_column : _get_line_length(line);
		for (const StyleRun &run : paragraph.runs) {
			const int run_from = MAX(run.from, from);
			const int run_to = MIN(run.to, to);
			if (run_from < run_to) {
				slice.runs.push_back({ run_from - from, run_to - from, run.style });
			}
		}
		for (const InlineObject &object : paragraph.objects) {
			if (object.column >= from && object.column < to) {
				InlineObject copy = object;
				copy.column -= from;
				slice.objects.push_back(copy);
			}
		}
		slice.eol_style = line < p_to_line ? paragraph.eol_style : 0;
		_set_paragraph_line_height(slice, paragraph.line_height);
	}
	return slices;
}

bool RichTextEdit::_slices_have_formatting(const LocalVector<Paragraph> &p_slices) const {
	for (const Paragraph &slice : p_slices) {
		if (!slice.runs.is_empty() || slice.eol_style != 0 || !slice.objects.is_empty() || !slice.line_height.is_empty()) {
			return true;
		}
	}
	return false;
}

void RichTextEdit::_insert_slices(int p_line, int p_column, const LocalVector<Paragraph> &p_slices, bool p_restore_line_heights) {
	ERR_FAIL_COND(p_slices.is_empty());
	ERR_FAIL_INDEX(p_line, (int)paragraphs.size());

	// Split the target paragraph at the insertion column.
	Paragraph &target = paragraphs[p_line];
	LocalVector<StyleRun> left_runs;
	LocalVector<StyleRun> right_runs;
	for (const StyleRun &run : target.runs) {
		if (run.to <= p_column) {
			left_runs.push_back(run);
		} else if (run.from >= p_column) {
			right_runs.push_back({ run.from - p_column, run.to - p_column, run.style });
		} else {
			left_runs.push_back({ run.from, p_column, run.style });
			right_runs.push_back({ 0, run.to - p_column, run.style });
		}
	}
	LocalVector<InlineObject> left_objects;
	LocalVector<InlineObject> right_objects;
	for (const InlineObject &object : target.objects) {
		if (object.column < p_column) {
			left_objects.push_back(object);
		} else {
			InlineObject moved = object;
			moved.column -= p_column;
			right_objects.push_back(moved);
		}
	}
	const int original_eol = target.eol_style;
	const String original_line_height = target.line_height;

	auto append = [this](Paragraph &r_to, const Paragraph &p_from, int p_shift) {
		for (const StyleRun &run : p_from.runs) {
			r_to.runs.push_back({ run.from + p_shift, run.to + p_shift, run.style });
		}
		for (const InlineObject &object : p_from.objects) {
			InlineObject moved = object;
			moved.column += p_shift;
			if (moved.id == 0) {
				moved.id = next_object_id++;
			}
			r_to.objects.push_back(moved);
		}
	};
	auto append_split = [](Paragraph &r_to, const LocalVector<StyleRun> &p_runs, const LocalVector<InlineObject> &p_objects, int p_shift) {
		for (const StyleRun &run : p_runs) {
			r_to.runs.push_back({ run.from + p_shift, run.to + p_shift, run.style });
		}
		for (const InlineObject &object : p_objects) {
			InlineObject moved = object;
			moved.column += p_shift;
			r_to.objects.push_back(moved);
		}
	};

	const int new_lines = p_slices.size() - 1;
	if (new_lines == 0) {
		Paragraph merged;
		append_split(merged, left_runs, left_objects, 0);
		append(merged, p_slices[0], p_column);
		append_split(merged, right_runs, right_objects, p_column + p_slices[0].length);
		merged.eol_style = original_eol;
		_set_paragraph_line_height(merged, original_line_height);
		_normalize_runs(merged.runs);
		paragraphs[p_line] = merged;
		return;
	}

	Paragraph first;
	append_split(first, left_runs, left_objects, 0);
	append(first, p_slices[0], p_column);
	first.eol_style = p_slices[0].eol_style;
	Paragraph last;
	append(last, p_slices[new_lines], 0);
	append_split(last, right_runs, right_objects, p_slices[new_lines].length);
	last.eol_style = original_eol;

	// Paragraph metadata travels with the paragraph's first character: text
	// inserted at the start of a paragraph pushes it down to the last line.
	if (p_restore_line_heights) {
		_set_paragraph_line_height(first, p_slices[0].line_height);
		_set_paragraph_line_height(last, p_slices[new_lines].line_height);
	} else if (p_column == 0) {
		_set_paragraph_line_height(first, p_slices[0].line_height);
		_set_paragraph_line_height(last, original_line_height);
	} else {
		_set_paragraph_line_height(first, original_line_height);
		_set_paragraph_line_height(last, p_slices[new_lines].line_height);
	}
	_normalize_runs(first.runs);
	_normalize_runs(last.runs);

	// Open a gap of new_lines paragraphs after the target in one move.
	const int old_size = paragraphs.size();
	paragraphs.resize(old_size + new_lines);
	for (int i = old_size - 1; i > p_line; i--) {
		paragraphs[i + new_lines] = std::move(paragraphs[i]);
	}
	paragraphs[p_line] = first;
	for (int i = 1; i < new_lines; i++) {
		Paragraph middle;
		append(middle, p_slices[i], 0);
		middle.eol_style = p_slices[i].eol_style;
		_set_paragraph_line_height(middle, p_slices[i].line_height);
		_normalize_runs(middle.runs);
		paragraphs[p_line + i] = middle;
	}
	paragraphs[p_line + new_lines] = last;
}

void RichTextEdit::_remove_model_range(int p_from_line, int p_from_column, int p_to_line, int p_to_column) {
	Paragraph &first = paragraphs[p_from_line];
	const Paragraph &last = paragraphs[p_to_line];
	const int removed = p_from_line == p_to_line ? p_to_column - p_from_column : 0;

	Paragraph merged;
	for (const StyleRun &run : first.runs) {
		if (run.from < p_from_column) {
			merged.runs.push_back({ run.from, MIN(run.to, p_from_column), run.style });
		}
	}
	for (const InlineObject &object : first.objects) {
		if (object.column < p_from_column) {
			merged.objects.push_back(object);
		}
	}
	const int shift = p_from_line == p_to_line ? -removed : p_from_column - p_to_column;
	for (const StyleRun &run : last.runs) {
		if (run.to > p_to_column) {
			merged.runs.push_back({ MAX(run.from, p_to_column) + shift, run.to + shift, run.style });
		}
	}
	for (const InlineObject &object : last.objects) {
		if (object.column >= p_to_column) {
			InlineObject moved = object;
			moved.column += shift;
			merged.objects.push_back(moved);
		}
	}
	merged.eol_style = last.eol_style;
	// Paragraph metadata travels with the paragraph's first character, as on
	// insertion: removing from the start of a line leaves the joined text
	// starting with the last paragraph's text.
	_set_paragraph_line_height(merged, p_from_column > 0 ? first.line_height : last.line_height);
	_normalize_runs(merged.runs);
	paragraphs[p_from_line] = merged;

	const int removed_lines = p_to_line - p_from_line;
	if (removed_lines > 0) {
		const int old_size = paragraphs.size();
		for (int i = p_to_line + 1; i < old_size; i++) {
			paragraphs[i - removed_lines] = std::move(paragraphs[i]);
		}
		paragraphs.resize(old_size - removed_lines);
	}
}

void RichTextEdit::_text_inserting(int p_line, int p_column, const String &p_text) {
	if (setting_bbcode_text) {
		return;
	}
	_ensure_model_size();
	ERR_FAIL_INDEX(p_line, (int)paragraphs.size());

	const Vector<String> pieces = p_text.split("\n");
	LocalVector<Paragraph> slices;
	bool restore_line_heights = false;
	bool have_slices = false;
	bool inherit_style = false;
	const bool undo_redo = _is_applying_undo_redo();
	if (undo_redo) {
		// TextEdit puts back exactly the text an edit removed: restore its
		// formatting. Text without a record was unformatted.
		for (int i = (int)removed_fragments.size() - 1; i >= 0; i--) {
			RemovedFragment &fragment = removed_fragments[i];
			if (fragment.line == p_line && fragment.column == p_column && fragment.text == p_text) {
				slices = fragment.slices;
				removed_fragments_chars -= fragment.text.length();
				removed_fragments.remove_at(i);
				have_slices = true;
				restore_line_heights = true;
				break;
			}
		}
	} else if (pending_fragment_active && p_text == pending_fragment_text) {
		slices = pending_fragment;
		inherit_style = pending_fragment_inherit_style;
		pending_fragment_active = false;
		pending_fragment.clear();
		have_slices = slices.size() == (uint32_t)pieces.size();
	}

	int style = 0;
	if (!undo_redo) {
		style = typing_style_override ? _intern_style(typing_style) : _get_insertion_style_id(p_line, p_column);
	}
	if (!have_slices) {
		slices.clear();
		slices.resize(pieces.size());
	}
	for (int i = 0; i < pieces.size(); i++) {
		Paragraph &slice = slices[i];
		slice.length = pieces[i].length();
		// Typed text, and inserted objects, take the style typing would use.
		if ((!have_slices || inherit_style) && style != 0) {
			if (slice.length > 0 && slice.runs.is_empty()) {
				slice.runs.push_back({ 0, slice.length, style });
			}
			if (i < pieces.size() - 1 && slice.eol_style == 0) {
				slice.eol_style = style;
			}
		}
	}

	// An unstyled paragraph break typed in front of a block paragraph mark
	// leaves the new empty paragraph plain instead of extending the block.
	bool plain_moved_mark = false;
	if (!have_slices && !undo_redo && p_text == "\n" && styles[style].block_tag.is_empty() && p_line < (int)paragraphs.size() - 1 && p_column == _get_line_length(p_line)) {
		const int mark = paragraphs[p_line].eol_style;
		const int before = p_column > 0 ? _get_style_id_at(p_line, p_column - 1) : (p_line > 0 ? paragraphs[p_line - 1].eol_style : 0);
		plain_moved_mark = mark != 0 && !styles[mark].block_tag.is_empty() && before == mark;
	}

	_insert_slices(p_line, p_column, slices, restore_line_heights);
	if (plain_moved_mark) {
		paragraphs[p_line + 1].eol_style = 0;
	}
	typing_style_override = false;
	_model_changed(p_line, p_line + pieces.size() - 1, false);
}

void RichTextEdit::_text_removing(int p_from_line, int p_from_column, int p_to_line, int p_to_column) {
	if (setting_bbcode_text) {
		return;
	}
	_ensure_model_size();
	ERR_FAIL_INDEX(p_to_line, (int)paragraphs.size());

	// Remember the removed formatting for undo and redo. Plain removals are
	// recorded too, so an older record at the same place cannot be matched.
	RemovedFragment fragment;
	fragment.line = p_from_line;
	fragment.column = p_from_column;
	fragment.text = _get_text_range(p_from_line, p_from_column, p_to_line, p_to_column);
	fragment.slices = _extract_slices(p_from_line, p_from_column, p_to_line, p_to_column);
	for (int i = 0; i < (int)fragment.slices.size(); i++) {
		const int from = i == 0 ? p_from_column : 0;
		const int to = p_from_line + i == p_to_line ? p_to_column : _get_line_length(p_from_line + i);
		fragment.slices[i].length = to - from;
	}
	bool removes_selected_object = false;
	for (const Paragraph &slice : fragment.slices) {
		for (const InlineObject &object : slice.objects) {
			removes_selected_object = removes_selected_object || object.id == selected_object_id;
		}
	}
	removed_fragments_chars += fragment.text.length();
	removed_fragments.push_back(fragment);
	while (!removed_fragments.is_empty() && ((int)removed_fragments.size() > MAX_REMOVED_FRAGMENTS || removed_fragments_chars > MAX_REMOVED_FRAGMENT_CHARS)) {
		removed_fragments_chars -= removed_fragments[0].text.length();
		removed_fragments.remove_at(0);
	}

	_remove_model_range(p_from_line, p_from_column, p_to_line, p_to_column);
	_model_changed(p_from_line, p_from_line, false);
	if (removes_selected_object) {
		_clear_selected_image();
	}
}

/* Documents. */

RichTextDocument RichTextEdit::_make_range_document(int p_from_line, int p_from_column, int p_to_line, int p_to_column) const {
	RichTextDocument document;
	document.text = _get_text_range(p_from_line, p_from_column, p_to_line, p_to_column);

	// Runs of equal style that touch across paragraph breaks become one span.
	RichTextDocument::StyleSpan open;
	int open_style = 0;
	auto flush = [&]() {
		if (open_style != 0) {
			open.style = styles[open_style];
			document.spans.push_back(open);
			open_style = 0;
		}
	};
	auto emit = [&](int p_from, int p_to, int p_style) {
		if (p_style == 0 || p_from >= p_to) {
			return;
		}
		if (open_style == p_style && open.to == p_from) {
			open.to = p_to;
			return;
		}
		flush();
		open.from = p_from;
		open.to = p_to;
		open_style = p_style;
	};

	int base = 0;
	for (int line = p_from_line; line <= p_to_line; line++) {
		const Paragraph &paragraph = paragraphs[line];
		const int from = line == p_from_line ? p_from_column : 0;
		const int to = line == p_to_line ? p_to_column : _get_line_length(line);
		for (const StyleRun &run : paragraph.runs) {
			const int run_from = MAX(run.from, from);
			const int run_to = MIN(run.to, to);
			if (run_from < run_to) {
				emit(base + run_from - from, base + run_to - from, run.style);
			}
		}
		for (const InlineObject &object : paragraph.objects) {
			if (object.column < from || object.column >= to) {
				continue;
			}
			const int offset = base + object.column - from;
			if (object.is_image) {
				RichTextDocument::InlineImage image = object.image;
				image.offset = offset;
				document.images.push_back(image);
			} else {
				RichTextDocument::RawInline raw_inline;
				raw_inline.offset = offset;
				raw_inline.bbcode = object.raw_bbcode;
				document.raw_inlines.push_back(raw_inline);
			}
		}
		if (!paragraph.line_height.is_empty()) {
			RichTextDocument::LineHeightSpan line_height;
			line_height.line_start = base;
			line_height.value = paragraph.line_height;
			document.line_heights.push_back(line_height);
		}
		base += to - from;
		if (line < p_to_line) {
			emit(base, base + 1, paragraph.eol_style);
			base++;
		}
	}
	flush();
	return document;
}

RichTextDocument RichTextEdit::_make_document() const {
	const int last_line = get_line_count() - 1;
	if (last_line < 0 || (int)paragraphs.size() != get_line_count()) {
		RichTextDocument document;
		document.text = TextEdit::get_text();
		return document;
	}
	RichTextDocument document = _make_range_document(0, 0, last_line, _get_line_length(last_line));
	document.canonicalize();
	return document;
}

LocalVector<RichTextEdit::Paragraph> RichTextEdit::_document_to_slices(const RichTextDocument &p_document) {
	const Vector<String> pieces = p_document.text.split("\n");
	LocalVector<Paragraph> slices;
	slices.resize(pieces.size());
	LocalVector<int> starts;
	starts.resize(pieces.size() + 1);
	starts[0] = 0;
	for (int i = 0; i < pieces.size(); i++) {
		slices[i].length = pieces[i].length();
		starts[i + 1] = starts[i] + pieces[i].length() + 1;
	}
	auto line_at = [&](int p_offset) {
		int low = 0;
		int high = pieces.size() - 1;
		while (low < high) {
			const int mid = (low + high + 1) / 2;
			if (starts[mid] <= p_offset) {
				low = mid;
			} else {
				high = mid - 1;
			}
		}
		return low;
	};

	for (const RichTextDocument::StyleSpan &span : p_document.spans) {
		const int style = _intern_style(span.style);
		if (style == 0 || span.from >= span.to) {
			continue;
		}
		for (int line = line_at(span.from); line < pieces.size() && starts[line] < span.to; line++) {
			const int line_end = starts[line] + pieces[line].length();
			const int from = MAX(span.from, starts[line]) - starts[line];
			const int to = MIN(span.to, line_end) - starts[line];
			if (from < to) {
				slices[line].runs.push_back({ from, to, style });
			}
			// The span covers this paragraph's break.
			if (line < pieces.size() - 1 && span.from <= line_end && span.to > line_end) {
				slices[line].eol_style = style;
			}
		}
	}
	for (Paragraph &slice : slices) {
		_normalize_runs(slice.runs);
	}

	for (const RichTextDocument::InlineImage &image : p_document.images) {
		if (image.offset < 0 || image.offset >= p_document.text.length()) {
			continue;
		}
		const int line = line_at(image.offset);
		InlineObject object;
		object.column = image.offset - starts[line];
		object.is_image = true;
		object.image = image;
		if (object.image.texture.is_null()) {
			object.image.texture = RichTextDocument::load_image_texture(object.image.source);
		}
		slices[line].objects.push_back(object);
	}
	for (const RichTextDocument::RawInline &raw_inline : p_document.raw_inlines) {
		if (raw_inline.offset < 0 || raw_inline.offset >= p_document.text.length()) {
			continue;
		}
		const int line = line_at(raw_inline.offset);
		InlineObject object;
		object.column = raw_inline.offset - starts[line];
		object.is_image = false;
		object.raw_bbcode = raw_inline.bbcode;
		slices[line].objects.push_back(object);
	}
	struct ObjectCompare {
		_FORCE_INLINE_ bool operator()(const InlineObject &p_a, const InlineObject &p_b) const {
			return p_a.column < p_b.column;
		}
	};
	for (Paragraph &slice : slices) {
		slice.objects.sort_custom<ObjectCompare>();
	}

	for (const RichTextDocument::LineHeightSpan &line_height : p_document.line_heights) {
		const int line = line_at(line_height.line_start);
		if (starts[line] == line_height.line_start) {
			_set_paragraph_line_height(slices[line], line_height.value);
		}
	}
	return slices;
}

void RichTextEdit::_load_document(const RichTextDocument &p_document, const String &p_bbcode) {
	_clear_selected_image();
	_reset_model();
	LocalVector<Paragraph> loaded = _document_to_slices(p_document);
	for (Paragraph &paragraph : loaded) {
		for (InlineObject &object : paragraph.objects) {
			object.id = next_object_id++;
		}
	}
	paragraphs = loaded;

	// TextEdit shapes every line once while inserting the text; the style
	// parser already reads the final paragraphs.
	setting_bbcode_text = true;
	TextEdit::set_text(p_document.text);
	setting_bbcode_text = false;
	_ensure_model_size();

	typing_style_override = false;
	removed_fragments.clear();
	removed_fragments_chars = 0;
	clear_undo_history();
	_model_changed(0, 0, false);
	bbcode_text = p_bbcode;
	bbcode_revision = use_bbcode ? model_revision : 0;
}

String RichTextEdit::_serialize_bbcode() const {
	return _make_document().to_bbcode();
}

void RichTextEdit::_mark_bbcode_dirty() {
	model_revision++;
}

void RichTextEdit::_lines_edited(int p_from_line, int p_to_line) {
	if (setting_bbcode_text) {
		return;
	}
	// An empty paragraph is formatted by paragraph marks. TextEdit skips
	// reshaping a line whose text did not change (a break typed at its end),
	// and the empty paragraph after an edit reads the mark before it.
	const int first = MIN(p_from_line, p_to_line);
	for (int line : { first, MAX(p_from_line, p_to_line) + 1 }) {
		if (line >= 0 && line < get_line_count() && _get_line_length(line) == 0) {
			_invalidate_lines(line, line);
		}
	}
}

void RichTextEdit::_update_bbcode_from_text() {
	// The edit hooks already updated the paragraphs; announce the change.
	if (!setting_bbcode_text) {
		_style_changed();
	}
}

/* TextEdit callbacks. */

Array RichTextEdit::_get_line_style_spans(int p_line) const {
	Array result;
	if (p_line < 0 || p_line >= (int)paragraphs.size() || p_line >= get_line_count()) {
		return result;
	}
	auto add = [&](int p_style, int p_from, int p_to) {
		const Dictionary base = _get_style_render_info(p_style);
		if (base.is_empty()) {
			return;
		}
		Dictionary info = base.duplicate();
		info["from"] = p_from;
		info["to"] = p_to;
		result.push_back(info);
	};
	const Paragraph &paragraph = paragraphs[p_line];
	const int length = _get_line_length(p_line);
	if (length == 0) {
		// An empty paragraph is formatted by its paragraph mark; a mark that
		// ends here (the previous one) still contributes its block formatting.
		const int previous = p_line > 0 ? paragraphs[p_line - 1].eol_style : 0;
		const int own = p_line < (int)paragraphs.size() - 1 ? paragraph.eol_style : 0;
		if (previous != 0 && previous != own) {
			add(previous, 0, 0);
		}
		if (own != 0) {
			add(own, 0, 1);
		}
		return result;
	}
	for (const StyleRun &run : paragraph.runs) {
		if (run.from < length) {
			add(run.style, run.from, MIN(run.to, length));
		}
	}
	return result;
}

String RichTextEdit::_get_line_height_for_line(int p_line) const {
	if (p_line < 0 || p_line >= (int)paragraphs.size()) {
		return String();
	}
	return paragraphs[p_line].line_height;
}

int RichTextEdit::_get_visual_line_wrap_height(int p_line, int p_wrap_index) const {
	if (p_line < 0 || p_line >= (int)paragraphs.size() || paragraphs[p_line].line_height_kind == 0) {
		return TextEdit::_get_visual_line_wrap_height(p_line, p_wrap_index);
	}
	const Paragraph &paragraph = paragraphs[p_line];
	const int top_margin = _get_visual_line_wrap_top_margin(p_line, p_wrap_index);
	const int bottom_margin = _get_visual_line_wrap_bottom_margin(p_line, p_wrap_index);
	const int content_height = MAX(_get_visual_line_wrap_content_height(p_line, p_wrap_index) - top_margin - bottom_margin, 1);
	int line_box_height = content_height;
	switch (paragraph.line_height_kind) {
		case 1:
			line_box_height = MAX(0, (int)Math::ceil(paragraph.line_height_value));
			break;
		case 2:
			line_box_height = Math::ceil(content_height * paragraph.line_height_value / 100.0);
			break;
		default:
			line_box_height = Math::ceil(content_height * paragraph.line_height_value);
			break;
	}
	return MAX(line_box_height + top_margin + bottom_margin, 1);
}

int RichTextEdit::_get_list_item_number(int p_line, int p_list_type, int p_list_start) const {
	if (p_list_start > 0) {
		return p_list_start;
	}
	if (p_list_type < 1 || p_list_type > 4 || p_line < 0) {
		return 1;
	}
	// Each paragraph advances the counter of every list type it uses; an
	// explicit start resets that counter. Counters are kept per line boundary.
	const int line_count = paragraphs.size();
	p_line = MIN(p_line, line_count);
	if ((int)list_counters.size() < (line_count + 1) * 4) {
		list_counters.resize((line_count + 1) * 4);
	}
	if (list_numbers_valid <= 0) {
		for (int t = 0; t < 4; t++) {
			list_counters[t] = 1;
		}
		list_numbers_valid = 0;
	}
	list_numbers_valid = MIN(list_numbers_valid, line_count);
	while (list_numbers_valid < p_line) {
		const int line = list_numbers_valid;
		int counters[4];
		for (int t = 0; t < 4; t++) {
			counters[t] = list_counters[line * 4 + t];
		}
		bool seen[4] = { false, false, false, false };
		auto count = [&](int p_style) {
			const TextStyle &style = styles[p_style];
			if (style.list_type < 1 || style.list_type > 4 || seen[style.list_type - 1]) {
				return;
			}
			seen[style.list_type - 1] = true;
			if (style.list_start > 0) {
				counters[style.list_type - 1] = MAX(1, style.list_start);
			}
			counters[style.list_type - 1]++;
		};
		const Paragraph &paragraph = paragraphs[line];
		if (_get_line_length(line) == 0) {
			const int previous = line > 0 ? paragraphs[line - 1].eol_style : 0;
			const int own = line < line_count - 1 ? paragraph.eol_style : 0;
			if (previous != 0 && previous != own) {
				count(previous);
			}
			if (own != 0) {
				count(own);
			}
		} else {
			for (const StyleRun &run : paragraph.runs) {
				count(run.style);
			}
		}
		for (int t = 0; t < 4; t++) {
			list_counters[(line + 1) * 4 + t] = counters[t];
		}
		list_numbers_valid++;
	}
	return list_counters[p_line * 4 + p_list_type - 1];
}

/* Style queries. */

RichTextEdit::TextStyle RichTextEdit::_get_selection_common_style() const {
	int from_line = 0;
	int from_column = 0;
	int to_line = 0;
	int to_column = 0;
	if (!_get_selection_range(from_line, from_column, to_line, to_column)) {
		return typing_style;
	}
	return styles[_get_style_id_at(from_line, from_column)];
}

RichTextEdit::TextStyle RichTextEdit::get_current_style() const {
	// The selection start, or the character before the caret.
	const int line = has_selection() ? get_selection_from_line() : get_caret_line();
	int column = 0;
	if (has_selection()) {
		column = get_selection_from_column();
	} else {
		column = MAX(get_caret_column() - 1, 0);
	}
	return styles[_get_style_id_at(line, column)];
}

int RichTextEdit::_get_default_font_size() const {
	const int text_edit_font_size = get_theme_font_size(SceneStringName(font_size));
	if (text_edit_font_size > 0) {
		return text_edit_font_size;
	}

	return 16;
}

void RichTextEdit::_caret_style_context_changed() {
	typing_style_override = false;
	// Moving the caret retargets the edit, so it ends any run in progress.
	_break_style_run();
}

const LocalVector<RichTextEdit::StyleSpan> &RichTextEdit::get_style_spans() const {
	if (style_spans_revision != model_revision) {
		style_spans_cache.clear();
		const RichTextDocument document = _make_document();
		for (const RichTextDocument::StyleSpan &span : document.spans) {
			StyleSpan style_span;
			style_span.from = span.from;
			style_span.to = span.to;
			style_span.style = span.style;
			style_spans_cache.push_back(style_span);
		}
		style_spans_revision = model_revision;
	}
	return style_spans_cache;
}

/* Undo. */

Ref<RichTextEdit::StyleState> RichTextEdit::_capture_style_state(int p_from_line, int p_to_line) const {
	Ref<StyleState> state;
	state.instantiate();
	p_from_line = CLAMP(p_from_line, 0, (int)paragraphs.size() - 1);
	p_to_line = CLAMP(p_to_line, p_from_line, (int)paragraphs.size() - 1);
	state->first_line = p_from_line;
	state->paragraphs.resize(p_to_line - p_from_line + 1);
	for (int line = p_from_line; line <= p_to_line; line++) {
		state->paragraphs[line - p_from_line] = paragraphs[line];
	}
	state->typing_style = typing_style;
	state->typing_style_override = typing_style_override;
	return state;
}

void RichTextEdit::_restore_style_state(const Ref<StyleState> &p_state) {
	ERR_FAIL_COND(p_state.is_null());
	const int count = p_state->paragraphs.size();
	ERR_FAIL_COND_MSG(p_state->first_line + count > (int)paragraphs.size(), "Rich text style history does not match the document.");
	bool selected_object_exists = false;
	for (int i = 0; i < count; i++) {
		paragraphs[p_state->first_line + i] = p_state->paragraphs[i];
	}
	if (selected_object_id != 0) {
		selected_object_exists = _find_object(selected_object_id) != nullptr;
	}
	typing_style = p_state->typing_style;
	typing_style_override = p_state->typing_style_override;
	// A changed paragraph mark also formats the empty paragraph after it.
	_model_changed(p_state->first_line, p_state->first_line + count, true);
	if (selected_object_id != 0 && !selected_object_exists) {
		_clear_selected_image();
	}
}

// Properties whose repeated application is treated as one editing gesture: the
// two spinner-driven sizes, which are clicked or dragged in bursts, and the
// outline color, which streams values while its picker is dragged.
bool RichTextEdit::_is_coalescing_style_property(StyleProperty p_property) {
	return p_property == STYLE_PROPERTY_FONT_SIZE || p_property == STYLE_PROPERTY_OUTLINE_SIZE || p_property == STYLE_PROPERTY_OUTLINE_COLOR;
}

void RichTextEdit::_break_style_run() {
	style_run_active = false;
	style_run_before.unref();
}

void RichTextEdit::_push_style_undo(const Ref<StyleState> &p_before, bool p_isolate) {
	const int count = p_before->paragraphs.size();
	const Ref<StyleState> after = _capture_style_state(p_before->first_line, p_before->first_line + count - 1);
	if (p_isolate) {
		push_custom_undo_operation_isolated(SNAME("rich_text_edit_style"), p_before, after);
	} else {
		push_custom_undo_operation(SNAME("rich_text_edit_style"), p_before, after);
	}
}

void RichTextEdit::_apply_custom_undo_operation(const StringName &p_type, const Variant &p_data) {
	if (p_type == SNAME("rich_text_edit_style")) {
		_restore_style_state(p_data);
		_sync_selected_image_edit_controls();
	}
}

/* Style edits. */

template <typename F>
bool RichTextEdit::_restyle_lines(int p_from_line, int p_to_line, F p_restyle) {
	// p_restyle(line, paragraph) edits one paragraph and reports a change.
	bool changed = false;
	for (int line = p_from_line; line <= p_to_line; line++) {
		changed = p_restyle(line, paragraphs[line]) || changed;
	}
	return changed;
}

// The style ids covering [p_from, p_to) of a paragraph, gaps included as 0.
static void _collect_segments(const LocalVector<RichTextEdit::StyleRun> &p_runs, int p_from, int p_to, LocalVector<RichTextEdit::StyleRun> &r_segments) {
	r_segments.clear();
	int cursor = p_from;
	for (const RichTextEdit::StyleRun &run : p_runs) {
		if (run.to <= cursor) {
			continue;
		}
		if (run.from >= p_to) {
			break;
		}
		if (run.from > cursor) {
			r_segments.push_back({ cursor, run.from, 0 });
		}
		const int to = MIN(run.to, p_to);
		r_segments.push_back({ MAX(run.from, cursor), to, run.style });
		cursor = to;
	}
	if (cursor < p_to) {
		r_segments.push_back({ cursor, p_to, 0 });
	}
}

void RichTextEdit::_apply_style_property_to_selection(StyleProperty p_property, const Variant &p_value, bool p_record_undo) {
	int from_line = 0;
	int from_column = 0;
	int to_line = 0;
	int to_column = 0;
	if (!_get_selection_range(from_line, from_column, to_line, to_column)) {
		// Without a selection the property applies to what typing would insert
		// at the caret, on top of the formatting already there.
		const TextStyle current = typing_style_override ? typing_style : styles[_get_insertion_style_id(get_caret_line(), get_caret_column())];
		TextStyle style = current;
		_apply_style_property(style, style, p_property, p_value);
		if (style == current) {
			return;
		}

		typing_style = style;
		typing_style_override = true;
		if (p_record_undo) {
			_style_changed();
		}
		return;
	}

	const TextStyle reference = _get_selection_common_style();
	HashMap<int, int> restyled;
	auto restyle = [&](int p_style) {
		const int *existing = restyled.getptr(p_style);
		if (existing != nullptr) {
			return *existing;
		}
		TextStyle style = styles[p_style];
		_apply_style_property(style, reference, p_property, p_value);
		const int id = _intern_style(style);
		restyled.insert(p_style, id);
		return id;
	};

	const Ref<StyleState> before = _capture_style_state(from_line, to_line);
	LocalVector<StyleRun> segments;
	const bool changed = _restyle_lines(from_line, to_line, [&](int p_line, Paragraph &r_paragraph) {
		const int from = p_line == from_line ? from_column : 0;
		const int to = p_line == to_line ? to_column : _get_line_length(p_line);
		bool line_changed = false;
		_collect_segments(r_paragraph.runs, from, to, segments);
		for (const StyleRun &segment : segments) {
			const int style = restyle(segment.style);
			if (style != segment.style) {
				_set_runs_range(r_paragraph.runs, segment.from, segment.to, style);
				line_changed = true;
			}
		}
		// The paragraph break is part of the selection.
		if (p_line < to_line) {
			const int style = restyle(r_paragraph.eol_style);
			if (style != r_paragraph.eol_style) {
				r_paragraph.eol_style = style;
				line_changed = true;
			}
		}
		return line_changed;
	});
	if (!changed) {
		return;
	}
	_model_changed(from_line, to_line + 1);
	if (p_record_undo) {
		const Vector4i range(from_line, from_column, to_line, to_column);
		const bool continues_run = style_run_active && style_run_property == p_property && style_run_range == range && _is_coalescing_style_property(p_property);
		// A run keeps the state from before its first edit, so one undo returns
		// to the value the field held before the burst began.
		if (!continues_run || !amend_custom_undo_operation(SNAME("rich_text_edit_style"), _capture_style_state(from_line, to_line))) {
			_push_style_undo(before);
			style_run_active = _is_coalescing_style_property(p_property);
			style_run_property = p_property;
			style_run_range = range;
		}
	}
}

void RichTextEdit::_apply_style_property_to_url_ranges(StyleProperty p_property, const Variant &p_value) {
	int from_line = 0;
	int from_column = 0;
	int to_line = 0;
	int to_column = 0;
	const bool has_range = _get_selection_range(from_line, from_column, to_line, to_column);
	if (!has_range) {
		from_line = to_line = get_caret_line();
		from_column = to_column = get_caret_column();
	}

	// Links are edited as a whole: every link the selection (or the caret, or
	// the character before it) touches, extended over adjacent paragraphs.
	struct Position {
		int line = 0;
		int column = 0;
	};
	LocalVector<Position> seeds;
	auto consider = [&](int p_line, int p_column) {
		const int style = _get_style_id_at(p_line, p_column);
		if (style != 0 && !styles[style].url.is_empty()) {
			seeds.push_back({ p_line, p_column });
		}
	};
	if (has_range) {
		for (int line = from_line; line <= to_line; line++) {
			const int from = line == from_line ? from_column : 0;
			const int to = line == to_line ? to_column : _get_line_length(line) + 1;
			for (const StyleRun &run : paragraphs[line].runs) {
				if (run.from < to && run.to > from) {
					consider(line, MAX(run.from, from));
				}
			}
			if (line < to_line) {
				consider(line, _get_line_length(line));
			}
		}
	} else {
		consider(from_line, from_column);
		if (from_column > 0) {
			consider(from_line, from_column - 1);
		}
	}
	if (seeds.is_empty()) {
		return;
	}

	int first_line = from_line;
	int last_line = to_line;
	HashMap<int, int> restyled;
	auto restyle = [&](int p_style) {
		const int *existing = restyled.getptr(p_style);
		if (existing != nullptr) {
			return *existing;
		}
		TextStyle style = styles[p_style];
		_apply_style_property(style, style, p_property, p_value);
		const int id = _intern_style(style);
		restyled.insert(p_style, id);
		return id;
	};

	// Walk each link's characters in both directions while the style holds.
	struct Edit {
		int line = 0;
		int from = 0;
		int to = 0;
		bool eol = false;
		int style = 0;
	};
	LocalVector<Edit> edits;
	auto collect = [&](int p_line, int p_column) {
		const int style = _get_style_id_at(p_line, p_column);
		int line = p_line;
		int column = p_column;
		// Backward to the first character of the link.
		while (true) {
			if (column > 0 && _get_style_id_at(line, column - 1) == style) {
				column--;
				continue;
			}
			if (column == 0 && line > 0 && paragraphs[line - 1].eol_style == style) {
				line--;
				column = _get_line_length(line);
				continue;
			}
			break;
		}
		first_line = MIN(first_line, line);
		// Forward over its characters and paragraph marks.
		while (line < (int)paragraphs.size()) {
			const int length = _get_line_length(line);
			int end = column;
			while (end < length && _get_style_id_at(line, end) == style) {
				end++;
			}
			if (end > column) {
				edits.push_back({ line, column, end, false, style });
			}
			if (end == length && line < (int)paragraphs.size() - 1 && paragraphs[line].eol_style == style) {
				edits.push_back({ line, length, length + 1, true, style });
				line++;
				column = 0;
				continue;
			}
			break;
		}
		last_line = MAX(last_line, line);
	};
	for (const Position &seed : seeds) {
		collect(seed.line, seed.column);
	}

	const Ref<StyleState> before = _capture_style_state(first_line, last_line);
	bool changed = false;
	for (const Edit &edit : edits) {
		const int style = restyle(edit.style);
		if (style == edit.style) {
			continue;
		}
		if (edit.eol) {
			paragraphs[edit.line].eol_style = style;
		} else {
			_set_runs_range(paragraphs[edit.line].runs, edit.from, edit.to, style);
		}
		changed = true;
	}
	if (!changed) {
		return;
	}
	_model_changed(first_line, last_line + 1);
	_push_style_undo(before);
}

void RichTextEdit::_apply_block_tag_to_selected_lines(const String &p_tag) {
	int from_line = 0;
	int to_line = 0;
	_get_selected_line_range(from_line, to_line);

	// Every call is one editing gesture. Lines it touches share a single group
	// id so their quote borders draw as one continuous box; a later call (even
	// over lines directly adjacent to this one) gets a new id, so the two
	// groups stay visually separate instead of merging.
	const int block_group = p_tag == "quote" ? ++next_block_group_id : 0;
	HashMap<int, int> restyled;
	const Ref<StyleState> before = _capture_style_state(from_line, to_line);
	LocalVector<StyleRun> segments;
	const bool changed = _restyle_lines(from_line, to_line, [&](int p_line, Paragraph &r_paragraph) {
		const int length = _get_line_length(p_line);
		if (length == 0) {
			return false;
		}
		bool line_changed = false;
		_collect_segments(r_paragraph.runs, 0, length, segments);
		for (const StyleRun &segment : segments) {
			int style = 0;
			const int *existing = restyled.getptr(segment.style);
			if (existing != nullptr) {
				style = *existing;
			} else {
				TextStyle updated = styles[segment.style];
				_apply_style_property(updated, updated, STYLE_PROPERTY_BLOCK_TAG, p_tag);
				updated.block_group = block_group;
				style = _intern_style(updated);
				restyled.insert(segment.style, style);
			}
			if (style != segment.style) {
				_set_runs_range(r_paragraph.runs, segment.from, segment.to, style);
				line_changed = true;
			}
		}
		return line_changed;
	});
	if (!changed) {
		return;
	}
	_model_changed(from_line, to_line + 1);
	_push_style_undo(before);
}

void RichTextEdit::_clear_block_tag_from_selected_lines(const String &p_tag) {
	int from_line = 0;
	int to_line = 0;
	_get_selected_line_range(from_line, to_line);
	const Ref<StyleState> before = _capture_style_state(from_line, to_line);
	LocalVector<StyleRun> segments;
	const bool changed = _restyle_lines(from_line, to_line, [&](int p_line, Paragraph &r_paragraph) {
		const int length = _get_line_length(p_line);
		if (length == 0) {
			return false;
		}
		bool line_changed = false;
		_collect_segments(r_paragraph.runs, 0, length, segments);
		for (const StyleRun &segment : segments) {
			TextStyle style = styles[segment.style];
			if (style.block_tag != p_tag) {
				continue;
			}
			style.block_tag = "";
			style.block_group = 0;
			if (p_tag == "ul" || p_tag == "ol") {
				style.indent_level = MAX(0, style.indent_level - 1);
				style.list_type = 0;
				style.list_start = -1;
				style.list_capitalize = false;
				if (style.indent_level > 0) {
					style.block_tag = "indent";
				}
			}
			_set_runs_range(r_paragraph.runs, segment.from, segment.to, _intern_style(style));
			line_changed = true;
		}
		return line_changed;
	});
	if (!changed) {
		return;
	}
	_model_changed(from_line, to_line + 1);
	_push_style_undo(before);
}

void RichTextEdit::_toggle_block_tag_on_selected_lines(const String &p_tag) {
	int from_line = 0;
	int to_line = 0;
	_get_selected_line_range(from_line, to_line);
	bool all_lines_have_tag = true;
	bool has_text_line = false;

	for (int line = from_line; line <= to_line; line++) {
		if (_get_line_length(line) == 0) {
			continue;
		}
		has_text_line = true;
		if (styles[_get_style_id_at(line, 0)].block_tag != p_tag) {
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

void RichTextEdit::decrease_indent() {
	int from_line = 0;
	int to_line = 0;
	_get_selected_line_range(from_line, to_line);
	const Ref<StyleState> before = _capture_style_state(from_line, to_line);
	LocalVector<StyleRun> segments;
	const bool changed = _restyle_lines(from_line, to_line, [&](int p_line, Paragraph &r_paragraph) {
		const int length = _get_line_length(p_line);
		if (length == 0) {
			return false;
		}
		bool line_changed = false;
		_collect_segments(r_paragraph.runs, 0, length, segments);
		for (const StyleRun &segment : segments) {
			TextStyle style = styles[segment.style];
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
			_set_runs_range(r_paragraph.runs, segment.from, segment.to, _intern_style(style));
			line_changed = true;
		}
		return line_changed;
	});
	if (!changed) {
		return;
	}
	_model_changed(from_line, to_line + 1);
	_push_style_undo(before);
}

void RichTextEdit::begin_selection_color_preview() {
	if (style_preview_active) {
		return;
	}

	style_preview_active = true;
	int from_line = 0;
	int from_column = 0;
	int to_line = 0;
	int to_column = 0;
	style_preview_had_selection = _get_selection_range(from_line, from_column, to_line, to_column);
	if (!style_preview_had_selection) {
		from_line = to_line = get_caret_line();
	}
	style_preview_before = _capture_style_state(from_line, to_line);
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

	const Ref<StyleState> before = style_preview_before;
	const bool had_selection = style_preview_had_selection;
	style_preview_active = false;
	style_preview_had_selection = false;
	style_preview_before.unref();
	if (before.is_null()) {
		return;
	}

	if (p_commit) {
		if (had_selection) {
			_push_style_undo(before);
		}
		_style_changed();
	} else {
		_restore_style_state(before);
	}
}

void RichTextEdit::begin_selection_bg_color_preview() {
	begin_selection_color_preview();
}

void RichTextEdit::preview_selection_bg_color(const Color &p_color) {
	if (!style_preview_active) {
		begin_selection_bg_color_preview();
	}
	_apply_style_property_to_selection(STYLE_PROPERTY_BG_COLOR, p_color, false);
}

void RichTextEdit::preview_selection_outline_color(const Color &p_color) {
	if (!style_preview_active) {
		begin_selection_color_preview();
	}
	_apply_style_property_to_selection(STYLE_PROPERTY_OUTLINE_COLOR, p_color, false);
}

void RichTextEdit::end_selection_bg_color_preview(bool p_commit) {
	end_selection_color_preview(p_commit);
}

void RichTextEdit::set_selection_url(const String &p_url) {
	const Vector<uint64_t> image_ids = _get_selected_image_ids();
	if (!image_ids.is_empty()) {
		int first_line = INT_MAX;
		int last_line = -1;
		for (uint64_t id : image_ids) {
			int line = 0;
			if (_find_object(id, &line) != nullptr) {
				first_line = MIN(first_line, line);
				last_line = MAX(last_line, line);
			}
		}
		if (last_line < 0) {
			return;
		}
		const Ref<StyleState> before = _capture_style_state(first_line, last_line);
		bool changed = false;
		for (uint64_t id : image_ids) {
			InlineObject *object = _find_object(id);
			if (object == nullptr) {
				continue;
			}
			const String previous = object->image.options.has("link") ? object->image.options["link"] : String();
			if (p_url.is_empty()) {
				object->image.options.erase("link");
			} else {
				object->image.options["link"] = p_url;
			}
			changed = changed || previous != p_url;
		}
		if (changed) {
			_push_style_undo(before, true);
			_model_changed(first_line, last_line);
		}
		return;
	}
	_apply_style_property_to_selection(STYLE_PROPERTY_URL, p_url);
}

void RichTextEdit::set_selection_line_height(const String &p_line_height) {
	const String line_height = p_line_height.strip_edges();
	if (!RichTextDocument::is_valid_line_height(line_height)) {
		return;
	}

	int first_line = get_caret_line();
	int last_line = first_line;
	if (has_selection()) {
		first_line = get_selection_from_line();
		last_line = get_selection_to_line();
		if (last_line > first_line && get_selection_to_column() == 0) {
			last_line--;
		}
	}

	const Ref<StyleState> before = _capture_style_state(first_line, last_line);
	bool changed = false;
	for (int line = first_line; line <= last_line; line++) {
		if (paragraphs[line].line_height != line_height) {
			_set_paragraph_line_height(paragraphs[line], line_height);
			changed = true;
		}
	}
	if (!changed) {
		return;
	}
	_push_style_undo(before);
	_model_changed(first_line, last_line);
}

void RichTextEdit::clear_selection_line_height() {
	int first_line = get_caret_line();
	int last_line = first_line;
	if (has_selection()) {
		first_line = get_selection_from_line();
		last_line = get_selection_to_line();
		if (last_line > first_line && get_selection_to_column() == 0) {
			last_line--;
		}
	}
	const Ref<StyleState> before = _capture_style_state(first_line, last_line);
	bool changed = false;
	for (int line = first_line; line <= last_line; line++) {
		if (!paragraphs[line].line_height.is_empty()) {
			_set_paragraph_line_height(paragraphs[line], String());
			changed = true;
		}
	}
	if (!changed) {
		return;
	}
	_push_style_undo(before);
	_model_changed(first_line, last_line);
}

String RichTextEdit::get_current_line_height() const {
	const String value = _get_line_height_for_line(has_selection() ? get_selection_from_line() : get_caret_line());
	return value.is_empty() ? String("normal") : value;
}

int RichTextEdit::get_current_font_size() const {
	const TextStyle style = get_current_style();
	return style.font_size > 0 ? style.font_size : _get_default_font_size();
}

/* Style templates. */

// A style template is the BBCode of a one-character document ("X") whose run
// carries the inline style and whose paragraph carries the line height.
static bool _parse_style_template(const String &p_template, RichTextDocument &r_document) {
	if (p_template.is_empty()) {
		r_document = RichTextDocument();
		r_document.text = "X";
		return true;
	}
	r_document = RichTextDocument::parse_bbcode(p_template);
	return r_document.text == "X";
}

Dictionary RichTextEdit::get_style_template_values(const String &p_template) {
	RichTextDocument document;
	if (p_template.is_empty() || !_parse_style_template(p_template, document)) {
		return Dictionary();
	}
	const TextStyle style = document.spans.is_empty() ? TextStyle() : document.spans[0].style;
	Dictionary values;
	String decoration;
	if (style.has_underline && style.underline) {
		decoration = "underline";
	}
	if (style.overline) {
		decoration += decoration.is_empty() ? "overline" : " overline";
	}
	if (style.strikethrough) {
		decoration += decoration.is_empty() ? "line-through" : " line-through";
	}
	values["TD"] = decoration;
	values["I"] = style.italic ? "1" : "";
	values["BG"] = style.has_bg_color ? "#" + style.bg_color.to_html(style.bg_color.a < 1) : "";
	values["OC"] = style.has_outline_color ? "#" + style.outline_color.to_html(style.outline_color.a < 1) : "";
	values["OW"] = style.has_outline_size ? itos(style.outline_size) + "px" : "";
	return values;
}

String RichTextEdit::get_selection_style_template() const {
	int from_line = 0;
	int from_column = 0;
	int to_line = 0;
	int to_column = 0;
	if (!_get_selection_range(from_line, from_column, to_line, to_column)) {
		return String();
	}
	RichTextDocument template_document;
	template_document.text = "X";
	const int style = _get_style_id_at(from_line, from_column);
	if (style != 0) {
		RichTextDocument::StyleSpan span;
		span.from = 0;
		span.to = 1;
		span.style = styles[style];
		template_document.spans.push_back(span);
	}
	const String line_height = _get_line_height_for_line(from_line);
	if (!line_height.is_empty()) {
		RichTextDocument::LineHeightSpan span;
		span.line_start = 0;
		span.value = line_height;
		template_document.line_heights.push_back(span);
	}
	return template_document.to_bbcode();
}

bool RichTextEdit::apply_style_template(const String &p_template, const Dictionary &p_overrides) {
	RichTextDocument template_document;
	// CSS-only Label templates use the same atomic replacement as native
	// Article templates. An empty template starts from plain inline formatting.
	if (!_parse_style_template(p_template, template_document)) {
		ERR_PRINT("Unable to apply rich-text style template: it must format a single \"X\".");
		return false;
	}
	int from_line = 0;
	int from_column = 0;
	int to_line = 0;
	int to_column = 0;
	if (!_get_selection_range(from_line, from_column, to_line, to_column)) {
		return false;
	}
	const bool full_lines = from_column == 0 &&
			(to_column == _get_line_length(to_line) || (to_column == 0 && to_line > from_line));
	const int final_line = to_column == 0 && to_line > from_line ? to_line - 1 : to_line;

	TextStyle desired;
	if (!template_document.spans.is_empty()) {
		desired = template_document.spans[0].style;
	}
	if (p_overrides.has("FS")) {
		const String size = String(p_overrides["FS"]).strip_edges();
		if (size.ends_with("px")) {
			desired.font_size = size.substr(0, size.length() - 2).to_int();
		}
	}
	if (p_overrides.has("FF")) {
		desired.font = String(p_overrides["FF"]);
	}
	if (p_overrides.has("FW")) {
		desired.bold = String(p_overrides["FW"]).to_int() >= 600;
	}
	if (p_overrides.has("I")) {
		desired.italic = String(p_overrides["I"]) == "1";
	}
	if (p_overrides.has("BG")) {
		const String color_text = String(p_overrides["BG"]);
		desired.has_bg_color = Color::html_is_valid(color_text);
		if (desired.has_bg_color) {
			desired.bg_color = Color::html(color_text);
		}
	}
	if (p_overrides.has("OC")) {
		const String color_text = String(p_overrides["OC"]);
		desired.has_outline_color = Color::html_is_valid(color_text);
		if (desired.has_outline_color) {
			desired.outline_color = Color::html(color_text);
		}
	}
	if (p_overrides.has("OW")) {
		desired.outline_size = MAX(0, String(p_overrides["OW"]).to_int());
		desired.has_outline_size = desired.outline_size > 0;
	}
	if (p_overrides.has("COL")) {
		const String color_text = String(p_overrides["COL"]);
		if (Color::html_is_valid(color_text)) {
			desired.has_color = true;
			desired.color = Color::html(color_text);
		}
	}
	if (p_overrides.has("TD")) {
		const String decoration = String(p_overrides["TD"]);
		desired.has_underline = decoration.contains("underline");
		desired.underline = desired.has_underline;
		desired.overline = decoration.contains("overline");
		desired.strikethrough = decoration.contains("line-through");
	}
	if (p_overrides.has("TA")) {
		const String alignment = String(p_overrides["TA"]);
		desired.alignment = HORIZONTAL_ALIGNMENT_LEFT;
		if (alignment == "center") {
			desired.alignment = HORIZONTAL_ALIGNMENT_CENTER;
		} else if (alignment == "right") {
			desired.alignment = HORIZONTAL_ALIGNMENT_RIGHT;
		} else if (alignment == "justify") {
			desired.alignment = HORIZONTAL_ALIGNMENT_FILL;
		}
		if (desired.block_tag.is_empty() || desired.block_tag == "left" || desired.block_tag == "center" || desired.block_tag == "right" || desired.block_tag == "fill") {
			desired.block_tag = alignment == "justify" ? "fill" : alignment;
		}
	}
	if (full_lines && desired.block_tag == "quote") {
		desired.block_group = ++next_block_group_id;
	}

	HashMap<int, int> replaced;
	auto replacement = [&](int p_style) {
		if (full_lines) {
			return _intern_style(desired);
		}
		const int *existing = replaced.getptr(p_style);
		if (existing != nullptr) {
			return *existing;
		}
		// A partial selection takes the inline style only; the paragraph keeps
		// its block formatting.
		const TextStyle &original = styles[p_style];
		TextStyle style = desired;
		style.alignment = original.alignment;
		style.indent_level = original.indent_level;
		style.list_type = original.list_type;
		style.list_start = original.list_start;
		style.list_capitalize = original.list_capitalize;
		style.block_tag = original.block_tag;
		style.block_group = original.block_group;
		const int id = _intern_style(style);
		replaced.insert(p_style, id);
		return id;
	};

	const Ref<StyleState> before = _capture_style_state(from_line, to_line);
	LocalVector<StyleRun> segments;
	bool changed = _restyle_lines(from_line, to_line, [&](int p_line, Paragraph &r_paragraph) {
		const int from = p_line == from_line ? from_column : 0;
		const int to = p_line == to_line ? to_column : _get_line_length(p_line);
		bool line_changed = false;
		_collect_segments(r_paragraph.runs, from, to, segments);
		for (const StyleRun &segment : segments) {
			const int style = replacement(segment.style);
			if (style != segment.style) {
				_set_runs_range(r_paragraph.runs, segment.from, segment.to, style);
				line_changed = true;
			}
		}
		if (p_line < to_line) {
			const int style = replacement(r_paragraph.eol_style);
			if (style != r_paragraph.eol_style) {
				r_paragraph.eol_style = style;
				line_changed = true;
			}
		}
		return line_changed;
	});
	if (full_lines) {
		String line_height = template_document.line_heights.is_empty() ? String() : template_document.line_heights[0].value;
		if (p_overrides.has("LH")) {
			const String override_height = String(p_overrides["LH"]);
			line_height = override_height == "normal" ? String() : override_height;
		}
		for (int line = from_line; line <= final_line; line++) {
			if (paragraphs[line].line_height != line_height) {
				_set_paragraph_line_height(paragraphs[line], line_height);
				changed = true;
			}
		}
	}
	if (!changed) {
		return true;
	}
	_model_changed(from_line, to_line + 1);
	_push_style_undo(before);
	return true;
}

/* Rich clipboard. */

void RichTextEdit::copy_rich_text() {
	int from_line = 0;
	int from_column = 0;
	int to_line = 0;
	int to_column = 0;
	if (!_get_selection_range(from_line, from_column, to_line, to_column)) {
		rich_text_clipboard = RichTextDocument();
		rich_text_clipboard_valid = false;
		rich_text_clipboard_plain_text = String();
		TextEdit::copy();
		return;
	}
	RichTextDocument selection = _make_range_document(from_line, from_column, to_line, to_column);
	for (RichTextDocument::InlineImage &image : selection.images) {
		image.texture.unref();
	}
	selection.canonicalize();
	rich_text_clipboard = selection;
	rich_text_clipboard_valid = true;
	if (DisplayServer::get_singleton()->has_feature(DisplayServerEnums::FEATURE_CLIPBOARD)) {
		TextEdit::copy();
		rich_text_clipboard_plain_text = DisplayServer::get_singleton()->clipboard_get();
	} else {
		// Headless tests and platforms without a system clipboard still retain
		// the process-local structured clipboard.
		rich_text_clipboard_plain_text = selection.text;
	}
}

bool RichTextEdit::paste_rich_text() {
	const bool has_clipboard = DisplayServer::get_singleton()->has_feature(DisplayServerEnums::FEATURE_CLIPBOARD);
	if (!rich_text_clipboard_valid || (has_clipboard && DisplayServer::get_singleton()->clipboard_get() != rich_text_clipboard_plain_text)) {
		return false;
	}
	// The insertion hook formats the pasted text with the copied paragraphs.
	pending_fragment = _document_to_slices(rich_text_clipboard);
	pending_fragment_text = rich_text_clipboard.text;
	pending_fragment_inherit_style = false;
	pending_fragment_active = true;
	begin_complex_operation();
	if (has_clipboard) {
		TextEdit::paste();
	} else {
		insert_text_at_caret(rich_text_clipboard.text);
	}
	end_complex_operation();
	pending_fragment_active = false;
	pending_fragment.clear();
	return true;
}

/* Text. */

void RichTextEdit::set_text(const String &p_text) {
	RichTextDocument document;
	if (use_bbcode) {
		document = RichTextDocument::parse_bbcode(p_text);
	} else {
		document.text = p_text.remove_char('\r');
	}
	_load_document(document, use_bbcode ? p_text : String());
}

String RichTextEdit::get_text() const {
	return use_bbcode ? get_bbcode_text() : TextEdit::get_text();
}

void RichTextEdit::set_bbcode_text(const String &p_bbcode) {
	use_bbcode = true;
	_load_document(RichTextDocument::parse_bbcode(p_bbcode), p_bbcode);
}

String RichTextEdit::get_bbcode_text() const {
	if (bbcode_revision != model_revision) {
		bbcode_text = _serialize_bbcode();
		bbcode_revision = model_revision;
	}
	return bbcode_text;
}

void RichTextEdit::set_style_font_resolver(const Callable &p_resolver) {
	// Reinstalling the same resolver refreshes asynchronously loaded faces.
	style_font_resolver = p_resolver;
	_refresh_all_styles();
}

Callable RichTextEdit::get_style_font_resolver() const {
	return style_font_resolver;
}

void RichTextEdit::set_outline_over_fill(bool p_enabled) {
	if (outline_over_fill == p_enabled) {
		return;
	}
	outline_over_fill = p_enabled;
	queue_redraw();
}

bool RichTextEdit::is_outline_over_fill() const {
	return outline_over_fill;
}

void RichTextEdit::set_decoration_paint(const Dictionary &p_paint) {
	if (decoration_paint == p_paint) {
		return;
	}
	decoration_paint = p_paint.duplicate();
	// Paint lives in the per-run render information; reshape with it.
	_refresh_all_styles();
}

void RichTextEdit::_style_changed() {
	_mark_bbcode_dirty();
	if (!is_inside_tree() || text_style_changed_dirty) {
		return;
	}

	text_style_changed_dirty = true;
	callable_mp(this, &RichTextEdit::_emit_text_style_changed).call_deferred();
}

bool RichTextEdit::has_pending_layout() const {
	return _has_pending_shaping();
}

void RichTextEdit::_deferred_layout_progressed(bool p_finished) {
	// Background shaping refines the content height over many frames. Layouts
	// that follow it hear about it a few times a second and when it is done.
	if (content_height_signal_queued || !is_inside_tree()) {
		return;
	}
	const uint64_t now = OS::get_singleton()->get_ticks_usec();
	if (!p_finished && now - content_height_signal_usec < 150000) {
		return;
	}
	content_height_signal_usec = now;
	content_height_signal_queued = true;
	callable_mp(this, &RichTextEdit::_emit_content_height_changed).call_deferred();
}

void RichTextEdit::_emit_content_height_changed() {
	content_height_signal_queued = false;
	emit_signal(SNAME("content_height_changed"));
}

void RichTextEdit::_emit_text_style_changed() {
	text_style_changed_dirty = false;
	emit_signal(SNAME("text_style_changed"));
}

void RichTextEdit::_refresh_all_styles() {
	_clear_style_render_infos();
	if (get_line_count() > 0) {
		_invalidate_lines(0, get_line_count() - 1);
	}
}

void RichTextEdit::_queue_full_style_refresh() {
	if (style_refresh_queued) {
		return;
	}
	style_refresh_queued = true;
	callable_mp(this, &RichTextEdit::_refresh_queued_styles).call_deferred();
}

void RichTextEdit::_refresh_queued_styles() {
	style_refresh_queued = false;
	_refresh_all_styles();
}

void RichTextEdit::_update_theme_item_cache() {
	TextEdit::_update_theme_item_cache();
	// Theme updates happen for any change up the tree; only reshape when an
	// item the run styles use actually changed.
	bool changed = false;
	auto check = [&](const StringName &p_name, const Variant &p_value) {
		const Variant *previous = style_theme_values.getptr(p_name);
		if (previous == nullptr || *previous != p_value) {
			style_theme_values[p_name] = p_value;
			changed = true;
		}
	};
	for (const char *name : STYLE_THEME_FONTS) {
		check(name, get_theme_font(name));
	}
	for (const char *name : STYLE_THEME_COLORS) {
		check(name, get_theme_color(name, SNAME("RichTextEdit")));
	}
	for (const char *name : STYLE_THEME_CONSTANTS) {
		check(name, get_theme_constant(name, SNAME("RichTextEdit")));
	}
	if (!changed) {
		return;
	}
	// TextEdit reshapes every line itself when its own font or size changes;
	// otherwise reshape here, but only if runs were rendered with old values.
	bool base_changed = false;
	const Variant base_font = get_theme_font(SceneStringName(font));
	const Variant base_size = get_theme_font_size(SceneStringName(font_size));
	for (const KeyValue<StringName, Variant> &base : { KeyValue<StringName, Variant>(SNAME("font"), base_font), KeyValue<StringName, Variant>(SNAME("font_size"), base_size) }) {
		const Variant *previous = style_theme_values.getptr(base.key);
		base_changed = base_changed || previous == nullptr || *previous != base.value;
		style_theme_values[base.key] = base.value;
	}
	bool rendered = false;
	for (uint32_t i = 1; i < style_render_info_valid.size() && !rendered; i++) {
		rendered = style_render_info_valid[i];
	}
	_clear_style_render_infos();
	if (rendered && !base_changed) {
		_queue_full_style_refresh();
	}
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
		case STYLE_PROPERTY_OVERLINE:
			r_style.overline = p_value.get_type() == Variant::BOOL ? bool(p_value) : !p_toggle_reference.overline;
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

/* URLs. */

bool RichTextEdit::_get_url_style_at(int p_line, int p_column, TextStyle &r_style) const {
	const int style = _get_style_id_at(p_line, p_column);
	if (style == 0 || styles[style].url.is_empty()) {
		return false;
	}
	r_style = styles[style];
	return true;
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

	const int column = CLAMP(pos.x, 0, _get_line_length(line));
	if (_get_url_style_at(line, column, r_style)) {
		return true;
	}
	if (column > 0) {
		return _get_url_style_at(line, column - 1, r_style);
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
			_clear_style_render_infos();
			emit_signal(SNAME("meta_hover_started"), meta);
			queue_redraw();
		}
	} else if (meta_hovering) {
		meta_hovering = false;
		emit_signal(SNAME("meta_hover_ended"), current_meta);
		current_meta = String();
		_clear_style_render_infos();
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
		_clear_style_render_infos();
		queue_redraw();
		emit_signal(SNAME("meta_clicked"), meta);
		return true;
	}

	if (_is_url_auto_open_allowed(meta)) {
		active_meta = meta;
		visited_links.insert(meta);
		_clear_style_render_infos();
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

/* Paragraph breaks. */

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
	if (line < 0 || line >= get_line_count()) {
		return;
	}
	_ensure_model_size();

	const int line_length = _get_line_length(line);
	const bool at_line_beginning = column == 0 || line_length == 0;
	int style_at_caret = 0;
	if (line_length == 0) {
		style_at_caret = typing_style_override ? _intern_style(typing_style) : _get_style_id_at(line, column);
	} else if (at_line_beginning) {
		style_at_caret = _get_style_id_at(line, 0);
	} else {
		style_at_caret = _get_style_id_at(line, column - 1);
	}

	if (p_shift_pressed && column == line_length && line < get_line_count() - 1 && style_at_caret != 0) {
		// Shift+Enter at the end of a formatted paragraph starts a plain one:
		// neither paragraph break keeps the formatting.
		typing_style = TextStyle();
		typing_style_override = true;
		insert_text_at_caret("\n");
		if (line + 1 < (int)paragraphs.size() && paragraphs[line + 1].eol_style != 0) {
			paragraphs[line + 1].eol_style = 0;
			_model_changed(line + 1, line + 2);
		}
		typing_style = TextStyle();
		typing_style_override = true;
		accept_event();
		return;
	}

	typing_style = p_shift_pressed ? TextStyle() : styles[style_at_caret];
	typing_style_override = true;
	insert_text_at_caret("\n");
	if (p_shift_pressed && at_line_beginning) {
		typing_style = styles[style_at_caret];
	} else {
		typing_style = p_shift_pressed ? TextStyle() : styles[style_at_caret];
	}
	typing_style_override = true;
	_style_changed();
	accept_event();
}

/* Inline objects. */

Dictionary RichTextEdit::_make_image_inline_info(const InlineObject &p_object) const {
	const RichTextDocument::InlineImage &image = p_object.image;
	Dictionary info;
	info["column"] = p_object.column;
	info["object_id"] = (int64_t)p_object.id;
	info["cursor_shape"] = CURSOR_ARROW;

	Size2 texture_size;
	if (image.texture.is_valid()) {
		texture_size = image.texture->get_size();
	}

	float width = 0.0;
	float height = 0.0;
	const String size_text = image.options.has("size") ? image.options["size"] : String();
	if (!size_text.is_empty()) {
		Vector<String> parts = size_text.to_lower().split("x");
		if (parts.size() == 2) {
			width = parts[0].strip_edges().to_float();
			height = parts[1].strip_edges().to_float();
		} else {
			width = size_text.to_float();
		}
	}
	if (image.options.has("width")) {
		width = image.options["width"].to_float();
	}
	if (image.options.has("height")) {
		height = image.options["height"].to_float();
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

RichTextEdit::InlineObject *RichTextEdit::_find_object(uint64_t p_id, int *r_line) {
	if (p_id == 0) {
		return nullptr;
	}
	for (uint32_t line = 0; line < paragraphs.size(); line++) {
		for (InlineObject &object : paragraphs[line].objects) {
			if (object.id == p_id) {
				if (r_line != nullptr) {
					*r_line = line;
				}
				return &object;
			}
		}
	}
	return nullptr;
}

const RichTextEdit::InlineObject *RichTextEdit::_get_object_at(int p_line, int p_column) const {
	if (p_line < 0 || p_line >= (int)paragraphs.size()) {
		return nullptr;
	}
	for (const InlineObject &object : paragraphs[p_line].objects) {
		if (object.column == p_column) {
			return &object;
		}
	}
	return nullptr;
}

Size2 RichTextEdit::_get_image_option_size(const RichTextDocument::InlineImage &p_image, const Size2 &p_fallback_size) const {
	Size2 size = p_fallback_size;
	bool has_x = false;
	bool has_y = false;
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
		has_x = true;
		has_y = true;
	}
	if (p_image.options.has("width")) {
		size.x = p_image.options["width"].to_float();
		has_x = true;
	}
	if (p_image.options.has("height")) {
		size.y = p_image.options["height"].to_float();
		has_y = true;
	}
	// The image itself is the source of truth for its size. Only when it has
	// no explicit size (or an invalid one) do we fall back to the texture's
	// natural size, and only after that to the caller-supplied fallback — a
	// stale value (e.g. a previous size shown in the width/height fields)
	// must never win over the image's actual current size.
	if (p_image.texture.is_valid()) {
		const Size2 texture_size = p_image.texture->get_size();
		if (!has_x || size.x <= 0.0f) {
			size.x = texture_size.x;
		}
		if (!has_y || size.y <= 0.0f) {
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

void RichTextEdit::_insert_objects(const String &p_text, const LocalVector<Paragraph> &p_slices) {
	// The insertion hook places the objects; the text takes the style typing
	// would give it there.
	pending_fragment = p_slices;
	pending_fragment_text = p_text;
	pending_fragment_inherit_style = true;
	pending_fragment_active = true;
	begin_complex_operation();
	insert_text_at_caret(p_text);
	end_complex_operation();
	pending_fragment_active = false;
	pending_fragment.clear();
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

	LocalVector<Paragraph> slices;
	slices.resize(1);
	slices[0].length = 1;
	InlineObject object;
	object.is_image = true;
	object.image.source = p_source;
	object.image.options = options;
	object.image.texture = RichTextDocument::load_image_texture(p_source);
	slices[0].objects.push_back(object);
	_insert_objects(String::chr(RichTextDocument::OBJECT_REPLACEMENT_CHAR), slices);
}

void RichTextEdit::insert_horizontal_rule() {
	// A rule always sits alone on its own line: break out of any text before
	// the caret, and always leave the caret on a fresh line after it.
	const bool break_before = get_caret_column() > 0;
	LocalVector<Paragraph> slices;
	slices.resize(break_before ? 3 : 2);
	Paragraph &rule = slices[break_before ? 1 : 0];
	rule.length = 1;
	InlineObject object;
	object.is_image = false;
	object.raw_bbcode = "[hr]";
	rule.objects.push_back(object);
	_insert_objects((break_before ? String("\n") : String()) + String::chr(RichTextDocument::OBJECT_REPLACEMENT_CHAR) + "\n", slices);
}

Vector<uint64_t> RichTextEdit::_get_selected_image_ids() const {
	Vector<uint64_t> ids;
	int from_line = 0;
	int from_column = 0;
	int to_line = 0;
	int to_column = 0;
	if (_get_selection_range(from_line, from_column, to_line, to_column)) {
		for (int line = from_line; line <= to_line; line++) {
			const int from = line == from_line ? from_column : 0;
			const int to = line == to_line ? to_column : _get_line_length(line);
			for (const InlineObject &object : paragraphs[line].objects) {
				if (object.is_image && object.column >= from && object.column < to) {
					ids.push_back(object.id);
				}
			}
		}
		return ids;
	}

	if (selected_object_id != 0) {
		ids.push_back(selected_object_id);
		return ids;
	}

	const int line = get_caret_line();
	const int column = get_caret_column();
	for (int candidate : { column, column - 1 }) {
		const InlineObject *object = _get_object_at(line, candidate);
		if (object != nullptr && object->is_image) {
			ids.push_back(object->id);
			break;
		}
	}
	return ids;
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
	bar_panel->set_bg_color(Color::html("33373d"));
	bar_panel->set_border_color(Color::html("1a1c1f"));
	bar_panel->set_border_width_all(1);
	bar_panel->set_corner_radius_all(5);
	bar_panel->set_content_margin_individual(10, 10, 10, 10);
	bar_panel->set_shadow_color(Color(0, 0, 0, 0.5));
	bar_panel->set_shadow_size(12);
	bar_panel->set_shadow_offset(Vector2(0, 4));
	image_edit_bar->add_theme_style_override(SceneStringName(panel), bar_panel);

	VBoxContainer *bar = memnew(VBoxContainer);
	bar->add_theme_constant_override("separation", 8);
	image_edit_bar->add_child(bar);
	HBoxContainer *properties_row = memnew(HBoxContainer);
	properties_row->add_theme_constant_override("separation", 7);
	properties_row->set_alignment(BoxContainer::ALIGNMENT_BEGIN);
	bar->add_child(properties_row);

	// Keep the existing fill-width behavior available to the implementation,
	// but omit it from the visible design panel. The toolbar's image panel is
	// intentionally limited to width, aspect lock, height, reset, and alt text.
	image_fill_width_button = memnew(Button);
	image_fill_width_button->set_toggle_mode(true);
	image_fill_width_button->set_tooltip_text(RTR("Fill editor width while keeping image ratio"));
	image_fill_width_button->connect(SceneStringName(toggled), callable_mp(this, &RichTextEdit::_selected_image_fill_width_toggled));

	auto add_property_label = [](VBoxContainer *p_parent, Label *&r_label, const String &p_text, Node::InternalMode p_internal_mode) {
		MarginContainer *label_margin = memnew(MarginContainer);
		label_margin->add_theme_constant_override("margin_top", 4);
		label_margin->add_theme_constant_override("margin_bottom", 6);
		r_label = memnew(Label(p_text));
		label_margin->add_child(r_label, false, p_internal_mode);
		p_parent->add_child(label_margin, false, p_internal_mode);
	};

	VBoxContainer *width_field = memnew(VBoxContainer);
	width_field->add_theme_constant_override("separation", 0);
	properties_row->add_child(width_field, false, INTERNAL_MODE_FRONT);
	Label *width_label = nullptr;
	add_property_label(width_field, width_label, RTR("WIDTH"), INTERNAL_MODE_FRONT);

	image_width_spin = memnew(SpinBox);
	image_width_spin->set_min(1);
	image_width_spin->set_max(16384);
	image_width_spin->set_step(1);
	image_width_spin->set_custom_minimum_size(Size2(54, 28));
	width_field->add_child(image_width_spin, false, INTERNAL_MODE_FRONT);
	image_width_spin->connect(SceneStringName(value_changed), callable_mp(this, &RichTextEdit::_selected_image_width_changed));
	image_width_spin->connect(SceneStringName(gui_input), callable_mp(this, &RichTextEdit::_image_edit_control_gui_input));
	image_width_spin->get_line_edit()->connect(SceneStringName(gui_input), callable_mp(this, &RichTextEdit::_image_edit_control_gui_input));

	image_ratio_button = memnew(Button);
	if (has_theme_icon(SNAME("lock"), SNAME("RichTextEdit"))) {
		image_ratio_button->set_button_icon(get_theme_icon(SNAME("lock"), SNAME("RichTextEdit")));
	}
	image_ratio_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	image_ratio_button->set_vertical_icon_alignment(VERTICAL_ALIGNMENT_CENTER);
	image_ratio_button->set_toggle_mode(true);
	image_ratio_button->set_tooltip_text(RTR("Keep image width and height ratio"));
	image_ratio_button->set_custom_minimum_size(Size2(28, 28));
	image_ratio_button->set_h_size_flags(SIZE_SHRINK_CENTER);
	image_ratio_button->set_v_size_flags(SIZE_SHRINK_END);
	properties_row->add_child(image_ratio_button, false, INTERNAL_MODE_FRONT);
	image_ratio_button->connect(SceneStringName(toggled), callable_mp(this, &RichTextEdit::_selected_image_ratio_toggled));
	image_ratio_button->connect(SceneStringName(gui_input), callable_mp(this, &RichTextEdit::_image_edit_control_gui_input));

	VBoxContainer *height_field = memnew(VBoxContainer);
	height_field->add_theme_constant_override("separation", 0);
	properties_row->add_child(height_field, false, INTERNAL_MODE_FRONT);
	Label *height_label = nullptr;
	add_property_label(height_field, height_label, RTR("HEIGHT"), INTERNAL_MODE_FRONT);

	image_height_spin = memnew(SpinBox);
	image_height_spin->set_min(1);
	image_height_spin->set_max(16384);
	image_height_spin->set_step(1);
	image_height_spin->set_custom_minimum_size(Size2(54, 28));
	height_field->add_child(image_height_spin, false, INTERNAL_MODE_FRONT);
	image_height_spin->connect(SceneStringName(value_changed), callable_mp(this, &RichTextEdit::_selected_image_height_changed));
	image_height_spin->connect(SceneStringName(gui_input), callable_mp(this, &RichTextEdit::_image_edit_control_gui_input));
	image_height_spin->get_line_edit()->connect(SceneStringName(gui_input), callable_mp(this, &RichTextEdit::_image_edit_control_gui_input));

	image_reset_button = memnew(Button(RTR("Reset")));
	image_reset_button->set_tooltip_text(RTR("Restore the original image size"));
	image_reset_button->set_custom_minimum_size(Size2(50, 28));
	image_reset_button->set_v_size_flags(SIZE_SHRINK_END);
	properties_row->add_child(image_reset_button);
	image_reset_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEdit::_selected_image_reset));
	image_reset_button->connect(SceneStringName(gui_input), callable_mp(this, &RichTextEdit::_image_edit_control_gui_input));

	Label *alt_label = nullptr;
	add_property_label(bar, alt_label, RTR("ALT TEXT"), INTERNAL_MODE_BACK);
	image_alt_edit = memnew(LineEdit);
	image_alt_edit->set_placeholder(RTR("Describe this image"));
	image_alt_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	bar->add_child(image_alt_edit, false, INTERNAL_MODE_BACK);
	image_alt_edit->connect(SNAME("text_submitted"), callable_mp(this, &RichTextEdit::_apply_selected_image_details).unbind(1));
	image_alt_edit->connect(SNAME("focus_exited"), callable_mp(this, &RichTextEdit::_apply_selected_image_details));
	image_alt_edit->connect(SceneStringName(gui_input), callable_mp(this, &RichTextEdit::_image_edit_control_gui_input));

	auto make_panel_style = [](const Color &p_bg, const Color &p_border, int p_radius = 3, int p_margin_x = 6, int p_margin_y = 3) {
		Ref<StyleBoxFlat> style;
		style.instantiate();
		style->set_bg_color(p_bg);
		style->set_border_color(p_border);
		style->set_border_width_all(1);
		style->set_corner_radius_all(p_radius);
		style->set_content_margin_individual(p_margin_x, p_margin_y, p_margin_x, p_margin_y);
		return style;
	};
	const Color field_bg = Color::html("2c3035");
	const Color field_border = Color::html("1a1c1f");
	const Color button_normal = Color(0, 0, 0, 0);
	const Color button_hover = Color::html("42464d");
	const Color button_pressed = Color(0.37, 0.70, 0.94, 0.20);
	const Color text_color = Color::html("dfe1e4");
	const Color muted_color = Color::html("9a9ea3");
	Ref<StyleBoxFlat> button_normal_style = make_panel_style(button_normal, Color(0, 0, 0, 0), 3, 10, 5);
	Ref<StyleBoxFlat> button_hover_style = make_panel_style(button_hover, field_border, 3, 10, 5);
	Ref<StyleBoxFlat> button_pressed_style = make_panel_style(button_pressed, Color::html("5fb2f0"), 3, 10, 5);
	Ref<StyleBoxFlat> reset_normal_style = make_panel_style(button_normal, field_border, 3, 10, 5);
	for (Button *button : { image_ratio_button, image_reset_button }) {
		button->add_theme_style_override(CoreStringName(normal), button_normal_style);
		button->add_theme_style_override(SceneStringName(hover), button_hover_style);
		button->add_theme_style_override(SceneStringName(pressed), button_pressed_style);
		button->add_theme_color_override(SceneStringName(font_color), text_color);
		button->add_theme_color_override("font_hover_color", text_color);
		button->add_theme_color_override("font_pressed_color", Color::html("5fb2f0"));
	}
	Ref<StyleBoxFlat> ratio_normal_style = make_panel_style(button_normal, Color(0, 0, 0, 0), 3, 0, 0);
	Ref<StyleBoxFlat> ratio_hover_style = make_panel_style(button_hover, field_border, 3, 0, 0);
	Ref<StyleBoxFlat> ratio_pressed_style = make_panel_style(button_pressed, Color::html("5fb2f0"), 3, 0, 0);
	image_ratio_button->add_theme_style_override(CoreStringName(normal), ratio_normal_style);
	image_ratio_button->add_theme_style_override(SceneStringName(hover), ratio_hover_style);
	image_ratio_button->add_theme_style_override(SceneStringName(pressed), ratio_pressed_style);
	image_ratio_button->add_theme_style_override("hover_pressed", ratio_pressed_style);
	image_reset_button->add_theme_style_override(CoreStringName(normal), reset_normal_style);
	image_reset_button->add_theme_color_override(SceneStringName(font_color), muted_color);
	image_reset_button->add_theme_color_override("font_hover_color", text_color);
	image_reset_button->add_theme_color_override("font_pressed_color", text_color);
	image_reset_button->add_theme_font_size_override(SceneStringName(font_size), 11);
	for (SpinBox *spin : { image_width_spin, image_height_spin }) {
		spin->add_theme_constant_override("buttons_width", 16);
		spin->add_theme_constant_override("field_and_buttons_separation", 0);
		spin->add_theme_constant_override("set_min_buttons_width_from_icons", 0);
		spin->get_line_edit()->set_custom_minimum_size(Size2(38, 28));
		spin->get_line_edit()->add_theme_style_override(CoreStringName(normal), make_panel_style(field_bg, field_border, 3, 4, 3));
		spin->get_line_edit()->add_theme_style_override("focus", make_panel_style(field_bg, Color::html("5fb2f0"), 3, 4, 3));
		spin->get_line_edit()->add_theme_color_override(SceneStringName(font_color), text_color);
		spin->get_line_edit()->add_theme_font_size_override(SceneStringName(font_size), 12);
		spin->get_line_edit()->add_theme_constant_override("minimum_character_width", 0);
	}
	image_alt_edit->add_theme_style_override(CoreStringName(normal), make_panel_style(Color::html("232629"), field_border, 3, 8, 5));
	image_alt_edit->add_theme_style_override("focus", make_panel_style(Color::html("232629"), Color::html("5fb2f0"), 3, 8, 5));
	image_alt_edit->add_theme_color_override(SceneStringName(font_color), text_color);
	image_alt_edit->add_theme_color_override("font_placeholder_color", muted_color);
	image_alt_edit->add_theme_font_size_override(SceneStringName(font_size), 12);
	for (Label *label : { width_label, height_label }) {
		label->add_theme_color_override(SceneStringName(font_color), muted_color);
		label->add_theme_font_size_override(SceneStringName(font_size), 10);
	}
	alt_label->add_theme_color_override(SceneStringName(font_color), muted_color);
	alt_label->add_theme_font_size_override(SceneStringName(font_size), 10);

	image_details_popup = memnew(PopupPanel);
	add_child(image_details_popup, false, INTERNAL_MODE_FRONT);

	VBoxContainer *details = memnew(VBoxContainer);
	image_details_popup->add_child(details);

	details->add_child(memnew(Label(RTR("Alt Text"))));
	LineEdit *details_alt_edit = memnew(LineEdit);
	details_alt_edit->set_custom_minimum_size(Size2(260, 0));
	details->add_child(details_alt_edit);

	details->add_child(memnew(Label(RTR("Link"))));
	image_link_edit = memnew(LineEdit);
	image_link_edit->set_custom_minimum_size(Size2(260, 0));
	details->add_child(image_link_edit);
	image_link_edit->connect(SceneStringName(gui_input), callable_mp(this, &RichTextEdit::_image_edit_control_gui_input));

	Button *apply_button = memnew(Button(RTR("Apply")));
	details->add_child(apply_button);
	apply_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEdit::_apply_selected_image_details));
	apply_button->connect(SceneStringName(gui_input), callable_mp(this, &RichTextEdit::_image_edit_control_gui_input));
}

void RichTextEdit::_sync_selected_image_edit_controls() {
	if (image_edit_bar == nullptr || selected_object_id == 0) {
		return;
	}

	InlineObject *object = _find_object(selected_object_id);
	if (object == nullptr || !object->is_image) {
		_clear_selected_image();
		return;
	}

	const RichTextDocument::InlineImage &image = object->image;
	updating_image_edit_controls = true;
	const Size2 fallback_size(image_width_spin->get_value(), image_height_spin->get_value());
	const Size2 image_size = _get_image_option_size(image, fallback_size);
	image_width_spin->set_value(image_size.x);
	image_height_spin->set_value(image_size.y);
	image_alt_edit->set_text(image.options.has("alt") ? image.options["alt"] : String());
	image_link_edit->set_text(image.options.has("link") ? image.options["link"] : String());
	image_ratio_button->set_pressed_no_signal(!image.options.has("_keep_ratio") || image.options["_keep_ratio"] == "true");
	image_fill_width_button->set_pressed_no_signal(image.options.has("_fill_width") && image.options["_fill_width"] == "true");
	updating_image_edit_controls = false;
}

void RichTextEdit::_image_edit_control_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> key = p_event;
	if (key.is_null() || !key->is_pressed() || key->is_echo()) {
		return;
	}

	if (key->is_action("ui_undo", true)) {
		undo();
		accept_event();
	} else if (key->is_action("ui_redo", true)) {
		redo();
		accept_event();
	} else if (key->is_action("ui_cancel", true)) {
		_clear_selected_image();
		accept_event();
	}
}

void RichTextEdit::_select_image(uint64_t p_id, const Rect2 &p_screen_rect) {
	InlineObject *object = _find_object(p_id);
	if (object == nullptr || !object->is_image) {
		_clear_selected_image();
		return;
	}
	selected_object_id = p_id;
	_ensure_image_edit_controls();
	object = _find_object(p_id);

	const RichTextDocument::InlineImage &image = object->image;
	updating_image_edit_controls = true;
	const Size2 image_size = _get_image_option_size(image, p_screen_rect.size);
	image_width_spin->set_value(image_size.x);
	image_height_spin->set_value(image_size.y);
	image_alt_edit->set_text(image.options.has("alt") ? image.options["alt"] : String());
	image_link_edit->set_text(image.options.has("link") ? image.options["link"] : String());
	image_ratio_button->set_pressed_no_signal(!image.options.has("_keep_ratio") || image.options["_keep_ratio"] == "true");
	image_fill_width_button->set_pressed_no_signal(image.options.has("_fill_width") && image.options["_fill_width"] == "true");
	updating_image_edit_controls = false;

	// The design keeps the property panel at the same fixed width as the image
	// demo (320px) and aligns its left edge with the selected image.
	image_edit_bar->set_custom_minimum_size(Size2(320, 0));
	image_edit_bar->reset_size();
	Size2 bar_size = image_edit_bar->get_size();
	if (bar_size.x <= 0.0f || bar_size.y <= 0.0f) {
		bar_size = image_edit_bar->get_combined_minimum_size();
		image_edit_bar->set_size(bar_size);
	}

	const Rect2 local_image_rect(p_screen_rect.position - get_screen_position(), p_screen_rect.size);
	Point2 bar_position(local_image_rect.position.x, local_image_rect.position.y + local_image_rect.size.y + 8.0f);
	bar_position.x = CLAMP(bar_position.x, 0.0f, MAX(0.0f, get_size().x - bar_size.x));
	if (bar_position.y + bar_size.y > get_size().y) {
		bar_position.y = local_image_rect.position.y - bar_size.y - 4.0f;
	}
	bar_position.y = CLAMP(bar_position.y, 0.0f, MAX(0.0f, get_size().y - bar_size.y));
	image_edit_bar->set_position(bar_position);
	image_edit_bar->show();
	_sync_selected_image_edit_controls();
	queue_redraw();
}

void RichTextEdit::_clear_selected_image() {
	selected_object_id = 0;
	if (image_edit_bar != nullptr) {
		image_edit_bar->hide();
	}
	if (image_details_popup != nullptr) {
		image_details_popup->hide();
	}
	queue_redraw();
}

void RichTextEdit::_commit_selected_image_options(const String &p_width, const String &p_height, const String &p_alt, const String &p_link, bool p_record_undo) {
	int line = 0;
	InlineObject *object = _find_object(selected_object_id, &line);
	if (object == nullptr) {
		_clear_selected_image();
		return;
	}
	const Ref<StyleState> before = _capture_style_state(line, line);
	object = _find_object(selected_object_id);
	HashMap<String, String> &options = object->image.options;

	options.erase("size");
	if (p_width.is_empty()) {
		options.erase("width");
	} else {
		options["width"] = p_width;
	}
	if (p_height.is_empty()) {
		options.erase("height");
	} else {
		options["height"] = p_height;
	}
	if (p_alt.is_empty()) {
		options.erase("alt");
	} else {
		options["alt"] = p_alt;
	}
	if (p_link.is_empty()) {
		options.erase("link");
	} else {
		options["link"] = p_link;
	}

	if (p_record_undo) {
		_push_style_undo(before, true);
	}
	_model_changed(line, line);
}

void RichTextEdit::_selected_image_width_changed(double p_value) {
	if (updating_image_edit_controls || selected_object_id == 0) {
		return;
	}
	if (image_ratio_button != nullptr && image_ratio_button->is_pressed()) {
		InlineObject *object = _find_object(selected_object_id);
		if (object != nullptr) {
			const Size2 current_size = _get_image_option_size(object->image, Size2(image_width_spin->get_value(), image_height_spin->get_value()));
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
	if (updating_image_edit_controls || selected_object_id == 0) {
		return;
	}
	if (image_ratio_button != nullptr && image_ratio_button->is_pressed()) {
		InlineObject *object = _find_object(selected_object_id);
		if (object != nullptr) {
			const Size2 current_size = _get_image_option_size(object->image, Size2(image_width_spin->get_value(), image_height_spin->get_value()));
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
	if (updating_image_edit_controls || selected_object_id == 0) {
		return;
	}
	int line = 0;
	InlineObject *object = _find_object(selected_object_id, &line);
	if (object == nullptr) {
		_clear_selected_image();
		return;
	}
	const Ref<StyleState> before = _capture_style_state(line, line);
	object = _find_object(selected_object_id);

	if (p_pressed) {
		object->image.options.erase("_keep_ratio"); // Kept ratio is the default; absence means on.
	} else {
		object->image.options["_keep_ratio"] = "false";
	}
	_commit_selected_image_options(String::num_real(image_width_spin->get_value(), false), String::num_real(image_height_spin->get_value(), false), image_alt_edit->get_text(), image_link_edit->get_text(), false);
	_push_style_undo(before, true);
}

void RichTextEdit::_selected_image_fill_width_toggled(bool p_pressed) {
	if (updating_image_edit_controls || selected_object_id == 0) {
		return;
	}
	int line = 0;
	InlineObject *object = _find_object(selected_object_id, &line);
	if (object == nullptr) {
		_clear_selected_image();
		return;
	}
	const Ref<StyleState> before = _capture_style_state(line, line);
	object = _find_object(selected_object_id);
	HashMap<String, String> &options = object->image.options;

	Size2 target_size;
	if (p_pressed) {
		const Size2 current_size = _get_image_option_size(object->image, Size2(image_width_spin->get_value(), image_height_spin->get_value()));
		options["_fill_width"] = "true";
		options["_previous_width"] = String::num_real(current_size.x, false);
		options["_previous_height"] = String::num_real(current_size.y, false);
		const float fill_width = _get_fill_image_width();
		target_size = Size2(fill_width, MAX(1.0f, fill_width * current_size.y / current_size.x));
	} else {
		options.erase("_fill_width");
		if (options.has("_previous_width") && options.has("_previous_height")) {
			target_size = Size2(options["_previous_width"].to_float(), options["_previous_height"].to_float());
		} else {
			target_size = _get_image_option_size(object->image, Size2(image_width_spin->get_value(), image_height_spin->get_value()));
		}
		options.erase("_previous_width");
		options.erase("_previous_height");
	}

	updating_image_edit_controls = true;
	image_width_spin->set_value(MAX(1.0f, target_size.x));
	image_height_spin->set_value(MAX(1.0f, target_size.y));
	updating_image_edit_controls = false;
	_commit_selected_image_options(String::num_real(image_width_spin->get_value(), false), String::num_real(image_height_spin->get_value(), false), image_alt_edit->get_text(), image_link_edit->get_text(), false);
	_push_style_undo(before, true);
}

void RichTextEdit::_selected_image_reset() {
	if (selected_object_id == 0) {
		return;
	}
	_commit_selected_image_options(String(), String(), image_alt_edit != nullptr ? image_alt_edit->get_text() : String(), image_link_edit != nullptr ? image_link_edit->get_text() : String());
	InlineObject *object = _find_object(selected_object_id);
	if (object != nullptr) {
		const Size2 natural_size = _get_image_option_size(object->image, Size2(1, 1));
		updating_image_edit_controls = true;
		image_width_spin->set_value(natural_size.x);
		image_height_spin->set_value(natural_size.y);
		image_ratio_button->set_pressed_no_signal(!object->image.options.has("_keep_ratio") || object->image.options["_keep_ratio"] == "true");
		image_fill_width_button->set_pressed_no_signal(false);
		updating_image_edit_controls = false;
	}
}

void RichTextEdit::_pressed_selected_image_details() {
	if (selected_object_id == 0 || image_details_popup == nullptr) {
		return;
	}
	image_details_popup->set_position(get_screen_position() + image_edit_bar->get_position() + Vector2(0, image_edit_bar->get_size().y + 4));
	image_details_popup->reset_size();
	image_details_popup->popup();
}

void RichTextEdit::_apply_selected_image_details() {
	if (selected_object_id == 0) {
		return;
	}
	_commit_selected_image_options(String::num_real(image_width_spin->get_value(), false), String::num_real(image_height_spin->get_value(), false), image_alt_edit->get_text(), image_link_edit->get_text());
	if (image_details_popup != nullptr) {
		image_details_popup->hide();
	}
}

void RichTextEdit::set_selection_image_size(int p_width, int p_height) {
	if (p_width < 0 && p_height < 0) {
		return;
	}

	const Vector<uint64_t> ids = _get_selected_image_ids();
	int first_line = INT_MAX;
	int last_line = -1;
	for (uint64_t id : ids) {
		int line = 0;
		if (_find_object(id, &line) != nullptr) {
			first_line = MIN(first_line, line);
			last_line = MAX(last_line, line);
		}
	}
	if (last_line < 0) {
		return;
	}

	const Ref<StyleState> before = _capture_style_state(first_line, last_line);
	for (uint64_t id : ids) {
		InlineObject *object = _find_object(id);
		if (object == nullptr) {
			continue;
		}
		HashMap<String, String> &options = object->image.options;
		options.erase("size");
		if (p_width >= 0) {
			options["width"] = itos(p_width);
		} else {
			options.erase("width");
		}
		if (p_height >= 0) {
			options["height"] = itos(p_height);
		} else {
			options.erase("height");
		}
	}
	_push_style_undo(before);
	_model_changed(first_line, last_line);
}

bool RichTextEdit::_is_horizontal_rule_bbcode(const String &p_bbcode) {
	return p_bbcode == "[hr]" || p_bbcode.begins_with("[hr ") || p_bbcode.begins_with("[hr=");
}

Dictionary RichTextEdit::_make_horizontal_rule_inline_info(const InlineObject &p_object) const {
	Dictionary info;
	info["column"] = p_object.column;
	info["object_id"] = (int64_t)p_object.id;
	info["cursor_shape"] = CURSOR_ARROW;
	info["horizontal_rule"] = true;
	info["width"] = MAX(1.0f, _get_fill_image_width());
	const int margin_top = get_theme_constant(SNAME("divider_margin_top"), SNAME("RichTextEdit"));
	const int margin_bottom = get_theme_constant(SNAME("divider_margin_bottom"), SNAME("RichTextEdit"));
	const int width = get_theme_constant(SNAME("divider_width"), SNAME("RichTextEdit"));
	info["height"] = MAX(1, margin_top + MAX(1, width) + margin_bottom);
	return info;
}

Array RichTextEdit::_get_line_inline_objects(int p_line, const String &p_line_text) const {
	Array result;
	if (p_line < 0 || p_line >= (int)paragraphs.size()) {
		return result;
	}
	for (const InlineObject &object : paragraphs[p_line].objects) {
		if (object.column >= p_line_text.length() || p_line_text[object.column] != RichTextDocument::OBJECT_REPLACEMENT_CHAR) {
			continue;
		}
		if (object.is_image) {
			result.push_back(_make_image_inline_info(object));
		} else if (_is_horizontal_rule_bbcode(object.raw_bbcode)) {
			result.push_back(_make_horizontal_rule_inline_info(object));
		}
	}
	return result;
}

void RichTextEdit::_draw_inline_object(const Dictionary &p_info, const Rect2 &p_rect) {
	RID text_canvas_item = get_text_canvas_item();
	if (p_info.get("horizontal_rule", false)) {
		const Color divider_color = get_theme_color(SNAME("divider_color"), SNAME("RichTextEdit"));
		const int margin_top = get_theme_constant(SNAME("divider_margin_top"), SNAME("RichTextEdit"));
		const int divider_width = MAX(1, get_theme_constant(SNAME("divider_width"), SNAME("RichTextEdit")));
		RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(p_rect.position.x, p_rect.position.y + margin_top, p_rect.size.x, divider_width), divider_color);
		return;
	}
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

	const InlineObject *object = _get_object_at(p_info.get("line", -1), p_info.get("column", -1));
	if (object == nullptr || !object->is_image) {
		draw_placeholder();
		return;
	}
	const RichTextDocument::InlineImage &image = object->image;
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
	if (selected_object_id == object->id) {
		const Color selection_color = Color(0.25, 0.55, 1.0, 1.0);
		RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(draw_rect.position, Size2(draw_rect.size.x, 2)), selection_color);
		RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(Point2(draw_rect.position.x, draw_rect.position.y + draw_rect.size.y - 2), Size2(draw_rect.size.x, 2)), selection_color);
		RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(draw_rect.position, Size2(2, draw_rect.size.y)), selection_color);
		RS::get_singleton()->canvas_item_add_rect(text_canvas_item, Rect2(Point2(draw_rect.position.x + draw_rect.size.x - 2, draw_rect.position.y), Size2(2, draw_rect.size.y)), selection_color);
	}
}

void RichTextEdit::_inline_object_clicked(const Dictionary &p_info, const Rect2 &p_rect) {
	if (p_info.get("horizontal_rule", false)) {
		return;
	}
	const uint64_t id = (int64_t)p_info.get("object_id", 0);
	if (id != 0) {
		_select_image(id, p_rect);
	}
}

bool RichTextEdit::_is_text_clipping_enabled() const {
	// Content only draws outside the control when both axes are visible.
	return overflow_x != OVERFLOW_VISIBLE || overflow_y != OVERFLOW_VISIBLE;
}

TextEdit::ScrollBarMode RichTextEdit::_get_axis_scroll_bar_mode(Overflow p_axis, Overflow p_other_axis) {
	// Like CSS, a visible axis next to a clipped one computes to auto.
	if (p_axis == OVERFLOW_VISIBLE && p_other_axis != OVERFLOW_VISIBLE) {
		return SCROLL_BAR_MODE_AUTO;
	}
	switch (p_axis) {
		case OVERFLOW_VISIBLE:
		case OVERFLOW_HIDDEN:
			return SCROLL_BAR_MODE_NEVER;
		case OVERFLOW_SCROLL:
			return SCROLL_BAR_MODE_ALWAYS;
		case OVERFLOW_AUTO:
			break;
	}
	return SCROLL_BAR_MODE_AUTO;
}

TextEdit::ScrollBarMode RichTextEdit::_get_h_scroll_bar_mode() const {
	return _get_axis_scroll_bar_mode(overflow_x, overflow_y);
}

TextEdit::ScrollBarMode RichTextEdit::_get_v_scroll_bar_mode() const {
	return _get_axis_scroll_bar_mode(overflow_y, overflow_x);
}

void RichTextEdit::set_overflow_x(Overflow p_overflow) {
	ERR_FAIL_INDEX((int)p_overflow, OVERFLOW_SCROLL + 1);
	if (overflow_x == p_overflow) {
		return;
	}
	overflow_x = p_overflow;
	_update_text_clipping();
}

RichTextEdit::Overflow RichTextEdit::get_overflow_x() const {
	return overflow_x;
}

void RichTextEdit::set_overflow_y(Overflow p_overflow) {
	ERR_FAIL_INDEX((int)p_overflow, OVERFLOW_SCROLL + 1);
	if (overflow_y == p_overflow) {
		return;
	}
	overflow_y = p_overflow;
	_update_text_clipping();
}

RichTextEdit::Overflow RichTextEdit::get_overflow_y() const {
	return overflow_y;
}

int RichTextEdit::get_content_height() const {
	return _get_visible_content_height();
}

void RichTextEdit::_notification(int p_what) {
	if (p_what == NOTIFICATION_MOUSE_EXIT && meta_hovering) {
		meta_hovering = false;
		emit_signal(SNAME("meta_hover_ended"), current_meta);
		current_meta = String();
		_clear_style_render_infos();
	}
	if (p_what == NOTIFICATION_MOUSE_EXIT && !active_meta.is_empty()) {
		active_meta = String();
		_clear_style_render_infos();
		queue_redraw();
	}
}

void RichTextEdit::gui_input(const Ref<InputEvent> &p_gui_input) {
	ERR_FAIL_COND(p_gui_input.is_null());

	Ref<InputEventKey> key = p_gui_input;
	if (key.is_valid() && key->is_pressed() && !key->is_echo() && key->is_action("ui_copy", true)) {
		copy_rich_text();
		accept_event();
		return;
	}
	if (key.is_valid() && key->is_pressed() && !key->is_echo() && key->is_action("ui_paste", true) && paste_rich_text()) {
		accept_event();
		return;
	}
	if (key.is_valid() && key->is_pressed() && !key->is_echo() && !key->is_command_or_control_pressed() && !key->is_alt_pressed() && key->is_action("ui_text_newline", true)) {
		_insert_newline(key->is_shift_pressed());
		return;
	}
	if (key.is_valid() && key->is_pressed() && !key->is_echo() && key->is_action("ui_cancel", true) && selected_object_id != 0) {
		_clear_selected_image();
		accept_event();
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
	} else if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT && selected_object_id != 0) {
		_clear_selected_image();
	} else if (mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT && !active_meta.is_empty()) {
		active_meta = String();
		_clear_style_render_infos();
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

void RichTextEdit::set_visible_characters(int p_visible) {
	_set_visible_characters(p_visible);
}

int RichTextEdit::get_visible_characters() const {
	return _get_visible_characters();
}

void RichTextEdit::set_visible_ratio(float p_ratio) {
	_set_visible_ratio(p_ratio);
}

float RichTextEdit::get_visible_ratio() const {
	return _get_visible_ratio();
}

void RichTextEdit::set_visible_characters_behavior(TextServer::VisibleCharactersBehavior p_behavior) {
	_set_visible_characters_behavior(p_behavior);
}

TextServer::VisibleCharactersBehavior RichTextEdit::get_visible_characters_behavior() const {
	return _get_visible_characters_behavior();
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

void RichTextEdit::set_overline() {
	_apply_style_property_to_selection(STYLE_PROPERTY_OVERLINE, true);
}

void RichTextEdit::clear_overline() {
	_apply_style_property_to_selection(STYLE_PROPERTY_OVERLINE, false);
}

void RichTextEdit::toggle_overline() {
	_apply_style_property_to_selection(STYLE_PROPERTY_OVERLINE);
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

void RichTextEdit::clear_selection_url() {
	set_selection_url(String());
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

void RichTextEdit::set_selection_line_spacing(int p_spacing) {
	set_selection_line_height(itos(MAX(0, p_spacing)) + "px");
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

void RichTextEdit::increase_indent() {
	_apply_block_tag_to_selected_lines("indent");
}

void RichTextEdit::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "syntax_highlighter") {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void RichTextEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_use_bbcode", "enable"), &RichTextEdit::set_use_bbcode);
	ClassDB::bind_method(D_METHOD("is_using_bbcode"), &RichTextEdit::is_using_bbcode);
	ClassDB::bind_method(D_METHOD("set_overflow_x", "overflow"), &RichTextEdit::set_overflow_x);
	ClassDB::bind_method(D_METHOD("get_overflow_x"), &RichTextEdit::get_overflow_x);
	ClassDB::bind_method(D_METHOD("set_overflow_y", "overflow"), &RichTextEdit::set_overflow_y);
	ClassDB::bind_method(D_METHOD("get_overflow_y"), &RichTextEdit::get_overflow_y);
	ClassDB::bind_method(D_METHOD("get_content_height"), &RichTextEdit::get_content_height);
	ClassDB::bind_method(D_METHOD("has_pending_layout"), &RichTextEdit::has_pending_layout);
	ClassDB::bind_method(D_METHOD("set_link_activation_mode", "mode"), &RichTextEdit::set_link_activation_mode);
	ClassDB::bind_method(D_METHOD("get_link_activation_mode"), &RichTextEdit::get_link_activation_mode);
	ClassDB::bind_method(D_METHOD("set_bbcode_text", "text"), &RichTextEdit::set_bbcode_text);
	ClassDB::bind_method(D_METHOD("set_style_font_resolver", "resolver"), &RichTextEdit::set_style_font_resolver);
	ClassDB::bind_method(D_METHOD("get_style_font_resolver"), &RichTextEdit::get_style_font_resolver);
	ClassDB::bind_method(D_METHOD("get_bbcode_text"), &RichTextEdit::get_bbcode_text);
	ClassDB::bind_method(D_METHOD("set_visible_characters", "amount"), &RichTextEdit::set_visible_characters);
	ClassDB::bind_method(D_METHOD("get_visible_characters"), &RichTextEdit::get_visible_characters);
	ClassDB::bind_method(D_METHOD("set_visible_characters_behavior", "behavior"), &RichTextEdit::set_visible_characters_behavior);
	ClassDB::bind_method(D_METHOD("get_visible_characters_behavior"), &RichTextEdit::get_visible_characters_behavior);
	ClassDB::bind_method(D_METHOD("set_visible_ratio", "ratio"), &RichTextEdit::set_visible_ratio);
	ClassDB::bind_method(D_METHOD("get_visible_ratio"), &RichTextEdit::get_visible_ratio);
	ClassDB::bind_static_method("RichTextEdit", D_METHOD("get_style_template_values", "template"), &RichTextEdit::get_style_template_values);
	ClassDB::bind_method(D_METHOD("set_decoration_paint", "paint"), &RichTextEdit::set_decoration_paint);
	ClassDB::bind_method(D_METHOD("set_outline_over_fill", "enabled"), &RichTextEdit::set_outline_over_fill);
	ClassDB::bind_method(D_METHOD("is_outline_over_fill"), &RichTextEdit::is_outline_over_fill);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "outline_over_fill"), "set_outline_over_fill", "is_outline_over_fill");
	ClassDB::bind_method(D_METHOD("get_selection_style_template"), &RichTextEdit::get_selection_style_template);
	ClassDB::bind_method(D_METHOD("apply_style_template", "template", "overrides"), &RichTextEdit::apply_style_template, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("copy_rich_text"), &RichTextEdit::copy_rich_text);
	ClassDB::bind_method(D_METHOD("paste_rich_text"), &RichTextEdit::paste_rich_text);
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
	ClassDB::bind_method(D_METHOD("set_overline"), &RichTextEdit::set_overline);
	ClassDB::bind_method(D_METHOD("clear_overline"), &RichTextEdit::clear_overline);
	ClassDB::bind_method(D_METHOD("toggle_overline"), &RichTextEdit::toggle_overline);
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
	ClassDB::bind_method(D_METHOD("set_selection_line_height", "line_height"), &RichTextEdit::set_selection_line_height);
	ClassDB::bind_method(D_METHOD("clear_selection_line_height"), &RichTextEdit::clear_selection_line_height);
	ClassDB::bind_method(D_METHOD("get_current_line_height"), &RichTextEdit::get_current_line_height);
	ClassDB::bind_method(D_METHOD("set_selection_line_spacing", "spacing"), &RichTextEdit::set_selection_line_spacing);
	ClassDB::bind_method(D_METHOD("insert_image", "source", "width", "height", "alt"), &RichTextEdit::insert_image, DEFVAL(-1), DEFVAL(-1), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("set_selection_image_size", "width", "height"), &RichTextEdit::set_selection_image_size);
	ClassDB::bind_method(D_METHOD("insert_horizontal_rule"), &RichTextEdit::insert_horizontal_rule);
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

	BIND_ENUM_CONSTANT(OVERFLOW_AUTO);
	BIND_ENUM_CONSTANT(OVERFLOW_VISIBLE);
	BIND_ENUM_CONSTANT(OVERFLOW_HIDDEN);
	BIND_ENUM_CONSTANT(OVERFLOW_SCROLL);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bbcode_enabled"), "set_use_bbcode", "is_using_bbcode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "link_activation_mode", PROPERTY_HINT_ENUM, "Auto,Ctrl Click,Click,Disabled"), "set_link_activation_mode", "get_link_activation_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bbcode_text", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_bbcode_text", "get_bbcode_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "overflow_x", PROPERTY_HINT_ENUM, "Auto,Visible,Hidden,Scroll"), "set_overflow_x", "get_overflow_x");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "overflow_y", PROPERTY_HINT_ENUM, "Auto,Visible,Hidden,Scroll"), "set_overflow_y", "get_overflow_y");
	// These properties must follow text/bbcode_text so their values are applied
	// against the final parsed character count when a scene is restored.
	ADD_PROPERTY(PropertyInfo(Variant::INT, "visible_characters", PROPERTY_HINT_RANGE, "-1,128000,1"), "set_visible_characters", "get_visible_characters");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "visible_characters_behavior", PROPERTY_HINT_ENUM, "Characters Before Shaping,Characters After Shaping,Glyphs (Layout Direction),Glyphs (Left-to-Right),Glyphs (Right-to-Left)"), "set_visible_characters_behavior", "get_visible_characters_behavior");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "visible_ratio", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_visible_ratio", "get_visible_ratio");

	ADD_SIGNAL(MethodInfo("meta_clicked", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("meta_hover_started", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("meta_hover_ended", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("text_style_changed"));
	ADD_SIGNAL(MethodInfo("content_height_changed"));

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, RichTextEdit, tooltip_panel);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, RichTextEdit, tooltip_font_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT, RichTextEdit, tooltip_font);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT_SIZE, RichTextEdit, tooltip_font_size);

	// The run styles read these items; _update_theme_item_cache() reshapes
	// the document when one of them actually changes.
	const auto registered_only = [](Node *, const StringName &, const StringName &) {};
	for (const char *name : STYLE_THEME_FONTS) {
		ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_FONT, get_class_static(), name, name, registered_only);
	}
	for (const char *name : STYLE_THEME_COLORS) {
		ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_COLOR, get_class_static(), name, name, registered_only);
	}
	for (const char *name : STYLE_THEME_CONSTANTS) {
		ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_CONSTANT, get_class_static(), name, name, registered_only);
	}
}

RichTextEdit::RichTextEdit() {
	styles.push_back(TextStyle());
	style_render_infos.push_back(Dictionary());
	style_render_info_valid.push_back(true);
	paragraphs.resize(1);

	// The parsers are installed once; edits reshape only the lines they touch,
	// and bulk changes of long documents shape lines as they are drawn.
	_set_deferred_shaping_enabled(true);
	set_style_parser(callable_mp(this, &RichTextEdit::_get_line_style_spans));
	set_inline_object_handlers(callable_mp(this, &RichTextEdit::_get_line_inline_objects), callable_mp(this, &RichTextEdit::_draw_inline_object), callable_mp(this, &RichTextEdit::_inline_object_clicked));
	set_deselect_on_focus_loss_enabled(false);

	connect(SceneStringName(text_changed), callable_mp(this, &RichTextEdit::_update_bbcode_from_text));
	connect("lines_edited_from", callable_mp(this, &RichTextEdit::_lines_edited));
	connect("caret_changed", callable_mp(this, &RichTextEdit::_caret_style_context_changed));
}
