/**************************************************************************/
/*  rich_text_edit.h                                                      */
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

#include "core/templates/hash_set.h"
#include "scene/gui/rich_text_document.h"
#include "scene/gui/text_edit.h"

class Button;
class LineEdit;
class PanelContainer;
class PopupPanel;
class SpinBox;

class RichTextEdit : public TextEdit {
	GDCLASS(RichTextEdit, TextEdit);

public:
	enum LinkActivationMode {
		LINK_ACTIVATION_AUTO,
		LINK_ACTIVATION_CTRL_CLICK,
		LINK_ACTIVATION_CLICK,
		LINK_ACTIVATION_DISABLED,
	};

	enum Overflow {
		OVERFLOW_AUTO,
		OVERFLOW_VISIBLE,
		OVERFLOW_HIDDEN,
		OVERFLOW_SCROLL,
	};

	using TextStyle = RichTextDocument::Style;

	struct StyleSpan {
		int from = 0;
		int to = 0;
		TextStyle style;

		bool operator==(const StyleSpan &p_other) const { return from == p_other.from && to == p_other.to && style == p_other.style; }
		bool operator!=(const StyleSpan &p_other) const { return !(*this == p_other); }
	};

	// A run of styled characters inside one paragraph. Columns are relative
	// to the paragraph; the style is an id into the editor's style table and
	// is never 0, the default style.
	struct StyleRun {
		int from = 0;
		int to = 0;
		int style = 0;

		bool operator==(const StyleRun &p_other) const { return from == p_other.from && to == p_other.to && style == p_other.style; }
		bool operator!=(const StyleRun &p_other) const { return !(*this == p_other); }
	};

	// An image or raw inline element sitting on an object replacement
	// character of its paragraph.
	struct InlineObject {
		int column = 0;
		uint64_t id = 0;
		bool is_image = true;
		RichTextDocument::InlineImage image;
		String raw_bbcode;
	};

	// The formatting of one TextEdit line. The paragraph break that follows a
	// line carries its own style, like a word processor's paragraph mark, so
	// empty paragraphs keep their list, quote and alignment.
	struct Paragraph {
		// Text length of a fragment slice; live paragraphs read TextEdit.
		int length = 0;
		LocalVector<StyleRun> runs;
		int eol_style = 0;
		LocalVector<InlineObject> objects;
		String line_height;
		// Parsed line_height: 0 = normal, 1 = px, 2 = percent, 3 = multiplier.
		uint8_t line_height_kind = 0;
		double line_height_value = 0.0;
	};

	// The paragraphs a style change touched, captured for undo.
	class StyleState : public RefCounted {
		GDSOFTCLASS(StyleState, RefCounted);

	public:
		int first_line = 0;
		LocalVector<Paragraph> paragraphs;
		TextStyle typing_style;
		bool typing_style_override = false;
	};

private:
	Dictionary decoration_paint;

	// Formatting of text removed by an edit, so undo and redo can restore it
	// when TextEdit puts the same text back at the same place.
	struct RemovedFragment {
		int line = 0;
		int column = 0;
		String text;
		LocalVector<Paragraph> slices;
	};

	enum StyleProperty {
		STYLE_PROPERTY_BOLD,
		STYLE_PROPERTY_ITALIC,
		STYLE_PROPERTY_UNDERLINE,
		STYLE_PROPERTY_STRIKETHROUGH,
		STYLE_PROPERTY_OVERLINE,
		STYLE_PROPERTY_COLOR,
		STYLE_PROPERTY_CLEAR_COLOR,
		STYLE_PROPERTY_BG_COLOR,
		STYLE_PROPERTY_CLEAR_BG_COLOR,
		STYLE_PROPERTY_OUTLINE_COLOR,
		STYLE_PROPERTY_CLEAR_OUTLINE_COLOR,
		STYLE_PROPERTY_OUTLINE_SIZE,
		STYLE_PROPERTY_CLEAR_OUTLINE_SIZE,
		STYLE_PROPERTY_FONT_SIZE,
		STYLE_PROPERTY_CLEAR_FONT_SIZE,
		STYLE_PROPERTY_FONT,
		STYLE_PROPERTY_CLEAR_FONT,
		STYLE_PROPERTY_URL,
		STYLE_PROPERTY_CLEAR_URL,
		STYLE_PROPERTY_URL_TOOLTIP,
		STYLE_PROPERTY_CLEAR_URL_TOOLTIP,
		STYLE_PROPERTY_URL_VISITED,
		STYLE_PROPERTY_CLEAR_URL_VISITED,
		STYLE_PROPERTY_CODE,
		STYLE_PROPERTY_BLOCK_TAG,
	};

	// Line-aligned document model: paragraphs[i] formats TextEdit line i.
	LocalVector<Paragraph> paragraphs;
	// Interned styles; style 0 is the default style.
	LocalVector<TextStyle> styles;
	HashMap<uint32_t, LocalVector<int>> style_ids_by_hash;
	mutable LocalVector<Dictionary> style_render_infos;
	mutable LocalVector<bool> style_render_info_valid;
	uint64_t next_object_id = 1;
	uint64_t model_revision = 1;

	// Ordered-list counters per line boundary (four list types), extended on
	// demand and valid for boundaries [0, list_numbers_valid].
	mutable LocalVector<int> list_counters;
	mutable int list_numbers_valid = 0;

	LocalVector<RemovedFragment> removed_fragments;
	int removed_fragments_chars = 0;
	bool pending_fragment_active = false;
	bool pending_fragment_inherit_style = false;
	String pending_fragment_text;
	LocalVector<Paragraph> pending_fragment;

	mutable String bbcode_text;
	mutable uint64_t bbcode_revision = 0;
	mutable LocalVector<StyleSpan> style_spans_cache;
	mutable uint64_t style_spans_revision = 0;
	bool setting_bbcode_text = false;
	bool use_bbcode = false;
	Callable style_font_resolver;
	Overflow overflow_x = OVERFLOW_AUTO;
	Overflow overflow_y = OVERFLOW_AUTO;
	LinkActivationMode link_activation_mode = LINK_ACTIVATION_AUTO;
	bool text_style_changed_dirty = false;
	bool typing_style_override = false;
	bool style_preview_active = false;
	bool style_preview_had_selection = false;
	Ref<StyleState> style_preview_before;
	// Tracks a run of repeated edits of one property (the toolbar's number fields
	// are spun in bursts) so the whole run collapses into a single undo step.
	bool style_run_active = false;
	StyleProperty style_run_property = STYLE_PROPERTY_BOLD;
	Vector4i style_run_range;
	Ref<StyleState> style_run_before;
	// Monotonically increasing id handed out to each quote-apply gesture; see
	// Style::block_group and _apply_block_tag_to_selected_lines.
	int next_block_group_id = 0;
	bool meta_hovering = false;
	bool updating_image_edit_controls = false;
	HashSet<String> visited_links;
	uint64_t selected_object_id = 0;
	String current_meta;
	String active_meta;
	TextStyle typing_style;
	PanelContainer *image_edit_bar = nullptr;
	PopupPanel *image_details_popup = nullptr;
	SpinBox *image_width_spin = nullptr;
	SpinBox *image_height_spin = nullptr;
	Button *image_ratio_button = nullptr;
	Button *image_fill_width_button = nullptr;
	Button *image_reset_button = nullptr;
	LineEdit *image_alt_edit = nullptr;
	LineEdit *image_link_edit = nullptr;

	struct ThemeCache {
		Ref<StyleBox> tooltip_panel;
		Color tooltip_font_color;
		Ref<Font> tooltip_font;
		int tooltip_font_size = 0;
	} theme_cache;
	// Theme items the per-run render information depends on.
	HashMap<StringName, Variant> style_theme_values;
	bool style_refresh_queued = false;

	static bool _is_url_auto_open_allowed(const String &p_url);
	static uint32_t _hash_style(const TextStyle &p_style);
	static void _set_paragraph_line_height(Paragraph &r_paragraph, const String &p_line_height);

	// Style table.
	int _intern_style(const TextStyle &p_style);
	const TextStyle &_get_style(int p_style) const { return styles[p_style]; }
	Dictionary _get_style_render_info(int p_style) const;
	void _clear_style_render_infos();

	// Model access.
	void _reset_model();
	void _ensure_model_size();
	int _get_line_length(int p_line) const { return get_line(p_line).length(); }
	int _get_style_id_at(int p_line, int p_column) const;
	int _get_insertion_style_id(int p_line, int p_column) const;
	static void _normalize_runs(LocalVector<StyleRun> &r_runs);
	static void _set_runs_range(LocalVector<StyleRun> &r_runs, int p_from, int p_to, int p_style);
	void _model_changed(int p_from_line, int p_to_line, bool p_invalidate = true);
	int _get_line_start(int p_line) const;
	void _offset_to_line_column(int p_offset, int &r_line, int &r_column) const;
	int _get_caret_offset() const;
	bool _get_selection_range(int &r_from_line, int &r_from_column, int &r_to_line, int &r_to_column) const;
	bool _get_selection_offsets(int &r_from, int &r_to) const;

	// Fragments: paragraph slices of a range, as removed text or clipboard.
	LocalVector<Paragraph> _extract_slices(int p_from_line, int p_from_column, int p_to_line, int p_to_column) const;
	void _insert_slices(int p_line, int p_column, const LocalVector<Paragraph> &p_slices, bool p_restore_line_heights);
	void _remove_model_range(int p_from_line, int p_from_column, int p_to_line, int p_to_column);
	String _get_text_range(int p_from_line, int p_from_column, int p_to_line, int p_to_column) const;
	bool _slices_have_formatting(const LocalVector<Paragraph> &p_slices) const;
	RichTextDocument _make_range_document(int p_from_line, int p_from_column, int p_to_line, int p_to_column) const;
	LocalVector<Paragraph> _document_to_slices(const RichTextDocument &p_document);
	void _load_document(const RichTextDocument &p_document, const String &p_bbcode);

	String _serialize_bbcode() const;
	RichTextDocument _make_document() const;
	void _mark_bbcode_dirty();
	void _update_bbcode_from_text();
	void _lines_edited(int p_from_line, int p_to_line);

	// Style edits.
	void _apply_style_property(TextStyle &r_style, const TextStyle &p_toggle_reference, StyleProperty p_property, const Variant &p_value) const;
	void _apply_style_property_to_selection(StyleProperty p_property, const Variant &p_value = Variant(), bool p_record_undo = true);
	void _apply_style_property_to_url_ranges(StyleProperty p_property, const Variant &p_value = Variant());
	template <typename F>
	bool _restyle_lines(int p_from_line, int p_to_line, F p_restyle);
	void _apply_block_tag_to_selected_lines(const String &p_tag);
	void _clear_block_tag_from_selected_lines(const String &p_tag);
	void _toggle_block_tag_on_selected_lines(const String &p_tag);
	void _get_selected_line_range(int &r_from_line, int &r_to_line) const;
	TextStyle _get_style_at_offset(int p_offset) const;
	TextStyle _get_selection_common_style() const;
	int _get_default_font_size() const;
	void _caret_style_context_changed();
	Ref<StyleState> _capture_style_state(int p_from_line, int p_to_line) const;
	void _restore_style_state(const Ref<StyleState> &p_state);
	static bool _is_coalescing_style_property(StyleProperty p_property);
	void _break_style_run();
	void _push_style_undo(const Ref<StyleState> &p_before, bool p_isolate = false);
	void _style_changed();
	void _emit_text_style_changed();
	void _queue_full_style_refresh();
	void _refresh_queued_styles();
	void _refresh_all_styles();

	// TextEdit callbacks.
	Array _get_line_style_spans(int p_line) const;
	virtual int _get_visual_line_wrap_height(int p_line, int p_wrap_index) const override;
	virtual int _get_list_item_number(int p_line, int p_list_type, int p_list_start) const override;
	virtual void _text_inserting(int p_line, int p_column, const String &p_text) override;
	virtual void _deferred_layout_progressed(bool p_finished) override;
	bool content_height_signal_queued = false;
	uint64_t content_height_signal_usec = 0;
	void _emit_content_height_changed();
	virtual void _text_removing(int p_from_line, int p_from_column, int p_to_line, int p_to_column) override;
	String _get_line_height_for_line(int p_line) const;
	bool _get_url_style_at(int p_line, int p_column, TextStyle &r_style) const;
	bool _get_url_style_at_position(const Point2 &p_position, TextStyle &r_style) const;
	bool _get_url_at_position(const Point2 &p_position, String &r_url) const;
	void _update_meta_hover(const Point2 &p_position);
	bool _should_activate_url(const Ref<InputEventMouseButton> &p_mouse_button) const;
	bool _activate_url_at_position(const Point2 &p_position);
	void _insert_newline(bool p_shift_pressed);
	Array _get_line_inline_objects(int p_line, const String &p_line_text) const;
	Dictionary _make_image_inline_info(const InlineObject &p_object) const;
	static bool _is_horizontal_rule_bbcode(const String &p_bbcode);
	Dictionary _make_horizontal_rule_inline_info(const InlineObject &p_object) const;
	InlineObject *_find_object(uint64_t p_id, int *r_line = nullptr);
	const InlineObject *_get_object_at(int p_line, int p_column) const;
	Size2 _get_image_option_size(const RichTextDocument::InlineImage &p_image, const Size2 &p_fallback_size) const;
	float _get_fill_image_width() const;
	void _insert_objects(const String &p_text, const LocalVector<Paragraph> &p_slices);
	Vector<uint64_t> _get_selected_image_ids() const;
	void _ensure_image_edit_controls();
	void _sync_selected_image_edit_controls();
	void _image_edit_control_gui_input(const Ref<InputEvent> &p_event);
	void _select_image(uint64_t p_id, const Rect2 &p_screen_rect);
	void _clear_selected_image();
	void _commit_selected_image_options(const String &p_width, const String &p_height, const String &p_alt, const String &p_link, bool p_record_undo = true);
	void _selected_image_width_changed(double p_value);
	void _selected_image_height_changed(double p_value);
	void _selected_image_ratio_toggled(bool p_pressed);
	void _selected_image_fill_width_toggled(bool p_pressed);
	void _selected_image_reset();
	void _pressed_selected_image_details();
	void _apply_selected_image_details();
	void _draw_inline_object(const Dictionary &p_info, const Rect2 &p_rect);
	void _inline_object_clicked(const Dictionary &p_info, const Rect2 &p_rect);
	static ScrollBarMode _get_axis_scroll_bar_mode(Overflow p_axis, Overflow p_other_axis);

protected:
	virtual bool _is_text_clipping_enabled() const override;
	virtual ScrollBarMode _get_h_scroll_bar_mode() const override;
	virtual ScrollBarMode _get_v_scroll_bar_mode() const override;
	void _notification(int p_what);
	void _validate_property(PropertyInfo &p_property) const;
	virtual void _apply_custom_undo_operation(const StringName &p_type, const Variant &p_data) override;
	virtual void _update_theme_item_cache() override;
	static void _bind_methods();

public:
	virtual void set_text(const String &p_text) override;
	virtual String get_text() const override;
	virtual void gui_input(const Ref<InputEvent> &p_gui_input) override;
	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;
	virtual String get_tooltip(const Point2 &p_pos) const override;
	virtual Control *make_custom_tooltip(const String &p_text) const override;
	void set_use_bbcode(bool p_enable);
	bool is_using_bbcode() const;
	void set_overflow_x(Overflow p_overflow);
	Overflow get_overflow_x() const;
	void set_overflow_y(Overflow p_overflow);
	Overflow get_overflow_y() const;
	int get_content_height() const;
	bool has_pending_layout() const;
	void set_link_activation_mode(LinkActivationMode p_mode);
	LinkActivationMode get_link_activation_mode() const;

	void set_bbcode_text(const String &p_bbcode);
	String get_bbcode_text() const;
	void set_style_font_resolver(const Callable &p_resolver);
	Callable get_style_font_resolver() const;
	void set_visible_characters(int p_visible);
	int get_visible_characters() const;
	void set_visible_ratio(float p_ratio);
	float get_visible_ratio() const;
	void set_visible_characters_behavior(TextServer::VisibleCharactersBehavior p_behavior);
	TextServer::VisibleCharactersBehavior get_visible_characters_behavior() const;
	String get_selection_style_template() const;
	bool apply_style_template(const String &p_template, const Dictionary &p_overrides = Dictionary());
	void copy_rich_text();
	bool paste_rich_text();

	// Global-offset view of the formatting (rebuilt on demand).
	const LocalVector<StyleSpan> &get_style_spans() const;
	// The style the toolbar reflects: the selection start or the character
	// before the caret.
	TextStyle get_current_style() const;

	void set_bold();
	void clear_bold();
	void toggle_bold();
	void set_italic();
	void clear_italic();
	void toggle_italic();
	void set_underline();
	void clear_underline();
	void toggle_underline();
	void set_strikethrough();
	void clear_strikethrough();
	void toggle_strikethrough();
	void set_overline();
	void clear_overline();
	void toggle_overline();
	void set_selection_color(const Color &p_color);
	void clear_selection_color();
	void set_selection_bg_color(const Color &p_color);
	void clear_selection_bg_color();
	void set_selection_outline_color(const Color &p_color);
	void clear_selection_outline_color();
	void set_selection_outline_size(int p_size);
	void clear_selection_outline_size();
	void begin_selection_color_preview();
	void preview_selection_color(const Color &p_color);
	void end_selection_color_preview(bool p_commit);
	void begin_selection_bg_color_preview();
	void preview_selection_bg_color(const Color &p_color);
	void preview_selection_outline_color(const Color &p_color);
	void end_selection_bg_color_preview(bool p_commit);
	void set_selection_font_size(int p_size);
	void clear_selection_font_size();
	void set_selection_font(const String &p_font);
	void clear_selection_font();
	void set_selection_url(const String &p_url);
	void clear_selection_url();
	void set_selection_url_tooltip(const String &p_tooltip);
	void clear_selection_url_tooltip();
	void set_selection_url_visited();
	void clear_selection_url_visited();
	void toggle_code();
	void set_selection_line_height(const String &p_line_height);
	void clear_selection_line_height();
	String get_current_line_height() const;
	void set_decoration_paint(const Dictionary &p_paint);
	void set_outline_over_fill(bool p_enabled);
	bool is_outline_over_fill() const;
	static Dictionary get_style_template_values(const String &p_template);
	void set_selection_line_spacing(int p_spacing);
	void insert_image(const String &p_source, int p_width = -1, int p_height = -1, const String &p_alt = String());
	void set_selection_image_size(int p_width, int p_height);
	void insert_horizontal_rule();
	void set_alignment(HorizontalAlignment p_alignment);
	void toggle_quote();
	void set_quote();
	void clear_quote();
	void toggle_unordered_list();
	void set_unordered_list();
	void clear_unordered_list();
	void toggle_ordered_list();
	void set_ordered_list();
	void clear_ordered_list();
	int get_current_font_size() const;
	void increase_indent();
	void decrease_indent();

	RichTextEdit();
};

VARIANT_ENUM_CAST(RichTextEdit::LinkActivationMode);
VARIANT_ENUM_CAST(RichTextEdit::Overflow);
