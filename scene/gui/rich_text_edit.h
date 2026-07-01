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

	using TextStyle = RichTextDocument::Style;

	struct StyleSpan {
		int from = 0;
		int to = 0;
		TextStyle style;

		bool operator==(const StyleSpan &p_other) const { return from == p_other.from && to == p_other.to && style == p_other.style; }
		bool operator!=(const StyleSpan &p_other) const { return !(*this == p_other); }
	};

private:
	struct DeletedStyleRange {
		int offset = 0;
		String text;
		Vector<StyleSpan> spans;
	};

	enum StyleProperty {
		STYLE_PROPERTY_BOLD,
		STYLE_PROPERTY_ITALIC,
		STYLE_PROPERTY_UNDERLINE,
		STYLE_PROPERTY_STRIKETHROUGH,
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

	mutable String bbcode_text;
	String source_text;
	Vector<StyleSpan> style_spans;
	Vector<RichTextDocument::InlineImage> images;
	Vector<RichTextDocument::RawInline> raw_inlines;
	Vector<DeletedStyleRange> deleted_style_ranges_for_undo;
	Vector<RichTextDocument::InlineImage> deleted_images_for_undo;
	Vector<RichTextDocument::RawInline> deleted_raw_inlines_for_undo;
	bool setting_bbcode_text = false;
	bool use_bbcode = false;
	mutable bool bbcode_dirty = true;
	bool syncing_text_change = false;
	bool pending_inline_metadata_restore_sync = false;
	LinkActivationMode link_activation_mode = LINK_ACTIVATION_AUTO;
	bool text_style_changed_dirty = false;
	bool typing_style_override = false;
	bool style_preview_active = false;
	bool style_preview_had_selection = false;
	Vector<StyleSpan> style_preview_before_spans;
	TextStyle style_preview_before_typing_style;
	bool style_preview_before_typing_style_override = false;
	bool meta_hovering = false;
	bool updating_image_edit_controls = false;
	HashSet<String> visited_links;
	int selected_image_offset = -1;
	String tracked_text;
	String current_meta;
	String active_meta;
	TextStyle typing_style;
	PanelContainer *image_edit_bar = nullptr;
	PopupPanel *image_details_popup = nullptr;
	SpinBox *image_width_spin = nullptr;
	SpinBox *image_height_spin = nullptr;
	Button *image_ratio_button = nullptr;
	Button *image_fill_width_button = nullptr;
	LineEdit *image_alt_edit = nullptr;
	LineEdit *image_link_edit = nullptr;

	struct ThemeCache {
		Ref<StyleBox> tooltip_panel;
		Color tooltip_font_color;
		Ref<Font> tooltip_font;
		int tooltip_font_size = 0;
	} theme_cache;

	static bool _is_url_auto_open_allowed(const String &p_url);

	String _serialize_bbcode() const;
	void _mark_bbcode_dirty();
	void _apply_document(const RichTextDocument &p_document);
	void _update_bbcode_from_text();
	void _sync_style_spans_for_text_change(const String &p_old_text, const String &p_new_text);
	void _sync_inline_objects_for_text_change(const String &p_old_text, const String &p_new_text, int p_old_from, int p_old_to, int p_new_to);
	void _replace_style_range(int p_from, int p_to, const TextStyle &p_style);
	void _apply_style_property(TextStyle &r_style, const TextStyle &p_toggle_reference, StyleProperty p_property, const Variant &p_value) const;
	void _apply_style_property_to_selection(StyleProperty p_property, const Variant &p_value = Variant(), bool p_record_undo = true);
	void _apply_style_property_to_url_ranges(StyleProperty p_property, const Variant &p_value = Variant());
	void _apply_block_tag_to_selected_lines(const String &p_tag);
	void _clear_block_tag_from_selected_lines(const String &p_tag);
	void _toggle_block_tag_on_selected_lines(const String &p_tag);
	Vector<int> _get_style_boundaries_for_range(int p_from, int p_to) const;
	TextStyle _get_insertion_style_at_offset(int p_offset) const;
	TextStyle _get_style_at_offset(int p_offset) const;
	TextStyle _get_selection_common_style() const;
	int _get_default_font_size() const;
	void _merge_adjacent_spans();
	int _get_line_start_offset(int p_line) const;
	int _get_caret_offset() const;
	bool _get_selection_offsets(int &r_from, int &r_to) const;
	void _apply_style_to_selection(const TextStyle &p_style);
	void _caret_style_context_changed();
	Variant _make_style_state_variant(const Vector<StyleSpan> &p_spans, const Vector<RichTextDocument::InlineImage> &p_images, const Vector<RichTextDocument::RawInline> &p_raw_inlines, const TextStyle &p_typing_style, bool p_typing_style_override) const;
	void _restore_style_state_variant(const Variant &p_state);
	void _push_style_undo_snapshot(const Vector<StyleSpan> &p_before_spans, const Vector<RichTextDocument::InlineImage> &p_before_images, const Vector<RichTextDocument::RawInline> &p_before_raw_inlines, const TextStyle &p_before_typing_style, bool p_before_typing_style_override);
	void _refresh_style_rendering();
	void _style_changed();
	void _emit_text_style_changed();
	Array _get_line_style_spans(int p_line) const;
	bool _get_url_style_at_offset(int p_offset, TextStyle &r_style) const;
	bool _get_url_style_at_position(const Point2 &p_position, TextStyle &r_style) const;
	bool _get_url_at_offset(int p_offset, String &r_url) const;
	bool _get_url_at_position(const Point2 &p_position, String &r_url) const;
	void _update_meta_hover(const Point2 &p_position);
	bool _should_activate_url(const Ref<InputEventMouseButton> &p_mouse_button) const;
	bool _activate_url_at_position(const Point2 &p_position);
	void _insert_newline(bool p_shift_pressed);
	Array _get_line_inline_objects(int p_line, const String &p_line_text) const;
	Dictionary _make_image_inline_info(const RichTextDocument::InlineImage &p_image, int p_column) const;
	RichTextDocument::InlineImage *_get_image_at_offset(int p_offset);
	Size2 _get_image_option_size(const RichTextDocument::InlineImage &p_image, const Size2 &p_fallback_size) const;
	float _get_fill_image_width() const;
	void _insert_image_with_options(const String &p_source, const HashMap<String, String> &p_options);
	Vector<int> _get_selected_image_offsets() const;
	void _ensure_image_edit_controls();
	void _select_image(int p_offset, const Rect2 &p_screen_rect);
	void _clear_selected_image();
	void _commit_selected_image_options(const String &p_width, const String &p_height, const String &p_alt, const String &p_link, bool p_record_undo = true);
	void _selected_image_width_changed(double p_value);
	void _selected_image_height_changed(double p_value);
	void _selected_image_ratio_toggled(bool p_pressed);
	void _selected_image_fill_width_toggled(bool p_pressed);
	void _pressed_selected_image_details();
	void _apply_selected_image_details();
	void _draw_inline_object(const Dictionary &p_info, const Rect2 &p_rect);
	void _inline_object_clicked(const Dictionary &p_info, const Rect2 &p_rect);

protected:
	void _notification(int p_what);
	void _validate_property(PropertyInfo &p_property) const;
	virtual void _apply_custom_undo_operation(const StringName &p_type, const Variant &p_data) override;
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
	void set_link_activation_mode(LinkActivationMode p_mode);
	LinkActivationMode get_link_activation_mode() const;

	void set_bbcode_text(const String &p_bbcode);
	String get_bbcode_text() const;

	const Vector<StyleSpan> &get_style_spans() const;

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
	void insert_image(const String &p_source, int p_width = -1, int p_height = -1, const String &p_alt = String());
	void set_selection_image_size(int p_width, int p_height);
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
