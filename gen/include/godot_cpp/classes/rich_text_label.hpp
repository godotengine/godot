/**************************************************************************/
/*  rich_text_label.hpp                                                   */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/rect2i.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Font;
class PackedStringArray;
class PopupMenu;
class RichTextEffect;
class Texture2D;
class VScrollBar;
struct Vector2;

class RichTextLabel : public Control {
	GDEXTENSION_CLASS(RichTextLabel, Control)

public:
	enum ListType {
		LIST_NUMBERS = 0,
		LIST_LETTERS = 1,
		LIST_ROMAN = 2,
		LIST_DOTS = 3,
	};

	enum MenuItems {
		MENU_COPY = 0,
		MENU_SELECT_ALL = 1,
		MENU_MAX = 2,
	};

	enum MetaUnderline {
		META_UNDERLINE_NEVER = 0,
		META_UNDERLINE_ALWAYS = 1,
		META_UNDERLINE_ON_HOVER = 2,
	};

	enum ImageUpdateMask : uint64_t {
		UPDATE_TEXTURE = 1,
		UPDATE_SIZE = 2,
		UPDATE_COLOR = 4,
		UPDATE_ALIGNMENT = 8,
		UPDATE_REGION = 16,
		UPDATE_PAD = 32,
		UPDATE_TOOLTIP = 64,
		UPDATE_WIDTH_IN_PERCENT = 128,
	};

	String get_parsed_text() const;
	void add_text(const String &p_text);
	void set_text(const String &p_text);
	void add_hr(int32_t p_width = 90, int32_t p_height = 2, const Color &p_color = Color(1, 1, 1, 1), HorizontalAlignment p_alignment = (HorizontalAlignment)1, bool p_width_in_percent = true, bool p_height_in_percent = false);
	void add_image(const Ref<Texture2D> &p_image, int32_t p_width = 0, int32_t p_height = 0, const Color &p_color = Color(1, 1, 1, 1), InlineAlignment p_inline_align = (InlineAlignment)5, const Rect2 &p_region = Rect2(0, 0, 0, 0), const Variant &p_key = nullptr, bool p_pad = false, const String &p_tooltip = String(), bool p_width_in_percent = false, bool p_height_in_percent = false, const String &p_alt_text = String());
	void update_image(const Variant &p_key, BitField<RichTextLabel::ImageUpdateMask> p_mask, const Ref<Texture2D> &p_image, int32_t p_width = 0, int32_t p_height = 0, const Color &p_color = Color(1, 1, 1, 1), InlineAlignment p_inline_align = (InlineAlignment)5, const Rect2 &p_region = Rect2(0, 0, 0, 0), bool p_pad = false, const String &p_tooltip = String(), bool p_width_in_percent = false, bool p_height_in_percent = false);
	void newline();
	bool remove_paragraph(int32_t p_paragraph, bool p_no_invalidate = false);
	bool invalidate_paragraph(int32_t p_paragraph);
	void push_font(const Ref<Font> &p_font, int32_t p_font_size = 0);
	void push_font_size(int32_t p_font_size);
	void push_normal();
	void push_bold();
	void push_bold_italics();
	void push_italics();
	void push_mono();
	void push_color(const Color &p_color);
	void push_outline_size(int32_t p_outline_size);
	void push_outline_color(const Color &p_color);
	void push_paragraph(HorizontalAlignment p_alignment, Control::TextDirection p_base_direction = (Control::TextDirection)0, const String &p_language = String(), TextServer::StructuredTextParser p_st_parser = (TextServer::StructuredTextParser)0, BitField<TextServer::JustificationFlag> p_justification_flags = (BitField<TextServer::JustificationFlag>)163, const PackedFloat32Array &p_tab_stops = PackedFloat32Array());
	void push_indent(int32_t p_level);
	void push_list(int32_t p_level, RichTextLabel::ListType p_type, bool p_capitalize, const String &p_bullet = "•");
	void push_meta(const Variant &p_data, RichTextLabel::MetaUnderline p_underline_mode = (RichTextLabel::MetaUnderline)1, const String &p_tooltip = String());
	void push_hint(const String &p_description);
	void push_language(const String &p_language);
	void push_underline(const Color &p_color = Color(0, 0, 0, 0));
	void push_strikethrough(const Color &p_color = Color(0, 0, 0, 0));
	void push_table(int32_t p_columns, InlineAlignment p_inline_align = (InlineAlignment)0, int32_t p_align_to_row = -1, const String &p_name = String());
	void push_dropcap(const String &p_string, const Ref<Font> &p_font, int32_t p_size, const Rect2 &p_dropcap_margins = Rect2(0, 0, 0, 0), const Color &p_color = Color(1, 1, 1, 1), int32_t p_outline_size = 0, const Color &p_outline_color = Color(0, 0, 0, 0));
	void set_table_column_expand(int32_t p_column, bool p_expand, int32_t p_ratio = 1, bool p_shrink = true);
	void set_table_column_name(int32_t p_column, const String &p_name);
	void set_cell_row_background_color(const Color &p_odd_row_bg, const Color &p_even_row_bg);
	void set_cell_border_color(const Color &p_color);
	void set_cell_size_override(const Vector2 &p_min_size, const Vector2 &p_max_size);
	void set_cell_padding(const Rect2 &p_padding);
	void push_cell();
	void push_fgcolor(const Color &p_fgcolor);
	void push_bgcolor(const Color &p_bgcolor);
	void push_customfx(const Ref<RichTextEffect> &p_effect, const Dictionary &p_env);
	void push_context();
	void pop_context();
	void pop();
	void pop_all();
	void clear();
	void set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser);
	TextServer::StructuredTextParser get_structured_text_bidi_override() const;
	void set_structured_text_bidi_override_options(const Array &p_args);
	Array get_structured_text_bidi_override_options() const;
	void set_text_direction(Control::TextDirection p_direction);
	Control::TextDirection get_text_direction() const;
	void set_language(const String &p_language);
	String get_language() const;
	void set_horizontal_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_horizontal_alignment() const;
	void set_vertical_alignment(VerticalAlignment p_alignment);
	VerticalAlignment get_vertical_alignment() const;
	void set_justification_flags(BitField<TextServer::JustificationFlag> p_justification_flags);
	BitField<TextServer::JustificationFlag> get_justification_flags() const;
	void set_tab_stops(const PackedFloat32Array &p_tab_stops);
	PackedFloat32Array get_tab_stops() const;
	void set_autowrap_mode(TextServer::AutowrapMode p_autowrap_mode);
	TextServer::AutowrapMode get_autowrap_mode() const;
	void set_autowrap_trim_flags(BitField<TextServer::LineBreakFlag> p_autowrap_trim_flags);
	BitField<TextServer::LineBreakFlag> get_autowrap_trim_flags() const;
	void set_meta_underline(bool p_enable);
	bool is_meta_underlined() const;
	void set_hint_underline(bool p_enable);
	bool is_hint_underlined() const;
	void set_scroll_active(bool p_active);
	bool is_scroll_active() const;
	void set_scroll_follow_visible_characters(bool p_follow);
	bool is_scroll_following_visible_characters() const;
	void set_scroll_follow(bool p_follow);
	bool is_scroll_following() const;
	VScrollBar *get_v_scroll_bar();
	void scroll_to_line(int32_t p_line);
	void scroll_to_paragraph(int32_t p_paragraph);
	void scroll_to_selection();
	void set_tab_size(int32_t p_spaces);
	int32_t get_tab_size() const;
	void set_fit_content(bool p_enabled);
	bool is_fit_content_enabled() const;
	void set_selection_enabled(bool p_enabled);
	bool is_selection_enabled() const;
	void set_context_menu_enabled(bool p_enabled);
	bool is_context_menu_enabled() const;
	void set_shortcut_keys_enabled(bool p_enabled);
	bool is_shortcut_keys_enabled() const;
	void set_deselect_on_focus_loss_enabled(bool p_enable);
	bool is_deselect_on_focus_loss_enabled() const;
	void set_drag_and_drop_selection_enabled(bool p_enable);
	bool is_drag_and_drop_selection_enabled() const;
	int32_t get_selection_from() const;
	int32_t get_selection_to() const;
	float get_selection_line_offset() const;
	void select_all();
	String get_selected_text() const;
	void deselect();
	void parse_bbcode(const String &p_bbcode);
	void append_text(const String &p_bbcode);
	String get_text() const;
	bool is_ready() const;
	bool is_finished() const;
	void set_threaded(bool p_threaded);
	bool is_threaded() const;
	void set_progress_bar_delay(int32_t p_delay_ms);
	int32_t get_progress_bar_delay() const;
	void set_visible_characters(int32_t p_amount);
	int32_t get_visible_characters() const;
	TextServer::VisibleCharactersBehavior get_visible_characters_behavior() const;
	void set_visible_characters_behavior(TextServer::VisibleCharactersBehavior p_behavior);
	void set_visible_ratio(float p_ratio);
	float get_visible_ratio() const;
	int32_t get_character_line(int32_t p_character);
	int32_t get_character_paragraph(int32_t p_character);
	int32_t get_total_character_count() const;
	void set_use_bbcode(bool p_enable);
	bool is_using_bbcode() const;
	int32_t get_line_count() const;
	Vector2i get_line_range(int32_t p_line);
	int32_t get_visible_line_count() const;
	int32_t get_paragraph_count() const;
	int32_t get_visible_paragraph_count() const;
	int32_t get_content_height() const;
	int32_t get_content_width() const;
	int32_t get_line_height(int32_t p_line) const;
	int32_t get_line_width(int32_t p_line) const;
	Rect2i get_visible_content_rect() const;
	float get_line_offset(int32_t p_line);
	float get_paragraph_offset(int32_t p_paragraph);
	Dictionary parse_expressions_for_values(const PackedStringArray &p_expressions);
	void set_effects(const Array &p_effects);
	Array get_effects();
	void install_effect(const Variant &p_effect);
	void reload_effects();
	PopupMenu *get_menu() const;
	bool is_menu_visible() const;
	void menu_option(int32_t p_option);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Control::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(RichTextLabel::ListType);
VARIANT_ENUM_CAST(RichTextLabel::MenuItems);
VARIANT_ENUM_CAST(RichTextLabel::MetaUnderline);
VARIANT_BITFIELD_CAST(RichTextLabel::ImageUpdateMask);

