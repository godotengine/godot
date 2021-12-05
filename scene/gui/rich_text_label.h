/*************************************************************************/
/*  rich_text_label.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef RICH_TEXT_LABEL_H
#define RICH_TEXT_LABEL_H

#include "rich_text_effect.h"
#include "scene/gui/scroll_bar.h"
#include "scene/resources/text_paragraph.h"

class RichTextLabel : public Control {
	GDCLASS(RichTextLabel, Control);

public:
	enum Align {
		ALIGN_LEFT,
		ALIGN_CENTER,
		ALIGN_RIGHT,
		ALIGN_FILL
	};

	enum ListType {
		LIST_NUMBERS,
		LIST_LETTERS,
		LIST_ROMAN,
		LIST_DOTS
	};

	enum ItemType {
		ITEM_FRAME,
		ITEM_TEXT,
		ITEM_IMAGE,
		ITEM_NEWLINE,
		ITEM_FONT,
		ITEM_FONT_SIZE,
		ITEM_FONT_FEATURES,
		ITEM_COLOR,
		ITEM_OUTLINE_SIZE,
		ITEM_OUTLINE_COLOR,
		ITEM_UNDERLINE,
		ITEM_STRIKETHROUGH,
		ITEM_PARAGRAPH,
		ITEM_INDENT,
		ITEM_LIST,
		ITEM_TABLE,
		ITEM_FADE,
		ITEM_SHAKE,
		ITEM_WAVE,
		ITEM_TORNADO,
		ITEM_RAINBOW,
		ITEM_BGCOLOR,
		ITEM_FGCOLOR,
		ITEM_META,
		ITEM_DROPCAP,
		ITEM_CUSTOMFX
	};

	enum VisibleCharactersBehavior {
		VC_CHARS_BEFORE_SHAPING,
		VC_CHARS_AFTER_SHAPING,
		VC_GLYPHS_AUTO,
		VC_GLYPHS_LTR,
		VC_GLYPHS_RTL,
	};

protected:
	void _notification(int p_what);
	static void _bind_methods();

private:
	struct Item;

	struct Line {
		Item *from = nullptr;

		Ref<TextParagraph> text_buf;
		Color dc_color;
		int dc_ol_size = 0;
		Color dc_ol_color;

		Vector2 offset;
		int char_offset = 0;
		int char_count = 0;

		Line() { text_buf.instantiate(); }
	};

	struct Item {
		int index = 0;
		int char_ofs = 0;
		Item *parent = nullptr;
		ItemType type = ITEM_FRAME;
		List<Item *> subitems;
		List<Item *>::Element *E = nullptr;
		int line = 0;

		void _clear_children() {
			while (subitems.size()) {
				memdelete(subitems.front()->get());
				subitems.pop_front();
			}
		}

		virtual ~Item() { _clear_children(); }
	};

	struct ItemFrame : public Item {
		bool cell = false;

		Vector<Line> lines;
		int first_invalid_line = 0;
		int first_resized_line = 0;

		ItemFrame *parent_frame = nullptr;

		Color odd_row_bg = Color(0, 0, 0, 0);
		Color even_row_bg = Color(0, 0, 0, 0);
		Color border = Color(0, 0, 0, 0);
		Size2 min_size_over = Size2(-1, -1);
		Size2 max_size_over = Size2(-1, -1);
		Rect2 padding;

		ItemFrame() { type = ITEM_FRAME; }
	};

	struct ItemText : public Item {
		String text;
		ItemText() { type = ITEM_TEXT; }
	};

	struct ItemDropcap : public Item {
		String text;
		Ref<Font> font;
		int font_size = 0;
		Color color;
		int ol_size = 0;
		Color ol_color;
		Rect2 dropcap_margins;
		ItemDropcap() { type = ITEM_DROPCAP; }
	};

	struct ItemImage : public Item {
		Ref<Texture2D> image;
		InlineAlign inline_align = INLINE_ALIGN_CENTER;
		Size2 size;
		Color color;
		ItemImage() { type = ITEM_IMAGE; }
	};

	struct ItemFont : public Item {
		Ref<Font> font;
		ItemFont() { type = ITEM_FONT; }
	};

	struct ItemFontSize : public Item {
		int font_size = 16;
		ItemFontSize() { type = ITEM_FONT_SIZE; }
	};

	struct ItemFontFeatures : public Item {
		Dictionary opentype_features;
		ItemFontFeatures() { type = ITEM_FONT_FEATURES; }
	};

	struct ItemColor : public Item {
		Color color;
		ItemColor() { type = ITEM_COLOR; }
	};

	struct ItemOutlineSize : public Item {
		int outline_size = 0;
		ItemOutlineSize() { type = ITEM_OUTLINE_SIZE; }
	};

	struct ItemOutlineColor : public Item {
		Color color;
		ItemOutlineColor() { type = ITEM_OUTLINE_COLOR; }
	};

	struct ItemUnderline : public Item {
		ItemUnderline() { type = ITEM_UNDERLINE; }
	};

	struct ItemStrikethrough : public Item {
		ItemStrikethrough() { type = ITEM_STRIKETHROUGH; }
	};

	struct ItemMeta : public Item {
		Variant meta;
		ItemMeta() { type = ITEM_META; }
	};

	struct ItemParagraph : public Item {
		Align align = ALIGN_LEFT;
		String language;
		Control::TextDirection direction = Control::TEXT_DIRECTION_AUTO;
		Control::StructuredTextParser st_parser = STRUCTURED_TEXT_DEFAULT;
		ItemParagraph() { type = ITEM_PARAGRAPH; }
	};

	struct ItemIndent : public Item {
		int level = 0;
		ItemIndent() { type = ITEM_INDENT; }
	};

	struct ItemList : public Item {
		ListType list_type = LIST_DOTS;
		bool capitalize = false;
		int level = 0;
		ItemList() { type = ITEM_LIST; }
	};

	struct ItemNewline : public Item {
		ItemNewline() { type = ITEM_NEWLINE; }
	};

	struct ItemTable : public Item {
		struct Column {
			bool expand = false;
			int expand_ratio = 0;
			int min_width = 0;
			int max_width = 0;
			int width = 0;
		};

		Vector<Column> columns;
		Vector<float> rows;

		int total_width = 0;
		int total_height = 0;
		InlineAlign inline_align = INLINE_ALIGN_TOP;
		ItemTable() { type = ITEM_TABLE; }
	};

	struct ItemFade : public Item {
		int starting_index = 0;
		int length = 0;

		ItemFade() { type = ITEM_FADE; }
	};

	struct ItemFX : public Item {
		double elapsed_time = 0.f;
	};

	struct ItemShake : public ItemFX {
		int strength = 0;
		float rate = 0.0f;
		uint64_t _current_rng = 0;
		uint64_t _previous_rng = 0;
		Vector2 prev_off;

		ItemShake() { type = ITEM_SHAKE; }

		void reroll_random() {
			_previous_rng = _current_rng;
			_current_rng = Math::rand();
		}

		uint64_t offset_random(int index) {
			return (_current_rng >> (index % 64)) |
					(_current_rng << (64 - (index % 64)));
		}

		uint64_t offset_previous_random(int index) {
			return (_previous_rng >> (index % 64)) |
					(_previous_rng << (64 - (index % 64)));
		}
	};

	struct ItemWave : public ItemFX {
		float frequency = 1.0f;
		float amplitude = 1.0f;
		Vector2 prev_off;

		ItemWave() { type = ITEM_WAVE; }
	};

	struct ItemTornado : public ItemFX {
		float radius = 1.0f;
		float frequency = 1.0f;
		Vector2 prev_off;

		ItemTornado() { type = ITEM_TORNADO; }
	};

	struct ItemRainbow : public ItemFX {
		float saturation = 0.8f;
		float value = 0.8f;
		float frequency = 1.0f;

		ItemRainbow() { type = ITEM_RAINBOW; }
	};

	struct ItemBGColor : public Item {
		Color color;
		ItemBGColor() { type = ITEM_BGCOLOR; }
	};

	struct ItemFGColor : public Item {
		Color color;
		ItemFGColor() { type = ITEM_FGCOLOR; }
	};

	struct ItemCustomFX : public ItemFX {
		Ref<CharFXTransform> char_fx_transform;
		Ref<RichTextEffect> custom_effect;

		ItemCustomFX() {
			type = ITEM_CUSTOMFX;
			char_fx_transform.instantiate();
		}

		virtual ~ItemCustomFX() {
			_clear_children();

			char_fx_transform.unref();
			custom_effect.unref();
		}
	};

	ItemFrame *main = nullptr;
	Item *current = nullptr;
	ItemFrame *current_frame = nullptr;

	VScrollBar *vscroll = nullptr;

	bool scroll_visible = false;
	bool scroll_follow = false;
	bool scroll_following = false;
	bool scroll_active = true;
	int scroll_w = 0;
	bool scroll_updated = false;
	bool updating_scroll = false;
	int current_idx = 1;
	int current_char_ofs = 0;
	int visible_paragraph_count = 0;
	int visible_line_count = 0;

	int tab_size = 4;
	bool underline_meta = true;
	bool override_selected_font_color = false;

	Align default_align = ALIGN_LEFT;

	ItemMeta *meta_hovering = nullptr;
	Variant current_meta;

	Array custom_effects;

	void _invalidate_current_line(ItemFrame *p_frame);
	void _validate_line_caches(ItemFrame *p_frame);

	void _add_item(Item *p_item, bool p_enter = false, bool p_ensure_newline = false);
	void _remove_item(Item *p_item, const int p_line, const int p_subitem_line);

	String language;
	TextDirection text_direction = TEXT_DIRECTION_AUTO;
	Control::StructuredTextParser st_parser = STRUCTURED_TEXT_DEFAULT;
	Array st_args;

	struct Selection {
		ItemFrame *click_frame = nullptr;
		int click_line = 0;
		Item *click_item = nullptr;
		int click_char = 0;

		ItemFrame *from_frame = nullptr;
		int from_line = 0;
		Item *from_item = nullptr;
		int from_char = 0;

		ItemFrame *to_frame = nullptr;
		int to_line = 0;
		Item *to_item = nullptr;
		int to_char = 0;

		bool active = false; // anything selected? i.e. from, to, etc. valid?
		bool enabled = false; // allow selections?
	};

	Selection selection;
	bool deselect_on_focus_loss_enabled = true;

	int visible_characters = -1;
	float percent_visible = 1.0;
	VisibleCharactersBehavior visible_chars_behavior = VC_CHARS_BEFORE_SHAPING;

	void _find_click(ItemFrame *p_frame, const Point2i &p_click, ItemFrame **r_click_frame = nullptr, int *r_click_line = nullptr, Item **r_click_item = nullptr, int *r_click_char = nullptr, bool *r_outside = nullptr);

	String _get_line_text(ItemFrame *p_frame, int p_line, Selection p_sel) const;
	bool _search_line(ItemFrame *p_frame, int p_line, const String &p_string, int p_char_idx, bool p_reverse_search);
	bool _search_table(ItemTable *p_table, List<Item *>::Element *p_from, const String &p_string, bool p_reverse_search);

	void _shape_line(ItemFrame *p_frame, int p_line, const Ref<Font> &p_base_font, int p_base_font_size, int p_width, int *r_char_offset);
	void _resize_line(ItemFrame *p_frame, int p_line, const Ref<Font> &p_base_font, int p_base_font_size, int p_width);
	int _draw_line(ItemFrame *p_frame, int p_line, const Vector2 &p_ofs, int p_width, const Color &p_base_color, int p_outline_size, const Color &p_outline_color, const Color &p_font_shadow_color, int p_shadow_outline_size, const Point2 &p_shadow_ofs, int &r_processed_glyphs);
	float _find_click_in_line(ItemFrame *p_frame, int p_line, const Vector2 &p_ofs, int p_width, const Point2i &p_click, ItemFrame **r_click_frame = nullptr, int *r_click_line = nullptr, Item **r_click_item = nullptr, int *r_click_char = nullptr);

	String _roman(int p_num, bool p_capitalize) const;
	String _letters(int p_num, bool p_capitalize) const;

	Item *_get_item_at_pos(Item *p_item_from, Item *p_item_to, int p_position);
	void _find_frame(Item *p_item, ItemFrame **r_frame, int *r_line);
	Ref<Font> _find_font(Item *p_item);
	int _find_font_size(Item *p_item);
	Dictionary _find_font_features(Item *p_item);
	int _find_outline_size(Item *p_item, int p_default);
	ItemList *_find_list_item(Item *p_item);
	ItemDropcap *_find_dc_item(Item *p_item);
	int _find_list(Item *p_item, Vector<int> &r_index, Vector<ItemList *> &r_list);
	int _find_margin(Item *p_item, const Ref<Font> &p_base_font, int p_base_font_size);
	Align _find_align(Item *p_item);
	TextServer::Direction _find_direction(Item *p_item);
	Control::StructuredTextParser _find_stt(Item *p_item);
	String _find_language(Item *p_item);
	Color _find_color(Item *p_item, const Color &p_default_color);
	Color _find_outline_color(Item *p_item, const Color &p_default_color);
	bool _find_underline(Item *p_item);
	bool _find_strikethrough(Item *p_item);
	bool _find_meta(Item *p_item, Variant *r_meta, ItemMeta **r_item = nullptr);
	Color _find_bgcolor(Item *p_item);
	Color _find_fgcolor(Item *p_item);
	bool _find_layout_subitem(Item *from, Item *to);
	void _fetch_item_fx_stack(Item *p_item, Vector<ItemFX *> &r_stack);

	void _update_scroll();
	void _update_fx(ItemFrame *p_frame, double p_delta_time);
	void _scroll_changed(double);

	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	Item *_get_next_item(Item *p_item, bool p_free = false) const;
	Item *_get_prev_item(Item *p_item, bool p_free = false) const;

	Rect2 _get_text_rect();
	Ref<RichTextEffect> _get_custom_effect_by_code(String p_bbcode_identifier);
	virtual Dictionary parse_expressions_for_values(Vector<String> p_expressions);

	void _draw_fbg_boxes(RID p_ci, RID p_rid, Vector2 line_off, Item *it_from, Item *it_to, int start, int end, int fbg_flag);
#ifndef DISABLE_DEPRECATED
	// Kept for compatibility from 3.x to 4.0.
	bool _set(const StringName &p_name, const Variant &p_value);
#endif
	bool use_bbcode = false;
	String text;

	int fixed_width = -1;

	bool fit_content_height = false;

public:
	String get_parsed_text() const;
	void add_text(const String &p_text);
	void add_image(const Ref<Texture2D> &p_image, const int p_width = 0, const int p_height = 0, const Color &p_color = Color(1.0, 1.0, 1.0), InlineAlign p_align = INLINE_ALIGN_CENTER);
	void add_newline();
	bool remove_line(const int p_line);
	void push_dropcap(const String &p_string, const Ref<Font> &p_font, int p_size, const Rect2 &p_dropcap_margins = Rect2(), const Color &p_color = Color(1, 1, 1), int p_ol_size = 0, const Color &p_ol_color = Color(0, 0, 0, 0));
	void push_font(const Ref<Font> &p_font);
	void push_font_size(int p_font_size);
	void push_font_features(const Dictionary &p_features);
	void push_outline_size(int p_font_size);
	void push_normal();
	void push_bold();
	void push_bold_italics();
	void push_italics();
	void push_mono();
	void push_color(const Color &p_color);
	void push_outline_color(const Color &p_color);
	void push_underline();
	void push_strikethrough();
	void push_paragraph(Align p_align, Control::TextDirection p_direction = Control::TEXT_DIRECTION_INHERITED, const String &p_language = "", Control::StructuredTextParser p_st_parser = STRUCTURED_TEXT_DEFAULT);
	void push_indent(int p_level);
	void push_list(int p_level, ListType p_list, bool p_capitalize);
	void push_meta(const Variant &p_meta);
	void push_table(int p_columns, InlineAlign p_align = INLINE_ALIGN_TOP);
	void push_fade(int p_start_index, int p_length);
	void push_shake(int p_strength, float p_rate);
	void push_wave(float p_frequency, float p_amplitude);
	void push_tornado(float p_frequency, float p_radius);
	void push_rainbow(float p_saturation, float p_value, float p_frequency);
	void push_bgcolor(const Color &p_color);
	void push_fgcolor(const Color &p_color);
	void push_customfx(Ref<RichTextEffect> p_custom_effect, Dictionary p_environment);
	void set_table_column_expand(int p_column, bool p_expand, int p_ratio = 1);
	void set_cell_row_background_color(const Color &p_odd_row_bg, const Color &p_even_row_bg);
	void set_cell_border_color(const Color &p_color);
	void set_cell_size_override(const Size2 &p_min_size, const Size2 &p_max_size);
	void set_cell_padding(const Rect2 &p_padding);
	int get_current_table_column() const;
	void push_cell();
	void pop();

	void clear();

	void set_offset(int p_pixel);

	void set_meta_underline(bool p_underline);
	bool is_meta_underlined() const;

	void set_override_selected_font_color(bool p_override_selected_font_color);
	bool is_overriding_selected_font_color() const;

	void set_scroll_active(bool p_active);
	bool is_scroll_active() const;

	void set_scroll_follow(bool p_follow);
	bool is_scroll_following() const;

	void set_tab_size(int p_spaces);
	int get_tab_size() const;

	void set_fit_content_height(bool p_enabled);
	bool is_fit_content_height_enabled() const;

	bool search(const String &p_string, bool p_from_selection = false, bool p_search_previous = false);

	void scroll_to_paragraph(int p_paragraph);
	int get_paragraph_count() const;
	int get_visible_paragraph_count() const;

	void scroll_to_line(int p_line);
	int get_line_count() const;
	int get_visible_line_count() const;

	int get_content_height() const;

	VScrollBar *get_v_scroll() { return vscroll; }

	virtual CursorShape get_cursor_shape(const Point2 &p_pos) const override;

	void set_selection_enabled(bool p_enabled);
	bool is_selection_enabled() const;
	int get_selection_from() const;
	int get_selection_to() const;
	String get_selected_text() const;
	void selection_copy();
	void set_deselect_on_focus_loss_enabled(const bool p_enabled);
	bool is_deselect_on_focus_loss_enabled() const;

	void parse_bbcode(const String &p_bbcode);
	void append_text(const String &p_bbcode);

	void set_use_bbcode(bool p_enable);
	bool is_using_bbcode() const;

	void set_text(const String &p_bbcode);
	String get_text() const;

	void set_text_direction(TextDirection p_text_direction);
	TextDirection get_text_direction() const;

	void set_language(const String &p_language);
	String get_language() const;

	void set_structured_text_bidi_override(Control::StructuredTextParser p_parser);
	Control::StructuredTextParser get_structured_text_bidi_override() const;

	void set_structured_text_bidi_override_options(Array p_args);
	Array get_structured_text_bidi_override_options() const;

	void set_visible_characters(int p_visible);
	int get_visible_characters() const;
	int get_total_character_count() const;
	int get_total_glyph_count() const;

	void set_percent_visible(float p_percent);
	float get_percent_visible() const;

	VisibleCharactersBehavior get_visible_characters_behavior() const;
	void set_visible_characters_behavior(VisibleCharactersBehavior p_behavior);

	void set_effects(Array p_effects);
	Array get_effects();

	void install_effect(const Variant effect);

	void set_fixed_size_to_width(int p_width);
	virtual Size2 get_minimum_size() const override;

	RichTextLabel();
	~RichTextLabel();
};

VARIANT_ENUM_CAST(RichTextLabel::Align);
VARIANT_ENUM_CAST(RichTextLabel::ListType);
VARIANT_ENUM_CAST(RichTextLabel::ItemType);
VARIANT_ENUM_CAST(RichTextLabel::VisibleCharactersBehavior);

#endif // RICH_TEXT_LABEL_H
