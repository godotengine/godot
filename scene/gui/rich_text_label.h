/**************************************************************************/
/*  rich_text_label.h                                                     */
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

#ifndef RICH_TEXT_LABEL_H
#define RICH_TEXT_LABEL_H

#include "rich_text_effect.h"
#include "scene/gui/scroll_bar.h"

class RichTextLabel : public Control {
	GDCLASS(RichTextLabel, Control);

public:
	enum Align {

		ALIGN_LEFT,
		ALIGN_CENTER,
		ALIGN_RIGHT,
		ALIGN_FILL
	};

	enum InlineAlign {

		INLINE_ALIGN_TOP,
		INLINE_ALIGN_CENTER,
		INLINE_ALIGN_BASELINE,
		INLINE_ALIGN_BOTTOM
	};

	enum ListType {

		LIST_NUMBERS,
		LIST_LETTERS,
		LIST_DOTS
	};

	enum ItemType {

		ITEM_FRAME,
		ITEM_TEXT,
		ITEM_IMAGE,
		ITEM_NEWLINE,
		ITEM_FONT,
		ITEM_COLOR,
		ITEM_UNDERLINE,
		ITEM_STRIKETHROUGH,
		ITEM_ALIGN,
		ITEM_INDENT,
		ITEM_LIST,
		ITEM_TABLE,
		ITEM_FADE,
		ITEM_SHAKE,
		ITEM_WAVE,
		ITEM_TORNADO,
		ITEM_RAINBOW,
		ITEM_META,
		ITEM_CUSTOMFX
	};

protected:
	static void _bind_methods();

private:
	struct Item;

	struct Line {
		Item *from;
		Vector<int> offset_caches;
		Vector<int> height_caches;
		Vector<int> ascent_caches;
		Vector<int> descent_caches;
		Vector<int> space_caches;
		int height_cache;
		int height_accum_cache;
		int char_count;
		int minimum_width;
		int maximum_width;

		Line() {
			from = nullptr;
			char_count = 0;
		}
	};

	struct Item {
		int index;
		Item *parent;
		ItemType type;
		List<Item *> subitems;
		List<Item *>::Element *E;
		int line;

		void _clear_children() {
			while (subitems.size()) {
				memdelete(subitems.front()->get());
				subitems.pop_front();
			}
		}

		Item() {
			parent = nullptr;
			E = nullptr;
			line = 0;
		}
		virtual ~Item() { _clear_children(); }
	};

	struct ItemFrame : public Item {
		int parent_line;
		bool cell;
		Vector<Line> lines;
		int first_invalid_line;
		ItemFrame *parent_frame;

		ItemFrame() {
			type = ITEM_FRAME;
			parent_frame = nullptr;
			cell = false;
			parent_line = 0;
		}
	};

	struct ItemText : public Item {
		String text;
		ItemText() { type = ITEM_TEXT; }
	};

	struct ItemImage : public Item {
		Ref<Texture> image;
		Size2 size;
		InlineAlign align;
		ItemImage() {
			type = ITEM_IMAGE;
			align = INLINE_ALIGN_BASELINE;
		}
	};

	struct ItemFont : public Item {
		Ref<Font> font;
		ItemFont() { type = ITEM_FONT; }
	};

	struct ItemColor : public Item {
		Color color;
		ItemColor() { type = ITEM_COLOR; }
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

	struct ItemAlign : public Item {
		Align align;
		ItemAlign() { type = ITEM_ALIGN; }
	};

	struct ItemIndent : public Item {
		int level;
		ItemIndent() { type = ITEM_INDENT; }
	};

	struct ItemList : public Item {
		ListType list_type;
		ItemList() { type = ITEM_LIST; }
	};

	struct ItemNewline : public Item {
		ItemNewline() { type = ITEM_NEWLINE; }
	};

	struct ItemTable : public Item {
		struct Column {
			bool expand;
			int expand_ratio;
			int min_width;
			int max_width;
			int width;
		};

		Vector<Column> columns;
		int total_width;
		ItemTable() { type = ITEM_TABLE; }
	};

	struct ItemFade : public Item {
		int starting_index;
		int length;

		ItemFade() { type = ITEM_FADE; }
	};

	struct ItemFX : public Item {
		float elapsed_time;

		ItemFX() {
			elapsed_time = 0.0f;
		}
	};

	struct ItemShake : public ItemFX {
		int strength;
		float rate;
		uint64_t _current_rng;
		uint64_t _previous_rng;

		ItemShake() {
			strength = 0;
			rate = 0.0f;
			_current_rng = 0;
			type = ITEM_SHAKE;
		}

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
		float frequency;
		float amplitude;

		ItemWave() {
			frequency = 1.0f;
			amplitude = 1.0f;
			type = ITEM_WAVE;
		}
	};

	struct ItemTornado : public ItemFX {
		float radius;
		float frequency;

		ItemTornado() {
			radius = 1.0f;
			frequency = 1.0f;
			type = ITEM_TORNADO;
		}
	};

	struct ItemRainbow : public ItemFX {
		float saturation;
		float value;
		float frequency;

		ItemRainbow() {
			saturation = 0.8f;
			value = 0.8f;
			frequency = 1.0f;
			type = ITEM_RAINBOW;
		}
	};

	struct ItemCustomFX : public ItemFX {
		Ref<CharFXTransform> char_fx_transform;
		Ref<RichTextEffect> custom_effect;

		ItemCustomFX() {
			type = ITEM_CUSTOMFX;

			char_fx_transform.instance();
		}

		virtual ~ItemCustomFX() {
			_clear_children();

			char_fx_transform.unref();
			custom_effect.unref();
		}
	};

	ItemFrame *main;
	Item *current;
	ItemFrame *current_frame;

	VScrollBar *vscroll;

	bool scroll_visible;
	bool scroll_follow;
	bool scroll_following;
	bool scroll_active;
	int scroll_w;
	bool scroll_updated;
	bool updating_scroll;
	int current_idx;
	int visible_line_count;

	int tab_size;
	bool underline_meta;
	bool override_selected_font_color;

	Align default_align;

	ItemMeta *meta_hovering;
	Variant current_meta;

	Vector<Ref<RichTextEffect>> custom_effects;

	void _invalidate_current_line(ItemFrame *p_frame);
	void _validate_line_caches(ItemFrame *p_frame);

	void _add_item(Item *p_item, bool p_enter = false, bool p_ensure_newline = false);
	void _remove_item(Item *p_item, const int p_line, const int p_subitem_line);

	struct ProcessState {
		int line_width;
	};

	enum ProcessMode {

		PROCESS_CACHE,
		PROCESS_DRAW,
		PROCESS_POINTER
	};

	struct Selection {
		Item *click;
		int click_char;

		Item *from;
		int from_char;
		Item *to;
		int to_char;

		bool active; // anything selected? i.e. from, to, etc. valid?
		bool enabled; // allow selections?
		bool drag_attempt;
	};

	Selection selection;
	bool deselect_on_focus_loss_enabled;

	int visible_characters;
	float percent_visible;

	bool _is_click_inside_selection() const;
	int _process_line(ItemFrame *p_frame, const Vector2 &p_ofs, int &y, int p_width, int p_line, ProcessMode p_mode, const Ref<Font> &p_base_font, const Color &p_base_color, const Color &p_font_color_shadow, bool p_shadow_as_outline, const Point2 &shadow_ofs, const Point2i &p_click_pos = Point2i(), Item **r_click_item = nullptr, int *r_click_char = nullptr, bool *r_outside = nullptr, int p_char_count = 0);
	void _find_click(ItemFrame *p_frame, const Point2i &p_click, Item **r_click_item = nullptr, int *r_click_char = nullptr, bool *r_outside = nullptr);

	Ref<Font> _find_font(Item *p_item);
	int _find_margin(Item *p_item, const Ref<Font> &p_base_font);
	Align _find_align(Item *p_item);
	Color _find_color(Item *p_item, const Color &p_default_color);
	bool _find_underline(Item *p_item);
	bool _find_strikethrough(Item *p_item);
	bool _find_meta(Item *p_item, Variant *r_meta, ItemMeta **r_item = nullptr);
	bool _find_layout_subitem(Item *from, Item *to);
	bool _find_by_type(Item *p_item, ItemType p_type);
	void _fetch_item_fx_stack(Item *p_item, Vector<ItemFX *> &r_stack);

	void _update_scroll();
	void _update_fx(ItemFrame *p_frame, float p_delta_time);
	void _scroll_changed(double);

	void _gui_input(Ref<InputEvent> p_event);
	Item *_get_next_item(Item *p_item, bool p_free = false);
	Item *_get_prev_item(Item *p_item, bool p_free = false);

	Rect2 _get_text_rect();
	Ref<RichTextEffect> _get_custom_effect_by_code(String p_bbcode_identifier);
	virtual Dictionary parse_expressions_for_values(Vector<String> p_expressions);

	bool use_bbcode;
	String bbcode;

	int fixed_width;

	bool fit_content_height;

protected:
	virtual void _validate_property(PropertyInfo &p_property) const;
	void _notification(int p_what);

public:
	String get_text();
	void add_text(const String &p_text);
	void add_image(const Ref<Texture> &p_image, const int p_width = 0, const int p_height = 0, RichTextLabel::InlineAlign p_align = INLINE_ALIGN_BASELINE);
	void add_newline();
	bool remove_line(const int p_line);
	void push_font(const Ref<Font> &p_font);
	void push_normal();
	void push_bold();
	void push_bold_italics();
	void push_italics();
	void push_mono();
	void push_color(const Color &p_color);
	void push_underline();
	void push_strikethrough();
	void push_align(Align p_align);
	void push_indent(int p_level);
	void push_list(ListType p_list);
	void push_meta(const Variant &p_meta);
	void push_table(int p_columns);
	void push_fade(int p_start_index, int p_length);
	void push_shake(int p_strength, float p_rate);
	void push_wave(float p_frequency, float p_amplitude);
	void push_tornado(float p_frequency, float p_radius);
	void push_rainbow(float p_saturation, float p_value, float p_frequency);
	void push_customfx(Ref<RichTextEffect> p_custom_effect, Dictionary p_environment);
	void set_table_column_expand(int p_column, bool p_expand, int p_ratio = 1);
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

	void scroll_to_line(int p_line);
	int get_line_count() const;
	int get_visible_line_count() const;

	int get_content_height() const;

	VScrollBar *get_v_scroll() { return vscroll; }

	virtual CursorShape get_cursor_shape(const Point2 &p_pos) const;
	virtual Variant get_drag_data(const Point2 &p_point);

	void set_selection_enabled(bool p_enabled);
	bool is_selection_enabled() const;
	String get_selected_text();
	void selection_copy();
	void set_deselect_on_focus_loss_enabled(const bool p_enabled);
	bool is_deselect_on_focus_loss_enabled() const;
	void deselect();

	Error parse_bbcode(const String &p_bbcode);
	Error append_bbcode(const String &p_bbcode);

	void set_use_bbcode(bool p_enable);
	bool is_using_bbcode() const;

	void set_bbcode(const String &p_bbcode);
	String get_bbcode() const;

	void set_text(const String &p_string);

	void set_visible_characters(int p_visible);
	int get_visible_characters() const;
	int get_total_character_count() const;

	void set_percent_visible(float p_percent);
	float get_percent_visible() const;

	void set_effects(const Vector<Variant> &effects);
	Vector<Variant> get_effects();

	void install_effect(const Variant effect);

	void set_fixed_size_to_width(int p_width);
	virtual Size2 get_minimum_size() const;

	RichTextLabel();
	~RichTextLabel();
};

VARIANT_ENUM_CAST(RichTextLabel::Align);
VARIANT_ENUM_CAST(RichTextLabel::InlineAlign);
VARIANT_ENUM_CAST(RichTextLabel::ListType);
VARIANT_ENUM_CAST(RichTextLabel::ItemType);

#endif // RICH_TEXT_LABEL_H
