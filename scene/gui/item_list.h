/*************************************************************************/
/*  item_list.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef ITEMLIST_H
#define ITEMLIST_H

#include "scene/gui/control.h"
#include "scene/gui/scroll_bar.h"

class ItemList : public Control {

	GDCLASS(ItemList, Control);

public:
	enum IconMode {
		ICON_MODE_TOP,
		ICON_MODE_LEFT
	};

	enum SelectMode {
		SELECT_SINGLE,
		SELECT_MULTI
	};

private:
	struct Item {

		Ref<Texture> icon;
		Rect2i icon_region;
		Ref<Texture> tag_icon;
		String text;
		bool selectable;
		bool selected;
		bool disabled;
		bool tooltip_enabled;
		Variant metadata;
		String tooltip;
		Color custom_bg;

		Rect2 rect_cache;
		Rect2 min_rect_cache;

		Size2 get_icon_size() const;

		bool operator<(const Item &p_another) const { return text < p_another.text; }
	};

	int current;

	bool shape_changed;

	bool ensure_selected_visible;
	bool same_column_width;

	Vector<Item> items;
	Vector<int> separators;

	SelectMode select_mode;
	IconMode icon_mode;
	VScrollBar *scroll_bar;

	uint64_t search_time_msec;
	String search_string;

	int current_columns;
	int fixed_column_width;
	int max_text_lines;
	int max_columns;

	Size2 fixed_icon_size;

	Size2 max_item_size_cache;

	int defer_select_single;

	bool allow_rmb_select;

	real_t icon_scale;

	void _scroll_changed(double);
	void _gui_input(const InputEvent &p_event);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void add_item(const String &p_item, const Ref<Texture> &p_texture = Ref<Texture>(), bool p_selectable = true);
	void add_icon_item(const Ref<Texture> &p_item, bool p_selectable = true);

	void set_item_text(int p_idx, const String &p_text);
	String get_item_text(int p_idx) const;

	void set_item_icon(int p_idx, const Ref<Texture> &p_icon);
	Ref<Texture> get_item_icon(int p_idx) const;

	void set_item_icon_region(int p_idx, const Rect2 &p_region);
	Rect2 get_item_icon_region(int p_idx) const;

	void set_item_selectable(int p_idx, bool p_selectable);
	bool is_item_selectable(int p_idx) const;

	void set_item_disabled(int p_idx, bool p_disabled);
	bool is_item_disabled(int p_idx) const;

	void set_item_metadata(int p_idx, const Variant &p_metadata);
	Variant get_item_metadata(int p_idx) const;

	void set_item_tag_icon(int p_idx, const Ref<Texture> &p_tag_icon);
	Ref<Texture> get_item_tag_icon(int p_idx) const;

	void set_item_tooltip_enabled(int p_idx, const bool p_enabled);
	bool is_item_tooltip_enabled(int p_idx) const;

	void set_item_tooltip(int p_idx, const String &p_tooltip);
	String get_item_tooltip(int p_idx) const;

	void set_item_custom_bg_color(int p_idx, const Color &p_custom_bg_color);
	Color get_item_custom_bg_color(int p_idx) const;

	void select(int p_idx, bool p_single = true);
	void unselect(int p_idx);
	bool is_selected(int p_idx) const;
	Vector<int> get_selected_items();

	void set_current(int p_current);
	int get_current() const;

	void move_item(int p_item, int p_to_pos);

	int get_item_count() const;
	void remove_item(int p_idx);

	void clear();

	void set_fixed_column_width(int p_size);
	int get_fixed_column_width() const;

	void set_same_column_width(bool p_enable);
	int is_same_column_width() const;

	void set_max_text_lines(int p_amount);
	int get_max_text_lines() const;

	void set_max_columns(int p_amount);
	int get_max_columns() const;

	void set_select_mode(SelectMode p_mode);
	SelectMode get_select_mode() const;

	void set_icon_mode(IconMode p_mode);
	IconMode get_icon_mode() const;

	void set_fixed_icon_size(const Size2 &p_size);
	Size2 get_fixed_icon_size() const;

	void set_allow_rmb_select(bool p_allow);
	bool get_allow_rmb_select() const;

	void ensure_current_is_visible();

	void sort_items_by_text();
	int find_metadata(const Variant &p_metadata) const;

	virtual String get_tooltip(const Point2 &p_pos) const;
	int get_item_at_pos(const Point2 &p_pos, bool p_exact = false) const;
	bool is_pos_at_end_of_items(const Point2 &p_pos) const;

	void set_icon_scale(real_t p_scale);
	real_t get_icon_scale() const;

	VScrollBar *get_v_scroll() { return scroll_bar; }

	ItemList();
	~ItemList();
};

VARIANT_ENUM_CAST(ItemList::SelectMode);
VARIANT_ENUM_CAST(ItemList::IconMode);

#endif // ITEMLIST_H
