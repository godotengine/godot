/*************************************************************************/
/*  item_list.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "item_list.h"
#include "os/os.h"
#include "project_settings.h"

void ItemList::add_item(const String &p_item, const Ref<Texture> &p_texture, bool p_selectable) {

	Item item;
	item.icon = p_texture;
	item.icon_region = Rect2i();
	item.text = p_item;
	item.selectable = p_selectable;
	item.selected = false;
	item.disabled = false;
	item.tooltip_enabled = true;
	item.custom_bg = Color(0, 0, 0, 0);
	items.push_back(item);

	update();
	shape_changed = true;
}

void ItemList::add_icon_item(const Ref<Texture> &p_item, bool p_selectable) {

	Item item;
	item.icon = p_item;
	item.icon_region = Rect2i();
	//item.text=p_item;
	item.selectable = p_selectable;
	item.selected = false;
	item.disabled = false;
	item.tooltip_enabled = true;
	item.custom_bg = Color(0, 0, 0, 0);
	items.push_back(item);

	update();
	shape_changed = true;
}

void ItemList::set_item_text(int p_idx, const String &p_text) {

	ERR_FAIL_INDEX(p_idx, items.size());

	items[p_idx].text = p_text;
	update();
	shape_changed = true;
}

String ItemList::get_item_text(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), String());
	return items[p_idx].text;
}

void ItemList::set_item_tooltip_enabled(int p_idx, const bool p_enabled) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items[p_idx].tooltip_enabled = p_enabled;
}

bool ItemList::is_item_tooltip_enabled(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].tooltip_enabled;
}

void ItemList::set_item_tooltip(int p_idx, const String &p_tooltip) {

	ERR_FAIL_INDEX(p_idx, items.size());

	items[p_idx].tooltip = p_tooltip;
	update();
	shape_changed = true;
}

String ItemList::get_item_tooltip(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), String());
	return items[p_idx].tooltip;
}

void ItemList::set_item_icon(int p_idx, const Ref<Texture> &p_icon) {

	ERR_FAIL_INDEX(p_idx, items.size());

	items[p_idx].icon = p_icon;
	update();
	shape_changed = true;
}

Ref<Texture> ItemList::get_item_icon(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), Ref<Texture>());

	return items[p_idx].icon;
}

void ItemList::set_item_icon_region(int p_idx, const Rect2 &p_region) {

	ERR_FAIL_INDEX(p_idx, items.size());

	items[p_idx].icon_region = p_region;
	update();
	shape_changed = true;
}

Rect2 ItemList::get_item_icon_region(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), Rect2());

	return items[p_idx].icon_region;
}

void ItemList::set_item_custom_bg_color(int p_idx, const Color &p_custom_bg_color) {

	ERR_FAIL_INDEX(p_idx, items.size());

	items[p_idx].custom_bg = p_custom_bg_color;
}

Color ItemList::get_item_custom_bg_color(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), Color());

	return items[p_idx].custom_bg;
}

void ItemList::set_item_custom_fg_color(int p_idx, const Color &p_custom_fg_color) {

	ERR_FAIL_INDEX(p_idx, items.size());

	items[p_idx].custom_fg = p_custom_fg_color;
}

Color ItemList::get_item_custom_fg_color(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), Color());

	return items[p_idx].custom_fg;
}

void ItemList::set_item_tag_icon(int p_idx, const Ref<Texture> &p_tag_icon) {

	ERR_FAIL_INDEX(p_idx, items.size());

	items[p_idx].tag_icon = p_tag_icon;
	update();
	shape_changed = true;
}
Ref<Texture> ItemList::get_item_tag_icon(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), Ref<Texture>());

	return items[p_idx].tag_icon;
}

void ItemList::set_item_selectable(int p_idx, bool p_selectable) {

	ERR_FAIL_INDEX(p_idx, items.size());

	items[p_idx].selectable = p_selectable;
}

bool ItemList::is_item_selectable(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].selectable;
}

void ItemList::set_item_disabled(int p_idx, bool p_disabled) {

	ERR_FAIL_INDEX(p_idx, items.size());

	items[p_idx].disabled = p_disabled;
	update();
}

bool ItemList::is_item_disabled(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].disabled;
}

void ItemList::set_item_metadata(int p_idx, const Variant &p_metadata) {

	ERR_FAIL_INDEX(p_idx, items.size());

	items[p_idx].metadata = p_metadata;
	update();
	shape_changed = true;
}

Variant ItemList::get_item_metadata(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), Variant());
	return items[p_idx].metadata;
}
void ItemList::select(int p_idx, bool p_single) {

	ERR_FAIL_INDEX(p_idx, items.size());

	if (p_single || select_mode == SELECT_SINGLE) {

		if (!items[p_idx].selectable || items[p_idx].disabled) {
			return;
		}

		for (int i = 0; i < items.size(); i++) {
			items[i].selected = p_idx == i;
		}

		current = p_idx;
		ensure_selected_visible = false;
	} else {

		if (items[p_idx].selectable && !items[p_idx].disabled) {
			items[p_idx].selected = true;
		}
	}
	update();
}
void ItemList::unselect(int p_idx) {

	ERR_FAIL_INDEX(p_idx, items.size());

	if (select_mode != SELECT_MULTI) {
		items[p_idx].selected = false;
		current = -1;
	} else {
		items[p_idx].selected = false;
	}
	update();
}
bool ItemList::is_selected(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), false);

	return items[p_idx].selected;
}

void ItemList::set_current(int p_current) {
	ERR_FAIL_INDEX(p_current, items.size());

	if (select_mode == SELECT_SINGLE)
		select(p_current, true);
	else {
		current = p_current;
		update();
	}
}

int ItemList::get_current() const {

	return current;
}

void ItemList::move_item(int p_item, int p_to_pos) {

	ERR_FAIL_INDEX(p_item, items.size());
	ERR_FAIL_INDEX(p_to_pos, items.size() + 1);

	Item it = items[p_item];
	items.remove(p_item);

	if (p_to_pos > p_item) {
		p_to_pos--;
	}

	if (p_to_pos >= items.size()) {
		items.push_back(it);
	} else {
		items.insert(p_to_pos, it);
	}

	if (current < 0) {
		//do none
	} else if (p_item == current) {
		current = p_to_pos;
	} else if (p_to_pos > p_item && current > p_item && current < p_to_pos) {
		current--;
	} else if (p_to_pos < p_item && current < p_item && current > p_to_pos) {
		current++;
	}

	update();
}

int ItemList::get_item_count() const {

	return items.size();
}
void ItemList::remove_item(int p_idx) {

	ERR_FAIL_INDEX(p_idx, items.size());

	items.remove(p_idx);
	update();
	shape_changed = true;
	defer_select_single = -1;
}

void ItemList::clear() {

	items.clear();
	current = -1;
	ensure_selected_visible = false;
	update();
	shape_changed = true;
	defer_select_single = -1;
}

void ItemList::set_fixed_column_width(int p_size) {

	ERR_FAIL_COND(p_size < 0);
	fixed_column_width = p_size;
	update();
	shape_changed = true;
}
int ItemList::get_fixed_column_width() const {

	return fixed_column_width;
}

void ItemList::set_same_column_width(bool p_enable) {

	same_column_width = p_enable;
	update();
	shape_changed = true;
}
bool ItemList::is_same_column_width() const {

	return same_column_width;
}

void ItemList::set_max_text_lines(int p_lines) {

	ERR_FAIL_COND(p_lines < 1);
	max_text_lines = p_lines;
	update();
	shape_changed = true;
}
int ItemList::get_max_text_lines() const {

	return max_text_lines;
}

void ItemList::set_max_columns(int p_amount) {

	ERR_FAIL_COND(p_amount < 0);
	max_columns = p_amount;
	update();
}
int ItemList::get_max_columns() const {

	return max_columns;
}

void ItemList::set_select_mode(SelectMode p_mode) {

	select_mode = p_mode;
	update();
}

ItemList::SelectMode ItemList::get_select_mode() const {

	return select_mode;
}

void ItemList::set_icon_mode(IconMode p_mode) {

	icon_mode = p_mode;
	update();
	shape_changed = true;
}
ItemList::IconMode ItemList::get_icon_mode() const {

	return icon_mode;
}

void ItemList::set_fixed_icon_size(const Size2 &p_size) {

	fixed_icon_size = p_size;
	update();
}

Size2 ItemList::get_fixed_icon_size() const {

	return fixed_icon_size;
}
Size2 ItemList::Item::get_icon_size() const {

	if (icon.is_null())
		return Size2();
	if (icon_region.has_no_area())
		return icon->get_size();

	return icon_region.size;
}

void ItemList::_gui_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseMotion> mm = p_event;
	if (defer_select_single >= 0 && mm.is_valid()) {
		defer_select_single = -1;
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (defer_select_single >= 0 && mb.is_valid() && mb->get_button_index() == BUTTON_LEFT && !mb->is_pressed()) {

		select(defer_select_single, true);

		emit_signal("multi_selected", defer_select_single, true);
		defer_select_single = -1;
		return;
	}

	if (mb.is_valid() && (mb->get_button_index() == BUTTON_LEFT || (allow_rmb_select && mb->get_button_index() == BUTTON_RIGHT)) && mb->is_pressed()) {

		search_string = ""; //any mousepress cancels
		Vector2 pos = mb->get_position();
		Ref<StyleBox> bg = get_stylebox("bg");
		pos -= bg->get_offset();
		pos.y += scroll_bar->get_value();

		int closest = -1;

		for (int i = 0; i < items.size(); i++) {

			Rect2 rc = items[i].rect_cache;
			if (i % current_columns == current_columns - 1) {
				rc.size.width = get_size().width; //not right but works
			}

			if (rc.has_point(pos)) {
				closest = i;
				break;
			}
		}

		if (closest != -1) {

			int i = closest;

			if (select_mode == SELECT_MULTI && items[i].selected && mb->get_command()) {
				unselect(i);
				emit_signal("multi_selected", i, false);

			} else if (select_mode == SELECT_MULTI && mb->get_shift() && current >= 0 && current < items.size() && current != i) {

				int from = current;
				int to = i;
				if (i < current) {
					SWAP(from, to);
				}
				for (int j = from; j <= to; j++) {
					bool selected = !items[j].selected;
					select(j, false);
					if (selected)
						emit_signal("multi_selected", i, true);
				}

				if (mb->get_button_index() == BUTTON_RIGHT) {

					emit_signal("item_rmb_selected", i, get_local_mouse_position());
				}
			} else {

				if (!mb->is_doubleclick() && !mb->get_command() && select_mode == SELECT_MULTI && items[i].selectable && !items[i].disabled && items[i].selected && mb->get_button_index() == BUTTON_LEFT) {
					defer_select_single = i;
					return;
				}

				if (items[i].selected && mb->get_button_index() == BUTTON_RIGHT) {

					emit_signal("item_rmb_selected", i, get_local_mouse_position());
				} else {
					bool selected = !items[i].selected;

					select(i, select_mode == SELECT_SINGLE || !mb->get_command());

					if (selected) {
						if (select_mode == SELECT_SINGLE) {
							emit_signal("item_selected", i);
						} else
							emit_signal("multi_selected", i, true);
					}

					if (mb->get_button_index() == BUTTON_RIGHT) {

						emit_signal("item_rmb_selected", i, get_local_mouse_position());
					} else if (/*select_mode==SELECT_SINGLE &&*/ mb->is_doubleclick()) {

						emit_signal("item_activated", i);
					}
				}
			}

			return;
		}
	}
	if (mb.is_valid() && mb->get_button_index() == BUTTON_WHEEL_UP && mb->is_pressed()) {

		scroll_bar->set_value(scroll_bar->get_value() - scroll_bar->get_page() * mb->get_factor() / 8);
	}
	if (mb.is_valid() && mb->get_button_index() == BUTTON_WHEEL_DOWN && mb->is_pressed()) {

		scroll_bar->set_value(scroll_bar->get_value() + scroll_bar->get_page() * mb->get_factor() / 8);
	}

	if (p_event->is_pressed() && items.size() > 0) {
		if (p_event->is_action("ui_up")) {

			if (search_string != "") {

				uint64_t now = OS::get_singleton()->get_ticks_msec();
				uint64_t diff = now - search_time_msec;

				if (diff < uint64_t(ProjectSettings::get_singleton()->get("gui/timers/incremental_search_max_interval_msec")) * 2) {

					for (int i = current - 1; i >= 0; i--) {

						if (items[i].text.begins_with(search_string)) {

							set_current(i);
							ensure_current_is_visible();
							if (select_mode == SELECT_SINGLE) {
								emit_signal("item_selected", current);
							}

							break;
						}
					}
					accept_event();
					return;
				}
			}

			if (current >= current_columns) {
				set_current(current - current_columns);
				ensure_current_is_visible();
				if (select_mode == SELECT_SINGLE) {
					emit_signal("item_selected", current);
				}
				accept_event();
			}
		} else if (p_event->is_action("ui_down")) {

			if (search_string != "") {

				uint64_t now = OS::get_singleton()->get_ticks_msec();
				uint64_t diff = now - search_time_msec;

				if (diff < uint64_t(ProjectSettings::get_singleton()->get("gui/timers/incremental_search_max_interval_msec")) * 2) {

					for (int i = current + 1; i < items.size(); i++) {

						if (items[i].text.begins_with(search_string)) {

							set_current(i);
							ensure_current_is_visible();
							if (select_mode == SELECT_SINGLE) {
								emit_signal("item_selected", current);
							}
							break;
						}
					}
					accept_event();
					return;
				}
			}

			if (current < items.size() - current_columns) {
				set_current(current + current_columns);
				ensure_current_is_visible();
				if (select_mode == SELECT_SINGLE) {
					emit_signal("item_selected", current);
				}
				accept_event();
			}
		} else if (p_event->is_action("ui_page_up")) {

			search_string = ""; //any mousepress cancels

			for (int i = 4; i > 0; i--) {
				if (current - current_columns * i >= 0) {
					set_current(current - current_columns * i);
					ensure_current_is_visible();
					if (select_mode == SELECT_SINGLE) {
						emit_signal("item_selected", current);
					}
					accept_event();
					break;
				}
			}
		} else if (p_event->is_action("ui_page_down")) {

			search_string = ""; //any mousepress cancels

			for (int i = 4; i > 0; i--) {
				if (current + current_columns * i < items.size()) {
					set_current(current + current_columns * i);
					ensure_current_is_visible();
					if (select_mode == SELECT_SINGLE) {
						emit_signal("item_selected", current);
					}
					accept_event();

					break;
				}
			}
		} else if (p_event->is_action("ui_left")) {

			search_string = ""; //any mousepress cancels

			if (current % current_columns != 0) {
				set_current(current - 1);
				ensure_current_is_visible();
				if (select_mode == SELECT_SINGLE) {
					emit_signal("item_selected", current);
				}
				accept_event();
			}
		} else if (p_event->is_action("ui_right")) {

			search_string = ""; //any mousepress cancels

			if (current % current_columns != (current_columns - 1)) {
				set_current(current + 1);
				ensure_current_is_visible();
				if (select_mode == SELECT_SINGLE) {
					emit_signal("item_selected", current);
				}
				accept_event();
			}
		} else if (p_event->is_action("ui_cancel")) {
			search_string = "";
		} else if (p_event->is_action("ui_select")) {

			if (select_mode == SELECT_MULTI && current >= 0 && current < items.size()) {
				if (items[current].selectable && !items[current].disabled && !items[current].selected) {
					select(current, false);
					emit_signal("multi_selected", current, true);
				} else if (items[current].selected) {
					unselect(current);
					emit_signal("multi_selected", current, false);
				}
			}
		} else if (p_event->is_action("ui_accept")) {
			search_string = ""; //any mousepress cance

			if (current >= 0 && current < items.size()) {
				emit_signal("item_activated", current);
			}
		} else {

			Ref<InputEventKey> k = p_event;

			if (k.is_valid() && k->get_unicode()) {

				uint64_t now = OS::get_singleton()->get_ticks_msec();
				uint64_t diff = now - search_time_msec;
				uint64_t max_interval = uint64_t(GLOBAL_DEF("gui/timers/incremental_search_max_interval_msec", 2000));
				search_time_msec = now;

				if (diff > max_interval) {
					search_string = "";
				}

				search_string += String::chr(k->get_unicode());
				for (int i = 0; i < items.size(); i++) {
					if (items[i].text.begins_with(search_string)) {
						set_current(i);
						ensure_current_is_visible();
						if (select_mode == SELECT_SINGLE) {
							emit_signal("item_selected", current);
						}
						break;
					}
				}
			}
		}
	}
}

void ItemList::ensure_current_is_visible() {

	ensure_selected_visible = true;
	update();
}

static Rect2 _adjust_to_max_size(Size2 p_size, Size2 p_max_size) {

	Size2 size = p_max_size;
	int tex_width = p_size.width * size.height / p_size.height;
	int tex_height = size.height;

	if (tex_width > size.width) {
		tex_width = size.width;
		tex_height = p_size.height * tex_width / p_size.width;
	}

	int ofs_x = (size.width - tex_width) / 2;
	int ofs_y = (size.height - tex_height) / 2;

	return Rect2(ofs_x, ofs_y, tex_width, tex_height);
}

void ItemList::_notification(int p_what) {

	if (p_what == NOTIFICATION_RESIZED) {
		shape_changed = true;
		update();
	}

	if (p_what == NOTIFICATION_DRAW) {

		Ref<StyleBox> bg = get_stylebox("bg");

		int mw = scroll_bar->get_minimum_size().x;
		scroll_bar->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_END, -mw);
		scroll_bar->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
		scroll_bar->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, bg->get_margin(MARGIN_TOP));
		scroll_bar->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, -bg->get_margin(MARGIN_BOTTOM));

		Size2 size = get_size();

		int width = size.width - bg->get_minimum_size().width;
		if (scroll_bar->is_visible()) {
			width -= mw + bg->get_margin(MARGIN_RIGHT);
		}

		draw_style_box(bg, Rect2(Point2(), size));

		int hseparation = get_constant("hseparation");
		int vseparation = get_constant("vseparation");
		int icon_margin = get_constant("icon_margin");
		int line_separation = get_constant("line_separation");

		Ref<StyleBox> sbsel = has_focus() ? get_stylebox("selected_focus") : get_stylebox("selected");
		Ref<StyleBox> cursor = has_focus() ? get_stylebox("cursor") : get_stylebox("cursor_unfocused");

		Ref<Font> font = get_font("font");
		Color guide_color = get_color("guide_color");
		Color font_color = get_color("font_color");
		Color font_color_selected = get_color("font_color_selected");
		int font_height = font->get_height();
		Vector<int> line_size_cache;
		Vector<int> line_limit_cache;

		if (max_text_lines) {
			line_size_cache.resize(max_text_lines);
			line_limit_cache.resize(max_text_lines);
		}

		if (has_focus()) {
			VisualServer::get_singleton()->canvas_item_add_clip_ignore(get_canvas_item(), true);
			draw_style_box(get_stylebox("bg_focus"), Rect2(Point2(), size));
			VisualServer::get_singleton()->canvas_item_add_clip_ignore(get_canvas_item(), false);
		}

		if (shape_changed) {

			float max_column_width = 0;

			//1- compute item minimum sizes
			for (int i = 0; i < items.size(); i++) {

				Size2 minsize;
				if (items[i].icon.is_valid()) {

					if (fixed_icon_size.x > 0 && fixed_icon_size.y > 0) {
						minsize = fixed_icon_size * icon_scale;
					} else {
						minsize = items[i].get_icon_size() * icon_scale;
					}

					if (items[i].text != "") {
						if (icon_mode == ICON_MODE_TOP) {
							minsize.y += icon_margin;
						} else {
							minsize.x += icon_margin;
						}
					}
				}

				if (items[i].text != "") {

					Size2 s = font->get_string_size(items[i].text);
					//s.width=MIN(s.width,fixed_column_width);

					if (icon_mode == ICON_MODE_TOP) {
						minsize.x = MAX(minsize.x, s.width);
						if (max_text_lines > 0) {
							minsize.y += (font_height + line_separation) * max_text_lines;
						} else {
							minsize.y += s.height;
						}

					} else {
						minsize.y = MAX(minsize.y, s.height);
						minsize.x += s.width;
					}
				}

				if (fixed_column_width > 0)
					minsize.x = fixed_column_width;
				max_column_width = MAX(max_column_width, minsize.x);

				// elements need to adapt to the selected size
				minsize.y += vseparation;
				minsize.x += hseparation;
				items[i].rect_cache.size = minsize;
				items[i].min_rect_cache.size = minsize;
			}

			int fit_size = size.x - bg->get_minimum_size().width - mw;

			//2-attempt best fit
			current_columns = 0x7FFFFFFF;
			if (max_columns > 0)
				current_columns = max_columns;

			while (true) {
				//repeat util all fits
				bool all_fit = true;
				Vector2 ofs;
				int col = 0;
				int max_h = 0;
				separators.clear();
				for (int i = 0; i < items.size(); i++) {

					if (current_columns > 1 && items[i].rect_cache.size.width + ofs.x > fit_size) {
						//went past
						current_columns = MAX(col, 1);
						all_fit = false;
						break;
					}

					if (same_column_width)
						items[i].rect_cache.size.x = max_column_width;
					items[i].rect_cache.position = ofs;
					max_h = MAX(max_h, items[i].rect_cache.size.y);
					ofs.x += items[i].rect_cache.size.x + hseparation;
					col++;
					if (col == current_columns) {

						if (i < items.size() - 1)
							separators.push_back(ofs.y + max_h + vseparation / 2);

						for (int j = i; j >= 0 && col > 0; j--, col--) {
							items[j].rect_cache.size.y = max_h;
						}

						ofs.x = 0;
						ofs.y += max_h + vseparation;
						col = 0;
						max_h = 0;
					}
				}

				for (int j = items.size() - 1; j >= 0 && col > 0; j--, col--) {
					items[j].rect_cache.size.y = max_h;
				}

				if (all_fit) {
					float page = size.height - bg->get_minimum_size().height;
					float max = MAX(page, ofs.y + max_h);
					if (auto_height)
						auto_height_value = ofs.y + max_h + bg->get_minimum_size().height;
					scroll_bar->set_max(max);
					scroll_bar->set_page(page);
					if (max <= page) {
						scroll_bar->set_value(0);
						scroll_bar->hide();
					} else {
						scroll_bar->show();
					}
					break;
				}
			}

			shape_changed = false;
		}

		//ensure_selected_visible needs to be checked before we draw the list.
		if (ensure_selected_visible && current >= 0 && current <= items.size()) {

			Rect2 r = items[current].rect_cache;
			int from = scroll_bar->get_value();
			int to = from + scroll_bar->get_page();

			if (r.position.y < from) {
				scroll_bar->set_value(r.position.y);
			} else if (r.position.y + r.size.y > to) {
				scroll_bar->set_value(r.position.y + r.size.y - (to - from));
			}
		}

		ensure_selected_visible = false;

		Vector2 base_ofs = bg->get_offset();
		base_ofs.y -= int(scroll_bar->get_value());

		Rect2 clip(Point2(), size - bg->get_minimum_size() + Vector2(0, scroll_bar->get_value()));

		for (int i = 0; i < items.size(); i++) {

			Rect2 rcache = items[i].rect_cache;

			if (!clip.intersects(rcache))
				continue;

			if (current_columns == 1) {
				rcache.size.width = width - rcache.position.x;
			}

			if (items[i].selected) {
				Rect2 r = rcache;
				r.position += base_ofs;
				r.position.y -= vseparation / 2;
				r.size.y += vseparation;
				r.position.x -= hseparation / 2;
				r.size.x += hseparation;

				draw_style_box(sbsel, r);
			}
			if (items[i].custom_bg.a > 0.001) {
				Rect2 r = rcache;
				r.position += base_ofs;

				// Size rect to make the align the temperature colors
				r.position.y -= vseparation / 2;
				r.size.y += vseparation;
				r.position.x -= hseparation / 2;
				r.size.x += hseparation;

				draw_rect(r, items[i].custom_bg);
			}

			Vector2 text_ofs;
			if (items[i].icon.is_valid()) {

				Size2 icon_size;
				//= _adjust_to_max_size(items[i].get_icon_size(),fixed_icon_size) * icon_scale;

				if (fixed_icon_size.x > 0 && fixed_icon_size.y > 0) {
					icon_size = fixed_icon_size * icon_scale;
				} else {
					icon_size = items[i].get_icon_size() * icon_scale;
				}

				Vector2 icon_ofs;

				Point2 pos = items[i].rect_cache.position + icon_ofs + base_ofs;

				if (icon_mode == ICON_MODE_TOP) {

					pos.x += Math::floor((items[i].rect_cache.size.width - icon_size.width) / 2);
					pos.y += MIN(
							Math::floor((items[i].rect_cache.size.height - icon_size.height) / 2),
							items[i].rect_cache.size.height - items[i].min_rect_cache.size.height);
					text_ofs.y = icon_size.height + icon_margin;
					text_ofs.y += items[i].rect_cache.size.height - items[i].min_rect_cache.size.height;
				} else {

					pos.y += Math::floor((items[i].rect_cache.size.height - icon_size.height) / 2);
					text_ofs.x = icon_size.width + icon_margin;
				}

				Rect2 draw_rect = Rect2(pos, icon_size);

				if (fixed_icon_size.x > 0 && fixed_icon_size.y > 0) {
					Rect2 adj = _adjust_to_max_size(items[i].get_icon_size() * icon_scale, icon_size);
					draw_rect.position += adj.position;
					draw_rect.size = adj.size;
				}

				Color modulate = Color(1, 1, 1, 1);
				if (items[i].disabled)
					modulate.a *= 0.5;

				if (items[i].icon_region.has_no_area())
					draw_texture_rect(items[i].icon, draw_rect, false, modulate);
				else
					draw_texture_rect_region(items[i].icon, draw_rect, items[i].icon_region, modulate);
			}

			if (items[i].tag_icon.is_valid()) {

				draw_texture(items[i].tag_icon, items[i].rect_cache.position + base_ofs);
			}

			if (items[i].text != "") {

				int max_len = -1;

				Vector2 size = font->get_string_size(items[i].text);
				if (fixed_column_width)
					max_len = fixed_column_width;
				else if (same_column_width)
					max_len = items[i].rect_cache.size.x;
				else
					max_len = size.x;

				Color modulate = items[i].selected ? font_color_selected : (items[i].custom_fg != Color() ? items[i].custom_fg : font_color);
				if (items[i].disabled)
					modulate.a *= 0.5;

				if (icon_mode == ICON_MODE_TOP && max_text_lines > 0) {

					int ss = items[i].text.length();
					float ofs = 0;
					int line = 0;
					for (int j = 0; j <= ss; j++) {

						int cs = j < ss ? font->get_char_size(items[i].text[j], items[i].text[j + 1]).x : 0;
						if (ofs + cs > max_len || j == ss) {
							line_limit_cache[line] = j;
							line_size_cache[line] = ofs;
							line++;
							ofs = 0;
							if (line >= max_text_lines)
								break;
						} else {
							ofs += cs;
						}
					}

					line = 0;
					ofs = 0;

					text_ofs.y += font->get_ascent();
					text_ofs = text_ofs.floor();
					text_ofs += base_ofs;
					text_ofs += items[i].rect_cache.position;

					for (int j = 0; j < ss; j++) {

						if (j == line_limit_cache[line]) {
							line++;
							ofs = 0;
							if (line >= max_text_lines)
								break;
						}
						ofs += font->draw_char(get_canvas_item(), text_ofs + Vector2(ofs + (max_len - line_size_cache[line]) / 2, line * (font_height + line_separation)).floor(), items[i].text[j], items[i].text[j + 1], modulate);
					}

					//special multiline mode
				} else {

					if (fixed_column_width > 0)
						size.x = MIN(size.x, fixed_column_width);

					if (icon_mode == ICON_MODE_TOP) {
						text_ofs.x += (items[i].rect_cache.size.width - size.x) / 2;
					} else {
						text_ofs.y += (items[i].rect_cache.size.height - size.y) / 2;
					}

					text_ofs.y += font->get_ascent();
					text_ofs = text_ofs.floor();
					text_ofs += base_ofs;
					text_ofs += items[i].rect_cache.position;

					draw_string(font, text_ofs, items[i].text, modulate, max_len + 1);
				}
			}

			if (select_mode == SELECT_MULTI && i == current) {

				Rect2 r = rcache;
				r.position += base_ofs;
				r.position.y -= vseparation / 2;
				r.size.y += vseparation;
				r.position.x -= hseparation / 2;
				r.size.x += hseparation;
				draw_style_box(cursor, r);
			}
		}

		for (int i = 0; i < separators.size(); i++) {
			draw_line(Vector2(bg->get_margin(MARGIN_LEFT), base_ofs.y + separators[i]), Vector2(size.width - bg->get_margin(MARGIN_RIGHT), base_ofs.y + separators[i]), guide_color);
		}
	}
}

void ItemList::_scroll_changed(double) {
	update();
}

int ItemList::get_item_at_position(const Point2 &p_pos, bool p_exact) const {

	Vector2 pos = p_pos;
	Ref<StyleBox> bg = get_stylebox("bg");
	pos -= bg->get_offset();
	pos.y += scroll_bar->get_value();

	int closest = -1;
	int closest_dist = 0x7FFFFFFF;

	for (int i = 0; i < items.size(); i++) {

		Rect2 rc = items[i].rect_cache;
		if (i % current_columns == current_columns - 1) {
			rc.size.width = get_size().width; //not right but works
		}

		if (rc.has_point(pos)) {
			closest = i;
			break;
		}

		float dist = rc.distance_to(pos);
		if (!p_exact && dist < closest_dist) {
			closest = i;
			closest_dist = dist;
		}
	}

	return closest;
}

bool ItemList::is_pos_at_end_of_items(const Point2 &p_pos) const {

	if (items.empty())
		return true;

	Vector2 pos = p_pos;
	Ref<StyleBox> bg = get_stylebox("bg");
	pos -= bg->get_offset();
	pos.y += scroll_bar->get_value();

	Rect2 endrect = items[items.size() - 1].rect_cache;
	return (pos.y > endrect.position.y + endrect.size.y);
}

String ItemList::get_tooltip(const Point2 &p_pos) const {

	int closest = get_item_at_position(p_pos);

	if (closest != -1) {
		if (!items[closest].tooltip_enabled) {
			return "";
		}
		if (items[closest].tooltip != "") {
			return items[closest].tooltip;
		}
		if (items[closest].text != "") {
			return items[closest].text;
		}
	}

	return Control::get_tooltip(p_pos);
}

void ItemList::sort_items_by_text() {

	items.sort();
	update();
	shape_changed = true;

	if (select_mode == SELECT_SINGLE) {
		for (int i = 0; i < items.size(); i++) {
			if (items[i].selected) {
				select(i);
				return;
			}
		}
	}
}

int ItemList::find_metadata(const Variant &p_metadata) const {

	for (int i = 0; i < items.size(); i++) {
		if (items[i].metadata == p_metadata) {
			return i;
		}
	}

	return -1;
}

void ItemList::set_allow_rmb_select(bool p_allow) {
	allow_rmb_select = p_allow;
}

bool ItemList::get_allow_rmb_select() const {

	return allow_rmb_select;
}

void ItemList::set_icon_scale(real_t p_scale) {
	icon_scale = p_scale;
}

real_t ItemList::get_icon_scale() const {
	return icon_scale;
}

Vector<int> ItemList::get_selected_items() {
	Vector<int> selected;
	for (int i = 0; i < items.size(); i++) {
		if (items[i].selected) {
			selected.push_back(i);
			if (this->select_mode == SELECT_SINGLE) {
				break;
			}
		}
	}
	return selected;
}

void ItemList::_set_items(const Array &p_items) {

	ERR_FAIL_COND(p_items.size() % 3);
	clear();

	for (int i = 0; i < p_items.size(); i += 3) {

		String text = p_items[i + 0];
		Ref<Texture> icon = p_items[i + 1];
		bool disabled = p_items[i + 2];

		int idx = get_item_count();
		add_item(text, icon);
		set_item_disabled(idx, disabled);
	}
}

Array ItemList::_get_items() const {

	Array items;
	for (int i = 0; i < get_item_count(); i++) {

		items.push_back(get_item_text(i));
		items.push_back(get_item_icon(i));
		items.push_back(is_item_disabled(i));
	}

	return items;
}

Size2 ItemList::get_minimum_size() const {

	if (auto_height) {
		return Size2(0, auto_height_value);
	}
	return Size2();
}

void ItemList::set_auto_height(bool p_enable) {

	auto_height = p_enable;
	shape_changed = true;
	update();
}

bool ItemList::has_auto_height() const {

	return auto_height;
}

void ItemList::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_item", "text", "icon", "selectable"), &ItemList::add_item, DEFVAL(Variant()), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("add_icon_item", "icon", "selectable"), &ItemList::add_icon_item, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("set_item_text", "idx", "text"), &ItemList::set_item_text);
	ClassDB::bind_method(D_METHOD("get_item_text", "idx"), &ItemList::get_item_text);

	ClassDB::bind_method(D_METHOD("set_item_icon", "idx", "icon"), &ItemList::set_item_icon);
	ClassDB::bind_method(D_METHOD("get_item_icon", "idx"), &ItemList::get_item_icon);

	ClassDB::bind_method(D_METHOD("set_item_icon_region", "idx", "rect"), &ItemList::set_item_icon_region);
	ClassDB::bind_method(D_METHOD("get_item_icon_region", "idx"), &ItemList::get_item_icon_region);

	ClassDB::bind_method(D_METHOD("set_item_selectable", "idx", "selectable"), &ItemList::set_item_selectable);
	ClassDB::bind_method(D_METHOD("is_item_selectable", "idx"), &ItemList::is_item_selectable);

	ClassDB::bind_method(D_METHOD("set_item_disabled", "idx", "disabled"), &ItemList::set_item_disabled);
	ClassDB::bind_method(D_METHOD("is_item_disabled", "idx"), &ItemList::is_item_disabled);

	ClassDB::bind_method(D_METHOD("set_item_metadata", "idx", "metadata"), &ItemList::set_item_metadata);
	ClassDB::bind_method(D_METHOD("get_item_metadata", "idx"), &ItemList::get_item_metadata);

	ClassDB::bind_method(D_METHOD("set_item_custom_bg_color", "idx", "custom_bg_color"), &ItemList::set_item_custom_bg_color);
	ClassDB::bind_method(D_METHOD("get_item_custom_bg_color", "idx"), &ItemList::get_item_custom_bg_color);

	ClassDB::bind_method(D_METHOD("set_item_tooltip_enabled", "idx", "enable"), &ItemList::set_item_tooltip_enabled);
	ClassDB::bind_method(D_METHOD("is_item_tooltip_enabled", "idx"), &ItemList::is_item_tooltip_enabled);

	ClassDB::bind_method(D_METHOD("set_item_tooltip", "idx", "tooltip"), &ItemList::set_item_tooltip);
	ClassDB::bind_method(D_METHOD("get_item_tooltip", "idx"), &ItemList::get_item_tooltip);

	ClassDB::bind_method(D_METHOD("select", "idx", "single"), &ItemList::select, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("unselect", "idx"), &ItemList::unselect);
	ClassDB::bind_method(D_METHOD("is_selected", "idx"), &ItemList::is_selected);
	ClassDB::bind_method(D_METHOD("get_selected_items"), &ItemList::get_selected_items);

	ClassDB::bind_method(D_METHOD("get_item_count"), &ItemList::get_item_count);
	ClassDB::bind_method(D_METHOD("remove_item", "idx"), &ItemList::remove_item);

	ClassDB::bind_method(D_METHOD("clear"), &ItemList::clear);
	ClassDB::bind_method(D_METHOD("sort_items_by_text"), &ItemList::sort_items_by_text);

	ClassDB::bind_method(D_METHOD("set_fixed_column_width", "width"), &ItemList::set_fixed_column_width);
	ClassDB::bind_method(D_METHOD("get_fixed_column_width"), &ItemList::get_fixed_column_width);

	ClassDB::bind_method(D_METHOD("set_same_column_width", "enable"), &ItemList::set_same_column_width);
	ClassDB::bind_method(D_METHOD("is_same_column_width"), &ItemList::is_same_column_width);

	ClassDB::bind_method(D_METHOD("set_max_text_lines", "lines"), &ItemList::set_max_text_lines);
	ClassDB::bind_method(D_METHOD("get_max_text_lines"), &ItemList::get_max_text_lines);

	ClassDB::bind_method(D_METHOD("set_max_columns", "amount"), &ItemList::set_max_columns);
	ClassDB::bind_method(D_METHOD("get_max_columns"), &ItemList::get_max_columns);

	ClassDB::bind_method(D_METHOD("set_select_mode", "mode"), &ItemList::set_select_mode);
	ClassDB::bind_method(D_METHOD("get_select_mode"), &ItemList::get_select_mode);

	ClassDB::bind_method(D_METHOD("set_icon_mode", "mode"), &ItemList::set_icon_mode);
	ClassDB::bind_method(D_METHOD("get_icon_mode"), &ItemList::get_icon_mode);

	ClassDB::bind_method(D_METHOD("set_fixed_icon_size", "size"), &ItemList::set_fixed_icon_size);
	ClassDB::bind_method(D_METHOD("get_fixed_icon_size"), &ItemList::get_fixed_icon_size);

	ClassDB::bind_method(D_METHOD("set_icon_scale", "scale"), &ItemList::set_icon_scale);
	ClassDB::bind_method(D_METHOD("get_icon_scale"), &ItemList::get_icon_scale);

	ClassDB::bind_method(D_METHOD("set_allow_rmb_select", "allow"), &ItemList::set_allow_rmb_select);
	ClassDB::bind_method(D_METHOD("get_allow_rmb_select"), &ItemList::get_allow_rmb_select);

	ClassDB::bind_method(D_METHOD("set_auto_height", "enable"), &ItemList::set_auto_height);
	ClassDB::bind_method(D_METHOD("has_auto_height"), &ItemList::has_auto_height);

	ClassDB::bind_method(D_METHOD("get_item_at_position", "position", "exact"), &ItemList::get_item_at_position, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("ensure_current_is_visible"), &ItemList::ensure_current_is_visible);

	ClassDB::bind_method(D_METHOD("get_v_scroll"), &ItemList::get_v_scroll);

	ClassDB::bind_method(D_METHOD("_scroll_changed"), &ItemList::_scroll_changed);
	ClassDB::bind_method(D_METHOD("_gui_input"), &ItemList::_gui_input);

	ClassDB::bind_method(D_METHOD("_set_items"), &ItemList::_set_items);
	ClassDB::bind_method(D_METHOD("_get_items"), &ItemList::_get_items);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "items", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_items", "_get_items");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "select_mode", PROPERTY_HINT_ENUM, "Single,Multi"), "set_select_mode", "get_select_mode");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "allow_rmb_select"), "set_allow_rmb_select", "get_allow_rmb_select");
	ADD_PROPERTYNO(PropertyInfo(Variant::INT, "max_text_lines"), "set_max_text_lines", "get_max_text_lines");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "auto_height"), "set_auto_height", "has_auto_height");
	ADD_GROUP("Columns", "");
	ADD_PROPERTYNO(PropertyInfo(Variant::INT, "max_columns"), "set_max_columns", "get_max_columns");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "same_column_width"), "set_same_column_width", "is_same_column_width");
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "fixed_column_width"), "set_fixed_column_width", "get_fixed_column_width");
	ADD_GROUP("Icon", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "icon_mode", PROPERTY_HINT_ENUM, "Top,Left"), "set_icon_mode", "get_icon_mode");
	ADD_PROPERTYNO(PropertyInfo(Variant::REAL, "icon_scale"), "set_icon_scale", "get_icon_scale");

	BIND_ENUM_CONSTANT(ICON_MODE_TOP);
	BIND_ENUM_CONSTANT(ICON_MODE_LEFT);

	BIND_ENUM_CONSTANT(SELECT_SINGLE);
	BIND_ENUM_CONSTANT(SELECT_MULTI);

	ADD_SIGNAL(MethodInfo("item_selected", PropertyInfo(Variant::INT, "index")));
	ADD_SIGNAL(MethodInfo("item_rmb_selected", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::VECTOR2, "at_position")));
	ADD_SIGNAL(MethodInfo("multi_selected", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::BOOL, "selected")));
	ADD_SIGNAL(MethodInfo("item_activated", PropertyInfo(Variant::INT, "index")));

	GLOBAL_DEF("gui/timers/incremental_search_max_interval_msec", 2000);
}

ItemList::ItemList() {

	current = -1;

	select_mode = SELECT_SINGLE;
	icon_mode = ICON_MODE_LEFT;

	fixed_column_width = 0;
	same_column_width = false;
	max_text_lines = 1;
	max_columns = 1;
	auto_height = false;
	auto_height_value = 0.0f;

	scroll_bar = memnew(VScrollBar);
	add_child(scroll_bar);

	shape_changed = true;
	scroll_bar->connect("value_changed", this, "_scroll_changed");

	set_focus_mode(FOCUS_ALL);
	current_columns = 1;
	search_time_msec = 0;
	ensure_selected_visible = false;
	defer_select_single = -1;
	allow_rmb_select = false;

	icon_scale = 1.0f;
	set_clip_contents(true);
}

ItemList::~ItemList() {
}
