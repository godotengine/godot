/*************************************************************************/
/*  item_list.cpp                                                        */
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

#include "item_list.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/string/translation.h"

void ItemList::_shape(int p_idx) {
	Item &item = items.write[p_idx];

	item.text_buf->clear();
	if (item.text_direction == Control::TEXT_DIRECTION_INHERITED) {
		item.text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		item.text_buf->set_direction((TextServer::Direction)item.text_direction);
	}
	item.text_buf->add_string(item.text, get_theme_font(SNAME("font")), get_theme_font_size(SNAME("font_size")), item.opentype_features, (item.language != "") ? item.language : TranslationServer::get_singleton()->get_tool_locale());
	if (icon_mode == ICON_MODE_TOP && max_text_lines > 0) {
		item.text_buf->set_flags(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::BREAK_GRAPHEME_BOUND);
	} else {
		item.text_buf->set_flags(TextServer::BREAK_NONE);
	}
	item.text_buf->set_text_overrun_behavior(text_overrun_behavior);
	item.text_buf->set_max_lines_visible(max_text_lines);
}

int ItemList::add_item(const String &p_item, const Ref<Texture2D> &p_texture, bool p_selectable) {
	Item item;
	item.icon = p_texture;
	item.text = p_item;
	item.selectable = p_selectable;
	items.push_back(item);
	int item_id = items.size() - 1;

	_shape(items.size() - 1);

	update();
	shape_changed = true;
	notify_property_list_changed();
	return item_id;
}

int ItemList::add_icon_item(const Ref<Texture2D> &p_item, bool p_selectable) {
	Item item;
	item.icon = p_item;
	item.selectable = p_selectable;
	items.push_back(item);
	int item_id = items.size() - 1;

	update();
	shape_changed = true;
	notify_property_list_changed();
	return item_id;
}

void ItemList::set_item_text(int p_idx, const String &p_text) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].text = p_text;
	_shape(p_idx);
	update();
	shape_changed = true;
}

String ItemList::get_item_text(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), String());
	return items[p_idx].text;
}

void ItemList::set_item_text_direction(int p_idx, Control::TextDirection p_text_direction) {
	ERR_FAIL_INDEX(p_idx, items.size());
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (items[p_idx].text_direction != p_text_direction) {
		items.write[p_idx].text_direction = p_text_direction;
		_shape(p_idx);
		update();
	}
}

Control::TextDirection ItemList::get_item_text_direction(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), TEXT_DIRECTION_INHERITED);
	return items[p_idx].text_direction;
}

void ItemList::clear_item_opentype_features(int p_idx) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].opentype_features.clear();
	_shape(p_idx);
	update();
}

void ItemList::set_item_opentype_feature(int p_idx, const String &p_name, int p_value) {
	ERR_FAIL_INDEX(p_idx, items.size());
	int32_t tag = TS->name_to_tag(p_name);
	if (!items[p_idx].opentype_features.has(tag) || (int)items[p_idx].opentype_features[tag] != p_value) {
		items.write[p_idx].opentype_features[tag] = p_value;
		_shape(p_idx);
		update();
	}
}

int ItemList::get_item_opentype_feature(int p_idx, const String &p_name) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), -1);
	int32_t tag = TS->name_to_tag(p_name);
	if (!items[p_idx].opentype_features.has(tag)) {
		return -1;
	}
	return items[p_idx].opentype_features[tag];
}

void ItemList::set_item_language(int p_idx, const String &p_language) {
	ERR_FAIL_INDEX(p_idx, items.size());
	if (items[p_idx].language != p_language) {
		items.write[p_idx].language = p_language;
		_shape(p_idx);
		update();
	}
}

String ItemList::get_item_language(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), "");
	return items[p_idx].language;
}

void ItemList::set_item_tooltip_enabled(int p_idx, const bool p_enabled) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].tooltip_enabled = p_enabled;
}

bool ItemList::is_item_tooltip_enabled(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].tooltip_enabled;
}

void ItemList::set_item_tooltip(int p_idx, const String &p_tooltip) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].tooltip = p_tooltip;
	update();
	shape_changed = true;
}

String ItemList::get_item_tooltip(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), String());
	return items[p_idx].tooltip;
}

void ItemList::set_item_icon(int p_idx, const Ref<Texture2D> &p_icon) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].icon = p_icon;
	update();
	shape_changed = true;
}

Ref<Texture2D> ItemList::get_item_icon(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), Ref<Texture2D>());

	return items[p_idx].icon;
}

void ItemList::set_item_icon_transposed(int p_idx, const bool p_transposed) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].icon_transposed = p_transposed;
	update();
	shape_changed = true;
}

bool ItemList::is_item_icon_transposed(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);

	return items[p_idx].icon_transposed;
}

void ItemList::set_item_icon_region(int p_idx, const Rect2 &p_region) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].icon_region = p_region;
	update();
	shape_changed = true;
}

Rect2 ItemList::get_item_icon_region(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), Rect2());

	return items[p_idx].icon_region;
}

void ItemList::set_item_icon_modulate(int p_idx, const Color &p_modulate) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].icon_modulate = p_modulate;
	update();
}

Color ItemList::get_item_icon_modulate(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), Color());

	return items[p_idx].icon_modulate;
}

void ItemList::set_item_custom_bg_color(int p_idx, const Color &p_custom_bg_color) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].custom_bg = p_custom_bg_color;
	update();
}

Color ItemList::get_item_custom_bg_color(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), Color());

	return items[p_idx].custom_bg;
}

void ItemList::set_item_custom_fg_color(int p_idx, const Color &p_custom_fg_color) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].custom_fg = p_custom_fg_color;
	update();
}

Color ItemList::get_item_custom_fg_color(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), Color());

	return items[p_idx].custom_fg;
}

void ItemList::set_item_tag_icon(int p_idx, const Ref<Texture2D> &p_tag_icon) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].tag_icon = p_tag_icon;
	update();
	shape_changed = true;
}

Ref<Texture2D> ItemList::get_item_tag_icon(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), Ref<Texture2D>());

	return items[p_idx].tag_icon;
}

void ItemList::set_item_selectable(int p_idx, bool p_selectable) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].selectable = p_selectable;
}

bool ItemList::is_item_selectable(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].selectable;
}

void ItemList::set_item_disabled(int p_idx, bool p_disabled) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].disabled = p_disabled;
	update();
}

bool ItemList::is_item_disabled(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].disabled;
}

void ItemList::set_item_metadata(int p_idx, const Variant &p_metadata) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].metadata = p_metadata;
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
			items.write[i].selected = p_idx == i;
		}

		current = p_idx;
		ensure_selected_visible = false;
	} else {
		if (items[p_idx].selectable && !items[p_idx].disabled) {
			items.write[p_idx].selected = true;
		}
	}
	update();
}

void ItemList::deselect(int p_idx) {
	ERR_FAIL_INDEX(p_idx, items.size());

	if (select_mode != SELECT_MULTI) {
		items.write[p_idx].selected = false;
		current = -1;
	} else {
		items.write[p_idx].selected = false;
	}
	update();
}

void ItemList::deselect_all() {
	if (items.size() < 1) {
		return;
	}

	for (int i = 0; i < items.size(); i++) {
		items.write[i].selected = false;
	}
	current = -1;
	update();
}

bool ItemList::is_selected(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);

	return items[p_idx].selected;
}

void ItemList::set_current(int p_current) {
	ERR_FAIL_INDEX(p_current, items.size());

	if (select_mode == SELECT_SINGLE) {
		select(p_current, true);
	} else {
		current = p_current;
		update();
	}
}

int ItemList::get_current() const {
	return current;
}

void ItemList::move_item(int p_from_idx, int p_to_idx) {
	ERR_FAIL_INDEX(p_from_idx, items.size());
	ERR_FAIL_INDEX(p_to_idx, items.size());

	if (is_anything_selected() && get_selected_items()[0] == p_from_idx) {
		current = p_to_idx;
	}

	Item item = items[p_from_idx];
	items.remove_at(p_from_idx);
	items.insert(p_to_idx, item);

	update();
	shape_changed = true;
	notify_property_list_changed();
}

void ItemList::set_item_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);
	items.resize(p_count);
	update();
	shape_changed = true;
	notify_property_list_changed();
}

int ItemList::get_item_count() const {
	return items.size();
}

void ItemList::remove_item(int p_idx) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.remove_at(p_idx);
	if (current == p_idx) {
		current = -1;
	}
	update();
	shape_changed = true;
	defer_select_single = -1;
	notify_property_list_changed();
}

void ItemList::clear() {
	items.clear();
	current = -1;
	ensure_selected_visible = false;
	update();
	shape_changed = true;
	defer_select_single = -1;
	notify_property_list_changed();
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
	if (max_text_lines != p_lines) {
		max_text_lines = p_lines;
		for (int i = 0; i < items.size(); i++) {
			if (icon_mode == ICON_MODE_TOP && max_text_lines > 0) {
				items.write[i].text_buf->set_flags(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::BREAK_GRAPHEME_BOUND);
				items.write[i].text_buf->set_max_lines_visible(p_lines);
			} else {
				items.write[i].text_buf->set_flags(TextServer::BREAK_NONE);
			}
		}
		shape_changed = true;
		update();
	}
}

int ItemList::get_max_text_lines() const {
	return max_text_lines;
}

void ItemList::set_max_columns(int p_amount) {
	ERR_FAIL_COND(p_amount < 0);
	max_columns = p_amount;
	update();
	shape_changed = true;
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
	ERR_FAIL_INDEX((int)p_mode, 2);
	if (icon_mode != p_mode) {
		icon_mode = p_mode;
		for (int i = 0; i < items.size(); i++) {
			if (icon_mode == ICON_MODE_TOP && max_text_lines > 0) {
				items.write[i].text_buf->set_flags(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::BREAK_GRAPHEME_BOUND);
			} else {
				items.write[i].text_buf->set_flags(TextServer::BREAK_NONE);
			}
		}
		shape_changed = true;
		update();
	}
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
	if (icon.is_null()) {
		return Size2();
	}

	Size2 size_result = Size2(icon_region.size).abs();
	if (icon_region.size.x == 0 || icon_region.size.y == 0) {
		size_result = icon->get_size();
	}

	if (icon_transposed) {
		Size2 size_tmp = size_result;
		size_result.x = size_tmp.y;
		size_result.y = size_tmp.x;
	}

	return size_result;
}

void ItemList::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	double prev_scroll = scroll_bar->get_value();

	Ref<InputEventMouseMotion> mm = p_event;
	if (defer_select_single >= 0 && mm.is_valid()) {
		defer_select_single = -1;
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (defer_select_single >= 0 && mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && !mb->is_pressed()) {
		select(defer_select_single, true);

		emit_signal(SNAME("multi_selected"), defer_select_single, true);
		defer_select_single = -1;
		return;
	}

	if (mb.is_valid() && (mb->get_button_index() == MouseButton::LEFT || (allow_rmb_select && mb->get_button_index() == MouseButton::RIGHT)) && mb->is_pressed()) {
		search_string = ""; //any mousepress cancels
		Vector2 pos = mb->get_position();
		Ref<StyleBox> bg = get_theme_stylebox(SNAME("bg"));
		pos -= bg->get_offset();
		pos.y += scroll_bar->get_value();

		if (is_layout_rtl()) {
			pos.x = get_size().width - pos.x;
		}

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

			if (select_mode == SELECT_MULTI && items[i].selected && mb->is_command_pressed()) {
				deselect(i);
				emit_signal(SNAME("multi_selected"), i, false);

			} else if (select_mode == SELECT_MULTI && mb->is_shift_pressed() && current >= 0 && current < items.size() && current != i) {
				int from = current;
				int to = i;
				if (i < current) {
					SWAP(from, to);
				}
				for (int j = from; j <= to; j++) {
					bool selected = !items[j].selected;
					select(j, false);
					if (selected) {
						emit_signal(SNAME("multi_selected"), j, true);
					}
				}

				if (mb->get_button_index() == MouseButton::RIGHT) {
					emit_signal(SNAME("item_rmb_selected"), i, get_local_mouse_position());
				}
			} else {
				if (!mb->is_double_click() && !mb->is_command_pressed() && select_mode == SELECT_MULTI && items[i].selectable && !items[i].disabled && items[i].selected && mb->get_button_index() == MouseButton::LEFT) {
					defer_select_single = i;
					return;
				}

				if (items[i].selected && mb->get_button_index() == MouseButton::RIGHT) {
					emit_signal(SNAME("item_rmb_selected"), i, get_local_mouse_position());
				} else {
					bool selected = items[i].selected;

					select(i, select_mode == SELECT_SINGLE || !mb->is_command_pressed());

					if (!selected || allow_reselect) {
						if (select_mode == SELECT_SINGLE) {
							emit_signal(SNAME("item_selected"), i);
						} else {
							emit_signal(SNAME("multi_selected"), i, true);
						}
					}

					if (mb->get_button_index() == MouseButton::RIGHT) {
						emit_signal(SNAME("item_rmb_selected"), i, get_local_mouse_position());
					} else if (/*select_mode==SELECT_SINGLE &&*/ mb->is_double_click()) {
						emit_signal(SNAME("item_activated"), i);
					}
				}
			}

			return;
		}
		if (mb->get_button_index() == MouseButton::RIGHT) {
			emit_signal(SNAME("rmb_clicked"), mb->get_position());

			return;
		}

		// Since closest is null, more likely we clicked on empty space, so send signal to interested controls. Allows, for example, implement items deselecting.
		emit_signal(SNAME("nothing_selected"));
	}
	if (mb.is_valid() && mb->get_button_index() == MouseButton::WHEEL_UP && mb->is_pressed()) {
		scroll_bar->set_value(scroll_bar->get_value() - scroll_bar->get_page() * mb->get_factor() / 8);
	}
	if (mb.is_valid() && mb->get_button_index() == MouseButton::WHEEL_DOWN && mb->is_pressed()) {
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
								emit_signal(SNAME("item_selected"), current);
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
					emit_signal(SNAME("item_selected"), current);
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
								emit_signal(SNAME("item_selected"), current);
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
					emit_signal(SNAME("item_selected"), current);
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
						emit_signal(SNAME("item_selected"), current);
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
						emit_signal(SNAME("item_selected"), current);
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
					emit_signal(SNAME("item_selected"), current);
				}
				accept_event();
			}
		} else if (p_event->is_action("ui_right")) {
			search_string = ""; //any mousepress cancels

			if (current % current_columns != (current_columns - 1) && current + 1 < items.size()) {
				set_current(current + 1);
				ensure_current_is_visible();
				if (select_mode == SELECT_SINGLE) {
					emit_signal(SNAME("item_selected"), current);
				}
				accept_event();
			}
		} else if (p_event->is_action("ui_cancel")) {
			search_string = "";
		} else if (p_event->is_action("ui_select") && select_mode == SELECT_MULTI) {
			if (current >= 0 && current < items.size()) {
				if (items[current].selectable && !items[current].disabled && !items[current].selected) {
					select(current, false);
					emit_signal(SNAME("multi_selected"), current, true);
				} else if (items[current].selected) {
					deselect(current);
					emit_signal(SNAME("multi_selected"), current, false);
				}
			}
		} else if (p_event->is_action("ui_accept")) {
			search_string = ""; //any mousepress cancels

			if (current >= 0 && current < items.size()) {
				emit_signal(SNAME("item_activated"), current);
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

				if (String::chr(k->get_unicode()) != search_string) {
					search_string += String::chr(k->get_unicode());
				}

				for (int i = current + 1; i <= items.size(); i++) {
					if (i == items.size()) {
						if (current == 0 || current == -1) {
							break;
						} else {
							i = 0;
						}
					}

					if (i == current) {
						break;
					}

					if (items[i].text.findn(search_string) == 0) {
						set_current(i);
						ensure_current_is_visible();
						if (select_mode == SELECT_SINGLE) {
							emit_signal(SNAME("item_selected"), current);
						}
						break;
					}
				}
			}
		}
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid()) {
		scroll_bar->set_value(scroll_bar->get_value() + scroll_bar->get_page() * pan_gesture->get_delta().y / 8);
	}

	if (scroll_bar->get_value() != prev_scroll) {
		accept_event(); //accept event if scroll changed
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

	if ((p_what == NOTIFICATION_LAYOUT_DIRECTION_CHANGED) || (p_what == NOTIFICATION_TRANSLATION_CHANGED) || (p_what == NOTIFICATION_THEME_CHANGED)) {
		for (int i = 0; i < items.size(); i++) {
			_shape(i);
		}
		shape_changed = true;
		update();
	}

	if (p_what == NOTIFICATION_DRAW) {
		Ref<StyleBox> bg = get_theme_stylebox(SNAME("bg"));

		int mw = scroll_bar->get_minimum_size().x;
		scroll_bar->set_anchor_and_offset(SIDE_LEFT, ANCHOR_END, -mw);
		scroll_bar->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, 0);
		scroll_bar->set_anchor_and_offset(SIDE_TOP, ANCHOR_BEGIN, bg->get_margin(SIDE_TOP));
		scroll_bar->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, -bg->get_margin(SIDE_BOTTOM));

		Size2 size = get_size();

		int width = size.width - bg->get_minimum_size().width;
		if (scroll_bar->is_visible()) {
			width -= mw;
		}

		draw_style_box(bg, Rect2(Point2(), size));

		int hseparation = get_theme_constant(SNAME("hseparation"));
		int vseparation = get_theme_constant(SNAME("vseparation"));
		int icon_margin = get_theme_constant(SNAME("icon_margin"));
		int line_separation = get_theme_constant(SNAME("line_separation"));
		Color font_outline_color = get_theme_color(SNAME("font_outline_color"));
		int outline_size = get_theme_constant(SNAME("outline_size"));

		Ref<StyleBox> sbsel = has_focus() ? get_theme_stylebox(SNAME("selected_focus")) : get_theme_stylebox(SNAME("selected"));
		Ref<StyleBox> cursor = has_focus() ? get_theme_stylebox(SNAME("cursor")) : get_theme_stylebox(SNAME("cursor_unfocused"));
		bool rtl = is_layout_rtl();

		Color guide_color = get_theme_color(SNAME("guide_color"));
		Color font_color = get_theme_color(SNAME("font_color"));
		Color font_selected_color = get_theme_color(SNAME("font_selected_color"));

		if (has_focus()) {
			RenderingServer::get_singleton()->canvas_item_add_clip_ignore(get_canvas_item(), true);
			draw_style_box(get_theme_stylebox(SNAME("bg_focus")), Rect2(Point2(), size));
			RenderingServer::get_singleton()->canvas_item_add_clip_ignore(get_canvas_item(), false);
		}

		if (shape_changed) {
			float max_column_width = 0.0;

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
					int max_width = -1;
					if (fixed_column_width) {
						max_width = fixed_column_width;
					} else if (same_column_width) {
						max_width = items[i].rect_cache.size.x;
					}
					items.write[i].text_buf->set_width(max_width);
					Size2 s = items[i].text_buf->get_size();

					if (icon_mode == ICON_MODE_TOP) {
						minsize.x = MAX(minsize.x, s.width);
						if (max_text_lines > 0) {
							minsize.y += s.height + line_separation * max_text_lines;
						} else {
							minsize.y += s.height;
						}

					} else {
						minsize.y = MAX(minsize.y, s.height);
						minsize.x += s.width;
					}
				}

				if (fixed_column_width > 0) {
					minsize.x = fixed_column_width;
				}
				max_column_width = MAX(max_column_width, minsize.x);

				// elements need to adapt to the selected size
				minsize.y += vseparation;
				minsize.x += hseparation;
				items.write[i].rect_cache.size = minsize;
				items.write[i].min_rect_cache.size = minsize;
			}

			int fit_size = size.x - bg->get_minimum_size().width - mw;

			//2-attempt best fit
			current_columns = 0x7FFFFFFF;
			if (max_columns > 0) {
				current_columns = max_columns;
			}

			while (true) {
				//repeat until all fits
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

					if (same_column_width) {
						items.write[i].rect_cache.size.x = max_column_width;
					}
					items.write[i].rect_cache.position = ofs;
					max_h = MAX(max_h, items[i].rect_cache.size.y);
					ofs.x += items[i].rect_cache.size.x + hseparation;
					col++;
					if (col == current_columns) {
						if (i < items.size() - 1) {
							separators.push_back(ofs.y + max_h + vseparation / 2);
						}

						for (int j = i; j >= 0 && col > 0; j--, col--) {
							items.write[j].rect_cache.size.y = max_h;
						}

						ofs.x = 0;
						ofs.y += max_h + vseparation;
						col = 0;
						max_h = 0;
					}
				}

				for (int j = items.size() - 1; j >= 0 && col > 0; j--, col--) {
					items.write[j].rect_cache.size.y = max_h;
				}

				if (all_fit) {
					float page = MAX(0, size.height - bg->get_minimum_size().height);
					float max = MAX(page, ofs.y + max_h);
					if (auto_height) {
						auto_height_value = ofs.y + max_h + bg->get_minimum_size().height;
					}
					scroll_bar->set_max(max);
					scroll_bar->set_page(page);
					if (max <= page) {
						scroll_bar->set_value(0);
						scroll_bar->hide();
					} else {
						scroll_bar->show();

						if (do_autoscroll_to_bottom) {
							scroll_bar->set_value(max);
						}
					}
					break;
				}
			}

			update_minimum_size();
			shape_changed = false;
		}

		//ensure_selected_visible needs to be checked before we draw the list.
		if (ensure_selected_visible && current >= 0 && current < items.size()) {
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

		const Rect2 clip(-base_ofs, size); // visible frame, don't need to draw outside of there

		int first_item_visible;
		{
			// do a binary search to find the first item whose rect reaches below clip.position.y
			int lo = 0;
			int hi = items.size();
			while (lo < hi) {
				const int mid = (lo + hi) / 2;
				const Rect2 &rcache = items[mid].rect_cache;
				if (rcache.position.y + rcache.size.y < clip.position.y) {
					lo = mid + 1;
				} else {
					hi = mid;
				}
			}
			// we might have ended up with column 2, or 3, ..., so let's find the first column
			while (lo > 0 && items[lo - 1].rect_cache.position.y == items[lo].rect_cache.position.y) {
				lo -= 1;
			}
			first_item_visible = lo;
		}

		for (int i = first_item_visible; i < items.size(); i++) {
			Rect2 rcache = items[i].rect_cache;

			if (rcache.position.y > clip.position.y + clip.size.y) {
				break; // done
			}

			if (!clip.intersects(rcache)) {
				continue;
			}

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

				if (rtl) {
					r.position.x = size.width - r.position.x - r.size.x;
				}

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

				if (rtl) {
					r.position.x = size.width - r.position.x - r.size.x;
				}

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
					pos.y += icon_margin;
					text_ofs.y = icon_size.height + icon_margin * 2;
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

				Color modulate = items[i].icon_modulate;
				if (items[i].disabled) {
					modulate.a *= 0.5;
				}

				// If the icon is transposed, we have to switch the size so that it is drawn correctly
				if (items[i].icon_transposed) {
					Size2 size_tmp = draw_rect.size;
					draw_rect.size.x = size_tmp.y;
					draw_rect.size.y = size_tmp.x;
				}

				Rect2 region = (items[i].icon_region.size.x == 0 || items[i].icon_region.size.y == 0) ? Rect2(Vector2(), items[i].icon->get_size()) : Rect2(items[i].icon_region);

				if (rtl) {
					draw_rect.position.x = size.width - draw_rect.position.x - draw_rect.size.x;
				}
				draw_texture_rect_region(items[i].icon, draw_rect, region, modulate, items[i].icon_transposed);
			}

			if (items[i].tag_icon.is_valid()) {
				Point2 draw_pos = items[i].rect_cache.position;
				if (rtl) {
					draw_pos.x = size.width - draw_pos.x - items[i].tag_icon->get_width();
				}
				draw_texture(items[i].tag_icon, draw_pos + base_ofs);
			}

			if (items[i].text != "") {
				int max_len = -1;

				Vector2 size2 = items[i].text_buf->get_size();
				if (fixed_column_width) {
					max_len = fixed_column_width;
				} else if (same_column_width) {
					max_len = items[i].rect_cache.size.x;
				} else {
					max_len = size2.x;
				}

				Color modulate = items[i].selected ? font_selected_color : (items[i].custom_fg != Color() ? items[i].custom_fg : font_color);
				if (items[i].disabled) {
					modulate.a *= 0.5;
				}

				if (icon_mode == ICON_MODE_TOP && max_text_lines > 0) {
					text_ofs += base_ofs;
					text_ofs += items[i].rect_cache.position;

					if (rtl) {
						text_ofs.x = size.width - text_ofs.x - max_len;
					}

					items.write[i].text_buf->set_align(HALIGN_CENTER);

					if (outline_size > 0 && font_outline_color.a > 0) {
						items[i].text_buf->draw_outline(get_canvas_item(), text_ofs, outline_size, font_outline_color);
					}

					items[i].text_buf->draw(get_canvas_item(), text_ofs, modulate);
				} else {
					if (fixed_column_width > 0) {
						size2.x = MIN(size2.x, fixed_column_width);
					}

					if (icon_mode == ICON_MODE_TOP) {
						text_ofs.x += (items[i].rect_cache.size.width - size2.x) / 2;
					} else {
						text_ofs.y += (items[i].rect_cache.size.height - size2.y) / 2;
					}

					text_ofs += base_ofs;
					text_ofs += items[i].rect_cache.position;

					if (rtl) {
						text_ofs.x = size.width - text_ofs.x - max_len;
					}

					items.write[i].text_buf->set_width(max_len);

					if (rtl) {
						items.write[i].text_buf->set_align(HALIGN_RIGHT);
					} else {
						items.write[i].text_buf->set_align(HALIGN_LEFT);
					}

					if (outline_size > 0 && font_outline_color.a > 0) {
						items[i].text_buf->draw_outline(get_canvas_item(), text_ofs, outline_size, font_outline_color);
					}

					items[i].text_buf->draw(get_canvas_item(), text_ofs, modulate);
				}
			}

			if (select_mode == SELECT_MULTI && i == current) {
				Rect2 r = rcache;
				r.position += base_ofs;
				r.position.y -= vseparation / 2;
				r.size.y += vseparation;
				r.position.x -= hseparation / 2;
				r.size.x += hseparation;

				if (rtl) {
					r.position.x = size.width - r.position.x - r.size.x;
				}

				draw_style_box(cursor, r);
			}
		}

		int first_visible_separator = 0;
		{
			// do a binary search to find the first separator that is below clip_position.y
			int lo = 0;
			int hi = separators.size();
			while (lo < hi) {
				const int mid = (lo + hi) / 2;
				if (separators[mid] < clip.position.y) {
					lo = mid + 1;
				} else {
					hi = mid;
				}
			}
			first_visible_separator = lo;
		}

		for (int i = first_visible_separator; i < separators.size(); i++) {
			if (separators[i] > clip.position.y + clip.size.y) {
				break; // done
			}

			const int y = base_ofs.y + separators[i];
			draw_line(Vector2(bg->get_margin(SIDE_LEFT), y), Vector2(width, y), guide_color);
		}
	}
}

void ItemList::_scroll_changed(double) {
	update();
}

int ItemList::get_item_at_position(const Point2 &p_pos, bool p_exact) const {
	Vector2 pos = p_pos;
	Ref<StyleBox> bg = get_theme_stylebox(SNAME("bg"));
	pos -= bg->get_offset();
	pos.y += scroll_bar->get_value();

	if (is_layout_rtl()) {
		pos.x = get_size().width - pos.x;
	}

	int closest = -1;
	int closest_dist = 0x7FFFFFFF;

	for (int i = 0; i < items.size(); i++) {
		Rect2 rc = items[i].rect_cache;
		if (i % current_columns == current_columns - 1) {
			rc.size.width = get_size().width - rc.position.x; //make sure you can still select the last item when clicking past the column
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
	if (items.is_empty()) {
		return true;
	}

	Vector2 pos = p_pos;
	Ref<StyleBox> bg = get_theme_stylebox(SNAME("bg"));
	pos -= bg->get_offset();
	pos.y += scroll_bar->get_value();

	if (is_layout_rtl()) {
		pos.x = get_size().width - pos.x;
	}

	Rect2 endrect = items[items.size() - 1].rect_cache;
	return (pos.y > endrect.position.y + endrect.size.y);
}

String ItemList::get_tooltip(const Point2 &p_pos) const {
	int closest = get_item_at_position(p_pos, true);

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

void ItemList::set_allow_reselect(bool p_allow) {
	allow_reselect = p_allow;
}

bool ItemList::get_allow_reselect() const {
	return allow_reselect;
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

bool ItemList::is_anything_selected() {
	for (int i = 0; i < items.size(); i++) {
		if (items[i].selected) {
			return true;
		}
	}

	return false;
}

Size2 ItemList::get_minimum_size() const {
	if (auto_height) {
		return Size2(0, auto_height_value);
	}
	return Size2();
}

void ItemList::set_autoscroll_to_bottom(const bool p_enable) {
	do_autoscroll_to_bottom = p_enable;
}

void ItemList::set_auto_height(bool p_enable) {
	auto_height = p_enable;
	shape_changed = true;
	update();
}

bool ItemList::has_auto_height() const {
	return auto_height;
}

void ItemList::set_text_overrun_behavior(TextParagraph::OverrunBehavior p_behavior) {
	if (text_overrun_behavior != p_behavior) {
		text_overrun_behavior = p_behavior;
		for (int i = 0; i < items.size(); i++) {
			items.write[i].text_buf->set_text_overrun_behavior(p_behavior);
		}
		shape_changed = true;
		update();
	}
}

TextParagraph::OverrunBehavior ItemList::get_text_overrun_behavior() const {
	return text_overrun_behavior;
}

bool ItemList::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (components.size() >= 2 && components[0].begins_with("item_") && components[0].trim_prefix("item_").is_valid_int()) {
		int item_index = components[0].trim_prefix("item_").to_int();
		if (components[1] == "text") {
			set_item_text(item_index, p_value);
			return true;
		} else if (components[1] == "icon") {
			set_item_icon(item_index, p_value);
			return true;
		} else if (components[1] == "disabled") {
			set_item_disabled(item_index, p_value);
			return true;
		}
	}
#ifndef DISABLE_DEPRECATED
	// Compatibility.
	if (p_name == "items") {
		Array arr = p_value;
		ERR_FAIL_COND_V(arr.size() % 3, false);
		clear();

		for (int i = 0; i < arr.size(); i += 3) {
			String text = arr[i + 0];
			Ref<Texture2D> icon = arr[i + 1];
			bool disabled = arr[i + 2];

			int idx = get_item_count();
			add_item(text, icon);
			set_item_disabled(idx, disabled);
		}
	}
#endif
	return false;
}

bool ItemList::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (components.size() >= 2 && components[0].begins_with("item_") && components[0].trim_prefix("item_").is_valid_int()) {
		int item_index = components[0].trim_prefix("item_").to_int();
		if (components[1] == "text") {
			r_ret = get_item_text(item_index);
			return true;
		} else if (components[1] == "icon") {
			r_ret = get_item_icon(item_index);
			return true;
		} else if (components[1] == "disabled") {
			r_ret = is_item_disabled(item_index);
			return true;
		}
	}
	return false;
}

void ItemList::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < items.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::STRING, vformat("item_%d/text", i)));

		PropertyInfo pi = PropertyInfo(Variant::OBJECT, vformat("item_%d/icon", i), PROPERTY_HINT_RESOURCE_TYPE, "Texture2D");
		pi.usage &= ~(get_item_icon(i).is_null() ? PROPERTY_USAGE_STORAGE : 0);
		p_list->push_back(pi);

		pi = PropertyInfo(Variant::BOOL, vformat("item_%d/disabled", i));
		pi.usage &= ~(!is_item_disabled(i) ? PROPERTY_USAGE_STORAGE : 0);
		p_list->push_back(pi);
	}
}

void ItemList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_item", "text", "icon", "selectable"), &ItemList::add_item, DEFVAL(Variant()), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("add_icon_item", "icon", "selectable"), &ItemList::add_icon_item, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("set_item_text", "idx", "text"), &ItemList::set_item_text);
	ClassDB::bind_method(D_METHOD("get_item_text", "idx"), &ItemList::get_item_text);

	ClassDB::bind_method(D_METHOD("set_item_icon", "idx", "icon"), &ItemList::set_item_icon);
	ClassDB::bind_method(D_METHOD("get_item_icon", "idx"), &ItemList::get_item_icon);

	ClassDB::bind_method(D_METHOD("set_item_text_direction", "idx", "direction"), &ItemList::set_item_text_direction);
	ClassDB::bind_method(D_METHOD("get_item_text_direction", "idx"), &ItemList::get_item_text_direction);

	ClassDB::bind_method(D_METHOD("set_item_opentype_feature", "idx", "tag", "value"), &ItemList::set_item_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_item_opentype_feature", "idx", "tag"), &ItemList::get_item_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_item_opentype_features", "idx"), &ItemList::clear_item_opentype_features);

	ClassDB::bind_method(D_METHOD("set_item_language", "idx", "language"), &ItemList::set_item_language);
	ClassDB::bind_method(D_METHOD("get_item_language", "idx"), &ItemList::get_item_language);

	ClassDB::bind_method(D_METHOD("set_item_icon_transposed", "idx", "transposed"), &ItemList::set_item_icon_transposed);
	ClassDB::bind_method(D_METHOD("is_item_icon_transposed", "idx"), &ItemList::is_item_icon_transposed);

	ClassDB::bind_method(D_METHOD("set_item_icon_region", "idx", "rect"), &ItemList::set_item_icon_region);
	ClassDB::bind_method(D_METHOD("get_item_icon_region", "idx"), &ItemList::get_item_icon_region);

	ClassDB::bind_method(D_METHOD("set_item_icon_modulate", "idx", "modulate"), &ItemList::set_item_icon_modulate);
	ClassDB::bind_method(D_METHOD("get_item_icon_modulate", "idx"), &ItemList::get_item_icon_modulate);

	ClassDB::bind_method(D_METHOD("set_item_selectable", "idx", "selectable"), &ItemList::set_item_selectable);
	ClassDB::bind_method(D_METHOD("is_item_selectable", "idx"), &ItemList::is_item_selectable);

	ClassDB::bind_method(D_METHOD("set_item_disabled", "idx", "disabled"), &ItemList::set_item_disabled);
	ClassDB::bind_method(D_METHOD("is_item_disabled", "idx"), &ItemList::is_item_disabled);

	ClassDB::bind_method(D_METHOD("set_item_metadata", "idx", "metadata"), &ItemList::set_item_metadata);
	ClassDB::bind_method(D_METHOD("get_item_metadata", "idx"), &ItemList::get_item_metadata);

	ClassDB::bind_method(D_METHOD("set_item_custom_bg_color", "idx", "custom_bg_color"), &ItemList::set_item_custom_bg_color);
	ClassDB::bind_method(D_METHOD("get_item_custom_bg_color", "idx"), &ItemList::get_item_custom_bg_color);

	ClassDB::bind_method(D_METHOD("set_item_custom_fg_color", "idx", "custom_fg_color"), &ItemList::set_item_custom_fg_color);
	ClassDB::bind_method(D_METHOD("get_item_custom_fg_color", "idx"), &ItemList::get_item_custom_fg_color);

	ClassDB::bind_method(D_METHOD("set_item_tooltip_enabled", "idx", "enable"), &ItemList::set_item_tooltip_enabled);
	ClassDB::bind_method(D_METHOD("is_item_tooltip_enabled", "idx"), &ItemList::is_item_tooltip_enabled);

	ClassDB::bind_method(D_METHOD("set_item_tooltip", "idx", "tooltip"), &ItemList::set_item_tooltip);
	ClassDB::bind_method(D_METHOD("get_item_tooltip", "idx"), &ItemList::get_item_tooltip);

	ClassDB::bind_method(D_METHOD("select", "idx", "single"), &ItemList::select, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("deselect", "idx"), &ItemList::deselect);
	ClassDB::bind_method(D_METHOD("deselect_all"), &ItemList::deselect_all);

	ClassDB::bind_method(D_METHOD("is_selected", "idx"), &ItemList::is_selected);
	ClassDB::bind_method(D_METHOD("get_selected_items"), &ItemList::get_selected_items);

	ClassDB::bind_method(D_METHOD("move_item", "from_idx", "to_idx"), &ItemList::move_item);

	ClassDB::bind_method(D_METHOD("set_item_count", "count"), &ItemList::set_item_count);
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

	ClassDB::bind_method(D_METHOD("set_allow_reselect", "allow"), &ItemList::set_allow_reselect);
	ClassDB::bind_method(D_METHOD("get_allow_reselect"), &ItemList::get_allow_reselect);

	ClassDB::bind_method(D_METHOD("set_auto_height", "enable"), &ItemList::set_auto_height);
	ClassDB::bind_method(D_METHOD("has_auto_height"), &ItemList::has_auto_height);

	ClassDB::bind_method(D_METHOD("is_anything_selected"), &ItemList::is_anything_selected);

	ClassDB::bind_method(D_METHOD("get_item_at_position", "position", "exact"), &ItemList::get_item_at_position, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("ensure_current_is_visible"), &ItemList::ensure_current_is_visible);

	ClassDB::bind_method(D_METHOD("get_v_scroll"), &ItemList::get_v_scroll);

	ClassDB::bind_method(D_METHOD("set_text_overrun_behavior", "overrun_behavior"), &ItemList::set_text_overrun_behavior);
	ClassDB::bind_method(D_METHOD("get_text_overrun_behavior"), &ItemList::get_text_overrun_behavior);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "select_mode", PROPERTY_HINT_ENUM, "Single,Multi"), "set_select_mode", "get_select_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_reselect"), "set_allow_reselect", "get_allow_reselect");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_rmb_select"), "set_allow_rmb_select", "get_allow_rmb_select");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_text_lines", PROPERTY_HINT_RANGE, "1,10,1,or_greater"), "set_max_text_lines", "get_max_text_lines");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_height"), "set_auto_height", "has_auto_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_overrun_behavior", PROPERTY_HINT_ENUM, "Trim Nothing,Trim Characters,Trim Words,Ellipsis,Word Ellipsis"), "set_text_overrun_behavior", "get_text_overrun_behavior");
	ADD_ARRAY_COUNT("Items", "item_count", "set_item_count", "get_item_count", "item_");
	ADD_GROUP("Columns", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_columns", PROPERTY_HINT_RANGE, "0,10,1,or_greater"), "set_max_columns", "get_max_columns");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "same_column_width"), "set_same_column_width", "is_same_column_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fixed_column_width", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_fixed_column_width", "get_fixed_column_width");
	ADD_GROUP("Icon", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "icon_mode", PROPERTY_HINT_ENUM, "Top,Left"), "set_icon_mode", "get_icon_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "icon_scale"), "set_icon_scale", "get_icon_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "fixed_icon_size"), "set_fixed_icon_size", "get_fixed_icon_size");

	BIND_ENUM_CONSTANT(ICON_MODE_TOP);
	BIND_ENUM_CONSTANT(ICON_MODE_LEFT);

	BIND_ENUM_CONSTANT(SELECT_SINGLE);
	BIND_ENUM_CONSTANT(SELECT_MULTI);

	ADD_SIGNAL(MethodInfo("item_selected", PropertyInfo(Variant::INT, "index")));
	ADD_SIGNAL(MethodInfo("item_rmb_selected", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::VECTOR2, "at_position")));
	ADD_SIGNAL(MethodInfo("multi_selected", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::BOOL, "selected")));
	ADD_SIGNAL(MethodInfo("item_activated", PropertyInfo(Variant::INT, "index")));
	ADD_SIGNAL(MethodInfo("rmb_clicked", PropertyInfo(Variant::VECTOR2, "at_position")));
	ADD_SIGNAL(MethodInfo("nothing_selected"));

	GLOBAL_DEF("gui/timers/incremental_search_max_interval_msec", 2000);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/timers/incremental_search_max_interval_msec", PropertyInfo(Variant::INT, "gui/timers/incremental_search_max_interval_msec", PROPERTY_HINT_RANGE, "0,10000,1,or_greater")); // No negative numbers
}

ItemList::ItemList() {
	scroll_bar = memnew(VScrollBar);
	add_child(scroll_bar, false, INTERNAL_MODE_FRONT);

	scroll_bar->connect("value_changed", callable_mp(this, &ItemList::_scroll_changed));

	set_focus_mode(FOCUS_ALL);
	set_clip_contents(true);
}

ItemList::~ItemList() {
}
