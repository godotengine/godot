/*************************************************************************/
/*  tabs.cpp                                                             */
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
#include "tabs.h"

#include "message_queue.h"

Size2 Tabs::get_minimum_size() const {

	Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
	Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
	Ref<StyleBox> tab_disabled = get_stylebox("tab_disabled");
	Ref<Font> font = get_font("font");

	Size2 ms(0, MAX(MAX(tab_bg->get_minimum_size().height, tab_fg->get_minimum_size().height), tab_disabled->get_minimum_size().height) + font->get_height());

	for (int i = 0; i < tabs.size(); i++) {

		Ref<Texture> tex = tabs[i].icon;
		if (tex.is_valid()) {
			ms.height = MAX(ms.height, tex->get_size().height);
			if (tabs[i].text != "")
				ms.width += get_constant("hseparation");
		}

		ms.width += font->get_string_size(tabs[i].text).width;

		if (tabs[i].disabled)
			ms.width += tab_disabled->get_minimum_size().width;
		else if (current == i)
			ms.width += tab_fg->get_minimum_size().width;
		else
			ms.width += tab_bg->get_minimum_size().width;

		if (tabs[i].right_button.is_valid()) {
			Ref<Texture> rb = tabs[i].right_button;
			Size2 bms = rb->get_size();
			bms.width += get_constant("hseparation");
			ms.width += bms.width;
			ms.height = MAX(bms.height + tab_bg->get_minimum_size().height, ms.height);
		}

		if (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && i == current)) {
			Ref<Texture> cb = get_icon("close");
			Size2 bms = cb->get_size();
			bms.width += get_constant("hseparation");
			ms.width += bms.width;
			ms.height = MAX(bms.height + tab_bg->get_minimum_size().height, ms.height);
		}
	}

	ms.width = 0; //TODO: should make this optional
	return ms;
}

void Tabs::_gui_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {

		Point2 pos = mm->get_position();

		highlight_arrow = -1;
		if (buttons_visible) {

			Ref<Texture> incr = get_icon("increment");
			Ref<Texture> decr = get_icon("decrement");

			int limit = get_size().width - incr->get_width() - decr->get_width();

			if (pos.x > limit + decr->get_width()) {
				highlight_arrow = 1;
			} else if (pos.x > limit) {
				highlight_arrow = 0;
			}
		}

		// test hovering to display right or close button
		int hover_now = -1;
		int hover_buttons = -1;
		for (int i = 0; i < tabs.size(); i++) {

			if (i < offset)
				continue;

			Rect2 rect = get_tab_rect(i);
			if (rect.has_point(pos)) {
				hover_now = i;
			}
			if (tabs[i].rb_rect.has_point(pos)) {
				rb_hover = i;
				cb_hover = -1;
				hover_buttons = i;
				break;
			} else if (!tabs[i].disabled && tabs[i].cb_rect.has_point(pos)) {
				cb_hover = i;
				rb_hover = -1;
				hover_buttons = i;
				break;
			}
		}
		if (hover != hover_now) {
			hover = hover_now;
			emit_signal("tab_hover", hover);
		}

		if (hover_buttons == -1) { // no hover
			rb_hover = hover_buttons;
			cb_hover = hover_buttons;
		}
		update();

		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {

		if (mb->is_pressed() && mb->get_button_index() == BUTTON_WHEEL_UP && !mb->get_command()) {

			if (scrolling_enabled && buttons_visible) {
				if (offset > 0) {
					offset--;
					update();
				}
			}
		}

		if (mb->is_pressed() && mb->get_button_index() == BUTTON_WHEEL_DOWN && !mb->get_command()) {
			if (scrolling_enabled && buttons_visible) {
				if (missing_right) {
					offset++;
					update();
				}
			}
		}

		if (rb_pressing && !mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {

			if (rb_hover != -1) {
				//pressed
				emit_signal("right_button_pressed", rb_hover);
			}

			rb_pressing = false;
			update();
		}

		if (cb_pressing && !mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {

			if (cb_hover != -1) {
				//pressed
				emit_signal("tab_close", cb_hover);
			}

			cb_pressing = false;
			update();
		}

		if (mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {

			// clicks
			Point2 pos(mb->get_position().x, mb->get_position().y);

			if (buttons_visible) {

				Ref<Texture> incr = get_icon("increment");
				Ref<Texture> decr = get_icon("decrement");

				int limit = get_size().width - incr->get_width() - decr->get_width();

				if (pos.x > limit + decr->get_width()) {
					if (missing_right) {
						offset++;
						update();
					}
					return;
				} else if (pos.x > limit) {
					if (offset > 0) {
						offset--;
						update();
					}
					return;
				}
			}

			int found = -1;
			for (int i = 0; i < tabs.size(); i++) {

				if (i < offset)
					continue;

				if (tabs[i].rb_rect.has_point(pos)) {
					rb_pressing = true;
					update();
					return;
				}

				if (tabs[i].cb_rect.has_point(pos)) {
					cb_pressing = true;
					update();
					return;
				}

				if (pos.x >= tabs[i].ofs_cache && pos.x < tabs[i].ofs_cache + tabs[i].size_cache) {
					if (!tabs[i].disabled) {
						found = i;
					}
					break;
				}
			}

			if (found != -1) {

				set_current_tab(found);
				emit_signal("tab_clicked", found);
			}
		}
	}
}

void Tabs::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_MOUSE_EXIT: {
			rb_hover = -1;
			cb_hover = -1;
			hover = -1;
			update();
		} break;
		case NOTIFICATION_RESIZED: {
			_update_cache();
			_ensure_no_over_offset();
			ensure_tab_visible(current);

		} break;
		case NOTIFICATION_DRAW: {
			_update_cache();
			RID ci = get_canvas_item();

			Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
			Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
			Ref<StyleBox> tab_disabled = get_stylebox("tab_disabled");
			Ref<Font> font = get_font("font");
			Color color_fg = get_color("font_color_fg");
			Color color_bg = get_color("font_color_bg");
			Color color_disabled = get_color("font_color_disabled");
			Ref<Texture> close = get_icon("close");

			int h = get_size().height;
			int w = 0;
			int mw = 0;

			for (int i = 0; i < tabs.size(); i++) {

				tabs[i].ofs_cache = mw;
				mw += get_tab_width(i);
			}

			if (tab_align == ALIGN_CENTER) {
				w = (get_size().width - mw) / 2;
			} else if (tab_align == ALIGN_RIGHT) {
				w = get_size().width - mw;
			}

			if (w < 0) {
				w = 0;
			}

			Ref<Texture> incr = get_icon("increment");
			Ref<Texture> decr = get_icon("decrement");
			Ref<Texture> incr_hl = get_icon("increment_highlight");
			Ref<Texture> decr_hl = get_icon("decrement_highlight");

			int limit = get_size().width - incr->get_size().width - decr->get_size().width;

			missing_right = false;

			for (int i = 0; i < tabs.size(); i++) {

				if (i < offset)
					continue;

				tabs[i].ofs_cache = w;

				int lsize = tabs[i].size_cache;

				Ref<StyleBox> sb;
				Color col;

				if (tabs[i].disabled) {
					sb = tab_disabled;
					col = color_disabled;
				} else if (i == current) {
					sb = tab_fg;
					col = color_fg;
				} else {
					sb = tab_bg;
					col = color_bg;
				}

				if (w + lsize > limit) {
					max_drawn_tab = i - 1;
					missing_right = true;
					break;
				} else {
					max_drawn_tab = i;
				}

				Rect2 sb_rect = Rect2(w, 0, tabs[i].size_cache, h);
				sb->draw(ci, sb_rect);

				w += sb->get_margin(MARGIN_LEFT);

				Size2i sb_ms = sb->get_minimum_size();
				Ref<Texture> icon = tabs[i].icon;
				if (icon.is_valid()) {

					icon->draw(ci, Point2i(w, sb->get_margin(MARGIN_TOP) + ((sb_rect.size.y - sb_ms.y) - icon->get_height()) / 2));
					if (tabs[i].text != "")
						w += icon->get_width() + get_constant("hseparation");
				}

				font->draw(ci, Point2i(w, sb->get_margin(MARGIN_TOP) + ((sb_rect.size.y - sb_ms.y) - font->get_height()) / 2 + font->get_ascent()), tabs[i].text, col, tabs[i].size_text);

				w += tabs[i].size_text;

				if (tabs[i].right_button.is_valid()) {

					Ref<StyleBox> style = get_stylebox("button");
					Ref<Texture> rb = tabs[i].right_button;

					w += get_constant("hseparation");

					Rect2 rb_rect;
					rb_rect.size = style->get_minimum_size() + rb->get_size();
					rb_rect.position.x = w;
					rb_rect.position.y = sb->get_margin(MARGIN_TOP) + ((sb_rect.size.y - sb_ms.y) - (rb_rect.size.y)) / 2;

					if (rb_hover == i) {
						if (rb_pressing)
							get_stylebox("button_pressed")->draw(ci, rb_rect);
						else
							style->draw(ci, rb_rect);
					}

					rb->draw(ci, Point2i(w + style->get_margin(MARGIN_LEFT), rb_rect.position.y + style->get_margin(MARGIN_TOP)));
					w += rb->get_width();
					tabs[i].rb_rect = rb_rect;
				}

				if (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && i == current)) {

					Ref<StyleBox> style = get_stylebox("button");
					Ref<Texture> cb = close;

					w += get_constant("hseparation");

					Rect2 cb_rect;
					cb_rect.size = style->get_minimum_size() + cb->get_size();
					cb_rect.position.x = w;
					cb_rect.position.y = sb->get_margin(MARGIN_TOP) + ((sb_rect.size.y - sb_ms.y) - (cb_rect.size.y)) / 2;

					if (!tabs[i].disabled && cb_hover == i) {
						if (cb_pressing)
							get_stylebox("button_pressed")->draw(ci, cb_rect);
						else
							style->draw(ci, cb_rect);
					}

					cb->draw(ci, Point2i(w + style->get_margin(MARGIN_LEFT), cb_rect.position.y + style->get_margin(MARGIN_TOP)));
					w += cb->get_width();
					tabs[i].cb_rect = cb_rect;
				}

				w += sb->get_margin(MARGIN_RIGHT);
			}

			if (offset > 0 || missing_right) {

				int vofs = (get_size().height - incr->get_size().height) / 2;

				if (offset > 0)
					draw_texture(highlight_arrow == 0 ? decr_hl : decr, Point2(limit, vofs));
				else
					draw_texture(decr, Point2(limit, vofs), Color(1, 1, 1, 0.5));

				if (missing_right)
					draw_texture(highlight_arrow == 1 ? incr_hl : incr, Point2(limit + decr->get_size().width, vofs));
				else
					draw_texture(incr, Point2(limit + decr->get_size().width, vofs), Color(1, 1, 1, 0.5));

				buttons_visible = true;
			} else {
				buttons_visible = false;
			}

		} break;
	}
}

int Tabs::get_tab_count() const {

	return tabs.size();
}

void Tabs::set_current_tab(int p_current) {

	if (current == p_current) return;
	ERR_FAIL_INDEX(p_current, get_tab_count());

	current = p_current;

	_change_notify("current_tab");
	_update_cache();
	update();

	emit_signal("tab_changed", p_current);
}

int Tabs::get_current_tab() const {

	return current;
}

int Tabs::get_hovered_tab() const {
	return hover;
}

int Tabs::get_tab_offset() const {
	return offset;
}

bool Tabs::get_offset_buttons_visible() const {
	return buttons_visible;
}

void Tabs::set_tab_title(int p_tab, const String &p_title) {

	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs[p_tab].text = p_title;
	update();
	minimum_size_changed();
}

String Tabs::get_tab_title(int p_tab) const {

	ERR_FAIL_INDEX_V(p_tab, tabs.size(), "");
	return tabs[p_tab].text;
}

void Tabs::set_tab_icon(int p_tab, const Ref<Texture> &p_icon) {

	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs[p_tab].icon = p_icon;
	update();
	minimum_size_changed();
}

Ref<Texture> Tabs::get_tab_icon(int p_tab) const {

	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Ref<Texture>());
	return tabs[p_tab].icon;
}

void Tabs::set_tab_disabled(int p_tab, bool p_disabled) {

	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs[p_tab].disabled = p_disabled;
	update();
}
bool Tabs::get_tab_disabled(int p_tab) const {

	ERR_FAIL_INDEX_V(p_tab, tabs.size(), false);
	return tabs[p_tab].disabled;
}

void Tabs::set_tab_right_button(int p_tab, const Ref<Texture> &p_right_button) {

	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs[p_tab].right_button = p_right_button;
	_update_cache();
	update();
	minimum_size_changed();
}
Ref<Texture> Tabs::get_tab_right_button(int p_tab) const {

	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Ref<Texture>());
	return tabs[p_tab].right_button;
}

void Tabs::_update_cache() {
	Ref<StyleBox> tab_disabled = get_stylebox("tab_disabled");
	Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
	Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
	Ref<Font> font = get_font("font");
	Ref<Texture> incr = get_icon("increment");
	Ref<Texture> decr = get_icon("decrement");
	int limit = get_size().width - incr->get_width() - decr->get_width();

	int w = 0;
	int mw = 0;
	int size_fixed = 0;
	int count_resize = 0;
	for (int i = 0; i < tabs.size(); i++) {
		tabs[i].ofs_cache = mw;
		tabs[i].size_cache = get_tab_width(i);
		tabs[i].size_text = font->get_string_size(tabs[i].text).width;
		mw += tabs[i].size_cache;
		if (tabs[i].size_cache <= min_width || i == current) {
			size_fixed += tabs[i].size_cache;
		} else {
			count_resize++;
		}
	}
	int m_width = min_width;
	if (count_resize > 0) {
		m_width = MAX((limit - size_fixed) / count_resize, min_width);
	}
	for (int i = 0; i < tabs.size(); i++) {
		if (i < offset)
			continue;
		Ref<StyleBox> sb;
		if (tabs[i].disabled) {
			sb = tab_disabled;
		} else if (i == current) {
			sb = tab_fg;
		} else {
			sb = tab_bg;
		}
		int lsize = tabs[i].size_cache;
		int slen = tabs[i].size_text;
		if (min_width > 0 && mw > limit && i != current) {
			if (lsize > m_width) {
				slen = m_width - (sb->get_margin(MARGIN_LEFT) + sb->get_margin(MARGIN_RIGHT));
				if (tabs[i].icon.is_valid()) {
					slen -= tabs[i].icon->get_width();
					slen -= get_constant("hseparation");
				}
				if (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && i == current)) {
					Ref<Texture> cb = get_icon("close");
					slen -= cb->get_width();
					slen -= get_constant("hseparation");
				}
				slen = MAX(slen, 1);
				lsize = m_width;
			}
		}
		tabs[i].ofs_cache = w;
		tabs[i].size_cache = lsize;
		tabs[i].size_text = slen;
		w += lsize;
	}
}

void Tabs::add_tab(const String &p_str, const Ref<Texture> &p_icon) {

	Tab t;
	t.text = p_str;
	t.icon = p_icon;
	t.disabled = false;
	t.ofs_cache = 0;
	t.size_cache = 0;

	tabs.push_back(t);
	_update_cache();
	update();
	minimum_size_changed();
}

void Tabs::clear_tabs() {
	tabs.clear();
	current = 0;
	update();
}

void Tabs::remove_tab(int p_idx) {

	ERR_FAIL_INDEX(p_idx, tabs.size());
	tabs.remove(p_idx);
	if (current >= p_idx)
		current--;
	_update_cache();
	update();
	minimum_size_changed();

	if (current < 0)
		current = 0;
	if (current >= tabs.size())
		current = tabs.size() - 1;

	_ensure_no_over_offset();
}

Variant Tabs::get_drag_data(const Point2 &p_point) {

	return get_tab_idx_at_point(p_point);
}

bool Tabs::can_drop_data(const Point2 &p_point, const Variant &p_data) const {

	return get_tab_idx_at_point(p_point) > -1;
}

void Tabs::drop_data(const Point2 &p_point, const Variant &p_data) {

	int hover_now = get_tab_idx_at_point(p_point);

	ERR_FAIL_INDEX(hover_now, tabs.size());
	emit_signal("reposition_active_tab_request", hover_now);
}

int Tabs::get_tab_idx_at_point(const Point2 &p_point) const {

	int hover_now = -1;
	for (int i = 0; i < tabs.size(); i++) {

		if (i < offset)
			continue;

		Rect2 rect = get_tab_rect(i);
		if (rect.has_point(p_point)) {
			hover_now = i;
		}
	}

	return hover_now;
}

void Tabs::set_tab_align(TabAlign p_align) {

	ERR_FAIL_INDEX(p_align, ALIGN_MAX);
	tab_align = p_align;
	update();
}

Tabs::TabAlign Tabs::get_tab_align() const {

	return tab_align;
}

void Tabs::move_tab(int from, int to) {

	if (from == to)
		return;

	ERR_FAIL_INDEX(from, tabs.size());
	ERR_FAIL_INDEX(to, tabs.size());

	Tab tab_from = tabs[from];
	tabs.remove(from);
	tabs.insert(to, tab_from);

	_update_cache();
	update();
}

int Tabs::get_tab_width(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, tabs.size(), 0);

	Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
	Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
	Ref<StyleBox> tab_disabled = get_stylebox("tab_disabled");
	Ref<Font> font = get_font("font");

	int x = 0;

	Ref<Texture> tex = tabs[p_idx].icon;
	if (tex.is_valid()) {
		x += tex->get_width();
		if (tabs[p_idx].text != "")
			x += get_constant("hseparation");
	}

	x += font->get_string_size(tabs[p_idx].text).width;

	if (tabs[p_idx].disabled)
		x += tab_disabled->get_minimum_size().width;
	else if (current == p_idx)
		x += tab_fg->get_minimum_size().width;
	else
		x += tab_bg->get_minimum_size().width;

	if (tabs[p_idx].right_button.is_valid()) {
		Ref<Texture> rb = tabs[p_idx].right_button;
		x += rb->get_width();
		x += get_constant("hseparation");
	}

	if (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && p_idx == current)) {
		Ref<Texture> cb = get_icon("close");
		x += cb->get_width();
		x += get_constant("hseparation");
	}

	return x;
}

void Tabs::_ensure_no_over_offset() {

	if (!is_inside_tree())
		return;

	Ref<Texture> incr = get_icon("increment");
	Ref<Texture> decr = get_icon("decrement");

	int limit = get_size().width - incr->get_width() - decr->get_width();

	while (offset > 0) {

		int total_w = 0;
		for (int i = 0; i < tabs.size(); i++) {

			if (i < offset - 1)
				continue;

			total_w += tabs[i].size_cache;
		}

		if (total_w < limit) {
			offset--;
			update();
		} else {
			break;
		}
	}
}

void Tabs::ensure_tab_visible(int p_idx) {

	if (!is_inside_tree())
		return;

	if (tabs.size() == 0) return;
	ERR_FAIL_INDEX(p_idx, tabs.size());

	if (p_idx == offset) {
		return;
	}
	if (p_idx < offset) {
		offset = p_idx;
		update();
		return;
	}

	int prev_offset = offset;
	Ref<Texture> incr = get_icon("increment");
	Ref<Texture> decr = get_icon("decrement");
	int limit = get_size().width - incr->get_width() - decr->get_width();
	for (int i = offset; i <= p_idx; i++) {
		if (tabs[i].ofs_cache + tabs[i].size_cache > limit) {
			offset++;
		}
	}

	if (prev_offset != offset) {
		update();
	}
}

Rect2 Tabs::get_tab_rect(int p_tab) const {
	return Rect2(tabs[p_tab].ofs_cache, 0, tabs[p_tab].size_cache, get_size().height);
}

void Tabs::set_tab_close_display_policy(CloseButtonDisplayPolicy p_policy) {

	ERR_FAIL_INDEX(p_policy, CLOSE_BUTTON_MAX);
	cb_displaypolicy = p_policy;
	update();
}

Tabs::CloseButtonDisplayPolicy Tabs::get_tab_close_display_policy() const {

	return cb_displaypolicy;
}

void Tabs::set_min_width(int p_width) {
	min_width = p_width;
}

void Tabs::set_scrolling_enabled(bool p_enabled) {
	scrolling_enabled = p_enabled;
}

bool Tabs::get_scrolling_enabled() const {
	return scrolling_enabled;
}

void Tabs::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &Tabs::_gui_input);
	ClassDB::bind_method(D_METHOD("get_tab_count"), &Tabs::get_tab_count);
	ClassDB::bind_method(D_METHOD("set_current_tab", "tab_idx"), &Tabs::set_current_tab);
	ClassDB::bind_method(D_METHOD("get_current_tab"), &Tabs::get_current_tab);
	ClassDB::bind_method(D_METHOD("set_tab_title", "tab_idx", "title"), &Tabs::set_tab_title);
	ClassDB::bind_method(D_METHOD("get_tab_title", "tab_idx"), &Tabs::get_tab_title);
	ClassDB::bind_method(D_METHOD("set_tab_icon", "tab_idx", "icon"), &Tabs::set_tab_icon);
	ClassDB::bind_method(D_METHOD("get_tab_icon", "tab_idx"), &Tabs::get_tab_icon);
	ClassDB::bind_method(D_METHOD("set_tab_disabled", "tab_idx", "disabled"), &Tabs::set_tab_disabled);
	ClassDB::bind_method(D_METHOD("get_tab_disabled", "tab_idx"), &Tabs::get_tab_disabled);
	ClassDB::bind_method(D_METHOD("remove_tab", "tab_idx"), &Tabs::remove_tab);
	ClassDB::bind_method(D_METHOD("add_tab", "title", "icon"), &Tabs::add_tab, DEFVAL(""), DEFVAL(Ref<Texture>()));
	ClassDB::bind_method(D_METHOD("set_tab_align", "align"), &Tabs::set_tab_align);
	ClassDB::bind_method(D_METHOD("get_tab_align"), &Tabs::get_tab_align);
	ClassDB::bind_method(D_METHOD("get_tab_offset"), &Tabs::get_tab_offset);
	ClassDB::bind_method(D_METHOD("get_offset_buttons_visible"), &Tabs::get_offset_buttons_visible);
	ClassDB::bind_method(D_METHOD("ensure_tab_visible", "idx"), &Tabs::ensure_tab_visible);
	ClassDB::bind_method(D_METHOD("get_tab_rect", "tab_idx"), &Tabs::get_tab_rect);
	ClassDB::bind_method(D_METHOD("move_tab", "from", "to"), &Tabs::move_tab);
	ClassDB::bind_method(D_METHOD("set_tab_close_display_policy", "policy"), &Tabs::set_tab_close_display_policy);
	ClassDB::bind_method(D_METHOD("get_tab_close_display_policy"), &Tabs::get_tab_close_display_policy);
	ClassDB::bind_method(D_METHOD("set_scrolling_enabled", "enabled"), &Tabs::set_scrolling_enabled);
	ClassDB::bind_method(D_METHOD("get_scrolling_enabled"), &Tabs::get_scrolling_enabled);

	ADD_SIGNAL(MethodInfo("tab_changed", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("right_button_pressed", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_close", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_hover", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("reposition_active_tab_request", PropertyInfo(Variant::INT, "idx_to")));
	ADD_SIGNAL(MethodInfo("tab_clicked", PropertyInfo(Variant::INT, "tab")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_tab", PROPERTY_HINT_RANGE, "-1,4096,1", PROPERTY_USAGE_EDITOR), "set_current_tab", "get_current_tab");
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "tab_close_display_policy", PROPERTY_HINT_ENUM, "Show Never,Show Active Only,Show Always"), "set_tab_close_display_policy", "get_tab_close_display_policy");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scrolling_enabled"), "set_scrolling_enabled", "get_scrolling_enabled");

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);
	BIND_ENUM_CONSTANT(ALIGN_MAX);

	BIND_ENUM_CONSTANT(CLOSE_BUTTON_SHOW_NEVER);
	BIND_ENUM_CONSTANT(CLOSE_BUTTON_SHOW_ACTIVE_ONLY);
	BIND_ENUM_CONSTANT(CLOSE_BUTTON_SHOW_ALWAYS);
	BIND_ENUM_CONSTANT(CLOSE_BUTTON_MAX);
}

Tabs::Tabs() {

	current = 0;
	tab_align = ALIGN_CENTER;
	rb_hover = -1;
	rb_pressing = false;
	highlight_arrow = -1;

	cb_hover = -1;
	cb_pressing = false;
	cb_displaypolicy = CLOSE_BUTTON_SHOW_NEVER;
	offset = 0;
	max_drawn_tab = 0;

	min_width = 0;
	scrolling_enabled = true;
}
