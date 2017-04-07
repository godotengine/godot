/*************************************************************************/
/*  tabs.cpp                                                             */
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

void Tabs::_gui_input(const InputEvent &p_event) {

	if (p_event.type == InputEvent::MOUSE_MOTION) {

		Point2 pos(p_event.mouse_motion.x, p_event.mouse_motion.y);

		hilite_arrow = -1;
		if (buttons_visible) {

			Ref<Texture> incr = get_icon("increment");
			Ref<Texture> decr = get_icon("decrement");

			int limit = get_size().width - incr->get_width() - decr->get_width();

			if (pos.x > limit + decr->get_width()) {
				hilite_arrow = 1;
			} else if (pos.x > limit) {
				hilite_arrow = 0;
			}
		}

		// test hovering to display right or close button
		int hover_buttons = -1;
		hover = -1;
		for (int i = 0; i < tabs.size(); i++) {

			if (i < offset)
				continue;

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

		if (hover_buttons == -1) { // no hover
			rb_hover = hover_buttons;
			cb_hover = hover_buttons;
		}
		update();

		return;
	}

	if (rb_pressing && p_event.type == InputEvent::MOUSE_BUTTON &&
			!p_event.mouse_button.pressed &&
			p_event.mouse_button.button_index == BUTTON_LEFT) {

		if (rb_hover != -1) {
			//pressed
			emit_signal("right_button_pressed", rb_hover);
		}

		rb_pressing = false;
		update();
	}

	if (cb_pressing && p_event.type == InputEvent::MOUSE_BUTTON &&
			!p_event.mouse_button.pressed &&
			p_event.mouse_button.button_index == BUTTON_LEFT) {

		if (cb_hover != -1) {
			//pressed
			emit_signal("tab_close", cb_hover);
		}

		cb_pressing = false;
		update();
	}

	if (p_event.type == InputEvent::MOUSE_BUTTON &&
			p_event.mouse_button.pressed &&
			p_event.mouse_button.button_index == BUTTON_LEFT) {

		// clicks
		Point2 pos(p_event.mouse_button.x, p_event.mouse_button.y);

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
			emit_signal("tab_changed", found);
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

			_ensure_no_over_offset();
		} break;
		case NOTIFICATION_DRAW: {

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
			Ref<Texture> incr_hl = get_icon("increment_hilite");
			Ref<Texture> decr_hl = get_icon("decrement_hilite");

			int limit = get_size().width - incr->get_size().width - decr->get_size().width;

			missing_right = false;

			for (int i = 0; i < tabs.size(); i++) {

				if (i < offset)
					continue;

				tabs[i].ofs_cache = w;

				int lsize = get_tab_width(i);

				String text = tabs[i].text;
				int slen = font->get_string_size(text).width;

				if (w + lsize > limit) {
					max_drawn_tab = i - 1;
					missing_right = true;
					break;
				} else {
					max_drawn_tab = i;
				}

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

				Rect2 sb_rect = Rect2(w, 0, lsize, h);
				sb->draw(ci, sb_rect);

				w += sb->get_margin(MARGIN_LEFT);

				Size2i sb_ms = sb->get_minimum_size();
				Ref<Texture> icon = tabs[i].icon;
				if (icon.is_valid()) {

					icon->draw(ci, Point2i(w, sb->get_margin(MARGIN_TOP) + ((sb_rect.size.y - sb_ms.y) - icon->get_height()) / 2));
					if (text != "")
						w += icon->get_width() + get_constant("hseparation");
				}

				font->draw(ci, Point2i(w, sb->get_margin(MARGIN_TOP) + ((sb_rect.size.y - sb_ms.y) - font->get_height()) / 2 + font->get_ascent()), text, col);

				w += slen;

				if (tabs[i].right_button.is_valid()) {

					Ref<StyleBox> style = get_stylebox("button");
					Ref<Texture> rb = tabs[i].right_button;

					w += get_constant("hseparation");

					Rect2 rb_rect;
					rb_rect.size = style->get_minimum_size() + rb->get_size();
					rb_rect.pos.x = w;
					rb_rect.pos.y = sb->get_margin(MARGIN_TOP) + ((sb_rect.size.y - sb_ms.y) - (rb_rect.size.y)) / 2;

					if (rb_hover == i) {
						if (rb_pressing)
							get_stylebox("button_pressed")->draw(ci, rb_rect);
						else
							style->draw(ci, rb_rect);
					}

					rb->draw(ci, Point2i(w + style->get_margin(MARGIN_LEFT), rb_rect.pos.y + style->get_margin(MARGIN_TOP)));
					w += rb->get_width();
					tabs[i].rb_rect = rb_rect;
				}

				if (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && i == current)) {

					Ref<StyleBox> style = get_stylebox("button");
					Ref<Texture> cb = close;

					w += get_constant("hseparation");

					Rect2 cb_rect;
					cb_rect.size = style->get_minimum_size() + cb->get_size();
					cb_rect.pos.x = w;
					cb_rect.pos.y = sb->get_margin(MARGIN_TOP) + ((sb_rect.size.y - sb_ms.y) - (cb_rect.size.y)) / 2;

					if (!tabs[i].disabled && cb_hover == i) {
						if (cb_pressing)
							get_stylebox("button_pressed")->draw(ci, cb_rect);
						else
							style->draw(ci, cb_rect);
					}

					cb->draw(ci, Point2i(w + style->get_margin(MARGIN_LEFT), cb_rect.pos.y + style->get_margin(MARGIN_TOP)));
					w += cb->get_width();
					tabs[i].cb_rect = cb_rect;
				}

				w += sb->get_margin(MARGIN_RIGHT);

				tabs[i].size_cache = w - tabs[i].ofs_cache;
			}

			if (offset > 0 || missing_right) {

				int vofs = (get_size().height - incr->get_size().height) / 2;

				if (offset > 0)
					draw_texture(hilite_arrow == 0 ? decr_hl : decr, Point2(limit, vofs));
				else
					draw_texture(decr, Point2(limit, vofs), Color(1, 1, 1, 0.5));

				if (missing_right)
					draw_texture(hilite_arrow == 1 ? incr_hl : incr, Point2(limit + decr->get_size().width, vofs));
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

	ERR_FAIL_INDEX(p_current, get_tab_count());

	current = p_current;

	_change_notify("current_tab");
	update();
}

int Tabs::get_current_tab() const {

	return current;
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
	update();
	minimum_size_changed();
}
Ref<Texture> Tabs::get_tab_right_button(int p_tab) const {

	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Ref<Texture>());
	return tabs[p_tab].right_button;
}

void Tabs::add_tab(const String &p_str, const Ref<Texture> &p_icon) {

	Tab t;
	t.text = p_str;
	t.icon = p_icon;
	t.disabled = false;

	tabs.push_back(t);

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
	update();
	minimum_size_changed();

	if (current < 0)
		current = 0;
	if (current >= tabs.size())
		current = tabs.size() - 1;

	_ensure_no_over_offset();
}

void Tabs::set_tab_align(TabAlign p_align) {

	tab_align = p_align;
	update();
}

Tabs::TabAlign Tabs::get_tab_align() const {

	return tab_align;
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

			total_w += get_tab_width(i);
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

	ERR_FAIL_INDEX(p_idx, tabs.size());

	_ensure_no_over_offset();

	if (p_idx <= offset) {
		offset = p_idx;
		update();
		return;
	}

	Ref<Texture> incr = get_icon("increment");
	Ref<Texture> decr = get_icon("decrement");
	int limit = get_size().width - incr->get_width() - decr->get_width();

	int x = 0;
	for (int i = 0; i < tabs.size(); i++) {

		if (i < offset)
			continue;

		int sz = get_tab_width(i);
		tabs[i].x_cache = x;
		tabs[i].x_size_cache = sz;
		x += sz;
	}

	while (offset < tabs.size() && ((tabs[p_idx].x_cache + tabs[p_idx].x_size_cache) - tabs[offset].x_cache) > limit) {
		offset++;
	}

	update();
}

void Tabs::set_tab_close_display_policy(CloseButtonDisplayPolicy p_policy) {
	cb_displaypolicy = p_policy;
	update();
}

void Tabs::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &Tabs::_gui_input);
	ClassDB::bind_method(D_METHOD("get_tab_count"), &Tabs::get_tab_count);
	ClassDB::bind_method(D_METHOD("set_current_tab", "tab_idx"), &Tabs::set_current_tab);
	ClassDB::bind_method(D_METHOD("get_current_tab"), &Tabs::get_current_tab);
	ClassDB::bind_method(D_METHOD("set_tab_title", "tab_idx", "title"), &Tabs::set_tab_title);
	ClassDB::bind_method(D_METHOD("get_tab_title", "tab_idx"), &Tabs::get_tab_title);
	ClassDB::bind_method(D_METHOD("set_tab_icon", "tab_idx", "icon:Texture"), &Tabs::set_tab_icon);
	ClassDB::bind_method(D_METHOD("get_tab_icon:Texture", "tab_idx"), &Tabs::get_tab_icon);
	ClassDB::bind_method(D_METHOD("set_tab_disabled", "tab_idx", "disabled"), &Tabs::set_tab_disabled);
	ClassDB::bind_method(D_METHOD("get_tab_disabled", "tab_idx"), &Tabs::get_tab_disabled);
	ClassDB::bind_method(D_METHOD("remove_tab", "tab_idx"), &Tabs::remove_tab);
	ClassDB::bind_method(D_METHOD("add_tab", "title", "icon:Texture"), &Tabs::add_tab);
	ClassDB::bind_method(D_METHOD("set_tab_align", "align"), &Tabs::set_tab_align);
	ClassDB::bind_method(D_METHOD("get_tab_align"), &Tabs::get_tab_align);
	ClassDB::bind_method(D_METHOD("ensure_tab_visible", "idx"), &Tabs::ensure_tab_visible);

	ADD_SIGNAL(MethodInfo("tab_changed", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("right_button_pressed", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_close", PropertyInfo(Variant::INT, "tab")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_tab", PROPERTY_HINT_RANGE, "-1,4096,1", PROPERTY_USAGE_EDITOR), "set_current_tab", "get_current_tab");

	BIND_CONSTANT(ALIGN_LEFT);
	BIND_CONSTANT(ALIGN_CENTER);
	BIND_CONSTANT(ALIGN_RIGHT);

	BIND_CONSTANT(CLOSE_BUTTON_SHOW_ACTIVE_ONLY);
	BIND_CONSTANT(CLOSE_BUTTON_SHOW_ALWAYS);
	BIND_CONSTANT(CLOSE_BUTTON_SHOW_NEVER);
}

Tabs::Tabs() {

	current = 0;
	tab_align = ALIGN_CENTER;
	rb_hover = -1;
	rb_pressing = false;
	hilite_arrow = -1;

	cb_hover = -1;
	cb_pressing = false;
	cb_displaypolicy = CLOSE_BUTTON_SHOW_NEVER;
	offset = 0;
	max_drawn_tab = 0;
}
