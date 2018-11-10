/*************************************************************************/
/*  tab_container.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "tab_container.h"

#include "core/message_queue.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"

int TabContainer::_get_top_margin() const {

	if (!tabs_visible)
		return 0;

	// Respect the minimum tab height.
	Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
	Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
	Ref<StyleBox> tab_disabled = get_stylebox("tab_disabled");

	int tab_height = MAX(MAX(tab_bg->get_minimum_size().height, tab_fg->get_minimum_size().height), tab_disabled->get_minimum_size().height);

	// Font height or higher icon wins.
	Ref<Font> font = get_font("font");
	int content_height = font->get_height();

	Vector<Control *> tabs = _get_tabs();
	for (int i = 0; i < tabs.size(); i++) {

		Control *c = tabs[i];
		if (!c->has_meta("_tab_icon"))
			continue;

		Ref<Texture> tex = c->get_meta("_tab_icon");
		if (!tex.is_valid())
			continue;
		content_height = MAX(content_height, tex->get_size().height);
	}

	return tab_height + content_height;
}

void TabContainer::_gui_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {

		Point2 pos(mb->get_position().x, mb->get_position().y);
		Size2 size = get_size();

		// Click must be on tabs in the tab header area.
		if (pos.x < tabs_ofs_cache || pos.y > _get_top_margin())
			return;

		// Handle menu button.
		Ref<Texture> menu = get_icon("menu");
		if (popup && pos.x > size.width - menu->get_width()) {
			emit_signal("pre_popup_pressed");

			Vector2 popup_pos = get_global_position();
			popup_pos.x += size.width - popup->get_size().width;
			popup_pos.y += menu->get_height();

			popup->set_global_position(popup_pos);
			popup->popup();
			return;
		}

		// Do not activate tabs when tabs is empty
		if (get_tab_count() == 0)
			return;

		Vector<Control *> tabs = _get_tabs();

		// Handle navigation buttons.
		if (buttons_visible_cache) {
			int popup_ofs = 0;
			if (popup) {
				popup_ofs = menu->get_width();
			}

			Ref<Texture> increment = get_icon("increment");
			Ref<Texture> decrement = get_icon("decrement");
			if (pos.x > size.width - increment->get_width() - popup_ofs) {
				if (last_tab_cache < tabs.size() - 1) {
					first_tab_cache += 1;
					update();
				}
				return;
			} else if (pos.x > size.width - increment->get_width() - decrement->get_width() - popup_ofs) {
				if (first_tab_cache > 0) {
					first_tab_cache -= 1;
					update();
				}
				return;
			}
		}

		// Activate the clicked tab.
		pos.x -= tabs_ofs_cache;
		for (int i = first_tab_cache; i <= last_tab_cache; i++) {
			int tab_width = _get_tab_width(i);
			if (pos.x < tab_width) {
				if (!get_tab_disabled(i)) {
					set_current_tab(i);
				}
				break;
			}
			pos.x -= tab_width;
		}
	}
}

void TabContainer::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_RESIZED: {

			Vector<Control *> tabs = _get_tabs();
			int side_margin = get_constant("side_margin");
			Ref<Texture> menu = get_icon("menu");
			Ref<Texture> increment = get_icon("increment");
			Ref<Texture> decrement = get_icon("decrement");
			int header_width = get_size().width - side_margin * 2;

			// Find the width of the header area.
			if (popup)
				header_width -= menu->get_width();
			if (buttons_visible_cache)
				header_width -= increment->get_width() + decrement->get_width();
			if (popup || buttons_visible_cache)
				header_width += side_margin;

			// Find the width of all tabs after first_tab_cache.
			int all_tabs_width = 0;
			for (int i = first_tab_cache; i < tabs.size(); i++) {
				int tab_width = _get_tab_width(i);
				all_tabs_width += tab_width;
			}

			// Check if tabs before first_tab_cache would fit into the header area.
			for (int i = first_tab_cache - 1; i >= 0; i--) {
				int tab_width = _get_tab_width(i);

				if (all_tabs_width + tab_width > header_width)
					break;

				all_tabs_width += tab_width;
				first_tab_cache--;
			}
		} break;

		case NOTIFICATION_DRAW: {

			RID canvas = get_canvas_item();
			Size2 size = get_size();

			// Draw only the tab area if the header is hidden.
			Ref<StyleBox> panel = get_stylebox("panel");
			if (!tabs_visible) {
				panel->draw(canvas, Rect2(0, 0, size.width, size.height));
				return;
			}

			Vector<Control *> tabs = _get_tabs();
			Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
			Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
			Ref<StyleBox> tab_disabled = get_stylebox("tab_disabled");
			Ref<Texture> increment = get_icon("increment");
			Ref<Texture> decrement = get_icon("decrement");
			Ref<Texture> menu = get_icon("menu");
			Ref<Texture> menu_hl = get_icon("menu_hl");
			Ref<Font> font = get_font("font");
			Color font_color_fg = get_color("font_color_fg");
			Color font_color_bg = get_color("font_color_bg");
			Color font_color_disabled = get_color("font_color_disabled");
			int side_margin = get_constant("side_margin");
			int icon_text_distance = get_constant("hseparation");

			// Find out start and width of the header area.
			int header_x = side_margin;
			int header_width = size.width - side_margin * 2;
			int header_height = _get_top_margin();
			if (popup)
				header_width -= menu->get_width();

			// Check if all tabs would fit into the header area.
			int all_tabs_width = 0;
			for (int i = 0; i < tabs.size(); i++) {
				int tab_width = _get_tab_width(i);
				all_tabs_width += tab_width;

				if (all_tabs_width > header_width) {
					// Not all tabs are visible at the same time - reserve space for navigation buttons.
					buttons_visible_cache = true;
					header_width -= decrement->get_width() + increment->get_width();
					break;
				} else {
					buttons_visible_cache = false;
				}
			}
			// With buttons, a right side margin does not need to be respected.
			if (popup || buttons_visible_cache) {
				header_width += side_margin;
			}

			if (!buttons_visible_cache) {
				first_tab_cache = 0;
			}

			// Go through the visible tabs to find the width they occupy.
			all_tabs_width = 0;
			Vector<int> tab_widths;
			for (int i = first_tab_cache; i < tabs.size(); i++) {
				int tab_width = _get_tab_width(i);
				if (all_tabs_width + tab_width > header_width && tab_widths.size() > 0)
					break;
				all_tabs_width += tab_width;
				tab_widths.push_back(tab_width);
			}

			// Find the offset at which to draw tabs, according to the alignment.
			switch (align) {
				case ALIGN_LEFT:
					tabs_ofs_cache = header_x;
					break;
				case ALIGN_CENTER:
					tabs_ofs_cache = header_x + (header_width / 2) - (all_tabs_width / 2);
					break;
				case ALIGN_RIGHT:
					tabs_ofs_cache = header_x + header_width - all_tabs_width;
					break;
			}

			// Draw the tab area.
			panel->draw(canvas, Rect2(0, header_height, size.width, size.height - header_height));

			// Draw all visible tabs.
			int x = 0;
			for (int i = 0; i < tab_widths.size(); i++) {
				Ref<StyleBox> tab_style;
				Color font_color;
				if (get_tab_disabled(i + first_tab_cache)) {
					tab_style = tab_disabled;
					font_color = font_color_disabled;
				} else if (i + first_tab_cache == current) {
					tab_style = tab_fg;
					font_color = font_color_fg;
				} else {
					tab_style = tab_bg;
					font_color = font_color_bg;
				}

				// Draw the tab background.
				int tab_width = tab_widths[i];
				Rect2 tab_rect(tabs_ofs_cache + x, 0, tab_width, header_height);
				tab_style->draw(canvas, tab_rect);

				// Draw the tab contents.
				Control *control = Object::cast_to<Control>(tabs[i + first_tab_cache]);
				String text = control->has_meta("_tab_name") ? String(tr(String(control->get_meta("_tab_name")))) : String(control->get_name());

				int x_content = tab_rect.position.x + tab_style->get_margin(MARGIN_LEFT);
				int top_margin = tab_style->get_margin(MARGIN_TOP);
				int y_center = top_margin + (tab_rect.size.y - tab_style->get_minimum_size().y) / 2;

				// Draw the tab icon.
				if (control->has_meta("_tab_icon")) {
					Ref<Texture> icon = control->get_meta("_tab_icon");
					if (icon.is_valid()) {
						int y = y_center - (icon->get_height() / 2);
						icon->draw(canvas, Point2i(x_content, y));
						if (text != "")
							x_content += icon->get_width() + icon_text_distance;
					}
				}

				// Draw the tab text.
				Point2i text_pos(x_content, y_center - (font->get_height() / 2) + font->get_ascent());
				font->draw(canvas, text_pos, text, font_color);

				x += tab_width;
				last_tab_cache = i + first_tab_cache;
			}

			// Draw the popup menu.
			x = get_size().width;
			if (popup) {
				x -= menu->get_width();
				if (mouse_x_cache > x)
					menu_hl->draw(get_canvas_item(), Size2(x, (header_height - menu_hl->get_height()) / 2));
				else
					menu->draw(get_canvas_item(), Size2(x, (header_height - menu->get_height()) / 2));
			}

			// Draw the navigation buttons.
			if (buttons_visible_cache) {
				int y_center = header_height / 2;

				x -= increment->get_width();
				increment->draw(canvas,
						Point2(x, y_center - (increment->get_height() / 2)),
						Color(1, 1, 1, last_tab_cache < tabs.size() - 1 ? 1.0 : 0.5));

				x -= decrement->get_width();
				decrement->draw(canvas,
						Point2(x, y_center - (decrement->get_height() / 2)),
						Color(1, 1, 1, first_tab_cache > 0 ? 1.0 : 0.5));
			}
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			call_deferred("_on_theme_changed"); //wait until all changed theme
		} break;
	}
}

void TabContainer::_on_theme_changed() {
	if (get_tab_count() > 0) {
		set_current_tab(get_current_tab());
	}
}

int TabContainer::_get_tab_width(int p_index) const {

	ERR_FAIL_INDEX_V(p_index, get_tab_count(), 0);
	Control *control = Object::cast_to<Control>(_get_tabs()[p_index]);
	if (!control || control->is_set_as_toplevel())
		return 0;

	// Get the width of the text displayed on the tab.
	Ref<Font> font = get_font("font");
	String text = control->has_meta("_tab_name") ? String(tr(String(control->get_meta("_tab_name")))) : String(control->get_name());
	int width = font->get_string_size(text).width;

	// Add space for a tab icon.
	if (control->has_meta("_tab_icon")) {
		Ref<Texture> icon = control->get_meta("_tab_icon");
		if (icon.is_valid()) {
			width += icon->get_width();
			if (text != "")
				width += get_constant("hseparation");
		}
	}

	// Respect a minimum size.
	Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
	Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
	Ref<StyleBox> tab_disabled = get_stylebox("tab_disabled");
	if (get_tab_disabled(p_index)) {
		width += tab_disabled->get_minimum_size().width;
	} else if (p_index == current) {
		width += tab_fg->get_minimum_size().width;
	} else {
		width += tab_bg->get_minimum_size().width;
	}

	return width;
}

Vector<Control *> TabContainer::_get_tabs() const {

	Vector<Control *> controls;
	for (int i = 0; i < get_child_count(); i++) {

		Control *control = Object::cast_to<Control>(get_child(i));
		if (!control || control->is_toplevel_control())
			continue;

		controls.push_back(control);
	}
	return controls;
}

void TabContainer::_child_renamed_callback() {

	update();
}

void TabContainer::add_child_notify(Node *p_child) {

	Container::add_child_notify(p_child);

	Control *c = Object::cast_to<Control>(p_child);
	if (!c)
		return;
	if (c->is_set_as_toplevel())
		return;

	bool first = false;

	if (get_tab_count() != 1)
		c->hide();
	else {
		c->show();
		//call_deferred("set_current_tab",0);
		first = true;
		current = 0;
		previous = 0;
	}
	c->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	if (tabs_visible)
		c->set_margin(MARGIN_TOP, _get_top_margin());
	Ref<StyleBox> sb = get_stylebox("panel");
	c->set_margin(Margin(MARGIN_TOP), c->get_margin(Margin(MARGIN_TOP)) + sb->get_margin(Margin(MARGIN_TOP)));
	c->set_margin(Margin(MARGIN_LEFT), c->get_margin(Margin(MARGIN_LEFT)) + sb->get_margin(Margin(MARGIN_LEFT)));
	c->set_margin(Margin(MARGIN_RIGHT), c->get_margin(Margin(MARGIN_RIGHT)) - sb->get_margin(Margin(MARGIN_RIGHT)));
	c->set_margin(Margin(MARGIN_BOTTOM), c->get_margin(Margin(MARGIN_BOTTOM)) - sb->get_margin(Margin(MARGIN_BOTTOM)));

	update();
	p_child->connect("renamed", this, "_child_renamed_callback");
	if (first)
		emit_signal("tab_changed", current);
}

int TabContainer::get_tab_count() const {

	return _get_tabs().size();
}

void TabContainer::set_current_tab(int p_current) {

	ERR_FAIL_INDEX(p_current, get_tab_count());

	int pending_previous = current;
	current = p_current;

	Ref<StyleBox> sb = get_stylebox("panel");
	Vector<Control *> tabs = _get_tabs();
	for (int i = 0; i < tabs.size(); i++) {

		Control *c = tabs[i];
		if (i == current) {
			c->show();
			c->set_anchors_and_margins_preset(Control::PRESET_WIDE);
			if (tabs_visible)
				c->set_margin(MARGIN_TOP, _get_top_margin());
			c->set_margin(Margin(MARGIN_TOP), c->get_margin(Margin(MARGIN_TOP)) + sb->get_margin(Margin(MARGIN_TOP)));
			c->set_margin(Margin(MARGIN_LEFT), c->get_margin(Margin(MARGIN_LEFT)) + sb->get_margin(Margin(MARGIN_LEFT)));
			c->set_margin(Margin(MARGIN_RIGHT), c->get_margin(Margin(MARGIN_RIGHT)) - sb->get_margin(Margin(MARGIN_RIGHT)));
			c->set_margin(Margin(MARGIN_BOTTOM), c->get_margin(Margin(MARGIN_BOTTOM)) - sb->get_margin(Margin(MARGIN_BOTTOM)));

		} else
			c->hide();
	}

	_change_notify("current_tab");

	if (pending_previous == current)
		emit_signal("tab_selected", current);
	else {
		previous = pending_previous;
		emit_signal("tab_selected", current);
		emit_signal("tab_changed", current);
	}

	update();
}

int TabContainer::get_current_tab() const {

	return current;
}

int TabContainer::get_previous_tab() const {

	return previous;
}

Control *TabContainer::get_tab_control(int p_idx) const {

	Vector<Control *> tabs = _get_tabs();
	if (p_idx >= 0 && p_idx < tabs.size())
		return tabs[p_idx];
	else
		return NULL;
}

Control *TabContainer::get_current_tab_control() const {

	Vector<Control *> tabs = _get_tabs();
	if (current >= 0 && current < tabs.size())
		return tabs[current];
	else
		return NULL;
}

void TabContainer::remove_child_notify(Node *p_child) {

	Container::remove_child_notify(p_child);

	call_deferred("_update_current_tab");

	p_child->disconnect("renamed", this, "_child_renamed_callback");

	update();
}

void TabContainer::_update_current_tab() {

	int tc = get_tab_count();
	if (current >= tc)
		current = tc - 1;
	if (current < 0)
		current = 0;
	else
		set_current_tab(current);
}

Variant TabContainer::get_drag_data(const Point2 &p_point) {

	if (!drag_to_rearrange_enabled)
		return Variant();

	int tab_over = get_tab_idx_at_point(p_point);

	if (tab_over < 0)
		return Variant();

	HBoxContainer *drag_preview = memnew(HBoxContainer);

	Ref<Texture> icon = get_tab_icon(tab_over);
	if (!icon.is_null()) {
		TextureRect *tf = memnew(TextureRect);
		tf->set_texture(icon);
		drag_preview->add_child(tf);
	}
	Label *label = memnew(Label(get_tab_title(tab_over)));
	drag_preview->add_child(label);
	set_drag_preview(drag_preview);

	Dictionary drag_data;
	drag_data["type"] = "tabc_element";
	drag_data["tabc_element"] = tab_over;
	drag_data["from_path"] = get_path();
	return drag_data;
}

bool TabContainer::can_drop_data(const Point2 &p_point, const Variant &p_data) const {

	if (!drag_to_rearrange_enabled)
		return false;

	Dictionary d = p_data;
	if (!d.has("type"))
		return false;

	if (String(d["type"]) == "tabc_element") {

		NodePath from_path = d["from_path"];
		NodePath to_path = get_path();
		if (from_path == to_path) {
			return true;
		} else if (get_tabs_rearrange_group() != -1) {
			// drag and drop between other TabContainers
			Node *from_node = get_node(from_path);
			TabContainer *from_tabc = Object::cast_to<TabContainer>(from_node);
			if (from_tabc && from_tabc->get_tabs_rearrange_group() == get_tabs_rearrange_group()) {
				return true;
			}
		}
	}
	return false;
}

void TabContainer::drop_data(const Point2 &p_point, const Variant &p_data) {

	if (!drag_to_rearrange_enabled)
		return;

	int hover_now = get_tab_idx_at_point(p_point);

	Dictionary d = p_data;
	if (!d.has("type"))
		return;

	if (String(d["type"]) == "tabc_element") {

		int tab_from_id = d["tabc_element"];
		NodePath from_path = d["from_path"];
		NodePath to_path = get_path();
		if (from_path == to_path) {
			if (hover_now < 0)
				hover_now = get_tab_count() - 1;
			move_child(get_tab_control(tab_from_id), hover_now);
			set_current_tab(hover_now);
		} else if (get_tabs_rearrange_group() != -1) {
			// drag and drop between TabContainers
			Node *from_node = get_node(from_path);
			TabContainer *from_tabc = Object::cast_to<TabContainer>(from_node);
			if (from_tabc && from_tabc->get_tabs_rearrange_group() == get_tabs_rearrange_group()) {
				Control *moving_tabc = from_tabc->get_tab_control(tab_from_id);
				from_tabc->remove_child(moving_tabc);
				add_child(moving_tabc);
				if (hover_now < 0)
					hover_now = get_tab_count() - 1;
				move_child(moving_tabc, hover_now);
				set_current_tab(hover_now);
				emit_signal("tab_changed", hover_now);
			}
		}
	}
	update();
}

int TabContainer::get_tab_idx_at_point(const Point2 &p_point) const {

	if (get_tab_count() == 0)
		return -1;

	// must be on tabs in the tab header area.
	if (p_point.x < tabs_ofs_cache || p_point.y > _get_top_margin())
		return -1;

	Size2 size = get_size();
	int right_ofs = 0;

	if (popup) {
		Ref<Texture> menu = get_icon("menu");
		right_ofs += menu->get_width();
	}
	if (buttons_visible_cache) {
		Ref<Texture> increment = get_icon("increment");
		Ref<Texture> decrement = get_icon("decrement");
		right_ofs += increment->get_width() + decrement->get_width();
	}
	if (p_point.x > size.width - right_ofs) {
		return -1;
	}

	// get the tab at the point
	Vector<Control *> tabs = _get_tabs();
	int px = p_point.x;
	px -= tabs_ofs_cache;
	for (int i = first_tab_cache; i <= last_tab_cache; i++) {
		int tab_width = _get_tab_width(i);
		if (px < tab_width) {
			return i;
		}
		px -= tab_width;
	}
	return -1;
}

void TabContainer::set_tab_align(TabAlign p_align) {

	ERR_FAIL_INDEX(p_align, 3);
	align = p_align;
	update();

	_change_notify("tab_align");
}

TabContainer::TabAlign TabContainer::get_tab_align() const {

	return align;
}

void TabContainer::set_tabs_visible(bool p_visibe) {

	if (p_visibe == tabs_visible)
		return;

	tabs_visible = p_visibe;

	Vector<Control *> tabs = _get_tabs();
	for (int i = 0; i < tabs.size(); i++) {

		Control *c = tabs[i];
		if (p_visibe)
			c->set_margin(MARGIN_TOP, _get_top_margin());
		else
			c->set_margin(MARGIN_TOP, 0);
	}
	update();
}

bool TabContainer::are_tabs_visible() const {

	return tabs_visible;
}

Control *TabContainer::_get_tab(int p_idx) const {

	return get_tab_control(p_idx);
}

void TabContainer::set_tab_title(int p_tab, const String &p_title) {

	Control *child = _get_tab(p_tab);
	ERR_FAIL_COND(!child);
	child->set_meta("_tab_name", p_title);
}

String TabContainer::get_tab_title(int p_tab) const {

	Control *child = _get_tab(p_tab);
	ERR_FAIL_COND_V(!child, "");
	if (child->has_meta("_tab_name"))
		return child->get_meta("_tab_name");
	else
		return child->get_name();
}

void TabContainer::set_tab_icon(int p_tab, const Ref<Texture> &p_icon) {

	Control *child = _get_tab(p_tab);
	ERR_FAIL_COND(!child);
	child->set_meta("_tab_icon", p_icon);
}
Ref<Texture> TabContainer::get_tab_icon(int p_tab) const {

	Control *child = _get_tab(p_tab);
	ERR_FAIL_COND_V(!child, Ref<Texture>());
	if (child->has_meta("_tab_icon"))
		return child->get_meta("_tab_icon");
	else
		return Ref<Texture>();
}

void TabContainer::set_tab_disabled(int p_tab, bool p_disabled) {

	Control *child = _get_tab(p_tab);
	ERR_FAIL_COND(!child);
	child->set_meta("_tab_disabled", p_disabled);
	update();
}

bool TabContainer::get_tab_disabled(int p_tab) const {

	Control *child = _get_tab(p_tab);
	ERR_FAIL_COND_V(!child, false);
	if (child->has_meta("_tab_disabled"))
		return child->get_meta("_tab_disabled");
	else
		return false;
}

void TabContainer::get_translatable_strings(List<String> *p_strings) const {

	Vector<Control *> tabs = _get_tabs();
	for (int i = 0; i < tabs.size(); i++) {

		Control *c = tabs[i];

		if (!c->has_meta("_tab_name"))
			continue;

		String name = c->get_meta("_tab_name");

		if (name != "")
			p_strings->push_back(name);
	}
}

Size2 TabContainer::get_minimum_size() const {

	Size2 ms;

	Vector<Control *> tabs = _get_tabs();
	for (int i = 0; i < tabs.size(); i++) {

		Control *c = tabs[i];

		if (!c->is_visible_in_tree())
			continue;

		Size2 cms = c->get_combined_minimum_size();
		ms.x = MAX(ms.x, cms.x);
		ms.y = MAX(ms.y, cms.y);
	}

	Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
	Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
	Ref<StyleBox> tab_disabled = get_stylebox("tab_disabled");
	Ref<Font> font = get_font("font");

	ms.y += MAX(MAX(tab_bg->get_minimum_size().y, tab_fg->get_minimum_size().y), tab_disabled->get_minimum_size().y);
	ms.y += font->get_height();

	Ref<StyleBox> sb = get_stylebox("panel");
	ms += sb->get_minimum_size();

	return ms;
}

void TabContainer::set_popup(Node *p_popup) {
	ERR_FAIL_NULL(p_popup);
	popup = Object::cast_to<Popup>(p_popup);
	update();
}

Popup *TabContainer::get_popup() const {
	return popup;
}

void TabContainer::set_drag_to_rearrange_enabled(bool p_enabled) {
	drag_to_rearrange_enabled = p_enabled;
}

bool TabContainer::get_drag_to_rearrange_enabled() const {
	return drag_to_rearrange_enabled;
}
void TabContainer::set_tabs_rearrange_group(int p_group_id) {
	tabs_rearrange_group = p_group_id;
}

int TabContainer::get_tabs_rearrange_group() const {
	return tabs_rearrange_group;
}

void TabContainer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &TabContainer::_gui_input);
	ClassDB::bind_method(D_METHOD("get_tab_count"), &TabContainer::get_tab_count);
	ClassDB::bind_method(D_METHOD("set_current_tab", "tab_idx"), &TabContainer::set_current_tab);
	ClassDB::bind_method(D_METHOD("get_current_tab"), &TabContainer::get_current_tab);
	ClassDB::bind_method(D_METHOD("get_previous_tab"), &TabContainer::get_previous_tab);
	ClassDB::bind_method(D_METHOD("get_current_tab_control"), &TabContainer::get_current_tab_control);
	ClassDB::bind_method(D_METHOD("get_tab_control", "idx"), &TabContainer::get_tab_control);
	ClassDB::bind_method(D_METHOD("set_tab_align", "align"), &TabContainer::set_tab_align);
	ClassDB::bind_method(D_METHOD("get_tab_align"), &TabContainer::get_tab_align);
	ClassDB::bind_method(D_METHOD("set_tabs_visible", "visible"), &TabContainer::set_tabs_visible);
	ClassDB::bind_method(D_METHOD("are_tabs_visible"), &TabContainer::are_tabs_visible);
	ClassDB::bind_method(D_METHOD("set_tab_title", "tab_idx", "title"), &TabContainer::set_tab_title);
	ClassDB::bind_method(D_METHOD("get_tab_title", "tab_idx"), &TabContainer::get_tab_title);
	ClassDB::bind_method(D_METHOD("set_tab_icon", "tab_idx", "icon"), &TabContainer::set_tab_icon);
	ClassDB::bind_method(D_METHOD("get_tab_icon", "tab_idx"), &TabContainer::get_tab_icon);
	ClassDB::bind_method(D_METHOD("set_tab_disabled", "tab_idx", "disabled"), &TabContainer::set_tab_disabled);
	ClassDB::bind_method(D_METHOD("get_tab_disabled", "tab_idx"), &TabContainer::get_tab_disabled);
	ClassDB::bind_method(D_METHOD("set_popup", "popup"), &TabContainer::set_popup);
	ClassDB::bind_method(D_METHOD("get_popup"), &TabContainer::get_popup);
	ClassDB::bind_method(D_METHOD("set_drag_to_rearrange_enabled", "enabled"), &TabContainer::set_drag_to_rearrange_enabled);
	ClassDB::bind_method(D_METHOD("get_drag_to_rearrange_enabled"), &TabContainer::get_drag_to_rearrange_enabled);
	ClassDB::bind_method(D_METHOD("set_tabs_rearrange_group", "group_id"), &TabContainer::set_tabs_rearrange_group);
	ClassDB::bind_method(D_METHOD("get_tabs_rearrange_group"), &TabContainer::get_tabs_rearrange_group);

	ClassDB::bind_method(D_METHOD("_child_renamed_callback"), &TabContainer::_child_renamed_callback);
	ClassDB::bind_method(D_METHOD("_on_theme_changed"), &TabContainer::_on_theme_changed);
	ClassDB::bind_method(D_METHOD("_update_current_tab"), &TabContainer::_update_current_tab);

	ADD_SIGNAL(MethodInfo("tab_changed", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_selected", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("pre_popup_pressed"));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "tab_align", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_tab_align", "get_tab_align");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_tab", PROPERTY_HINT_RANGE, "-1,4096,1", PROPERTY_USAGE_EDITOR), "set_current_tab", "get_current_tab");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "tabs_visible"), "set_tabs_visible", "are_tabs_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "drag_to_rearrange_enabled"), "set_drag_to_rearrange_enabled", "get_drag_to_rearrange_enabled");

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);
}

TabContainer::TabContainer() {

	first_tab_cache = 0;
	last_tab_cache = 0;
	buttons_visible_cache = false;
	tabs_ofs_cache = 0;
	current = 0;
	previous = 0;
	mouse_x_cache = 0;
	align = ALIGN_CENTER;
	tabs_visible = true;
	popup = NULL;
	drag_to_rearrange_enabled = false;
	tabs_rearrange_group = -1;
}
