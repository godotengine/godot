/*************************************************************************/
/*  tab_container.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/object/message_queue.h"
#include "core/string/translation.h"

#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"

int TabContainer::_get_top_margin() const {
	if (!tabs_visible) {
		return 0;
	}

	// Respect the minimum tab height.
	Ref<StyleBox> tab_unselected = get_theme_stylebox(SNAME("tab_unselected"));
	Ref<StyleBox> tab_selected = get_theme_stylebox(SNAME("tab_selected"));
	Ref<StyleBox> tab_disabled = get_theme_stylebox(SNAME("tab_disabled"));

	int tab_height = MAX(MAX(tab_unselected->get_minimum_size().height, tab_selected->get_minimum_size().height), tab_disabled->get_minimum_size().height);

	// Font height or higher icon wins.
	int content_height = 0;

	Vector<Control *> tabs = _get_tabs();
	for (int i = 0; i < tabs.size(); i++) {
		content_height = MAX(content_height, text_buf[i]->get_size().y);

		Control *c = tabs[i];
		if (!c->has_meta("_tab_icon")) {
			continue;
		}

		Ref<Texture2D> tex = c->get_meta("_tab_icon");
		if (!tex.is_valid()) {
			continue;
		}
		content_height = MAX(content_height, tex->get_size().height);
	}

	return tab_height + content_height;
}

void TabContainer::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseButton> mb = p_event;

	Popup *popup = get_popup();

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		Point2 pos = mb->get_position();
		Size2 size = get_size();

		// Click must be on tabs in the tab header area.
		if (pos.y > _get_top_margin()) {
			return;
		}

		// Handle menu button.
		Ref<Texture2D> menu = get_theme_icon(SNAME("menu"));

		if (is_layout_rtl()) {
			if (popup && pos.x < menu->get_width()) {
				emit_signal(SNAME("pre_popup_pressed"));

				Vector2 popup_pos = get_screen_position();
				popup_pos.y += menu->get_height();

				popup->set_position(popup_pos);
				popup->popup();
				return;
			}
		} else {
			if (popup && pos.x > size.width - menu->get_width()) {
				emit_signal(SNAME("pre_popup_pressed"));

				Vector2 popup_pos = get_screen_position();
				popup_pos.x += size.width - popup->get_size().width;
				popup_pos.y += menu->get_height();

				popup->set_position(popup_pos);
				popup->popup();
				return;
			}
		}

		// Do not activate tabs when tabs is empty.
		if (get_tab_count() == 0) {
			return;
		}

		Vector<Control *> tabs = _get_tabs();

		// Handle navigation buttons.
		if (buttons_visible_cache) {
			int popup_ofs = 0;
			if (popup) {
				popup_ofs = menu->get_width();
			}

			Ref<Texture2D> increment = get_theme_icon(SNAME("increment"));
			Ref<Texture2D> decrement = get_theme_icon(SNAME("decrement"));
			if (is_layout_rtl()) {
				if (pos.x < popup_ofs + decrement->get_width()) {
					if (last_tab_cache < tabs.size() - 1) {
						first_tab_cache += 1;
						update();
					}
					return;
				} else if (pos.x < popup_ofs + increment->get_width() + decrement->get_width()) {
					if (first_tab_cache > 0) {
						first_tab_cache -= 1;
						update();
					}
					return;
				}
			} else {
				if (pos.x > size.width - increment->get_width() - popup_ofs && pos.x) {
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
		}

		// Activate the clicked tab.
		if (is_layout_rtl()) {
			pos.x = size.width - pos.x;
		}

		if (pos.x < tabs_ofs_cache) {
			return;
		}

		pos.x -= tabs_ofs_cache;
		for (int i = first_tab_cache; i <= last_tab_cache; i++) {
			if (get_tab_hidden(i)) {
				continue;
			}
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

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		Point2 pos = mm->get_position();
		Size2 size = get_size();

		// Mouse must be on tabs in the tab header area.
		if (pos.y > _get_top_margin()) {
			if (menu_hovered || highlight_arrow > -1) {
				menu_hovered = false;
				highlight_arrow = -1;
				update();
			}
			return;
		}

		Ref<Texture2D> menu = get_theme_icon(SNAME("menu"));
		if (popup) {
			if (is_layout_rtl()) {
				if (pos.x <= menu->get_width()) {
					if (!menu_hovered) {
						menu_hovered = true;
						highlight_arrow = -1;
						update();
						return;
					}
				} else if (menu_hovered) {
					menu_hovered = false;
					update();
				}
			} else {
				if (pos.x >= size.width - menu->get_width()) {
					if (!menu_hovered) {
						menu_hovered = true;
						highlight_arrow = -1;
						update();
						return;
					}
				} else if (menu_hovered) {
					menu_hovered = false;
					update();
				}
			}

			if (menu_hovered) {
				return;
			}
		}

		// Do not activate tabs when tabs is empty.
		if ((get_tab_count() == 0 || !buttons_visible_cache) && menu_hovered) {
			highlight_arrow = -1;
			update();
			return;
		}

		int popup_ofs = 0;
		if (popup) {
			popup_ofs = menu->get_width();
		}

		Ref<Texture2D> increment = get_theme_icon(SNAME("increment"));
		Ref<Texture2D> decrement = get_theme_icon(SNAME("decrement"));

		if (is_layout_rtl()) {
			if (pos.x <= popup_ofs + decrement->get_width()) {
				if (highlight_arrow != 1) {
					highlight_arrow = 1;
					update();
				}
			} else if (pos.x <= popup_ofs + increment->get_width() + decrement->get_width()) {
				if (highlight_arrow != 0) {
					highlight_arrow = 0;
					update();
				}
			} else if (highlight_arrow > -1) {
				highlight_arrow = -1;
				update();
			}
		} else {
			if (pos.x >= size.width - increment->get_width() - popup_ofs) {
				if (highlight_arrow != 1) {
					highlight_arrow = 1;
					update();
				}
			} else if (pos.x >= size.width - increment->get_width() - decrement->get_width() - popup_ofs) {
				if (highlight_arrow != 0) {
					highlight_arrow = 0;
					update();
				}
			} else if (highlight_arrow > -1) {
				highlight_arrow = -1;
				update();
			}
		}
	}
}

void TabContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_RESIZED: {
			Vector<Control *> tabs = _get_tabs();
			int side_margin = get_theme_constant(SNAME("side_margin"));
			Ref<Texture2D> menu = get_theme_icon(SNAME("menu"));
			Ref<Texture2D> increment = get_theme_icon(SNAME("increment"));
			Ref<Texture2D> decrement = get_theme_icon(SNAME("decrement"));
			int header_width = get_size().width - side_margin * 2;

			// Find the width of the header area.
			Popup *popup = get_popup();
			if (popup) {
				header_width -= menu->get_width();
			}
			if (buttons_visible_cache) {
				header_width -= increment->get_width() + decrement->get_width();
			}
			if (popup || buttons_visible_cache) {
				header_width += side_margin;
			}

			// Find the width of all tabs after first_tab_cache.
			int all_tabs_width = 0;
			for (int i = first_tab_cache; i < tabs.size(); i++) {
				int tab_width = _get_tab_width(i);
				all_tabs_width += tab_width;
			}

			// Check if tabs before first_tab_cache would fit into the header area.
			for (int i = first_tab_cache - 1; i >= 0; i--) {
				int tab_width = _get_tab_width(i);

				if (all_tabs_width + tab_width > header_width) {
					break;
				}

				all_tabs_width += tab_width;
				first_tab_cache--;
			}
		} break;
		case NOTIFICATION_DRAW: {
			RID canvas = get_canvas_item();
			Size2 size = get_size();
			bool rtl = is_layout_rtl();

			// Draw only the tab area if the header is hidden.
			Ref<StyleBox> panel = get_theme_stylebox(SNAME("panel"));
			if (!tabs_visible) {
				panel->draw(canvas, Rect2(0, 0, size.width, size.height));
				return;
			}

			Vector<Control *> tabs = _get_tabs();
			Ref<StyleBox> tab_unselected = get_theme_stylebox(SNAME("tab_unselected"));
			Ref<StyleBox> tab_selected = get_theme_stylebox(SNAME("tab_selected"));
			Ref<StyleBox> tab_disabled = get_theme_stylebox(SNAME("tab_disabled"));
			Ref<Texture2D> increment = get_theme_icon(SNAME("increment"));
			Ref<Texture2D> increment_hl = get_theme_icon(SNAME("increment_highlight"));
			Ref<Texture2D> decrement = get_theme_icon(SNAME("decrement"));
			Ref<Texture2D> decrement_hl = get_theme_icon(SNAME("decrement_highlight"));
			Ref<Texture2D> menu = get_theme_icon(SNAME("menu"));
			Ref<Texture2D> menu_hl = get_theme_icon(SNAME("menu_highlight"));
			Color font_selected_color = get_theme_color(SNAME("font_selected_color"));
			Color font_unselected_color = get_theme_color(SNAME("font_unselected_color"));
			Color font_disabled_color = get_theme_color(SNAME("font_disabled_color"));
			int side_margin = get_theme_constant(SNAME("side_margin"));

			// Find out start and width of the header area.
			int header_x = side_margin;
			int header_width = size.width - side_margin * 2;
			int header_height = _get_top_margin();
			Popup *popup = get_popup();
			if (popup) {
				header_width -= menu->get_width();
			}

			// Check if all tabs would fit into the header area.
			int all_tabs_width = 0;
			for (int i = 0; i < tabs.size(); i++) {
				if (get_tab_hidden(i)) {
					continue;
				}
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
				if (get_tab_hidden(i)) {
					tab_widths.push_back(0);
					continue;
				}
				int tab_width = _get_tab_width(i);
				if (all_tabs_width + tab_width > header_width && tab_widths.size() > 0) {
					break;
				}
				all_tabs_width += tab_width;
				tab_widths.push_back(tab_width);
			}

			// Find the offset at which to draw tabs, according to the alignment.
			switch (alignment) {
				case ALIGNMENT_LEFT:
					tabs_ofs_cache = header_x;
					break;
				case ALIGNMENT_CENTER:
					tabs_ofs_cache = header_x + (header_width / 2) - (all_tabs_width / 2);
					break;
				case ALIGNMENT_RIGHT:
					tabs_ofs_cache = header_x + header_width - all_tabs_width;
					break;
			}

			if (all_tabs_in_front) {
				// Draw the tab area.
				panel->draw(canvas, Rect2(0, header_height, size.width, size.height - header_height));
			}

			// Draw unselected tabs in back
			int x = 0;
			int x_current = 0;
			int index = 0;
			for (int i = 0; i < tab_widths.size(); i++) {
				index = i + first_tab_cache;
				if (get_tab_hidden(index)) {
					continue;
				}

				int tab_width = tab_widths[i];
				if (index == current) {
					x_current = x;
				} else if (get_tab_disabled(index)) {
					if (rtl) {
						_draw_tab(tab_disabled, font_disabled_color, index, size.width - (tabs_ofs_cache + x) - tab_width);
					} else {
						_draw_tab(tab_disabled, font_disabled_color, index, tabs_ofs_cache + x);
					}
				} else {
					if (rtl) {
						_draw_tab(tab_unselected, font_unselected_color, index, size.width - (tabs_ofs_cache + x) - tab_width);
					} else {
						_draw_tab(tab_unselected, font_unselected_color, index, tabs_ofs_cache + x);
					}
				}

				x += tab_width;
				last_tab_cache = index;
			}

			if (!all_tabs_in_front) {
				// Draw the tab area.
				panel->draw(canvas, Rect2(0, header_height, size.width, size.height - header_height));
			}

			// Draw selected tab in front. Only draw selected tab when it's in visible range.
			if (tabs.size() > 0 && current - first_tab_cache < tab_widths.size() && current >= first_tab_cache) {
				Ref<StyleBox> current_style_box = get_tab_disabled(current) ? tab_disabled : tab_selected;
				if (rtl) {
					_draw_tab(current_style_box, font_selected_color, current, size.width - (tabs_ofs_cache + x_current) - tab_widths[current]);
				} else {
					_draw_tab(current_style_box, font_selected_color, current, tabs_ofs_cache + x_current);
				}
			}

			// Draw the popup menu.
			if (rtl) {
				x = 0;
			} else {
				x = get_size().width;
			}
			if (popup) {
				if (!rtl) {
					x -= menu->get_width();
				}
				if (menu_hovered) {
					menu_hl->draw(get_canvas_item(), Size2(x, (header_height - menu_hl->get_height()) / 2));
				} else {
					menu->draw(get_canvas_item(), Size2(x, (header_height - menu->get_height()) / 2));
				}
				if (rtl) {
					x += menu->get_width();
				}
			}

			// Draw the navigation buttons.
			if (buttons_visible_cache) {
				if (rtl) {
					if (last_tab_cache < tabs.size() - 1) {
						draw_texture(highlight_arrow == 1 ? decrement_hl : decrement, Point2(x, (header_height - increment->get_height()) / 2));
					} else {
						draw_texture(decrement, Point2(x, (header_height - increment->get_height()) / 2), Color(1, 1, 1, 0.5));
					}
					x += increment->get_width();

					if (first_tab_cache > 0) {
						draw_texture(highlight_arrow == 0 ? increment_hl : increment, Point2(x, (header_height - decrement->get_height()) / 2));
					} else {
						draw_texture(increment, Point2(x, (header_height - decrement->get_height()) / 2), Color(1, 1, 1, 0.5));
					}
					x += decrement->get_width();
				} else {
					x -= increment->get_width();
					if (last_tab_cache < tabs.size() - 1) {
						draw_texture(highlight_arrow == 1 ? increment_hl : increment, Point2(x, (header_height - increment->get_height()) / 2));
					} else {
						draw_texture(increment, Point2(x, (header_height - increment->get_height()) / 2), Color(1, 1, 1, 0.5));
					}

					x -= decrement->get_width();
					if (first_tab_cache > 0) {
						draw_texture(highlight_arrow == 0 ? decrement_hl : decrement, Point2(x, (header_height - decrement->get_height()) / 2));
					} else {
						draw_texture(decrement, Point2(x, (header_height - decrement->get_height()) / 2), Color(1, 1, 1, 0.5));
					}
				}
			}
		} break;
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			Vector<Control *> tabs = _get_tabs();
			for (int i = 0; i < tabs.size(); i++) {
				text_buf.write[i]->clear();
			}
			_theme_changing = true;
			call_deferred(SNAME("_on_theme_changed")); // Wait until all changed theme.
		} break;
	}
}

void TabContainer::_draw_tab(Ref<StyleBox> &p_tab_style, Color &p_font_color, int p_index, float p_x) {
	Control *control = get_tab_control(p_index);
	RID canvas = get_canvas_item();
	Color font_outline_color = get_theme_color(SNAME("font_outline_color"));
	int outline_size = get_theme_constant(SNAME("outline_size"));
	int icon_text_distance = get_theme_constant(SNAME("icon_separation"));
	int tab_width = _get_tab_width(p_index);
	int header_height = _get_top_margin();

	// Draw the tab background.
	Rect2 tab_rect(p_x, 0, tab_width, header_height);
	p_tab_style->draw(canvas, tab_rect);

	// Draw the tab contents.
	String text = control->has_meta("_tab_name") ? String(atr(String(control->get_meta("_tab_name")))) : String(atr(control->get_name()));

	int x_content = tab_rect.position.x + p_tab_style->get_margin(SIDE_LEFT);
	int top_margin = p_tab_style->get_margin(SIDE_TOP);
	int y_center = top_margin + (tab_rect.size.y - p_tab_style->get_minimum_size().y) / 2;

	// Draw the tab icon.
	if (control->has_meta("_tab_icon")) {
		Ref<Texture2D> icon = control->get_meta("_tab_icon");
		if (icon.is_valid()) {
			int y = y_center - (icon->get_height() / 2);
			icon->draw(canvas, Point2i(x_content, y));
			if (!text.is_empty()) {
				x_content += icon->get_width() + icon_text_distance;
			}
		}
	}

	// Draw the tab text.
	Point2i text_pos(x_content, y_center - text_buf[p_index]->get_size().y / 2);
	if (outline_size > 0 && font_outline_color.a > 0) {
		text_buf[p_index]->draw_outline(canvas, text_pos, outline_size, font_outline_color);
	}
	text_buf[p_index]->draw(canvas, text_pos, p_font_color);
}

void TabContainer::_refresh_texts() {
	text_buf.clear();
	Vector<Control *> tabs = _get_tabs();
	bool rtl = is_layout_rtl();
	Ref<Font> font = get_theme_font(SNAME("font"));
	int font_size = get_theme_font_size(SNAME("font_size"));
	for (int i = 0; i < tabs.size(); i++) {
		Control *control = Object::cast_to<Control>(tabs[i]);
		String text = control->has_meta("_tab_name") ? String(atr(String(control->get_meta("_tab_name")))) : String(atr(control->get_name()));

		Ref<TextLine> name;
		name.instantiate();
		name->set_direction(rtl ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
		name->add_string(text, font, font_size, Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
		text_buf.push_back(name);
	}
}

void TabContainer::_on_theme_changed() {
	if (!_theme_changing) {
		return;
	}

	_refresh_texts();

	update_minimum_size();
	if (get_tab_count() > 0) {
		_repaint();
		update();
	}
	_theme_changing = false;
}

void TabContainer::_repaint() {
	Ref<StyleBox> sb = get_theme_stylebox(SNAME("panel"));
	Vector<Control *> tabs = _get_tabs();
	for (int i = 0; i < tabs.size(); i++) {
		Control *c = tabs[i];
		if (i == current) {
			c->show();
			c->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
			if (tabs_visible) {
				c->set_offset(SIDE_TOP, _get_top_margin());
			}
			c->set_offset(SIDE_TOP, c->get_offset(SIDE_TOP) + sb->get_margin(SIDE_TOP));
			c->set_offset(SIDE_LEFT, c->get_offset(SIDE_LEFT) + sb->get_margin(SIDE_LEFT));
			c->set_offset(SIDE_RIGHT, c->get_offset(SIDE_RIGHT) - sb->get_margin(SIDE_RIGHT));
			c->set_offset(SIDE_BOTTOM, c->get_offset(SIDE_BOTTOM) - sb->get_margin(SIDE_BOTTOM));

		} else {
			c->hide();
		}
	}
}

void TabContainer::_on_mouse_exited() {
	if (menu_hovered || highlight_arrow > -1) {
		menu_hovered = false;
		highlight_arrow = -1;
		update();
	}
}

int TabContainer::_get_tab_width(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, get_tab_count(), 0);
	Control *control = get_tab_control(p_index);
	if (!control || get_tab_hidden(p_index)) {
		return 0;
	}

	// Get the width of the text displayed on the tab.
	Ref<Font> font = get_theme_font(SNAME("font"));
	int font_size = get_theme_font_size(SNAME("font_size"));
	String text = control->has_meta("_tab_name") ? String(atr(String(control->get_meta("_tab_name")))) : String(atr(control->get_name()));
	int width = font->get_string_size(text, font_size).width;

	// Add space for a tab icon.
	if (control->has_meta("_tab_icon")) {
		Ref<Texture2D> icon = control->get_meta("_tab_icon");
		if (icon.is_valid()) {
			width += icon->get_width();
			if (!text.is_empty()) {
				width += get_theme_constant(SNAME("icon_separation"));
			}
		}
	}

	// Respect a minimum size.
	Ref<StyleBox> tab_unselected = get_theme_stylebox(SNAME("tab_unselected"));
	Ref<StyleBox> tab_selected = get_theme_stylebox(SNAME("tab_selected"));
	Ref<StyleBox> tab_disabled = get_theme_stylebox(SNAME("tab_disabled"));
	if (get_tab_disabled(p_index)) {
		width += tab_disabled->get_minimum_size().width;
	} else if (p_index == current) {
		width += tab_selected->get_minimum_size().width;
	} else {
		width += tab_unselected->get_minimum_size().width;
	}

	return width;
}

Vector<Control *> TabContainer::_get_tabs() const {
	Vector<Control *> controls;
	for (int i = 0; i < get_child_count(); i++) {
		Control *control = Object::cast_to<Control>(get_child(i));
		if (!control || control->is_set_as_top_level()) {
			continue;
		}

		controls.push_back(control);
	}
	return controls;
}

void TabContainer::_child_renamed_callback() {
	_refresh_texts();
	update();
}

void TabContainer::add_child_notify(Node *p_child) {
	Container::add_child_notify(p_child);

	Control *c = Object::cast_to<Control>(p_child);
	if (!c || c->is_set_as_top_level()) {
		return;
	}

	Vector<Control *> tabs = _get_tabs();
	_refresh_texts();

	bool first = false;

	if (tabs.size() != 1) {
		c->hide();
	} else {
		c->show();
		//call_deferred(SNAME("set_current_tab"),0);
		first = true;
		current = 0;
		previous = 0;
	}
	c->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	if (tabs_visible) {
		c->set_offset(SIDE_TOP, _get_top_margin());
	}
	Ref<StyleBox> sb = get_theme_stylebox(SNAME("panel"));
	c->set_offset(SIDE_TOP, c->get_offset(SIDE_TOP) + sb->get_margin(SIDE_TOP));
	c->set_offset(SIDE_LEFT, c->get_offset(SIDE_LEFT) + sb->get_margin(SIDE_LEFT));
	c->set_offset(SIDE_RIGHT, c->get_offset(SIDE_RIGHT) - sb->get_margin(SIDE_RIGHT));
	c->set_offset(SIDE_BOTTOM, c->get_offset(SIDE_BOTTOM) - sb->get_margin(SIDE_BOTTOM));
	update();
	p_child->connect("renamed", callable_mp(this, &TabContainer::_child_renamed_callback));
	if (first && is_inside_tree()) {
		emit_signal(SNAME("tab_changed"), current);
	}
}

void TabContainer::move_child_notify(Node *p_child) {
	Container::move_child_notify(p_child);

	Control *c = Object::cast_to<Control>(p_child);
	if (!c || c->is_set_as_top_level()) {
		return;
	}

	_update_current_tab();
	update();
}

int TabContainer::get_tab_count() const {
	return _get_tabs().size();
}

void TabContainer::set_current_tab(int p_current) {
	ERR_FAIL_INDEX(p_current, get_tab_count());

	int pending_previous = current;
	current = p_current;

	_repaint();

	if (pending_previous == current) {
		emit_signal(SNAME("tab_selected"), current);
	} else {
		previous = pending_previous;
		emit_signal(SNAME("tab_selected"), current);
		emit_signal(SNAME("tab_changed"), current);
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
	if (p_idx >= 0 && p_idx < tabs.size()) {
		return tabs[p_idx];
	} else {
		return nullptr;
	}
}

Control *TabContainer::get_current_tab_control() const {
	return get_tab_control(current);
}

void TabContainer::remove_child_notify(Node *p_child) {
	Container::remove_child_notify(p_child);

	Control *c = Object::cast_to<Control>(p_child);
	if (!c || c->is_set_as_top_level()) {
		return;
	}

	// Defer the call because tab is not yet removed (remove_child_notify is called right before p_child is actually removed).
	call_deferred(SNAME("_update_current_tab"));

	p_child->disconnect("renamed", callable_mp(this, &TabContainer::_child_renamed_callback));

	update();
}

void TabContainer::_update_current_tab() {
	_refresh_texts();

	int tc = get_tab_count();
	if (current >= tc) {
		current = tc - 1;
	}
	if (current < 0) {
		current = 0;
	} else {
		set_current_tab(current);
	}
}

Variant TabContainer::get_drag_data(const Point2 &p_point) {
	if (!drag_to_rearrange_enabled) {
		return Variant();
	}

	int tab_over = get_tab_idx_at_point(p_point);

	if (tab_over < 0) {
		return Variant();
	}

	HBoxContainer *drag_preview = memnew(HBoxContainer);

	Ref<Texture2D> icon = get_tab_icon(tab_over);
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
	if (!drag_to_rearrange_enabled) {
		return false;
	}

	Dictionary d = p_data;
	if (!d.has("type")) {
		return false;
	}

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
	if (!drag_to_rearrange_enabled) {
		return;
	}

	int hover_now = get_tab_idx_at_point(p_point);

	Dictionary d = p_data;
	if (!d.has("type")) {
		return;
	}

	if (String(d["type"]) == "tabc_element") {
		int tab_from_id = d["tabc_element"];
		NodePath from_path = d["from_path"];
		NodePath to_path = get_path();
		if (from_path == to_path) {
			if (hover_now < 0) {
				hover_now = get_tab_count() - 1;
			}
			move_child(get_tab_control(tab_from_id), get_tab_control(hover_now)->get_index());
			set_current_tab(hover_now);
		} else if (get_tabs_rearrange_group() != -1) {
			// drag and drop between TabContainers
			Node *from_node = get_node(from_path);
			TabContainer *from_tabc = Object::cast_to<TabContainer>(from_node);
			if (from_tabc && from_tabc->get_tabs_rearrange_group() == get_tabs_rearrange_group()) {
				Control *moving_tabc = from_tabc->get_tab_control(tab_from_id);
				from_tabc->remove_child(moving_tabc);
				add_child(moving_tabc, false, INTERNAL_MODE_FRONT);
				if (hover_now < 0) {
					hover_now = get_tab_count() - 1;
				}
				move_child(moving_tabc, get_tab_control(hover_now)->get_index());
				set_current_tab(hover_now);
				emit_signal(SNAME("tab_changed"), hover_now);
			}
		}
	}
	update();
}

int TabContainer::get_tab_idx_at_point(const Point2 &p_point) const {
	if (get_tab_count() == 0) {
		return -1;
	}

	// must be on tabs in the tab header area.
	if (p_point.y > _get_top_margin()) {
		return -1;
	}

	Size2 size = get_size();
	int button_ofs = 0;
	int px = p_point.x;

	if (is_layout_rtl()) {
		px = size.width - px;
	}

	if (px < tabs_ofs_cache) {
		return -1;
	}

	Popup *popup = get_popup();
	if (popup) {
		Ref<Texture2D> menu = get_theme_icon(SNAME("menu"));
		button_ofs += menu->get_width();
	}
	if (buttons_visible_cache) {
		Ref<Texture2D> increment = get_theme_icon(SNAME("increment"));
		Ref<Texture2D> decrement = get_theme_icon(SNAME("decrement"));
		button_ofs += increment->get_width() + decrement->get_width();
	}
	if (px > size.width - button_ofs) {
		return -1;
	}

	// get the tab at the point
	Vector<Control *> tabs = _get_tabs();
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

void TabContainer::set_tab_alignment(AlignmentMode p_alignment) {
	ERR_FAIL_INDEX(p_alignment, 3);
	alignment = p_alignment;
	update();
}

TabContainer::AlignmentMode TabContainer::get_tab_alignment() const {
	return alignment;
}

void TabContainer::set_tabs_visible(bool p_visible) {
	if (p_visible == tabs_visible) {
		return;
	}

	tabs_visible = p_visible;

	Vector<Control *> tabs = _get_tabs();
	for (int i = 0; i < tabs.size(); i++) {
		Control *c = tabs[i];
		if (p_visible) {
			c->set_offset(SIDE_TOP, _get_top_margin());
		} else {
			c->set_offset(SIDE_TOP, 0);
		}
	}

	update();
	update_minimum_size();
}

bool TabContainer::are_tabs_visible() const {
	return tabs_visible;
}

void TabContainer::set_all_tabs_in_front(bool p_in_front) {
	if (p_in_front == all_tabs_in_front) {
		return;
	}

	all_tabs_in_front = p_in_front;

	update();
}

bool TabContainer::is_all_tabs_in_front() const {
	return all_tabs_in_front;
}

void TabContainer::set_tab_title(int p_tab, const String &p_title) {
	Control *child = get_tab_control(p_tab);
	ERR_FAIL_COND(!child);
	child->set_meta("_tab_name", p_title);
	_refresh_texts();
	update();
}

String TabContainer::get_tab_title(int p_tab) const {
	Control *child = get_tab_control(p_tab);
	ERR_FAIL_COND_V(!child, "");
	if (child->has_meta("_tab_name")) {
		return child->get_meta("_tab_name");
	} else {
		return child->get_name();
	}
}

void TabContainer::set_tab_icon(int p_tab, const Ref<Texture2D> &p_icon) {
	Control *child = get_tab_control(p_tab);
	ERR_FAIL_COND(!child);
	child->set_meta("_tab_icon", p_icon);
	update();
}

Ref<Texture2D> TabContainer::get_tab_icon(int p_tab) const {
	Control *child = get_tab_control(p_tab);
	ERR_FAIL_COND_V(!child, Ref<Texture2D>());
	if (child->has_meta("_tab_icon")) {
		return child->get_meta("_tab_icon");
	} else {
		return Ref<Texture2D>();
	}
}

void TabContainer::set_tab_disabled(int p_tab, bool p_disabled) {
	Control *child = get_tab_control(p_tab);
	ERR_FAIL_COND(!child);
	child->set_meta("_tab_disabled", p_disabled);
	update();
}

bool TabContainer::get_tab_disabled(int p_tab) const {
	Control *child = get_tab_control(p_tab);
	ERR_FAIL_COND_V(!child, false);
	if (child->has_meta("_tab_disabled")) {
		return child->get_meta("_tab_disabled");
	} else {
		return false;
	}
}

void TabContainer::set_tab_hidden(int p_tab, bool p_hidden) {
	Control *child = get_tab_control(p_tab);
	ERR_FAIL_COND(!child);
	child->set_meta("_tab_hidden", p_hidden);
	update();
	for (int i = 0; i < get_tab_count(); i++) {
		int try_tab = (p_tab + 1 + i) % get_tab_count();
		if (get_tab_disabled(try_tab) || get_tab_hidden(try_tab)) {
			continue;
		}

		set_current_tab(try_tab);
		return;
	}

	//assumed no other tab can be switched to, just hide
	child->hide();
}

bool TabContainer::get_tab_hidden(int p_tab) const {
	Control *child = get_tab_control(p_tab);
	ERR_FAIL_COND_V(!child, false);
	if (child->has_meta("_tab_hidden")) {
		return child->get_meta("_tab_hidden");
	} else {
		return false;
	}
}

void TabContainer::get_translatable_strings(List<String> *p_strings) const {
	Vector<Control *> tabs = _get_tabs();
	for (int i = 0; i < tabs.size(); i++) {
		Control *c = tabs[i];

		if (!c->has_meta("_tab_name")) {
			continue;
		}

		String name = c->get_meta("_tab_name");

		if (!name.is_empty()) {
			p_strings->push_back(name);
		}
	}
}

Size2 TabContainer::get_minimum_size() const {
	Size2 ms;

	Vector<Control *> tabs = _get_tabs();
	for (int i = 0; i < tabs.size(); i++) {
		Control *c = tabs[i];

		if (!c->is_visible_in_tree() && !use_hidden_tabs_for_min_size) {
			continue;
		}

		Size2 cms = c->get_combined_minimum_size();
		ms.x = MAX(ms.x, cms.x);
		ms.y = MAX(ms.y, cms.y);
	}

	Ref<StyleBox> tab_unselected = get_theme_stylebox(SNAME("tab_unselected"));
	Ref<StyleBox> tab_selected = get_theme_stylebox(SNAME("tab_selected"));
	Ref<StyleBox> tab_disabled = get_theme_stylebox(SNAME("tab_disabled"));

	if (tabs_visible) {
		ms.y += MAX(MAX(tab_unselected->get_minimum_size().y, tab_selected->get_minimum_size().y), tab_disabled->get_minimum_size().y);
		ms.y += _get_top_margin();
	}

	Ref<StyleBox> sb = get_theme_stylebox(SNAME("panel"));
	ms += sb->get_minimum_size();

	return ms;
}

void TabContainer::set_popup(Node *p_popup) {
	ERR_FAIL_NULL(p_popup);
	Popup *popup = Object::cast_to<Popup>(p_popup);
	popup_obj_id = popup ? popup->get_instance_id() : ObjectID();
	update();
}

Popup *TabContainer::get_popup() const {
	if (popup_obj_id.is_valid()) {
		Popup *popup = Object::cast_to<Popup>(ObjectDB::get_instance(popup_obj_id));
		if (popup) {
			return popup;
		} else {
#ifdef DEBUG_ENABLED
			ERR_PRINT("Popup assigned to TabContainer is gone!");
#endif
			popup_obj_id = ObjectID();
		}
	}
	return nullptr;
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

void TabContainer::set_use_hidden_tabs_for_min_size(bool p_use_hidden_tabs) {
	use_hidden_tabs_for_min_size = p_use_hidden_tabs;
}

bool TabContainer::get_use_hidden_tabs_for_min_size() const {
	return use_hidden_tabs_for_min_size;
}

void TabContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_tab_count"), &TabContainer::get_tab_count);
	ClassDB::bind_method(D_METHOD("set_current_tab", "tab_idx"), &TabContainer::set_current_tab);
	ClassDB::bind_method(D_METHOD("get_current_tab"), &TabContainer::get_current_tab);
	ClassDB::bind_method(D_METHOD("get_previous_tab"), &TabContainer::get_previous_tab);
	ClassDB::bind_method(D_METHOD("get_current_tab_control"), &TabContainer::get_current_tab_control);
	ClassDB::bind_method(D_METHOD("get_tab_control", "tab_idx"), &TabContainer::get_tab_control);
	ClassDB::bind_method(D_METHOD("set_tab_alignment", "alignment"), &TabContainer::set_tab_alignment);
	ClassDB::bind_method(D_METHOD("get_tab_alignment"), &TabContainer::get_tab_alignment);
	ClassDB::bind_method(D_METHOD("set_tabs_visible", "visible"), &TabContainer::set_tabs_visible);
	ClassDB::bind_method(D_METHOD("are_tabs_visible"), &TabContainer::are_tabs_visible);
	ClassDB::bind_method(D_METHOD("set_all_tabs_in_front", "is_front"), &TabContainer::set_all_tabs_in_front);
	ClassDB::bind_method(D_METHOD("is_all_tabs_in_front"), &TabContainer::is_all_tabs_in_front);
	ClassDB::bind_method(D_METHOD("set_tab_title", "tab_idx", "title"), &TabContainer::set_tab_title);
	ClassDB::bind_method(D_METHOD("get_tab_title", "tab_idx"), &TabContainer::get_tab_title);
	ClassDB::bind_method(D_METHOD("set_tab_icon", "tab_idx", "icon"), &TabContainer::set_tab_icon);
	ClassDB::bind_method(D_METHOD("get_tab_icon", "tab_idx"), &TabContainer::get_tab_icon);
	ClassDB::bind_method(D_METHOD("set_tab_disabled", "tab_idx", "disabled"), &TabContainer::set_tab_disabled);
	ClassDB::bind_method(D_METHOD("get_tab_disabled", "tab_idx"), &TabContainer::get_tab_disabled);
	ClassDB::bind_method(D_METHOD("set_tab_hidden", "tab_idx", "hidden"), &TabContainer::set_tab_hidden);
	ClassDB::bind_method(D_METHOD("get_tab_hidden", "tab_idx"), &TabContainer::get_tab_hidden);
	ClassDB::bind_method(D_METHOD("get_tab_idx_at_point", "point"), &TabContainer::get_tab_idx_at_point);
	ClassDB::bind_method(D_METHOD("set_popup", "popup"), &TabContainer::set_popup);
	ClassDB::bind_method(D_METHOD("get_popup"), &TabContainer::get_popup);
	ClassDB::bind_method(D_METHOD("set_drag_to_rearrange_enabled", "enabled"), &TabContainer::set_drag_to_rearrange_enabled);
	ClassDB::bind_method(D_METHOD("get_drag_to_rearrange_enabled"), &TabContainer::get_drag_to_rearrange_enabled);
	ClassDB::bind_method(D_METHOD("set_tabs_rearrange_group", "group_id"), &TabContainer::set_tabs_rearrange_group);
	ClassDB::bind_method(D_METHOD("get_tabs_rearrange_group"), &TabContainer::get_tabs_rearrange_group);

	ClassDB::bind_method(D_METHOD("set_use_hidden_tabs_for_min_size", "enabled"), &TabContainer::set_use_hidden_tabs_for_min_size);
	ClassDB::bind_method(D_METHOD("get_use_hidden_tabs_for_min_size"), &TabContainer::get_use_hidden_tabs_for_min_size);

	ClassDB::bind_method(D_METHOD("_on_theme_changed"), &TabContainer::_on_theme_changed);
	ClassDB::bind_method(D_METHOD("_update_current_tab"), &TabContainer::_update_current_tab);

	ADD_SIGNAL(MethodInfo("tab_changed", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_selected", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("pre_popup_pressed"));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "tab_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_tab_alignment", "get_tab_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_tab", PROPERTY_HINT_RANGE, "-1,4096,1", PROPERTY_USAGE_EDITOR), "set_current_tab", "get_current_tab");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "tabs_visible"), "set_tabs_visible", "are_tabs_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "all_tabs_in_front"), "set_all_tabs_in_front", "is_all_tabs_in_front");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "drag_to_rearrange_enabled"), "set_drag_to_rearrange_enabled", "get_drag_to_rearrange_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_hidden_tabs_for_min_size"), "set_use_hidden_tabs_for_min_size", "get_use_hidden_tabs_for_min_size");

	BIND_ENUM_CONSTANT(ALIGNMENT_LEFT);
	BIND_ENUM_CONSTANT(ALIGNMENT_CENTER);
	BIND_ENUM_CONSTANT(ALIGNMENT_RIGHT);
}

TabContainer::TabContainer() {
	connect("mouse_exited", callable_mp(this, &TabContainer::_on_mouse_exited));
}
