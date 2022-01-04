/*************************************************************************/
/*  tab_bar.cpp                                                          */
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

#include "tab_bar.h"

#include "core/object/message_queue.h"
#include "core/string/translation.h"

#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"

Size2 TabBar::get_minimum_size() const {
	Ref<StyleBox> tab_unselected = get_theme_stylebox(SNAME("tab_unselected"));
	Ref<StyleBox> tab_selected = get_theme_stylebox(SNAME("tab_selected"));
	Ref<StyleBox> tab_disabled = get_theme_stylebox(SNAME("tab_disabled"));

	int y_margin = MAX(MAX(tab_unselected->get_minimum_size().height, tab_selected->get_minimum_size().height), tab_disabled->get_minimum_size().height);

	Size2 ms(0, 0);

	for (int i = 0; i < tabs.size(); i++) {
		Ref<Texture2D> tex = tabs[i].icon;
		if (tex.is_valid()) {
			ms.height = MAX(ms.height, tex->get_size().height);
			if (!tabs[i].text.is_empty()) {
				ms.width += get_theme_constant(SNAME("hseparation"));
			}
		}

		ms.width += Math::ceil(tabs[i].text_buf->get_size().x);
		ms.height = MAX(ms.height, tabs[i].text_buf->get_size().y + y_margin);

		if (tabs[i].disabled) {
			ms.width += tab_disabled->get_minimum_size().width;
		} else if (current == i) {
			ms.width += tab_selected->get_minimum_size().width;
		} else {
			ms.width += tab_unselected->get_minimum_size().width;
		}

		if (tabs[i].right_button.is_valid()) {
			Ref<Texture2D> rb = tabs[i].right_button;
			Size2 bms = rb->get_size();
			bms.width += get_theme_constant(SNAME("hseparation"));
			ms.width += bms.width;
			ms.height = MAX(bms.height + tab_unselected->get_minimum_size().height, ms.height);
		}

		if (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && i == current)) {
			Ref<Texture2D> cb = get_theme_icon(SNAME("close"));
			Size2 bms = cb->get_size();
			bms.width += get_theme_constant(SNAME("hseparation"));
			ms.width += bms.width;
			ms.height = MAX(bms.height + tab_unselected->get_minimum_size().height, ms.height);
		}
	}

	if (clip_tabs) {
		ms.width = 0;
	}

	return ms;
}

void TabBar::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		Point2 pos = mm->get_position();

		if (buttons_visible) {
			Ref<Texture2D> incr = get_theme_icon(SNAME("increment"));
			Ref<Texture2D> decr = get_theme_icon(SNAME("decrement"));

			if (is_layout_rtl()) {
				if (pos.x < decr->get_width()) {
					if (highlight_arrow != 1) {
						highlight_arrow = 1;
						update();
					}
				} else if (pos.x < incr->get_width() + decr->get_width()) {
					if (highlight_arrow != 0) {
						highlight_arrow = 0;
						update();
					}
				} else if (highlight_arrow != -1) {
					highlight_arrow = -1;
					update();
				}
			} else {
				int limit_minus_buttons = get_size().width - incr->get_width() - decr->get_width();
				if (pos.x > limit_minus_buttons + decr->get_width()) {
					if (highlight_arrow != 1) {
						highlight_arrow = 1;
						update();
					}
				} else if (pos.x > limit_minus_buttons) {
					if (highlight_arrow != 0) {
						highlight_arrow = 0;
						update();
					}
				} else if (highlight_arrow != -1) {
					highlight_arrow = -1;
					update();
				}
			}
		}

		_update_hover();
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->is_pressed() && mb->get_button_index() == MouseButton::WHEEL_UP && !mb->is_command_pressed()) {
			if (scrolling_enabled && buttons_visible) {
				if (offset > 0) {
					offset--;
					update();
				}
			}
		}

		if (mb->is_pressed() && mb->get_button_index() == MouseButton::WHEEL_DOWN && !mb->is_command_pressed()) {
			if (scrolling_enabled && buttons_visible) {
				if (missing_right) {
					offset++;
					_ensure_no_over_offset(); // Avoid overreaching when scrolling fast.
					update();
				}
			}
		}

		if (rb_pressing && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
			if (rb_hover != -1) {
				// Right mouse button clicked.
				emit_signal(SNAME("tab_rmb_clicked"), rb_hover);
			}

			rb_pressing = false;
			update();
		}

		if (cb_pressing && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
			if (cb_hover != -1) {
				// Close button pressed.
				emit_signal(SNAME("tab_close_pressed"), cb_hover);
			}

			cb_pressing = false;
			update();
		}

		if (mb->is_pressed() && (mb->get_button_index() == MouseButton::LEFT || (select_with_rmb && mb->get_button_index() == MouseButton::RIGHT))) {
			// Clicks.
			Point2 pos = mb->get_position();

			if (buttons_visible) {
				Ref<Texture2D> incr = get_theme_icon(SNAME("increment"));
				Ref<Texture2D> decr = get_theme_icon(SNAME("decrement"));

				if (is_layout_rtl()) {
					if (pos.x < decr->get_width()) {
						if (missing_right) {
							offset++;
							update();
						}
						return;
					} else if (pos.x < incr->get_width() + decr->get_width()) {
						if (offset > 0) {
							offset--;
							update();
						}
						return;
					}
				} else {
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
			}

			if (tabs.is_empty()) {
				// Return early if there are no actual tabs to handle input for.
				return;
			}

			int found = -1;
			for (int i = offset; i <= max_drawn_tab; i++) {
				if (tabs[i].rb_rect.has_point(pos)) {
					rb_pressing = true;
					update();
					return;
				}

				if (tabs[i].cb_rect.has_point(pos) && (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && i == current))) {
					cb_pressing = true;
					update();
					return;
				}

				if (pos.x >= get_tab_rect(i).position.x && pos.x < get_tab_rect(i).position.x + tabs[i].size_cache) {
					if (!tabs[i].disabled) {
						found = i;
					}
					break;
				}
			}

			if (found != -1) {
				set_current_tab(found);
				emit_signal(SNAME("tab_clicked"), found);
			}
		}
	}
}

void TabBar::_shape(int p_tab) {
	Ref<Font> font = get_theme_font(SNAME("font"));
	int font_size = get_theme_font_size(SNAME("font_size"));

	tabs.write[p_tab].xl_text = atr(tabs[p_tab].text);
	tabs.write[p_tab].text_buf->clear();
	tabs.write[p_tab].text_buf->set_width(-1);
	if (tabs[p_tab].text_direction == Control::TEXT_DIRECTION_INHERITED) {
		tabs.write[p_tab].text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		tabs.write[p_tab].text_buf->set_direction((TextServer::Direction)tabs[p_tab].text_direction);
	}

	tabs.write[p_tab].text_buf->add_string(tabs.write[p_tab].xl_text, font, font_size, tabs[p_tab].opentype_features, !tabs[p_tab].language.is_empty() ? tabs[p_tab].language : TranslationServer::get_singleton()->get_tool_locale());
}

void TabBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			_update_cache();
			update();
		} break;
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED: {
			for (int i = 0; i < tabs.size(); ++i) {
				_shape(i);
			}
			_update_cache();
			update_minimum_size();
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

			Ref<StyleBox> tab_unselected = get_theme_stylebox(SNAME("tab_unselected"));
			Ref<StyleBox> tab_selected = get_theme_stylebox(SNAME("tab_selected"));
			Ref<StyleBox> tab_disabled = get_theme_stylebox(SNAME("tab_disabled"));
			Color font_selected_color = get_theme_color(SNAME("font_selected_color"));
			Color font_unselected_color = get_theme_color(SNAME("font_unselected_color"));
			Color font_disabled_color = get_theme_color(SNAME("font_disabled_color"));
			Ref<Texture2D> close = get_theme_icon(SNAME("close"));
			Color font_outline_color = get_theme_color(SNAME("font_outline_color"));
			int outline_size = get_theme_constant(SNAME("outline_size"));

			Vector2 size = get_size();
			bool rtl = is_layout_rtl();

			int h = get_size().height;
			int w = 0;
			int mw = 0;

			for (int i = 0; i < tabs.size(); i++) {
				tabs.write[i].ofs_cache = mw;
				mw += get_tab_width(i);
			}

			if (tab_alignment == ALIGNMENT_CENTER) {
				w = (get_size().width - mw) / 2;
			} else if (tab_alignment == ALIGNMENT_RIGHT) {
				w = get_size().width - mw;
			}

			if (w < 0) {
				w = 0;
			}

			Ref<Texture2D> incr = get_theme_icon(SNAME("increment"));
			Ref<Texture2D> decr = get_theme_icon(SNAME("decrement"));
			Ref<Texture2D> incr_hl = get_theme_icon(SNAME("increment_highlight"));
			Ref<Texture2D> decr_hl = get_theme_icon(SNAME("decrement_highlight"));

			int limit = get_size().width;
			int limit_minus_buttons = get_size().width - incr->get_width() - decr->get_width();

			missing_right = false;

			for (int i = offset; i < tabs.size(); i++) {
				tabs.write[i].ofs_cache = w;

				int lsize = tabs[i].size_cache;

				Ref<StyleBox> sb;
				Color col;

				if (tabs[i].disabled) {
					sb = tab_disabled;
					col = font_disabled_color;
				} else if (i == current) {
					sb = tab_selected;
					col = font_selected_color;
				} else {
					sb = tab_unselected;
					col = font_unselected_color;
				}

				int new_width = w + lsize;
				if (new_width > limit || (i < tabs.size() - 1 && new_width > limit_minus_buttons)) { // For the last tab, we accept if the tab covers the buttons.
					max_drawn_tab = i - 1;
					missing_right = true;
					break;
				} else {
					max_drawn_tab = i;
				}

				Rect2 sb_rect;
				if (rtl) {
					sb_rect = Rect2(size.width - w - tabs[i].size_cache, 0, tabs[i].size_cache, h);
				} else {
					sb_rect = Rect2(w, 0, tabs[i].size_cache, h);
				}
				sb->draw(ci, sb_rect);

				w += sb->get_margin(SIDE_LEFT);

				Size2i sb_ms = sb->get_minimum_size();
				Ref<Texture2D> icon = tabs[i].icon;
				if (icon.is_valid()) {
					if (rtl) {
						icon->draw(ci, Point2i(size.width - w - icon->get_width(), sb->get_margin(SIDE_TOP) + ((sb_rect.size.y - sb_ms.y) - icon->get_height()) / 2));
					} else {
						icon->draw(ci, Point2i(w, sb->get_margin(SIDE_TOP) + ((sb_rect.size.y - sb_ms.y) - icon->get_height()) / 2));
					}
					if (!tabs[i].text.is_empty()) {
						w += icon->get_width() + get_theme_constant(SNAME("hseparation"));
					}
				}

				if (rtl) {
					Vector2 text_pos = Point2i(size.width - w - tabs[i].text_buf->get_size().x, sb->get_margin(SIDE_TOP) + ((sb_rect.size.y - sb_ms.y) - tabs[i].text_buf->get_size().y) / 2);
					if (outline_size > 0 && font_outline_color.a > 0) {
						tabs[i].text_buf->draw_outline(ci, text_pos, outline_size, font_outline_color);
					}
					tabs[i].text_buf->draw(ci, text_pos, col);
				} else {
					Vector2 text_pos = Point2i(w, sb->get_margin(SIDE_TOP) + ((sb_rect.size.y - sb_ms.y) - tabs[i].text_buf->get_size().y) / 2);
					if (outline_size > 0 && font_outline_color.a > 0) {
						tabs[i].text_buf->draw_outline(ci, text_pos, outline_size, font_outline_color);
					}
					tabs[i].text_buf->draw(ci, text_pos, col);
				}

				w += tabs[i].size_text;

				if (tabs[i].right_button.is_valid()) {
					Ref<StyleBox> style = get_theme_stylebox(SNAME("close_bg_highlight"));
					Ref<Texture2D> rb = tabs[i].right_button;

					w += get_theme_constant(SNAME("hseparation"));

					Rect2 rb_rect;
					rb_rect.size = style->get_minimum_size() + rb->get_size();
					if (rtl) {
						rb_rect.position.x = size.width - w - rb_rect.size.x;
					} else {
						rb_rect.position.x = w;
					}
					rb_rect.position.y = sb->get_margin(SIDE_TOP) + ((sb_rect.size.y - sb_ms.y) - (rb_rect.size.y)) / 2;

					if (rb_hover == i) {
						if (rb_pressing) {
							get_theme_stylebox(SNAME("button_pressed"))->draw(ci, rb_rect);
						} else {
							style->draw(ci, rb_rect);
						}
					}

					if (rtl) {
						rb->draw(ci, Point2i(size.width - w - rb_rect.size.x + style->get_margin(SIDE_LEFT), rb_rect.position.y + style->get_margin(SIDE_TOP)));
					} else {
						rb->draw(ci, Point2i(w + style->get_margin(SIDE_LEFT), rb_rect.position.y + style->get_margin(SIDE_TOP)));
					}
					w += rb->get_width();
					tabs.write[i].rb_rect = rb_rect;
				}

				if (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && i == current)) {
					Ref<StyleBox> style = get_theme_stylebox(SNAME("close_bg_highlight"));
					Ref<Texture2D> cb = close;

					w += get_theme_constant(SNAME("hseparation"));

					Rect2 cb_rect;
					cb_rect.size = style->get_minimum_size() + cb->get_size();
					if (rtl) {
						cb_rect.position.x = size.width - w - cb_rect.size.x;
					} else {
						cb_rect.position.x = w;
					}
					cb_rect.position.y = sb->get_margin(SIDE_TOP) + ((sb_rect.size.y - sb_ms.y) - (cb_rect.size.y)) / 2;

					if (!tabs[i].disabled && cb_hover == i) {
						if (cb_pressing) {
							get_theme_stylebox(SNAME("close_bg_pressed"))->draw(ci, cb_rect);
						} else {
							style->draw(ci, cb_rect);
						}
					}

					if (rtl) {
						cb->draw(ci, Point2i(size.width - w - cb_rect.size.x + style->get_margin(SIDE_LEFT), cb_rect.position.y + style->get_margin(SIDE_TOP)));
					} else {
						cb->draw(ci, Point2i(w + style->get_margin(SIDE_LEFT), cb_rect.position.y + style->get_margin(SIDE_TOP)));
					}
					w += cb->get_width();
					tabs.write[i].cb_rect = cb_rect;
				}

				w += sb->get_margin(SIDE_RIGHT);
			}

			if (offset > 0 || missing_right) {
				int vofs = (get_size().height - incr->get_size().height) / 2;

				if (rtl) {
					if (missing_right) {
						draw_texture(highlight_arrow == 1 ? decr_hl : decr, Point2(0, vofs));
					} else {
						draw_texture(decr, Point2(0, vofs), Color(1, 1, 1, 0.5));
					}

					if (offset > 0) {
						draw_texture(highlight_arrow == 0 ? incr_hl : incr, Point2(incr->get_size().width, vofs));
					} else {
						draw_texture(incr, Point2(incr->get_size().width, vofs), Color(1, 1, 1, 0.5));
					}
				} else {
					if (offset > 0) {
						draw_texture(highlight_arrow == 0 ? decr_hl : decr, Point2(limit_minus_buttons, vofs));
					} else {
						draw_texture(decr, Point2(limit_minus_buttons, vofs), Color(1, 1, 1, 0.5));
					}

					if (missing_right) {
						draw_texture(highlight_arrow == 1 ? incr_hl : incr, Point2(limit_minus_buttons + decr->get_size().width, vofs));
					} else {
						draw_texture(incr, Point2(limit_minus_buttons + decr->get_size().width, vofs), Color(1, 1, 1, 0.5));
					}
				}

				buttons_visible = true;
			} else {
				buttons_visible = false;
			}
		} break;
	}
}

int TabBar::get_tab_count() const {
	return tabs.size();
}

void TabBar::set_current_tab(int p_current) {
	if (current == p_current) {
		return;
	}
	ERR_FAIL_INDEX(p_current, get_tab_count());

	previous = current;
	current = p_current;

	_update_cache();
	update();

	emit_signal(SNAME("tab_changed"), p_current);
}

int TabBar::get_current_tab() const {
	return current;
}

int TabBar::get_previous_tab() const {
	return previous;
}

int TabBar::get_hovered_tab() const {
	return hover;
}

int TabBar::get_tab_offset() const {
	return offset;
}

bool TabBar::get_offset_buttons_visible() const {
	return buttons_visible;
}

void TabBar::set_tab_title(int p_tab, const String &p_title) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs.write[p_tab].text = p_title;
	_shape(p_tab);
	update();
	update_minimum_size();
}

String TabBar::get_tab_title(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), "");
	return tabs[p_tab].text;
}

void TabBar::set_tab_text_direction(int p_tab, Control::TextDirection p_text_direction) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (tabs[p_tab].text_direction != p_text_direction) {
		tabs.write[p_tab].text_direction = p_text_direction;
		_shape(p_tab);
		update();
	}
}

Control::TextDirection TabBar::get_tab_text_direction(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Control::TEXT_DIRECTION_INHERITED);
	return tabs[p_tab].text_direction;
}

void TabBar::clear_tab_opentype_features(int p_tab) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs.write[p_tab].opentype_features.clear();
	_shape(p_tab);
	update();
}

void TabBar::set_tab_opentype_feature(int p_tab, const String &p_name, int p_value) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	int32_t tag = TS->name_to_tag(p_name);
	if (!tabs[p_tab].opentype_features.has(tag) || (int)tabs[p_tab].opentype_features[tag] != p_value) {
		tabs.write[p_tab].opentype_features[tag] = p_value;
		_shape(p_tab);
		update();
	}
}

int TabBar::get_tab_opentype_feature(int p_tab, const String &p_name) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), -1);
	int32_t tag = TS->name_to_tag(p_name);
	if (!tabs[p_tab].opentype_features.has(tag)) {
		return -1;
	}
	return tabs[p_tab].opentype_features[tag];
}

void TabBar::set_tab_language(int p_tab, const String &p_language) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	if (tabs[p_tab].language != p_language) {
		tabs.write[p_tab].language = p_language;
		_shape(p_tab);
		update();
	}
}

String TabBar::get_tab_language(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), "");
	return tabs[p_tab].language;
}

void TabBar::set_tab_icon(int p_tab, const Ref<Texture2D> &p_icon) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs.write[p_tab].icon = p_icon;
	update();
	update_minimum_size();
}

Ref<Texture2D> TabBar::get_tab_icon(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Ref<Texture2D>());
	return tabs[p_tab].icon;
}

void TabBar::set_tab_disabled(int p_tab, bool p_disabled) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs.write[p_tab].disabled = p_disabled;
	update();
}

bool TabBar::get_tab_disabled(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), false);
	return tabs[p_tab].disabled;
}

void TabBar::set_tab_right_button(int p_tab, const Ref<Texture2D> &p_right_button) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs.write[p_tab].right_button = p_right_button;
	_update_cache();
	update();
	update_minimum_size();
}

Ref<Texture2D> TabBar::get_tab_right_button(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Ref<Texture2D>());
	return tabs[p_tab].right_button;
}

void TabBar::_update_hover() {
	if (!is_inside_tree()) {
		return;
	}

	const Point2 &pos = get_local_mouse_position();
	// test hovering to display right or close button.
	int hover_now = -1;
	int hover_buttons = -1;
	for (int i = offset; i < tabs.size(); i++) {
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
		emit_signal(SNAME("tab_hovered"), hover);
	}

	if (hover_buttons == -1) { // No hover.
		rb_hover = hover_buttons;
		cb_hover = hover_buttons;
	}
}

void TabBar::_update_cache() {
	Ref<StyleBox> tab_disabled = get_theme_stylebox(SNAME("tab_disabled"));
	Ref<StyleBox> tab_unselected = get_theme_stylebox(SNAME("tab_unselected"));
	Ref<StyleBox> tab_selected = get_theme_stylebox(SNAME("tab_selected"));
	Ref<Texture2D> incr = get_theme_icon(SNAME("increment"));
	Ref<Texture2D> decr = get_theme_icon(SNAME("decrement"));
	int limit_minus_buttons = get_size().width - incr->get_width() - decr->get_width();

	int w = 0;
	int mw = 0;
	int size_fixed = 0;
	int count_resize = 0;
	for (int i = 0; i < tabs.size(); i++) {
		tabs.write[i].ofs_cache = mw;
		tabs.write[i].size_cache = get_tab_width(i);
		tabs.write[i].size_text = Math::ceil(tabs[i].text_buf->get_size().x);
		tabs.write[i].text_buf->set_width(-1);
		mw += tabs[i].size_cache;
		if (tabs[i].size_cache <= min_width || i == current) {
			size_fixed += tabs[i].size_cache;
		} else {
			count_resize++;
		}
	}
	int m_width = min_width;
	if (count_resize > 0) {
		m_width = MAX((limit_minus_buttons - size_fixed) / count_resize, min_width);
	}
	for (int i = offset; i < tabs.size(); i++) {
		Ref<StyleBox> sb;
		if (tabs[i].disabled) {
			sb = tab_disabled;
		} else if (i == current) {
			sb = tab_selected;
		} else {
			sb = tab_unselected;
		}
		int lsize = tabs[i].size_cache;
		int slen = tabs[i].size_text;
		if (min_width > 0 && mw > limit_minus_buttons && i != current) {
			if (lsize > m_width) {
				slen = m_width - (sb->get_margin(SIDE_LEFT) + sb->get_margin(SIDE_RIGHT));
				if (tabs[i].icon.is_valid()) {
					slen -= tabs[i].icon->get_width();
					slen -= get_theme_constant(SNAME("hseparation"));
				}
				if (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && i == current)) {
					Ref<Texture2D> cb = get_theme_icon(SNAME("close"));
					slen -= cb->get_width();
					slen -= get_theme_constant(SNAME("hseparation"));
				}
				slen = MAX(slen, 1);
				lsize = m_width;
			}
		}
		tabs.write[i].ofs_cache = w;
		tabs.write[i].size_cache = lsize;
		tabs.write[i].size_text = slen;
		tabs.write[i].text_buf->set_width(slen);
		w += lsize;
	}
}

void TabBar::_on_mouse_exited() {
	rb_hover = -1;
	cb_hover = -1;
	hover = -1;
	highlight_arrow = -1;
	update();
}

void TabBar::add_tab(const String &p_str, const Ref<Texture2D> &p_icon) {
	Tab t;
	t.text = p_str;
	t.xl_text = atr(p_str);
	t.text_buf.instantiate();
	t.text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	t.text_buf->add_string(t.xl_text, get_theme_font(SNAME("font")), get_theme_font_size(SNAME("font_size")), Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
	t.icon = p_icon;
	t.disabled = false;
	t.ofs_cache = 0;
	t.size_cache = 0;

	tabs.push_back(t);
	_update_cache();
	call_deferred(SNAME("_update_hover"));
	update();
	update_minimum_size();
}

void TabBar::clear_tabs() {
	tabs.clear();
	current = 0;
	previous = 0;
	call_deferred(SNAME("_update_hover"));
	update();
}

void TabBar::remove_tab(int p_idx) {
	ERR_FAIL_INDEX(p_idx, tabs.size());
	tabs.remove_at(p_idx);
	if (current >= p_idx) {
		current--;
	}
	_update_cache();
	call_deferred(SNAME("_update_hover"));
	update();
	update_minimum_size();

	if (current < 0) {
		current = 0;
		previous = 0;
	}
	if (current >= tabs.size()) {
		current = tabs.size() - 1;
	}

	_ensure_no_over_offset();
}

Variant TabBar::get_drag_data(const Point2 &p_point) {
	if (!drag_to_rearrange_enabled) {
		return Variant();
	}

	int tab_over = get_tab_idx_at_point(p_point);

	if (tab_over < 0) {
		return Variant();
	}

	HBoxContainer *drag_preview = memnew(HBoxContainer);

	if (!tabs[tab_over].icon.is_null()) {
		TextureRect *tf = memnew(TextureRect);
		tf->set_texture(tabs[tab_over].icon);
		drag_preview->add_child(tf);
	}
	Label *label = memnew(Label(tabs[tab_over].xl_text));
	drag_preview->add_child(label);
	if (!tabs[tab_over].right_button.is_null()) {
		TextureRect *tf = memnew(TextureRect);
		tf->set_texture(tabs[tab_over].right_button);
		drag_preview->add_child(tf);
	}
	set_drag_preview(drag_preview);

	Dictionary drag_data;
	drag_data["type"] = "tab_element";
	drag_data["tab_element"] = tab_over;
	drag_data["from_path"] = get_path();
	return drag_data;
}

bool TabBar::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	if (!drag_to_rearrange_enabled) {
		return false;
	}

	Dictionary d = p_data;
	if (!d.has("type")) {
		return false;
	}

	if (String(d["type"]) == "tab_element") {
		NodePath from_path = d["from_path"];
		NodePath to_path = get_path();
		if (from_path == to_path) {
			return true;
		} else if (get_tabs_rearrange_group() != -1) {
			// Drag and drop between other TabBars.
			Node *from_node = get_node(from_path);
			TabBar *from_tabs = Object::cast_to<TabBar>(from_node);
			if (from_tabs && from_tabs->get_tabs_rearrange_group() == get_tabs_rearrange_group()) {
				return true;
			}
		}
	}
	return false;
}

void TabBar::drop_data(const Point2 &p_point, const Variant &p_data) {
	if (!drag_to_rearrange_enabled) {
		return;
	}

	int hover_now = get_tab_idx_at_point(p_point);

	Dictionary d = p_data;
	if (!d.has("type")) {
		return;
	}

	if (String(d["type"]) == "tab_element") {
		int tab_from_id = d["tab_element"];
		NodePath from_path = d["from_path"];
		NodePath to_path = get_path();
		if (from_path == to_path) {
			if (hover_now < 0) {
				hover_now = get_tab_count() - 1;
			}
			move_tab(tab_from_id, hover_now);
			emit_signal(SNAME("active_tab_rearranged"), hover_now);
			set_current_tab(hover_now);
		} else if (get_tabs_rearrange_group() != -1) {
			// Drag and drop between Tabs.
			Node *from_node = get_node(from_path);
			TabBar *from_tabs = Object::cast_to<TabBar>(from_node);
			if (from_tabs && from_tabs->get_tabs_rearrange_group() == get_tabs_rearrange_group()) {
				if (tab_from_id >= from_tabs->get_tab_count()) {
					return;
				}
				Tab moving_tab = from_tabs->tabs[tab_from_id];
				if (hover_now < 0) {
					hover_now = get_tab_count();
				}
				tabs.insert(hover_now, moving_tab);
				from_tabs->remove_tab(tab_from_id);
				set_current_tab(hover_now);
				emit_signal(SNAME("tab_changed"), hover_now);
				_update_cache();
			}
		}
	}
	update();
}

int TabBar::get_tab_idx_at_point(const Point2 &p_point) const {
	int hover_now = -1;
	for (int i = offset; i <= max_drawn_tab; i++) {
		Rect2 rect = get_tab_rect(i);
		if (rect.has_point(p_point)) {
			hover_now = i;
		}
	}

	return hover_now;
}

void TabBar::set_tab_alignment(AlignmentMode p_alignment) {
	ERR_FAIL_INDEX(p_alignment, ALIGNMENT_MAX);
	tab_alignment = p_alignment;
	update();
}

TabBar::AlignmentMode TabBar::get_tab_alignment() const {
	return tab_alignment;
}

void TabBar::set_clip_tabs(bool p_clip_tabs) {
	if (clip_tabs == p_clip_tabs) {
		return;
	}
	clip_tabs = p_clip_tabs;
	update();
	update_minimum_size();
}

bool TabBar::get_clip_tabs() const {
	return clip_tabs;
}

void TabBar::move_tab(int from, int to) {
	if (from == to) {
		return;
	}

	ERR_FAIL_INDEX(from, tabs.size());
	ERR_FAIL_INDEX(to, tabs.size());

	Tab tab_from = tabs[from];
	tabs.remove_at(from);
	tabs.insert(to, tab_from);

	_update_cache();
	update();
}

int TabBar::get_tab_width(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, tabs.size(), 0);

	Ref<StyleBox> tab_unselected = get_theme_stylebox(SNAME("tab_unselected"));
	Ref<StyleBox> tab_selected = get_theme_stylebox(SNAME("tab_selected"));
	Ref<StyleBox> tab_disabled = get_theme_stylebox(SNAME("tab_disabled"));

	int x = 0;

	Ref<Texture2D> tex = tabs[p_idx].icon;
	if (tex.is_valid()) {
		x += tex->get_width();
		if (!tabs[p_idx].text.is_empty()) {
			x += get_theme_constant(SNAME("hseparation"));
		}
	}

	x += Math::ceil(tabs[p_idx].text_buf->get_size().x);

	if (tabs[p_idx].disabled) {
		x += tab_disabled->get_minimum_size().width;
	} else if (current == p_idx) {
		x += tab_selected->get_minimum_size().width;
	} else {
		x += tab_unselected->get_minimum_size().width;
	}

	if (tabs[p_idx].right_button.is_valid()) {
		Ref<Texture2D> rb = tabs[p_idx].right_button;
		x += rb->get_width();
		x += get_theme_constant(SNAME("hseparation"));
	}

	if (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && p_idx == current)) {
		Ref<Texture2D> cb = get_theme_icon(SNAME("close"));
		x += cb->get_width();
		x += get_theme_constant(SNAME("hseparation"));
	}

	return x;
}

void TabBar::_ensure_no_over_offset() {
	if (!is_inside_tree()) {
		return;
	}

	Ref<Texture2D> incr = get_theme_icon(SNAME("increment"));
	Ref<Texture2D> decr = get_theme_icon(SNAME("decrement"));

	int limit = get_size().width;
	int limit_minus_buttons = get_size().width - incr->get_width() - decr->get_width();

	while (offset > 0) {
		int total_w = 0;
		for (int i = offset - 1; i < tabs.size(); i++) {
			total_w += tabs[i].size_cache;
		}

		if ((buttons_visible && total_w < limit_minus_buttons) || total_w < limit) { // For the last tab, we accept if the tab covers the buttons.
			offset--;
			update();
		} else {
			break;
		}
	}
}

void TabBar::ensure_tab_visible(int p_idx) {
	if (!is_inside_tree()) {
		return;
	}

	if (tabs.size() == 0) {
		return;
	}
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
	Ref<Texture2D> incr = get_theme_icon(SNAME("increment"));
	Ref<Texture2D> decr = get_theme_icon(SNAME("decrement"));
	int limit = get_size().width;
	int limit_minus_buttons = get_size().width - incr->get_width() - decr->get_width();

	for (int i = offset; i <= p_idx; i++) {
		int total_w = tabs[i].ofs_cache + tabs[i].size_cache;
		if (total_w > limit || (buttons_visible && total_w > limit_minus_buttons)) {
			offset++;
		}
	}

	if (prev_offset != offset) {
		update();
	}
}

Rect2 TabBar::get_tab_rect(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Rect2());
	if (is_layout_rtl()) {
		return Rect2(get_size().width - tabs[p_tab].ofs_cache - tabs[p_tab].size_cache, 0, tabs[p_tab].size_cache, get_size().height);
	} else {
		return Rect2(tabs[p_tab].ofs_cache, 0, tabs[p_tab].size_cache, get_size().height);
	}
}

void TabBar::set_tab_close_display_policy(CloseButtonDisplayPolicy p_policy) {
	ERR_FAIL_INDEX(p_policy, CLOSE_BUTTON_MAX);
	cb_displaypolicy = p_policy;
	update();
}

TabBar::CloseButtonDisplayPolicy TabBar::get_tab_close_display_policy() const {
	return cb_displaypolicy;
}

void TabBar::set_min_width(int p_width) {
	min_width = p_width;
}

void TabBar::set_scrolling_enabled(bool p_enabled) {
	scrolling_enabled = p_enabled;
}

bool TabBar::get_scrolling_enabled() const {
	return scrolling_enabled;
}

void TabBar::set_drag_to_rearrange_enabled(bool p_enabled) {
	drag_to_rearrange_enabled = p_enabled;
}

bool TabBar::get_drag_to_rearrange_enabled() const {
	return drag_to_rearrange_enabled;
}

void TabBar::set_tabs_rearrange_group(int p_group_id) {
	tabs_rearrange_group = p_group_id;
}

int TabBar::get_tabs_rearrange_group() const {
	return tabs_rearrange_group;
}

void TabBar::set_select_with_rmb(bool p_enabled) {
	select_with_rmb = p_enabled;
}

bool TabBar::get_select_with_rmb() const {
	return select_with_rmb;
}

void TabBar::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_hover"), &TabBar::_update_hover);
	ClassDB::bind_method(D_METHOD("get_tab_count"), &TabBar::get_tab_count);
	ClassDB::bind_method(D_METHOD("set_current_tab", "tab_idx"), &TabBar::set_current_tab);
	ClassDB::bind_method(D_METHOD("get_current_tab"), &TabBar::get_current_tab);
	ClassDB::bind_method(D_METHOD("get_previous_tab"), &TabBar::get_previous_tab);
	ClassDB::bind_method(D_METHOD("set_tab_title", "tab_idx", "title"), &TabBar::set_tab_title);
	ClassDB::bind_method(D_METHOD("get_tab_title", "tab_idx"), &TabBar::get_tab_title);
	ClassDB::bind_method(D_METHOD("set_tab_text_direction", "tab_idx", "direction"), &TabBar::set_tab_text_direction);
	ClassDB::bind_method(D_METHOD("get_tab_text_direction", "tab_idx"), &TabBar::get_tab_text_direction);
	ClassDB::bind_method(D_METHOD("set_tab_opentype_feature", "tab_idx", "tag", "values"), &TabBar::set_tab_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_tab_opentype_feature", "tab_idx", "tag"), &TabBar::get_tab_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_tab_opentype_features", "tab_idx"), &TabBar::clear_tab_opentype_features);
	ClassDB::bind_method(D_METHOD("set_tab_language", "tab_idx", "language"), &TabBar::set_tab_language);
	ClassDB::bind_method(D_METHOD("get_tab_language", "tab_idx"), &TabBar::get_tab_language);
	ClassDB::bind_method(D_METHOD("set_tab_icon", "tab_idx", "icon"), &TabBar::set_tab_icon);
	ClassDB::bind_method(D_METHOD("get_tab_icon", "tab_idx"), &TabBar::get_tab_icon);
	ClassDB::bind_method(D_METHOD("set_tab_disabled", "tab_idx", "disabled"), &TabBar::set_tab_disabled);
	ClassDB::bind_method(D_METHOD("get_tab_disabled", "tab_idx"), &TabBar::get_tab_disabled);
	ClassDB::bind_method(D_METHOD("remove_tab", "tab_idx"), &TabBar::remove_tab);
	ClassDB::bind_method(D_METHOD("add_tab", "title", "icon"), &TabBar::add_tab, DEFVAL(""), DEFVAL(Ref<Texture2D>()));
	ClassDB::bind_method(D_METHOD("set_tab_alignment", "alignment"), &TabBar::set_tab_alignment);
	ClassDB::bind_method(D_METHOD("get_tab_alignment"), &TabBar::get_tab_alignment);
	ClassDB::bind_method(D_METHOD("set_clip_tabs", "clip_tabs"), &TabBar::set_clip_tabs);
	ClassDB::bind_method(D_METHOD("get_clip_tabs"), &TabBar::get_clip_tabs);
	ClassDB::bind_method(D_METHOD("get_tab_offset"), &TabBar::get_tab_offset);
	ClassDB::bind_method(D_METHOD("get_offset_buttons_visible"), &TabBar::get_offset_buttons_visible);
	ClassDB::bind_method(D_METHOD("ensure_tab_visible", "idx"), &TabBar::ensure_tab_visible);
	ClassDB::bind_method(D_METHOD("get_tab_rect", "tab_idx"), &TabBar::get_tab_rect);
	ClassDB::bind_method(D_METHOD("move_tab", "from", "to"), &TabBar::move_tab);
	ClassDB::bind_method(D_METHOD("set_tab_close_display_policy", "policy"), &TabBar::set_tab_close_display_policy);
	ClassDB::bind_method(D_METHOD("get_tab_close_display_policy"), &TabBar::get_tab_close_display_policy);
	ClassDB::bind_method(D_METHOD("set_scrolling_enabled", "enabled"), &TabBar::set_scrolling_enabled);
	ClassDB::bind_method(D_METHOD("get_scrolling_enabled"), &TabBar::get_scrolling_enabled);
	ClassDB::bind_method(D_METHOD("set_drag_to_rearrange_enabled", "enabled"), &TabBar::set_drag_to_rearrange_enabled);
	ClassDB::bind_method(D_METHOD("get_drag_to_rearrange_enabled"), &TabBar::get_drag_to_rearrange_enabled);
	ClassDB::bind_method(D_METHOD("set_tabs_rearrange_group", "group_id"), &TabBar::set_tabs_rearrange_group);
	ClassDB::bind_method(D_METHOD("get_tabs_rearrange_group"), &TabBar::get_tabs_rearrange_group);

	ClassDB::bind_method(D_METHOD("set_select_with_rmb", "enabled"), &TabBar::set_select_with_rmb);
	ClassDB::bind_method(D_METHOD("get_select_with_rmb"), &TabBar::get_select_with_rmb);

	ADD_SIGNAL(MethodInfo("tab_changed", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_rmb_clicked", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_close_pressed", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_hovered", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("active_tab_rearranged", PropertyInfo(Variant::INT, "idx_to")));
	ADD_SIGNAL(MethodInfo("tab_clicked", PropertyInfo(Variant::INT, "tab")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_tab", PROPERTY_HINT_RANGE, "-1,4096,1", PROPERTY_USAGE_EDITOR), "set_current_tab", "get_current_tab");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tab_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_tab_alignment", "get_tab_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_tabs"), "set_clip_tabs", "get_clip_tabs");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tab_close_display_policy", PROPERTY_HINT_ENUM, "Show Never,Show Active Only,Show Always"), "set_tab_close_display_policy", "get_tab_close_display_policy");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scrolling_enabled"), "set_scrolling_enabled", "get_scrolling_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "drag_to_rearrange_enabled"), "set_drag_to_rearrange_enabled", "get_drag_to_rearrange_enabled");

	BIND_ENUM_CONSTANT(ALIGNMENT_LEFT);
	BIND_ENUM_CONSTANT(ALIGNMENT_CENTER);
	BIND_ENUM_CONSTANT(ALIGNMENT_RIGHT);
	BIND_ENUM_CONSTANT(ALIGNMENT_MAX);

	BIND_ENUM_CONSTANT(CLOSE_BUTTON_SHOW_NEVER);
	BIND_ENUM_CONSTANT(CLOSE_BUTTON_SHOW_ACTIVE_ONLY);
	BIND_ENUM_CONSTANT(CLOSE_BUTTON_SHOW_ALWAYS);
	BIND_ENUM_CONSTANT(CLOSE_BUTTON_MAX);
}

TabBar::TabBar() {
	connect("mouse_exited", callable_mp(this, &TabBar::_on_mouse_exited));
}
