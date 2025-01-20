/**************************************************************************/
/*  foldable_container.cpp                                                */
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

#include "foldable_container.h"

#include "scene/theme/theme_db.h"

Size2 FoldableContainer::get_minimum_size() const {
	Size2 title_ms = _get_title_panel_min_size();

	if (folded) {
		return title_ms;
	}
	Size2 ms;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c || !c->is_visible()) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}
		ms = ms.max(c->get_combined_minimum_size());
	}
	ms += theme_cache.panel_style->get_minimum_size();

	return Size2(MAX(ms.width, title_ms.width), ms.height + title_ms.height);
}

Vector<int> FoldableContainer::get_allowed_size_flags_horizontal() const {
	return Vector<int>();
}

Vector<int> FoldableContainer::get_allowed_size_flags_vertical() const {
	return Vector<int>();
}

void FoldableContainer::fold() {
	set_folded(true);
	emit_signal(SNAME("folding_changed"), folded);
}

void FoldableContainer::expand() {
	set_folded(false);
	emit_signal(SNAME("folding_changed"), folded);
}

void FoldableContainer::set_folded(bool p_folded) {
	if (folded != p_folded) {
		if (!changing_group && foldable_group.is_valid()) {
			if (!p_folded) {
				_update_group();
				foldable_group->emit_signal(SNAME("expanded"), this);
			} else if (!foldable_group->updating_group && foldable_group->get_expanded_container() == this && !foldable_group->is_allow_folding_all()) {
				return;
			}
		}
		folded = p_folded;

		for (Button &E : buttons) {
			if (E.auto_hide) {
				E.visible = !folded;
			}
		}

		update_minimum_size();
		queue_sort();
		queue_redraw();
	}
}

bool FoldableContainer::is_folded() const {
	return folded;
}

void FoldableContainer::set_expanded(bool p_expanded) {
	set_folded(!p_expanded);
}

bool FoldableContainer::is_expanded() const {
	return !folded;
}

void FoldableContainer::set_foldable_group(const Ref<FoldableGroup> &p_group) {
	if (foldable_group.is_valid()) {
		foldable_group->containers.erase(this);
	}

	foldable_group = p_group;

	if (foldable_group.is_valid()) {
		changing_group = true;
		if (folded && !foldable_group->get_expanded_container() && !foldable_group->is_allow_folding_all()) {
			set_folded(false);
		} else if (is_expanded() && foldable_group->get_expanded_container()) {
			set_folded(true);
		}
		foldable_group->containers.insert(this);
		changing_group = false;
	}

	queue_redraw();
}

Ref<FoldableGroup> FoldableContainer::get_foldable_group() const {
	return foldable_group;
}

void FoldableContainer::set_text(const String &p_text) {
	if (text != p_text) {
		text = p_text;
		_shape();
		update_minimum_size();
		queue_redraw();
	}
}

String FoldableContainer::get_text() const {
	return text;
}

void FoldableContainer::set_text_alignment(HorizontalAlignment p_alignment) {
	ERR_FAIL_INDEX((int)p_alignment, 3);
	text_alignment = p_alignment;

	if (_get_actual_alignment() != text_buf->get_horizontal_alignment()) {
		_shape();
		queue_redraw();
	}
}

HorizontalAlignment FoldableContainer::get_text_alignment() const {
	return text_alignment;
}

void FoldableContainer::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;

		_shape();
		update_minimum_size();
		queue_redraw();
	}
}

String FoldableContainer::get_language() const {
	return language;
}

void FoldableContainer::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < 0 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;

		_shape();
		queue_redraw();
	}
}

Control::TextDirection FoldableContainer::get_text_direction() const {
	return text_direction;
}

void FoldableContainer::set_text_overrun_behavior(TextServer::OverrunBehavior p_overrun_behavior) {
	if (overrun_behavior != p_overrun_behavior) {
		overrun_behavior = p_overrun_behavior;

		_shape();
		update_minimum_size();
		queue_redraw();
	}
}

TextServer::OverrunBehavior FoldableContainer::get_text_overrun_behavior() const {
	return overrun_behavior;
}

void FoldableContainer::set_title_position(FoldableContainer::TitlePosition p_title_position) {
	ERR_FAIL_INDEX(p_title_position, POSITION_MAX);
	if (title_position != p_title_position) {
		title_position = p_title_position;
		queue_redraw();
		queue_sort();
	}
}

FoldableContainer::TitlePosition FoldableContainer::get_title_position() const {
	return title_position;
}

void FoldableContainer::add_button(const Ref<Texture2D> &p_icon, int p_position, int p_id) {
	Button button = Button();
	if (p_icon.is_valid()) {
		button.icon = p_icon;
	}
	button.id = p_id == -1 ? buttons.size() - 1 : p_id;
	p_position = p_position < 0 ? MAX(buttons.size(), 0) : CLAMP(p_position, 0, buttons.size());

	buttons.insert(p_position, button);
	update_minimum_size();
	queue_redraw();
	notify_property_list_changed();
}

void FoldableContainer::remove_button(int p_index) {
	ERR_FAIL_INDEX(p_index, buttons.size());

	bool redraw = buttons[p_index].visible;
	buttons.remove_at(p_index);

	if (redraw) {
		update_minimum_size();
		queue_redraw();
	}
	notify_property_list_changed();
}

void FoldableContainer::set_button_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);

	if (buttons.size() != p_count) {
		buttons.resize(p_count);

		update_minimum_size();
		queue_redraw();
		notify_property_list_changed();
	}
}

int FoldableContainer::get_button_count() const {
	return buttons.size();
}

Rect2 FoldableContainer::get_button_rect(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, buttons.size(), Rect2());

	return buttons[p_index].rect;
}

void FoldableContainer::clear() {
	buttons.clear();
	_hovered = -1;

	update_minimum_size();
	queue_redraw();
	notify_property_list_changed();
}

void FoldableContainer::set_button_id(int p_index, int p_id) {
	ERR_FAIL_INDEX(p_index, buttons.size());

	buttons.write[p_index].id = p_id;
}

int FoldableContainer::get_button_id(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, buttons.size(), -1);

	return buttons[p_index].id;
}

int FoldableContainer::move_button(int p_from_index, int p_to_index) {
	int arr_size = buttons.size();
	ERR_FAIL_INDEX_V(p_from_index, arr_size, -1);
	ERR_FAIL_COND_V(arr_size < 2, -1);
	p_to_index = p_to_index == -1 ? arr_size - 1 : CLAMP(p_to_index, 0, arr_size - 1);
	ERR_FAIL_COND_V(p_from_index == p_to_index, -1);

	Button button = buttons[p_from_index];
	buttons.remove_at(p_from_index);
	arr_size--;
	p_to_index = CLAMP(p_to_index, 0, arr_size);

	int idx = buttons.insert(p_to_index, button) == OK ? p_to_index : -1;
	notify_property_list_changed();
	return idx;
}

int FoldableContainer::get_button_index(int p_id) const {
	for (int i = 0; i < buttons.size(); i++) {
		if (buttons[i].id == p_id) {
			return i;
		}
	}
	return -1;
}

void FoldableContainer::set_button_toggle_mode(int p_index, bool p_mode) {
	ERR_FAIL_INDEX(p_index, buttons.size());

	buttons.write[p_index].toggle_mode = p_mode;
	if (!p_mode && buttons[p_index].toggled_on) {
		buttons.write[p_index].toggled_on = false;
		queue_redraw();
	}
}

bool FoldableContainer::get_button_toggle_mode(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, buttons.size(), false);

	return buttons[p_index].toggle_mode;
}

void FoldableContainer::set_button_toggled(int p_index, bool p_toggled_on) {
	ERR_FAIL_INDEX(p_index, buttons.size());
	if (!buttons[p_index].toggle_mode) {
		return;
	}

	if (buttons[p_index].toggled_on != p_toggled_on) {
		buttons.write[p_index].toggled_on = p_toggled_on;
		queue_redraw();
	}
}

bool FoldableContainer::is_button_toggled(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, buttons.size(), false);

	return buttons[p_index].toggled_on;
}

void FoldableContainer::set_button_icon(int p_index, const Ref<Texture2D> &p_icon) {
	ERR_FAIL_INDEX(p_index, buttons.size());

	buttons.write[p_index].icon = p_icon;

	if (buttons[p_index].visible) {
		update_minimum_size();
		queue_redraw();
	}
}

Ref<Texture2D> FoldableContainer::get_button_icon(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, buttons.size(), Ref<Texture2D>());

	return buttons[p_index].icon;
}

void FoldableContainer::set_button_tooltip(int p_index, String p_tooltip) {
	ERR_FAIL_INDEX(p_index, buttons.size());

	buttons.write[p_index].tooltip = p_tooltip;
}

String FoldableContainer::get_button_tooltip(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, buttons.size(), "");

	return buttons[p_index].tooltip;
}

void FoldableContainer::set_button_disabled(int p_index, bool p_disabled) {
	ERR_FAIL_INDEX(p_index, buttons.size());

	if (buttons[p_index].disabled != p_disabled) {
		buttons.write[p_index].disabled = p_disabled;

		if (buttons[p_index].visible) {
			update_minimum_size();
			queue_redraw();
		}
	}
}

bool FoldableContainer::is_button_disabled(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, buttons.size(), false);

	return buttons[p_index].disabled;
}

void FoldableContainer::set_button_visible(int p_index, bool p_visible) {
	ERR_FAIL_INDEX(p_index, buttons.size());

	if (buttons[p_index].visible != p_visible) {
		if (buttons[p_index].auto_hide) {
			if ((p_visible && folded) || (!p_visible && !folded)) {
				return;
			}
		}

		buttons.write[p_index].visible = p_visible;

		update_minimum_size();
		queue_redraw();
	}
}

bool FoldableContainer::is_button_visible(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, buttons.size(), false);

	return buttons[p_index].visible;
}

void FoldableContainer::set_button_auto_hide(int p_index, bool p_auto_hide) {
	ERR_FAIL_INDEX(p_index, buttons.size());

	if (buttons[p_index].auto_hide != p_auto_hide) {
		buttons.write[p_index].auto_hide = p_auto_hide;

		if (p_auto_hide) {
			bool changed = false;

			if (folded && buttons[p_index].visible) {
				buttons.write[p_index].visible = false;
				changed = true;
			} else if (!folded && !buttons[p_index].visible) {
				buttons.write[p_index].visible = true;
				changed = true;
			}

			if (changed) {
				update_minimum_size();
				queue_redraw();
			}
		}
	}
}

bool FoldableContainer::is_button_auto_hide(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, buttons.size(), false);

	return buttons[p_index].auto_hide;
}

void FoldableContainer::set_button_metadata(int p_index, Variant p_metadata) {
	ERR_FAIL_INDEX(p_index, buttons.size());

	buttons.write[p_index].metadata = p_metadata;
}

Variant FoldableContainer::get_button_metadata(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, buttons.size(), Variant());

	return buttons[p_index].metadata;
}

int FoldableContainer::get_button_at_position(const Point2 &p_pos) const {
	for (int i = 0; i < buttons.size(); i++) {
		if (!buttons[i].visible) {
			continue;
		}
		if (buttons[i].rect.has_point(p_pos)) {
			return i;
		}
	}
	return -1;
}

void FoldableContainer::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseMotion> m = p_event;

	if (m.is_valid()) {
		int _last_hovered = _hovered;
		_hovered = -1;
		Rect2 title_rect = Rect2(0, (title_position == POSITION_TOP) ? 0 : get_size().height - title_panel_height, get_size().width, title_panel_height);
		if (title_rect.has_point(m->get_position())) {
			if (!is_hovering) {
				is_hovering = true;
				queue_redraw();
			}
			_hovered = get_button_at_position(m->get_position());
		} else if (is_hovering) {
			is_hovering = false;
			queue_redraw();
		}
		if (_last_hovered != _hovered) {
			queue_redraw();
		}
	}

	if (has_focus() && p_event->is_action_pressed("ui_accept", false, true)) {
		set_folded(!folded);
		emit_signal(SNAME("folding_changed"), folded);
		accept_event();
		return;
	}

	Ref<InputEventMouseButton> b = p_event;
	if (b.is_valid()) {
		int last_pressed = _pressed;
		_pressed = -1;
		Rect2 title_rect = Rect2(0, (title_position == POSITION_TOP) ? 0 : get_size().height - title_panel_height, get_size().width, title_panel_height);
		if (b->get_button_index() == MouseButton::LEFT && title_rect.has_point(b->get_position())) {
			int button = get_button_at_position(b->get_position());
			if (b->is_pressed()) {
				_pressed = button;
				if (_pressed == -1) {
					set_folded(!folded);
					emit_signal(SNAME("folding_changed"), folded);
				}
				accept_event();
			} else {
				if (button > -1 && !buttons[button].disabled && button == last_pressed) {
					if (buttons[last_pressed].toggle_mode) {
						bool toggled_on = buttons[last_pressed].toggled_on;
						buttons.write[last_pressed].toggled_on = !toggled_on;
						emit_signal(SNAME("button_toggled"), !toggled_on, last_pressed);
					} else {
						emit_signal(SNAME("button_pressed"), last_pressed);
					}
				}
				accept_event();
			}
		}
		if (last_pressed != _pressed) {
			queue_redraw();
		}
	}
}

String FoldableContainer::get_tooltip(const Point2 &p_pos) const {
	if (_hovered != -1) {
		return buttons[_hovered].tooltip;
	} else if (Rect2(0, (title_position == POSITION_TOP) ? 0 : get_size().height - title_panel_height, get_size().width, title_panel_height).has_point(p_pos)) {
		return Control::get_tooltip(p_pos);
	}
	return "";
}

void FoldableContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			Size2 size = get_size();
			title_panel_height = _get_title_panel_min_size().height;
			Ref<StyleBox> title_style = _get_title_style();

			Point2 title_ofs = (title_position == POSITION_TOP) ? Point2() : Point2(0, size.height - title_panel_height);
			Rect2 title_rect = Rect2(-title_ofs, Size2(size.width, title_panel_height));
			if (title_position == POSITION_BOTTOM) {
				draw_set_transform(Point2(0.0, title_style->get_draw_rect(title_rect).size.height), 0.0, Size2(1.0, -1.0));
			}
			title_style->draw(ci, title_rect);
			if (title_position == POSITION_BOTTOM) {
				draw_set_transform(Point2(), 0.0, Size2(1.0, 1.0));
			}

			Size2 title_ms = title_style->get_minimum_size();
			int title_width = size.width - title_ms.width;

			int h_separation = MAX(theme_cache.h_separation, 0);
			int title_style_ofs = (title_position == POSITION_TOP) ? title_style->get_margin(SIDE_TOP) : title_style->get_margin(SIDE_BOTTOM);
			Point2 title_pos;
			title_pos.y = MAX((title_panel_height - title_ms.height - text_buf->get_size().height) * 0.5, 0) + title_style_ofs;

			Ref<Texture2D> icon = _get_title_icon();

			title_width -= icon->get_width() + h_separation;
			Point2 icon_pos;
			icon_pos.x = title_style->get_margin(SIDE_LEFT);
			icon_pos.y = MAX((title_panel_height - title_ms.height - icon->get_height()) * 0.5, 0) + title_style_ofs;

			bool rtl = is_layout_rtl();
			if (rtl) {
				icon_pos.x = size.width - icon_pos.x - icon->get_width();
			} else {
				title_pos.x = title_style->get_margin(SIDE_LEFT) + icon->get_width() + h_separation;
			}
			icon->draw(ci, title_ofs + icon_pos);

			Size2 button_ms = theme_cache.button_normal_style->get_minimum_size();
			int offset = 0;
			for (int i = buttons.size() - 1; i > -1; i--) {
				Button button = buttons[i];
				if (!button.visible || button.icon.is_null()) {
					continue;
				}

				Ref<StyleBox> button_style;
				Color icon_color;
				if (button.disabled) {
					button_style = theme_cache.button_disabled_style;
					icon_color = theme_cache.button_icon_normal;
				} else if (i == _pressed || button.toggled_on) {
					button_style = theme_cache.button_pressed_style;
					icon_color = theme_cache.button_icon_pressed;
				} else if (i == _hovered) {
					button_style = theme_cache.button_hovered_style;
					icon_color = button.toggled_on ? theme_cache.button_icon_pressed : theme_cache.button_icon_hovered;
				} else {
					button_style = theme_cache.button_normal_style;
					icon_color = theme_cache.button_icon_normal;
				}

				Size2 icon_size = button.icon->get_size();
				Size2 button_size = icon_size + button_ms;
				Point2 button_pos;
				button_pos.x = rtl ? title_style->get_margin(SIDE_LEFT) + offset : size.width - button_size.width - title_style->get_margin(SIDE_RIGHT) - offset;
				button_pos.y = MAX((title_panel_height - title_ms.height - button_size.height) * 0.5, 0) + title_style_ofs;
				button_style->draw(ci, Rect2(title_ofs + button_pos, button_size));
				button.icon->draw(ci, title_ofs + button_pos + button_style->get_offset(), icon_color);
				buttons.write[i].rect = Rect2(title_ofs + button_pos, button_size);
				offset += icon_size.width + button_ms.width + h_separation;
			}

			title_width -= offset;
			if (rtl) {
				title_pos.x = offset + h_separation;
			}

			Color font_color = folded ? theme_cache.title_collapsed_font_color : theme_cache.title_font_color;
			if (is_hovering) {
				font_color = theme_cache.title_hovered_font_color;
			}
			text_buf->set_width(title_width);

			if (title_width > 0) {
				if (theme_cache.title_font_outline_size > 0 && theme_cache.title_font_outline_color.a > 0) {
					text_buf->draw_outline(ci, title_ofs + title_pos, theme_cache.title_font_outline_size, theme_cache.title_font_outline_color);
				}
				text_buf->draw(ci, title_ofs + title_pos, font_color);
			}

			if (is_expanded()) {
				Rect2 panel_rect = Rect2(Point2(0, (title_position == POSITION_TOP) ? title_panel_height : 0), Size2(size.width, size.height - title_panel_height));
				if (title_position == POSITION_BOTTOM) {
					draw_set_transform(Point2(0.0, title_style->get_draw_rect(panel_rect).size.height), 0.0, Size2(1.0, -1.0));
				}
				theme_cache.panel_style->draw(ci, panel_rect);
				if (title_position == POSITION_BOTTOM) {
					draw_set_transform(Point2(), 0.0, Size2(1.0, 1.0));
				}
			}

			if (has_focus()) {
				Rect2 focus_rect = folded ? Rect2(-title_ofs, Size2(size.width, title_panel_height)) : Rect2(Point2(), size);
				if (title_position == POSITION_BOTTOM) {
					draw_set_transform(Point2(0.0, title_style->get_draw_rect(focus_rect).size.height), 0.0, Size2(1.0, -1.0));
				}
				theme_cache.focus_style->draw(ci, focus_rect);
				if (title_position == POSITION_BOTTOM) {
					draw_set_transform(Point2(), 0.0, Size2(1.0, 1.0));
				}
			}
		} break;

		case NOTIFICATION_SORT_CHILDREN: {
			Point2 start_ofs;
			Point2 end_ofs;
			start_ofs.x = is_layout_rtl() ? theme_cache.panel_style->get_margin(SIDE_RIGHT) : theme_cache.panel_style->get_margin(SIDE_LEFT);
			end_ofs.x = is_layout_rtl() ? theme_cache.panel_style->get_margin(SIDE_LEFT) : theme_cache.panel_style->get_margin(SIDE_RIGHT);
			if (title_position == POSITION_TOP) {
				start_ofs.y = _get_title_panel_min_size().height + theme_cache.panel_style->get_margin(SIDE_TOP);
				end_ofs.y = theme_cache.panel_style->get_margin(SIDE_BOTTOM);
			} else {
				start_ofs.y = theme_cache.panel_style->get_margin(SIDE_BOTTOM);
				end_ofs.y = _get_title_panel_min_size().height + theme_cache.panel_style->get_margin(SIDE_TOP);
			}

			for (int i = 0; i < get_child_count(); i++) {
				Control *c = Object::cast_to<Control>(get_child(i));
				if (!c || c->is_set_as_top_level()) {
					continue;
				}
				c->set_visible(is_expanded());

				if (is_expanded()) {
					c->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
					c->set_offset(SIDE_LEFT, c->get_offset(SIDE_LEFT) + start_ofs.x);
					c->set_offset(SIDE_TOP, c->get_offset(SIDE_TOP) + start_ofs.y);
					c->set_offset(SIDE_RIGHT, c->get_offset(SIDE_RIGHT) - end_ofs.x);
					c->set_offset(SIDE_BOTTOM, c->get_offset(SIDE_BOTTOM) - end_ofs.y);
				}
			}
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			if (is_hovering || _hovered != -1) {
				is_hovering = false;
				_hovered = -1;
				queue_redraw();
			}
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			_shape();
			update_minimum_size();
			queue_redraw();
		} break;
	}
}

Size2 FoldableContainer::_get_title_panel_min_size() const {
	Ref<StyleBox> title_style = folded ? theme_cache.title_collapsed_style : theme_cache.title_style;
	Ref<Texture2D> icon = _get_title_icon();
	Size2 title_ms = title_style->get_minimum_size();
	Size2 s = title_ms;
	s.width += icon->get_width();

	if (text.length() > 0) {
		s.width += MAX(0, theme_cache.h_separation);
		Size2 text_size = text_buf->get_size();
		s.height += text_size.height;
		if (overrun_behavior == TextServer::OverrunBehavior::OVERRUN_NO_TRIMMING) {
			s.width += text_size.width;
		}
	}

	Size2 button_ms = theme_cache.button_normal_style->get_minimum_size();
	int icon_height = 0;
	for (Button button : buttons) {
		if (!button.visible) {
			continue;
		}
		s.width += theme_cache.h_separation + button_ms.width;

		if (button.icon.is_valid()) {
			Size2 icon_size = button.icon->get_size();
			s.width += icon_size.width;
			icon_height = MAX(icon_height, icon_size.height);
		}
	}

	if (icon_height > 0) {
		s.height = MAX(s.height, title_ms.height + button_ms.height + icon_height);
	}
	s.height = MAX(s.height, title_ms.height + icon->get_height());

	return s;
}

Ref<StyleBox> FoldableContainer::_get_title_style() const {
	if (is_hovering) {
		return folded ? theme_cache.title_collapsed_hover_style : theme_cache.title_hover_style;
	}
	return folded ? theme_cache.title_collapsed_style : theme_cache.title_style;
}

Ref<Texture2D> FoldableContainer::_get_title_icon() const {
	if (is_expanded()) {
		return (title_position == POSITION_TOP) ? theme_cache.expanded_arrow : theme_cache.expanded_arrow_mirrored;
	} else if (is_layout_rtl()) {
		return theme_cache.folded_arrow_mirrored;
	}
	return theme_cache.folded_arrow;
}

void FoldableContainer::_shape() {
	Ref<Font> font = theme_cache.title_font;
	int font_size = theme_cache.title_font_size;
	if (font.is_null() || font_size == 0) {
		return;
	}

	text_buf->clear();
	text_buf->set_width(-1);

	if (text_direction == TEXT_DIRECTION_INHERITED) {
		text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		text_buf->set_direction((TextServer::Direction)text_direction);
	}

	text_buf->set_horizontal_alignment(_get_actual_alignment());

	text_buf->set_text_overrun_behavior(overrun_behavior);

	text_buf->add_string(atr(text), font, font_size, language);
}

HorizontalAlignment FoldableContainer::_get_actual_alignment() const {
	if (is_layout_rtl()) {
		if (text_alignment == HORIZONTAL_ALIGNMENT_RIGHT) {
			return HORIZONTAL_ALIGNMENT_LEFT;
		} else if (text_alignment == HORIZONTAL_ALIGNMENT_LEFT) {
			return HORIZONTAL_ALIGNMENT_RIGHT;
		}
	}
	return text_alignment;
}

bool FoldableContainer::_set(const StringName &p_name, const Variant &p_value) {
	return property_helper.property_set_value(p_name, p_value);
}

void FoldableContainer::_update_group() {
	foldable_group->updating_group = true;
	for (FoldableContainer *E : foldable_group->containers) {
		if (E == this) {
			continue;
		}

		E->set_folded(true);
	}
	foldable_group->updating_group = false;
}

void FoldableContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("fold"), &FoldableContainer::fold);
	ClassDB::bind_method(D_METHOD("expand"), &FoldableContainer::expand);
	ClassDB::bind_method(D_METHOD("set_folded", "folded"), &FoldableContainer::set_folded);
	ClassDB::bind_method(D_METHOD("is_folded"), &FoldableContainer::is_folded);
	ClassDB::bind_method(D_METHOD("set_expanded", "expanded"), &FoldableContainer::set_expanded);
	ClassDB::bind_method(D_METHOD("is_expanded"), &FoldableContainer::is_expanded);
	ClassDB::bind_method(D_METHOD("set_foldable_group", "button_group"), &FoldableContainer::set_foldable_group);
	ClassDB::bind_method(D_METHOD("get_foldable_group"), &FoldableContainer::get_foldable_group);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &FoldableContainer::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &FoldableContainer::get_text);
	ClassDB::bind_method(D_METHOD("set_title_alignment", "alignment"), &FoldableContainer::set_text_alignment);
	ClassDB::bind_method(D_METHOD("get_title_alignment"), &FoldableContainer::get_text_alignment);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &FoldableContainer::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &FoldableContainer::get_language);
	ClassDB::bind_method(D_METHOD("set_text_direction", "text_direction"), &FoldableContainer::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &FoldableContainer::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_text_overrun_behavior", "overrun_behavior"), &FoldableContainer::set_text_overrun_behavior);
	ClassDB::bind_method(D_METHOD("get_text_overrun_behavior"), &FoldableContainer::get_text_overrun_behavior);
	ClassDB::bind_method(D_METHOD("set_title_position", "title_position"), &FoldableContainer::set_title_position);
	ClassDB::bind_method(D_METHOD("get_title_position"), &FoldableContainer::get_title_position);
	ClassDB::bind_method(D_METHOD("add_button", "icon", "position", "id"), &FoldableContainer::add_button, DEFVAL(Ref<Texture2D>()), DEFVAL(-1), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("remove_button", "index"), &FoldableContainer::remove_button);
	ClassDB::bind_method(D_METHOD("set_button_count", "count"), &FoldableContainer::set_button_count);
	ClassDB::bind_method(D_METHOD("get_button_count"), &FoldableContainer::get_button_count);
	ClassDB::bind_method(D_METHOD("get_button_rect", "index"), &FoldableContainer::get_button_rect);
	ClassDB::bind_method(D_METHOD("clear"), &FoldableContainer::clear);
	ClassDB::bind_method(D_METHOD("set_button_id", "index", "id"), &FoldableContainer::set_button_id);
	ClassDB::bind_method(D_METHOD("get_button_id", "index"), &FoldableContainer::get_button_id);
	ClassDB::bind_method(D_METHOD("move_button", "from", "to"), &FoldableContainer::move_button);
	ClassDB::bind_method(D_METHOD("get_button_index", "id"), &FoldableContainer::get_button_index);
	ClassDB::bind_method(D_METHOD("set_button_toggle_mode", "index", "enabled"), &FoldableContainer::set_button_toggle_mode);
	ClassDB::bind_method(D_METHOD("get_button_toggle_mode", "index"), &FoldableContainer::get_button_toggle_mode);
	ClassDB::bind_method(D_METHOD("set_button_toggled", "index", "toggled_on"), &FoldableContainer::set_button_toggled);
	ClassDB::bind_method(D_METHOD("is_button_toggled", "index"), &FoldableContainer::is_button_toggled);
	ClassDB::bind_method(D_METHOD("set_button_icon", "index", "icon"), &FoldableContainer::set_button_icon);
	ClassDB::bind_method(D_METHOD("get_button_icon", "index"), &FoldableContainer::get_button_icon);
	ClassDB::bind_method(D_METHOD("set_button_tooltip", "index", "tooltip"), &FoldableContainer::set_button_tooltip);
	ClassDB::bind_method(D_METHOD("get_button_tooltip", "index"), &FoldableContainer::get_button_tooltip);
	ClassDB::bind_method(D_METHOD("set_button_disabled", "index", "disabled"), &FoldableContainer::set_button_disabled);
	ClassDB::bind_method(D_METHOD("is_button_disabled", "index"), &FoldableContainer::is_button_disabled);
	ClassDB::bind_method(D_METHOD("set_button_visible", "index", "hidden"), &FoldableContainer::set_button_visible);
	ClassDB::bind_method(D_METHOD("is_button_visible", "index"), &FoldableContainer::is_button_visible);
	ClassDB::bind_method(D_METHOD("set_button_auto_hide", "index", "auto_hide"), &FoldableContainer::set_button_auto_hide);
	ClassDB::bind_method(D_METHOD("is_button_auto_hide", "index"), &FoldableContainer::is_button_auto_hide);
	ClassDB::bind_method(D_METHOD("set_button_metadata", "index", "metadata"), &FoldableContainer::set_button_metadata);
	ClassDB::bind_method(D_METHOD("get_button_metadata", "index"), &FoldableContainer::get_button_metadata);
	ClassDB::bind_method(D_METHOD("get_button_at_position", "position"), &FoldableContainer::get_button_at_position);

	ADD_SIGNAL(MethodInfo("folding_changed", PropertyInfo(Variant::BOOL, "is_folded")));
	ADD_SIGNAL(MethodInfo("button_pressed", PropertyInfo(Variant::INT, "index")));
	ADD_SIGNAL(MethodInfo("button_toggled", PropertyInfo(Variant::BOOL, "toggled_on"), PropertyInfo(Variant::INT, "index")));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "folded"), "set_folded", "is_folded");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text"), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "title_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_title_alignment", "get_title_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "title_position", PROPERTY_HINT_ENUM, "Top,Bottom"), "set_title_position", "get_title_position");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_overrun_behavior", PROPERTY_HINT_ENUM, "Trim Nothing,Trim Characters,Trim Words,Ellipsis,Word Ellipsis"), "set_text_overrun_behavior", "get_text_overrun_behavior");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "foldable_group", PROPERTY_HINT_RESOURCE_TYPE, "FoldableGroup"), "set_foldable_group", "get_foldable_group");

	ADD_ARRAY_COUNT("Buttons", "button_count", "set_button_count", "get_button_count", "button_");

	ADD_GROUP("BiDi", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID), "set_language", "get_language");

	BIND_ENUM_CONSTANT(POSITION_TOP);
	BIND_ENUM_CONSTANT(POSITION_BOTTOM);

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, title_style, "title_panel");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, title_hover_style, "title_hover_panel");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, title_collapsed_style, "title_collapsed_panel");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, title_collapsed_hover_style, "title_collapsed_hover_panel");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, focus_style, "focus");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, panel_style, "panel");

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, button_normal_style);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, button_hovered_style);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, button_pressed_style);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, button_disabled_style);

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_FONT, FoldableContainer, title_font, "font");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_FONT_SIZE, FoldableContainer, title_font_size, "font_size");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_CONSTANT, FoldableContainer, title_font_outline_size, "outline_size");

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, FoldableContainer, title_font_color, "font_color");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, FoldableContainer, title_hovered_font_color, "hover_font_color");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, FoldableContainer, title_collapsed_font_color, "collapsed_font_color");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, FoldableContainer, title_font_outline_color, "font_outline_color");

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, FoldableContainer, button_icon_normal);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, FoldableContainer, button_icon_hovered);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, FoldableContainer, button_icon_pressed);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, FoldableContainer, button_icon_disabled);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FoldableContainer, expanded_arrow);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FoldableContainer, expanded_arrow_mirrored);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FoldableContainer, folded_arrow);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FoldableContainer, folded_arrow_mirrored);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, FoldableContainer, h_separation);

	Button defaults;
	base_property_helper.set_prefix("button_");
	base_property_helper.set_array_length_getter(&FoldableContainer::get_button_count);
	base_property_helper.register_property(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), defaults.icon, &FoldableContainer::set_button_icon, &FoldableContainer::get_button_icon);
	base_property_helper.register_property(PropertyInfo(Variant::INT, "id", PROPERTY_HINT_RANGE, "-1,8,1,or_greater"), defaults.id, &FoldableContainer::set_button_id, &FoldableContainer::get_button_id);
	base_property_helper.register_property(PropertyInfo(Variant::BOOL, "toggle_mode"), defaults.toggle_mode, &FoldableContainer::set_button_toggle_mode, &FoldableContainer::get_button_toggle_mode);
	base_property_helper.register_property(PropertyInfo(Variant::BOOL, "toggled_on"), defaults.toggled_on, &FoldableContainer::set_button_toggled, &FoldableContainer::is_button_toggled);
	base_property_helper.register_property(PropertyInfo(Variant::BOOL, "disabled"), defaults.disabled, &FoldableContainer::set_button_disabled, &FoldableContainer::is_button_disabled);
	base_property_helper.register_property(PropertyInfo(Variant::BOOL, "auto_hide"), defaults.auto_hide, &FoldableContainer::set_button_auto_hide, &FoldableContainer::is_button_auto_hide);
	base_property_helper.register_property(PropertyInfo(Variant::BOOL, "visible"), defaults.visible, &FoldableContainer::set_button_visible, &FoldableContainer::is_button_visible);
	base_property_helper.register_property(PropertyInfo(Variant::STRING, "tooltip"), defaults.tooltip, &FoldableContainer::set_button_tooltip, &FoldableContainer::get_button_tooltip);
	PropertyListHelper::register_base_helper(&base_property_helper);
}

FoldableContainer::FoldableContainer(const String &p_text) {
	text_buf.instantiate();
	set_text(p_text);
	set_focus_mode(FOCUS_ALL);
	set_mouse_filter(MOUSE_FILTER_STOP);
	property_helper.setup_for_instance(base_property_helper, this);
}

FoldableContainer::~FoldableContainer() {
	if (foldable_group.is_valid()) {
		foldable_group->containers.erase(this);
	}
}

FoldableContainer *FoldableGroup::get_expanded_container() {
	for (FoldableContainer *E : containers) {
		if (E->is_expanded()) {
			return E;
		}
	}

	return nullptr;
}

void FoldableGroup::set_allow_folding_all(bool p_enabled) {
	allow_folding_all = p_enabled;
	if (!allow_folding_all && !get_expanded_container() && containers.size() > 0) {
		updating_group = true;
		for (FoldableContainer *E : containers) {
			E->set_folded(false);
			break;
		}
		updating_group = false;
	}
}

bool FoldableGroup::is_allow_folding_all() {
	return allow_folding_all;
}

void FoldableGroup::get_containers(List<FoldableContainer *> *r_containers) {
	for (FoldableContainer *E : containers) {
		r_containers->push_back(E);
	}
}

TypedArray<FoldableContainer> FoldableGroup::_get_containers() {
	TypedArray<FoldableContainer> foldable_containers;
	for (const FoldableContainer *E : containers) {
		foldable_containers.push_back(E);
	}

	return foldable_containers;
}

void FoldableGroup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_expanded_container"), &FoldableGroup::get_expanded_container);
	ClassDB::bind_method(D_METHOD("get_containers"), &FoldableGroup::_get_containers);
	ClassDB::bind_method(D_METHOD("set_allow_folding_all", "enabled"), &FoldableGroup::set_allow_folding_all);
	ClassDB::bind_method(D_METHOD("is_allow_folding_all"), &FoldableGroup::is_allow_folding_all);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_folding_all"), "set_allow_folding_all", "is_allow_folding_all");

	ADD_SIGNAL(MethodInfo("expanded", PropertyInfo(Variant::OBJECT, "container", PROPERTY_HINT_RESOURCE_TYPE, "FoldableContainer")));
}

FoldableGroup::FoldableGroup() {
	set_local_to_scene(true);
}
